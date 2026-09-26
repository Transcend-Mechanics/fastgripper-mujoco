"""Run a physics-backed FastGripper bridge following a URSim joint stream.

URSim owns the arm reference trajectory. MuJoCo owns the simulated mechanism
and free objects; gripper status is returned through the bridge registers.
No command is sent to the RTDE source. Use the companion Docker demo to create
an explicitly isolated simulator and a normal PolyScope/URCap program.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import threading
import time

import mujoco
import numpy as np

from .arm_demo import ARM_SETTINGS, ContactAudit
from .bridge import BridgeDevice, ThreadedBridgeServer
from .fixture import ensure_valid_state, physics_batch
from .rtde import RTDEClient
from .ursim_scene import URSimScene


PHASES = {0: "INITIALIZE", 1: "OPEN", 2: "APPROACH", 3: "CLOSE", 4: "GRIP CHECK",
          5: "LIFT", 6: "RAISED HOLD", 7: "LOWER", 8: "RELEASE", 9: "RETRACT", 10: "DONE"}


class URSimBridge:
    """Single physics owner; protocol threads acquire the same state lock."""

    def __init__(self, *, arm="ur5e", rtde_host="127.0.0.1", rtde_port=30004,
                 listen="127.0.0.1", port=55431, max_reference_age=.5, max_lag=.5):
        if not math.isfinite(max_lag) or max_lag <= 0:
            raise ValueError("max_lag must be finite and positive")
        if not math.isfinite(max_reference_age) or max_reference_age <= 0:
            raise ValueError("max_reference_age must be finite and positive")
        self.scene = URSimScene(arm=arm)
        self.arm = arm
        self.lock = threading.RLock()
        self.device = BridgeDevice(self.scene.controller)
        self.server = ThreadedBridgeServer(self.device, self.lock, host=listen, port=port)
        self.rtde = RTDEClient(rtde_host, port=rtde_port)
        self.max_reference_age, self.max_lag = max_reference_age, max_lag
        self.batch, self.dt = physics_batch(self.scene.model)
        self.audit = ContactAudit(self.scene.model)
        self.samples = deque(maxlen=20_000)
        self.events = []
        self.error = None
        self.max_observed_lag = 0.
        self.overrun_count = 0
        self._stop = threading.Event()
        self._thread = None
        self._initial_height = float(self.scene.data.body("test_block").xpos[2])

    def start(self, timeout=30.):
        try:
            return self._start(timeout)
        except BaseException:
            self.close()
            raise

    def _start(self, timeout):
        self.rtde.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reference = self.rtde.latest(max_age=self.max_reference_age)
            if reference is not None:
                break
            time.sleep(.02)
        else:
            self.rtde.stop()
            raise TimeoutError("No fresh URSim RTDE reference; inspect the controller and port")
        with self.lock:
            self.scene.reset_from_reference(reference)
            self.scene.controller.set_enabled(False)
            self._initial_height = float(self.scene.data.body("test_block").xpos[2])
            self._controller_epoch = reference.timestamp - float(self.scene.data.time)
        try:
            self.server.start()
        except Exception:
            self.rtde.stop()
            raise
        self._thread = threading.Thread(target=self._run, name="fastgripper-physics", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        scene = self.scene
        controller_epoch = self._controller_epoch
        fallback_epoch = time.monotonic() - float(scene.data.time)
        last_controller_time = None
        previous_runtime_state = None
        next_sample = 0.
        lost = False
        clock_fault = False
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                reference = self.rtde.latest(max_age=self.max_reference_age)
                with self.lock:
                    if reference is None and not lost:
                        self.device.stop(reason="rtde_reference_lost")
                        self.events.append({"time_s": float(scene.data.time), "event": "rtde_reference_lost"})
                        fallback_epoch = now - float(scene.data.time)
                    if reference is not None and lost:
                        self.events.append({"time_s": float(scene.data.time), "event": "rtde_reference_recovered"})
                        # Preserve physical state across reconnection. Only the
                        # mapping between controller and simulation time changes.
                        controller_epoch = reference.timestamp - float(scene.data.time)
                    if reference is not None:
                        if last_controller_time is not None and reference.timestamp < last_controller_time:
                            clock_fault = True
                            controller_epoch = reference.timestamp - float(scene.data.time)
                            self.events.append({"time_s": float(scene.data.time), "event": "rtde_controller_clock_reset"})
                        last_controller_time = reference.timestamp
                        # URSim can run slower than wall time under emulation.
                        # Integrate one physics second per controller second so
                        # target_qd, gravity and motor dynamics share a clock.
                        target_time = reference.timestamp - controller_epoch
                        state = reference.runtime_state
                        if state != previous_runtime_state:
                            self.events.append({"time_s": float(scene.data.time), "event": "ur_program_state", "state": state})
                        previous_runtime_state = state
                        # The real URCap daemon owns program pause/resume/stop
                        # semantics, including its paused-command bookkeeping.
                        # Preempting its stop here would hide the moving state
                        # from that watcher and break its explicit resume path.
                    else:
                        # Keep gravity/contact physics alive during an outage.
                        target_time = now - fallback_epoch
                    lag = max(0., target_time - float(scene.data.time))
                    self.max_observed_lag = max(self.max_observed_lag, lag)
                    if lag > self.max_lag and reference is not None:
                        self.overrun_count += 1
                        clock_fault = True
                        self.device.stop(reason="simulation_clock_overrun")
                        self.events.append({"time_s": float(scene.data.time), "event": "simulation_clock_overrun", "lag_s": lag})
                        if reference is not None:
                            controller_epoch = reference.timestamp - float(scene.data.time)
                        fallback_epoch = now - float(scene.data.time)
                        target_time = float(scene.data.time)
                    # With no controller stream the gripper is already gated
                    # and the arm is holding. Catch up passive outage physics;
                    # a host scheduling delay here is not a missed UR trajectory.
                    lost = reference is None
                    if lost or clock_fault:
                        # A new socket command must not bypass a missing arm
                        # reference or a latched simulation-clock failure.
                        self.device.stop(reason="simulation_clock_overrun" if clock_fault else "rtde_reference_lost")
                    advance = target_time + 1e-10 >= float(scene.data.time) + self.dt
                    if advance:
                        scene.update_arm(None if clock_fault else reference, self.dt)
                        self.device.update(self.dt)
                        scene.controller.update(self.dt)
                        mujoco.mj_step(scene.model, scene.data, nstep=self.batch)
                        ensure_valid_state(scene.data)
                        self.audit.update(scene.data)
                        if scene.data.time + 1e-10 >= next_sample:
                            self.samples.append(self._sample(reference, lag))
                            next_sample = float(scene.data.time) + .05
                if not advance:
                    self._stop.wait(.001)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.device.stop(reason="physics_error")
            self._stop.set()

    def _sample(self, reference, lag):
        scene = self.scene
        state = scene.controller.snapshot()
        phase = reference.task_phase if reference is not None else -1
        return {"time_s": float(scene.data.time), "phase": PHASES.get(phase, str(phase)), "phase_code": phase,
                "ur_timestamp": reference.timestamp if reference is not None else None,
                "ur_runtime_state": reference.runtime_state if reference is not None else None,
                "reference_age_s": time.monotonic()-reference.received_at if reference is not None else None,
                "clock_lag_s": lag, "object_rise_m": state["object_height_m"]-self._initial_height,
                "object_height_m": state["object_height_m"], "aperture_m": state["aperture_m"],
                "motor_torque_nm": state["motor_torque_nm"], "torque_cap_nm": state["torque_cap_nm"],
                "state": state["status"], "object_state": state["object_state"],
                "contact_normal_force_n_by_side": state["contact_normal_force_n_by_side"],
                "qpos": scene.data.qpos.tolist(), "qvel": scene.data.qvel.tolist(),
                "arm_reference_rad": list(reference.target_q) if reference is not None else None,
                "arm_position_rad": scene.data.qpos[scene.qpos].tolist(),
                "arm_torque_nm": scene.data.actuator_force[scene.actuators].tolist(),
                "closure_error_m": max(float(np.linalg.norm(scene.data.site(a).xpos-scene.data.site(b).xpos))
                    for a,b in (("finger_pivot_B", "follower_tip_B"), ("finger_negative_pivot_B", "follower_negative_tip_B")))}

    def report(self):
        with self.lock:
            return {"schema_version": 1, "backend": "mujoco", "arm": self.arm,
                    "arm_coupling": "one-way URSim joint reference; bounded MuJoCo actuators",
                    "gripper_coupling": "bidirectional raw Modbus RTU over TCP",
                    "physics_clock": "UR controller timestamp; wall-time fallback while RTDE is unavailable",
                    "physics_timestep_s": self.scene.model.opt.timestep,
                    "control_timestep_s": self.dt, "error": self.error,
                    "device": self.device.snapshot(),
                    "warning_count": int(self.scene.data.warning.number.sum()),
                    "max_clock_lag_s": self.max_observed_lag, "overrun_count": self.overrun_count,
                    "rtde": self.rtde.status(), "events": list(self.events),
                    "contact_pairs": self.audit.summary(), "samples": list(self.samples),
                    "sample_retention": "latest 20000 samples at 20 Hz"}

    def close(self):
        self._stop.set()
        stuck = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            stuck = self._thread.is_alive()
        try:
            self.server.close()
        finally:
            self.rtde.stop()
        if stuck:
            raise RuntimeError("Physics owner did not stop")


def render_trace(report, output: Path, *, fps=15):
    """Render recorded physics states after execution, outside the live clock."""
    from PIL import Image, ImageDraw, ImageFont
    scene = URSimScene(arm=report["arm"])
    renderer = mujoco.Renderer(scene.model, height=850, width=1100)
    camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
    camera.lookat[:] = [.3 if report["arm"] == "ur5e" else .16, 0, .30]
    camera.distance = ARM_SETTINGS[report["arm"]]["camera_distance"]
    camera.azimuth, camera.elevation = 135, -22
    options.sitegroup[:] = 0
    title, font = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    frames, still = [], None
    rows = [row for row in report["samples"] if 1 <= row["phase_code"] <= 10]
    if not rows:
        rows = report["samples"]
    if not rows:
        renderer.close()
        raise ValueError("No recorded simulation states")
    next_time = rows[0]["time_s"]
    try:
        for row in rows:
            if row["time_s"] + 1e-8 < next_time:
                continue
            next_time += 1/fps
            scene.data.qpos[:] = row["qpos"]
            scene.data.qvel[:] = row["qvel"]
            mujoco.mj_forward(scene.model, scene.data)
            renderer.update_scene(scene.data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,1100,76), fill="#142231")
            draw.text((18,12), f"URSim + FastGripper URCap + MuJoCo | {report['arm'].upper()}", font=title, fill="white")
            draw.text((18,46), "Actual PolyScope program | Contact-driven gripper and free object", font=font, fill="#b3c8d8")
            draw.rectangle((0,762,1100,850), fill="#142231")
            draw.text((18,778), f"{row['phase']} | object rise {row['object_rise_m']*1000:.1f} mm | gap {row['aperture_m']*1000:.1f} mm", font=font, fill="white")
            draw.text((18,807), "Recorded physics states | provisional mount and physical parameters", font=font, fill="#b3c8d8")
            frames.append(frame.quantize(colors=160))
            if row["phase_code"] == 6:
                still = frames[-1]
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    (still or frames[len(frames)//2]).save(output.with_suffix(".png"))
    delays = [10*(round((i+1)*100/fps)-round(i*100/fps)) for i in range(len(frames))]
    frames[0].save(output.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=delays, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_SETTINGS), default="ur5e")
    parser.add_argument("--rtde-host", default="127.0.0.1")
    parser.add_argument("--rtde-port", type=int, default=30004)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55431)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/ursim/session.json"))
    parser.add_argument("--render-recording", type=Path, help="Render a completed JSON session instead of starting a bridge")
    args = parser.parse_args()
    if args.render_recording:
        render_trace(json.loads(args.render_recording.read_text()), args.output)
        return
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    bridge = URSimBridge(arm=args.arm, rtde_host=args.rtde_host, rtde_port=args.rtde_port,
                         listen=args.listen, port=args.port)
    try:
        bridge.start()
        print(json.dumps({"ready": True, "simulation_backend": "mujoco", "address": bridge.server.address, "arm": args.arm}), flush=True)
        started = time.monotonic()
        while not stop.wait(.25):
            if bridge.error:
                raise RuntimeError(bridge.error)
            if args.duration is not None and time.monotonic()-started >= args.duration:
                break
    finally:
        bridge.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(bridge.report(), indent=2)+"\n")


if __name__ == "__main__":
    main()
