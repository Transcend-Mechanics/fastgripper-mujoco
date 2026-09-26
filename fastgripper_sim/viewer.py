#!/usr/bin/env python3
"""Run the FastGripper gravity/contact fixture in the native MuJoCo viewer.

G: grasp, O: open, D: demo, M: motor enable, Space: pause, R: reset.
The motor is torque controlled. The demo lifts only the external mocap fixture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import sys
import time

from .assets import asset_path


def main() -> None:
    import mujoco
    import mujoco.viewer

    from .fixture import (GraspDemo, configure_camera, ensure_valid_state,
                            lift_distance, percent, physics_batch, positive_seconds, reset_scene,
                            status_lines)
    from .controller import ControllerConfig, GripperController

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=asset_path())
    parser.add_argument("--demo", action="store_true", help="repeat the grasp/hold/lift/release cycle")
    parser.add_argument("--duration", type=positive_seconds, help="close after this many wall-clock seconds")
    parser.add_argument("--speed", type=percent, default=100, help="command speed percentage (default: 100)")
    parser.add_argument("--force", type=percent, default=30, help="command motor force percentage (default: 30)")
    parser.add_argument("--lift-height", type=lift_distance, default=0.05,
                        help="external fixture lift in metres, 0..0.05 (default: 0.05)")
    args = parser.parse_args()

    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        launcher = Path(sys.executable).parent / "mjpython"
        if not launcher.is_file():
            parser.exit(1, f"macOS requires {launcher}; run uv sync first.\n")
        os.execv(str(launcher), [str(launcher), "-m", "fastgripper_sim.viewer", *sys.argv[1:]])

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.exit(1, f"Model not found: {model_path}. Reinstall the runtime asset package.\n")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    batch_steps, control_timestep = physics_batch(model)
    data = mujoco.MjData(model)
    controller = GripperController(model, data, ControllerConfig.from_model(model))
    reset_scene(model, data, controller, args.speed, args.force)
    demo = GraspDemo(model, data, controller, speed_pct=args.speed,
                     force_pct=args.force, lift_height=args.lift_height)
    if args.demo:
        demo.start()
    paused = False
    keys: queue.SimpleQueue[int] = queue.SimpleQueue()
    print(f"Loaded {model_path}: torque control, gravity and contacts.", flush=True)
    print("G: grasp | O: open | D: cycle | M: motor on/off | Space: pause | R: reset | Esc: exit", flush=True)
    print("The cycle moves the external fixture; object motion comes from contact and gravity.", flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            configure_camera(viewer.cam, model, args.lift_height)
            viewer.opt.sitegroup[:] = 0
        viewer.sync()
        started = previous = time.monotonic()
        accumulated = 0.0
        while viewer.is_running():
            frame_start = time.monotonic()
            if args.duration is not None and frame_start - started >= args.duration:
                break
            elapsed = min(frame_start - previous, 0.1)
            previous = frame_start
            viewer.sync()
            with viewer.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key in (ord("G"), ord("O")):
                        demo.stop()
                        controller.set_enabled(True)
                        controller.command(position_pct=100 if key == ord("G") else 0,
                                           speed_pct=args.speed, force_pct=args.force)
                    elif key == ord("D"):
                        demo.stop() if demo.active else demo.start()
                    elif key == ord("M"):
                        demo.stop()
                        controller.set_enabled(not controller.snapshot()["enabled"])
                    elif key == ord(" "):
                        paused = not paused
                        accumulated = 0.0
                    elif key == ord("R"):
                        reset_scene(model, data, controller, args.speed, args.force)
                        demo.reset_origin()
                        accumulated = 0.0
                    elif key == 256:
                        viewer.close()
                if not paused:
                    accumulated += elapsed
                    while accumulated >= control_timestep:
                        demo.update()
                        controller.update(dt=control_timestep)
                        mujoco.mj_step(model, data, nstep=batch_steps)
                        accumulated -= control_timestep
                    ensure_valid_state(data)
                else:
                    mujoco.mj_forward(model, data)
                snapshot = controller.snapshot()
            if hasattr(viewer, "set_texts"):
                phase = "PAUSED" if paused else demo.phase
                lines = ["FastGripper | Grasp physics prototype", *status_lines(snapshot, phase, demo.fixture_lift_m),
                         "G grasp | O open | D cycle | M motor | Space pause | R reset",
                         "Provisional motor, mass and friction parameters; externally moved fixture"]
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                  mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "\n".join(lines), ""))
            viewer.sync()
            remaining = 1 / 60 - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    print(f"Closed after {data.time:.2f} simulated seconds.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
