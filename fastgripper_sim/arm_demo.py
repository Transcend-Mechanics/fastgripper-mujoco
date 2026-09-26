"""Actuator-driven UR approach, grasp, lift, lower and release example.

Run with ``python -m fastgripper_sim.arm_demo --output outputs/ur5e --render``.
The mount is provisional. This is an ideal joint-servo simulation, not a UR
controller emulator. No hardware connection is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assets import asset_path
from .controller import ControllerConfig, GripperController
from .fixture import contact_sides, ensure_valid_state, physics_batch


ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HOME = np.array([-math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.])
DOWN_QUAT = np.array([0., 1., 0., 0.])  # w, x, y, z; flange +Z points down.
LIFT_HEIGHT = .10
MOUNT_POSITION = (.0279919658976, -.00243382114975, .11)
ARM_SETTINGS = {
    "ur5e": {"label": "UR5e", "object_x": .50, "table_top": .12, "camera_distance": 1.65},
    "ur3e": {"label": "UR3e", "object_x": .26, "table_top": .06, "camera_distance": 1.35},
}
PHASES = (
    ("SETTLE", 1., 1., 0.),
    ("APPROACH", 2., 0., None),
    ("CLOSE", 4., 0., 100.),
    ("CONTACT HOLD", 1., 0., None),
    ("LIFT", 2., 1., None),
    ("RAISED HOLD", 2., 1., None),
    ("LOWER", 2., 0., None),
    ("RELEASE", 3., 0., 0.),
    ("RETRACT", 2., 1., None),
    ("REST", 1., 1., None),
)
DURATION = sum(phase[1] for phase in PHASES)


def _merge(root: ET.Element, element: ET.Element) -> None:
    existing = root.find(element.tag)
    if existing is None:
        root.append(copy.deepcopy(element))
    elif element.tag in ("option", "compiler", "statistic", "visual"):
        existing.attrib.update(element.attrib)
        for child in element:
            existing.append(copy.deepcopy(child))
    else:
        for child in element:
            existing.append(copy.deepcopy(child))


def _absolute_asset_files(root: ET.Element, directory: Path, meshdir: str = "") -> None:
    for element in root.iter():
        if "file" not in element.attrib:
            continue
        relative = Path(element.attrib["file"])
        path = directory / (meshdir if element.tag == "mesh" else "") / relative
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping bundled asset: {relative}")
        element.set("file", str(path.resolve()))


def make_arm_model(*, arm: str = "ur5e", object_xy: tuple[float, float] | None = None) -> mujoco.MjModel:
    """Compose bundled assets in memory; no absolute paths are exported."""
    if arm not in ARM_SETTINGS:
        raise ValueError(f"Unsupported arm: {arm}")
    settings = ARM_SETTINGS[arm]
    table_top = settings["table_top"]
    object_xy = object_xy or (settings["object_x"], 0.)
    arm_path = asset_path(f"{arm}.xml", model=arm)
    gripper_dir = asset_path("body.xml").parent
    root = ET.parse(arm_path).getroot()
    compiler = root.find("compiler")
    _absolute_asset_files(root, arm_path.parent, compiler.get("meshdir", ""))
    compiler.attrib.pop("meshdir", None)
    compiler.set("inertiafromgeom", "false")
    root.set("model", f"{settings['label']} with FastGripper — provisional adapter")
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    mechanism = ET.parse(gripper_dir / "mechanism.xml").getroot()
    _absolute_asset_files(mechanism, gripper_dir)
    for element in mechanism:
        if element.tag not in ("compiler", "keyframe", "worldbody", "statistic"):
            _merge(root, element)
    option = root.find("option")
    option.attrib.update(timestep="0.00005", gravity="0 0 -9.81", integrator="implicitfast",
                         solver="Newton", iterations="100", tolerance="1e-10",
                         cone="elliptic", impratio="10", noslip_iterations="3")
    wrist = root.find(".//body[@name='wrist_3_link']")
    attachment = wrist.find("site[@name='attachment_site']")
    flange = ET.SubElement(wrist, "body", name="tool_flange",
                           pos=attachment.get("pos", "0 0 0"), quat=attachment.get("quat", "1 0 0 0"))
    adapter = ET.SubElement(flange, "body", name="provisional_adapter")
    ET.SubElement(adapter, "inertial", mass="0.10", pos="0 0 0.015", diaginertia="0.0000331 0.0000331 0.0000512")
    ET.SubElement(adapter, "geom", name="adapter_envelope", type="cylinder", size="0.032 0.015", pos="0 0 0.015",
                  contype="1", conaffinity="7", rgba="0.25 0.27 0.30 1")
    mount = ET.SubElement(adapter, "body", name="gripper_mount", childclass="fastgripper",
                          pos=" ".join(map(str, MOUNT_POSITION)))
    body = ET.parse(gripper_dir / "body.xml").getroot()
    for child in body:
        mount.append(copy.deepcopy(child))
    world = root.find("worldbody")
    x, y = object_xy
    ET.SubElement(world, "light", pos="0.3 -0.7 1.8", dir="0 0 -1", diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1.2 1.2 .01", rgba=".12 .17 .23 1",
                  contype="4", conaffinity="3")
    table_half_width = .18 if arm == "ur5e" else .12
    ET.SubElement(world, "geom", name="table", type="box", pos=f"{x} {y} {table_top / 2}",
                  size=f"{table_half_width} .18 {table_top / 2}", rgba=".32 .38 .43 1", contype="4", conaffinity="3",
                  friction=".5 .001 .0001")
    block = ET.SubElement(world, "body", name="test_block", pos=f"{x} {y} {table_top + .015}")
    ET.SubElement(block, "freejoint", name="block_free")
    ET.SubElement(block, "inertial", mass=".05", pos="0 0 0", diaginertia=".00000615 .000010416667 .000009066667")
    ET.SubElement(block, "geom", name="test_block_collision", type="box", size=".02 .012 .015",
                  rgba=".94 .64 .25 1", contype="2", conaffinity="5", friction=".35 .001 .0001",
                  condim="4", solref=".004 1", solimp=".95 .99 .001")
    ET.SubElement(block, "site", name="object_center", size=".001", group="3")
    ET.SubElement(root, "statistic", center=".25 0 .34", extent="1.15")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1100", offheight="850")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class ArmTrajectory:
    """Solve flange waypoints using separate scratch state; actuate live state."""

    def __init__(self, model: mujoco.MjModel, object_xy: tuple[float, float], *, arm: str = "ur5e"):
        self.model = model
        self.qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.dof = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self.actuators = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
        self.site = model.site("attachment_site").id
        self.scratch = mujoco.MjData(model)
        self.scratch.qpos[self.qpos] = HOME
        self.object_xy = object_xy
        self.grasp_flange_z = ARM_SETTINGS[arm]["table_top"] + .15 + MOUNT_POSITION[2]
        self.arm = arm
        self.times = np.linspace(0., 1., 51)
        self.path = []
        seed = HOME.copy()
        # Solve from the high pose toward the table for a continuous IK branch.
        for lift in self.times[::-1]:
            goal = np.array([*object_xy, self.grasp_flange_z + lift * LIFT_HEIGHT])
            seed = self.solve(goal, seed)
            self.path.append(seed.copy())
        self.path = np.asarray(self.path[::-1])
        self.path_derivative = np.gradient(self.path, self.times, axis=0)

    def solve(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        data = self.scratch
        data.qpos[self.qpos] = seed
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        quat, inverse, error_quat, orientation = (np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3))
        for _ in range(250):
            mujoco.mj_forward(self.model, data)
            mujoco.mju_mat2Quat(quat, data.site_xmat[self.site])
            mujoco.mju_negQuat(inverse, quat)
            mujoco.mju_mulQuat(error_quat, DOWN_QUAT, inverse)
            mujoco.mju_quat2Vel(orientation, error_quat, 1.)
            position = target - data.site_xpos[self.site]
            error = np.concatenate((position, orientation))
            if np.linalg.norm(position) < 1e-7 and np.linalg.norm(orientation) < 1e-7:
                return data.qpos[self.qpos].copy()
            mujoco.mj_jacSite(self.model, data, jp, jr, self.site)
            jac = np.vstack((jp[:, self.dof], jr[:, self.dof]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            delta *= min(1., .2 / max(np.max(np.abs(delta)), 1e-12))
            data.qpos[self.qpos] += delta
            for index, name in enumerate(ARM_JOINTS):
                limits = self.model.joint(name).range
                data.qpos[self.qpos[index]] = np.clip(data.qpos[self.qpos[index]], limits[0] + .001, limits[1] - .001)
        raise RuntimeError(f"{self.arm} inverse kinematics did not converge; position error {np.linalg.norm(position):.6g} m")

    def target(self, lift: float) -> np.ndarray:
        return np.array([np.interp(lift, self.times, self.path[:, joint]) for joint in range(6)])

    def update(self, data: mujoco.MjData, lift: float, lift_velocity: float = 0.) -> None:
        target = self.target(lift)
        velocity = np.array([np.interp(lift, self.times, self.path_derivative[:, joint])
                             for joint in range(6)]) * lift_velocity
        # Feed gravity through the existing bounded servos, never through an
        # unbounded external force or direct state assignment.
        gains = self.model.actuator_gainprm[self.actuators, 0]
        damping = -self.model.actuator_biasprm[self.actuators, 2]
        compensated = target + (damping * velocity + data.qfrc_bias[self.dof]) / gains
        limits = self.model.actuator_ctrlrange[self.actuators]
        data.ctrl[self.actuators] = np.clip(compensated, limits[:, 0], limits[:, 1])


class ArmDemo:
    duration = DURATION

    def __init__(self, *, arm: str = "ur5e", seed: int = 0, randomize: bool = False, force: float = 30.):
        if not math.isfinite(force) or not 1 <= force <= 100:
            raise ValueError("force must be in 1..100 percent")
        if arm not in ARM_SETTINGS:
            raise ValueError(f"Unsupported arm: {arm}")
        self.arm = arm
        rng = np.random.default_rng(seed)
        self.object_xy = (ARM_SETTINGS[arm]["object_x"] + (rng.uniform(-.015, .015) if randomize else 0.),
                          rng.uniform(-.02, .02) if randomize else 0.)
        self.model = make_arm_model(arm=arm, object_xy=self.object_xy)
        self.data = mujoco.MjData(self.model)
        self.controller = GripperController(self.model, self.data, ControllerConfig.from_model(self.model))
        self.trajectory = ArmTrajectory(self.model, self.object_xy, arm=arm)
        self.force = force
        self.reset()

    def reset(self) -> None:
        """The only assignment to the live arm/object joint configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.trajectory.qpos] = self.trajectory.target(1.)
        open_angle = self.controller.config.open_motor_rad
        numeric = self.model.numeric("signed_worm_ratio")
        ratio = float(numeric.data[0])
        angle = open_angle / ratio
        initial = {"worm_joint": open_angle, "driver_positive_joint": angle,
                   "finger_joint": -angle, "follower_joint": angle,
                   "driver_negative_joint": -angle, "finger_negative_joint": angle,
                   "follower_negative_joint": -angle}
        for name, value in initial.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.controller.set_enabled(True)
        self.controller.command(position_pct=0., speed_pct=100., force_pct=self.force)
        self.trajectory.update(self.data, 1.)
        mujoco.mj_forward(self.model, self.data)
        self.initial_object_height = float(self.data.body("test_block").xpos[2])
        self.phase_index = -1
        self.phase = "SETTLE"
        self.lift_fraction = 1.

    def update(self, dt: float) -> None:
        remaining = min(float(self.data.time), DURATION - 1e-9)
        previous_lift = 1.
        for index, (name, seconds, lift, command) in enumerate(PHASES):
            if remaining < seconds:
                alpha = np.clip(remaining / seconds, 0., 1.)
                smooth = alpha * alpha * (3 - 2 * alpha)
                self.lift_fraction = previous_lift + (lift - previous_lift) * smooth
                lift_velocity = (lift - previous_lift) * 6 * alpha * (1 - alpha) / seconds
                if self.phase_index != index:
                    self.phase_index, self.phase = index, name
                    if command is not None:
                        self.controller.command(position_pct=command, speed_pct=100., force_pct=self.force)
                break
            remaining -= seconds
            previous_lift = lift
        self.trajectory.update(self.data, self.lift_fraction, lift_velocity)
        self.controller.update(dt)

    def snapshot(self) -> dict:
        state = self.controller.snapshot()
        positive, negative = contact_sides(state)
        target = np.array([*self.object_xy, self.trajectory.grasp_flange_z + self.lift_fraction * LIFT_HEIGHT])
        closure = max(np.linalg.norm(self.data.site(a).xpos - self.data.site(b).xpos) for a, b in
                      (("finger_pivot_B", "follower_tip_B"), ("finger_negative_pivot_B", "follower_negative_tip_B")))
        return {"time_s": float(self.data.time), "phase": self.phase,
                "object_rise_m": state["object_height_m"] - self.initial_object_height,
                "object_height_m": state["object_height_m"], "aperture_m": state["aperture_m"],
                "positive_contact_n": positive, "negative_contact_n": negative,
                "motor_torque_nm": state["motor_torque_nm"], "object_state": state["object_state"],
                "flange_position_error_m": float(np.linalg.norm(self.data.site("attachment_site").xpos - target)),
                "max_closure_error_m": float(closure),
                "arm_position_rad": self.data.qpos[self.trajectory.qpos].tolist(),
                "arm_torque_nm": self.data.actuator_force[self.trajectory.actuators].tolist()}


class ContactAudit:
    """Record active contacts at control boundaries, including unwanted pairs."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.pairs: dict[tuple[int, int], dict] = {}
        self.force = np.zeros(6)
        block = model.geom("test_block_collision").id
        self.expected = {tuple(sorted((block, model.geom(name).id))) for name in
                         ("table", "finger_positive_contact", "finger_negative_contact")}

    def update(self, data: mujoco.MjData) -> None:
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.efc_address < 0:
                continue
            key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if key not in self.pairs:
                names = [self.model.geom(g).name or
                         f"{self.model.body(int(self.model.geom_bodyid[g])).name}/geom{g}" for g in key]
                self.pairs[key] = {"geoms": names, "task_contact": key in self.expected,
                                   "max_normal_force_n": 0., "max_penetration_m": 0., "observations": 0}
            mujoco.mj_contactForce(self.model, data, index, self.force)
            record = self.pairs[key]
            record["observations"] += 1
            record["max_normal_force_n"] = max(record["max_normal_force_n"], abs(float(self.force[0])))
            record["max_penetration_m"] = max(record["max_penetration_m"], -float(contact.dist))

    def summary(self) -> list[dict]:
        return [self.pairs[key] for key in sorted(self.pairs)]


def run_demo(*, output: Path | None = None, render: bool = False,
             arm: str = "ur5e", seed: int = 0, randomize: bool = False, fps: int = 15) -> dict:
    if not 1 <= fps <= 60:
        raise ValueError("fps must be in 1..60")
    if render and output is None:
        raise ValueError("Rendering needs an output prefix")
    demo = ArmDemo(arm=arm, seed=seed, randomize=randomize)
    model, data = demo.model, demo.data
    batch, _ = physics_batch(model)
    dt = float(model.opt.timestep)
    contact_audit = ContactAudit(model)
    samples, frames, preview = [], [], None
    renderer = None
    if render:
        from PIL import Image, ImageDraw, ImageFont
        width, height = 1100, 850
        renderer = mujoco.Renderer(model, width=width, height=height)
        camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
        camera.lookat[:] = [ARM_SETTINGS[arm]["object_x"] * .56, 0, .29 if arm == "ur5e" else .24]
        camera.distance, camera.azimuth, camera.elevation = ARM_SETTINGS[arm]["camera_distance"], 135, -23
        options.sitegroup[:] = 0
        options.geomgroup[2] = 1  # UR visual meshes use group 2.
        options.geomgroup[3] = 0  # Hide UR collision proxies and reference sites.
        title_font, font = ImageFont.load_default(size=21), ImageFont.load_default(size=16)
    started = time.perf_counter()
    try:
        for frame in range(round(DURATION * fps)):
            remaining = max(0, round(((frame + 1) / fps - float(data.time)) / dt))
            while remaining:
                count = min(batch, remaining)
                demo.update(count * dt)
                mujoco.mj_step(model, data, nstep=count)
                contact_audit.update(data)
                remaining -= count
            ensure_valid_state(data)
            snap = demo.snapshot()
            samples.append(snap)
            if renderer:
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame_image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame_image)
                draw.rectangle((0, 0, width, 76), fill="#132233")
                draw.text((18, 10), f"{ARM_SETTINGS[arm]['label']} + FastGripper | Actuator-driven grasp and lift", font=title_font, fill="#f5f7fa")
                draw.text((18, 45), "Joint servos + contact dynamics | provisional adapter and physical parameters", font=font, fill="#b8cbdc")
                draw.rectangle((0, height - 82, width, height), fill="#132233")
                draw.text((18, height - 72), f"{snap['phase']} | object rise {snap['object_rise_m']*1000:.1f} mm | aperture {snap['aperture_m']*1000:.1f} mm", font=font, fill="#f5f7fa")
                draw.text((18, height - 46), f"Finger contact {snap['positive_contact_n']:.2f} / {snap['negative_contact_n']:.2f} N | motor {snap['motor_torque_nm']:+.3f} N m", font=font, fill="#b4e3d7")
                draw.text((18, height - 23), "The object is free: no object attachment or pose playback.", font=font, fill="#b8cbdc")
                if snap["phase"] == "RAISED HOLD":
                    preview = frame_image.copy()
                frames.append(frame_image.quantize(colors=160))
    finally:
        if renderer:
            renderer.close()
    held = [s for s in samples if s["phase"] == "RAISED HOLD" and s["time_s"] >= 10.5]
    final = samples[-1]
    success = bool(held and min(s["object_rise_m"] for s in held) > .08
                   and min(min(s["positive_contact_n"], s["negative_contact_n"]) for s in held) > .1
                   and abs(final["object_rise_m"]) < .002
                   and final["positive_contact_n"] + final["negative_contact_n"] < .01
                   and all(pair["task_contact"] for pair in contact_audit.summary()))
    result = {"schema_version": 1, "arm": ARM_SETTINGS[arm]["label"], "success": success, "seed": seed,
              "arm_source_commit": json.loads(asset_path("manifest.json", model=arm).read_text())["commit"],
              "table_top_m": ARM_SETTINGS[arm]["table_top"],
              "object_xy_m": list(demo.object_xy), "duration_s": DURATION,
              "timestep_s": dt, "wall_seconds": time.perf_counter() - started,
              "mount_status": "Provisional rigid adapter; not a measured installation transform.",
              "mount_reference_frame": "Menagerie attachment_site" if arm == "ur5e" else "UR tool0, aliased as attachment_site",
              "attachment_to_gripper_translation_m": list(MOUNT_POSITION),
              "attachment_to_gripper_quaternion_wxyz": [1, 0, 0, 0],
              "max_object_rise_m": max(s["object_rise_m"] for s in samples),
              "minimum_raised_hold_rise_m": min((s["object_rise_m"] for s in held), default=0.),
              "max_flange_position_error_m": max(s["flange_position_error_m"] for s in samples),
              "max_abs_arm_torque_nm": [max(abs(s["arm_torque_nm"][joint]) for s in samples) for joint in range(6)],
              "arm_actuator_force_limits_nm": model.actuator_forcerange[demo.trajectory.actuators].tolist(),
              "arm_joint_planning_range_rad": [model.joint(name).range.tolist() for name in ARM_JOINTS],
              "arm_joint_limited": [bool(model.joint(name).limited[0]) for name in ARM_JOINTS],
              "max_closure_error_m": max(s["max_closure_error_m"] for s in samples),
              "contact_audit_interval_s": batch * dt,
              "contact_pairs": contact_audit.summary(),
              "warning_count": sum(int(w.number) for w in data.warning), "final": final, "samples": samples}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        if frames:
            # GIF delays are whole centiseconds; alternate frame durations so
            # the exported animation retains the simulated cycle duration.
            delays = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps)) for i in range(len(frames))]
            frames[0].save(output.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=delays, loop=0, optimize=False)
            (preview or frames[-1]).save(output.with_suffix(".png"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_SETTINGS), default="ur5e")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true", help="Seeded small object-position variation, with matching planned approach")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    result = run_demo(output=args.output or Path("outputs") / args.arm, render=args.render,
                      arm=args.arm, seed=args.seed, randomize=args.randomize, fps=args.fps)
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "final")}, indent=2))
    if not result["success"]:
        raise SystemExit(f"{args.arm} demo did not satisfy grasp/lift/release checks; inspect telemetry.")


if __name__ == "__main__":
    main()
