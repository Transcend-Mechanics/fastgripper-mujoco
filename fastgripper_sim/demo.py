"""Run a complete contact-driven grasp, hold, lift and release experiment.

python -m fastgripper_sim.demo --output outputs/grasp --render
python -m fastgripper_sim.viewer --demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .assets import asset_path
from .controller import ControllerConfig, GripperController
from .fixture import (GraspDemo, configure_camera, contact_sides,
                      ensure_valid_state, percent, physics_batch, reset_scene)


def run_demo(*, output: Path | None = None, render: bool = False,
             model_path: Path | None = None, force: float = 30.0,
             fps: int = 20, width: int = 800, height: int = 680) -> dict:
    """Return telemetry; rendering is optional and never affects the physics."""
    if not 1 <= fps <= 100:
        raise ValueError("fps must be in 1..100")
    if render and output is None:
        raise ValueError("Rendering requires an output prefix")
    model = mujoco.MjModel.from_xml_path(str(model_path or asset_path()))
    data = mujoco.MjData(model)
    controller = GripperController(model, data, ControllerConfig.from_model(model))
    reset_scene(model, data, controller, 100, force)
    demo = GraspDemo(model, data, controller, force_pct=force)
    initial_height = float(data.body("test_block").xpos[2])
    demo.start(repeat=False)
    batch, _ = physics_batch(model)
    dt = float(model.opt.timestep)
    samples, frames = [], []
    renderer = None
    if render:
        from PIL import Image, ImageDraw, ImageFont
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
        configure_camera(camera, model)
        options.sitegroup[:] = 0
        options.geomgroup[2] = 0
        title_font = ImageFont.load_default(size=20)
        font = ImageFont.load_default(size=15)
    started = time.perf_counter()
    max_closure = 0.0
    try:
        for frame in range(round(demo.duration * fps)):
            remaining = max(0, round(((frame + 1) / fps - float(data.time)) / dt))
            while remaining:
                count = min(batch, remaining)
                demo.update()
                controller.update(count * dt)
                mujoco.mj_step(model, data, nstep=count)
                remaining -= count
            ensure_valid_state(data)
            snap = controller.snapshot()
            forces = contact_sides(snap)
            for a, b in (("finger_pivot_B", "follower_tip_B"),
                         ("finger_negative_pivot_B", "follower_negative_tip_B")):
                max_closure = max(max_closure, float(np.linalg.norm(
                    data.site(a).xpos - data.site(b).xpos)))
            row = {"time_s": float(data.time), "phase": demo.phase,
                   "aperture_m": snap["aperture_m"],
                   "motor_torque_nm": snap["motor_torque_nm"],
                   "torque_cap_nm": snap["torque_cap_nm"],
                   "object_height_m": snap["object_height_m"],
                   "object_rise_m": snap["object_height_m"] - initial_height,
                   "fixture_lift_m": demo.fixture_lift_m,
                   "positive_contact_n": forces[0], "negative_contact_n": forces[1],
                   "object_state": snap["object_state"]}
            samples.append(row)
            if renderer:
                renderer.update_scene(data, camera=camera, scene_option=options)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, width, 72), fill="#141b26")
                draw.text((16, 10), "FastGripper | Grasp, lift and release", font=title_font, fill="#f3f6fc")
                draw.text((16, 43), "Free object | contact physics | prescribed fixture lift", font=font, fill="#b2c2d3")
                draw.rectangle((0, height - 82, width, height), fill="#141b26")
                draw.text((16, height - 73), f"{demo.phase} | gap {snap['aperture_m']*1000:.1f} mm | rise {row['object_rise_m']*1000:.1f} mm", font=font, fill="#f3f6fc")
                draw.text((16, height - 49), f"Contact {forces[0]:.2f} / {forces[1]:.2f} N | motor {snap['motor_torque_nm']:+.3f} N m", font=font, fill="#b4e3d7")
                draw.text((16, height - 25), "Provisional physics parameters; external geometry simplified.", font=font, fill="#b2c2d3")
                frames.append(im.quantize(colors=160))
    finally:
        if renderer:
            renderer.close()
    held = [r for r in samples if r["phase"] == "RAISED HOLD" and r["time_s"] >= 7.5]
    retained = bool(held) and all(r["object_rise_m"] >= .045 and
        min(r["positive_contact_n"], r["negative_contact_n"]) > .05 for r in held)
    released = abs(samples[-1]["object_rise_m"]) < .001 and max(
        samples[-1]["positive_contact_n"], samples[-1]["negative_contact_n"]) < .01
    summary = {"schema_version": 1, "scope": "provisional_simulation",
               "fixture_motion": "prescribed; object motion derives from contact and gravity",
               "success": retained and released and max_closure < .00005,
               "retained": retained, "released": released,
               "simulation_seconds": float(data.time), "wall_seconds": time.perf_counter() - started,
               "physics_timestep_s": dt, "command_force_pct": force,
               "peak_rise_m": max(r["object_rise_m"] for r in samples),
               "max_closure_error_m": max_closure,
               "warning_count": int(data.warning.number.sum()), "samples": samples}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        if frames:
            still = next((i for i, r in enumerate(samples) if r["phase"] == "RAISED HOLD"), len(frames)//2)
            frames[still].save(output.with_suffix(".png"))
            frames[0].save(output.with_suffix(".gif"), save_all=True, append_images=frames[1:],
                           duration=round(1000/fps), loop=0, optimize=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp"))
    parser.add_argument("--render", action="store_true", help="also save a GIF and still image")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--force", type=percent, default=30.0)
    args = parser.parse_args()
    result = run_demo(output=args.output, render=args.render, model_path=args.model, force=args.force)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    if not result["success"]:
        raise SystemExit("Grasp/lift/release validation failed; see the telemetry")


if __name__ == "__main__":
    main()
