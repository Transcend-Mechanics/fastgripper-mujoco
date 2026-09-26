#!/usr/bin/env python3
"""Render a repeatable motor-driven stock fingertip opening/closing preview.

Run from an installed checkout with ``python scripts/render_preview.py``.
The default is hardware v1.1; --model selects an explicit reusable model XML.
Outputs are a GIF, a PNG still and a JSON record of actual simulation steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastgripper_sim.controller import GripperController


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_files(path):
    """Load only local MJCF dependencies and record their relative filenames."""
    base = path.parent.resolve()
    files = {}

    def visit(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Preview models must reference assets inside their own directory")
        name = relative.as_posix()
        if name in files:
            return
        source = base/name
        if source.is_symlink() or not source.resolve().is_relative_to(base):
            raise ValueError("Preview dependency is outside the model directory")
        files[name] = source.read_bytes()
        if source.suffix == ".xml":
            for element in ET.fromstring(files[name]).iter():
                if element.tag == "compiler" and any(element.get(key) for key in ("meshdir", "texturedir", "assetdir")):
                    raise ValueError("Use explicit local asset references without compiler directory overrides")
                for key, value in element.attrib.items():
                    if key == "file" or key.startswith("file"):
                        visit((relative.parent/PurePosixPath(value)).as_posix())

    visit(path.name)
    return files


def studio_model(path, width, height):
    files = model_files(path)
    root = ET.fromstring(files[path.name])
    if root.find("worldbody/body[@name='gripper_mount']") is None:
        raise ValueError("Use the reusable fixed-base gripper model")
    world = root.find("worldbody")
    if any(item.tag == "freejoint" for item in world.iter()):
        raise ValueError("Use the reusable fixed-base model, rather than the grasp fixture")
    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", active="1", ambient="0.32 0.32 0.34",
                  diffuse="0.55 0.56 0.59", specular="0.22 0.22 0.22")
    ET.SubElement(visual, "rgba", haze="0.055 0.075 0.095 1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="preview_sky", type="skybox", builtin="gradient",
                  rgb1="0.035 0.05 0.07", rgb2="0.14 0.18 0.23", width="512", height="3072")
    ET.SubElement(asset, "material", name="preview_floor_material", rgba="0.16 0.20 0.24 1",
                  specular="0.2", shininess="0.3", reflectance="0.06")
    ET.SubElement(world, "geom", name="preview_floor", type="plane", pos="0 0 -0.0995",
                  size="2 2 0.01", material="preview_floor_material", contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "geom", name="preview_plinth", type="cylinder",
                  pos="-0.0275 0.0024 -0.090", size="0.045 0.0095", rgba="0.13 0.17 0.21 1",
                  contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "light", name="preview_key", pos="-0.2 -0.28 0.42", dir="0.35 0.45 -1",
                  diffuse="0.82 0.84 0.88", specular="0.6 0.6 0.6", castshadow="true")
    ET.SubElement(world, "light", name="preview_fill", pos="0.24 0.1 0.25", dir="-0.7 -0.2 -1",
                  diffuse="0.4 0.54 0.6", specular="0.25 0.3 0.3", castshadow="false")
    compiled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=files)
    # Appearance changes have no contact or mass effect.
    for side in ("positive", "negative"):
        compiled.geom(f"stock_metal_{side}_visual").rgba[:] = [.26, .77, .69, 1]
        compiled.geom(f"carrier_{side}_visual").rgba[:] = [.54, .62, .70, 1]
    return compiled, files


def caption(frame, aperture, state):
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    title = ImageFont.load_default(size=29)
    body = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    label = ImageFont.load_default(size=11)
    white, muted, accent = "#eef4f5", "#acbac5", "#63cdbc"
    draw.text((34, 26), "FASTGRIPPER", font=title, fill=white)
    draw.text((36, 65), "v1.1  /  Stock fingertips", font=body, fill=muted)
    draw.line((36, 94, 85, 94), fill=accent, width=3)
    draw.rounded_rectangle((width-155, 31, width-31, 61), radius=14, fill="#213c40")
    draw.text((width-139, 40), "MUJOCO PHYSICS", font=label, fill=accent)
    draw.line((34, height-86, width-34, height-86), fill="#425364", width=1)
    draw.text((35, height-67), state.upper(), font=body, fill=white)
    draw.text((width-210, height-67), f"Jaw opening  {aperture*1000:5.1f} mm", font=body, fill=accent)
    draw.text((35, height-37), "Motor-driven simulation  /  Nominal command range 3-104 mm", font=small, fill=muted)
    draw.text((35, height-20), "Provisional parameters; hardware travel and forces await calibration.", font=small, fill=muted)
    return frame


def render(path, output, *, width=800, height=700, fps=20, seconds=12):
    model, sources = studio_model(path, width, height)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)
    mujoco.mj_forward(model, data)
    controller = GripperController(model, data)
    controller.set_enabled(True)
    controller.command(0, speed_pct=100, force_pct=30)
    step = float(model.opt.timestep)
    batch = max(1, round(.001/step))
    max_torque, total_steps = 0., 0

    def advance(until):
        nonlocal max_torque, total_steps
        remaining = max(0, round((until-float(data.time))/step))
        while remaining:
            count = min(batch, remaining)
            controller.update(count*step)
            mujoco.mj_step(model, data, nstep=count)
            max_torque = max(max_torque, abs(controller.motor_torque_nm))
            total_steps += count
            remaining -= count
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.any(data.warning.number):
            raise RuntimeError("Simulation produced invalid state or warnings")

    warmup = .5
    advance(warmup)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.0275, .0024, .032]
    camera.distance = .46
    camera.azimuth = 108
    camera.elevation = -16
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[2] = 0
    frame_count = round(seconds*fps)
    close_at, reopen_at = 1., 6.
    commands = [{"time_s": 0., "position_pct": 0}, {"time_s": close_at, "position_pct": 100},
                {"time_s": reopen_at, "position_pct": 0}]
    phase, frames, samples, palette, still = 0, [], [], None, None
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index in range(frame_count):
            elapsed = index/fps
            next_phase = 0 if elapsed < close_at else 1 if elapsed < reopen_at else 2
            if next_phase != phase:
                controller.command(commands[next_phase]["position_pct"], speed_pct=100, force_pct=30)
                phase = next_phase
            advance(warmup + (index+1)/fps)
            snapshot = controller.snapshot()
            aperture = float(snapshot["aperture_m"])
            target = commands[phase]["position_pct"]
            settled = abs(controller.motor_position_rad-controller.target_motor_rad) < .35
            state = ("Open" if target == 0 else "Closed") if settled else ("Opening" if target == 0 else "Closing")
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = caption(Image.fromarray(renderer.render()), aperture, state)
            if still is None:
                still = frame.copy()
                palette = frame.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            samples.append({"time_s": (index+1)/fps, "aperture_m": aperture,
                            "motor_position_rad": controller.motor_position_rad,
                            "motor_torque_nm": controller.motor_torque_nm,
                            "target_position_pct": target})
    output.parent.mkdir(parents=True, exist_ok=True)
    gif, png, record = (Path(str(output) + suffix) for suffix in (".gif", ".png", ".json"))
    still.save(png)
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=round(1000/fps), optimize=True, disposal=1)
    aperture_range = [min(item["aperture_m"] for item in samples), max(item["aperture_m"] for item in samples)]
    if aperture_range[1]-aperture_range[0] < .09:
        raise RuntimeError("Preview did not demonstrate the expected opening/closing travel")
    report = {"schema_version": 1, "hardware_revision": "v1.1", "fingertip": "stock-metal",
              "model": path.name, "renderer": "scripts/render_preview.py",
              "model_files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sources.items())},
              "gif_sha256": digest(gif),
              "media_sha256": {gif.name: digest(gif)},
              "mujoco_version": mujoco.__version__, "width": width, "height": height, "fps": fps,
              "rendered_frames": frame_count, "playback_seconds": frame_count/fps,
              "simulation_seconds": float(data.time), "warmup_seconds": warmup,
              "physics_timestep_s": step, "native_physics_steps": total_steps,
              "controller_update_period_s": batch*step, "controller_driven_motion": True,
              "state_animation": False, "commands": commands,
              "speed_pct": 100, "force_pct": 30, "maximum_motor_torque_nm": max_torque,
              "warning_count": int(data.warning.number.sum()), "measured_aperture_range_m": aperture_range,
              "nominal_command_aperture_range_m": model.numeric("nominal_command_gap_range_m").data.tolist(),
              "scope": "Rigid stock fingertips; conservative nominal command span, not calibrated hardware travel.",
              "studio": "Visual ground, plinth and lights; contacts disabled on studio geometry.",
              "samples": samples}
    record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gif": gif.name, "gif_bytes": gif.stat().st_size, "frames": frame_count,
                      "measured_aperture_range_mm": [value*1000 for value in aperture_range],
                      "warning_count": report["warning_count"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reusable fixed-base stock v1.1 model.xml")
    parser.add_argument("--output", type=Path, default=Path("build/media/gripper-v1.1"), help="Output filename prefix")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=12.)
    args = parser.parse_args()
    if not 640 <= args.width <= 1920 or not 600 <= args.height <= 1440 or not 10 <= args.fps <= 40:
        parser.error("Use width 640..1920, height 600..1440 and fps 10..40")
    if not np.isfinite(args.seconds) or not 11 <= args.seconds <= 30:
        parser.error("--seconds must be 11..30 so both motor moves can finish")
    if args.model is None:
        from fastgripper_sim.models import model_path
        args.model = model_path(hardware="v1.1", tip="stock-metal")
    render(args.model.expanduser().resolve(), args.output, width=args.width, height=args.height,
           fps=args.fps, seconds=args.seconds)


if __name__ == "__main__":
    main()
