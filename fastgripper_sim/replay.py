"""Render a deterministic rollout of the bundled learned gripper policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC
from PIL import Image, ImageDraw, ImageFont

from .assets import example_path
from .env import GripLiftEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("outputs/grip_lift/rollout"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or example_path("policy.zip")
    torch.set_num_threads(1)
    policy = SAC.load(checkpoint, device="cpu")
    env = GripLiftEnv(render_mode="rgb_array")
    frames, rows = [], []
    font = ImageFont.load_default(size=16)
    try:
        obs, info = env.reset(seed=args.seed)
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            im = Image.fromarray(env.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, im.width, 55), fill="#141b26")
            draw.text((14, 9), "FastGripper | Learned SAC policy", font=font, fill="#f3f6fc")
            draw.text((14, 31), "Policy: gripper commands | Fixture: prescribed lift", font=font, fill="#b2c2d3")
            draw.rectangle((0, im.height-35, im.width, im.height), fill="#141b26")
            draw.text((14, im.height-26),
                      f"t={info['simulation_seconds']:.2f}s | rise {info['object_lift_m']*1000:.1f} mm | held {info['held_seconds']:.2f}s",
                      font=font, fill="#b4e3d7")
            frames.append(im.quantize(colors=160))
            rows.append({"action": action.tolist(), "reward": reward, **info})
            if terminated or truncated:
                break
        report = {"seed": args.seed, "checkpoint": checkpoint.name,
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "environment": env.task_description(), "success": bool(info["is_success"]),
                  "policy_controls": "gripper closure and torque limit",
                  "fixture_motion": "prescribed", "samples": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[-1].save(args.output.with_suffix(".png"))
        frames[0].save(args.output.with_suffix(".gif"), save_all=True, append_images=frames[1:],
                       duration=round(1000/env.config.policy_hz), loop=0, optimize=False)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"success": report["success"], "seed": args.seed,
                          "frames": len(frames), "output": str(args.output)}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
