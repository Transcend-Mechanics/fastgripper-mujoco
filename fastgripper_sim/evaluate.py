"""Evaluate a saved SAC checkpoint and scripted controls on matched held-out seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .env import GripLiftEnv
from .assets import example_path


def run_episode(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    total = 0.0
    started = time.perf_counter()
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        total += reward
        if terminated or truncated:
            return {"seed": seed, "return": total, **info,
                    "wall_seconds": time.perf_counter()-started}


def summarize(rows):
    return {
        "episodes": len(rows), "successes": sum(int(row["is_success"]) for row in rows),
        "success_rate": float(np.mean([row["is_success"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_force_pct": float(np.mean([row["mean_force_pct"] for row in rows])),
        "mean_torque_squared_integral_nm2_s": float(np.mean([row["torque_squared_integral_nm2_s"] for row in rows])),
        "mean_final_object_lift_m": float(np.mean([row["object_lift_m"] for row in rows])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the bundled trained checkpoint")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--first-seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/grip_lift/evaluation.json"))
    parser.add_argument("--model", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.checkpoint is None:
        args.checkpoint = example_path("policy.zip")
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    env = GripLiftEnv(model_path=args.model)
    learned = SAC.load(args.checkpoint, device="cpu")
    policies = {
        "sac": lambda observation: learned.predict(observation, deterministic=True)[0],
        "scripted_30pct": lambda observation: np.asarray([1.0, 0.25], dtype=np.float32),
        "open_jaws": lambda observation: np.asarray([-1.0, 0.25], dtype=np.float32),
        "insufficient_5pct": lambda observation: np.asarray([1.0, -1.0], dtype=np.float32),
    }
    results = {}
    try:
        for name, policy in policies.items():
            rows = [run_episode(env, policy, seed) for seed in
                    range(args.first_seed, args.first_seed+args.episodes)]
            results[name] = {"summary": summarize(rows), "episodes": rows}
            print(name, json.dumps(results[name]["summary"]), flush=True)
        report = {
            "checkpoint": args.checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            **env.asset_provenance,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("mujoco", "numpy", "gymnasium", "stable-baselines3", "torch")},
            "first_seed": args.first_seed, "episodes_per_policy": args.episodes,
            "environment": env.task_description(), "policies": results,
            "interpretation": "Held-out simulator seeds within the training parameter ranges; no hardware or out-of-distribution claim. The scripted baseline is intentionally strong. Failure controls verify that success depends on a physical grasp.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
