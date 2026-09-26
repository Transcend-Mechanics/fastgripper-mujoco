"""Train a small genuine SAC policy for GripLiftEnv; run with the learn extra."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .env import GripLiftEnv


class EpisodeLog(BaseCallback):
    def __init__(self, output: Path):
        super().__init__()
        self.output = output
        self.rows = []
        self.started = time.perf_counter()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                row = {"timesteps": self.num_timesteps, "episode": len(self.rows)+1,
                       "return": info["episode"]["r"], "success": int(info["is_success"]),
                       "mean_force_pct": info["mean_force_pct"],
                       "torque_squared_integral_nm2_s": info["torque_squared_integral_nm2_s"],
                       "wall_seconds": time.perf_counter()-self.started}
                self.rows.append(row)
                with (self.output/"training.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(self.rows)
                if len(self.rows) % 10 == 0:
                    window = self.rows[-10:]
                    print(f"step={self.num_timesteps} episodes={len(self.rows)} "
                          f"return10={np.mean([r['return'] for r in window]):.2f} "
                          f"success10={np.mean([r['success'] for r in window]):.0%}", flush=True)
        return True


def plot_training(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    x = np.asarray([r["timesteps"] for r in rows])
    returns = np.asarray([r["return"] for r in rows])
    success = np.asarray([r["success"] for r in rows])
    mean = lambda a: np.asarray([a[max(0, i-9):i+1].mean() for i in range(len(a))])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    axes[0].plot(x, returns, color="#b4c9e8", linewidth=0.8, label="Training episode")
    axes[0].plot(x, mean(returns), color="#2459a6", label="Trailing 10 episodes")
    axes[0].set_ylabel("Return")
    axes[0].legend(frameon=False)
    axes[1].plot(x, mean(success), color="#168477")
    axes[1].set_ylabel("Training success\n(trailing 10)")
    axes[1].set_xlabel("20 Hz policy steps")
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle("SAC: grip, prescribed 50 mm lift, sustained hold")
    fig.savefig(output/"learning_curve.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/grip_lift"))
    parser.add_argument("--model", type=Path, help="Optional alternate scene; omitted from portable metadata")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory; existing training artifacts are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = GripLiftEnv(model_path=args.model)
    hyperparameters = dict(learning_rate=3e-4, buffer_size=50_000, learning_starts=200,
                          batch_size=64, tau=0.005, gamma=0.98, train_freq=1,
                          gradient_steps=1, ent_coef="auto", policy_kwargs={"net_arch": [64, 64]})
    callback = EpisodeLog(args.output)
    model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0, **hyperparameters)
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        model.save(args.output/"policy")
        plot_training(callback.rows, args.output)
        metadata = {
            "algorithm": "SAC", "seed": args.seed, "requested_timesteps": args.steps,
            "trained_timesteps": model.num_timesteps, "completed_episodes": len(callback.rows),
            "wall_seconds": time.perf_counter()-started,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **env.asset_provenance,
            "checkpoint_sha256": hashlib.sha256((args.output/"policy.zip").read_bytes()).hexdigest(),
            "hyperparameters": hyperparameters, "environment": env.task_description(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("mujoco", "numpy", "gymnasium", "stable-baselines3", "torch")},
            "training_curve_note": "Stochastic training episodes, not held-out evaluation.",
        }
        (args.output/"training.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(json.dumps({k: metadata[k] for k in
                          ("algorithm", "trained_timesteps", "completed_episodes", "wall_seconds")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
