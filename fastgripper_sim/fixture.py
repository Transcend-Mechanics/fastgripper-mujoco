"""Shared command sequence and display helpers for the FastGripper grasp fixture.

Only the external fixture's mocap target is moved here. Object and gripper
joint states evolve through MuJoCo; reset_scene is the sole state reset path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from .controller import GripperController


def positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def percent(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 100:
        raise argparse.ArgumentTypeError("must be in 0..100")
    return result


def lift_distance(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 0.05:
        raise argparse.ArgumentTypeError("must be in 0..0.05 metres")
    return result


def reset_scene(model, data, controller: GripperController, speed_pct: float,
                force_pct: float) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "open")
    if key < 0 and model.nkey:
        key = 0
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    controller.reset()
    controller.set_enabled(True)
    controller.command(position_pct=0, speed_pct=speed_pct, force_pct=force_pct)


@dataclass(frozen=True)
class Phase:
    name: str
    seconds: float
    position_pct: float | None
    lift_fraction: float


PHASES = (
    Phase("OPEN / SETTLE", 1.0, 0.0, 0.0),
    Phase("CLOSE", 4.0, 100.0, 0.0),
    Phase("HOLD", 1.0, None, 0.0),
    Phase("LIFT FIXTURE", 1.0, None, 1.0),
    Phase("RAISED HOLD", 2.0, None, 1.0),
    Phase("LOWER FIXTURE", 1.0, None, 0.0),
    Phase("RELEASE / OPEN", 3.0, 0.0, 0.0),
    Phase("REST", 1.0, None, 0.0),
)


class GraspDemo:
    """Issue phase-entry commands and smoothly move the external lift fixture."""

    duration = sum(phase.seconds for phase in PHASES)

    def __init__(self, model, data, controller: GripperController, *,
                 speed_pct: float = 100, force_pct: float = 30,
                 lift_height: float = 0.05):
        self.model, self.data, self.controller = model, data, controller
        self.speed_pct, self.force_pct = speed_pct, force_pct
        self.lift_height = lift_height
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mount_target")
        self.mocap_id = int(model.body_mocapid[body]) if body >= 0 else -1
        if self.mocap_id < 0 and lift_height > 0:
            raise ValueError("Fixture lift requires a mocap body named mount_target")
        self.active = False
        self.repeat = True
        self.phase = "MANUAL"
        self._phase_token = None
        self.reset_origin()

    def reset_origin(self) -> None:
        self.origin = (self.data.mocap_pos[self.mocap_id].copy()
                       if self.mocap_id >= 0 else np.zeros(3))
        self.active = False
        self.phase = "MANUAL"
        self._phase_token = None

    def start(self, *, repeat: bool = True) -> None:
        self.started = float(self.data.time)
        self.repeat = repeat
        self.active = True
        self._phase_token = None
        self.controller.set_enabled(True)
        self.update()

    def stop(self) -> None:
        self.active = False
        self.phase = "MANUAL"
        self._phase_token = None

    @property
    def fixture_lift_m(self) -> float:
        if self.mocap_id < 0:
            return 0.0
        return float(self.data.mocap_pos[self.mocap_id, 2] - self.origin[2])

    def update(self) -> None:
        if not self.active:
            return
        elapsed = max(0.0, float(self.data.time) - self.started)
        if not self.repeat and elapsed >= self.duration:
            self.active = False
            self.phase = "COMPLETE"
            return
        cycle = int(elapsed // self.duration)
        remaining = elapsed % self.duration
        phase_index = 0
        for phase_index, phase in enumerate(PHASES):
            if remaining < phase.seconds:
                break
            remaining -= phase.seconds
        token = (cycle, phase_index)
        if token != self._phase_token:
            self._phase_token = token
            self.phase = phase.name
            self._lift_start = self.fixture_lift_m
            if phase.position_pct is not None:
                self.controller.command(position_pct=phase.position_pct,
                                        speed_pct=self.speed_pct, force_pct=self.force_pct)
        if self.mocap_id >= 0:
            alpha = np.clip(remaining / phase.seconds, 0.0, 1.0)
            smooth = alpha * alpha * (3 - 2 * alpha)
            target = phase.lift_fraction * self.lift_height
            height = self._lift_start + (target - self._lift_start) * smooth
            self.data.mocap_pos[self.mocap_id] = self.origin
            self.data.mocap_pos[self.mocap_id, 2] += height


def contact_sides(snapshot: dict) -> tuple[float, float]:
    forces = snapshot["contact_normal_force_n_by_side"]
    return float(forces["positive_x"]), float(forces["negative_x"])


def status_lines(snapshot: dict, phase: str, lift_m: float) -> list[str]:
    positive, negative = contact_sides(snapshot)
    enabled = "ON" if snapshot["enabled"] else "OFF"
    detection = {"closing": "closing contact", "opening": "opening contact",
                 "at_position": "target reached", "contact_lost": "contact lost",
                 "none": "none"}.get(snapshot["object_state"], snapshot["object_state"])
    return [
        f"{phase} | {snapshot['status']} | detection: {detection}",
        f"Aperture {snapshot['aperture_m'] * 1000:.1f} mm | motor {enabled}: "
        f"{snapshot['motor_torque_nm']:+.3f} N m / cap {snapshot['torque_cap_nm']:.3f} N m",
        f"Finger normal force +X {positive:.2f} N / -X {negative:.2f} N | "
        f"object Z {snapshot['object_height_m'] * 1000:.1f} mm | fixture +{lift_m * 1000:.0f} mm",
    ]


def configure_camera(camera, model, lift_height: float = 0.05) -> None:
    camera.lookat[:] = model.stat.center
    camera.lookat[2] += lift_height / 2
    camera.distance = max(float(model.stat.extent) * 1.65, 0.50)
    camera.azimuth = 110
    camera.elevation = -18


def physics_batch(model) -> tuple[int, float]:
    """Keep command updates near 1 kHz while MuJoCo runs finer native steps."""
    timestep = float(model.opt.timestep)
    if not math.isfinite(timestep) or not 0 < timestep <= 0.01:
        raise ValueError("Grasp fixture timestep must be finite and in (0, 0.01] seconds")
    steps = max(1, round(0.001 / timestep))
    return steps, steps * timestep


def ensure_valid_state(data) -> None:
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or np.any(data.warning.number)):
        raise RuntimeError("MuJoCo reported an invalid state or solver warning")
