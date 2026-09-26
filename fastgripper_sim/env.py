"""A small, state-observed grip / prescribed lift / hold learning task.

The policy chooses gripper target angle and FORCE; it does not move an arm.
All observations and rewards include simulator-only measurements. Reset places
the mechanism in a consistent pregrasp pose, and only MuJoCo moves its joints
and the free block thereafter. The external fixture follows a fixed 50 mm lift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .assets import asset_path
from .controller import GripperController
from .fixture import ensure_valid_state, physics_batch


@dataclass(frozen=True)
class TaskConfig:
    policy_hz: int = 20
    close_seconds: float = 0.8
    lift_seconds: float = 0.6
    hold_seconds: float = 0.8
    lift_m: float = 0.05
    success_height_m: float = 0.045
    success_hold_seconds: float = 0.5
    pregrasp_driver_rad: float = -0.23
    min_force_pct: float = 5.0
    max_force_pct: float = 45.0
    min_mass_kg: float = 0.03
    max_mass_kg: float = 0.07
    min_friction: float = 0.25
    max_friction: float = 0.45

    def __post_init__(self):
        if isinstance(self.policy_hz, bool) or not isinstance(self.policy_hz, int) or self.policy_hz <= 0:
            raise ValueError("policy_hz must be a positive integer")
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("close_seconds", "lift_seconds", "hold_seconds", "lift_m",
                     "success_height_m", "success_hold_seconds", "min_mass_kg"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.min_force_pct <= self.max_force_pct <= 100:
            raise ValueError("FORCE range must lie within (0, 100]")
        if self.max_mass_kg < self.min_mass_kg or not 0 <= self.min_friction <= self.max_friction:
            raise ValueError("Mass and friction ranges must be ordered and nonnegative")
        if self.success_height_m > self.lift_m or self.success_hold_seconds > self.hold_seconds:
            raise ValueError("Success thresholds must fit within the prescribed lift and hold")

    @property
    def duration(self):
        return self.close_seconds + self.lift_seconds + self.hold_seconds


OBSERVATIONS = (
    "phase_fraction", "aperture_m/0.1", "motor_angle_rad/12",
    "motor_velocity_rad_s/4", "motor_torque_nm/1.5", "block_lift_m/0.05",
    "block_vertical_velocity_m_s/0.2", "fixture_lift_m/0.05",
    "positive_normal_n/30", "negative_normal_n/30", "mass_kg/0.1",
    "friction", "held_target_fraction", "held_force_fraction", "seeking",
)


class GripLiftEnv(gym.Env):
    """Continuous 20 Hz actions with held, quantized firmware commands.

    action[0]: -1=pregrasp angle, +1=closed endpoint; quantized to 2%.
    action[1]: -1=5% FORCE, +1=45% FORCE; quantized to 2 percentage points.
    An unchanged command is held, preserving firmware stall detection. A new
    command is a real retarget and restarts seeking as in the bridge firmware.
    SPEED is fixed at 100%. No learning policy or reward overrides physics.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, *, model_path: str | Path | None = None,
                 config: TaskConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or TaskConfig()
        self.model_path = Path(model_path) if model_path else asset_path()
        self.asset_provenance = {
            "model_reference": "custom_scene" if model_path else "assets/fastgripper/scene.xml",
            "model_xml_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            "runtime_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("env.py", "controller.py", "fixture.py")},
        }
        if model_path is None:
            manifest = self.model_path.parent / "manifest.json"
            self.asset_provenance["asset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        else:
            self.asset_provenance["custom_model_hash_scope"] = "Entry XML only; external includes and assets are not covered."
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = GripperController(self.model, self.data)
        if abs(self.model.opt.timestep - 0.00005) > 1e-12:
            raise ValueError("This task requires the validated 50 microsecond physics timestep")
        self.batch, self.update_dt = physics_batch(self.model)
        self.policy_dt = 1 / self.config.policy_hz
        self.batches_per_action = round(self.policy_dt / self.update_dt)
        if abs(self.batches_per_action * self.update_dt - self.policy_dt) > 1e-10:
            raise ValueError("Policy period must be a whole number of native physics batches")
        self.max_steps = round(self.config.duration / self.policy_dt)
        if abs(self.max_steps * self.policy_dt - self.config.duration) > 1e-10:
            raise ValueError("Episode duration must contain a whole number of policy steps")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        self._renderer = None
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBSERVATIONS),), dtype=np.float32)
        self._block_id = self.model.body("test_block").id
        self._block_geom = self.model.geom("test_block_collision").id
        self._block_dof = int(self.model.joint("block_free").dofadr[0])
        self._fixture_id = int(self.model.body("mount_target").mocapid[0])
        self._contact_ids = [self.model.geom(n).id for n in
                             ("finger_positive_contact", "finger_negative_contact", "test_block_collision")]
        self._pregrasp_motor = -20 * self.config.pregrasp_driver_rad
        if not min(self.controller.config.open_motor_rad, self.controller.config.closed_motor_rad) <= self._pregrasp_motor <= max(self.controller.config.open_motor_rad, self.controller.config.closed_motor_rad):
            raise ValueError("Pregrasp pose must be within the model's motor endpoints")
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        cfg = self.config
        self.mass_kg = float(options.get("mass_kg", self.np_random.uniform(cfg.min_mass_kg, cfg.max_mass_kg)))
        self.friction = float(options.get("friction", self.np_random.uniform(cfg.min_friction, cfg.max_friction)))
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        self.model.body_mass[self._block_id] = self.mass_kg
        halfsize = self.model.geom_size[self._block_geom]
        self.model.body_inertia[self._block_id] = self.mass_kg / 3 * (np.sum(halfsize**2) - halfsize**2)
        self.model.geom_friction[self._contact_ids, 0] = self.friction
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("open").id)
        q = cfg.pregrasp_driver_rad
        for name, angle in {
            "worm_joint": -20*q, "driver_positive_joint": q, "finger_joint": -q,
            "follower_joint": q, "driver_negative_joint": -q,
            "finger_negative_joint": q, "follower_negative_joint": -q,
        }.items():
            self.data.joint(name).qpos[0] = angle
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.controller.set_enabled(True)
        self._origin = self.data.mocap_pos[self._fixture_id].copy()
        self._initial_block_z = float(self.data.body("test_block").xpos[2])
        self._step_count = 0
        self._held_command = None
        self._hold_time = 0.0
        self._return = 0.0
        self._torque_squared_integral = 0.0
        self._force_pct_sum = 0.0
        self._command_count = 0
        self._needs_reset = False
        self._snapshot = self.controller.snapshot()
        if self._snapshot["aperture_m"] <= 2*self.model.geom_size[self._block_geom, 0]+0.001:
            raise ValueError("Pregrasp pose must clear the block by at least 1 mm")
        return self._observation(), self._info()

    def _command(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain two finite numbers")
        action = np.clip(action, -1, 1)
        target_fraction = float(np.round((action[0]+1) / 2 * 50) / 50)
        force = self.config.min_force_pct + (action[1]+1) / 2 * (self.config.max_force_pct-self.config.min_force_pct)
        force_pct = float(np.clip(2 * np.round(force / 2), self.config.min_force_pct, self.config.max_force_pct))
        command = (target_fraction, force_pct)
        if command != self._held_command:
            target = self._pregrasp_motor + target_fraction * (self.controller.config.closed_motor_rad-self._pregrasp_motor)
            self.controller.command_angle(target, speed_pct=100, force_pct=force_pct)
            self._held_command = command
            self._command_count += 1
        return command

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset before stepping a new episode")
        _, force_pct = self._command(action)
        cfg = self.config
        starting_effort = self._torque_squared_integral
        for _ in range(self.batches_per_action):
            t = self.data.time + self.update_dt
            fraction = float(np.clip((t-cfg.close_seconds) / cfg.lift_seconds, 0, 1))
            self.data.mocap_pos[self._fixture_id, 2] = self._origin[2] + cfg.lift_m * (3*fraction**2 - 2*fraction**3)
            self.controller.update(self.update_dt)
            mujoco.mj_step(self.model, self.data, nstep=self.batch)
            self._torque_squared_integral += self.controller.motor_torque_nm**2 * self.update_dt
        ensure_valid_state(self.data)
        self._step_count += 1
        self._force_pct_sum += force_pct
        self._snapshot = self.controller.snapshot()
        lift = self._snapshot["object_height_m"] - self._initial_block_z
        normal = min(self._snapshot["contact_normal_force_n_by_side"].values())
        vertical_speed = float(self.data.qvel[self._block_dof+2])
        stable_hold = (lift >= cfg.success_height_m and normal > 0.05 and
                       abs(vertical_speed) < 0.025 and self.data.time >= cfg.close_seconds+cfg.lift_seconds)
        self._hold_time = self._hold_time+self.policy_dt if stable_hold else 0.0
        lifted_fraction = float(np.clip(lift / cfg.lift_m, 0, 1))
        # Penalize measured actuator effort as well as the requested cap. A
        # lower cap alone need not save effort if repeated retargets keep the
        # motor seeking, whereas a settled worm can retain the grasp passively.
        mean_torque_squared = (self._torque_squared_integral-starting_effort) / self.policy_dt
        max_command_cap = self.controller.config.torque_cap_nm * cfg.max_force_pct/100
        reward = (2.0*lifted_fraction + 0.15*float(normal > 0.05)
                  - 0.08*(force_pct/cfg.max_force_pct)**2
                  - 0.2*mean_torque_squared/max_command_cap**2)
        # Time/phase is observed and this finite task ends after its prescribed
        # hold. Terminate rather than mark an external time-limit truncation;
        # SAC must not bootstrap beyond the terminal success reward.
        terminated = self._step_count >= self.max_steps
        success = self._hold_time + 1e-9 >= cfg.success_hold_seconds
        if terminated:
            reward += 10.0 * success
            self._needs_reset = True
        self._return += reward
        info = self._info()
        if terminated:
            info["episode"] = {"r": self._return, "l": self._step_count}
        return self._observation(), float(reward), terminated, False, info

    def _observation(self):
        s = self._snapshot
        normal = s["contact_normal_force_n_by_side"]
        command = self._held_command or (0.0, 0.0)
        return np.asarray([
            self._step_count / self.max_steps, s["aperture_m"] / 0.1,
            s["motor_position_rad"] / 12, s["motor_velocity_rad_s"] / 4,
            s["motor_torque_nm"] / 1.5,
            (s["object_height_m"]-self._initial_block_z) / 0.05,
            self.data.qvel[self._block_dof+2] / 0.2,
            (self.data.mocap_pos[self._fixture_id, 2]-self._origin[2]) / 0.05,
            normal["positive_x"] / 30, normal["negative_x"] / 30,
            self.mass_kg / 0.1, self.friction, command[0], command[1] / 100,
            float(s["seeking"]),
        ], dtype=np.float32)

    def _info(self):
        return {
            "is_success": self._hold_time+1e-9 >= self.config.success_hold_seconds,
            "held_seconds": self._hold_time,
            "object_lift_m": self._snapshot["object_height_m"]-self._initial_block_z,
            "mass_kg": self.mass_kg, "friction": self.friction,
            "mean_force_pct": self._force_pct_sum / max(1, self._step_count),
            "torque_squared_integral_nm2_s": self._torque_squared_integral,
            "commands_issued": self._command_count,
            "simulation_seconds": float(self.data.time),
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0, 0, 0.14)
        camera.distance, camera.azimuth, camera.elevation = 0.48, 135, -15
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def task_description(self):
        return {"task": "grip-lift-hold", "config": asdict(self.config),
                "observations": list(OBSERVATIONS), "physics_timestep_s": self.model.opt.timestep,
                "actions": ["pregrasp-to-closed target", f"{self.config.min_force_pct:g}-to-{self.config.max_force_pct:g} percent FORCE"],
                "episode_end": "finite_horizon_termination",
                "reward": "2*clipped_lift_fraction + 0.15*bilateral_contact - 0.08*(FORCE/max_FORCE)^2 - 0.2*mean_step_torque_squared/max_command_torque_squared + 10*terminal_success",
                "privileged_simulator_observations": True, "prescribed_fixture_lift": True}
