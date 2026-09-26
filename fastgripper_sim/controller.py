"""Torque-driven grasp controls and measurements for the FastGripper runtime model.

The seek/stall rules follow fastgripper-cobot's TCAN485/Teensy firmware. This
models its 100 Hz position-to-velocity loop, feedback-relative velocity clamp,
and an ideal MIT velocity servo. A native hard torque limit represents the
intended cap; the hardware's sampled clamp cannot guarantee this instantaneous
bound between feedback updates. CAN timing, electronics, homing, faults, and
autonomous re-grasp are not simulated.
Motor geometry, inertia, friction and the endpoint calibration remain
provisional. Contact evidence is not a payload-presence sensor.

Call controller.update(model.opt.timestep) before each mujoco.mj_step. The
controller sets worm_motor's control and native actuator parameters. MuJoCo's
implicit integrator handles unsaturated velocity feedback. Saturated forces
still require a small physics timestep; this fixture uses 50 microseconds. It
never changes qpos/qvel, constraints, body poses, or mocap targets, and never
advances physics itself. Use a separate model instance for each controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import mujoco
import numpy as np


class Status(IntEnum):
    INACTIVE = 0
    READY = 2
    MOVING = 3


class ObjectState(IntEnum):
    NONE = 0
    CLOSING = 1
    OPENING = 2
    AT_POSITION = 3
    CONTACT_LOST = 4


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def model_numeric(model: mujoco.MjModel, name: str, default: float) -> float:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if index < 0:
        return default
    if model.numeric_size[index] != 1:
        raise ValueError(f"Model numeric {name!r} must be scalar")
    return float(model.numeric_data[model.numeric_adr[index]])


@dataclass(frozen=True)
class ControllerConfig:
    # Nominal simulation endpoints, not measured hardware calibration.
    open_motor_rad: float = -3.0
    closed_motor_rad: float = 10.4
    seek_gain: float = 3.0
    max_speed_rad_s: float = 4.0
    motor_kd: float = 1.0
    torque_cap_nm: float = 1.5
    control_hz: float = 100.0
    settle_rad: float = 0.30
    stall_window_s: float = 0.150
    stall_motion_rad: float = 0.02
    stall_cap_fraction: float = 0.65
    object_body_name: str = "test_block"

    def __post_init__(self):
        for name in ("open_motor_rad", "closed_motor_rad", "seek_gain", "max_speed_rad_s",
                     "motor_kd", "torque_cap_nm", "control_hz", "settle_rad",
                     "stall_window_s", "stall_motion_rad", "stall_cap_fraction"):
            value = _finite(getattr(self, name), name)
            if name not in ("open_motor_rad", "closed_motor_rad") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.open_motor_rad == self.closed_motor_rad:
            raise ValueError("Open and closed motor endpoints must differ")
        if self.stall_cap_fraction > 1:
            raise ValueError("stall_cap_fraction must not exceed 1")

    @classmethod
    def from_model(cls, model: mujoco.MjModel, **overrides) -> ControllerConfig:
        values = {name: model_numeric(model, name, getattr(cls, name))
                  for name in ("open_motor_rad", "closed_motor_rad")}
        values["torque_cap_nm"] = model_numeric(model, "motor_torque_cap_nm", cls.torque_cap_nm)
        return cls(**(values | overrides))


@dataclass(frozen=True)
class GripperCommand:
    position_pct: float
    speed_pct: float = 100.0
    force_pct: float = 50.0


def contact_normal_forces(model: mujoco.MjModel, data: mujoco.MjData,
                          geom_names: tuple[str, ...], other_body_name: str | None = None) -> dict[str, float]:
    """Sum solved compressive normals by finger geom, optionally against a body.

    These are simulator contact measurements, independent of motor-stall flags.
    Call after stepping/forward dynamics; no solver or state changes are made.
    """
    result = dict.fromkeys(geom_names, 0.0)
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): name for name in geom_names}
    ids.pop(-1, None)
    body = -1 if other_body_name is None else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other_body_name)
    if other_body_name is not None and body < 0:
        return result
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        for finger, other in ((int(contact.geom1), int(contact.geom2)), (int(contact.geom2), int(contact.geom1))):
            if finger not in ids or other < 0:
                continue
            if body >= 0 and model.geom_bodyid[other] != body:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            result[ids[finger]] += max(0.0, float(wrench[0]))
    return result


class GripperController:
    """Firmware-style motion intent with a separately selectable torque mode."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config: ControllerConfig | None = None):
        self.model, self.data = model, data
        self.config = config or ControllerConfig.from_model(model)
        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "worm_motor")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "worm_joint")
        if self.actuator_id < 0 or joint < 0:
            raise ValueError("Model requires worm_joint and the worm_motor actuator")
        if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or model.actuator_trntype[self.actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[self.actuator_id, 0] != joint
                or not np.isclose(model.actuator_gear[self.actuator_id, 0], 1.0)):
            raise ValueError("worm_motor must drive worm_joint with unit gear for motor-side angle and torque units")
        if (int(model.actuator_biastype[self.actuator_id]) not in
                (int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE))
                or model.actuator_dyntype[self.actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[self.actuator_id] != mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError("worm_motor must be a stateless fixed-gain motor or affine actuator")
        self.qpos_address = int(model.jnt_qposadr[joint])
        self.dof_address = int(model.jnt_dofadr[joint])
        self.reset()

    @property
    def motor_position_rad(self) -> float:
        return float(self.data.qpos[self.qpos_address])

    @property
    def motor_velocity_rad_s(self) -> float:
        return float(self.data.qvel[self.dof_address])

    @property
    def motor_torque_nm(self) -> float:
        return float(self.data.actuator_force[self.actuator_id])

    @property
    def torque_cap_nm(self) -> float:
        return self.config.torque_cap_nm * (1.0 if self.mode == "torque" else self.force_fraction)

    def reset(self) -> None:
        """Reset controller bookkeeping and disable motor; leave physics intact."""
        self.enabled, self.seeking = False, False
        self.mode, self.status, self.object_state = "idle", Status.INACTIVE, ObjectState.NONE
        self.target_motor_rad = self.motor_position_rad
        self.speed_fraction, self.force_fraction = 1.0, 0.5
        self.grip_torque_nm, self.peak_torque_nm = 0.0, 0.0
        self.grip_motor_rad = None
        self._clock, self._next_tick = 0.0, 0.0
        self._velocity_setpoint, self._direct_torque = 0.0, 0.0
        self._stall_position, self._stall_since = self.motor_position_rad, 0.0
        self._closing = False
        self._configure_actuator()
        self.data.ctrl[self.actuator_id] = 0.0

    def set_enabled(self, enabled: bool) -> None:
        if bool(enabled) == self.enabled:
            return
        self.enabled = bool(enabled)
        self._next_tick = self._clock
        self.stop()

    def stop(self) -> None:
        self.seeking, self.mode, self.object_state = False, "idle", ObjectState.NONE
        self.status = Status.READY if self.enabled else Status.INACTIVE
        self._velocity_setpoint, self._direct_torque = 0.0, 0.0
        self._next_tick = self._clock
        self._configure_actuator()
        self._apply_control()

    def apply_command(self, command: GripperCommand) -> None:
        self.command(command.position_pct, command.speed_pct, command.force_pct)

    def command(self, position_pct: float, speed_pct: float = 100, force_pct: float = 50) -> None:
        pct = _clip(_finite(position_pct, "position_pct"), 0.0, 100.0)
        span = self.config.closed_motor_rad - self.config.open_motor_rad
        self.command_angle(self.config.open_motor_rad + span * pct / 100, speed_pct, force_pct)

    def command_angle(self, target_rad: float, speed_pct: float = 100, force_pct: float = 50) -> None:
        if not self.enabled:
            raise RuntimeError("Enable the motor before commanding motion")
        target = _finite(target_rad, "target_rad")
        speed = _clip(_finite(speed_pct, "speed_pct"), 0.0, 100.0)
        force = _clip(_finite(force_pct, "force_pct"), 0.0, 100.0)
        self.speed_fraction, self.force_fraction = (speed or 100) / 100, (force or 50) / 100
        self.target_motor_rad = _clip(target, min(self.config.open_motor_rad, self.config.closed_motor_rad),
                                     max(self.config.open_motor_rad, self.config.closed_motor_rad))
        self._closing = ((self.target_motor_rad - self.motor_position_rad)
                         * (self.config.closed_motor_rad - self.config.open_motor_rad) > 0)
        self.mode, self.status, self.object_state = "position", Status.MOVING, ObjectState.NONE
        self.seeking = True
        self.grip_torque_nm, self.peak_torque_nm, self.grip_motor_rad = 0.0, 0.0, None
        self._stall_position, self._stall_since = self.motor_position_rad, self._clock
        self._next_tick = self._clock
        self._configure_actuator()

    def set_torque(self, torque_nm: float) -> None:
        """Validation mode: bypass seek/stall logic, retain the full motor cap."""
        if not self.enabled:
            raise RuntimeError("Enable the motor before commanding torque")
        self._direct_torque = _clip(_finite(torque_nm, "torque_nm"),
                                   -self.config.torque_cap_nm, self.config.torque_cap_nm)
        self.mode, self.seeking, self.object_state = "torque", False, ObjectState.NONE
        self.status = Status.MOVING if self._direct_torque else Status.READY
        self._configure_actuator()
        self._apply_control()

    def _update_motion_intent(self) -> None:
        self._velocity_setpoint = 0.0
        if not self.seeking:
            if self.grip_motor_rad is not None and self.object_state in (ObjectState.CLOSING, ObjectState.OPENING):
                drift_limit = max(0.30, abs(self.config.closed_motor_rad - self.config.open_motor_rad) * 0.01)
                if abs(self.motor_position_rad - self.grip_motor_rad) > drift_limit:
                    self.object_state = ObjectState.CONTACT_LOST
            return
        error = self.target_motor_rad - self.motor_position_rad
        if abs(error) < self.config.settle_rad:
            self.seeking, self.status, self.object_state = False, Status.READY, ObjectState.AT_POSITION
            return
        vmax = self.config.max_speed_rad_s * self.speed_fraction
        self._velocity_setpoint = _clip(self.config.seek_gain * error, -vmax, vmax)
        measured_torque = abs(self.motor_torque_nm)
        self.peak_torque_nm = max(self.peak_torque_nm, measured_torque)
        moved = abs(self.motor_position_rad - self._stall_position)
        at_cap = measured_torque >= self.torque_cap_nm * self.config.stall_cap_fraction
        if moved > self.config.stall_motion_rad or not at_cap:
            self._stall_position, self._stall_since = self.motor_position_rad, self._clock
        elif self._clock - self._stall_since > self.config.stall_window_s + 1e-12:
            self.seeking, self.status = False, Status.READY
            self.object_state = ObjectState.CLOSING if self._closing else ObjectState.OPENING
            self.grip_torque_nm, self.grip_motor_rad = measured_torque, self.motor_position_rad
            # As in firmware, this tick finishes its seek command. Subsequent
            # ticks request zero velocity. At rest KD therefore gives ZERO
            # active holding torque; retention must come from the mechanism.

    def _control_tick(self) -> None:
        self._update_motion_intent()
        # Match controlTick's final feedback-relative clamp, including idle,
        # settling, and the tick after detecting a stall. The transmitted MIT
        # velocity stays fixed until the next 100 Hz update. A separate native
        # force cap bounds the idealized inner servo between those updates.
        delta = self.torque_cap_nm / self.config.motor_kd
        velocity = self.motor_velocity_rad_s
        self._velocity_setpoint = _clip(self._velocity_setpoint, velocity-delta, velocity+delta)

    def _configure_actuator(self) -> None:
        """Use the engine's implicit velocity derivative, with native force caps."""
        actuator = self.actuator_id
        model = self.model
        model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator, :] = 0.0
        model.actuator_biasprm[actuator, :] = 0.0
        if self.enabled:
            if self.mode == "torque":
                model.actuator_gainprm[actuator, 0] = 1.0
            else:
                model.actuator_gainprm[actuator, 0] = self.config.motor_kd
                model.actuator_biasprm[actuator, 2] = -self.config.motor_kd
        # Disabled means no motor damping either. Mechanical friction remains
        # in the MJCF, independent of the actuator enable state.
        model.actuator_forcelimited[actuator] = True
        model.actuator_forcerange[actuator] = (-self.torque_cap_nm, self.torque_cap_nm)
        # The feedback-relative clamp can temporarily request a velocity above
        # seek speed when braking external backdrive. Only the requested seek
        # speed is bounded; re-clipping the transmitted value would differ from
        # the firmware. Torque mode has an ordinary force-valued control cap.
        model.actuator_ctrllimited[actuator] = self.mode == "torque"
        limit = self.config.torque_cap_nm if self.mode == "torque" else self.config.max_speed_rad_s
        model.actuator_ctrlrange[actuator] = (-limit, limit)

    def _apply_control(self) -> None:
        if not self.enabled:
            control = 0.0
        elif self.mode == "torque":
            control = self._direct_torque
        else:
            control = self._velocity_setpoint
        self.data.ctrl[self.actuator_id] = control

    def update(self, dt: float) -> None:
        """Sample intent at 100 Hz; MuJoCo integrates native velocity feedback."""
        dt = _finite(dt, "dt")
        period = 1 / self.config.control_hz
        if not 0 < dt <= period + 1e-12:
            raise ValueError("dt must be positive and no larger than the controller period")
        if self.enabled and self.mode != "torque" and self._clock + 1e-12 >= self._next_tick:
            self._control_tick()
            self._next_tick += period
        self._apply_control()
        self._clock += dt

    def snapshot(self) -> dict:
        geom_names = ("finger_positive_contact", "finger_negative_contact")
        forces = contact_normal_forces(self.model, self.data, geom_names, self.config.object_body_name)
        sides = {side: forces[name] for side, name in zip(("positive_x", "negative_x"), geom_names)}
        sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                 for name in ("finger_positive_contact_surface", "finger_negative_contact_surface")]
        aperture = float(np.linalg.norm(self.data.site_xpos[sites[0]] - self.data.site_xpos[sites[1]])) if min(sites) >= 0 else None
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.object_body_name)
        span = self.config.closed_motor_rad - self.config.open_motor_rad
        command_torque = (0.0 if not self.enabled else self._direct_torque if self.mode == "torque" else
                          _clip(self.config.motor_kd * (self._velocity_setpoint - self.motor_velocity_rad_s),
                                -self.torque_cap_nm, self.torque_cap_nm))
        return {"time_s": float(self.data.time), "enabled": self.enabled, "mode": self.mode,
                "status": self.status.name.lower(), "status_code": int(self.status),
                "object_state": self.object_state.name.lower(), "object_code": int(self.object_state),
                "seeking": self.seeking, "target_motor_rad": self.target_motor_rad,
                "motor_position_rad": self.motor_position_rad, "motor_velocity_rad_s": self.motor_velocity_rad_s,
                "motor_torque_nm": self.motor_torque_nm, "commanded_torque_nm": command_torque,
                "velocity_setpoint_rad_s": self._velocity_setpoint, "torque_cap_nm": self.torque_cap_nm,
                "position_pct": _clip(100 * (self.motor_position_rad - self.config.open_motor_rad) / span, 0, 100),
                "grip_torque_nm": self.grip_torque_nm, "grip_motor_rad": self.grip_motor_rad,
                "peak_torque_nm": self.peak_torque_nm, "aperture_m": aperture,
                "contact_forces_n_by_geom": forces, "contact_normal_force_n_by_side": sides,
                "total_finger_normal_force_n": sum(forces.values()),
                "object_height_m": float(self.data.xpos[body, 2]) if body >= 0 else None}
