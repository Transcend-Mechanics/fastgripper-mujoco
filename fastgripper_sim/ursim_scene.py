"""UR-controller joint references tracked by bounded MuJoCo arm actuators.

The scene uses UR controller base coordinates and nominal manufacturer DH
geometry. Arm trajectories come from URSim; contact forces remain in MuJoCo.
This module does not implement the proprietary UR servo or safety controller.
"""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

from .arm_demo import ARM_ACTUATORS, ARM_JOINTS, ARM_SETTINGS, ArmTrajectory, make_arm_model
from .controller import ControllerConfig, GripperController
from .rtde import RTDEReference


def align_ur_controller_frames(model: mujoco.MjModel, arm: str) -> None:
    """Apply nominal UR geometry only to this in-memory integration scene.

    UR3e assets use ROS base_link (pi about Z from UR base). Menagerie UR5e
    already has the controller convention but rounds several link dimensions.
    Published DH reference:
    https://www.universal-robots.com/developer/hardware-and-motion/robot-motion-dh-parameters/
    """
    if arm == "ur3e":
        model.body_quat[model.body("base").id] = (1., 0., 0., 0.)
    elif arm == "ur5e":
        model.body_pos[model.body("shoulder_link").id, 2] = .1625
        model.body_pos[model.body("wrist_1_link").id, 2] = .3922
        # d4 = upper_arm offset .138 + forearm offset -.131 + wrist_2.
        model.body_pos[model.body("wrist_2_link").id, 1] = .1263
        model.body_pos[model.body("wrist_3_link").id, 2] = .0997
        model.site_pos[model.site("attachment_site").id, 1] = .0996
        model.body_pos[model.body("tool_flange").id, 1] = .0996
    else:
        raise ValueError(f"Unsupported UR arm: {arm}")
    mujoco.mj_setConst(model, mujoco.MjData(model))


class URSimScene:
    """Mutable simulation state owned and locked by the runtime main loop.

    update_arm only changes arm controls. The owner updates the gripper and
    advances physics. reset is the sole live joint-state assignment; calling it
    with a first RTDE sample is appropriate before enabling the bridge only.
    """

    def __init__(self, *, arm: str = "ur5e"):
        if arm not in ARM_SETTINGS:
            raise ValueError(f"Unsupported UR arm: {arm}")
        self.arm = arm
        self.object_xy = (ARM_SETTINGS[arm]["object_x"], 0.)
        self.model = make_arm_model(arm=arm, object_xy=self.object_xy)
        align_ur_controller_frames(self.model, arm)
        self.data = mujoco.MjData(self.model)
        self.controller = GripperController(self.model, self.data, ControllerConfig.from_model(self.model))
        self.trajectory = ArmTrajectory(self.model, self.object_xy, arm=arm)
        self.qpos, self.dof, self.actuators = self.trajectory.qpos, self.trajectory.dof, self.trajectory.actuators
        # Manufacturer maximum reference rates. These bound command slew;
        # physical velocity and collisions remain consequences of dynamics.
        self.max_joint_speed = np.full(6, math.pi)
        if arm == "ur3e":
            self.max_joint_speed[3:] = 2 * math.pi
        self.joint_limits = np.array([self.model.joint(name).range for name in ARM_JOINTS])
        self.reference_q = np.zeros(6)
        self.reference_qd = np.zeros(6)
        self.raw_reference_q = np.zeros(6)
        self.holding = True
        self.reference_clipped = False
        self.reference_rate_limited = False
        self.reset()

    def _joint_vector(self, values: Sequence[float]) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError("Arm reference must contain six finite joint values")
        return result

    def reset(self, initial_q: Sequence[float] | None = None) -> None:
        pose = self.trajectory.target(1.) if initial_q is None else self._joint_vector(initial_q)
        if np.any(pose < self.joint_limits[:, 0]) or np.any(pose > self.joint_limits[:, 1]):
            raise ValueError("Initial UR joint pose exceeds the bundled model joint ranges")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = pose
        angle = self.controller.config.open_motor_rad
        wheel = angle / float(self.model.numeric("signed_worm_ratio").data[0])
        initial = {"worm_joint": angle, "driver_positive_joint": wheel,
                   "finger_joint": -wheel, "follower_joint": wheel,
                   "driver_negative_joint": -wheel, "finger_negative_joint": wheel,
                   "follower_negative_joint": -wheel}
        for name, value in initial.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()  # Bridge activation explicitly enables it later.
        self.reference_q = pose.copy()
        self.raw_reference_q = pose.copy()
        self.reference_qd[:] = 0.
        self.reference_clipped = self.reference_rate_limited = False
        self.holding = True
        self.initial_object_height = float(self.data.body("test_block").xpos[2])
        self._apply_arm_controls()
        mujoco.mj_forward(self.model, self.data)

    def reset_from_reference(self, reference: RTDEReference) -> None:
        # Synchronize to the simulated robot's current position, then follow
        # target_q during stepping. This is never called automatically on a
        # reconnect, where resetting would teleport arm, gripper and object.
        self.reset(reference.actual_q)

    def update_arm(self, reference: RTDEReference | None, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Arm update dt must be positive")
        if reference is None:
            if not self.holding:
                self.reference_q = self.data.qpos[self.qpos].copy()
            self.reference_qd[:] = 0.
            self.holding = True
            self.reference_rate_limited = False
        else:
            self.raw_reference_q = self._joint_vector(reference.target_q)
            desired = np.clip(self.raw_reference_q, self.joint_limits[:, 0], self.joint_limits[:, 1])
            self.reference_clipped = bool(np.any(desired != self.raw_reference_q))
            delta = desired - self.reference_q
            bounded_delta = np.clip(delta, -self.max_joint_speed * dt, self.max_joint_speed * dt)
            self.reference_rate_limited = bool(np.any(np.abs(delta - bounded_delta) > 1e-12))
            self.reference_q += bounded_delta
            self.reference_qd = np.clip(reference.target_qd, -self.max_joint_speed, self.max_joint_speed)
            # On a discontinuity/reconnect, feed forward only the bounded
            # approach speed until the position reference has caught up.
            limited = np.abs(delta - bounded_delta) > 1e-12
            self.reference_qd[limited] = bounded_delta[limited] / dt
            self.holding = False
        self._apply_arm_controls()

    def _apply_arm_controls(self) -> None:
        gains = self.model.actuator_gainprm[self.actuators, 0]
        damping = -self.model.actuator_biasprm[self.actuators, 2]
        compensated = self.reference_q + (damping * self.reference_qd + self.data.qfrc_bias[self.dof]) / gains
        limits = self.model.actuator_ctrlrange[self.actuators]
        self.data.ctrl[self.actuators] = np.clip(compensated, limits[:, 0], limits[:, 1])

    def snapshot(self) -> dict:
        position = self.data.qpos[self.qpos]
        return {"arm": self.arm, "frame": "UR controller base; nominal DH",
                "arm_position_rad": position.tolist(), "arm_reference_rad": self.reference_q.tolist(),
                "arm_reference_velocity_rad_s": self.reference_qd.tolist(),
                "arm_raw_reference_rad": self.raw_reference_q.tolist(),
                "max_arm_tracking_error_rad": float(np.max(np.abs(position - self.reference_q))),
                "arm_torque_nm": self.data.actuator_force[self.actuators].tolist(),
                "arm_force_limits_nm": self.model.actuator_forcerange[self.actuators].tolist(),
                "holding": self.holding, "reference_clipped": self.reference_clipped,
                "reference_rate_limited": self.reference_rate_limited,
                "flange_position_m": self.data.site("attachment_site").xpos.tolist(),
                "flange_rotation_matrix": self.data.site("attachment_site").xmat.reshape(3, 3).tolist()}
