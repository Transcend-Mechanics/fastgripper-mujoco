"""Commands must drive forces without assigning poses."""
import mujoco
import numpy as np
import pytest

from fastgripper_sim.assets import asset_path
from fastgripper_sim.controller import GripperController
from fastgripper_sim.fixture import reset_scene


@pytest.fixture
def rig():
    model = mujoco.MjModel.from_xml_path(str(asset_path()))
    data = mujoco.MjData(model)
    controller = GripperController(model, data)
    reset_scene(model, data, controller, 100, 30)
    return model, data, controller


def test_command_updates_do_not_teleport_and_torque_is_bounded(rig):
    model, data, controller = rig
    before = [array.copy() for array in (data.qpos, data.qvel, data.mocap_pos)]
    controller.set_torque(100)
    controller.update(model.opt.timestep)
    for old, actual in zip(before, (data.qpos, data.qvel, data.mocap_pos)):
        np.testing.assert_array_equal(old, actual)
    initial = controller.motor_position_rad
    torques = []
    for _ in range(round(.015 / model.opt.timestep)):
        controller.update(model.opt.timestep)
        mujoco.mj_step(model, data)
        torques.append(controller.motor_torque_nm)
    assert controller.motor_position_rad > initial + .001
    assert max(abs(value) for value in torques) <= controller.config.torque_cap_nm + 1e-9
    controller.set_enabled(False)
    controller.update(model.opt.timestep)
    mujoco.mj_step(model, data)
    assert controller.motor_torque_nm == pytest.approx(0, abs=1e-12)
    assert not np.any(data.warning.number)


def test_zero_speed_and_force_select_firmware_defaults(rig):
    _, _, controller = rig
    controller.command(100, speed_pct=0, force_pct=0)
    assert controller.torque_cap_nm == pytest.approx(.75)
    assert controller.speed_fraction == 1


@pytest.mark.parametrize("field", ["position_pct", "speed_pct", "force_pct"])
def test_invalid_commands_are_rejected(rig, field):
    _, data, controller = rig
    command = dict(position_pct=100, speed_pct=100, force_pct=30)
    command[field] = float("nan")
    with pytest.raises(ValueError):
        controller.command(**command)
    assert data.time == 0
