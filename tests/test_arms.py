"""Both supported arms carry a free object using bounded joint actuators."""
import mujoco
import numpy as np
import pytest

from fastgripper_sim.arm_demo import ArmDemo, run_demo


@pytest.mark.parametrize("arm", ["ur5e", "ur3e"])
def test_mount_is_rigid_and_object_is_free(arm):
    demo = ArmDemo(arm=arm, seed=8, randomize=True)
    assert demo.model.nmocap == 0
    assert demo.model.nu == 7
    assert demo.model.body("gripper_mount").jntnum[0] == 0
    assert demo.model.joint("block_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert mujoco.mj_name2id(demo.model, mujoco.mjtObj.mjOBJ_EQUALITY, "fixture_weld") == -1


@pytest.mark.parametrize("arm", ["ur5e", "ur3e"])
def test_arm_grasp_lift_and_release(arm):
    result = run_demo(arm=arm, fps=10)
    assert result["success"]
    assert result["warning_count"] == 0
    assert result["minimum_raised_hold_rise_m"] > .09
    assert abs(result["final"]["object_rise_m"]) < .002
    assert result["max_closure_error_m"] < 1e-4
    assert result["max_flange_position_error_m"] < .003
    limits = np.array(result["arm_actuator_force_limits_nm"])[:, 1]
    assert np.all(np.array(result["max_abs_arm_torque_nm"]) <= limits)
    assert all(np.isfinite(sample["arm_position_rad"]).all() for sample in result["samples"])
