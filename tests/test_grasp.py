"""Exercise an actual contact-driven grasp, lift, lower and release."""
import numpy as np
import pytest

from fastgripper_sim.demo import run_demo
from fastgripper_sim.models import model_path


@pytest.mark.parametrize("hardware", ["v1.0", "v1.1"])
def test_grasp_lift_release(hardware):
    result = run_demo(fps=10, model_path=model_path(hardware=hardware, scene=True))
    assert result["success"] and result["retained"] and result["released"]
    assert result["warning_count"] == 0
    assert result["peak_rise_m"] > .045
    assert result["max_closure_error_m"] < 5e-5
    assert abs(result["samples"][-1]["object_rise_m"]) < .001
    for sample in result["samples"]:
        assert abs(sample["motor_torque_nm"]) <= sample["torque_cap_nm"] + 1e-9
        assert np.isfinite(sample["aperture_m"])


@pytest.mark.parametrize("hardware", ["v1.0", "v1.1"])
def test_insufficient_motor_torque_cannot_pass_the_grasp(hardware):
    result = run_demo(force=5, fps=10, model_path=model_path(hardware=hardware, scene=True))
    assert not result["success"]
    assert not result["retained"]
    assert result["warning_count"] == 0
