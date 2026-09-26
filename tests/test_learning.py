"""Optional learning extra: repeatable resets, genuine grasp and failure controls."""
import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="Install the learn extra for learning checks")

from fastgripper_sim.env import GripLiftEnv
from fastgripper_sim.evaluate import run_episode


def test_seeded_reset_and_gym_contract():
    from gymnasium.utils.env_checker import check_env

    env = GripLiftEnv()
    try:
        first, info = env.reset(seed=817)
        positions = env.data.qpos.copy()
        env.step([1, .25])
        second, repeated_info = env.reset(seed=817)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(positions, env.data.qpos)
        assert info == repeated_info
        assert env.data.time == 0
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_grasp_and_failure_controls():
    env = GripLiftEnv()
    try:
        good = run_episode(env, lambda _: np.asarray([1, .25]), 9201)
        opened = run_episode(env, lambda _: np.asarray([-1, .25]), 9201)
        weak = run_episode(env, lambda _: np.asarray([1, -1]), 9201)
        assert good["is_success"] and good["object_lift_m"] > .045
        assert not opened["is_success"] and not weak["is_success"]
    finally:
        env.close()


def test_bundled_policy_loads_and_grasps():
    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import SAC
    from fastgripper_sim.assets import example_path

    torch.set_num_threads(1)
    policy = SAC.load(example_path("policy.zip"), device="cpu")
    env = GripLiftEnv()
    try:
        result = run_episode(env, lambda obs: policy.predict(obs, deterministic=True)[0], 10000)
        assert result["is_success"]
        assert result["object_lift_m"] > .045
    finally:
        env.close()
