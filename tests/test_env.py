"""Integration tests for ch02.env — needs ManiSkill + SAPIEN/Vulkan.

Marked integration; skipped in CI by default. Run locally with
`pytest -m integration` once the sim extra is installed and Vulkan
(software or hardware) is available.
"""

import pytest

pytest.importorskip("mani_skill")

from ch02.env import _episode_success, make_env, run_random_agent


def test_episode_success_reads_maniskill_key():
    """Helper reads ManiSkill's `success` key and returns a Python bool."""
    assert _episode_success({"success": True}) is True
    assert _episode_success({"success": False}) is False
    assert _episode_success({}) is False  # missing key → False


@pytest.mark.integration
def test_make_env_constructs():
    """Env constructs and reset returns a non-empty observation."""
    env = make_env(obs_mode="state", render_mode=None)
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
    finally:
        env.close()


@pytest.mark.integration
def test_random_agent_one_episode():
    """Random agent completes an episode and returns sane metrics."""
    env = make_env(obs_mode="state", render_mode=None)
    try:
        success_rate, mean_return = run_random_agent(env, n_episodes=1)
        assert 0.0 <= success_rate <= 1.0
        assert isinstance(mean_return, float)
    finally:
        env.close()
