"""SO-100 pick-and-place environment + random-agent baseline.

Wraps ManiSkill3's PickCubeSO100-v1 task in a thin factory so the
notebook, tests, and the scripted-agent eval all construct the env
the same way. The random-agent loop is Listing 2.2 from the chapter
and establishes the performance floor every learned policy must clear.
"""

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers PickCubeSO100-v1
import numpy as np


def make_env(
    obs_mode: str = "rgb",
    control_mode: str = "pd_joint_delta_pos",
    render_mode: str = "rgb_array",
) -> gym.Env:
    """Construct the PickCubeSO100-v1 environment.

    obs_mode is one of "state", "rgb", "rgbd", or "state_dict".
    Use "state" for CPU-only local development where Vulkan is
    unavailable; the chapter's notebook uses "rgb". Seeding is the
    caller's responsibility via `env.reset(seed=...)`.
    """
    return gym.make(
        "PickCubeSO100-v1",
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )


def _episode_success(info: dict) -> bool:
    """Read the success flag out of an env step's info dict.

    ManiSkill envs typically expose `success`; gym-lowcostrobot used
    `is_success`. We check both so the function works across
    embodiments without forcing callers to know which.
    """
    if "success" in info:
        return bool(info["success"])
    if "is_success" in info:
        return bool(info["is_success"])
    return False


def run_random_agent(
    env: gym.Env,
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> tuple[float, float]:
    """Run uniformly random actions; return (success_rate, mean_return)."""
    successes, returns = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + seed_offset)
        ep_return = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated) or bool(truncated)
        successes += int(_episode_success(info))
        returns.append(ep_return)
    return successes / n_episodes, float(np.mean(returns))
