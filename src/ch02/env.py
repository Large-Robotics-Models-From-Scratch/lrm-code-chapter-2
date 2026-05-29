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

    Args:
        obs_mode: "state", "rgb", "rgbd", or "state_dict". Use "state"
            for CPU-only local dev; the chapter's notebook uses "rgb".
        control_mode: ManiSkill controller; default is joint-space deltas.
        render_mode: Gymnasium render mode (use `None` to skip rendering).

    Returns:
        A Gymnasium env. Seed via `env.reset(seed=...)`.
    """
    return gym.make(
        "PickCubeSO100-v1",
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )


def _episode_success(info: dict) -> bool:
    """Coerce `info["success"]` (a (1,) torch.bool) to a Python bool.

    Args:
        info: ManiSkill step's info dict.

    Returns:
        True iff the episode reported success.
    """
    # TODO: return np.ndarray if env is ever built with num_envs > 1.
    return bool(info.get("success", False))


def run_random_agent(
    env: gym.Env,
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> tuple[float, float]:
    """Run uniformly random actions and report the performance floor.

    Performance floor only; §2.4 collects actions separately for viz.

    Args:
        env: A Gymnasium env from `make_env`.
        n_episodes: Number of episodes to roll out.
        seed_offset: Added to the per-episode seed (default → seeds 0..N-1).

    Returns:
        (success_rate, mean_return) — success_rate ∈ [0, 1]; mean_return
        is the per-episode sum of step rewards, averaged over `n_episodes`.
    """
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
