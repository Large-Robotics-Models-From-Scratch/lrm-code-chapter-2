"""Visualization helpers for Chapter 2 datasets and policies.

All `plot_*` and `render_*` functions return matplotlib Figures so
the caller can render inline and save in the same call. None of them
call ``plt.savefig`` as a side effect; the caller passes ``save_path``
when persistence is wanted.
"""

from collections.abc import Callable, Iterable

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def render_keyframes(
    dataset,
    episode_idx: int = 0,
    n_frames: int = 6,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Two-row filmstrip of up/side camera keyframes from one episode."""
    # Lazy import: ch02.dataset pulls lerobot which is in the [data]
    # extra; viz callers that supply their own dataset shouldn't be
    # forced to install lerobot just to plot.
    from ch02.dataset import episode_frames

    ep = episode_frames(dataset, episode_idx)
    if len(ep) == 0:
        raise ValueError(f"Episode {episode_idx} not in dataset")

    idxs = np.linspace(0, len(ep) - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
    for col, i in enumerate(idxs):
        up = ep[i]["observation.images.up"]
        side = ep[i]["observation.images.side"]
        axes[0, col].imshow(up.permute(1, 2, 0).numpy())
        axes[0, col].set_title(f"step {i}", fontsize=10)
        axes[1, col].imshow(side.permute(1, 2, 0).numpy())
        for r in (0, 1):
            axes[r, col].axis("off")
    axes[0, 0].set_ylabel("up view", fontsize=10)
    axes[1, 0].set_ylabel("side view", fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def collect_actions(
    env,
    policy_fn: Callable | None = None,
    n_episodes: int = 5,
    seed_offset: int = 200,
) -> np.ndarray:
    """Run a step-level policy and stack the actions it issues.

    ``policy_fn=None`` means uniform sampling from ``env.action_space``.
    For the trajectory-level scripted policy (motion-planner based),
    use ``collect_scripted_actions`` instead.
    """
    actions = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            if policy_fn is None:
                action = env.action_space.sample()
            else:
                action = policy_fn(env, obs)
            actions.append(np.asarray(action).copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated) or bool(truncated)
    return np.array(actions)


def collect_scripted_actions(
    env,
    n_episodes: int = 5,
) -> np.ndarray:
    """Capture the joint commands the scripted policy issues.

    Wraps ``env.unwrapped.step`` to record each action without
    modifying its return value, then runs ``run_scripted_agent``.
    Restores the original ``step`` even if the run errors out.
    """
    # Lazy import: ch02.scripted pulls the ManiSkill motion planner
    # (sim extra); viz callers using only the plot helpers shouldn't
    # be forced to install the sim stack.
    from ch02.scripted import run_scripted_agent

    captured: list[np.ndarray] = []
    unwrapped = env.unwrapped
    original_step = unwrapped.step

    def capture_step(action):
        captured.append(np.asarray(action).copy())
        return original_step(action)

    unwrapped.step = capture_step
    try:
        run_scripted_agent(env, n_episodes=n_episodes)
    finally:
        unwrapped.step = original_step
    return np.array(captured)


def collect_expert_actions(dataset) -> np.ndarray:
    """Stack actions from every frame in the dataset."""
    return np.stack(
        [np.asarray(dataset[i]["action"]) for i in range(len(dataset))]
    )


def plot_action_distributions(
    expert: np.ndarray,
    scripted: np.ndarray | None,
    random_: np.ndarray,
    save_path: str | None = None,
    joint_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Overlay per-dimension histograms of expert / scripted / random.

    ``scripted=None`` skips that source. Useful when the scripted
    policy can't be run (no Vulkan, planner segfault) but expert and
    random comparison is still wanted.
    """
    names = joint_names or JOINT_NAMES
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    sources = [("expert", expert), ("random", random_)]
    if scripted is not None:
        sources.insert(1, ("scripted", scripted))
    for j, name in enumerate(names):
        ax = axes.flat[j]
        for label, arr in sources:
            ax.hist(
                arr[:, j], bins=40, alpha=0.5,
                label=label, density=True,
            )
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_joint_trajectories(
    dataset,
    episode_indices: Iterable[int] = range(5),
    save_path: str | None = None,
    joint_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Per-joint angle over time, overlaid across episodes.

    Note: iterates each episode via ``episode_frames`` which decodes
    full image tensors per frame even though only ``observation.state``
    is plotted. Slow on the real LeRobot dataset (minutes for 5
    episodes); consider passing a sliced/state-only dataset variant
    for speed.
    """
    # Lazy import — same reason as render_keyframes.
    from ch02.dataset import episode_frames

    names = joint_names or JOINT_NAMES
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    for ep_idx in episode_indices:
        ep = episode_frames(dataset, ep_idx)
        states = np.stack(
            [np.asarray(f["observation.state"]) for f in ep]
        )
        for j in range(n):
            axes.flat[j].plot(
                states[:, j], alpha=0.5, label=f"ep {ep_idx}"
            )
    for j, name in enumerate(names):
        axes.flat[j].set_title(name, fontsize=10)
        axes.flat[j].set_xlabel("step")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
