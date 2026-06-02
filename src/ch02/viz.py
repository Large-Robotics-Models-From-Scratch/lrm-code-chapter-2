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
    # Lazy on purpose: ch02.dataset → lerobot ([data] extra). Top-level
    # import here would make `import ch02.viz` require [data], breaking
    # consumers that plot using their own dataset object.
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
    """Run a step-level policy in `env` and stack the actions issued.

    Used by the §2.4 histogram (Listing 2.8) to gather the random branch.
    For the trajectory-level scripted policy (motion-planner based),
    use `capture_scripted_actions` instead (it intercepts `env.step`
    because the motion planner doesn't expose a per-step callable).

    Args:
        env: Gymnasium env from `make_env`.
        policy_fn: `policy_fn(env, obs) -> action`. `None` means uniform
            sampling from `env.action_space` — the random branch.
        n_episodes: Number of episodes to roll out.
        seed_offset: Added to the per-episode seed (default 200 keeps
            these episodes disjoint from `run_random_agent`'s seeds 0..N).

    Returns:
        `(n_steps, action_dim)` float32 array of every action issued.
    """
    actions = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        done = False
        while not done:
            if policy_fn is None:
                action = env.action_space.sample()  # random branch
            else:
                action = policy_fn(env, obs)
            actions.append(np.asarray(action).copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated) or bool(truncated)
    return np.array(actions)


def capture_scripted_actions(
    env,
    n_episodes: int = 5,
) -> np.ndarray:
    """Intercept `env.step` to record actions the scripted policy issues.

    The scripted policy runs through ManiSkill's motion planner, which
    emits actions inside `run_scripted_agent` rather than exposing them
    as a per-step callable — so we monkey-patch `env.unwrapped.step` for
    the duration of the run. Restores the original `step` even on error.

    Args:
        env: Gymnasium env from `make_env`.
        n_episodes: Number of episodes to roll out.

    Returns:
        `(n_steps, action_dim)` float32 array of every action the motion
        planner issued across the rollouts.
    """
    # Lazy on purpose: ch02.scripted → ManiSkill ([sim] extra). Top-level
    # import here would make `import ch02.viz` require [sim] just for the
    # plot helpers. Same reasoning as render_keyframes.
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
    # Lazy on purpose — same [data]-extra rationale as render_keyframes.
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


def render_env_filmstrip(
    env,
    n_steps: int = 200,
    n_frames: int = 6,
    action_fn: Callable | None = None,
    seed: int = 0,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Run env for `n_steps` and tile `n_frames` evenly-spaced RGB renders.

    Args:
        env: Gymnasium env with `render_mode="rgb_array"` set.
        n_steps: Total step budget for the rollout.
        n_frames: Number of frames to display in the filmstrip.
        action_fn: `action_fn(env, obs) -> action`. None → uniform random.
        seed: Seed for `env.reset` (deterministic filmstrip across runs).
        save_path: If set, also `savefig` at 300 dpi.

    Returns:
        1×n_frames matplotlib Figure with the rendered frames.
    """
    capture_idxs = np.linspace(0, n_steps - 1, n_frames, dtype=int).tolist()
    capture_at = set(capture_idxs)
    obs, _ = env.reset(seed=seed)
    frames = []
    for step in range(n_steps):
        if step in capture_at:
            img = env.render()
            if hasattr(img, "cpu"):
                arr = img.cpu().numpy()
            else:
                arr = np.asarray(img)
            if arr.ndim == 4:  # vectorized envs return (n_envs, H, W, 3)
                arr = arr[0]
            frames.append((step, arr))
        if action_fn is not None:
            action = action_fn(env, obs)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated) or bool(truncated):
            break

    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]
    for ax, (step, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(f"step {step}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_phase_keyframes(
    grasp_pose,
    goal_pos: np.ndarray,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """3-D scatter of scripted-policy keyframes from `run_scripted_episode`.

    Uses a capturing mock-planner to extract the (x, y, z) target of each
    keyframe without running any actual motion planning — works without
    `[sim]` extras.

    Args:
        grasp_pose: A `sapien.Pose` for the cube grasp moment.
        goal_pos: `(3,)` xyz drop point.
        save_path: If set, also `savefig` at 300 dpi.

    Returns:
        3-D matplotlib Figure showing the keyframe trajectory.
    """
    # Lazy: scripted.py uses numpy/sapien but lazy-loads mani_skill.
    from ch02.scripted import run_scripted_episode

    class _Capture:
        def __init__(self):
            self.targets: list[tuple[float, float, float]] = []
            self.labels: list[str] = []

        def move_to_pose_with_screw(self, pose):
            self.targets.append(tuple(pose.p))

        def close_gripper(self, gripper_state=None):
            pass

        def open_gripper(self):
            pass

    cap = _Capture()
    run_scripted_episode(cap, grasp_pose, goal_pos)
    pts = np.asarray(cap.targets)

    # 6 move_to_pose_with_screw calls map to: approach, descend, grasp,
    # lift, transport, place. (open is the 7th phase, not a target pose.)
    phase_labels = [
        "approach", "descend", "grasp",
        "lift", "transport", "place",
    ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="tab:blue")
    for i, label in enumerate(phase_labels[: len(pts)]):
        ax.text(pts[i, 0], pts[i, 1], pts[i, 2] + 0.01, label, fontsize=9)
    ax.scatter(*goal_pos, color="tab:red", s=80, label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Scripted policy keyframes (6 motion targets + release)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_episode_lengths(
    dataset,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Bar chart of per-episode frame counts across the dataset.

    Reads `dataset.episode_data_index` if available (fast path); falls
    back to scanning `episode_index` (slow on real LeRobot — see
    `episode_frames`).

    Args:
        dataset: LeRobotDataset (or any object with episode-grouped frames).
        save_path: If set, also `savefig` at 300 dpi.

    Returns:
        1-D matplotlib Figure: episode index on x, frame count on y.
    """
    # Fast path: LeRobot exposes `episode_data_index` (per-episode `from`/
    # `to` cumulative offsets); lengths = to - from. Skips per-frame decode.
    meta = getattr(dataset, "meta", None)
    edi = getattr(meta, "episode_data_index", None)
    if edi is not None and "from" in edi and "to" in edi:
        lengths = np.asarray(edi["to"]) - np.asarray(edi["from"])
    else:
        # Slow fallback: scan episode_index.
        from collections import Counter

        counts: Counter = Counter()
        for i in range(len(dataset)):
            counts[int(dataset[i]["episode_index"])] += 1
        lengths = np.array([counts[i] for i in sorted(counts)])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(lengths)), lengths, color="tab:blue", alpha=0.7)
    ax.set_xlabel("episode index")
    ax.set_ylabel("frames")
    ax.set_title(
        f"Episode lengths ({len(lengths)} episodes; "
        f"min={lengths.min()}, max={lengths.max()}, "
        f"mean={lengths.mean():.0f})"
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_stats(
    stats: dict,
    key: str = "action",
    save_path: str | None = None,
    joint_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Bar chart of per-dimension mean ± std for one feature in `stats`.

    Args:
        stats: StatsDict from `compute_stats` (or LeRobot's `meta.stats`).
        key: Which feature to plot — `"action"` or `"observation.state"`.
        save_path: If set, also `savefig` at 300 dpi.
        joint_names: Optional override for the x-axis labels.

    Returns:
        1-D matplotlib Figure with mean as a bar and std as an error bar.
    """
    means = np.asarray(stats[key]["mean"])
    stds = np.asarray(stats[key]["std"])
    names = joint_names or JOINT_NAMES[: len(means)]
    x = np.arange(len(means))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel(f"{key} (mean ± std)")
    ax.set_title(f"Per-dimension statistics for `{key}`")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_normalization_effect(
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    dim_name: str = "joint",
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Before/after histograms showing what z-score normalization did.

    Args:
        raw_values: 1-D array of raw values for one feature dimension.
        normalized_values: Corresponding normalized values (same length).
        dim_name: Label for the dimension (e.g., `"shoulder_pan"`).
        save_path: If set, also `savefig` at 300 dpi.

    Returns:
        1×2 matplotlib Figure: raw histogram on the left, normalized on
        the right; the normalized version centers at 0 with unit spread.
    """
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_raw.hist(raw_values, bins=50, color="tab:blue", alpha=0.7)
    ax_raw.set_title(f"Raw `{dim_name}` (mean={raw_values.mean():.2f}, "
                     f"std={raw_values.std():.2f})")
    ax_raw.set_xlabel("value")
    ax_norm.hist(normalized_values, bins=50, color="tab:orange", alpha=0.7)
    ax_norm.set_title(f"Normalized `{dim_name}` "
                      f"(mean={normalized_values.mean():.2f}, "
                      f"std={normalized_values.std():.2f})")
    ax_norm.set_xlabel("value")
    ax_norm.axvline(0, color="grey", linewidth=0.8)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
