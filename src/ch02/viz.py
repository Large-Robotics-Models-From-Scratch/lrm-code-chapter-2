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
    normalize_per_source: bool = True,
) -> matplotlib.figure.Figure:
    """Overlay per-dimension histograms of expert / scripted / random.

    The three sources live in different units — expert actions are
    degrees of joint position (LeRobot teleop convention, ~[-100, +100]),
    scripted actions are radians (`pd_joint_pos`, ~[-π, +π]), and random
    actions are normalized deltas (`pd_joint_delta_pos`, [-1, +1]). When
    `normalize_per_source=True` (the default) each source is min-max
    rescaled into [-1, +1] *independently* before plotting so the
    distribution *shapes* are visible on a common axis without pretending
    the absolute values are comparable.

    Pass `normalize_per_source=False` to plot raw values — useful only
    when you've already converted every source to the same unit.

    ``scripted=None`` skips that source. Useful when the scripted policy
    can't be run (no Vulkan, planner segfault) but expert and random
    comparison is still wanted.
    """
    names = joint_names or JOINT_NAMES
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    sources = [("expert", expert), ("random", random_)]
    if scripted is not None:
        sources.insert(1, ("scripted", scripted))

    def _rescale(arr: np.ndarray) -> np.ndarray:
        """Per-source, per-joint min-max → [-1, +1]; preserves shape."""
        if not normalize_per_source:
            return arr
        lo = arr.min(axis=0, keepdims=True)
        hi = arr.max(axis=0, keepdims=True)
        span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        return 2.0 * (arr - lo) / span - 1.0

    rescaled = [(label, _rescale(arr)) for label, arr in sources]
    xlabel = (
        "normalized action (per-source min-max → [-1, +1])"
        if normalize_per_source
        else "action value (raw units)"
    )

    for j, name in enumerate(names):
        ax = axes.flat[j]
        for label, arr in rescaled:
            ax.hist(
                arr[:, j], bins=40, alpha=0.5,
                label=label, density=True,
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=8)
    # Hide any empty cells if n_joints isn't a multiple of cols.
    for k in range(n, rows * cols):
        axes.flat[k].set_visible(False)
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

    Auto-resets on episode end (seed varied per episode) so the filmstrip
    spans the full step budget even when the env truncates early — useful
    for short-horizon tasks like PickCube (horizon ~50 steps). Panel titles
    include the episode index so the reader can see the resets even when
    per-frame motion is subtle.

    Args:
        env: Gymnasium env with `render_mode="rgb_array"` set.
        n_steps: Total step budget (may cover many episodes).
        n_frames: Number of frames to display in the filmstrip.
        action_fn: `action_fn(env, obs) -> action`. None → uniform random.
        seed: Base seed for `env.reset`; episode N uses `seed + N`.
        save_path: If set, also `savefig` at 300 dpi.

    Returns:
        1×n_frames matplotlib Figure with the rendered frames.
    """
    capture_idxs = np.linspace(0, n_steps - 1, n_frames, dtype=int).tolist()
    capture_at = set(capture_idxs)
    ep = 0
    ep_step = 0
    obs, _ = env.reset(seed=seed + ep)
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
            frames.append((ep, ep_step, np.asarray(arr).copy()))
        if action_fn is not None:
            action = action_fn(env, obs)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        ep_step += 1
        if bool(terminated) or bool(truncated):
            ep += 1
            ep_step = 0
            obs, _ = env.reset(seed=seed + ep)

    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]
    for ax, (ep_idx, ep_s, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(f"ep {ep_idx} · step {ep_s}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def record_env_video(
    env,
    n_steps: int = 200,
    action_fn: Callable | None = None,
    seed: int = 0,
    fps: int = 20,
    save_path: str | None = None,
):
    """Record a rollout as MP4 video, auto-resetting on episode end.

    Every step's `env.render()` frame is captured and written to an
    MP4 via imageio + ffmpeg. The default policy is uniform random;
    pass `action_fn` to swap in a scripted/learned controller.

    Args:
        env: Gymnasium env with `render_mode="rgb_array"` set.
        n_steps: Total step budget (may cover many episodes).
        action_fn: `action_fn(env, obs) -> action`. None → uniform random.
        seed: Base seed for `env.reset`; episode N uses `seed + N`.
        fps: Output video frame rate.
        save_path: Where to write the MP4. If None, a temp file is used.

    Returns:
        `IPython.display.Video` for inline notebook display. The `.filename`
        attribute holds the on-disk path.
    """
    import tempfile

    import imageio.v2 as imageio
    from IPython.display import Video

    if save_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        save_path = tf.name
        tf.close()

    ep = 0
    obs, _ = env.reset(seed=seed + ep)
    frames = []
    for _ in range(n_steps):
        img = env.render()
        if hasattr(img, "cpu"):
            arr = img.cpu().numpy()
        else:
            arr = np.asarray(img)
        if arr.ndim == 4:  # vectorized envs return (n_envs, H, W, 3)
            arr = arr[0]
        frames.append(np.asarray(arr, dtype=np.uint8).copy())
        if action_fn is not None:
            action = action_fn(env, obs)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated) or bool(truncated):
            ep += 1
            obs, _ = env.reset(seed=seed + ep)

    imageio.mimsave(save_path, frames, fps=fps, macro_block_size=1)
    return Video(save_path, embed=True)


def record_dataset_episode_video(
    dataset,
    episode_idx: int = 0,
    cameras: Iterable[str] = ("up", "side"),
    fps: int = 30,
    save_path: str | None = None,
):
    """Replay one LeRobot episode as MP4, stacking camera views.

    Each frame's `observation.images.<cam>` tensors (torch float32 in
    `[0, 1]`, layout `(3, H, W)`) are converted to uint8 RGB and
    horizontally concatenated across the requested cameras. Output frame
    rate defaults to 30 fps to match the SO-101 dataset's native cadence.

    Args:
        dataset: LeRobotDataset (frames must include the cameras requested).
        episode_idx: Zero-based episode index.
        cameras: Camera suffixes to stack left-to-right (e.g. `("up",)`
            for a single view or `("up", "side")` for both).
        fps: Output video frame rate.
        save_path: Where to write the MP4. If None, a temp file is used.

    Returns:
        `IPython.display.Video` for inline notebook display.
    """
    import tempfile

    import imageio.v2 as imageio
    from IPython.display import Video

    # Lazy on purpose: ch02.dataset → lerobot ([data] extra). See
    # render_keyframes for the same lazy-import rationale.
    from ch02.dataset import episode_frames

    if save_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        save_path = tf.name
        tf.close()

    ep = episode_frames(dataset, episode_idx)
    if len(ep) == 0:
        raise ValueError(f"Episode {episode_idx} not in dataset")

    cams = list(cameras)
    frames = []
    for frame in ep:
        panels = []
        for cam in cams:
            t = frame[f"observation.images.{cam}"]
            arr = t.permute(1, 2, 0).numpy()  # (H, W, 3) float32 in [0,1]
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            panels.append(arr)
        if len(panels) > 1:
            stacked = np.concatenate(panels, axis=1)
        else:
            stacked = panels[0]
        frames.append(stacked)

    imageio.mimsave(save_path, frames, fps=fps, macro_block_size=1)
    return Video(save_path, embed=True)


def record_scripted_episode_video(
    env,
    n_episodes: int = 1,
    fps: int = 20,
    save_path: str | None = None,
):
    """Record scripted-policy rollouts as MP4 by intercepting `env.step`.

    The motion planner emits actions inside `run_scripted_agent`, so we
    monkey-patch `env.unwrapped.step` to also call `env.render()` after
    each physics step and accumulate frames (same pattern as
    `capture_scripted_actions`). Restores the original `step` even on error.

    Requires `env` constructed with `render_mode="rgb_array"`.

    Args:
        env: Gymnasium env from `make_env(render_mode="rgb_array")`.
        n_episodes: Number of scripted rollouts to record.
        fps: Output video frame rate.
        save_path: Where to write the MP4. If None, a temp file is used.

    Returns:
        `IPython.display.Video` for inline notebook display.
    """
    import tempfile

    import imageio.v2 as imageio
    from IPython.display import Video

    # Lazy: scripted.py → ManiSkill ([sim] extra). Same reason as
    # capture_scripted_actions.
    from ch02.scripted import run_scripted_agent

    if save_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        save_path = tf.name
        tf.close()

    frames: list[np.ndarray] = []
    unwrapped = env.unwrapped
    original_step = unwrapped.step

    def capture_step(action):
        result = original_step(action)
        img = env.render()
        if hasattr(img, "cpu"):
            arr = img.cpu().numpy()
        else:
            arr = np.asarray(img)
        if arr.ndim == 4:
            arr = arr[0]
        frames.append(np.asarray(arr, dtype=np.uint8).copy())
        return result

    unwrapped.step = capture_step
    try:
        run_scripted_agent(env, n_episodes=n_episodes)
    finally:
        unwrapped.step = original_step

    imageio.mimsave(save_path, frames, fps=fps, macro_block_size=1)
    return Video(save_path, embed=True)


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

        def hold(self, n_steps=30):
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
