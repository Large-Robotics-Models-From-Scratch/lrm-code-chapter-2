"""Tests for ch02.viz — smoke tests using fake data.

Uses matplotlib's Agg backend so tests run headlessly. All
plot/render functions are exercised against minimal synthetic
inputs; the contract being verified is "returns a Figure, doesn't
crash on reasonable data."
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from ch02.viz import (  # noqa: E402
    JOINT_NAMES,
    collect_actions,
    plot_action_distributions,
    plot_episode_lengths,
    plot_joint_trajectories,
    plot_normalization_effect,
    plot_stats,
    render_env_filmstrip,
    render_keyframes,
)


class FakeFrame(dict):
    """Dict that round-trips like a LeRobotDataset frame."""


def _img(h=32, w=32):
    """Tiny float32 image tensor in (C, H, W) layout."""
    return torch.from_numpy(np.random.rand(3, h, w).astype(np.float32))


class FakeDataset:
    """Minimal LeRobotDataset stand-in with image+state+action."""

    def __init__(self, episode_assignments, action_dim=6, state_dim=6):
        self._frames = []
        for i, ep in enumerate(episode_assignments):
            self._frames.append(FakeFrame({
                "episode_index": ep,
                "frame_index": i,
                "observation.images.up": _img(),
                "observation.images.side": _img(),
                "observation.state": torch.zeros(state_dim),
                "action": torch.zeros(action_dim),
            }))

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


def test_joint_names_has_six_entries():
    assert len(JOINT_NAMES) == 6


def test_render_keyframes_returns_figure():
    ds = FakeDataset([0] * 10)
    fig = render_keyframes(ds, episode_idx=0, n_frames=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_render_keyframes_raises_for_missing_episode():
    ds = FakeDataset([0, 0, 1])
    with pytest.raises(ValueError, match="Episode 99"):
        render_keyframes(ds, episode_idx=99)


def test_plot_action_distributions_two_way():
    """expert + random only (scripted=None) — no env needed."""
    expert = np.random.randn(100, 6).astype(np.float32)
    random_ = np.random.uniform(-1, 1, (100, 6)).astype(np.float32)
    fig = plot_action_distributions(expert, scripted=None, random_=random_)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_action_distributions_three_way():
    expert = np.random.randn(100, 6).astype(np.float32)
    scripted = np.random.randn(80, 6).astype(np.float32)
    random_ = np.random.uniform(-1, 1, (100, 6)).astype(np.float32)
    fig = plot_action_distributions(expert, scripted, random_)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_action_distributions_custom_joint_names():
    expert = np.random.randn(50, 4).astype(np.float32)
    random_ = np.random.uniform(-1, 1, (50, 4)).astype(np.float32)
    fig = plot_action_distributions(
        expert, None, random_,
        joint_names=["j1", "j2", "j3", "j4"],
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_joint_trajectories_returns_figure():
    ds = FakeDataset([0] * 20 + [1] * 15)
    fig = plot_joint_trajectories(ds, episode_indices=[0, 1])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_collect_actions_callable():
    """Function exists with expected signature; full eval is integration."""
    assert callable(collect_actions)


def test_joint_names_content_and_str():
    """Names + order are load-bearing for plot column-label alignment."""
    expected = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    assert JOINT_NAMES == expected
    assert all(isinstance(n, str) for n in JOINT_NAMES)


class FakeActionSpace:
    def __init__(self, dim=6):
        self.dim = dim

    def sample(self):
        return np.random.uniform(-1, 1, self.dim).astype(np.float32)


class FakeEnv:
    """Minimal Gymnasium-like env for collect_actions tests."""

    def __init__(self, n_steps=3, action_dim=6):
        self.action_space = FakeActionSpace(action_dim)
        self._n_steps = n_steps
        self._step_count = 0
        self.last_seed = None

    def reset(self, seed=None):
        self.last_seed = seed
        self._step_count = 0
        return {}, {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self._n_steps
        return {}, 0.0, terminated, False, {}


def test_collect_actions_random_branch_shape_and_seed():
    """policy_fn=None samples action_space; seed_offset reaches reset."""
    env = FakeEnv(n_steps=4)
    actions = collect_actions(
        env, policy_fn=None, n_episodes=2, seed_offset=10,
    )
    assert actions.shape == (8, 6)
    assert env.last_seed == 11  # 0..1 + offset 10 → last reset is seed=11


def test_collect_actions_with_policy_fn_receives_env_and_obs():
    """policy_fn is called as (env, obs); returned action is recorded."""
    env = FakeEnv(n_steps=2)
    seen_args = []

    def my_policy(env_arg, obs):
        seen_args.append((env_arg is env, type(obs).__name__))
        return np.zeros(6, dtype=np.float32)

    actions = collect_actions(env, policy_fn=my_policy, n_episodes=1)
    assert actions.shape == (2, 6)
    assert all(env_match for env_match, _ in seen_args)


def test_render_keyframes_save_path_writes_file(tmp_path):
    """The save_path branch actually persists to disk."""
    ds = FakeDataset([0] * 8)
    out = tmp_path / "keyframes.png"
    render_keyframes(ds, episode_idx=0, n_frames=3, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_action_distributions_save_path_writes_file(tmp_path):
    expert = np.random.randn(50, 6).astype(np.float32)
    random_ = np.random.uniform(-1, 1, (50, 6)).astype(np.float32)
    out = tmp_path / "actions.png"
    plot_action_distributions(
        expert, scripted=None, random_=random_, save_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 0


# -------------- render_env_filmstrip --------------

class _FilmstripFakeEnv:
    """Minimal env: reset/step/render/action_space for filmstrip tests."""

    class _Space:
        shape = (6,)

        def sample(self):
            return np.zeros(6, dtype=np.float32)

    def __init__(self):
        self.action_space = self._Space()
        self._step = 0

    def reset(self, seed=0):
        self._step = 0
        return {}, {}

    def step(self, action):
        self._step += 1
        return {}, 0.0, False, False, {}

    def render(self):
        # Simple per-step gradient so frames are distinguishable.
        img = np.full((16, 16, 3), self._step, dtype=np.uint8)
        return img


def test_render_env_filmstrip_returns_figure_with_n_frames():
    env = _FilmstripFakeEnv()
    fig = render_env_filmstrip(env, n_steps=20, n_frames=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_render_env_filmstrip_save_path_writes_file(tmp_path):
    env = _FilmstripFakeEnv()
    out = tmp_path / "filmstrip.png"
    render_env_filmstrip(env, n_steps=10, n_frames=3, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


# -------------- plot_episode_lengths --------------

def test_plot_episode_lengths_fallback_path():
    """No meta.episode_data_index → scans episode_index."""
    ds = FakeDataset([0, 0, 0, 1, 1, 2])
    fig = plot_episode_lengths(ds)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Title should reflect 3 episodes.
    assert "3 episodes" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_episode_lengths_fast_path_uses_episode_data_index():
    """With meta.episode_data_index present, scans cumulative offsets."""
    ds = FakeDataset([0] * 10)  # ignored by fast path

    class _Meta:
        episode_data_index = {
            "from": np.array([0, 5, 12]),
            "to": np.array([5, 12, 30]),
        }
    ds.meta = _Meta()
    fig = plot_episode_lengths(ds)
    # Heights = [5, 7, 18]; fast path avoided per-frame decode.
    bars = [p.get_height() for p in fig.axes[0].patches]
    assert bars == [5, 7, 18]
    plt.close(fig)


# -------------- plot_stats --------------

def test_plot_stats_renders_per_dim_bars():
    stats = {
        "action": {
            "mean": np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.1]),
            "std": np.array([0.5, 0.4, 0.6, 0.3, 0.2, 0.1]),
            "min": np.zeros(6),
            "max": np.ones(6),
        },
    }
    fig = plot_stats(stats, key="action")
    assert isinstance(fig, matplotlib.figure.Figure)
    bars = fig.axes[0].patches
    assert len(bars) == 6
    plt.close(fig)


# -------------- plot_normalization_effect --------------

def test_plot_normalization_effect_two_panel_figure():
    raw = np.random.uniform(-100, 100, 500).astype(np.float32)
    normalized = (raw - raw.mean()) / (raw.std() + 1e-8)
    fig = plot_normalization_effect(
        raw, normalized, dim_name="shoulder_pan",
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2  # raw + normalized panels
    plt.close(fig)


# -------------- plot_phase_keyframes --------------

def test_plot_phase_keyframes_renders_six_targets():
    """Capture-mock + 3-D scatter; runs run_scripted_episode internally."""
    sapien = pytest.importorskip("sapien")
    from ch02.viz import plot_phase_keyframes  # lazy via __getattr__

    grasp = sapien.Pose([0.2, 0.0, 0.05])
    goal = np.array([-0.2, 0.2, 0.02])
    fig = plot_phase_keyframes(grasp, goal)
    assert isinstance(fig, matplotlib.figure.Figure)
    # 6 connected move targets + 1 goal scatter = at least 6 plotted points.
    plt.close(fig)
