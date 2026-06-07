#!/usr/bin/env python
"""Render Figure 2.3: PickCubeSO100 shaped-reward decomposition.

Run from the repo root:

    python scripts/generate_figure_2_3_reward.py

Rolls out the scripted IK policy until a successful episode is found,
records per-step state, decomposes the env's dense reward into its four
shaped components plus the success bonus, and writes a stacked-area
chart to figures/figure_2_3_pickcube_task.png.

The decomposition mirrors PickCubeEnv.compute_dense_reward exactly:

    reaching = 1 - tanh(5 * |tcp - cube|)
    reward   = reaching + is_grasped
    place    = (1 - tanh(5 * |cube - goal|)) * is_grasped
    static   = (1 - tanh(5 * |qvel|))       * is_obj_placed
    reward   = 5  (flat) on success

so the stacked components sum to the per-step reward the env returns.
"""

import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers PickCubeSO100-v1
    import matplotlib.pyplot as plt
    import numpy as np

    from ch02.scripted import (
        IKMotionPlanner,
        _compute_top_down_grasp_pose,
        run_scripted_episode,
    )
except ImportError as _exc:
    print(f"FAIL: import error: {_exc}")
    print('hint: did you `pip install -e ".[data,sim]"`?')
    sys.exit(1)

# Output lands in the repo's figures/ dir, created if missing.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "figures" / "figure_2_3_pickcube_task.png"
MAX_SEEDS = 40
DPI = 300

# Stacked component order + legend labels + colors (colorblind-safe).
_LABELS = [
    "reaching: 1 − tanh(5·‖tcp − cube‖)",
    "grasp bonus: +1 once grasped",
    "place: (1 − tanh(5·‖cube − goal‖)) · is_grasped",
    "static: (1 − tanh(5·‖qvel‖)) · is_obj_placed",
    "success: flat +5 once cube settles at goal",
]
_COLORS = ["#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756"]


def capture_states(env, seed):
    """Run the scripted policy with per-step state recording."""
    env.reset(seed=seed)
    states = []
    unwrapped = env.unwrapped
    original_step = unwrapped.step

    def record_step(action):
        result = original_step(action)
        tcp_pos = unwrapped.agent.tcp_pos
        if hasattr(tcp_pos, "cpu"):
            tcp_pos = tcp_pos.cpu().numpy()
        qvel = unwrapped.agent.robot.get_qvel()
        if hasattr(qvel, "cpu"):
            qvel = qvel.cpu().numpy()
        info = dict(unwrapped.evaluate())
        states.append({
            "tcp_pos": np.asarray(tcp_pos).reshape(-1)[:3].copy(),
            "cube_pos": np.asarray(unwrapped.cube.pose.sp.p).copy(),
            "goal_pos": np.asarray(unwrapped.goal_site.pose.sp.p).copy(),
            "qvel": np.asarray(qvel).reshape(-1)[:-1].copy(),
            "is_grasped": bool(info.get("is_grasped", False)),
            "is_obj_placed": bool(info.get("is_obj_placed", False)),
            "success": bool(info.get("success", False)),
        })
        return result

    unwrapped.step = record_step
    try:
        planner = IKMotionPlanner(env)
        grasp_pose = _compute_top_down_grasp_pose(env)
        goal_pos = unwrapped.goal_site.pose.sp.p
        run_scripted_episode(planner, grasp_pose, goal_pos)
    finally:
        unwrapped.step = original_step
    return states, dict(env.unwrapped.evaluate())


def decompose_reward(states):
    """Per-step reward components matching compute_dense_reward."""
    reach, grasp, place, static, success = [], [], [], [], []
    for s in states:
        if s["success"]:
            # Env overwrites all components with a flat 5 on success.
            reach.append(0.0)
            grasp.append(0.0)
            place.append(0.0)
            static.append(0.0)
            success.append(5.0)
            continue
        tcp_to_obj = float(np.linalg.norm(s["cube_pos"] - s["tcp_pos"]))
        is_grasped = float(s["is_grasped"])
        obj_to_goal = float(np.linalg.norm(s["goal_pos"] - s["cube_pos"]))
        qvel_norm = float(np.linalg.norm(s["qvel"]))
        reach.append(1.0 - np.tanh(5.0 * tcp_to_obj))
        grasp.append(is_grasped)
        place.append((1.0 - np.tanh(5.0 * obj_to_goal)) * is_grasped)
        static.append(
            (1.0 - np.tanh(5.0 * qvel_norm)) * float(s["is_obj_placed"])
        )
        success.append(0.0)
    return [np.array(c) for c in (reach, grasp, place, static, success)]


def first_step_where(states, key):
    """Index of the first state where `state[key]` is truthy, or None."""
    return next((i for i, s in enumerate(states) if s[key]), None)


def render(states, seed):
    components = decompose_reward(states)
    steps = np.arange(len(states))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(
        steps, *components, labels=_LABELS, colors=_COLORS,
        alpha=0.92, edgecolor="white", linewidth=0.3,
    )

    events = [
        (first_step_where(states, "is_grasped"), "grasp", _COLORS[1]),
        (first_step_where(states, "is_obj_placed"), "placed", _COLORS[3]),
        (first_step_where(states, "success"), "success", _COLORS[4]),
    ]
    y_top = ax.get_ylim()[1]
    for step_idx, label, color in events:
        if step_idx is None:
            continue
        ax.axvline(
            step_idx, color=color, linestyle="--", linewidth=1.0, alpha=0.6
        )
        ax.annotate(
            label, xy=(step_idx, y_top * 0.92),
            xytext=(4, 0), textcoords="offset points",
            fontsize=9, color=color, alpha=0.85,
        )

    ax.set_xlim(0, len(steps) - 1)
    ax.set_xlabel("environment step")
    ax.set_ylabel("per-step reward (stacked components)")
    ax.set_title(
        "PickCubeSO100 shaped reward across one scripted-policy episode "
        f"(seed {seed})"
    )
    ax.legend(
        loc="upper left", fontsize=8, framealpha=0.95,
        title="reward terms (sum = per-step reward)", title_fontsize=8,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    env = gym.make(
        "PickCubeSO100-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
    )

    print(f"Searching seeds [0, {MAX_SEEDS}) for a successful episode ...")
    chosen_seed, chosen_states = None, None
    for seed in range(MAX_SEEDS):
        states, final = capture_states(env, seed)
        if bool(final.get("success", False)):
            chosen_seed, chosen_states = seed, states
            print(f"  seed {seed} succeeded ({len(states)} steps)")
            break

    if chosen_states is None:
        print(
            f"  no success in [0, {MAX_SEEDS}); using seed 0 "
            "(figure will show approach + grasp only)"
        )
        chosen_seed = 0
        chosen_states, _ = capture_states(env, 0)

    render(chosen_states, chosen_seed)
    print(f"Saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
