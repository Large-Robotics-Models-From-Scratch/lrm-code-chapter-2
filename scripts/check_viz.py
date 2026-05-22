#!/usr/bin/env python
"""Reader-style end-to-end check for the viz helpers (PR 5).

Run from the repo root:

    python scripts/check_viz.py

Loads the dataset, renders keyframes from episode 0, plots joint
trajectories from the first three episodes, and saves two PNGs
to /tmp/. Skips the scripted-action collection (needs Vulkan/sim).
"""

import sys

try:
    from ch02.dataset import load_dataset
    from ch02.viz import (
        collect_expert_actions,
        plot_joint_trajectories,
        render_keyframes,
    )
except ImportError as _exc:
    print(f"FAIL: import error: {_exc}")
    print('hint: did you `pip install -e ".[dev,data,sim]"`?')
    sys.exit(1)


KEYFRAMES_OUT = "/tmp/figure_2_4_expert_keyframes.png"
TRAJECTORIES_OUT = "/tmp/figure_2_6_joint_trajectories.png"


def main() -> int:
    print("=== PR 5 viz smoke check ===\n")

    print("[1/4] Loading dataset ...")
    try:
        dataset = load_dataset()
    except Exception as exc:
        print(f"  FAIL: dataset load error: {exc}")
        return 1
    print(f"  ok — {len(dataset)} frames, {dataset.num_episodes} episodes")

    print("[2/4] Stacking all expert actions ...")
    try:
        expert = collect_expert_actions(dataset)
    except Exception as exc:
        print(f"  FAIL: collect_expert_actions error: {exc}")
        return 1
    print(f"  expert actions shape: {expert.shape}, dtype: {expert.dtype}")

    print("[3/4] Rendering keyframes from episode 0 ...")
    try:
        render_keyframes(
            dataset, episode_idx=0, n_frames=6,
            save_path=KEYFRAMES_OUT,
        )
    except Exception as exc:
        print(f"  FAIL: render_keyframes error: {exc}")
        return 1
    print(f"  saved to {KEYFRAMES_OUT}")

    print("[4/4] Plotting joint trajectories for episode 0 ...")
    print(
        "  (per-episode video decode is slow on the real dataset;"
        " using 1 episode to keep this script under a minute)"
    )
    try:
        plot_joint_trajectories(
            dataset, episode_indices=[0],
            save_path=TRAJECTORIES_OUT,
        )
    except Exception as exc:
        print(f"  FAIL: plot_joint_trajectories error: {exc}")
        return 1
    print(f"  saved to {TRAJECTORIES_OUT}")

    print("\n=== PASS — viz helpers run end-to-end ===")
    print(f"open {KEYFRAMES_OUT} and {TRAJECTORIES_OUT} to inspect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
