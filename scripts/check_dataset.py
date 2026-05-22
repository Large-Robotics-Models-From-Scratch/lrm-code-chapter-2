#!/usr/bin/env python
"""Reader-style end-to-end check for the LeRobot dataset (PR 4).

Run from the repo root:

    python scripts/check_dataset.py

Downloads ``lerobot/svla_so101_pickplace`` on first run (cached
afterwards), reports total frames + episode count + feature schema,
and inspects frame 0's tensor shapes. Exits non-zero on any failure.
"""

import sys

try:
    from ch02.dataset import (
        DEFAULT_DATASET_ID,
        episode_frames,
        load_dataset,
    )
except ImportError as _exc:
    print(f"FAIL: import error: {_exc}")
    print('hint: did you `pip install -e ".[dev,data,sim]"`?')
    sys.exit(1)


def main() -> int:
    print("=== PR 4 dataset smoke check ===\n")

    print(f"[1/3] Loading {DEFAULT_DATASET_ID} ...")
    print("  (first run downloads from HF Hub; later runs hit cache)")
    try:
        dataset = load_dataset()
    except Exception as exc:
        print(f"  FAIL: dataset load error: {exc}")
        return 1
    n_frames = len(dataset)
    n_episodes = dataset.num_episodes
    print(f"  ok — {n_frames} frames across {n_episodes} episodes")

    print("[2/3] Inspecting frame 0 schema ...")
    try:
        frame = dataset[0]
        for key, val in frame.items():
            shape = getattr(val, "shape", None)
            dtype = getattr(val, "dtype", None)
            if shape is not None:
                print(f"  {key}: shape={tuple(shape)}, dtype={dtype}")
            else:
                print(f"  {key}: {val}")
    except Exception as exc:
        print(f"  FAIL: frame inspect error: {exc}")
        return 1

    print("[3/3] Counting frames in episode 0 ...")
    try:
        ep0 = episode_frames(dataset, 0)
    except Exception as exc:
        print(f"  FAIL: episode_frames error: {exc}")
        return 1
    print(f"  episode 0 has {len(ep0)} frames")

    print("\n=== PASS — dataset loads + indexes cleanly ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
