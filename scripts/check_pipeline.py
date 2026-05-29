#!/usr/bin/env python
"""Reader-style end-to-end check for the Ch3 export contract (PR 6).

Run from the repo root:

    python scripts/check_pipeline.py

Loads the dataset, builds the normalized DataLoader, draws one batch,
and verifies the contract: state has ~zero mean across the batch,
image pixels are in [0, 1], denormalize is the exact inverse.
"""

import sys

try:
    from ch02 import (
        denormalize,
        make_pickplace_dataloader,
        normalize,
    )
    from ch02.dataset import DEFAULT_DATASET_ID, load_dataset
    from ch02.pipeline import compute_stats
except ImportError as _exc:
    print(f"FAIL: import error: {_exc}")
    print('hint: did you `pip install -e ".[dev,data,sim]"`?')
    sys.exit(1)

import torch  # noqa: E402


def main() -> int:
    print("=== PR 6 pipeline smoke check ===\n")

    print("[1/4] Loading dataset + computing stats ...")
    print("  (full-dataset iteration; expect several minutes)")
    try:
        dataset = load_dataset()
        stats = compute_stats(dataset)
    except Exception as exc:
        print(f"  FAIL: load/stats error: {exc}")
        return 1
    print(
        f"  ok — stats keys: {list(stats.keys())}, "
        f"state mean shape: {stats['observation.state']['mean'].shape}"
    )

    print("[2/4] Round-trip normalize/denormalize on a sample ...")
    try:
        sample = dataset[0]["observation.state"]
        recovered = denormalize(
            normalize(sample, stats, "observation.state"),
            stats, "observation.state",
        )
        max_err = (recovered - sample).abs().max().item()
        ok = torch.allclose(sample, recovered, atol=1e-5)
    except Exception as exc:
        print(f"  FAIL: round-trip error: {exc}")
        return 1
    print(f"  max abs error: {max_err:.2e}, allclose: {ok}")
    if not ok:
        return 1

    print("[3/4] Building DataLoader (uses default dataset_id) ...")
    print(f"  dataset_id: {DEFAULT_DATASET_ID}")
    try:
        loader, _stats2 = make_pickplace_dataloader(batch_size=4)
    except Exception as exc:
        print(f"  FAIL: dataloader build error: {exc}")
        return 1
    print(f"  ok — loader has {len(loader)} batches at batch_size=4")

    print("[4/4] Drawing one batch and checking the contract ...")
    try:
        batch = next(iter(loader))
        state = batch["observation.state"]
        action = batch["action"]
        img_up = batch["observation.images.up"]
        print(f"  state: shape={tuple(state.shape)}, "
              f"mean per dim={state.mean(0).numpy().round(3)}")
        print(f"  action: shape={tuple(action.shape)}, "
              f"mean per dim={action.mean(0).numpy().round(3)}")
        print(
            f"  image range: [{img_up.min().item():.2f}, "
            f"{img_up.max().item():.2f}], "
            f"dtype={img_up.dtype}"
        )
    except Exception as exc:
        print(f"  FAIL: batch read error: {exc}")
        return 1

    print("\n=== PASS — Ch3 export contract works end-to-end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
