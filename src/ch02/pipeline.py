"""Normalized DataLoader pipeline — the Chapter 3 export contract.

The three functions Chapter 3 imports directly from ``ch02``:

- ``make_pickplace_dataloader(dataset_id, batch_size, shuffle)`` →
  ``(DataLoader, stats)``; the API Chapter 3 calls.
- ``normalize(x, stats, key)`` — z-score normalize a state or action
  tensor using precomputed stats.
- ``denormalize(x, stats, key)`` — inverse, used to recover
  environment-scale predictions before ``env.step``.

Plus ``compute_stats(dataset)`` (Listing 2.9, type-along) which
iterates the dataset once and returns per-feature mean / std / min /
max for ``observation.state`` and ``action``.

These signatures are frozen. Renaming or re-ordering breaks
Chapter 3 and every downstream chapter.
"""

import torch
from torch.utils.data import DataLoader

from ch02.dataset import DEFAULT_DATASET_ID, load_dataset

StatsDict = dict[str, dict[str, torch.Tensor]]


def compute_stats(dataset) -> StatsDict:
    """Per-feature mean / std / min / max for state and action.

    Iterates the entire dataset once. On the real LeRobot dataset
    this can take a minute or two because each ``dataset[i]`` triggers
    video decode for the unused image streams — accept the cost or
    use LeRobot's precomputed ``meta/stats.json`` for repeated runs.
    """
    states, actions = [], []
    for i in range(len(dataset)):
        frame = dataset[i]
        states.append(frame["observation.state"])
        actions.append(frame["action"])
    states = torch.stack(states)
    actions = torch.stack(actions)
    return {
        "observation.state": {
            "mean": states.mean(0),
            "std": states.std(0),
            "min": states.min(0).values,
            "max": states.max(0).values,
        },
        "action": {
            "mean": actions.mean(0),
            "std": actions.std(0),
            "min": actions.min(0).values,
            "max": actions.max(0).values,
        },
    }


def normalize(
    x: torch.Tensor, stats: StatsDict, key: str,
) -> torch.Tensor:
    """Z-score normalize ``x`` using ``stats[key]``."""
    return (x - stats[key]["mean"]) / (stats[key]["std"] + 1e-8)


def denormalize(
    x: torch.Tensor, stats: StatsDict, key: str,
) -> torch.Tensor:
    """Inverse z-score; recovers environment-scale values."""
    return x * (stats[key]["std"] + 1e-8) + stats[key]["mean"]


def _build_collate_fn(stats: StatsDict):
    """Closure over `stats` for a single-process DataLoader (num_workers=0).

    Not picklable — switch to a class with ``__call__`` if you ever
    raise the worker count (which would also require torch + closure
    re-entrancy thinking on Colab).
    """

    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            vals = [b[key] for b in batch]
            first = vals[0]
            if isinstance(first, str):
                out[key] = vals
                continue
            if not isinstance(first, torch.Tensor):
                out[key] = torch.tensor(vals)
                continue
            stacked = torch.stack(vals)
            if key in stats:
                stacked = normalize(stacked, stats, key)
            elif key.startswith("observation.images"):
                # LeRobot 0.5.x normally returns images as float32 in
                # [0, 1]; handle the legacy uint8 path defensively and
                # clamp to enforce the Ch3 [0, 1] image contract.
                if stacked.dtype == torch.uint8:
                    stacked = stacked.float() / 255.0
                if stacked.dtype.is_floating_point:
                    stacked = stacked.clamp(0.0, 1.0)
            out[key] = stacked
        return out

    return collate_fn


def _stats_from_meta(dataset) -> StatsDict | None:
    """Return LeRobot's precomputed `meta/stats.json`, or None if absent."""
    meta_stats = getattr(getattr(dataset, "meta", None), "stats", None)
    if meta_stats is None:
        return None
    if not all(k in meta_stats for k in ("observation.state", "action")):
        return None
    return {k: meta_stats[k] for k in ("observation.state", "action")}


def make_pickplace_dataloader(
    dataset_id: str = DEFAULT_DATASET_ID,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tuple[DataLoader, StatsDict]:
    """Return a normalized DataLoader and stats for the SO-101 task.

    The Ch3 contract. Do not rename or reorder these parameters.
    ``dataset_id`` is the first positional so later chapters can swap
    in custom datasets without changing the call shape.

    Stats come from LeRobot's precomputed `meta/stats.json` when
    available (the §2.5.3 reveal — saves 60-90s per fresh kernel on the
    real dataset). Falls back to `compute_stats` (Listing 2.9) for
    datasets without it.
    """
    dataset = load_dataset(dataset_id)
    stats = _stats_from_meta(dataset) or compute_stats(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_build_collate_fn(stats),
        num_workers=0,
    )
    return loader, stats
