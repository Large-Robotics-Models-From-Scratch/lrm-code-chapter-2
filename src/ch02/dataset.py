"""LeRobot dataset loader for the SO-101 pick-and-place expert data.

Wraps ``LeRobotDataset`` so the notebook, tests, and the downstream
pipeline all reference the same dataset ID by name. The chapter's
expected dataset is ``lerobot/svla_so101_pickplace`` — 50 episodes
of teleoperated SO-101 pick-and-place, 11,939 frames at 30 fps,
with `up` and `side` camera streams.

``episode_frames`` factors out the "all frames where
``episode_index == k``" pattern used by both Listing 2.6 and the
keyframe-rendering helper in PR 5.
"""

from lerobot.datasets import LeRobotDataset

DEFAULT_DATASET_ID: str = "lerobot/svla_so101_pickplace"


def load_dataset(dataset_id: str = DEFAULT_DATASET_ID) -> LeRobotDataset:
    """Load the SO-101 pick-and-place expert dataset from Hugging Face Hub.

    First call downloads parquet shards + video archives (~1.5 GB);
    subsequent calls hit the local HF cache.

    Args:
        dataset_id: HF Hub dataset ID. Defaults to the chapter's pin
            (`lerobot/svla_so101_pickplace`).

    Returns:
        LeRobotDataset; `len(ds)` is the total frame count, `ds[i]` is
        one frame (see `episode_frames` for the frame schema).
    """
    return LeRobotDataset(dataset_id)


def episode_frames(
    dataset: LeRobotDataset,
    episode_idx: int,
) -> list[dict]:
    """Return all frames of one episode, in trajectory order.

    A plain `list[dict]` matches LeRobotDataset's per-frame `__getitem__`
    return shape — each frame is already a dict — and lets callers iterate
    or index without learning a new wrapper. Costs one pass over the
    dataset; callers iterating many episodes should batch via the
    dataset's episode metadata instead.

    Args:
        dataset: LeRobotDataset from `load_dataset`.
        episode_idx: Zero-based episode index.

    Returns:
        Frames where `frame["episode_index"] == episode_idx`, ordered by
        `frame_index`. Each frame is a dict with these keys:
            observation.state: torch.float32 (6,) — joint positions
            observation.images.up: torch.uint8 (3, H, W) — top camera
            observation.images.side: torch.uint8 (3, H, W) — side camera
            action: torch.float32 (6,) — recorded teleop command
            episode_index, frame_index: int64 scalars
            timestamp: float32 — seconds from episode start
            next.done: bool — terminal-step flag
    """
    out: list[dict] = []
    for i in range(len(dataset)):
        frame = dataset[i]
        if int(frame["episode_index"]) == episode_idx:
            out.append(frame)
    return out
