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
    """Construct the SO-101 pick-and-place dataset from HF Hub.

    Downloads parquet shards and video archives on first call;
    subsequent calls hit the local Hugging Face cache. Returns a
    ``LeRobotDataset`` whose ``__getitem__`` yields a per-frame dict
    of torch tensors (state, images, action) and scalar metadata
    (episode_index, frame_index, timestamp).
    """
    return LeRobotDataset(dataset_id)


def episode_frames(
    dataset: LeRobotDataset,
    episode_idx: int,
) -> list[dict]:
    """Return all frames belonging to one episode, in trajectory order.

    Indexes each frame exactly once. Callers that need many episodes
    are better off slicing via the dataset's episode metadata
    directly — this is the readable per-episode primitive used by
    Listing 2.6 and the keyframe-rendering helper in PR 5.
    """
    out: list[dict] = []
    for i in range(len(dataset)):
        frame = dataset[i]
        if int(frame["episode_index"]) == episode_idx:
            out.append(frame)
    return out
