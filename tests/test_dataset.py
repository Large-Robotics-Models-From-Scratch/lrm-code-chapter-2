"""Tests for ch02.dataset — LeRobot dataset loading.

The non-integration tests use a fake dataset (lightweight dict-of-list
fixture) so we exercise ``episode_frames`` without downloading from
Hugging Face Hub. Integration tests that actually load
``lerobot/svla_so101_pickplace`` are marked ``@pytest.mark.integration``
and require network access plus the HF cache.
"""

import inspect

import numpy as np
import pytest

pytest.importorskip("lerobot")

from ch02.dataset import (  # noqa: E402
    DEFAULT_DATASET_ID,
    episode_frames,
    load_dataset,
)


class FakeDataset:
    """Minimal LeRobotDataset stand-in for unit tests."""

    def __init__(self, episode_assignments: list[int]):
        self._frames = [
            {"episode_index": ep, "frame_index": i, "data": i * 10}
            for i, ep in enumerate(episode_assignments)
        ]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


def test_default_dataset_id():
    assert DEFAULT_DATASET_ID == "lerobot/svla_so101_pickplace"


def test_load_dataset_is_callable():
    assert callable(load_dataset)


def test_episode_frames_returns_only_matching_episode():
    """Frames with the target episode_index are returned in order."""
    fake = FakeDataset([0, 0, 1, 0, 2, 1, 1])
    ep0 = episode_frames(fake, 0)
    assert len(ep0) == 3
    assert all(f["episode_index"] == 0 for f in ep0)
    assert [f["frame_index"] for f in ep0] == [0, 1, 3]


def test_episode_frames_for_missing_episode_is_empty():
    fake = FakeDataset([0, 0, 1])
    assert episode_frames(fake, 99) == []


def test_episode_frames_preserves_frame_order():
    """The 'in trajectory order' contract: frames come back as found."""
    fake = FakeDataset([2, 2, 2])
    ep2 = episode_frames(fake, 2)
    assert [f["frame_index"] for f in ep2] == [0, 1, 2]


def test_episode_frames_accepts_numpy_episode_index():
    """LeRobotDataset returns episode_index as torch/numpy scalar; the
    int() coercion in episode_frames keeps the comparison correct."""
    fake = FakeDataset([0, 0, 1])
    fake._frames[0]["episode_index"] = np.int64(0)
    fake._frames[1]["episode_index"] = np.int64(0)
    fake._frames[2]["episode_index"] = np.int64(1)
    ep0 = episode_frames(fake, 0)
    assert len(ep0) == 2


def test_episode_frames_empty_dataset_returns_empty():
    """Edge case: empty dataset yields empty list, no errors."""
    assert episode_frames(FakeDataset([]), 0) == []


def test_episode_frames_sparse_interleaved_episodes():
    """Order preserved when matches are non-contiguous."""
    fake = FakeDataset([0, 1, 0, 1, 0])
    ep0 = episode_frames(fake, 0)
    assert [f["frame_index"] for f in ep0] == [0, 2, 4]


def test_load_dataset_default_signature():
    """Default arg is DEFAULT_DATASET_ID — signature drift guard."""
    sig = inspect.signature(load_dataset)
    assert sig.parameters["dataset_id"].default == DEFAULT_DATASET_ID


@pytest.mark.integration
def test_load_dataset_real_hub_download():
    """Pull the actual dataset; verify schema and basic counts."""
    import torch

    dataset = load_dataset()
    assert len(dataset) > 0
    assert dataset.num_episodes > 0
    keys = list(dataset.features.keys())
    assert "observation.state" in keys
    assert "action" in keys
    assert "observation.images.up" in keys
    assert "observation.images.side" in keys

    frame = dataset[0]
    assert frame["observation.state"].shape == (6,)
    assert frame["observation.state"].dtype == torch.float32
    assert frame["action"].shape == (6,)
    assert frame["action"].dtype == torch.float32
