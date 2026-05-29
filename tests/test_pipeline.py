"""Tests for ch02.pipeline — the Chapter 3 export contract.

Heaviest test coverage of any module in ch02 because these
signatures (`make_pickplace_dataloader`, `normalize`, `denormalize`)
are what Chapter 3 imports. A silent drift here breaks every
downstream chapter.

FakeDataset returns torch tensors shaped like the real
`svla_so101_pickplace` schema so tests don't need lerobot's actual
DataLoader path.
"""

import inspect

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from ch02 import (  # noqa: E402
    denormalize,
    make_pickplace_dataloader,
    normalize,
)
from ch02.pipeline import (  # noqa: E402
    StatsDict,
    _build_collate_fn,
    compute_stats,
)


class FakeDataset:
    """Stand-in for LeRobotDataset with the svla_so101 schema."""

    def __init__(self, n_frames=20, state_dim=6, action_dim=6):
        torch.manual_seed(0)
        self.n_frames = n_frames
        self._frames = []
        for i in range(n_frames):
            self._frames.append({
                "observation.state": torch.randn(state_dim),
                "action": torch.randn(action_dim),
                "observation.images.up": torch.rand(3, 32, 32),
                "observation.images.side": torch.rand(3, 32, 32),
                "episode_index": torch.tensor(i // 5, dtype=torch.int64),
                "frame_index": torch.tensor(i % 5, dtype=torch.int64),
                "timestamp": torch.tensor(float(i) / 30.0),
                "task": "pink lego brick into the transparent box",
            })

    def __len__(self):
        return self.n_frames

    def __getitem__(self, i):
        return self._frames[i]


# -- contract: re-exports at package root -----------------------------

def test_normalize_reexported_at_package_root():
    from ch02 import normalize as nrm
    assert nrm is normalize


def test_denormalize_reexported_at_package_root():
    from ch02 import denormalize as dnrm
    assert dnrm is denormalize


def test_make_pickplace_dataloader_reexported_at_package_root():
    from ch02 import make_pickplace_dataloader as mpd
    assert mpd is make_pickplace_dataloader


def test_make_pickplace_dataloader_signature_is_frozen():
    """The three positional args are the Ch3 contract; do not reorder."""
    sig = inspect.signature(make_pickplace_dataloader)
    params = list(sig.parameters.keys())
    assert params == ["dataset_id", "batch_size", "shuffle"]
    assert sig.parameters["batch_size"].default == 64
    assert sig.parameters["shuffle"].default is True


# -- compute_stats --------------------------------------------------

def test_compute_stats_returns_expected_keys():
    stats = compute_stats(FakeDataset(n_frames=10))
    assert set(stats.keys()) == {"observation.state", "action"}
    for sub in stats.values():
        assert set(sub.keys()) == {"mean", "std", "min", "max"}


def test_compute_stats_shapes_match_feature_dims():
    stats = compute_stats(FakeDataset(n_frames=10))
    for key in ("observation.state", "action"):
        assert stats[key]["mean"].shape == (6,)
        assert stats[key]["std"].shape == (6,)
        assert stats[key]["min"].shape == (6,)
        assert stats[key]["max"].shape == (6,)


def test_compute_stats_mean_matches_manual_calculation():
    ds = FakeDataset(n_frames=10)
    stats = compute_stats(ds)
    expected_mean = torch.stack(
        [ds[i]["observation.state"] for i in range(len(ds))]
    ).mean(0)
    assert torch.allclose(stats["observation.state"]["mean"], expected_mean)


# -- normalize / denormalize round-trip ----------------------------

def test_normalize_denormalize_round_trip():
    """The load-bearing invariant: denorm(norm(x)) ≈ x."""
    ds = FakeDataset(n_frames=10)
    stats = compute_stats(ds)
    sample = ds[0]["observation.state"]
    recovered = denormalize(
        normalize(sample, stats, "observation.state"),
        stats, "observation.state",
    )
    assert torch.allclose(sample, recovered, atol=1e-5)


def test_normalize_yields_approximately_zero_mean():
    """Normalizing all state vectors → batch mean ≈ 0 per dim."""
    ds = FakeDataset(n_frames=100)
    stats = compute_stats(ds)
    states = torch.stack(
        [ds[i]["observation.state"] for i in range(len(ds))]
    )
    normalized = normalize(states, stats, "observation.state")
    assert torch.allclose(
        normalized.mean(0), torch.zeros(6), atol=1e-5,
    )


def test_normalize_eps_handles_zero_variance_feature():
    """Constant feature has std=0; epsilon prevents NaN."""
    stats = {
        "action": {
            "mean": torch.tensor([1.0, 1.0]),
            "std": torch.tensor([0.0, 0.0]),
            "min": torch.tensor([1.0, 1.0]),
            "max": torch.tensor([1.0, 1.0]),
        }
    }
    x = torch.tensor([1.0, 1.0])
    out = normalize(x, stats, "action")
    assert torch.isfinite(out).all()


# -- collate function ----------------------------------------------

def test_collate_normalizes_state_and_action():
    ds = FakeDataset(n_frames=10)
    stats = compute_stats(ds)
    collate = _build_collate_fn(stats)
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["observation.state"].shape == (3, 6)
    assert batch["action"].shape == (3, 6)
    # mean across batch should not match raw (because normalized)
    raw_state = torch.stack(
        [ds[i]["observation.state"] for i in range(3)]
    )
    assert not torch.allclose(
        batch["observation.state"], raw_state, atol=1e-3,
    )


def test_collate_passes_through_float32_images():
    ds = FakeDataset(n_frames=4)
    stats = compute_stats(ds)
    collate = _build_collate_fn(stats)
    batch = collate([ds[0], ds[1]])
    img = batch["observation.images.up"]
    assert img.shape == (2, 3, 32, 32)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min().item() and img.max().item() <= 1.0


def test_collate_handles_uint8_images_legacy_path():
    """Defensive: a uint8 image batch gets divided by 255."""
    frames = [
        {
            "observation.state": torch.zeros(6),
            "action": torch.zeros(6),
            "observation.images.up": torch.full(
                (3, 4, 4), 255, dtype=torch.uint8,
            ),
        }
        for _ in range(2)
    ]
    stats = {
        "observation.state": {
            "mean": torch.zeros(6), "std": torch.ones(6),
            "min": torch.zeros(6), "max": torch.zeros(6),
        },
        "action": {
            "mean": torch.zeros(6), "std": torch.ones(6),
            "min": torch.zeros(6), "max": torch.zeros(6),
        },
    }
    out = _build_collate_fn(stats)(frames)
    img = out["observation.images.up"]
    assert img.dtype == torch.float32
    assert torch.allclose(img, torch.ones_like(img))


def test_collate_keeps_task_strings_as_list():
    ds = FakeDataset(n_frames=3)
    stats = compute_stats(ds)
    batch = _build_collate_fn(stats)([ds[0], ds[1], ds[2]])
    assert batch["task"] == ["pink lego brick into the transparent box"] * 3


# -- stats dict shape contract --------------------------------------

def test_stats_dict_type_alias_resolves():
    """StatsDict is importable + the type annotation works at runtime."""
    sample: StatsDict = {
        "observation.state": {
            "mean": torch.zeros(6), "std": torch.ones(6),
            "min": torch.zeros(6), "max": torch.zeros(6),
        },
    }
    assert sample["observation.state"]["mean"].shape == (6,)


def test_chapter3_imports_canonical_paths():
    """These exact import statements are the Ch3 contract."""
    from ch02 import denormalize as d  # noqa: F401
    from ch02 import make_pickplace_dataloader as m  # noqa: F401
    from ch02 import normalize as n  # noqa: F401


def test_stats_dict_alias_reexported():
    """Ch3 will type-annotate against StatsDict — accessible at root."""
    from ch02 import StatsDict as SD  # noqa: F401


def test_make_pickplace_dataloader_end_to_end(monkeypatch):
    """Fallback path: dataset has no meta.stats; compute_stats runs."""
    fake = FakeDataset(n_frames=12)
    monkeypatch.setattr("ch02.pipeline.load_dataset", lambda _id: fake)
    loader, stats = make_pickplace_dataloader(
        dataset_id="ignored", batch_size=4, shuffle=False,
    )
    from torch.utils.data import DataLoader as _DL
    assert isinstance(loader, _DL)
    assert set(stats.keys()) == {"observation.state", "action"}
    batch = next(iter(loader))
    assert batch["observation.state"].shape == (4, 6)
    assert batch["action"].shape == (4, 6)


def test_make_pickplace_dataloader_uses_meta_stats_fast_path(monkeypatch):
    """Fast path: precomputed meta.stats short-circuits compute_stats."""
    fake = FakeDataset(n_frames=12)
    # Inject a sentinel meta.stats so we can prove compute_stats was skipped
    sentinel = {
        "mean": torch.full((6,), 7.0),
        "std": torch.ones(6),
        "min": torch.zeros(6),
        "max": torch.ones(6),
    }
    fake.meta = type("Meta", (), {"stats": {
        "observation.state": sentinel,
        "action": sentinel,
    }})()
    monkeypatch.setattr("ch02.pipeline.load_dataset", lambda _id: fake)

    # Boom if compute_stats is called — proves we took the fast path.
    def boom(_):
        raise AssertionError("compute_stats ran despite meta.stats")
    monkeypatch.setattr("ch02.pipeline.compute_stats", boom)

    _, stats = make_pickplace_dataloader(dataset_id="ignored")
    assert torch.equal(stats["observation.state"]["mean"], sentinel["mean"])


def test_collate_clamps_float_images_out_of_range():
    """Defensive clamp keeps the Ch3 [0,1] image contract."""
    frames = [
        {
            "observation.state": torch.zeros(6),
            "action": torch.zeros(6),
            "observation.images.up": torch.tensor(
                [[[1.5, -0.2]]], dtype=torch.float32
            ),
        }
        for _ in range(2)
    ]
    stats = {
        "observation.state": {
            "mean": torch.zeros(6), "std": torch.ones(6),
            "min": torch.zeros(6), "max": torch.zeros(6),
        },
        "action": {
            "mean": torch.zeros(6), "std": torch.ones(6),
            "min": torch.zeros(6), "max": torch.zeros(6),
        },
    }
    out = _build_collate_fn(stats)(frames)
    img = out["observation.images.up"]
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_denormalize_round_trip_on_action_key():
    """Symmetric coverage — action is what Ch3 denormalizes for env.step."""
    ds = FakeDataset(n_frames=10)
    stats = compute_stats(ds)
    sample = ds[0]["action"]
    recovered = denormalize(
        normalize(sample, stats, "action"),
        stats, "action",
    )
    assert torch.allclose(sample, recovered, atol=1e-5)


def test_normalize_preserves_batched_shape():
    """(B, D) in → (B, D) out via broadcasting against (D,) stats."""
    ds = FakeDataset(n_frames=10)
    stats = compute_stats(ds)
    batch = torch.randn(8, 6)
    out = normalize(batch, stats, "observation.state")
    assert out.shape == (8, 6)
    assert out.dtype == torch.float32
