# Chapter 2 Software Design

Companion to `chapter_2_plan.md`. The plan locks the *what* — listings, exports, prose mapping. This doc locks the *how* — module APIs, notebook architecture, test strategy, dependency pinning, and the agent prompt.

If you change anything in this doc that affects the Ch3 export contract, update `chapter_2_plan.md` and `../lrm-book/chapter_2/chapter_2_structure_and_plan.md` in the same commit.

**Authoritative upstream sources** per `MANNING_STYLE.md`: `../../lrm-book/STYLEGUIDE.md`, `../../lrm-book/FIGURE_STYLE_GUIDE.md`, `../../lrm-book/writing_instructions/writing_instructions.md`, and `../../lrm-book/book_learnings.md` (§2 covers craft patterns for coding-heavy chapters). When this doc conflicts with any upstream, upstream wins.

---

## 1. Reader Experience Workflow

Two paths, same end state — the reader runs `notebooks/ch02.ipynb` against the `ch02` package.

**Local:** `git clone`, `pip install -e ".[dev,data,sim]"`, open the notebook in Jupyter. Editable install picks up edits to `src/ch02/*.py` on the next import (use `%load_ext autoreload` for live-reload).

**Colab:** README's "Open in Colab" badge loads `notebooks/ch02.ipynb` directly in Colab. The notebook's first cell detects Colab and runs the Vulkan + `pip install` recipe (full version in README):

```python
import sys
if "google.colab" in sys.modules:
    # ... 6 lines of Vulkan ICD setup (see README) ...
    !pip install -q "lrm-ch02[data,sim] @ git+https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2.git@<release-tag>"
```

Reader never uploads anything. Colab pulls the notebook from GitHub; pip pulls the `ch02` package from the same repo.

### Notebook ↔ package

**Type-along** listings (random agent, scripted policy, normalize functions) live in the notebook namespace — the reader writes the code in the cell, subsequent cells call their version. A byte-for-byte equivalent exists in `src/ch02/<module>.py` but is independent (no live binding) and serves tests, Ch3 imports, and editor cross-reference.

**Provided utility** listings (viz helpers, dataloader plumbing) are one-line imports from the package — the reader doesn't retype.

The isolation means reader experiments in the notebook don't affect tests or downstream chapters.

### Figures

Inline in both paths. `fig.savefig("figures/...")` is author-only — gated off in Colab, used locally to regenerate print-quality PNGs.

---

## 2. Module APIs

The chapter plan's module mapping is the starting point. This section pins every public function signature (the agents' `listing-check` enforces these against the plan's listings).

### `src/ch02/__init__.py`

Re-exports the Ch3 contract symbols at the package root so downstream chapters can `from ch02 import normalize` without knowing the submodule layout.

```python
from ch02.pipeline import make_pickplace_dataloader, normalize, denormalize

__all__ = ["make_pickplace_dataloader", "normalize", "denormalize"]
```

### `src/ch02/env.py` (listings 2.1, 2.2)

```python
def make_env(
    observation_mode: str = "both",
    render_mode: str = "rgb_array",
    seed: int | None = None,
) -> gym.Env: ...

def run_random_agent(
    env: gym.Env,
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> tuple[float, float]:
    """Returns (success_rate, mean_return)."""
```

`make_env` wraps the listing 2.1 `gym.make(...)` call so the notebook, tests, and the scripted-agent eval all construct the env the same way. Listing 2.1 in the notebook shows the `gym.make` line verbatim; the function in `env.py` is the canonical version everything else imports.

### `src/ch02/scripted.py` (listings 2.3, 2.4)

```python
PHASES: list[str] = [
    "approach", "descend", "grasp",
    "lift", "transport", "place", "release",
]

def scripted_policy(
    obs: dict,
    state: dict,
) -> np.ndarray:
    """Returns a (7,) float32 action. Mutates `state` to advance phases."""

def run_scripted_agent(
    env: gym.Env,
    n_episodes: int = 10,
) -> float:
    """Returns success_rate."""
```

`state` is a plain dict that the caller owns — keeps the function pure-ish (no module-level state), matches listing 2.3 exactly, and makes the per-episode reset trivial (`state = {"phase": "approach"}`).

### `src/ch02/dataset.py` (listings 2.5, 2.6)

```python
DEFAULT_DATASET_ID: str = "lerobot/so100_pick_place"  # pin at implementation

def load_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
) -> LeRobotDataset: ...

def episode_frames(
    dataset: LeRobotDataset,
    episode_idx: int,
) -> list[dict]:
    """All frames belonging to one episode, in order."""
```

`episode_frames` is extracted as a helper because both listing 2.6 and listing 2.7 (`render_keyframes`) re-derive it inline in the plan — DRY it once here so the notebook doesn't show the same list comprehension twice.

`load_dataset` is a one-line wrapper that exists mainly so tests can inject a fixture path without monkeypatching `LeRobotDataset`.

### `src/ch02/viz.py` (listings 2.7, 2.8)

```python
def render_keyframes(
    dataset: LeRobotDataset,
    episode_idx: int = 0,
    n_frames: int = 6,
    save_path: str | None = None,
) -> matplotlib.figure.Figure: ...

def collect_actions(
    env: gym.Env,
    policy_fn: Callable | None,
    n_episodes: int = 10,
) -> np.ndarray:
    """policy_fn=None means random. Returns (n_steps, 7) array."""

def plot_action_distributions(
    expert: np.ndarray,
    scripted: np.ndarray,
    random_: np.ndarray,
    save_path: str | None = None,
) -> matplotlib.figure.Figure: ...

def plot_joint_trajectories(
    dataset: LeRobotDataset,
    episode_indices: Iterable[int] = range(5),
    save_path: str | None = None,
) -> matplotlib.figure.Figure: ...
```

Two changes from the chapter plan worth flagging:

1. `save_path` is a keyword arg, not hardcoded. The chapter plan's listings hardcode the path (`"figures/figure_2_4_expert_keyframes.png"`) — that's fine for the book's prose, but tests need to redirect to a tmp path. The notebook cell will pass the canonical path explicitly.
2. All four functions return the `Figure` instead of saving as a side effect, so the notebook can both inline-render *and* save in the same call.

`plot_joint_trajectories` supports figure 2.6, which the plan describes but doesn't tie to a listing. It belongs in `viz.py` for symmetry with the other plotting helpers.

### `src/ch02/pipeline.py` (listings 2.9, 2.10, 2.11)

```python
StatsDict = dict[str, dict[str, torch.Tensor]]

def compute_stats(dataset: LeRobotDataset) -> StatsDict: ...

def normalize(
    x: torch.Tensor,
    stats: StatsDict,
    key: str,
) -> torch.Tensor: ...

def denormalize(
    x: torch.Tensor,
    stats: StatsDict,
    key: str,
) -> torch.Tensor: ...

def make_pickplace_dataloader(
    dataset_id: str = DEFAULT_DATASET_ID,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tuple[DataLoader, StatsDict]: ...
```

`StatsDict` is type-aliased so Chapter 3's signature annotations can import it. The `stats` dict shape (`stats["observation.state"]["mean"]`) is part of the contract — codified here as a type, not just prose.

`make_pickplace_dataloader` is the frozen export. Default `num_workers=4` per listing 2.11; we may need to bump down to `0` on Colab if multiprocessing-with-CUDA gets cranky.

---

## 3. Notebook Architecture

### Layout

One notebook: `notebooks/ch02.ipynb`. Section headers mirror the chapter (`## 2.1 The SO-100 Pick-and-Place Environment`, etc.). Within each section:

- Markdown cell introducing the listing — caption format is **noun phrase, no colon, no "this listing..."** (e.g., `### Listing 2.5 Loading the SO-101 expert dataset`). Per `MANNING_STYLE.md` §1.5 and `book_learnings.md` craft patterns.
- A **lead-in sentence** before the code cell, keyed to the listing's role:
  - **Type-along:** name the function and what it teaches.
  - **API illustration:** name what API surface is being shown.
  - **Provided utility:** explicit `from ch02.<module> import <function>` mention, plus "shown here for transparency, not for typing."
- Code cell containing the listing.
- Output cell (cleared before commit).

### What lives inline vs imported, per listing role

| Role | Notebook cell contents |
|------|------------------------|
| Type-along (2.2, 2.3, 2.9, 2.10) | Full code from the book listing, inline. Reader can read, retype, or modify. |
| API illustration (2.1, 2.4, 2.5, 2.6) | Full code, inline. They're short and the reader benefits from running them as-is. |
| Provided utility (2.7, 2.8, 2.11) | `from ch02.viz import render_keyframes` (etc.) + a usage call. Implementation stays in the package. |

Notebook/package shadowing mechanics are covered in §1.

### Figure conventions

Every figure-producing cell:

```python
fig = plot_X(...)
fig.savefig("../figures/figure_2_<n>_<slug>.png", dpi=300, bbox_inches="tight")
```

Paths are relative to `notebooks/`. Figure 2.1 is reused from Chapter 1 — no code cell, just a markdown reference.

### Reproducibility

- Every random call seeded. `env.reset(seed=ep)` per episode; `np.random.seed(0)` at the top of any cell that does its own sampling.
- Dataset loads pinned to `DEFAULT_DATASET_ID` from `ch02.dataset`. No mid-notebook string literals for dataset IDs.

### Output discipline

- Pre-commit hook: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.
- Notebook diffs without output stay reviewable; outputs regenerate on a fresh run.

### Hazards

- **Stale `env`.** If a cell crashes mid-episode, `env` may be in a bad state. Each major cell does its own `env.reset()` rather than relying on prior state.
- **Dataset size.** First load is multi-GB. Load once at the start of §2.3 and reuse the `dataset` variable; don't reload per cell.
- **Image-observation memory.** `observation_mode="both"` keeps images in obs dicts. On a 16 GB laptop running multiple kernels, this can OOM. A `observation_mode="state"` fallback is documented in §2.1 prose for low-memory readers.

---

## 4. Test Strategy

### Layout

```
tests/
├── conftest.py            # shared fixtures (tiny synthetic dataset, env factory)
├── test_pipeline.py       # full coverage — this is the export contract
├── test_scripted.py       # phase-transition logic, no MuJoCo needed
├── test_dataset.py        # smoke — load fixture, count frames
├── test_viz.py            # smoke — returns Figure, no crash
└── test_env.py            # integration, @pytest.mark.integration
```

### Coverage by module

| Module | Target | Approach |
|--------|--------|----------|
| `pipeline.py` | 100% of public API | Synthetic mini-dataset fixture; assert stats math, round-trip equality, dataloader yields correctly shaped batches. |
| `scripted.py` | Phase-transition logic | Construct synthetic `obs` dicts that should trigger each transition; assert `state["phase"]` advances. No env needed. |
| `dataset.py` | Smoke only | `load_dataset(fixture_id)` returns something with `len()` > 0. |
| `viz.py` | Smoke only | Each function returns a `Figure`; nothing crashes on the fixture dataset. |
| `env.py` | Integration, gated | `make_env()` instantiates, `step()` returns a 5-tuple. Marked `@pytest.mark.integration`, skipped by default. |

### Fixtures (`conftest.py`)

```python
@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A 20-frame synthetic LeRobotDataset-compatible mock."""
    ...

@pytest.fixture
def env_or_skip():
    """Returns make_env() or skips if ManiSkill/SAPIEN unavailable."""
    ...
```

The tiny dataset is the load-bearing fixture — it lets `pipeline.py` tests run in <1 second without downloading anything.

### What we explicitly don't test

- Notebook execution end-to-end. That's verified by the author running it from a fresh kernel before each chapter release, not by CI. (Optional future: `nbmake` in CI.)
- The actual LeRobot Hub download path. Tested manually; CI uses the fixture.
- Figure pixel-equivalence. Image diffs are flaky; we test that `Figure` objects are returned, not what's in them.

### CI

GitHub Actions: `pytest tests/ -m "not integration"` on push. Integration suite runs on a manual trigger or nightly cron with the ManiSkill + Vulkan setup.

### Reader-facing test instruction

One line in the README: "Run `pytest tests/` after install to smoke-test your setup." That's it — readers aren't expected to engage with the test suite as curriculum.

---

## 5. Dependency Strategy

### Python version

**3.12 only** (`>=3.12,<3.13`). `lerobot` 0.5.1 declares `requires_python = ">=3.12"`, and `lrm-code-agents/defaults.yml` locks 3.12 for resource-check. Colab supports 3.12 in current runtimes.

### Packaging

`pyproject.toml` with PEP 621 metadata + optional dependency groups:

```toml
[project]
name = "lrm-ch02"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.1,<3.0",
    "numpy>=1.24,<3.0",
    "gymnasium>=0.29,<2.0",
    "matplotlib>=3.7,<4.0",
]

[project.optional-dependencies]
sim = [
    "mani-skill==3.0.1",
]
data = [
    "lerobot==0.5.1",
]
dev = [
    "pytest>=7.4",
    "pytest-cov",
    "ruff",
    "jupyter",
    "nbconvert",
    "nbstripout",
    "pre-commit",
]
```

Pins:
- **`lerobot==0.5.1`** — exact pin. Released 2026-04-07; the book's listings target this API.
- **`mani-skill==3.0.1`** — exact pin. Apache 2.0, SAPIEN-based, GPU-parallelized. SO-100 is a first-class robot in upstream (`mani_skill/agents/robots/so100/`); ships `PickCubeSO100-v1` and the `SO100GraspCube-v1` sim2real digital twin. Maintained by UCSD's Hao Su lab. Bundled SO-100 URDF/meshes Apache 2.0 (sourced from `TheRobotStudio/SO-ARM100`). SAPIEN dependency is MIT.
- **`torch`, `numpy`, etc.** — loose pins; foundational libraries change less aggressively.

**Sim coverage strategy across the book:**
- Ch 2–5: ManiSkill3, SO-100 carrier embodiment, `PickCubeSO100-v1` task family.
- Ch 6–7 (scaling / RL): ManiSkill3's GPU-parallelized rollouts are the enabling feature — RL training in minutes on a 4090 rather than hours on CPU MuJoCo. Larger task family (PickCube, StackCube, PushCube, PegInsertion, PlugCharger, YCB pick) available as scaling demonstrations without changing simulators.
- Ch 8–9 (sim-to-real): `SO100GraspCube-v1` digital twin with built-in domain randomization is the teaching artifact. This is materially richer than a one-line API swap — readers learn what sim2real actually is.
- Ch 10–11 (deployment): real SO-101 hardware via lerobot; no sim.

**Why not `lerobot-sim2real`:** Stone Tao's `lerobot-sim2real` repo (https://github.com/StoneT2000/lerobot-sim2real) is a useful reference implementation that demonstrates ManiSkill + LeRobot integration, but cannot be a dependency: (a) it has no LICENSE file (`license: null` on the GitHub API), making redistribution legally ambiguous, and (b) it pins lerobot to a pre-0.4.0 commit using `lerobot.common.robots.*` imports that no longer exist in 0.5.1. We read it for technique, vendor any small glue scripts we need under our own Apache 2.0, and depend only on `mani-skill` and `lerobot` directly.

**Forking plan:** if later chapters need SO-101 in sim or custom tasks that the upstream ManiSkill doesn't cover, fork `haosulab/ManiSkill` under the book's GitHub organization and re-point the pin. The third-party `aalmuzairee/squint` repo (MIT) already has an SO-101 ManiSkill port we can reference. Default plan: stay on upstream until concrete need forces the fork.

### Lockfile

Generated as a **release-time artifact**, not maintained during development. Strict pins on `mani-skill==3.0.1` and `lerobot==0.5.1` plus loose pins on stable foundations cover the dev loop. At chapter release: `pip-compile pyproject.toml --extra dev --extra data --extra sim -o requirements.txt`. Readers who hit dep-drift get a one-line escape hatch.

Install paths and the Colab recipe live in §1 and the README.

---

## 6. Chapter-2 Agent Prompt

### Location

Canonical source: `agents/chapter-02-guide.md` — zero-padded chapter number per `MANNING_STYLE.md` §1.5 and §6. The file ships in `agents/` alongside `notebooks/`, `src/`, and `tests/` as a first-class chapter deliverable.

For Claude Code to discover it as a subagent, the reader's setup symlinks it into `.claude/agents/`:

```bash
mkdir -p .claude/agents
ln -s ../../agents/chapter-02-guide.md .claude/agents/chapter-02-guide.md
```

The README's quick-start documents this step. (This mirrors the author-side convention in `program.md` §2, where `lrm-code-agents/agents/` is symlinked into `.claude/agents/` — same pattern, different agents.)

### Structure

```markdown
---
name: chapter-2-guide
description: Walks readers through Chapter 2 of *Build a Large Robot Model*.
tools: Read, Bash, WebFetch
---

# Role

You are the Chapter 2 guide for *Build a Large Robot Model (From Scratch)*.
Chapter 2 is "Simulation & Data" — building the SO-100 pick-and-place
simulation environment, a scripted policy, the LeRobot SO-101 expert
dataset loader, and a normalized DataLoader.

The reader is working through this chapter with the book on one side and a
terminal on the other. They have already installed dependencies and cloned
the repo. Your job is to guide them through the same eleven listings the
notebook covers, with the added value of conversational clarification.

# What you know

- The chapter plan: docs/chapter_2_plan.md
- The package source: src/ch02/*.py
- The notebook (for structural reference, not for running): notebooks/ch02.ipynb

You may read any of these. Quote from them when explaining; don't paraphrase
when the source is clearer.

# How to interact

1. Greet the reader briefly. Ask which listing they want to start with (default:
   2.1 from the top).
2. For each listing, in order:
   - Open with a **role-keyed lead-in sentence** per MANNING_STYLE.md §1.5:
     - Type-along: name the function and what it teaches.
     - API illustration: name what API surface is being shown.
     - Provided utility: explicit `from ch02.<module> import <function>` and
       note the implementation is shown for transparency, not for typing.
   - Present the code from the listing exactly as it appears in the chapter.
     Do not rewrite or "improve" it.
   - Wait for the reader to run it or ask a question.
   - If they ran it successfully, ask one comprehension-check question before
     moving on. ("What happens if you change `seed=42` to a different value?")
   - If they hit an error, help debug. Common failure modes: SAPIEN/Vulkan
     setup, dataset download, version mismatch.
3. At the end of each section (2.1–2.5):
   - Summarize what was built in 2–3 sentences.
   - State an **honest scoping disclaimer** — what this section *did not*
     cover and where in the book that material lives. (Per
     book_learnings.md, this is the single biggest reader-trust device
     across the reference books.)
   - Confirm the reader is ready to move on.

# Boundaries

- This chapter ends at the normalized DataLoader. If the reader asks about
  training a model, behavior cloning, or anything else from Chapter 3+, defer:
  "That's Chapter 3 material — let's finish the data pipeline first." Don't
  improvise model code.
- Don't rewrite listings. The book's versions are canonical. If a reader
  thinks one is wrong, surface it as an issue rather than editing on the fly.
- Don't run destructive commands without confirmation (rm, git reset, etc.).
- If a reader wants to skip ahead, point them to the book's chapter preview,
  not your own summary.

# When to defer back to the book or notebook

- Long-form prose explanations of *why* a design choice was made → quote the
  callout box from the chapter rather than synthesizing.
- Figure interpretation → point to the figure caption.
- Anything you're not confident about → say so and recommend the book.
```

### What gets loaded as context

The agent loads files lazily (via the `Read` tool) rather than stuffing everything into the system prompt. That keeps the prompt small and lets the agent re-read updated files in future sessions.

### Testing the agent

Hard to automate. Pragmatic approach:

- A small `tests/agent_prompts.md` with 5–10 sanity questions the agent should answer correctly ("Why z-score and not min-max for actions?", "What's the role of `delta_timestamps`?"). Manually run before each chapter release.
- No CI gating — agent behavior changes with model versions and can't be locked down deterministically.

### Maintenance flag

This file is the experimental layer. If readers find it confusing or it drifts from the notebook, we cut it without breaking the canonical path. The notebook is the contract; the agent is the bonus.

---

## Open Questions

Things this doc deliberately doesn't pin yet, and where the decision will be made:

1. **The exact `DEFAULT_DATASET_ID`.** Pinned during §2.3 implementation, when we verify which SO-100 dataset on the Hub actually loads cleanly with the pinned `lerobot` version.
2. **`num_workers` default in `make_pickplace_dataloader`.** Listing 2.11 says 4. May drop to 0 if Colab multiprocessing is unhappy. Verified during §2.5 implementation.
3. **CI integration-test runner.** Whether the nightly MuJoCo integration suite runs on GitHub Actions (free runners can be flaky with OpenGL) or a self-hosted runner. Decided when CI is actually wired up.
4. **Notebook → book sync mechanism.** Right now, drift is caught by author re-reading both. Long-term we may want a `nbconvert`-based diff tool. Defer until the second chapter is built.
