# lrm-code-chapter-2

Companion code for **Chapter 2: Simulation & Data** of *Build a Large Robot Model (From Scratch)*.

This repo contains the runnable code, tests, and figure-generation notebooks for everything the reader builds in Chapter 2: the SO-100 pick-and-place sim wrapper, a multi-phase scripted policy baseline, the LeRobot dataset loader, and the normalized DataLoader that downstream chapters depend on. The simulator is **ManiSkill3** (SAPIEN-based, GPU-parallelized); the carrier embodiment is the **SO-100 arm** in sim transitioning to **SO-101 hardware** in the later deployment chapters — the same low-cost 6-DOF arm with parallel-jaw gripper used through Chapter 11.

## Two reader paths

This chapter ships **two ways to work through the material**:

1. **`notebooks/ch02.ipynb` — the canonical path.** Maps cell-by-cell to the book's prose. Version-pinned and kept green; this is what we promise will work end-to-end.
2. **`agents/chapter-02-guide.md` — the experimental companion.** A Claude Code subagent that walks the reader through the same listings in dialogue. Scope is strictly Chapter 2 — it will not skate into later-chapter material. Optional and unsupported beyond best-effort.

**To use the companion agent**, symlink it into Claude Code's per-project agent directory so the subagent is auto-discovered, then start Claude Code from this repo root:

```bash
mkdir -p .claude/agents
ln -s ../../agents/chapter-02-guide.md .claude/agents/chapter-02-guide.md
# Then start Claude Code; "chapter-02-guide" appears as an invocable subagent.
```

Tests in `tests/` are author/CI infrastructure. Readers may run them as an install smoke check; engagement is not expected.

`scripts/check_*.py` are reader-friendly smoke scripts for each section's package code (env, scripted policy, dataset, viz helpers, pipeline). Run any of them from the repo root after install to verify a section works end-to-end.

## Layout

```
lrm-code-chapter-2/
├── docs/                    # Chapter plan + Manning style reference
├── src/ch02/                # Importable Python package — the export contract
├── notebooks/               # ch02.ipynb — reader's canonical walkthrough
├── agents/                  # chapter-02-guide.md — reader's optional companion agent
├── tests/                   # pytest suite (author/CI infrastructure, not for readers)
├── figures/                 # Rendered figures referenced by the chapter
├── program.md               # How to use this repo with the chapter plan + agents
└── README.md
```

This repo follows the **Raschka-style hybrid model** for code (importable Python in `src/ch02/`, notebook walkthrough in `notebooks/`), plus a per-chapter agent for dialogic learning. This is Manning's first book to ship a chapter agent alongside the standard book + notebook artifacts.

## Reader install — local

This chapter runs on **Linux, WSL2, or Colab**. macOS works for state-mode code (the dataset listings, the pipeline) but rendering (RGB observations, the §2.2 random-rollout video) needs Vulkan, which is unreliable under MoltenVK — use Linux/WSL2 or Colab for those sections.

### Step 1 — System prerequisites

Two native libraries the Python extras can't install themselves:

- **FFmpeg** — `lerobot` decodes dataset videos via `torchcodec`, a wrapper around system FFmpeg. Without it, dataset iteration crashes at `libavutil.so.* cannot open shared object file`.
- **Vulkan loader + driver** — `mani-skill` renders via SAPIEN, which uses Vulkan.

```bash
# Ubuntu / Debian / WSL2
sudo apt install ffmpeg libvulkan1 mesa-vulkan-drivers vulkan-tools

# macOS (state-mode only; rendering caveat above)
brew install ffmpeg
```

Software rendering via mesa's `llvmpipe` is enough for state-mode env stepping. A real GPU is recommended for RGB rendering at speed.

### Step 2 — Python 3.12

Pinned to match `lerobot==0.5.1`'s metadata constraints. 3.10/3.11 won't resolve; 3.13 is too new for `lerobot`.

### Step 3 — Clone the repo

```bash
git clone --depth 1 https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2.git
cd lrm-code-chapter-2
```

`--depth 1` skips git history for a faster, lighter clone.

### Step 4 — Install dependencies (uv recommended)

We recommend [**uv**](https://docs.astral.sh/uv/) — significantly faster than pip, resolves dependency graphs more reliably, and creates an isolated `.venv/` automatically.

1. Install uv: see [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Sync the project (from inside the repo root):

   ```bash
   uv sync --extra data --extra sim
   ```

   This reads `pyproject.toml`, creates `.venv/`, and installs the chapter package along with the `[data]` extra (`lerobot==0.5.1`) and `[sim]` extra (`mani-skill==3.0.1`). Expect ~4 GB on disk and 2–5 minutes the first time.

3. Launch JupyterLab:

   ```bash
   uv run jupyter lab notebooks/ch02.ipynb
   ```

   `uv run` activates the project's `.venv/` for that single command, so you don't have to `source .venv/bin/activate` manually.

#### pip fallback

If you'd rather use pip:

```bash
pip install -e ".[data,sim]"
jupyter lab notebooks/ch02.ipynb
```

Same dependency graph; slower resolver. Activate your own venv first if you want isolation.

### Reader install — Colab

ManiSkill needs Vulkan ICDs configured before `mani-skill` can render. The Colab path uses a single bootstrap cell — already present as the first cell of `notebooks/ch02.ipynb`:

```bash
!mkdir -p /usr/share/vulkan/icd.d
!wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
!wget -q https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json
!mv nvidia_icd.json /usr/share/vulkan/icd.d
!mv 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
!apt-get install -y --no-install-recommends libvulkan-dev
!pip install -q "lrm-ch02[data,sim] @ git+https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2.git"
```

The recipe is verbatim from ManiSkill's `examples/tutorials/1_quickstart.ipynb`. Exercised on the free T4 tier; 15 GB GPU memory is sufficient for everything in this chapter.

## Contributor setup

If you're adding to the chapter (not just reading), there are extra steps:

1. Read `program.md` end-to-end — it is the operating manual.
2. Read `docs/chapter_2_plan.md` for the chapter blueprint (listings, figures, exports).
3. Read `docs/design.md` for module APIs, test strategy, dep pins, and the agent prompt.
4. Set up the `.claude/agents` symlink from `lrm-code-agents` (instructions in `program.md`).
5. Install with the `[dev]` extra too:

   ```bash
   uv sync --extra dev --extra data --extra sim
   # or:
   pip install -e ".[dev,data,sim]"
   ```

### Tests

```bash
pytest -m "not integration"   # unit tests; fast; what CI runs
pytest -m integration          # spins up ManiSkill / SAPIEN; needs sim extra installed
```

### Pre-commit hooks

```bash
pre-commit install
```

Installs `nbstripout` (clears notebook outputs before commit) and `ruff` (Python lint + autofix).

## Exports for Chapter 3

After implementation, this repo will export:

| Symbol | Purpose |
|--------|---------|
| `make_pickplace_dataloader(dataset_id, batch_size, shuffle)` | Normalized DataLoader + stats; `dataset_id` lets later chapters swap in custom datasets without changing the signature |
| `normalize(x, stats, key)` | Z-score normalize state and actions |
| `denormalize(x, stats, key)` | Inverse normalize (model output → environment scale) |

Chapter 3 imports these directly — the function signatures are the API contract.
