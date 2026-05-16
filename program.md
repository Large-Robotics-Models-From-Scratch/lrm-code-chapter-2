# Program: How to Implement Chapter 2 Code

This is the **operating manual** for this repo. It tells Claude Code (or a human collaborator) how to use the three inputs available — the chapter plan, the agent toolkit, and the Manning style guide — to go from this empty scaffold to a finished, validated Chapter 2 codebase.

Read this end-to-end before writing any Python.

---

## 1. The Three Inputs

| Input | Where | Role |
|-------|-------|------|
| **Chapter plan** | `docs/chapter_2_plan.md` | The *what*. 11 listings, 8 figures (incl. roadmap recap), 6 callouts, 2 tables, and the Ch2→Ch3 export contract. |
| **Agent toolkit** | `../lrm-code-agents/` | The *check*. Five Claude Code subagents that enforce style, listing format, continuity, resources, and tests. |
| **Style guide** | `docs/MANNING_STYLE.md` + `../lrm-book/STYLEGUIDE.md` | The *how*. Manning conventions — annotations, line widths, banned words, code style. |

Implementation is a tight loop between these three. The plan tells you the next listing to build. The style guide tells you how to write it. The agents tell you whether you got it right.

---

## 2. One-Time Setup

Before the first implementation pass, link the author toolkit and the reader-facing chapter agent into `.claude/agents/` so Claude Code discovers everything at startup.

```bash
cd /home/siddharth/workspaces/LRM/src/lrm-code-chapter-2

mkdir -p .claude/agents

# Author toolkit (lrm-code-agents): style-check, listing-check, etc.
# These ship as a directory; symlink the whole agents folder once and
# then add the reader-facing agent file individually.
for f in ../../lrm-code-agents/agents/*; do
    ln -sf "../$f" .claude/agents/
done
ln -s ../../lrm-code-agents/CLAUDE.md .claude/CLAUDE.md

# Reader-facing chapter agent (committed in this repo at agents/)
ln -s ../../agents/chapter-02-guide.md .claude/agents/chapter-02-guide.md
```

Optionally, copy the agent toolkit defaults to override settings:

```bash
cp ../lrm-code-agents/defaults.yml .lrm-agents.yml
# edit .lrm-agents.yml — e.g., set reminder_percent: 50
```

**Then restart Claude Code from this directory.** Subagents are loaded at session start; the new ones will not appear until then.

Verify by checking that the subagent list now includes both the author toolkit (`style-check`, `listing-check`, `chapter-continuity`, `test-gen`, `resource-check`) and the reader agent (`chapter-02-guide`).

---

## 3. The Build Loop

This repo follows the **Raschka-style hybrid model** used across Manning's "from scratch" series:

- **`src/ch02/*.py`** — the importable Python package. Holds every function and class that downstream chapters or tests touch. This is where the export contract lives.
- **`notebooks/ch02.ipynb`** — the reader's primary artifact. Imports from `src/ch02/`, runs the code cell-by-cell, produces inline plots, and contains the prose that mirrors the book.

Every listing in the chapter plan has a home in **both** artifacts: it is defined as a function/class in `src/ch02/` and *also* appears (or is called) in the notebook so the reader can execute it inline. The notebook is the experience; the package is the substance.

For each of the 11 listings in `docs/chapter_2_plan.md`, in order:

1. **Read the listing spec.** Open `docs/chapter_2_plan.md` and find the listing (e.g., Listing 2.1). Note the code, the `#A`–`#Z` annotations, and the section it belongs to.

2. **Locate the module.** Decide which file in `src/ch02/` the listing's code lives in. Group related listings into the same module:

   | Listings | Module | Reader role per listing |
   |----------|--------|-------------------------|
   | 2.1, 2.2 | `src/ch02/env.py` (`gym-lowcostrobot` PickPlaceCube setup, random agent) | 2.1 API illustration; 2.2 type-along |
   | 2.3, 2.4 | `src/ch02/scripted.py` (multi-phase pick-and-place state machine + eval) | 2.3 type-along; 2.4 API illustration |
   | 2.5, 2.6 | `src/ch02/dataset.py` (LeRobot dataset loading) | both API illustrations |
   | 2.7, 2.8 | `src/ch02/viz.py` (keyframe + per-joint distribution plotting helpers) | both provided utilities (imported by the reader) |
   | 2.9, 2.10, 2.11 | `src/ch02/pipeline.py` (stats, normalize, `make_pickplace_dataloader`) | 2.9 and 2.10 type-along; 2.11 provided utility |

   These are starting suggestions, not contracts — adjust based on what reads cleanly. The export contract is the only hard boundary.

3. **Mirror in the notebook.** For each listing implemented in `src/ch02/`, add a cell to `notebooks/ch02.ipynb` that imports and exercises it. The notebook cells should follow the same numbered order as the listings (2.1, 2.2, ...) and include short prose cells that match the chapter plan's section headings.

   For visualization listings (2.7, 2.8), the helper goes in `src/ch02/viz.py` but the notebook is where it actually renders inline — that is the reader's "see it work" moment.

4. **Write the code.** Match the chapter plan listing exactly: same variable names, same annotations, same line ordering. The chapter prose references them.

5. **Validate with agents.** After the file compiles and runs locally:

   ```
   # In Claude Code
   "Run style-check on src/ch02/<file>.py"
   "Run listing-check on src/ch02/<file>.py"
   ```

   Fix anything flagged before moving on. ERROR-level findings block; WARNING-level should be addressed unless there is a deliberate reason.

6. **Run the notebook end-to-end.** After updating `notebooks/ch02.ipynb` with the new cell, restart the kernel and run all cells. This catches import drift between the package and the notebook early — the most common failure in a hybrid setup.

7. **Generate tests.** Once a module's listings are in place:

   ```
   "Run test-gen on src/ch02/<file>.py"
   ```

   `test-gen` produces shape/dtype/interface stubs in `tests/`. Flesh them out or commit them as-is — they are scaffolding, not the final test suite. Tests target the `src/ch02/` modules, not the notebook.

8. **Update the chapter-guide agent.** Whenever a listing's content or its role (type-along / API illustration / provided utility) changes, mirror the change in `agents/chapter-02-guide.md`. The agent's listing roadmap table is the contract between the chapter's pedagogy and the reader's companion — drift here silently degrades the experimental reader path.

9. **Run resource-check at major milestones.** After completing a section (e.g., all of §2.5 done), ask:

   ```
   "Resource-check this — will it run on Colab T4?"
   ```

   The MuJoCo sim itself is light, but image observations and dataset caches can balloon GPU memory once you start batching at scale. Catch this early.

---

## 4. The Export Contract

`make_pickplace_dataloader()`, `normalize()`, and `denormalize()` are the **frozen interface** between this chapter's code and Chapter 3. Their signatures are specified in `docs/chapter_2_plan.md` §2.5.5 and in the README.

**Implications:**
- Do not rename them.
- Do not add or remove required arguments without updating the chapter plan first.
- The `stats` dict structure is part of the contract — Chapter 3 expects `stats["observation.state"]["mean"]` etc.
- The `dataset_id` parameter on `make_pickplace_dataloader` is the extension point for later chapters that swap in custom datasets. Keep it as the first positional argument.

When implementation reveals a problem with this interface, update the plan in `docs/chapter_2_plan.md` *and* in `../lrm-book/chapter_2/chapter_2_structure_and_plan.md` *together*, in the same commit. Both must stay in sync.

Run `chapter-continuity` after any change to the export surface:

```
"Run chapter-continuity"
```

---

## 5. Figures

The chapter plan lists 8 figures (2.1–2.8). Figure 2.1 is the book-wide roadmap recap with the Chapter 2 stage highlighted — reuse the figure-1.7 source rather than re-rendering. The other seven are produced inside `notebooks/ch02.ipynb` using helper functions defined in `src/ch02/viz.py`, and exported to `figures/` for the chapter draft.

For each figure:
- The plotting helper (matplotlib code, style choices, axis labels) lives in `src/ch02/viz.py` so it is importable and testable.
- The notebook cell calls the helper, sets a fixed random seed, and calls `plt.savefig("figures/figure_2_<n>_<slug>.png", dpi=300)`.
- Verify the rendered image matches the description in `docs/chapter_2_plan.md`.

This split keeps the visual logic under version control as plain Python while still letting the reader see plots inline in the notebook.

`MANNING_STYLE.md` §4 has the production rules. `../lrm-book/FIGURE_STYLE_GUIDE.md` has full details on palette, axis labels, and caption format.

---

## 6. Definition of Done

A listing is done when:
- [ ] The code matches the plan's listing content
- [ ] Annotations are contiguous from `#A` and explanations are 1–2 sentences each
- [ ] `style-check` passes
- [ ] `listing-check` passes
- [ ] The corresponding notebook cell imports it and runs cleanly
- [ ] `test-gen` has produced a stub test (even if minimal)

A module is done when:
- [ ] All of its listings are done
- [ ] Tests in `tests/` cover its public interface
- [ ] `resource-check` flags no critical issues

The chapter is done when:
- [ ] All 11 listings implemented in `src/ch02/` and validated
- [ ] `notebooks/ch02.ipynb` runs top-to-bottom from a fresh kernel
- [ ] All 8 figures rendered and saved to `figures/` (figure 2.1 reused from chapter 1's roadmap)
- [ ] `agents/chapter-02-guide.md` listing roadmap matches `docs/chapter_2_plan.md` Listings Summary (titles, order, roles)
- [ ] `chapter-continuity` confirms the Ch3 exports are stable
- [ ] A "full check" pass (all five agents) reports clean at ERROR severity

---

## 7. Working Notes

- **One listing at a time.** Don't write three files in parallel — the agents are sharper when invoked on a small surface, and rework is cheap.
- **Update the plan when reality diverges.** If a listing as written in the plan won't actually run, fix the plan first, then write the code. Code-first, plan-after creates silent drift.
- **Cross-reference the export contract before any rename.** Chapter 3's plan locks the function signatures — renaming `make_pickplace_dataloader` breaks the next chapter without warning.
- **Don't pre-emptively optimize.** Chapter 2's job is to be the simplest data pipeline that works on a laptop. Performance work belongs in later chapters.
- **The notebook is not a dumping ground.** Anything more than ~10 lines or used in more than one cell belongs in `src/ch02/`. The notebook should read like a walkthrough, not a script.
- **Clear notebook outputs before committing.** `jupyter nbconvert --clear-output` (or a pre-commit hook) keeps `ch02.ipynb` diffs reviewable.

---

## 8. Where Things Live

| Need | Path |
|------|------|
| What to build | `docs/chapter_2_plan.md` |
| How to format it | `docs/MANNING_STYLE.md` |
| Importable Python (export contract) | `src/ch02/` |
| Reader's canonical walkthrough | `notebooks/ch02.ipynb` |
| Reader's optional companion agent | `agents/chapter-02-guide.md` (committed; symlinked into `.claude/agents/` at setup) |
| Author-tooling agents | `../lrm-code-agents/agents/` (symlinked into `.claude/agents/` after setup) |
| Author agent router / config | `../lrm-code-agents/CLAUDE.md`, `defaults.yml` |
| Rendered figures | `figures/` |
| Tests (author/CI infrastructure) | `tests/` |
| Full prose/code style rules | `../lrm-book/STYLEGUIDE.md` |
| Figure rendering rules | `../lrm-book/FIGURE_STYLE_GUIDE.md` |
| Chapter prose draft (future) | `../lrm-book/chapter_2/` |
