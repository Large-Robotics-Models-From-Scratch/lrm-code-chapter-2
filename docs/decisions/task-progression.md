# Decision: Task Progression Across Chapters

**Date:** 2026-05-16
**Status:** Accepted
**Scope:** How the carrier task and simulator are used from Ch 2 through Ch 11.

## TL;DR

**One simulator, one embodiment, one task family — cover to cover.** The carrier task is `PickCubeSO100-v1` (ManiSkill3, SO-100 arm). It anchors Chapters 2–5, expands to a multi-task suite in Chapter 6, drives RL in Chapter 7, becomes the sim-to-real digital twin in Chapter 9, and hands off to real SO-101 hardware in Chapters 10–11.

## The carrier task and why it's the carrier

`PickCubeSO100-v1` is single-object, fixed start, fixed target, no language conditioning. It is the simplest version of the task family the book ends on. Pick-and-place is complex enough to expose why hand-coded rules struggle (contact dynamics, gripper timing, recovery from misalignment) and simple enough to train on a laptop in an evening.

The SO-100 arm in sim is kinematically very close to the SO-101 hardware the reader will deploy on in Chapters 10–11 — same 6-DOF action space, same gripper command, same observation structure. The mental model transfers without re-internalization. Chapter 9 is the explicit place where the small remaining gap (servo tuning, slight kinematic differences) gets named and bridged.

## Per-chapter progression

| Chapter | Task / data | Sim role | What's new in this chapter |
|---|---|---|---|
| **Ch 2 — Simulation & Data** | `PickCubeSO100-v1` env + `lerobot/svla_so101_pickplace` dataset | Introduce the Gymnasium API; produce action arrays for distribution visualization | Data pipeline; `normalize`/`denormalize`; the `make_pickplace_dataloader` Ch3 contract |
| **Ch 3 — Imitation Learning (BC)** | Same task, same dataset | Eval rollouts | Behavior cloning architecture |
| **Ch 4 — ACT** | Same | Eval rollouts | Action chunking via `delta_timestamps` |
| **Ch 5 — Diffusion Policy** | Same | Eval rollouts | Stochastic action generation |
| **Ch 6 — Scaling & Multi-task** | `PickCubeSO100-v1`, `StackCube-v1`, `PushCube-v1`, `PegInsertionSide-v1`, `PlugCharger-v1`, YCB-pick variants | Task breadth experiments | Multi-task heads; language conditioning where applicable |
| **Ch 7 — Reinforcement Learning** | `PickCubeSO100-v1` (PPO) | GPU-parallel rollouts — minutes on a 4090 | PPO/SAC fundamentals; reward shaping |
| **Ch 8 — Reasoning / Advanced** | TBD at chapter-plan time | Likely same task family | Architecture-specific |
| **Ch 9 — Sim-to-Real** | `SO100GraspCube-v1` digital twin | Sim with built-in domain randomization; deploy zero-shot to hardware | Domain randomization, greenscreen, the sim-to-real gap |
| **Ch 10 — Real Hardware** | SO-101 via lerobot's `SO100Follower` | None — physical arm | Hardware integration |
| **Ch 11 — Deployment** | SO-101 in production loop | None | End-to-end inference pipeline |

## Where sim might not carry forward

If a future chapter explores tasks too complex for sim — deformable objects, fluid manipulation, real-world visual diversity that's hard to randomize — we fall back to **dataset-only training**. The reader uses HF Hub datasets via the same `LeRobotDataset` and normalized `DataLoader` machinery built in Chapter 2. Sim rollout eval is replaced by held-out-trajectory eval (action MSE, success-rate-from-data metrics).

The data-pipeline skills from Chapter 2 carry through unchanged. Only the rollout loop is missing, and the book is honest about that when the situation arises.

## Optional escape hatch: SO-101 in sim

ManiSkill upstream does not ship SO-101 today. If a future chapter genuinely needs SO-101 in sim — for example, if the servo differences between SO-100 and SO-101 turn out to materially affect a learned policy's behavior — we have three options, in order of preference:

1. **Stay on SO-100 in sim.** The kinematic gap is small. Chapter 9's sim-to-real chapter is explicitly the place where the gap is named and addressed. This is the default.
2. **Reference `aalmuzairee/squint`.** MIT-licensed third-party SO-101 ManiSkill port. Use as-is, cite in the chapter's references.
3. **Fork `haosulab/ManiSkill` under the book's org.** Add SO-101 ourselves. ~1–2 days of work — copy the `so100/` directory, swap the URDF from `TheRobotStudio`, add a motion-planning subclass, add a task config entry. Maintain the fork for the book's shelf life.

Default plan: stay on SO-100 until concrete need forces a change. Re-evaluation, if it happens, is documented as a new decision record.

## References

- Carrier-task rationale (longer form): `docs/chapter_2_plan.md` § "Why pick-and-place from Chapter 2"
- Simulator choice and trade-offs: `docs/decisions/simulator-choice.md`
- Per-module API plan: `docs/design.md`
- Chapter dependency graph used by the agent toolkit: `lrm-code-agents/defaults.yml` `chapter_dependencies`
