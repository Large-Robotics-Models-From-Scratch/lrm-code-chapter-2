# Decision: Simulator Choice for the Book

**Date:** 2026-05-16
**Status:** Accepted
**Scope:** All chapters that use a robotics simulator (Ch 2–9).

## TL;DR

**Chosen:** ManiSkill3 (`mani-skill==3.0.1`), used standalone. We do **not** depend on `lerobot-sim2real` despite its apparent fit — that repo has no LICENSE file and pins lerobot to a pre-0.4.0 commit incompatible with our `lerobot==0.5.1` target. We read it for technique and vendor any glue scripts we need into this repo under Apache 2.0.

## Constraints

A simulator for this book must satisfy *all* of:

1. **SO-100 or SO-101 supported in sim today** — no relying on PRs that haven't merged
2. **Permissive license** — Apache 2.0, MIT, or BSD (no copyleft)
3. **Actively maintained** by a credible org or lab — not dormant
4. **Light enough for Colab T4 free tier** — readers without GPUs must be able to follow along
5. **Compatible with `lerobot==0.5.1`** — the dataset library we have committed to
6. **Pick-and-place task family** — the book's carrier task
7. **Extensible** — we can fork and add custom assets/tasks for later chapters
8. **RL-capable with reasonable wall-clock training time** — Chapter 7 teaches RL; multi-hour training per run is a disengagement risk

## Options considered

| Option | SO-101 today | License | Maintained | Colab T4 | RL speed | Verdict |
|---|---|---|---|---|---|---|
| **ManiSkill3** | SO-100 ✅ (SO-101 needs port) | Apache 2.0 | Active (UCSD) | Works — 7-cell Vulkan setup | GPU-parallel, minutes | **Chosen** |
| gym-lowcostrobot | SO-100 ✅ (close to SO-101) | Apache 2.0 | Dormant since Jan 2025 | Light install | CPU MuJoCo — hours | Rejected: maintenance risk + slow RL |
| LeIsaac (IsaacLab) | SO-101 native | Apache 2.0 | Active (HF) | Heavy CUDA install | Fast | Rejected: gates book to GPU users; pins `lerobot==0.4.1` |
| MetaWorld | ❌ Sawyer only | MIT | Active | Light install | Moderate | Rejected: wrong embodiment, breaks SO-101 continuity |
| LIBERO | ❌ Franka only | MIT | Active | Light install | Moderate | Rejected: wrong embodiment |
| No-sim / data-only | N/A | N/A | N/A | N/A | None | Rejected: RL chapter requires sim rollouts |

## Why ManiSkill3 wins

- **SO-100 is first-class.** `mani_skill/agents/robots/so100/` ships the robot config, URDF, and meshes. `PickCubeSO100-v1` and `SO100GraspCube-v1` are registered envs. A motion-planning solver and PPO baseline scripts are bundled.
- **GPU-parallel rollouts via SAPIEN.** Chapter 7's RL training becomes minutes on a 4090 instead of hours on CPU MuJoCo. This was the decisive factor once we confirmed RL is in the book.
- **Sim-to-real digital twin built in.** `SO100GraspCube-v1` includes domain randomization and greenscreen support out of the box. Chapter 9 teaches the actual sim2real techniques rather than just an API switch.
- **Active maintenance.** UCSD Hao Su lab, monthly commit cadence, v3.0.1 released 2026-04-21.
- **Larger task family for scaling.** `PickCube`, `StackCube`, `PushCube`, `PegInsertionSide`, `PlugCharger`, YCB-pick. Chapter 6's multi-task lessons stay in the same simulator.
- **Permissive licensing throughout.** ManiSkill Apache 2.0; SAPIEN MIT; bundled SO-100 URDF Apache 2.0 (from `TheRobotStudio/SO-ARM100`).

## Trade-offs we accept

- **Colab install is 7 cells, not 1.** Vulkan ICDs need manual placement before `pip install mani-skill`. The README documents the exact recipe (verbatim from ManiSkill's official quickstart notebook). Cost is one extra cell of reader friction in any sim-using notebook.
- **SO-101 is not in upstream ManiSkill.** We use SO-100 in sim throughout. The SO-100/SO-101 kinematic gap is small (same arm family, slightly different servos) and gets explicitly addressed in Chapter 9. If a future chapter genuinely needs SO-101 in sim, we fork ManiSkill or reference `aalmuzairee/squint` (MIT, a third-party SO-101 port).
- **No `lerobot-sim2real` dependency.** Two hard blockers: no LICENSE (legally non-redistributable) and stale lerobot pin (`lerobot.common.robots.*` imports broken in 0.5.1). We vendor what we need under Apache 2.0.
- **SAPIEN/GPU-parallel concepts intrude slightly earlier in the book** than vanilla MuJoCo would. We introduce them as advanced topics in Chapter 7, not as prerequisites.

## What would trigger a re-evaluation

- ManiSkill goes dormant (no commits for 6+ months) or loses UCSD backing.
- Colab drops Vulkan support, breaking the install path for free-tier readers.
- LeRobot itself ships an SO-100/SO-101 sim env in `lerobot.envs.*` (currently absent in 0.5.1).
- A new simulator emerges with SO-100 + light install + lerobot integration + active maintenance.

Re-evaluation means a new decision record, not a silent switch.

## References

- ManiSkill3 repo: <https://github.com/haosulab/ManiSkill>
- ManiSkill3 Colab quickstart: `examples/tutorials/1_quickstart.ipynb`
- SO-ARM100 (upstream URDF source): <https://github.com/TheRobotStudio/SO-ARM100>
- `lerobot-sim2real` (reference, not dependency): <https://github.com/StoneT2000/lerobot-sim2real>
- `aalmuzairee/squint` (third-party SO-101 port): <https://github.com/aalmuzairee/squint>
- Full dep pin rationale: `docs/design.md` §4
