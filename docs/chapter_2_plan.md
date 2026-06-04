# Chapter 2: Simulation & Data — Structure & Content Plan

## Archetype

**Primary:** Hands-on Setup (Raschka-style) — environment, data, and pipeline construction.

Code-heavy chapter. The reader installs tools, launches an SO-100 pick-and-place simulation, writes a scripted policy, loads expert data (recorded from real SO-101 teleop), and builds a reusable DataLoader. The embodiment family chosen here — a low-cost 6-DOF arm with a parallel-jaw gripper — is the same one used through Chapter 11: SO-100 in sim, SO-101 on hardware, with the small kinematic gap addressed explicitly in Chapter 9.

### A note on the listings

Listings in this chapter fall into three roles:

- **Type-along teaching code** — the reader writes, runs, and understands these line by line. The scripted policy, the random-agent loop, the normalization functions, and the stats computation are the conceptual core of the chapter.
- **API illustrations** — short library calls that show how Gymnasium and LeRobot work. The reader runs these but does not need to memorize the API surface; the framework documentation is the long-term reference.
- **Provided utilities** — visualization helpers and the DataLoader collate plumbing live in the `ch02` package and are imported, not typed. The book shows the implementation for transparency.

The Listings Summary table near the end of the chapter marks the role of each listing.

---

## Reader Experience: Notebook + Optional Agent Companion

This chapter ships with two reader-facing paths. The same convention applies to every later chapter — when scaffolding Chapter 3+, lift this section verbatim and adjust the chapter number.

- **`notebooks/chXX.ipynb` — the canonical path.** The book's prose maps cell-by-cell to this notebook. Type-along listings appear inline so the reader can read and rerun (or retype) them. API illustrations are short calls in their own cells. Provided utilities are imported from `src/chXX/` with a one-liner. This notebook is version-pinned and kept green at the author end; the reader should never have to debug dependency drift to make it run.
- **`agents/chapter-XX-guide.md` — an experimental companion.** A per-chapter Claude Code subagent definition (zero-padded chapter number, e.g. `chapter-02` for this chapter, per `MANNING_STYLE.md`). Canonical source lives in `agents/`; symlinked into `.claude/agents/` at reader setup time so Claude Code auto-discovers it. The agent walks the reader through the same listings in the same order. The added value over the notebook is dialogue: the reader can ask clarification questions, ask "why this," request a sharper explanation in their own framing, or get unstuck when something breaks. Scope is strictly self-contained to the chapter — the agent is not a tutor for the rest of the book and should not skate into the next chapter's material.

The agent's system prompt is a first-class deliverable, on par with the notebook. It must encode:

- The chapter's listing order and the role of each (type-along / API illustration / provided utility).
- Where to pause and check understanding before moving on (typically at the end of each section).
- When to defer back to the book or notebook rather than improvise an answer.
- What this chapter does *not* cover, so the agent declines to wander into later-chapter material.

This is Manning's first book to ship a chapter agent alongside the standard book + notebook artifacts. The workflow is explicitly framed as experimental in the front matter: the notebook is the default and is what we promise will work end-to-end; the agent is offered to readers who want guided, conversational learning. Tests in `tests/` are author/CI infrastructure — readers may run them as an install smoke check but are not expected to engage with them.

---

## Why pick-and-place from Chapter 2

The book's endpoint is a physical SO-101 on the reader's desk performing tasks like "hand me the pen" or "put the pen in the stand." For the reader's mental model to transfer cleanly from simulation through reinforcement learning, reasoning, sim-to-real, and deployment, the carrier task must share the same embodiment, action space, and observation pipeline at every step.

A 2D toy task like PushT would force the reader to absorb a gripper, 6-DOF control, 3D perception, language conditioning, and camera fusion all at once around Chapter 6 — exactly the point where readers fall off. Starting with single-object pick-and-place on the same arm the reader will eventually deploy means each subsequent chapter adds one axis of complexity rather than asking the reader to re-internalize the world.

This is the simplest version of the task family the reader will end on. One cube, fixed start pose, fixed target zone, no language. The full embodiment is in place from day one.

---

## Chapter Opening

### "This chapter covers" block (5 bullets)
- Setting up the SO-100 simulation environment using ManiSkill3 and the Gymnasium interface
- Understanding observations, actions, episodes, and rewards for a 6-DOF arm with a gripper
- Writing a scripted pick-and-place policy as a state machine, and observing its failure modes
- Loading, inspecting, and visualizing expert demonstration data from the LeRobot Hub
- Building a normalized DataLoader that becomes the data contract for Chapter 3

### Hook paragraphs (2 paragraphs)
- **Paragraph 1:** A robot policy is only as good as the data that trains it — and only as transferable as the embodiment that produced it. Before writing a single line of model code, you need three things: a simulated arm to act in, demonstrations from that arm to learn from, and a pipeline that feeds both into training. This chapter builds all three, on the embodiment you will use for the rest of the book.
- **Paragraph 2:** You will work with `PickCubeSO100-v1`, a task where an SO-100 arm in simulation must grasp a cube from a starting position and release it inside a target zone. It looks simple, and that is the point. Pick-and-place is complex enough to expose why hand-coded heuristics struggle (contact dynamics, gripper timing, recovery from misalignment) yet simple enough to train on a laptop in an evening. By the end of this chapter, you will have a working data pipeline that Chapter 3 plugs directly into — and an embodiment family that does not change again until Chapter 11 (SO-100 in sim, SO-101 on hardware, bridged explicitly in Chapter 9).

### Section preview paragraph

Section 2.1 sets up the ManiSkill3 simulator and the Gymnasium interface, then runs a random agent to establish a performance floor. Section 2.2 introduces a scripted state-machine policy and observes its failure modes — the motivation for learning from data. Section 2.3 loads expert demonstrations from the LeRobot Hub and walks through the dataset's feature schema, including the `delta_timestamps` mechanism that later chapters use for action chunking. Section 2.4 visualizes the expert data side-by-side with the scripted and random baselines, making the policy-gap concrete. Section 2.5 closes the loop: normalization statistics, the `normalize`/`denormalize` functions, and the `make_pickplace_dataloader` export that Chapter 3 imports unchanged.

**Figure 2.1: Where this chapter sits in the book**
- Reuse the book-wide roadmap diagram from Figure 1.7 with the Chapter 2 stage highlighted ("Simulation & Data"). Stages: Foundations (Ch 1-2, current) → Architecture & Imitation (Ch 3-5) → Scaling (Ch 6-7) → Advanced (Ch 8-9) → Deployment (Ch 10-11).
- Caption: "The book's five-part progression with the current chapter highlighted. Chapter 2 builds the simulation environment, scripted-policy baseline, and normalized data pipeline that every later chapter consumes. By the end of this chapter, the embodiment and data interface are fixed for the remaining nine chapters."

---

## Section 2.1: The SO-100 Pick-and-Place Environment

**Purpose:** Install the tooling, launch the simulation, and teach the Gymnasium API through direct interaction with the pick-and-place task.

**Target length:** ~5 pages

**Content:**

### 2.1.1 Installation and First Launch
- Install `lerobot` and `mani-skill` (and their SAPIEN dependency)
- Create a `PickCubeSO100-v1` environment instance configured for image observations
- Call `env.reset()` and inspect the observation dictionary
- Call `env.step(action)` and inspect the return tuple

### 2.1.2 The Gymnasium Interface
- Define the core loop: `reset → step → step → ... → done`
- **Observation space:** What the agent sees — joint positions and velocities for the six arm joints, end-effector pose, a grasped flag, the goal-zone center, the cube's world pose (under `state_dict` mode), and an RGB camera view (under `rgb` mode); see Table 2.1 for the exact keys and shapes
- **Action space:** What the agent can do — a 6-DOF continuous action in `[-1, 1]`, one delta per SO-100 joint (the gripper is joint 6, not a separate dimension)
- **Episode:** One complete pick-and-place attempt from reset to termination
- **Step:** A single (observation, action, reward, next_observation) transition at the sim's control frequency
- **Reward:** Pick-and-place uses a shaped reward — distance to cube while approaching, lift bonus during grasp, distance-to-target while transporting, and a success bonus when the cube enters the target zone

### 2.1.3 Running a Random Agent
- Sample random actions from the action space and run several episodes
- Random agents almost never solve pick-and-place — the success rate is essentially zero
- This establishes the performance floor and motivates everything that follows

Listing 2.1 installs the simulation libraries and constructs a `PickCubeSO100-v1` environment instance with image observations enabled. The action space matches the SO-100's 6-DOF joint command structure and carries forward to every learned policy in later chapters.

**Listing 2.1: Installing the SO-100 sim and creating the environment**
```python
# pip install lerobot mani-skill
import gymnasium as gym                          #A
import mani_skill.envs                           #B
import matplotlib.pyplot as plt

env = gym.make(                                  #C
    "PickCubeSO100-v1",
    obs_mode="rgb",                              #D
    control_mode="pd_joint_delta_pos",           #E
    render_mode="rgb_array",
)
obs, info = env.reset(seed=42)                   #F
print(f"Observation keys: {list(obs.keys())}")
print(f"Action space: {env.action_space}")      #G

frame = env.render()[0].cpu().numpy()            #H
plt.imshow(frame); plt.axis("off"); plt.show()
```
- #A Provides the standard `reset` / `step` interface used by every env in this book — sim, real-hardware wrapper, or benchmark
- #B Importing `mani_skill.envs` registers `PickCubeSO100-v1` and the rest of the SO-100 task family
- #C Create the SO-100 pick-and-place environment
- #D Return RGB camera observations (alternatives: `"state"`, `"rgbd"`, `"state_dict"`)
- #E Joint-space delta actions in the same format the SO-100 hardware expects
- #F Reset returns the initial observation dictionary and an info dict
- #G Action is `Box(6,)` — one delta per SO-100 joint, gripper included as joint 6
- #H `env.render()` returns `(num_envs, H, W, 3)`; `[0]` drops the leading vector-env dim, `.cpu().numpy()` makes it matplotlib-friendly. The reader sees the arm, the cube, and the target zone in the same view their later policies will get

**Expected output:**
```
Observation keys: ['agent', 'extra', 'sensor_param', 'sensor_data']
Action space: Box(-1.0, 1.0, (6,), float32)
```
*Verified against `mani-skill==3.0.1`. `sensor_param` holds camera intrinsics/extrinsics; `sensor_data.base_camera.rgb` is the actual image. See Table 2.1 for the per-key shapes.*

**Editorial note (implementation):** Table 2.1 above is ground-truthed against the live `PickCubeSO100-v1` env. Listings 2.3–2.4 (§2.2) still show the scripted policy against gym-lowcostrobot-style keys (`arm_qpos`, `cube_pos`); those listings will be aligned to the ManiSkill `state_dict` schema when PR 3 lands.

The `run_random_agent` function in listing 2.2 executes the Gymnasium interaction loop with uniformly sampled actions and reports the success rate over a fixed number of episodes. This is the performance floor every learned policy must clear.

**Listing 2.2: Running a random agent on PickCubeSO100-v1**
```python
import numpy as np
from ch02.viz import record_env_video

def run_random_agent(env, n_episodes=10):
    successes, returns = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_return = 0.0
        done = False
        while not done:
            action = env.action_space.sample()           #A
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        successes += int(info.get("success", False))    #B
        returns.append(ep_return)
    return successes / n_episodes, np.mean(returns)

success_rate, mean_return = run_random_agent(env)
print(f"Random agent: success={success_rate:.0%} "
      f"return={mean_return:.2f}")                       #C

record_env_video(env, n_steps=200, seed=0, fps=20) #D
```
- #A Sample uniformly from the 6-DOF continuous action space
- #B Tally a success if the env's terminal info reports the cube landed in the target zone
- #C Expect near-zero success — flailing the arm rarely grasps anything
- #D Record a ~10-second MP4 of the random policy (auto-reset on truncation, so the clip spans ~4 episodes since the env's horizon is ~50 steps). Returns `IPython.display.Video` for inline playback in Jupyter/Colab. Watching the arm flail makes "random actions" concrete in a way the success-rate number never can

**Expected output:**
```
Random agent: success=0% return=-0.42
```
*Numbers vary by seed; what matters is that success rate is effectively zero and return is negative (the env's distance-shaping reward dominates).*

**Callout Box: "WHAT IS GYMNASIUM?"**
- Gymnasium (formerly OpenAI Gym) is the standard Python API for simulation and reinforcement-learning environments.
- Every env exposes `reset()` and `step(action)` — a universal interface regardless of the task or embodiment.
- LeRobot and `mani-skill` register their environments as Gymnasium envs, so the same code patterns work for sim, real-hardware wrappers, and benchmark tasks across the ecosystem.

**Callout Box: "WHY SO-100 IN SIM, SO-101 ON HARDWARE?"**
- ManiSkill3 ships the SO-100 as a first-class robot (`mani_skill/agents/robots/so100/`). SO-101 is the newer revision with slightly different servos and tuning.
- Observation and action interfaces are identical, so policy training is unaffected.
- The residual kinematic gap is real but small, and isolating it to a single sim-to-real chapter (Chapter 9) is cleaner pedagogy than pretending it does not exist.
- This is the canonical sim-to-real problem in miniature, and Chapter 9 is built around it.

**Callout Box: "PITFALL — SAPIEN/Vulkan setup on Colab"**
- ManiSkill3 renders through SAPIEN, which needs Vulkan installable client drivers (ICDs). Colab runtimes don't ship them by default.
- The symptom is a confusing `vk::Result::eErrorIncompatibleDriver` or `Cannot find any valid ICD` traceback the first time `env.reset()` runs — *not* when ManiSkill imports.
- Run the seven-line Vulkan setup recipe documented in the repo's README before installing `mani-skill`. The recipe is copied verbatim from ManiSkill's official quickstart notebook and works on the free-tier T4.
- On local Linux/macOS this is rarely an issue — system Vulkan drivers are usually already installed.

**Figure 2.2: The SO-100 Pick-and-Place Task**
- Annotated screenshot showing the arm in start pose, the cube on the workspace, the target zone outlined on the table, and an inset of the env's `base_camera` RGB view (the single sim camera; the expert dataset in §2.3 has two real-world views recorded from teleop).
- Caption: "The `PickCubeSO100-v1` task. A 6-DOF arm with a parallel-jaw gripper (SO-100 in ManiSkill3) must grasp the cube and release it inside the target zone. The reward combines approach distance, grasp success, transport distance, and a discrete success bonus on placement. SO-100 is the simulation embodiment; SO-101 is the hardware sibling the chapter's dataset was recorded on and the deployment chapters target — the small kinematic gap is addressed in Chapter 9."

**Figure 2.3: The Gymnasium Loop**
- Flow diagram: `reset()` → observation → agent selects action → `step(action)` → (obs, reward, terminated, truncated, info) → loop back or done.
- Caption: "The Gymnasium interaction loop. The agent receives an observation, selects an action, and the environment returns the next observation, a scalar reward, and termination flags. Every environment in this book — sim and real — exposes this interface."

**Table 2.1: SO-100 Observation and Action Spaces**

Observations are nested dicts; keys below are dotted paths (e.g., `agent.qpos`). ManiSkill envs are GPU-vectorizable, so each tensor carries a leading `(num_envs,)` dim — `1` for the single-env construction in Listing 2.1, dropped from the per-env shapes below for readability.

Observation keys depend on the env's `obs_mode`. With `obs_mode="state_dict"` (used by the scripted policy in §2.2) ManiSkill exposes:

| Component | Shape | Type | Description |
|-----------|-------|------|-------------|
| `agent.qpos` | (6,) | float32 | Joint positions in radians (6 SO-100 joints; gripper is joint 6) |
| `agent.qvel` | (6,) | float32 | Joint velocities |
| `extra.tcp_pose` | (7,) | float32 | End-effector world pose `(x, y, z, qw, qx, qy, qz)` |
| `extra.goal_pos` | (3,) | float32 | Target zone center `(x, y, z)` |
| `extra.is_grasped` | scalar | bool | Whether the gripper is currently holding the cube |
| `extra.obj_pose` | (7,) | float32 | Cube world pose; exposed under `obs_mode="state_dict"` (used by the §2.2 scripted policy) |
| `sensor_data.base_camera.rgb` | (128, 128, 3) | uint8 | RGB camera under `obs_mode="rgb"`; the default sim env ships a single view (the dataset in §2.3 has two — see §2.3 for the asymmetry) |
| Action | (6,) | float32 | One position delta per SO-100 joint (gripper is joint 6) |

**Exercise 2.1: Joint-space vs. end-effector-space control.** Call `make_env(control_mode="pd_ee_delta_pose")` instead of the default joint-space mode and re-run the random agent. The action space shape changes from `Box(6,)` (one delta per SO-100 joint) to a 7-dim Cartesian delta pose plus gripper command. Do random rollouts succeed any more often? Why or why not? *Tip: end-effector control hides one form of difficulty (joint coordination) while exposing another (workspace boundaries).*

**Transition:** "The random agent has no chance. Can a simple rule do better?"

---

## Section 2.2: A Scripted Policy

**Purpose:** Give the reader agency by writing a heuristic pick-and-place controller as an explicit state machine. Show how even a "reasonable" hand-coded approach has sharp failure modes — the motivation for learned policies.

**Target length:** ~4 pages

**Content:**

### 2.2.1 Designing the Heuristic
- Pick-and-place is naturally multi-phase. Decompose it: **approach** the cube from above → **descend** to grasp height → **close** the gripper → **lift** clear of the table → **transport** above the target → **descend** → **release**.
- Each phase is a target end-effector pose. PickCubeSO100-v1 ships with joint-space control modes only, so we lean on ManiSkill's bundled `SO100ArmMotionPlanningSolver` to convert Cartesian targets into joint trajectories. The scripted controller stays declarative — "here is where the gripper should go next" — and the planner handles IK.

### 2.2.2 Implementation
- Define `run_scripted_episode(planner, grasp_pose, goal_pos)` that issues one motion-planner call per phase: three keyframes to descend onto the cube, one gripper close, one lift, two keyframes to the goal, one gripper open.
- Wrap that in `run_scripted_agent(env, n_episodes)` which constructs the planner per episode, computes a top-down grasp pose from the cube spawn position, runs the seven phases, and tallies success from `env.unwrapped.evaluate()`.

### 2.2.3 Where the Heuristic Fails
- The scripted policy succeeds in nominal conditions (cube placed cleanly in the workspace) but fails on:
  - Grasps where the gripper closes a frame too early or too late
  - Cube positions near the workspace edge where the arm's reachability is limited
  - Cases where contact pushes the cube out of position during descent
- It is open-loop within each phase — no recovery once the gripper misses
- Key insight: even a "smart" heuristic plateaus far below expert performance. Learned policies can capture the subtle contact dynamics and recovery behaviors that rules miss.

Listing 2.3 defines `run_scripted_episode`, the seven-phase controller expressed as keyframe poses for the chapter's IK planner (`IKMotionPlanner` in `ch02.scripted`). Each `move_to_pose_with_screw` call solves SAPIEN's pinocchio IK at evenly spaced Cartesian waypoints between the current and target end-effector poses, clamps the result to joint limits, and feeds the per-waypoint qpos straight to the `pd_joint_pos` controller. The final `hold` lets the arm settle so the env's `is_robot_static` flag flips true. The scripted code itself stays declarative — "here is where the gripper should go next" — and the planner handles IK.

**Listing 2.3: A multi-phase scripted pick-and-place policy**
```python
import numpy as np
import sapien
from ch02.viz import plot_phase_keyframes

def run_scripted_episode(
        planner, grasp_pose, goal_pos):
    """Seven phases via the IK planner.

    approach → descend → grasp → lift → transport → place → hold.
    """
    quat = grasp_pose.q
    goal = np.asarray(goal_pos)

    planner.move_to_pose_with_screw(                #A
        sapien.Pose([0, 0, 0.10]) * grasp_pose)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.03]) * grasp_pose)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.01]) * grasp_pose)
    planner.close_gripper(gripper_state=-0.8)       #B
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.15]) * grasp_pose)
    planner.move_to_pose_with_screw(                #C
        sapien.Pose(goal + np.array([0, 0, 0.05]), quat))
    planner.move_to_pose_with_screw(
        sapien.Pose(goal, quat))
    planner.hold(n_steps=30)                        #D

sample_grasp = sapien.Pose([0.0, 0.0, 0.05])        #E
sample_goal = np.array([-0.2, 0.2, 0.02])
plot_phase_keyframes(sample_grasp, sample_goal)
```
- #A Three target poses at decreasing Z above the grasp point (approach, descend, grasp). The planner solves IK and follows the joint trajectory for each
- #B Partial close (`gripper_state=-0.8`) applies contact pressure on the cube without overclosing
- #C Transport and place keep the grasp orientation. Transport hovers 5 cm above goal; place sets the cube down at goal. (mplib used 15 cm + 2 cm; our pure-IK path is less reach-efficient, so we tuck the keyframes closer to the workspace.)
- #D Hold instead of release. The success check only requires `is_obj_placed AND is_robot_static` — dropping the cube from any height bounces it past the 1.56 cm goal threshold, so we keep the gripper closed and let the arm settle
- #E The function defines the policy but doesn't execute it. Plot the six motion-planner targets in 3-D for a sample `(grasp_pose, goal_pos)` so the reader can see the keyframe trajectory the planner will follow (approach high → descend → grasp → lift → transport → place)

Listing 2.4 evaluates the scripted policy. The package's `run_scripted_agent` constructs an `IKMotionPlanner` once per episode, computes a top-down grasp pose from the cube's spawn position, runs the seven phases, and tallies success from the env's `evaluate()`. The env must be built with `control_mode="pd_joint_pos"` so the IK qpos targets feed straight to the controller. Reported success rates land in the 20–40% band — well above the random floor but well below expert teleoperation. The chapter's pure-IK planner can't always thread the workspace as cleanly as a sampling-based planner (it has no collision avoidance, no retry-from-elsewhere), and that's the gap learned policies eventually close.

**Listing 2.4: Evaluating the scripted policy**
```python
from ch02.scripted import run_scripted_agent           #A
from ch02.viz import record_scripted_episode_video

env = make_env(
    obs_mode="state", control_mode="pd_joint_pos", render_mode=None,
)                                                     #B
rate = run_scripted_agent(env, n_episodes=10)
print(f"Scripted agent success rate: {rate:.0%}")

video_env = make_env(                                 #C
    obs_mode="state", control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
record_scripted_episode_video(video_env, n_episodes=1, fps=20)
```
- #A Provided utility from `ch02.scripted`. Internally constructs the IK planner, computes the grasp pose per episode, and counts successes
- #B `pd_joint_pos` is required — the IK planner emits per-waypoint absolute joint targets, not deltas. `state` mode is fine since the scripted policy reads cube and goal positions directly from `env.unwrapped`
- #C Record one scripted rollout as MP4 for inline playback. The helper monkey-patches `env.step` to capture `env.render()` after each physics step — same pattern as `capture_scripted_actions`. A separate `rgb_array`-mode env is needed because the success-rate env above was built with `render_mode=None` for throughput

**Expected output:**
```
Scripted agent success rate: 50%
```
*Anywhere from ~30% to ~70% is normal; the exact rate depends on cube spawn positions across seeds. The point is "substantially better than 0%, substantially below 100%" — the heuristic plateau.*

**Callout Box: "WHY NOT JUST ENGINEER A BETTER HEURISTIC?"**
- You could add error recovery, force feedback, retry-on-miss, and a finer phase decomposition.
- But every improvement requires more hand-coded rules, and each new edge case (different cube color, novel target position, occlusion) compounds the complexity.
- This is the "long tail" argument from Chapter 1 in miniature: heuristics plateau, learned policies keep improving with more data.

**Exercise 2.2: Grasp-failure recovery.** Extend `scripted_policy` with a new phase that detects when the cube has not risen above a height threshold after the `lift` phase, returns to `approach`, and retries the grasp. How much does success rate improve? At what point does adding phases stop helping? *Tip: instrument `state` with a `retries` counter and bail out at three retries.*

**Transition:** "The scripted policy shows what one person's intuition can achieve in an afternoon. Expert demonstrations show what practiced teleoperation looks like as data."

---

## Section 2.3: The LeRobot Dataset Standard

**Purpose:** Load expert demonstrations from the Hub, understand the LeRobot format, and see what successful pick-and-place looks like numerically.

**Target length:** ~5 pages

**Content:**

### 2.3.1 Loading from the Hub
- Use `lerobot.datasets.LeRobotDataset` to load the SO-101 pick-and-place expert dataset.
- The dataset is hosted on Hugging Face Hub — a single line downloads the parquet shards and video archives.
- Inspect dataset length, number of episodes, and feature names.
- The chapter pins to `lerobot/svla_so101_pickplace`: 50 episodes of real-hardware teleoperated pick-and-place at 30 fps, 11,939 frames, two USB-camera streams (`up` and `side`).

### 2.3.2 The Feature Schema
- Each sample is a dictionary with keys including `observation.state`, `observation.images.up`, `observation.images.side`, `action`, `episode_index`, `frame_index`, `index`, `timestamp`, `task`, `task_index`.
- `observation.state` is the 6-DOF SO-101 joint state (gripper is joint 6). `action` is the recorded 6-DOF teleoperation command — same shape, same convention.
- Images come from two USB cameras: `up` (looking down at the workspace) and `side` (across the workspace). LeRobot 0.5.x returns them as `float32` in `[0, 1]` (already normalized on read).
- `episode_index` groups frames into episodes; `frame_index` orders them within each episode; `index` is the position in the full dataset.
- `task` is the natural-language task description (`"pink lego brick into the transparent box"`); `task_index` is its integer id. Late chapters use these for language conditioning.

### 2.3.3 Understanding delta_timestamps
- LeRobot's `delta_timestamps` mechanism: request observation/action at relative time offsets from the current frame.
- Example: `delta_timestamps = {"observation.state": [-0.1, 0.0], "action": [0.0, 0.033, 0.067]}` returns the previous and current observations plus the current and two future actions (offsets in seconds; 0.033 ≈ one frame at the dataset's 30 fps control rate).
- This is the mechanism for **action chunking** — predicting multiple future actions from a single observation, which is essential for high-frequency, smooth control in later chapters.

### 2.3.4 Episode Structure
- Iterate through one episode and print shapes at each step.
- Show how episodes begin (reset state) and end (success or timeout).
- Count steps per episode — episodes have variable length depending on how quickly the teleoperator completed the task.

Listing 2.5 instantiates the `LeRobotDataset` for the SO-101 pick-and-place expert dataset and prints its overall size and feature schema. The download is one-time; subsequent calls hit the local Hugging Face cache.

**Listing 2.5: Loading the SO-101 pick-and-place expert dataset**
```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",                  #A
)
print(f"Total frames: {len(dataset)}")               #B
print(f"Episodes: {dataset.num_episodes}")
print(f"Features: {list(dataset.features.keys())}") #C
```
- #A 50 episodes of teleoperated SO-101 pick-and-place collected on real hardware — the chapter's canonical expert dataset
- #B Total number of (observation, action) frames across all episodes
- #C Feature names include the state vector, two camera streams, the action, and episode/frame metadata

**Expected output:**
```
Total frames: 11939
Episodes: 50
Features: ['action', 'observation.state', 'observation.images.up',
           'observation.images.side', 'timestamp', 'frame_index',
           'episode_index', 'index', 'task_index']
```
*Counts and feature names match the v3.0 codebase version of `svla_so101_pickplace` (verified against `lerobot==0.5.1`; 30 fps; `task` is added by `__getitem__` but not part of `features.keys()`). If the dataset is republished under a different revision, both may shift.*

Listing 2.6 indexes into the dataset to inspect a single frame's shape and dtype, then collects every frame belonging to episode zero to confirm episode-length variation. This concretizes the abstract feature schema for the reader.

**Listing 2.6: Inspecting a single frame and one episode**
```python
frame = dataset[0]                                   #A
for key, val in frame.items():
    if hasattr(val, "shape"):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {key}: {val}")

ep_indices = [i for i in range(len(dataset))         #B
              if dataset[i]["episode_index"] == 0]
print(f"\nEpisode 0 length: {len(ep_indices)} steps")
```
- #A Access a single frame by integer index — returns a dictionary of tensors and scalars
- #B Collect all frame indices belonging to episode 0 to inspect a complete trajectory

**Expected output:**
```
  observation.images.up: shape=torch.Size([3, 480, 640]), dtype=torch.float32
  observation.images.side: shape=torch.Size([3, 480, 640]), dtype=torch.float32
  action: shape=torch.Size([6]), dtype=torch.float32
  observation.state: shape=torch.Size([6]), dtype=torch.float32
  timestamp: shape=torch.Size([]), dtype=torch.float32
  frame_index: shape=torch.Size([]), dtype=torch.int64
  episode_index: shape=torch.Size([]), dtype=torch.int64
  index: shape=torch.Size([]), dtype=torch.int64
  task_index: shape=torch.Size([]), dtype=torch.int64
  task: pink lego brick into the transparent box

Episode 0 length: 303 steps
```
*Image channel-first ordering (C, H, W) is the LeRobot convention; matplotlib needs the transpose. Images come back as `float32` in `[0, 1]` — LeRobot 0.5.x normalizes on read. The `task` field carries the natural-language goal for the demo, set up for the language-conditioned policies in later chapters.*

**Table 2.2: LeRobot SO-101 Pick-and-Place Dataset Features**

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `observation.state` | (6,) | float32 | Six SO-101 joint positions (gripper is joint 6) |
| `observation.images.up` | (3, 480, 640) | float32 | Up-mounted camera (looking down at workspace), normalized to `[0, 1]` |
| `observation.images.side` | (3, 480, 640) | float32 | Side-mounted camera (across workspace), normalized to `[0, 1]` |
| `action` | (6,) | float32 | Recorded teleoperation joint command |
| `episode_index` | scalar | int64 | Episode this frame belongs to |
| `frame_index` | scalar | int64 | Position within the episode |
| `index` | scalar | int64 | Position within the full dataset (across episodes) |
| `timestamp` | scalar | float32 | Seconds from episode start |
| `task` | string | — | Natural-language task description, e.g. `"pink lego brick into the transparent box"` |
| `task_index` | scalar | int64 | Integer id for the task string (1 unique task in this dataset) |

**Callout Box: "WHAT IS delta_timestamps?"**
- LeRobot's mechanism for requesting data at relative time offsets from a given frame.
- Setting `delta_timestamps={"action": [0.0, 0.033, 0.067]}` returns the current action plus the next two future actions, stacked into shape `(3, 6)` (offsets in seconds at the dataset's 30 fps rate).
- This enables **action chunking** — predicting a short sequence of future actions instead of one at a time.
- Action chunking improves policy smoothness and is the foundation for the ACT and diffusion-policy heads in Chapters 4 and 5.

**Exercise 2.3: Single-episode statistics.** Compute mean and std of `observation.state` across the frames of a single episode, then compare to the statistics across the whole dataset. How large is the discrepancy on the joint dimensions? On the gripper? *Tip: this is the basic argument for normalizing on the *dataset*, not per-batch — and it foreshadows why a tiny fine-tuning dataset can destabilize a model that was trained with population-level statistics.*

**Transition:** "Numbers in a table tell you the data's shape. Plots and rendered frames tell you what success actually looks like."

---

## Section 2.4: Visualizing the Data

**Purpose:** Build visual intuition for expert pick-and-place behavior. Compare expert, scripted, and random action distributions per-joint to make the quality gap concrete.

**Target length:** ~4 pages

**Content:**

### 2.4.1 Rendering Expert Episodes
- Extract keyframes from one expert episode at regular intervals.
- Display `up` and `side` camera views side by side as a two-row filmstrip — the reader sees what the cube looked like at each phase of a successful grasp-transport-release.
- Annotate with the action taken at each keyframe.

### 2.4.2 Action Distributions: Three-Way Per-Joint Comparison
- Collect actions from the expert dataset, the scripted policy, and a random agent.
- Mechanism note: the scripted policy is trajectory-level (the motion planner steps multiple frames per call), so its per-step actions are recovered by **intercepting `env.step`** — see `capture_scripted_actions` in `ch02.viz`. Random and expert actions use the natural sources (env action-space sampling, dataset frame indexing).
- For each of the six action dimensions (five SO-100 arm joints + gripper as joint 6), plot a histogram with the three sources overlaid.
- Expert distributions show structured, multi-modal patterns — different approach directions produce different joint trajectories. Scripted actions cluster around a few values. Random actions are uniform.
- The point is the **shape of the action-space distribution** — random is uniform, scripted is concentrated, expert is structured and multi-modal — not a same-task quality comparison. The expert dataset is teleoperated "lego in box" on real SO-101 hardware; the scripted policy is solving the sim's cube-in-zone task. See §2.6 for the task-gap discussion and the Chapter 4 BC eval contract that follows from it.

### 2.4.3 Expert Joint Trajectories
- Plot per-joint angle over time for several expert episodes on the same axes.
- Episodes overlap in shape (all start, lift, transport, place) but differ in timing and amplitude depending on initial cube pose.
- This visually confirms that a successful policy must be conditional on the observation, not a single memorized trajectory.
- Figure 2.6 is produced by `plot_joint_trajectories(dataset, episode_indices)` from `ch02.viz`; the notebook §2.4 cell calls it directly.

Listing 2.7 replays one expert episode as an MP4 with both camera views stacked side-by-side, using `record_dataset_episode_video` from `ch02.viz`. The helper pulls per-frame images from the dataset in trajectory order, converts each tensor from `(3, H, W)` float `[0, 1]` to `(H, W, 3)` uint8, concatenates the cameras horizontally, and writes an MP4 at the dataset's native 30 fps. The result is roughly a 10-second clip that plays inline in Jupyter or Colab.

**Listing 2.7: Playing an expert episode with both camera views**
```python
from ch02.viz import record_dataset_episode_video

record_dataset_episode_video(                   #A
    dataset, episode_idx=0,
    cameras=("up", "side"), fps=30,             #B
)
```
- #A Returns `IPython.display.Video` (embed=True, base64-encoded into the notebook) so the cell renders the player inline without an external file dependency
- #B Stacking `("up", "side")` produces a 480×1280 frame per timestep; pass `("up",)` for a single-camera preview (used by the post-Listing 2.6 cell)

Listing 2.8 collects actions from three sources — the dataset's expert teleoperation, the scripted IK policy from §2.2, and uniform random sampling — and overlays per-dimension histograms via `plot_action_distributions`. The dataset-side actions are a simple `np.stack`; the env-side actions go through `ch02.viz`.

A subtle gotcha the helper resolves: the three sources live in *different units*. Expert actions are degrees of joint position (LeRobot's teleop convention, roughly [-100, +100]). Scripted actions are radians (`pd_joint_pos`, roughly [-π, +π]). Random actions are normalized deltas (`pd_joint_delta_pos`, [-1, +1]). Overlaying raw values on a common x-axis crushes scripted and random into a single bar near zero on the expert's degree scale; nothing about the distribution shapes is readable. `plot_action_distributions` defaults to *per-source min-max rescaling into [-1, +1]* so the shapes line up without lying about absolute magnitudes.

**Listing 2.8: Per-joint action distributions — expert vs. scripted vs. random**
```python
import numpy as np
from ch02.viz import (
    collect_actions,
    capture_scripted_actions,
    plot_action_distributions,
)

expert = np.stack(                                   #A
    [np.asarray(dataset[i]["action"]) for i in range(len(dataset))]
)
random_ = collect_actions(env, None, n_episodes=3)   #B
scripted = capture_scripted_actions(env, n_episodes=3)

plot_action_distributions(                           #C
    expert, scripted, random_,
    save_path="figures/figure_2_5_action_distributions.png",
)
```
- #A Expert actions are pure dataset indexing — no env, no rollout — so they're an inline `np.stack`, not a wrapped helper
- #B `collect_actions(env, None, ...)` samples uniformly from `env.action_space` (random branch); `capture_scripted_actions` intercepts `env.step` to record the joint commands the scripted policy issues mid-trajectory
- #C The helper handles the 2×3 grid layout, per-source min-max normalization into [-1, +1] (so shapes are comparable on a common axis), axis labels, and savefig at 300 DPI

**Figure 2.4: Expert Pick-and-Place Keyframes**
- A 2x6 grid: top row is the `up` camera, bottom row is the `side` camera, columns are six keyframes from one expert episode spanning approach → grasp → lift → transport → place → release.
- Caption: "Keyframes from one expert episode. The `up` view shows the workspace from above; the `side` view shows the gripper-to-cube contact geometry. A learned policy must capture both perspectives to handle objects whose position is only partially visible from above."

**Figure 2.5: Per-Joint Action Distributions — Expert vs. Scripted vs. Random**
- Six overlapping histograms in a 2x3 grid, one per SO-101 joint, comparing the three policies.
- Caption: "Action distributions for each of the six action dimensions. Expert actions show structured, multi-modal clusters that reflect different grasp strategies. The scripted policy produces a simpler, lower-variance pattern. Random actions are uniform across the range. The gap between the scripted and expert histograms is what a learned policy must close."

**Figure 2.6: Expert Joint Trajectories**
- Six small line plots, one per arm joint, showing joint angle over time for five overlaid expert episodes.
- Caption: "Joint trajectories from five expert episodes. Episodes share the same coarse structure (approach, lift, transport, place) but diverge in timing and amplitude depending on the initial cube pose. A successful policy must be conditional on the current observation, not a single memorized trajectory."

**Transition:** "You have the environment, a baseline, and expert data. The final step is packaging that data into a form your neural network can consume."

---

## Section 2.5: The Data Pipeline

**Purpose:** Build the DataLoader that Chapter 3 will use. Implement normalization from first principles, then connect to LeRobot's stats. This section's output is the chapter's API contract with downstream chapters.

**Target length:** ~5 pages

**Content:**

### 2.5.1 Why Normalize?
- Neural networks train faster and more stably when inputs are zero-centered and unit-scaled.
- Joint angles range over a few radians while gripper commands range over `[-1, 1]` — different scales cause uneven gradient flow.
- Image pixels (0–255) and state vectors live on completely different scales and need separate treatment.
- Two normalization strategies: z-score (mean/std) for state and actions, min-max scaling to `[0, 1]` for image pixels.

### 2.5.2 Computing Statistics from Scratch
- Iterate through the dataset and compute per-feature mean, std, min, max for `observation.state` and `action`.
- Implement `normalize(x, stats, key)` and `denormalize(x, stats, key)` functions.
- Verify round-trip: `denormalize(normalize(x)) ≈ x` to floating-point precision.
- Images are not z-scored. LeRobot 0.5.x already returns them as `float32` in `[0, 1]`, so the collate function passes them through. The legacy `uint8` path is kept for callers who feed in an older dataset version.

### 2.5.3 LeRobot's meta/stats.json
- Reveal that LeRobot ships precomputed statistics in each dataset's `meta/stats.json`.
- Load them and compare against the manually computed values — they should agree.
- Going forward, use the precomputed stats for convenience, but the reader now understands what they contain and why.

### 2.5.4 The DataLoader Wrapper
- Wrap the LeRobotDataset in a PyTorch DataLoader with batching and shuffling.
- Apply normalization in a collate function: z-score state and action; images pass through (already `float32` in `[0, 1]` from LeRobot 0.5.x); the `task` strings stay as a list; everything else gets `torch.tensor`-stacked.
- Export `make_pickplace_dataloader(dataset_id, batch_size, shuffle)` — the API Chapter 3 imports.

### 2.5.5 Verifying the Pipeline
- Draw a batch, print shapes, and confirm: state has mean ≈ 0 and std ≈ 1 across the batch, action similarly, and image pixels are in `[0, 1]`.
- Denormalize a sample action and verify it is in the original radian/gripper range.
- This is the smoke test that confirms the pipeline is correct before any training begins.

The `compute_stats` function in listing 2.9 iterates through every frame in the dataset and computes per-dimension mean, std, min, and max for both `observation.state` and `action`. These statistics are the only piece of training-time state that has to survive into inference.

**Listing 2.9: Computing normalization statistics manually**
```python
import torch

def compute_stats(dataset):
    """Compute per-feature mean, std, min, max for state and action."""
    states, actions = [], []
    for i in range(len(dataset)):
        frame = dataset[i]
        states.append(frame["observation.state"])  #A
        actions.append(frame["action"])
    states = torch.stack(states)
    actions = torch.stack(actions)

    return {
        "observation.state": {
            "mean": states.mean(0), "std": states.std(0),
            "min": states.min(0).values, "max": states.max(0).values,
        },
        "action": {
            "mean": actions.mean(0), "std": actions.std(0),
            "min": actions.min(0).values, "max": actions.max(0).values,
        },
    }
```
- #A Collect every state and action across the entire dataset to compute exact statistics

Listing 2.10 defines the `normalize` and `denormalize` functions used everywhere in the book and verifies the round-trip is lossless to floating-point precision. Both functions take the same `(x, stats, key)` signature so they compose cleanly with the dataloader's collate function.

**Listing 2.10: Normalize and denormalize functions**
```python
import torch

def normalize(x, stats, key):
    """Z-score normalization: (x - mean) / std."""
    return (x - stats[key]["mean"]) / (
        stats[key]["std"] + 1e-8)            #A

def denormalize(x, stats, key):
    """Inverse z-score normalization."""
    return x * (stats[key]["std"] + 1e-8) + stats[key]["mean"]

sample = dataset[0]["observation.state"]
normed = normalize(sample, stats, "observation.state")
recovered = denormalize(normed, stats, "observation.state")
assert torch.allclose(sample, recovered, atol=1e-5)  #B
```
- #A The small epsilon prevents division by zero for features that are constant across the dataset
- #B Verify the round-trip is lossless to floating-point precision

Listing 2.11 ties the dataset, the statistics, and the normalization functions into the chapter's primary export: `make_pickplace_dataloader`. The function lives in `ch02.pipeline` and is imported as `from ch02.pipeline import make_pickplace_dataloader`. Chapter 3 imports it the same way and treats the signature as frozen — the listing is shown here so the reader understands what the function does, not because the reader writes it from scratch.

**Listing 2.11: Building the DataLoader — the Chapter 3 API contract**
```python
import torch
from torch.utils.data import DataLoader
from ch02.dataset import DEFAULT_DATASET_ID, load_dataset
from ch02.pipeline import compute_stats, normalize

def make_pickplace_dataloader(
    dataset_id=DEFAULT_DATASET_ID,            #A
    batch_size=64,
    shuffle=True,
):
    """Return a normalized DataLoader and stats."""
    dataset = load_dataset(dataset_id)
    stats = compute_stats(dataset)

    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            vals = [b[key] for b in batch]
            first = vals[0]
            if isinstance(first, str):        #B
                out[key] = vals
                continue
            if not isinstance(first, torch.Tensor):
                out[key] = torch.tensor(vals)
                continue
            stacked = torch.stack(vals)
            if key in stats:                  #C
                stacked = normalize(stacked, stats, key)
            elif key.startswith("observation.images"):
                if stacked.dtype == torch.uint8:
                    stacked = stacked.float() / 255.0
                if stacked.dtype.is_floating_point:
                    stacked = stacked.clamp(0, 1)  #D
            out[key] = stacked
        return out

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate_fn,
        num_workers=0,
    )
    return loader, stats

loader, stats = make_pickplace_dataloader(batch_size=4)
batch = next(iter(loader))
print(f"state mean: "
      f"{batch['observation.state'].mean(0)}")  #E
print(f"image range: "
      f"[{batch['observation.images.up'].min():.2f}, "
      f"{batch['observation.images.up'].max():.2f}]")
```
- #A `DEFAULT_DATASET_ID` is `lerobot/svla_so101_pickplace`; the parameter is the extension point later chapters use to swap in their own datasets
- #B `task` is a string field; keep it as a list rather than trying to stack into a tensor
- #C Z-score normalize state and action using precomputed stats
- #D Enforces the Ch3 `[0, 1]` image contract — handles the legacy uint8 path and clamps any float drift
- #E After normalization, per-dimension state mean should be near zero across a batch

**Expected output:**
```
state mean per dim: tensor([-0.01,  0.02,  0.00, -0.03,  0.01,  0.00])
image range: [0.00, 1.00]
```
*Each state-mean component should sit within ~±0.1 of zero (within-batch noise around the dataset-level mean of zero); image pixel range should land in `[0, 1]`. Any wildly off value here means a stat dict was mis-keyed or a feature slipped past the collate function.*

**Callout Box: "Z-SCORE vs. MIN-MAX NORMALIZATION"**
- **Z-score** — `(x - mean) / std`. Centers at zero, scales by spread. Preferred when features are roughly Gaussian, as joint angles and recorded actions tend to be.
- **Min-max** — `(x - min) / (max - min)`. Scales to `[0, 1]`. Preferred when bounded outputs are needed, as with image pixels handed to a vision encoder expecting `[0, 1]`.
- This book uses z-score for state and actions and `x / 255` for images. The choice is consistent across all chapters.

**Callout Box: "THE CHAPTER 3 CONTRACT"**
- Chapter 3 imports `make_pickplace_dataloader()`, `normalize()`, and `denormalize()` directly from `ch02`.
- It expects batches with keys `observation.state` (normalized), `observation.images.up`, `observation.images.side` (in `[0, 1]`), and `action` (normalized). The `task` string and per-frame metadata (`episode_index`, `frame_index`, `index`, `timestamp`, `task_index`) ride along untouched.
- After the model predicts a normalized action, `denormalize()` converts it back to environment scale before calling `env.step()`.
- Treat the function signature `make_pickplace_dataloader(dataset_id, batch_size, shuffle)` as frozen — renaming or re-ordering arguments breaks every downstream chapter.

**Figure 2.7: The Normalization Round-Trip**
- Flow diagram: raw observation/action → `normalize(x, stats)` → zero-centered input → model prediction (normalized) → `denormalize(ŷ, stats)` → environment-scale action → `env.step()`.
- Caption: "The normalization round-trip. Observations and actions are z-score normalized before entering the model. Predicted actions are denormalized back to radians and gripper commands before being sent to the simulator. The stats dictionary bridges both directions and is the only piece of training-time state that has to survive into inference."

**Figure 2.8: End-to-End Data Pipeline**
- Flow diagram: Hugging Face Hub → LeRobotDataset → compute_stats → DataLoader with normalize-in-collate → training batch {state, images, action} → Chapter 3 model.
- Caption: "The complete data pipeline built in this chapter. Expert demonstrations are loaded from the Hub, normalization statistics are computed once, and a DataLoader applies the right normalization to each feature type at batch time. `make_pickplace_dataloader()` encapsulates the entire flow as the API contract for Chapter 3."

**Exercise 2.4: Min-max normalization on actions.** Replace `compute_stats` and `normalize` with min-max scaling — `(x - min) / (max - min)` — for `observation.state` and `action` only. Verify the round-trip still works to floating-point precision. Then draw one batch and compare per-dimension batch means against z-score: which is closer to zero? Why does that matter for downstream learning? *Tip: think about how the loss gradient flows through a denormalized action prediction at the start of training, before the model has learned anything.*

### 2.5.6 Pipeline performance across hardware

**Table 2.3: Chapter 2 pipeline timings**

The point of this table is to calibrate the reader's expectations before they hit Chapter 3's training loop. None of these numbers are large; everything in Chapter 2 fits comfortably on a free-tier T4.

| Step | Colab T4 (free) | RTX 4090 | A100 (Colab Pro) |
|------|-----------------|----------|------------------|
| First-time dataset download (~1.5 GB) | ~3-5 min (network bound) | ~1-2 min | ~1-2 min |
| `compute_stats` (single pass over dataset) | ~30-60 s | ~10-20 s | ~10-20 s |
| One pipeline smoke (batch of 32 with images) | < 2 s | < 1 s | < 1 s |
| Random-agent rollout (10 episodes) | ~20-40 s | ~10-15 s | ~10-15 s |
| Scripted-agent rollout (10 episodes) | ~30-60 s | ~15-25 s | ~15-25 s |

Numbers are wall-clock and approximate; rerun on first install and pin in the README on chapter release. The Vulkan setup on Colab adds a one-time ~20-30 second cell. Subsequent runs in the same kernel session have no setup overhead.

---

## Section 2.6: Summary

**Target length:** ~1 page

Comprehensive bulleted summary:

- `PickCubeSO100-v1` is a single-object pick-and-place task on a 6-DOF arm with a parallel-jaw gripper, served by ManiSkill3 over SAPIEN. It is the carrier task and the carrier embodiment for the rest of the book.
- The Gymnasium API provides a universal interface — `reset()` returns an initial observation, `step(action)` returns the next observation, reward, and termination flags. Every environment in this book, in sim and on hardware, exposes this interface.
- A random agent on a 6-DOF arm essentially never succeeds. A multi-phase scripted policy (approach → descend → grasp → lift → transport → place → release) raises the success rate but plateaus far below expert performance because it cannot recover from misalignment or adapt to contact dynamics.
- Expert demonstrations from teleoperation are stored in the LeRobot dataset format on Hugging Face Hub. Each frame includes joint state, two camera views, the recorded action, episode/frame metadata, and a natural-language task string (used from Ch 6 for language conditioning).
- LeRobot's `delta_timestamps` mechanism enables requesting data at relative time offsets — the foundation for action chunking in later chapters.
- Visualizing expert action distributions per-joint reveals structured, multi-modal patterns that neither random sampling nor a hand-coded heuristic can reproduce. Visualizing expert joint trajectories shows that successful policies must be conditional, not memorized.
- Neural networks need normalized inputs. Z-score normalization is applied to state and action; images are scaled to `[0, 1]`. Denormalization recovers environment-scale actions for use with `env.step()`.
- The chapter's primary export is `make_pickplace_dataloader(dataset_id, batch_size, shuffle)`, parameterized on `dataset_id` so later chapters can swap in custom datasets without changing the interface.
- **The sim env and the dataset are different tasks.** `PickCubeSO100-v1` spawns a generic colored cube with a fixed target zone; `svla_so101_pickplace` is real-hardware teleop of "pink lego brick into a transparent box." They share the 6-DOF action interface and not much else, which determines the Chapter 4 eval strategy — see the callout below.
- Chapter 3 picks up exactly where this chapter ends: same DataLoader, same normalization conventions, same SO-100 embodiment. It adds the part this chapter deliberately left out — a model that learns to predict actions from observations, using a vision-language backbone and the first incarnation of a generative robot policy.

**Callout Box: "SIM ENV ≠ DATASET TASK — THE CH4 BC EVAL CONTRACT"**
- The sim env (`PickCubeSO100-v1`) and the expert dataset (`svla_so101_pickplace`) live in different scenes: different object (generic cube vs. pink lego), different goal geometry (target zone vs. transparent box), different visual distribution (sim render vs. real hardware), different camera setup. They share **only** the 6-DOF SO-100/SO-101 action interface and the 30 Hz control rate.
- Therefore: **Chapter 4's behavior-cloning eval is action MSE on a held-out dataset split**, not rollout success against the sim env. A BC policy trained on dataset actions cannot be fairly scored by attempting the env's cube task — that would conflate policy quality with task transfer.
- The §2.4 per-joint histogram is illustrative of action-space *structure* (random uniform → scripted concentrated → expert multi-modal), not a same-task policy benchmark.
- What Chapter 3+ inherits from this chapter is the **data-and-action contract** (normalized DataLoader, action-space scale, denormalization for `env.step()`), not a task-level metric. Building a sim env that mirrors the dataset task is out of scope and not on the book's roadmap.

---

## Further Reading

A short, opinionated reading list for readers who want to dig deeper into the topics this chapter touched. References are grouped by chapter section; ordering within each group is "start here" first.

**Simulator and embodiment**
- *ManiSkill3: GPU-Parallelized Robotics Simulation and Rendering* (Tao et al., 2024) — the paper behind the simulator we use. arXiv:2410.00425.
- *SO-ARM100 hardware design* (The Robot Studio, ongoing) — the open-source CAD and BOM for the physical arm. <https://github.com/TheRobotStudio/SO-ARM100>
- *SO-101 release notes* (Hugging Face LeRobot, 2026) — what changed from SO-100 and why. Linked from the LeRobot docs.

**LeRobot dataset format**
- *LeRobot: State-of-the-art ML for real-world robotics in PyTorch* (Cadene et al., 2024) — the framework's design document. <https://github.com/huggingface/lerobot>
- *Hub datasets for robot learning* — the SO-100/SO-101 pick-and-place datasets on the Hub, with episode-count and quality notes per dataset card. <https://huggingface.co/datasets?other=lerobot>

**Pick-and-place as a learning benchmark**
- *Implicit Behavioral Cloning* (Florence et al., CoRL 2021) — the PushT task and an early energy-based BC formulation. We chose pick-and-place over PushT for the reasons in §2.2's "Why pick-and-place from Chapter 2," but PushT is a useful foil. arXiv:2109.00137.
- *Action Chunking with Transformers* (Zhao et al., RSS 2023) — the original ACT paper. The `delta_timestamps` mechanism in §2.3 is the data-side enabler for ACT, which Chapter 4 builds. arXiv:2304.13705.
- *Diffusion Policy* (Chi et al., RSS 2023) — Chapter 5's continuous-action approach uses the same normalized DataLoader we built here. arXiv:2303.04137.

**Normalization for robot policies**
- *Behavior Cloning, Pitfalls, and Lessons* — surveys of why action-distribution normalization matters more for robot policies than for image classifiers. Pointer in the chapter's source repo.

---

## Listings Summary

| Listing | Title | Section | Mode |
|---------|-------|---------|------|
| 2.1 | Installing the SO-100 sim and creating the environment | 2.1 | API illustration |
| 2.2 | Running a random agent on PickPlaceCube | 2.1 | **Type-along** |
| 2.3 | A multi-phase scripted pick-and-place policy | 2.2 | **Type-along** |
| 2.4 | Evaluating the scripted policy | 2.2 | API illustration |
| 2.5 | Loading the SO-101 pick-and-place expert dataset | 2.3 | API illustration |
| 2.6 | Inspecting a single frame and one episode | 2.3 | API illustration |
| 2.7 | Rendering expert keyframes from both camera views | 2.4 | Provided utility (`ch02.viz`) |
| 2.8 | Per-joint action distributions — expert vs. scripted vs. random | 2.4 | Provided utility (`ch02.viz`) |
| 2.9 | Computing normalization statistics manually | 2.5 | **Type-along** |
| 2.10 | Normalize and denormalize functions | 2.5 | **Type-along** |
| 2.11 | Building the DataLoader — the Chapter 3 API contract | 2.5 | Provided utility (`ch02.pipeline`) |

## Figure Summary

| Figure | Description | Type | Section |
|--------|------------|------|---------|
| 2.1 | Where this chapter sits in the book (roadmap recap, Ch2 highlighted) | Stage diagram | Opening |
| 2.2 | The SO-100 Pick-and-Place Task | Annotated screenshot | 2.1 |
| 2.3 | The Gymnasium Loop | Flow diagram | 2.1 |
| 2.4 | Expert Pick-and-Place Keyframes | Two-row image filmstrip | 2.4 |
| 2.5 | Per-Joint Action Distributions | Overlaid histograms | 2.4 |
| 2.6 | Expert Joint Trajectories | Line plots | 2.4 |
| 2.7 | The Normalization Round-Trip | Flow diagram | 2.5 |
| 2.8 | End-to-End Data Pipeline | Flow diagram | 2.5 |

## Callout Box Summary

| Callout | Section | Purpose |
|---------|---------|---------|
| "WHAT IS GYMNASIUM?" | 2.1 | Define the simulation API for ML readers |
| "WHY SO-100 IN SIM, SO-101 ON HARDWARE?" | 2.1 | Address the embodiment-gap question up front |
| "PITFALL — SAPIEN/Vulkan setup on Colab" | 2.1 | Surface the most common first-time install failure mode |
| "WHY NOT JUST ENGINEER A BETTER HEURISTIC?" | 2.2 | Connect to Chapter 1's long-tail argument |
| "WHAT IS delta_timestamps?" | 2.3 | Explain LeRobot's temporal indexing mechanism |
| "Z-SCORE vs. MIN-MAX NORMALIZATION" | 2.5 | Normalization strategy rationale |
| "THE CHAPTER 3 CONTRACT" | 2.5 | Define the API boundary between chapters |
| "SIM ENV ≠ DATASET TASK — THE CH4 BC EVAL CONTRACT" | 2.6 | Flag the sim/dataset task gap and pin Ch4's eval as held-out action MSE |

## Exercises Summary

| Exercise | Title | Section | Difficulty | Solution location |
|----------|-------|---------|------------|-------------------|
| 2.1 | Joint-space vs. end-effector-space control | 2.1 (end) | Light | Appendix |
| 2.2 | Grasp-failure recovery in the scripted policy | 2.2 (end) | Light-moderate | Appendix |
| 2.3 | Single-episode vs. dataset-level statistics | 2.3 (end) | Light | Appendix |
| 2.4 | Min-max normalization on actions | 2.5 (end) | Light-moderate | Appendix |

## Table Summary

| Table | Description | Section |
|-------|-------------|---------|
| 2.1 | SO-100 Observation and Action Spaces | 2.1 |
| 2.2 | LeRobot SO-101 Pick-and-Place Dataset Features | 2.3 |
| 2.3 | Chapter 2 pipeline timings across hardware | 2.5.6 |

## Chapter 2 Exports for Chapter 3

| Export | Type | Purpose |
|--------|------|---------|
| `make_pickplace_dataloader(dataset_id, batch_size, shuffle)` | function | Returns `(DataLoader, stats_dict)` with normalized batches; `dataset_id` parameterized so later chapters can swap datasets without changing the signature |
| `normalize(x, stats, key)` | function | Z-score normalize a tensor using precomputed stats |
| `denormalize(x, stats, key)` | function | Inverse z-score to recover environment-scale values |
| `stats` | dict | Per-feature `{mean, std, min, max}` for `observation.state` and `action` |

---

## Estimated Length: 22–25 pages (Manning format)
## Estimated Word Count: ~9,000–11,000 words
