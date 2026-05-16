# Chapter 2: Simulation & Data — Structure & Content Plan

## Archetype

**Primary:** Hands-on Setup (Raschka-style) — environment, data, and pipeline construction.

Code-heavy chapter. The reader installs tools, launches an SO-101 pick-and-place simulation, writes a scripted policy, loads expert data, and builds a reusable DataLoader. The embodiment chosen here — a low-cost 6-DOF arm with a parallel-jaw gripper — is the same embodiment used through Chapter 11, so every concept the reader learns about observations, actions, and rewards transfers directly.

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
- Setting up the SO-101 simulation environment using `gym-lowcostrobot` and the Gymnasium interface
- Understanding observations, actions, episodes, and rewards for a 6-DOF arm with a gripper
- Writing a scripted pick-and-place policy as a state machine, and observing its failure modes
- Loading, inspecting, and visualizing expert demonstration data from the LeRobot Hub
- Building a normalized DataLoader that becomes the data contract for Chapter 3

### Hook paragraphs (2 paragraphs)
- **Paragraph 1:** A robot policy is only as good as the data that trains it — and only as transferable as the embodiment that produced it. Before writing a single line of model code, you need three things: a simulated arm to act in, demonstrations from that arm to learn from, and a pipeline that feeds both into training. This chapter builds all three, on the embodiment you will use for the rest of the book.
- **Paragraph 2:** You will work with `PickPlaceCube`, a task where an SO-101 arm in simulation must grasp a cube from a starting position and release it inside a target zone. It looks simple, and that is the point. Pick-and-place is complex enough to expose why hand-coded heuristics struggle (contact dynamics, gripper timing, recovery from misalignment) yet simple enough to train on a laptop in an evening. By the end of this chapter, you will have a working data pipeline that Chapter 3 plugs directly into — and an embodiment that does not change again until Chapter 11.

**Figure 2.1: Where this chapter sits in the book**
- Reuse the book-wide roadmap diagram from Figure 1.7 with the Chapter 2 stage highlighted ("Simulation & Data"). Stages: Foundations (Ch 1-2, current) → Architecture & Imitation (Ch 3-5) → Scaling (Ch 6-7) → Advanced (Ch 8-9) → Deployment (Ch 10-11).
- Caption: "The book's five-part progression with the current chapter highlighted. Chapter 2 builds the simulation environment, scripted-policy baseline, and normalized data pipeline that every later chapter consumes. By the end of this chapter, the embodiment and data interface are fixed for the remaining nine chapters."

---

## Section 2.1: The SO-101 Pick-and-Place Environment

**Purpose:** Install the tooling, launch the simulation, and teach the Gymnasium API through direct interaction with the pick-and-place task.

**Target length:** ~5 pages

**Content:**

### 2.1.1 Installation and First Launch
- Install `lerobot` and `gym-lowcostrobot` (and their MuJoCo dependency)
- Create a `PickPlaceCube-v0` environment instance configured for image observations
- Call `env.reset()` and inspect the observation dictionary
- Call `env.step(action)` and inspect the return tuple

### 2.1.2 The Gymnasium Interface
- Define the core loop: `reset → step → step → ... → done`
- **Observation space:** What the agent sees — joint positions and velocities for six arm joints, gripper state, cube pose, target pose, and (optionally) two camera images (top-down and wrist-mounted)
- **Action space:** What the agent can do — a 7-DOF continuous action (six joint position deltas plus one gripper command in `[-1, 1]`)
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

env = gym.make(                                  #C
    "PickCubeSO100-v1",
    obs_mode="rgb",                              #D
    control_mode="pd_joint_delta_pos",           #E
    render_mode="rgb_array",
)
obs, info = env.reset(seed=42)                   #F
print(f"Observation keys: {list(obs.keys())}")
print(f"Action space: {env.action_space}")      #G
```
- #A Gymnasium API
- #B Importing `mani_skill.envs` registers `PickCubeSO100-v1` and the rest of the SO-100 task family
- #C Create the SO-100 pick-and-place environment
- #D Return RGB camera observations (alternatives: `"state"`, `"rgbd"`, `"state_dict"`)
- #E Joint-space delta actions in the same format the SO-100 hardware expects
- #F Reset returns the initial observation dictionary and an info dict
- #G Action is `Box(6,)` — one delta per SO-100 joint, gripper included as joint 6

**Editorial note (implementation):** Table 2.1's observation/action schema and listings 2.3–2.4's scripted policy reference gym-lowcostrobot-style keys (`arm_qpos`, `cube_pos`, etc.); ManiSkill's actual observation dict uses different keys (`agent.qpos`, `extra.tcp_pose`, etc.) which will be confirmed and updated when PR 2 and PR 3 land.

The `run_random_agent` function in listing 2.2 executes the Gymnasium interaction loop with uniformly sampled actions and reports the success rate over a fixed number of episodes. This is the performance floor every learned policy must clear.

**Listing 2.2: Running a random agent on PickPlaceCube**
```python
import numpy as np

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
        successes += int(info.get("is_success", False)) #B
        returns.append(ep_return)
    return successes / n_episodes, np.mean(returns)

success_rate, mean_return = run_random_agent(env)
print(f"Random agent: success={success_rate:.0%} "
      f"return={mean_return:.2f}")                       #C
```
- #A Sample uniformly from the 7-DOF continuous action space
- #B The env reports success when the cube reaches the target zone
- #C Expect near-zero success — flailing the arm rarely grasps anything

**Callout Box: "WHAT IS GYMNASIUM?"**
- Gymnasium (formerly OpenAI Gym) is the standard Python API for simulation and reinforcement-learning environments.
- Every env exposes `reset()` and `step(action)` — a universal interface regardless of the task or embodiment.
- LeRobot and `gym-lowcostrobot` register their environments as Gymnasium envs, so the same code patterns work for sim, real-hardware wrappers, and benchmark tasks across the ecosystem.

**Callout Box: "WHY SO-100 IN SIM, SO-101 ON HARDWARE?"**
- The `gym-lowcostrobot` simulator was built for the SO-100 arm. SO-101 is the newer revision with slightly different servos and tuning.
- Observation and action interfaces are identical, so policy training is unaffected.
- The residual kinematic gap is real but small, and isolating it to a single sim-to-real chapter (Chapter 9) is cleaner pedagogy than pretending it does not exist.
- This is the canonical sim-to-real problem in miniature, and Chapter 9 is built around it.

**Figure 2.2: The SO-101 Pick-and-Place Task**
- Annotated screenshot showing the arm in start pose, the cube on the workspace, the target zone outlined on the table, and a small inset of the wrist-camera view.
- Caption: "The PickPlaceCube task. A 6-DOF arm with a parallel-jaw gripper (SO-100 URDF, branded as SO-101 for the book) must grasp the cube and release it inside the target zone. The reward combines approach distance, grasp success, transport distance, and a discrete success bonus on placement."

**Figure 2.3: The Gymnasium Loop**
- Flow diagram: `reset()` → observation → agent selects action → `step(action)` → (obs, reward, terminated, truncated, info) → loop back or done.
- Caption: "The Gymnasium interaction loop. The agent receives an observation, selects an action, and the environment returns the next observation, a scalar reward, and termination flags. Every environment in this book — sim and real — exposes this interface."

**Table 2.1: SO-101 Observation and Action Spaces**

| Component | Shape | Type | Description |
|-----------|-------|------|-------------|
| `arm_qpos` | (6,) | float32 | Joint positions in radians |
| `arm_qvel` | (6,) | float32 | Joint velocities |
| `cube_pos` | (3,) | float32 | Cube (x, y, z) world coordinates |
| `target_pos` | (3,) | float32 | Target zone (x, y, z) |
| `image_top` | (224, 224, 3) | uint8 | Top-down RGB camera |
| `image_wrist` | (224, 224, 3) | uint8 | Wrist-mounted RGB camera |
| Action | (7,) | float32 | Six joint position deltas + gripper command |

**Transition:** "The random agent has no chance. Can a simple rule do better?"

---

## Section 2.2: A Scripted Policy

**Purpose:** Give the reader agency by writing a heuristic pick-and-place controller as an explicit state machine. Show how even a "reasonable" hand-coded approach has sharp failure modes — the motivation for learned policies.

**Target length:** ~4 pages

**Content:**

### 2.2.1 Designing the Heuristic
- Pick-and-place is naturally multi-phase. Decompose it: **approach** the cube from above → **descend** to grasp height → **close** the gripper → **lift** clear of the table → **transport** above the target → **descend** → **release**.
- Represent each phase as a state with a target end-effector pose and a transition condition (distance threshold or gripper contact).
- Use the env's `end_effector` action mode if available, or compute a simple Cartesian-to-joint mapping using the env's exposed kinematics. Either way the scripted controller is short and explicit.

### 2.2.2 Implementation
- Track a phase index (0–6) across steps.
- At each step, compute the target end-effector position for the current phase, derive an action that moves toward it, and advance the phase index when the transition condition fires.
- Run the controller for multiple episodes and report success rate.

### 2.2.3 Where the Heuristic Fails
- The scripted policy succeeds in nominal conditions (cube placed cleanly in the workspace) but fails on:
  - Grasps where the gripper closes a frame too early or too late
  - Cube positions near the workspace edge where the arm's reachability is limited
  - Cases where contact pushes the cube out of position during descent
- It is open-loop within each phase — no recovery once the gripper misses
- Key insight: even a "smart" heuristic plateaus far below expert performance. Learned policies can capture the subtle contact dynamics and recovery behaviors that rules miss.

Listing 2.3 defines the seven-phase scripted controller as a single `scripted_policy` function. State is carried across calls in a small dictionary that records the current phase and frame counters — exactly the kind of bookkeeping a learned policy will replace with weights.

**Listing 2.3: A multi-phase scripted pick-and-place policy**
```python
import numpy as np

PHASES = ["approach", "descend", "grasp",
          "lift", "transport", "place", "release"]      #A

def scripted_policy(obs, state):
    """A simple state-machine controller for pick-and-place."""
    phase = state["phase"]
    ee_pos = obs["arm_qpos"][-3:]                       #B
    cube = obs["cube_pos"]
    target = obs["target_pos"]

    if phase == "approach":                              #C
        goal = cube + np.array([0.0, 0.0, 0.10])
        gripper = -1.0
        if np.linalg.norm(ee_pos - goal) < 0.01:
            state["phase"] = "descend"
    elif phase == "descend":
        goal = cube + np.array([0.0, 0.0, 0.005])
        gripper = -1.0
        if abs(ee_pos[2] - goal[2]) < 0.005:
            state["phase"] = "grasp"
    elif phase == "grasp":                               #D
        goal = ee_pos
        gripper = 1.0
        state["grasp_steps"] = state.get("grasp_steps", 0) + 1
        if state["grasp_steps"] >= 5:
            state["phase"] = "lift"
    elif phase == "lift":
        goal = cube + np.array([0.0, 0.0, 0.15])
        gripper = 1.0
        if ee_pos[2] >= goal[2] - 0.01:
            state["phase"] = "transport"
    elif phase == "transport":
        goal = target + np.array([0.0, 0.0, 0.15])
        gripper = 1.0
        if np.linalg.norm(ee_pos[:2] - goal[:2]) < 0.01:
            state["phase"] = "place"
    elif phase == "place":
        goal = target + np.array([0.0, 0.0, 0.02])
        gripper = 1.0
        if abs(ee_pos[2] - goal[2]) < 0.005:
            state["phase"] = "release"
    else:  # release
        goal = ee_pos
        gripper = -1.0                                   #E

    direction = goal - ee_pos
    joint_delta = np.clip(direction * 5.0, -1.0, 1.0)   #F
    return np.concatenate([joint_delta,
                           np.zeros(3),
                           [gripper]]).astype(np.float32)
```
- #A Seven phases of the state machine, in execution order
- #B End-effector position is the last three components of `arm_qpos` in this env's convention
- #C Approach: hover 10 cm above the cube with the gripper open
- #D Grasp: hold position and close the gripper for several frames to ensure contact
- #E Release the cube at the target
- #F Convert the desired Cartesian motion into a joint-space action, clipped to the env's range

The `run_scripted_agent` function in listing 2.4 mirrors the random-agent loop but threads the per-episode state dictionary through `scripted_policy`. Reported success rates climb above zero but settle well below what teleoperators achieve, motivating the data-driven approach.

**Listing 2.4: Evaluating the scripted policy**
```python
def run_scripted_agent(env, n_episodes=10):
    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        state = {"phase": "approach"}                    #A
        done = False
        while not done:
            action = scripted_policy(obs, state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(info.get("is_success", False))
    return successes / n_episodes

rate = run_scripted_agent(env)
print(f"Scripted agent success rate: {rate:.0%}")       #B
```
- #A The state machine carries its phase across steps via this dict
- #B Expect a moderate success rate — better than random, far from expert

**Callout Box: "WHY NOT JUST ENGINEER A BETTER HEURISTIC?"**
- You could add error recovery, force feedback, retry-on-miss, and a finer phase decomposition.
- But every improvement requires more hand-coded rules, and each new edge case (different cube color, novel target position, occlusion) compounds the complexity.
- This is the "long tail" argument from Chapter 1 in miniature: heuristics plateau, learned policies keep improving with more data.

**Transition:** "The scripted policy shows what one person's intuition can achieve in an afternoon. Expert demonstrations show what practiced teleoperation looks like as data."

---

## Section 2.3: The LeRobot Dataset Standard

**Purpose:** Load expert demonstrations from the Hub, understand the LeRobot format, and see what successful pick-and-place looks like numerically.

**Target length:** ~5 pages

**Content:**

### 2.3.1 Loading from the Hub
- Use `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` to load an SO-100 pick-and-place dataset.
- The dataset is hosted on Hugging Face Hub — a single line downloads the parquet shards and image archives.
- Inspect dataset length, number of episodes, and feature names.
- **Implementor note:** the exact dataset ID is pinned at implementation time from the available SO-100 pick-and-place datasets on the LeRobot Hub. The plan uses `lerobot/so100_pick_place` as a placeholder.

### 2.3.2 The Feature Schema
- Each sample is a dictionary with keys including `observation.state`, `observation.images.top`, `observation.images.wrist`, `action`, `episode_index`, `frame_index`, `timestamp`.
- `observation.state` is the 6-DOF joint state plus gripper. `action` is the recorded 7-DOF teleoperation command.
- Images come in two views (top-down third-person and wrist-mounted), matching the simulation observation structure.
- `episode_index` groups frames into episodes; `frame_index` orders them within each episode.

### 2.3.3 Understanding delta_timestamps
- LeRobot's `delta_timestamps` mechanism: request observation/action at relative time offsets from the current frame.
- Example: `delta_timestamps = {"observation.state": [-0.1, 0.0], "action": [0.0, 0.04, 0.08]}` returns the previous and current observations plus the current and two future actions.
- This is the mechanism for **action chunking** — predicting multiple future actions from a single observation, which is essential for high-frequency, smooth control in later chapters.

### 2.3.4 Episode Structure
- Iterate through one episode and print shapes at each step.
- Show how episodes begin (reset state) and end (success or timeout).
- Count steps per episode — episodes have variable length depending on how quickly the teleoperator completed the task.

Listing 2.5 instantiates the `LeRobotDataset` for the SO-100 pick-and-place dataset and prints its overall size and feature schema. The dataset ID is a placeholder — at implementation time, pin to a specific dataset from the LeRobot Hub.

**Listing 2.5: Loading the SO-101 pick-and-place expert dataset**
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "lerobot/so100_pick_place",                         #A
)
print(f"Total frames: {len(dataset)}")                  #B
print(f"Episodes: {dataset.num_episodes}")
print(f"Features: {list(dataset.features.keys())}")     #C
```
- #A Placeholder dataset ID — verify and replace with the chosen SO-100 pick-and-place dataset at implementation time
- #B Total number of (observation, action) frames across all episodes
- #C Feature names include the state vector, two camera streams, the action, and episode/frame metadata

Listing 2.6 indexes into the dataset to inspect a single frame's shape and dtype, then collects every frame belonging to episode zero to confirm episode-length variation. This concretizes the abstract feature schema for the reader.

**Listing 2.6: Inspecting a single frame and one episode**
```python
frame = dataset[0]                                      #A
for key, val in frame.items():
    if hasattr(val, "shape"):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {key}: {val}")

ep_indices = [i for i in range(len(dataset))            #B
              if dataset[i]["episode_index"] == 0]
print(f"\nEpisode 0 length: {len(ep_indices)} steps")
```
- #A Access a single frame by integer index — returns a dictionary of tensors and scalars
- #B Collect all frame indices belonging to episode 0 to inspect a complete trajectory

**Table 2.2: LeRobot SO-101 Pick-and-Place Dataset Features**

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `observation.state` | (7,) | float32 | Six joint positions + gripper state |
| `observation.images.top` | (3, 224, 224) | uint8 | Top-down third-person camera |
| `observation.images.wrist` | (3, 224, 224) | uint8 | Wrist-mounted camera |
| `action` | (7,) | float32 | Recorded teleoperation command |
| `episode_index` | scalar | int64 | Episode this frame belongs to |
| `frame_index` | scalar | int64 | Position within the episode |
| `timestamp` | scalar | float32 | Seconds from episode start |

**Callout Box: "WHAT IS delta_timestamps?"**
- LeRobot's mechanism for requesting data at relative time offsets from a given frame.
- Setting `delta_timestamps={"action": [0.0, 0.04, 0.08]}` returns the current action plus the next two future actions, stacked into shape `(3, 7)`.
- This enables **action chunking** — predicting a short sequence of future actions instead of one at a time.
- Action chunking improves policy smoothness and is the foundation for the ACT and diffusion-policy heads in Chapters 4 and 5.

**Transition:** "Numbers in a table tell you the data's shape. Plots and rendered frames tell you what success actually looks like."

---

## Section 2.4: Visualizing the Data

**Purpose:** Build visual intuition for expert pick-and-place behavior. Compare expert, scripted, and random action distributions per-joint to make the quality gap concrete.

**Target length:** ~4 pages

**Content:**

### 2.4.1 Rendering Expert Episodes
- Extract keyframes from one expert episode at regular intervals.
- Display top-down and wrist-camera views side by side as a two-row filmstrip — the reader sees what the cube looked like at each phase of a successful grasp-transport-release.
- Annotate with the action taken at each keyframe.

### 2.4.2 Action Distributions: Three-Way Per-Joint Comparison
- Collect actions from the expert dataset, the scripted policy, and a random agent.
- For each of the seven action dimensions (six joints + gripper), plot a histogram with the three sources overlaid.
- Expert distributions show structured, multi-modal patterns — different approach directions produce different joint trajectories. Scripted actions cluster around a few values. Random actions are uniform.
- The gap between scripted and expert histograms is what behavior cloning has to close.

### 2.4.3 Expert Joint Trajectories
- Plot per-joint angle over time for several expert episodes on the same axes.
- Episodes overlap in shape (all start, lift, transport, place) but differ in timing and amplitude depending on initial cube pose.
- This visually confirms that a successful policy must be conditional on the observation, not a single memorized trajectory.

Listing 2.7 implements `render_keyframes` as a utility in `ch02.viz`. Import it as `from ch02.viz import render_keyframes` and call it on the dataset; the source is shown here for transparency about how the helper tiles the two camera views.

**Listing 2.7: Rendering expert keyframes from both camera views**
```python
import matplotlib.pyplot as plt

def render_keyframes(dataset, episode_idx=0, n_frames=6):
    """Display top and wrist camera keyframes from one episode."""
    ep = [dataset[i] for i in range(len(dataset))
          if dataset[i]["episode_index"] == episode_idx]
    idxs = np.linspace(0, len(ep) - 1,
                       n_frames, dtype=int)              #A

    fig, axes = plt.subplots(2, n_frames, figsize=(18, 6))
    for col, i in enumerate(idxs):
        top = ep[i]["observation.images.top"]
        wrist = ep[i]["observation.images.wrist"]
        axes[0, col].imshow(top.permute(1, 2, 0).numpy())  #B
        axes[0, col].set_title(f"step {i}")
        axes[1, col].imshow(wrist.permute(1, 2, 0).numpy())
        for r in (0, 1):
            axes[r, col].axis("off")
    axes[0, 0].set_ylabel("top view")
    axes[1, 0].set_ylabel("wrist view")
    plt.tight_layout()
    plt.savefig("figures/figure_2_4_expert_keyframes.png",
                dpi=300)                                  #C
```
- #A Sample `n_frames` evenly spaced indices across the episode
- #B LeRobot stores images as (C, H, W) — permute for matplotlib
- #C Save at print resolution (300 DPI) per the figure style guide

Listing 2.8 lives in `ch02.viz` alongside `render_keyframes`. It collects actions from all three policies and overlays per-dimension histograms in figure 2.5. Import it as `from ch02.viz import plot_action_distributions` — the structural gap between the scripted and expert distributions is the gap a learned policy has to close.

**Listing 2.8: Per-joint action distributions — expert vs. scripted vs. random**
```python
def collect_actions(env, policy_fn, n_episodes=10):
    """Run a policy and collect all actions taken."""
    actions = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 100)
        state = {"phase": "approach"} if policy_fn else None
        done = False
        while not done:
            if policy_fn is None:
                action = env.action_space.sample()       #A
            else:
                action = policy_fn(obs, state)
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(actions)

expert = np.stack([dataset[i]["action"].numpy()         #B
                   for i in range(len(dataset))])
scripted = collect_actions(env, scripted_policy)
random_ = collect_actions(env, None)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
joint_names = [f"joint_{i}" for i in range(6)] + ["gripper"]
for j, name in enumerate(joint_names):
    ax = axes.flat[j]
    for arr, label in [(expert, "expert"),
                        (scripted, "scripted"),
                        (random_, "random")]:
        ax.hist(arr[:, j], bins=40, alpha=0.4,
                label=label, density=True)               #C
    ax.set_title(name)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/figure_2_5_action_distributions.png",
            dpi=300)
```
- #A `None` indicates the random policy
- #B Stack all expert actions from the dataset into a single array
- #C Overlapping histograms reveal the distributional gap per joint

**Figure 2.4: Expert Pick-and-Place Keyframes**
- A 2x6 grid: top row is the top-down camera, bottom row is the wrist-mounted camera, columns are six keyframes from one expert episode spanning approach → grasp → lift → transport → place → release.
- Caption: "Keyframes from one expert episode. The top-down view shows the macroscopic motion of the arm; the wrist view shows the contact-level detail of the grasp. A learned policy must capture both perspectives to handle objects whose position is only partially visible from above."

**Figure 2.5: Per-Joint Action Distributions — Expert vs. Scripted vs. Random**
- Seven overlapping histograms, one per action dimension, comparing the three policies.
- Caption: "Action distributions for each of the seven action dimensions. Expert actions show structured, multi-modal clusters that reflect different grasp strategies. The scripted policy produces a simpler, lower-variance pattern. Random actions are uniform across the range. The gap between the scripted and expert histograms is what a learned policy must close."

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
- Images are not z-scored — they are divided by 255 to land in `[0, 1]` and handed to the vision encoder.

### 2.5.3 LeRobot's meta/stats.json
- Reveal that LeRobot ships precomputed statistics in each dataset's `meta/stats.json`.
- Load them and compare against the manually computed values — they should agree.
- Going forward, use the precomputed stats for convenience, but the reader now understands what they contain and why.

### 2.5.4 The DataLoader Wrapper
- Wrap the LeRobotDataset in a PyTorch DataLoader with batching and shuffling.
- Apply normalization in a collate function: z-score state and action, scale images to `[0, 1]`.
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
        states.append(frame["observation.state"])         #A
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
def normalize(x, stats, key):
    """Z-score normalization: (x - mean) / std."""
    return (x - stats[key]["mean"]) / (stats[key]["std"] + 1e-8)  #A

def denormalize(x, stats, key):
    """Inverse z-score normalization."""
    return x * (stats[key]["std"] + 1e-8) + stats[key]["mean"]

sample = dataset[0]["observation.state"]
normed = normalize(sample, stats, "observation.state")
recovered = denormalize(normed, stats, "observation.state")
assert torch.allclose(sample, recovered, atol=1e-5)       #B
```
- #A The small epsilon prevents division by zero for features that are constant across the dataset
- #B Verify the round-trip is lossless to floating-point precision

Listing 2.11 ties the dataset, the statistics, and the normalization functions into the chapter's primary export: `make_pickplace_dataloader`. The function lives in `ch02.pipeline` and is imported as `from ch02.pipeline import make_pickplace_dataloader`. Chapter 3 imports it the same way and treats the signature as frozen — the listing is shown here so the reader understands what the function does, not because the reader writes it from scratch.

**Listing 2.11: Building the DataLoader — the Chapter 3 API contract**
```python
from torch.utils.data import DataLoader

def make_pickplace_dataloader(
    dataset_id="lerobot/so100_pick_place",                #A
    batch_size=64,
    shuffle=True,
):
    """Return a normalized DataLoader and stats for the SO-101 task."""
    dataset = LeRobotDataset(dataset_id)
    stats = compute_stats(dataset)

    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            vals = [b[key] for b in batch]
            if not isinstance(vals[0], torch.Tensor):
                out[key] = torch.tensor(vals)
                continue
            stacked = torch.stack(vals)
            if key in stats:                              #B
                stacked = normalize(stacked, stats, key)
            elif key.startswith("observation.images"):    #C
                stacked = stacked.float() / 255.0
            out[key] = stacked
        return out

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, collate_fn=collate_fn,
                        num_workers=4)
    return loader, stats

loader, stats = make_pickplace_dataloader(batch_size=32)
batch = next(iter(loader))
print(f"state mean per dim: "
      f"{batch['observation.state'].mean(0)}")            #D
print(f"image range: [{batch['observation.images.top'].min():.2f},"
      f" {batch['observation.images.top'].max():.2f}]")
```
- #A Parameterized on dataset ID so later chapters can swap in their own datasets without changing the function signature
- #B Z-score normalize state and action using precomputed stats
- #C Image features are scaled to `[0, 1]` — a different normalization path
- #D After normalization, per-dimension state mean should be near zero across a batch

**Callout Box: "Z-SCORE vs. MIN-MAX NORMALIZATION"**
- **Z-score** — `(x - mean) / std`. Centers at zero, scales by spread. Preferred when features are roughly Gaussian, as joint angles and recorded actions tend to be.
- **Min-max** — `(x - min) / (max - min)`. Scales to `[0, 1]`. Preferred when bounded outputs are needed, as with image pixels handed to a vision encoder expecting `[0, 1]`.
- This book uses z-score for state and actions and `x / 255` for images. The choice is consistent across all chapters.

**Callout Box: "THE CHAPTER 3 CONTRACT"**
- Chapter 3 imports `make_pickplace_dataloader()`, `normalize()`, and `denormalize()` directly from `ch02`.
- It expects batches with keys `observation.state` (normalized), `observation.images.top`, `observation.images.wrist` (in `[0, 1]`), and `action` (normalized).
- After the model predicts a normalized action, `denormalize()` converts it back to environment scale before calling `env.step()`.
- Treat the function signature `make_pickplace_dataloader(dataset_id, batch_size, shuffle)` as frozen — renaming or re-ordering arguments breaks every downstream chapter.

**Figure 2.7: The Normalization Round-Trip**
- Flow diagram: raw observation/action → `normalize(x, stats)` → zero-centered input → model prediction (normalized) → `denormalize(ŷ, stats)` → environment-scale action → `env.step()`.
- Caption: "The normalization round-trip. Observations and actions are z-score normalized before entering the model. Predicted actions are denormalized back to radians and gripper commands before being sent to the simulator. The stats dictionary bridges both directions and is the only piece of training-time state that has to survive into inference."

**Figure 2.8: End-to-End Data Pipeline**
- Flow diagram: Hugging Face Hub → LeRobotDataset → compute_stats → DataLoader with normalize-in-collate → training batch {state, images, action} → Chapter 3 model.
- Caption: "The complete data pipeline built in this chapter. Expert demonstrations are loaded from the Hub, normalization statistics are computed once, and a DataLoader applies the right normalization to each feature type at batch time. `make_pickplace_dataloader()` encapsulates the entire flow as the API contract for Chapter 3."

---

## Section 2.6: Summary

**Target length:** ~1 page

Comprehensive bulleted summary:

- `PickPlaceCube` is a single-object pick-and-place task on a 6-DOF arm with a parallel-jaw gripper, served by `gym-lowcostrobot` over MuJoCo. It is the carrier task and the carrier embodiment for the rest of the book.
- The Gymnasium API provides a universal interface — `reset()` returns an initial observation, `step(action)` returns the next observation, reward, and termination flags. Every environment in this book, in sim and on hardware, exposes this interface.
- A random agent on a 6-DOF arm essentially never succeeds. A multi-phase scripted policy (approach → descend → grasp → lift → transport → place → release) raises the success rate but plateaus far below expert performance because it cannot recover from misalignment or adapt to contact dynamics.
- Expert demonstrations from teleoperation are stored in the LeRobot dataset format on Hugging Face Hub. Each frame includes joint state, two camera views, the recorded action, and episode/frame metadata.
- LeRobot's `delta_timestamps` mechanism enables requesting data at relative time offsets — the foundation for action chunking in later chapters.
- Visualizing expert action distributions per-joint reveals structured, multi-modal patterns that neither random sampling nor a hand-coded heuristic can reproduce. Visualizing expert joint trajectories shows that successful policies must be conditional, not memorized.
- Neural networks need normalized inputs. Z-score normalization is applied to state and action; images are scaled to `[0, 1]`. Denormalization recovers environment-scale actions for use with `env.step()`.
- The chapter's primary export is `make_pickplace_dataloader(dataset_id, batch_size, shuffle)`. It is the API contract for Chapter 3 and is parameterized on `dataset_id` so later chapters can swap in custom datasets without changing the interface.

---

## Listings Summary

| Listing | Title | Section | Mode |
|---------|-------|---------|------|
| 2.1 | Installing the SO-101 sim and creating the environment | 2.1 | API illustration |
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
| 2.2 | The SO-101 Pick-and-Place Task | Annotated screenshot | 2.1 |
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
| "WHY NOT JUST ENGINEER A BETTER HEURISTIC?" | 2.2 | Connect to Chapter 1's long-tail argument |
| "WHAT IS delta_timestamps?" | 2.3 | Explain LeRobot's temporal indexing mechanism |
| "Z-SCORE vs. MIN-MAX NORMALIZATION" | 2.5 | Normalization strategy rationale |
| "THE CHAPTER 3 CONTRACT" | 2.5 | Define the API boundary between chapters |

## Table Summary

| Table | Description | Section |
|-------|-------------|---------|
| 2.1 | SO-101 Observation and Action Spaces | 2.1 |
| 2.2 | LeRobot SO-101 Pick-and-Place Dataset Features | 2.3 |

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
