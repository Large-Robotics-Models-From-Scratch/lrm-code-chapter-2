"""Multi-phase scripted pick-and-place policy.

A seven-phase state machine that approaches the cube, descends,
closes the gripper, lifts, transports, places, and releases. It is
intentionally simple and operates in end-effector space, so the
controller can compute Cartesian deltas without running inverse
kinematics on the SO-100.

The policy plateaus well below expert teleoperation performance —
that is the chapter's pedagogical point. Learned policies pick up
the subtle contact dynamics and recovery behaviors that this
heuristic cannot encode.

Requires the env to be constructed with:
- ``obs_mode='state_dict'`` so positions are addressable by name
- ``control_mode='pd_ee_delta_pose'`` so the action is a 7-tuple
  (xyz delta + rpy delta + gripper)
"""

import numpy as np

PHASES: list[str] = [
    "approach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "place",
    "release",
]


def _episode_success(info: dict) -> bool:
    """Read the success flag, tolerating ManiSkill vs legacy keys."""
    if "success" in info:
        return bool(info["success"])
    if "is_success" in info:
        return bool(info["is_success"])
    return False


def scripted_policy(obs: dict, state: dict) -> np.ndarray:
    """Seven-phase state-machine controller for PickCubeSO100-v1.

    ``state`` is the caller's per-episode dict (start with
    ``{"phase": "approach"}``). The function mutates it to track
    phase transitions and the grasp-hold counter.

    Expects an env constructed with ``obs_mode='state_dict'`` so that
    end-effector, cube, and target positions are addressable by name
    under ``obs['extra']``.
    """
    phase = state["phase"]
    extra = obs["extra"]
    ee_pos = np.array(extra["tcp_pose"][:3])
    cube = np.array(extra["obj_pose"][:3])
    target = np.array(extra["goal_pos"])

    if phase == "approach":
        goal = cube + np.array([0.0, 0.0, 0.10])
        gripper = -1.0
        if np.linalg.norm(ee_pos - goal) < 0.01:
            state["phase"] = "descend"
    elif phase == "descend":
        goal = cube + np.array([0.0, 0.0, 0.005])
        gripper = -1.0
        if abs(ee_pos[2] - goal[2]) < 0.005:
            state["phase"] = "grasp"
    elif phase == "grasp":
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
        gripper = -1.0

    ee_delta_xyz = np.clip((goal - ee_pos) * 5.0, -1.0, 1.0)
    return np.concatenate(
        [ee_delta_xyz, np.zeros(3), [gripper]]
    ).astype(np.float32)


def run_scripted_agent(env, n_episodes: int = 10) -> float:
    """Run the scripted policy for n_episodes; return success_rate."""
    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        state = {"phase": "approach"}
        done = False
        while not done:
            action = scripted_policy(obs, state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
        successes += int(_episode_success(info))
    return successes / n_episodes
