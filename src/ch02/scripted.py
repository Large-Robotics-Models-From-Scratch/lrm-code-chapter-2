"""Multi-phase scripted pick-and-place via ManiSkill's motion planner.

The seven phases — approach, descend, grasp, lift, transport, place,
release — are expressed as Cartesian keyframe poses. ManiSkill's
``SO100ArmMotionPlanningSolver`` solves IK and steps the joint
trajectory for each keyframe, so the scripted code itself stays
declarative.

The policy plateaus below expert teleoperation: the keyframe offsets
are hard-coded, grasp-pose orientation is fixed top-down, and there
is no recovery. That gap is the chapter's motivation for learned
policies.
"""

import numpy as np
import sapien
from transforms3d.euler import euler2quat

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


def run_scripted_episode(planner, grasp_pose, goal_pos) -> None:
    """Execute the 7 phases via the motion planner.

    ``planner`` is an ``SO100ArmMotionPlanningSolver`` already bound
    to a fresh-reset env. ``grasp_pose`` is the target end-effector
    pose at the moment of closure. ``goal_pos`` is the xyz drop
    point. The function returns nothing; success is read off the env
    afterwards via ``env.unwrapped.evaluate()``.
    """
    quat = grasp_pose.q
    goal = np.asarray(goal_pos)

    # 1. approach (10cm above grasp pose)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.10]) * grasp_pose
    )
    # 2. descend (3cm above)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.03]) * grasp_pose
    )
    # 3. grasp (at pose, partial close)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.01]) * grasp_pose
    )
    planner.close_gripper(gripper_state=-0.8)
    # 4. lift (15cm above)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.15]) * grasp_pose
    )
    # 5. transport (15cm above goal)
    planner.move_to_pose_with_screw(
        sapien.Pose(goal + np.array([0, 0, 0.15]), quat)
    )
    # 6. place (2cm above goal)
    planner.move_to_pose_with_screw(
        sapien.Pose(goal + np.array([0, 0, 0.02]), quat)
    )
    # 7. release
    planner.open_gripper()


def _compute_top_down_grasp_pose(env):
    """Build a top-down grasp pose for the cube spawn position."""
    unwrapped = env.unwrapped
    cube_pos = unwrapped.cube.pose.sp.p
    base = unwrapped.agent.build_grasp_pose(
        approaching=np.array([0, 0, -1]),
        closing=np.array([1, 0, 0]),
        center=cube_pos,
    )
    # SO-100's gripper frame requires a rotation correction so the
    # closing axis lines up with the cube faces.
    return base * sapien.Pose(
        q=euler2quat(-np.pi / 2, 0, np.pi / 2)
    )


def run_scripted_agent(env, n_episodes: int = 10) -> float:
    """Run the scripted policy n_episodes; return success_rate."""
    # Lazy import: keeps the module importable in environments without
    # the [sim] extra so tests using the MockPlanner pattern can run.
    from mani_skill.examples.motionplanning.so100.motionplanner import (
        SO100ArmMotionPlanningSolver,
    )

    successes = 0
    for ep in range(n_episodes):
        env.reset(seed=ep)
        planner = SO100ArmMotionPlanningSolver(
            env,
            base_pose=env.unwrapped.agent.robot.pose,
            print_env_info=False,
        )
        try:
            grasp_pose = _compute_top_down_grasp_pose(env)
            goal_pos = env.unwrapped.goal_site.pose.sp.p
            run_scripted_episode(planner, grasp_pose, goal_pos)
            info = env.unwrapped.evaluate()
            if _episode_success(info):
                successes += 1
        finally:
            try:
                planner.close()
            except Exception:
                pass
    return successes / n_episodes
