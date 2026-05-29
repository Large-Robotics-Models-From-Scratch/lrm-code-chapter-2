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

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from ch02._metrics import _episode_success

if TYPE_CHECKING:
    from mani_skill.examples.motionplanning.so100.motionplanner import (
        SO100ArmMotionPlanningSolver,
    )


def run_scripted_episode(
    planner: "SO100ArmMotionPlanningSolver",
    grasp_pose: sapien.Pose,
    goal_pos: np.ndarray,
) -> None:
    """Execute the 7 keyframe phases via the motion planner.

    approach → descend → grasp → lift → transport → place → release.
    Returns nothing; read success via `env.unwrapped.evaluate()`.

    Args:
        planner: SO100ArmMotionPlanningSolver bound to a fresh-reset env.
        grasp_pose: Target end-effector pose at gripper closure.
        goal_pos: `(3,)` xyz drop point in world coordinates.
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


def _compute_top_down_grasp_pose(env: gym.Env) -> sapien.Pose:
    """Build a top-down grasp pose for the cube's current spawn position.

    Args:
        env: Gymnasium env wrapping `PickCubeSO100-v1`.

    Returns:
        SAPIEN pose for the gripper at grasp time (rotated so the SO-100
        closing axis aligns with the cube faces).
    """
    unwrapped = env.unwrapped
    cube_pos = unwrapped.cube.pose.sp.p
    # NOTE: closing=[1, 0, 0] assumes an axis-aligned cube; PickCubeSO100-v1
    # spawns this way. For arbitrary cube yaw, derive from the OBB (see
    # ManiSkill `motionplanning/so100/solutions/pick_cube.py`).
    base = unwrapped.agent.build_grasp_pose(
        approaching=np.array([0, 0, -1]),
        closing=np.array([1, 0, 0]),
        center=cube_pos,
    )
    # SO-100's gripper frame needs this rotation so the closing axis
    # lines up with the cube faces. Matches upstream verbatim:
    # mani_skill/examples/motionplanning/so100/solutions/pick_cube.py:44
    return base * sapien.Pose(
        q=euler2quat(-np.pi / 2, 0, np.pi / 2)
    )


def run_scripted_agent(env: gym.Env, n_episodes: int = 10) -> float:
    """Run the scripted policy `n_episodes` times; report the success rate.

    Args:
        env: Gymnasium env from `make_env`. Must expose `env.unwrapped`
            for cube/goal pose reads and `agent.build_grasp_pose`.
        n_episodes: Number of episodes to roll out.

    Returns:
        Fraction of episodes where the cube landed in the goal zone.
    """
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
