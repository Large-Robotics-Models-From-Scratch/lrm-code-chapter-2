"""Unit tests for the scripted-policy state machine.

Tests use a MockPlanner — no env needed, no SAPIEN simulation.
``sapien.Pose`` is required for synthetic pose construction; if
sapien isn't installed, the whole module is skipped.
"""

import numpy as np
import pytest

pytest.importorskip("sapien")

import sapien  # noqa: E402

from ch02._metrics import _episode_success  # noqa: E402
from ch02.scripted import (  # noqa: E402
    run_scripted_agent,
    run_scripted_episode,
)


class MockPlanner:
    """Records ManiSkill motion-planner calls for assertion."""

    def __init__(self):
        self.calls: list[tuple] = []

    def move_to_pose_with_screw(self, pose):
        self.calls.append(("move_to_pose_with_screw", pose))
        return "success"

    def close_gripper(self, gripper_state=None):
        self.calls.append(("close_gripper", gripper_state))

    def open_gripper(self):
        self.calls.append(("open_gripper",))

    def close(self):
        self.calls.append(("close",))


def _grasp_and_goal():
    """Synthetic grasp pose and goal position for tests."""
    grasp_pose = sapien.Pose([0.2, 0.0, 0.05])
    goal_pos = np.array([-0.2, 0.2, 0.02])
    return grasp_pose, goal_pos


def test_episode_success_helper():
    assert _episode_success({"success": True}) is True
    assert _episode_success({"success": False}) is False
    assert _episode_success({}) is False  # missing key → False


def test_episode_call_sequence():
    """The 8 planner calls fire in the documented order."""
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)

    expected = [
        "move_to_pose_with_screw",  # 1 approach
        "move_to_pose_with_screw",  # 2 descend
        "move_to_pose_with_screw",  # 3 grasp pose
        "close_gripper",            # close
        "move_to_pose_with_screw",  # 4 lift
        "move_to_pose_with_screw",  # 5 transport
        "move_to_pose_with_screw",  # 6 place
        "open_gripper",             # 7 release
    ]
    assert [c[0] for c in planner.calls] == expected


def test_episode_uses_six_move_calls():
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    moves = [c for c in planner.calls if c[0] == "move_to_pose_with_screw"]
    assert len(moves) == 6


def test_close_gripper_uses_partial_state():
    """Partial close (-0.8) applies contact pressure without overclosing."""
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    close_calls = [c for c in planner.calls if c[0] == "close_gripper"]
    assert len(close_calls) == 1
    assert close_calls[0][1] == -0.8


def test_open_gripper_called_exactly_once():
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    open_calls = [c for c in planner.calls if c[0] == "open_gripper"]
    assert len(open_calls) == 1


def test_close_fires_after_third_move():
    """Grasp = approach + descend + grasp-pose then close."""
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    assert planner.calls[3][0] == "close_gripper"


def test_open_fires_last():
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    assert planner.calls[-1][0] == "open_gripper"


def test_run_scripted_agent_is_callable():
    """Smoke: importable + callable; full eval is integration."""
    assert callable(run_scripted_agent)


def test_move_z_offsets_match_phase_intent():
    """6 move calls hit the documented Z offsets in order."""
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    moves = [
        c[1] for c in planner.calls
        if c[0] == "move_to_pose_with_screw"
    ]
    # moves 1-4 are grasp-relative (Z offset above grasp_pose.z)
    grasp_z = grasp_pose.p[2]
    assert moves[0].p[2] == pytest.approx(grasp_z + 0.10)
    assert moves[1].p[2] == pytest.approx(grasp_z + 0.03)
    assert moves[2].p[2] == pytest.approx(grasp_z + 0.01)
    assert moves[3].p[2] == pytest.approx(grasp_z + 0.15)
    # moves 5-6 are goal-relative
    assert moves[4].p[2] == pytest.approx(goal_pos[2] + 0.15)
    assert moves[5].p[2] == pytest.approx(goal_pos[2] + 0.02)


def test_quaternion_preserved_across_goal_moves():
    """Transport + place reuse grasp orientation so the cube stays put."""
    planner = MockPlanner()
    grasp_pose, goal_pos = _grasp_and_goal()
    run_scripted_episode(planner, grasp_pose, goal_pos)
    moves = [
        c[1] for c in planner.calls
        if c[0] == "move_to_pose_with_screw"
    ]
    assert np.allclose(moves[4].q, grasp_pose.q)
    assert np.allclose(moves[5].q, grasp_pose.q)
