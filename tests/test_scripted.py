"""Unit tests for the scripted-policy state machine.

These tests construct synthetic ManiSkill-style observations and
verify the phase transitions fire under the right geometric
conditions. No env needed; ch02.scripted has no ManiSkill imports.
"""

import numpy as np

from ch02.scripted import (
    PHASES,
    _episode_success,
    run_scripted_agent,
    scripted_policy,
)


def _obs(ee_xyz, cube_xyz=(0.2, 0.0, 0.02), target_xyz=(-0.2, 0.2, 0.02)):
    """Build a state_dict-shaped obs with the three positions we read."""
    return {
        "extra": {
            "tcp_pose": np.array(
                [*ee_xyz, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "obj_pose": np.array(
                [*cube_xyz, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "goal_pos": np.array(target_xyz, dtype=np.float32),
        }
    }


def test_phases_in_execution_order():
    assert PHASES == [
        "approach",
        "descend",
        "grasp",
        "lift",
        "transport",
        "place",
        "release",
    ]


def test_action_shape_and_dtype():
    state = {"phase": "approach"}
    action = scripted_policy(_obs(ee_xyz=(0.0, 0.0, 0.3)), state)
    assert action.shape == (7,)
    assert action.dtype == np.float32


def test_approach_advances_to_descend_when_at_hover_height():
    """Approach goal is cube + 10cm Z; reach it and we should advance."""
    cube = (0.2, 0.0, 0.02)
    hover_goal = (cube[0], cube[1], cube[2] + 0.10)
    state = {"phase": "approach"}
    scripted_policy(_obs(ee_xyz=hover_goal, cube_xyz=cube), state)
    assert state["phase"] == "descend"


def test_grasp_advances_after_five_steps():
    """Grasp phase counts frames and advances at 5."""
    state = {"phase": "grasp"}
    for _ in range(4):
        scripted_policy(_obs(ee_xyz=(0.2, 0.0, 0.02)), state)
        assert state["phase"] == "grasp"
    scripted_policy(_obs(ee_xyz=(0.2, 0.0, 0.02)), state)
    assert state["phase"] == "lift"


def test_grasp_gripper_command_is_closed():
    state = {"phase": "grasp"}
    action = scripted_policy(_obs(ee_xyz=(0.2, 0.0, 0.02)), state)
    assert action[-1] == 1.0  # gripper closed


def test_approach_gripper_command_is_open():
    state = {"phase": "approach"}
    action = scripted_policy(_obs(ee_xyz=(0.0, 0.0, 0.3)), state)
    assert action[-1] == -1.0  # gripper open


def test_episode_success_helper():
    assert _episode_success({"success": True}) is True
    assert _episode_success({"success": False}) is False
    assert _episode_success({"is_success": True}) is True
    assert _episode_success({}) is False


def test_run_scripted_agent_is_callable():
    """Sanity: run_scripted_agent exists and is exposed at module top."""
    assert callable(run_scripted_agent)
