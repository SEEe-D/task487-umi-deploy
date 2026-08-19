import threading

import pytest
import numpy as np

from umi.real_world.umi_env import UmiEnv, center_square_resize, resize_with_pad


class FakeRobot:
    def __init__(self, index, events, hold_done=True, home_done=True):
        self.index = index
        self.events = events
        self.hold_done = hold_done
        self.home_done = home_done

    def hold(self):
        self.events.append(("hold", self.index))

    def wait_for_hold(self, timeout):
        self.events.append(("wait_hold", self.index))
        return self.hold_done

    def go_home(self):
        self.events.append(("home", self.index))

    def wait_for_home(self, timeout):
        self.events.append(("wait_home", self.index))
        return self.home_done


def make_env(*robots):
    env = UmiEnv.__new__(UmiEnv)
    env.robots = list(robots)
    return env


def test_hold_commands_both_arms_before_waiting():
    events = []
    env = make_env(FakeRobot(0, events), FakeRobot(1, events))

    env.hold(wait=True, timeout=2.0)

    assert events == [("hold", 0), ("hold", 1), ("wait_hold", 0), ("wait_hold", 1)]


def test_go_home_commands_both_arms_before_waiting():
    events = []
    env = make_env(FakeRobot(0, events), FakeRobot(1, events))

    env.go_home(wait=True, timeout=10.0)

    assert events == [("home", 0), ("home", 1), ("wait_home", 0), ("wait_home", 1)]


def test_hold_timeout_is_not_silently_ignored():
    events = []
    env = make_env(FakeRobot(0, events, hold_done=False), FakeRobot(1, events))

    with pytest.raises(TimeoutError, match="Robot 0 did not confirm HOLD"):
        env.hold(wait=True, timeout=2.0)

    assert events[:2] == [("hold", 0), ("hold", 1)]


def test_end_episode_is_safe_during_partial_startup():
    env = UmiEnv.__new__(UmiEnv)
    env.obs_accumulator = None
    env.action_accumulator = None

    env.end_episode()


def test_thor_training_geometry_crops_wide_frame_before_resize():
    frame = np.broadcast_to(
        np.arange(6, dtype=np.uint8)[None, :, None],
        (4, 6, 3),
    ).copy()

    result = center_square_resize(frame, (4, 4))

    assert result.shape == (4, 4, 3)
    np.testing.assert_array_equal(result[0, :, 0], [1, 2, 3, 4])


def test_raw_umi_training_geometry_preserves_wide_fov_and_pads_height():
    frame = np.full((4, 6, 3), 255, dtype=np.uint8)

    result = resize_with_pad(frame, (4, 4))

    assert result.shape == (4, 4, 3)
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(result[1:3], 255)
    np.testing.assert_array_equal(result[3], 0)


def make_task_env(guard):
    env = UmiEnv.__new__(UmiEnv)
    env.task_instructions = [
        {"task_index": 0, "task_key": "vegetable", "task": "vegetable prompt"},
        {"task_index": 1, "task_key": "fruit", "task": "fruit prompt"},
    ]
    env._task_instruction_lock = threading.Lock()
    env._task_selection_guard = guard
    env.task_instruction = dict(env.task_instructions[0])
    return env


def test_task_ui_accepts_switch_when_guard_allows_it():
    env = make_task_env(lambda: True)

    selected = env.set_task_instruction(1)

    assert selected["task_key"] == "fruit"
    assert env.get_task_instruction()["task_key"] == "fruit"


def test_task_ui_rejects_switch_when_guard_blocks_it():
    env = make_task_env(lambda: False)

    selected = env.set_task_instruction(1)

    assert selected["task_key"] == "vegetable"
    assert env.get_task_instruction()["task_key"] == "vegetable"
