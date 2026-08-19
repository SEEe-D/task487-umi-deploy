import numpy as np

from task487_client import grippers_are_ready, task_ui_configuration
from task487_runtime.contract import (
    DEFAULT_POLICY_RUNTIME,
    POLICY_RUNTIME_CONTRACTS,
    task_prompts_for_runtime,
)


class FakeGripperEnv:
    def __init__(self, right, left):
        self._positions = (right, left)

    def get_gripper_state(self):
        return [
            {"gripper_position": np.asarray([position], dtype=np.float64)}
            for position in self._positions
        ]


def test_task_ui_uses_runtime_specific_training_prompts():
    prompts = task_prompts_for_runtime(DEFAULT_POLICY_RUNTIME)

    instructions, indices = task_ui_configuration(prompts)

    assert instructions[0]["task"] == "Pick Up Vegetable and Place Vegetable on the Pink Plate on the Right"
    assert indices == {"vegetable": 0, "fruit": 1}


def test_gripper_readiness_follows_runtime_boundary_state():
    raw = POLICY_RUNTIME_CONTRACTS[DEFAULT_POLICY_RUNTIME]
    old = POLICY_RUNTIME_CONTRACTS["pi05_umi_task487_v1"]

    assert grippers_are_ready(FakeGripperEnv(1.0, 1.0), raw)
    assert not grippers_are_ready(FakeGripperEnv(35.0, 35.0), raw)
    assert grippers_are_ready(FakeGripperEnv(30.1, 24.0), old)
    assert not grippers_are_ready(FakeGripperEnv(34.0, 35.0), old)
    assert not grippers_are_ready(FakeGripperEnv(1.0, 1.0), old)
