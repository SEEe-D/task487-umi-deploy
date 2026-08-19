"""Exact Task487 observation/action contract shared by tests and the real client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


TASK_PROMPTS = {
    "vegetable": "Pick Up Vegetable and Place Vegetable on the Pink  Plate on the Right",
    "fruit": "Pick Up Fruit and Place Fruit on the Blue Plate on the Left",
}
RAW_12_5_TASK_PROMPTS = {
    # The newly curated UMI dataset uses one space between Pink and Plate.
    # PaliGemma does not collapse the old dataset's doubled space.
    "vegetable": "Pick Up Vegetable and Place Vegetable on the Pink Plate on the Right",
    "fruit": "Pick Up Fruit and Place Fruit on the Blue Plate on the Left",
}

CONTROL_HZ = 12.5
# Current Task487 checkpoint horizon. Callers connected to a policy server
# must pass the horizon advertised by that server instead of assuming this
# value; B1 uses 16.
ACTION_HORIZON = 20
ACTION_DIM = 20
ROBOT_ACTION_DIM = 14
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
# Independently calibrated safe physical opening endpoints on this Marvin.
# Keep model radians/degrees one-to-one and clip only beyond these endpoints;
# do not claim that the left gripper can physically reach the old 35° label.
MARVIN_GRIPPER_SAFE_OPEN_DEGREES = (34.000000247628236, 24.035200207677658)


@dataclass(frozen=True)
class PolicyRuntimeContract:
    """Server metadata and timing contract for one Task487 policy runtime."""

    runtime: str
    action_horizon: int
    control_hz: float
    action_first_target_offset_s: float
    state_history_offsets_s: tuple[float, float]
    rtc_prefix_steps: int
    # Local deployment preprocessing that is intentionally not advertised by
    # older policy servers.  The 12.5 Hz UMI set contains full 280x224 frames
    # and trains through resize-with-pad; the older real-robot set contains
    # center-square 224x224 frames.
    image_geometry: str
    # Dataset-aligned gripper state at the HOME/HOLD boundary.  Values remain
    # logical opening angles in degrees and are converted to/from model radians
    # one-to-one; these fields do not rescale model outputs.
    gripper_start_degrees: tuple[float, float]
    gripper_ready_tolerance_deg: float
    mask_enabled: bool = False
    camera_order: tuple[str, ...] = ("cam_head", "cam_left_top", "cam_right_top")
    head_enabled: bool | None = None

    def expected_metadata(self) -> dict[str, Any]:
        metadata = {
            "runtime": self.runtime,
            "camera_order": list(self.camera_order),
            "control_hz": self.control_hz,
            "action_first_target_offset_s": self.action_first_target_offset_s,
            "state_history_offsets_s": list(self.state_history_offsets_s),
            "action_horizon": self.action_horizon,
            "action_dim": ACTION_DIM,
            "state_dim": ACTION_DIM,
            "mask_enabled": self.mask_enabled,
            "rtc_enabled": True,
            "rtc_mode": "action_prefill_hard_inpainting_v1",
            "rtc_prefix_steps": self.rtc_prefix_steps,
        }
        if self.head_enabled is not None:
            metadata["head_enabled"] = self.head_enabled
        return metadata


DEFAULT_POLICY_RUNTIME = "pi05_umi_task487_12_5_v1"
MASKED_POLICY_RUNTIME = "pi05_umi_task487_masked_12_5_v1"
WRIST_ONLY_POLICY_RUNTIME = "pi05_umi_task487_wrist_only_12_5_v1"
POLICY_RUNTIME_CONTRACTS = {
    DEFAULT_POLICY_RUNTIME: PolicyRuntimeContract(
        runtime=DEFAULT_POLICY_RUNTIME,
        action_horizon=20,
        control_hz=12.5,
        action_first_target_offset_s=0.08,
        state_history_offsets_s=(-0.08, 0.0),
        rtc_prefix_steps=5,
        image_geometry="resize_with_pad",
        gripper_start_degrees=(1.0, 1.0),
        gripper_ready_tolerance_deg=2.5,
    ),
    MASKED_POLICY_RUNTIME: PolicyRuntimeContract(
        runtime=MASKED_POLICY_RUNTIME,
        action_horizon=20,
        control_hz=12.5,
        action_first_target_offset_s=0.08,
        state_history_offsets_s=(-0.08, 0.0),
        rtc_prefix_steps=5,
        image_geometry="resize_with_pad",
        gripper_start_degrees=(1.0, 1.0),
        gripper_ready_tolerance_deg=2.5,
        mask_enabled=True,
    ),
    WRIST_ONLY_POLICY_RUNTIME: PolicyRuntimeContract(
        runtime=WRIST_ONLY_POLICY_RUNTIME,
        action_horizon=20,
        control_hz=12.5,
        action_first_target_offset_s=0.08,
        state_history_offsets_s=(-0.08, 0.0),
        rtc_prefix_steps=5,
        image_geometry="resize_with_pad",
        gripper_start_degrees=(1.0, 1.0),
        gripper_ready_tolerance_deg=2.5,
        camera_order=("cam_left_top", "cam_right_top"),
        head_enabled=False,
    ),
    # Keep the two known 25 Hz runtimes explicit so an operator can roll back
    # without weakening metadata validation.
    "pi05_umi_task487_v1": PolicyRuntimeContract(
        runtime="pi05_umi_task487_v1",
        action_horizon=20,
        control_hz=25.0,
        action_first_target_offset_s=0.04,
        state_history_offsets_s=(-0.04, 0.0),
        rtc_prefix_steps=10,
        image_geometry="center_square",
        gripper_start_degrees=MARVIN_GRIPPER_SAFE_OPEN_DEGREES,
        gripper_ready_tolerance_deg=5.0,
    ),
    "stage2_b1_robot_jax_task487_rtc_v1": PolicyRuntimeContract(
        runtime="stage2_b1_robot_jax_task487_rtc_v1",
        action_horizon=16,
        control_hz=25.0,
        action_first_target_offset_s=0.04,
        state_history_offsets_s=(-0.04, 0.0),
        rtc_prefix_steps=10,
        image_geometry="center_square",
        gripper_start_degrees=MARVIN_GRIPPER_SAFE_OPEN_DEGREES,
        gripper_ready_tolerance_deg=5.0,
    ),
}


def task_prompts_for_runtime(runtime: str) -> dict[str, str]:
    """Return the exact task strings used to train one policy runtime."""
    if runtime not in POLICY_RUNTIME_CONTRACTS:
        raise ValueError(f"Unsupported Task487 runtime {runtime!r}")
    prompts = (
        RAW_12_5_TASK_PROMPTS
        if runtime in (DEFAULT_POLICY_RUNTIME, MASKED_POLICY_RUNTIME, WRIST_ONLY_POLICY_RUNTIME)
        else TASK_PROMPTS
    )
    return dict(prompts)


def validate_policy_metadata(metadata: dict[str, Any]) -> PolicyRuntimeContract:
    """Fail closed unless all timing and RTC fields match a known runtime."""

    runtime = metadata.get("runtime")
    contract = POLICY_RUNTIME_CONTRACTS.get(runtime)
    if contract is None:
        raise RuntimeError(
            f"Unsupported Task487 server runtime {runtime!r}; "
            f"expected one of {sorted(POLICY_RUNTIME_CONTRACTS)}"
        )
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in contract.expected_metadata().items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Task487 server contract mismatch: {mismatches}")
    return contract


# UMI model/tool-axis convention used by the Task487 Marvin data and the
# known-good B1 Marvin deployment.  This is Marvin-specific; the old value in
# this file came from the UR deployment and rotates an absolute replay pose by
# roughly 90 degrees even though the recorded video starts from a normal pose.
R_M2R = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
R_R2M = R_M2R.T

# Dataset robot0 (right arm) was expressed in the left-arm base frame.
T_LEFT_FROM_RIGHT = np.array(
    [
        [0.99996206, 0.00661996, 0.00566226, -0.01676012],
        [-0.00663261, 0.99997554, 0.00221860, -0.605552492],
        [-0.00564743, -0.00225607, 0.99998151, -0.00727700],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PolicyRequest:
    observation: dict
    observation_time: float
    tcp_bases: np.ndarray
    # Identifies the operator-started ACTIVE round.  An asynchronous result
    # from an earlier round must never be merged after HOME/HOLD + a new [d].
    round_id: int = 0


def _pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose shape (6,), got {pose.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _matrix_to_pose6(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()))


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    row0 = rot6d[..., :3]
    row1 = rot6d[..., 3:6]
    norm0 = np.linalg.norm(row0, axis=-1, keepdims=True)
    if np.any(norm0 < 1e-8):
        raise ValueError("Invalid rotation_6d first row")
    row0 = row0 / norm0
    row1 = row1 - np.sum(row1 * row0, axis=-1, keepdims=True) * row0
    norm1 = np.linalg.norm(row1, axis=-1, keepdims=True)
    if np.any(norm1 < 1e-8):
        raise ValueError("Invalid rotation_6d rows")
    row1 = row1 / norm1
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=-2)


def right_pose_in_dataset_frame(pose: np.ndarray) -> np.ndarray:
    return _matrix_to_pose6(T_LEFT_FROM_RIGHT @ _pose6_to_matrix(pose))


def robot_pose_to_model_pose(pose: np.ndarray) -> np.ndarray:
    """Express a robot TCP pose using the UMI gripper/tag local-axis convention."""
    pose = np.asarray(pose, dtype=np.float64)
    robot_rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    model_rotation = robot_rotation @ R_M2R
    return np.concatenate((pose[:3], _matrix_to_rot6d(model_rotation)))


def _state20(obs: dict, index: int) -> np.ndarray:
    result = np.empty(ACTION_DIM, dtype=np.float32)
    for robot_index, offset in ((0, 0), (1, 10)):
        raw_pose = np.concatenate(
            (
                np.asarray(obs[f"robot{robot_index}_eef_pos"][index], dtype=np.float64),
                np.asarray(obs[f"robot{robot_index}_eef_rot_axis_angle"][index], dtype=np.float64),
            )
        )
        dataset_pose = right_pose_in_dataset_frame(raw_pose) if robot_index == 0 else raw_pose
        result[offset : offset + 9] = robot_pose_to_model_pose(dataset_pose)
        gripper_degrees = float(np.asarray(obs[f"robot{robot_index}_gripper_angle"][index]).reshape(-1)[0])
        result[offset + 9] = np.deg2rad(gripper_degrees)
    return result


def robot_targets_to_model_absolute_actions(targets: np.ndarray) -> np.ndarray:
    """Encode Hx14 robot targets as absolute 20D UMI actions for RTC prefill.

    Feeding these through ``UMIBimanualInputs`` at the next observation turns
    them into the normalized body-frame representation expected inside Pi0.5.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != ROBOT_ACTION_DIM:
        raise ValueError(f"Expected robot targets shape (H, 14), got {targets.shape}")
    if not np.isfinite(targets).all():
        raise ValueError("RTC targets contain NaN or infinity")
    actions = np.empty((len(targets), ACTION_DIM), dtype=np.float32)
    for index, target in enumerate(targets):
        for arm, (model_offset, robot_offset) in enumerate(((0, 0), (10, 7))):
            pose = target[robot_offset : robot_offset + 6]
            dataset_pose = right_pose_in_dataset_frame(pose) if arm == 0 else pose
            actions[index, model_offset : model_offset + 9] = robot_pose_to_model_pose(dataset_pose)
            actions[index, model_offset + 9] = np.deg2rad(target[robot_offset + 6])
    return actions


def build_policy_request(
    obs: dict,
    prompt: str,
    rtc_prefix_targets: np.ndarray | None = None,
    round_id: int = 0,
    action_horizon: int = ACTION_HORIZON,
    fixed_head_mask: np.ndarray | None = None,
) -> PolicyRequest:
    """Build the exact three-camera, two-state input expected by pi05_umi_task487."""
    required_images = {
        # UmiEnv camera5 is cam_head_left, the source of training head_main.
        # camera4 is cam_head_right / head_main_stereo_right and is not the
        # head stream used by this three-camera checkpoint.
        "cam_head": "camera5_rgb",
        "cam_left_top": "camera2_rgb",
        "cam_right_top": "camera0_rgb",
    }
    images = {}
    for policy_key, obs_key in required_images.items():
        frames = np.asarray(obs[obs_key])
        expected_shape = (2, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if frames.shape != expected_shape:
            raise ValueError(
                f"{obs_key} must match the Task487 training image shape "
                f"{expected_shape}, got {frames.shape}"
            )
        image = frames[-1]
        if image.dtype != np.uint8:
            raise ValueError(f"{obs_key} must be uint8, got {image.dtype}")
        images[policy_key] = np.ascontiguousarray(image)

    timestamps = np.asarray(obs["timestamp"], dtype=np.float64)
    if timestamps.shape != (2,) or not np.isfinite(timestamps).all():
        raise ValueError(f"Expected two finite observation timestamps, got {timestamps}")
    if timestamps[1] <= timestamps[0]:
        raise ValueError(f"Observation timestamps are not increasing: {timestamps}")

    tcp_bases = np.empty((2, 6), dtype=np.float64)
    for robot_index in (0, 1):
        tcp_bases[robot_index] = np.concatenate(
            (
                np.asarray(obs[f"robot{robot_index}_eef_pos"][-1], dtype=np.float64),
                np.asarray(obs[f"robot{robot_index}_eef_rot_axis_angle"][-1], dtype=np.float64),
            )
        )

    policy_obs = {
        **images,
        "pre_state": _state20(obs, -2),
        "state": _state20(obs, -1),
        "prompt": prompt,
    }
    if fixed_head_mask is not None:
        mask = np.asarray(fixed_head_mask)
        if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                "fixed_head_mask must match the final Task487 head-image shape "
                f"{(IMAGE_HEIGHT, IMAGE_WIDTH)}, got {mask.shape}"
            )
        if mask.dtype != np.uint8:
            raise ValueError(f"fixed_head_mask must be uint8, got {mask.dtype}")
        values = np.unique(mask)
        if not np.isin(values, (0, 255)).all():
            raise ValueError(f"fixed_head_mask must be binary 0/255, got values {values[:8]}")
        policy_obs["fixed_head_mask"] = np.ascontiguousarray(mask)
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if rtc_prefix_targets is not None and len(rtc_prefix_targets):
        prefix = np.asarray(rtc_prefix_targets, dtype=np.float64)
        if len(prefix) > action_horizon:
            raise ValueError(
                f"RTC prefix has {len(prefix)} steps, exceeds horizon {action_horizon}"
            )
        absolute_actions = robot_targets_to_model_absolute_actions(prefix)
        if len(absolute_actions) < action_horizon:
            absolute_actions = np.concatenate(
                (
                    absolute_actions,
                    np.repeat(
                        absolute_actions[-1:],
                        action_horizon - len(absolute_actions),
                        axis=0,
                    ),
                ),
                axis=0,
            )
        policy_obs["actions"] = absolute_actions
        policy_obs["action_prefill_len"] = np.int32(len(prefix))
    return PolicyRequest(policy_obs, float(timestamps[-1]), tcp_bases, round_id)


def body_actions_to_robot_targets(
    actions: np.ndarray,
    tcp_bases: np.ndarray,
    action_horizon: int | None = None,
) -> np.ndarray:
    """Convert Hx20 body-frame model targets to Hx14 UmiEnv absolute targets."""
    actions = np.asarray(actions, dtype=np.float64)
    tcp_bases = np.asarray(tcp_bases, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shape (H, {ACTION_DIM}), got {actions.shape}")
    if action_horizon is not None and actions.shape[0] != int(action_horizon):
        raise ValueError(
            f"Expected action horizon {int(action_horizon)}, got actions shape {actions.shape}"
        )
    if tcp_bases.shape != (2, 6):
        raise ValueError(f"Expected tcp_bases shape (2, 6), got {tcp_bases.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Model actions contain NaN or infinity")

    targets = np.empty((len(actions), ROBOT_ACTION_DIM), dtype=np.float64)
    for arm, (model_offset, robot_offset) in enumerate(((0, 0), (10, 7))):
        base = tcp_bases[arm]
        robot_rotation = Rotation.from_rotvec(base[3:]).as_matrix()
        current_model_rotation = robot_rotation @ R_M2R
        delta_rotation = rot6d_to_matrix(actions[:, model_offset + 3 : model_offset + 9])
        target_position = base[:3] + np.einsum(
            "ij,nj->ni", current_model_rotation, actions[:, model_offset : model_offset + 3]
        )
        target_rotation = np.einsum("ij,njk,kl->nil", current_model_rotation, delta_rotation, R_R2M)
        target_rotvec = Rotation.from_matrix(target_rotation).as_rotvec()
        targets[:, robot_offset : robot_offset + 3] = target_position
        targets[:, robot_offset + 3 : robot_offset + 6] = target_rotvec
        targets[:, robot_offset + 6] = np.rad2deg(actions[:, model_offset + 9])
    return targets
