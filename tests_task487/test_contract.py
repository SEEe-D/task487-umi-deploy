import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from task487_runtime.contract import (
    ACTION_HORIZON,
    CONTROL_HZ,
    DEFAULT_POLICY_RUNTIME,
    MASKED_POLICY_RUNTIME,
    POLICY_RUNTIME_CONTRACTS,
    RAW_12_5_TASK_PROMPTS,
    R_M2R,
    R_R2M,
    TASK_PROMPTS,
    WRIST_ONLY_POLICY_RUNTIME,
    body_actions_to_robot_targets,
    build_policy_request,
    right_pose_in_dataset_frame,
    robot_pose_to_model_pose,
    robot_targets_to_model_absolute_actions,
    rot6d_to_matrix,
    task_prompts_for_runtime,
    validate_policy_metadata,
)
from openpi.models import model as model_module
from openpi.policies.umi_policy import UMIBimanualInputs


def _fake_obs():
    obs = {"timestamp": np.array([10.0, 10.08])}
    for key in ("camera0_rgb", "camera2_rgb", "camera5_rgb"):
        obs[key] = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    for robot in (0, 1):
        obs[f"robot{robot}_eef_pos"] = np.array([[0.1, -0.2, 0.3], [0.101, -0.2, 0.3]])
        obs[f"robot{robot}_eef_rot_axis_angle"] = np.zeros((2, 3))
        obs[f"robot{robot}_gripper_angle"] = np.array([[10.0], [11.0]])
    return obs


def test_marvin_model_tool_axis_contract_matches_task487_b1_runtime():
    expected = np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(R_M2R, expected)
    np.testing.assert_array_equal(R_R2M, expected.T)
    np.testing.assert_allclose(R_M2R @ R_R2M, np.eye(3), atol=0.0)
    assert np.isclose(np.linalg.det(R_M2R), 1.0)


def test_request_has_exact_task487_keys_and_units():
    request = build_policy_request(_fake_obs(), TASK_PROMPTS["vegetable"], round_id=7)
    assert set(request.observation) == {
        "cam_head",
        "cam_left_top",
        "cam_right_top",
        "pre_state",
        "state",
        "prompt",
    }
    assert request.observation["cam_head"].shape == (224, 224, 3)
    assert request.observation["state"].shape == (20,)
    assert np.isclose(request.observation["state"][9], np.deg2rad(11.0))
    assert np.isclose(request.observation["state"][19], np.deg2rad(11.0))
    assert request.observation_time == 10.08
    assert request.tcp_bases.shape == (2, 6)
    assert request.round_id == 7


def test_request_rejects_deployment_image_geometry_mismatch():
    obs = _fake_obs()
    obs["camera5_rgb"] = np.zeros((2, 512, 640, 3), dtype=np.uint8)
    with np.testing.assert_raises_regex(ValueError, "training image shape"):
        build_policy_request(obs, TASK_PROMPTS["vegetable"])


def test_task_prompts_match_dataset_text_exactly():
    assert TASK_PROMPTS["vegetable"] == "Pick Up Vegetable and Place Vegetable on the Pink  Plate on the Right"
    assert TASK_PROMPTS["fruit"] == "Pick Up Fruit and Place Fruit on the Blue Plate on the Left"
    assert RAW_12_5_TASK_PROMPTS["vegetable"] == (
        "Pick Up Vegetable and Place Vegetable on the Pink Plate on the Right"
    )
    assert RAW_12_5_TASK_PROMPTS["fruit"] == TASK_PROMPTS["fruit"]
    assert task_prompts_for_runtime(DEFAULT_POLICY_RUNTIME) == RAW_12_5_TASK_PROMPTS
    assert task_prompts_for_runtime("pi05_umi_task487_v1") == TASK_PROMPTS


def test_default_runtime_uses_exact_12_5_hz_downsampled_contract():
    contract = POLICY_RUNTIME_CONTRACTS[DEFAULT_POLICY_RUNTIME]
    assert CONTROL_HZ == 12.5
    assert contract.action_horizon == 20
    assert contract.control_hz == 12.5
    assert contract.action_first_target_offset_s == 0.08
    assert contract.state_history_offsets_s == (-0.08, 0.0)
    assert contract.rtc_prefix_steps == 5
    assert contract.image_geometry == "resize_with_pad"
    assert contract.gripper_start_degrees == (1.0, 1.0)
    assert contract.gripper_ready_tolerance_deg == 2.5
    assert validate_policy_metadata(contract.expected_metadata()) == contract


def test_wrist_only_runtime_disables_head_and_preserves_12_5_hz_contract():
    contract = POLICY_RUNTIME_CONTRACTS[WRIST_ONLY_POLICY_RUNTIME]
    metadata = contract.expected_metadata()

    assert metadata["camera_order"] == ["cam_left_top", "cam_right_top"]
    assert metadata["head_enabled"] is False
    assert metadata["control_hz"] == 12.5
    assert metadata["rtc_prefix_steps"] == 5
    assert task_prompts_for_runtime(WRIST_ONLY_POLICY_RUNTIME) == RAW_12_5_TASK_PROMPTS
    assert validate_policy_metadata(metadata) == contract


def test_masked_runtime_requires_token_mask_and_preserves_12_5_hz_contract():
    contract = POLICY_RUNTIME_CONTRACTS[MASKED_POLICY_RUNTIME]
    metadata = contract.expected_metadata()

    assert metadata["camera_order"] == ["cam_head", "cam_left_top", "cam_right_top"]
    assert metadata["mask_enabled"] is True
    assert metadata["control_hz"] == 12.5
    assert metadata["rtc_prefix_steps"] == 5
    assert contract.image_geometry == "resize_with_pad"
    assert task_prompts_for_runtime(MASKED_POLICY_RUNTIME) == RAW_12_5_TASK_PROMPTS
    assert validate_policy_metadata(metadata) == contract


def test_masked_request_preserves_binary_head_mask_and_builds_token_mask():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:112, :112] = 255
    request = build_policy_request(
        _fake_obs(),
        TASK_PROMPTS["vegetable"],
        fixed_head_mask=mask,
    )

    np.testing.assert_array_equal(request.observation["fixed_head_mask"], mask)
    transformed = UMIBimanualInputs(model_module.ModelType.PI05)(request.observation)
    assert set(transformed["image_token_mask"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert transformed["image_token_mask"]["base_0_rgb"].shape == (256,)
    assert np.count_nonzero(transformed["image_token_mask"]["base_0_rgb"]) == 192
    assert transformed["image_token_mask"]["left_wrist_0_rgb"].all()
    assert transformed["image_token_mask"]["right_wrist_0_rgb"].all()


def test_masked_request_rejects_non_binary_or_misaligned_mask():
    with np.testing.assert_raises_regex(ValueError, "final Task487 head-image shape"):
        build_policy_request(
            _fake_obs(),
            TASK_PROMPTS["vegetable"],
            fixed_head_mask=np.zeros((512, 640), dtype=np.uint8),
        )
    invalid = np.zeros((224, 224), dtype=np.uint8)
    invalid[0, 0] = 127
    with np.testing.assert_raises_regex(ValueError, "binary 0/255"):
        build_policy_request(
            _fake_obs(),
            TASK_PROMPTS["vegetable"],
            fixed_head_mask=invalid,
        )


def test_old_real_runtime_preserves_center_crop_and_calibrated_open_home_contract():
    contract = POLICY_RUNTIME_CONTRACTS["pi05_umi_task487_v1"]
    assert contract.image_geometry == "center_square"
    assert contract.gripper_start_degrees == pytest.approx(
        (34.000000247628236, 24.035200207677658)
    )
    assert contract.gripper_ready_tolerance_deg == 5.0


def test_runtime_metadata_rejects_old_25_hz_timing_under_new_runtime_name():
    metadata = POLICY_RUNTIME_CONTRACTS[DEFAULT_POLICY_RUNTIME].expected_metadata()
    metadata["control_hz"] = 25.0
    metadata["action_first_target_offset_s"] = 0.04
    with np.testing.assert_raises_regex(RuntimeError, "contract mismatch"):
        validate_policy_metadata(metadata)


def test_rtc_robot_prefix_round_trips_through_policy_input_transform():
    obs = _fake_obs()
    plain_request = build_policy_request(obs, TASK_PROMPTS["vegetable"])
    targets = np.zeros((10, 14), dtype=np.float64)
    for arm, offset in ((0, 0), (1, 7)):
        targets[:, offset : offset + 6] = plain_request.tcp_bases[arm]
        targets[:, offset] += np.arange(1, 11) * 0.001
        targets[:, offset + 6] = 12.0 + arm

    absolute = robot_targets_to_model_absolute_actions(targets)
    assert absolute.shape == (10, 20)
    request = build_policy_request(
        obs,
        TASK_PROMPTS["vegetable"],
        rtc_prefix_targets=targets,
    )
    assert request.observation["actions"].shape == (ACTION_HORIZON, 20)
    assert request.observation["action_prefill_len"] == 10

    transformed = UMIBimanualInputs(model_module.ModelType.PI05)(request.observation)
    recovered = body_actions_to_robot_targets(
        transformed["actions"],
        request.tcp_bases,
    )
    np.testing.assert_allclose(recovered[:10], targets, atol=2e-7)


def test_b1_horizon_16_rtc_prefix_is_padded_and_converted_dynamically():
    obs = _fake_obs()
    plain_request = build_policy_request(obs, TASK_PROMPTS["vegetable"])
    targets = np.zeros((5, 14), dtype=np.float64)
    for arm, offset in ((0, 0), (1, 7)):
        targets[:, offset : offset + 6] = plain_request.tcp_bases[arm]
        targets[:, offset + 1] += np.arange(1, 6) * 0.001
        targets[:, offset + 6] = 10.0 + arm

    request = build_policy_request(
        obs,
        TASK_PROMPTS["vegetable"],
        rtc_prefix_targets=targets,
        action_horizon=16,
    )
    assert request.observation["actions"].shape == (16, 20)
    assert request.observation["action_prefill_len"] == 5

    transformed = UMIBimanualInputs(model_module.ModelType.PI05)(request.observation)
    recovered = body_actions_to_robot_targets(
        transformed["actions"],
        request.tcp_bases,
        action_horizon=16,
    )
    assert recovered.shape == (16, 14)
    np.testing.assert_allclose(recovered[:5], targets, atol=2e-7)


def test_identity_body_action_returns_observed_tcp():
    bases = np.array(
        [
            [0.2, -0.1, 0.4, 0.1, -0.2, 0.3],
            [-0.2, 0.1, 0.5, -0.2, 0.1, -0.1],
        ]
    )
    actions = np.zeros((ACTION_HORIZON, 20))
    actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
    actions[:, 13:19] = [1, 0, 0, 0, 1, 0]
    actions[:, 9] = np.deg2rad(12.0)
    actions[:, 19] = np.deg2rad(18.0)
    targets = body_actions_to_robot_targets(actions, bases)
    np.testing.assert_allclose(targets[:, :6], np.broadcast_to(bases[0], (ACTION_HORIZON, 6)), atol=1e-7)
    np.testing.assert_allclose(targets[:, 7:13], np.broadcast_to(bases[1], (ACTION_HORIZON, 6)), atol=1e-7)
    np.testing.assert_allclose(targets[:, 6], 12.0)
    np.testing.assert_allclose(targets[:, 13], 18.0)


def test_robot_pose_rotation_uses_first_two_rows():
    pose = np.array([0.1, 0.2, 0.3, *Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_rotvec()])
    model = robot_pose_to_model_pose(pose)
    recovered = rot6d_to_matrix(model[3:9])
    assert recovered.shape == (3, 3)
    np.testing.assert_allclose(recovered @ recovered.T, np.eye(3), atol=1e-7)


def _body_delta(target_model_pose, base_model_pose):
    base_rotation = rot6d_to_matrix(base_model_pose[3:9])
    target_rotation = rot6d_to_matrix(target_model_pose[3:9])
    return np.concatenate(
        (
            base_rotation.T @ (target_model_pose[:3] - base_model_pose[:3]),
            (base_rotation.T @ target_rotation)[:2].reshape(-1),
        )
    )


def test_nonidentity_body_actions_round_trip_for_both_arms():
    bases = np.array(
        [
            [0.22, -0.18, 0.42, *Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_rotvec()],
            [-0.17, 0.16, 0.47, *Rotation.from_euler("xyz", [-0.2, 0.1, -0.15]).as_rotvec()],
        ]
    )
    desired = np.array(
        [
            [0.235, -0.162, 0.414, *Rotation.from_euler("xyz", [0.24, -0.13, 0.32]).as_rotvec()],
            [-0.158, 0.141, 0.481, *Rotation.from_euler("xyz", [-0.17, 0.08, -0.11]).as_rotvec()],
        ]
    )
    actions = np.zeros((ACTION_HORIZON, 20), dtype=np.float64)

    for arm, offset in ((0, 0), (1, 10)):
        base_pose = right_pose_in_dataset_frame(bases[arm]) if arm == 0 else bases[arm]
        desired_pose = right_pose_in_dataset_frame(desired[arm]) if arm == 0 else desired[arm]
        body = _body_delta(
            robot_pose_to_model_pose(desired_pose),
            robot_pose_to_model_pose(base_pose),
        )
        actions[:, offset : offset + 9] = body

    targets = body_actions_to_robot_targets(actions, bases)
    for arm, offset in ((0, 0), (1, 7)):
        np.testing.assert_allclose(
            targets[:, offset : offset + 3],
            np.broadcast_to(desired[arm, :3], (ACTION_HORIZON, 3)),
            atol=1e-7,
        )
        target_rotation = Rotation.from_rotvec(targets[:, offset + 3 : offset + 6])
        desired_rotation = Rotation.from_rotvec(desired[arm, 3:])
        np.testing.assert_allclose((target_rotation.inv() * desired_rotation).magnitude(), 0.0, atol=1e-7)
