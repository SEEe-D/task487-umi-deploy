"""Regression checks for physical view identity and the new checkpoint contract."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.models.model import ModelType
from openpi.policies.umi_policy import UMI_IMAGE_KEYS, UMIBimanualInputs
from openpi.training import config as config_lib
from task487_client import CameraPreview, validate_thor, task_ui_configuration
from task487_runtime.contract import (
    CAMERA_SOURCES, build_policy_request, task_prompts_for_runtime,
    thor_cameras_for_order, validate_policy_metadata,
    resolve_task_for_runtime, POLICY_RUNTIME_CONTRACTS, FOUR_WRIST_POLICY_RUNTIMES,
)
from umi.real_world.umi_env import thor_observation_key
from task487_runtime.scheduler import RollingScheduler, SchedulerConfig


@pytest.mark.parametrize("variant,head", [("raw", True), ("cnc", True), ("wrist_only", False)])
def test_four_wrist_end_to_end_view_identity(variant, head):
    cfg = config_lib.get_config(f"pi05_umi_task487_{variant}_4w_12_5")
    contract = validate_policy_metadata(cfg.policy_metadata)
    assert cfg.model.image_keys == UMI_IMAGE_KEYS
    assert tuple(cfg.model.inputs_spec()[0].images) == UMI_IMAGE_KEYS
    assert cfg.data.use_four_wrist_cameras
    assert cfg.data.use_head_camera is head
    assert contract.mask_enabled is False
    assert contract.control_hz == 12.5
    assert contract.action_horizon == 20
    assert not contract.complete_chunk_before_replan
    scheduler = RollingScheduler(SchedulerConfig.for_policy_rate(
        contract.control_hz, contract.action_horizon,
        complete_chunk_before_replan=contract.complete_chunk_before_replan))
    targets = np.zeros((20, 14))
    targets[:, 0] = np.arange(1, 21) * 0.01
    live = np.zeros(14)
    scheduler.activate(live)
    scheduler.mark_request_started(1.0)
    scheduler.merge_chunk(targets, 1.0, 1.0, live)
    assert len(scheduler.pop_batch(live, now=1.0)) == 1
    assert scheduler.queued_steps == 20
    # Replan with the slow, unsent tail still present; do not wait for it
    # to finish as in the previous full-chunk experiment.
    assert scheduler.request_due(1.32)
    assert task_prompts_for_runtime(contract.runtime) == {"sorting": "Vegetable and Fruit Sorting"}
    for alias in ("vegetable", "fruit", "sorting"):
        assert resolve_task_for_runtime(contract.runtime, alias) == "sorting"
    instructions, indices = task_ui_configuration(task_prompts_for_runtime(contract.runtime))
    assert len(instructions) == 1
    assert indices == {"sorting": 0}
    cameras = thor_cameras_for_order(contract.camera_order)
    assert [c["video_port"] for c in cameras] == ([5000] if head else []) + [5002, 5003, 5004, 5005]
    assert [c["meta_port"] for c in cameras] == ([6000] if head else []) + [6002, 6003, 6004, 6005]
    obs = {"timestamp": np.array([10.0, 10.08])}
    for key, camera in zip(contract.camera_order, cameras):
        obs_key = thor_observation_key(camera["label"])
        assert obs_key == CAMERA_SOURCES[key][0]
        # Unique colour for each *physical* source detects swaps and aliasing.
        obs[obs_key] = np.full((2, 224, 224, 3), camera["video_port"] - 4980, dtype=np.uint8)
    for arm in (0, 1):
        obs[f"robot{arm}_eef_pos"] = np.zeros((2, 3))
        obs[f"robot{arm}_eef_rot_axis_angle"] = np.zeros((2, 3))
        obs[f"robot{arm}_gripper_angle"] = np.ones((2, 1))
    request = build_policy_request(
        obs, task_prompts_for_runtime(contract.runtime)["sorting"], camera_order=contract.camera_order)
    assert tuple(k for k in request.observation if k.startswith("cam_")) == contract.camera_order
    assert request.observation["prompt"] == "Vegetable and Fruit Sorting"
    transformed = UMIBimanualInputs(ModelType.PI05, use_head_camera=head, use_four_wrist_cameras=True)(
        request.observation)
    assert tuple(transformed["image"]) == UMI_IMAGE_KEYS
    for model_key, colour in (("left_wrist_0_rgb", 22), ("left_wrist_1_rgb", 23),
                              ("right_wrist_0_rgb", 24), ("right_wrist_1_rgb", 25)):
        assert np.all(transformed["image"][model_key] == colour)
    assert bool(transformed["image_mask"]["base_0_rgb"]) is head
    assert np.all(transformed["image"]["base_0_rgb"] == (20 if head else 0))
    assert "image_token_mask" not in transformed
    preview = CameraPreview.compose_processed(request.observation, "test")
    assert preview.shape[1] == 224 * (5 if head else 4)
    rgb_y = 68 if head else 40
    for index, key in enumerate(contract.camera_order):
        np.testing.assert_array_equal(preview[rgb_y:rgb_y+224, index*224:(index+1)*224],
                                      request.observation[key][..., ::-1])
    del obs["camera3_rgb"]
    with pytest.raises(ValueError, match="Missing required camera cam_left_down"):
        build_policy_request(obs, "test", camera_order=contract.camera_order)
    stale_metadata = dict(cfg.policy_metadata, camera_order=["cam_head", "cam_left_top", "cam_right_top"])
    with pytest.raises(RuntimeError, match="contract mismatch"):
        validate_policy_metadata(stale_metadata)


def test_legacy_runtimes_keep_separate_tasks_and_rolling_replan():
    for runtime, contract in POLICY_RUNTIME_CONTRACTS.items():
        if runtime in FOUR_WRIST_POLICY_RUNTIMES:
            continue
        assert not contract.complete_chunk_before_replan
        assert set(task_prompts_for_runtime(runtime)) == {"vegetable", "fruit"}
        assert resolve_task_for_runtime(runtime, "fruit") == "fruit"
        with pytest.raises(ValueError, match="not supported"):
            resolve_task_for_runtime(runtime, "sorting")


def test_thor_bottom_aliases_are_explicit_and_unknown_views_rejected():
    assert thor_observation_key("cam_hand_l_bottom") == thor_observation_key("cam_hand_l_btm") == "camera3_rgb"
    assert thor_observation_key("cam_hand_r_bottom") == thor_observation_key("cam_hand_r_btm") == "camera1_rgb"
    with pytest.raises(ValueError, match="Unknown Thor"):
        thor_observation_key("cam_hand_r_botom")


def test_all_four_wrist_streams_must_be_fresh_even_without_head():
    cfg = config_lib.get_config("pi05_umi_task487_wrist_only_4w_12_5")
    order = validate_policy_metadata(cfg.policy_metadata).camera_order
    receivers = {
        cam["label"]: SimpleNamespace(lock=threading.Lock(), frame=np.zeros((1, 1, 3)),
                                     meta=SimpleNamespace(latest_ts_us=time.time()*1e6, clock_offset_ms=0))
        for cam in thor_cameras_for_order(order)
    }
    env = SimpleNamespace(thor_receivers=receivers)
    validate_thor(env, 0.25, 0.05, order)
    receivers["cam_hand_l_bottom"].meta.latest_ts_us -= 1e6
    with pytest.raises(RuntimeError, match="cam_hand_l_bottom age"):
        validate_thor(env, 0.25, 0.05, order)
    del receivers["cam_hand_l_bottom"]
    with pytest.raises(RuntimeError, match="Missing required Thor receivers"):
        validate_thor(env, 0.25, 0.05, order)
