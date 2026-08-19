import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import umi_policy

_IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _pose(position, gripper=0.0):
    return np.concatenate((np.asarray(position, dtype=np.float32), _IDENTITY_6D, [gripper])).astype(np.float32)


def _input(state, *, pre_state=None, actions=None):
    data = {
        "cam_head": np.full((3, 8, 9), 4, dtype=np.uint8),
        "cam_left_top": np.zeros((3, 8, 9), dtype=np.uint8),
        "cam_right_top": np.full((3, 8, 9), 2, dtype=np.uint8),
        "state": state,
        "prompt": b"wipe the table",
    }
    if pre_state is not None:
        data["pre_state"] = pre_state
    if actions is not None:
        data["actions"] = actions
    return data


def test_bimanual_inputs_camera_and_state_contract():
    previous = np.concatenate((_pose([1, 0, 0], 0.1), _pose([0, 2, 0], 0.2)))
    current = np.concatenate((_pose([1.1, 0, 0], 0.3), _pose([0, 2.2, 0], 0.4)))

    result = umi_policy.UMIBimanualInputs(_model.ModelType.PI05)(_input(np.stack((previous, current))))

    assert tuple(result["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert all(image.shape == (8, 9, 3) for image in result["image"].values())
    assert all(result["image_mask"].values())
    assert "image_token_mask" not in result
    np.testing.assert_allclose(result["state"][:3], [0.1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result["state"][3:9], _IDENTITY_6D, atol=1e-6)
    assert result["state"][9] == pytest.approx(0.3)
    np.testing.assert_allclose(result["state"][10:13], [0, 0.2, 0], atol=1e-6)
    assert result["state"][19] == pytest.approx(0.4)
    assert result["prompt"] == "wipe the table"


def test_bimanual_actions_are_expressed_in_current_body_frame():
    rotation_z_90 = np.array([0, -1, 0, 1, 0, 0], dtype=np.float32)
    right = np.concatenate(([1, 2, 3], rotation_z_90, [0.25])).astype(np.float32)
    left = _pose([0, 0, 0], 0.5)
    current = np.concatenate((right, left))
    target_right = np.concatenate(([2, 2, 3], rotation_z_90, [0.8])).astype(np.float32)
    actions = np.stack((np.concatenate((target_right, left)),))

    result = umi_policy.UMIBimanualInputs(_model.ModelType.PI05)(_input(current, pre_state=current, actions=actions))

    np.testing.assert_allclose(result["actions"][0, :3], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(result["actions"][0, 3:9], _IDENTITY_6D, atol=1e-6)
    assert result["actions"][0, 9] == pytest.approx(0.8)
    np.testing.assert_allclose(result["actions"][0, 10:19], np.r_[np.zeros(3), _IDENTITY_6D], atol=1e-6)
    assert result["actions"][0, 19] == pytest.approx(0.5)


def test_single_state_requires_previous_state():
    state = np.concatenate((_pose([0, 0, 0]), _pose([0, 0, 0])))
    with pytest.raises(ValueError, match="requires pre_state"):
        umi_policy.UMIBimanualInputs(_model.ModelType.PI05)(_input(state))


def test_fixed_head_mask_is_applied_only_to_head_tokens():
    state = np.concatenate((_pose([0, 0, 0]), _pose([0, 0, 0])))
    data = _input(np.stack((state, state)))
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :112] = 255
    data["fixed_head_mask"] = mask

    result = umi_policy.UMIBimanualInputs(_model.ModelType.PI05)(data)

    assert result["image_token_mask"]["base_0_rgb"].sum() == 128
    for name, token_mask in result["image_token_mask"].items():
        if name != "base_0_rgb":
            assert token_mask.all()


def test_raw_head_repack_selects_raw_stream_and_disables_mask():
    raw = np.full((3, 4, 5), 7, dtype=np.uint8)
    data = {
        "observation.images.head_main": np.zeros_like(raw),
        "observation.images.head_raw": raw,
        "observation.images.right_hand_up": np.ones_like(raw),
        "observation.images.left_hand_up": np.full_like(raw, 2),
        "observation.images.fixed_head_mask": np.full((4, 5), 255, dtype=np.uint8),
        "observation.state": np.zeros(20, dtype=np.float32),
        "action": np.zeros((20, 20), dtype=np.float32),
        "prompt": "test",
    }

    result = umi_policy.UMIRepackTransform(
        head_feature="observation.images.head_raw",
        use_head_mask=False,
    )(data)

    np.testing.assert_array_equal(result["cam_head"], raw)
    assert "fixed_head_mask" not in result


def test_wrist_only_omits_and_masks_head_camera():
    state = np.concatenate((_pose([0, 0, 0]), _pose([0, 0, 0])))
    data = _input(np.stack((state, state)))
    data.pop("cam_head")

    result = umi_policy.UMIBimanualInputs(_model.ModelType.PI05, use_head_camera=False)(data)

    assert not result["image"]["base_0_rgb"].any()
    assert not result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]


def test_wrist_only_repack_does_not_require_head_feature():
    frame = np.zeros((3, 4, 5), dtype=np.uint8)
    data = {
        "observation.images.right_hand_up": frame,
        "observation.images.left_hand_up": frame,
        "observation.state": np.zeros(20, dtype=np.float32),
        "action": np.zeros((20, 20), dtype=np.float32),
        "prompt": "test",
    }

    result = umi_policy.UMIRepackTransform(use_head_camera=False, use_head_mask=False)(data)

    assert "cam_head" not in result
    assert "fixed_head_mask" not in result


def test_outputs_keep_only_robot_action_dimensions():
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)
    result = umi_policy.UMIBimanualOutputs()({"actions": actions})
    np.testing.assert_array_equal(result["actions"], actions[:, :20])
