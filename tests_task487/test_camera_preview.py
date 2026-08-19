import numpy as np
from openpi.policies.umi_policy import fixed_head_mask_to_token_keep_mask

from task487_client import CameraPreview


def test_camera_preview_preserves_exact_policy_pixels_and_order():
    observation = {
        "cam_head": np.full((224, 224, 3), [10, 20, 30], dtype=np.uint8),
        "cam_left_top": np.full((224, 224, 3), [40, 50, 60], dtype=np.uint8),
        "cam_right_top": np.full((224, 224, 3), [70, 80, 90], dtype=np.uint8),
    }
    originals = {key: value.copy() for key, value in observation.items()}

    canvas = CameraPreview.compose(observation, "test")

    assert canvas.shape == (264, 672, 3)
    np.testing.assert_array_equal(canvas[40:, 0:224], observation["cam_head"][..., ::-1])
    np.testing.assert_array_equal(canvas[40:, 224:448], observation["cam_left_top"][..., ::-1])
    np.testing.assert_array_equal(canvas[40:, 448:672], observation["cam_right_top"][..., ::-1])
    for key in observation:
        np.testing.assert_array_equal(observation[key], originals[key])


def test_processed_camera_preview_shows_mask_without_changing_policy_pixels():
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[-28:, :] = 255
    observation = {
        "cam_head": np.full((224, 224, 3), [10, 20, 30], dtype=np.uint8),
        "cam_left_top": np.full((224, 224, 3), [40, 50, 60], dtype=np.uint8),
        "cam_right_top": np.full((224, 224, 3), [70, 80, 90], dtype=np.uint8),
        "fixed_head_mask": mask,
    }
    originals = {key: value.copy() for key, value in observation.items()}

    canvas = CameraPreview.compose_processed(observation, "processed test")

    assert canvas.shape == (548, 672, 3)
    np.testing.assert_array_equal(canvas[68:292, 0:224], observation["cam_head"][..., ::-1])
    np.testing.assert_array_equal(canvas[68:292, 224:448], observation["cam_left_top"][..., ::-1])
    np.testing.assert_array_equal(canvas[68:292, 448:672], observation["cam_right_top"][..., ::-1])
    # The lower overlay leaves unmasked pixels unchanged and tints masked pixels red.
    np.testing.assert_array_equal(canvas[324 + 20, 20], observation["cam_head"][20, 20, ::-1])
    assert not np.array_equal(canvas[324 + 210, 20], observation["cam_head"][210, 20, ::-1])
    for key in observation:
        np.testing.assert_array_equal(observation[key], originals[key])


def test_preview_token_pooling_matches_masked_policy_transform():
    rng = np.random.default_rng(7)
    for _ in range(8):
        mask = (rng.random((224, 224)) < rng.uniform(0.05, 0.8)).astype(np.uint8) * 255
        expected = fixed_head_mask_to_token_keep_mask(mask).reshape(16, 16)
        np.testing.assert_array_equal(CameraPreview.head_token_keep_mask(mask), expected)


def test_processed_preview_reports_joint_and_mask_motion_from_first_frame():
    preview = CameraPreview(enabled=False, show_processed=True)
    first_mask = np.zeros((224, 224), dtype=np.uint8)
    moved_mask = first_mask.copy()
    moved_mask[10:15, 20:24] = 255

    first = preview.motion_status({"fixed_head_mask": first_mask}, np.zeros(14))
    moved_q = np.zeros(14)
    moved_q[3] = np.deg2rad(2.5)
    moved = preview.motion_status({"fixed_head_mask": moved_mask}, moved_q)

    assert first == "q_from_start=0.00deg mask_from_start=0px"
    assert moved == "q_from_start=2.50deg mask_from_start=20px"
