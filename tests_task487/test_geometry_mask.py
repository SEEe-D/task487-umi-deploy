import json
from pathlib import Path

import numpy as np

from pi05_geometry_mask import camera_matrix_for_image_geometry


def _source_calibration():
    path = Path(__file__).parents[1] / "geometry_mask/configs/camera_head_main.json"
    config = json.loads(path.read_text())
    return np.asarray(config["K_camera_head_main_source"]), config["source_image_size_wh"]


def test_resize_with_pad_camera_matrix_matches_umi_env_pixel_geometry():
    source, source_size = _source_calibration()
    mapped = camera_matrix_for_image_geometry(source, source_size, (224, 224), "resize_with_pad")

    # The 1920x1536 calibration and 640x512 Thor stream have the same aspect
    # ratio. UmiEnv maps either one to 224x179, then pads 22 pixels above and
    # 23 below.
    source_width, source_height = source_size
    expected = source.copy().astype(np.float64)
    expected[0] *= 224 / source_width
    expected[1] *= 179 / source_height
    expected[1, 2] += 22
    np.testing.assert_allclose(mapped, expected, atol=1e-12)


def test_stretch_geometry_keeps_legacy_mapping_and_unknown_mode_fails():
    source, source_size = _source_calibration()
    mapped = camera_matrix_for_image_geometry(source, source_size, (224, 224), "stretch")
    source_width, source_height = source_size
    expected = source.copy().astype(np.float64)
    expected[0] *= 224 / source_width
    expected[1] *= 224 / source_height
    np.testing.assert_allclose(mapped, expected, atol=1e-12)

    with np.testing.assert_raises_regex(ValueError, "Unsupported geometry-mask image geometry"):
        camera_matrix_for_image_geometry(source, source_size, (224, 224), "center_square")
