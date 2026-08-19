from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .transforms import apply_camera_frame_adjustment, validate_homogeneous


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_processed_fisheye_camera(
    path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, tuple[int, int]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("distortion_model") != "equidistant":
        raise ValueError("camera config must use distortion_model: equidistant")
    if config.get("distortion_api") != "cv2.fisheye":
        raise ValueError("camera config must use distortion_api: cv2.fisheye")
    if config.get("frames_are_undistorted") is not False:
        raise ValueError("projection expects distorted source frames")
    source_width, source_height = config["source_image_size_wh"]
    output_width, output_height = config["image_processing"][
        "output_image_size_wh"
    ]
    K_source = np.asarray(config["K_camera_head_main_source"], dtype=np.float64)
    operation = config["image_processing"]["operation"]
    if operation == "center_square_then_resize":
        crop_size = min(source_width, source_height)
        crop_x_px = (source_width - crop_size) / 2.0
        crop_y_px = (source_height - crop_size) / 2.0
        resized_width, resized_height = output_width, output_height
        pad_left = pad_top = 0
        principal_x = K_source[0, 2] - crop_x_px
        principal_y = K_source[1, 2] - crop_y_px
        scale_x = output_width / crop_size
        scale_y = output_height / crop_size
    elif operation == "resize_with_pad":
        ratio = max(source_width / output_width, source_height / output_height)
        resized_width = int(source_width / ratio)
        resized_height = int(source_height / ratio)
        pad_left = (output_width - resized_width) // 2
        pad_top = (output_height - resized_height) // 2
        principal_x = K_source[0, 2]
        principal_y = K_source[1, 2]
        scale_x = resized_width / source_width
        scale_y = resized_height / source_height
    else:
        raise ValueError(f"unsupported image_processing.operation: {operation!r}")
    K_camera_image = np.array(
        [
            [
                K_source[0, 0] * scale_x,
                K_source[0, 1] * scale_x,
                principal_x * scale_x + pad_left,
            ],
            [
                0.0,
                K_source[1, 1] * scale_y,
                principal_y * scale_y + pad_top,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D_camera_fisheye = np.asarray(
        config["D_camera_head_main_source"], dtype=np.float64
    ).reshape(4, 1)
    return (
        config,
        K_camera_image,
        D_camera_fisheye,
        (int(output_width), int(output_height)),
    )


def load_initial_camera_from_base(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    config = load_yaml(path)
    if config.get("transform_direction") != "camera_from_base":
        raise ValueError("extrinsic YAML must declare transform_direction: camera_from_base")
    T_camera_left_base = np.asarray(config["T_camera_left_base"], dtype=np.float64)
    T_camera_right_base = np.asarray(config["T_camera_right_base"], dtype=np.float64)
    validate_homogeneous(T_camera_left_base, "T_camera_left_base")
    validate_homogeneous(T_camera_right_base, "T_camera_right_base")
    return config, T_camera_left_base, T_camera_right_base


def current_camera_from_base(
    T_camera_base_initial: np.ndarray,
    translation_camera_base_current_m: np.ndarray,
    rotation_adjustment_camera_rpy_rad: np.ndarray,
) -> np.ndarray:
    return apply_camera_frame_adjustment(
        T_camera_base_initial,
        translation_camera_base_current_m,
        rotation_adjustment_camera_rpy_rad,
    )


def save_adjusted_camera_from_base(
    output_path: Path,
    initial_config_path: Path,
    initial_config: dict,
    T_camera_left_base: np.ndarray,
    T_camera_right_base: np.ndarray,
    left_rotation_adjustment_rpy_rad: np.ndarray,
    right_rotation_adjustment_rpy_rad: np.ndarray,
) -> None:
    if output_path.resolve() == initial_config_path.resolve():
        raise ValueError("adjusted extrinsics must not overwrite the initial YAML")
    output = deepcopy(initial_config)
    output["calibration_status"] = "manually_adjusted"
    output["parent_initial_yaml"] = str(initial_config_path.resolve())
    output["T_camera_left_base"] = T_camera_left_base.tolist()
    output["T_camera_right_base"] = T_camera_right_base.tolist()
    output["left_current_translation_m"] = T_camera_left_base[:3, 3].tolist()
    output["right_current_translation_m"] = T_camera_right_base[:3, 3].tolist()
    output["left_rotation_adjustment_camera_rpy_rad"] = (
        left_rotation_adjustment_rpy_rad.tolist()
    )
    output["right_rotation_adjustment_camera_rpy_rad"] = (
        right_rotation_adjustment_rpy_rad.tolist()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
