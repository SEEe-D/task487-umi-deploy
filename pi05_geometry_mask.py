from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


_ROOT = Path(__file__).resolve().parent
_GEOMETRY_ROOT = _ROOT / "geometry_mask"


def camera_matrix_for_image_geometry(
    source_matrix: Any,
    source_size_wh: tuple[int, int] | list[int],
    output_size_wh: tuple[int, int],
    image_geometry: str,
) -> np.ndarray:
    """Map calibrated source intrinsics into the exact deployed RGB geometry."""
    matrix = np.asarray(source_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Camera matrix must have shape (3, 3), got {matrix.shape}")
    source_width, source_height = (int(value) for value in source_size_wh)
    output_width, output_height = (int(value) for value in output_size_wh)
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("Camera source and output dimensions must be positive")

    if image_geometry == "stretch":
        resized_width, resized_height = output_width, output_height
        pad_left = pad_top = 0
    elif image_geometry == "resize_with_pad":
        ratio = max(source_width / output_width, source_height / output_height)
        resized_width = int(source_width / ratio)
        resized_height = int(source_height / ratio)
        pad_left = (output_width - resized_width) // 2
        pad_top = (output_height - resized_height) // 2
    else:
        raise ValueError(
            f"Unsupported geometry-mask image geometry {image_geometry!r}; "
            "expected 'stretch' or 'resize_with_pad'"
        )

    scale_x = resized_width / float(source_width)
    scale_y = resized_height / float(source_height)
    return np.array(
        [
            [matrix[0, 0] * scale_x, matrix[0, 1] * scale_x, matrix[0, 2] * scale_x + pad_left],
            [0.0, matrix[1, 1] * scale_y, matrix[1, 2] * scale_y + pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def geometry_q14_from_observation(observation: dict[str, Any]) -> np.ndarray:
    """Read camera-timestamp-aligned joints from UmiEnv in renderer order."""
    joints = []
    for robot_index, side in enumerate(("right", "left")):
        key = f"robot{robot_index}_joint_pos"
        if key not in observation:
            raise KeyError(f"Observation has no synchronized {key}")
        history = np.asarray(observation[key], dtype=np.float64)
        if history.ndim != 2 or history.shape[1] != 7 or history.shape[0] == 0:
            raise ValueError(f"{side} synchronized joints must have shape (T, 7), got {history.shape}")
        if not np.isfinite(history[-1]).all():
            raise ValueError(f"{side} synchronized joints contain NaN or infinity")
        joints.append(history[-1])
    return np.concatenate(joints)


class GeometryMasker:
    """Render the calibrated dual-arm silhouette in the Pi0.5 head-image frame."""

    def __init__(
        self,
        image_size: int = 224,
        mask_dilation_px: int = 2,
        image_geometry: str = "stretch",
    ) -> None:
        self.image_size = int(image_size)
        self.mask_dilation_px = int(mask_dilation_px)
        self.image_geometry = str(image_geometry)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self._initialize()

    def _initialize(self) -> None:
        from geometry_mask.tools.dual_arm_projection.rasterizer import FisheyeArmRasterizer
        from geometry_mask.tools.dual_arm_projection.urdf_model import load_arm_model

        camera_config = json.loads(
            (_GEOMETRY_ROOT / "configs/camera_head_main.json").read_text(encoding="utf-8")
        )
        if camera_config.get("distortion_model") != "equidistant":
            raise ValueError("Geometry camera must use equidistant distortion")
        k_source = np.asarray(camera_config["K_camera_head_main_source"], dtype=np.float64)
        k_image = camera_matrix_for_image_geometry(
            k_source,
            camera_config["source_image_size_wh"],
            (self.image_size, self.image_size),
            self.image_geometry,
        )
        distortion = np.asarray(camera_config["D_camera_head_main_source"], dtype=np.float64)

        extrinsics = yaml.safe_load(
            (_GEOMETRY_ROOT / "configs/dual_arm_extrinsics_calibrated.yaml").read_text(
                encoding="utf-8"
            )
        )
        if extrinsics.get("transform_direction") != "camera_from_base":
            raise ValueError("Extrinsics must declare camera_from_base")
        self._t_camera_left_base = np.asarray(extrinsics["T_camera_left_base"], dtype=np.float64)
        self._t_camera_right_base = np.asarray(extrinsics["T_camera_right_base"], dtype=np.float64)

        urdfs = list((_GEOMETRY_ROOT / "robot_model").rglob("*.urdf"))
        if len(urdfs) != 1:
            raise RuntimeError(f"Expected one geometry URDF, found {urdfs}")
        model_args = {
            "max_faces_per_link": 600,
            "geometry_mode": "capsule_proxy",
            "capsule_radius_margin_m": 0.002,
            "capsule_sections": 16,
            "sphere_subdivisions": 2,
            "terminal_link_radius_m": 0.04,
            "terminal_distal_sphere_scale": 0.25,
        }
        left = load_arm_model(urdfs[0], "L", **model_args)
        right = load_arm_model(urdfs[0], "R", **model_args)
        self._rasterizer = FisheyeArmRasterizer(
            k_image,
            distortion,
            (self.image_size, self.image_size),
            left,
            right,
            mask_dilation_px=self.mask_dilation_px,
        )

    @staticmethod
    def _joint_dicts(q14: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        right = {f"Joint{index + 1}_R": float(q14[index]) for index in range(7)}
        left = {f"Joint{index + 1}_L": float(q14[index + 7]) for index in range(7)}
        return left, right

    def render(self, head_rgb: Any, q14: Any) -> dict[str, Any]:
        image = np.asarray(head_rgb)
        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError(
                f"Head image must be {self.image_size}x{self.image_size} RGB, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Head image must be uint8, got {image.dtype}")
        q = np.asarray(q14, dtype=np.float64)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError(f"Geometry joint vector must be finite shape (14,), got {q.shape}")

        left, right = self._joint_dicts(q)
        result = self._rasterizer.render(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            self._t_camera_left_base,
            self._t_camera_right_base,
            left,
            right,
        )
        return {
            "combined_mask": result.combined_mask,
            "overlay_rgb": cv2.cvtColor(result.overlay_bgr, cv2.COLOR_BGR2RGB),
            "statistics": result.statistics,
        }
