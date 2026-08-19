from __future__ import annotations

import math

import numpy as np


def rotation_from_rpy(rpy_rad: list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy_rad]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    R_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_y @ R_x


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / axis_norm
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [x * x * one_minus_cosine + cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine, y * y * one_minus_cosine + cosine, y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, z * z * one_minus_cosine + cosine],
        ],
        dtype=np.float64,
    )


def homogeneous_from_rotation_translation(
    rotation_target_from_source: np.ndarray,
    translation_target_from_source_m: np.ndarray | list[float],
) -> np.ndarray:
    T_target_source = np.eye(4, dtype=np.float64)
    T_target_source[:3, :3] = np.asarray(rotation_target_from_source, dtype=np.float64)
    T_target_source[:3, 3] = np.asarray(translation_target_from_source_m, dtype=np.float64)
    return T_target_source


def homogeneous_from_xyz_rpy(
    xyz_target_from_source_m: list[float],
    rpy_target_from_source_rad: list[float],
) -> np.ndarray:
    return homogeneous_from_rotation_translation(
        rotation_from_rpy(rpy_target_from_source_rad),
        xyz_target_from_source_m,
    )


def transform_points(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    points_source = np.asarray(points_source, dtype=np.float64)
    return points_source @ T_target_source[:3, :3].T + T_target_source[:3, 3]


def validate_homogeneous(T_target_source: np.ndarray, name: str) -> None:
    T_target_source = np.asarray(T_target_source, dtype=np.float64)
    if T_target_source.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T_target_source.shape}")
    if not np.allclose(T_target_source[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    R_target_source = T_target_source[:3, :3]
    if not np.allclose(R_target_source.T @ R_target_source, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(R_target_source)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")


def apply_camera_frame_adjustment(
    T_camera_base_initial: np.ndarray,
    translation_camera_base_current_m: np.ndarray | list[float],
    rotation_adjustment_camera_rpy_rad: np.ndarray | list[float],
) -> np.ndarray:
    """Left-multiply a camera-frame RPY correction onto the initial rotation."""
    R_camera_adjustment = rotation_from_rpy(rotation_adjustment_camera_rpy_rad)
    R_camera_base_current = R_camera_adjustment @ T_camera_base_initial[:3, :3]
    return homogeneous_from_rotation_translation(
        R_camera_base_current,
        translation_camera_base_current_m,
    )

