from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit

from .transforms import transform_points, validate_homogeneous
from .urdf_model import ArmModel


@njit(cache=True)
def _rasterize_projected_triangles(
    projected_uv: np.ndarray,
    vertex_distance_m: np.ndarray,
    vertex_z_camera_m: np.ndarray,
    faces: np.ndarray,
    link_id: int,
    near_plane_m: float,
    z_buffer_m: np.ndarray,
    link_id_image: np.ndarray,
) -> tuple[int, int]:
    height, width = z_buffer_m.shape
    rendered_triangle_count = 0
    rejected_near_triangle_count = 0
    for face_index in range(faces.shape[0]):
        i0, i1, i2 = faces[face_index]
        if (
            vertex_z_camera_m[i0] <= near_plane_m
            or vertex_z_camera_m[i1] <= near_plane_m
            or vertex_z_camera_m[i2] <= near_plane_m
        ):
            rejected_near_triangle_count += 1
            continue
        u0, v0 = projected_uv[i0]
        u1, v1 = projected_uv[i1]
        u2, v2 = projected_uv[i2]
        if not (
            np.isfinite(u0)
            and np.isfinite(v0)
            and np.isfinite(u1)
            and np.isfinite(v1)
            and np.isfinite(u2)
            and np.isfinite(v2)
        ):
            continue
        denominator = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denominator) < 1e-10:
            continue
        x_min = max(0, int(np.floor(min(u0, u1, u2))))
        x_max = min(width - 1, int(np.ceil(max(u0, u1, u2))))
        y_min = max(0, int(np.floor(min(v0, v1, v2))))
        y_max = min(height - 1, int(np.ceil(max(v0, v1, v2))))
        if x_min > x_max or y_min > y_max:
            continue
        rendered_triangle_count += 1
        inverse_d0 = 1.0 / vertex_distance_m[i0]
        inverse_d1 = 1.0 / vertex_distance_m[i1]
        inverse_d2 = 1.0 / vertex_distance_m[i2]
        for y_pixel in range(y_min, y_max + 1):
            y_center = y_pixel + 0.5
            for x_pixel in range(x_min, x_max + 1):
                x_center = x_pixel + 0.5
                weight0 = (
                    (v1 - v2) * (x_center - u2)
                    + (u2 - u1) * (y_center - v2)
                ) / denominator
                weight1 = (
                    (v2 - v0) * (x_center - u2)
                    + (u0 - u2) * (y_center - v2)
                ) / denominator
                weight2 = 1.0 - weight0 - weight1
                if weight0 < -1e-7 or weight1 < -1e-7 or weight2 < -1e-7:
                    continue
                inverse_distance = (
                    weight0 * inverse_d0
                    + weight1 * inverse_d1
                    + weight2 * inverse_d2
                )
                if inverse_distance <= 0.0:
                    continue
                distance_m = 1.0 / inverse_distance
                if distance_m < z_buffer_m[y_pixel, x_pixel]:
                    z_buffer_m[y_pixel, x_pixel] = distance_m
                    link_id_image[y_pixel, x_pixel] = link_id
    return rendered_triangle_count, rejected_near_triangle_count


@dataclass(frozen=True)
class RenderResult:
    left_mask: np.ndarray
    right_mask: np.ndarray
    combined_mask: np.ndarray
    link_id_image: np.ndarray
    z_buffer_m: np.ndarray
    link_id_color_bgr: np.ndarray
    overlay_bgr: np.ndarray
    statistics: dict


class FisheyeArmRasterizer:
    def __init__(
        self,
        K_camera_image: np.ndarray,
        D_camera_fisheye: np.ndarray,
        image_size_wh: tuple[int, int],
        left_arm: ArmModel,
        right_arm: ArmModel,
        near_plane_m: float = 0.02,
        mask_dilation_px: int = 0,
    ) -> None:
        self.K_camera_image = np.asarray(K_camera_image, dtype=np.float64)
        self.D_camera_fisheye = np.asarray(D_camera_fisheye, dtype=np.float64).reshape(4, 1)
        self.image_size_wh = image_size_wh
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.near_plane_m = float(near_plane_m)
        if mask_dilation_px < 0:
            raise ValueError("mask_dilation_px must be non-negative")
        self.mask_dilation_px = int(mask_dilation_px)
        self.link_name_by_id = {
            link.link_id: link.link_name
            for arm in (left_arm, right_arm)
            for link in arm.render_links
        }

    def render(
        self,
        image_bgr: np.ndarray,
        T_camera_left_base: np.ndarray,
        T_camera_right_base: np.ndarray,
        left_joint_angles_rad: dict[str, float],
        right_joint_angles_rad: dict[str, float],
        overlay_alpha: float = 0.48,
    ) -> RenderResult:
        validate_homogeneous(T_camera_left_base, "T_camera_left_base")
        validate_homogeneous(T_camera_right_base, "T_camera_right_base")
        width, height = self.image_size_wh
        if image_bgr.shape[:2] != (height, width):
            raise ValueError(
                f"expected image {(height, width)}, got {image_bgr.shape[:2]}"
            )
        z_buffer_m = np.full((height, width), np.inf, dtype=np.float64)
        link_id_image = np.zeros((height, width), dtype=np.uint16)
        statistics = {
            "coordinate_system": "OpenCV camera: +x right, +y down, +z forward",
            "projection_model": "cv2.fisheye.projectPoints / equidistant",
            "renderer": "CPU triangle rasterization with per-pixel Z-buffer",
            "rendered_triangle_count": 0,
            "rejected_near_triangle_count": 0,
            "links": [],
        }
        for arm, T_camera_base, joint_angles_rad in (
            (self.left_arm, T_camera_left_base, left_joint_angles_rad),
            (self.right_arm, T_camera_right_base, right_joint_angles_rad),
        ):
            T_base_link_by_name = arm.forward_kinematics(joint_angles_rad)
            for render_link in arm.render_links:
                T_base_link = T_base_link_by_name[render_link.link_name]
                vertices_base_m = transform_points(
                    T_base_link, render_link.vertices_link_m
                )
                vertices_camera_m = transform_points(
                    T_camera_base, vertices_base_m
                )
                projected_uv, _ = cv2.fisheye.projectPoints(
                    vertices_camera_m.reshape(-1, 1, 3),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    self.K_camera_image,
                    self.D_camera_fisheye,
                )
                projected_uv = projected_uv.reshape(-1, 2)
                vertex_distance_m = np.linalg.norm(vertices_camera_m, axis=1)
                rendered_count, rejected_count = _rasterize_projected_triangles(
                    projected_uv,
                    vertex_distance_m,
                    vertices_camera_m[:, 2],
                    render_link.faces,
                    render_link.link_id,
                    self.near_plane_m,
                    z_buffer_m,
                    link_id_image,
                )
                statistics["rendered_triangle_count"] += int(rendered_count)
                statistics["rejected_near_triangle_count"] += int(rejected_count)
                statistics["links"].append(
                    {
                        "side": arm.side,
                        "link_name": render_link.link_name,
                        "link_id": render_link.link_id,
                        "source_face_count": render_link.source_face_count,
                        "rendered_face_count": render_link.rendered_face_count,
                        "visual_mesh_path": str(render_link.mesh_path),
                        "visual_scale": render_link.visual_scale.tolist(),
                        "T_link_visual": render_link.T_link_visual.tolist(),
                        "geometry_mode": render_link.geometry_mode,
                        "proxy_metadata": render_link.proxy_metadata,
                    }
                )

        left_mask_raw = (
            ((link_id_image > 0) & (link_id_image < 100)).astype(np.uint8) * 255
        )
        right_mask_raw = (link_id_image >= 100).astype(np.uint8) * 255
        left_mask = left_mask_raw
        right_mask = right_mask_raw
        if self.mask_dilation_px > 0:
            kernel_size = 2 * self.mask_dilation_px + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            left_mask = cv2.dilate(left_mask_raw, kernel, iterations=1)
            right_mask = cv2.dilate(right_mask_raw, kernel, iterations=1)
        combined_mask = ((left_mask > 0) | (right_mask > 0)).astype(np.uint8) * 255
        link_id_color_bgr = self.colorize_link_ids(link_id_image)
        link_id_color_bgr[(left_mask > 0) & (left_mask_raw == 0)] = (80, 220, 255)
        link_id_color_bgr[(right_mask > 0) & (right_mask_raw == 0)] = (255, 180, 80)
        overlay_bgr = image_bgr.copy()
        foreground = combined_mask > 0
        overlay_bgr[foreground] = cv2.addWeighted(
            image_bgr[foreground],
            1.0 - overlay_alpha,
            link_id_color_bgr[foreground],
            overlay_alpha,
            0.0,
        )
        statistics["left_mask_pixel_count"] = int(np.count_nonzero(left_mask))
        statistics["right_mask_pixel_count"] = int(np.count_nonzero(right_mask))
        statistics["combined_mask_pixel_count"] = int(np.count_nonzero(combined_mask))
        statistics["raw_combined_mask_pixel_count"] = int(
            np.count_nonzero((left_mask_raw > 0) | (right_mask_raw > 0))
        )
        statistics["mask_dilation_px"] = self.mask_dilation_px
        statistics["visible_link_ids"] = [
            int(value) for value in np.unique(link_id_image) if value != 0
        ]
        return RenderResult(
            left_mask=left_mask,
            right_mask=right_mask,
            combined_mask=combined_mask,
            link_id_image=link_id_image,
            z_buffer_m=z_buffer_m,
            link_id_color_bgr=link_id_color_bgr,
            overlay_bgr=overlay_bgr,
            statistics=statistics,
        )

    @staticmethod
    def colorize_link_ids(link_id_image: np.ndarray) -> np.ndarray:
        color_bgr = np.zeros((*link_id_image.shape, 3), dtype=np.uint8)
        palette = [
            (80, 220, 255),
            (40, 180, 255),
            (40, 230, 120),
            (255, 180, 60),
            (255, 100, 100),
            (220, 80, 220),
            (180, 220, 80),
            (100, 140, 255),
        ]
        for link_id in np.unique(link_id_image):
            if link_id == 0:
                continue
            local_index = (int(link_id) - 1) % 100
            color = palette[local_index % len(palette)]
            if link_id >= 100:
                color = (color[2], color[1], color[0])
            color_bgr[link_id_image == link_id] = color
        return color_bgr
