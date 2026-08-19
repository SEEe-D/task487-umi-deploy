# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "trimesh==4.7.1",
#   "fast-simplification",
#   "opencv-python==4.10.0.84",
#   "pyyaml==6.0.2",
#   "pyarrow==24.0.0",
#   "pandas==2.2.3",
#   "numba",
# ]
# ///

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.dual_arm_projection.configuration import (
    load_initial_camera_from_base,
    load_processed_fisheye_camera,
)
from tools.dual_arm_projection.dataset import LeRobotProjectionDataset, ProjectionFrame
from tools.dual_arm_projection.rasterizer import FisheyeArmRasterizer, RenderResult
from tools.dual_arm_projection.urdf_model import load_arm_model


def write_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    encoded_ok, encoded = cv2.imencode(path.suffix, image)
    if not encoded_ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(path)


def parse_frame_indices(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def write_frame_outputs(
    output_dir: Path,
    projection_frame: ProjectionFrame,
    render_result: RenderResult,
) -> dict:
    frame_dir = output_dir / f"frame_{projection_frame.relative_frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_image": frame_dir / "source.png",
        "left_mask": frame_dir / "left_mask.png",
        "right_mask": frame_dir / "right_mask.png",
        "combined_mask": frame_dir / "combined_mask.png",
        "link_id_uint16": frame_dir / "link_id_uint16.png",
        "link_id_color": frame_dir / "link_id_color.png",
        "overlay": frame_dir / "overlay.png",
        "z_buffer": frame_dir / "z_buffer_m.npy",
    }
    image_values = {
        "source_image": projection_frame.image_bgr,
        "left_mask": render_result.left_mask,
        "right_mask": render_result.right_mask,
        "combined_mask": render_result.combined_mask,
        "link_id_uint16": render_result.link_id_image,
        "link_id_color": render_result.link_id_color_bgr,
        "overlay": render_result.overlay_bgr,
    }
    for name, image in image_values.items():
        write_image_unicode_safe(paths[name], image)
    np.save(paths["z_buffer"], render_result.z_buffer_m)
    return {name: str(path.resolve()) for name, path in paths.items()}


def render_one(
    rasterizer: FisheyeArmRasterizer,
    projection_frame: ProjectionFrame,
    T_camera_left_base: np.ndarray,
    T_camera_right_base: np.ndarray,
    overlay_alpha: float,
) -> RenderResult:
    return rasterizer.render(
        projection_frame.image_bgr,
        T_camera_left_base,
        T_camera_right_base,
        projection_frame.left_joint_angles_rad,
        projection_frame.right_joint_angles_rad,
        overlay_alpha=overlay_alpha,
    )


def write_sequence(
    output_dir: Path,
    rasterizer: FisheyeArmRasterizer,
    projection_frames: list[ProjectionFrame],
    T_camera_left_base: np.ndarray,
    T_camera_right_base: np.ndarray,
    fps: float,
    overlay_alpha: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = rasterizer.image_size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path = output_dir / "episode_overlay.mp4"
    combined_mask_path = output_dir / "episode_combined_mask.mp4"
    link_id_path = output_dir / "episode_link_id_color.mp4"
    frame_statistics = []
    started_at = perf_counter()
    with tempfile.TemporaryDirectory(prefix="dual_arm_mask_video_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_overlay_path = temporary_root / overlay_path.name
        temporary_mask_path = temporary_root / combined_mask_path.name
        temporary_link_id_path = temporary_root / link_id_path.name
        overlay_writer = cv2.VideoWriter(
            str(temporary_overlay_path), fourcc, fps, (width, height), True
        )
        mask_writer = cv2.VideoWriter(
            str(temporary_mask_path), fourcc, fps, (width, height), False
        )
        link_writer = cv2.VideoWriter(
            str(temporary_link_id_path), fourcc, fps, (width, height), True
        )
        if (
            not overlay_writer.isOpened()
            or not mask_writer.isOpened()
            or not link_writer.isOpened()
        ):
            raise RuntimeError("failed to open one or more sequence video writers")
        try:
            for projection_frame in projection_frames:
                render_result = render_one(
                    rasterizer,
                    projection_frame,
                    T_camera_left_base,
                    T_camera_right_base,
                    overlay_alpha,
                )
                overlay_writer.write(render_result.overlay_bgr)
                mask_writer.write(render_result.combined_mask)
                link_writer.write(render_result.link_id_color_bgr)
                frame_statistics.append(
                    {
                        "relative_frame_index": projection_frame.relative_frame_index,
                        "dataset_frame_index": projection_frame.dataset_frame_index,
                        "statistics": render_result.statistics,
                    }
                )
        finally:
            overlay_writer.release()
            mask_writer.release()
            link_writer.release()
        shutil.copyfile(temporary_overlay_path, overlay_path)
        shutil.copyfile(temporary_mask_path, combined_mask_path)
        shutil.copyfile(temporary_link_id_path, link_id_path)
    elapsed_seconds = perf_counter() - started_at
    return {
        "overlay_video": str(overlay_path.resolve()),
        "combined_mask_video": str(combined_mask_path.resolve()),
        "link_id_color_video": str(link_id_path.resolve()),
        "frame_count": len(projection_frames),
        "fps": fps,
        "render_elapsed_seconds": elapsed_seconds,
        "render_frames_per_second": (
            len(projection_frames) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "frame_statistics": frame_statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--camera-config", type=Path, default=Path("configs/camera_head_main.json")
    )
    parser.add_argument(
        "--extrinsics-yaml",
        type=Path,
        default=Path("configs/dual_arm_extrinsics_initial.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/dual_arm_projection")
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-indices", default="0")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--sequence-max-frames", type=int, default=0)
    parser.add_argument("--max-faces-per-link", type=int, default=600)
    parser.add_argument(
        "--geometry-mode",
        choices=("visual_mesh", "capsule_proxy"),
        default="capsule_proxy",
    )
    parser.add_argument("--capsule-radius-margin-m", type=float, default=0.002)
    parser.add_argument("--capsule-sections", type=int, default=16)
    parser.add_argument("--sphere-subdivisions", type=int, default=2)
    parser.add_argument("--terminal-link-radius-m", type=float, default=0.04)
    parser.add_argument(
        "--terminal-distal-sphere-scale", type=float, default=0.25
    )
    parser.add_argument("--mask-dilation-px", type=int, default=2)
    parser.add_argument("--overlay-alpha", type=float, default=0.48)
    args = parser.parse_args()

    camera_config, K_camera_image, D_camera_fisheye, image_size_wh = (
        load_processed_fisheye_camera(args.camera_config)
    )
    extrinsic_config, T_camera_left_base, T_camera_right_base = (
        load_initial_camera_from_base(args.extrinsics_yaml)
    )
    left_arm = load_arm_model(
        args.urdf.resolve(),
        "L",
        args.max_faces_per_link,
        geometry_mode=args.geometry_mode,
        capsule_radius_margin_m=args.capsule_radius_margin_m,
        capsule_sections=args.capsule_sections,
        sphere_subdivisions=args.sphere_subdivisions,
        terminal_link_radius_m=args.terminal_link_radius_m,
        terminal_distal_sphere_scale=args.terminal_distal_sphere_scale,
    )
    right_arm = load_arm_model(
        args.urdf.resolve(),
        "R",
        args.max_faces_per_link,
        geometry_mode=args.geometry_mode,
        capsule_radius_margin_m=args.capsule_radius_margin_m,
        capsule_sections=args.capsule_sections,
        sphere_subdivisions=args.sphere_subdivisions,
        terminal_link_radius_m=args.terminal_link_radius_m,
        terminal_distal_sphere_scale=args.terminal_distal_sphere_scale,
    )
    rasterizer = FisheyeArmRasterizer(
        K_camera_image,
        D_camera_fisheye,
        image_size_wh,
        left_arm,
        right_arm,
        mask_dilation_px=args.mask_dilation_px,
    )
    dataset = LeRobotProjectionDataset(args.dataset_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = parse_frame_indices(args.frame_indices)
    selected_frames = dataset.read_frames(args.episode_index, selected_indices)
    selected_reports = []
    for projection_frame in selected_frames:
        render_result = render_one(
            rasterizer,
            projection_frame,
            T_camera_left_base,
            T_camera_right_base,
            args.overlay_alpha,
        )
        selected_reports.append(
            {
                "episode_index": projection_frame.episode_index,
                "source_episode": projection_frame.source_episode,
                "relative_frame_index": projection_frame.relative_frame_index,
                "dataset_frame_index": projection_frame.dataset_frame_index,
                "timestamp_s": projection_frame.timestamp_s,
                "outputs": write_frame_outputs(
                    output_dir, projection_frame, render_result
                ),
                "statistics": render_result.statistics,
            }
        )

    sequence_report = None
    if args.sequence:
        episode_length = dataset.episode_length(args.episode_index)
        sequence_indices = list(range(0, episode_length, args.sequence_stride))
        if sequence_indices[-1] != episode_length - 1:
            sequence_indices.append(episode_length - 1)
        if args.sequence_max_frames > 0:
            sequence_indices = sequence_indices[: args.sequence_max_frames]
        sequence_frames = dataset.read_frames(args.episode_index, sequence_indices)
        sequence_report = write_sequence(
            output_dir / "sequence",
            rasterizer,
            sequence_frames,
            T_camera_left_base,
            T_camera_right_base,
            fps=25.0 / args.sequence_stride,
            overlay_alpha=args.overlay_alpha,
        )

    report = {
        "status": "pass",
        "scope": "offline projection only; StarVLA and token masks are unchanged",
        "dataset_root": str(args.dataset_root.resolve()),
        "urdf_path": str(args.urdf.resolve()),
        "camera_config_path": str(args.camera_config.resolve()),
        "extrinsics_yaml_path": str(args.extrinsics_yaml.resolve()),
        "projection_model": camera_config["distortion_model"],
        "projection_api": camera_config["distortion_api"],
        "image_size_wh": list(image_size_wh),
        "K_camera_image": K_camera_image.tolist(),
        "D_camera_fisheye": D_camera_fisheye.reshape(-1).tolist(),
        "coordinate_system": "OpenCV camera: +x right, +y down, +z forward",
        "T_camera_left_base": T_camera_left_base.tolist(),
        "T_camera_right_base": T_camera_right_base.tolist(),
        "transform_direction": extrinsic_config["transform_direction"],
        "excluded_links": {
            "left": list(left_arm.excluded_links),
            "right": list(right_arm.excluded_links),
        },
        "max_faces_per_link": args.max_faces_per_link,
        "geometry_mode": args.geometry_mode,
        "capsule_radius_margin_m": args.capsule_radius_margin_m,
        "terminal_link_radius_m": args.terminal_link_radius_m,
        "terminal_distal_sphere_scale": args.terminal_distal_sphere_scale,
        "mask_dilation_px": args.mask_dilation_px,
        "selected_frames": selected_reports,
        "sequence": sequence_report,
    }
    report_path = output_dir / "projection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_frame_count": len(selected_reports),
                "sequence_frame_count": sequence_report["frame_count"]
                if sequence_report
                else 0,
                "report": str(report_path),
                "excluded_links": report["excluded_links"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
