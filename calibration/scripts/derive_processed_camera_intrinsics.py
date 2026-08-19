from __future__ import annotations

import argparse
import json
from pathlib import Path


def derive_center_square_resize(config: dict) -> dict:
    source_width, source_height = config["source_image_size_wh"]
    output_width, output_height = config["image_processing"]["output_image_size_wh"]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output image dimensions must be positive")

    crop_size = min(source_width, source_height)
    crop_x_px = (source_width - crop_size) / 2.0
    crop_y_px = (source_height - crop_size) / 2.0
    scale_x = output_width / crop_size
    scale_y = output_height / crop_size

    K_source = config["K_camera_head_main_source"]
    fx_source = float(K_source[0][0])
    fy_source = float(K_source[1][1])
    cx_source = float(K_source[0][2])
    cy_source = float(K_source[1][2])
    skew_source = float(K_source[0][1])

    K_processed = [
        [fx_source * scale_x, skew_source * scale_x, (cx_source - crop_x_px) * scale_x],
        [0.0, fy_source * scale_y, (cy_source - crop_y_px) * scale_y],
        [0.0, 0.0, 1.0],
    ]
    principal_point_inside_image = (
        0.0 <= K_processed[0][2] < output_width
        and 0.0 <= K_processed[1][2] < output_height
    )

    return {
        "status": "pass" if principal_point_inside_image else "warning",
        "camera_stream": config["camera_stream"],
        "coordinate_convention": {
            "image_origin": "top_left",
            "pixel_axes": "+u right, +v down",
            "K_direction": "camera coordinates to processed image pixels",
        },
        "source_image_size_wh": [source_width, source_height],
        "crop": {
            "x_px": crop_x_px,
            "y_px": crop_y_px,
            "width_px": crop_size,
            "height_px": crop_size,
        },
        "resize": {
            "output_image_size_wh": [output_width, output_height],
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
        "K_camera_head_main_source": K_source,
        "K_camera_head_main_processed": K_processed,
        "D_camera_head_main_processed": config["D_camera_head_main_source"],
        "distortion_model": config["distortion_model"],
        "distortion_api": config["distortion_api"],
        "distortion_order": config["distortion_order"],
        "calibration_provenance": config["calibration_provenance"],
        "validation": {
            "principal_point_inside_processed_image": principal_point_inside_image,
            "frames_are_undistorted": config["frames_are_undistorted"],
            "per_camera_calibration_confirmed": not config["requires_per_camera_confirmation"],
        },
        "limitations": config["notes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/camera_head_main.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/camera_intrinsics_inspection.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    operation = config["image_processing"]["operation"]
    if operation != "center_square_then_resize":
        raise ValueError(f"unsupported image processing operation: {operation}")

    report = derive_center_square_resize(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
