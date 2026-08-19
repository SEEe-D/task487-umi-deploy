"""Build the minimal LeRobot projection dataset from a live recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


def process_head_frame(frame: np.ndarray, image_geometry: str) -> np.ndarray:
    """Match the deployed Task487 640x512 -> 224x224 image transform."""
    if frame.shape[:2] != (512, 640):
        raise ValueError(f"expected 640x512 head frame, got {frame.shape[:2]}")
    if image_geometry == "center_square":
        return cv2.resize(frame[:, 64:576], (224, 224), interpolation=cv2.INTER_AREA)
    if image_geometry != "resize_with_pad":
        raise ValueError(f"unsupported image geometry: {image_geometry}")

    ratio = max(640 / 224, 512 / 224)
    resized_width = int(640 / ratio)
    resized_height = int(512 / ratio)
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_left = (224 - resized_width) // 2
    pad_top = (224 - resized_height) // 2
    output = np.zeros((224, 224, 3), dtype=frame.dtype)
    output[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-geometry",
        choices=("center_square", "resize_with_pad"),
        default="resize_with_pad",
    )
    args = parser.parse_args()

    recording_dir = args.recording_dir.resolve()
    pose_data = json.loads(args.pose_json.read_text(encoding="utf-8"))
    if not pose_data.get("success") or len(pose_data.get("poses", [])) != 5:
        raise RuntimeError("pose sequence is not a successful five-pose record")
    timestamp_rows = [
        json.loads(line)
        for line in (recording_dir / "head_main_timestamps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not timestamp_rows:
        raise RuntimeError("camera timestamp log is empty")
    frame_times = np.asarray([row["arrival_time_ns"] for row in timestamp_rows], dtype=np.int64)

    selected = []
    for pose in pose_data["poses"]:
        target_ns = int(pose["capture_time_ns"])
        frame_position = int(np.argmin(np.abs(frame_times - target_ns)))
        row = timestamp_rows[frame_position]
        skew_ms = abs(int(row["arrival_time_ns"]) - target_ns) / 1e6
        if skew_ms > 250.0:
            raise RuntimeError(f"pose {pose['pose_index']} nearest camera frame is {skew_ms:.1f}ms away")
        selected.append(
            {
                "pose": pose,
                "source_video_frame_index": int(row["video_frame_index"]),
                "camera_pose_skew_ms": skew_ms,
            }
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    (output_dir / "meta/episodes/chunk-000").mkdir(parents=True)
    (output_dir / "data/chunk-000").mkdir(parents=True)
    video_dir = output_dir / "videos/observation.images.head_main/chunk-000"
    video_dir.mkdir(parents=True)
    preview_dir = output_dir / "live_capture_preview"
    preview_dir.mkdir(parents=True)

    source_video = cv2.VideoCapture(str(recording_dir / "head_main_640x512.mp4"))
    if not source_video.isOpened():
        raise RuntimeError("failed to open recorded head video")
    output_video_path = video_dir / "file-000.mp4"
    writer = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (224, 224), True
    )
    if not writer.isOpened():
        raise RuntimeError("failed to create calibration video")

    try:
        for index, item in enumerate(selected):
            source_video.set(cv2.CAP_PROP_POS_FRAMES, item["source_video_frame_index"])
            ok, frame = source_video.read()
            if not ok or frame is None or frame.shape[:2] != (512, 640):
                raise RuntimeError(f"failed to read source frame for pose {index}")
            processed = process_head_frame(frame, args.image_geometry)
            writer.write(processed)
            cv2.imwrite(str(preview_dir / f"pose_{index}.png"), processed)
    finally:
        source_video.release()
        writer.release()

    joint_rows = []
    for index, item in enumerate(selected):
        q14 = np.asarray(item["pose"]["q14_rad"], dtype=np.float32)
        if q14.shape != (14,) or not np.isfinite(q14).all():
            raise ValueError(f"pose {index} has invalid q14")
        joint16 = np.zeros(16, dtype=np.float32)
        joint16[:7] = q14[:7]
        joint16[8:15] = q14[7:]
        joint_rows.append(
            {
                "observation.joint_position": joint16,
                "timestamp": np.float32(index / 5.0),
                "episode_index": 0,
                "frame_index": index,
            }
        )
    pd.DataFrame(joint_rows).to_parquet(output_dir / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(
        [{"episode_index": 0, "length": 5, "dataset_from_index": 0, "dataset_to_index": 5}]
    ).to_parquet(output_dir / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame([{"episode_index": 0, "source_episode": "live_calibration_20260811"}]).to_parquet(
        output_dir / "meta/source_index.parquet", index=False
    )
    (output_dir / "capture_alignment.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.pose_json, output_dir / "pose_sequence.json")
    print(json.dumps({"dataset_root": str(output_dir), "frames": 5, "alignment": selected}, indent=2))


if __name__ == "__main__":
    main()
