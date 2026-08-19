from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_JOINT_COLUMNS = [
    "Joint71",
    "Joint72",
    "Joint73",
    "Joint74",
    "Joint75",
    "Joint76",
    "Joint77",
    "Joint79",
    "Joint61",
    "Joint62",
    "Joint63",
    "Joint64",
    "Joint65",
    "Joint66",
    "Joint67",
    "Joint69",
]
REQUIRED_DATA_COLUMNS = {
    "observation.state",
    "observation.state_valid",
    "action",
    "action_valid",
    "timestamp",
    "frame_index",
    "episode_index",
    "observation.joint_position",
    "observation.joint_position_valid",
    "joint.t_wall",
    "joint.t_monotonic",
    "valid.frame",
}


def parse_rate(value: str) -> float:
    return float(Fraction(value)) if value else 0.0


def probe_video(ffprobe: str, path: Path) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(completed.stdout)["streams"][0]
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "fps": parse_rate(stream["avg_frame_rate"]),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_s": float(stream["duration"]) if stream.get("duration") else None,
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
    }


def decode_probe_frame(ffmpeg: str, path: Path, timestamp_s: float) -> dict:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{timestamp_s:.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "timestamp_s": timestamp_s,
        "success": completed.returncode == 0,
        "error": completed.stderr.strip() or None,
    }


def array_column(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def explicit_joint_mapping() -> dict:
    right = {
        f"Joint7{index}": f"Joint{index}_R" for index in range(1, 8)
    }
    left = {
        f"Joint6{index}": f"Joint{index}_L" for index in range(1, 8)
    }
    return {
        "status": "candidate_requires_robot_owner_confirmation",
        "rule": "Mapping is explicit; data order is never assumed to equal URDF order.",
        "data_to_urdf_revolute_joint": {**right, **left},
        "unmapped_data_joints": {
            "Joint79": "right gripper; no matching movable joint in this URDF",
            "Joint69": "left gripper; no matching movable joint in this URDF",
        },
        "evidence": (
            "Data metadata identifies Joint79/Joint69 as right/left grippers. "
            "The URDF contains Joint1_R..Joint7_R and Joint1_L..Joint7_L only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    pyarrow_major = int(pa.__version__.split(".", 1)[0])
    if pyarrow_major < 24:
        raise RuntimeError(
            f"PyArrow >= 24 is required for these Parquet files; found {pa.__version__}."
        )
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise RuntimeError("ffprobe and ffmpeg must be available on PATH.")

    skill_root = args.dataset_root.resolve()
    dataset_dirs = sorted(path for path in skill_root.glob("*/*") if path.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError(f"No date/task datasets under {skill_root}")

    dataset_records = []
    episode_population = []
    global_joint_min = np.full(16, np.inf, dtype=np.float64)
    global_joint_max = np.full(16, -np.inf, dtype=np.float64)
    global_eef_min = np.full(20, np.inf, dtype=np.float64)
    global_eef_max = np.full(20, -np.inf, dtype=np.float64)
    total_frames = 0
    total_episodes = 0
    total_missing_frames = 0
    total_duplicate_frames = 0
    total_invalid_frames = 0
    video_failures = []

    for dataset_dir in dataset_dirs:
        date, task = dataset_dir.parent.name, dataset_dir.name
        info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
        source_index = pd.read_parquet(dataset_dir / "meta/source_index.parquet").sort_values(
            "episode_index"
        )
        episodes = pd.read_parquet(
            dataset_dir / "meta/episodes/chunk-000/file-000.parquet"
        ).sort_values("episode_index")
        table = pq.read_table(dataset_dir / "data/chunk-000/file-000.parquet")
        missing_columns = REQUIRED_DATA_COLUMNS - set(table.column_names)
        if missing_columns:
            raise ValueError(f"{date}/{task}: missing columns {sorted(missing_columns)}")

        joint_positions = array_column(table, "observation.joint_position", np.float32)
        joint_valid = array_column(table, "observation.joint_position_valid", bool)
        eef_states = array_column(table, "observation.state", np.float32)
        eef_valid = array_column(table, "observation.state_valid", bool)
        frame_indices = table["frame_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy().astype(np.float64)
        episode_indices = table["episode_index"].to_numpy()
        valid_frames = table["valid.frame"].to_numpy()
        global_joint_min = np.minimum(global_joint_min, np.nanmin(joint_positions, axis=0))
        global_joint_max = np.maximum(global_joint_max, np.nanmax(joint_positions, axis=0))
        global_eef_min = np.minimum(global_eef_min, np.nanmin(eef_states, axis=0))
        global_eef_max = np.maximum(global_eef_max, np.nanmax(eef_states, axis=0))

        dataset_missing_frames = 0
        dataset_duplicate_frames = 0
        episode_checks = []
        for source_row in source_index.itertuples(index=False):
            episode_index = int(source_row.episode_index)
            start, end = int(source_row.dataset_from_index), int(source_row.dataset_to_index)
            frame_count = int(source_row.source_num_frames)
            local_frame_index = frame_indices[start:end]
            unique_count = len(np.unique(local_frame_index))
            missing_count = frame_count - unique_count
            duplicate_count = len(local_frame_index) - unique_count
            expected_frame_index = np.arange(frame_count, dtype=np.int64)
            expected_timestamp = expected_frame_index / float(source_row.output_fps)
            frame_index_contiguous = np.array_equal(local_frame_index, expected_frame_index)
            timestamp_max_abs_error = float(
                np.max(np.abs(timestamps[start:end] - expected_timestamp))
            )
            episode_index_consistent = bool(
                np.all(episode_indices[start:end] == episode_index)
            )
            joint_csv = dataset_dir / "joint_states" / f"{source_row.source_episode}.csv"
            csv_frame_count = sum(1 for _ in joint_csv.open("r", encoding="utf-8")) - 1
            counts_consistent = bool(
                end - start == frame_count == csv_frame_count
            )
            check = {
                "episode_index": episode_index,
                "source_episode": str(source_row.source_episode),
                "dataset_from_index": start,
                "dataset_to_index": end,
                "frame_count": frame_count,
                "joint_csv_frame_count": csv_frame_count,
                "counts_consistent": counts_consistent,
                "frame_index_contiguous": frame_index_contiguous,
                "episode_index_consistent": episode_index_consistent,
                "timestamp_max_abs_error_s": timestamp_max_abs_error,
                "missing_frame_count": missing_count,
                "duplicate_frame_count": duplicate_count,
            }
            episode_checks.append(check)
            episode_population.append(
                {
                    "dataset_dir": dataset_dir,
                    "date": date,
                    "task": task,
                    "source_row": source_row,
                    "episode_row": episodes.iloc[episode_index],
                    "check": check,
                    "joint_positions": joint_positions[start:end],
                    "joint_valid": joint_valid[start:end],
                    "eef_states": eef_states[start:end],
                    "eef_valid": eef_valid[start:end],
                    "timestamps": timestamps[start:end],
                    "wall": table["joint.t_wall"].slice(start, end - start).to_numpy(),
                    "monotonic": table["joint.t_monotonic"].slice(start, end - start).to_numpy(),
                }
            )
            dataset_missing_frames += missing_count
            dataset_duplicate_frames += duplicate_count
            if not (
                counts_consistent
                and frame_index_contiguous
                and episode_index_consistent
                and timestamp_max_abs_error <= 1e-5
            ):
                raise ValueError(f"Failed episode alignment: {date}/{task}/{check}")

        video_records = []
        videos_root = dataset_dir / "videos"
        for stream_dir in sorted(path for path in videos_root.glob("*") if path.is_dir()):
            video_path = stream_dir / "chunk-000/file-000.mp4"
            record = probe_video(ffprobe, video_path)
            record["stream_key"] = stream_dir.name
            record["matches_dataset_frame_count"] = record["nb_frames"] == table.num_rows
            record["matches_declared_fps"] = abs(record["fps"] - float(info["fps"])) < 1e-9
            record["matches_declared_image_size"] = [
                record["height"],
                record["width"],
            ] == info["features"][stream_dir.name]["shape"][1:]
            if not all(
                [
                    record["matches_dataset_frame_count"],
                    record["matches_declared_fps"],
                    record["matches_declared_image_size"],
                ]
            ):
                video_failures.append(record)
            video_records.append(record)

        invalid_frame_count = int(np.count_nonzero(~valid_frames))
        dataset_records.append(
            {
                "date": date,
                "task": task,
                "dataset_path": str(dataset_dir.resolve()),
                "episode_count": len(source_index),
                "frame_count": table.num_rows,
                "declared_fps": float(info["fps"]),
                "data_columns": table.column_names,
                "joint_state_shape": list(joint_positions.shape),
                "eef_state_shape": list(eef_states.shape),
                "all_joint_values_finite": bool(np.isfinite(joint_positions).all()),
                "all_joint_values_valid": bool(joint_valid.all()),
                "all_eef_values_finite": bool(np.isfinite(eef_states).all()),
                "all_eef_values_valid": bool(eef_valid.all()),
                "invalid_frame_count": invalid_frame_count,
                "missing_frame_count": dataset_missing_frames,
                "duplicate_frame_count": dataset_duplicate_frames,
                "video_streams": video_records,
                "episode_frame_count_min": int(source_index["source_num_frames"].min()),
                "episode_frame_count_max": int(source_index["source_num_frames"].max()),
                "episode_frame_count_mean": float(source_index["source_num_frames"].mean()),
            }
        )
        total_frames += table.num_rows
        total_episodes += len(source_index)
        total_missing_frames += dataset_missing_frames
        total_duplicate_frames += dataset_duplicate_frames
        total_invalid_frames += invalid_frame_count

    rng = random.Random(args.seed)
    by_dataset: dict[tuple[str, str], list[dict]] = {}
    for episode in episode_population:
        by_dataset.setdefault((episode["date"], episode["task"]), []).append(episode)
    sampled = [rng.choice(entries) for entries in by_dataset.values()]
    sampled_ids = {id(episode) for episode in sampled}
    remaining = [episode for episode in episode_population if id(episode) not in sampled_ids]
    target_count = min(max(args.sample_count, len(sampled)), len(episode_population))
    sampled.extend(rng.sample(remaining, target_count - len(sampled)))
    sampled.sort(key=lambda item: (item["date"], item["task"], item["check"]["episode_index"]))

    sample_records = []
    for episode in sampled:
        dataset_dir = episode["dataset_dir"]
        source_row = episode["source_row"]
        episode_row = episode["episode_row"]
        source_episode = str(source_row.source_episode)
        joint_csv_path = dataset_dir / "joint_states" / f"{source_episode}.csv"
        raw_joint = pd.read_csv(joint_csv_path)
        raw_joint_fields = [column for column in raw_joint.columns if column.startswith("Joint")]
        head_video = (
            dataset_dir
            / "videos/observation.images.head_main/chunk-000/file-000.mp4"
        )
        video_start = float(
            episode_row["videos/observation.images.head_main/from_timestamp"]
        )
        video_midpoint = video_start + float(source_row.source_num_frames) / (
            2.0 * float(source_row.output_fps)
        )
        decode = decode_probe_frame(ffmpeg, head_video, video_midpoint)
        if not decode["success"]:
            raise RuntimeError(
                f"Could not decode sampled frame {source_episode}: {decode['error']}"
            )
        sample_records.append(
            {
                "date": episode["date"],
                "task": episode["task"],
                "episode_index": int(source_row.episode_index),
                "source_episode": source_episode,
                "frame_count": int(source_row.source_num_frames),
                "fps": float(source_row.output_fps),
                "image_paths": [
                    str(path.resolve())
                    for path in sorted((dataset_dir / "videos").glob("*/chunk-000/file-000.mp4"))
                ],
                "image_size_hw": [224, 224],
                "decoded_head_main_frame": decode,
                "joint_csv_path": str(joint_csv_path.resolve()),
                "raw_joint_angle_fields": raw_joint_fields,
                "raw_recorded_joint_count": len(raw_joint_fields),
                "model_joint_fields": EXPECTED_JOINT_COLUMNS,
                "right_joint_count_including_gripper": 8,
                "left_joint_count_including_gripper": 8,
                "joint_unit": "radian",
                "joint_min_rad": np.min(episode["joint_positions"], axis=0).tolist(),
                "joint_max_rad": np.max(episode["joint_positions"], axis=0).tolist(),
                "eef_state_fields": [
                    "right: xyz_m + rotation_6d + gripper_rad",
                    "left: xyz_m + rotation_6d + gripper_rad",
                ],
                "eef_state_shape": list(episode["eef_states"].shape),
                "eef_state_min": np.min(episode["eef_states"], axis=0).tolist(),
                "eef_state_max": np.max(episode["eef_states"], axis=0).tolist(),
                "timestamp_fields": [
                    "timestamp",
                    "joint.t_wall",
                    "joint.t_monotonic",
                ],
                "timestamp_start_s": float(episode["timestamps"][0]),
                "timestamp_end_s": float(episode["timestamps"][-1]),
                "wall_time_start_s": float(episode["wall"][0]),
                "wall_time_end_s": float(episode["wall"][-1]),
                "alignment_check": episode["check"],
            }
        )

    report = {
        "status": "pass" if not video_failures else "warning",
        "dataset_root": str(skill_root),
        "inspection_environment": {
            "pyarrow_version": pa.__version__,
            "ffprobe": ffprobe,
            "ffmpeg": ffmpeg,
            "sample_seed": args.seed,
            "sample_count": len(sample_records),
        },
        "summary": {
            "dataset_count": len(dataset_records),
            "episode_count": total_episodes,
            "frame_count": total_frames,
            "video_file_count": sum(len(item["video_streams"]) for item in dataset_records),
            "missing_frame_count": total_missing_frames,
            "duplicate_frame_count": total_duplicate_frames,
            "invalid_frame_count": total_invalid_frames,
            "video_validation_failure_count": len(video_failures),
            "all_image_sizes_hw": sorted(
                {
                    (video["height"], video["width"])
                    for dataset in dataset_records
                    for video in dataset["video_streams"]
                }
            ),
            "all_video_fps": sorted(
                {
                    video["fps"]
                    for dataset in dataset_records
                    for video in dataset["video_streams"]
                }
            ),
        },
        "joint_data": {
            "model_joint_fields": EXPECTED_JOINT_COLUMNS,
            "model_joint_count": 16,
            "right_count_including_gripper": 8,
            "left_count_including_gripper": 8,
            "unit": "radian",
            "global_min_rad": global_joint_min.tolist(),
            "global_max_rad": global_joint_max.tolist(),
            "explicit_urdf_mapping": explicit_joint_mapping(),
        },
        "eef_data": {
            "shape_per_frame": [20],
            "layout": "right_10d then left_10d",
            "per_side": "xyz_m + rotation_6d_dimensionless + gripper_rad",
            "global_min": global_eef_min.tolist(),
            "global_max": global_eef_max.tolist(),
        },
        "datasets": dataset_records,
        "sampled_episodes": sample_records,
        "video_failures": video_failures,
        "limitations": [
            "This stage validates stored synchronization and decodability only.",
            "It does not validate camera intrinsics or T_camera_left_base/T_camera_right_base.",
            "It does not validate the proposed data-to-URDF joint mapping on rendered images.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
