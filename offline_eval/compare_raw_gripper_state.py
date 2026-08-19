#!/usr/bin/env python3
"""Read-only A/B test for Task487 gripper-state distribution shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from openpi_client import websocket_client_policy

from task487_runtime.contract import body_actions_to_robot_targets


def _rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _observation(capture: Path, record: dict, index: int, grippers_deg: tuple[float, float]) -> dict:
    frame = f"{index:04d}.jpg"
    pre_state = np.asarray(record["pre_state"], dtype=np.float32).copy()
    state = np.asarray(record["state"], dtype=np.float32).copy()
    for offset, value in zip((9, 19), grippers_deg, strict=True):
        pre_state[offset] = np.deg2rad(value)
        state[offset] = np.deg2rad(value)
    return {
        "cam_head": _rgb(capture / "cam_head_right" / frame),
        "cam_left_top": _rgb(capture / "cam_hand_l_top" / frame),
        "cam_right_top": _rgb(capture / "cam_hand_r_top" / frame),
        "pre_state": pre_state,
        "state": state,
        "prompt": record["prompt"],
    }


def _condition_summary(samples: np.ndarray, input_grippers_deg: tuple[float, float]) -> dict:
    # samples: repeats x horizon x 14 robot targets
    result: dict[str, object] = {
        "input_grippers_deg": list(input_grippers_deg),
        "repeats": int(samples.shape[0]),
        "horizon": int(samples.shape[1]),
    }
    for side, offset in (("right", 0), ("left", 7)):
        grip = samples[:, :, offset + 6]
        xyz = samples[:, :, offset : offset + 3]
        mean_grip = grip.mean(axis=0)
        result[side] = {
            "gripper_mean_deg_by_step": mean_grip.tolist(),
            "gripper_std_deg_by_step": grip.std(axis=0).tolist(),
            "first_step_mean_deg": float(mean_grip[0]),
            "last_step_mean_deg": float(mean_grip[-1]),
            "minimum_mean_deg": float(mean_grip.min()),
            "maximum_mean_deg": float(mean_grip.max()),
            "mean_path_range_deg": float(mean_grip.max() - mean_grip.min()),
            "mean_xyz_by_step_m": xyz.mean(axis=0).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads((args.capture / "observations.json").read_text())
    record = records[args.index]
    tcp_bases = np.asarray(record["tcp_bases"], dtype=np.float64)
    conditions = {
        "home_35deg": (35.0, 35.0),
        # q99 from the exact norm_stats shipped with raw_seed42/29999.
        "raw_q99": (np.rad2deg(0.38035363673679534), np.rad2deg(0.4751590864706784)),
    }

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata()
    action_horizon = int(metadata["action_horizon"])
    samples = {name: [] for name in conditions}
    timing_ms = {name: [] for name in conditions}

    # Alternate conditions so server-side sampling drift cannot favor one arm of the A/B.
    for _ in range(args.repeats):
        for name, grippers in conditions.items():
            result = policy.infer(_observation(args.capture, record, args.index, grippers))
            actions = np.asarray(result["actions"], dtype=np.float64)
            targets = body_actions_to_robot_targets(
                actions,
                tcp_bases,
                action_horizon=action_horizon,
            )
            samples[name].append(targets)
            timing = result.get("policy_timing", {}).get("infer_ms")
            if timing is not None:
                timing_ms[name].append(float(timing))

    stacked = {name: np.stack(values) for name, values in samples.items()}
    summaries = {
        name: _condition_summary(stacked[name], conditions[name])
        for name in conditions
    }
    for name in conditions:
        summaries[name]["infer_ms_mean"] = (
            float(np.mean(timing_ms[name])) if timing_ms[name] else None
        )

    home = stacked["home_35deg"]
    in_dist = stacked["raw_q99"]
    comparison = {}
    for side, offset in (("right", 0), ("left", 7)):
        home_grip = home[:, :, offset + 6].mean(axis=0)
        in_dist_grip = in_dist[:, :, offset + 6].mean(axis=0)
        home_xyz = home[:, :, offset : offset + 3].mean(axis=0)
        in_dist_xyz = in_dist[:, :, offset : offset + 3].mean(axis=0)
        comparison[side] = {
            "q99_minus_home_gripper_deg_by_step": (in_dist_grip - home_grip).tolist(),
            "max_abs_gripper_effect_deg": float(np.max(np.abs(in_dist_grip - home_grip))),
            "max_abs_xyz_effect_mm": float(
                np.max(np.linalg.norm(in_dist_xyz - home_xyz, axis=1)) * 1000.0
            ),
        }

    report = {
        "read_only": True,
        "robot_commands_published": False,
        "capture": str(args.capture),
        "capture_index": args.index,
        "server_metadata": metadata,
        "conditions": summaries,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
