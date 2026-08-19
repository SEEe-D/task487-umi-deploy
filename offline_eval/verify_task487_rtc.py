#!/usr/bin/env python3
"""Read-only checkpoint verification for Task487 action-prefill RTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation

from task487_runtime.contract import (
    body_actions_to_robot_targets,
    robot_targets_to_model_absolute_actions,
)


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _observation(capture: Path, records: list[dict], index: int) -> dict:
    record = records[index]
    frame = f"{index:04d}.jpg"
    return {
        "cam_head": _image(capture / "cam_head_right" / frame),
        "cam_left_top": _image(capture / "cam_hand_l_top" / frame),
        "cam_right_top": _image(capture / "cam_hand_r_top" / frame),
        "pre_state": np.asarray(record["pre_state"], dtype=np.float32),
        "state": np.asarray(record["state"], dtype=np.float32),
        "prompt": record["prompt"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rtc-index", type=int)
    parser.add_argument("--prefix-offset", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=10)
    args = parser.parse_args()

    observations = json.loads((args.capture / "observations.json").read_text())
    if args.prefix_offset < 0:
        raise ValueError("--prefix-offset must be non-negative")
    rtc_index = args.index if args.rtc_index is None else args.rtc_index
    record = observations[args.index]
    observation = _observation(args.capture, observations, args.index)
    rtc_base_observation = _observation(args.capture, observations, rtc_index)
    tcp_bases = np.asarray(record["tcp_bases"], dtype=np.float64)
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata()
    if not metadata.get("rtc_enabled"):
        raise RuntimeError(f"Server does not advertise RTC: {metadata}")
    action_horizon = int(metadata.get("action_horizon", 0))
    if action_horizon <= 0:
        raise RuntimeError(f"Server does not advertise a valid action_horizon: {metadata}")

    baseline_result = policy.infer(observation)
    baseline_actions = np.asarray(baseline_result["actions"], dtype=np.float64)
    baseline_targets = body_actions_to_robot_targets(
        baseline_actions,
        tcp_bases,
        action_horizon=action_horizon,
    )
    prefix_len = min(args.prefix_steps, action_horizon - args.prefix_offset)
    if prefix_len <= 0:
        raise ValueError("RTC prefix is empty after applying --prefix-offset")
    prefix_targets = baseline_targets[args.prefix_offset : args.prefix_offset + prefix_len]
    absolute_prefill = robot_targets_to_model_absolute_actions(prefix_targets)
    absolute_prefill = np.concatenate(
        (
            absolute_prefill,
            np.repeat(absolute_prefill[-1:], action_horizon - prefix_len, axis=0),
        ),
        axis=0,
    )
    rtc_observation = {
        **rtc_base_observation,
        "actions": absolute_prefill,
        "action_prefill_len": np.int32(prefix_len),
    }
    rtc_result = policy.infer(rtc_observation)
    rtc_actions = np.asarray(rtc_result["actions"], dtype=np.float64)
    rtc_targets = body_actions_to_robot_targets(
        rtc_actions,
        tcp_bases,
        action_horizon=action_horizon,
    )

    translation_errors = []
    rotation_errors = []
    gripper_errors = []
    seam_translation_steps = []
    seam_rotation_steps = []
    seam_gripper_steps = []
    for arm_offset in (0, 7):
        translation_errors.extend(
            np.linalg.norm(
                rtc_targets[:prefix_len, arm_offset : arm_offset + 3]
                - prefix_targets[:, arm_offset : arm_offset + 3],
                axis=1,
            ).tolist()
        )
        rotation_errors.extend(
            (
                Rotation.from_rotvec(rtc_targets[:prefix_len, arm_offset + 3 : arm_offset + 6]).inv()
                * Rotation.from_rotvec(
                    prefix_targets[:, arm_offset + 3 : arm_offset + 6]
                )
            ).magnitude().tolist()
        )
        gripper_errors.extend(
            np.abs(
                rtc_targets[:prefix_len, arm_offset + 6]
                - prefix_targets[:, arm_offset + 6]
            ).tolist()
        )
        if prefix_len < len(rtc_targets):
            seam_translation_steps.append(
                float(
                    np.linalg.norm(
                        rtc_targets[prefix_len, arm_offset : arm_offset + 3]
                        - rtc_targets[prefix_len - 1, arm_offset : arm_offset + 3]
                    )
                )
            )
            seam_rotation_steps.append(
                float(
                    (
                        Rotation.from_rotvec(
                            rtc_targets[prefix_len - 1, arm_offset + 3 : arm_offset + 6]
                        ).inv()
                        * Rotation.from_rotvec(
                            rtc_targets[prefix_len, arm_offset + 3 : arm_offset + 6]
                        )
                    ).magnitude()
                )
            )
            seam_gripper_steps.append(
                float(
                    abs(
                        rtc_targets[prefix_len, arm_offset + 6]
                        - rtc_targets[prefix_len - 1, arm_offset + 6]
                    )
                )
            )

    result = {
        "read_only": True,
        "robot_commands_published": False,
        "rtc_mode": metadata.get("rtc_mode"),
        "action_horizon": action_horizon,
        "baseline_index": args.index,
        "rtc_index": rtc_index,
        "prefix_offset": args.prefix_offset,
        "prefix_steps": prefix_len,
        "baseline_model_infer_ms": baseline_result.get("policy_timing", {}).get("infer_ms"),
        "rtc_model_infer_ms": rtc_result.get("policy_timing", {}).get("infer_ms"),
        "max_prefix_translation_error_m": max(translation_errors, default=0.0),
        "max_prefix_rotation_error_rad": max(rotation_errors, default=0.0),
        "max_prefix_gripper_error_deg": max(gripper_errors, default=0.0),
        "max_rtc_seam_translation_step_m": max(seam_translation_steps, default=0.0),
        "max_rtc_seam_rotation_step_rad": max(seam_rotation_steps, default=0.0),
        "max_rtc_seam_gripper_step_deg": max(seam_gripper_steps, default=0.0),
    }
    print(json.dumps(result, indent=2))
    if result["max_prefix_translation_error_m"] > 0.002:
        raise RuntimeError("RTC translation prefill was not preserved")
    if result["max_prefix_rotation_error_rad"] > 0.02:
        raise RuntimeError("RTC rotation prefill was not preserved")
    if result["max_prefix_gripper_error_deg"] > 0.5:
        raise RuntimeError("RTC gripper prefill was not preserved")
    if result["max_rtc_seam_translation_step_m"] > 0.030:
        raise RuntimeError("RTC generated suffix has an unsafe translation seam")
    if result["max_rtc_seam_rotation_step_rad"] > 0.12:
        raise RuntimeError("RTC generated suffix has an unsafe rotation seam")


if __name__ == "__main__":
    main()
