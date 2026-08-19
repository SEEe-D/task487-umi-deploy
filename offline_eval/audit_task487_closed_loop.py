#!/usr/bin/env python3
"""Read-only synthetic closed-loop audit for Task487 image geometry and RTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from openpi_client import websocket_client_policy

from task487_runtime.contract import (
    ACTION_HORIZON,
    PolicyRequest,
    TASK_PROMPTS,
    body_actions_to_robot_targets,
    build_policy_request,
)


def _rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _fake_env_obs(
    images: dict[str, np.ndarray],
    previous: np.ndarray,
    current: np.ndarray,
    previous_grippers: np.ndarray,
    current_grippers: np.ndarray,
    timestamp: float,
) -> dict:
    result = {
        "timestamp": np.asarray([timestamp - 0.04, timestamp], dtype=np.float64),
        "camera4_rgb": np.stack([images["cam_head"]] * 2),
        "camera2_rgb": np.stack([images["cam_left_top"]] * 2),
        "camera0_rgb": np.stack([images["cam_right_top"]] * 2),
    }
    for arm in (0, 1):
        result[f"robot{arm}_eef_pos"] = np.stack(
            [previous[arm, :3], current[arm, :3]]
        )
        result[f"robot{arm}_eef_rot_axis_angle"] = np.stack(
            [previous[arm, 3:], current[arm, 3:]]
        )
        result[f"robot{arm}_gripper_angle"] = np.asarray(
            [[previous_grippers[arm]], [current_grippers[arm]]], dtype=np.float64
        )
    return result


def _request(
    native_images: dict[str, np.ndarray],
    image_mode: str,
    previous: np.ndarray,
    current: np.ndarray,
    previous_grippers: np.ndarray,
    current_grippers: np.ndarray,
    timestamp: float,
    prefix: np.ndarray | None,
) -> PolicyRequest:
    square_images = {}
    for key, value in native_images.items():
        height, width = value.shape[:2]
        size = min(height, width)
        y0 = (height - size) // 2
        x0 = (width - size) // 2
        square_images[key] = cv2.resize(
            value[y0 : y0 + size, x0 : x0 + size],
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )
    env_obs = _fake_env_obs(
        square_images,
        previous,
        current,
        previous_grippers,
        current_grippers,
        timestamp,
    )
    request = build_policy_request(
        env_obs,
        TASK_PROMPTS["vegetable"],
        rtc_prefix_targets=prefix,
    )
    if image_mode == "native_padded":
        policy_obs = dict(request.observation)
        policy_obs.update(native_images)
        request = PolicyRequest(policy_obs, request.observation_time, request.tcp_bases)
    return request


def _rollout(policy, capture: Path, image_mode: str, chunks: int, execute_steps: int) -> dict:
    records = json.loads((capture / "observations.json").read_text())
    base = np.asarray(records[0]["tcp_bases"], dtype=np.float64)
    images = {
        "cam_head": _rgb(capture / "cam_head_right" / "0000.jpg"),
        "cam_left_top": _rgb(capture / "cam_hand_l_top" / "0000.jpg"),
        "cam_right_top": _rgb(capture / "cam_hand_r_top" / "0000.jpg"),
    }
    initial = base.copy()
    previous = base.copy()
    current = base.copy()
    previous_grippers = np.full(2, 35.0)
    current_grippers = np.full(2, 35.0)
    prefix = None
    floor = np.maximum(np.full(2, 0.60), initial[:, 2] - 0.12)
    rows = []
    first_floor_violation = None

    for chunk_index in range(chunks):
        request = _request(
            images,
            image_mode,
            previous,
            current,
            previous_grippers,
            current_grippers,
            timestamp=1000.0 + chunk_index,
            prefix=prefix,
        )
        result = policy.infer(request.observation)
        actions = np.asarray(result["actions"], dtype=np.float64)
        targets = body_actions_to_robot_targets(actions, request.tcp_bases)
        positions = np.stack((targets[:, 0:3], targets[:, 7:10]), axis=1)
        violations = np.argwhere(positions[:, :, 2] < floor[None, :])
        if len(violations) and first_floor_violation is None:
            first_floor_violation = {
                "chunk": chunk_index,
                "action": int(violations[0, 0]),
                "arm": "right" if int(violations[0, 1]) == 0 else "left",
                "z_m": float(positions[tuple(violations[0])][2]),
            }
        rows.append(
            {
                "chunk": chunk_index,
                "current_z_m": current[:, 2].tolist(),
                "delta_action4_mm": np.concatenate(
                    (targets[4, 0:3] - current[0, :3], targets[4, 7:10] - current[1, :3])
                ).__mul__(1000.0).tolist(),
                "delta_action14_mm": np.concatenate(
                    (targets[14, 0:3] - current[0, :3], targets[14, 7:10] - current[1, :3])
                ).__mul__(1000.0).tolist(),
                "delta_action19_mm": np.concatenate(
                    (targets[19, 0:3] - current[0, :3], targets[19, 7:10] - current[1, :3])
                ).__mul__(1000.0).tolist(),
                "chunk_min_z_m": positions[:, :, 2].min(axis=0).tolist(),
                "rtc_prefix_len": 0 if prefix is None else len(prefix),
            }
        )

        previous = current.copy()
        previous_grippers = current_grippers.copy()
        executed_index = execute_steps - 1
        current = np.stack((targets[executed_index, 0:6], targets[executed_index, 7:13]))
        current_grippers = np.asarray(
            [targets[executed_index, 6], targets[executed_index, 13]]
        )
        # The proposed 5/10 scheduler has five already-committed targets left
        # when replanning. The current 15/20 scheduler has the same prefix
        # length, so both modes exercise an identical RTC hard prefix.
        prefix = targets[execute_steps : execute_steps + 5].copy()

    return {
        "image_mode": image_mode,
        "execute_steps_per_replan": execute_steps,
        "input_shape": list(next(iter(images.values())).shape) if image_mode == "native_padded" else [224, 224, 3],
        "workspace_floor_m": floor.tolist(),
        "first_floor_violation": first_floor_violation,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute-steps", type=int, choices=(5, 15), default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    result = {
        "read_only": True,
        "robot_commands_published": False,
        "server_metadata": policy.get_server_metadata(),
        "capture": str(args.capture),
        "rollouts": [],
    }
    for repeat in range(args.repeats):
        for mode in ("native_padded", "training_square"):
            rollout = _rollout(policy, args.capture, mode, args.chunks, args.execute_steps)
            rollout["repeat"] = repeat
            result["rollouts"].append(rollout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
