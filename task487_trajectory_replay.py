#!/usr/bin/env python3
"""Replay a recorded Task487 motion to validate Marvin tracking.

This path does not start a policy server, decode cameras, or run inference.
Recorded motion is expressed relative to its first pose and anchored at the
live TCP poses, so dataset/world-frame offsets are never sent to the robot.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

from task487_runtime.contract import R_R2M, body_actions_to_robot_targets, rot6d_to_matrix
from umi.real_world.umi_env import UmiEnv


CLIENT_LOCK_PATH = "/tmp/task487_client.lock"
GRIPPER_SAFE_OPEN_RAD = (-0.59341195, -0.41949338)
GRIPPER_SAFE_OPEN_DEG = tuple(
    float(np.rad2deg(abs(open_rad))) for open_rad in GRIPPER_SAFE_OPEN_RAD
)


class TerminalKeys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd) if sys.stdin.isatty() else None
        if self.settings is not None:
            tty.setcbreak(self.fd)
        return self

    def poll(self) -> list[str]:
        if self.settings is None:
            return []
        result = []
        while select.select([self.fd], [], [], 0.0)[0]:
            result.append(os.read(self.fd, 1).decode(errors="ignore").lower())
        return result

    def __exit__(self, *_args):
        if self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


def acquire_lock():
    handle = open(CLIENT_LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another client owns {CLIENT_LOCK_PATH}") from None
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:2].reshape(6)


def recorded_body_actions(actions: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float64).copy()
    initial_state = np.asarray(initial_state, dtype=np.float64)
    for offset in (0, 10):
        initial_rotation = rot6d_to_matrix(initial_state[offset + 3 : offset + 9])
        for action in result:
            action[offset : offset + 3] = initial_rotation.T @ (
                action[offset : offset + 3] - initial_state[offset : offset + 3]
            )
            target_rotation = rot6d_to_matrix(action[offset + 3 : offset + 9])
            action[offset + 3 : offset + 9] = matrix_to_rot6d(
                initial_rotation.T @ target_rotation
            )
    return result


def load_segment(dataset: Path, file_index: int, episode: int, start: int, steps: int) -> np.ndarray:
    path = dataset / "data/chunk-000" / f"file-{file_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"])
    episode_indices = table.column("episode_index").to_numpy()
    rows = np.flatnonzero(episode_indices == episode)
    if not len(rows):
        raise ValueError(f"Episode {episode} is not present in {path}")
    if steps == 0:
        steps = len(rows) - start
    if start < 0 or steps <= 0 or start + steps > len(rows):
        raise ValueError(f"Requested [{start}, {start + steps}) outside episode length {len(rows)}")
    states = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
    actions = np.stack(table.column("action").to_pylist()).astype(np.float64)
    first = int(rows[start])
    return recorded_body_actions(actions[first : first + steps], states[first])


def model_absolute_to_robot(values: np.ndarray) -> np.ndarray:
    """Decode dataset pose9 blocks in the shared Task487 world frame."""
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    values = values[None] if squeeze else values
    targets = np.empty((len(values), 14), dtype=np.float64)
    for model_offset, robot_offset in ((0, 0), (10, 7)):
        targets[:, robot_offset : robot_offset + 3] = values[:, model_offset : model_offset + 3]
        model_rotations = np.stack(
            [rot6d_to_matrix(value) for value in values[:, model_offset + 3 : model_offset + 9]]
        )
        targets[:, robot_offset + 3 : robot_offset + 6] = Rotation.from_matrix(
            model_rotations @ R_R2M
        ).as_rotvec()
        targets[:, robot_offset + 6] = np.rad2deg(values[:, model_offset + 9])
    return targets[0] if squeeze else targets


def load_absolute_segment(
    dataset: Path, file_index: int, episode: int, start: int, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    path = dataset / "data/chunk-000" / f"file-{file_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
    rows = np.flatnonzero(table.column("episode_index").to_numpy() == episode)
    if not len(rows):
        raise ValueError(f"Episode {episode} is not present in {path}")
    if steps == 0:
        steps = len(rows) - start
    if start < 0 or steps <= 0 or start + steps > len(rows):
        raise ValueError(f"Requested [{start}, {start + steps}) outside episode length {len(rows)}")
    states = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
    actions = np.stack(table.column("action").to_pylist()).astype(np.float64)
    first = int(rows[start])
    return model_absolute_to_robot(states[first]), model_absolute_to_robot(actions[first : first + steps])


def live_targets(env: UmiEnv) -> tuple[np.ndarray, np.ndarray]:
    target = np.empty(14, dtype=np.float64)
    bases = np.empty((2, 6), dtype=np.float64)
    robot_states = env.get_robot_state()
    gripper_states = env.get_gripper_state()
    for arm in (0, 1):
        pose = np.asarray(robot_states[arm]["ActualTCPPose"], dtype=np.float64)
        gripper = float(np.asarray(gripper_states[arm]["gripper_position"]).reshape(-1)[-1])
        bases[arm] = pose
        target[7 * arm : 7 * arm + 6] = pose
        target[7 * arm + 6] = gripper
    return target, bases


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())


def validate_targets(
    targets: np.ndarray,
    anchor: np.ndarray,
    frequency: float,
    max_excursion_m: float,
    max_translation_speed_m_s: float = 0.05,
    max_rotation_speed_rad_s: float = 0.20,
    min_z_m: float = 0.25,
    max_z_m: float = 0.90,
) -> dict:
    sequence = np.concatenate((anchor[None], targets), axis=0)
    report = {}
    for name, offset in (("right", 0), ("left", 7)):
        position_steps = np.linalg.norm(np.diff(sequence[:, offset : offset + 3], axis=0), axis=1)
        rotation_steps = np.asarray(
            [rotation_distance(a, b) for a, b in zip(sequence[:-1, offset + 3 : offset + 6], sequence[1:, offset + 3 : offset + 6])]
        )
        excursion = np.linalg.norm(targets[:, offset : offset + 3] - anchor[offset : offset + 3], axis=1)
        report[name] = {
            "max_step_mm": float(position_steps.max() * 1000.0),
            "max_rotation_step_deg": float(np.rad2deg(rotation_steps.max())),
            "max_translation_speed_m_s": float(position_steps.max() * frequency),
            "max_rotation_speed_rad_s": float(rotation_steps.max() * frequency),
            "max_excursion_mm": float(excursion.max() * 1000.0),
            "z_range_m": [
                float(targets[:, offset + 2].min()),
                float(targets[:, offset + 2].max()),
            ],
        }
        if position_steps.max() * frequency > max_translation_speed_m_s:
            raise RuntimeError(
                f"{name} trajectory exceeds {max_translation_speed_m_s:.2f} m/s at {frequency:g} Hz"
            )
        if rotation_steps.max() * frequency > max_rotation_speed_rad_s:
            raise RuntimeError(
                f"{name} trajectory exceeds {max_rotation_speed_rad_s:.2f} rad/s at {frequency:g} Hz"
            )
        if excursion.max() > max_excursion_m:
            raise RuntimeError(f"{name} trajectory excursion exceeds {max_excursion_m:.3f} m")
        if targets[:, offset + 2].min() < min_z_m or targets[:, offset + 2].max() > max_z_m:
            raise RuntimeError(
                f"{name} trajectory z range is outside [{min_z_m:.3f}, {max_z_m:.3f}] m"
            )
    return report


def workspace_fit_bases(
    body_actions: np.ndarray,
    live_bases: np.ndarray,
    min_z_m: float = 0.27,
    max_z_m: float = 0.88,
) -> tuple[np.ndarray, dict]:
    """Find the smallest per-arm vertical translation that fits the full path."""
    desired = np.asarray(live_bases, dtype=np.float64).copy()
    provisional = body_actions_to_robot_targets(body_actions, desired)
    offsets = {}
    for name, arm_offset, base_index in (("right", 0, 0), ("left", 7, 1)):
        path_z = provisional[:, arm_offset + 2]
        lower = min_z_m - float(path_z.min())
        upper = max_z_m - float(path_z.max())
        if lower > upper:
            raise RuntimeError(
                f"{name} full path height span cannot fit [{min_z_m:.3f}, {max_z_m:.3f}] m"
            )
        shift = min(max(0.0, lower), upper)
        desired[base_index, 2] += shift
        offsets[name] = float(shift)
    return desired, offsets


def prepare_start(
    env: UmiEnv,
    live: np.ndarray,
    desired: np.ndarray,
) -> None:
    """Move to the recorded replay anchor at conservative Cartesian speed."""
    desired = np.asarray(desired, dtype=np.float64)
    distance = max(
        np.linalg.norm(desired[0:3] - live[0:3]),
        np.linalg.norm(desired[7:10] - live[7:10]),
    )
    rotation = max(
        rotation_distance(live[3:6], desired[3:6]),
        rotation_distance(live[10:13], desired[10:13]),
    )
    duration = max(1.0, distance / 0.030, rotation / 0.15)
    steps = max(2, int(np.ceil(duration * 25.0)))
    alpha = np.linspace(0.0, 1.0, steps + 1)[1:]
    targets = live[None] + alpha[:, None] * (desired - live)[None]
    # Rotvec components cannot be interpolated linearly for a 90-degree
    # preparation move. Use the shortest SO(3) path independently per arm.
    for offset in (0, 7):
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_rotvec(np.stack((live[offset + 3 : offset + 6], desired[offset + 3 : offset + 6]))),
        )
        targets[:, offset + 3 : offset + 6] = slerp(alpha).as_rotvec()
    timestamps = time.time() + 0.30 + (np.arange(steps) + 1) / 25.0
    try:
        for begin in range(0, steps, 100):
            end = min(begin + 100, steps)
            env.exec_actions(
                targets[begin:end].copy(),
                timestamps[begin:end],
                compensate_latency=False,
                time_is_new=True,
                bimanual=True,
            )
            time.sleep(0.08)
        time.sleep(max(0.0, timestamps[-1] - time.time()) + 0.25)
    finally:
        env.hold(wait=True, timeout=2.0)


def execute_and_measure(env: UmiEnv, targets: np.ndarray, frequency: float, keys: TerminalKeys) -> dict:
    lead = 0.50
    dt = 1.0 / frequency
    started = time.time() + lead
    timestamps = started + (np.arange(len(targets)) + 1) * dt

    actual_rows = []
    emitted_rows = []
    sample_times = []
    interrupted = False
    sent = 0
    next_progress = time.monotonic() + 30.0
    deadline = timestamps[-1] + 0.50
    try:
        while time.time() < deadline:
            now = time.time()
            # Keep 10 seconds of future waypoints in each controller.  This
            # avoids overflowing the 256-entry shared-memory command queue.
            future_end = int(np.searchsorted(timestamps, now + 10.0, side="right"))
            if future_end > sent:
                end = min(future_end, sent + 64)
                env.exec_actions(
                    targets[sent:end].copy(),
                    timestamps[sent:end],
                    compensate_latency=False,
                    time_is_new=True,
                    bimanual=True,
                )
                sent = end
            if "s" in keys.poll():
                interrupted = True
                break
            states = env.get_robot_state()
            actual_rows.append(np.concatenate([np.asarray(state["ActualTCPPose"]) for state in states]))
            emitted_rows.append(np.concatenate([np.asarray(state["TargetTCPPose"]) for state in states]))
            sample_times.append(now)
            if time.monotonic() >= next_progress:
                complete = min(len(targets), int(np.searchsorted(timestamps, now, side="right")))
                logging.info("Replay progress: %d/%d waypoints (%.1f%%)", complete, len(targets), 100 * complete / len(targets))
                next_progress += 30.0
            time.sleep(0.02)
    finally:
        env.hold(wait=True, timeout=2.0)

    actual = np.asarray(actual_rows)
    emitted = np.asarray(emitted_rows)
    position_errors = {}
    rotation_errors = {}
    for name, offset in (("right", 0), ("left", 6)):
        position = np.linalg.norm(actual[:, offset : offset + 3] - emitted[:, offset : offset + 3], axis=1)
        rotation = np.asarray(
            [rotation_distance(a, b) for a, b in zip(actual[:, offset + 3 : offset + 6], emitted[:, offset + 3 : offset + 6])]
        )
        position_errors[name] = {
            "rms_mm": float(np.sqrt(np.mean(position**2)) * 1000.0),
            "max_mm": float(position.max() * 1000.0),
        }
        rotation_errors[name] = {
            "rms_deg": float(np.rad2deg(np.sqrt(np.mean(rotation**2)))),
            "max_deg": float(np.rad2deg(rotation.max())),
        }
    return {
        "interrupted_by_s": interrupted,
        "samples": len(actual),
        "position_tracking": position_errors,
        "rotation_tracking": rotation_errors,
        "sample_times": sample_times,
        "actual_pose6": actual.tolist(),
        "emitted_pose6": emitted.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/task487_cloud_sample"))
    parser.add_argument("--file-index", type=int, default=2)
    parser.add_argument("--episode", type=int, default=66)
    parser.add_argument("--start", type=int, default=26)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--full-episode", action="store_true")
    parser.add_argument("--absolute-dataset", action="store_true")
    parser.add_argument("--frequency", type=float, default=25.0)
    parser.add_argument("--move-grippers", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("task487_logs/trajectory_replay_latest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_episode:
        args.start = 0
        args.steps = 0
    if args.frequency <= 0:
        raise ValueError("--frequency must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client_lock = acquire_lock()
    body_actions = load_segment(args.dataset, args.file_index, args.episode, args.start, args.steps)
    absolute_start, absolute_targets = load_absolute_segment(
        args.dataset, args.file_index, args.episode, args.start, args.steps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env_output = args.output.parent / "trajectory_replay_env"
    env_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with SharedMemoryManager() as shm_manager, UmiEnv(
            output_dir=env_output,
            robot_ip=[None, None],
            gripper_ip=None,
            gripper_type="livelybot",
            gripper_can_if=[None, None],
            gripper_device_id=[9, 10],
            gripper_deg_open=GRIPPER_SAFE_OPEN_DEG,
            gripper_deg_closed=0.0,
            gripper_open_rad=GRIPPER_SAFE_OPEN_RAD,
            gripper_closed_rad=(0.0, 0.0),
            gripper_max_speed_dps=35.0,
            frequency=args.frequency,
            obs_image_resolution=(224, 224),
            init_joints=False,
            enable_task_ui=False,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.0,
            gripper_action_latency=0.0,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            max_pos_speed=(0.10 if args.absolute_dataset else 0.05) / np.sqrt(3.0),
            max_rot_speed=(0.30 if args.absolute_dataset else 0.20) / np.sqrt(3.0),
            robot_type="Marvin",
            shm_manager=shm_manager,
            task_name="task487",
            thor_enabled=True,
            thor_cameras=[],
        ) as env, TerminalKeys() as keys:
            env.hold(wait=True, timeout=2.0)
            anchor, bases = live_targets(env)
            if args.absolute_dataset:
                preview_anchor = absolute_start.copy()
                preview_targets = absolute_targets.copy()
                base_offsets = {"right": 0.0, "left": 0.0}
            else:
                replay_bases, base_offsets = workspace_fit_bases(body_actions, bases)
                preview_targets = body_actions_to_robot_targets(body_actions, replay_bases)
                preview_anchor = anchor.copy()
                preview_anchor[:6] = replay_bases[0]
                preview_anchor[7:13] = replay_bases[1]
            if not args.move_grippers:
                preview_targets[:, 6] = anchor[6]
                preview_targets[:, 13] = anchor[13]
                preview_anchor[6] = anchor[6]
                preview_anchor[13] = anchor[13]
            safety = validate_targets(
                preview_targets,
                preview_anchor,
                args.frequency,
                max_excursion_m=0.45 if args.full_episode else 0.03,
                max_translation_speed_m_s=0.10 if args.absolute_dataset else 0.05,
                max_rotation_speed_rad_s=0.30 if args.absolute_dataset else 0.20,
            )
            logging.warning(
                "Ready in HOLD (%s). Episode=%d frames=[%d,%s), replay=%.2fHz. "
                "Keys: [p] prepare start, [d] track once, [s] HOLD, Ctrl+C exit",
                "REAL COMMAND" if args.execute else "DRY RUN",
                args.episode,
                args.start,
                "end" if args.steps == 0 else str(args.start + args.steps),
                args.frequency,
            )
            if args.absolute_dataset:
                for name, offset in (("right", 0), ("left", 7)):
                    logging.info(
                        "%s preparation: translation=%.1fmm rotation=%.1fdeg",
                        name,
                        np.linalg.norm(absolute_start[offset : offset + 3] - anchor[offset : offset + 3]) * 1000.0,
                        np.rad2deg(rotation_distance(anchor[offset + 3 : offset + 6], absolute_start[offset + 3 : offset + 6])),
                    )
            else:
                logging.info("Workspace-fit vertical offsets: %s", base_offsets)
            logging.info("Validated trajectory: %s", safety)
            prepared = False
            while True:
                pressed = keys.poll()
                if "s" in pressed:
                    env.hold(wait=True, timeout=2.0)
                    logging.warning("HOLD")
                if "p" in pressed:
                    if not args.execute:
                        logging.info("DRY RUN: start preparation not sent")
                        prepared = True
                    else:
                        anchor, bases = live_targets(env)
                        if args.absolute_dataset:
                            desired_start = absolute_start.copy()
                            if not args.move_grippers:
                                desired_start[6] = anchor[6]
                                desired_start[13] = anchor[13]
                            logging.warning("Preparing the recorded absolute Task487 start pose")
                        else:
                            replay_bases, base_offsets = workspace_fit_bases(body_actions, bases)
                            desired_start = anchor.copy()
                            desired_start[:6] = replay_bases[0]
                            desired_start[7:13] = replay_bases[1]
                            desired_start[6] = preview_targets[0, 6]
                            desired_start[13] = preview_targets[0, 13]
                            logging.warning("Preparing replay anchor; vertical offsets=%s", base_offsets)
                        prepare_start(env, anchor, desired_start)
                        reached, _ = live_targets(env)
                        for name, offset in (("right", 0), ("left", 7)):
                            position_error = np.linalg.norm(
                                reached[offset : offset + 3] - desired_start[offset : offset + 3]
                            )
                            rotation_error = rotation_distance(
                                reached[offset + 3 : offset + 6],
                                desired_start[offset + 3 : offset + 6],
                            )
                            if position_error > 0.010 or rotation_error > np.deg2rad(5.0):
                                raise RuntimeError(
                                    f"{name} did not reach the recorded start pose: "
                                    f"position_error={position_error * 1000:.1f}mm "
                                    f"rotation_error={np.rad2deg(rotation_error):.1f}deg"
                                )
                        prepared = True
                        logging.warning("Replay anchor ready in HOLD; press [d] to execute")
                if "d" in pressed:
                    if not prepared:
                        logging.error("Start refused: press [p] to prepare the workspace-fitted anchor first")
                        time.sleep(0.02)
                        continue
                    if not args.execute:
                        logging.info("DRY RUN: trajectory is valid; no command sent")
                        break
                    anchor, bases = live_targets(env)
                    targets = (
                        absolute_targets.copy()
                        if args.absolute_dataset
                        else body_actions_to_robot_targets(body_actions, bases)
                    )
                    if not args.move_grippers:
                        targets[:, 6] = anchor[6]
                        targets[:, 13] = anchor[13]
                    safety = validate_targets(
                        targets,
                        anchor,
                        args.frequency,
                        max_excursion_m=0.45 if args.full_episode else 0.03,
                        max_translation_speed_m_s=0.10 if args.absolute_dataset else 0.05,
                        max_rotation_speed_rad_s=0.30 if args.absolute_dataset else 0.20,
                    )
                    logging.warning("Tracking recorded trajectory now")
                    report = {
                        "model_used": False,
                        "camera_used": False,
                        "dataset": str(args.dataset.resolve()),
                        "episode": args.episode,
                        "start": args.start,
                        "steps": args.steps,
                        "frequency_hz": args.frequency,
                        "full_episode": args.full_episode,
                        "move_grippers": args.move_grippers,
                        "absolute_dataset": args.absolute_dataset,
                        "trajectory_safety": safety,
                        "tracking": execute_and_measure(env, targets, args.frequency, keys),
                    }
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    logging.warning("Trajectory complete; robot returned to HOLD")
                    logging.info("Tracking report: %s", report["tracking"] | {"actual_pose6": "saved", "emitted_pose6": "saved", "sample_times": "saved"})
                    break
                time.sleep(0.02)
    except KeyboardInterrupt:
        logging.warning("Interrupted; UmiEnv is stopping controllers in HOLD")
    finally:
        client_lock.close()


if __name__ == "__main__":
    main()
