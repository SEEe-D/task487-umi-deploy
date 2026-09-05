#!/usr/bin/env python3
"""Task487 Pi0.5 multi-camera client for the Tianji/Marvin Mink stack."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import logging
from multiprocessing.managers import SharedMemoryManager
import os
import pathlib
import select
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from openpi_client import websocket_client_policy
from pi05_geometry_mask import GeometryMasker, geometry_q14_from_observation
from task487_runtime.contract import (
    TASK_PROMPTS,
    FOUR_WRIST_TASK_PROMPTS,
    CAMERA_SOURCES,
    LEGACY_CAMERA_ORDER,
    PolicyRuntimeContract,
    build_policy_request,
    task_prompts_for_runtime,
    resolve_task_for_runtime,
    thor_cameras_for_order,
    validate_policy_metadata,
)
from task487_runtime.diagnostics import ChunkDiagnosticRecorder
from task487_runtime.policy_logging import PolicyExchangeRecorder
from task487_runtime.telemetry_process import PassiveGripperTelemetry
from task487_runtime.gripper_dynamics import DEFAULT_GRIPPER_CONFIG, load_gripper_dynamics
from task487_runtime.scheduler import RollingScheduler, RunState, SchedulerConfig, UnsafeChunkError
from task487_runtime.worker import InferenceWorker
from task487_runtime.author_sync import (
    AuthorSyncScheduler, DEFAULT_SYNC_MAX_POS_SPEED, DEFAULT_SYNC_MAX_ROT_SPEED,
    DEFAULT_SYNC_SPEED_SCALE,
    ROBOT_LATENCY, GRIPPER_LATENCY,
)
from umi.real_world.umi_env import UmiEnv


CLIENT_LOCK_PATH = "/tmp/task487_client.lock"
# Order follows UmiEnv: right (Joint79), then left (Joint69). After a fresh
# closed-stop auto-zero on 2026-08-15, the right gripper reached its mechanical
# limit at -0.610865 rad with sustained high torque. Its safe endpoint is one
# degree inside that limit; the left endpoint remains its independently
# verified low-torque setting.
GRIPPER_SAFE_OPEN_RAD = (-0.59341195, -0.41949338)
# The Task487 dataset stores absolute Joint79/Joint69 opening angles in
# radians.  Use the magnitude of each verified physical endpoint as that
# side's degree limit so commands remain one-to-one (model rad -> physical
# joint rad) until the safe endpoint, where they are clipped.
GRIPPER_SAFE_OPEN_DEG = tuple(
    float(np.rad2deg(abs(open_rad))) for open_rad in GRIPPER_SAFE_OPEN_RAD
)
# Temporarily disabled for Task487 diagnosis.  Step/tracking guards and the
# physical velocity limiter remain active; no absolute or activation-relative
# TCP-z floor is applied.
MIN_TCP_Z_M = None
MAX_DOWNWARD_EXCURSION_M = None


def task_ui_configuration(task_prompts: dict[str, str]) -> tuple[tuple[dict, ...], dict[str, int]]:
    instructions = tuple(
        {
            "task_index": index,
            "task_key": task_key,
            "task": prompt,
        }
        for index, (task_key, prompt) in enumerate(task_prompts.items())
    )
    indices = {
        instruction["task_key"]: instruction["task_index"]
        for instruction in instructions
    }
    return instructions, indices


def acquire_client_lock():
    """Guarantee that only one process can publish Task487 robot targets."""
    lock_file = open(CLIENT_LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(
            f"Another Task487 client already owns {CLIENT_LOCK_PATH}; "
            "stop it before starting a second client"
        ) from None
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


class TerminalKeys:
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._settings = termios.tcgetattr(self._fd) if sys.stdin.isatty() else None
        if self._settings is not None:
            tty.setcbreak(self._fd)
        return self

    def poll(self) -> list[str]:
        if self._settings is None:
            return []
        keys = []
        while select.select([self._fd], [], [], 0.0)[0]:
            keys.append(os.read(self._fd, 1).decode("utf-8", errors="ignore").lower())
        return keys

    def __exit__(self, *_args):
        if self._settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)


class CameraPreview:
    """Display the exact 224x224 RGB images placed in each policy request."""

    RGB_WINDOW_NAME = "Task487 Pi0.5 exact model input (RGB before normalization)"
    PROCESSED_WINDOW_NAME = "Task487 Pi0.5 processed camera input (RGB + head mask tokens)"
    IMAGE_KEYS = tuple(CAMERA_SOURCES)
    TOKEN_GRID = 16
    MIN_KEEP_HEAD_TOKENS = 64

    def __init__(self, enabled: bool, show_processed: bool = False):
        self.enabled = bool(enabled)
        self.show_processed = bool(show_processed)
        self.request_count = 0
        self.frame_count = 0
        self._initial_preview_q14 = None
        self._initial_preview_mask = None

    @property
    def window_name(self) -> str:
        return self.PROCESSED_WINDOW_NAME if self.show_processed else self.RGB_WINDOW_NAME

    def __enter__(self):
        if self.enabled:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError(
                    "camera preview requires a graphical display; set DISPLAY "
                    "(normally :1 on this machine)"
                )
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, 1344, 900 if self.show_processed else 528)
        return self

    @classmethod
    def compose(cls, observation: dict, caption: str) -> np.ndarray:
        """Build a BGR display canvas without modifying policy input pixels."""
        image_keys = tuple(key for key in cls.IMAGE_KEYS if key in observation)
        if not image_keys:
            raise ValueError("Preview observation has no policy camera images")
        panels = []
        for key in image_keys:
            image = np.asarray(observation[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"Preview expected {key} uint8[224,224,3], got {image.shape} {image.dtype}")
            panels.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image_row = np.concatenate(panels, axis=1)
        header = np.zeros((40, image_row.shape[1], 3), dtype=np.uint8)
        for index, key in enumerate(image_keys):
            cv2.putText(
                header,
                key,
                (index * 224 + 8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            header,
            caption,
            (8, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.concatenate((header, image_row), axis=0)

    @classmethod
    def head_token_keep_mask(cls, mask_frame: np.ndarray) -> np.ndarray:
        """Mirror the server's 16x16 white-is-masked token pooling for preview."""
        mask = np.asarray(mask_frame)
        if mask.ndim == 3:
            if mask.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Preview expected a 2D or HWC mask, got {mask.shape}")
            mask = mask[..., :3].mean(axis=-1)
        elif mask.ndim != 2:
            raise ValueError(f"Preview expected a 2D or HWC mask, got {mask.shape}")
        if mask.shape != (224, 224):
            raise ValueError(f"Preview expected fixed_head_mask uint8[224,224], got {mask.shape}")
        if not np.isfinite(mask).all():
            raise ValueError("Preview head mask contains NaN or infinity")

        masked = mask > 127.0
        edges = np.linspace(0, 224, cls.TOKEN_GRID + 1, dtype=np.int64)
        ratios = np.empty((cls.TOKEN_GRID, cls.TOKEN_GRID), dtype=np.float32)
        for row in range(cls.TOKEN_GRID):
            for column in range(cls.TOKEN_GRID):
                ratios[row, column] = masked[
                    edges[row] : edges[row + 1], edges[column] : edges[column + 1]
                ].mean()
        keep = ratios < 0.5
        if int(keep.sum()) < cls.MIN_KEEP_HEAD_TOKENS:
            indices = np.argsort(ratios.reshape(-1), kind="stable")[: cls.MIN_KEEP_HEAD_TOKENS]
            keep.reshape(-1)[indices] = True
        return keep.reshape(cls.TOKEN_GRID, cls.TOKEN_GRID)

    @classmethod
    def compose_processed(cls, observation: dict, caption: str) -> np.ndarray:
        """Add pixel- and token-level mask views below the unmodified RGB row."""
        if "cam_head" not in observation:
            return cls.compose(observation, caption)
        rgb_canvas = cls.compose(observation, "")
        status_header = np.zeros((28, rgb_canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            status_header,
            caption,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        head_bgr = rgb_canvas[40:, :224].copy()
        mask = observation.get("fixed_head_mask")

        processed_header = np.zeros((32, rgb_canvas.shape[1], 3), dtype=np.uint8)
        labels = ("head + pixel mask", "head after 16x16 token mask", "fixed_head_mask")
        for index, label in enumerate(labels):
            cv2.putText(
                processed_header,
                label,
                (index * 224 + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if mask is None:
            panels = [head_bgr, head_bgr.copy(), np.zeros_like(head_bgr)]
            cv2.putText(
                panels[2],
                "MASK DISABLED",
                (40, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            mask = np.asarray(mask)
            if mask.shape != (224, 224) or mask.dtype != np.uint8:
                raise ValueError(
                    "Processed preview expected fixed_head_mask uint8[224,224], "
                    f"got {mask.shape} {mask.dtype}"
                )
            masked_pixels = mask > 127
            keep = cls.head_token_keep_mask(mask)
            masked_tokens = ~keep
            token_pixels = masked_tokens.repeat(14, axis=0).repeat(14, axis=1)

            pixel_overlay = head_bgr.astype(np.float32)
            pixel_overlay[masked_pixels] = (
                0.35 * pixel_overlay[masked_pixels] + 0.65 * np.array([0, 0, 255])
            )
            pixel_overlay = pixel_overlay.clip(0, 255).astype(np.uint8)

            token_view = head_bgr.copy()
            token_view[token_pixels] = (token_view[token_pixels].astype(np.float32) * 0.18).astype(
                np.uint8
            )
            for coordinate in range(0, 225, 14):
                endpoint = min(coordinate, 223)
                cv2.line(token_view, (endpoint, 0), (endpoint, 223), (70, 70, 70), 1)
                cv2.line(token_view, (0, endpoint), (223, endpoint), (70, 70, 70), 1)

            binary_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                binary_mask,
                f"masked px={int(masked_pixels.sum())}",
                (7, 198),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                binary_mask,
                f"masked tokens={int(masked_tokens.sum())}/256",
                (7, 217),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels = [pixel_overlay, token_view, binary_mask]

        processed_row = np.concatenate(panels, axis=1)
        if processed_row.shape[1] < rgb_canvas.shape[1]:
            processed_row = np.pad(
                processed_row, ((0, 0), (0, rgb_canvas.shape[1] - processed_row.shape[1]), (0, 0))
            )
        return np.concatenate((status_header, rgb_canvas, processed_header, processed_row), axis=0)

    def show(self, request, stage: str) -> None:
        if not self.enabled:
            return
        self.request_count += 1
        caption = f"request={self.request_count} round={request.round_id} {stage}"
        canvas = (
            self.compose_processed(request.observation, caption)
            if self.show_processed
            else self.compose(request.observation, caption)
        )
        cv2.imshow(self.window_name, canvas)
        self.pump()

    def motion_status(self, observation: dict, q14: np.ndarray) -> str:
        """Report movement from the first preview frame for joint/mask diagnostics."""
        q14 = np.asarray(q14, dtype=np.float64)
        if q14.shape != (14,) or not np.all(np.isfinite(q14)):
            raise ValueError(f"Preview expected finite q14, got {q14.shape}")
        mask = observation.get("fixed_head_mask")
        if mask is None:
            return "q_from_start=n/a mask_from_start=n/a"
        mask = np.asarray(mask)
        if mask.shape != (224, 224):
            raise ValueError(f"Preview expected fixed_head_mask[224,224], got {mask.shape}")

        if self._initial_preview_q14 is None:
            self._initial_preview_q14 = q14.copy()
            self._initial_preview_mask = (mask > 127).copy()
        q_delta_deg = float(np.rad2deg(np.max(np.abs(q14 - self._initial_preview_q14))))
        mask_delta_px = int(np.count_nonzero((mask > 127) != self._initial_preview_mask))
        return f"q_from_start={q_delta_deg:.2f}deg mask_from_start={mask_delta_px}px"

    def show_live(self, observation: dict, caption: str, q14: np.ndarray) -> None:
        """Refresh processed inputs at the control-loop rate, including in HOLD."""
        if not self.enabled or not self.show_processed:
            return
        self.frame_count += 1
        status = f"frame={self.frame_count} {caption} {self.motion_status(observation, q14)}"
        cv2.imshow(self.window_name, self.compose_processed(observation, status))
        self.pump()

    def pump(self) -> None:
        if self.enabled:
            cv2.waitKey(1)

    def __exit__(self, *_args):
        if self.enabled:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--task", choices=(*TASK_PROMPTS, *FOUR_WRIST_TASK_PROMPTS), default="vegetable")
    parser.add_argument(
        "--no-task-ui",
        action="store_true",
        help="Disable the runtime-specific task-selection window",
    )
    parser.add_argument("--execute", action="store_true", help="Enable real robot command output")
    parser.add_argument("--continuous", action="store_true", help="Continue beyond the first five-waypoint round")
    parser.add_argument("--execution-mode", choices=("rolling", "author-sync"), default="rolling")
    parser.add_argument("--gripper-config", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get("TASK487_GRIPPER_CONFIG", DEFAULT_GRIPPER_CONFIG)),
                        help="The same gripper_can.yaml loaded by the robot backend; used for joint timing.")
    parser.add_argument("--sync-steps-per-inference", type=int, default=6)
    parser.add_argument("--sync-max-pos-speed", type=float, default=DEFAULT_SYNC_MAX_POS_SPEED,
                        help="Effective Cartesian m/s cap for author-sync (no further sqrt(3) multiplier)")
    parser.add_argument("--sync-max-rot-speed", type=float, default=DEFAULT_SYNC_MAX_ROT_SPEED,
                        help="Effective rad/s cap for author-sync")
    parser.add_argument("--sync-speed-scale", type=float, default=DEFAULT_SYNC_SPEED_SCALE,
                        help="Author-sync path playback rate in (0, 1], default 0.5; "
                             "keeps physical speed caps, gripper travel time and replan cadence")
    parser.add_argument("--sync-right-gripper-preload-deg", type=float, default=0.0,
                        help="Optional right closing preload, 0..2 degrees (default 0/off); "
                             "withdrawn on opening, retimed with both arms; not force control")
    parser.add_argument("--sync-gripper-close-compensation-deg", type=float, default=0.0,
                        help="Both hands: subtract 0..5 degrees on closing (default 0/off)")
    parser.add_argument("--sync-gripper-open-compensation-deg", type=float, default=0.0,
                        help="Both hands: add 0..5 degrees on opening (default 0/off); "
                             "physical endpoints and shared speed limits retained")
    parser.add_argument("--sync-right-before-left", action="store_true",
                        help="Hold left arm/gripper until right close-then-open target is reached; requires compensation")
    parser.add_argument(
        "--max-waypoints",
        type=int,
        help="Automatically HOLD after N executed waypoints (also bounds --continuous RTC tests)",
    )
    parser.add_argument("--thor-host", default="192.168.2.178")
    parser.add_argument("--thor-receiver-path", default=".")
    parser.add_argument("--output", default="/tmp/pi05_task487_eval")
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Show all camera images selected by the server contract in each policy request",
    )
    parser.add_argument(
        "--show-processed-cameras",
        action="store_true",
        help=(
            "Show exact policy RGB plus the aligned head pixel mask, 16x16 token mask, "
            "and binary mask at the live control-loop rate"
        ),
    )
    parser.add_argument("--max-camera-age", type=float, default=0.25)
    # Receivers expose only the latest frame per camera.  During the interval
    # in which a new frame has reached some streams but not the others,
    # latest timestamps may differ by one frame period even though capture is
    # synchronized.  50 ms admits that arrival window and still rejects a
    # stream lagging by more than one training frame.
    parser.add_argument("--max-camera-skew", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_waypoints is not None and args.max_waypoints <= 0:
        parser.error("--max-waypoints must be positive")
    if not np.isfinite(args.sync_right_gripper_preload_deg) or not 0 <= args.sync_right_gripper_preload_deg <= 2:
        parser.error("--sync-right-gripper-preload-deg must be finite and in [0, 2]")
    if args.sync_right_gripper_preload_deg and args.execution_mode != "author-sync":
        parser.error("--sync-right-gripper-preload-deg requires --execution-mode author-sync")
    for name in ("sync_gripper_close_compensation_deg", "sync_gripper_open_compensation_deg"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 5:
            parser.error(f"--{name.replace('_', '-')} must be finite and in [0, 5]")
    if args.sync_gripper_close_compensation_deg or args.sync_gripper_open_compensation_deg:
        if args.execution_mode != "author-sync":
            parser.error("gripper compensation requires --execution-mode author-sync")
        if args.sync_right_gripper_preload_deg:
            parser.error("gripper compensation cannot be combined with --sync-right-gripper-preload-deg")
    if args.sync_right_before_left and not (args.sync_gripper_close_compensation_deg or args.sync_gripper_open_compensation_deg):
        parser.error("--sync-right-before-left requires gripper compensation")
    return args


def validate_server_metadata(metadata: dict, *, require_rtc: bool = True) -> PolicyRuntimeContract:
    return validate_policy_metadata(metadata, require_rtc=require_rtc)


def validate_thor(
    env: UmiEnv, max_age: float, max_skew: float,
    camera_order: tuple[str, ...] = LEGACY_CAMERA_ORDER,
) -> None:
    now = time.time()
    expected = tuple(camera["label"] for camera in thor_cameras_for_order(camera_order))
    missing = set(expected) - env.thor_receivers.keys()
    if missing:
        raise RuntimeError(f"Missing required Thor receivers: {sorted(missing)}")
    capture_times = []
    for label in expected:
        receiver = env.thor_receivers[label]
        with receiver.lock:
            has_frame = receiver.frame is not None
        capture_time = receiver.meta.latest_ts_us / 1_000_000.0 - receiver.meta.clock_offset_ms / 1_000.0
        age = now - capture_time
        if not has_frame or receiver.meta.latest_ts_us <= 0:
            raise RuntimeError(f"Thor camera {label} has no frame or metadata")
        if age < -0.05 or age > max_age:
            raise RuntimeError(f"Thor camera {label} age {age:.3f}s is outside [-0.05, {max_age:.2f}]s")
        capture_times.append(capture_time)
    skew = max(capture_times) - min(capture_times)
    if skew > max_skew:
        raise RuntimeError(f"Thor {len(expected)}-camera skew {skew:.4f}s exceeds {max_skew:.4f}s")


def wait_for_thor(
    env: UmiEnv, max_age: float, max_skew: float, timeout: float = 15.0,
    camera_order: tuple[str, ...] = LEGACY_CAMERA_ORDER,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            validate_thor(env, max_age, max_skew, camera_order)
            return
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Thor cameras were not ready within {timeout:.1f}s: {last_error}")


def read_live_targets(env: UmiEnv) -> tuple[np.ndarray, np.ndarray]:
    """Return actual robot feedback and the controller's emitted setpoint."""
    actual_target = np.empty(14, dtype=np.float64)
    controller_target = np.empty(14, dtype=np.float64)
    robot_states = env.get_robot_state()
    gripper_states = env.get_gripper_state()
    for arm, (robot_state, gripper_state) in enumerate(zip(robot_states, gripper_states)):
        offset = 7 * arm
        actual_target[offset : offset + 6] = np.asarray(
            robot_state["ActualTCPPose"], dtype=np.float64
        )
        controller_target[offset : offset + 6] = np.asarray(
            robot_state["TargetTCPPose"], dtype=np.float64
        )
        gripper = float(np.asarray(gripper_state["gripper_position"]).reshape(-1)[-1])
        actual_target[offset + 6] = gripper
        controller_target[offset + 6] = gripper
    return actual_target, controller_target


def preview_model_intent(result, now: float, config: SchedulerConfig, live_target: np.ndarray) -> tuple:
    """Summarize the next executable policy window without altering it."""
    targets = np.asarray(result.targets, dtype=np.float64)
    target_times = result.request.observation_time + (np.arange(len(targets)) + 1) * config.dt
    valid_indices = np.flatnonzero(target_times > now + config.dispatch_lead_s)
    if len(valid_indices) == 0:
        return (-1, 0.0, 0.0, 0.0, 0.0, *(0.0,) * 6)
    # Keep this diagnostic as a five-action local intent preview even though
    # the scheduler now executes most of a 20-action chunk before replanning.
    window_indices = valid_indices[:5]
    window = targets[window_indices]
    values = []
    for arm_offset in (0, 7):
        position_m = np.linalg.norm(
            window[:, arm_offset : arm_offset + 3] - live_target[arm_offset : arm_offset + 3],
            axis=1,
        ).max()
        live_rotation = Rotation.from_rotvec(live_target[arm_offset + 3 : arm_offset + 6])
        target_rotations = Rotation.from_rotvec(window[:, arm_offset + 3 : arm_offset + 6])
        rotation_rad = (live_rotation.inv() * target_rotations).magnitude().max()
        values.extend((float(position_m), float(rotation_rad)))
    signed_endpoint_delta_mm = np.concatenate(
        (
            window[-1, 0:3] - live_target[0:3],
            window[-1, 7:10] - live_target[7:10],
        )
    ) * 1000.0
    return (int(window_indices[0]), *values, *signed_endpoint_delta_mm.tolist())


def hold_robot(env: UmiEnv) -> None:
    env.hold(wait=True, timeout=2.0)


def activate_scheduler(scheduler, env, live):
    if isinstance(scheduler, AuthorSyncScheduler) and scheduler.compensation_enabled:
        policy_angles = [float(np.asarray(state["gripper_policy_position"]).reshape(-1)[-1])
                         for state in env.get_gripper_state()]
        scheduler.activate(live, gripper_policy_live=policy_angles)
    else:
        scheduler.activate(live)


@contextmanager
def physical_hold_on_exit(env, execute):
    """HOLD before UmiEnv tears down its processes, including exception paths."""
    try:
        yield
    finally:
        if execute:
            try:
                hold_robot(env)
            except Exception:
                logging.exception("Failed to confirm final HOLD; inspect backend / use hardware E-stop")


def read_gripper_degrees(env: UmiEnv) -> np.ndarray:
    return np.asarray(
        [
            float(np.asarray(state["gripper_position"]).reshape(-1)[-1])
            for state in env.get_gripper_state()
        ],
        dtype=np.float64,
    )


def grippers_are_ready(env: UmiEnv, runtime_contract: PolicyRuntimeContract) -> bool:
    positions = read_gripper_degrees(env)
    targets = np.asarray(runtime_contract.gripper_start_degrees, dtype=np.float64)
    return bool(np.all(np.abs(positions - targets) <= runtime_contract.gripper_ready_tolerance_deg))


def prepare_grippers_for_runtime(
    env: UmiEnv,
    runtime_contract: PolicyRuntimeContract,
    timeout: float = 5.0,
) -> None:
    """Clear a previous grasp latch by opening before preparing the boundary.

    The installed driver rearms on a request >1 degree beyond the measured
    opening. Merely sending a smaller angle cannot leave a latched hold.
    """
    targets = np.asarray(runtime_contract.gripper_start_degrees, dtype=np.float64)
    positions = read_gripper_degrees(env)
    if positions.shape != (2,) or not np.isfinite(positions).all():
        raise RuntimeError("gripper preparation requires finite measured positions")
    release_targets = np.minimum(positions + 2.0, GRIPPER_SAFE_OPEN_DEG)
    logging.info("Gripper prepare: open to clear previous stall latch, right=%.2fdeg left=%.2fdeg",
                 *release_targets)
    failure = None
    for stage, goal, tolerance in (
        ("open-to-rearm", release_targets, .25),
        ("checkpoint-close", targets, min(.5, runtime_contract.gripper_ready_tolerance_deg)),
    ):
        target_time = time.time() + .10
        for gripper, target in zip(env.grippers, goal, strict=True):
            # No compensation during preparation; reset the paired reference.
            gripper.schedule_waypoint(float(target), target_time)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            measured = read_gripper_degrees(env)
            if measured.shape != (2,) or not np.isfinite(measured).all():
                failure = f"{stage}: invalid gripper feedback"
                break
            positions = measured
            if np.all(np.abs(positions - goal) <= tolerance):
                logging.info("Gripper prepare %s reached: right=%.2fdeg left=%.2fdeg", stage, *positions)
                break
            time.sleep(.05)
        else:
            failure = f"{stage}: targets not reached"
        if failure:
            break
    if failure is None:
        logging.info("Grippers prepared at right=%.1fdeg left=%.1fdeg (targets %.1f/%.1fdeg)",
                     *positions, *targets)
        return

    # Freeze at measured positions before failing.  Without this, the gripper
    # interpolation processes keep streaming an unreachable target
    # and the motor driver remains torque-saturated even though ACTIVE is
    # blocked.
    freeze_time = time.time() + 0.10
    for gripper, position in zip(env.grippers, positions):
        gripper.schedule_waypoint(float(position), freeze_time)
    time.sleep(0.20)
    raise RuntimeError(
        "grippers did not reach the checkpoint boundary state: "
        f"{failure}; "
        f"right={positions[0]:.1f}deg left={positions[1]:.1f}deg "
        f"(targets={targets[0]:.1f}/{targets[1]:.1f}deg, "
        f"stage_tolerance={tolerance:.2f}deg)"
    )


def return_home(env: UmiEnv) -> None:
    logging.warning("Requesting Marvin HOME through Mink FSM")
    env.go_home(wait=True, timeout=10.0)
    logging.info("HOME complete; state remains HOLD")


def wait_for_warmup(worker: InferenceWorker, request, timeout: float = 30.0):
    worker.submit(request)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            worker.record_disposition(result, "warmup_error" if result.error is not None else "warmup_discarded")
            if result.error is not None:
                raise RuntimeError("Task487 warmup inference failed") from result.error
            logging.info("Warmup inference complete in %.1fms", result.latency_s * 1000)
            return result
        time.sleep(0.02)
    raise TimeoutError(f"Task487 warmup did not finish within {timeout:.1f}s")


def render_runtime_head_mask(
    obs: dict,
    runtime_contract: PolicyRuntimeContract,
    geometry_masker: GeometryMasker | None,
) -> dict | None:
    """Render the synchronized robot silhouette required by masked policies."""
    if not runtime_contract.mask_enabled:
        return None
    if geometry_masker is None:
        raise RuntimeError("Masked Task487 runtime requires a geometry masker")
    head_history = np.asarray(obs["camera5_rgb"])
    if head_history.ndim != 4 or head_history.shape[-3:] != (224, 224, 3):
        raise ValueError(f"camera5_rgb must contain 224x224 RGB history, got {head_history.shape}")
    return geometry_masker.render(
        np.ascontiguousarray(head_history[-1]),
        geometry_q14_from_observation(obs),
    )


def submit_rtc_request(
    worker: InferenceWorker,
    scheduler: RollingScheduler,
    obs: dict,
    prompt: str,
    round_id: int,
    action_horizon: int,
    camera_preview: CameraPreview,
    runtime_contract: PolicyRuntimeContract,
    geometry_masker: GeometryMasker | None,
) -> None:
    if worker.busy:
        return  # Drain/invalidate the previous round before submitting another.
    plain = isinstance(scheduler, AuthorSyncScheduler)
    prefix = None if plain else scheduler.rtc_prefix_targets()
    mask_result = render_runtime_head_mask(obs, runtime_contract, geometry_masker)
    request = build_policy_request(
        obs,
        prompt,
        rtc_prefix_targets=prefix,
        round_id=round_id,
        action_horizon=action_horizon,
        fixed_head_mask=None if mask_result is None else mask_result["combined_mask"],
        camera_order=runtime_contract.camera_order,
        gripper_policy_coordinates=plain and scheduler.compensation_enabled,
    )
    camera_preview.show(request, "plain_author_sync" if plain else f"rtc_prefix={len(prefix)}")
    worker.submit(request)
    scheduler.mark_request_started(time.time())


def main() -> None:
    args = parse_args()
    client_lock = acquire_client_lock()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    selected_task_key = args.task
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    policy = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    server_metadata = policy.get_server_metadata()
    author_sync = args.execution_mode == "author-sync"
    runtime_contract = validate_server_metadata(server_metadata, require_rtc=not author_sync)
    geometry_masker = (
        GeometryMasker(
            image_size=224,
            image_geometry=runtime_contract.image_geometry,
        )
        if runtime_contract.mask_enabled
        else None
    )
    task_prompts = task_prompts_for_runtime(runtime_contract.runtime)
    selected_task_key = resolve_task_for_runtime(runtime_contract.runtime, args.task)
    if selected_task_key != args.task:
        logging.warning(
            "Task alias %r maps to %r for %s: this checkpoint was trained on one sorting prompt",
            args.task, selected_task_key, runtime_contract.runtime,
        )
    prompt = task_prompts[selected_task_key]
    task_ui_instructions, task_ui_index = task_ui_configuration(task_prompts)
    action_horizon = runtime_contract.action_horizon
    control_hz = runtime_contract.control_hz
    gripper_dynamics = load_gripper_dynamics(args.gripper_config)
    scheduler = RollingScheduler(
        SchedulerConfig.for_policy_rate(
            control_hz,
            action_horizon,
            min_tcp_z_m=MIN_TCP_Z_M,
            max_downward_excursion_m=MAX_DOWNWARD_EXCURSION_M,
            complete_chunk_before_replan=runtime_contract.complete_chunk_before_replan,
            max_physical_gripper_speed_deg_s=gripper_dynamics.joint_speed_deg_s,
        )
    )
    if author_sync:
        scheduler = AuthorSyncScheduler(
            scheduler.config, action_horizon=action_horizon,
            steps_per_inference=args.sync_steps_per_inference,
            max_pos_speed=args.sync_max_pos_speed, max_rot_speed=args.sync_max_rot_speed,
            speed_scale=args.sync_speed_scale,
            right_gripper_preload_deg=args.sync_right_gripper_preload_deg,
            gripper_close_compensation_deg=args.sync_gripper_close_compensation_deg,
            gripper_open_compensation_deg=args.sync_gripper_open_compensation_deg,
            gripper_open_limits_deg=GRIPPER_SAFE_OPEN_DEG,
            right_before_left=args.sync_right_before_left)
    logging.warning(
        "Execution=%s; effective speed limits=%.3fm/s %.3frad/s; policy rate=%.1fHz",
        args.execution_mode, scheduler.config.max_physical_translation_speed_m_s,
        scheduler.config.max_physical_rotation_speed_rad_s, control_hz)
    logging.info("Gripper timing: %.3f deg/s joint (CAN %.3f rad/s motor, gear 20); config=%s",
                 gripper_dynamics.joint_speed_deg_s, gripper_dynamics.motor_speed_rad_s,
                 gripper_dynamics.config_path)
    diagnostic_root = pathlib.Path(os.environ.get("UMI_ACTION_LOG_DIR", args.output))
    chunk_diagnostics = ChunkDiagnosticRecorder(diagnostic_root)
    dt = scheduler.config.dt
    latest_obs = None
    preview_error = None
    logging.info(
        "Task487 task=%s prompt=%r runtime=%s control_hz=%.1f action_horizon=%d "
        "image_geometry=%s gripper_start=%.1f/%.1fdeg",
        selected_task_key,
        prompt,
        server_metadata["runtime"],
        control_hz,
        action_horizon,
        runtime_contract.image_geometry,
        *runtime_contract.gripper_start_degrees,
    )
    logging.info(
        "Policy chunk diagnostics: %s (exact arrays in %s)",
        chunk_diagnostics.summary_path,
        chunk_diagnostics.chunk_dir,
    )
    logging.info("Policy camera order: %s; Thor streams: %s",
                 runtime_contract.camera_order, thor_cameras_for_order(runtime_contract.camera_order))
    if author_sync:
        logging.info("Plain Bezier: replan=%d steps (%.3fs); device latency=%.3f/%.3fs; no RTC prefill",
                     scheduler.steps_per_inference, scheduler.steps_per_inference * dt,
                     ROBOT_LATENCY, GRIPPER_LATENCY)
        logging.info("Path playback=%.2fx; continuous future splice; extra local timing correction <=150ms; "
                     "controller speed caps retained", scheduler.speed_scale)
        logging.info("Right gripper closing preload=%.2fdeg (0=off); original policy values remain in diagnostics; "
                     "feedback remains measured", scheduler.right_gripper_preload_deg)
        logging.info("Both grippers compensation: closing=-%.2fdeg opening=+%.2fdeg (0=off); "
                     "opening limits right=%.3fdeg left=%.3fdeg; direction hysteresis=0.5deg; "
                     "original policy values and measured feedback retained",
                     scheduler.gripper_close_compensation_deg, scheduler.gripper_open_compensation_deg,
                     *GRIPPER_SAFE_OPEN_DEG)
        if scheduler.compensation_enabled:
            logging.info("Compensation feedback=v2 paired physical/policy trajectories; "
                         "policy state=policy reference + measured tracking error; raw feedback separately logged; "
                         "right-before-left=%s", scheduler.right_before_left)
    else:
        logging.info("Scheduler complete_chunk_before_replan=%s; commit_window_s=%s; commit_steps=%d",
                     scheduler.config.complete_chunk_before_replan,
                     scheduler.config.commit_window_s, scheduler.config.commit_steps)

    with CameraPreview(
        args.show_cameras or args.show_processed_cameras,
        show_processed=args.show_processed_cameras,
    ) as camera_preview, SharedMemoryManager() as shm_manager:
        with UmiEnv(
            output_dir=args.output,
            robot_ip=[None, None],
            gripper_ip=None,
            gripper_type="livelybot",
            gripper_can_if=[None, None],
            gripper_device_id=[9, 10],
            gripper_width_open_m=0.09,
            gripper_deg_open=GRIPPER_SAFE_OPEN_DEG,
            gripper_deg_closed=0.0,
            # Keep the model's 0..35 contract while mapping each side to its
            # independently measured safe physical opening endpoint.
            gripper_open_rad=GRIPPER_SAFE_OPEN_RAD,
            gripper_closed_rad=(0.0, 0.0),
            gripper_kp=10.0,
            gripper_kd=1.0,
            gripper_max_speed_dps=gripper_dynamics.joint_speed_deg_s,
            frequency=control_hz,
            # Select the exact camera geometry tied to the validated runtime:
            # old pure-real checkpoints center-crop, raw UMI preserves the full
            # 1.25:1 field of view and pads it to a square.
            obs_image_resolution=(224, 224),
            obs_float32=False,
            thor_center_crop=runtime_contract.image_geometry == "center_square",
            thor_resize_with_pad=runtime_contract.image_geometry == "resize_with_pad",
            init_joints=False,
            enable_task_ui=not args.no_task_ui,
            task_instructions=task_ui_instructions,
            initial_task_index=task_ui_index[selected_task_key],
            camera_obs_latency=0.0,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=ROBOT_LATENCY if author_sync else 0.0,
            gripper_action_latency=GRIPPER_LATENCY if author_sync else 0.0,
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # UmiEnv multiplies these values by cube_diag before constructing
            # each controller. Derive them from the scheduler so both layers
            # enforce the same executable-time contract.
            max_pos_speed=scheduler.config.max_physical_translation_speed_m_s / np.sqrt(3.0),
            max_rot_speed=scheduler.config.max_physical_rotation_speed_rad_s / np.sqrt(3.0),
            robot_type="Marvin",
            tcp_offset_x=-0.016,
            tcp_offset_y=-0.028,
            tcp_offset_z=0.2105,
            tcp_rot_x=0.0,
            tcp_rot_y=-0.1745,
            tcp_rot_z=0.0,
            shm_manager=shm_manager,
            task_name="task487",
            thor_enabled=True,
            thor_host=args.thor_host,
            thor_receiver_path=args.thor_receiver_path,
            thor_cameras=thor_cameras_for_order(runtime_contract.camera_order),
            thor_tile_w=640,
            thor_tile_h=512,
        ) as env, TerminalKeys() as keys, \
                PolicyExchangeRecorder(diagnostic_root, metadata={
                    "server_metadata": server_metadata, "client_args": vars(args),
                    "runtime_contract": vars(runtime_contract),
                    "gripper_config_path": str(gripper_dynamics.config_path),
                    "gripper_config_text": gripper_dynamics.config_path.read_text(),
                    "angle_convention": "model gripper radians; robot_targets opening degrees; right then left",
                }) as policy_recorder, PassiveGripperTelemetry(diagnostic_root) as telemetry, \
                InferenceWorker(policy, action_horizon=action_horizon, recorder=policy_recorder) as worker, \
                physical_hold_on_exit(env, args.execute):
            logging.info("Exact policy RGB/state/responses: %s (async, lossless; request IDs link to chunk logs)",
                         policy_recorder.events_path)

            wait_for_thor(env, args.max_camera_age, args.max_camera_skew,
                          camera_order=runtime_contract.camera_order)
            latest_obs = env.get_obs()
            warmup_mask = render_runtime_head_mask(latest_obs, runtime_contract, geometry_masker)
            warmup_request = build_policy_request(
                latest_obs,
                prompt,
                action_horizon=action_horizon,
                fixed_head_mask=None if warmup_mask is None else warmup_mask["combined_mask"],
                camera_order=runtime_contract.camera_order,
            )
            if warmup_mask is not None:
                mask_preview_path = pathlib.Path(args.output) / "geometry_mask_preview.png"
                cv2.imwrite(
                    str(mask_preview_path),
                    cv2.cvtColor(warmup_mask["overlay_rgb"], cv2.COLOR_RGB2BGR),
                )
                logging.info(
                    "Masked runtime geometry ready: preview=%s masked_pixels=%d triangles=%s",
                    mask_preview_path,
                    int(np.count_nonzero(warmup_mask["combined_mask"])),
                    warmup_mask["statistics"].get("rendered_triangle_count"),
                )
            camera_preview.show(warmup_request, "plain_warmup")
            warmup_result = wait_for_warmup(worker, warmup_request)
            # The JAX function has a separate compiled signature when action
            # prefill is present. Compile it in HOLD, never on the first live
            # RTC handoff.
            if not author_sync:
                rtc_warmup_request = build_policy_request(
                    latest_obs, prompt,
                    rtc_prefix_targets=warmup_result.targets[: scheduler.config.commit_steps],
                    action_horizon=action_horizon,
                    fixed_head_mask=None if warmup_mask is None else warmup_mask["combined_mask"],
                    camera_order=runtime_contract.camera_order,
                )
                camera_preview.show(rtc_warmup_request, "rtc_warmup")
                wait_for_warmup(worker, rtc_warmup_request)
            logging.info("Warmup outputs discarded (RTC prefill=%s)", not author_sync)
            # The window starts together with UmiEnv, so an operator may have
            # selected the other Task487 prompt while JAX warmup was running.
            selected_instruction = env.get_task_instruction()
            if selected_instruction is not None:
                selected_task_key = selected_instruction["task_key"]
                prompt = task_prompts[selected_task_key]
            grippers_prepared = grippers_are_ready(env, runtime_contract)
            mode = "REAL COMMAND" if args.execute else "DRY RUN"
            waypoint_limit = args.max_waypoints
            if waypoint_limit is None and not args.continuous:
                waypoint_limit = 5
            round_mode = (
                f"bounded {args.execution_mode} round ({waypoint_limit} waypoints)"
                if waypoint_limit is not None
                else "continuous"
            )
            logging.warning(
                "Ready in HOLD (%s, %s). Task=%s. Grippers=%s. "
                "Keys: [r] HOME+PREPARE, [d] start, [s] HOLD",
                mode,
                round_mode,
                selected_task_key,
                "READY" if grippers_prepared else "NOT PREPARED",
            )

            stop_requested = threading.Event()
            previous_sigint = signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
            executed_in_round = 0
            active_round_id = 0
            next_tick = time.monotonic()
            last_sensor_error = None
            while not stop_requested.is_set():
                telemetry.check()
                next_tick += dt
                camera_preview.pump()
                selected_instruction = env.get_task_instruction()
                if (
                    selected_instruction is not None
                    and selected_instruction["task_key"] != selected_task_key
                ):
                    # A live task switch is an automatic safe handoff, never a
                    # prompt mutation inside an existing trajectory: first
                    # enter physical HOLD, invalidate the old inference round
                    # and timeline, then reactivate from the measured pose.
                    previous_task_key = selected_task_key
                    selected_task_key = selected_instruction["task_key"]
                    prompt = task_prompts[selected_task_key]
                    was_active = scheduler.state is RunState.ACTIVE
                    scheduler.hold(f"task switched to {selected_task_key}")
                    active_round_id += 1
                    executed_in_round = 0
                    resume_after_switch = was_active
                    if was_active and args.execute:
                        try:
                            hold_robot(env)
                        except Exception:
                            resume_after_switch = False
                            logging.exception(
                                "Task switch entered software HOLD but physical HOLD failed; "
                                "automatic resume cancelled"
                            )
                    if resume_after_switch:
                        try:
                            activation_live, _ = read_live_targets(env)
                            activate_scheduler(scheduler, env, activation_live)
                        except Exception:
                            scheduler.hold("task switch activation failed")
                            logging.exception(
                                "Task switched but activation from live pose failed; "
                                "state remains HOLD"
                            )
                        else:
                            logging.warning(
                                "Task switched during ACTIVE: %s -> %s; physical HOLD "
                                "completed, old trajectory/inference invalidated, fresh "
                                "round auto-resumed; prompt=%r",
                                previous_task_key,
                                selected_task_key,
                                prompt,
                            )
                    else:
                        logging.warning(
                            "Task switched in HOLD: %s -> %s; old trajectory and "
                            "inference round invalidated; prompt=%r",
                            previous_task_key,
                            selected_task_key,
                            prompt,
                        )
                for key in keys.poll():
                    if key == "s":
                        scheduler.hold("operator pause")
                        if args.execute and latest_obs is not None:
                            hold_robot(env)
                        logging.warning("HOLD: operator pause")
                    elif key == "d" and scheduler.state is RunState.HOLD:
                        if args.execute and not grippers_prepared:
                            logging.error(
                                "Start refused: press [r] to HOME and prepare the checkpoint-specific "
                                "gripper boundary state first"
                            )
                        else:
                            activation_live, _ = read_live_targets(env)
                            active_round_id += 1
                            activate_scheduler(scheduler, env, activation_live)
                            executed_in_round = 0
                            workspace_floors = scheduler.workspace_min_z
                            if workspace_floors is None:
                                logging.warning(
                                    "ACTIVE: beginning a fresh inference round; "
                                    "workspace height limit DISABLED"
                                )
                            else:
                                logging.warning(
                                    "ACTIVE: beginning a fresh inference round; "
                                    "workspace floors right=%.3fm left=%.3fm",
                                    *workspace_floors,
                                )
                    elif key == "r" and scheduler.state is RunState.HOLD:
                        if args.execute:
                            latest_obs = env.get_obs()
                            validate_thor(env, args.max_camera_age, args.max_camera_skew,
                                          runtime_contract.camera_order)
                            return_home(env)
                            try:
                                prepare_grippers_for_runtime(env, runtime_contract)
                            except RuntimeError as exc:
                                grippers_prepared = False
                                logging.error("HOME complete but start remains blocked: %s", exc)
                            else:
                                grippers_prepared = True
                                if author_sync:
                                    scheduler.reset_gripper_episode()
                        else:
                            logging.info("DRY RUN: HOME command ignored")

                try:
                    validate_thor(env, args.max_camera_age, args.max_camera_skew,
                                  runtime_contract.camera_order)
                    latest_obs = env.get_obs()
                    live, controller_target = read_live_targets(env)
                except Exception as exc:
                    scheduler.hold(f"observation fault: {exc}")
                    message = str(exc)
                    if message != last_sensor_error:
                        if args.execute:
                            try:
                                hold_robot(env)
                            except Exception:
                                logging.exception("Failed to command HOLD after observation fault")
                        logging.error("Observation fault; HOLD: %s", message)
                        last_sensor_error = message
                    time.sleep(0.10)
                    next_tick = time.monotonic()
                    continue
                if last_sensor_error is not None:
                    logging.info("Observation stream recovered; state remains HOLD")
                    last_sensor_error = None

                if camera_preview.show_processed:
                    try:
                        live_preview_mask = render_runtime_head_mask(
                            latest_obs, runtime_contract, geometry_masker
                        )
                        live_preview_request = build_policy_request(
                            latest_obs,
                            prompt,
                            round_id=active_round_id,
                            camera_order=runtime_contract.camera_order,
                            action_horizon=action_horizon,
                            fixed_head_mask=(
                                None
                                if live_preview_mask is None
                                else live_preview_mask["combined_mask"]
                            ),
                        )
                        camera_preview.show_live(
                            live_preview_request.observation,
                            f"round={active_round_id} state={scheduler.state.name} "
                            f"task={selected_task_key}",
                            geometry_q14_from_observation(latest_obs),
                        )
                    except Exception as exc:
                        message = str(exc)
                        if message != preview_error:
                            logging.exception(
                                "Processed camera preview update failed; control continues"
                            )
                            preview_error = message
                    else:
                        if preview_error is not None:
                            logging.info("Processed camera preview recovered")
                        preview_error = None

                if scheduler.state is RunState.ACTIVE:
                    try:
                        # Keep hardware health independent from RTC indexing:
                        # controller_target says which action was consumed;
                        # ActualTCPPose proves that Marvin really followed it.
                        scheduler.validate_physical_tracking(live, controller_target)
                        if author_sync:
                            scheduler.observe_controller(controller_target)
                        # Queue progress follows the setpoint actually emitted by
                        # RosTargetInterpolationController. ActualTCPPose may lag
                        # that setpoint during Mink's anchor release and must not
                        # make RTC retire/hold on the wrong action index.
                        was_left_allowed = scheduler._left_released if author_sync else None
                        completed = scheduler.advance(time.time(), controller_target)
                        if author_sync and scheduler.right_before_left and not was_left_allowed and scheduler._left_released:
                            logging.info("Gripper handoff: right opening reached %.2fdeg; left arm/gripper allowed",
                                         scheduler._right_release_goal)
                    except UnsafeChunkError as exc:
                        if args.execute:
                            hold_robot(env)
                        logging.error("Physical/controller tracking failed; HOLD: %s", exc)
                    else:
                        executed_in_round += completed
                        if waypoint_limit is not None and executed_in_round >= waypoint_limit:
                            scheduler.hold("bounded round complete")
                            if args.execute:
                                hold_robot(env)
                            logging.warning(
                                "%d physically-timed waypoints complete; HOLD. "
                                "Press [d] for a fresh round",
                                waypoint_limit,
                            )

                result = worker.poll()
                if result is not None:
                    if result.request.round_id != active_round_id:
                        worker.record_disposition(result, "stale_round", active_round=active_round_id)
                        logging.warning(
                            "Discarded stale inference from round=%d; active round=%d",
                            result.request.round_id,
                            active_round_id,
                        )
                    elif scheduler.state is RunState.HOLD:
                        worker.record_disposition(result, "discarded_while_hold")
                        logging.info("Discarded inference completed while HOLD")
                    elif result.error is not None:
                        worker.record_disposition(result, "inference_error", error=repr(result.error))
                        scheduler.mark_request_failed(f"inference failed: {result.error}")
                        if args.execute:
                            hold_robot(env)
                        logging.exception("Inference failed; entered HOLD", exc_info=result.error)
                    else:
                        before_merge = scheduler.diagnostic_snapshot()
                        merge_now = time.time()
                        try:
                            intent = preview_model_intent(result, merge_now, scheduler.config, live)
                            stats = scheduler.merge_chunk(
                                result.targets,
                                result.request.observation_time,
                                merge_now,
                                live,
                            )
                            after_merge = scheduler.diagnostic_snapshot()
                            worker.record_disposition(result, "accepted", accepted=stats.accepted)
                            try:
                                chunk_diagnostics.record(
                                    policy_request_id=result.request.request_id,
                                    policy_session_id=policy_recorder.session_id,
                                    round_id=result.request.round_id,
                                    observation_time=result.request.observation_time,
                                    merge_time=merge_now,
                                    inference_latency_s=result.latency_s,
                                    raw_targets=result.targets,
                                    live_target=live,
                                    controller_target=controller_target,
                                    tcp_bases=result.request.tcp_bases,
                                    before=before_merge,
                                    after=after_merge,
                                    stats=stats,
                                    **({"gripper_policy_targets": scheduler._queue_policy_grippers,
                                        "gripper_handoff": scheduler.gripper_handoff_status()} if author_sync else {}),
                                    **({"splice_time": scheduler.replace_from_time,
                                        "splice_anchor": scheduler._trajectory_targets[0]}
                                       if author_sync else {}),
                                )
                            except Exception:
                                logging.exception(
                                    "Failed to persist policy chunk diagnostics; control continues"
                                )
                            logging.info(
                                "chunk mode=%s round=%d obs_age=%.3fs accepted=%d expired=%d "
                                "rtc_prefix=%d infer_delay=%d unsafe_tail=%d "
                                "retimed=%d max_delay=%.3fs queue_cut=%d blend=%d "
                                "infer=%.1fms queue=%d committed=%d "
                                "intent5[first=%d R=%.1fmm/%.1fdeg L=%.1fmm/%.1fdeg "
                                "Rxyz=(%+.1f,%+.1f,%+.1f)mm Lxyz=(%+.1f,%+.1f,%+.1f)mm]",
                                args.execution_mode, result.request.round_id,
                                merge_now - result.request.observation_time,
                                stats.accepted,
                                stats.expired,
                                stats.rtc_prefix_preserved,
                                stats.inference_delay_steps,
                                stats.unsafe_truncated,
                                stats.speed_retimed,
                                stats.max_retime_delay_s,
                                stats.queue_truncated,
                                stats.handoff_blended,
                                result.latency_s * 1000,
                                scheduler.queued_steps,
                                scheduler.committed_steps,
                                intent[0],
                                intent[1] * 1000,
                                np.rad2deg(intent[2]),
                                intent[3] * 1000,
                                np.rad2deg(intent[4]),
                                *intent[5:11],
                            )
                        except UnsafeChunkError as exc:
                            worker.record_disposition(result, "rejected", error=str(exc))
                            try:
                                chunk_diagnostics.record(
                                    policy_request_id=result.request.request_id,
                                    policy_session_id=policy_recorder.session_id,
                                    round_id=result.request.round_id,
                                    observation_time=result.request.observation_time,
                                    merge_time=merge_now,
                                    inference_latency_s=result.latency_s,
                                    raw_targets=result.targets,
                                    live_target=live,
                                    controller_target=controller_target,
                                    tcp_bases=result.request.tcp_bases,
                                    before=before_merge,
                                    after=scheduler.diagnostic_snapshot(),
                                    stats=None,
                                    error=str(exc),
                                )
                            except Exception:
                                logging.exception(
                                    "Failed to persist rejected chunk diagnostics; control continues"
                                )
                            if args.execute:
                                hold_robot(env)
                            logging.error(
                                "Chunk rejected; HOLD: %s (round=%d obs_age=%.3fs)",
                                exc,
                                result.request.round_id,
                                merge_now - result.request.observation_time,
                            )

                if scheduler.state is RunState.ACTIVE:
                    if (scheduler.request_due(time.time()) and
                            (waypoint_limit is None or executed_in_round + scheduler.committed_steps < waypoint_limit)):
                        submit_rtc_request(
                            worker,
                            scheduler,
                            latest_obs,
                            prompt,
                            active_round_id,
                            action_horizon,
                            camera_preview,
                            runtime_contract,
                            geometry_masker,
                        )
                    try:
                        remaining = (None if waypoint_limit is None else
                                     waypoint_limit - executed_in_round - scheduler.committed_steps)
                        scheduled_batch = ([] if remaining is not None and remaining <= 0 else
                                           scheduler.pop_batch(live, now=time.time(), max_actions=remaining))
                    except UnsafeChunkError as exc:
                        if args.execute:
                            hold_robot(env)
                        logging.error("Live target rejected; HOLD: %s", exc)
                        scheduled_batch = []
                    else:
                        # pop_batch() fail-closes the software scheduler when an
                        # executed queue runs dry. Mirror that transition to
                        # the physical Mink FSM; otherwise the robot remains
                        # ACTIVE indefinitely on its final latched target.
                        if not scheduled_batch and scheduler.state is RunState.HOLD:
                            if args.execute:
                                hold_robot(env)
                            logging.warning("Action queue exhausted; physical HOLD: %s", scheduler.reason)
                    if scheduled_batch:
                        if args.execute:
                            env.exec_actions(
                                actions=np.stack(
                                    [scheduled.target for scheduled in scheduled_batch]
                                ),
                                timestamps=np.asarray(
                                    [scheduled.target_time for scheduled in scheduled_batch],
                                    dtype=np.float64,
                                ),
                                compensate_latency=author_sync,
                                time_is_new=True,
                                **({"gripper_policy_angles": np.stack([a.gripper_policy_target for a in scheduled_batch])}
                                   if author_sync and scheduler.compensation_enabled else {}),
                                **({"replace_from_time": scheduler.replace_from_time}
                                   if author_sync else {}),
                            )

                if (scheduler.request_due(time.time()) and
                        (waypoint_limit is None or executed_in_round + scheduler.committed_steps < waypoint_limit)):
                    submit_rtc_request(
                        worker,
                        scheduler,
                        latest_obs,
                        prompt,
                        active_round_id,
                        action_horizon,
                        camera_preview,
                        runtime_contract,
                        geometry_masker,
                    )

                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    logging.warning("Control tick overrun %.1fms", -delay * 1000)
                    next_tick = time.monotonic()

            scheduler.hold("Ctrl+C")
            if args.execute:
                hold_robot(env)
            logging.warning("Ctrl+C: commanded HOLD, shutting down")
            signal.signal(signal.SIGINT, previous_sigint)

    # Keep the descriptor alive for the full UmiEnv/controller lifetime.
    client_lock.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted; UmiEnv shutdown will stop the RTDE controllers")
