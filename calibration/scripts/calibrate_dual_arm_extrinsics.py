from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.dual_arm_projection.configuration import (
    current_camera_from_base,
    load_initial_camera_from_base,
    load_processed_fisheye_camera,
    save_adjusted_camera_from_base,
)
from tools.dual_arm_projection.dataset import LeRobotProjectionDataset
from tools.dual_arm_projection.rasterizer import FisheyeArmRasterizer
from tools.dual_arm_projection.urdf_model import load_arm_model


PARAMETER_NAMES = [
    "L tx",
    "L ty",
    "L tz",
    "L roll",
    "L pitch",
    "L yaw",
    "R tx",
    "R ty",
    "R tz",
    "R roll",
    "R pitch",
    "R yaw",
]
TRANSLATION_STEP_M = 0.001
TRANSLATION_CENTER = 400
TRANSLATION_MAXIMUM = 800
ROTATION_STEP_DEG = 0.2
ROTATION_CENTER = 450
ROTATION_MAXIMUM = 900
# Start with the most diagnostic view: every URDF Link has a distinct color
# and every revolute-joint origin is explicitly marked.  The older arm-level
# outline/fill modes remain available through the M key.
DISPLAY_MODES = ("joint_links", "outline", "fill_outline", "fill")
JOINT_LINK_VIEWS = (
    [("ALL", None)]
    + [(f"L{index}", index + 1) for index in range(1, 8)]
    + [(f"R{index}", 100 + index + 1) for index in range(1, 8)]
)
JOINT_LINK_FILL_ALPHAS = (0.0, 0.18, 0.35, 0.60)
CONFIRM_BUTTON_RECT = (522, 112, 662, 144)


def write_image_unicode(path: Path, image_bgr: np.ndarray) -> None:
    """Write an OpenCV image on Windows even when the path contains Unicode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image_bgr)
    if not success:
        raise RuntimeError(f"failed to encode preview: {path}")
    try:
        encoded.tofile(path)
    except OSError as error:
        raise RuntimeError(f"failed to write preview: {path}") from error


def parse_vector3(value: str) -> np.ndarray:
    values = [float(part.strip()) for part in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return np.asarray(values, dtype=np.float64)


def put_lines(
    image_bgr: np.ndarray,
    lines: list[str],
    origin_xy: tuple[int, int],
    color_bgr: tuple[int, int, int] = (235, 235, 235),
) -> None:
    x, y = origin_xy
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
        y += 18


def make_mosaic(images: list[np.ndarray], columns: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("cannot make an empty mosaic")
    height, width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    mosaic = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        mosaic[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return mosaic


def link_color_bgr(link_id: int) -> tuple[int, int, int]:
    sample = np.asarray([[int(link_id)]], dtype=np.uint16)
    colorized = FisheyeArmRasterizer.colorize_link_ids(sample)
    return tuple(int(value) for value in colorized[0, 0])


def draw_projected_joint_origins(
    image_bgr: np.ndarray,
    rasterizer: FisheyeArmRasterizer,
    T_camera_left_base: np.ndarray,
    T_camera_right_base: np.ndarray,
    left_joint_angles_rad: dict[str, float],
    right_joint_angles_rad: dict[str, float],
    selected_link_id: int | None = None,
) -> None:
    """Draw L1..L7/R1..R7 at the corresponding URDF joint origins."""

    height, width = image_bgr.shape[:2]
    for arm, T_camera_base, joint_angles_rad in (
        (rasterizer.left_arm, T_camera_left_base, left_joint_angles_rad),
        (rasterizer.right_arm, T_camera_right_base, right_joint_angles_rad),
    ):
        transforms = arm.forward_kinematics(joint_angles_rad)
        for joint_index, joint in enumerate(arm.chain_joints, start=1):
            point_base = transforms[joint.child_link][:3, 3]
            point_camera = (
                T_camera_base
                @ np.asarray([point_base[0], point_base[1], point_base[2], 1.0])
            )[:3]
            if point_camera[2] <= rasterizer.near_plane_m:
                continue
            projected_uv, _ = cv2.fisheye.projectPoints(
                point_camera.reshape(1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                rasterizer.K_camera_image,
                rasterizer.D_camera_fisheye,
            )
            u, v = projected_uv.reshape(2)
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < width and 0 <= y < height):
                continue
            link_id = joint_index + 1 + (0 if arm.side == "L" else 100)
            if selected_link_id is not None and link_id != selected_link_id:
                continue
            color = link_color_bgr(link_id)
            cv2.circle(image_bgr, (x, y), 5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(image_bgr, (x, y), 4, color, -1, cv2.LINE_AA)
            label = f"{arm.side}{joint_index}"
            label_origin = (min(width - 18, x + 5), max(9, y - 4))
            cv2.putText(
                image_bgr,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                color,
                1,
                cv2.LINE_AA,
            )


def slider_state(
    slider_positions: list[int],
    T_camera_left_base_initial: np.ndarray,
    T_camera_right_base_initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_translation_m = T_camera_left_base_initial[:3, 3].copy()
    right_translation_m = T_camera_right_base_initial[:3, 3].copy()
    for axis in range(3):
        left_translation_m[axis] += (
            slider_positions[axis] - TRANSLATION_CENTER
        ) * TRANSLATION_STEP_M
        right_translation_m[axis] += (
            slider_positions[6 + axis] - TRANSLATION_CENTER
        ) * TRANSLATION_STEP_M
    left_adjustment_rpy_rad = np.radians(
        [
            (slider_positions[3 + axis] - ROTATION_CENTER) * ROTATION_STEP_DEG
            for axis in range(3)
        ]
    )
    right_adjustment_rpy_rad = np.radians(
        [
            (slider_positions[9 + axis] - ROTATION_CENTER) * ROTATION_STEP_DEG
            for axis in range(3)
        ]
    )
    T_camera_left_base = current_camera_from_base(
        T_camera_left_base_initial,
        left_translation_m,
        left_adjustment_rpy_rad,
    )
    T_camera_right_base = current_camera_from_base(
        T_camera_right_base_initial,
        right_translation_m,
        right_adjustment_rpy_rad,
    )
    return (
        T_camera_left_base,
        T_camera_right_base,
        left_translation_m,
        right_translation_m,
        left_adjustment_rpy_rad,
        right_adjustment_rpy_rad,
    )


def render_calibration_view(
    rasterizer: FisheyeArmRasterizer,
    frames,
    T_camera_left_base: np.ndarray,
    T_camera_right_base: np.ndarray,
    T_camera_left_base_initial: np.ndarray,
    T_camera_right_base_initial: np.ndarray,
    left_adjustment_rpy_rad: np.ndarray,
    right_adjustment_rpy_rad: np.ndarray,
    selected_parameter: int,
    display_mode: str,
    selected_joint_link: tuple[str, int | None],
    joint_link_fill_alpha: float,
    confirmation_progress: tuple[int, int] | None = None,
) -> np.ndarray:
    rendered_frames = []
    active_frame_position = None
    for frame_position, frame in enumerate(frames):
        result = rasterizer.render(
            frame.image_bgr,
            T_camera_left_base,
            T_camera_right_base,
            frame.left_joint_angles_rad,
            frame.right_joint_angles_rad,
        )
        if display_mode == "joint_links":
            overlay = frame.image_bgr.copy()
            _, selected_link_id = selected_joint_link
            foreground = (
                result.link_id_image > 0
                if selected_link_id is None
                else result.link_id_image == selected_link_id
            )
            if joint_link_fill_alpha > 0.0:
                overlay[foreground] = cv2.addWeighted(
                    frame.image_bgr[foreground],
                    1.0 - joint_link_fill_alpha,
                    result.link_id_color_bgr[foreground],
                    joint_link_fill_alpha,
                    0.0,
                )
            for link_id in np.unique(result.link_id_image[foreground]):
                if link_id == 0:
                    continue
                link_mask = (
                    (result.link_id_image == link_id).astype(np.uint8) * 255
                )
                contours, _ = cv2.findContours(
                    link_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    link_color_bgr(int(link_id)),
                    2,
                    cv2.LINE_AA,
                )
            draw_projected_joint_origins(
                overlay,
                rasterizer,
                T_camera_left_base,
                T_camera_right_base,
                frame.left_joint_angles_rad,
                frame.right_joint_angles_rad,
                selected_link_id,
            )
        elif display_mode == "fill":
            overlay = result.overlay_bgr.copy()
        else:
            overlay = (
                result.overlay_bgr.copy()
                if display_mode == "fill_outline"
                else frame.image_bgr.copy()
            )
            for mask, color_bgr in (
                (result.left_mask, (0, 255, 255)),
                (result.right_mask, (255, 255, 0)),
            ):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color_bgr, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"ep={frame.episode_index} frame={frame.relative_frame_index}",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if active_frame_position == frame_position:
            cv2.rectangle(
                overlay,
                (1, 1),
                (overlay.shape[1] - 2, overlay.shape[0] - 2),
                (80, 255, 80),
                3,
            )
        rendered_frames.append(overlay)
    mosaic_columns = len(rendered_frames) if confirmation_progress is not None else 3
    mosaic = make_mosaic(rendered_frames, columns=mosaic_columns)
    panel_height = 170 if confirmation_progress is not None else 150
    panel = np.full((panel_height, mosaic.shape[1], 3), 24, dtype=np.uint8)
    left_initial = T_camera_left_base_initial[:3, 3]
    right_initial = T_camera_right_base_initial[:3, 3]
    left_current = T_camera_left_base[:3, 3]
    right_current = T_camera_right_base[:3, 3]
    left_lines = [
        "LEFT camera_from_base",
        f"initial t m: {left_initial[0]:+.3f} {left_initial[1]:+.3f} {left_initial[2]:+.3f}",
        f"current t m: {left_current[0]:+.3f} {left_current[1]:+.3f} {left_current[2]:+.3f}",
        f"delta t mm: {(left_current[0]-left_initial[0])*1000:+.0f} {(left_current[1]-left_initial[1])*1000:+.0f} {(left_current[2]-left_initial[2])*1000:+.0f}",
        "rotation correction deg: " + " ".join(f"{v:+.1f}" for v in np.degrees(left_adjustment_rpy_rad)),
    ]
    right_lines = [
        "RIGHT camera_from_base",
        f"initial t m: {right_initial[0]:+.3f} {right_initial[1]:+.3f} {right_initial[2]:+.3f}",
        f"current t m: {right_current[0]:+.3f} {right_current[1]:+.3f} {right_current[2]:+.3f}",
        f"delta t mm: {(right_current[0]-right_initial[0])*1000:+.0f} {(right_current[1]-right_initial[1])*1000:+.0f} {(right_current[2]-right_initial[2])*1000:+.0f}",
        "rotation correction deg: " + " ".join(f"{v:+.1f}" for v in np.degrees(right_adjustment_rpy_rad)),
    ]
    put_lines(panel, left_lines, (8, 18), (80, 230, 255))
    put_lines(panel, right_lines, (mosaic.shape[1] // 2, 18), (255, 180, 80))
    put_lines(
        panel,
        [
            f"selected: {PARAMETER_NAMES[selected_parameter]} | view: {display_mode}",
            (
                f"joint Link: {selected_joint_link[0]} | [ / ] choose | a all | "
                f"t fill {joint_link_fill_alpha * 100:.0f}%"
            ),
            (
                "click CONFIRM / Enter | m view | h/l fine | u/o coarse | q quit"
                if confirmation_progress is not None
                else "m view | j/k select | h/l fine | u/o coarse | r reset | s save | q quit"
            ),
        ],
        (8, 112),
        (230, 230, 230),
    )
    if confirmation_progress is not None:
        _, total_count = confirmation_progress
        x0, y0, x1, y1 = CONFIRM_BUTTON_RECT
        cv2.rectangle(panel, (x0, y0), (x1, y1), (40, 150, 70), -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (100, 240, 130), 1)
        cv2.putText(
            panel,
            f"SAVE ALL {total_count}",
            (x0 + 9, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, mosaic])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--camera-config", type=Path, default=Path("configs/camera_head_main.json")
    )
    parser.add_argument(
        "--initial-yaml",
        type=Path,
        default=Path("configs/dual_arm_extrinsics_initial.yaml"),
    )
    parser.add_argument(
        "--save-yaml",
        type=Path,
        default=Path("outputs/dual_arm_extrinsic_calibration/adjusted_extrinsics.yaml"),
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=Path("outputs/dual_arm_extrinsic_calibration/multi_frame_preview.png"),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--sequential-confirm", action="store_true")
    parser.add_argument("--max-faces-per-link", type=int, default=600)
    parser.add_argument(
        "--left-translation-offset-m", type=parse_vector3, default=np.zeros(3)
    )
    parser.add_argument(
        "--right-translation-offset-m", type=parse_vector3, default=np.zeros(3)
    )
    parser.add_argument(
        "--left-rotation-adjustment-deg", type=parse_vector3, default=np.zeros(3)
    )
    parser.add_argument(
        "--right-rotation-adjustment-deg", type=parse_vector3, default=np.zeros(3)
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    _, K_camera_image, D_camera_fisheye, image_size_wh = (
        load_processed_fisheye_camera(args.camera_config)
    )
    initial_config, T_camera_left_base_initial, T_camera_right_base_initial = (
        load_initial_camera_from_base(args.initial_yaml)
    )
    left_arm = load_arm_model(args.urdf.resolve(), "L", args.max_faces_per_link)
    right_arm = load_arm_model(args.urdf.resolve(), "R", args.max_faces_per_link)
    rasterizer = FisheyeArmRasterizer(
        K_camera_image,
        D_camera_fisheye,
        image_size_wh,
        left_arm,
        right_arm,
    )
    dataset = LeRobotProjectionDataset(args.dataset_root)
    frame_indices = dataset.evenly_spaced_indices(args.episode_index, args.frame_count)
    frames = dataset.read_frames(args.episode_index, frame_indices)

    slider_positions = [
        TRANSLATION_CENTER,
        TRANSLATION_CENTER,
        TRANSLATION_CENTER,
        ROTATION_CENTER,
        ROTATION_CENTER,
        ROTATION_CENTER,
    ] * 2
    translation_limit_m = TRANSLATION_CENTER * TRANSLATION_STEP_M
    rotation_limit_deg = ROTATION_CENTER * ROTATION_STEP_DEG
    for side, translation_offset_m, rotation_adjustment_deg in (
        ("left", args.left_translation_offset_m, args.left_rotation_adjustment_deg),
        ("right", args.right_translation_offset_m, args.right_rotation_adjustment_deg),
    ):
        if np.any(np.abs(translation_offset_m) > translation_limit_m + 1e-12):
            raise ValueError(
                f"{side} translation offset exceeds +/-{translation_limit_m:.3f} m: "
                f"{translation_offset_m.tolist()}"
            )
        if np.any(np.abs(rotation_adjustment_deg) > rotation_limit_deg + 1e-12):
            raise ValueError(
                f"{side} rotation adjustment exceeds +/-{rotation_limit_deg:.1f} deg: "
                f"{rotation_adjustment_deg.tolist()}"
            )
    for axis in range(3):
        slider_positions[axis] = int(
            np.clip(
                TRANSLATION_CENTER
                + round(args.left_translation_offset_m[axis] / TRANSLATION_STEP_M),
                0,
                TRANSLATION_MAXIMUM,
            )
        )
        slider_positions[6 + axis] = int(
            np.clip(
                TRANSLATION_CENTER
                + round(args.right_translation_offset_m[axis] / TRANSLATION_STEP_M),
                0,
                TRANSLATION_MAXIMUM,
            )
        )
        slider_positions[3 + axis] = int(
            np.clip(
                ROTATION_CENTER
                + round(args.left_rotation_adjustment_deg[axis] / ROTATION_STEP_DEG),
                0,
                ROTATION_MAXIMUM,
            )
        )
        slider_positions[9 + axis] = int(
            np.clip(
                ROTATION_CENTER
                + round(args.right_rotation_adjustment_deg[axis] / ROTATION_STEP_DEG),
                0,
                ROTATION_MAXIMUM,
            )
        )
    startup_slider_positions = tuple(slider_positions)
    selected_parameter = 0
    display_mode_index = 0
    joint_link_view_index = 0
    joint_link_fill_alpha_index = 0
    def build_view() -> tuple[np.ndarray, tuple]:
        state = slider_state(
            slider_positions,
            T_camera_left_base_initial,
            T_camera_right_base_initial,
        )
        confirmation_progress = (
            (1, len(frames))
            if args.sequential_confirm
            else None
        )
        view = render_calibration_view(
            rasterizer,
            frames,
            state[0],
            state[1],
            T_camera_left_base_initial,
            T_camera_right_base_initial,
            state[4],
            state[5],
            selected_parameter,
            DISPLAY_MODES[display_mode_index],
            JOINT_LINK_VIEWS[joint_link_view_index],
            JOINT_LINK_FILL_ALPHAS[joint_link_fill_alpha_index],
            confirmation_progress,
        )
        return view, state

    if args.headless:
        view, state = build_view()
        write_image_unicode(args.preview_output, view)
        save_adjusted_camera_from_base(
            args.save_yaml,
            args.initial_yaml,
            initial_config,
            state[0],
            state[1],
            state[4],
            state[5],
        )
        print(f"preview={args.preview_output.resolve()}")
        print(f"saved_yaml={args.save_yaml.resolve()}")
        return

    image_window_name = "Dual-arm multi-frame calibration"
    controls_window_name = "Extrinsic controls"
    cv2.namedWindow(image_window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(controls_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_window_name, 720, 520)
    slider_specs = [
        ("L tx mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("L ty mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("L tz mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("L roll x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
        ("L pitch x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
        ("L yaw x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
        ("R tx mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("R ty mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("R tz mm", TRANSLATION_MAXIMUM, TRANSLATION_CENTER),
        ("R roll x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
        ("R pitch x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
        ("R yaw x0.2deg", ROTATION_MAXIMUM, ROTATION_CENTER),
    ]
    for index, (name, maximum, _) in enumerate(slider_specs):
        cv2.createTrackbar(
            name,
            controls_window_name,
            slider_positions[index],
            maximum,
            lambda _: None,
        )
    controls_canvas = np.full((80, 720, 3), 30, dtype=np.uint8)
    cv2.putText(
        controls_canvas,
        "Adjust sliders here; image window keeps native aspect ratio.",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(controls_window_name, controls_canvas)

    confirm_requested = False

    def on_image_mouse(event, x, y, _flags, _userdata) -> None:
        nonlocal confirm_requested
        if event != cv2.EVENT_LBUTTONUP or not args.sequential_confirm:
            return
        x0, y0, x1, y1 = CONFIRM_BUTTON_RECT
        if x0 <= x <= x1 and y0 <= y <= y1:
            confirm_requested = True

    cv2.setMouseCallback(image_window_name, on_image_mouse)

    def save_interactive_snapshot(view: np.ndarray, state: tuple) -> dict:
        save_adjusted_camera_from_base(
            args.save_yaml,
            args.initial_yaml,
            initial_config,
            state[0],
            state[1],
            state[4],
            state[5],
        )
        write_image_unicode(args.preview_output, view)

        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        timestamp_for_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        history_dir = args.save_yaml.parent / f"{args.save_yaml.stem}_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_yaml = history_dir / f"extrinsics_{timestamp_for_name}.yaml"
        history_preview = history_dir / f"preview_{timestamp_for_name}.png"
        save_adjusted_camera_from_base(
            history_yaml,
            args.initial_yaml,
            initial_config,
            state[0],
            state[1],
            state[4],
            state[5],
        )
        write_image_unicode(history_preview, view)

        event = {
            "saved_at": timestamp,
            "episode_index": args.episode_index,
            "confirmation_number": 1,
            "confirmed_relative_frame_indices": frame_indices,
            "context_relative_frame_indices": frame_indices,
            "T_camera_left_base": state[0].tolist(),
            "T_camera_right_base": state[1].tolist(),
            "left_translation_m": state[2].tolist(),
            "right_translation_m": state[3].tolist(),
            "left_rotation_adjustment_rpy_rad": state[4].tolist(),
            "right_rotation_adjustment_rpy_rad": state[5].tolist(),
            "current_yaml": str(args.save_yaml.resolve()),
            "current_preview": str(args.preview_output.resolve()),
            "history_yaml": str(history_yaml.resolve()),
            "history_preview": str(history_preview.resolve()),
        }
        history_jsonl = history_dir / "calibration_history.jsonl"
        with history_jsonl.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"saved_yaml={args.save_yaml.resolve()}")
        print(f"saved_preview={args.preview_output.resolve()}")
        print(f"history_yaml={history_yaml.resolve()}")
        print(f"history_preview={history_preview.resolve()}")
        print(f"history_jsonl={history_jsonl.resolve()}")
        return event

    previous_positions = None
    state = None
    while True:
        slider_positions[:] = [
            cv2.getTrackbarPos(name, controls_window_name)
            for name, _, _ in slider_specs
        ]
        current_positions = tuple(slider_positions)
        if current_positions != previous_positions:
            view, state = build_view()
            cv2.imshow(image_window_name, view)
            previous_positions = current_positions
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            display_mode_index = (display_mode_index + 1) % len(DISPLAY_MODES)
            previous_positions = None
        if key == ord("["):
            joint_link_view_index = (joint_link_view_index - 1) % len(JOINT_LINK_VIEWS)
            previous_positions = None
        elif key == ord("]"):
            joint_link_view_index = (joint_link_view_index + 1) % len(JOINT_LINK_VIEWS)
            previous_positions = None
        elif key == ord("a"):
            joint_link_view_index = 0
            previous_positions = None
        elif key == ord("t"):
            joint_link_fill_alpha_index = (
                joint_link_fill_alpha_index + 1
            ) % len(JOINT_LINK_FILL_ALPHAS)
            previous_positions = None
        if key == ord("j"):
            selected_parameter = (selected_parameter - 1) % len(slider_specs)
            previous_positions = None
        elif key == ord("k"):
            selected_parameter = (selected_parameter + 1) % len(slider_specs)
            previous_positions = None
        elif key in {ord("h"), ord("l"), ord("u"), ord("o")}:
            direction = -1 if key in {ord("h"), ord("u")} else 1
            coarse = key in {ord("u"), ord("o")}
            step = 10 if coarse else 1
            name, maximum, _ = slider_specs[selected_parameter]
            next_value = int(
                np.clip(slider_positions[selected_parameter] + direction * step, 0, maximum)
            )
            cv2.setTrackbarPos(name, controls_window_name, next_value)
        elif key == ord("r"):
            for index, (name, _, _) in enumerate(slider_specs):
                cv2.setTrackbarPos(
                    name, controls_window_name, startup_slider_positions[index]
                )
        elif key == ord("s") and state is not None:
            confirm_requested = args.sequential_confirm
            if not args.sequential_confirm:
                save_interactive_snapshot(view, state)

        if (
            args.sequential_confirm
            and state is not None
            and (confirm_requested or key in {10, 13, ord("c")})
        ):
            confirm_requested = False
            record = save_interactive_snapshot(view, state)
            history_dir = args.save_yaml.parent / f"{args.save_yaml.stem}_history"
            summary_path = history_dir / "confirmation_summary.json"
            summary = {
                "status": "complete",
                "confirmation_count": 1,
                "frames_used": len(frames),
                "episode_index": args.episode_index,
                "records": [record],
                "final_yaml": str(args.save_yaml.resolve()),
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"confirmation_summary={summary_path.resolve()}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
