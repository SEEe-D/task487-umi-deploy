"""Collect five static Marvin joint poses using small, low-speed motions.

The sequence is deliberately independent from policy/model output.  It checks
controller state, joint limits and FK displacement before enabling either arm,
then returns to the exact measured start pose and disables both arms.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time

import numpy as np


WRAPPER_ROOT = Path("/home/simpleai/Code/eval_benchmark/Marvin/marvin_wrapper")
DEFAULT_CONFIG = WRAPPER_ROOT / "configs" / "ccs_m6_31.yaml"
EXECUTE_CONFIRMATION = "ENABLE_CALIBRATION_POSE_SEQUENCE"


def state_dict(snapshot) -> dict:
    return {
        "cur_state": int(snapshot.cur_state),
        "err_code": int(snapshot.err_code),
        "joints_deg": [float(value) for value in snapshot.joints],
        "joint_vel_deg_s": [float(value) for value in snapshot.joint_vel],
    }


def assert_clear(label: str, snapshot, expected_state: int | None = None) -> None:
    if int(snapshot.err_code) != 0:
        raise RuntimeError(f"{label} err_code={snapshot.err_code}")
    if expected_state is not None and int(snapshot.cur_state) != expected_state:
        raise RuntimeError(
            f"{label} cur_state={snapshot.cur_state}, expected {expected_state}"
        )


def add(start: list[float], offsets: list[float]) -> list[float]:
    return [float(a + b) for a, b in zip(start, offsets)]


def pose_record(index: int, label: str, left, right) -> dict:
    left_state = state_dict(left)
    right_state = state_dict(right)
    assert_clear("left/A", left)
    assert_clear("right/B", right)
    q14_rad = np.radians(
        np.asarray(right_state["joints_deg"] + left_state["joints_deg"], dtype=np.float64)
    )
    record = {
        "pose_index": index,
        "label": label,
        "capture_time_ns": time.time_ns(),
        "q14_rad": q14_rad.tolist(),
        "left": left_state,
        "right": right_state,
    }
    logging.info(
        "POSE_READY index=%d label=%s qL1=%.3f qR1=%.3f",
        index,
        label,
        left_state["joints_deg"][0],
        right_state["joints_deg"][0],
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=1.2)
    parser.add_argument("--speed-ratio", type=int, default=10)
    parser.add_argument("--max-joint-offset-deg", type=float, default=2.0)
    parser.add_argument("--max-tcp-displacement-mm", type=float, default=45.0)
    parser.add_argument("--execute-confirmation", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.execute_confirmation != EXECUTE_CONFIRMATION:
        raise RuntimeError(f"execution requires --execute-confirmation {EXECUTE_CONFIRMATION}")
    if not 4.0 <= args.duration_s <= 12.0:
        raise ValueError("duration-s must be within [4, 12]")
    if not 5 <= args.speed_ratio <= 15:
        raise ValueError("speed-ratio must be within [5, 15]")

    wrapper_root = str(WRAPPER_ROOT.resolve())
    if wrapper_root not in sys.path:
        sys.path.insert(0, wrapper_root)
    from marvin_wrapper import MarvinRobot
    from marvin_wrapper.config import load_config

    config = load_config(args.config)
    config.connection.auto_clear_on_connect = False
    config.safety.auto_recover = False
    config.safety.estop_on_servo_error = True
    robot = MarvinRobot(config)
    enabled = False
    output = {"success": False, "poses": [], "returned_to_start": False}
    try:
        robot.connect()
        if robot.dual is None:
            raise RuntimeError("dual-arm Marvin config is required")
        start_left, start_right = robot.dual.snapshot()
        assert_clear("left/A", start_left, 0)
        assert_clear("right/B", start_right, 0)
        ql0 = [float(value) for value in start_left.joints]
        qr0 = [float(value) for value in start_right.joints]
        output["start"] = {"left": state_dict(start_left), "right": state_dict(start_right)}

        # Local lower-arm/wrist perturbations.  At most 2 degrees per joint.
        left_plus = [0.0, 1.5, 0.0, -2.0, 0.0, 2.0, 0.0]
        left_minus = [0.0, -1.5, 0.0, 2.0, 0.0, -2.0, 0.0]
        right_plus = [0.0, 1.5, 0.0, 2.0, 0.0, 2.0, 0.0]
        right_minus = [0.0, -1.5, 0.0, -2.0, 0.0, -2.0, 0.0]
        targets = [
            ("start", ql0, qr0),
            ("left_plus", add(ql0, left_plus), qr0),
            ("left_minus", add(ql0, left_minus), qr0),
            ("right_plus", ql0, add(qr0, right_plus)),
            ("right_minus", ql0, add(qr0, right_minus)),
        ]

        start_pose_left = np.asarray(robot.left.cart.fk(ql0)[:3], dtype=np.float64)
        start_pose_right = np.asarray(robot.right.cart.fk(qr0)[:3], dtype=np.float64)
        plans = []
        for label, ql, qr in targets:
            for arm, q, q0, arm_label in (
                (robot.left, ql, ql0, "left/A"),
                (robot.right, qr, qr0, "right/B"),
            ):
                if not all(math.isfinite(value) for value in q):
                    raise RuntimeError(f"{label} {arm_label} target is non-finite")
                max_offset = max(abs(value - start) for value, start in zip(q, q0))
                if max_offset > args.max_joint_offset_deg + 1e-9:
                    raise RuntimeError(f"{label} {arm_label} offset {max_offset:.3f}deg exceeds limit")
                violation = arm.check_joint_limits(q)
                if violation is not None:
                    raise RuntimeError(f"{label} {arm_label} joint limit: {violation[1]}")
            left_disp = float(
                np.linalg.norm(np.asarray(robot.left.cart.fk(ql)[:3]) - start_pose_left)
            )
            right_disp = float(
                np.linalg.norm(np.asarray(robot.right.cart.fk(qr)[:3]) - start_pose_right)
            )
            if max(left_disp, right_disp) > args.max_tcp_displacement_mm:
                raise RuntimeError(
                    f"{label} TCP displacement {max(left_disp, right_disp):.2f}mm exceeds limit"
                )
            plans.append(
                {"label": label, "left_tcp_displacement_mm": left_disp, "right_tcp_displacement_mm": right_disp}
            )
        output["plans"] = plans
        logging.info("all five pose plans passed joint-limit/FK checks: %s", plans)

        # Pose zero is the measured disabled start pose.
        time.sleep(args.settle_s)
        output["poses"].append(pose_record(0, "start", *robot.dual.snapshot()))

        robot.dual.enable()
        enabled = True
        robot.dual.set_speed(args.speed_ratio, args.speed_ratio)
        time.sleep(0.4)
        assert_clear("left/A", robot.left.snapshot(), 1)
        assert_clear("right/B", robot.right.snapshot(), 1)

        for pose_index, (label, ql, qr) in enumerate(targets[1:], start=1):
            logging.info("moving to pose %d/%d: %s", pose_index + 1, len(targets), label)
            robot.dual.joint.move_in(
                ql, qr, duration_s=args.duration_s, profile="cubic", rate_hz=100, blocking=True
            )
            time.sleep(args.settle_s)
            left, right = robot.dual.snapshot()
            max_error = max(
                max(abs(a - b) for a, b in zip(left.joints, ql)),
                max(abs(a - b) for a, b in zip(right.joints, qr)),
            )
            if max_error > 0.5:
                raise RuntimeError(f"pose {label} max joint error {max_error:.3f}deg exceeds 0.5deg")
            output["poses"].append(pose_record(pose_index, label, left, right))

        logging.info("returning both arms to exact measured start pose")
        robot.dual.joint.move_in(
            ql0, qr0, duration_s=args.duration_s, profile="cubic", rate_hz=100, blocking=True
        )
        time.sleep(args.settle_s)
        returned_left, returned_right = robot.dual.snapshot()
        return_error = max(
            max(abs(a - b) for a, b in zip(returned_left.joints, ql0)),
            max(abs(a - b) for a, b in zip(returned_right.joints, qr0)),
        )
        if return_error > 0.5:
            raise RuntimeError(f"return max joint error {return_error:.3f}deg exceeds 0.5deg")
        output["return_max_joint_error_deg"] = return_error
        output["returned_to_start"] = True
        robot.dual.disable()
        enabled = False
        output["success"] = True
        return 0
    except KeyboardInterrupt:
        output["error"] = "operator interrupt"
        if robot.is_alive():
            robot.estop("AB")
        return 130
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("calibration pose sequence failed")
        if robot.is_alive() and enabled:
            try:
                robot.estop("AB")
            except Exception:
                logging.exception("soft estop failed")
        return 1
    finally:
        if robot.is_alive():
            if enabled:
                try:
                    robot.dual.disable()
                except Exception:
                    logging.exception("dual disable failed")
            try:
                output["final"] = {
                    "left": state_dict(robot.left.snapshot()),
                    "right": state_dict(robot.right.snapshot()),
                }
            except Exception as exc:
                output["final_state_error"] = f"{type(exc).__name__}: {exc}"
            robot.disconnect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logging.info("pose record saved: %s", args.output)


if __name__ == "__main__":
    raise SystemExit(main())
