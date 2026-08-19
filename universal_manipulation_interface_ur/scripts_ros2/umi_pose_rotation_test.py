#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂末端旋转测试脚本（ROS2 rclpy）：

功能：
- 订阅 /left_current_pose 与 /right_current_pose，等待收到后记录当前位姿到 JSON 文件
- 保持平移不变，仅修改左右末端的旋转（四元数）
- 发布 /left_target 与 /right_target（geometry_msgs/Pose）**多帧**（默认 10Hz），用于观察双臂末端运动

默认旋转模式（absolute）：
  q_target = q_rot
也就是直接将目标姿态设为你给定的四元数（xyzw）。平移保持当前不变。

示例：
  # 左臂绕 Z 轴旋转 +90deg，右臂保持不变，10Hz 发布 3 秒
  python3 umi_pose_rotation_test.py --left-rot-xyzw 0 0 0.7071068 0.7071068 --publish-secs 3 --publish-rate 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Pose


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f"invalid quaternion norm: {n}")
    return q / n


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product, both in xyzw convention.
    """
    x1, y1, z1, w1 = map(float, np.asarray(q1, dtype=np.float64).reshape(4))
    x2, y2, z2, w2 = map(float, np.asarray(q2, dtype=np.float64).reshape(4))
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return _quat_normalize_xyzw(np.array([x, y, z, w], dtype=np.float64))


def _pose_stamped_to_xyzw(p: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    pos = p.pose.position
    ori = p.pose.orientation
    xyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
    q = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float64)
    return xyz, _quat_normalize_xyzw(q)


def _make_pose(xyz: np.ndarray, q_xyzw: np.ndarray) -> Pose:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    q = _quat_normalize_xyzw(q_xyzw)
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = map(float, xyz)
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = map(float, q)
    return msg


@dataclass
class Snapshot:
    stamp_unix_sec: float
    left_current: dict
    right_current: dict
    left_target: dict
    right_target: dict
    mode: str
    left_rot_xyzw: list
    right_rot_xyzw: list


class UmiPoseRotationTestNode(Node):
    def __init__(
        self,
        *,
        mode: str,
        left_rot_xyzw: np.ndarray,
        right_rot_xyzw: np.ndarray,
        out_path: str,
        publish_rate: float,
        publish_secs: float,
        wait_timeout_sec: float,
    ) -> None:
        super().__init__("umi_pose_rotation_test")

        self._mode = str(mode)
        if self._mode not in ("relative", "absolute"):
            raise ValueError("mode must be relative|absolute")
        self._left_rot_xyzw = _quat_normalize_xyzw(left_rot_xyzw)
        self._right_rot_xyzw = _quat_normalize_xyzw(right_rot_xyzw)
        self._out_path = str(out_path)
        self._publish_rate = float(publish_rate)
        self._publish_secs = float(publish_secs)
        self._wait_timeout_sec = float(wait_timeout_sec)
        if self._publish_rate <= 0:
            raise ValueError("publish_rate must be > 0")
        if self._publish_secs <= 0:
            raise ValueError("publish_secs must be > 0")
        if self._wait_timeout_sec <= 0:
            raise ValueError("wait_timeout_sec must be > 0")

        pose_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._left_msg: Optional[PoseStamped] = None
        self._right_msg: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, "/left_current_pose", self._on_left, pose_qos)
        self.create_subscription(PoseStamped, "/right_current_pose", self._on_right, pose_qos)

        self._pub_left = self.create_publisher(Pose, "/left_target", 10)
        self._pub_right = self.create_publisher(Pose, "/right_target", 10)

    def _on_left(self, msg: PoseStamped) -> None:
        self._left_msg = msg

    def _on_right(self, msg: PoseStamped) -> None:
        self._right_msg = msg

    def _wait_for_both(self) -> Tuple[PoseStamped, PoseStamped]:
        t0 = time.monotonic()
        while rclpy.ok():
            if self._left_msg is not None and self._right_msg is not None:
                return self._left_msg, self._right_msg
            if (time.monotonic() - t0) > self._wait_timeout_sec:
                raise TimeoutError(
                    f"wait timeout: left={self._left_msg is not None} right={self._right_msg is not None}"
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("rclpy not ok")

    def run_once(self) -> None:
        self.get_logger().info(
            "waiting for /left_current_pose and /right_current_pose... "
            f"(timeout={self._wait_timeout_sec:.1f}s)"
        )
        left_ps, right_ps = self._wait_for_both()

        left_xyz, left_q = _pose_stamped_to_xyzw(left_ps)
        right_xyz, right_q = _pose_stamped_to_xyzw(right_ps)

        if self._mode == "absolute":
            left_target_q = self._left_rot_xyzw
            right_target_q = self._right_rot_xyzw
        else:
            left_target_q = _quat_mul_xyzw(self._left_rot_xyzw, left_q)
            right_target_q = _quat_mul_xyzw(self._right_rot_xyzw, right_q)

        left_target = _make_pose(left_xyz, left_target_q)
        right_target = _make_pose(right_xyz, right_target_q)

        snap = Snapshot(
            stamp_unix_sec=float(time.time()),
            left_current={
                "xyz": left_xyz.tolist(),
                "xyzw": left_q.tolist(),
                "frame_id": str(left_ps.header.frame_id),
            },
            right_current={
                "xyz": right_xyz.tolist(),
                "xyzw": right_q.tolist(),
                "frame_id": str(right_ps.header.frame_id),
            },
            left_target={"xyz": left_xyz.tolist(), "xyzw": left_target_q.tolist()},
            right_target={"xyz": right_xyz.tolist(), "xyzw": right_target_q.tolist()},
            mode=self._mode,
            left_rot_xyzw=self._left_rot_xyzw.tolist(),
            right_rot_xyzw=self._right_rot_xyzw.tolist(),
        )

        try:
            os.makedirs(os.path.dirname(self._out_path) or ".", exist_ok=True)
            with open(self._out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(snap), f, ensure_ascii=False, indent=2)
            self.get_logger().info(f"snapshot saved: {self._out_path}")
        except Exception as e:
            self.get_logger().warn(f"failed to write snapshot json: {e}")

        self.get_logger().info(
            "publishing targets... "
            f"mode={self._mode} rate={self._publish_rate:.1f}Hz secs={self._publish_secs:.2f}"
        )
        dt = 1.0 / self._publish_rate
        t_end = time.monotonic() + self._publish_secs
        n = 0
        while rclpy.ok() and time.monotonic() < t_end:
            self._pub_left.publish(left_target)
            self._pub_right.publish(right_target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
            n += 1
        self.get_logger().info(f"done. published {n} messages per arm.")


def _parse_quat4(args4) -> np.ndarray:
    if args4 is None or len(args4) != 4:
        raise ValueError("quaternion must have 4 numbers: x y z w")
    return _quat_normalize_xyzw(np.array([float(x) for x in args4], dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="absolute",
        choices=["relative", "absolute"],
        help="absolute: q_target=q_rot; relative: q_target=q_rot⊗q_current",
    )
    ap.add_argument("--left-rot-xyzw", nargs=4, type=float, default=[0.0, 0.0, 0.0, 1.0], help="left rotation quaternion")
    ap.add_argument("--right-rot-xyzw", nargs=4, type=float, default=[0.0, 0.0, 0.0, 1.0], help="right rotation quaternion")
    ap.add_argument("--publish-rate", type=float, default=10.0, help="发布频率(Hz)，默认10Hz")
    ap.add_argument("--publish-secs", type=float, default=3.0, help="发布时长(秒)，默认3秒")
    ap.add_argument("--wait-timeout-sec", type=float, default=10.0)
    ap.add_argument(
        "--out",
        default=f"/tmp/umi_pose_rotation_test_{int(time.time())}.json",
        help="snapshot output json path",
    )
    args = ap.parse_args()

    rclpy.init()
    node = UmiPoseRotationTestNode(
        mode=args.mode,
        left_rot_xyzw=_parse_quat4(args.left_rot_xyzw),
        right_rot_xyzw=_parse_quat4(args.right_rot_xyzw),
        out_path=args.out,
        publish_rate=args.publish_rate,
        publish_secs=args.publish_secs,
        wait_timeout_sec=args.wait_timeout_sec,
    )
    try:
        node.run_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

