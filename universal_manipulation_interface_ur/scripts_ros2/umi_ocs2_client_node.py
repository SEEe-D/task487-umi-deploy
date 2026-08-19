#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Jazzy client 节点（Python3.12 + rclpy）：
- 相机：直接用设备端口号读取（cv2.VideoCapture）
- 反馈：订阅 /left_current_pose、/right_current_pose、/joint_states
- 推理：把观测通过 TCP 发给 conda 推理服务端（umi_conda_inference_server.py）
- 执行：默认 dry-run，只打印将要下发的末端/夹爪命令；--no-dry-run 才真实发布

注意：
- 这个脚本不 import torch/hydra/diffusion_policy，避免 ROS Jazzy Python3.12 与 conda Python3.9 冲突。
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.spatial.transform as st
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class Cv2FisheyeRemapper:
    """
    ROS 侧 remap：只用 OpenCV（不依赖 torch），直接用 map npz 里的 map1/map2 做 cv2.remap。
    兼容 scripts_ros2/do_remap.py 导出的 npz 字段：
      - map1, map2, src_w, src_h, render_w, render_h, align_corners (align_corners 对 cv2.remap 无影响)
      - 可选 out_w/out_h：若存在，会在 remap 后做 INTER_AREA 下采样
    """

    def __init__(self, map_path: str):
        self.map_path = map_path
        data = np.load(map_path, allow_pickle=False)
        for k in ("map1", "map2", "src_w", "src_h", "render_w", "render_h", "align_corners"):
            if k not in data:
                raise ValueError(f"map 文件缺少字段 '{k}': {map_path}")
        self.map1 = data["map1"].astype(np.float32)
        self.map2 = data["map2"].astype(np.float32)
        self.src_w = int(np.array(data["src_w"]).reshape(()))
        self.src_h = int(np.array(data["src_h"]).reshape(()))
        self.render_w = int(np.array(data["render_w"]).reshape(()))
        self.render_h = int(np.array(data["render_h"]).reshape(()))
        self.out_w = int(np.array(data["out_w"]).reshape(())) if "out_w" in data else self.render_w
        self.out_h = int(np.array(data["out_h"]).reshape(())) if "out_h" in data else self.render_h

    def remap_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"frame 必须是 BGR 三通道图像，当前 shape={getattr(frame_bgr, 'shape', None)}")
        h, w = frame_bgr.shape[:2]
        if (w != self.src_w) or (h != self.src_h):
            raise ValueError(
                f"输入帧分辨率 {w}x{h} 与 map 期望的 src {self.src_w}x{self.src_h} 不一致"
            )
        out = cv2.remap(frame_bgr, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (self.out_w, self.out_h) != (self.render_w, self.render_h):
            out = cv2.resize(out, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return out


def _send_msg(sock: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def _recv_msg(sock: socket.socket) -> dict:
    (n,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, n))


def _pose_stamped_to_pose6(msg: PoseStamped) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
    return np.concatenate([pos, rotvec], axis=-1)


def _rotate_pose6_rotvec_about_base_y(pose6: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    仅对 pose6 的旋转部分（axis-angle rotvec）施加绕 **末端自身坐标系（body/self）Y 轴** 的旋转。

    语义：若原姿态为 R（base->eef），则输出为：
      R' = R * Ry(angle)

    其中 Ry(angle) 定义在 eef/self 坐标系下；因此这是“绕自身 Y 轴”的旋转（右乘 / body-fixed）。
    注意：不改变平移（pos）。
    """
    x = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if x.shape[0] != 6:
        raise ValueError(f"pose6 must have 6 elements, got shape={x.shape}")
    pos = x[:3]
    rotvec = x[3:6]
    R = st.Rotation.from_rotvec(rotvec)
    Ry = st.Rotation.from_euler("y", float(angle_deg), degrees=True)
    # post-multiply: rotate about self/body axes
    R2 = R * Ry
    out = np.empty((6,), dtype=np.float64)
    out[:3] = pos
    out[3:6] = R2.as_rotvec()
    return out


@dataclass
class LatestMsg:
    stamp_sec: float = 0.0
    data: object = None


class UmiOcs2ClientNode(Node):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        frequency: float,
        steps_per_inference: int,
        execute_step_start: int,
        execute_step_stop: Optional[int],
        action_base: str,
        pose_offset_x: float,
        pose_offset_y: float,
        pose_offset_z: float,
        num_cameras: int,
        camera0_dev: str,
        camera1_dev: str,
        camera2_dev: Optional[str],
        camera_width: Optional[int],
        camera_height: Optional[int],
        fisheye_map0: Optional[str],
        fisheye_map1: Optional[str],
        fisheye_map2: Optional[str],
        show_cameras: bool,
        save_camera_debug_dir: Optional[str],
        save_camera_debug_rate_hz: float,
        gripper_joint_name_left: str,
        gripper_joint_name_right: str,
        gripper_position_index_left: int,
        gripper_position_index_right: int,
        gripper_deg_min: float,
        gripper_deg_max: float,
        dry_run: bool,
    ) -> None:
        super().__init__("umi_ocs2_client")

        self._host = str(host)
        self._port = int(port)
        self._dt = 1.0 / float(frequency)
        self._steps_per_inference = int(steps_per_inference)
        # Execute only a sub-range of the predicted action horizon.
        # Use 1-based indices for CLI friendliness: [start, stop] inclusive.
        self._execute_step_start = int(execute_step_start)
        self._execute_step_stop = None if execute_step_stop is None else int(execute_step_stop)
        if self._execute_step_start < 1:
            raise ValueError("execute_step_start must be >= 1 (1-based index)")
        if self._execute_step_stop is not None and self._execute_step_stop < self._execute_step_start:
            raise ValueError("execute_step_stop must be >= execute_step_start")
        self._action_base = str(action_base)
        # Pose offset applied BEFORE inference in XYZ (model frame shift).
        # User request: x -= 0.51, y += 0.26, z -= 0.31
        # Implement as: pose_model = pose_robot + offset; publish uses pose_robot = pose_model - offset
        self._pose_offset_xyz = np.array([float(pose_offset_x), float(pose_offset_y), float(pose_offset_z)], dtype=np.float64)
        self._show_cameras = bool(show_cameras)
        self._save_camera_debug_dir = save_camera_debug_dir
        self._save_camera_debug_rate_hz = float(save_camera_debug_rate_hz)
        if self._save_camera_debug_rate_hz <= 0:
            raise ValueError("save_camera_debug_rate_hz must be > 0")
        self._last_save_t = 0.0
        self._save_period_s = 1.0 / self._save_camera_debug_rate_hz
        self._dry_run = bool(dry_run)

        self._num_cameras = int(num_cameras)
        if self._num_cameras not in (2, 3):
            raise ValueError("num_cameras must be 2 or 3")

        self._gripper_joint_name_left = str(gripper_joint_name_left)
        self._gripper_joint_name_right = str(gripper_joint_name_right)
        self._gripper_position_index_left = int(gripper_position_index_left)
        self._gripper_position_index_right = int(gripper_position_index_right)
        self._gripper_deg_min = float(gripper_deg_min)
        self._gripper_deg_max = float(gripper_deg_max)

        # resolved indices (by name) once we see /joint_states
        self._resolved_gripper_index_left: Optional[int] = None
        self._resolved_gripper_index_right: Optional[int] = None

        # subs
        pose_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.RELIABLE)
        sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._latest_left_pose = LatestMsg()
        self._latest_right_pose = LatestMsg()
        self._latest_joint_state = LatestMsg()
        self.create_subscription(PoseStamped, "/left_current_pose", self._on_left_pose, pose_qos)
        self.create_subscription(PoseStamped, "/right_current_pose", self._on_right_pose, pose_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, sensor_qos)

        # publishers
        self._pub_gripper_left = self.create_publisher(Float64, "/Joint69/position_command", 10)
        self._pub_gripper_right = self.create_publisher(Float64, "/Joint79/position_command", 10)
        self._pub_left_pose = self.create_publisher(Pose, "/left_target", 10)
        self._pub_right_pose = self.create_publisher(Pose, "/right_target", 10)

        # services
        self._active = False
        self._episode_start_pose6: Optional[list] = None
        self.create_service(Trigger, "umi_ocs2_inference/start", self._srv_start)
        self.create_service(Trigger, "umi_ocs2_inference/stop", self._srv_stop)

        # camera
        self._remapper0 = Cv2FisheyeRemapper(fisheye_map0) if fisheye_map0 else None
        self._remapper1 = Cv2FisheyeRemapper(fisheye_map1) if fisheye_map1 else None
        self._remapper2 = Cv2FisheyeRemapper(fisheye_map2) if fisheye_map2 else None

        self._cam0 = self._open_cap(camera0_dev, camera_width, camera_height, self._remapper0, "camera0")
        self._cam1 = self._open_cap(camera1_dev, camera_width, camera_height, self._remapper1, "camera1")
        self._cam2: Optional[cv2.VideoCapture] = None
        if self._num_cameras == 3:
            if not camera2_dev:
                raise ValueError("num_cameras=3 requires --camera2-dev")
            self._cam2 = self._open_cap(camera2_dev, camera_width, camera_height, self._remapper2, "camera2")

        # show-cameras safety: require DISPLAY (X11/Wayland) or disable to avoid hang
        if self._show_cameras:
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                self.get_logger().warn(
                    "--show-cameras 需要图形界面（DISPLAY/WAYLAND_DISPLAY）。当前未检测到，已自动禁用；"
                    "可改用 --save-camera-debug-dir /tmp/cam_debug 落盘查看。"
                )
                self._show_cameras = False
            else:
                cv2.namedWindow("camera0", cv2.WINDOW_NORMAL)
                cv2.namedWindow("camera1", cv2.WINDOW_NORMAL)
                if self._cam2 is not None:
                    cv2.namedWindow("camera2", cv2.WINDOW_NORMAL)

        # tcp client
        self._sock: Optional[socket.socket] = None
        self._connect()

        # action queue from server
        self._action_queue: Optional[np.ndarray] = None
        self._action_queue_idx = 0
        # For logging: original index offset into the full predicted horizon (0-based)
        self._action_queue_orig_start_idx = 0

        # log throttle
        self._last_log_t = 0.0
        self._log_period_s = 0.2
        self._last_warmup_log_t = 0.0
        self._warmup_log_period_s = 1.0
        # debug: throttle model input payload logs (poses/images ordering)
        self._last_model_input_log_t = 0.0
        self._model_input_log_period_s = 1.0
        # debug: throttle subscription prints (per-topic, so one topic doesn't suppress others)
        self._sub_log_period_s = 1.0
        self._last_sub_log_t_left = 0.0
        self._last_sub_log_t_right = 0.0
        self._last_sub_log_t_joint = 0.0

        self.create_timer(self._dt, self._on_timer)

    def _pose6_to_model_frame(self, pose6_robot: np.ndarray) -> np.ndarray:
        pose6_robot = np.asarray(pose6_robot, dtype=np.float64)
        out = pose6_robot.copy()
        out[:3] = out[:3] + self._pose_offset_xyz
        return out

    def _pose6_to_robot_frame(self, pose6_model: np.ndarray) -> np.ndarray:
        pose6_model = np.asarray(pose6_model, dtype=np.float64)
        out = pose6_model.copy()
        out[:3] = out[:3] - self._pose_offset_xyz
        return out

    def _now(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._host, self._port))
        self._sock = s

        # best-effort handshake: detect server version/capabilities
        try:
            _send_msg(self._sock, {"type": "ping"})
            resp = _recv_msg(self._sock)
            if resp.get("ok", False):
                ver = resp.get("server_version", "unknown")
                horizons = resp.get("env_obs_horizons", None)
                self.get_logger().info(
                    f"connected to inference server {ver}; env_obs_horizons keys={list(horizons.keys()) if isinstance(horizons, dict) else horizons}"
                )
            else:
                self.get_logger().warn(f"server ping failed: {resp}")
        except Exception as e:
            # old server may not support ping
            self.get_logger().warn(f"server ping not supported (or failed): {e}")

    @staticmethod
    def _open_cap(dev: str, w: Optional[int], h: Optional[int], remapper, name: str) -> cv2.VideoCapture:
        src = int(dev) if str(dev).isdigit() else dev
        # Prefer V4L2 backend on Linux for /dev/video* or integer device id (avoids some gstreamer hangs/warnings)
        if (isinstance(src, int)) or (isinstance(src, str) and str(src).startswith("/dev/video")):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 {name} 设备: {dev}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # if remap enabled and w/h not provided, set to map src
        if remapper is not None and (w is None or h is None):
            w = int(remapper.src_w) if w is None else w
            h = int(remapper.src_h) if h is None else h
        if w is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        if h is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        return cap

    def _on_left_pose(self, msg: PoseStamped) -> None:
        self._latest_left_pose = LatestMsg(self._now(), msg)
        now = time.time()
        if (now - self._last_sub_log_t_left) >= self._sub_log_period_s:
            self._last_sub_log_t_left = now
            self.get_logger().info(
                f"rx /left_current_pose frame={msg.header.frame_id} "
                f"pos=[{msg.pose.position.x:.3f} {msg.pose.position.y:.3f} {msg.pose.position.z:.3f}] "
                f"quat=[{msg.pose.orientation.x:.3f} {msg.pose.orientation.y:.3f} "
                f"{msg.pose.orientation.z:.3f} {msg.pose.orientation.w:.3f}]"
            )

    def _on_right_pose(self, msg: PoseStamped) -> None:
        self._latest_right_pose = LatestMsg(self._now(), msg)
        now = time.time()
        if (now - self._last_sub_log_t_right) >= self._sub_log_period_s:
            self._last_sub_log_t_right = now
            self.get_logger().info(
                f"rx /right_current_pose frame={msg.header.frame_id} "
                f"pos=[{msg.pose.position.x:.3f} {msg.pose.position.y:.3f} {msg.pose.position.z:.3f}] "
                f"quat=[{msg.pose.orientation.x:.3f} {msg.pose.orientation.y:.3f} "
                f"{msg.pose.orientation.z:.3f} {msg.pose.orientation.w:.3f}]"
            )

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = LatestMsg(self._now(), msg)
        # resolve gripper indices by name (best-effort)
        if msg.name:
            if self._resolved_gripper_index_left is None:
                try:
                    self._resolved_gripper_index_left = list(msg.name).index(self._gripper_joint_name_left)
                except ValueError:
                    # fallback: case-insensitive match
                    try:
                        names_lower = [str(n).lower() for n in msg.name]
                        self._resolved_gripper_index_left = names_lower.index(self._gripper_joint_name_left.lower())
                    except ValueError:
                        pass
            if self._resolved_gripper_index_right is None:
                try:
                    self._resolved_gripper_index_right = list(msg.name).index(self._gripper_joint_name_right)
                except ValueError:
                    # fallback: case-insensitive match
                    try:
                        names_lower = [str(n).lower() for n in msg.name]
                        self._resolved_gripper_index_right = names_lower.index(self._gripper_joint_name_right.lower())
                    except ValueError:
                        pass

        now = time.time()
        if (now - self._last_sub_log_t_joint) >= self._sub_log_period_s:
            self._last_sub_log_t_joint = now
            n = len(msg.position) if msg.position is not None else 0
            p69 = None
            p79 = None
            try:
                idx_l = self._resolved_gripper_index_left
                idx_r = self._resolved_gripper_index_right
                if idx_l is None:
                    idx_l = self._gripper_position_index_left
                if idx_r is None:
                    idx_r = self._gripper_position_index_right
                if msg.position is not None and idx_l is not None and idx_l < len(msg.position):
                    p69 = float(msg.position[idx_l])
                if msg.position is not None and idx_r is not None and idx_r < len(msg.position):
                    p79 = float(msg.position[idx_r])
            except Exception:
                pass
            self.get_logger().info(
                f"rx /joint_states n_pos={int(n)} "
                f"{self._gripper_joint_name_left}="
                f"{'None' if p69 is None else f'{p69:.4f}'} "
                f"{self._gripper_joint_name_right}="
                f"{'None' if p79 is None else f'{p79:.4f}'} "
                f"(idx_by_name L={self._resolved_gripper_index_left} R={self._resolved_gripper_index_right})"
            )
            if msg.name and (p69 is None or p79 is None):
                # print joint names once per throttle window to help debugging
                self.get_logger().info(f"/joint_states names={list(msg.name)}")

    def _get_gripper_rad(self) -> Tuple[Optional[float], Optional[float]]:
        js: JointState = self._latest_joint_state.data
        if js is None or js.position is None:
            return None, None
        pos = list(js.position)
        def by_index(i: Optional[int]) -> Optional[float]:
            if i is None:
                return None
            if i < 0 or i >= len(pos):
                return None
            return float(pos[i])
        idx_l = self._resolved_gripper_index_left if self._resolved_gripper_index_left is not None else self._gripper_position_index_left
        idx_r = self._resolved_gripper_index_right if self._resolved_gripper_index_right is not None else self._gripper_position_index_right
        return by_index(idx_l), by_index(idx_r)

    def _rad_to_obs_deg(self, rad: float) -> float:
        deg = -float(rad) * (180.0 / float(np.pi))
        return float(np.clip(deg, self._gripper_deg_min, self._gripper_deg_max))

    def _action_deg_to_cmd_rad(self, deg: float) -> float:
        d = float(np.clip(deg, self._gripper_deg_min, self._gripper_deg_max))
        return float(-(d * float(np.pi) / 180.0))

    def _read_frame_bgr(self, cap: cv2.VideoCapture, remapper, name: str) -> np.ndarray:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"{name} read() failed")
        frame = frame.astype(np.uint8, copy=False)
        if remapper is not None:
            frame = remapper.remap_frame(frame)
        return frame

    def _encode_jpg(self, bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("jpg encode failed")
        return bytes(buf)

    def _maybe_save_debug_frames(
        self, cam0_bgr: np.ndarray, cam1_bgr: np.ndarray, cam2_bgr: Optional[np.ndarray] = None
    ) -> None:
        if not self._save_camera_debug_dir:
            return
        try:
            now = time.time()
            if (now - self._last_save_t) < self._save_period_s:
                return
            self._last_save_t = now
            os.makedirs(self._save_camera_debug_dir, exist_ok=True)
            ts = int(now * 1000)
            cv2.imwrite(os.path.join(self._save_camera_debug_dir, f"cam0_{ts}.jpg"), cam0_bgr)
            cv2.imwrite(os.path.join(self._save_camera_debug_dir, f"cam1_{ts}.jpg"), cam1_bgr)
            if cam2_bgr is not None:
                cv2.imwrite(os.path.join(self._save_camera_debug_dir, f"cam2_{ts}.jpg"), cam2_bgr)
        except Exception as e:
            self.get_logger().warn(f"保存调试图像失败: {e}")

    def _srv_start(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._latest_left_pose.data is None or self._latest_right_pose.data is None:
            res.success = False
            res.message = "未收到 /left_current_pose 或 /right_current_pose"
            return res
        left_pose6_robot = _pose_stamped_to_pose6(self._latest_left_pose.data)
        right_pose6_robot = _pose_stamped_to_pose6(self._latest_right_pose.data)
        # send episode_start in model frame (offset applied) for consistency with inference inputs/outputs
        left_pose6 = self._pose6_to_model_frame(left_pose6_robot)
        right_pose6 = self._pose6_to_model_frame(right_pose6_robot)
        # User request: before sending to server, rotate both hands' orientation about base Y by -90 deg.
        left_pose6 = _rotate_pose6_rotvec_about_base_y(left_pose6, -90.0)
        right_pose6 = _rotate_pose6_rotvec_about_base_y(right_pose6, -90.0)
        # IMPORTANT: model expects RIGHT then LEFT ordering
        self._episode_start_pose6 = [right_pose6.tolist(), left_pose6.tolist()]
        self._active = True
        self._action_queue = None
        self._action_queue_idx = 0
        try:
            _send_msg(self._sock, {"type": "start", "episode_start_pose6": self._episode_start_pose6, "action_base": self._action_base})
            resp = _recv_msg(self._sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error", "unknown"))
        except Exception as e:
            res.success = False
            res.message = f"start->server 失败: {e}"
            return res
        res.success = True
        res.message = "Started."
        return res

    def _srv_stop(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._active = False
        self._action_queue = None
        self._action_queue_idx = 0
        try:
            _send_msg(self._sock, {"type": "stop"})
            _ = _recv_msg(self._sock)
        except Exception:
            pass
        res.success = True
        res.message = "Stopped."
        return res

    def _ready(self) -> bool:
        return (
            (self._latest_left_pose.data is not None)
            and (self._latest_right_pose.data is not None)
            and (self._latest_joint_state.data is not None)
        )

    def _on_timer(self) -> None:
        # Always try to read camera frames (for debug saving / showing), even before start/server.
        cam0_bgr = None
        cam1_bgr = None
        cam2_bgr = None
        try:
            cam0_bgr = self._read_frame_bgr(self._cam0, self._remapper0, "camera0")
            cam1_bgr = self._read_frame_bgr(self._cam1, self._remapper1, "camera1")
            if self._cam2 is not None:
                cam2_bgr = self._read_frame_bgr(self._cam2, self._remapper2, "camera2")
            self._maybe_save_debug_frames(cam0_bgr, cam1_bgr, cam2_bgr)
            if self._show_cameras:
                cv2.imshow("camera0", cam0_bgr)
                cv2.imshow("camera1", cam1_bgr)
                if cam2_bgr is not None:
                    cv2.imshow("camera2", cam2_bgr)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"camera read/show/save error: {e}")
            return

        # If not started yet, stop here (still saved debug frames above).
        if not self._ready() or (not self._active) or (self._sock is None):
            return

        try:
            # refill queue from server if needed
            if self._action_queue is None or self._action_queue_idx >= len(self._action_queue):
                # IMPORTANT: model expects camera order RIGHT then LEFT.
                # We treat (cam1 -> right, cam0 -> left), so swap before sending.
                cam0_jpg = self._encode_jpg(cam1_bgr)  # right
                cam1_jpg = self._encode_jpg(cam0_bgr)  # left
                cam2_jpg = self._encode_jpg(cam2_bgr) if cam2_bgr is not None else None
                left_pose6_robot = _pose_stamped_to_pose6(self._latest_left_pose.data)
                right_pose6_robot = _pose_stamped_to_pose6(self._latest_right_pose.data)
                # apply requested offset BEFORE inference
                left_pose6 = self._pose6_to_model_frame(left_pose6_robot)
                right_pose6 = self._pose6_to_model_frame(right_pose6_robot)
                # User request: before sending to server, rotate orientation about base Y by -90 deg.
                left_pose6 = _rotate_pose6_rotvec_about_base_y(left_pose6, -90.0)
                right_pose6 = _rotate_pose6_rotvec_about_base_y(right_pose6, -90.0)
                gl_rad, gr_rad = self._get_gripper_rad()
                gl_deg = self._rad_to_obs_deg(gl_rad if gl_rad is not None else 0.0)
                gr_deg = self._rad_to_obs_deg(gr_rad if gr_rad is not None else 0.0)

                req = {
                    "type": "infer",
                    "cam0_jpg": cam0_jpg,
                    "cam1_jpg": cam1_jpg,
                    **({} if cam2_jpg is None else {"cam2_jpg": cam2_jpg}),
                    # IMPORTANT: model expects ee state order RIGHT then LEFT.
                    # Server uses fields left_pose6->robot0, right_pose6->robot1, so we swap payload.
                    "left_pose6": right_pose6.tolist(),   # robot0 := right
                    "right_pose6": left_pose6.tolist(),   # robot1 := left
                    "left_grip_deg": float(gr_deg),       # robot0 := right gripper
                    "right_grip_deg": float(gl_deg),      # robot1 := left gripper
                    "timestamp": time.time(),
                }
                # Debug: print EXACT model inputs for the two pose fields (throttled).
                # Note: payload is intentionally swapped to match model's expected order (RIGHT then LEFT).
                now = time.time()
                if (now - self._last_model_input_log_t) >= self._model_input_log_period_s:
                    self._last_model_input_log_t = now
                    try:
                        self.get_logger().info(
                            "model_input poses | "
                            f"left_pose6(robot0:=right)={req['left_pose6']} | "
                            f"right_pose6(robot1:=left)={req['right_pose6']}"
                        )
                    except Exception as e:
                        self.get_logger().warn(f"failed to log model_input poses: {e}")
                _send_msg(self._sock, req)
                resp = _recv_msg(self._sock)
                if resp.get("warming_up", False):
                    now = time.time()
                    if (now - self._last_warmup_log_t) >= self._warmup_log_period_s:
                        self._last_warmup_log_t = now
                        self.get_logger().info(f"server warming up: {resp.get('message','')}")
                    return
                if not resp.get("ok", False):
                    # backward-compat: old server returned ok=false with this error string during warmup
                    err = str(resp.get("error", "unknown"))
                    if "not enough history yet" in err or "not enough history" in err:
                        now = time.time()
                        if (now - self._last_warmup_log_t) >= self._warmup_log_period_s:
                            self._last_warmup_log_t = now
                            self.get_logger().info(f"server warming up (legacy): {err}")
                        return
                    raise RuntimeError(err)
                env_action = resp["env_action"]
                if isinstance(env_action, np.ndarray):
                    a = env_action
                else:
                    a = np.asarray(env_action, dtype=np.float32)
                if a.ndim == 1:
                    a = a[None, :]
                # First: keep at most steps_per_inference from server response.
                a_full = a[: self._steps_per_inference]
                # Then: select execution window within that horizon: [start, stop] (1-based, inclusive).
                start0 = self._execute_step_start - 1
                stop0_excl = self._execute_step_stop if self._execute_step_stop is not None else int(a_full.shape[0])
                stop0_excl = min(int(stop0_excl), int(a_full.shape[0]))
                if start0 >= int(a_full.shape[0]):
                    raise RuntimeError(
                        f"execute_step_start={self._execute_step_start} exceeds predicted horizon={int(a_full.shape[0])}"
                    )
                aq = a_full[start0:stop0_excl]
                self._action_queue_orig_start_idx = int(start0)
                # safety: filter abnormal values (nan/inf) from inference output
                if not np.isfinite(aq).all():
                    bad_mask = ~np.isfinite(aq)
                    bad_count = int(bad_mask.sum())
                    self.get_logger().warn(
                        f"inference returned non-finite values (nan/inf). "
                        f"drop this action queue. bad_count={bad_count} shape={tuple(aq.shape)}"
                    )
                    self._action_queue = None
                    self._action_queue_idx = 0
                    self._action_queue_orig_start_idx = 0
                    return
                self._action_queue = aq
                self._action_queue_idx = 0
                # debug: print the whole predicted action queue once per inference
                try:
                    aq = self._action_queue
                    aq_str = np.array2string(
                        aq,
                        formatter={"float_kind": lambda x: f"{float(x):.6f}"},
                        separator=", ",
                        max_line_width=200,
                    )
                    s1 = int(self._action_queue_orig_start_idx) + 1
                    s2 = int(self._action_queue_orig_start_idx) + int(aq.shape[0])
                    self.get_logger().info(
                        f"_action_queue shape={tuple(aq.shape)} exec_steps=[{s1}..{s2}] values={aq_str}"
                    )
                except Exception as e:
                    self.get_logger().warn(f"failed to print _action_queue: {e}")

            # execute next step (or dry-run log)
            a = self._action_queue[self._action_queue_idx]
            orig_step_1based = int(self._action_queue_orig_start_idx) + int(self._action_queue_idx) + 1
            self._action_queue_idx += 1
            # safety: double-check per-step
            if not np.isfinite(a).all():
                self.get_logger().warn(
                    f"non-finite action at exec_step(orig_1based)={orig_step_1based}. "
                    f"drop queue and do not publish."
                )
                self._action_queue = None
                self._action_queue_idx = 0
                self._action_queue_orig_start_idx = 0
                return

            # IMPORTANT: model outputs are RIGHT then LEFT ordering
            right_pose6 = a[0:6].astype(np.float64)
            right_grip_deg = float(a[6])
            left_pose6 = a[7:13].astype(np.float64)
            left_grip_deg = float(a[13])

            l_cmd = self._action_deg_to_cmd_rad(left_grip_deg)
            r_cmd = self._action_deg_to_cmd_rad(right_grip_deg)

            # User request: after inference, rotate orientation about base Y by +90 deg before publishing.
            left_pose6 = _rotate_pose6_rotvec_about_base_y(left_pose6, 90.0)
            right_pose6 = _rotate_pose6_rotvec_about_base_y(right_pose6, 90.0)

            # inverse transform BEFORE publishing (back to robot frame)
            left_pose6_robot = self._pose6_to_robot_frame(left_pose6)
            right_pose6_robot = self._pose6_to_robot_frame(right_pose6)
            if not np.isfinite(left_pose6_robot).all() or not np.isfinite(right_pose6_robot).all():
                self.get_logger().warn("non-finite pose after inverse offset transform; drop queue and do not publish.")
                self._action_queue = None
                self._action_queue_idx = 0
                return

            now = self._now()
            if (now - self._last_log_t) >= self._log_period_s:
                self._last_log_t = now
                self.get_logger().info(
                    "cmd | "
                    f"L_pose6=[{left_pose6_robot[0]:.4f} {left_pose6_robot[1]:.4f} {left_pose6_robot[2]:.4f} "
                    f"{left_pose6_robot[3]:.4f} {left_pose6_robot[4]:.4f} {left_pose6_robot[5]:.4f}] | "
                    f"R_pose6=[{right_pose6_robot[0]:.4f} {right_pose6_robot[1]:.4f} {right_pose6_robot[2]:.4f} "
                    f"{right_pose6_robot[3]:.4f} {right_pose6_robot[4]:.4f} {right_pose6_robot[5]:.4f}] | "
                    f"grip_deg(L,R)=({left_grip_deg:.2f}, {right_grip_deg:.2f}) "
                    f"-> cmd_rad(L,R)=({l_cmd:.6f}, {r_cmd:.6f})"
                    + (" [DRY_RUN]" if self._dry_run else "")
                )

            if self._dry_run:
                return

            # publish pose targets (continuous control chain expects Pose)
            lp = Pose()
            lp.position.x, lp.position.y, lp.position.z = map(float, left_pose6_robot[:3])
            ql = st.Rotation.from_rotvec(left_pose6_robot[3:6]).as_quat()
            lp.orientation.x, lp.orientation.y, lp.orientation.z, lp.orientation.w = map(float, ql)
            rp = Pose()
            rp.position.x, rp.position.y, rp.position.z = map(float, right_pose6_robot[:3])
            qr = st.Rotation.from_rotvec(right_pose6_robot[3:6]).as_quat()
            rp.orientation.x, rp.orientation.y, rp.orientation.z, rp.orientation.w = map(float, qr)
            self._pub_left_pose.publish(lp)
            self._pub_right_pose.publish(rp)
            self._pub_gripper_left.publish(Float64(data=float(l_cmd)))
            self._pub_gripper_right.publish(Float64(data=float(r_cmd)))
        except Exception as e:
            self.get_logger().error(f"on_timer error: {e}")
            try:
                self._connect()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=18080)
    ap.add_argument("--frequency", type=float, default=2.0)
    ap.add_argument("--steps-per-inference", type=int, default=6)
    ap.add_argument(
        "--execute-step-start",
        type=int,
        default=1,
        help="仅执行推理结果中的第 start 步（1-based, inclusive）。例如 5 表示从第5步开始执行。",
    )
    ap.add_argument(
        "--execute-step-stop",
        type=int,
        default=None,
        help="仅执行推理结果中的第 stop 步（1-based, inclusive）。例如 10 表示执行到第10步；不填则执行到最后一步。",
    )
    ap.add_argument("--action-base", default="current", choices=["current", "episode_start"])
    ap.add_argument("--pose-offset-x", type=float, default=-0.51, help="推理前对输入 pose 的 x 平移偏置（model=robot+offset）")
    ap.add_argument("--pose-offset-y", type=float, default=0.26, help="推理前对输入 pose 的 y 平移偏置（model=robot+offset）")
    ap.add_argument("--pose-offset-z", type=float, default=-0.31, help="推理前对输入 pose 的 z 平移偏置（model=robot+offset）")
    ap.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        choices=[2, 3],
        help="相机路数：默认2；设为3则额外读取 camera2 并在请求中携带 cam2_jpg。",
    )
    ap.add_argument("--camera0-dev", required=True)
    ap.add_argument("--camera1-dev", required=True)
    ap.add_argument("--camera2-dev", default=None, help="第三路相机设备（仅在 --num-cameras 3 时需要）。")
    ap.add_argument("--camera-width", type=int, default=None)
    ap.add_argument("--camera-height", type=int, default=None)
    ap.add_argument("--fisheye-map0", default=None)
    ap.add_argument("--fisheye-map1", default=None)
    ap.add_argument("--fisheye-map2", default=None, help="camera2 的 fisheye remap map(.npz)，字段同 map0/map1。")
    ap.add_argument("--show-cameras", action="store_true", help="调试用：imshow 两路相机（需要X11）")
    ap.add_argument("--save-camera-debug-dir", default=None, help="无GUI时用于调试：将两路相机帧周期性保存到该目录（jpg）")
    ap.add_argument("--save-camera-debug-rate-hz", type=float, default=1.0, help="保存调试图像频率(Hz)，默认1Hz")
    ap.add_argument("--gripper-joint-name-left", type=str, default="Joint69", help="从 /joint_states.name 匹配左夹爪 joint 名称")
    ap.add_argument("--gripper-joint-name-right", type=str, default="Joint79", help="从 /joint_states.name 匹配右夹爪 joint 名称")
    ap.add_argument("--gripper-position-index-left", type=int, default=69)
    ap.add_argument("--gripper-position-index-right", type=int, default=79)
    ap.add_argument("--gripper-deg-min", type=float, default=0.0)
    ap.add_argument("--gripper-deg-max", type=float, default=35.0)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()

    rclpy.init()
    node = UmiOcs2ClientNode(
        host=args.server_host,
        port=args.server_port,
        frequency=args.frequency,
        steps_per_inference=args.steps_per_inference,
        execute_step_start=args.execute_step_start,
        execute_step_stop=args.execute_step_stop,
        action_base=args.action_base,
        pose_offset_x=args.pose_offset_x,
        pose_offset_y=args.pose_offset_y,
        pose_offset_z=args.pose_offset_z,
        num_cameras=args.num_cameras,
        camera0_dev=args.camera0_dev,
        camera1_dev=args.camera1_dev,
        camera2_dev=args.camera2_dev,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        fisheye_map0=args.fisheye_map0,
        fisheye_map1=args.fisheye_map1,
        fisheye_map2=args.fisheye_map2,
        show_cameras=args.show_cameras,
        save_camera_debug_dir=args.save_camera_debug_dir,
        save_camera_debug_rate_hz=args.save_camera_debug_rate_hz,
        gripper_joint_name_left=args.gripper_joint_name_left,
        gripper_joint_name_right=args.gripper_joint_name_right,
        gripper_position_index_left=args.gripper_position_index_left,
        gripper_position_index_right=args.gripper_position_index_right,
        gripper_deg_min=args.gripper_deg_min,
        gripper_deg_max=args.gripper_deg_max,
        dry_run=args.dry_run,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

