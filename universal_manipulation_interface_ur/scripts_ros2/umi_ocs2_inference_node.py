#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 推理节点：在 universal_manipulation_interface 中加载 diffusion policy checkpoint，
订阅双相机 + 双臂末端当前位姿(/left_current_pose,/right_current_pose)，并将策略输出转换为
OCS2 Arm Controller 可接收的目标 PoseStamped，发布到：
  - /left_target/stamped  (以及兼容的 /ocs2_arm_controller/left_target/stamped)
  - /right_target/stamped (以及兼容的 /ocs2_arm_controller/right_target/stamped)

重要说明（与 ocs2_arm_controller 对齐）：
  - /left_current_target,/right_current_target 是控制器“回发布”的当前目标（可视化/误差统计），
    并不是控制器输入。控制器输入话题是 left_target/stamped 与 right_target/stamped。

启动方式（示例）：
  1) 先启动控制器（你给的命令）：
     ros2 launch ocs2_arm_controller demo.launch.py robot:=simpleai_x3 hardware:=real \
       can_interface_left:=can2 can_interface_right:=can3 can_fd:=false enable_protect_keyboard:=false

  2) 再启动本节点（需要已 source ROS2 环境）：
     python3 scripts_ros2/umi_ocs2_inference_node.py \
       --ckpt /path/to/latest.ckpt \
       --camera0-dev 0 --camera1-dev 1

  3) 在 RViz/ArmsTargetManager 把机器人拖到期望起始姿态后，调用开始服务锁定起始 pose：
     ros2 service call /umi_ocs2_inference/start std_srvs/srv/Trigger {}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# --- make repo importable when running as a script ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_ROS2_DIR = _REPO_ROOT / "scripts_ros2"
if str(_SCRIPTS_ROS2_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROS2_DIR))

import dill  # noqa: E402
import torch  # noqa: E402
import scipy.spatial.transform as st  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from geometry_msgs.msg import PoseStamped, Pose  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from umi.real_world.real_inference_util import (  # noqa: E402
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from umi.common.pose_util import pose_to_mat, mat_to_pose  # noqa: E402
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep  # noqa: E402
from umi.common.pose_util import pose10d_to_mat  # noqa: E402

try:
    # scripts_ros2/do_remap.py
    from do_remap import FisheyeRemapper  # type: ignore  # noqa: E402
except Exception:
    FisheyeRemapper = None  # type: ignore


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _pose_stamped_to_pos_axis_angle(msg: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    """PoseStamped -> (pos(3,), axis_angle(3,))"""
    p = msg.pose.position
    q = msg.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float32)
    quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return pos, rotvec


def _axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    quat = st.Rotation.from_rotvec(axis_angle.astype(np.float64)).as_quat()
    return quat.astype(np.float64)


@dataclass
class LatestMsg:
    stamp_sec: float = 0.0
    data: object = None


class _AsyncRecorder:
    """Best-effort async writer for images/json/npz to reduce impact on control loop."""

    def __init__(self, run_dir: str, *, max_queue: int = 512, enable: bool = True) -> None:
        self.enable = bool(enable)
        self.run_dir = str(run_dir)
        self.max_queue = int(max_queue)
        self._q: "queue.Queue[Tuple[str, str, object]]" = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()
        self._dropped = 0
        self._warned_drop = False

        # subfolders
        self.dir_images = os.path.join(self.run_dir, "images")
        self.dir_frames = os.path.join(self.run_dir, "frames")
        self.dir_infer = os.path.join(self.run_dir, "inference")
        self.dir_steps = os.path.join(self.run_dir, "steps")
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_frames, exist_ok=True)
        os.makedirs(self.dir_infer, exist_ok=True)
        os.makedirs(self.dir_steps, exist_ok=True)

        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def close(self, timeout_s: float = 2.0) -> None:
        if not self.enable:
            return
        self._stop.set()
        try:
            self._th.join(timeout=float(timeout_s))
        except Exception:
            pass

    def _try_put(self, kind: str, path: str, payload: object) -> None:
        if not self.enable:
            return
        try:
            self._q.put_nowait((kind, path, payload))
        except queue.Full:
            self._dropped += 1

    def save_image_bgr(self, *, name: str, t_ns: int, image_bgr: np.ndarray) -> str:
        """Return the absolute image path (even if dropped)."""
        fname = f"{name}_{int(t_ns)}.png"
        path = os.path.join(self.dir_images, fname)
        self._try_put("image", path, image_bgr)
        return path

    def save_json(self, *, subdir: str, name: str, t_ns: int, data: dict) -> str:
        fname = f"{name}_{int(t_ns)}.json"
        path = os.path.join(self.run_dir, subdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._try_put("json", path, data)
        return path

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                kind, path, payload = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                if kind == "image":
                    img = payload
                    # cv2.imwrite expects BGR uint8
                    cv2.imwrite(path, img)
                elif kind == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                # best-effort: ignore write failures (disk full, permission, etc.)
                pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass


class UmiOcs2InferenceNode(Node):
    def __init__(
        self,
        *,
        ckpt_path: str,
        frequency: float,
        steps_per_inference: int,
        device: str,
        target_frame_id: str,
        camera0_dev: Optional[str],
        camera1_dev: Optional[str],
        camera_width: Optional[int],
        camera_height: Optional[int],
        fisheye_map0: Optional[str],
        fisheye_map1: Optional[str],
        gripper_joint_left: str,
        gripper_joint_right: str,
        gripper_position_index_left: int,
        gripper_position_index_right: int,
        gripper_deg_min: float,
        gripper_deg_max: float,
        left_pose_candidates: List[str],
        right_pose_candidates: List[str],
        left_target_candidates: List[str],
        right_target_candidates: List[str],
        dual_target_candidates: List[str],
        publish_to_current_target: bool,
        publish_to_pose_target: bool,
        dry_run: bool,
        action_index: int,
        action_base: str,
        # IO logging
        save_io: bool,
        io_dir: Optional[str],
        io_async_max_queue: int,
        debug_imshow: bool,
    ) -> None:
        super().__init__("umi_ocs2_inference")

        self._camera0_dev = camera0_dev
        self._camera1_dev = camera1_dev
        self._camera_width = camera_width
        self._camera_height = camera_height

        # 已按你的要求禁用 topic 模式：必须使用设备端口号直接读取
        if (self._camera0_dev is None) or (self._camera1_dev is None):
            raise ValueError("请同时提供 --camera0-dev 与 --camera1-dev（已禁用 topic 模式）。")

        self._dt = 1.0 / float(frequency)
        self._steps_per_inference = int(steps_per_inference)
        self._target_frame_id = str(target_frame_id)
        self._action_index = int(action_index)
        self._action_base = str(action_base)
        self._fisheye_map0 = fisheye_map0
        self._fisheye_map1 = fisheye_map1
        self._publish_to_current_target = bool(publish_to_current_target)
        self._publish_to_pose_target = bool(publish_to_pose_target)

        # safety: dry-run disables ALL publishing (EEF + gripper); only logs commands for debugging
        self._dry_run = bool(dry_run)
        self._last_cmd_log_t = 0.0
        self._cmd_log_period_s = 0.2  # log at most 5Hz
        self._debug_imshow = bool(debug_imshow)

        # --- IO recorder (per-frame input/output dump) ---
        self._save_io = bool(save_io)
        self._io_run_dir: Optional[str] = None
        self._recorder: Optional[_AsyncRecorder] = None
        self._last_chunk_t_ns: Optional[int] = None
        base = io_dir if io_dir is not None else "/tmp/umi_ocs2_inference_io"
        self.get_logger().info(f"IO dump setting: save_io={self._save_io} io_dir={base}")
        if self._save_io:
            run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            try:
                self._io_run_dir = os.path.join(str(base), run_id)
                os.makedirs(self._io_run_dir, exist_ok=True)
                self._recorder = _AsyncRecorder(self._io_run_dir, max_queue=int(io_async_max_queue), enable=True)
                # write meta once
                meta = {
                    "created_localtime": run_id,
                    "ckpt_path": ckpt_path,
                    "frequency_hz": float(frequency),
                    "steps_per_inference": int(steps_per_inference),
                    "device": str(device),
                    "target_frame_id": str(target_frame_id),
                    "camera0_dev": camera0_dev,
                    "camera1_dev": camera1_dev,
                    "camera_width": camera_width,
                    "camera_height": camera_height,
                    "fisheye_map0": fisheye_map0,
                    "fisheye_map1": fisheye_map1,
                    "save_io": True,
                    "note": "images are saved as BGR PNG; timestamps are ROS clock nanoseconds.",
                }
                self._recorder.save_json(subdir=".", name="meta", t_ns=int(time.time() * 1e9), data=meta)
                self.get_logger().info(f"IO dump enabled. run_dir={self._io_run_dir}")
            except Exception as e:
                # if cannot create, disable to avoid crashing the node
                self._save_io = False
                self._io_run_dir = None
                self._recorder = None
                self.get_logger().error(f"IO dump init failed, disabled. err={e}")

        # gripper mapping (Joint angle <-> policy "width")
        self._gripper_joint_left = str(gripper_joint_left)
        self._gripper_joint_right = str(gripper_joint_right)
        self._gripper_position_index_left = int(gripper_position_index_left)
        self._gripper_position_index_right = int(gripper_position_index_right)
        # 约定：策略输出的是 0~35 的角度值（单位：deg），发布时需转换成负弧度下发：
        #   cmd_rad = -deg * pi/180
        # 同时观测侧从 /joint_states 读到的是弧度（范围约 -0.61086524~0），转换成正角度给策略：
        #   obs_deg = clamp(-rad * 180/pi, [deg_min, deg_max])
        self._gripper_deg_min = float(gripper_deg_min)
        self._gripper_deg_max = float(gripper_deg_max)
        if self._gripper_deg_max <= self._gripper_deg_min:
            raise ValueError("gripper_deg_max must be > gripper_deg_min")

        # --- robot config (tx_left_right) ---
        # 你当前不需要从 yaml 传入；默认假设左右臂已经在同一 base frame 下表达。
        # get_real_umi_obs_dict 仍需要一个 tx_robot1_robot0（用于构造“互为参照”的观测项），这里置为单位矩阵。
        self._tx_robot1_robot0 = np.eye(4, dtype=np.float64)

        # --- load checkpoint & build policy ---
        ckpt_path = ckpt_path
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
        payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]

        self.get_logger().info(f"Loaded ckpt: {ckpt_path}")
        self.get_logger().info(f"model_name: {cfg.policy.obs_encoder.model_name}")
        self.get_logger().info(f"dataset_path: {cfg.task.dataset.dataset_path}")

        self._cfg = cfg
        self._obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
        self._action_pose_repr = cfg.task.pose_repr.action_pose_repr
        self.get_logger().info(f"obs_pose_repr: {self._obs_pose_repr}")
        self.get_logger().info(f"action_pose_repr: {self._action_pose_repr}")

        obs_res = get_real_obs_resolution(cfg.task.shape_meta)
        self.get_logger().info(f"policy obs image resolution: {obs_res}")

        cls = __import__("hydra").utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.num_inference_steps = 16

        self._device = torch.device(device)
        self._policy = policy.eval().to(self._device)

        # --- ROS IO ---
        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self._latest_left_pose = LatestMsg()
        self._latest_right_pose = LatestMsg()

        # gripper feedback from /joint_states
        self._latest_joint_state = LatestMsg()
        self._joint_state_warned = False
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, sensor_qos)

        # camera input: device only
        self._cam0_cap = None
        self._cam1_cap = None

        # subscribe pose (support both /xxx and /ocs2_arm_controller/xxx)
        for t in left_pose_candidates:
            self.create_subscription(PoseStamped, t, self._on_left_pose, pose_qos)
        for t in right_pose_candidates:
            self.create_subscription(PoseStamped, t, self._on_right_pose, pose_qos)

        # publishers: publish to both candidate topics to be robust to namespace
        self._left_target_pubs = [self.create_publisher(PoseStamped, t, 10) for t in left_target_candidates]
        self._right_target_pubs = [self.create_publisher(PoseStamped, t, 10) for t in right_target_candidates]
        self._dual_target_pubs = [self.create_publisher(Path, t, 10) for t in dual_target_candidates]

        # optional: publish to Pose topics (continuous mode in arms_target_manager uses left_target/right_target)
        self._left_pose_target_pubs: List = []
        self._right_pose_target_pubs: List = []
        if self._publish_to_pose_target:
            self._left_pose_target_pubs = [
                self.create_publisher(Pose, t, 10) for t in ["/left_target", "/ocs2_arm_controller/left_target"]
            ]
            self._right_pose_target_pubs = [
                self.create_publisher(Pose, t, 10) for t in ["/right_target", "/ocs2_arm_controller/right_target"]
            ]

        # optional: some deployments may remap controller input to *current_target.
        # NOTE: in upstream ocs2_arm_controller implementation, *current_target is a publisher, not a subscriber.
        # Enable this only if you verified your live system expects it.
        self._left_current_target_pubs: List = []
        self._right_current_target_pubs: List = []
        if self._publish_to_current_target:
            self._left_current_target_pubs = [
                self.create_publisher(PoseStamped, t, 10)
                for t in ["/left_current_target", "/ocs2_arm_controller/left_current_target"]
            ]
            self._right_current_target_pubs = [
                self.create_publisher(PoseStamped, t, 10)
                for t in ["/right_current_target", "/ocs2_arm_controller/right_current_target"]
            ]

        # gripper command publishers (std_msgs/Float64)
        self._pub_gripper_left = self.create_publisher(
            Float64, f"/{self._gripper_joint_left}/position_command", 10
        )
        self._pub_gripper_right = self.create_publisher(
            Float64, f"/{self._gripper_joint_right}/position_command", 10
        )

        # services: start/stop
        self._active = False
        self._episode_start_pose: Optional[List[np.ndarray]] = None  # each is (6,)
        self.create_service(Trigger, "umi_ocs2_inference/start", self._srv_start)
        self.create_service(Trigger, "umi_ocs2_inference/stop", self._srv_stop)

        # Observation ring buffer sampled at control frequency.
        self._obs_buf: Dict[str, List[np.ndarray]] = {}
        self._ts_buf: List[float] = []

        # Determine required horizons & keys from shape_meta
        self._shape_meta = cfg.task.shape_meta
        self._obs_horizons = self._infer_obs_horizons(self._shape_meta)
        self.get_logger().info(f"obs horizons: {self._obs_horizons}")

        # optional fisheye remapper(s)
        self._remapper0 = None
        self._remapper1 = None
        self._remap_error_logged = False
        if (self._fisheye_map0 is not None) or (self._fisheye_map1 is not None):
            if FisheyeRemapper is None:
                raise RuntimeError(
                    "已指定 fisheye_map，但无法导入 FisheyeRemapper（scripts_ros2/do_remap.py）。"
                )
        if self._fisheye_map0:
            self.get_logger().info(f"Loading fisheye_map0: {self._fisheye_map0}")
            self._remapper0 = FisheyeRemapper(self._fisheye_map0)
        if self._fisheye_map1:
            self.get_logger().info(f"Loading fisheye_map1: {self._fisheye_map1}")
            self._remapper1 = FisheyeRemapper(self._fisheye_map1)
        # open device cameras (after remapper is ready so we can auto-set resolution)
        def _open_cap(dev: str, w: Optional[int], h: Optional[int]) -> cv2.VideoCapture:
            src = int(dev) if dev.isdigit() else dev
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开相机设备: {dev}")
            # try minimize latency
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if w is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            if h is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            return cap

        w0 = self._camera_width
        h0 = self._camera_height
        if (w0 is None or h0 is None) and (self._remapper0 is not None):
            # remapper requires exact src resolution
            w0 = int(self._remapper0.src_w) if w0 is None else w0
            h0 = int(self._remapper0.src_h) if h0 is None else h0
        self.get_logger().info(f"Opening camera0 device={self._camera0_dev} w={w0} h={h0}")
        self._cam0_cap = _open_cap(self._camera0_dev, w0, h0)

        w1 = self._camera_width
        h1 = self._camera_height
        if (w1 is None or h1 is None) and (self._remapper1 is not None):
            w1 = int(self._remapper1.src_w) if w1 is None else w1
            h1 = int(self._remapper1.src_h) if h1 is None else h1
        self.get_logger().info(f"Opening camera1 device={self._camera1_dev} w={w1} h={h1}")
        self._cam1_cap = _open_cap(self._camera1_dev, w1, h1)

        # timer loop
        self.create_timer(self._dt, self._on_timer)

        # warmup when enough data arrives
        self._did_warmup = False

        # action queue produced by policy inference, executed step-by-step at control frequency
        self._action_queue: Optional[np.ndarray] = None  # shape (H, 14) or (14,)
        self._action_queue_idx: int = 0

    @staticmethod
    def _infer_obs_horizons(shape_meta: dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for key, attr in shape_meta["obs"].items():
            h = int(attr.get("horizon", 1))
            out[key] = h
        # we also always provide timestamp
        out["timestamp"] = max(out.values()) if len(out) else 1
        return out

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _msg_stamp_sec(self, msg) -> float:
        # use ROS header stamp if present, else fallback to now
        if hasattr(msg, "header"):
            t = msg.header.stamp
            return float(t.sec) + float(t.nanosec) * 1e-9
        return self._now_sec()

    def _on_left_pose(self, msg: PoseStamped) -> None:
        self._latest_left_pose = LatestMsg(self._msg_stamp_sec(msg), msg)

    def _on_right_pose(self, msg: PoseStamped) -> None:
        self._latest_right_pose = LatestMsg(self._msg_stamp_sec(msg), msg)

    def _on_joint_state(self, msg: JointState) -> None:
        # no reliable stamp for hardware; use node time
        self._latest_joint_state = LatestMsg(self._now_sec(), msg)

    def _get_gripper_angles(self) -> Tuple[Optional[float], Optional[float]]:
        """
        返回 (left_angle_rad, right_angle_rad)；读不到则返回 None。
        优先使用 user 指定的 position index（69/79），否则按 joint name 查找。
        """
        js: JointState = self._latest_joint_state.data
        if js is None or js.position is None:
            return None, None
        pos = list(js.position)
        name = list(js.name) if js.name is not None else []

        def _by_index(idx: int) -> Optional[float]:
            if idx < 0 or idx >= len(pos):
                return None
            return float(pos[idx])

        l = _by_index(self._gripper_position_index_left)
        r = _by_index(self._gripper_position_index_right)
        if (l is not None) and (r is not None):
            return l, r

        # fallback by name
        if name:
            try:
                li = name.index(self._gripper_joint_left)
                l = float(pos[li]) if li < len(pos) else l
            except ValueError:
                pass
            try:
                ri = name.index(self._gripper_joint_right)
                r = float(pos[ri]) if ri < len(pos) else r
            except ValueError:
                pass
        return l, r

    def _angle_rad_to_obs_deg(self, angle_rad: float) -> float:
        # joint angle is negative rad; convert to positive degrees for policy
        deg = -float(angle_rad) * (180.0 / float(np.pi))
        return float(np.clip(deg, self._gripper_deg_min, self._gripper_deg_max))

    def _action_deg_to_cmd_rad(self, deg: float) -> float:
        # policy outputs degrees in [deg_min, deg_max]; command expects negative radians
        deg_c = float(np.clip(deg, self._gripper_deg_min, self._gripper_deg_max))
        rad = deg_c * (float(np.pi) / 180.0)
        return float(-rad)

    def _srv_start(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if (self._latest_left_pose.data is None) or (self._latest_right_pose.data is None):
            res.success = False
            res.message = "未收到 left/right 当前位姿（/left_current_pose,/right_current_pose），无法开始。"
            return res

        left_pos, left_rot = _pose_stamped_to_pos_axis_angle(self._latest_left_pose.data)
        right_pos, right_rot = _pose_stamped_to_pos_axis_angle(self._latest_right_pose.data)
        self._episode_start_pose = [
            np.concatenate([left_pos, left_rot], axis=-1),
            np.concatenate([right_pos, right_rot], axis=-1),
        ]
        self._active = True
        self._policy.reset()
        self._action_queue = None
        self._action_queue_idx = 0
        res.success = True
        res.message = "Started. 已锁定 episode_start_pose，并开始推理发布目标。"
        self.get_logger().info(res.message)
        return res

    def _srv_stop(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._active = False
        self._action_queue = None
        self._action_queue_idx = 0
        res.success = True
        res.message = "Stopped."
        self.get_logger().info(res.message)
        return res

    def _push_obs(self, key: str, x: np.ndarray, horizon: int) -> None:
        buf = self._obs_buf.setdefault(key, [])
        buf.append(x)
        if len(buf) > horizon:
            del buf[0 : len(buf) - horizon]

    def _ready(self) -> bool:
        # require camera & pose
        return (
            (self._cam0_cap is not None)
            and (self._cam1_cap is not None)
            and (self._latest_left_pose.data is not None)
            and (self._latest_right_pose.data is not None)
        )

    def _sample_and_buffer(self) -> None:
        """在控制频率下把 latest_* 采样进 ring buffer，以满足 horizon。"""
        now_ns = self._now_ns()
        now = float(now_ns) * 1e-9
        self._ts_buf.append(now)
        max_h = self._obs_horizons.get("timestamp", 1)
        if len(self._ts_buf) > max_h:
            del self._ts_buf[0 : len(self._ts_buf) - max_h]

        # camera: 先解码为 BGR，再（可选）fisheye remap，然后转成 RGB 喂给策略。
        # 原因：FisheyeRemapper.remap_frame(frame_bgr) 要求输入/输出都是 BGR；
        #      而 UMI 的 get_image_transform(... bgr_to_rgb=False) 不会替我们换通道，
        #      因此进入 obs 的 camera*_rgb 必须是真正的 RGB 排列。
        def _read_cap(cap: cv2.VideoCapture, name: str) -> np.ndarray:
            ok, frame = cap.read()
            if (not ok) or frame is None:
                raise RuntimeError(f"{name} 读取失败（VideoCapture.read() 返回空）。")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"{name} 帧格式异常，shape={getattr(frame, 'shape', None)}")
            return frame.astype(np.uint8, copy=False)

        if self._cam0_cap is None or self._cam1_cap is None:
            raise RuntimeError("相机设备未打开。")
        cam0_bgr = _read_cap(self._cam0_cap, "camera0")
        cam1_bgr = _read_cap(self._cam1_cap, "camera1")

        if self._remapper0 is not None:
            try:
                cam0_bgr = self._remapper0.remap_frame(cam0_bgr)
            except Exception as e:
                if not self._remap_error_logged:
                    self._remap_error_logged = True
                    self.get_logger().error(
                        f"camera0 fisheye remap 失败: {e}. "
                        "常见原因：输入帧分辨率与 map 的 src_w/src_h 不一致。"
                    )
                raise
        if self._remapper1 is not None:
            try:
                cam1_bgr = self._remapper1.remap_frame(cam1_bgr)
            except Exception as e:
                if not self._remap_error_logged:
                    self._remap_error_logged = True
                    self.get_logger().error(
                        f"camera1 fisheye remap 失败: {e}. "
                        "常见原因：输入帧分辨率与 map 的 src_w/src_h 不一致。"
                    )
                raise

        # optional debug display (requires X11; may block)
        if self._debug_imshow:
            try:
                cv2.imshow("camera0", cam0_bgr)
                cv2.imshow("camera1", cam1_bgr)
                cv2.waitKey(1)
            except Exception:
                pass

        # dump raw per-frame inputs (best-effort)
        cam0_path = None
        cam1_path = None
        if self._save_io and (self._recorder is not None):
            try:
                cam0_path = self._recorder.save_image_bgr(name="camera0", t_ns=now_ns, image_bgr=cam0_bgr)
                cam1_path = self._recorder.save_image_bgr(name="camera1", t_ns=now_ns, image_bgr=cam1_bgr)
            except Exception:
                cam0_path = None
                cam1_path = None

        cam0_img = cam0_bgr[..., ::-1]
        cam1_img = cam1_bgr[..., ::-1]

        # pose
        left_pos, left_rot = _pose_stamped_to_pos_axis_angle(self._latest_left_pose.data)
        right_pos, right_rot = _pose_stamped_to_pos_axis_angle(self._latest_right_pose.data)

        # dump state (poses + gripper) aligned to this sampled frame
        if self._save_io and (self._recorder is not None):
            try:
                left_a, right_a = self._get_gripper_angles()
                frame = {
                    "t_ns": int(now_ns),
                    "t_sec": float(now),
                    "camera0_png": os.path.basename(cam0_path) if cam0_path else None,
                    "camera1_png": os.path.basename(cam1_path) if cam1_path else None,
                    "left_pose": {
                        "pos_xyz": [float(x) for x in left_pos.tolist()],
                        "rot_axis_angle": [float(x) for x in left_rot.tolist()],
                        "frame_id": getattr(getattr(self._latest_left_pose.data, "header", None), "frame_id", ""),
                        "stamp_sec": float(self._latest_left_pose.stamp_sec),
                    },
                    "right_pose": {
                        "pos_xyz": [float(x) for x in right_pos.tolist()],
                        "rot_axis_angle": [float(x) for x in right_rot.tolist()],
                        "frame_id": getattr(getattr(self._latest_right_pose.data, "header", None), "frame_id", ""),
                        "stamp_sec": float(self._latest_right_pose.stamp_sec),
                    },
                    "gripper": {
                        "left_angle_rad": None if left_a is None else float(left_a),
                        "right_angle_rad": None if right_a is None else float(right_a),
                        "left_obs_deg": None if left_a is None else float(self._angle_rad_to_obs_deg(left_a)),
                        "right_obs_deg": None if right_a is None else float(self._angle_rad_to_obs_deg(right_a)),
                        "joint_left": self._gripper_joint_left,
                        "joint_right": self._gripper_joint_right,
                        "position_index_left": int(self._gripper_position_index_left),
                        "position_index_right": int(self._gripper_position_index_right),
                    },
                }
                self._recorder.save_json(subdir="frames", name="frame", t_ns=now_ns, data=frame)
            except Exception:
                pass

        # fill required keys from shape_meta
        # cameras are usually camera0_rgb / camera1_rgb
        if "camera0_rgb" in self._shape_meta["obs"]:
            self._push_obs("camera0_rgb", cam0_img, self._obs_horizons["camera0_rgb"])
        if "camera1_rgb" in self._shape_meta["obs"]:
            self._push_obs("camera1_rgb", cam1_img, self._obs_horizons["camera1_rgb"])

        # robot0 = left, robot1 = right
        self._push_obs("robot0_eef_pos", left_pos, self._obs_horizons.get("robot0_eef_pos", 1))
        self._push_obs(
            "robot0_eef_rot_axis_angle", left_rot, self._obs_horizons.get("robot0_eef_rot_axis_angle", 1)
        )
        self._push_obs("robot1_eef_pos", right_pos, self._obs_horizons.get("robot1_eef_pos", 1))
        self._push_obs(
            "robot1_eef_rot_axis_angle", right_rot, self._obs_horizons.get("robot1_eef_rot_axis_angle", 1)
        )

        # optional: gripper width, if policy expects it
        for k in ("robot0_gripper_width", "robot1_gripper_width"):
            if k in self._shape_meta["obs"]:
                left_a, right_a = self._get_gripper_angles()
                if left_a is None or right_a is None:
                    if not self._joint_state_warned:
                        self._joint_state_warned = True
                        self.get_logger().warn(
                            "未能从 /joint_states 解析 Joint69/Joint79 夹爪角度；将使用 0 作为占位。"
                        )
                    left_w = 0.0
                    right_w = 0.0
                else:
                    # 这里虽然 key 叫 gripper_width，但对齐你的模型语义：实际填的是“夹爪角度(deg)”
                    left_w = self._angle_rad_to_obs_deg(left_a)
                    right_w = self._angle_rad_to_obs_deg(right_a)
                if k == "robot0_gripper_width":
                    self._push_obs(k, np.array([left_w], dtype=np.float32), self._obs_horizons[k])
                else:
                    self._push_obs(k, np.array([right_w], dtype=np.float32), self._obs_horizons[k])

        # timestamp for env_obs (T,)
        # keep as float64
        self._push_obs("timestamp", np.array([now], dtype=np.float64), self._obs_horizons["timestamp"])

    def _build_env_obs(self) -> Optional[Dict[str, np.ndarray]]:
        # ensure every required key has enough horizon length
        for key, h in self._obs_horizons.items():
            if key not in self._obs_buf:
                return None
            if len(self._obs_buf[key]) < h:
                return None

        env_obs: Dict[str, np.ndarray] = {}
        for key, h in self._obs_horizons.items():
            xs = self._obs_buf[key][-h:]
            x = np.stack(xs, axis=0)
            # special-case timestamp: stored as (1,) each -> stack -> (T,1) -> squeeze to (T,)
            if key == "timestamp":
                x = x.reshape(-1)
            env_obs[key] = x
        return env_obs

    def _publish_target(self, left_pose6: np.ndarray, right_pose6: np.ndarray, left_grip: float, right_grip: float) -> None:
        # left/right pose6: [x,y,z, rx,ry,rz] (axis-angle rotvec)
        now = self.get_clock().now().to_msg()
        now_s = self._now_sec()

        # 允许 target_frame_id="auto"：沿用反馈 pose 的 frame_id，减少 TF/坐标系配置负担。
        frame_id = self._target_frame_id
        if frame_id == "auto":
            try:
                frame_id = str(self._latest_left_pose.data.header.frame_id)
            except Exception:
                frame_id = "base_link"

        def to_msg(pose6: np.ndarray) -> PoseStamped:
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = frame_id
            msg.pose.position.x = float(pose6[0])
            msg.pose.position.y = float(pose6[1])
            msg.pose.position.z = float(pose6[2])
            q = _axis_angle_to_quat_xyzw(pose6[3:6])
            msg.pose.orientation.x = float(q[0])
            msg.pose.orientation.y = float(q[1])
            msg.pose.orientation.z = float(q[2])
            msg.pose.orientation.w = float(q[3])
            return msg

        lmsg = to_msg(left_pose6)
        rmsg = to_msg(right_pose6)

        # publish gripper commands (angle rad)
        try:
            l_cmd = self._action_deg_to_cmd_rad(float(left_grip))
            r_cmd = self._action_deg_to_cmd_rad(float(right_grip))
            if self._dry_run:
                # print the EXACT command that would be sent
                if (now_s - self._last_cmd_log_t) >= self._cmd_log_period_s:
                    self._last_cmd_log_t = now_s
                    self.get_logger().info(
                        "DRY_RUN cmd | frame=%s | "
                        "L_pose6=[%.4f %.4f %.4f %.4f %.4f %.4f] | "
                        "R_pose6=[%.4f %.4f %.4f %.4f %.4f %.4f] | "
                        "grip_deg(L,R)=(%.2f, %.2f) -> cmd_rad(L,R)=(%.6f, %.6f)",
                        frame_id,
                        float(left_pose6[0]), float(left_pose6[1]), float(left_pose6[2]),
                        float(left_pose6[3]), float(left_pose6[4]), float(left_pose6[5]),
                        float(right_pose6[0]), float(right_pose6[1]), float(right_pose6[2]),
                        float(right_pose6[3]), float(right_pose6[4]), float(right_pose6[5]),
                        float(left_grip), float(right_grip),
                        float(l_cmd), float(r_cmd),
                    )
                return

            self._pub_gripper_left.publish(Float64(data=l_cmd))
            self._pub_gripper_right.publish(Float64(data=r_cmd))
        except Exception as e:
            self.get_logger().error(f"发布夹爪指令失败: {e}")
            if self._dry_run:
                return

        # dry-run: do NOT publish end-effector targets either
        if self._dry_run:
            return

        # optional: publish Pose (no frame) to left_target/right_target topics.
        # NOTE: PoseBasedReferenceManager will treat these as already in base_frame_ (no TF transform),
        # so only enable if you are sure the pose is expressed in the controller base frame.
        if self._publish_to_pose_target:
            lp = Pose()
            lp.position = lmsg.pose.position
            lp.orientation = lmsg.pose.orientation
            rp = Pose()
            rp.position = rmsg.pose.position
            rp.orientation = rmsg.pose.orientation
            for pub in self._left_pose_target_pubs:
                pub.publish(lp)
            for pub in self._right_pose_target_pubs:
                pub.publish(rp)

        # optional override: publish to *current_target topics (only if explicitly enabled)
        if self._publish_to_current_target:
            for pub in self._left_current_target_pubs:
                pub.publish(lmsg)
            for pub in self._right_current_target_pubs:
                pub.publish(rmsg)
            # still also publish to standard command topics for safety

        for pub in self._left_target_pubs:
            pub.publish(lmsg)
        for pub in self._right_target_pubs:
            pub.publish(rmsg)

        # also publish dual_target/stamped (Path with 2 poses) for atomic dual-arm update
        if len(self._dual_target_pubs) > 0:
            path = Path()
            path.header.stamp = now
            path.header.frame_id = frame_id
            path.poses = [lmsg, rmsg]
            for pub in self._dual_target_pubs:
                pub.publish(path)

    def _raw_action_to_env_action(
        self, raw_action: np.ndarray, env_obs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        将策略输出 raw_action (.., 20) 转为 env_action (.., 14)：
          [left_pose6, left_grip, right_pose6, right_grip]

        - action_pose_repr 由 ckpt cfg.task.pose_repr.action_pose_repr 决定。
        - action_base:
            - "current": 与 UMI 现有实现一致，动作相对“当前观测 pose”解释（默认）。
            - "episode_start": 将相对动作解释为相对 start pose（满足你的需求描述）。
              注意：前提是训练时 action_pose_repr 的语义也确实是相对 start，否则会不一致。
        """
        if self._action_base == "current":
            return get_real_umi_action(raw_action, env_obs, self._action_pose_repr)

        if self._action_base != "episode_start":
            raise RuntimeError(f"Unsupported action_base: {self._action_base}")
        if self._episode_start_pose is None:
            raise RuntimeError("episode_start_pose is None; call /start first.")

        n_robots = int(raw_action.shape[-1] // 10)
        if n_robots != 2:
            raise RuntimeError(f"Expected 2 robots, got {n_robots}")

        env_action_parts: List[np.ndarray] = []
        for robot_idx in range(n_robots):
            start_pose6 = self._episode_start_pose[robot_idx].astype(np.float64)
            base_pose_mat = pose_to_mat(start_pose6)

            start = robot_idx * 10
            action_pose10d = raw_action[..., start : start + 9]
            action_grip = raw_action[..., start + 9 : start + 10]
            action_pose_mat = pose10d_to_mat(action_pose10d)

            # convert pose representation back to absolute in base frame
            action_abs_mat = convert_pose_mat_rep(
                action_pose_mat, base_pose_mat=base_pose_mat, pose_rep=self._action_pose_repr, backward=True
            )
            action_pose6 = mat_to_pose(action_abs_mat)  # (...,6)
            env_action_parts.append(action_pose6)
            env_action_parts.append(action_grip)

        return np.concatenate(env_action_parts, axis=-1)

    def _on_timer(self) -> None:
        if not self._ready():
            return

        # sample into horizon buffers
        self._sample_and_buffer()
        env_obs = self._build_env_obs()
        if env_obs is None:
            return

        # warmup once before active
        if not self._did_warmup:
            try:
                with torch.no_grad():
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=env_obs,
                        shape_meta=self._shape_meta,
                        obs_pose_repr=self._obs_pose_repr,
                        tx_robot1_robot0=self._tx_robot1_robot0,
                        episode_start_pose=self._episode_start_pose,
                    )
                    obs_dict = dict_apply(
                        obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self._device)
                    )
                    _ = self._policy.predict_action(obs_dict)
                self._did_warmup = True
                self.get_logger().info("Policy warmup done.")
            except Exception as e:
                self.get_logger().error(f"Policy warmup failed: {e}")
            return

        if not self._active:
            return

        if self._episode_start_pose is None:
            self.get_logger().warn("active=True but episode_start_pose is None; call /start first.")
            return

        try:
            timer_ns = self._now_ns()
            # if queue empty/exhausted, run inference to refill
            if (self._action_queue is None) or (self._action_queue_idx >= len(self._action_queue)):
                with torch.no_grad():
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=env_obs,
                        shape_meta=self._shape_meta,
                        obs_pose_repr=self._obs_pose_repr,
                        tx_robot1_robot0=self._tx_robot1_robot0,
                        episode_start_pose=self._episode_start_pose,
                    )
                    obs_dict = dict_apply(
                        obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self._device)
                    )
                    result = self._policy.predict_action(obs_dict)
                    raw_action = result["action_pred"][0].detach().to("cpu").numpy()
                    env_action = self._raw_action_to_env_action(raw_action, env_obs)

                def _jsonify(x):
                    # numpy -> {"dtype","shape","data"} (flattened), scalars -> python scalar, lists/tuples -> list
                    if isinstance(x, np.ndarray):
                        return {
                            "dtype": str(x.dtype),
                            "shape": list(x.shape),
                            "data": x.reshape(-1).tolist(),
                        }
                    if isinstance(x, (np.floating, np.integer)):
                        return x.item()
                    if isinstance(x, (list, tuple)):
                        return [_jsonify(v) for v in x]
                    if isinstance(x, dict):
                        return {str(k): _jsonify(v) for k, v in x.items()}
                    return x

                # record model input/output for this inference (action chunk)
                if self._save_io and (self._recorder is not None):
                    try:
                        payload = {
                            "t_ns": int(timer_ns),
                            "env_obs": {str(k): _jsonify(np.asarray(v)) for k, v in env_obs.items()},
                            "model_in": {str(k): _jsonify(np.asarray(v)) for k, v in obs_dict_np.items()},
                            "raw_action": _jsonify(np.asarray(raw_action)),
                            "env_action": _jsonify(np.asarray(env_action)),
                        }
                        self._recorder.save_json(subdir="inference", name="chunk", t_ns=timer_ns, data=payload)
                        self._last_chunk_t_ns = int(timer_ns)
                    except Exception:
                        pass

                # normalize to (H,14)
                if env_action.ndim == 1:
                    env_action = env_action[None, :]
                if env_action.shape[-1] != 14:
                    raise RuntimeError(f"Expected env_action (...,14), got {env_action.shape}")

                # limit horizon if user asks (steps_per_inference)
                H = min(int(self._steps_per_inference), int(env_action.shape[0]))
                self._action_queue = env_action[:H]
                self._action_queue_idx = 0

            # publish next step
            assert self._action_queue is not None
            a = self._action_queue[self._action_queue_idx]
            self._action_queue_idx += 1

            # record per-step output (what we actually execute this frame)
            if self._save_io and (self._recorder is not None):
                try:
                    step = {
                        "t_ns": int(timer_ns),
                        "action_step": _jsonify(np.asarray(a)),
                        "queue_idx_after_inc": int(self._action_queue_idx),
                        "queue_len": int(len(self._action_queue)),
                        "last_chunk_t_ns": int(self._last_chunk_t_ns) if self._last_chunk_t_ns is not None else -1,
                    }
                    self._recorder.save_json(subdir="steps", name="step", t_ns=timer_ns, data=step)
                except Exception:
                    pass

            left_pose6 = a[0:6].astype(np.float64)
            left_grip = float(a[6])
            right_pose6 = a[7:13].astype(np.float64)
            right_grip = float(a[13])
            self._publish_target(left_pose6, right_pose6, left_grip, right_grip)
        except Exception as e:
            self.get_logger().error(f"Inference/publish failed: {e}")

    def destroy_node(self) -> bool:
        # close cameras & recorder
        try:
            if self._cam0_cap is not None:
                self._cam0_cap.release()
        except Exception:
            pass
        try:
            if self._cam1_cap is not None:
                self._cam1_cap.release()
        except Exception:
            pass
        try:
            if self._recorder is not None:
                self._recorder.close(timeout_s=2.0)
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .ckpt or experiment dir (will use checkpoints/latest.ckpt)")
    ap.add_argument("--frequency", type=float, default=10.0)
    ap.add_argument("--steps-per-inference", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--target-frame-id",
        default="auto",
        help="PoseStamped header.frame_id to publish. 建议 auto（沿用 /left_current_pose 的 frame_id）",
    )
    ap.add_argument("--camera0-dev", required=True, help="直接用设备读取 camera0：如 0 或 /dev/video0。")
    ap.add_argument("--camera1-dev", required=True, help="直接用设备读取 camera1：如 1 或 /dev/video1。")
    ap.add_argument("--camera-width", type=int, default=None, help="设备模式下强制设置相机宽度（两路共用）。若使用 fisheye map 且未指定，会自动用 map 的 src_w。")
    ap.add_argument("--camera-height", type=int, default=None, help="设备模式下强制设置相机高度（两路共用）。若使用 fisheye map 且未指定，会自动用 map 的 src_h。")
    ap.add_argument(
        "--fisheye-map0",
        default=None,
        help="camera0 的 fisheye remap map(.npz)。使用 do_remap.FisheyeRemapper；输入/输出均为BGR。",
    )
    ap.add_argument(
        "--fisheye-map1",
        default=None,
        help="camera1 的 fisheye remap map(.npz)。使用 do_remap.FisheyeRemapper；输入/输出均为BGR。",
    )
    # gripper config
    ap.add_argument("--gripper-joint-left", default="Joint69")
    ap.add_argument("--gripper-joint-right", default="Joint79")
    ap.add_argument("--gripper-position-index-left", type=int, default=69, help="从 /joint_states.position 读取左夹爪角度的索引")
    ap.add_argument("--gripper-position-index-right", type=int, default=79, help="从 /joint_states.position 读取右夹爪角度的索引")
    ap.add_argument("--gripper-deg-min", type=float, default=0.0, help="策略侧夹爪最小角度(deg)")
    ap.add_argument("--gripper-deg-max", type=float, default=35.0, help="策略侧夹爪最大角度(deg)")
    # safety: default dry-run enabled
    ap.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="安全调试：不下发任何末端/夹爪指令，仅打印将要下发的命令（默认开启）。",
    )
    ap.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="关闭 dry-run：开始真实下发末端与夹爪指令。",
    )
    ap.add_argument("--action-index", type=int, default=0, help="If policy returns multi-step, which step to publish")
    ap.add_argument(
        "--publish-to-current-target",
        action="store_true",
        help="若你的现场系统确实把控制器输入接在 /left_current_target,/right_current_target，则打开此开关；默认不打开。",
    )
    # 默认改为：发布 geometry_msgs/Pose 到 /left_target,/right_target（RViz 连续控制同路）
    # 同时保留可禁用选项，便于切回 stamped/path 通路。
    ap.add_argument(
        "--publish-to-pose-target",
        dest="publish_to_pose_target",
        action="store_true",
        default=True,
        help="发布 geometry_msgs/Pose 到 /left_target,/right_target（无 frame）。默认开启。",
    )
    ap.add_argument(
        "--no-publish-to-pose-target",
        dest="publish_to_pose_target",
        action="store_false",
        help="禁用发布 /left_target,/right_target（Pose）。",
    )
    ap.add_argument(
        "--action-base",
        default="current",
        choices=["current", "episode_start"],
        help="如何解释相对动作：current=相对当前pose（与UMI现有实现一致），episode_start=相对/start锁定pose（符合你的需求描述）",
    )
    # IO dump (per-frame input/output)
    ap.add_argument(
        "--save-io",
        dest="save_io",
        action="store_true",
        default=True,
        help="保存每帧输入(相机图像+state)与输出(action chunk/step)到文件（默认开启）。",
    )
    ap.add_argument(
        "--no-save-io",
        dest="save_io",
        action="store_false",
        help="关闭保存IO（不落盘）。",
    )
    ap.add_argument(
        "--io-dir",
        default="/tmp/umi_ocs2_inference_io",
        help="IO保存根目录；会自动创建 run 子目录（如 /tmp/umi_ocs2_inference_io/20260225_123456）。",
    )
    ap.add_argument(
        "--io-async-max-queue",
        type=int,
        default=512,
        help="异步写盘队列上限（满了会丢数据，避免阻塞控制循环）。",
    )
    ap.add_argument(
        "--debug-imshow",
        action="store_true",
        default=False,
        help="调试：显示两路相机窗口（需要X11，可能阻塞）。默认关闭。",
    )
    args = ap.parse_args()

    rclpy.init()
    node = UmiOcs2InferenceNode(
        ckpt_path=args.ckpt,
        frequency=args.frequency,
        steps_per_inference=args.steps_per_inference,
        device=args.device,
        target_frame_id=args.target_frame_id,
        camera0_dev=args.camera0_dev,
        camera1_dev=args.camera1_dev,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        fisheye_map0=args.fisheye_map0,
        fisheye_map1=args.fisheye_map1,
        gripper_joint_left=args.gripper_joint_left,
        gripper_joint_right=args.gripper_joint_right,
        gripper_position_index_left=args.gripper_position_index_left,
        gripper_position_index_right=args.gripper_position_index_right,
        gripper_deg_min=args.gripper_deg_min,
        gripper_deg_max=args.gripper_deg_max,
        left_pose_candidates=["/left_current_pose", "/ocs2_arm_controller/left_current_pose"],
        right_pose_candidates=["/right_current_pose", "/ocs2_arm_controller/right_current_pose"],
        left_target_candidates=["/left_target/stamped", "/ocs2_arm_controller/left_target/stamped"],
        right_target_candidates=["/right_target/stamped", "/ocs2_arm_controller/right_target/stamped"],
        dual_target_candidates=["/dual_target/stamped", "/ocs2_arm_controller/dual_target/stamped"],
        publish_to_current_target=args.publish_to_current_target,
        publish_to_pose_target=args.publish_to_pose_target,
        dry_run=args.dry_run,
        action_index=args.action_index,
        action_base=args.action_base,
        save_io=args.save_io,
        io_dir=args.io_dir,
        io_async_max_queue=args.io_async_max_queue,
        debug_imshow=args.debug_imshow,
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

