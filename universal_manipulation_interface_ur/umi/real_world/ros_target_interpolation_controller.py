"""RosTargetInterpolationController — 把"模型动作"变成"遥操作 VR target"的 drop-in。

背景 / 重构思路
==============
原来模型回放走天机(Marvin) SDK: ``MarvinInterpolationController`` 把策略端下发的
**末端(tip) 6D 位姿** 经 ``tcp_offset_pose`` (flange→tip) 转成 flange 位姿, 再以
mm+四元数推给厂家控制器。

新架构: 模型层 **只发左右手 target**, 底层控制全部交给 zkpan 的 mink 遥操作程序:
  * 本控制器把 tip 6D 位姿 (米 + 轴角) 发到 ROS 话题 ``/left_target`` /
    ``/right_target`` (``geometry_msgs/Pose``, 米 + 四元数 xyzw), 与 VR 节点
    ``xr_target_node`` 发的话题完全一致。
  * mink_arm_controller 订阅这两个话题, 解 IK 到 **L610/L710 夹爪末端**, 驱动电机。
  * 因为 mink 解的就是夹爪末端, 天机那套 flange→tip 的 ``tcp_offset`` (±0.028m /
    ±0.1745rad) **直接去掉** —— 模型输出的 tip 位姿原样发出去。

为什么经过 UDP bridge (不直接 rclpy)
====================================
策略端跑在 conda 的 openpi(py3.11) / starvla(py3.10) 环境里, 而 ROS2 Jazzy 的
``rclpy`` C 扩展只为 **py3.12** 编译 (ABI 不兼容, 在 py3.10/3.11 里 import 必失败)。
所以 ROS 不能在策略进程里跑。

解法: 本控制器 (策略进程, conda) 只用 **UDP** 和一个独立的桥接进程
``ros_target_bridge.py`` 通信; 桥接进程跑在 **系统 python3.12 + ROS2 Jazzy** 下
(由 replay_mink.launch.py 一起拉起), 负责真正的 ROS 收发:
  * 本控制器 → bridge: 发 target tip 6D + FSM 命令 (UDP)。
  * bridge → 本控制器: 回当前末端位姿和左/右 7 轴关节角
    (来自 mink 的 /left|right_current_pose 和 /joint_states), 分别写进
    ring buffer 的 ``ActualTCPPose`` / ``ActualQ``。
这与现有 VR 链路 (xr_target_node 也是从 UDP/TCP 读手柄位姿) 思路一致。

接口对齐
========
对外暴露与 ``MarvinInterpolationController`` 完全相同的接口:
``servoL`` / ``schedule_waypoint`` / ``get_state`` / ``get_all_state`` /
``start`` / ``stop`` / ``start_wait`` / ``stop_wait`` / ``is_ready`` /
``teach_mode`` / ``end_teach_mode`` (后两个 no-op)。
ring buffer key 名沿用 RTDE/Marvin, 所以 ``umi_env_Marvin.py`` 一行不改即可工作。

FSM
===
``manage_fsm=True`` 的实例 (默认右臂/idx0) 启动时只发 HOLD。HOME 只能由
操作员明确请求；首批模型 waypoint 预装后才发 ACTIVE。
ACTIVE 切入瞬间 mink 会把 target 锚定到当前末端 FK + 3cm anchor-release 死区,
模型第一段运动累计超 3cm 后正常接管 (已知的小启动死区)。
"""
from __future__ import annotations

import os
import time
import enum
import signal
import socket
import struct
import multiprocessing as mp

import numpy as np
import scipy.spatial.transform as st

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.action_sync_logger import BufferedCsvLogger
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.precise_sleep import precise_wait


# ── UDP bridge 协议 (与 ros_target_bridge.py 严格一致) ──
# 端口可用环境变量覆盖, 默认本机回环。
BRIDGE_HOST = os.environ.get("ROS_TARGET_BRIDGE_HOST", "127.0.0.1")
BRIDGE_INGEST_PORT = int(os.environ.get("ROS_TARGET_BRIDGE_PORT", "6010"))   # 模型→bridge
LEFT_FEEDBACK_PORT = int(os.environ.get("ROS_TARGET_LEFT_FB_PORT", "6011"))  # bridge→模型(左)
RIGHT_FEEDBACK_PORT = int(os.environ.get("ROS_TARGET_RIGHT_FB_PORT", "6012"))  # bridge→模型(右)

SIDE_LEFT, SIDE_RIGHT = 0, 1
# 模型→bridge 包: 1字节类型 + 1字节side + payload
#   b'T' target: <c B 6d>  (side, x,y,z,rx,ry,rz)  米+轴角
#   b'F' fsm   : <c B i>    (side忽略, cmd)
PKT_TARGET = b"T"
PKT_FSM = b"F"
TARGET_FMT = "<cB6d"   # 1+1+48 = 50
FSM_FMT = "<cBi"       # 1+1+4 = 6
FEEDBACK_FMT = "<13d"  # 104: bridge→模型 末端 pose6 + 当前关节 q7

# Marvin arm 'A' = 左臂, 'B' = 右臂 (与 umi_env_Marvin.py 的 r_id→arm 映射一致)
ARM_TO_SIDE = {"A": "left", "B": "right"}


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    TEACH_MODE = 3        # no-op
    END_TEACH_MODE = 4    # no-op
    GO_HOME = 5           # 重发 FSM HOLD→HOME→HOLD, 回 home 后等待下一段模型动作
    HOLD = 6              # 立即进 HOLD，并清空当前臂尚未执行的插值轨迹


def _pose6_to_xyz_quat(pose6):
    pos = np.asarray(pose6[:3], dtype=np.float64)
    quat = st.Rotation.from_rotvec(np.asarray(pose6[3:6], dtype=np.float64)).as_quat()
    return pos, quat  # quat = [x,y,z,w]


def _unpack_feedback(data):
    """解码 bridge 反馈，并拒绝旧的仅 pose6 包或非有限数值。"""
    size = struct.calcsize(FEEDBACK_FMT)
    if len(data) < size:
        return None
    values = np.asarray(struct.unpack(FEEDBACK_FMT, data[:size]), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return values[:6], values[6:]


class RosTargetInterpolationController(mp.Process):
    """发布 VR target 到 ROS (经 UDP bridge), 替代天机 SDK 直连。"""

    def __init__(self,
                 shm_manager,
                 robot_ip=None,          # 兼容签名, 不用
                 frequency=125,
                 lookahead_time=0.1,     # 兼容, 不用
                 gain=300,               # 兼容, 不用
                 max_pos_speed=0.25,
                 max_rot_speed=0.16,
                 launch_timeout=35,
                 tcp_offset_pose=None,   # ★ 接收但忽略: mink 直接解夹爪末端
                 payload_mass=None,      # 兼容, 不用
                 payload_cog=None,       # 兼容, 不用
                 joints_init=None,       # 兼容, 不用
                 joints_init_speed=1.05, # 兼容, 不用
                 soft_real_time=False,   # 兼容, 不用
                 verbose=False,
                 receive_keys=None,      # 兼容, 不用
                 receive_latency=0.0,
                 arm="B",                # 'A'=左, 'B'=右
                 side=None,              # 显式覆盖 'left'/'right'
                 manage_fsm=False,       # True 实例负责发 FSM 1→2→3
                 passive_hold=False,     # True: 无模型命令时冻结后不再刷 target (单臂任务左臂)
                 ):
        super().__init__(name=f"RosTargetController-{arm}")

        if side is None:
            side = ARM_TO_SIDE.get(arm, "right")
        assert side in ("left", "right"), f"bad side {side}"
        self.side = side
        self.side_id = SIDE_LEFT if side == "left" else SIDE_RIGHT
        self.arm = arm
        self.manage_fsm = manage_fsm
        self.passive_hold = bool(passive_hold)

        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.launch_timeout = float(launch_timeout)
        self.receive_latency = float(receive_latency)
        self.verbose = bool(verbose)

        if tcp_offset_pose is not None and self.verbose:
            print(f"[RosTargetController-{arm}] tcp_offset_pose={tcp_offset_pose} "
                  f"已忽略 (mink 解夹爪末端, 无需 flange→tip 偏置)")

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros(6, dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256,
        )

        state_example = {
            'ActualTCPPose': np.zeros(6, dtype=np.float64),
            'ActualTCPSpeed': np.zeros(6, dtype=np.float64),
            'ActualQ': np.zeros(7, dtype=np.float64),
            'ActualQd': np.zeros(7, dtype=np.float64),
            'TargetTCPPose': np.zeros(6, dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=128,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency,
        )

        self.ready_event = mp.Event()
        self._stop_event = mp.Event()
        # 上层不能靠固定 sleep 猜测命令何时执行完成。每个 controller 各自确认，
        # UmiEnv 再等待双臂全部完成。
        self.home_done_event = mp.Event()
        self.hold_done_event = mp.Event()
        self.home_done_event.set()
        self.hold_done_event.set()

    # ───────── 生命周期 ─────────
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RosTargetController-{self.arm}] started ({self.side})")

    def stop(self, wait=True):
        if self.pid is None:
            return
        self.input_queue.put({'cmd': Command.STOP.value})
        self._stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            if self.ready_event.wait(timeout=min(0.1, deadline - time.monotonic())):
                if not self.is_alive():
                    raise RuntimeError(
                        f"RosTargetController-{self.arm} exited during startup "
                        f"(exitcode={self.exitcode})")
                return
            if not self.is_alive():
                self.join(timeout=0)
                raise RuntimeError(
                    f"RosTargetController-{self.arm} exited before ready "
                    f"(exitcode={self.exitcode})")
        raise TimeoutError(
            f"RosTargetController-{self.arm} did not receive valid EE feedback "
            f"within {self.launch_timeout:.1f}s")

    def stop_wait(self):
        if self.pid is None:
            return
        self.join(timeout=5.0)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)
        if self.is_alive():
            self.kill()
            self.join(timeout=2.0)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ───────── 命令 API (与 Marvin 一致) ─────────
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert duration >= self.dt
        pose = np.asarray(pose)
        assert pose.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': float(duration),
        })

    def schedule_waypoint(self, pose, target_time):
        pose = np.asarray(pose)
        assert pose.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': float(target_time),
        })

    def teach_mode(self):
        pass

    def end_teach_mode(self):
        pass

    def hold(self):
        """进入 HOLD，并把本臂插值器重置到最新实测位姿。"""
        assert self.is_alive()
        self.hold_done_event.clear()
        self.input_queue.put({
            'cmd': Command.HOLD.value,
            'target_pose': np.zeros(6, dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
        })

    def wait_for_hold(self, timeout=None):
        return self.hold_done_event.wait(timeout)

    def go_home(self):
        """回 home 姿态并停在 HOLD；下一段模型动作再触发 ACTIVE。"""
        assert self.is_alive()
        self.home_done_event.clear()
        self.input_queue.put({
            'cmd': Command.GO_HOME.value,
            'target_pose': np.zeros(6, dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
        })

    def wait_for_home(self, timeout=None):
        return self.home_done_event.wait(timeout)

    # ───────── 状态读取 ─────────
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ───────── 子进程主体 (纯 UDP, 无 rclpy) ─────────
    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # ── 诊断: 打印生效限速 + 每 tick 最大理论跳变 (排查"突然甩动"用) ──
        # 目标每 tick 位移上限 = max_pos_speed * dt。若这个值 ≳5mm, 真机会看到台阶式甩动。
        print(f"[RosTargetController-{self.arm}] 生效限速: max_pos_speed={self.max_pos_speed:.3f} m/s "
              f"max_rot_speed={self.max_rot_speed:.3f} rad/s | dt={self.dt*1000:.1f}ms → "
              f"每 tick 最大跳变 = {self.max_pos_speed*self.dt*1000:.1f} mm")
        # 发: 模型→bridge (target/fsm)
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        bridge_addr = (BRIDGE_HOST, BRIDGE_INGEST_PORT)
        # 收: bridge→模型 (当前末端位姿)
        fb_port = LEFT_FEEDBACK_PORT if self.side == "left" else RIGHT_FEEDBACK_PORT
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.bind(("127.0.0.1", fb_port))
        recv_sock.setblocking(False)

        cur = {"pose6": None, "q7": None}
        trace_log = BufferedCsvLogger(
            f"robot_{self.side}_trace.csv",
            (
                "wall_time", "monotonic_time", "side", "active",
                "active_pending", "first_cmd_received",
                "target_x", "target_y", "target_z",
                "target_rx", "target_ry", "target_rz",
                "actual_x", "actual_y", "actual_z",
                "actual_rx", "actual_ry", "actual_rz",
            ))
        waypoint_log = BufferedCsvLogger(
            f"robot_{self.side}_waypoints.csv",
            (
                "receive_wall_time", "receive_monotonic_time", "side",
                "requested_wall_time", "converted_monotonic_time",
                "controller_curr_time", "is_future",
                "actual_tail_monotonic_time", "actual_tail_wall_time",
                "limiter_extension_s",
                "target_x", "target_y", "target_z",
                "target_rx", "target_ry", "target_rz",
            ))
        event_log = BufferedCsvLogger(
            f"robot_{self.side}_events.csv",
            ("wall_time", "monotonic_time", "side", "event", "value"))

        def _drain_feedback():
            got = False
            while True:
                try:
                    data, _ = recv_sock.recvfrom(256)
                except BlockingIOError:
                    break
                except OSError:
                    break
                decoded = _unpack_feedback(data)
                if decoded is not None:
                    cur["pose6"], cur["q7"] = decoded
                    got = True
            return got

        def _send_target(pose6):
            pkt = struct.pack(TARGET_FMT, PKT_TARGET, self.side_id,
                              float(pose6[0]), float(pose6[1]), float(pose6[2]),
                              float(pose6[3]), float(pose6[4]), float(pose6[5]))
            try:
                send_sock.sendto(pkt, bridge_addr)
            except OSError:
                pass

        def _send_fsm(cmd):
            pkt = struct.pack(FSM_FMT, PKT_FSM, 0, int(cmd))
            try:
                send_sock.sendto(pkt, bridge_addr)
            except OSError:
                pass
            event_log.write({
                "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "side": self.side,
                "event": "fsm_command",
                "value": int(cmd),
            })

        # ── 等首帧末端位姿 + 真实关节角。两者缺一时不允许用零值启动。 ──
        t_wait0 = time.monotonic()
        while ((cur["pose6"] is None or cur["q7"] is None)
               and not self._stop_event.is_set()):
            _drain_feedback()
            time.sleep(0.02)
            if time.monotonic() - t_wait0 > 30.0:
                raise RuntimeError(
                    f"RosTargetController-{self.arm} received no complete pose+joint "
                    "feedback for 30s; refusing to seed zero robot state")

        # 轨迹执行: exec_actions() 会一次塞入一个 chunk 的多个 waypoint。
        # 旧逻辑每 tick 排空队列只保留 latest, 会把 20-step chunk 合并成最后一点。
        # 这里恢复 Marvin/RTDE controller 的语义: 把所有 waypoint 插入轨迹,
        # 每个控制周期采样一次并发给 mink。过期 waypoint 不丢弃, 顺延到当前时刻后
        # 依序执行, 保证 chunk 内每个动作都参与轨迹。
        latest = cur["pose6"].copy()

        # ── FSM 启动序列 (仅 manage_fsm 实例) ── 方案A: mink 是唯一逻辑权威 ──
        # /fsm_command 只进 mink; mink 发 /fsm_state 供 marvin_bridge 跟随。
        # 时序不再靠魔法 sleep 咬死 bridge 的 HOME_DURATION_S: mink 收到 ACTIVE(3) 时
        # 若 bridge 物理归位(move_in)未完成, 会自动 defer 到 /bridge_ready=True 再进,
        # 所以这里 1→2→3 背靠背发即可 (小 sleep 只为让 mink 依序处理 HOME→HOLD)。
        #   mink:  HOME(1) -> HOLD(2) -> ACTIVE(3, gate on /bridge_ready)
        #   bridge: 跟随 /fsm_state; 见 HOME 跑 move_in, 完成回报 /bridge_ready
        def _engage_fsm(initial=False):
            """启动只进入 HOLD。★不自动 HOME，不自动 ACTIVE ★
            改为"按需门控"(见主循环): 有新模型动作 → 发 ACTIVE, 之后【保持 ACTIVE】。
            ★已去掉 chunk 结尾回 HOLD★: 进入 ACTIVE 后不再自动退回 HOLD, 空闲期靠轨迹
            插值器持续贴住最后一个 target。s→d→重开任务时, 新动作会自动把 mink 拉回
            ACTIVE。仅 manage_fsm 实例发。"""
            if not self.manage_fsm:
                return
            _send_fsm(2)
            print(f"[RosTargetController-{self.arm}] FSM 2 (HOLD) 已发 "
                  f"(启动不自动归位；按 r 才 HOME)")

        if self.manage_fsm:
            _engage_fsm(initial=True)

        # ── 主循环 (轨迹采样 → UDP target) ──
        t_start = time.monotonic()
        pose_interp = PoseTrajectoryInterpolator(
            times=np.array([t_start], dtype=np.float64),
            poses=np.asarray([latest], dtype=np.float64),
        )
        last_waypoint_time = t_start
        iter_idx = 0
        keep_running = True
        # 方案A修: 收到模型首个真实命令前, 让发出的 target 持续贴住"当前反馈"
        # (见主循环采样前), 避免 target 停在 HOME 前的旧种子; 收到首命令后置 True。
        first_cmd_received = False
        # 按需 ACTIVE 门控 (仅 manage_fsm 实例真正发 FSM):
        #   有新模型 waypoint → ACTIVE; 进入 ACTIVE 后保持, 不再在 chunk 结尾回 HOLD。
        active = False
        # 首批 waypoint 先预装并发布几拍，再切 ACTIVE。否则 manage_fsm 的右臂先收到
        # 命令就会立即切状态，而左臂仍可能停留在上个 episode 的 ROS target。
        active_pending_since = None
        ACTIVE_PRIME_S = 0.05
        # 未收到模型命令前的"保持"状态: 归位期间贴反馈(mink 仍 HOLD 会忽略), 一旦反馈
        # 稳定(已到 home)就【冻结】成固定 hold_pose 只发这个固定值。★关键★: 绝不持续贴
        # 实时反馈 —— 否则该臂进 ACTIVE 后会和 mink 形成正反馈自激环(未被模型指挥的臂被
        # 重力/整体QP耦合带着漂走, 实测左手漂 332mm)。
        hold_pose = None
        hold_frozen = False
        hold_frozen_logged = False
        stable_since = None
        prev_fb = None
        HOLD_FREEZE_MOVE = 0.002   # m/tick; 反馈每帧移动 < 2mm 视为稳定
        HOLD_FREEZE_DUR = 0.5      # s; 稳定持续这么久 → 冻结 hold_pose
        try:
            while keep_running and not self._stop_event.is_set():
                t_now = time.monotonic()

                # 1) 刷新当前末端反馈
                _drain_feedback()

                # 2) 采样轨迹并发给 bridge → /right_target|/left_target
                # passive_hold: 冻结后不再刷 target, 让 mink ACTIVE 时 anchor 到 FK +
                # idle_hold 锁关节, 避免 hold_pose 流冲 anchor / 与 QP 串扰形成自激。
                if (self.passive_hold and not first_cmd_received
                        and hold_frozen and hold_pose is not None):
                    latest = np.asarray(hold_pose, dtype=np.float64)
                else:
                    latest = pose_interp(t_now)
                    _send_target(latest)

                # 3) 写状态 ring buffer (ActualTCPPose = mink 反馈的当前末端)
                actual = (cur["pose6"] if cur["pose6"] is not None
                          else np.zeros(6, dtype=np.float64))
                t_recv = time.time()
                self.ring_buffer.put({
                    'ActualTCPPose': np.asarray(actual, dtype=np.float64),
                    'ActualTCPSpeed': np.zeros(6, dtype=np.float64),
                    'ActualQ': np.asarray(cur["q7"], dtype=np.float64),
                    'ActualQd': np.zeros(7, dtype=np.float64),
                    'TargetTCPPose': np.asarray(latest, dtype=np.float64),
                    'robot_receive_timestamp': t_recv,
                    'robot_timestamp': t_recv - self.receive_latency,
                })
                trace_log.write({
                    "wall_time": t_recv,
                    "monotonic_time": t_now,
                    "side": self.side,
                    "active": int(active),
                    "active_pending": int(active_pending_since is not None),
                    "first_cmd_received": int(first_cmd_received),
                    "target_x": latest[0],
                    "target_y": latest[1],
                    "target_z": latest[2],
                    "target_rx": latest[3],
                    "target_ry": latest[4],
                    "target_rz": latest[5],
                    "actual_x": actual[0],
                    "actual_y": actual[1],
                    "actual_z": actual[2],
                    "actual_rx": actual[3],
                    "actual_ry": actual[4],
                    "actual_rz": actual[5],
                })

                # 4) 排干队列。单个 chunk 内的 waypoint 全部保留；新 chunk 的时间戳
                #    与旧未来轨迹重叠时，按 RTDE 语义覆盖旧轨迹而不是追加到末尾。
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                got_waypoint = False   # 本 tick 是否收到新模型 waypoint (→ 触发 ACTIVE)
                for i in range(n_cmd):
                    command = {key: commands[key][i] for key in commands}
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.HOLD.value:
                        # s 停止 policy 时的原子操作：
                        # 1) 全局 FSM 退出 ACTIVE；2) 本臂未来轨迹全部丢弃；
                        # 3) 从最新实测位姿建立单点插值器，避免再次 ACTIVE 时追旧 target。
                        if self.manage_fsm:
                            _send_fsm(2)
                            print(f"[RosTargetController-{self.arm}] HOLD: FSM 2 已发，清空旧轨迹")
                        _drain_feedback()
                        reset_t = time.monotonic()
                        reset_pose = np.asarray(
                            cur["pose6"] if cur["pose6"] is not None else latest,
                            dtype=np.float64).copy()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([reset_t], dtype=np.float64),
                            poses=np.asarray([reset_pose], dtype=np.float64))
                        last_waypoint_time = reset_t
                        active = False
                        active_pending_since = None
                        first_cmd_received = False
                        hold_pose = reset_pose.copy()
                        hold_frozen = True
                        stable_since = reset_t
                        prev_fb = reset_pose.copy()
                        hold_frozen_logged = False
                        # 即使 passive_hold 随后不再刷 topic，也先把 ROS 中残留的旧任务
                        # target 覆盖为当前位姿，避免下一次 ACTIVE 短暂追旧目标。
                        _send_target(reset_pose)
                        event_log.write({
                            "wall_time": time.time(),
                            "monotonic_time": reset_t,
                            "side": self.side,
                            "event": "hold_reset_complete",
                            "value": 1,
                        })
                        got_waypoint = False
                        self.hold_done_event.set()
                        # get_all() 已把这一批命令取出；HOLD 后面的旧 waypoint 必须丢弃。
                        break
                    elif cmd == Command.GO_HOME.value:
                        # 回 home 姿态并【停在那里】(不自动重进 ACTIVE)。
                        # ★ mink 在 ACTIVE 会忽略 HOME(1), 必须先 HOLD(2); 归位后停在 HOLD。
                        # 序列: 2(退出ACTIVE) → 1(HOME, mink归home+bridge move_in) → 等归位 → 2(HOLD)。
                        # 【故意不发 3】: 之前发 3 会让模型立刻恢复, 而模型的历史/集成缓存还是
                        #   归位前的旧动作, 一恢复就把 target 平滑拉向旧远处 → 手臂飞走(实测
                        #   103-114s target y 从 -0.01 一路飘到 -0.88)。停在 HOLD 最安全, 要
                        #   继续任务由上层显式重启 policy(它会重置缓存并重新发 3)。
                        if self.manage_fsm:
                            print(f"[RosTargetController-{self.arm}] GO_HOME: 2(HOLD)→1(HOME)→等归位→2(HOLD停住)")
                            _send_fsm(2); time.sleep(0.3)
                            _send_fsm(1)                    # HOME: 物理归位
                            time.sleep(4.0)                 # 等 move_in(~3s) 归位完成
                            _send_fsm(2)                    # HOLD: 停在 home, 不自动 ACTIVE
                        else:
                            # 非 FSM 臂也必须等同一轮物理 HOME 完成后再取反馈。旧实现立即
                            # 进入“稳定检测”，而 3s 慢速 move_in 每拍通常 <2mm，会在途中
                            # 被误判稳定并冻结成上个 episode 的旧 target。
                            time.sleep(4.3)
                        _drain_feedback()
                        reset_t = time.monotonic()
                        reset_pose = np.asarray(
                            cur["pose6"] if cur["pose6"] is not None else latest,
                            dtype=np.float64).copy()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([reset_t], dtype=np.float64),
                            poses=np.asarray([reset_pose], dtype=np.float64))
                        last_waypoint_time = reset_t
                        active = False                       # 停在 HOLD; 下个模型动作再触发 ACTIVE
                        active_pending_since = None
                        # 此时 HOME 已完成，直接冻结最新反馈，不再运行可能误判慢速运动的检测。
                        first_cmd_received = False
                        hold_frozen = True; hold_pose = reset_pose.copy()
                        stable_since = reset_t; prev_fb = reset_pose.copy()
                        hold_frozen_logged = False
                        # GO_HOME 内部阻塞了约 4.3s；重置控制环时基，禁止随后无 sleep
                        # 狂跑数百拍去“追赶”旧 deadline。
                        t_start = reset_t
                        iter_idx = 0
                        # 强制把 bridge/ROS 中最后一个 target 更新为 home；左臂随后即使
                        # passive_hold 停止刷 topic，也不会残留旧任务终点。
                        _send_target(reset_pose)
                        event_log.write({
                            "wall_time": time.time(),
                            "monotonic_time": reset_t,
                            "side": self.side,
                            "event": "home_reset_complete",
                            "value": 1,
                        })
                        got_waypoint = False
                        self.home_done_event.set()
                        # 丢弃与 GO_HOME 同批取出的旧 waypoint；新 policy 必须归位完成后再发。
                        break
                    elif cmd == Command.SERVOL.value:
                        first_cmd_received = True
                        got_waypoint = True
                        target_pose = np.asarray(command['target_pose'], dtype=np.float64)
                        duration = float(command['duration'])
                        curr_time = t_now + self.dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        first_cmd_received = True
                        got_waypoint = True
                        target_pose = np.asarray(command['target_pose'], dtype=np.float64)
                        requested_wall_time = float(command['target_time'])
                        # translate global wall-clock time to monotonic time
                        target_time = (
                            time.monotonic() - time.time() + requested_wall_time)
                        curr_time = t_now + self.dt
                        receive_wall_time = time.time()
                        receive_monotonic_time = time.monotonic()
                        # Match RTDEInterpolationController semantics: do not force
                        # a new waypoint behind the previous chunk's tail. When the
                        # timestamp overlaps last_waypoint_time, schedule_waypoint()
                        # trims the old future trajectory at curr_time, so the new
                        # chunk replaces it. Overdue waypoints are ignored by the
                        # interpolator; later future waypoints in the chunk remain.
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        # Speed limiting may extend completion beyond the
                        # requested timestamp.  Future inserts must use the
                        # interpolator's real tail; remembering target_time
                        # makes each following waypoint trim/rebuild from an
                        # already stale endpoint and produces a sawtooth path.
                        last_waypoint_time = float(pose_interp.times[-1])
                        actual_tail_wall_time = (
                            receive_wall_time
                            + last_waypoint_time
                            - receive_monotonic_time
                        )
                        waypoint_log.write({
                            "receive_wall_time": receive_wall_time,
                            "receive_monotonic_time": receive_monotonic_time,
                            "side": self.side,
                            "requested_wall_time": requested_wall_time,
                            "converted_monotonic_time": target_time,
                            "controller_curr_time": curr_time,
                            "is_future": int(target_time > curr_time),
                            "actual_tail_monotonic_time": last_waypoint_time,
                            "actual_tail_wall_time": actual_tail_wall_time,
                            "limiter_extension_s": last_waypoint_time - target_time,
                            "target_x": target_pose[0],
                            "target_y": target_pose[1],
                            "target_z": target_pose[2],
                            "target_rx": target_pose[3],
                            "target_ry": target_pose[4],
                            "target_rz": target_pose[5],
                        })
                    # TEACH_MODE / END_TEACH_MODE: 忽略
                if not keep_running:
                    break

                # ── 按需 ACTIVE 门控 (仅 manage_fsm 实例发全局 FSM) ──
                #   有新模型 waypoint 且当前未 ACTIVE → 发 ACTIVE (mink 从 HOLD→ACTIVE, 重锚当前位再跟踪)。
                #     兼修 s→d→重开: go_home 后停 HOLD, 重开任务首个动作即把 mink 拉回 ACTIVE。
                #   ★已去掉"chunk 播完 → HOLD": 进入 ACTIVE 后就保持 ACTIVE, 不在每个
                #     action chunk 结尾回 HOLD (空闲时靠轨迹插值器持续贴住最后一个 target)。
                if self.manage_fsm:
                    if got_waypoint and not active and active_pending_since is None:
                        active_pending_since = t_now
                    if (not active and active_pending_since is not None
                            and t_now - active_pending_since >= ACTIVE_PRIME_S):
                        _send_fsm(3)
                        active = True
                        active_pending_since = None
                        event_log.write({
                            "wall_time": time.time(),
                            "monotonic_time": t_now,
                            "side": self.side,
                            "event": "active_after_prime",
                            "value": ACTIVE_PRIME_S,
                        })
                        print(f"[RosTargetController-{self.arm}] 双臂目标预装完成 → FSM 3 (ACTIVE)")

                # 还没收到模型命令前的"保持"逻辑 (兼顾 36-37s 旧种子问题 + 防自激漂移):
                #   归位期间(反馈在动, mink 仍 HOLD 会忽略 /target)贴住实时反馈, 使 anchor
                #   baseline = 当前 home; 一旦反馈稳定(已到位)就【冻结】hold_pose, 之后只发
                #   这个固定值。这样该臂进 ACTIVE 时跟踪的是固定 home, 不会追实时反馈自激漂移。
                if not first_cmd_received and cur["pose6"] is not None:
                    fb = np.asarray(cur["pose6"], dtype=np.float64)
                    if not hold_frozen:
                        if prev_fb is not None and np.linalg.norm(fb[:3] - prev_fb[:3]) < HOLD_FREEZE_MOVE:
                            if stable_since is None:
                                stable_since = t_now
                            elif t_now - stable_since > HOLD_FREEZE_DUR:
                                hold_frozen = True
                                if self.passive_hold and not hold_frozen_logged:
                                    hold_frozen_logged = True
                                    print(f"[RosTargetController-{self.arm}] passive_hold: "
                                          f"hold_pose 已冻结, 停止刷 /{self.side}_target")
                        else:
                            stable_since = None
                        hold_pose = fb            # 未冻结前跟随反馈(归位跟随)
                        prev_fb = fb
                    pose_interp = PoseTrajectoryInterpolator(
                        times=np.array([t_now], dtype=np.float64),
                        poses=np.asarray([hold_pose], dtype=np.float64))
                    last_waypoint_time = t_now

                # 5) 控频
                t_wait_until = t_start + (iter_idx + 1) * self.dt
                precise_wait(t_wait_until, time_func=time.monotonic)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
        finally:
            self.home_done_event.set()
            self.hold_done_event.set()
            trace_log.close()
            waypoint_log.close()
            event_log.close()
            try:
                send_sock.close()
            except Exception:
                pass
            try:
                recv_sock.close()
            except Exception:
                pass
            if self.verbose:
                print(f"[RosTargetController-{self.arm}] 子进程退出")
