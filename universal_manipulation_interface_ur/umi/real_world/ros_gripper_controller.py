"""RosGripperController — 把模型的夹爪开合命令交给遥操作的 gripper_can_node 驱动。

重构思路 (与手臂同构)
=====================
模型层**完全不碰硬件**。原来夹爪走 LivelybotGripperController spawn 的
``x3arm-can-demo-gripper`` 二进制 (SET/GET 协议, 实测不稳: 经常不出 READY / 控制
循环卡死不喂状态 → get_obs 崩)。现在改成:
  * 命令: 模型夹爪开合值 (deg) --UDP--> ros_target_bridge --> 发
    ``/Joint79/position_command`` (右) / ``/Joint69/position_command`` (左)
    (Float64, rad) --> zkpan ``gripper_can_node`` 驱 CAN (= 录制时那个能动的驱动)。
  * 反馈: ``gripper_can_node`` 发 ``/gripper_joint_states`` (rad) --bridge--UDP-->
    本控制器, 转回 deg 写 ring buffer 的 ``gripper_position`` 供 get_obs / 模型用。

接口与 LivelybotGripperController 完全一致 (schedule_waypoint / get_all_state /
start / stop / start_wait / stop_wait / is_ready / restart_put), ring buffer key 名
也一致, 所以 umi_env_Marvin 只需把构造换掉, get_obs / exec_actions 一行不改。

单位映射 (可调)
==============
模型侧 gripper "angle" 用度: deg_closed=0(闭) .. deg_open=35(开)。
关节侧 (gripper_can_node) 用 rad；左右机械端点可能不同，由 UmiEnv 为每个
控制器分别传入 open_rad。两端线性互映 (端点对端点)。
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

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.action_sync_logger import BufferedCsvLogger
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.precise_sleep import precise_wait


# ── UDP 协议 (与 ros_target_bridge.py 严格一致) ──
BRIDGE_HOST = os.environ.get("ROS_TARGET_BRIDGE_HOST", "127.0.0.1")
BRIDGE_INGEST_PORT = int(os.environ.get("ROS_TARGET_BRIDGE_PORT", "6010"))      # 模型→bridge
LEFT_GRIP_FB_PORT = int(os.environ.get("ROS_TARGET_LEFT_GRIP_FB_PORT", "6013"))  # bridge→模型(左夹爪)
RIGHT_GRIP_FB_PORT = int(os.environ.get("ROS_TARGET_RIGHT_GRIP_FB_PORT", "6014"))  # bridge→模型(右夹爪)

SIDE_LEFT, SIDE_RIGHT = 0, 1
PKT_GRIP_CMD = b"G"          # 模型→bridge 夹爪命令: <c B d> (type, side, joint_rad)
GRIP_CMD_FMT = "<cBd"        # 1+1+8 = 10
GRIP_FB_FMT = "<d"           # 8: bridge→模型 当前夹爪关节 rad


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class RosGripperController(mp.Process):
    """发夹爪命令到 ROS (经 UDP bridge), 替代 spawn demo 二进制。drop-in 兼容 LivelybotGripperController。"""

    def __init__(self,
                 shm_manager,
                 frequency=120.0,
                 side="right",                # 'left'=Joint69(0x0A) / 'right'=Joint79(0x09)
                 # 单位映射端点
                 deg_open=35.0, deg_closed=0.0,
                 open_rad=-0.61086524, closed_rad=0.0,
                 # 兼容 LivelybotGripperController 的签名 (硬件参数全忽略)
                 executable_path=None, can_if=None, device_id=None,
                 receive_latency=0.0, width_open_m=0.09,
                 kp=10.0, kd=1.0, target_vel_deg=0.0, torque_nm=0.0,
                 max_speed_deg_per_sec=None,
                 max_pos_speed=None,
                 get_max_k=128, command_queue_size=1024,
                 launch_timeout=35.0, verbose=False,
                 ):
        super().__init__(name=f"RosGripperController-{side}")
        assert side in ("left", "right"), f"bad side {side}"
        self.side = side
        self.side_id = SIDE_LEFT if side == "left" else SIDE_RIGHT

        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency
        self.deg_open = float(deg_open)
        self.deg_closed = float(deg_closed)
        self.open_rad = float(open_rad)
        self.closed_rad = float(closed_rad)
        self.receive_latency = float(receive_latency)
        self.launch_timeout = float(launch_timeout)
        self.verbose = bool(verbose)
        # 命令限速 (deg/s); None = 不限
        self.max_speed_deg_per_sec = (
            float(max_speed_deg_per_sec) if max_speed_deg_per_sec else 1000.0)

        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=command_queue_size)

        state_example = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_target_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": 0.0,
            "gripper_measure_timestamp": time.time(),
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, examples=state_example,
            get_max_k=get_max_k, get_time_budget=0.2,
            put_desired_frequency=self.frequency)

        self.ready_event = mp.Event()
        self._stop_event = mp.Event()

    # ── 单位互映 (参考 LivelybotGripperController: 两端都夹到 [deg_closed, deg_open]) ──
    def _clip_gripper_deg(self, gripper_deg):
        low = min(self.deg_closed, self.deg_open)
        high = max(self.deg_closed, self.deg_open)
        return float(np.clip(gripper_deg, low, high))

    def _deg_to_rad(self, deg):
        # 命令端: 先把目标夹到物理量程再线性映射到关节 rad (对齐 livelybot 的
        # _gripper_deg_to_motor_deg, 防止插值过冲/越界命令打到硬件)。
        deg = self._clip_gripper_deg(deg)
        denom = (self.deg_open - self.deg_closed) or 1e-9
        return self.closed_rad + (self.open_rad - self.closed_rad) * (
            (float(deg) - self.deg_closed) / denom)

    def _rad_to_deg(self, rad):
        # 反馈端: 线性映射回 deg 后夹到物理量程 (对齐 livelybot 的
        # _motor_deg_to_gripper_deg, 滤掉标定误差/噪声导致的越界读数)。
        denom = (self.open_rad - self.closed_rad) or 1e-9
        deg = self.deg_closed + (self.deg_open - self.deg_closed) * (
            (float(rad) - self.closed_rad) / denom)
        return self._clip_gripper_deg(deg)

    # ── 生命周期 ──
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RosGripper-{self.side}] started")

    def stop(self, wait=True):
        if self.pid is None:
            return
        self.input_queue.put({"cmd": Command.SHUTDOWN.value})
        self._stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            if self.ready_event.wait(timeout=min(0.1, deadline - time.monotonic())):
                if not self.is_alive():
                    raise RuntimeError(
                        f"RosGripper-{self.side} exited during startup "
                        f"(exitcode={self.exitcode})")
                return
            if not self.is_alive():
                self.join(timeout=0)
                raise RuntimeError(
                    f"RosGripper-{self.side} exited before ready "
                    f"(exitcode={self.exitcode})")
        raise TimeoutError(
            f"RosGripper-{self.side} did not receive valid feedback "
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

    # ── 命令 API (与 LivelybotGripperController 一致) ──
    def schedule_waypoint(self, pos: float, target_time: float):
        self.input_queue.put({
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": float(pos),
            "target_time": float(target_time),
        })

    def restart_put(self, start_time):
        self.input_queue.put({
            "cmd": Command.RESTART_PUT.value,
            "target_pos": 0.0,
            "target_time": float(start_time),
        })

    # ── 状态读取 ──
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ── 子进程主体 (纯 UDP) ──
    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        bridge_addr = (BRIDGE_HOST, BRIDGE_INGEST_PORT)
        fb_port = LEFT_GRIP_FB_PORT if self.side == "left" else RIGHT_GRIP_FB_PORT
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.bind(("127.0.0.1", fb_port))
        recv_sock.setblocking(False)

        # 最新反馈: deg=当前夹爪角, t=该帧到达 monotonic 时刻, vel=deg/s。
        # (对齐 livelybot 上报真实 gripper_velocity: 这里对反馈位置做差分 + 轻度 EMA,
        #  因为 UDP 反馈里没有速度字段, 只能自己算。)
        cur = {"deg": None, "t": None, "vel": 0.0}
        trace_log = BufferedCsvLogger(
            f"gripper_{self.side}_trace.csv",
            (
                "wall_time", "monotonic_time", "side",
                "target_deg", "actual_deg", "actual_velocity_deg_s",
                "target_minus_actual_deg",
            ))
        waypoint_log = BufferedCsvLogger(
            f"gripper_{self.side}_waypoints.csv",
            (
                "receive_wall_time", "receive_monotonic_time", "side",
                "requested_wall_time", "converted_monotonic_time",
                "controller_curr_time", "is_future", "target_deg",
            ))

        def _drain_feedback():
            sz = struct.calcsize(GRIP_FB_FMT)
            while True:
                try:
                    data, _ = recv_sock.recvfrom(64)
                except (BlockingIOError, OSError):
                    break
                if len(data) >= sz:
                    (joint_rad,) = struct.unpack(GRIP_FB_FMT, data[:sz])
                    new_deg = self._rad_to_deg(joint_rad)
                    t_fb = time.monotonic()
                    if cur["deg"] is not None and cur["t"] is not None:
                        dt_fb = t_fb - cur["t"]
                        if dt_fb > 1e-6:
                            inst_vel = (new_deg - cur["deg"]) / dt_fb
                            # EMA(α=0.5): 反馈按帧到达, 纯差分抖, 轻度平滑
                            cur["vel"] = 0.5 * cur["vel"] + 0.5 * inst_vel
                    cur["deg"] = new_deg
                    cur["t"] = t_fb

        def _send_cmd_deg(deg):
            joint_rad = self._deg_to_rad(deg)
            pkt = struct.pack(GRIP_CMD_FMT, PKT_GRIP_CMD, self.side_id, float(joint_rad))
            try:
                send_sock.sendto(pkt, bridge_addr)
            except OSError:
                pass

        # 等首帧夹爪反馈 (gripper_can_node + bridge 起来才有)
        t_wait0 = time.monotonic()
        while cur["deg"] is None and not self._stop_event.is_set():
            _drain_feedback()
            time.sleep(0.02)
            if time.monotonic() - t_wait0 > 30.0:
                raise RuntimeError(
                    f"RosGripper-{self.side} received no feedback for 30s; "
                    "refusing to seed a synthetic closed position")

        # 用当前夹爪 deg 种子初始化插值器 (1-DOF 塞进 6D pose)
        curr_t = time.monotonic()
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t], poses=[[float(cur["deg"]), 0, 0, 0, 0, 0]])
        last_wp_time = curr_t

        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        try:
            while keep_running and not self._stop_event.is_set():
                t_now = time.monotonic()
                _drain_feedback()

                # 1) 发命令 = 插值器当前目标 deg → 夹到量程 → joint rad → bridge
                #    (对齐 livelybot: 读出插值目标后立刻 clip 到 [deg_closed, deg_open])
                target_deg = self._clip_gripper_deg(pose_interp(t_now)[0])

                # # (可选) 夹紧系数: 历史实验, 现关闭
                # # StarVLA Task485 small black
                # if target_deg > 11.0:
                #     target_deg = target_deg * 1.3
                # elif target_deg > 6.6:
                #     target_deg = target_deg * 0.95
                # else:
                #     target_deg = target_deg * 0.95

                # # StarVLA Task485 white
                # if target_deg > 11.5:
                #     target_deg = target_deg * 1.3
                # elif target_deg > 6.6:
                #     target_deg = target_deg * 0.95
                # else:
                #     target_deg = target_deg * 0.95

                # if target_deg > 11.0:
                #     target_deg = target_deg * 1.0
                # elif target_deg > 6.6:
                #     target_deg = target_deg * 0.95
                # else:
                #     target_deg = target_deg * 0.95

                # # StarVLA Task487
                # if target_deg > 12.5:
                #     target_deg = target_deg * 1.1
                # elif target_deg > 6.6:
                #     target_deg = target_deg * 0.95
                # else:
                #     target_deg = target_deg * 0.95

                # # Task-483 Pick Blocks
                # if target_deg > 6.0:
                #     target_deg = target_deg * 1.0
                # else:
                #     target_deg = target_deg * 0.5

                # # Task 487 PI05
                # if target_deg > 12.5:
                #     target_deg = target_deg * 1.1
                # elif target_deg > 11:
                #     target_deg = target_deg * 0.85
                # else:
                #     target_deg = target_deg * 0.8

                # # Task 487-Tele PI05
                # if target_deg > 12.5:
                #     target_deg = target_deg * 1.0
                # elif target_deg > 11:
                #     target_deg = target_deg * 0.8
                
                _send_cmd_deg(target_deg)

                # 2) 写状态 (position = 反馈当前 deg; velocity = 反馈位置差分, 对齐 livelybot)
                has_fb = cur["deg"] is not None
                pos_deg = float(cur["deg"]) if has_fb else self.deg_closed
                vel_deg = float(cur["vel"]) if has_fb else 0.0
                t_recv = time.time()
                self.ring_buffer.put({
                    "gripper_state": 0,
                    "gripper_position": pos_deg,
                    "gripper_target_position": target_deg,
                    "gripper_velocity": vel_deg,
                    # UDP 反馈只有关节 rad, 无力矩 → 保持 0 (需 bridge 扩展协议才能上报真实 force)
                    "gripper_force": 0.0,
                    "gripper_measure_timestamp": t_recv,
                    "gripper_receive_timestamp": t_recv,
                    "gripper_timestamp": t_recv - self.receive_latency,
                })
                trace_log.write({
                    "wall_time": t_recv,
                    "monotonic_time": t_now,
                    "side": self.side,
                    "target_deg": target_deg,
                    "actual_deg": pos_deg,
                    "actual_velocity_deg_s": vel_deg,
                    "target_minus_actual_deg": target_deg - pos_deg,
                })

                # 3) 取命令
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    command = {k: v[i] for k, v in commands.items()}
                    c = command["cmd"]
                    if c == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif c == Command.SCHEDULE_WAYPOINT.value:
                        wp = self._clip_gripper_deg(command["target_pos"])
                        requested_wall_time = float(command["target_time"])
                        wp_t = (time.monotonic() - time.time()
                                + requested_wall_time)
                        waypoint_log.write({
                            "receive_wall_time": time.time(),
                            "receive_monotonic_time": time.monotonic(),
                            "side": self.side,
                            "requested_wall_time": requested_wall_time,
                            "converted_monotonic_time": wp_t,
                            "controller_curr_time": t_now,
                            "is_future": int(wp_t > t_now),
                            "target_deg": wp,
                        })
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[wp, 0, 0, 0, 0, 0], time=wp_t,
                            max_pos_speed=self.max_speed_deg_per_sec,
                            max_rot_speed=self.max_speed_deg_per_sec,
                            curr_time=t_now, last_waypoint_time=last_wp_time)
                        last_wp_time = wp_t
                    elif c == Command.RESTART_PUT.value:
                        t_start = (float(command["target_time"])
                                   - time.time() + time.monotonic())
                        iter_idx = 1

                # 4) 控频 + ready
                t_wait_until = t_start + (iter_idx + 1) * self.dt
                precise_wait(t_wait_until, time_func=time.monotonic)
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
        finally:
            trace_log.close()
            waypoint_log.close()
            try:
                send_sock.close()
            except Exception:
                pass
            try:
                recv_sock.close()
            except Exception:
                pass
            if self.verbose:
                print(f"[RosGripper-{self.side}] 子进程退出")
