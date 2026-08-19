"""
LivelybotGripperController — drop-in replacement for WSGController.

Talks to `x3arm-can-demo-gripper --daemon` via stdin/stdout.
Exposes the same shared-memory ring-buffer / command-queue API that UmiEnv expects.
"""
import enum
import multiprocessing as mp
import subprocess
import time
import traceback

import numpy as np

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class LivelybotGripperController(mp.Process):
    def __init__(
        self,
        shm_manager,
        executable_path: str,
        can_if: str = "can3",
        device_id: int = 8,
        frequency: float = 30.0,
        get_max_k=None,
        command_queue_size: int = 1024,
        launch_timeout: float = 5.0,
        receive_latency: float = 0.0,
        width_open_m: float = 0.09,
        deg_open: float = 35.0,
        deg_closed: float = 0.0,
        kp: float = 10.0,
        kd: float = 1.0,
        target_vel_deg: float = 0.0,
        torque_nm: float = 0.0,
        daemon_loop_hz: float = 100.0,
        max_speed_deg_per_sec: float = 100.0,
        verbose: bool = False,
    ):
        super().__init__(name="LivelybotGripperController")
        self.executable_path = executable_path
        self.can_if = can_if
        self.device_id = int(device_id)
        self.frequency = float(frequency)
        self.launch_timeout = float(launch_timeout)
        self.receive_latency = float(receive_latency)
        self.width_open_m = float(width_open_m)
        self.gripper_deg_open = float(deg_open)
        self.gripper_deg_closed = float(deg_closed)
        self.motor_deg_open = -700.0
        self.motor_deg_closed = 0.0
        self.kp = float(kp)
        self.kd = float(kd)
        self.target_vel_deg = float(target_vel_deg)
        self.torque_nm = float(torque_nm)
        self.daemon_loop_hz = float(daemon_loop_hz)
        full_range  = abs(self.gripper_deg_open - self.gripper_deg_closed)
        self.max_pos_speed = float(max_speed_deg_per_sec) if max_speed_deg_per_sec is not None else full_range * 1.0
        self.verbose = bool(verbose)

        if get_max_k is None:
            get_max_k = int(self.frequency * 10)

        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size,
        )

        example_state = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": 0.0,
            "gripper_measure_timestamp": time.time(),
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency,
        )
        self.ready_event = mp.Event()

    def _clip_gripper_deg(self, gripper_deg: float) -> float:
        low = min(self.gripper_deg_closed, self.gripper_deg_open)
        high = max(self.gripper_deg_closed, self.gripper_deg_open)
        return float(np.clip(gripper_deg, low, high))

    def _gripper_deg_to_motor_deg(self, gripper_deg: float) -> float:
        clipped = self._clip_gripper_deg(gripper_deg)
        denom = self.gripper_deg_open - self.gripper_deg_closed
        if abs(denom) < 1e-9:
            return self.motor_deg_closed
        ratio = (clipped - self.gripper_deg_closed) / denom
        return self.motor_deg_closed + (
            self.motor_deg_open - self.motor_deg_closed
        ) * ratio

    def _motor_deg_to_gripper_deg(self, motor_deg: float) -> float:
        denom = self.motor_deg_open - self.motor_deg_closed
        if abs(denom) < 1e-9:
            return self.gripper_deg_closed
        ratio = (float(motor_deg) - self.motor_deg_closed) / denom
        gripper_deg = self.gripper_deg_closed + (
            self.gripper_deg_open - self.gripper_deg_closed
        ) * ratio
        return self._clip_gripper_deg(gripper_deg)

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.input_queue.put({"cmd": Command.SHUTDOWN.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pos: float, target_time: float):
        self.input_queue.put({
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": float(pos),
            "target_time": float(target_time),
        })

    def restart_put(self, start_time):
        self.input_queue.put({
            "cmd": Command.RESTART_PUT.value,
            "target_time": float(start_time),
        })

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    @staticmethod
    def _readline(proc: subprocess.Popen) -> str:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("gripper daemon stdout closed unexpectedly")
        return line.strip()

    @staticmethod
    def _send_cmd(proc: subprocess.Popen, cmd: str):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def run(self):
        proc = None
        try:
            cmd = [
                self.executable_path,
                self.can_if,
                str(self.device_id),
                "--daemon",
                "--kp", str(self.kp),
                "--kd", str(self.kd),
                "--target-vel-deg", str(self.target_vel_deg),
                "--torque", str(self.torque_nm),
                "--loop-hz", str(self.daemon_loop_hz),
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            while True:
                line = self._readline(proc)
                if self.verbose:
                    print(f"[LivelybotGripper] {line}")
                if line == "READY":
                    break

            self._send_cmd(proc, "GET")
            state_line = self._readline(proc)
            if not state_line.startswith("STATE "):
                raise RuntimeError(
                    f"unexpected daemon response during init: {state_line}")
            init_parts = state_line.split()
            init_motor_deg = float(init_parts[1])
            init_gripper_deg = self._motor_deg_to_gripper_deg(init_motor_deg)

            now = time.monotonic()
            last_waypoint_time = now
            pose_interp = PoseTrajectoryInterpolator(
                times=[now],
                poses=[[init_gripper_deg, 0, 0, 0, 0, 0]],
            )

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                t_now = time.monotonic()
                dt = 1.0 / self.frequency
                target_gripper_deg = self._clip_gripper_deg(pose_interp(t_now)[0])
                
                target_deg = self._gripper_deg_to_motor_deg(target_gripper_deg)

                self._send_cmd(proc, f"SET {target_deg:.6f}")
                _ = self._readline(proc)
                self._send_cmd(proc, "GET")
                state_line = self._readline(proc)
                if not state_line.startswith("STATE "):
                    raise RuntimeError(
                        f"unexpected daemon response: {state_line}")

                parts = state_line.split()
                pos_deg = float(parts[1])
                vel_dps = float(parts[2])
                tq_nm = float(parts[3])
                pos_gripper_deg = self._motor_deg_to_gripper_deg(pos_deg)
                vel_gripper_dps = (
                    self._motor_deg_to_gripper_deg(pos_deg + vel_dps * dt)
                    - pos_gripper_deg
                ) / max(dt, 1e-6)
                t_recv = time.time()

                state = {
                    "gripper_state": 0,
                    "gripper_position": pos_gripper_deg,
                    "gripper_velocity": vel_gripper_dps,
                    "gripper_force": tq_nm,
                    "gripper_measure_timestamp": t_recv,
                    "gripper_receive_timestamp": t_recv,
                    "gripper_timestamp": t_recv - self.receive_latency,
                }
                self.ring_buffer.put(state)

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
                    if c == Command.SCHEDULE_WAYPOINT.value:
                        wp = self._clip_gripper_deg(command["target_pos"])
                        wp_t = (time.monotonic() - time.time()
                                + float(command["target_time"]))
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[wp, 0, 0, 0, 0, 0],
                            time=wp_t,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_pos_speed,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = wp_t
                    elif c == Command.RESTART_PUT.value:
                        t_start = (float(command["target_time"])
                                   - time.time() + time.monotonic())
                        iter_idx = 1

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            self.ready_event.set()
            if proc is not None:
                try:
                    self._send_cmd(proc, "QUIT")
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
