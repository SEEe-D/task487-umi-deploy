"""UR7e + Pi0.5 部署 v3

v1 的直接 servoL (稳定) + TCP ring buffer 时间对齐 (修复晃动)
不使用 PoseTrajectoryInterpolator，避免双重插值冲突。

后台线程只读 TCP 记录到 ring buffer，不发 servoL。
主线程: 拍图 → 用时间对齐的 TCP 算 body delta → servoL 执行 → 等待。
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import threading
import collections

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

R_M2R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
R_R2M = R_M2R.T


def rot6d_to_matrix(rot6d):
    r1 = rot6d[:3].copy()
    r2 = rot6d[3:6].copy()
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=1)


def body_delta_to_target_tcp(action, current_tcp):
    R_robot = Rotation.from_rotvec(current_tcp[3:6]).as_matrix()
    R_current_model = R_robot @ R_M2R
    target_pos = current_tcp[:3] + R_current_model @ action[:3]
    R_delta = rot6d_to_matrix(action[3:9])
    R_target_model = R_current_model @ R_delta
    R_target_robot = R_target_model @ R_R2M
    target_rotvec = Rotation.from_matrix(R_target_robot).as_rotvec()
    if np.linalg.norm(target_rotvec - current_tcp[3:6]) > np.pi:
        R_diff = R_target_robot @ R_robot.T
        delta_rv = Rotation.from_matrix(R_diff).as_rotvec()
        target_rotvec = current_tcp[3:6] + delta_rv
    return np.concatenate([target_pos, target_rotvec])


class TcpRecorder(threading.Thread):
    """后台线程: 高频读 TCP 记录到 ring buffer, 供主线程时间戳插值"""

    def __init__(self, rtde_r, freq=50):
        super().__init__(daemon=True)
        self.rtde_r = rtde_r
        self.dt = 1.0 / freq
        self.running = True
        self._buf = collections.deque(maxlen=500)  # 50Hz * 10s
        # 初始记录
        tcp = np.array(rtde_r.getActualTCPPose())
        self._buf.append((time.time(), tcp))

    def run(self):
        while self.running:
            t0 = time.monotonic()
            try:
                tcp = np.array(self.rtde_r.getActualTCPPose())
                self._buf.append((time.time(), tcp))
            except:
                pass
            elapsed = time.monotonic() - t0
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

    def get_tcp_at_time(self, t):
        """插值获取指定时间的 TCP"""
        buf = list(self._buf)
        if len(buf) < 2:
            return buf[-1][1] if buf else np.zeros(6)
        times = np.array([b[0] for b in buf])
        poses = np.array([b[1] for b in buf])
        t_clamped = np.clip(t, times[0], times[-1])
        result = np.zeros(6)
        for dim in range(6):
            result[dim] = np.interp(t_clamped, times, poses[:, dim])
        return result

    def get_latest_tcp(self):
        return self._buf[-1][1] if self._buf else np.zeros(6)

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="UR7e + Pi0.5 v3 (direct servoL + TCP time align)")
    parser.add_argument("--robot-ip", type=str, default="192.168.3.254")
    parser.add_argument("--cam-right-id", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--server-host", type=str, default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="building blocks into box")
    parser.add_argument("--max-steps", type=int, default=9999)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--lookahead", type=float, default=0.2)
    parser.add_argument("--gain", type=int, default=100)
    parser.add_argument("--can-if", type=str, default="can3")
    parser.add_argument("--gripper-id", type=int, default=8)
    parser.add_argument("--motor-deg-open", type=float, default=-700.0)
    parser.add_argument("--motor-deg-closed", type=float, default=0.0)
    parser.add_argument("--grip-open-value", type=float, default=35.0/90.0)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--camera-latency", type=float, default=0.17)
    parser.add_argument("--steps-per-inference", type=int, default=8)
    parser.add_argument("--max-pos-step", type=float, default=0.02)
    parser.add_argument("--max-rot-step", type=float, default=0.1)
    parser.add_argument("--pos-bias", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()

    import rtde_control
    import rtde_receive

    logger.info(f"Connecting to robot {args.robot_ip}...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    rtde_c.reuploadScript()
    time.sleep(0.5)

    # 设置 TCP offset（和 DP eval_real_umi.py 一致：UMI 默认夹爪尖端偏移）
    rtde_c.setTcp([-0.016, -0.028, 0.2105, 0, -0.1745, 0])
    logger.info("TCP offset set: [-0.016, -0.028, 0.2105, 0, -0.1745, 0]")

    tcp = rtde_r.getActualTCPPose()
    logger.info(f"Connected. TCP: {[round(x,4) for x in tcp]}")

    # 夹爪
    gripper_proc = None
    if not args.no_gripper:
        gripper_proc = subprocess.Popen(
            ["./x3arm-can-demo-gripper", args.can_if, str(args.gripper_id),
             "--daemon", "--kp", "10.0", "--kd", "1.0", "--loop-hz", "100"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready = gripper_proc.stdout.readline().strip()
        if ready == "READY":
            logger.info("Gripper daemon ready")
            # 先 STOP 清故障，否则电机不响应
            gripper_proc.stdin.write("STOP\n")
            gripper_proc.stdin.flush()
            gripper_proc.stdout.readline()  # OK
            time.sleep(0.3)
            # 设到训练数据初始开度
            init_motor = -40.0
            gripper_proc.stdin.write(f"SET {init_motor:.2f}\n")
            gripper_proc.stdin.flush()
            gripper_proc.stdout.readline()  # OK
            logger.info(f"Gripper init: STOP + SET {init_motor:.0f}")
            time.sleep(1)
        else:
            logger.warning(f"Gripper daemon unexpected: {ready}")

    def gripper_set(grip_value):
        if gripper_proc is None or gripper_proc.poll() is not None:
            return
        # grip_value: 模型输出 (0=闭合, 0.389=全开)
        # 映射到电机度数: 0=闭合, -50=全开
        grip_clamped = np.clip(grip_value, 0, args.grip_open_value)
        ratio = grip_clamped / args.grip_open_value  # 0~1
        motor_deg = ratio * (-700.0)  # 0=闭, -700=全开(35°)
        gripper_proc.stdin.write(f"SET {motor_deg:.2f}\n")
        gripper_proc.stdin.flush()

    # 相机
    cam_right = cv2.VideoCapture(args.cam_right_id)
    cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cam_right.read()

    # Policy server
    from openpi_client import websocket_client_policy as ws_module
    ws_policy = ws_module.WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    logger.info("Policy server connected")

    # Warmup
    warmup_obs = {
        "cam_right": np.random.randint(256, size=(args.image_size, args.image_size, 3), dtype=np.uint8),
        "state": np.zeros(20, dtype=np.float32),
        "prompt": args.prompt,
    }
    warmup_obs["state"][3:9] = [1, 0, 0, 0, 1, 0]
    ws_policy.infer(warmup_obs)
    logger.info("Warm-up done")

    # TCP recorder (后台 50Hz 读 TCP)
    tcp_recorder = TcpRecorder(rtde_r, freq=50)
    tcp_recorder.start()
    logger.info("TCP recorder running (50Hz)")

    # 信号
    shutdown = False
    def handler(sig, frame):
        nonlocal shutdown
        if shutdown:
            sys.exit(1)
        shutdown = True
    signal.signal(signal.SIGINT, handler)

    dt = 1.0 / args.frequency
    gripper_pos = 0.02  # 初始小开度, 和 init_motor=-40 匹配
    logger.info("Starting!")
    input("Press Enter to start...")

    log_dir = "/tmp/pi05_run_log"
    os.makedirs(log_dir, exist_ok=True)

    total_infer = 0
    infer_count = 0
    step_count = 0
    for infer_idx in range(args.max_steps):
        if shutdown:
            break

        # 1. 采集图像 + 记录时间
        ret, frame = cam_right.read()
        if not ret:
            break
        obs_time = time.time()
        img_right = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_right = cv2.resize(img_right, (args.image_size, args.image_size))

        # 2. 用时间对齐的 TCP (图像实际时间 = obs_time - camera_latency)
        tcp_pose = tcp_recorder.get_tcp_at_time(obs_time - args.camera_latency)

        # 3. 构建 obs
        state = np.zeros(20, dtype=np.float32)
        state[3:9] = [1, 0, 0, 0, 1, 0]
        state[9] = gripper_pos

        # 4. 推理
        t0 = time.time()
        result = ws_policy.infer({"cam_right": img_right, "state": state, "prompt": args.prompt})
        infer_time = time.time() - t0
        total_infer += infer_time
        infer_count += 1

        actions = np.asarray(result["actions"])
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        # 5. 执行 chunk (直接 servoL, 和 v1 一致但用时间对齐的 tcp_pose)
        tcp_base = tcp_pose.copy()
        n_exec = min(args.steps_per_inference, len(actions) - 1)

        for ai in range(1, 1 + n_exec):
            if shutdown:
                break

            action_biased = actions[ai, :10].copy()
            action_biased[:3] += np.array(args.pos_bias)
            target_tcp = body_delta_to_target_tcp(action_biased, tcp_base)

            # Per-step clipping (用实时 TCP)
            tcp_now = tcp_recorder.get_latest_tcp()

            pos_diff = target_tcp[:3] - tcp_now[:3]
            pos_norm = np.linalg.norm(pos_diff)
            if pos_norm > args.max_pos_step:
                pos_diff = pos_diff / pos_norm * args.max_pos_step
            clipped_pos = tcp_now[:3] + pos_diff
            # Z 安全下限：防止撞桌面触发保护停止
            clipped_pos[2] = max(clipped_pos[2], 0.02)

            R_now = Rotation.from_rotvec(tcp_now[3:6]).as_matrix()
            R_target = Rotation.from_rotvec(target_tcp[3:6]).as_matrix()
            R_diff = R_target @ R_now.T
            angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            if angle > args.max_rot_step:
                ratio = args.max_rot_step / angle
                R_clipped = Rotation.from_rotvec(
                    Rotation.from_matrix(R_diff).as_rotvec() * ratio
                ).as_matrix() @ R_now
                clipped_rotvec = Rotation.from_matrix(R_clipped).as_rotvec()
            else:
                clipped_rotvec = target_tcp[3:6]

            servo_target = np.concatenate([clipped_pos, clipped_rotvec])
            success = rtde_c.servoL(servo_target.tolist(), 0, 0, dt, args.lookahead, args.gain)
            if not success:
                logger.error(f"servoL failed at chunk[{ai}]")
                shutdown = True
                break

            gripper_pos = float(actions[ai, 9])
            gripper_set(gripper_pos)
            step_count += 1
            time.sleep(dt)

        # 6. 日志
        if infer_idx % 5 == 0:
            avg = total_infer / max(infer_count, 1)
            logger.info(
                f"Infer {infer_idx:4d} | {infer_time:.3f}s (avg: {avg:.3f}s) | "
                f"tcp: [{tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}] | "
                f"steps: {step_count} | gripper: {gripper_pos:.2f}"
            )

        cv2.imwrite(f"{log_dir}/obs_{infer_idx:04d}.jpg", cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
        np.savez(f"{log_dir}/action_{infer_idx:04d}.npz", actions=actions, tcp=tcp_pose)

    # 清理
    logger.info(f"Done. {infer_count} inferences, {step_count} servo steps")
    tcp_recorder.stop()
    rtde_c.servoStop()
    rtde_c.disconnect()
    rtde_r.disconnect()
    cam_right.release()
    if gripper_proc is not None and gripper_proc.poll() is None:
        gripper_proc.stdin.write("QUIT\n")
        gripper_proc.stdin.flush()
        gripper_proc.wait(timeout=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)
    main()
