"""诊断 R_M2R 是否正确

测试方法:
1. 读取机器人 TCP pose
2. 分别用 WITH R_M2R 和 WITHOUT R_M2R 构建 state
3. 发送到 server 推理
4. 比较: 输出位置 vs 输入位置的距离
   - 如果 state 构建正确, 模型应预测小 delta, 输出位置 ≈ 输入位置
   - 如果 state 构建错误, 模型 confused, 输出位置偏离大

同时比较 state 旋转的 z-score vs 训练分布。
"""

import logging
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ============== 坐标系映射 ==============
R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)
R_R2M = R_M2R.T

# Training norm_stats (right_only checkpoint)
STATE_MEAN = np.array([-0.0365, 0.0826, 0.5422, 0.2180, 0.5630, 0.5677, 0.3011, -0.6236, 0.3530, 0.1453,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
STATE_STD = np.array([0.1531, 0.1543, 0.1149, 0.3065, 0.1945, 0.4259, 0.4893, 0.2669, 0.2919, 0.1166,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 后10D用1避免除零


def build_state_with_r_m2r(tcp_pose, gripper_pos):
    """使用 R_M2R 构建 state (当前代码的方式)"""
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = tcp_pose[:3]
    R_robot = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
    R_model = R_robot @ R_M2R
    state[3:9] = R_model[:, :2].T.flatten()
    state[9] = gripper_pos
    return state


def build_state_without_r_m2r(tcp_pose, gripper_pos):
    """不使用 R_M2R 构建 state (直接用 robot 旋转)"""
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = tcp_pose[:3]
    R_robot = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
    state[3:9] = R_robot[:, :2].T.flatten()
    state[9] = gripper_pos
    return state


def rot6d_to_rotvec(rot6d):
    r1 = rot6d[:3].copy()
    r2 = rot6d[3:6].copy()
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    return Rotation.from_matrix(R).as_rotvec()


def action_to_target_tcp_with_r2m(action, current_tcp):
    """使用 R_R2M 转换 action (当前代码的方式)"""
    target_pos = action[:3].copy()
    rotvec_model = rot6d_to_rotvec(action[3:9])
    R_target_model = Rotation.from_rotvec(rotvec_model).as_matrix()
    R_target_robot = R_target_model @ R_R2M
    target_rotvec = Rotation.from_matrix(R_target_robot).as_rotvec()
    current_rotvec = current_tcp[3:6]
    if np.linalg.norm(target_rotvec - current_rotvec) > np.pi:
        R_current = Rotation.from_rotvec(current_rotvec).as_matrix()
        R_delta = R_target_robot @ R_current.T
        delta_rotvec = Rotation.from_matrix(R_delta).as_rotvec()
        target_rotvec = current_rotvec + delta_rotvec
    return np.concatenate([target_pos, target_rotvec])


def action_to_target_tcp_without_r2m(action, current_tcp):
    """不使用 R_R2M 转换 action (直接转)"""
    target_pos = action[:3].copy()
    target_rotvec = rot6d_to_rotvec(action[3:9])
    current_rotvec = current_tcp[3:6]
    if np.linalg.norm(target_rotvec - current_rotvec) > np.pi:
        R_target = Rotation.from_rotvec(target_rotvec).as_matrix()
        R_current = Rotation.from_rotvec(current_rotvec).as_matrix()
        R_delta = R_target @ R_current.T
        delta_rotvec = Rotation.from_matrix(R_delta).as_rotvec()
        target_rotvec = current_rotvec + delta_rotvec
    return np.concatenate([target_pos, target_rotvec])


def compute_z_scores(state):
    return (state[:10] - STATE_MEAN[:10]) / STATE_STD[:10]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="192.168.3.254")
    parser.add_argument("--cam-right-id", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="building blocks into box")
    parser.add_argument("--n-infer", type=int, default=3,
                        help="推理次数 (取平均)")
    parser.add_argument("--home", action="store_true",
                        help="先移到 home pose (训练分布范围内)")
    parser.add_argument("--home-pose", type=float, nargs=6,
                        default=[0.0, -0.3, 0.45, 2.22, -2.22, 0.0],
                        help="Home pose [x,y,z,rx,ry,rz]")
    args = parser.parse_args()

    # ========== 初始化 ==========
    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    if args.home:
        import rtde_control
        rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        rtde_c.reuploadScript()
        time.sleep(0.5)
        logger.info(f"Moving to home: {args.home_pose}")
        rtde_c.moveL(args.home_pose, 0.1, 0.3)
        time.sleep(1.0)
        tcp = np.array(rtde_r.getActualTCPPose())
        logger.info(f"Home reached: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]")
        rtde_c.disconnect()

    cam = cv2.VideoCapture(args.cam_right_id)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cam.read()

    from openpi_client import websocket_client_policy as ws_module
    ws_policy = ws_module.WebsocketClientPolicy(
        host=args.server_host, port=args.server_port,
    )

    # Warm-up
    warmup_obs = {
        "cam_right": np.random.randint(256, size=(args.image_size, args.image_size, 3), dtype=np.uint8),
        "state": np.zeros(20, dtype=np.float32),
        "prompt": args.prompt,
    }
    ws_policy.infer(warmup_obs)
    logger.info("Warm-up done\n")

    # ========== 读取当前状态 ==========
    tcp_pose = np.array(rtde_r.getActualTCPPose())
    logger.info(f"TCP pose: pos=[{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}]  "
                f"rot=[{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}]")

    # 采集图像
    ret, frame = cam.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.image_size, args.image_size))

    # ========== 构建两种 state ==========
    gripper_pos = 0.0
    state_with = build_state_with_r_m2r(tcp_pose, gripper_pos)
    state_without = build_state_without_r_m2r(tcp_pose, gripper_pos)

    logger.info(f"\n{'='*60}")
    logger.info("State comparison (rot6d, indices 3:9):")
    logger.info(f"  WITH R_M2R:    {np.array2string(state_with[3:9], precision=4, suppress_small=True)}")
    logger.info(f"  WITHOUT R_M2R: {np.array2string(state_without[3:9], precision=4, suppress_small=True)}")
    logger.info(f"  Training mean: {np.array2string(STATE_MEAN[3:9], precision=4, suppress_small=True)}")
    logger.info(f"  Training std:  {np.array2string(STATE_STD[3:9], precision=4, suppress_small=True)}")

    z_with = compute_z_scores(state_with)
    z_without = compute_z_scores(state_without)
    logger.info(f"\nZ-scores (rotation 3:9):")
    logger.info(f"  WITH R_M2R:    {np.array2string(z_with[3:9], precision=2)}")
    logger.info(f"  WITHOUT R_M2R: {np.array2string(z_without[3:9], precision=2)}")
    logger.info(f"  |z| max WITH:    {np.max(np.abs(z_with[3:9])):.2f}")
    logger.info(f"  |z| max WITHOUT: {np.max(np.abs(z_without[3:9])):.2f}")
    logger.info(f"  |z| mean WITH:   {np.mean(np.abs(z_with[3:9])):.2f}")
    logger.info(f"  |z| mean WITHOUT:{np.mean(np.abs(z_without[3:9])):.2f}")

    z_pos = (tcp_pose[:3] - STATE_MEAN[:3]) / STATE_STD[:3]
    logger.info(f"\nZ-scores (position 0:3): {np.array2string(z_pos, precision=2)}")
    logger.info(f"  Position is {'IN' if np.max(np.abs(z_pos)) < 3 else 'OUT OF'} training distribution")

    # ========== 推理测试 ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"Running inference {args.n_infer}x for each variant...\n")

    for label, state_val, action_converter in [
        ("WITH R_M2R", state_with, action_to_target_tcp_with_r2m),
        ("WITHOUT R_M2R", state_without, action_to_target_tcp_without_r2m),
    ]:
        pos_diffs = []
        rot_diffs = []
        raw_actions = []

        for i in range(args.n_infer):
            # 每次重新采集图像
            ret, frame = cam.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (args.image_size, args.image_size))

            obs = {
                "cam_right": img,
                "state": state_val,
                "prompt": args.prompt,
            }
            result = ws_policy.infer(obs)
            actions = np.asarray(result["actions"])
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]

            # 只看第一个 action (最接近当前时刻)
            action = actions[0]
            raw_actions.append(action.copy())

            target_tcp = action_converter(action, tcp_pose)
            pos_diff = np.linalg.norm(target_tcp[:3] - tcp_pose[:3])
            rot_diff = np.linalg.norm(target_tcp[3:6] - tcp_pose[3:6])
            pos_diffs.append(pos_diff)
            rot_diffs.append(rot_diff)

        avg_action = np.mean(raw_actions, axis=0)
        logger.info(f"--- {label} ---")
        logger.info(f"  Raw action[0] (avg): pos=[{avg_action[0]:.4f}, {avg_action[1]:.4f}, {avg_action[2]:.4f}]")
        logger.info(f"                       rot6d=[{avg_action[3]:.4f}, {avg_action[4]:.4f}, {avg_action[5]:.4f}, "
                     f"{avg_action[6]:.4f}, {avg_action[7]:.4f}, {avg_action[8]:.4f}]")
        logger.info(f"                       gripper={avg_action[9]:.4f}")
        logger.info(f"  Input TCP pos:  [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}]")
        logger.info(f"  Action pos:     [{avg_action[0]:.4f}, {avg_action[1]:.4f}, {avg_action[2]:.4f}]")
        logger.info(f"  Pos diff (action vs TCP): {np.linalg.norm(avg_action[:3] - tcp_pose[:3]):.4f}m")
        logger.info(f"  Target TCP pos diffs: {[f'{d:.4f}' for d in pos_diffs]}")
        logger.info(f"  Target TCP rot diffs: {[f'{d:.4f}' for d in rot_diffs]}")
        logger.info(f"  Avg pos diff: {np.mean(pos_diffs):.4f}m")
        logger.info(f"  Avg rot diff: {np.mean(rot_diffs):.4f}rad")

        # 检查 action 位置 vs training state mean (理想情况: action ≈ current pos)
        action_vs_mean = np.linalg.norm(avg_action[:3] - STATE_MEAN[:3])
        tcp_vs_mean = np.linalg.norm(tcp_pose[:3] - STATE_MEAN[:3])
        logger.info(f"  Action pos dist to training mean: {action_vs_mean:.4f}m")
        logger.info(f"  TCP pos dist to training mean:    {tcp_vs_mean:.4f}m")
        logger.info(f"  (If action ≈ TCP, model is coherent; if action → training mean, model ignores input)")
        logger.info("")

    # ========== 清理 ==========
    rtde_r.disconnect()
    cam.release()
    logger.info("Done.")


if __name__ == "__main__":
    main()
