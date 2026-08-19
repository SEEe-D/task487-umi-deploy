"""
Pi0.5 + UMI env 部署脚本

直接加载 Pi0.5 checkpoint 本地推理，使用 UmiEnv 的 RTDEInterpolationController
(125Hz 独立进程) 实现平滑运动控制。

用法:
    python scripts_real/eval_real_pi05.py \
        --config pi05_umi6_bb_h20 \
        --checkpoint ~/pi05-deploy/checkpoint_bb_h20_31k/31000 \
        --prompt "building blocks into box" \
        -o data_local/pi05_test \
        --no_spacemouse
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# openpi source
OPENPI_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'openpi-official')
sys.path.insert(0, os.path.join(OPENPI_DIR, 'src'))

import time
import pathlib
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import scipy.spatial.transform as st

from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env import UmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)


# ============== 坐标系映射 ==============
R_M2R = np.array([[0, -1, 0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)
R_R2M = R_M2R.T


def rot6d_to_matrix(rot6d):
    r1 = rot6d[:3].copy()
    r2 = rot6d[3:6].copy()
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=1)


def body_delta_to_absolute(action_chunk, current_tcp):
    """Pi0.5 body-frame delta → 绝对 TCP 位姿

    Args:
        action_chunk: (N, 10+) [body_dpos(3) + body_drot6d(6) + gripper(1) + ...]
        current_tcp: (6,) [x,y,z,rx,ry,rz]

    Returns:
        actions: (N, 7) [x,y,z,rx,ry,rz,gripper]
    """
    R_robot = st.Rotation.from_rotvec(current_tcp[3:6]).as_matrix()
    R_current_model = R_robot @ R_M2R

    N = len(action_chunk)
    actions = np.zeros((N, 7))

    for i in range(N):
        # 位置
        target_pos = current_tcp[:3] + R_current_model @ action_chunk[i, :3]

        # 旋转
        R_delta = rot6d_to_matrix(action_chunk[i, 3:9])
        R_target_model = R_current_model @ R_delta
        R_target_robot = R_target_model @ R_R2M
        target_rotvec = st.Rotation.from_matrix(R_target_robot).as_rotvec()

        if np.linalg.norm(target_rotvec - current_tcp[3:6]) > np.pi:
            R_diff = R_target_robot @ R_robot.T
            delta_rv = st.Rotation.from_matrix(R_diff).as_rotvec()
            target_rotvec = current_tcp[3:6] + delta_rv

        actions[i, :3] = target_pos
        actions[i, 3:6] = target_rotvec
        actions[i, 6] = action_chunk[i, 9]

    return actions


class DummySpacemouse:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get_motion_state_transformed(self): return np.zeros(6)
    def is_button_pressed(self, button_id): return False


@click.command()
@click.option('--config', default='pi05_umi6_bb_h20', help='openpi config name')
@click.option('--checkpoint', required=True, help='Checkpoint directory (e.g. .../31000)')
@click.option('--prompt', default='building blocks into box')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='192.168.3.254')
@click.option('--gripper_ip', default='192.168.0.27')
@click.option('--camera_reorder', '-cr', default='23')
@click.option('--steps_per_inference', '-si', default=8, type=int)
@click.option('--max_duration', '-md', default=120, type=float)
@click.option('--frequency', '-f', default=10, type=float, help="Inference loop frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float)
@click.option('--image_size', default=256, type=int)
@click.option('--robot_type', default='ur5')
@click.option('--no_mirror', '-nm', is_flag=True, default=False)
@click.option('--gripper_type', type=click.Choice(['wsg', 'livelybot']), default='livelybot')
@click.option('--gripper_executable_path', default='x3arm_can/build_ws/x3arm-can-demo-gripper')
@click.option('--gripper_can_if', default='can3')
@click.option('--gripper_device_id', default=8, type=int)
@click.option('--gripper_width_open_m', default=0.09, type=float)
@click.option('--gripper_deg_open', default=35.0, type=float)
@click.option('--gripper_deg_closed', default=0.0, type=float)
@click.option('--gripper_kp', default=10.0, type=float)
@click.option('--gripper_kd', default=1.0, type=float)
@click.option('--tcp_offset_x', default=-0.016, type=float)
@click.option('--tcp_offset_y', default=-0.028, type=float)
@click.option('--tcp_offset_z', default=0.2105, type=float)
@click.option('--tcp_rot_x', default=0.0, type=float)
@click.option('--tcp_rot_y', default=-0.1745, type=float)
@click.option('--tcp_rot_z', default=0.0, type=float)
@click.option('--no_spacemouse', is_flag=True, default=False)
@click.option('--init_joints', '-j', is_flag=True, default=False)
def main(config, checkpoint, prompt, output, robot_ip, gripper_ip,
         camera_reorder, steps_per_inference, max_duration,
         frequency, command_latency, image_size, robot_type,
         no_mirror, gripper_type, gripper_executable_path,
         gripper_can_if, gripper_device_id,
         gripper_width_open_m, gripper_deg_open, gripper_deg_closed,
         gripper_kp, gripper_kd,
         tcp_offset_x, tcp_offset_y, tcp_offset_z,
         tcp_rot_x, tcp_rot_y, tcp_rot_z,
         no_spacemouse, init_joints):

    max_gripper_command = gripper_deg_open if gripper_type == 'livelybot' else gripper_width_open_m
    gripper_speed = 20.0 if gripper_type == 'livelybot' else 0.2
    dt = 1 / frequency
    obs_res = (image_size, image_size)

    # 模型在 UmiEnv fork 之后加载，避免 JAX 多线程 + fork 冲突
    policy = None

    # ========== 启动 UmiEnv ==========
    SpacemouseCls = DummySpacemouse if no_spacemouse else __import__(
        'umi.real_world.spacemouse_shared_memory', fromlist=['Spacemouse']).Spacemouse

    with SharedMemoryManager() as shm_manager:
        with SpacemouseCls(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            UmiEnv(
                output_dir=output,
                robot_ip=robot_ip,
                gripper_ip=gripper_ip,
                gripper_type=gripper_type,
                gripper_executable_path=gripper_executable_path,
                gripper_can_if=gripper_can_if,
                gripper_device_id=gripper_device_id,
                gripper_width_open_m=gripper_width_open_m,
                gripper_deg_open=gripper_deg_open,
                gripper_deg_closed=gripper_deg_closed,
                gripper_kp=gripper_kp,
                gripper_kd=gripper_kd,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                camera_name_mapping={0: 0, 1: 3},
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                camera_obs_latency=0.049,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.113,
                gripper_action_latency=0.1,
                camera_obs_horizon=1,
                robot_obs_horizon=1,
                gripper_obs_horizon=1,
                no_mirror=no_mirror,
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                robot_type=robot_type,
                tcp_offset_x=tcp_offset_x,
                tcp_offset_y=tcp_offset_y,
                tcp_offset_z=tcp_offset_z,
                tcp_rot_x=tcp_rot_x,
                tcp_rot_y=tcp_rot_y,
                tcp_rot_z=tcp_rot_z,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # 在 fork 之后加载模型（避免 JAX + fork 冲突）
            print(f"Loading Pi0.5 model: config={config}, checkpoint={checkpoint}")
            from openpi.training import config as _config
            from openpi.policies import policy_config as _policy_config

            train_config = _config.get_config(config)
            policy = _policy_config.create_trained_policy(
                train_config, checkpoint, default_prompt=prompt)
            print("Model loaded!")

            # Warmup
            print("Warming up...")
            warmup_obs = {
                "cam_right": np.random.randint(256, size=(image_size, image_size, 3), dtype=np.uint8),
                "state": np.zeros(20, dtype=np.float32),
                "prompt": prompt,
            }
            warmup_obs["state"][3:9] = [1, 0, 0, 0, 1, 0]
            policy.infer(warmup_obs)
            print("Warmup done!")

            print("Ready!")
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                print("  [t] toggle freedrive  [c] start policy  [q] quit")
                teach_mode_on = False
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                gripper_state = env.gripper.get_state()
                gripper_target_command = gripper_state['gripper_position']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    obs = env.get_obs()
                    vis_img = obs['camera0_rgb'][-1]
                    if vis_img.max() <= 1.0:
                        vis_img = (vis_img * 255).astype(np.uint8)
                    cv2.imshow('default', vis_img[..., ::-1])
                    _ = cv2.pollKey()

                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True
                        elif key_stroke == KeyCode(char='t'):
                            if not teach_mode_on:
                                env.robot.teach_mode()
                                teach_mode_on = True
                                print("Teach mode ON")
                            else:
                                env.robot.end_teach_mode()
                                teach_mode_on = False
                                time.sleep(0.5)
                                state = env.get_robot_state()
                                target_pose = state['ActualTCPPose']
                                print("Teach mode OFF")

                    if start_policy:
                        if teach_mode_on:
                            env.robot.end_teach_mode()
                            teach_mode_on = False
                            time.sleep(0.5)
                            state = env.get_robot_state()
                            target_pose = state['ActualTCPPose']
                        # 初始夹爪开度
                        gripper_target_command = 20.0
                        env.gripper.schedule_waypoint(
                            gripper_target_command, target_time=time.time() + 0.5)
                        time.sleep(0.5)
                        break

                    precise_wait(t_sample)
                    if teach_mode_on or no_spacemouse:
                        actual_state = env.get_robot_state()
                        target_pose = actual_state['ActualTCPPose'].copy()
                    else:
                        sm_state = sm.get_motion_state_transformed()
                        dpos = sm_state[:3] * (0.5 / frequency)
                        drot_xyz = sm_state[3:] * (1.5 / frequency)
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[:3] += dpos
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:])).as_rotvec()
                        dpos = 0
                        if sm.is_button_pressed(0):
                            dpos = -gripper_speed / frequency
                        if sm.is_button_pressed(1):
                            dpos = gripper_speed / frequency
                        gripper_target_command = np.clip(
                            gripper_target_command + dpos, 0, max_gripper_command)

                    if not teach_mode_on:
                        action = np.zeros((7,))
                        action[:6] = target_pose
                        action[-1] = gripper_target_command
                        env.exec_actions(
                            actions=[action],
                            timestamps=[t_command_target - time.monotonic() + time.time()],
                            compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                print("Robot in control!")
                try:
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    gripper_pos = 0.32
                    iter_idx = 0
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # 获取观测
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']

                        # 当前 TCP
                        robot_state = env.get_robot_state()
                        current_tcp = robot_state['ActualTCPPose']

                        # 相机图像
                        cam_img = obs['camera0_rgb'][-1]
                        if cam_img.max() <= 1.0:
                            cam_img_uint8 = (cam_img * 255).astype(np.uint8)
                        else:
                            cam_img_uint8 = cam_img.astype(np.uint8)
                        cam_img_uint8 = cv2.resize(cam_img_uint8, (image_size, image_size))

                        # Pi0.5 输入
                        pi05_state = np.zeros(20, dtype=np.float32)
                        pi05_state[3:9] = [1, 0, 0, 0, 1, 0]
                        pi05_state[9] = gripper_pos
                        pi05_obs = {
                            "cam_right": cam_img_uint8,
                            "state": pi05_state,
                            "prompt": prompt,
                        }

                        # 推理
                        s = time.time()
                        result = policy.infer(pi05_obs)
                        action_chunk = np.asarray(result["actions"])
                        inference_latency = time.time() - s
                        print(f"Inference: {inference_latency:.3f}s, chunk: {action_chunk.shape}")

                        # body delta → 绝对 TCP
                        abs_actions = body_delta_to_absolute(
                            action_chunk[:, :10], current_tcp)

                        # 夹爪: grip(0~0.389) → 电机角度(0~35°)
                        for i in range(len(abs_actions)):
                            grip_val = action_chunk[i, 9]
                            abs_actions[i, 6] = np.clip(
                                grip_val / (gripper_deg_open / 90.0) * gripper_deg_open,
                                gripper_deg_closed, gripper_deg_open)

                        # 时间戳调度
                        action_timestamps = (np.arange(len(abs_actions), dtype=np.float64)
                            ) * (1.0 / 30.0) + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)

                        if np.sum(is_new) == 0:
                            abs_actions = abs_actions[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            print("Over budget")
                        else:
                            abs_actions = abs_actions[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # 执行
                        env.exec_actions(
                            actions=abs_actions,
                            timestamps=action_timestamps,
                            compensate_latency=True)

                        gripper_pos = float(action_chunk[-1, 9])
                        print(f"  TCP: [{current_tcp[0]:.3f},{current_tcp[1]:.3f},{current_tcp[2]:.3f}]"
                              f"  grip: {gripper_pos:.2f}  submitted: {len(abs_actions)} steps")

                        # 可视化
                        vis_img = cam_img_uint8.copy()
                        cv2.putText(vis_img,
                            f"Pi0.5 iter={iter_idx} grip={gripper_pos:.2f}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.imshow('default', vis_img[..., ::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stopped.')
                                env.end_episode()
                                raise StopIteration

                        if time.time() - eval_t_start > max_duration:
                            print("Max duration reached.")
                            env.end_episode()
                            break

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except StopIteration:
                    pass
                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()

                print("Stopped.")


if __name__ == '__main__':
    main()
