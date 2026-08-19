"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

from moviepy.editor import VideoFileClip
import av
import click
import cv2
import dill
import hydra
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env import UmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import Spacemouse

class DummySpacemouse:
    """Drop-in replacement for Spacemouse when no device is available."""
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get_motion_state_transformed(self): return np.zeros(6)
    def is_button_pressed(self, button_id): return False

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
# @click.option('--robot_ip', '-ri', default='172.24.95.9')
# @click.option('--gripper_ip', '-gi', default='172.24.95.17')
@click.option('--robot_ip', default='192.168.3.254')
@click.option('--gripper_ip', default='192.168.0.27')
# @click.option('--robot_ip', default='172.16.0.3')
# @click.option('--gripper_ip', default='172.24.95.27')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='23')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=8, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=5, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('-rt', '--robot_type', default='ur5')
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--gripper_type', type=click.Choice(['wsg', 'livelybot']), default='livelybot', help='Gripper backend.')
@click.option('--gripper_executable_path', default='x3arm_can/build_ws/x3arm-can-demo-gripper', help='Path to livelybot gripper daemon executable.')
@click.option('--gripper_can_if', default='can3')
@click.option('--gripper_device_id', default=8, type=int)
@click.option('--gripper_width_open_m', default=0.09, type=float, help='Max gripper width in meters (fully open).')
@click.option('--gripper_deg_open', default=35.0, type=float, help='Gripper output angle (deg) at fully open.')
@click.option('--gripper_deg_closed', default=0.0, type=float, help='Gripper output angle (deg) at fully closed.')
@click.option('--gripper_kp', default=10.0, type=float)
@click.option('--gripper_kd', default=1.0, type=float)
@click.option('--gripper_target_vel_deg', default=0.0, type=float)
@click.option('--gripper_torque_nm', default=0.0, type=float)
@click.option('--tcp_offset_x', default=-0.016, type=float, help='TCP X offset in flange frame (meters).')
@click.option('--tcp_offset_y', default=-0.028, type=float, help='TCP Y offset in flange frame (meters).')
@click.option('--tcp_offset_z', default=0.2105, type=float, help='TCP Z offset in flange frame (meters).')
@click.option('--tcp_rot_x', default=0.0, type=float, help='TCP rotation rx in flange frame (radians).')
@click.option('--tcp_rot_y', default=-0.1745, type=float, help='TCP rotation ry in flange frame (radians).')
@click.option('--tcp_rot_z', default=0.0, type=float, help='TCP rotation rz in flange frame (radians).')
@click.option('--save_io', is_flag=True, default=False, help='Save inference IO logs into run directory.')
@click.option('--io_dir', default='/tmp/ur7e_inference_io', type=str, help='Root directory for inference IO logs.')
@click.option('--no_spacemouse', is_flag=True, default=False, help='Disable SpaceMouse; use manual dragging to position robot.')
@click.option('--thor', is_flag=True, default=False, help='Use Thor streaming camera')
@click.option('--thor_host', default='192.168.1.101')
@click.option('--thor_receiver_path', default='/home/simpleai/thor-camera-stream/receiver')
@click.option('--thor_cam_port', default=5002, type=int, help='Thor video port for wrist cam')
@click.option('--thor_meta_port', default=6002, type=int)
@click.option('--thor_tile_w', default=640, type=int)
@click.option('--thor_tile_h', default=512, type=int)
def main(input, output, robot_ip, gripper_ip,
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, command_latency,
    no_mirror, sim_fov, camera_intrinsics, robot_type,
    mirror_crop, mirror_swap,
    gripper_type, gripper_executable_path, gripper_can_if, gripper_device_id,
    gripper_width_open_m, gripper_deg_open, gripper_deg_closed,
    gripper_kp, gripper_kd, gripper_target_vel_deg, gripper_torque_nm,
    tcp_offset_x, tcp_offset_y, tcp_offset_z,
    tcp_rot_x, tcp_rot_y, tcp_rot_z,
    save_io, io_dir, no_spacemouse,
    thor, thor_host, thor_receiver_path, thor_cam_port, thor_meta_port, thor_tile_w, thor_tile_h):
    max_gripper_command = gripper_width_open_m
    gripper_speed = 0.2
    gripper_obs_key = 'robot0_gripper_width'
    if gripper_type == 'livelybot':
        max_gripper_command = gripper_deg_open
        gripper_speed = 20.0
        gripper_obs_key = 'robot0_gripper_angle'

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

    # image log directory
    run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    img_log_dir = os.path.join('logs', 'images', run_tag)
    os.makedirs(img_log_dir, exist_ok=True)
    print(f"[EVAL] Camera frames will be saved to: {img_log_dir}")

    # optional: save per-run inference metadata and event logs
    save_io = bool(save_io)
    io_run_dir = None
    inference_log_f = None
    print(f"[EVAL] IO dump setting: save_io={save_io} io_dir={io_dir}")
    if save_io:
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        try:
            io_run_dir = os.path.join(str(io_dir), run_id)
            os.makedirs(io_run_dir, exist_ok=True)
            meta = {
                "created_localtime": run_id,
                "script": "scripts_real/eval_real_umi.py",
                "ckpt_path": ckpt_path,
                "model_name": str(cfg.policy.obs_encoder.model_name),
                "dataset_path": str(cfg.task.dataset.dataset_path),
                "robot_ip": str(robot_ip),
                "robot_type": str(robot_type),
                "gripper_type": str(gripper_type),
                "steps_per_inference": int(steps_per_inference),
                "frequency": float(frequency),
                "camera_reorder": [int(x) for x in camera_reorder],
                "vis_camera_idx": int(vis_camera_idx),
                "obs_resolution_wh": [int(obs_res[0]), int(obs_res[1])],
                "sim_fov": None if sim_fov is None else float(sim_fov),
                "camera_intrinsics": camera_intrinsics,
                "note": "Per-step records are saved in inference_events.jsonl.",
            }
            meta_path = os.path.join(io_run_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            inference_log_f = open(
                os.path.join(io_run_dir, "inference_events.jsonl"),
                "a",
                encoding="utf-8"
            )
            print(f"[EVAL] IO dump enabled. run_dir={io_run_dir}")
        except Exception as e:
            save_io = False
            io_run_dir = None
            inference_log_f = None
            print(f"[EVAL][ERROR] IO dump init failed, disabled. err={e}")
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    # Thor 推流相机初始化
    thor_receiver = None
    if thor:
        import sys as _sys
        _sys.path.insert(0, thor_receiver_path)
        from receiver_gi import CameraReceiver, measure_clock_offset
        print(f"Thor: measuring clock offset to {thor_host}...")
        clock_offset = measure_clock_offset(thor_host, port=7777)
        thor_receiver = CameraReceiver(
            camera_id=0, label="thor_wrist",
            video_port=thor_cam_port, meta_port=thor_meta_port,
            listen_host="0.0.0.0",
            tile_w=thor_tile_w, tile_h=thor_tile_h,
            clock_offset_ms=clock_offset,
        )
        thor_receiver.start()
        print(f"Thor camera started (port {thor_cam_port}, {thor_tile_w}x{thor_tile_h})")
        import time as _time
        _time.sleep(3)

    print("steps_per_inference:", steps_per_inference)
    SpacemouseCls = DummySpacemouse if no_spacemouse else Spacemouse
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
                gripper_target_vel_deg=gripper_target_vel_deg,
                gripper_torque_nm=gripper_torque_nm,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                camera_name_mapping={0: 0, 1: 3},  # env camera0->model camera0(腕部), env camera1->model camera3(第三视角)
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.18,
                gripper_action_latency=0.1,
                # camera_obs_latency=0.0,
                # robot_obs_latency=0.0,
                # gripper_obs_latency=0.0,
                # robot_action_latency=0.0,
                # gripper_action_latency=0.0,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=getattr(
                    cfg.task.shape_meta.obs,
                    gripper_obs_key,
                    cfg.task.shape_meta.obs.robot0_gripper_width
                ).horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                # action
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

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        # img = VideoFileClip(str(match_video_path)).get_frame(0)

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent
            # duplicating CUDA context with ffmpeg nvenc
            # pretrained=False: no need to download base weights,
            # checkpoint will overwrite them anyway
            cfg.policy.obs_encoder.pretrained = False
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            # Coordinate frame transform: model frame <-> robot TCP frame
            # model x = robot z, model y = robot -x, model z = robot -y
            R_m2r = np.array([[0, -1,  0],
                               [0,  0, -1],
                               [1,  0,  0]], dtype=np.float64)  # model -> robot
            R_r2m = R_m2r.T  # robot -> model

            def obs_robot_to_model(obs):
                """Transform robot TCP rotation in obs to model frame.
                R_model = R_robot @ R_m2r (rearrange columns: model_x=flange_z, etc.)
                """
                obs = dict(obs)
                rot_aa = obs['robot0_eef_rot_axis_angle']  # [T, 3]
                rot_mat = st.Rotation.from_rotvec(rot_aa).as_matrix()  # [T, 3, 3]
                rot_mat_model = rot_mat @ R_m2r
                obs['robot0_eef_rot_axis_angle'] = st.Rotation.from_matrix(rot_mat_model).as_rotvec()
                return obs

            def action_model_to_robot(action):
                """Transform model frame action rotation to robot TCP frame.
                R_robot = R_model @ R_r2m (reverse of obs transform)
                """
                action = action.copy()
                rot_aa = action[:, 3:6]  # [N, 3]
                rot_mat = st.Rotation.from_rotvec(rot_aa).as_matrix()  # [N, 3, 3]
                rot_mat_robot = rot_mat @ R_r2m
                action[:, 3:6] = st.Rotation.from_matrix(rot_mat_robot).as_rotvec()
                return action

            print("Warming up policy inference")
            obs = env.get_obs()
            obs_model = obs_robot_to_model(obs)
            # use current pose as dummy start pose for warmup
            warmup_start_pose = [np.concatenate([
                obs_model['robot0_eef_pos'][-1],
                obs_model['robot0_eef_rot_axis_angle'][-1]
            ])]
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs_model, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=warmup_start_pose)
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs_model, action_pose_repr)
                action = action_model_to_robot(action)
                assert action.shape[-1] == 7
                del result

            print('Ready!')
            teach_mode_on = False
            auto_start = no_spacemouse  # --no_spacemouse 时自动跳过 human loop
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                gripper_state = env.gripper.get_state()
                gripper_target_command = gripper_state['gripper_position']
                if auto_start:
                    print("Auto-start: skipping human control, entering policy...")
                    gripper_target_command = 20.0
                    env.gripper.schedule_waypoint(gripper_target_command, target_time=time.time() + 0.5)
                    time.sleep(1.0)
                else:
                    print("  [t] toggle freedrive  [c] start policy  [q] quit")
                    print("  [m] move to episode start  [e/w] next/prev episode  [backspace] drop episode")
                key_counter.clear()
                t_start = time.monotonic()
                iter_idx = 0
                while not auto_start:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_img = obs['camera0_rgb'][-1]
                    if mirror_crop:
                        crop_img = obs['camera0_rgb_mirror_crop'][-1]
                        vis_img = np.concatenate([obs_img, crop_img, vis_img], axis=1)
                    elif match_episode_id in episode_first_frame_map:
                        vis_img = np.concatenate([obs_img, vis_img], axis=1)
                    else:
                        vis_img = obs_img
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            pos = ep['robot0_eef_pos'][0]
                            rot = ep['robot0_eef_rot_axis_angle'][0]
                            grip = ep[gripper_obs_key][0]
                            start_pose = np.concatenate([pos, rot])
                            start_grip = grip[0]
                            env.robot.servoL(start_pose, duration=duration)
                            env.gripper.schedule_waypoint(start_grip, target_time=time.time() + duration)
                            time.sleep(duration)
                            target_pose = start_pose
                            gripper_target_command = start_grip
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='t'):
                            if not teach_mode_on:
                                env.robot.teach_mode()
                                teach_mode_on = True
                                print("Teach mode ON (freedrive)")
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
                            print("Teach mode OFF (auto)")
                        # 推理前设置爪子开合度为20度
                        gripper_target_command = 20.0
                        env.gripper.schedule_waypoint(gripper_target_command, target_time=time.time() + 0.5)
                        print("Gripper set to 20 degrees")
                        time.sleep(0.5)
                        break

                    precise_wait(t_sample)
                    if teach_mode_on:
                        actual_state = env.get_robot_state()
                        target_pose = actual_state['ActualTCPPose'].copy()
                    elif no_spacemouse:
                        # Follow actual robot pose so manual dragging meets little resistance
                        actual_state = env.get_robot_state()
                        target_pose = actual_state['ActualTCPPose'].copy()
                    else:
                        # get teleop command from SpaceMouse
                        sm_state = sm.get_motion_state_transformed()
                        # print(sm_state)
                        dpos = sm_state[:3] * (0.5 / frequency)
                        drot_xyz = sm_state[3:] * (1.5 / frequency)

                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[:3] += dpos
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:])).as_rotvec()
                        target_pose[2] = np.maximum(target_pose[2], 0.055)

                        dpos = 0
                        if sm.is_button_pressed(0):
                            # close gripper
                            dpos = -gripper_speed / frequency
                        if sm.is_button_pressed(1):
                            dpos = gripper_speed / frequency
                        gripper_target_command = np.clip(
                            gripper_target_command + dpos, 0, max_gripper_command)

                    if not teach_mode_on:
                        action = np.zeros((7,))
                        action[:6] = target_pose
                        action[-1] = gripper_target_command

                        # execute teleop command
                        env.exec_actions(
                            actions=[action],
                            timestamps=[t_command_target-time.monotonic()+time.time()],
                            compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # capture episode start pose for wrt_start obs (in model frame)
                    ep_start_state = env.get_robot_state()
                    ep_start_pose = ep_start_state['ActualTCPPose'].copy()
                    ep_start_rot_model = st.Rotation.from_rotvec(ep_start_pose[3:]).as_matrix() @ R_m2r
                    ep_start_pose[3:] = st.Rotation.from_matrix(ep_start_rot_model).as_rotvec()
                    episode_start_pose = [ep_start_pose]
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # Thor: 用推流帧替换UmiEnv的相机帧
                        if thor_receiver is not None:
                            with thor_receiver.lock:
                                thor_frame = thor_receiver.frame
                            if thor_frame is not None:
                                # BGR→RGB, resize, float32 [0,1]
                                thor_rgb = cv2.cvtColor(thor_frame, cv2.COLOR_BGR2RGB)
                                obs_res_hw = obs['camera0_rgb'].shape[1:3]  # (H, W)
                                thor_resized = cv2.resize(thor_rgb, (obs_res_hw[1], obs_res_hw[0]))
                                thor_float = thor_resized.astype(np.float32) / 255.0
                                # 替换所有时间步
                                for t in range(obs['camera0_rgb'].shape[0]):
                                    obs['camera0_rgb'][t] = thor_float

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_model = obs_robot_to_model(obs)
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs_model, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs_model, action_pose_repr)
                            action = action_model_to_robot(action)
                            inference_latency = time.time() - s
                            print('Inference latency:', inference_latency)
                            # 实时打印模型输入（低维）
                            print(f'  eef_pos (t-1,t):      {obs_dict_np["robot0_eef_pos"]}')
                            print(f'  eef_rot (t-1,t):      {obs_dict_np["robot0_eef_rot_axis_angle"]}')
                            print(f'  gripper_width (t-1,t):{obs_dict_np["robot0_gripper_width"].flatten()}')
                            print(f'  rot_wrt_start (t-1,t):{obs_dict_np["robot0_eef_rot_axis_angle_wrt_start"]}')
                            print(f'  raw_action chunk ({len(raw_action)} steps, relative rot6d+gripper):')
                            for _i, _a in enumerate(raw_action):
                                print(f'    [{_i:02d}] {_a.round(4)}')
                            # save camera frame
                            frame_bgr = (obs['camera0_rgb'][-1] * 255).astype(np.uint8)[..., ::-1]
                            img_path = os.path.join(img_log_dir, f'ep{episode_id:03d}_iter{iter_idx:05d}.jpg')
                            cv2.imwrite(img_path, frame_bgr)
                        
                        # convert policy action to env actions
                        this_target_poses = action
                        # this_target_poses[:,2] = np.maximum(this_target_poses[:,2], 0.055)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                            over_budget = True
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]
                            over_budget = False

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")
                        print('------------')
                        print(f'  发给机器人的动作 ({len(this_target_poses)} steps):')
                        for _i, _a in enumerate(this_target_poses):
                            print(f'    [{_i:02d}] pos={_a[:3]}  rot={_a[3:6]}  gripper={_a[6]:.4f}')
                        current_state = env.get_robot_state()
                        eef = current_state['ActualTCPPose']
                        print(f'  末端实时位姿: pos={eef[:3]}  rot={eef[3:]}')
                        print('------------')
                        if inference_log_f is not None:
                            event = {
                                "time_sec": float(time.time()),
                                "episode_id": int(env.replay_buffer.n_episodes),
                                "iter_idx": int(iter_idx),
                                "obs_timestamp_last": float(obs_timestamps[-1]),
                                "inference_latency_sec": float(inference_latency),
                                "pred_action_steps": int(len(action)),
                                "submitted_action_steps": int(len(this_target_poses)),
                                "over_budget": bool(over_budget),
                                "run_dir": io_run_dir,
                            }
                            inference_log_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                            inference_log_f.flush()

                        # visualize: show all camera history frames
                        episode_id = env.replay_buffer.n_episodes
                        cam_keys = sorted([k for k in obs.keys() if k.endswith('_rgb') and not k.endswith('mirror_crop')])
                        cam_rows = []
                        for cam_key in cam_keys:
                            cam_name = cam_key.replace('_rgb', '')  # e.g. "camera0"
                            frames = obs[cam_key]  # [T, H, W, C]
                            labeled_frames = []
                            for t_idx, frame in enumerate(frames):
                                frame_copy = frame.copy()
                                if frame_copy.max() <= 1.0:
                                    frame_draw = (frame_copy * 255).astype(np.uint8)
                                else:
                                    frame_draw = frame_copy.astype(np.uint8)
                                label = f'{cam_name} t-{len(frames)-1-t_idx}'
                                cv2.putText(frame_draw, label, (5, 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                                labeled_frames.append(frame_draw)
                            cam_rows.append(np.concatenate(labeled_frames, axis=1))
                        # pad rows to same width
                        max_w = max(r.shape[1] for r in cam_rows)
                        padded = []
                        for r in cam_rows:
                            if r.shape[1] < max_w:
                                pad = np.zeros((r.shape[0], max_w - r.shape[1], r.shape[2]), dtype=r.dtype)
                                r = np.concatenate([r, pad], axis=1)
                            padded.append(r)
                        vis_img = np.concatenate(padded, axis=0)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(vis_img, text, (10, vis_img.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        # 等待机器人执行完毕停稳
                        # time.sleep(5)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")
    if inference_log_f is not None:
        inference_log_f.close()



# %%
if __name__ == '__main__':
    main()