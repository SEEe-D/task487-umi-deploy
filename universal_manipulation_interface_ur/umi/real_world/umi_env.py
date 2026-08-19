from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
import threading
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
# from umi.real_world.rtde_interpolation_controller_v1 import RTDEInterpolationController
from umi.real_world.ros_target_interpolation_controller import RosTargetInterpolationController
from umi.real_world.ros_gripper_controller import RosGripperController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.livelybot_gripper_controller import LivelybotGripperController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import (
    draw_predefined_mask, 
    get_mirror_crop_slices
)
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer


def center_square_resize(frame, output_res):
    """Match the Task487 recorder: center-square crop, then INTER_AREA resize."""
    height, width = frame.shape[:2]
    size = min(height, width)
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    square = frame[y0 : y0 + size, x0 : x0 + size]
    if square.shape[:2] == (output_res[1], output_res[0]):
        return np.ascontiguousarray(square)
    return cv2.resize(square, output_res, interpolation=cv2.INTER_AREA)


def resize_with_pad(frame, output_res):
    """Match OpenPI resize_with_pad while keeping the complete camera FOV."""
    height, width = frame.shape[:2]
    output_width, output_height = output_res
    ratio = max(width / output_width, height / output_height)
    resized_width = int(width / ratio)
    resized_height = int(height / ratio)
    if (width, height) == (resized_width, resized_height):
        resized = np.ascontiguousarray(frame)
    else:
        # OpenPI uses JAX LINEAR for the second-stage model resize.
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    pad_left = (output_width - resized_width) // 2
    pad_top = (output_height - resized_height) // 2
    result = np.zeros((output_height, output_width, *frame.shape[2:]), dtype=frame.dtype)
    result[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
        ...,
    ] = resized
    return result


def normalize_per_gripper_value(value, n_grippers, name):
    """Expand a scalar calibration or validate one value per gripper."""
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return [float(values)] * n_grippers
    if values.ndim != 1 or len(values) != n_grippers:
        raise ValueError(
            f"{name} must be a scalar or contain exactly {n_grippers} values; "
            f"got shape {values.shape}"
        )
    return [float(item) for item in values]


class ThorCameraVisAdapter:
    """Thor receivers 的 get_vis() 适配器，供 MultiCameraVisualizer 消费。

    MultiCameraVisualizer 要求 camera.get_vis() 返回
    {'color': (N, H, W, 3) uint8 RGB}。
    """
    def __init__(self, thor_receivers: dict, tile_hw=(512, 640)):
        self.thor_receivers = thor_receivers
        self.tile_hw = tile_hw  # (H, W)

    def get_vis(self, out=None):
        H, W = self.tile_hw
        frames = []
        for receiver in self.thor_receivers.values():
            with receiver.lock:
                frame = receiver.frame  # BGR uint8 or None
            if frame is not None:
                frame = cv2.resize(frame, (W, H))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            frames.append(frame)
        if out is None:
            out = {}
        out['color'] = np.stack(frames, axis=0)  # (N, H, W, 3)
        return out


from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator


class UmiEnv:
    DEFAULT_TASK_INSTRUCTIONS = (
        {"task_index": 0, "task": "Pick up the blocks, and place it into the box."},
        {"task_index": 1, "task": "Fold the towel."},
        {"task_index": 2, "task": "Grasp the towel from the left side of the table."},
        {"task_index": 3, "task": "Pick up the two corners of the towel."},
        {"task_index": 4, "task": "Flatten the towel and fold it in half."},
        {"task_index": 5, "task": "Fold the towel from left to right."},
        {"task_index": 6, "task": "Put the towel on the table in the upper right corner."},
        {"task_index": 7, "task": "Grasp the towel, and fold the towel from bottom to top."},
        {"task_index": 8, "task": "Flatten the towel."},
        {"task_index": 9, "task": "Tidy up the remote comtrols."},
    )

    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            gripper_ip,
            gripper_port=1000,
            gripper_type='wsg',
            gripper_executable_path='x3arm_can/build_ws/x3arm-can-demo-gripper',
            gripper_can_if='can3',       # str or list[str], e.g. ['can3','can4']
            gripper_device_id=8,         # int or list[int], e.g. [8, 8]
            gripper_width_open_m=0.09,
            gripper_deg_open=35.0,
            gripper_deg_closed=0.0,
            gripper_open_rad=-0.61086524,
            gripper_closed_rad=0.0,
            gripper_kp=10.0,
            gripper_kd=1.0,
            gripper_target_vel_deg=0.0,
            gripper_torque_nm=0.0,
            gripper_max_speed_dps=None,
            # env params
            frequency=20,
            robot_type='ur5',
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            camera_name_mapping=None,  # e.g. {0: 0, 1: 3} -> camera0_rgb, camera3_rgb
            no_mirror=False,
            fisheye_converter=None,
            mirror_crop=False,
            mirror_swap=False,
            # timing
            align_camera_idx=0,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            task_name="block",
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset_x=0.0,
            tcp_offset_y=0.0,
            tcp_offset_z=0.0,
            tcp_rot_x=0.0,
            tcp_rot_y=0.0,
            tcp_rot_z=0.0,
            init_joints=False,
            # vis params
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            # Thor 推流相机 (替代本地USB相机)
            thor_enabled=False,
            thor_host='192.168.1.101',
            thor_receiver_path='/home/simpleai/thor-camera-stream/receiver',
            thor_cameras=None,  # list of dicts: [{"label": "cam_hand_r_top", "video_port": 5004, "meta_port": 6004}]
            thor_tile_w=640,
            thor_tile_h=512,
            thor_center_crop=True,
            thor_resize_with_pad=False,
            enable_task_ui=True,
            task_instructions=None,
            initial_task_index=0,
            task_selection_guard=None,
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # ========== Thor 推流相机 ==========
        self.thor_enabled = thor_enabled
        self.thor_center_crop = bool(thor_center_crop)
        self.thor_resize_with_pad = bool(thor_resize_with_pad)
        if self.thor_center_crop and self.thor_resize_with_pad:
            raise ValueError("thor_center_crop and thor_resize_with_pad are mutually exclusive")
        self.thor_receivers = {}  # label -> CameraReceiver
        if thor_enabled and thor_cameras:
            import sys as _sys
            _sys.path.insert(0, thor_receiver_path)
            from receiver_gi import CameraReceiver, measure_clock_offset

            print(f"[Thor] Measuring clock offset to {thor_host}...")
            clock_offset = measure_clock_offset(thor_host, port=7777)
            print(f"[Thor] Clock offset: {clock_offset:.1f}ms")

            for cam_cfg in thor_cameras:
                r = CameraReceiver(
                    camera_id=len(self.thor_receivers),
                    label=cam_cfg["label"],
                    video_port=cam_cfg["video_port"],
                    meta_port=cam_cfg["meta_port"],
                    listen_host="0.0.0.0",
                    tile_w=thor_tile_w, tile_h=thor_tile_h,
                    clock_offset_ms=clock_offset,
                    total_cameras=len(thor_cameras),
                )
                r.start()
                self.thor_receivers[cam_cfg["label"]] = r
                print(f"[Thor] {cam_cfg['label']} started (port {cam_cfg['video_port']})")
            time.sleep(3)  # 等首帧
            # 启动健康检查：若某路长时间无帧，主动触发重连。
            unhealthy = [
                r for r in self.thor_receivers.values()
                if hasattr(r, "has_recent_frame") and (not r.has_recent_frame(within_sec=1.5))
            ]
            if unhealthy:
                labels = ", ".join(r.label for r in unhealthy)
                print(f"[Thor] Startup health check reconnect: {labels}")
                for r in unhealthy:
                    if hasattr(r, "force_reconnect"):
                        r.force_reconnect(reason="umi-startup-health-check")
                time.sleep(1.0)
            print(f"[Thor] {len(self.thor_receivers)} cameras ready")

        if not thor_enabled:
            # Find and reset all Elgato capture cards.
            # Required to workaround a firmware bug.
            reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths(by_id=False) if not thor_enabled else []
        print(f"[UmiEnv] Found {len(v4l_paths)} cameras v4l_paths:")
        for i, p in enumerate(v4l_paths):
            print(f"  [{i}] {p}")
        if camera_reorder is not None and len(v4l_paths) > 0:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths
            print(f"[UmiEnv] After reorder: {len(v4l_paths)} cameras")
            for i, p in enumerate(v4l_paths):
                print(f"  [{i}] {p}")
        self.camera_name_mapping = camera_name_mapping

        # compute resolution for vis
        if len(v4l_paths) > 0:
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(v4l_paths),
                in_wh_ratio=4/3,
                max_resolution=multi_cam_vis_resolution
            )
        else:
            rw, rh, col, row = 480, 480, 1, 1

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for idx, path in enumerate(v4l_paths):
            if 'Cam_Link_4K' in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = (1600, 1200)  # USB cameras
                fps = 60
                buf = 1
                bit_rate = 3000*1000
                stack_crop = (idx==0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)

                def tf(data, input_res=res, stack_crop=stack_crop, is_mirror=is_mirror):
                    img = data['color']
                    if fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:,::-1,::-1] # bgr to rgb
                        # center crop to square then resize
                        h, w = img.shape[:2]
                        s = min(h, w)
                        y0 = (h - s) // 2
                        x0 = (w - s) // 2
                        img = img[y0:y0+s, x0:x0+s]
                        f = get_image_transform(
                            input_res=(s, s),
                            output_res=obs_image_resolution,
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0),
                            mirror=no_mirror, gripper=False, finger=False, use_aa=True)
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data):
                img = data['color']
                h, w = img.shape[:2]
                f = get_image_transform(
                    input_res=(w, h),
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        if len(v4l_paths) > 0:
            camera = MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                cap_buffer_size=cap_buffer_size,
                transform=transform,
                vis_transform=vis_transform,
                video_recorder=video_recorder,
                verbose=False
            )
        else:
            camera = None  # Thor模式无本地相机

        multi_cam_vis = None
        if enable_multi_cam_vis:
            if thor_enabled and self.thor_receivers:
                n_cams = len(self.thor_receivers)
                _col = math.ceil(math.sqrt(n_cams))
                _row = math.ceil(n_cams / _col)
                vis_cam = ThorCameraVisAdapter(
                    self.thor_receivers,
                    tile_hw=(thor_tile_h, thor_tile_w)
                )
                multi_cam_vis = MultiCameraVisualizer(
                    camera=vis_cam,
                    row=_row,
                    col=_col,
                    rgb_to_bgr=False
                )
            elif camera is not None:
                multi_cam_vis = MultiCameraVisualizer(
                    camera=camera,
                    row=row,
                    col=col,
                    rgb_to_bgr=False
                )

        robots = []
        grippers = []
        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([-75.26,-56.85,-148.33,-63.75,44.30,75.92]) / 180 * np.pi # pose 1 block

        # j_init = np.array([-175.21,-102.99,-147.23,26.62,77.72,5.80]) / 180 * np.pi

        # j_init = np.array([-171.86,-110.15,-113.99,-10.30,77.97,5.81]) / 180 * np.pi # pose 2 

        # j_init = np.array([-222.83,-95.84,-114.78,-79.92,127.06,228.62]) / 180 * np.pi

        # j_inits = [np.array([-152.15,-95.59,-132.15,6.28,92.59,9.97]) / 180 * np.pi,
        #            np.array([-29.35,-94.81,137.86,195.24,278.10,0.36]) / 180 * np.pi] # pose fold towel
        
        # j_inits = [np.array([-138.15,-96.75,-142.21,6.25,92.59,9.99]) / 180 * np.pi,
        #            np.array([-29.35,-94.81,137.86,195.24,278.10,0.36]) / 180 * np.pi] # pose fold towel stage-2

        # j_inits = [np.array([-141.24,-109.30,-126.15,6.19,68.97,14.46]) / 180 * np.pi,
        #            np.array([-20.41,-75.31,119.58,195.24,278.10,0.36]) / 180 * np.pi] # pose fold towel stage-2

        j_inits = None
        print("task_name: ", task_name)
        if task_name == 'fold_towel':
            j_inits = [np.array([-333.34,-116.44,-138.51,54.77,91.35,1.03]) / 180 * np.pi,
                   np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi] # pose fold towel stage-1
            j_inits = [np.array([30.25,-121.97,-134.01,51.05,86.50,12.00]) / 180 * np.pi,
                    np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi] # pose fold towel stage-1
        elif task_name == 'block':
            j_inits = [np.array([24.31,-112.86,-125.80,29.31,91.50,-10.0]) / 180 * np.pi,
                   np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi] # pose block
            
            # j_inits = [np.array([24.31,-112.86,-125.80,29.31,91.50,-10.0]) / 180 * np.pi,
            #        np.array([157.21,-82.05,133.56,-176.65,266.57,8.65]) / 180 * np.pi] # pose block starVLA
        else:
             j_inits = [np.array([24.31,-112.86,-125.80,29.31,91.50,-10.0]) / 180 * np.pi,
                   np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi] # pose block

        # j_inits = [np.array([24.31,-112.86,-125.80,29.31,91.50,-10.0]) / 180 * np.pi,
        #            np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi] # pose block


        print(robot_ip)
        for r_id, r_ip in enumerate(robot_ip):
            j_init = j_inits[r_id] if init_joints else None
            print("Connect robot:", r_ip, j_init)
            if r_ip == '192.168.3.244':
                tcp_offset_y = 0.028
                tcp_rot_y = 0.1745
                if init_joints:
                    j_init = np.array([160.71,-65.50,137.87,-225.70,266.28,1.10]) / 180 * np.pi
            else:
                tcp_offset_y = -0.028
                tcp_rot_y = -0.1745
            
            # tcp_offset_y = 0.028
            # tcp_rot_y = 0.1745
            
            if robot_type.startswith('ur'):
                robot = RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=r_ip,
                    frequency=300, # UR5 CB3 RTDE
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=3,
                    tcp_offset_pose=[tcp_offset_x, tcp_offset_y, tcp_offset_z, tcp_rot_x, tcp_rot_y, tcp_rot_z],
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=j_init,
                    joints_init_speed=1.05,
                    soft_real_time=False,
                    verbose=False,
                    receive_keys=None,
                    receive_latency=robot_obs_latency
                    )
            elif robot_type == 'Marvin':
                robot = RosTargetInterpolationController(
                    shm_manager=shm_manager,
                    frequency=100,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=35,
                    tcp_offset_pose=None,
                    receive_latency=robot_obs_latency,
                    arm='B' if r_id == 0 else 'A',
                    manage_fsm=(r_id == 0),
                    passive_hold=(r_id == 1),
                )
            else:
                raise ValueError(f'Unsupported robot_type: {robot_type}')
            robots.append(robot)
        
        n_grippers = len(robots)
        if gripper_ip is not None:
            n_grippers = len(gripper_ip)

        can_ifs = gripper_can_if if isinstance(gripper_can_if, (list, tuple)) \
            else [gripper_can_if] * n_grippers
        dev_ids = gripper_device_id if isinstance(gripper_device_id, (list, tuple)) \
            else [gripper_device_id] * n_grippers
        open_rads = normalize_per_gripper_value(
            gripper_open_rad, n_grippers, "gripper_open_rad")
        closed_rads = normalize_per_gripper_value(
            gripper_closed_rad, n_grippers, "gripper_closed_rad")
        open_degrees = normalize_per_gripper_value(
            gripper_deg_open, n_grippers, "gripper_deg_open")
        closed_degrees = normalize_per_gripper_value(
            gripper_deg_closed, n_grippers, "gripper_deg_closed")

        for g_idx in range(n_grippers):
            g_can = can_ifs[g_idx] if g_idx < len(can_ifs) else can_ifs[-1]
            g_did = dev_ids[g_idx] if g_idx < len(dev_ids) else dev_ids[-1]
            print(f"Connect gripper {g_idx}: can_if={g_can}, device_id={g_did}")
            if robot_type == 'Marvin':
                gripper = RosGripperController(
                    shm_manager=shm_manager,
                    frequency=100,
                    side='right' if g_idx == 0 else 'left',
                    deg_open=open_degrees[g_idx],
                    deg_closed=closed_degrees[g_idx],
                    open_rad=open_rads[g_idx],
                    closed_rad=closed_rads[g_idx],
                    receive_latency=gripper_obs_latency,
                    max_speed_deg_per_sec=gripper_max_speed_dps,
                )
            elif gripper_type == 'livelybot':
                gripper = LivelybotGripperController(
                    shm_manager=shm_manager,
                    frequency=60.0,
                    executable_path=gripper_executable_path,
                    can_if=g_can,
                    device_id=g_did,
                    receive_latency=gripper_obs_latency,
                    width_open_m=gripper_width_open_m,
                    deg_open=open_degrees[g_idx],
                    deg_closed=closed_degrees[g_idx],
                    kp=gripper_kp,
                    kd=gripper_kd,
                    target_vel_deg=gripper_target_vel_deg,
                    torque_nm=gripper_torque_nm,
                    max_speed_deg_per_sec=gripper_max_speed_dps,
                )
            else:
                raise ValueError(f'Unsupported gripper_type: {gripper_type}')

            grippers.append(gripper)

        self.camera = camera
        self.robots = robots
        self.grippers = grippers
        self.obs_image_resolution = obs_image_resolution
        self.obs_float32 = obs_float32
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.gripper_obs_key = (
            'robot0_gripper_angle' if gripper_type == 'livelybot'
            else 'robot0_gripper_width'
        )
        self.gripper_legacy_obs_key = 'robot0_gripper_width'
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self._started = False
        self._stopped = False

        instruction_source = (
            self.DEFAULT_TASK_INSTRUCTIONS
            if task_instructions is None
            else task_instructions
        )
        self.task_instructions = [dict(task) for task in instruction_source]
        if not self.task_instructions:
            raise ValueError("task_instructions must contain at least one task")
        for index, task in enumerate(self.task_instructions):
            if "task" not in task:
                raise ValueError(f"task_instructions[{index}] is missing 'task'")
            # The Tk list selection is positional.  Keep task_index aligned
            # with that position even when callers only supply key/text.
            task["task_index"] = index
        initial_task_index = int(initial_task_index)
        if initial_task_index < 0 or initial_task_index >= len(self.task_instructions):
            raise ValueError(f"Invalid initial_task_index: {initial_task_index}")
        self._task_instruction_lock = threading.Lock()
        self._task_selection_guard = task_selection_guard
        self.task_instruction = dict(self.task_instructions[initial_task_index])
        self._task_ui_stop_event = threading.Event()
        self._task_ui_thread = None
        if enable_task_ui:
            self._start_task_instruction_ui()

        

    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        if self.thor_enabled:
            ready_flag = True
            for robot in self.robots:
                ready_flag = ready_flag and robot.is_ready
            for gripper in self.grippers:
                ready_flag = ready_flag and gripper.is_ready
            return ready_flag
        
        return self.camera.is_ready and self.robots[0].is_ready and self.robots[1].is_ready and self.grippers[0].is_ready and self.grippers[1].is_ready
    
    def start(self, wait=True):
        if self._started:
            return
        self._started = True
        if not self.thor_enabled:
            self.camera.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self._stopped:
            return
        self._stopped = True
        self.end_episode()
        self._stop_task_instruction_ui()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        
        for robot in self.robots:
            if robot.pid is not None:
                robot.stop(wait=False)
        for gripper in self.grippers:
            if gripper.pid is not None:
                gripper.stop(wait=False)
        
        if not self.thor_enabled:
            self.camera.stop(wait=False)
        # Thor 清理
        for r in self.thor_receivers.values():
            r.stop()
        if wait:
            self.stop_wait()

    # ========= task instruction UI ===========
    def get_task_instruction(self):
        with self._task_instruction_lock:
            if self.task_instruction is None:
                return None
            return dict(self.task_instruction)

    def set_task_instruction(self, task_index):
        task_index = int(task_index)
        if task_index < 0 or task_index >= len(self.task_instructions):
            raise IndexError(f"Invalid task_index: {task_index}")
        if self._task_selection_guard is not None:
            try:
                allowed = bool(self._task_selection_guard())
            except Exception as exc:
                print(f"[TaskUI] task switch guard failed; selection unchanged: {exc}")
                allowed = False
            if not allowed:
                selected = self.get_task_instruction()
                print("[TaskUI] Task switch rejected: enter HOLD first")
                return selected
        with self._task_instruction_lock:
            self.task_instruction = dict(self.task_instructions[task_index])
            selected = dict(self.task_instruction)
        print(f"[TaskUI] Selected task {selected['task_index']}: {selected['task']}")
        return selected

    def _start_task_instruction_ui(self):
        if self._task_ui_thread is not None:
            return
        self._task_ui_thread = threading.Thread(
            target=self._run_task_instruction_ui,
            name="TaskInstructionUI",
            daemon=True,
        )
        self._task_ui_thread.start()

    def _stop_task_instruction_ui(self):
        self._task_ui_stop_event.set()

    def _run_task_instruction_ui(self):
        try:
            import tkinter as tk
        except Exception as exc:
            print(f"[TaskUI] tkinter unavailable, task UI disabled: {exc}")
            return

        try:
            root = tk.Tk()
            root.title("Task Instruction")
            root.geometry("520x260")

            status_var = tk.StringVar()
            listbox = tk.Listbox(root, height=len(self.task_instructions), exportselection=False)
            listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
            for task in self.task_instructions:
                listbox.insert(tk.END, f"{task['task_index']}: {task['task']}")

            status_label = tk.Label(root, textvariable=status_var, anchor="w")
            status_label.pack(fill=tk.X, padx=8, pady=(0, 8))

            def refresh_selection():
                selected_task = self.get_task_instruction()
                if selected_task is None:
                    status_var.set("Current: None")
                    return
                task_index = selected_task["task_index"]
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(task_index)
                listbox.see(task_index)
                status_var.set(f"Current {task_index}: {selected_task['task']}")

            def select_from_list(_event=None):
                selection = listbox.curselection()
                if not selection:
                    return
                self.set_task_instruction(selection[0])
                refresh_selection()

            def poll_stop():
                if self._task_ui_stop_event.is_set():
                    root.destroy()
                    return
                root.after(200, poll_stop)

            listbox.bind("<<ListboxSelect>>", select_from_list)
            root.protocol("WM_DELETE_WINDOW", self._task_ui_stop_event.set)
            refresh_selection()
            poll_stop()
            root.mainloop()
        except Exception as exc:
            print(f"[TaskUI] task UI disabled: {exc}")

    def start_wait(self):
        if not self.thor_enabled:
            self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        for robot in self.robots:
            if robot.pid is not None:
                robot.stop_wait()
        for gripper in self.grippers:
            if gripper.pid is not None:
                gripper.stop_wait()
        if not self.thor_enabled:
            self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.stop()
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # both have more than n_obs_steps data
        last_robots_data = list()
        last_grippers_data = list()
        # 125/500 hz, robot_receive_timestamp
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())
        # 30 hz, gripper_receive_timestamp
        for gripper in self.grippers:
            last_grippers_data.append(gripper.get_all_state())

        dt = 1 / self.frequency

        if self.thor_enabled and self.thor_receivers:
            # ===== Thor 模式: 用推流帧替代本地相机 =====
            import cv2 as _cv2
            align_label = "cam_head_right" if "cam_head_right" in self.thor_receivers else next(iter(self.thor_receivers))
            align_receiver = self.thor_receivers[align_label]
            latest_ts_us = int(align_receiver.meta.latest_ts_us)
            if latest_ts_us <= 0:
                raise RuntimeError(f"Thor camera {align_label} has no metadata timestamp")
            last_timestamp = latest_ts_us / 1_000_000.0 - align_receiver.meta.clock_offset_ms / 1_000.0
            camera_obs_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
            camera_obs = dict()

            # Thor label -> camera key 映射
            thor_label_to_cam = { 
                "cam_hand_r_top": "camera0_rgb",
                "cam_hand_r_btm": "camera1_rgb",
                "cam_hand_l_top": "camera2_rgb",
                "cam_hand_l_btm": "camera3_rgb",
                "cam_head_right": "camera4_rgb",
                "cam_head_left": "camera5_rgb",
            }

            for label, receiver in self.thor_receivers.items():
                with receiver.lock:
                    frame = receiver.frame  # BGR uint8 or None
                if frame is not None:
                    frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    obs_res = self.obs_image_resolution if hasattr(self, 'obs_image_resolution') else (224, 224)
                    if self.thor_center_crop:
                        frame_rgb = center_square_resize(frame_rgb, obs_res)
                    elif self.thor_resize_with_pad:
                        frame_rgb = resize_with_pad(frame_rgb, obs_res)
                    elif frame_rgb.shape[0] != obs_res[1] or frame_rgb.shape[1] != obs_res[0]:
                        frame_rgb = _cv2.resize(frame_rgb, obs_res, interpolation=_cv2.INTER_AREA)
                    if self.obs_float32:
                        frame_rgb = frame_rgb.astype(np.float32) / 255.0
                    # 复制到所有时间步
                    frames = np.stack([frame_rgb] * self.camera_obs_horizon, axis=0)
                else:
                    raise RuntimeError(f"Thor camera {label} has no decoded frame")

                cam_key = thor_label_to_cam.get(label, f"camera0_rgb")
                camera_obs[cam_key] = frames

            # 确保 camera0_rgb 存在（推理需要）
            if 'camera0_rgb' not in camera_obs and len(camera_obs) > 0:
                camera_obs['camera0_rgb'] = list(camera_obs.values())[0]

        else:
            # ===== 本地相机模式 =====
            # get data
            # 60 Hz, camera_calibrated_timestamp
            k = math.ceil(
                self.camera_obs_horizon * self.camera_down_sample_steps \
                * (60 / self.frequency))
            self.last_camera_data = self.camera.get(
                k=k,
                out=self.last_camera_data)

            last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]

            # align camera obs timestamps
            camera_obs_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
            camera_obs = dict()
            for camera_idx, value in self.last_camera_data.items():
                this_timestamps = value['timestamp']
                this_idxs = list()
                for t in camera_obs_timestamps:
                    nn_idx = np.argmin(np.abs(this_timestamps - t))
                    this_idxs.append(nn_idx)
                obs_cam_idx = camera_idx
                if self.camera_name_mapping is not None and camera_idx in self.camera_name_mapping:
                    obs_cam_idx = self.camera_name_mapping[camera_idx]
                if obs_cam_idx == 0 and self.mirror_crop:
                    camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                    camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
                else:
                    camera_obs[f'camera{obs_cam_idx}_rgb'] = value['color'][this_idxs]

        # return obs
        obs_data = dict(camera_obs)

        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ActualTCPPose'])
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_joint_interpolator = get_interp1d(
                t=last_robot_data['robot_timestamp'],
                x=last_robot_data['ActualQ'])
            robot_joint = robot_joint_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:],
                f'robot{robot_idx}_joint_pos': robot_joint,
            }
            # update obs_data
            obs_data.update(robot_obs)

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_obs = {
                f'robot{robot_idx}_gripper_angle': gripper_interpolator(gripper_obs_timestamps)
            }
            # update obs_data
            obs_data.update(gripper_obs)

        # accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )

            for robot_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][...,None]
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False,
            time_is_new=True,
            bimanual=True):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        if time_is_new:
            new_actions = actions[is_new]
            new_timestamps = timestamps[is_new]
        else:
            new_actions = actions
            new_timestamps = timestamps

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # # schedule waypoints
        # for i in range(len(new_actions)):
        #     r_actions = new_actions[i,:6]
        #     g_actions = new_actions[i,6:]

        #     # r_actions[2] = np.clip(r_actions[2], 0.0, 0.5)  # z height limit
        #     self.robot.schedule_waypoint(
        #         pose=r_actions,
        #         target_time=new_timestamps[i]-r_latency
        #     )
        #     self.gripper.schedule_waypoint(
        #         pos=g_actions,
        #         target_time=new_timestamps[i]-g_latency
        #     )
        #     print(f"Scheduled action at {new_timestamps[i]:.3f}s {r_actions}, {g_actions}")

        # schedule waypoints
        for robot_idx, (robot, gripper) in enumerate(zip(self.robots, self.grippers)):
            if robot_idx == 1 and np.shape(new_actions)[1] == 7:
                continue
            
            if robot_idx == 1 and not bimanual:
                continue

            for i in range(len(new_actions)):
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                g_actions = new_actions[i, 7 * robot_idx + 6]

                if robot_idx == 0:
                    r_actions[2] = np.clip(r_actions[2], 0.025, 0.9)  # z height limit
                else:
                    r_actions[2] = np.clip(r_actions[2], 0.025, 0.9)  # z height limit
                
                robot.schedule_waypoint(
                    pose=r_actions,
                    target_time=new_timestamps[i] - r_latency
                )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )
         
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )

    def hold(self, wait=True, timeout=None):
        """Clear both arm trajectories and command the shared Mink FSM to HOLD."""
        for robot in self.robots:
            robot.hold()

        if not wait:
            return

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        for robot_idx, robot in enumerate(self.robots):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not robot.wait_for_hold(remaining):
                raise TimeoutError(f"Robot {robot_idx} did not confirm HOLD within {timeout}s")

    def go_home(self, wait=True, timeout=None):
        """Command both arms HOME and leave the shared Mink FSM in HOLD."""
        for robot in self.robots:
            robot.go_home()

        if not wait:
            return

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        for robot_idx, robot in enumerate(self.robots):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not robot.wait_for_home(remaining):
                raise TimeoutError(f"Robot {robot_idx} did not confirm HOME within {timeout}s")
    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        if self.camera is not None:
            n_cameras = self.camera.n_cameras
            video_paths = list()
            for i in range(n_cameras):
                video_paths.append(
                    str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
            # start recording on camera
            self.camera.restart_put(start_time=start_time)
            self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        if self.obs_accumulator is None and self.action_accumulator is None:
            return
        
        # stop video recorder
        # self.camera.stop_recording()

        # # TODO
        # if self.obs_accumulator is not None:
        #     # recording
        #     assert self.action_accumulator is not None

        #     # Since the only way to accumulate obs and action is by calling
        #     # get_obs and exec_actions, which will be in the same thread.
        #     # We don't need to worry new data come in here.
        #     end_time = float('inf')
        #     for key, value in self.obs_accumulator.timestamps.items():
        #         end_time = min(end_time, value[-1])
        #     if len(self.action_accumulator.timestamps) > 0:
        #         end_time = min(end_time, self.action_accumulator.timestamps[-1])

        #     actions = self.action_accumulator.actions
        #     action_timestamps = self.action_accumulator.timestamps
        #     n_steps = 0
        #     if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
        #         n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

        #     if n_steps > 0:
        #         timestamps = action_timestamps[:n_steps]
        #         episode = {
        #             'timestamp': timestamps,
        #             'action': actions[:n_steps],
        #         }
        #         robot_pose_interpolator = PoseInterpolator(
        #             t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
        #             x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
        #         )
        #         robot_pose = robot_pose_interpolator(timestamps)
        #         episode['robot0_eef_pos'] = robot_pose[:,:3]
        #         episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
        #         joint_pos_interpolator = get_interp1d(
        #             np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
        #             np.array(self.obs_accumulator.data['robot0_joint_pos'])
        #         )
        #         joint_vel_interpolator = get_interp1d(
        #             np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
        #             np.array(self.obs_accumulator.data['robot0_joint_vel'])
        #         )
        #         episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
        #         episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

        #         gripper_interpolator = get_interp1d(
        #             t=np.array(self.obs_accumulator.timestamps[self.gripper_obs_key]),
        #             x=np.array(self.obs_accumulator.data[self.gripper_obs_key])
        #         )
        #         gripper_values = gripper_interpolator(timestamps)
        #         episode[self.gripper_obs_key] = gripper_values
        #         if self.gripper_obs_key != self.gripper_legacy_obs_key:
        #             episode[self.gripper_legacy_obs_key] = gripper_values

        #         self.replay_buffer.add_episode(episode, compressors='disk')
        #         episode_id = self.replay_buffer.n_episodes - 1
        #         print(f'Episode {episode_id} saved!')
            
            # self.obs_accumulator = None
            # self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
