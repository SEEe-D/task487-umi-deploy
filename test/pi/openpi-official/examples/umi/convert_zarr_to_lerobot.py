"""convert_zarr_to_lerobot.py — UMI zarr (replay_buffer) → LeRobot v2

输入: UMI SLAM pipeline 输出的 zarr.zip (07_generate_replay_buffer.py 生成)
  - data/robot0_eef_pos: (N, 3) xyz 绝对位姿 (tag frame)
  - data/robot0_eef_rot_axis_angle: (N, 3) rotvec 绝对旋转 (tag frame)
  - data/robot0_gripper_width: (N, 1) 夹爪宽度 (mm)
  - data/camera0_rgb: (N, 224, 224, 3) 腕部相机
  - data/camera3_rgb: (N, 224, 224, 3) 头部相机 (可选)
  - meta/episode_ends: (num_episodes,) episode 结束索引

输出: LeRobot v2 (绝对 EE pose, 和官方 UMI 一致)
  - State/Action: 20D [pos(3) + rot6d(6) + gripper(1) + zeros(10)]
  - 训练时 RelativeState + DeltaActions + GlobalToBodyDelta 处理相对化

Usage:
    cd /path/to/openpi
    uv run examples/umi/convert_zarr_to_lerobot.py \
        --zarr_path /workspace/umi/mydata/cups-0-47-crop.zarr.zip \
        --task "pick up the cup"
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from numcodecs import registry
    import imagecodecs
    from numcodecs.abc import Codec

    class JpegXl(Codec):
        codec_id = "imagecodecs_jpegxl"
        def __init__(self, level=None, **kwargs):
            self.level = level
        def encode(self, buf):
            return imagecodecs.jpegxl_encode(np.asarray(buf), level=self.level)
        def decode(self, buf, out=None):
            return imagecodecs.jpegxl_decode(buf)

    registry.register_codec(JpegXl)
except Exception as e:
    print(f"Warning: JpegXl codec registration failed: {e}")

import zarr
from zarr.storage import ZipStore

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from tqdm import tqdm

REPO_NAME = "local/umi_zarr_single_arm_6d"


def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """axis-angle (3,) → rot6d (6,)"""
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return R[:, :2].T.flatten()


def main(
    zarr_path: str,
    task: str = "pick up the cup",
    repo_name: str = REPO_NAME,
    image_size: int = 224,
    fps: int = 30,
    cam_mode: str = "btm",  # btm / top / topbtm
    cam_btm: str = "camera0_rgb",
    cam_top: str = "camera3_rgb",
    gripper_max_width: float = 90.0,
    min_episode_length: int = 30,
):
    assert cam_mode in ("btm", "top", "topbtm"), f"cam_mode must be btm/top/topbtm, got {cam_mode}"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    store = ZipStore(zarr_path, mode='r')
    root = zarr.open_group(store, mode='r')

    episode_ends = np.array(root['meta']['episode_ends'])
    n_episodes = len(episode_ends)
    n_frames = int(episode_ends[-1])

    print(f"Episodes: {n_episodes}, Frames: {n_frames}, cam_mode: {cam_mode}")

    eef_pos = np.array(root['data']['robot0_eef_pos'])
    eef_rot = np.array(root['data']['robot0_eef_rot_axis_angle'])
    gripper = np.array(root['data']['robot0_gripper_width'])

    features = {
        "cam_right": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (20,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (20,),
            "names": ["action"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="umi_single_arm",
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    converted = 0

    # 列出可用的 data keys，自动检测相机名
    available_keys = list(root['data'].keys()) if hasattr(root['data'], 'keys') else list(root['data'].array_keys())
    print(f"Available data keys: {available_keys}")

    # 自动检测相机 key（兼容不同命名）
    def find_cam_key(preferred, fallback_patterns):
        if preferred in available_keys:
            return preferred
        for pat in fallback_patterns:
            for k in available_keys:
                if pat in k.lower():
                    return k
        return None

    # 找所有图像 key（含 rgb 的）
    img_keys = [k for k in available_keys if "rgb" in k.lower() or "image" in k.lower() or "camera" in k.lower()]
    print(f"Image keys: {img_keys}")

    cam_btm_key = find_cam_key(cam_btm, ["camera0", "cam0", "btm", "bottom", "wrist"])
    cam_top_key = find_cam_key(cam_top, ["camera3", "cam3", "top", "head"])

    # 如果只有一个相机 key，单相机模式直接用它
    if len(img_keys) == 1:
        only_cam = img_keys[0]
        if cam_mode == "btm" and cam_btm_key is None:
            cam_btm_key = only_cam
        elif cam_mode == "top" and cam_top_key is None:
            cam_top_key = only_cam
    # 如果有两个相机 key，按索引排序（小号=下方，大号=上方）
    elif len(img_keys) == 2 and cam_mode == "topbtm":
        img_keys_sorted = sorted(img_keys)
        if cam_btm_key is None:
            cam_btm_key = img_keys_sorted[0]
        if cam_top_key is None:
            cam_top_key = img_keys_sorted[1]

    # 根据 cam_mode 加载相机数据
    if cam_mode == "btm":
        assert cam_btm_key, f"找不到下方相机 key, available: {available_keys}"
        print(f"Using bottom cam: {cam_btm_key}")
        cam_data_primary = root['data'][cam_btm_key]
        cam_data_secondary = None
    elif cam_mode == "top":
        assert cam_top_key, f"找不到上方相机 key, available: {available_keys}"
        print(f"Using top cam: {cam_top_key}")
        cam_data_primary = root['data'][cam_top_key]
        cam_data_secondary = None
    else:  # topbtm
        assert cam_btm_key, f"找不到下方相机 key, available: {available_keys}"
        assert cam_top_key, f"找不到上方相机 key, available: {available_keys}"
        print(f"Using bottom cam: {cam_btm_key}, top cam: {cam_top_key}")
        cam_data_primary = root['data'][cam_btm_key]   # 下方
        cam_data_secondary = root['data'][cam_top_key]  # 上方

    for ep_idx in tqdm(range(n_episodes), desc="转换 episodes"):
        start = int(ep_starts[ep_idx])
        end = int(episode_ends[ep_idx])
        ep_len = end - start

        if ep_len < min_episode_length:
            continue

        BATCH = 100
        for batch_start in range(0, ep_len, BATCH):
            batch_end = min(batch_start + BATCH, ep_len)
            abs_start = start + batch_start
            abs_end = start + batch_end
            ep_imgs_primary = np.array(cam_data_primary[abs_start:abs_end])
            ep_imgs_secondary = None
            if cam_data_secondary is not None:
                ep_imgs_secondary = np.array(cam_data_secondary[abs_start:abs_end])

            for i in range(batch_end - batch_start):
                fid = abs_start + i
                pos = eef_pos[fid]
                rot6d = rotvec_to_rot6d(eef_rot[fid])
                grip = gripper[fid, 0] / gripper_max_width
                grip = np.clip(grip, 0, 1)

                pose_10d = np.concatenate([pos, rot6d, [grip]]).astype(np.float32)
                pose_20d = np.concatenate([pose_10d, np.zeros(10, dtype=np.float32)])

                if cam_mode == "topbtm":
                    # 上下拼接: top在上, btm在下 → resize 到 image_size x image_size
                    img_btm = ep_imgs_primary[i]
                    img_top = ep_imgs_secondary[i]
                    # 两张图 resize 到相同宽度后上下拼
                    h1, w1 = img_top.shape[:2]
                    h2, w2 = img_btm.shape[:2]
                    target_w = image_size
                    img_top_r = cv2.resize(img_top, (target_w, target_w // 2))
                    img_btm_r = cv2.resize(img_btm, (target_w, target_w // 2))
                    img = np.vstack([img_top_r, img_btm_r])  # (image_size, image_size, 3)
                else:
                    img = ep_imgs_primary[i]
                    if img.shape[0] != image_size or img.shape[1] != image_size:
                        img = cv2.resize(img, (image_size, image_size))

                dataset.add_frame({
                    "cam_right": img,
                    "state": pose_20d,
                    "action": pose_20d,
                    "task": task,
                })

            del ep_imgs_primary
            if ep_imgs_secondary is not None:
                del ep_imgs_secondary

        dataset.save_episode()
        converted += 1

    store.close()
    if hasattr(dataset, 'consolidate'):
        dataset.consolidate()
    elif hasattr(dataset, 'finalize'):
        dataset.finalize()
    print(f"\n转换完成: {converted}/{n_episodes} episodes -> {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="building blocks into box")
    parser.add_argument("--repo_name", type=str, default=REPO_NAME)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cam_mode", type=str, default="btm", choices=["btm", "top", "topbtm"],
                        help="btm=下方相机, top=上方相机, topbtm=上下拼图")
    parser.add_argument("--cam_btm", type=str, default="camera0_rgb")
    parser.add_argument("--cam_top", type=str, default="camera3_rgb")
    parser.add_argument("--gripper_max_width", type=float, default=90.0)
    args = parser.parse_args()

    main(args.zarr_path, args.task, args.repo_name, args.image_size, args.fps,
         args.cam_mode, args.cam_btm, args.cam_top, args.gripper_max_width)
