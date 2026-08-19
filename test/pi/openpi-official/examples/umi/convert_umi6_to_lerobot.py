"""convert_umi6_to_lerobot.py — UMI6 数据 -> LeRobot v2 (单臂, 只用右臂)

将 UMI6 数据 (JSONL + MP4) 转为 LeRobot v2 格式, 供 pi0.5 微调使用。
虽然 UMI6 记录了双臂数据, 但只使用右臂, 左臂零填充 (与 UMI5 一致)。

UMI6 输入格式:
- 每个 episode 目录 (如 20260306-093335/) 下:
  - aligned_pos_{ep_name}.json: JSONL, 每行含 right/left 的 {xyz, wxyz, gripper, valid}
  - aligned_right_{ep_name}.MP4: 右腕相机 (640x480)
  - aligned_left_{ep_name}.MP4: 左腕相机 (640x480)
  - aligned_head_{ep_name}.MP4: 头部相机 (1600x1200)

输出格式 (LeRobot v2):
- State: 20D [right_pos(3) + right_rot_6d(6) + right_gripper(1) + left_zeros(10)]
  - 绝对 EE pose (和官方 UMI 一致)
  - 训练时 RelativeState transform 把 pose 归零 (只保留 gripper)
- Action: 20D 绝对 EE pose (训练时 DeltaActions 转为 chunk-wise delta)
- 图像: cam_right, cam_head (均 resize 到 image_size, 无左腕相机)

Usage:
    cd /path/to/openpi
    uv run examples/umi/convert_umi6_to_lerobot.py \
        --umi6_dir /data/umi6/building-blocks-righthand/ \
        --task "building blocks into box"
"""
import json
import shutil
import glob
from pathlib import Path

import cv2
import numpy as np
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

REPO_NAME = "local/umi6_single_arm_6d_rel"


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """quaternion (w,x,y,z) -> rotation_6d (6D)"""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R[:, :2].T.flatten()  # (6,)


def frame_to_ee_pose(hand: dict) -> np.ndarray:
    """从单手数据提取 10D EE pose (绝对位姿, 和官方 UMI 一致)"""
    xyz = np.array(hand["xyz"])                         # (3,)
    rot6d = quat_wxyz_to_rot6d(np.array(hand["wxyz"]))  # (6,)
    gripper = np.array([hand["gripper"] / 90.0])         # (1,) 归一化到 [0,1]
    return np.concatenate([xyz, rot6d, gripper])          # (10,)



class VideoReader:
    """从 MP4 视频逐帧读取图像 (缓存已打开的 VideoCapture)"""

    def __init__(self):
        self._caps = {}

    def read_frame(self, video_path: Path, frame_id: int) -> np.ndarray:
        key = str(video_path)
        if key not in self._caps:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            self._caps[key] = cap

        cap = self._caps[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_id} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def find_episode_files(ep_dir: Path):
    """查找 episode 下的视频和位姿文件, 返回 (poses_path, right_video, head_video) 或 None"""
    ep_name = ep_dir.name

    # 位姿文件: aligned_pos_{ep_name}.json
    poses_path = ep_dir / f"aligned_pos_{ep_name}.json"
    if not poses_path.exists():
        # fallback: glob
        candidates = sorted(glob.glob(str(ep_dir / "aligned_pos_*.json")))
        if not candidates:
            return None
        poses_path = Path(candidates[0])

    # 视频文件 (只用右腕 + 头部, 不用左腕)
    right_video = ep_dir / f"aligned_right_{ep_name}.MP4"
    head_video = ep_dir / f"aligned_head_{ep_name}.MP4"

    # fallback: glob 匹配 (兼容大小写)
    if not right_video.exists():
        candidates = sorted(glob.glob(str(ep_dir / "aligned_right_*.[Mm][Pp]4")))
        right_video = Path(candidates[0]) if candidates else right_video
    if not head_video.exists():
        candidates = sorted(glob.glob(str(ep_dir / "aligned_head_*.[Mm][Pp]4")))
        head_video = Path(candidates[0]) if candidates else head_video

    if not right_video.exists():
        return None

    return poses_path, right_video, head_video


def main(
    umi6_dir: str,
    task: str = "building blocks into box",
    repo_name: str = REPO_NAME,
    image_size: int = 256,
    fps: int = 30,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        # 检查已有数据, 提示用户而非自动删除
        try:
            info_path = output_path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                existing_eps = info.get("total_episodes", 0)
                print(f"已有数据: {existing_eps} episodes, 删除后重新转换...")
        except Exception:
            pass
        shutil.rmtree(output_path)

    features = {
        "cam_right": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "cam_head": {
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
        robot_type="umi6_single_arm",
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    video_reader = VideoReader()

    episodes = sorted([d for d in Path(umi6_dir).iterdir() if d.is_dir()])
    print(f"找到 {len(episodes)} 个 episode 目录")

    converted = 0
    for ep_dir in tqdm(episodes, desc="转换 episodes"):
        result = find_episode_files(ep_dir)
        if result is None:
            print(f"  跳过 {ep_dir.name}: 缺少必要文件")
            continue

        poses_path, right_video, head_video = result
        has_head = head_video.exists()

        with open(poses_path) as f:
            frames = [json.loads(line) for line in f]

        if len(frames) < 2:
            continue

        for fid, frame in enumerate(frames):
            right_hand = frame.get("right", {})

            if not right_hand.get("valid", True):
                continue

            # 20D 绝对 EE pose: 右臂 10D + 左臂零填充 10D (单臂, 和官方 UMI 一致)
            right_pose = frame_to_ee_pose(right_hand)  # (10,)
            current_pose = np.concatenate([right_pose, np.zeros(10, dtype=np.float32)]).astype(np.float32)

            # 读取视频帧并 resize (只用右腕 + 头部)
            try:
                img_right = video_reader.read_frame(right_video, fid)
                img_head = video_reader.read_frame(head_video, fid) if has_head else np.zeros(
                    (image_size, image_size, 3), dtype=np.uint8
                )
            except RuntimeError:
                break

            img_right = cv2.resize(img_right, (image_size, image_size))
            if has_head:
                img_head = cv2.resize(img_head, (image_size, image_size))

            # action = 绝对 pose (训练时 DeltaActions 自动转为 chunk-wise delta)
            dataset.add_frame({
                "cam_right": img_right,
                "cam_head": img_head,
                "state": current_pose,
                "action": current_pose,
                "task": task,
            })

        dataset.save_episode()
        video_reader.close()  # 释放视频内存, 防止 OOM
        converted += 1

    video_reader.close()
    if hasattr(dataset, 'consolidate'):
        dataset.consolidate()
    elif hasattr(dataset, 'finalize'):
        dataset.finalize()
    print(f"\n转换完成: {converted}/{len(episodes)} episodes -> {output_path}")
    print(f"State (20D): [right_pos(3) + right_rot_6d(6) + right_gripper(1) + left_zeros(10)]")
    print(f"Action (20D): 绝对 EE pose (训练时 DeltaActions 转为 chunk-wise delta)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert UMI6 data to LeRobot v2 for pi0.5")
    parser.add_argument("--umi6_dir", type=str, required=True,
                        help="Parent directory containing episode subdirectories")
    parser.add_argument("--task", type=str, default="building blocks into box",
                        help="Task description prompt")
    parser.add_argument("--repo_name", type=str, default=REPO_NAME)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    main(args.umi6_dir, args.task, args.repo_name, args.image_size, args.fps)
