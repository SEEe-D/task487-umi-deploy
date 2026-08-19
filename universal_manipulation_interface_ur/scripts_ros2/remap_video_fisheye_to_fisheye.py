import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FisheyeCamera:
    """OpenCV fisheye 模型相机参数。"""

    K: np.ndarray  # (3, 3), float64
    D: np.ndarray  # (4,), float64
    calib_size: Tuple[int, int]  # (width, height) 对应这组 K 的标定分辨率


def get_dji_fisheye_camera_wide() -> FisheyeCamera:
    """从 DJI 脚本中拷贝的鱼眼相机参数（OpenCV fisheye 模型，wide）。"""
    fx = 770.0231323242188
    fy = 770.0231323242188
    cx = 960.0
    cy = 720.0

    k1 = 0.17545107007026672
    k2 = 0.11582709848880768
    k3 = -0.06586384773254395
    k4 = -0.008552291430532932

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    calib_size = (1920, 1440)
    return FisheyeCamera(K=K, D=D, calib_size=calib_size)


def get_dji_fisheye_camera_ultra_wide() -> FisheyeCamera:
    """DJI ultra wide 参数（OpenCV fisheye 模型）。"""
    fx = 820.50500460055071
    fy = 820.50500460055071
    cx = 960.0
    cy = 720.0

    k1 = 0.14506145705377738
    k2 = -0.10514431038330563
    k3 = 0.078496452737663425
    k4 = -0.01663638404997372

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    calib_size = (1920, 1440)
    return FisheyeCamera(K=K, D=D, calib_size=calib_size)


def get_hardcoded_fisheye_camera_1600x1200() -> FisheyeCamera:
    """
    另一套写死的 fisheye 相机内参（OpenCV fisheye 模型）。
    来源：UMI4/image_remap/undistort_image.py::get_hardcoded_fisheye_camera
    """
    fx = 769.81511546510023
    fy = 770.06382791792794
    cx = 800.0
    cy = 600.0

    k1 = -0.042495704417828087
    k2 = 0.014505478176843988
    k3 = -0.010619835551726717
    k4 = 0.002034250716162644

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    calib_size = (1600, 1200)
    return FisheyeCamera(K=K, D=D, calib_size=calib_size)


def _scale_K(K: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> np.ndarray:
    """按分辨率比例缩放内参 K（fx/fy/cx/cy）。"""
    from_w, from_h = from_size
    to_w, to_h = to_size
    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)

    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    K2[2, 2] = 1.0
    return K2


def _load_fisheye_camera_json(path: str) -> FisheyeCamera:
    """
    从 JSON 加载 fisheye 相机参数。

    JSON 格式示例：
    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "D": [k1, k2, k3, k4],
      "calib_size": [w, h]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    K = np.array(obj["K"], dtype=np.float64)
    D = np.array(obj["D"], dtype=np.float64).reshape(-1)
    if D.size != 4:
        raise ValueError(f"fisheye D 需要 4 个系数，当前是 {D.size}: {path}")
    calib_size = tuple(int(x) for x in obj["calib_size"][:2])
    if len(calib_size) != 2:
        raise ValueError(f"calib_size 需要 [w, h]: {path}")
    return FisheyeCamera(K=K, D=D, calib_size=(calib_size[0], calib_size[1]))


def _get_preset_camera(name: str) -> FisheyeCamera:
    name = name.strip().lower()
    if name in ("dji_wide", "dji"):
        return get_dji_fisheye_camera_wide()
    if name in ("dji_ultra", "dji_ultra_wide", "dji_uw"):
        return get_dji_fisheye_camera_ultra_wide()
    if name in ("hardcoded_1600x1200", "hardcoded", "fisheye_1600x1200"):
        return get_hardcoded_fisheye_camera_1600x1200()
    raise ValueError(f"未知 preset: {name}")


def create_map_fisheye_to_fisheye(
    out_h: int,
    out_w: int,
    src_camera: FisheyeCamera,
    dst_camera: FisheyeCamera,
    src_frame_size: Tuple[int, int],
    dst_frame_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成从目标 fisheye(畸变像素) -> 源 fisheye(畸变像素) 的 remap 映射。

    - 输出图（目标）遵循 dst_camera 的 fisheye 畸变模型
    - 输入图（源）为 src_camera 的 fisheye 畸变图

    Args:
        out_h/out_w: 目标输出（渲染）尺寸
        src_frame_size: (src_w, src_h)，用于把 src 的 K 从标定分辨率缩放到实际视频帧
        dst_frame_size: (dst_w, dst_h)，用于把 dst 的 K 从标定分辨率缩放到目标输出分辨率
    """
    src_w, src_h = src_frame_size
    dst_w, dst_h = dst_frame_size
    if out_w != dst_w or out_h != dst_h:
        raise ValueError("out_w/out_h 必须与 dst_frame_size 一致（内部约束）")

    K_src = _scale_K(src_camera.K, from_size=src_camera.calib_size, to_size=(src_w, src_h))
    D_src = src_camera.D.astype(np.float64).reshape(4)

    K_dst = _scale_K(dst_camera.K, from_size=dst_camera.calib_size, to_size=(dst_w, dst_h))
    D_dst = dst_camera.D.astype(np.float64).reshape(4)

    # 目标图像像素网格（dst fisheye 畸变像素坐标）
    u, v = np.meshgrid(
        np.arange(out_w, dtype=np.float64),
        np.arange(out_h, dtype=np.float64),
    )
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float64)

    # 1) 目标畸变像素 -> 归一化无畸变坐标 (x, y)
    # undistortPoints 输出为归一化平面坐标（相机坐标系 z=1 的投影）
    pts_undist = cv2.fisheye.undistortPoints(pts, K_dst, D_dst, R=np.eye(3), P=np.eye(3))
    pts_undist = pts_undist.reshape(-1, 2)

    # 2) 归一化平面坐标 -> 3D 射线方向（尺度不重要；投影只与方向有关）
    X = np.concatenate([pts_undist, np.ones((pts_undist.shape[0], 1), dtype=np.float64)], axis=1)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    X = X / norm
    pts3d = X.reshape(-1, 1, 3).astype(np.float64)

    # 3) 射线方向 -> 源 fisheye 畸变像素坐标
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img_pts, _ = cv2.fisheye.projectPoints(pts3d, rvec, tvec, K_src, D_src)
    img_pts = img_pts.reshape(out_h, out_w, 2)

    map1 = img_pts[:, :, 0].astype(np.float32)
    map2 = img_pts[:, :, 1].astype(np.float32)
    return map1, map2


def create_grid_from_map(
    map1: np.ndarray, map2: np.ndarray, src_h: int, src_w: int, align_corners: bool = True
) -> np.ndarray:
    """将 OpenCV remap 的 map1/map2 转换为 PyTorch grid_sample 的 grid。"""
    if align_corners:
        # x in [0, W-1] -> [-1, 1]
        denom_x = max(src_w - 1, 1)
        denom_y = max(src_h - 1, 1)
        x_normalized = 2.0 * (map1 / denom_x) - 1.0
        y_normalized = 2.0 * (map2 / denom_y) - 1.0
    else:
        # x in [0, W] roughly -> [-1, 1] with pixel-center convention
        x_normalized = 2.0 * ((map1 + 0.5) / src_w) - 1.0
        y_normalized = 2.0 * ((map2 + 0.5) / src_h) - 1.0

    grid = np.stack([x_normalized, y_normalized], axis=-1)[np.newaxis, ...]
    return grid.astype(np.float32)


class ParallelVideoReader:
    def __init__(self, path: str, num_workers: int = 4, queue_size: int = 256):
        self.path = path
        self.num_workers = max(1, num_workers)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.total_frames <= 0:
            print("[WARNING] 无法获取视频总帧数，回退到单线程读取。")
            self.num_workers = 1

        self.queues = [
            Queue(maxsize=max(1, queue_size // self.num_workers)) for _ in range(self.num_workers)
        ]
        self.threads = []
        self.stopped = False

        chunk_size = int(np.ceil(self.total_frames / self.num_workers)) if self.total_frames > 0 else 0
        for i in range(self.num_workers):
            start_frame = i * chunk_size
            end_frame = min((i + 1) * chunk_size, self.total_frames) if self.total_frames > 0 else 0

            if self.total_frames > 0 and start_frame >= self.total_frames:
                self.queues[i].put(None)
                continue

            t = threading.Thread(
                target=self._worker_loop, args=(i, start_frame, end_frame), daemon=True
            )
            t.start()
            self.threads.append(t)

        self.current_worker_idx = 0

    def _worker_loop(self, worker_idx: int, start_frame: int, end_frame: int):
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while not self.stopped and current_frame < end_frame:
            if not self.queues[worker_idx].full():
                ret, frame = cap.read()
                if not ret:
                    break
                self.queues[worker_idx].put(frame)
                current_frame += 1
            else:
                time.sleep(0.005)

        cap.release()
        if not self.stopped:
            self.queues[worker_idx].put(None)

    def read(self):
        while self.current_worker_idx < self.num_workers:
            q = self.queues[self.current_worker_idx]
            try:
                frame = q.get(timeout=1.0)
                if frame is None:
                    self.current_worker_idx += 1
                    continue
                return frame
            except Empty:
                if (
                    self.current_worker_idx < len(self.threads)
                    and (not self.threads[self.current_worker_idx].is_alive())
                    and q.empty()
                ):
                    self.current_worker_idx += 1
                    continue
                return None
        return None

    def running(self):
        return self.current_worker_idx < self.num_workers

    def release(self):
        self.stopped = True
        for t in self.threads:
            t.join()


class FFmpegVideoWriter:
    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        target_width: int | None = None,
        target_height: int | None = None,
        queue_size: int = 256,
        encoder: str = "libx264",
        crf: int = 15,
        gop: int = 1,
    ):
        self.width = width
        self.height = height
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg 命令，请先安装 ffmpeg。")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{fps}",
            "-i",
            "-",
        ]

        if target_width is not None and target_height is not None:
            if target_width != width or target_height != height:
                cmd.extend(["-vf", f"scale={target_width}:{target_height}:flags=lanczos"])

        cmd.extend(["-c:v", encoder, "-g", str(gop), "-bf", "0"])

        if "libx264" in encoder or "libx265" in encoder:
            cmd.extend(["-preset", "ultrafast", "-crf", str(crf)])
        elif "nvenc" in encoder:
            cmd.extend(
                [
                    "-preset",
                    "p1",
                    "-rc:v",
                    "constqp",
                    "-qp",
                    str(crf),
                    "-spatial-aq",
                    "0",
                    "-temporal-aq",
                    "0",
                ]
            )

        cmd.extend(["-pix_fmt", "yuv420p"])
        cmd.append(path)

        print(f"[INFO] FFmpeg cmd: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=64 * 1024 * 1024
        )

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while True:
            if self.stopped and self.queue.empty():
                break
            try:
                frame_bgr = self.queue.get(timeout=0.1)
                frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
                if self.process.poll() is None:
                    try:
                        self.process.stdin.write(frame_yuv.tobytes())
                    except BrokenPipeError:
                        print("[ERROR] FFmpeg stdin broken pipe")
                        break
                self.queue.task_done()
            except Empty:
                continue

        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()

    def write(self, frame):
        self.queue.put(frame)

    def release(self):
        self.stopped = True
        self.thread.join()


def remap_video_fisheye_to_fisheye(
    video_path: str,
    output_path: str,
    src_camera: FisheyeCamera,
    dst_camera: FisheyeCamera,
    out_width: int | None = None,
    out_height: int | None = None,
    render_scale: float = 1.0,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    encoder: str = "auto",
    crf: int = 28,
    gop: int = 1,
):
    """将一个 fisheye 畸变视频从 src fisheye 模型重映射成 dst fisheye 模型。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("[INFO] CUDA 不可用，检测到 macOS MPS，切换为 'mps' 设备")
            device = "mps"
        else:
            print("[WARNING] CUDA 不可用且无 MPS，使用 CPU")
            device = "cpu"

    reader = ParallelVideoReader(video_path, num_workers=num_workers)
    print(
        f"[INFO] 输入视频: {reader.width}x{reader.height}, FPS={reader.fps:.2f}, 总帧数={reader.total_frames}"
    )

    if (out_width is None) != (out_height is None):
        raise ValueError("out_width 和 out_height 要么同时指定，要么都不指定")

    if out_width is None:
        # 默认输出分辨率：目标相机的标定分辨率（更符合“输出到某个相机模型”的直觉）
        out_w, out_h = int(dst_camera.calib_size[0]), int(dst_camera.calib_size[1])
    else:
        out_w, out_h = int(out_width), int(out_height)

    render_scale = float(render_scale)
    if render_scale <= 0:
        raise ValueError("render_scale 必须 > 0")

    render_w = max(1, int(round(out_w * render_scale)))
    render_h = max(1, int(round(out_h * render_scale)))

    # 为了避免非常奇怪的形变（非等比例缩放），这里强制 render 与 out 同比例缩放
    # 允许 out_w/out_h 本身不是源视频尺寸，但内部 remap 基于目标分辨率 render_w/render_h 做 K 缩放
    print(
        f"[INFO] 渲染尺寸(Remap): {render_w}x{render_h} (render_scale={render_scale:.3f}), 最终输出尺寸(FFmpeg): {out_w}x{out_h}"
    )
    print(f"[INFO] Batch Size={batch_size}, Workers={reader.num_workers}, Device={device}")
    print(f"[INFO] src calib_size={src_camera.calib_size}, dst calib_size={dst_camera.calib_size}")

    if encoder == "auto":
        if device == "cuda":
            encoder = "hevc_nvenc"
            print("[INFO] Auto-detected CUDA, using 'hevc_nvenc' (H.265).")
        else:
            encoder = "libx264"
            print("[INFO] Using 'libx264' for encoding.")

    try:
        writer = FFmpegVideoWriter(
            output_path,
            reader.fps,
            render_w,
            render_h,
            target_width=out_w,
            target_height=out_h,
            encoder=encoder,
            crf=crf,
            gop=gop,
        )
    except Exception as e:
        print(f"[WARNING] 启动编码器 {encoder} 失败: {e}")
        print("[WARNING] 回退到 libx264...")
        writer = FFmpegVideoWriter(
            output_path,
            reader.fps,
            render_w,
            render_h,
            target_width=out_w,
            target_height=out_h,
            encoder="libx264",
            crf=crf,
            gop=gop,
        )

    print("[INFO] 生成 fisheye->fisheye remap 映射表...")
    map1, map2 = create_map_fisheye_to_fisheye(
        out_h=render_h,
        out_w=render_w,
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_frame_size=(reader.width, reader.height),
        dst_frame_size=(render_w, render_h),
    )

    align_corners = True
    grid_np = create_grid_from_map(map1, map2, src_h=reader.height, src_w=reader.width, align_corners=align_corners)
    grid_base = torch.from_numpy(grid_np).to(device)
    grid_batch = grid_base.expand(batch_size, -1, -1, -1)

    print("[INFO] 开始处理视频...")
    frame_count = 0
    t0 = time.time()

    try:
        frames_buffer = []
        while reader.running() or frames_buffer:
            while len(frames_buffer) < batch_size and reader.running():
                frame = reader.read()
                if frame is not None:
                    frames_buffer.append(frame)
                else:
                    break

            if not frames_buffer:
                break

            current_batch_size = len(frames_buffer)
            batch_np = np.stack(frames_buffer)  # (B, H, W, C)

            batch_tensor = (
                torch.from_numpy(batch_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            )

            current_grid = grid_batch[:current_batch_size]
            output_tensor = F.grid_sample(
                batch_tensor, current_grid, align_corners=align_corners, padding_mode="zeros"
            )

            output_np = (
                output_tensor.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
            )

            for i in range(current_batch_size):
                writer.write(output_np[i])

            frame_count += current_batch_size
            frames_buffer.clear()

            if frame_count % (batch_size * 5) < batch_size:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 1e-6)
                print(f"[INFO] 处理进度: {frame_count}/{reader.total_frames} | FPS: {fps:.1f}")

    finally:
        reader.release()
        writer.release()

    total_time = time.time() - t0
    print(
        f"[DONE] 视频处理完成，共处理 {frame_count} 帧，耗时 {total_time:.1f}s (FPS: {frame_count/max(total_time,1e-6):.1f})"
    )
    print(f"[DONE] 输出视频已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "读取一个 fisheye 畸变视频，基于 OpenCV fisheye 模型，"
            "将 src fisheye 畸变重映射为 dst fisheye 畸变（通过 PyTorch grid_sample）。"
        )
    )
    parser.add_argument("--video", required=True, help="输入视频路径（fisheye 畸变视频，mp4 等）")
    parser.add_argument(
        "--output_video",
        required=False,
        help="输出视频路径；若不指定，则保存在输入视频同目录，文件名加 _to_fisheye 后缀",
    )
    parser.add_argument(
        "--out_width",
        type=int,
        default=None,
        help="最终输出视频宽度；默认等于 dst 相机 calib_size 的宽度（例如 DJI=1920）",
    )
    parser.add_argument(
        "--out_height",
        type=int,
        default=None,
        help="最终输出视频高度；默认等于 dst 相机 calib_size 的高度（例如 DJI=1440）",
    )
    parser.add_argument(
        "--render_scale",
        type=float,
        default=1.0,
        help="remap 的渲染缩放（相对最终输出分辨率），<1 更快但会先低分辨率 remap 再缩放输出。默认 1.0",
    )

    parser.add_argument(
        "--src_preset",
        type=str,
        default="hardcoded_1600x1200",
        help="源 fisheye 相机 preset: dji_wide/dji_ultra/hardcoded_1600x1200；默认 hardcoded_1600x1200",
    )
    parser.add_argument(
        "--dst_preset",
        type=str,
        default="dji_ultra",
        help="目标 fisheye 相机 preset: dji_wide/dji_ultra/hardcoded_1600x1200；默认 dji_ultra",
    )
    parser.add_argument(
        "--src_json",
        type=str,
        default=None,
        help="源相机 JSON（覆盖 src_preset），格式见脚本注释",
    )
    parser.add_argument(
        "--dst_json",
        type=str,
        default=None,
        help="目标相机 JSON（覆盖 dst_preset），格式见脚本注释",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="使用的设备（默认 cuda）"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小（默认 16）")
    parser.add_argument("--num_workers", type=int, default=4, help="并行读取/解码线程数（默认 4）")
    parser.add_argument(
        "--encoder",
        type=str,
        default="auto",
        help="视频编码器 (例如: libx264, h264_nvenc, hevc_nvenc)。默认 'auto' 会在 CUDA 可用时尝试 hevc_nvenc。",
    )
    parser.add_argument("--crf", type=int, default=28, help="CRF/QP 质量值 (越小画质越好，默认 28)")
    parser.add_argument("--gop", type=int, default=1, help="GOP 大小 (关键帧间隔，默认 1 为 All-Intra)")
    return parser.parse_args()


def _resolve_camera(preset: str, json_path: Optional[str]) -> FisheyeCamera:
    if json_path is not None:
        return _load_fisheye_camera_json(json_path)
    return _get_preset_camera(preset)


if __name__ == "__main__":
    args = parse_args()

    if args.output_video is not None:
        out_video_path = args.output_video
    else:
        base_name = os.path.basename(args.video)
        name, ext = os.path.splitext(base_name)
        if ext == "":
            ext = ".mp4"
        dir_name = os.path.dirname(args.video) or "."
        out_video_path = os.path.join(dir_name, f"{name}_to_fisheye{ext}")

    src_cam = _resolve_camera(args.src_preset, args.src_json)
    dst_cam = _resolve_camera(args.dst_preset, args.dst_json)

    remap_video_fisheye_to_fisheye(
        video_path=args.video,
        output_path=out_video_path,
        src_camera=src_cam,
        dst_camera=dst_cam,
        out_width=args.out_width,
        out_height=args.out_height,
        render_scale=args.render_scale,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder=args.encoder,
        crf=args.crf,
        gop=args.gop,
    )


