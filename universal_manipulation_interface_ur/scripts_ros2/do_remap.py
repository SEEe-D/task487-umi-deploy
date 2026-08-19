import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import remap_video_fisheye_to_fisheye as rv


'''
其他模块调用方式：
from UMI4.image_remap.do_remap import FisheyeRemapper
remapper = FisheyeRemapper("/path/fisheye_map.npz")
out = remapper.remap_frame(frame_bgr)
'''



def _read_map_npz(map_path: str) -> dict:
    data = np.load(map_path, allow_pickle=False)
    required = ["map1", "map2", "src_w", "src_h", "render_w", "render_h", "align_corners"]
    for k in required:
        if k not in data:
            raise ValueError(f"map 文件缺少字段 '{k}': {map_path}")
    return {k: data[k] for k in data.files}


def _open_video(src: str) -> cv2.VideoCapture:
    # 允许传入 "0" / "1" 作为摄像头 ID
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    return cap


def _resize_for_display(img: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = float(max_w) / float(w)
    nh = int(round(h * scale))
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)


class FisheyeRemapper:
    """
    可复用的 remap 模块：
    - 读取 export_map.py 导出的 .npz
    - 预计算 grid_sample 的 grid（CPU）
    - 提供 remap_frame(frame_bgr) -> out_frame_bgr
    """

    def __init__(self, map_path: str):
        self.map_path = map_path
        self.device = "cpu"  # 按你的要求：固定 CPU

        m = _read_map_npz(map_path)
        self._m = m

        map1 = m["map1"].astype(np.float32)
        map2 = m["map2"].astype(np.float32)

        self.src_w = int(np.array(m["src_w"]).reshape(()))
        self.src_h = int(np.array(m["src_h"]).reshape(()))
        self.render_w = int(np.array(m["render_w"]).reshape(()))
        self.render_h = int(np.array(m["render_h"]).reshape(()))
        self.align_corners = bool(int(np.array(m["align_corners"]).reshape(())))

        # 可选：export_map.py 会写入 out_w/out_h（最终输出尺寸）。
        # 若存在且与 render 不同，则在 remap 后用 INTER_AREA 低通下采样，显著减少锯齿。
        self.out_w = int(np.array(m["out_w"]).reshape(())) if "out_w" in m else self.render_w
        self.out_h = int(np.array(m["out_h"]).reshape(())) if "out_h" in m else self.render_h

        grid_np = rv.create_grid_from_map(
            map1, map2, src_h=self.src_h, src_w=self.src_w, align_corners=self.align_corners
        )
        self.grid = torch.from_numpy(grid_np).to(self.device)

    def check_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            raise ValueError("frame is None")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"frame 必须是 BGR 三通道图像，当前 shape={frame_bgr.shape}")
        h, w = frame_bgr.shape[:2]
        if w != self.src_w or h != self.src_h:
            raise ValueError(
                f"输入帧分辨率 {w}x{h} 与 map 期望的 src {self.src_w}x{self.src_h} 不一致"
            )

    def remap_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """对单帧 BGR 图像做 remap，返回 BGR 输出帧。"""
        self.check_frame(frame_bgr)

        # batch=1, CPU
        batch_tensor = (
            torch.from_numpy(frame_bgr).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        )
        output_tensor = F.grid_sample(
            batch_tensor, self.grid, align_corners=self.align_corners, padding_mode="zeros"
        )
        out_frame = output_tensor[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

        # 关键：强下采样时，用 INTER_AREA 抗锯齿（便宜且有效）
        if (self.out_w, self.out_h) != (self.render_w, self.render_h):
            out_frame = cv2.resize(out_frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return out_frame


def preview_video(
    video_path: str,
    remapper: FisheyeRemapper,
    show_both: bool = False,
    max_display_width: int = 0,
):
    cap = _open_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频/设备: {video_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[INFO] 输入: {in_w}x{in_h}, FPS={in_fps:.2f}, 总帧数={total_frames}")
    print(
        f"[INFO] map: src={remapper.src_w}x{remapper.src_h}, render={remapper.render_w}x{remapper.render_h}, out={remapper.out_w}x{remapper.out_h}, align_corners={remapper.align_corners}"
    )

    if in_w != remapper.src_w or in_h != remapper.src_h:
        cap.release()
        raise ValueError(
            f"输入视频分辨率 {in_w}x{in_h} 与 map 期望的 src {remapper.src_w}x{remapper.src_h} 不一致；"
            f"请重新 export_map（用同一个输入分辨率）"
        )

    frame_count = 0
    import time

    t0 = time.time()
    try:
        print("[INFO] 按键：q/ESC 退出；空格 暂停/继续")
        win_out = "remap"
        win_in = "input"
        paused = False
        last_frame = None
        last_out_frame = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                last_frame = frame
            else:
                # 暂停时不读新帧，但仍保持窗口响应
                if last_frame is None or last_out_frame is None:
                    # 理论上不会发生（用户不可能在第一帧前暂停），做个兜底
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == 32:
                        paused = not paused
                    continue
                frame = last_frame
                out_frame = last_out_frame

            if not paused:
                out_frame = remapper.remap_frame(frame)
                last_out_frame = out_frame

            if show_both:
                disp_in = _resize_for_display(frame, max_display_width)
                disp_out = _resize_for_display(out_frame, max_display_width)
                cv2.imshow(win_in, disp_in)
                cv2.imshow(win_out, disp_out)
            else:
                disp_out = _resize_for_display(out_frame, max_display_width)
                cv2.imshow(win_out, disp_out)

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                paused = not paused

            if not paused:
                frame_count += 1

            if frame_count % 30 == 0 and frame_count > 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 1e-6)
                denom = total_frames if total_frames > 0 else -1
                if denom > 0:
                    print(f"[INFO] 处理进度: {frame_count}/{denom} | FPS: {fps:.1f}")
                else:
                    print(f"[INFO] 已处理: {frame_count} 帧 | FPS: {fps:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = time.time() - t0
    print(f"[DONE] 显示结束，共处理 {frame_count} 帧，耗时 {total:.1f}s (FPS: {frame_count/max(total,1e-6):.1f})")


def do_remap(video_path: str, map_path: str, show_both: bool = False, max_display_width: int = 0):
    """兼容入口：内部构建 remapper 并实时显示。"""
    remapper = FisheyeRemapper(map_path=map_path)
    preview_video(
        video_path=video_path,
        remapper=remapper,
        show_both=show_both,
        max_display_width=max_display_width,
    )


def parse_args():
    p = argparse.ArgumentParser(description="读取 export_map.py 生成的 .npz map 文件，实时 remap 并显示。")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--map", required=True, help="export_map.py 导出的 .npz map 文件路径")
    p.add_argument("--show_both", action="store_true", help="同时显示原图与 remap（两个独立窗口）")
    p.add_argument(
        "--max_display_width",
        type=int,
        default=0,
        help="限制显示窗口最大宽度（0=不限制）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    do_remap(
        video_path=args.video,
        map_path=args.map,
        show_both=args.show_both,
        max_display_width=args.max_display_width,
    )


