#!/usr/bin/env python3
"""
Thor 6路相机低延迟接收端
GStreamer硬件解码 + 2x3网格显示 + OSD + 帧同步指标
"""

import os
import sys
import time
import struct
import socket
import signal
import threading
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------------------
# 时钟偏移测量
# ---------------------------------------------------------------------------

def measure_clock_offset(sender_host, port=7777, samples=15, timeout=2.0):
    """UDP NTP-like时钟偏移测量, 返回offset_ms
    取RTT最小的样本对应的offset, 最准确"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        results = []  # (rtt_us, offset_us)
        for _ in range(samples):
            t1 = int(time.time() * 1_000_000)
            sock.sendto(struct.pack('<q', t1), (sender_host, port))
            try:
                data, _ = sock.recvfrom(32)
                t4 = int(time.time() * 1_000_000)
                if len(data) >= 24:
                    t1_echo, t2, t3 = struct.unpack('<qqq', data[:24])
                    offset_us = ((t2 - t1) + (t3 - t4)) / 2
                    rtt_us = (t4 - t1) - (t3 - t2)
                    results.append((rtt_us, offset_us))
            except socket.timeout:
                pass
            time.sleep(0.03)
        sock.close()
        if results:
            # 按RTT排序, 取RTT最小的前5个样本的中位数offset
            results.sort(key=lambda x: x[0])
            best = results[:min(5, len(results))]
            offsets = [r[1] for r in best]
            offsets.sort()
            median_us = offsets[len(offsets) // 2]
            best_rtt = best[0][0]
            print(f"Clock offset: {median_us / 1000:.1f}ms "
                  f"(best RTT={best_rtt / 1000:.1f}ms, "
                  f"{len(results)}/{samples} samples)")
            return median_us / 1000.0
        print("Clock offset measurement failed, using 0ms")
        return 0.0
    except Exception as e:
        print(f"Clock offset error: {e}, using 0ms")
        return 0.0


# ---------------------------------------------------------------------------
# OSD绘制 (纯cv2.putText, 黑色描边 + 彩色前景)
# ---------------------------------------------------------------------------

def draw_text(img, text, pos, scale=0.6, color=(0, 255, 0), thickness=1):
    """带黑色描边的文字绘制 (LINE_AA抗锯齿)"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 元数据接收
# ---------------------------------------------------------------------------

class MetaReceiver:
    """接收单路相机的帧时间戳元数据"""

    def __init__(self, camera_id, port, listen_host="0.0.0.0",
                 clock_offset_ms=0.0):
        self.camera_id = camera_id
        self.port = port
        self.clock_offset_ms = clock_offset_ms
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_host, port))
        self.sock.settimeout(0.5)

        self.latest_ts_us = 0
        self.latest_frame_id = 0
        self.latency_ema = 0.0
        self.meta_recv_count = 0     # 收到的元数据包数
        self.meta_drop_count = 0     # 通过frame_id检测到的丢帧数
        self._prev_frame_id = 0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        alpha = 0.15
        while self.running:
            try:
                data, _ = self.sock.recvfrom(64)
                if len(data) >= 24:
                    cam_id, frame_id, ts_us, w, h = struct.unpack(
                        '<IIqII', data[:24])
                    self.latest_ts_us = ts_us
                    self.latest_frame_id = frame_id
                    self.meta_recv_count += 1

                    # 通过frame_id连续性检测丢帧
                    if self._prev_frame_id > 0 and frame_id > self._prev_frame_id + 1:
                        self.meta_drop_count += frame_id - self._prev_frame_id - 1
                    self._prev_frame_id = frame_id

                    now_us = int(time.time() * 1_000_000)
                    adjusted_ts = ts_us - int(self.clock_offset_ms * 1000)
                    raw_ms = (now_us - adjusted_ts) / 1000.0

                    if -500 < raw_ms < 5000:
                        if self.latency_ema == 0:
                            self.latency_ema = raw_ms
                        else:
                            self.latency_ema = (alpha * raw_ms +
                                                (1 - alpha) * self.latency_ema)
            except socket.timeout:
                pass
            except OSError:
                break

    def close(self):
        self.running = False
        self.sock.close()
        if self.thread:
            self.thread.join(timeout=1)


# ---------------------------------------------------------------------------
# 单路相机接收
# ---------------------------------------------------------------------------

class CameraReceiver:
    """单路相机视频接收 + 解码"""

    def __init__(self, camera_id, label, video_port, meta_port,
                 listen_host, tile_w, tile_h, clock_offset_ms=0.0):
        self.camera_id = camera_id
        self.label = label
        self.video_port = video_port
        self.tile_w = tile_w
        self.tile_h = tile_h

        self.cap = None
        self.meta = MetaReceiver(camera_id, meta_port, listen_host,
                                 clock_offset_ms)
        self.decoder = "none"

        self.lock = threading.Lock()
        self.frame = None
        self.connected = False
        self.fps_counter = 0
        self.fps_value = 0.0
        self.fps_last_time = time.time()
        self.total_frames = 0

        self.running = False
        self.thread = None

    def _gst_pipeline(self, decoder="nvh264dec"):
        return (
            f'udpsrc port={self.video_port} buffer-size=2097152 '
            f'caps="application/x-rtp,media=video,'
            f'encoding-name=H264,payload=96" '
            f'! rtph264depay ! h264parse ! {decoder} '
            f'! queue max-size-buffers=5 leaky=downstream '
            f'! videoconvert ! video/x-raw,format=BGR '
            f'! appsink sync=false drop=false max-buffers=5'
        )

    def start(self):
        self.meta.start()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[{self.label}] Receiving on port {self.video_port}")

    def _capture_loop(self):
        # 默认优先用 CPU 解码器, 避免缺少 NVIDIA 插件时直接失败
        for dec in ["avdec_h264", "nvh264dec"]:
            self.cap = cv2.VideoCapture(self._gst_pipeline(dec),
                                        cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self.decoder = dec
                break
            print(f"[{self.label}] {dec} failed, trying next...")
        else:
            print(f"[{self.label}] ERROR: all decoders failed")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.002)
                continue

            self.connected = True
            self.total_frames += 1
            self.fps_counter += 1

            now = time.time()
            dt = now - self.fps_last_time
            if dt >= 1.0:
                self.fps_value = self.fps_counter / dt
                self.fps_counter = 0
                self.fps_last_time = now

            if frame.shape[1] != self.tile_w or frame.shape[0] != self.tile_h:
                frame = cv2.resize(frame, (self.tile_w, self.tile_h))

            with self.lock:
                self.frame = frame

        if self.cap:
            self.cap.release()

    def get_display_frame(self):
        with self.lock:
            f = self.frame

        if f is None:
            ph = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
            draw_text(ph, self.label, (20, self.tile_h // 2 - 10),
                      0.8, (100, 100, 100), 2)
            draw_text(ph, f"Port {self.video_port} waiting...",
                      (20, self.tile_h // 2 + 25), 0.6, (80, 80, 80), 1)
            return ph

        display = f.copy()

        # OSD布局: 字体和行距随tile高度自适应
        h = display.shape[0]
        s = h / 480.0  # 以480px为基准缩放
        line_gap = int(26 * s)
        fs_l = 0.6 * s   # 标签字号
        fs_m = 0.55 * s   # 中号
        fs_s = 0.5 * s   # 小号
        pad = int(8 * s)
        y = line_gap

        # Line 1: camera label
        draw_text(display, self.label, (pad, y), fs_l, (0, 255, 0), 1)
        y += line_gap

        # Line 2: latency
        lat = self.meta.latency_ema
        if lat != 0:
            if abs(lat) < 50:
                color = (0, 255, 0)
            elif abs(lat) < 100:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            draw_text(display, f"Latency:{lat:.0f}ms",
                      (pad, y), fs_m, color, 1)
            y += line_gap

        # Line 3: FPS + decoder
        draw_text(display, f"FPS:{self.fps_value:.1f} [{self.decoder}]",
                  (pad, y), fs_s, (200, 200, 200), 1)
        y += line_gap

        # Line 4: frame stats
        drop = self.meta.meta_drop_count
        if self.total_frames > 0:
            drop_color = (0, 255, 0) if drop == 0 else (0, 0, 255)
            draw_text(display, f"F:{self.total_frames} D:{drop}",
                      (pad, y), fs_s, drop_color, 1)

        return display

    def stop(self):
        self.running = False
        self.meta.close()
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# 主显示窗口
# ---------------------------------------------------------------------------

class StreamViewer:
    """6路视频 2x3 网格显示"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.receivers = []
        self.clock_offset_ms = 0.0
        self.running = False
        self._clock_thread = None

    def setup(self):
        listen_host = self.config['listen_host']
        video_base = self.config['video_base_port']
        meta_base = self.config['meta_base_port']
        display = self.config['display']
        tile_w = display['tile_width']
        tile_h = display['tile_height']

        # 时钟偏移测量
        sender_host = self.config.get('sender_host', '192.168.1.101')
        clock_port = self.config.get('clock_sync_port', 7777)
        print(f"Measuring clock offset to {sender_host}:{clock_port}...")
        self.clock_offset_ms = measure_clock_offset(
            sender_host, clock_port, samples=15, timeout=1.0)

        for i, cam_cfg in enumerate(self.config['cameras']):
            r = CameraReceiver(
                camera_id=i,
                label=cam_cfg['label'],
                video_port=video_base + i,
                meta_port=meta_base + i,
                listen_host=listen_host,
                tile_w=tile_w,
                tile_h=tile_h,
                clock_offset_ms=self.clock_offset_ms,
            )
            self.receivers.append(r)

        self._sender_host = sender_host
        self._clock_port = clock_port

        print(f"Configured {len(self.receivers)} cameras, "
              f"ports {video_base}-{video_base + len(self.receivers) - 1}")
        print(f"Display: {display['grid_cols']}x{display['grid_rows']}, "
              f"tile {tile_w}x{tile_h}")

    def _build_status_bar(self, width):
        """底部状态栏"""
        connected = sum(1 for r in self.receivers if r.connected)
        latencies = [r.meta.latency_ema for r in self.receivers
                     if r.meta.latency_ema != 0]

        # 帧同步: 6路latency的最大差值 (反映显示画面的时间同步程度)
        # GPIO FSIN保证采集同步, 此处衡量的是显示同步
        if len(latencies) >= 2:
            sync_ms = max(latencies) - min(latencies)
            sync_text = f"Sync: {sync_ms:.0f}ms"
        else:
            sync_text = "Sync: --"

        # 总丢帧统计
        total_drop = sum(r.meta.meta_drop_count for r in self.receivers)
        total_recv = sum(r.total_frames for r in self.receivers)
        if total_recv > 0:
            drop_rate = total_drop / (total_recv + total_drop) * 100 if (total_recv + total_drop) > 0 else 0
            drop_text = f"Drop: {total_drop} ({drop_rate:.1f}%)"
        else:
            drop_text = "Drop: --"

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            status = (f"Connected: {connected}/{len(self.receivers)}  |  "
                      f"Avg Latency: {avg_lat:.0f}ms  |  "
                      f"{sync_text}  |  "
                      f"{drop_text}  |  "
                      f"Offset: {self.clock_offset_ms:.1f}ms")
        else:
            status = (f"Connected: {connected}/{len(self.receivers)}  |  "
                      f"Waiting for data...")

        bar = np.zeros((44, width, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        cv2.putText(bar, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)
        return bar

    def _clock_sync_loop(self):
        """定期重新测量时钟偏移, 补偿时钟漂移"""
        while self.running:
            for _ in range(30):  # 每30秒测一次
                if not self.running:
                    return
                time.sleep(1)
            new_offset = measure_clock_offset(
                self._sender_host, self._clock_port, samples=15, timeout=1.0)
            if new_offset != 0.0:
                self.clock_offset_ms = new_offset
                for r in self.receivers:
                    r.meta.clock_offset_ms = new_offset

    def start(self):
        self.running = True
        for r in self.receivers:
            r.start()

        # 启动定期时钟同步
        self._clock_thread = threading.Thread(
            target=self._clock_sync_loop, daemon=True)
        self._clock_thread.start()

        display = self.config['display']
        cols = display['grid_cols']
        rows = display['grid_rows']
        tile_w = display['tile_width']
        tile_h = display['tile_height']
        title = display['window_title']

        canvas_w = cols * tile_w
        canvas_h = rows * tile_h + 44

        print(f"\n{'='*60}")
        print(f"Viewer: {canvas_w}x{canvas_h} | Press 'q' or ESC to quit")
        print(f"{'='*60}\n")

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, canvas_w, canvas_h)

        while self.running:
            grid = np.zeros((rows * tile_h, canvas_w, 3), dtype=np.uint8)

            for i, receiver in enumerate(self.receivers):
                r = i // cols
                c = i % cols
                tile = receiver.get_display_frame()
                if tile is not None:
                    grid[r * tile_h:(r + 1) * tile_h,
                         c * tile_w:(c + 1) * tile_w] = tile

            for c in range(1, cols):
                cv2.line(grid, (c * tile_w, 0), (c * tile_w, rows * tile_h),
                         (60, 60, 60), 1)
            for r in range(1, rows):
                cv2.line(grid, (0, r * tile_h), (canvas_w, r * tile_h),
                         (60, 60, 60), 1)

            status_bar = self._build_status_bar(canvas_w)
            canvas = np.vstack([grid, status_bar])

            cv2.imshow(title, canvas)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q') or key == 27:
                break

        self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        for r in self.receivers:
            r.stop()
        cv2.destroyAllWindows()
        print("Viewer stopped")


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.yaml')
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    viewer = StreamViewer(config_path)
    viewer.setup()

    def on_signal(sig, frame):
        print("\nStopping...")
        viewer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    viewer.start()


if __name__ == '__main__':
    main()
