#!/usr/bin/env python3
"""
Thor 相机接收端 — 使用 gi.repository.Gst 解码（不依赖 OpenCV GStreamer 后端）
其余逻辑与 receiver.py 完全一致
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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)


# ---------------------------------------------------------------------------
# 时钟偏移测量 (和 receiver.py 完全一致)
# ---------------------------------------------------------------------------

def measure_clock_offset(sender_host, port=7777, samples=15, timeout=2.0):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        results = []
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
# OSD
# ---------------------------------------------------------------------------

def draw_text(img, text, pos, scale=0.6, color=(0, 255, 0), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 元数据接收 (和 receiver.py 完全一致)
# ---------------------------------------------------------------------------

class MetaReceiver:
    def __init__(self, camera_id, port, listen_host="0.0.0.0", clock_offset_ms=0.0):
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
        self.meta_recv_count = 0
        self.meta_drop_count = 0
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
                    cam_id, frame_id, ts_us, w, h = struct.unpack('<IIqII', data[:24])
                    self.latest_ts_us = ts_us
                    self.latest_frame_id = frame_id
                    self.meta_recv_count += 1
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
                            self.latency_ema = alpha * raw_ms + (1 - alpha) * self.latency_ema
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
# 单路相机接收 — 用 gi.repository.Gst appsink 解码
# ---------------------------------------------------------------------------

class CameraReceiver:
    def __init__(self, camera_id, label, video_port, meta_port,
                 listen_host, tile_w, tile_h, clock_offset_ms=0.0, total_cameras=1):
        self.camera_id = camera_id
        self.label = label
        self.video_port = video_port
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.total_cameras = max(1, int(total_cameras))

        self.meta = MetaReceiver(camera_id, meta_port, listen_host, clock_offset_ms)
        self.decoder = "none"
        self.pipeline = None
        self._sink = None
        self._thread = None
        self._pipeline_lock = threading.Lock()
        self._decoder_candidates = self._build_decoder_candidates()
        self._decoder_idx = 0
        self._last_frame_time = 0.0
        self._pull_timeout_ns = int(1e8)  # 100ms
        self._no_frame_timeout_sec = 2.0

        self.lock = threading.Lock()
        self.frame = None
        self.connected = False
        self.fps_counter = 0
        self.fps_value = 0.0
        self.fps_last_time = time.time()
        self.total_frames = 0
        self.running = False

    def _build_decoder_candidates(self):
        # 5路及以上默认优先CPU解码，规避硬件解码会话/调度瓶颈。
        if self.total_cameras >= 5:
            preferred = ["avdec_h264", "nvh264dec"]
        else:
            preferred = ["nvh264dec", "avdec_h264"]
        available = [
            decoder for decoder in preferred
            if Gst.ElementFactory.find(decoder) is not None
        ]
        missing = [decoder for decoder in preferred if decoder not in available]
        if missing:
            print(f"[{self.label}] Skipping unavailable decoders: {', '.join(missing)}")
        if not available:
            raise RuntimeError("No supported GStreamer H.264 decoder is installed")
        return available

    def _create_pipeline_str(self, decoder):
        return (
            f'udpsrc port={self.video_port} buffer-size=2097152 '
            f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
            f'! rtph264depay ! h264parse ! {decoder} '
            f'! queue max-size-buffers=1 leaky=downstream '
            f'! videoconvert '
            f'! video/x-raw,format=BGR,width={self.tile_w},height={self.tile_h} '
            f'! appsink name=sink emit-signals=false sync=false drop=true max-buffers=1'
        )

    def _teardown_pipeline(self):
        with self._pipeline_lock:
            pipeline = self.pipeline
            self.pipeline = None
            self._sink = None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)

    def _start_pipeline_with_decoder(self, decoder):
        self._teardown_pipeline()
        pipeline = None
        try:
            pipeline = Gst.parse_launch(self._create_pipeline_str(decoder))
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline failed to PLAYING")
            sink = pipeline.get_by_name('sink')
            if sink is None:
                raise RuntimeError("appsink not found")
            with self._pipeline_lock:
                self.pipeline = pipeline
                self._sink = sink
            self.decoder = decoder
            self.connected = False
            self._last_frame_time = time.time()
            print(f"[{self.label}] Receiving on port {self.video_port} ({decoder})")
            return True
        except Exception as e:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
            print(f"[{self.label}] {decoder} failed: {e}")
            return False

    def _try_start_with_fallback(self, preferred_idx=0):
        n = len(self._decoder_candidates)
        for step in range(n):
            idx = (preferred_idx + step) % n
            dec = self._decoder_candidates[idx]
            if self._start_pipeline_with_decoder(dec):
                self._decoder_idx = idx
                return True
        print(f"[{self.label}] ERROR: all decoders failed")
        return False

    def _recover_pipeline(self, reason):
        if not self.running:
            return
        print(f"[{self.label}] Recovering pipeline ({reason})")
        next_idx = (self._decoder_idx + 1) % len(self._decoder_candidates)
        if self._try_start_with_fallback(preferred_idx=next_idx):
            print(
                f"[{self.label}] Pipeline restarted with {self.decoder}; "
                "awaiting frames"
            )
            return
        print(f"[{self.label}] Recovery failed, retrying in 0.3s")
        time.sleep(0.3)

    def has_recent_frame(self, within_sec=1.5):
        return (time.time() - self._last_frame_time) <= within_sec and self.connected

    def force_reconnect(self, reason="manual"):
        self._recover_pipeline(reason=reason)

    def start(self):
        self.meta.start()
        self.running = True
        if not self._try_start_with_fallback(preferred_idx=0):
            return

        # 轮询线程取帧
        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._thread.start()

    def _pull_loop(self):
        while self.running:
            with self._pipeline_lock:
                sink = self._sink
            if sink is None:
                time.sleep(0.01)
                continue
            sample = sink.emit('try-pull-sample', self._pull_timeout_ns)
            if sample is None:
                if (time.time() - self._last_frame_time) > self._no_frame_timeout_sec:
                    self.connected = False
                    self._recover_pipeline(reason="frame-timeout")
                time.sleep(0.002)
                continue
            buf = sample.get_buffer()
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if ok:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                    self.tile_h, self.tile_w, 3).copy()
                buf.unmap(mapinfo)

                stream_was_connected = self.connected
                self.connected = True
                self._last_frame_time = time.time()
                self.total_frames += 1
                self.fps_counter += 1
                if not stream_was_connected:
                    print(f"[{self.label}] Frame flow active ({self.decoder})")

                now = time.time()
                dt = now - self.fps_last_time
                if dt >= 1.0:
                    self.fps_value = self.fps_counter / dt
                    self.fps_counter = 0
                    self.fps_last_time = now

                with self.lock:
                    self.frame = frame

    def get_display_frame(self):
        with self.lock:
            f = self.frame

        if f is None:
            ph = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
            draw_text(ph, self.label, (20, self.tile_h // 2 - 10), 0.8, (100, 100, 100), 2)
            draw_text(ph, f"Port {self.video_port} waiting...",
                      (20, self.tile_h // 2 + 25), 0.6, (80, 80, 80), 1)
            return ph

        display = f.copy()
        h = display.shape[0]
        s = h / 480.0
        line_gap = int(26 * s)
        fs_l = 0.6 * s
        fs_m = 0.55 * s
        fs_s = 0.5 * s
        pad = int(8 * s)
        y = line_gap

        draw_text(display, self.label, (pad, y), fs_l, (0, 255, 0), 1)
        y += line_gap

        lat = self.meta.latency_ema
        if lat != 0:
            color = (0, 255, 0) if abs(lat) < 50 else (0, 255, 255) if abs(lat) < 100 else (0, 0, 255)
            draw_text(display, f"Latency:{lat:.0f}ms", (pad, y), fs_m, color, 1)
            y += line_gap

        draw_text(display, f"FPS:{self.fps_value:.1f} [{self.decoder}]",
                  (pad, y), fs_s, (200, 200, 200), 1)
        y += line_gap

        drop = self.meta.meta_drop_count
        if self.total_frames > 0:
            drop_color = (0, 255, 0) if drop == 0 else (0, 0, 255)
            draw_text(display, f"F:{self.total_frames} D:{drop}",
                      (pad, y), fs_s, drop_color, 1)

        return display

    def stop(self):
        self.running = False
        self.meta.close()
        if self._thread:
            self._thread.join(timeout=1)
        self._teardown_pipeline()


# ---------------------------------------------------------------------------
# 主显示窗口
# ---------------------------------------------------------------------------

class StreamViewer:
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

        sender_host = self.config.get('sender_host', '192.168.1.101')
        clock_port = self.config.get('clock_sync_port', 7777)
        print(f"Measuring clock offset to {sender_host}:{clock_port}...")
        self.clock_offset_ms = measure_clock_offset(sender_host, clock_port, samples=15, timeout=1.0)

        for i, cam_cfg in enumerate(self.config['cameras']):
            r = CameraReceiver(
                camera_id=i, label=cam_cfg['label'],
                video_port=video_base + i, meta_port=meta_base + i,
                listen_host=listen_host, tile_w=tile_w, tile_h=tile_h,
                clock_offset_ms=self.clock_offset_ms,
            )
            self.receivers.append(r)

        self._sender_host = sender_host
        self._clock_port = clock_port

        print(f"Configured {len(self.receivers)} cameras, "
              f"ports {video_base}-{video_base + len(self.receivers) - 1}")

    def _build_status_bar(self, width):
        connected = sum(1 for r in self.receivers if r.connected)
        latencies = [r.meta.latency_ema for r in self.receivers if r.meta.latency_ema != 0]
        if len(latencies) >= 2:
            sync_text = f"Sync: {max(latencies) - min(latencies):.0f}ms"
        else:
            sync_text = "Sync: --"
        total_drop = sum(r.meta.meta_drop_count for r in self.receivers)
        total_recv = sum(r.total_frames for r in self.receivers)
        if total_recv > 0:
            drop_rate = total_drop / (total_recv + total_drop) * 100
            drop_text = f"Drop: {total_drop} ({drop_rate:.1f}%)"
        else:
            drop_text = "Drop: --"
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            status = (f"Connected: {connected}/{len(self.receivers)}  |  "
                      f"Avg Latency: {avg_lat:.0f}ms  |  {sync_text}  |  "
                      f"{drop_text}  |  Offset: {self.clock_offset_ms:.1f}ms")
        else:
            status = f"Connected: {connected}/{len(self.receivers)}  |  Waiting for data..."
        bar = np.zeros((44, width, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return bar

    def _clock_sync_loop(self):
        while self.running:
            for _ in range(30):
                if not self.running:
                    return
                time.sleep(1)
            new_offset = measure_clock_offset(self._sender_host, self._clock_port, samples=15, timeout=1.0)
            if new_offset != 0.0:
                self.clock_offset_ms = new_offset
                for r in self.receivers:
                    r.meta.clock_offset_ms = new_offset

    def _startup_health_check(self, timeout_sec=6.0, poll_sec=0.2):
        deadline = time.time() + timeout_sec
        while self.running and time.time() < deadline:
            unhealthy = [r for r in self.receivers if not r.has_recent_frame(within_sec=1.5)]
            if not unhealthy:
                print("[Health] Startup check passed")
                return
            time.sleep(poll_sec)

        unhealthy = [r for r in self.receivers if not r.has_recent_frame(within_sec=1.5)]
        if not unhealthy:
            return
        labels = ", ".join(r.label for r in unhealthy)
        print(f"[Health] Startup check found unhealthy cameras: {labels}")
        for r in unhealthy:
            r.force_reconnect(reason="startup-health-check")
        time.sleep(1.0)

    def start(self):
        self.running = True

        for r in self.receivers:
            r.start()

        self._startup_health_check()

        self._clock_thread = threading.Thread(target=self._clock_sync_loop, daemon=True)
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
                    grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
            for c in range(1, cols):
                cv2.line(grid, (c * tile_w, 0), (c * tile_w, rows * tile_h), (60, 60, 60), 1)
            for r in range(1, rows):
                cv2.line(grid, (0, r * tile_h), (canvas_w, r * tile_h), (60, 60, 60), 1)
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
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
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
