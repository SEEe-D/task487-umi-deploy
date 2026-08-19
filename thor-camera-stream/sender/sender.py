#!/usr/bin/env python3
"""
Thor 6路相机低延迟推流端
使用GStreamer硬件编码 + UDP RTP推流
集成GPIO FSIN触发实现6路相机25fps同步采集
"""

import os
import sys
import time
import struct
import socket
import signal
import threading
import yaml
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from gpio_trigger import GPIOTrigger


class CameraStreamer:
    """单路相机的GStreamer推流管线"""

    def __init__(self, camera_id, config, camera_cfg, target_host,
                 video_port, meta_port):
        self.camera_id = camera_id
        self.label = camera_cfg['label']
        self.device = camera_cfg['device_path']
        self.cap_width = camera_cfg['width']
        self.cap_height = camera_cfg['height']
        self.cap_fps = config['cap_fps']
        self.stream_fps = config['stream_fps']
        self.target_host = target_host
        self.video_port = video_port
        self.meta_port = meta_port
        self.encoding = config['encoding']

        stream = config.get('stream_resolution', {})
        self.stream_width = stream.get('width', self.cap_width)
        self.stream_height = stream.get('height', self.cap_height)

        self.pipeline = None
        self.frame_count = 0
        self.meta_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False

    def build_pipeline(self):
        """构建GStreamer推流管线 (使用gst-launch语法, 已验证可靠)"""
        bitrate = self.encoding['bitrate']
        iframeinterval = self.encoding['iframeinterval']
        preset_id = self.encoding['preset_id']

        print(f"[{self.label}] {self.cap_width}x{self.cap_height}@{self.cap_fps} → "
              f"{self.stream_width}x{self.stream_height}@{self.stream_fps}fps → "
              f"{self.target_host}:{self.video_port}")

        # GPIO FSIN 25Hz触发相机采集, v4l2src以cap_fps协商但实际帧率由GPIO控制
        # 不使用videorate: 避免缓冲延迟, GPIO已保证实际帧率为stream_fps
        # nvvidconv做UYVY→NV12颜色转换 + 硬件缩放到stream_resolution
        # capsfilter用程序化方式设置NVMM caps (避免parse_launch括号语法问题)
        pipeline_desc = (
            f'v4l2src device={self.device} io-mode=2 do-timestamp=true name=src '
            f'! video/x-raw,format=UYVY,width={self.cap_width},'
            f'height={self.cap_height},framerate={self.cap_fps}/1 '
            f'! queue max-size-buffers=1 leaky=downstream '
            f'! nvvidconv name=scaler '
            f'! capsfilter name=nvmm_caps '
            f'! nvv4l2h264enc '
            f'  preset-id={preset_id} '
            f'  bitrate={bitrate} '
            f'  iframeinterval={iframeinterval} '
            f'  idrinterval={iframeinterval} '
            f'  num-B-Frames=0 '
            f'  control-rate=1 '
            f'  insert-sps-pps=true '
            f'! h264parse config-interval=1 '
            f'! rtph264pay config-interval=1 pt=96 mtu=1400 name=pay '
            f'! udpsink host={self.target_host} port={self.video_port} '
            f'  sync=false async=false name=sink '
        )

        self.pipeline = Gst.parse_launch(pipeline_desc)

        # 程序化设置nvvidconv输出caps: NVMM内存 + 缩放分辨率
        nvmm_caps = self.pipeline.get_by_name("nvmm_caps")
        caps_str = (f"video/x-raw(memory:NVMM),"
                    f"width={self.stream_width},"
                    f"height={self.stream_height},"
                    f"format=NV12")
        nvmm_caps.set_property("caps", Gst.Caps.from_string(caps_str))

        # bus消息监听
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        bus.connect("message::warning", self._on_warning)
        bus.connect("message::eos", self._on_eos)

        # Probe the payloader sink: one encoded access unit arrives per camera
        # frame. The payloader src can emit GstBufferList batches, so probing
        # it as BUFFER under-counts frames and makes metadata appear stale.
        pay = self.pipeline.get_by_name("pay")
        if pay:
            pad = pay.get_static_pad('sink')
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)

    def _on_buffer_probe(self, pad, info, user_data):
        """Send one metadata packet for each encoded camera frame."""
        self.frame_count += 1
        capture_ts = int(time.time() * 1_000_000)

        meta_packet = struct.pack('<IIqII',
                                  self.camera_id,
                                  self.frame_count,
                                  capture_ts,
                                  self.stream_width,
                                  self.stream_height)
        try:
            self.meta_sock.sendto(meta_packet,
                                  (self.target_host, self.meta_port))
        except Exception:
            pass

        if self.frame_count % (self.stream_fps * 5) == 0:
            print(f"[{self.label}] frames={self.frame_count}, "
                  f"port={self.video_port}")

        return Gst.PadProbeReturn.OK

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[{self.label}] ERROR: {err.message}")
        if debug:
            print(f"[{self.label}] DEBUG: {debug}")

    def _on_warning(self, bus, msg):
        warn, debug = msg.parse_warning()
        # 启动初期Signal lost是GPIO同步前的正常瞬态, 只在出帧后才报警
        if self.frame_count > 0:
            print(f"[{self.label}] WARNING: {warn.message}")

    def _on_eos(self, bus, msg):
        print(f"[{self.label}] End of stream")

    def start(self):
        """启动推流"""
        self.build_pipeline()
        self.running = True
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print(f"[{self.label}] Failed to start pipeline!")
            return False
        print(f"[{self.label}] Streaming started")
        return True

    def stop(self):
        """停止推流"""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.meta_sock.close()
        print(f"[{self.label}] Stopped")


class ClockSyncServer:
    """UDP时钟同步服务, 响应receiver的NTP-like探测请求"""

    def __init__(self, port=7777):
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.settimeout(1.0)
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        print(f"Clock sync server on port {self.port}")

    def _serve(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(32)
                t2 = int(time.time() * 1_000_000)
                if len(data) >= 8:
                    t1 = struct.unpack('<q', data[:8])[0]
                    t3 = int(time.time() * 1_000_000)
                    reply = struct.pack('<qqq', t1, t2, t3)
                    self.sock.sendto(reply, addr)
            except socket.timeout:
                pass
            except OSError:
                break

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.thread:
            self.thread.join(timeout=1)


class MultiCameraStreamer:
    """6路相机推流管理器 (集成GPIO FSIN同步触发)"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.streamers = []
        self.gpio_trigger = None
        self.clock_sync = None
        self.main_loop = None
        self.running = False

    def setup(self):
        target_host = self.config['target_host']
        video_base_port = self.config['video_base_port']
        meta_base_port = self.config['meta_base_port']

        for i, cam_cfg in enumerate(self.config['cameras']):
            if not cam_cfg.get('enabled', True):
                print(f"[{cam_cfg['label']}] Disabled by configuration")
                continue
            streamer = CameraStreamer(
                camera_id=i,
                config=self.config,
                camera_cfg=cam_cfg,
                target_host=target_host,
                video_port=video_base_port + i,
                meta_port=meta_base_port + i,
            )
            self.streamers.append(streamer)

        # 启动时钟同步服务 (receiver启动时会探测)
        clock_port = self.config.get('clock_sync_port', 7777)
        self.clock_sync = ClockSyncServer(port=clock_port)
        self.clock_sync.start()

        # 初始化GPIO FSIN触发器
        gpio_cfg = self.config.get('gpio_trigger', {})
        if gpio_cfg.get('enabled', True):
            cam_gpio = gpio_cfg.get('cam_gpio', GPIOTrigger.DEFAULT_CAM_GPIO)
            trigger_fps = self.config['stream_fps']
            self.gpio_trigger = GPIOTrigger(fps=trigger_fps, cam_gpio=cam_gpio)
            print(f"GPIO FSIN trigger: GPIO{cam_gpio}, {trigger_fps}Hz")
        else:
            print("GPIO FSIN trigger: DISABLED (using videorate fallback)")

        stream_res = self.config.get('stream_resolution', {})
        sw = stream_res.get('width', 'native')
        sh = stream_res.get('height', 'native')
        print(f"Configured {len(self.streamers)} camera streams")
        print(f"Target: {target_host}")
        print(f"Video ports: {video_base_port}-"
              f"{video_base_port + len(self.streamers) - 1}")
        print(f"Meta ports: {meta_base_port}-"
              f"{meta_base_port + len(self.streamers) - 1}")
        print(f"Capture: {self.config['cap_fps']}fps → "
              f"Stream: {self.config['stream_fps']}fps (GPIO FSIN), "
              f"Resolution: {sw}x{sh}")
        print(f"Bitrate: {self.config['encoding']['bitrate'] / 1_000_000:.1f} Mbps")

    def start(self):
        Gst.init(None)
        self.running = True

        # 先启动GPIO FSIN触发, 让相机以目标帧率同步采集
        if self.gpio_trigger:
            if not self.gpio_trigger.start():
                print("ERROR: GPIO FSIN trigger failed to start! "
                      "Need root permission (sudo).")
                self.stop()
                return
            # 等待触发稳定
            time.sleep(0.5)
            print(f"GPIO FSIN trigger running, waiting for cameras to sync...")

        for i, streamer in enumerate(self.streamers):
            if not streamer.start():
                print(f"Failed to start {streamer.label}, aborting")
                self.stop()
                return
            if i < len(self.streamers) - 1:
                time.sleep(0.1)

        print(f"\n{'='*60}")
        print(f"All {len(self.streamers)} cameras streaming!")
        if self.gpio_trigger:
            print(f"GPIO FSIN: {self.config['stream_fps']}Hz sync trigger active")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        self.main_loop = GLib.MainLoop()
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False

        # 先停GStreamer管线
        for streamer in self.streamers:
            streamer.stop()

        # 再停GPIO触发 (停触发后相机不再产生帧)
        if self.gpio_trigger:
            self.gpio_trigger.stop()
            self.gpio_trigger = None

        if self.clock_sync:
            self.clock_sync.stop()
            self.clock_sync = None

        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()
        print("All streams stopped")


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.yaml')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    streamer = MultiCameraStreamer(config_path)
    streamer.setup()

    def signal_handler(sig, frame):
        print("\nReceived signal, stopping...")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    streamer.start()


if __name__ == '__main__':
    main()
