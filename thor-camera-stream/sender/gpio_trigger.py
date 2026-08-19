"""
GPIO FSIN 相机同步触发模块
移植自 UMI dual_trigger.c, 仅保留相机触发部分

通过 GPIO sysfs 产生指定频率的方波,
触发 IMX307 GMSL 相机的 FSIN 引脚实现同步采集.

需要 root 权限操作 GPIO sysfs.
"""

import os
import time
import signal
import threading


class GPIOTrigger:
    """GPIO方波触发器, 控制相机FSIN同步采集帧率"""

    # Jetson Thor 默认GPIO编号
    DEFAULT_CAM_GPIO = 659   # PM.00, 相机FSIN触发

    def __init__(self, fps=25, cam_gpio=None):
        self.fps = fps
        self.cam_gpio = cam_gpio or self.DEFAULT_CAM_GPIO
        self._cam_fd = -1
        self._exported = False
        self._stop = threading.Event()
        self._thread = None
        self._frame_count = 0
        self._val_path = None
        self.last_trigger_us = 0  # 最近一次GPIO触发的时间戳 (微秒, epoch)

    def _sysfs_write(self, path, val, quiet=False):
        """写sysfs节点"""
        try:
            fd = os.open(path, os.O_WRONLY)
            os.write(fd, val.encode())
            os.close(fd)
            return True
        except OSError as e:
            if not quiet:
                print(f"[gpio] sysfs write {path} failed: {e}")
            return False

    def _gpio_export_find(self, gpio_num):
        """Export GPIO并找到sysfs value节点路径"""
        num_str = str(gpio_num)
        gpio_dir = "/sys/class/gpio/"
        skip = {"export", "unexport"}

        def _list_gpio_nodes():
            """列出当前所有非gpiochip的GPIO节点"""
            nodes = set()
            for name in os.listdir(gpio_dir):
                if name.startswith('.') or name.startswith('gpiochip'):
                    continue
                if name in skip:
                    continue
                nodes.add(name)
            return nodes

        # 先尝试unexport清理 (首次运行时会失败, 属正常)
        self._sysfs_write("/sys/class/gpio/unexport", num_str, quiet=True)
        time.sleep(0.15)

        # 记录export前的节点列表
        before = _list_gpio_nodes()

        if not self._sysfs_write("/sys/class/gpio/export", num_str):
            return None
        time.sleep(0.2)

        # 先试经典数字路径
        classic = f"/sys/class/gpio/gpio{gpio_num}/value"
        if os.path.exists(classic):
            return classic

        # 对比export前后的节点, 找到新增的那个 (Jetson Thor用标签命名)
        after = _list_gpio_nodes()
        new_nodes = after - before
        if new_nodes:
            name = new_nodes.pop()
            vp = os.path.join(gpio_dir, name, "value")
            if os.path.exists(vp):
                return vp

        print(f"[gpio] GPIO{gpio_num} sysfs node not found")
        return None

    def _setup_gpio_out(self, val_path):
        """设置GPIO方向为output, 返回value文件的fd"""
        dir_path = os.path.join(os.path.dirname(val_path), "direction")
        if not self._sysfs_write(dir_path, "out"):
            return -1
        try:
            return os.open(val_path, os.O_WRONLY)
        except OSError as e:
            print(f"[gpio] open value failed: {e}")
            return -1

    def _trigger_loop(self):
        """GPIO方波触发主循环 (在独立线程中运行)"""
        half_ns = 1_000_000_000 // self.fps // 2  # 半周期 (纳秒)
        half_s = half_ns / 1e9

        self._frame_count = 0

        while not self._stop.is_set():
            # 上升沿 — 此刻6路相机同时触发采集
            os.pwrite(self._cam_fd, b"1", 0)
            self.last_trigger_us = int(time.time() * 1_000_000)
            self._frame_count += 1

            if self._frame_count % (self.fps * 5) == 0:
                print(f"[gpio] camera {self._frame_count} frames triggered")

            # 半周期等待
            time.sleep(half_s)
            if self._stop.is_set():
                break

            # 下降沿
            os.pwrite(self._cam_fd, b"0", 0)

            # 半周期等待
            time.sleep(half_s)

    def start(self):
        """启动GPIO触发 (需要root权限)"""
        # 卸载可能占用GPIO的内核模块 (Thor系统自带的gpio_trigger_framerate)
        os.system("rmmod gpio_trigger_framerate 2>/dev/null")
        time.sleep(0.2)

        # Export并找到GPIO节点
        self._val_path = self._gpio_export_find(self.cam_gpio)
        if not self._val_path:
            print(f"[gpio] ERROR: Failed to export GPIO{self.cam_gpio}")
            return False
        self._exported = True

        # 设置输出方向
        self._cam_fd = self._setup_gpio_out(self._val_path)
        if self._cam_fd < 0:
            self.stop()
            return False

        half_ms = 1000.0 / self.fps / 2
        print(f"[gpio] Camera FSIN trigger: GPIO{self.cam_gpio} ({self._val_path}), "
              f"{self.fps}Hz, half-period={half_ms:.2f}ms")

        # 启动触发线程
        self._stop.clear()
        self._thread = threading.Thread(target=self._trigger_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止GPIO触发并清理"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        if self._cam_fd >= 0:
            try:
                os.pwrite(self._cam_fd, b"0", 0)
                os.close(self._cam_fd)
            except OSError:
                pass
            self._cam_fd = -1

        if self._exported:
            self._sysfs_write("/sys/class/gpio/unexport", str(self.cam_gpio))
            self._exported = False

        print(f"[gpio] Stopped after {self._frame_count} frames")

    @property
    def frame_count(self):
        return self._frame_count
