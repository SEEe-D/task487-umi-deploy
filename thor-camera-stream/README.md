# Thor 6路相机低延迟推流系统

Thor (Jetson Thor) 上的6路IMX307 GMSL相机通过网线直连推流到PC，为模型推理提供实时视频输入。

## 系统架构

```
Thor (推流端, 192.168.1.101)              PC (接收端, 192.168.1.102)
┌─────────────────────┐     网线直连      ┌─────────────────────┐
│ 6x IMX307 GMSL相机  │   UDP RTP x6    │ GStreamer接收解码     │
│ GPIO FSIN 25Hz同步   │ ─────────────> │ nvh264dec GPU硬件解码 │
│ nvv4l2h264enc编码    │   UDP meta x6  │ 2x3网格OpenCV显示    │
│ H264 RTP            │ ─────────────> │ OSD: 延迟/FPS/丢帧   │
└─────────────────────┘                 └─────────────────────┘
```

| 项目 | 规格 |
|------|------|
| 视频通道 | UDP RTP H264, 端口 5000-5005 |
| 元数据通道 | UDP 帧ID+时间戳, 端口 6000-6005 |
| 时钟同步 | chrony NTP (<0.1ms) + UDP辅助校准 (端口 7777) |
| 帧率 | 25fps (GPIO FSIN硬件同步) |
| 分辨率 | 可配置, 默认480×480 (从1920×1536硬件缩放) |
| 编码码率 | 可配置, 默认4Mbps/路 (高质量) |
| 目标延迟 | <50ms (网线直连) |

> 详细技术架构见 `架构设计.md`

---

## 1. 网线直连配置

一根网线直接连接Thor和PC（不需要路由器/交换机），配置静态IP。

### 1.1 查看有线网卡名

两端分别执行，找到有线网卡名称：

```bash
ip link show
```

- Thor: 一般是 `eth0` 或 `enp*`
- PC: 如果是USB转接器，一般是 `enx*` 或 `enp*`

> 插上网线后再执行，找状态为 `UP` 或 `NO-CARRIER` 的以太网口（排除 `lo`、`wl*`、`docker*`）

### 1.2 配置永久静态IP

**Thor端** — 新建文件 `/etc/netplan/99-direct-link.yaml`（`sudo vim` 编辑）：

```yaml
network:
  version: 2
  ethernets:
    eth6:                    # Thor实际有线网卡名 (通过1.1步骤确认)
      addresses:
        - 192.168.1.101/24
      dhcp4: false
```

**PC端** — 新建文件 `/etc/netplan/99-direct-link.yaml`（`sudo vim` 编辑）：

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enx6c1ff757a0e3:         # PC实际有线网卡名 (通过1.1步骤确认)
      addresses:
        - 192.168.1.102/24
      dhcp4: false
```

> PC使用NetworkManager管理网络，所以需要加 `renderer: NetworkManager`。Thor使用systemd-networkd，不需要加。
>
> 如果更换了USB网卡适配器，PC的网卡名会变，需要重新查看并修改。

两端分别执行：

```bash
sudo chmod 600 /etc/netplan/99-direct-link.yaml
sudo netplan apply
```

### 1.3 验证连通

在PC上执行：

```bash
ping 192.168.1.101
```

正常应显示延迟 <1ms。如果不通，检查网线是否插好、网卡名是否正确。

---

## 2. 安装依赖

### 2.1 PC (接收端)

```bash
# GStreamer解码插件
sudo apt install -y gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

# Python依赖
pip install numpy opencv-python pyyaml

# 时钟同步
sudo apt install -y chrony
```

> 也可以一键安装: `cd receiver && bash install_deps.sh`
>
> nvh264dec硬件解码需要NVIDIA GPU + 驱动。没有GPU会自动回退到avdec_h264软件解码。

### 2.2 Thor (推流端)

GStreamer + NVIDIA编码器已预装，只需安装chrony:

```bash
sudo apt install -y chrony
```

---

## 3. 配置时钟同步 (chrony)

chrony是Linux标准时钟同步服务，**不影响其他系统功能**，只是让两台机器的时间保持一致。配置一次后开机自动生效。

> 不配chrony的话，延迟显示会有秒级误差。

### 3.1 PC端 — 在已有配置末尾追加

文件: `/etc/chrony/chrony.conf`

在文件**末尾追加**以下两行（不要删除原有内容）：

```ini
allow 192.168.1.0/24
local stratum 8
```

追加后重启: `sudo systemctl restart chrony`

### 3.2 Thor端 — 替换整个配置文件

文件: `/etc/chrony/chrony.conf`

**替换**为以下内容：

```ini
confdir /etc/chrony/conf.d
server 192.168.1.102 iburst prefer minpoll 2 maxpoll 4
keyfile /etc/chrony/chrony.keys
driftfile /var/lib/chrony/chrony.drift
ntsdumpdir /var/lib/chrony
logdir /var/log/chrony
maxupdateskew 100.0
rtcsync
makestep 1 10
leapsectz right/UTC
```

替换后重启: `sudo systemctl restart chrony`

### 3.3 验证

在Thor上执行:

```bash
chronyc sources
```

看到 `*` 号表示同步成功：

```
^* 192.168.1.102             8   2   377     1   -201us[...] +/- 8ms
```

如果显示 `?` 或 `x`，等1-2分钟，或执行 `sudo chronyc makestep` 强制同步。

---

## 4. 启动

### 4.1 部署推流端代码到Thor

在PC上执行：

```bash
cd sender
bash deploy.sh 192.168.1.101
```

### 4.2 启动推流端 (Thor)

```bash
ssh simpleai@192.168.1.101
cd /home/simpleai/project/thor-stream-sender
sudo python3 sender.py
```

正常输出：

```
Clock sync server on port 7777
GPIO FSIN trigger: GPIO659, 25Hz
GPIO FSIN trigger running, waiting for cameras to sync...
[cam_head_left] Streaming started
...
All 6 cameras streaming!
```

> 首帧输出需要约20-30秒，属正常行为。

### 4.3 启动接收端 (PC)

**先确保推流端已有帧输出**，然后：

```bash
cd receiver
python3 receiver.py
```

正常输出：

```
Clock offset: 0.2ms (best RTT=0.3ms, 15/15 samples)
Configured 6 cameras, ports 5000-5005
Viewer: 1008x716 | Press 'q' or ESC to quit
```

### 4.4 停止

- 接收端: 按 `q` 或 `Ctrl+C`
- 推流端: `Ctrl+C`

---

## 5. IP配置汇总

默认配置: Thor `192.168.1.101` ↔ PC `192.168.1.102`（网线直连）

**如果IP变了，需要改以下4处：**

| # | 说明 | 文件 | 字段 | 当前值 |
|---|------|------|------|--------|
| 1 | 推流目标 (PC的IP) | Thor: `sender/config.yaml` | `target_host` | `192.168.1.102` |
| 2 | 时钟校准 (Thor的IP) | PC: `receiver/config.yaml` | `sender_host` | `192.168.1.101` |
| 3 | chrony同步 (PC的IP) | Thor: `/etc/chrony/chrony.conf` | `server` 行 | `192.168.1.102` |
| 4 | chrony允许网段 | PC: `/etc/chrony/chrony.conf` | `allow` 行 | `192.168.1.0/24` |

改完后两端重启chrony，重新部署并启动程序。

---

## 配置参数说明

### sender/config.yaml

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `target_host` | 接收端PC的IP | `192.168.1.102` |
| `video_base_port` | 视频起始端口 | `5000` |
| `meta_base_port` | 元数据起始端口 | `6000` |
| `stream_fps` | 推流帧率 (GPIO触发) | `25` |
| `stream_resolution` | 推流分辨率 | `480×480` |
| `encoding.bitrate` | H264码率 (bps) | `4000000` (4Mbps) |
| `encoding.iframeinterval` | I帧间隔 | `15` |
| `gpio_trigger.cam_gpio` | GPIO编号 | `659` (PM.00) |

> `bitrate` 控制压缩质量。`stream_resolution` 和 `bitrate` 可按需调整: 224×224+2Mbps(模型推理最小尺寸), 480×480+4Mbps(监控清晰), 960×768+8Mbps(高清)。修改后需重新部署到Thor。

### receiver/config.yaml

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `sender_host` | 推流端Thor的IP | `192.168.1.101` |
| `clock_sync_port` | 时钟同步端口 | `7777` |
| `video_base_port` | 视频起始端口 | `5000` |
| `meta_base_port` | 元数据起始端口 | `6000` |
| `display.tile_width/height` | 单路显示尺寸 | `480×480` |
| `display.grid_cols/rows` | 网格布局 | `3×2` |

---

## OSD显示说明

每路画面左上角：
- **第1行**: 相机标签 (cam_head_left等)
- **第2行**: 延迟 (绿<50ms, 黄<100ms, 红>100ms)
- **第3行**: FPS + 解码器
- **第4行**: F:帧数 D:丢帧数 (有丢帧变红)

底部状态栏：Connected | Avg Latency | Sync (6路延迟差值, 越小越同步) | Drop | Offset

---

## 模型集成

算法工程师将接收端代码集成到推理pipeline时，核心是 `receiver.py` 中的 `CameraReceiver` 类。

### 最简集成示例

```python
import sys
sys.path.append('/path/to/thor-stream/receiver')  # 加入receiver目录

from receiver import CameraReceiver, MetaReceiver, measure_clock_offset

# 1. 测量时钟偏移 (推流端需要已启动)
clock_offset = measure_clock_offset("192.168.1.101", port=7777)

# 2. 创建6路接收器
receivers = []
for i in range(6):
    r = CameraReceiver(
        camera_id=i,
        label=f"cam{i}",
        video_port=5000 + i,
        meta_port=6000 + i,
        listen_host="0.0.0.0",
        tile_w=480,         # 接收到的原始分辨率, 与sender config一致
        tile_h=480,
        clock_offset_ms=clock_offset,
    )
    r.start()
    receivers.append(r)

# 3. 在推理循环中取帧
while True:
    frames = []
    for r in receivers:
        with r.lock:
            frame = r.frame  # numpy array, shape=(H,W,3), dtype=uint8, BGR
        if frame is not None:
            frames.append(frame)

    if len(frames) == 6:
        # frames 是6个BGR numpy数组, 可直接:
        # - cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 转RGB
        # - torch.from_numpy(frame).permute(2,0,1).float() / 255.0 转tensor
        # - cv2.resize(frame, (224, 224)) 缩放到模型输入尺寸
        pass

# 4. 停止
for r in receivers:
    r.stop()
```

### 关键说明

- **frame格式**: `numpy.ndarray`, shape=`(H, W, 3)`, dtype=`uint8`, 颜色空间BGR (OpenCV默认)
- **分辨率**: 与 `sender/config.yaml` 中的 `stream_resolution` 一致 (当前480×480)。如果模型需要224×224，在推理侧用 `cv2.resize` 缩放即可
- **线程安全**: 取帧时必须用 `with r.lock` 加锁
- **无帧时**: `r.frame` 为 `None`，等推流端数据到达后自动更新
- **不需要UI**: 集成时不需要 `StreamViewer` 和 `cv2.imshow`，只用 `CameraReceiver`
- **依赖**: 只需 `numpy`, `opencv-python`, `pyyaml`，以及系统安装的GStreamer插件
- 详见 `架构设计.md` §6

---

## 常见问题

**Q: 启动后长时间没有画面?**
A: 推流端首帧需要20-30秒。确认sender日志有 `frames=125` 后再启动接收端。

**Q: Device or resource busy?**
A: Thor上有残留进程: `sudo pkill -9 -f sender.py; sleep 3`

**Q: 延迟显示为负数或几千ms?**
A: chrony未配好。在Thor上 `chronyc sources` 确认有 `*` 号。

**Q: PC卡顿?**
A: 确保NVIDIA GPU驱动正常，接收端优先用nvh264dec硬件解码。

**Q: 延迟>100ms?**
A: 检查是否走的WiFi而非网线。网线直连ping应<1ms。

**Q: 更换了Thor或PC?**
A: 按第5节修改4处IP配置，重启chrony和程序。

**Q: 如何调整分辨率和画质?**
A: 改 `sender/config.yaml` 的 `stream_resolution` 和 `encoding.bitrate`，改完后重新 `bash deploy.sh` 部署到Thor。接收端 `receiver/config.yaml` 的 `tile_width/height` 也建议同步调整。

**Q: 如何集成到模型推理?**
A: 见"模型集成"章节，完整示例代码可直接复制使用。

---

## 文件结构

```
thor-stream/
├── README.md              # 使用指南 (本文件)
├── 架构设计.md             # 技术架构文档
├── 进度与问题.md           # 开发进度与问题记录
├── sender/                # 推流端 (部署到Thor)
│   ├── sender.py          # 主程序
│   ├── gpio_trigger.py    # GPIO FSIN触发
│   ├── config.yaml        # 推流配置
│   └── deploy.sh          # 部署脚本
└── receiver/              # 接收端 (PC上运行)
    ├── receiver.py        # 主程序
    ├── config.yaml        # 接收配置
    └── install_deps.sh    # 一键安装依赖
    
    
    
    
