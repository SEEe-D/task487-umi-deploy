#!/bin/bash
# 接收端 (PC) 依赖安装脚本
set -e

echo "=== 安装接收端依赖 ==="

# GStreamer 插件 (H264解码等)
echo "[1/3] 安装 GStreamer 插件..."
sudo apt install -y \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-good \
    python3-gi \
    gir1.2-gstreamer-1.0

# chrony 时钟同步
echo "[2/3] 安装 chrony 时钟同步..."
sudo apt install -y chrony

# 配置PC为NTP服务器 (允许局域网设备同步)
if ! grep -q "Allow LAN devices to sync" /etc/chrony/chrony.conf 2>/dev/null; then
    echo '
# Allow LAN devices to sync
allow 192.168.1.0/24
local stratum 8' | sudo tee -a /etc/chrony/chrony.conf
    sudo systemctl restart chrony
    echo "  chrony: configured as NTP server"
else
    echo "  chrony: already configured"
fi

# 验证
echo ""
echo "[3/3] 验证..."
gst-inspect-1.0 avdec_h264 > /dev/null 2>&1 && echo "  avdec_h264: OK" || echo "  avdec_h264: FAILED"
gst-inspect-1.0 h264parse > /dev/null 2>&1 && echo "  h264parse: OK" || echo "  h264parse: FAILED"
gst-inspect-1.0 rtph264depay > /dev/null 2>&1 && echo "  rtph264depay: OK" || echo "  rtph264depay: FAILED"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" 2>&1
python3 -c "import yaml; print('  PyYAML: OK')" 2>&1 || {
    echo "  Installing PyYAML..."
    pip3 install pyyaml
}
chronyc tracking > /dev/null 2>&1 && echo "  chrony: running" || echo "  chrony: NOT running"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "注意: 如果更换了PC的IP网段, 需要修改 /etc/chrony/chrony.conf 中的 allow 行"
