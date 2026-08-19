#!/bin/bash
# 部署推流端代码到Thor
# 用法: bash deploy.sh [THOR_IP]
#
# 默认Thor IP: 192.168.1.101 (网线直连)
# 如果Thor IP不同: bash deploy.sh 192.168.x.x

set -e

THOR_HOST="${1:-192.168.1.101}"
THOR_USER="simpleai"
THOR_DIR="/home/simpleai/project/thor-stream-sender"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 部署推流端到 Thor ($THOR_HOST) ==="

# 创建远程目录
echo "[1/3] 创建远程目录..."
ssh "$THOR_USER@$THOR_HOST" "mkdir -p $THOR_DIR"

# 拷贝文件
echo "[2/3] 拷贝文件..."
scp "$SCRIPT_DIR/sender.py" \
    "$SCRIPT_DIR/gpio_trigger.py" \
    "$SCRIPT_DIR/config.yaml" \
    "$THOR_USER@$THOR_HOST:$THOR_DIR/"

# 检查依赖
echo "[3/3] 检查远程依赖..."
ssh "$THOR_USER@$THOR_HOST" "
    echo '--- Python GStreamer ---'
    python3 -c \"import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('OK')\" 2>&1
    echo '--- PyYAML ---'
    python3 -c 'import yaml; print(\"OK\")' 2>&1 || {
        echo 'Installing PyYAML...'
        pip3 install pyyaml --break-system-packages 2>&1 | tail -3
    }
    echo '--- GStreamer Elements ---'
    gst-inspect-1.0 nvv4l2h264enc > /dev/null 2>&1 && echo 'nvv4l2h264enc: OK' || echo 'nvv4l2h264enc: MISSING!'
    gst-inspect-1.0 nvvidconv > /dev/null 2>&1 && echo 'nvvidconv: OK' || echo 'nvvidconv: MISSING!'
    echo '--- chrony ---'
    chronyc tracking > /dev/null 2>&1 && echo 'chrony: running' || echo 'chrony: NOT configured (see README §2.2)'
"

echo ""
echo "=== 部署完成 ==="
echo "在Thor上运行: ssh $THOR_USER@$THOR_HOST"
echo "             cd $THOR_DIR && sudo python3 sender.py"
