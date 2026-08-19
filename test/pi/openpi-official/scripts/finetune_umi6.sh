#!/bin/bash
# ============================================================
# pi0.5 UMI6 单臂 (右臂) 微调 (全参数, ~40GB VRAM/卡)
#
# 多卡: JAX 自动检测所有 GPU, 做 data parallelism
#   batch_size 必须被 GPU 数量整除 (当前 64 / 8 GPU = 每卡 8)
# 指定 GPU: export CUDA_VISIBLE_DEVICES=0,1,2,3
#
# Usage:
#   cd /path/to/openpi-official
#   bash scripts/finetune_umi6.sh
# ============================================================

set -euo pipefail

# 数据集在项目目录下 data/，LeRobot 通过此环境变量定位
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_LEROBOT_HOME="${SCRIPT_DIR}/../data"

EXP_NAME="${1:-umi6_finetune}"

echo "===== pi0.5 UMI6 单任务微调 ====="
echo "Config: pi05_umi6_single_task"
echo "Exp: ${EXP_NAME}"

# 1. 计算归一化统计量 (首次需要, 已有则跳过)
echo ">>> 计算 norm stats..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py \
    --config-name=pi05_umi6_single_task

# 2. 启动训练
echo ">>> 启动训练..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_umi6_single_task \
    --exp-name="${EXP_NAME}" \
    --overwrite
