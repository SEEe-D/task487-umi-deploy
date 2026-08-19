#!/bin/bash
# ============================================================
# pi0.5 UMI6 单臂 (右臂) LoRA 微调 (~25GB VRAM/卡)
#
# 多卡: JAX 自动检测所有 GPU, 做 data parallelism
# 指定 GPU: export CUDA_VISIBLE_DEVICES=0,1,2,3
#
# Usage:
#   cd /path/to/openpi-official
#   bash scripts/finetune_umi6_lora.sh
# ============================================================

set -euo pipefail

# 数据集在项目目录下 data/，LeRobot 通过此环境变量定位
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_LEROBOT_HOME="${SCRIPT_DIR}/../data"

EXP_NAME="${1:-umi6_lora}"

echo "===== pi0.5 UMI6 LoRA 微调 ====="
echo "Config: pi05_umi6_lora"
echo "Exp: ${EXP_NAME}"

# 1. 计算归一化统计量
echo ">>> 计算 norm stats..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py \
    --config-name=pi05_umi6_lora

# 2. 启动训练
echo ">>> 启动训练..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_umi6_lora \
    --exp-name="${EXP_NAME}" \
    --overwrite
