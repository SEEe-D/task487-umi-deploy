#!/bin/bash
# ============================================================
# pi0.5 UMI5 单臂 LoRA 微调 (~25GB VRAM/卡)
#
# 多卡: JAX 自动检测所有 GPU, 做 data parallelism
# 指定 GPU: export CUDA_VISIBLE_DEVICES=0,1,2,3
#
# Usage:
#   cd /path/to/openpi-official
#   bash scripts/finetune_umi5_lora.sh
# ============================================================

set -euo pipefail

EXP_NAME="${1:-umi5_lora}"

echo "===== pi0.5 UMI5 LoRA 微调 ====="
echo "Config: pi05_umi5_lora"
echo "Exp: ${EXP_NAME}"

# 1. 计算归一化统计量
echo ">>> 计算 norm stats..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py \
    --config-name=pi05_umi5_lora

# 2. 启动训练
echo ">>> 启动训练..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_umi5_lora \
    --exp-name="${EXP_NAME}" \
    --overwrite
