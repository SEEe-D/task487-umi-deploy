#!/bin/bash
# ============================================================
# pi0.5 UMI6 单臂 LoRA train/test split 版微调 (~25GB VRAM/卡)
#
# 只用 train split 训练，训练完成后用 split_and_eval_lerobot.py eval
# 在 test split 上评估 MSE，验证管线正确性。
#
# 前置步骤:
#   cd /path/to/openpi-official
#   uv run examples/umi/split_and_eval_lerobot.py split \
#     --dataset-path data/local/umi6_single_arm_6d_rel
#
# Usage:
#   bash scripts/finetune_umi6_lora_split.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_LEROBOT_HOME="${SCRIPT_DIR}/../data"

EXP_NAME="${1:-umi6_lora_split}"

echo "===== pi0.5 UMI6 LoRA train split 微调 ====="
echo "Config: pi05_umi6_lora_split"
echo "Exp: ${EXP_NAME}"

# 1. 计算归一化统计量 (使用全量数据集, 与部署版一致)
echo ">>> 计算 norm stats..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py \
    --config-name=pi05_umi6_lora

# 2. 启动训练 (只用 train episodes)
echo ">>> 启动训练 (train split only)..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_umi6_lora_split \
    --exp-name="${EXP_NAME}" \
    --overwrite
