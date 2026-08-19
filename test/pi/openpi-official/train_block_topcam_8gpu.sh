#!/bin/sh
set -eu

cd /workspace/zt/openpi-official
export HF_LEROBOT_HOME="/workspace/zt/openpi-official/data"
export WANDB_MODE=disabled
export PATH="/workspace/zt/openpi-official/.venv/bin:$PATH"

CONFIG_NAME="pi05_block_topcam_h20"
CKPT_DIR="/workspace/zt/checkpoints/${CONFIG_NAME}/${CONFIG_NAME}"
STATS_DIR="assets/${CONFIG_NAME}/local/block_topcam"

# 1. 数据转换
echo ">>> 转换数据: blocktopcam (上方相机)"
python examples/umi/convert_zarr_to_lerobot.py \
    --zarr_path /workspace/umi/mydata/blocktopcam.zarr.zip \
    --task "building blocks into box" \
    --repo_name "local/block_topcam" \
    --cam_mode top \
    --image_size 224

# 2. 计算 norm_stats
if [ -f "$STATS_DIR/norm_stats.json" ]; then
    echo ">>> norm_stats 已存在，跳过"
else
    echo ">>> 计算 norm_stats..."
    python scripts/compute_norm_stats.py --config-name=$CONFIG_NAME
fi

# 3. 训练
if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR"/*/params 1>/dev/null 2>&1; then
    echo ">>> 发现已有 checkpoint，resume 训练"
    python scripts/train.py $CONFIG_NAME \
        --exp-name=${CONFIG_NAME} \
        --resume
else
    echo ">>> 无 checkpoint，从头训练"
    python scripts/train.py $CONFIG_NAME \
        --exp-name=${CONFIG_NAME} \
        --overwrite
fi

echo ">>> 训练完成!"
