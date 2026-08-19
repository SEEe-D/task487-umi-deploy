#!/bin/bash
# 积木任务 - 三种相机配置训练
# 在 S3 训练机上运行: bash scripts/train_block_3cam.sh
set -e

MYDATA="/workspace/umi/mydata"
TASK="building blocks into box"

echo "========== Step 1: 转换数据 =========="

echo "--- 1a: 下方相机 (blockbtmcam) ---"
uv run examples/umi/convert_zarr_to_lerobot.py \
    --zarr_path "$MYDATA/blockbtmcam.zarr.zip" \
    --task "$TASK" \
    --repo_name "local/block_btmcam" \
    --cam_mode btm \
    --image_size 224

echo "--- 1b: 上方相机 (blocktopcam) ---"
uv run examples/umi/convert_zarr_to_lerobot.py \
    --zarr_path "$MYDATA/blocktopcam.zarr.zip" \
    --task "$TASK" \
    --repo_name "local/block_topcam" \
    --cam_mode top \
    --image_size 224

echo "--- 1c: 上下拼图 (blocktopbottomcam) ---"
uv run examples/umi/convert_zarr_to_lerobot.py \
    --zarr_path "$MYDATA/blocktopbottomcam.zarr.zip" \
    --task "$TASK" \
    --repo_name "local/block_topbtmcam" \
    --cam_mode topbtm \
    --image_size 224

echo "========== Step 2: 训练 =========="

echo "--- 2a: 下方相机模型 ---"
uv run scripts/train.py --config pi05_block_btmcam_h20

echo "--- 2b: 上方相机模型 ---"
uv run scripts/train.py --config pi05_block_topcam_h20

echo "--- 2c: 上下拼图模型 ---"
uv run scripts/train.py --config pi05_block_topbtmcam_h20

echo "========== 全部完成 =========="
