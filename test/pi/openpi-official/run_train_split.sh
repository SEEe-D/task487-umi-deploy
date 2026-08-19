#!/bin/bash
cd /workspace/zt/openpi-official
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/workspace/zt/openpi-official/data
export WANDB_MODE=offline

# Install transformers_replace patch
pip install transformers==4.53.2 2>/dev/null
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_umi6_single_task_split --exp-name umi6_split
