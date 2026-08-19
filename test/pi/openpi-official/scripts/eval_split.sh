#!/bin/bash
cd /workspace/zt/openpi-official
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python examples/umi/split_and_eval_lerobot.py eval \
    --dataset-path /workspace/zt/openpi-official/data/local/umi6_single_arm_6d_rel \
    --policy-config pi05_umi6_single_task_split \
    --checkpoint-dir checkpoints/pi05_umi6_single_task_split/umi6_split/20000
