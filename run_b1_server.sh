#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
B1_RUNTIME="${SCRIPT_DIR}/cloud_b1_bundle/openpi-wrc-dataloader-opt-runtime"
B1_CHECKPOINT="${1:-${SCRIPT_DIR}/checkpoints/stage2_b1_robot_jax/B1}"
B1_PORT="${2:-8082}"

if [[ ! -d "${B1_RUNTIME}" ]]; then
    echo "B1 runtime not found: ${B1_RUNTIME}" >&2
    exit 1
fi

if [[ ! -d "${B1_CHECKPOINT}" ]]; then
    echo "B1 checkpoint not found: ${B1_CHECKPOINT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="${B1_RUNTIME}:${B1_RUNTIME}/src:${B1_RUNTIME}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${B1_RUNTIME}"
exec "${SCRIPT_DIR}/openpi-official/.venv/bin/python" scripts/serve_policy.py \
    --port "${B1_PORT}" \
    policy:checkpoint \
    --policy.config=stage2_b1_robot_jax \
    --policy.dir="${B1_CHECKPOINT}" \
    --policy.model-dtype=bfloat16
