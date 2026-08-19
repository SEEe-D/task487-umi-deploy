#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$ROOT/openpi-official"
DEFAULT_CHECKPOINT="$ROOT/checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999"
CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
PORT="${2:-8000}"
POLICY_CONFIG="${TASK487_POLICY_CONFIG:-pi05_umi_task487}"

if [[ ! -d "$CHECKPOINT/params" && ! -f "$CHECKPOINT/model.safetensors" ]]; then
    echo "Invalid checkpoint: $CHECKPOINT" >&2
    exit 2
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"

cd "$OPENPI_ROOT"
exec .venv/bin/python scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$CHECKPOINT"
