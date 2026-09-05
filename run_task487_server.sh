#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$ROOT/openpi-official"
DEFAULT_CHECKPOINT="$ROOT/checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999"
CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
PORT="${2:-8000}"
# Short names bind the exact experiment to its four-wrist deployment config.
case "$CHECKPOINT" in
    cnc) CHECKPOINT="$ROOT/checkpoints/pi05_umi_task487_cnc_12_5/cnc8gpu_seed42/29999" ;;
    raw4w) CHECKPOINT="$ROOT/checkpoints/pi05_umi_task487_raw_12_5/raw4w_seed42/29999" ;;
    wrist4w) CHECKPOINT="$ROOT/checkpoints/pi05_umi_task487_wrist_only_12_5/wrist4w_seed42/29999" ;;
esac

if [[ ! -d "$CHECKPOINT/params" && ! -f "$CHECKPOINT/model.safetensors" ]]; then
    echo "Invalid checkpoint: $CHECKPOINT" >&2
    exit 2
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
EXPECTED_CONFIG=""
case "$CHECKPOINT" in
    */pi05_umi_task487_cnc_12_5/cnc8gpu_seed42/29999)
        EXPECTED_CONFIG=pi05_umi_task487_cnc_4w_12_5 ;;
    */pi05_umi_task487_raw_12_5/raw4w_seed42/29999)
        EXPECTED_CONFIG=pi05_umi_task487_raw_4w_12_5 ;;
    */pi05_umi_task487_wrist_only_12_5/wrist4w_seed42/29999)
        EXPECTED_CONFIG=pi05_umi_task487_wrist_only_4w_12_5 ;;
esac
if [[ -n "$EXPECTED_CONFIG" && -n "${TASK487_POLICY_CONFIG:-}" && "$TASK487_POLICY_CONFIG" != "$EXPECTED_CONFIG" ]]; then
    echo "Checkpoint requires $EXPECTED_CONFIG; unset stale TASK487_POLICY_CONFIG=$TASK487_POLICY_CONFIG" >&2
    exit 2
fi
POLICY_CONFIG="${TASK487_POLICY_CONFIG:-${EXPECTED_CONFIG:-pi05_umi_task487}}"
echo "[Task487] config=$POLICY_CONFIG checkpoint=$CHECKPOINT port=$PORT"

cd "$OPENPI_ROOT"
exec .venv/bin/python scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$CHECKPOINT"
