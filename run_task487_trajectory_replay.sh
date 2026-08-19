#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/openpi-official/packages/openpi-client/src:$ROOT/universal_manipulation_interface_ur:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi
cd "$ROOT"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_trajectory_replay"
export UMI_ACTION_LOG_DIR="$ROOT/task487_logs/$RUN_TAG"
mkdir -p "$UMI_ACTION_LOG_DIR"

exec python -u task487_trajectory_replay.py \
  --dataset datasets/task487_cloud_sample \
  --file-index 2 --episode 66 --full-episode --absolute-dataset --frequency 5 --move-grippers \
  --output "$UMI_ACTION_LOG_DIR/tracking_report.json" "$@"
