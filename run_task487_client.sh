#!/usr/bin/env bash
set -euo pipefail

# The robot backend is Tianji/Marvin through the local ROS2 Mink UDP bridge.
# Start /home/simpleai/Code/mjm/eval_mink/start_teleop_replay.sh first.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK487_TASK:-vegetable}"
SERVER_HOST="${TASK487_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${TASK487_SERVER_PORT:-8000}"

# Preserve the original positional interface:
#   run_task487_client.sh [task] [host] [port] [client options...]
# If the first argument is an option, forward all options unchanged so the
# wrapper also accepts task487_client.py's documented --server-host/--task
# interface.
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    TASK="$1"
    shift
    if (( $# > 0 )) && [[ "$1" != -* ]]; then
        SERVER_HOST="$1"
        shift
    fi
    if (( $# > 0 )) && [[ "$1" != -* ]]; then
        SERVER_PORT="$1"
        shift
    fi
fi

export PYTHONPATH="$ROOT/openpi-official/packages/openpi-client/src:$ROOT/universal_manipulation_interface_ur:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi

cd "$ROOT"

# Persist enough evidence to distinguish policy/scheduler motion from the
# target actually emitted by each controller and the EE feedback returned by
# Mink.  Controller subprocesses inherit this directory and write 100 Hz CSV
# traces, waypoint insertion records, FSM events, and gripper traces.
if [[ -z "${UMI_ACTION_LOG_DIR:-}" ]]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
    export UMI_ACTION_LOG_DIR="$ROOT/task487_logs/$RUN_TAG"
fi
mkdir -p "$UMI_ACTION_LOG_DIR"
echo "[Task487] diagnostic logs: $UMI_ACTION_LOG_DIR"

set +e
python -u task487_client.py \
    --task "$TASK" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --thor-receiver-path "$ROOT" \
    "$@" 2>&1 | tee "$UMI_ACTION_LOG_DIR/client.log"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e
echo "[Task487] diagnostic logs saved: $UMI_ACTION_LOG_DIR"
exit "$CLIENT_STATUS"
