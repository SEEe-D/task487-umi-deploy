#!/usr/bin/env bash
# Trial requested on 2026-09-05: both hands close -5deg / open +5deg.
# Same [task] [host] [port] [options] interface; startup remains keyboard-controlled HOLD.
set -euo pipefail
task487_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$task487_root/run_task487_client_sync.sh" "$@" \
    --sync-gripper-close-compensation-deg 5 \
    --sync-gripper-open-compensation-deg 5 \
    --sync-right-before-left
