#!/usr/bin/env bash
# Ordinary WebSocket + author-style Bezier. Existing rolling launcher unchanged.
# Same positional interface: [task] [host] [port] [options]. No automatic motion.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_task487_client.sh" "$@" --execution-mode author-sync
