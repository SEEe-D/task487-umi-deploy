"""Low-overhead CSV logging for policy/controller synchronization diagnostics."""

from __future__ import annotations

import atexit
import csv
import os
import time
from pathlib import Path


class BufferedCsvLogger:
    """Buffered CSV writer enabled by ``UMI_ACTION_LOG_DIR``.

    The file is flushed at most once per second so 100 Hz controller traces do
    not introduce the timing jitter they are intended to diagnose.
    """

    def __init__(self, filename, fieldnames, flush_interval=1.0):
        self.path = None
        self._fh = None
        self._writer = None
        self._flush_interval = float(flush_interval)
        self._last_flush = time.monotonic()

        root = os.environ.get("UMI_ACTION_LOG_DIR")
        if not root:
            return

        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.path = root_path / filename
        self._fh = self.path.open("w", newline="", buffering=1024 * 1024)
        self._writer = csv.DictWriter(
            self._fh, fieldnames=list(fieldnames), extrasaction="ignore")
        self._writer.writeheader()
        atexit.register(self.close)

    @property
    def enabled(self):
        return self._writer is not None

    def write(self, row):
        if self._writer is None:
            return
        self._writer.writerow(row)
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._fh.flush()
            self._last_flush = now

    def close(self):
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._fh = None
            self._writer = None
