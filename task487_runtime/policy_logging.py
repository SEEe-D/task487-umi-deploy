"""Bounded asynchronous evidence recording, outside the motion-control path."""
from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import shutil
import threading
import time
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def array_tree(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Own every leaf; compression never races with policy input/output reuse."""
    if isinstance(value, dict):
        result = {}
        for name, child in value.items():
            result.update(array_tree(child, f"{prefix}/{name}" if prefix else str(name)))
        return result
    if value is None:
        return {}
    array = np.array(value, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot archive object array at {prefix}")
    return {prefix: array}


class PolicyExchangeRecorder:
    """Exact request RGB/state and pre-conversion response, with stable request IDs.

    Queue overflow and storage errors are visible in status/counters and logs;
    neither disk I/O nor compression runs in the control/inference thread.
    """

    def __init__(self, output_dir, *, metadata=None, queue_size=32,
                 image_budget_bytes=2 * 1024**3, min_free_bytes=512 * 1024**2):
        self.root = pathlib.Path(output_dir)
        self.session_id = f"{time.time_ns()}_{os.getpid()}"
        self.path = self.root / "policy_io" / self.session_id
        self.events_path = self.root / "policy_io.jsonl"
        self.status_path = self.root / "policy_logging_status.json"
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stats = dict(enqueued=0, written=0, dropped=0, errors=0,
                           images_omitted=0, last_dropped_request_id=None, last_error=None)
        self._image_bytes = 0
        self._image_budget = image_budget_bytes
        self._min_free = min_free_bytes
        self._metadata = metadata or {}
        self._thread = threading.Thread(target=self._run, name="task487-log-writer", daemon=True)
        self._thread.start()

    def _problem(self, field, message, request_id=None):
        with self._lock:
            self._stats[field] += 1
            if field == "dropped": self._stats["last_dropped_request_id"] = request_id
            if field == "errors": self._stats["last_error"] = message
            count = self._stats[field]
        if count == 1 or count % 100 == 0:
            logging.warning("Policy logging %s (%d): %s", field, count, message)

    def record(self, event, request_id=0, *, arrays=None, **metadata):
        try:
            if self._stop.is_set():
                self._problem("dropped", "recorder already closing", request_id)
                return
            item = (event, int(request_id), array_tree(arrays or {}),
                    json_value(dict(event_time=time.time(), event_monotonic=time.monotonic(), **metadata)))
            self._queue.put_nowait(item)
            with self._lock: self._stats["enqueued"] += 1
        except queue.Full:
            self._problem("dropped", f"writer queue full; event={event} request={request_id}", request_id)
        except Exception as exc:
            self._problem("errors", f"snapshot {event}/{request_id}: {exc}")

    def request(self, request, *, started_wall, started_monotonic):
        self.record("request", request.request_id,
            arrays={"observation": request.observation, "tcp_bases": request.tcp_bases},
            round_id=request.round_id, observation_time=request.observation_time,
            started_wall=started_wall, started_monotonic=started_monotonic,
            diagnostics=request.diagnostics)

    def _write_item(self, event, request_id, arrays, metadata):
        started = time.monotonic()
        image_keys = [k for k, a in arrays.items() if a.ndim >= 2 and
                      (k.startswith("observation/cam_") or k == "observation/fixed_head_mask")]
        image_bytes = sum(arrays[k].nbytes for k in image_keys)
        omitted = []
        if image_keys and (self._image_bytes + image_bytes > self._image_budget or
                           shutil.disk_usage(self.root).free < self._min_free):
            omitted = image_keys
            arrays = {k: v for k, v in arrays.items() if k not in image_keys}
            self._problem("images_omitted", "image budget/free-space limit reached; numeric records continue")
        file_name = None
        if arrays:
            dest = self.path / f"{event}_{request_id:06d}.npz"
            temporary = dest.with_suffix(".npz.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(dest)
            finally:
                temporary.unlink(missing_ok=True)
            file_name = str(dest.relative_to(self.root))
            if not omitted: self._image_bytes += image_bytes
        row = dict(schema_version=1, session_id=self.session_id, event=event, request_id=request_id,
                   array_file=file_name, omitted_image_keys=omitted, **metadata)
        row["writer_time_ms"] = (time.monotonic()-started)*1000
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        with self._lock: self._stats["written"] += 1

    def _write_status(self):
        with self._lock: status = dict(self._stats)
        status.update(session_id=self.session_id, updated_time=time.time(), queue_pending=self._queue.qsize(),
            state="closed" if self._stop.is_set() and self._queue.empty() else "running",
            image_uncompressed_bytes=self._image_bytes, image_budget_bytes=self._image_budget)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(self.status_path)

    def _run(self):
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._write_item("session", 0, {}, dict(metadata=json_value(self._metadata),
                created_time=time.time(), created_monotonic=time.monotonic(),
                image_format="lossless uint8 RGB as sent in request", npz_allow_pickle=False))
        except Exception as exc:
            self._problem("errors", f"initialize: {exc}")
        last_status = 0.0
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=.1)
            except queue.Empty:
                item = None
            if item is not None:
                try: self._write_item(*item)
                except Exception as exc: self._problem("errors", f"write {item[0]}/{item[1]}: {exc}")
                finally: self._queue.task_done()
            if time.monotonic()-last_status >= 1:
                try: self._write_status()
                except Exception as exc: self._problem("errors", f"status: {exc}")
                last_status = time.monotonic()
        try: self._write_status()
        except Exception as exc: self._problem("errors", f"final status: {exc}")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logging.warning("Policy log writer still draining; pending=%d", self._queue.qsize())

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
