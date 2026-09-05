"""Single in-flight asynchronous inference worker."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import queue
import threading
import time

import numpy as np

from task487_runtime.contract import ACTION_HORIZON, PolicyRequest, body_actions_to_robot_targets


@dataclass(frozen=True)
class InferenceResult:
    request: PolicyRequest
    targets: np.ndarray | None
    latency_s: float
    error: Exception | None


class InferenceWorker:
    def __init__(self, policy, action_horizon: int = ACTION_HORIZON, recorder=None):
        self._policy = policy
        self._recorder = recorder
        self._sequence = 0
        self._action_horizon = int(action_horizon)
        if self._action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._requests: queue.Queue[PolicyRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._run, name="task487-inference", daemon=True)
        self._thread.start()

    def submit(self, request: PolicyRequest) -> None:
        if self.busy:
            raise RuntimeError("Inference already in flight or awaiting poll")
        self._busy.set()
        try:
            self._sequence += 1
            request = replace(request, request_id=self._sequence)
            self._requests.put_nowait(request)
        except Exception:
            self._busy.clear()
            raise

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def poll(self) -> InferenceResult | None:
        try:
            result = self._results.get_nowait()
            self._busy.clear()
            return result
        except queue.Empty:
            return None

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.monotonic()
            started_wall = time.time()
            output = None
            targets = None
            if self._recorder is not None:
                self._recorder.request(request, started_wall=started_wall, started_monotonic=started)
            try:
                output = self._policy.infer(request.observation)
                actions = np.asarray(output["actions"], dtype=np.float64)
                targets = body_actions_to_robot_targets(
                    actions,
                    request.tcp_bases,
                    action_horizon=self._action_horizon,
                )
                result = InferenceResult(request, targets, time.monotonic() - started, None)
            except Exception as exc:
                result = InferenceResult(request, None, time.monotonic() - started, exc)
            if self._recorder is not None:
                self._recorder.record("response", request.request_id,
                    arrays={"output": output, "robot_targets": targets},
                    round_id=request.round_id, observation_time=request.observation_time,
                    completed_wall=time.time(), completed_monotonic=time.monotonic(),
                    latency_s=result.latency_s,
                    error=None if result.error is None else repr(result.error))
            self._results.put(result)

    def record_disposition(self, result, status, **details):
        if self._recorder is not None:
            self._recorder.record("disposition", result.request.request_id,
                round_id=result.request.round_id, status=status, **details)

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logging.warning("Inference still in flight on shutdown; response log may be incomplete")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
