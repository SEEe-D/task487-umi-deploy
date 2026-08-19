"""Single in-flight asynchronous inference worker."""

from __future__ import annotations

from dataclasses import dataclass
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
    def __init__(self, policy, action_horizon: int = ACTION_HORIZON):
        self._policy = policy
        self._action_horizon = int(action_horizon)
        if self._action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._requests: queue.Queue[PolicyRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, name="task487-inference", daemon=True)
        self._thread.start()

    def submit(self, request: PolicyRequest) -> None:
        self._requests.put_nowait(request)

    def poll(self) -> InferenceResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.monotonic()
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
            self._results.put(result)

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join(timeout=2.0)
