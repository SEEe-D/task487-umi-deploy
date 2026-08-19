import time

import numpy as np

from task487_runtime.contract import PolicyRequest
from task487_runtime.worker import InferenceWorker


class FakePolicy:
    def infer(self, _observation):
        actions = np.zeros((20, 20), dtype=np.float64)
        actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
        actions[:, 13:19] = [1, 0, 0, 0, 1, 0]
        return {"actions": actions}


class FakeB1Policy:
    def infer(self, _observation):
        actions = np.zeros((16, 20), dtype=np.float64)
        actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
        actions[:, 13:19] = [1, 0, 0, 0, 1, 0]
        return {"actions": actions}


def test_worker_keeps_request_anchor_with_converted_targets():
    request = PolicyRequest({"prompt": "test"}, 12.5, np.zeros((2, 6)), round_id=23)
    worker = InferenceWorker(FakePolicy())
    worker.submit(request)
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = worker.poll()
        time.sleep(0.001)
    worker.close()
    assert result is not None
    assert result.error is None
    assert result.request.observation_time == 12.5
    assert result.request.round_id == 23
    assert result.targets.shape == (20, 14)


def test_worker_accepts_server_advertised_b1_horizon():
    request = PolicyRequest({"prompt": "test"}, 12.5, np.zeros((2, 6)))
    worker = InferenceWorker(FakeB1Policy(), action_horizon=16)
    worker.submit(request)
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = worker.poll()
        time.sleep(0.001)
    worker.close()
    assert result is not None
    assert result.error is None
    assert result.targets.shape == (16, 14)
