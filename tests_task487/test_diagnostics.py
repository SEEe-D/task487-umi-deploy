import json

import numpy as np

from task487_runtime.diagnostics import ChunkDiagnosticRecorder
from task487_runtime.scheduler import RollingScheduler, SchedulerConfig


def _targets(count=20, step=0.001, start=0.0):
    result = np.zeros((count, 14), dtype=np.float64)
    values = start + np.arange(1, count + 1) * step
    result[:, 0] = values
    result[:, 7] = values
    return result


def _rtc_result(prefix, count=20, step=-0.004):
    result = np.zeros((count, 14), dtype=np.float64)
    result[: len(prefix)] = prefix
    anchor = float(prefix[-1, 0])
    for index in range(len(prefix), count):
        value = anchor + (index - len(prefix) + 1) * step
        result[index, 0] = value
        result[index, 7] = value
    return result


def test_chunk_diagnostics_separate_raw_blended_and_dispatched_paths(tmp_path):
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    scheduler.activate()
    scheduler.mark_request_started()
    scheduler.merge_chunk(
        _targets(), observation_time=1.0, now=1.0, live_target=np.zeros(14)
    )
    while scheduler.pop_next(np.zeros(14), now=1.0) is not None:
        pass

    completion_target = scheduler.diagnostic_snapshot().targets[4]
    completion_time = scheduler.diagnostic_snapshot().target_times[4] + 1e-6
    assert scheduler.advance(completion_time, completion_target) == 5
    scheduler.mark_request_started()
    before = scheduler.diagnostic_snapshot()
    raw_targets = _rtc_result(before.request_prefix_targets)
    stats = scheduler.merge_chunk(
        raw_targets,
        observation_time=2.0,
        now=2.0,
        live_target=before.request_prefix_targets[-1],
    )
    after = scheduler.diagnostic_snapshot()

    recorder = ChunkDiagnosticRecorder(tmp_path)
    array_path = recorder.record(
        round_id=3,
        observation_time=2.0,
        merge_time=2.1,
        inference_latency_s=0.085,
        raw_targets=raw_targets,
        live_target=before.request_prefix_targets[-1],
        controller_target=before.request_prefix_targets[-1],
        tcp_bases=np.zeros((2, 6)),
        before=before,
        after=after,
        stats=stats,
    )

    summary = json.loads((tmp_path / "chunk_diagnostics.jsonl").read_text())
    assert summary["status"] == "accepted"
    assert summary["requested_prefix"] == 5
    assert summary["generated_start"] == 5
    assert summary["handoff_blended"] == 5
    assert np.isclose(summary["right"]["old_vs_raw_direction_cos"], -1.0)
    assert np.isclose(summary["left"]["old_vs_raw_direction_cos"], -1.0)

    arrays = np.load(array_path)
    np.testing.assert_allclose(arrays["raw_targets"], raw_targets)
    np.testing.assert_allclose(arrays["request_prefix_targets"], before.request_prefix_targets)
    np.testing.assert_allclose(arrays["post_queue_targets"], after.targets)
    assert not np.allclose(arrays["raw_suffix"][:5], arrays["blended_suffix"][:5])


def test_scheduler_diagnostic_snapshot_does_not_alias_live_queue():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    scheduler.activate()
    scheduler.mark_request_started()
    scheduler.merge_chunk(
        _targets(), observation_time=1.0, now=1.0, live_target=np.zeros(14)
    )

    snapshot = scheduler.diagnostic_snapshot()
    snapshot.targets[0, 0] = 999.0
    snapshot.sent[:] = True
    fresh = scheduler.diagnostic_snapshot()

    assert fresh.targets[0, 0] != 999.0
    assert not fresh.sent.any()
