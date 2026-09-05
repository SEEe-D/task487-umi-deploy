import time
from types import SimpleNamespace

import numpy as np
import pytest

from task487_runtime.author_sync import (
    AUTHOR_MAX_POS_SPEED, AuthorSyncScheduler, GRIPPER_LATENCY, ROBOT_LATENCY,
    stitch_targets,
)
from task487_runtime.contract import (
    POLICY_RUNTIME_CONTRACTS, PolicyRequest, validate_policy_metadata,
)
from task487_runtime.scheduler import RunState, SchedulerConfig, UnsafeChunkError
from task487_runtime.worker import InferenceWorker


def scheduler(rate=12.5, horizon=20, **kwargs):
    result = AuthorSyncScheduler(SchedulerConfig.for_policy_rate(rate, horizon),
                                 action_horizon=horizon, **kwargs)
    result.activate(np.zeros(14))
    return result


def chunk(horizon=20):
    result = np.zeros((horizon, 14))
    result[:, 0] = np.arange(1, horizon+1) * .001
    result[:, 7] = -result[:, 0]
    result[:, 6] = np.linspace(0, 25, horizon)
    result[:, 13] = np.linspace(20, 0, horizon)
    return result


@pytest.mark.parametrize("rate,horizon", [(12.5, 20), (25., 16)])
def test_plain_cadence_and_dispatch_without_retiming_or_rtc(rate, horizon):
    s = scheduler(rate, horizon)
    s._retime = lambda *a, **k: pytest.fail("rolling retiming must not run")
    s._blend_chunk_handoff = lambda *a, **k: pytest.fail("old tail blending must not run")
    s.mark_request_started(10.)
    original = chunk(horizon)
    stats = s.merge_chunk(original, 10., 10.05, np.zeros(14))
    assert stats.expired == int(np.ceil((.05 + GRIPPER_LATENCY) * rate))
    assert stats.speed_retimed == stats.handoff_blended == stats.rtc_prefix_preserved == 0
    actions = s.pop_batch(np.zeros(14), now=10.05)
    assert len(actions) == stats.accepted
    assert all(a.target_time - GRIPPER_LATENCY > 10.05 for a in actions)
    assert all(a.target_time - ROBOT_LATENCY > 10.05 for a in actions)
    assert not s.pop_batch(np.zeros(14), now=10.06)
    assert len(s.rtc_prefix_targets()) == 0  # even with every waypoint dispatched
    assert not s.request_due(10. + 5/rate)
    assert s.request_due(10. + 6/rate + 1e-9)
    np.testing.assert_array_equal(original, chunk(horizon))


def test_gripper_knots_are_not_multiplied_or_skipped_by_cartesian_connection():
    raw = chunk()
    raw[:, 6] = [10, 0, 20, 5] + [15]*16
    raw[:, 13] = [0, 20, 0, 10] + [5]*16
    result, times = stitch_targets(raw, np.zeros(14), np.zeros(14), .08)
    for index in range(len(raw)):
        i = np.flatnonzero(np.isclose(times, (index+1)*.08))
        assert len(i) == 1
        np.testing.assert_allclose(result[i[0], [6, 13]], raw[index, [6, 13]], atol=1e-10)
    assert np.isfinite(result).all() and np.all(np.diff(times) > 0)
    np.testing.assert_allclose(result[-1], raw[-1], atol=1e-10)


def test_next_chunk_replaces_old_future_without_replaying_old_prefix():
    s = scheduler()
    s.mark_request_started(10)
    s.merge_chunk(chunk(), 10, 10.05, np.zeros(14))
    first = s.pop_batch(np.zeros(14), now=10.05)
    s.mark_request_started(10.49)
    next_chunk = chunk()
    next_chunk[:, 0] *= -1
    s.merge_chunk(next_chunk, 10.49, 10.54, np.zeros(14))
    second = s.pop_batch(np.zeros(14), now=10.54)
    assert first[0].target[0] > 0 and second[-1].target[0] < 0
    # The new connection starts on the moving old path; it must not jump
    # straight across zero just because the policy changed direction.
    assert second[0].target[0] > 0


@pytest.mark.parametrize("failure", ["expired", "nan", "shape", "jump", "future"])
def test_bad_chunk_holds_and_clears_timeline(failure):
    s = scheduler()
    s.mark_request_started(10)
    raw, obs, now = chunk(), 10., 10.05
    if failure == "expired":
        now = 12.
    elif failure == "nan":
        raw[-1, 3] = np.nan
    elif failure == "shape":
        raw = raw[:, :7]
    elif failure == "jump":
        raw[:, 0] += .2
    else:
        obs = 11.
    with pytest.raises(UnsafeChunkError):
        s.merge_chunk(raw, obs, now, np.zeros(14))
    assert s.state is RunState.HOLD and s.queued_steps == 0


def test_large_velocity_correction_exceeds_budget_and_holds():
    s = scheduler(max_pos_speed=.001)
    s.mark_request_started(10)
    with pytest.raises(UnsafeChunkError, match="velocity"):
        s.merge_chunk(chunk(), 10, 10.05, np.zeros(14))
    assert s.state is RunState.HOLD


def test_pending_timeout_and_empty_queue_hold_even_during_inference():
    s = scheduler()
    s.mark_request_started(10)
    with pytest.raises(UnsafeChunkError, match="timeout"):
        s.advance(12.01, np.zeros(14))
    s.activate(np.zeros(14))
    s.mark_request_started(20)
    s.merge_chunk(chunk(), 20, 20.05, np.zeros(14))
    actions = s.pop_batch(np.zeros(14), now=20.05)
    s.mark_request_started(20.5)
    s.advance(actions[-1].target_time + .01, np.zeros(14))
    assert not s.pop_batch(np.zeros(14), now=actions[-1].target_time + .01)
    assert s.state is RunState.HOLD


def test_batch_expiration_and_five_waypoint_cap():
    s = scheduler()
    s.mark_request_started(10)
    s.merge_chunk(chunk(), 10, 10.05, np.zeros(14))
    actions = s.pop_batch(np.zeros(14), now=10.05, max_actions=5)
    assert len(actions) == s.committed_steps == 5
    assert s.advance(actions[0].target_time, np.zeros(14)) == 1
    s.hold("operator")
    assert not s.pop_batch(np.zeros(14), now=11)
    s.activate(np.zeros(14))
    assert s.request_due(11)
    s.mark_request_started(11)
    s.merge_chunk(chunk(), 11, 11.05, np.zeros(14))
    with pytest.raises(UnsafeChunkError, match="expired before dispatch"):
        s.pop_batch(np.zeros(14), now=15)


def test_physical_following_guard_is_retained():
    s = scheduler()
    bad = np.zeros(14)
    bad[0] = .051
    with pytest.raises(UnsafeChunkError, match="tracking"):
        s.validate_physical_tracking(np.zeros(14), bad)
    assert s.state is RunState.HOLD


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf, AUTHOR_MAX_POS_SPEED+.01])
def test_speed_settings_cannot_remove_cap(speed):
    with pytest.raises(ValueError):
        scheduler(max_pos_speed=speed)


def test_plain_metadata_does_not_require_rtc_but_keeps_camera_and_timing_checks():
    runtime = POLICY_RUNTIME_CONTRACTS["pi05_umi_task487_wrist_only_4w_12_5_v1"]
    metadata = {k: v for k, v in runtime.expected_metadata().items() if not k.startswith("rtc_")}
    assert validate_policy_metadata(metadata, require_rtc=False) is runtime
    with pytest.raises(RuntimeError):
        validate_policy_metadata(metadata)
    metadata["control_hz"] = 2
    with pytest.raises(RuntimeError):
        validate_policy_metadata(metadata, require_rtc=False)


def test_worker_stays_busy_until_old_result_is_drained():
    class Policy:
        def infer(self, obs):
            assert not any(k.startswith("action_prefill") or k == "_rtc" for k in obs)
            result = np.zeros((20, 20))
            result[:, 3:9] = result[:, 13:19] = [1, 0, 0, 0, 1, 0]
            return {"actions": result}
    worker = InferenceWorker(Policy())
    try:
        req = PolicyRequest({"prompt": "Vegetable and Fruit Sorting"}, 10, np.zeros((2, 6)), 1)
        worker.submit(req)
        with pytest.raises(RuntimeError, match="in flight"):
            worker.submit(req)
        deadline, result = time.monotonic()+2, None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(.001)
        assert result is not None and result.error is None
        assert result.request.round_id == 1 and not worker.busy
    finally:
        worker.close()


def test_two_chunks_replay_through_real_pose_interpolator_without_time_stretch():
    from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
    s = scheduler()
    interpolator = PoseTrajectoryInterpolator([10.0], [np.zeros(6)])
    last_tail = 10.0
    for obs_time, merge_time in ((10., 10.05), (10.5, 10.55)):
        current = np.zeros(14)
        current[:6] = interpolator(merge_time)
        s.observe_controller(current)
        previous = current.copy()
        previous[:6] = interpolator(merge_time-.01)
        s._previous_controller = previous
        s.mark_request_started(obs_time)
        raw = chunk()
        raw[:, :6] += current[:6]
        s.merge_chunk(raw, obs_time, merge_time, current)
        batch = s.pop_batch(current, now=merge_time)
        for i, action in enumerate(batch):
            command_time = action.target_time - ROBOT_LATENCY
            interpolator = interpolator.schedule_waypoint(
                action.target[:6], command_time,
                max_pos_speed=s.config.max_physical_translation_speed_m_s,
                max_rot_speed=s.config.max_physical_rotation_speed_rad_s,
                curr_time=merge_time+.01, last_waypoint_time=last_tail,
                replace_from_time=s.replace_from_time-ROBOT_LATENCY if i == 0 else None)
            last_tail = float(interpolator.times[-1])
            assert last_tail == pytest.approx(command_time, abs=1e-9)
        np.testing.assert_allclose(interpolator(last_tail), batch[-1].target[:6], atol=1e-9)


def test_small_overspeed_is_locally_corrected_without_delaying_requests():
    s = scheduler(max_pos_speed=.5, max_rot_speed=.4)
    s.mark_request_started(10.)
    raw = chunk()
    # A 0.409 rad/s original tail, just above the selected 0.4 cap.
    raw[:, 5] = np.maximum(0, np.arange(20)-5) * .409 * .08
    stats = s.merge_chunk(raw, 10., 10.17, np.zeros(14))
    assert 0 < stats.max_retime_delay_s < .15
    assert stats.speed_retimed > 0 and s.state is RunState.ACTIVE
    assert s.request_due(10.48)
    prev, t = s._trajectory_targets[0], s.replace_from_time
    for action in s.pop_batch(np.zeros(14), now=10.17):
        assert s._minimum_duration(prev, action.target) <= action.target_time-t+1e-7
        prev, t = action.target, action.target_time


def test_fixed_phase_cadence_does_not_accumulate_late_control_ticks():
    s = scheduler()
    s.mark_request_started(10.)
    for now, expected in [(10.49, 10.96), (11.01, 11.44), (11.44-1e-8, 11.92)]:
        s._request_pending = False
        s.mark_request_started(now)
        assert s._next_request_at == pytest.approx(expected)


def test_future_splice_preserves_moving_prefix_through_real_interpolator():
    from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
    s = scheduler(max_pos_speed=.5, max_rot_speed=.5)
    # Both arms already moving at 0.05 m/s. The new command must not pull
    # them backwards toward a stale NOW anchor during the transport lead.
    times = np.array([10., 11.5])
    targets = np.zeros((2, 14))
    targets[1, [0, 7]] = .075
    s._trajectory_times, s._trajectory_targets = times, targets
    old = PoseTrajectoryInterpolator(times-ROBOT_LATENCY, targets[:, :6])
    now = 10.5
    current = np.zeros(14)
    current[:6] = current[7:13] = old(now)
    s.observe_controller(current)
    s.mark_request_started(now-.17)
    raw = np.zeros((20, 14))
    raw[:, 0] = raw[:, 7] = .02 + np.arange(20)*.004
    s.merge_chunk(raw, now-.17, now, current)
    cut = s.replace_from_time-ROBOT_LATENCY
    np.testing.assert_allclose(s._trajectory_targets[0, :6], old(cut), atol=1e-10)
    interpolation, tail = old, old.times[-1]
    before = old(np.linspace(now+.01, cut, 20))
    for i, action in enumerate(s.pop_batch(current, now=now)):
        interpolation = interpolation.schedule_waypoint(
            action.target[:6], action.target_time-ROBOT_LATENCY,
            max_pos_speed=.5, max_rot_speed=.5, curr_time=now+.01,
            last_waypoint_time=tail, replace_from_time=cut if i == 0 else None)
        tail = interpolation.times[-1]
    np.testing.assert_allclose(interpolation(np.linspace(now+.01, cut, 20)), before, atol=1e-10)
    t = np.linspace(now+.01, cut+.1, 100)
    assert np.all(np.diff(interpolation(t)[:, 0]) > 0)  # no reset/brake/reversal


def test_reactivation_discards_old_future_anchor():
    s = scheduler()
    s.mark_request_started(10)
    s.merge_chunk(chunk(), 10, 10.05, np.zeros(14))
    s.hold("operator")
    new = np.full(14, .01)
    s.activate(new)
    assert s._trajectory_times is None and s.replace_from_time is None
    np.testing.assert_array_equal(s._planned_target(20), new)


def test_replacement_boundary_rejects_nonfinite_or_backwards_timeline():
    from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
    p = PoseTrajectoryInterpolator([10.], [np.zeros(6)])
    for boundary in [np.nan, np.inf, 11.]:
        with pytest.raises(ValueError):
            p.schedule_waypoint(np.zeros(6), 11., curr_time=10., replace_from_time=boundary)
