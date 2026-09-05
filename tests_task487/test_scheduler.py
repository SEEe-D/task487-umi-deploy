import numpy as np
import pytest

from task487_runtime.scheduler import RollingScheduler, RunState, SchedulerConfig, UnsafeChunkError


def _target_sequence(count=20, step=0.001, start=0.0):
    targets = np.zeros((count, 14), dtype=np.float64)
    values = start + np.arange(1, count + 1) * step
    targets[:, 0] = values
    targets[:, 7] = values
    return targets


def _initial_chunk(scheduler, targets=None, observation_time=1.0, now=1.0):
    if targets is None:
        targets = _target_sequence()
    scheduler.activate()
    scheduler.mark_request_started()
    return scheduler.merge_chunk(targets, observation_time, now, np.zeros(14))


def _preload(scheduler, live=None):
    if live is None:
        live = np.zeros(14)
    actions = []
    while True:
        action = scheduler.pop_next(live, now=1.0)
        if action is None:
            break
        actions.append(action)
    return actions


def _complete(scheduler, count):
    assert count > 0
    live = scheduler._queue[count - 1].copy()
    now = float(scheduler._times[count - 1] + 1e-6)
    completed = scheduler.advance(now, live)
    assert completed == count
    return live, now


def _rtc_output(prefix, count=20, step=0.001):
    result = np.zeros((count, 14), dtype=np.float64)
    prefix_len = min(len(prefix), count)
    result[:prefix_len] = prefix[:prefix_len]
    anchor = 0.0 if not prefix_len else float(prefix[prefix_len - 1, 0])
    for index in range(prefix_len, count):
        value = anchor + (index - prefix_len + 1) * step
        result[index, 0] = value
        result[index, 7] = value
    return result


@pytest.mark.parametrize("step", [0.001, 0.01])
@pytest.mark.parametrize("inference_latency", [0.02, 1.0])
@pytest.mark.parametrize("tick_dt", [0.01, 0.08])
def test_complete_chunk_keeps_closing_tail_and_bounded_rtc_prefix(step, inference_latency, tick_dt):
    config = SchedulerConfig.for_policy_rate(12.5, 20, complete_chunk_before_replan=True)
    scheduler = RollingScheduler(config)
    targets = _target_sequence(step=step)
    targets[:, [6, 13]] = 15.0
    targets[-3:, [6, 13]] = 1.0
    live = targets[0].copy()
    live[[0, 7]] = 0.0
    scheduler.activate(live)
    assert scheduler.request_due(1.0)
    scheduler.mark_request_started(1.0)
    scheduler.merge_chunk(targets, 1.0, 1.0, live)
    dispatched = []
    for tick in range(2000):
        now = 1.0 + tick * tick_dt
        snapshot = scheduler.diagnostic_snapshot()
        due = snapshot.sent & (snapshot.target_times <= now)
        if due.any():
            live = snapshot.targets[np.flatnonzero(due)[-1]].copy()
        scheduler.advance(now, live)
        if scheduler.request_due(now):
            break
        batch = scheduler.pop_batch(live, now)
        dispatched.extend(batch)
        assert scheduler.committed_steps <= config.commit_steps
        after = scheduler.diagnostic_snapshot()
        # Existing progress exception: one long segment may exceed 0.4s,
        # but never enqueue the rest of the slow chunk with it.
        if scheduler.committed_steps > 1:
            assert after.target_times[after.sent].max() <= now + 0.4 + 1e-9
        if (~after.sent).any():
            assert not scheduler.request_due(now)
    else:
        pytest.fail("Replan never became due after dispatching the closing tail")
    np.testing.assert_allclose(np.stack([a.target for a in dispatched]), targets)
    prefix = scheduler.rtc_prefix_targets()
    assert 0 < len(prefix) <= 5
    np.testing.assert_allclose(prefix[-1, [6, 13]], [1.0, 1.0])
    scheduler.mark_request_started(now)
    assert not scheduler.request_due(now)
    assert scheduler.pop_batch(live, now) == []
    snapshot = scheduler.diagnostic_snapshot()
    arrival = now + inference_latency
    due = snapshot.sent & (snapshot.target_times <= arrival)
    if due.any():
        live = snapshot.targets[np.flatnonzero(due)[-1]].copy()
    scheduler.advance(arrival, live)
    new_chunk = np.repeat(prefix[-1:], 20, axis=0)
    new_chunk[:len(prefix)] = prefix
    stats = scheduler.merge_chunk(new_chunk, now, arrival, live)
    assert stats.rtc_prefix_preserved == len(prefix) - int(due.sum())
    assert not scheduler.request_due(arrival)
    assert scheduler.state is RunState.ACTIVE
    scheduler.hold("operator pause")
    assert scheduler.queued_steps == 0
    assert scheduler.pop_batch(live, arrival) == []
    assert not scheduler.request_due(arrival)


def test_complete_chunk_keeps_physical_tracking_guard():
    scheduler = RollingScheduler(SchedulerConfig.for_policy_rate(
        12.5, 20, complete_chunk_before_replan=True))
    _initial_chunk(scheduler)
    actual = np.zeros(14)
    actual[0] = 0.2
    with pytest.raises(UnsafeChunkError):
        scheduler.validate_physical_tracking(actual, np.zeros(14))
    assert scheduler.state is RunState.HOLD
    assert scheduler.queued_steps == 0


def test_initial_chunk_skips_expired_nominal_prefix_then_retimes_reachable():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    scheduler.activate()
    scheduler.mark_request_started()
    stats = scheduler.merge_chunk(
        _target_sequence(), observation_time=1.0, now=1.10, live_target=np.zeros(14)
    )

    assert stats.expired == 2
    assert stats.accepted == 18
    first = scheduler.pop_next(np.zeros(14))
    assert np.isclose(first.target[0], 0.003)
    assert np.isclose(first.nominal_time, 1.12)
    # 3 mm from live at 0.02 m/s needs 150 ms, so executable time is 1.250.
    assert np.isclose(first.target_time, 1.250)


def test_request_cadence_uses_physical_completion_not_preload_count():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    preloaded = _preload(scheduler)
    assert len(preloaded) == scheduler.config.commit_steps
    assert not scheduler.request_due()

    _complete(scheduler, 5)
    assert scheduler.request_due()


def test_12_5_hz_profile_preserves_wall_clock_commit_and_replan_windows():
    config = SchedulerConfig.for_policy_rate(12.5, action_horizon=20)
    assert config.control_hz == 12.5
    assert np.isclose(config.dt, 0.08)
    assert config.commit_steps == 5
    assert np.isclose(config.commit_window_s, 0.4)
    assert config.request_every_steps == 2
    assert config.replan_remaining_steps == 3
    assert config.handoff_blend_steps == 3
    assert config.max_queue_steps == 25
    assert config.max_translation_step_m == 0.045
    assert config.max_rotation_step_rad == 0.14

    scheduler = RollingScheduler(config)
    _initial_chunk(scheduler)
    preloaded = _preload(scheduler)
    assert len(preloaded) == 5
    assert np.isclose(preloaded[0].nominal_time, 1.08)
    assert np.isclose(preloaded[-1].nominal_time, 1.40)

    _complete(scheduler, 2)
    assert scheduler.request_due()
    assert len(scheduler.rtc_prefix_targets()) == 3


def test_12_5_hz_slow_path_caps_commit_but_retains_replaceable_horizon():
    config = SchedulerConfig.for_policy_rate(12.5, action_horizon=20)
    scheduler = RollingScheduler(config)
    stats = _initial_chunk(scheduler, _target_sequence(step=0.010))

    # At 20 mm/s each 10 mm segment takes 0.5 s. Keep the full replaceable
    # horizon for handoff continuity, but do not commit five points (2.5 s).
    assert stats.accepted == 20
    assert scheduler.queued_steps == 20
    batch = scheduler.pop_batch(np.zeros(14), now=1.0)
    assert len(batch) == 1
    assert np.isclose(batch[0].target_time, 1.55)

    # Replan once shortly before the physical prefix ends, rather than waiting
    # for two slow points to finish or repeatedly sampling the same prefix.
    assert not scheduler.request_due(now=1.30)
    assert scheduler.request_due(now=1.31)
    scheduler.mark_request_started(now=1.31)
    prefix = scheduler.rtc_prefix_targets()
    scheduler.merge_chunk(
        _rtc_output(prefix, step=0.010),
        observation_time=1.31,
        now=1.40,
        live_target=np.zeros(14),
    )
    assert not scheduler.request_due(now=1.40)


def test_12_5_hz_profile_accepts_measured_p999_step_but_rejects_larger_jump():
    config = SchedulerConfig.for_policy_rate(12.5, action_horizon=20)
    safe = RollingScheduler(config)
    _initial_chunk(safe, _target_sequence(count=3, step=0.039))
    assert safe.state is RunState.ACTIVE

    unsafe = RollingScheduler(config)
    with pytest.raises(UnsafeChunkError, match="translation step"):
        _initial_chunk(unsafe, _target_sequence(count=3, step=0.046))
    assert unsafe.state is RunState.HOLD


def test_pop_batch_atomically_fills_committed_window():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    targets = _target_sequence()
    _initial_chunk(scheduler, targets)

    actions = scheduler.pop_batch(np.zeros(14), now=1.0)

    assert len(actions) == scheduler.config.commit_steps == 10
    np.testing.assert_allclose(
        np.stack([action.target for action in actions]),
        targets[: scheduler.config.commit_steps],
    )
    np.testing.assert_allclose(
        [action.target_time for action in actions],
        scheduler._times[: scheduler.config.commit_steps],
    )
    assert scheduler.committed_steps == scheduler.config.commit_steps
    assert scheduler._sent[: scheduler.config.commit_steps].all()
    assert not scheduler._sent[scheduler.config.commit_steps :].any()


def test_batch_preload_preserves_lift_tail_after_near_stationary_grasp():
    """The first replan must retain a non-empty prefix and enqueue the lift.

    This reproduces the banana trace shape: five almost stationary closed-grip
    samples precede the useful upward suffix.  Single-point dispatch at 25 Hz
    let those first five points finish as fast as they were enqueued, so the
    next request could replace the lift before it reached the controller.
    """
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    targets = np.zeros((20, 14), dtype=np.float64)
    targets[:, [6, 13]] = 10.0
    lift = np.maximum(0, np.arange(20) - 4) * 0.004
    targets[:, 2] = lift
    targets[:, 9] = lift
    _initial_chunk(scheduler, targets)

    initial_batch = scheduler.pop_batch(np.zeros(14), now=1.0)
    assert len(initial_batch) == 10

    _complete(scheduler, 5)
    assert scheduler.request_due()
    request_prefix = scheduler.rtc_prefix_targets()
    assert len(request_prefix) == 5
    assert request_prefix[-1, 2] > request_prefix[0, 2]
    scheduler.mark_request_started()

    # Two committed points finish during inference. The remaining three are
    # immutable RTC context; the newly sampled suffix continues the lift.
    live, inference_done = _complete(scheduler, 2)
    result = np.zeros((20, 14), dtype=np.float64)
    result[:, [6, 13]] = 10.0
    result[: len(request_prefix)] = request_prefix
    for index in range(len(request_prefix), len(result)):
        rise = (index - len(request_prefix) + 1) * 0.004
        result[index] = request_prefix[-1]
        result[index, 2] += rise
        result[index, 9] += rise
    stats = scheduler.merge_chunk(
        result,
        observation_time=inference_done,
        now=inference_done,
        live_target=live,
    )

    assert stats.rtc_prefix_preserved == 3
    assert scheduler.committed_steps == 3
    refill = scheduler.pop_batch(live, now=inference_done)
    assert len(refill) == 7
    assert scheduler.committed_steps == scheduler.config.commit_steps
    assert max(action.target[9] for action in refill) > live[9]


def test_short_safe_chunk_replans_when_only_five_actions_remain():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler, _target_sequence(count=12))
    _preload(scheduler)

    _complete(scheduler, 7)

    assert scheduler.queued_steps == 5
    assert scheduler.request_due()


def test_rtc_prefix_excludes_replaceable_unsent_old_tail():
    # Exercise partial-commit behavior with no low-queue replan shortcut.
    scheduler = RollingScheduler(
        SchedulerConfig(
            dispatch_lead_s=0.0,
            request_every_steps=5,
            replan_remaining_steps=0,
            commit_steps=10,
        )
    )
    _initial_chunk(scheduler)
    _preload(scheduler)
    _complete(scheduler, 5)

    prefix = scheduler.rtc_prefix_targets()
    assert len(prefix) == 5
    np.testing.assert_allclose(prefix, scheduler._queue[:5])
    assert scheduler._sent[:5].all()
    assert not scheduler._sent[5:].any()


def test_advance_tracks_controller_setpoint_not_lagging_physical_pose():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler, _target_sequence(count=3, step=0.020))
    _preload(scheduler)

    # A physical arm may still be near the start while Mink releases its
    # anchor. RTC progress follows the interpolation setpoint, which has
    # consumed the first scheduled action at this timestamp.
    controller_target = scheduler._queue[0].copy()
    now = float(scheduler._times[0] + 1e-6)
    assert scheduler.advance(now, controller_target) == 1


def test_physical_tracking_watchdog_is_separate_from_rtc_progress():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    actual = np.zeros(14)
    controller_target = np.zeros(14)
    controller_target[0] = 0.051

    with pytest.raises(UnsafeChunkError, match="tracking error"):
        scheduler.validate_physical_tracking(actual, controller_target)
    assert scheduler.state is RunState.HOLD


def test_rtc_prefill_preserves_unconsumed_prefix_and_replaces_only_suffix():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    _preload(scheduler)
    _complete(scheduler, 5)

    prefix = scheduler.rtc_prefix_targets()
    prefix_times = scheduler._times[: len(prefix)].copy()
    scheduler.mark_request_started()
    assert scheduler.pop_next(prefix[0], now=1.0) is None

    # Two of the prefills complete while inference is running.
    _complete(scheduler, 2)
    result = _rtc_output(prefix)
    stats = scheduler.merge_chunk(
        result,
        observation_time=2.0,
        now=2.1,
        live_target=prefix[1],
    )

    assert stats.inference_delay_steps == 2
    assert stats.rtc_prefix_preserved == 3
    assert stats.handoff_blended == 5
    np.testing.assert_allclose(scheduler._queue[:3], prefix[2:])
    np.testing.assert_allclose(scheduler._times[:3], prefix_times[2:])
    # The first five generated actions ease through the replaceable old tail.
    np.testing.assert_allclose(scheduler._queue[8:], result[10:])
    assert scheduler.queued_steps == 18


def test_full_chunk_rtc_handoff_blends_from_last_prefix_target():
    scheduler = RollingScheduler(
        SchedulerConfig(
            dispatch_lead_s=0.0,
            request_every_steps=15,
            replan_remaining_steps=5,
            commit_steps=20,
        )
    )
    _initial_chunk(scheduler, _target_sequence(step=0.001))
    _preload(scheduler)
    _complete(scheduler, 15)
    prefix = scheduler.rtc_prefix_targets()
    scheduler.mark_request_started()

    result = _rtc_output(prefix, step=-0.008)
    raw_first_suffix = result[len(prefix)].copy()
    stats = scheduler.merge_chunk(
        result,
        observation_time=2.0,
        now=2.0,
        live_target=prefix[0],
    )

    assert stats.handoff_blended == 5
    anchor = prefix[-1]
    first_suffix = scheduler._queue[len(prefix)]
    expected_weight = 1.0 / 6.0
    expected_x = (1.0 - expected_weight) * anchor[0] + expected_weight * raw_first_suffix[0]
    assert np.isclose(first_suffix[0], expected_x)
    assert abs(first_suffix[0] - anchor[0]) < abs(raw_first_suffix[0] - anchor[0])


def test_rtc_merge_does_not_reject_valid_committed_prefix_when_arm_lags():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler, _target_sequence(step=0.010))
    _preload(scheduler)
    _complete(scheduler, 5)
    prefix = scheduler.rtc_prefix_targets()
    scheduler.mark_request_started()

    result = _rtc_output(prefix, step=0.010)
    lagging_actual = np.zeros(14)
    stats = scheduler.merge_chunk(
        result,
        observation_time=2.0,
        now=2.0,
        live_target=lagging_actual,
    )

    assert stats.rtc_prefix_preserved == len(prefix)
    assert stats.unsafe_truncated == 0
    assert scheduler.state is RunState.ACTIVE


def test_rtc_suffix_crossfades_from_safe_replaceable_old_tail():
    scheduler = RollingScheduler(
        SchedulerConfig(
            dispatch_lead_s=0.0,
            request_every_steps=5,
            replan_remaining_steps=0,
            commit_steps=10,
            handoff_blend_steps=5,
        )
    )
    _initial_chunk(scheduler, _target_sequence(step=0.001))
    _preload(scheduler)
    _complete(scheduler, 5)
    prefix = scheduler.rtc_prefix_targets()
    old_tail = scheduler._queue[len(prefix) :].copy()
    scheduler.mark_request_started()

    result = _rtc_output(prefix, step=-0.004)
    raw_first_suffix = result[len(prefix)].copy()
    stats = scheduler.merge_chunk(
        result,
        observation_time=2.0,
        now=2.0,
        live_target=prefix[0],
    )

    assert stats.handoff_blended == 5
    first_suffix = scheduler._queue[len(prefix)]
    expected_weight = 1.0 / 6.0
    expected_x = (
        (1.0 - expected_weight) * old_tail[0, 0]
        + expected_weight * raw_first_suffix[0]
    )
    assert np.isclose(first_suffix[0], expected_x)
    assert abs(first_suffix[0] - old_tail[0, 0]) < abs(
        raw_first_suffix[0] - old_tail[0, 0]
    )


def test_rtc_prefix_mismatch_fails_closed_instead_of_fake_blending():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    _preload(scheduler)
    _complete(scheduler, 5)
    prefix = scheduler.rtc_prefix_targets()
    scheduler.mark_request_started()

    result = _rtc_output(prefix)
    result[0, 0] += 0.003
    with pytest.raises(UnsafeChunkError, match="RTC prefix"):
        scheduler.merge_chunk(result, observation_time=2.0, now=2.0, live_target=prefix[0])
    assert scheduler.state is RunState.HOLD


def test_ten_mm_per_40ms_is_retimed_to_physical_speed_limit():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.05))
    stats = _initial_chunk(
        scheduler,
        _target_sequence(step=0.010),
        observation_time=1.02,
        now=1.0,
    )

    assert stats.speed_retimed == 20
    # Each 10 mm Cartesian segment requires at least 500 ms at 0.02 m/s.
    assert np.all(np.diff(scheduler._times) >= 0.50 - 1e-12)
    assert np.isclose(scheduler._queue[-1, 0], 0.2)


def test_rotation_uses_geodesic_distance_for_retiming():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    targets = _target_sequence(count=3, step=0.0)
    targets[:, 3] = [0.04, 0.08, 0.12]
    targets[:, 10] = [0.04, 0.08, 0.12]
    _initial_chunk(scheduler, targets)

    assert np.all(np.diff(scheduler._times) >= 0.50 - 1e-12)


def test_one_hundred_rtc_replans_keep_queue_bounded():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    max_seen = scheduler.queued_steps

    for iteration in range(100):
        _preload(scheduler, scheduler._queue[0])
        _complete(scheduler, 5)
        assert scheduler.request_due()
        prefix = scheduler.rtc_prefix_targets()
        scheduler.mark_request_started()
        _complete(scheduler, 2)
        result = _rtc_output(prefix, step=0.001)
        scheduler.merge_chunk(
            result,
            observation_time=10.0 + iteration,
            now=10.1 + iteration,
            live_target=prefix[1],
        )
        max_seen = max(max_seen, scheduler.queued_steps)

    assert max_seen <= 20
    assert scheduler.state is RunState.ACTIVE


def test_hold_clears_prefill_and_old_trajectory_cannot_revive():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler)
    _preload(scheduler)
    scheduler.hold("operator")

    assert scheduler.queued_steps == 0
    assert scheduler.rtc_prefix_targets().shape == (0, 14)
    assert scheduler.pop_next(np.zeros(14), now=2.0) is None
    scheduler.activate()
    assert scheduler.queued_steps == 0
    assert scheduler.request_due()


def test_unsafe_step_holds_fail_closed():
    scheduler = RollingScheduler()
    scheduler.activate()
    scheduler.mark_request_started()
    targets = _target_sequence()
    targets[4, 0] += 0.04
    with pytest.raises(UnsafeChunkError):
        scheduler.merge_chunk(targets, observation_time=1.0, now=1.0, live_target=np.zeros(14))
    assert scheduler.state is RunState.HOLD


def test_thirty_one_mm_seam_is_retimed_but_larger_outlier_is_rejected():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    targets = _target_sequence(count=3, step=0.001)
    targets[1:, 0] += 0.030
    targets[1:, 7] += 0.030

    stats = _initial_chunk(scheduler, targets)

    assert stats.accepted == 3
    assert stats.speed_retimed >= 2

    unsafe = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    unsafe_targets = _target_sequence(count=3, step=0.001)
    unsafe_targets[1:, 0] += 0.035
    unsafe_targets[1:, 7] += 0.035
    with pytest.raises(UnsafeChunkError, match="translation step"):
        _initial_chunk(unsafe, unsafe_targets)
    assert unsafe.state is RunState.HOLD


def test_unsafe_remote_tail_is_truncated_without_dropping_safe_prefix():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    scheduler.activate()
    scheduler.mark_request_started()
    targets = _target_sequence(count=15, step=0.001)
    targets[14, 3] = 0.323

    stats = scheduler.merge_chunk(targets, observation_time=1.0, now=1.0, live_target=np.zeros(14))

    assert stats.accepted == 14
    assert stats.unsafe_truncated == 1
    assert scheduler.state is RunState.ACTIVE


def test_excessive_live_tracking_error_holds_fail_closed():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    scheduler.activate()
    scheduler.mark_request_started()
    targets = _target_sequence(count=3, step=0.001)
    targets[:, 0] += 0.051

    with pytest.raises(UnsafeChunkError, match="tracking error"):
        scheduler.merge_chunk(targets, observation_time=1.0, now=1.0, live_target=np.zeros(14))
    assert scheduler.state is RunState.HOLD


def test_workspace_floor_rejects_below_table_chunk_and_holds():
    scheduler = RollingScheduler(
        SchedulerConfig(
            dispatch_lead_s=0.0,
            min_tcp_z_m=0.60,
            max_downward_excursion_m=0.12,
        )
    )
    live = np.zeros(14)
    live[[2, 9]] = [0.76, 0.79]
    scheduler.activate(live)
    np.testing.assert_allclose(scheduler.workspace_min_z, [0.64, 0.67])
    scheduler.mark_request_started()
    targets = np.repeat(live[None, :], 4, axis=0)
    targets[:, 2] = [0.72, 0.69, 0.66, 0.639]

    with pytest.raises(UnsafeChunkError, match="below workspace floor"):
        scheduler.merge_chunk(targets, observation_time=1.0, now=1.0, live_target=live)
    assert scheduler.state is RunState.HOLD


def test_workspace_floor_uses_absolute_minimum_for_low_activation_pose():
    scheduler = RollingScheduler(
        SchedulerConfig(min_tcp_z_m=0.60, max_downward_excursion_m=0.12)
    )
    live = np.zeros(14)
    live[[2, 9]] = [0.65, 0.68]
    scheduler.activate(live)
    np.testing.assert_allclose(scheduler.workspace_min_z, [0.60, 0.60])


def test_queue_exhaustion_after_physical_completion_enters_hold():
    scheduler = RollingScheduler(SchedulerConfig(dispatch_lead_s=0.0))
    _initial_chunk(scheduler, _target_sequence(count=1))
    assert scheduler.pop_next(np.zeros(14), now=1.0) is not None
    _complete(scheduler, 1)
    assert scheduler.pop_next(np.zeros(14), now=2.0) is None
    assert scheduler.state is RunState.HOLD
    assert scheduler.reason == "action queue exhausted"
