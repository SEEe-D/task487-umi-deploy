"""Slow playback must preserve action semantics and the real-time control loop."""
import numpy as np
import pytest

from task487_runtime.author_sync import AuthorSyncScheduler, GRIPPER_LATENCY
from task487_runtime.scheduler import RunState, SchedulerConfig, UnsafeChunkError


def make_scheduler(scale, rate=12.5, horizon=20):
    scheduler = AuthorSyncScheduler(SchedulerConfig.for_policy_rate(
        rate, horizon, max_physical_gripper_speed_deg_s=np.rad2deg(3 / 20)),
        action_horizon=horizon, max_pos_speed=.15, max_rot_speed=.35,
        speed_scale=scale)
    scheduler.activate(np.zeros(14))
    return scheduler


@pytest.mark.parametrize("scale", [0, -1, 1.01, np.nan, np.inf])
def test_invalid_playback_rate_is_rejected(scale):
    with pytest.raises(ValueError, match="speed scale"):
        make_scheduler(scale)


@pytest.mark.parametrize("rate,horizon", [(12.5, 20), (25., 16)])
def test_half_speed_keeps_stale_filter_cadence_and_short_physical_window(rate, horizon):
    raw = np.zeros((horizon, 14))
    raw[:, [0, 7]] = np.arange(1, horizon + 1)[:, None] * .0005
    raw[:, [5, 12]] = raw[:, [0, 7]] * 2
    raw[:, [6, 13]] = raw[:, [0, 7]] * 20
    original = raw.copy()
    batches = []
    for scale in (1., .5):
        s = make_scheduler(scale, rate, horizon)
        s.mark_request_started(10.)
        stats = s.merge_chunk(raw, 10., 10.05, np.zeros(14))
        assert stats.expired == int(np.ceil((.05 + GRIPPER_LATENCY) * rate))
        assert s.state is RunState.ACTIVE
        assert not s.request_due(10 + 5 / rate)
        assert s.request_due(10 + 6 / rate)
        batch = s.pop_batch(np.zeros(14), now=10.05)
        assert batch[-1].target_time - s.replace_from_time <= 7 / rate + .15 + 1e-9
        for action in batch:
            assert action.target_time - s.replace_from_time == pytest.approx(
                (action.nominal_time - s.replace_from_time) / scale)
        batches.append(batch)
    # The same early action values occur later. The short execution window
    # traverses less of the path; it must not scale the pose/gripper endpoints.
    np.testing.assert_allclose(batches[0][0].target, batches[1][0].target)
    assert batches[1][-1].target[0] < batches[0][-1].target[0]
    np.testing.assert_array_equal(raw, original)


def test_gripper_limited_segment_is_not_slowed_twice():
    anchor = np.zeros(14)
    targets = np.zeros((2, 14))
    targets[:, [0, 7]] = [[.01, .01], [.02, .02]]
    targets[:, [6, 13]] = [[10., 10.], [20., 20.]]
    nominal = np.array([.08, .16])
    full = make_scheduler(1)._limit_local_timing(targets, nominal, anchor)
    half = make_scheduler(.5)._limit_local_timing(targets, nominal, anchor)
    np.testing.assert_allclose(full, half)
    np.testing.assert_allclose(half, [10 / np.rad2deg(3 / 20), 20 / np.rad2deg(3 / 20)])


def test_half_speed_retains_local_outlier_and_physical_tracking_guards():
    s = make_scheduler(.5)
    targets = np.zeros((1, 14))
    targets[0, 0] = .04  # .267 s needed, above .002 s playback + .15 s correction.
    with pytest.raises(UnsafeChunkError, match="velocity"):
        s._limit_local_timing(targets, np.array([.001]), np.zeros(14))
    with pytest.raises(UnsafeChunkError, match="tracking"):
        s.validate_physical_tracking(np.zeros(14), np.r_[.051, np.zeros(13)])


def test_client_uses_slow_defaults_and_respects_explicit_overrides(monkeypatch):
    import task487_client as client

    monkeypatch.setattr(client.sys, "argv", ["task487_client.py"])
    args = client.parse_args()
    assert (args.sync_max_pos_speed, args.sync_max_rot_speed, args.sync_speed_scale) == (.15, .35, .5)
    monkeypatch.setattr(client.sys, "argv", ["task487_client.py", "--sync-max-pos-speed", ".1",
                                           "--sync-max-rot-speed", ".2", "--sync-speed-scale", ".4"])
    args = client.parse_args()
    assert (args.sync_max_pos_speed, args.sync_max_rot_speed, args.sync_speed_scale) == (.1, .2, .4)
