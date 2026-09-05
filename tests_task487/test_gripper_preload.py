import numpy as np
import pytest

from task487_runtime.author_sync import AuthorSyncScheduler, GRIPPER_LATENCY
from task487_runtime.gripper_preload import ClosureIntent, preload_right_gripper
from task487_runtime.scheduler import SchedulerConfig, UnsafeChunkError
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


def targets(angles):
    result = np.zeros((len(angles), 14))
    result[:, 6] = angles
    result[:, 13] = 3.
    return result


def make_scheduler(preload=2.):
    s = AuthorSyncScheduler(SchedulerConfig.for_policy_rate(
        12.5, 20, max_physical_gripper_speed_deg_s=np.rad2deg(3 / 20)),
        action_horizon=20, max_pos_speed=.15, max_rot_speed=.35,
        speed_scale=.5, right_gripper_preload_deg=preload)
    s.activate(targets([15.6])[0])
    return s


def test_closing_adds_bounded_preload_and_opening_withdraws_it():
    raw = targets([15.6, 15.4, 14.8, 13., 12.7, 12.8, 14.5, 16.])
    original = raw.copy()
    adjusted, _ = preload_right_gripper(raw, ClosureIntent(15.6), 2.)
    np.testing.assert_allclose(adjusted[:, 6], [15.6, 15.4, 12.8, 11., 10.7, 10.8, 14.5, 16.])
    assert np.all((raw[:, 6] - adjusted[:, 6] >= 0) & (raw[:, 6] - adjusted[:, 6] <= 2. + 1e-9))
    np.testing.assert_array_equal(adjusted[:, np.arange(14) != 6], original[:, np.arange(14) != 6])
    np.testing.assert_array_equal(raw, original)


def test_plateau_holds_preload_across_chunks_without_using_measured_compression():
    _, intent = preload_right_gripper(targets([13., 12.7]), ClosureIntent(15.6), 2.)
    held, intent = preload_right_gripper(targets([12.7, 12.8, 12.6]), intent, 2.)
    np.testing.assert_allclose(held[:, 6], [10.7, 10.8, 10.6])
    opened, _ = preload_right_gripper(targets([14., 15.]), intent, 2.)
    np.testing.assert_allclose(opened[:, 6], [14., 15.])


def test_speculative_tail_does_not_commit_open_or_closed_state():
    _, intent = preload_right_gripper(targets([15., 12.]), ClosureIntent(15.), 2.)
    assert not intent.closing
    _, intent = preload_right_gripper(targets([12., 17.]), ClosureIntent(15.), 2.)
    assert intent.closing
    held, _ = preload_right_gripper(targets([12.]), intent, 2.)
    assert held[0, 6] == 10.


def test_preload_cannot_push_past_closed_stop_and_small_noise_does_not_toggle():
    result, _ = preload_right_gripper(targets([1., .1, .2, .3, .1]), ClosureIntent(3.), 2.)
    np.testing.assert_array_equal(result[:, 6], 0.)
    raw = targets([15., 14.9, 15.1, 14.8, 15.])
    result, _ = preload_right_gripper(raw, ClosureIntent(15.), 2.)
    np.testing.assert_array_equal(result, raw)


@pytest.mark.parametrize("preload", [-1., 2.1, np.nan, np.inf])
def test_preload_bounds(preload):
    with pytest.raises(ValueError, match="preload"):
        make_scheduler(preload)


def test_preload_requires_gripper_timing_and_is_off_by_default():
    config = SchedulerConfig.for_policy_rate(12.5, 20)
    with pytest.raises(ValueError, match="physical gripper speed"):
        AuthorSyncScheduler(config, action_horizon=20, right_gripper_preload_deg=2.)
    assert AuthorSyncScheduler(config, action_horizon=20).right_gripper_preload_deg == 0


def test_stale_closure_cannot_enable_preload_and_activation_resets_intent():
    s = make_scheduler()
    current = targets([15.6])[0]
    raw = targets([12.] * 3 + [15.6] * 17)
    s.merge_chunk(raw, 10., 10.05, current)
    assert not s._right_closure_intent.closing
    assert np.allclose(s._queue[:, 6], 15.6)
    s._right_closure_intent = ClosureIntent(12., True)
    s.hold("operator")
    s.activate(current)
    assert s._right_closure_intent == ClosureIntent(15.6)


def test_rejected_chunk_cannot_commit_preload_intent(monkeypatch):
    s = make_scheduler()
    original = s._right_closure_intent
    def reject(*args):
        raise UnsafeChunkError("synthetic timing rejection")
    monkeypatch.setattr(s, "_limit_local_timing", reject)
    with pytest.raises(UnsafeChunkError, match="synthetic"):
        s.merge_chunk(targets([13.] * 20), 10., 10.05, targets([15.6])[0])
    assert s._right_closure_intent == original


def test_close_and_reopen_keep_short_shared_clock_and_real_interpolator_limits():
    s = make_scheduler()
    speed = s.config.max_physical_gripper_speed_deg_s
    interp = PoseTrajectoryInterpolator([10.], [[15.6, 0, 0, 0, 0, 0]])
    tail = 10.
    for now, requested in ((10.05, 13.), (10.53, 13.), (11.01, 16.)):
        current = targets([interp(now)[0]])[0]
        s.observe_controller(current)
        s.advance(now, current)
        raw = targets([requested] * 20)
        original = raw.copy()
        s.merge_chunk(raw, now-.05, now, current)
        batch = s.pop_batch(current, now=now)
        assert batch[-1].target_time-s.replace_from_time <= .71+1e-9
        cut = s.replace_from_time-GRIPPER_LATENCY
        grid = np.linspace(now+.01, cut, 20)
        prefix = interp(grid)
        previous = s._trajectory_targets[0]
        last = s.replace_from_time
        for i, action in enumerate(batch):
            assert s._minimum_duration(previous, action.target) <= action.target_time-last+1e-8
            due = action.target_time-GRIPPER_LATENCY
            interp = interp.schedule_waypoint(
                [action.target[6], 0, 0, 0, 0, 0], due, max_pos_speed=speed,
                max_rot_speed=speed, curr_time=now+.01, last_waypoint_time=tail,
                replace_from_time=cut if i == 0 else None)
            tail = interp.times[-1]
            assert tail == pytest.approx(due, abs=1e-8)
            previous, last = action.target, action.target_time
        np.testing.assert_allclose(interp(grid), prefix, atol=1e-9)
        np.testing.assert_array_equal(raw, original)
    assert not s._right_closure_intent.closing
    assert batch[-1].target[6] == pytest.approx(16.)


def test_cli_opt_in_and_mode_check(monkeypatch):
    import task487_client as client
    monkeypatch.setattr(client.sys, "argv", ["task487_client.py"])
    assert client.parse_args().sync_right_gripper_preload_deg == 0
    monkeypatch.setattr(client.sys, "argv", ["task487_client.py", "--sync-right-gripper-preload-deg", "2"])
    with pytest.raises(SystemExit):
        client.parse_args()
    monkeypatch.setattr(client.sys, "argv", ["task487_client.py", "--execution-mode", "author-sync",
                                           "--sync-right-gripper-preload-deg", "2"])
    assert client.parse_args().sync_right_gripper_preload_deg == 2
