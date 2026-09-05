import numpy as np
import pytest

from task487_runtime.author_sync import AuthorSyncScheduler, GRIPPER_LATENCY
from task487_runtime.gripper_compensation import GripperIntent, compensate_grippers
from task487_runtime.scheduler import SchedulerConfig, UnsafeChunkError
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


def targets(right, left):
    result = np.zeros((len(right), 14))
    result[:, 6], result[:, 13] = right, left
    return result


def config():
    return SchedulerConfig.for_policy_rate(12.5, 20, max_physical_gripper_speed_deg_s=np.rad2deg(3 / 20))


def make_scheduler(**kwargs):
    return AuthorSyncScheduler(config(), action_horizon=20, speed_scale=.5,
        max_pos_speed=.15, max_rot_speed=.35, gripper_open_limits_deg=(34., 24.), **kwargs)


def adjust(raw, intents):
    return compensate_grippers(raw, intents, close_deg=5., open_deg=5., open_limits_deg=(34., 24.))


def test_independent_directions_neutral_start_and_limits():
    raw = targets([15., 14.8, 14., 12.8, 13.5, 30., 34.],
                  [10., 10.2, 11., 12., 11., 4., 1.])
    original = raw.copy()
    result, _ = adjust(raw, (GripperIntent(15.), GripperIntent(10.)))
    np.testing.assert_allclose(result[:, 6], [15., 14.8, 9., 7.8, 18.5, 34., 34.])
    np.testing.assert_allclose(result[:, 13], [10., 10.2, 16., 17., 6., 0., 0.])
    np.testing.assert_array_equal(result[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], 0.)
    np.testing.assert_array_equal(raw, original)
    result, _ = adjust(targets([40.], [40.]), (GripperIntent(30., 1), GripperIntent(20., 1)))
    np.testing.assert_allclose(result[:, [6, 13]], [[34., 24.]])


def test_plateaus_do_not_accumulate_and_tail_does_not_commit():
    intents = (GripperIntent(15.), GripperIntent(10.))
    _, unchanged = adjust(targets([15., 12.], [10., 13.]), intents)
    assert unchanged == intents
    raw = targets([12.8, 12.9, 12.6, 16.], [13., 12.8, 13.1, 10.])
    for _ in range(10):
        result, intents = adjust(raw, intents)
        np.testing.assert_allclose(result[:, 6], [7.8, 7.9, 7.6, 21.])
        np.testing.assert_allclose(result[:, 13], [18., 17.8, 18.1, 5.])
        assert tuple(i.direction for i in intents) == (-1, 1)


@pytest.mark.parametrize("name", ["gripper_close_compensation_deg", "gripper_open_compensation_deg"])
@pytest.mark.parametrize("value", [-1., 5.1, np.nan, np.inf])
def test_invalid_offsets_rejected(name, value):
    with pytest.raises(ValueError, match="compensation"):
        make_scheduler(**{name: value})


def test_configuration_guards_and_default_off():
    assert not make_scheduler().compensation_enabled
    with pytest.raises(ValueError, match="combined"):
        make_scheduler(right_gripper_preload_deg=2., gripper_close_compensation_deg=5.)
    with pytest.raises(ValueError, match="physical gripper speed"):
        AuthorSyncScheduler(SchedulerConfig.for_policy_rate(12.5, 20), action_horizon=20,
                            gripper_close_compensation_deg=5., gripper_open_limits_deg=(34., 24.))
    with pytest.raises(ValueError, match="opening limits"):
        AuthorSyncScheduler(config(), action_horizon=20, gripper_open_compensation_deg=5.)


def test_stale_tail_rejected_chunk_and_reactivation_do_not_leak_state(monkeypatch):
    s = make_scheduler(gripper_close_compensation_deg=5., gripper_open_compensation_deg=5.)
    current = targets([15.], [10.])[0]
    s.activate(current)
    s.merge_chunk(targets([1.] * 3 + [15.] * 17, [20.] * 3 + [10.] * 17), 10., 10.05, current)
    assert tuple(i.direction for i in s._gripper_intents) == (0, 0)
    np.testing.assert_allclose(s._queue[:, [6, 13]], np.tile([15., 10.], (len(s._queue), 1)))
    original = s._gripper_intents
    def reject(*args):
        raise UnsafeChunkError("timing rejected")
    monkeypatch.setattr(s, "_limit_local_timing", reject)
    with pytest.raises(UnsafeChunkError, match="timing rejected"):
        s.merge_chunk(targets([12.] * 20, [13.] * 20), 10.48, 10.53, current)
    assert s._gripper_intents == original
    s._gripper_intents = (GripperIntent(12., -1), GripperIntent(13., 1))
    s.activate(current)
    assert s._gripper_intents == original


def test_large_close_to_open_switch_uses_real_interpolators_without_extra_delay():
    s = make_scheduler(gripper_close_compensation_deg=5., gripper_open_compensation_deg=5.)
    current = targets([15.], [10.])[0]
    s.activate(current)
    speed = s.config.max_physical_gripper_speed_deg_s
    interps = [PoseTrajectoryInterpolator([10.], [[angle, 0, 0, 0, 0, 0]]) for angle in (15., 10.)]
    tails = [10., 10.]
    max_rates = []
    for n in range(10):
        now = 10.05 + n * .48
        current[[6, 13]] = [interp(now)[0] for interp in interps]
        s.observe_controller(current)
        s.advance(now, current)
        right, left = (12.8, 13.) if n < 4 else (16., 10.)
        raw = targets([right] * 20, [left] * 20)
        s.merge_chunk(raw, now - .05, now, current)
        batch = s.pop_batch(current, now=now)
        assert batch[-1].target_time - s.replace_from_time <= .71 + 1e-9
        previous, previous_time = s._trajectory_targets[0], s.replace_from_time
        for action in batch:
            assert s._minimum_duration(previous, action.target) <= action.target_time - previous_time + 1e-8
            previous, previous_time = action.target, action.target_time
        for hand, column in enumerate((6, 13)):
            cut = s.replace_from_time - GRIPPER_LATENCY
            grid = np.linspace(now + .01, cut, 20)
            old = interps[hand](grid)
            for i, action in enumerate(batch):
                due = action.target_time - GRIPPER_LATENCY
                interps[hand] = interps[hand].schedule_waypoint(
                    [action.target[column], 0, 0, 0, 0, 0], due,
                    max_pos_speed=speed, max_rot_speed=speed, curr_time=now + .01,
                    last_waypoint_time=tails[hand], replace_from_time=cut if i == 0 else None)
                tails[hand] = interps[hand].times[-1]
                assert tails[hand] == pytest.approx(due, abs=1e-8)
            np.testing.assert_allclose(interps[hand](grid), old, atol=1e-9)
            ts = np.arange(now + .01, tails[hand], .001)
            max_rates.append(np.max(np.abs(np.diff(interps[hand](ts)[:, 0]) / np.diff(ts))))
    assert max(max_rates) <= speed + 1e-7
    np.testing.assert_allclose([interps[0](tails[0])[0], interps[1](tails[1])[0]], [21., 5.])
    assert tuple(i.direction for i in s._gripper_intents) == (1, -1)


def test_cli_compensation_and_incompatible_modes(monkeypatch):
    import task487_client as client
    options = ["--sync-gripper-close-compensation-deg", "5", "--sync-gripper-open-compensation-deg", "5"]
    monkeypatch.setattr(client.sys, "argv", ["task487_client.py", "--execution-mode", "author-sync", *options])
    args = client.parse_args()
    assert args.sync_gripper_close_compensation_deg == args.sync_gripper_open_compensation_deg == 5
    for extra in ([], ["--execution-mode", "author-sync", "--sync-right-gripper-preload-deg", "2"]):
        monkeypatch.setattr(client.sys, "argv", ["task487_client.py", *options, *extra])
        with pytest.raises(SystemExit):
            client.parse_args()
