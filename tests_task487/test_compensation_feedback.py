"""Compensation coordinate routing, time alignment and measured handoff guards."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from task487_runtime.author_sync import AuthorSyncScheduler, GRIPPER_LATENCY
from task487_runtime.contract import build_policy_request
from task487_runtime.scheduler import SchedulerConfig, UnsafeChunkError
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.real_world.umi_env import UmiEnv
from umi.real_world.ros_gripper_controller import RosGripperController


def scheduler(gate=False):
    return AuthorSyncScheduler(SchedulerConfig.for_policy_rate(12.5, 20,
        max_physical_gripper_speed_deg_s=np.rad2deg(3 / 20)), action_horizon=20,
        max_pos_speed=.15, max_rot_speed=.35, speed_scale=.5,
        gripper_close_compensation_deg=5., gripper_open_compensation_deg=5.,
        gripper_open_limits_deg=(34., 24.), right_before_left=gate)


def current(right=10., left=1.):
    a = np.zeros(14)
    a[[6, 13]] = [right, left]
    return a


def test_model_coordinates_are_opt_in_and_raw_measurements_are_archived():
    obs = {"timestamp": np.array([10., 10.08])}
    for camera in (0, 2, 5):
        obs[f"camera{camera}_rgb"] = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    for hand in (0, 1):
        obs[f"robot{hand}_eef_pos"] = np.zeros((2, 3))
        obs[f"robot{hand}_eef_rot_axis_angle"] = np.zeros((2, 3))
        obs[f"robot{hand}_gripper_angle"] = np.array([[7.8], [8.]])
        obs[f"robot{hand}_gripper_policy_angle"] = np.array([[12.8], [13.]])
    raw = build_policy_request(obs, "sorting")
    mapped = build_policy_request(obs, "sorting", gripper_policy_coordinates=True)
    np.testing.assert_allclose(np.rad2deg(raw.observation["state"][[9, 19]]), [8., 8.], atol=1e-6)
    np.testing.assert_allclose(np.rad2deg(mapped.observation["pre_state"][[9, 19]]), [12.8, 12.8], atol=1e-6)
    np.testing.assert_allclose(np.rad2deg(mapped.observation["state"][[9, 19]]), [13., 13.], atol=1e-6)
    np.testing.assert_array_equal(mapped.diagnostics["measured_gripper_degrees"], [[7.8, 7.8], [8., 8.]])
    np.testing.assert_array_equal(raw.observation["cam_right_top"], mapped.observation["cam_right_top"])
    del obs["robot1_gripper_policy_angle"]
    with pytest.raises(ValueError, match="coordinate feedback"):
        build_policy_request(obs, "sorting", gripper_policy_coordinates=True)


def test_reference_routing_matches_filtering_and_hardware_order():
    calls = [[], [], [], []]
    devices = [SimpleNamespace(schedule_waypoint=lambda _i=i, **kw: calls[_i].append(kw)) for i in range(4)]
    env = SimpleNamespace(is_ready=True, robots=devices[:2], grippers=devices[2:],
                         robot_action_latency=.11, gripper_action_latency=.176, action_accumulator=None)
    now = time.time()
    actions = np.tile(current(8., 18.), (3, 1))
    refs = np.array([[0., 0.], [13., 13.], [14., 12.]])
    UmiEnv.exec_actions(env, actions, [now - 1, now + 1, now + 2], compensate_latency=True,
                        gripper_policy_angles=refs)
    assert [c['policy_pos'] for c in calls[2]] == [13., 14.]
    assert [c['policy_pos'] for c in calls[3]] == [13., 12.]
    queue = []
    gripper = SimpleNamespace(input_queue=SimpleNamespace(put=queue.append))
    RosGripperController.schedule_waypoint(gripper, 8., now + 1, policy_pos=13.)
    RosGripperController.schedule_waypoint(gripper, 1., now + 2)
    assert queue[0]['policy_target_pos'] == 13.
    assert queue[1]['policy_target_pos'] == 1.
    with pytest.raises(ValueError, match="non-finite"):
        RosGripperController.schedule_waypoint(gripper, 8., now + 1, policy_pos=np.nan)


def test_paired_reference_survives_retiming_splicing_clipping_and_pause():
    s = scheduler()
    live = current(15., 10.)
    s.activate(live)
    phys = [PoseTrajectoryInterpolator([10.], [[q, 0, 0, 0, 0, 0]]) for q in (15., 10.)]
    logical = [PoseTrajectoryInterpolator([10.], [[q, 0, 0, 0, 0, 0]]) for q in (15., 10.)]
    tails = [10., 10.]
    for n in range(12):
        now = 10.05 + n * .48
        live[[6, 13]] = [p(now)[0] for p in phys]
        s.observe_controller(live)
        s.advance(now, live)
        requested = current(12.8, 13.) if n < 5 else current(33., 20.)
        s.merge_chunk(np.tile(requested, (20, 1)), now - .05, now, live)
        batch = s.pop_batch(live, now=now)
        for hand, column in enumerate((6, 13)):
            cut = s.replace_from_time - GRIPPER_LATENCY
            ts = np.linspace(now + .01, cut, 15)
            old_logical = logical[hand](ts)
            for i, a in enumerate(batch):
                due = a.target_time - GRIPPER_LATENCY
                params = dict(curr_time=now + .01, last_waypoint_time=tails[hand],
                              replace_from_time=cut if i == 0 else None)
                phys[hand] = phys[hand].schedule_waypoint([a.target[column], 0, 0, 0, 0, 0], due,
                    max_pos_speed=s.config.max_physical_gripper_speed_deg_s, **params)
                logical[hand] = logical[hand].schedule_waypoint([a.gripper_policy_target[hand], 0, 0, 0, 0, 0],
                    phys[hand].times[-1], **params)
                tails[hand] = phys[hand].times[-1]
                assert tails[hand] == pytest.approx(due, abs=1e-8)
            np.testing.assert_allclose(logical[hand](ts), old_logical, atol=1e-9)
            ts = np.linspace(cut, tails[hand], 100)
            offset = phys[hand](ts)[:, 0] - logical[hand](ts)[:, 0]
            assert np.max(np.abs(offset)) <= 5. + 1e-8
    # Endpoints clip the physical target, not the policy reference.
    np.testing.assert_allclose([p(t)[0] for p, t in zip(phys, tails)], [34., 24.])
    np.testing.assert_allclose([p(t)[0] for p, t in zip(logical, tails)], [33., 20.])
    s.hold("pause")
    s.reset_gripper_episode()
    s.activate(current(7.8, 18.), gripper_policy_live=[12.8, 13.])
    assert tuple(i.direction for i in s._gripper_intents) == (-1, 1)
    s.merge_chunk(np.tile(current(12.8, 13.), (20, 1)), 30., 30.05, current(7.8, 18.))
    np.testing.assert_allclose(s._queue[:, [6, 13]], np.tile([7.8, 18.], (len(s._queue), 1)))


def test_left_waits_for_measured_right_opening_not_predicted_opening_or_time_alone():
    s = scheduler(gate=True)
    live = current()
    s.activate(live)
    for n, right in enumerate((12., 11., 14.)):
        now = 10.05 + n * 2.
        if n:
            live = s._planned_target(now)
        s.observe_controller(live)
        s.advance(now, live)
        raw = np.tile(current(right, 14.), (20, 1))
        raw[:, 7] = .2  # An unsafe left movement is replaced by a stationary hold.
        s.merge_chunk(raw, now - .05, now, live)
        batch = s.pop_batch(live, now=now)
        for action in batch:
            np.testing.assert_allclose(action.target[7:], current()[7:])
    assert s._right_release_goal == 19.
    assert not s._left_released
    s.advance(15., current(18.4))
    s.advance(16., current(18.4))
    assert not s._left_released
    s.advance(16.1, current(19.))
    s.advance(16.2, current(18.))  # Below goal resets the dwell.
    s.advance(16.3, current(19.))
    assert not s._left_released
    s.advance(16.47, current(19.))
    assert s._left_released
    s.observe_controller(current(19.))
    s.merge_chunk(np.tile(current(14., 14.), (20, 1)), 16.5, 16.55, current(19.))
    assert np.max(s._queue[:, 13]) > 1.


def test_gating_does_not_weaken_active_arm_guards_or_accept_a_stale_release():
    s = scheduler(gate=True)
    live = current()
    s.activate(live)
    raw = np.tile(live, (20, 1))
    raw[:3, 6] = 20.
    s.merge_chunk(raw, 10., 10.05, live)
    assert not s._right_opened and s._right_release_goal is None
    old_reference = s._trajectory_policy_grippers.copy()
    raw[:, 0] = .2
    with pytest.raises(UnsafeChunkError, match="right tracking error"):
        s.merge_chunk(raw, 11., 11.05, live)
    np.testing.assert_array_equal(s._trajectory_policy_grippers, old_reference)
    assert not s._left_released
