"""Real UmiEnv routing and controller queue API, without constructing hardware."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from umi.real_world.umi_env import UmiEnv
from umi.real_world.ros_target_interpolation_controller import RosTargetInterpolationController
from umi.real_world.ros_gripper_controller import RosGripperController
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


@pytest.mark.parametrize("replace", [False, True])
def test_umi_routes_splice_boundary_once_with_independent_device_latencies(replace):
    calls = [[], [], [], []]
    devices = [SimpleNamespace(schedule_waypoint=lambda _i=i, **kw: calls[_i].append(kw)) for i in range(4)]
    env = SimpleNamespace(is_ready=True, robots=devices[:2], grippers=devices[2:],
                          robot_action_latency=.11, gripper_action_latency=.176,
                          action_accumulator=None)
    cut = time.time()+1
    times = cut+np.arange(1, 4)*.08
    actions = np.zeros((3, 14))
    actions[:, [2, 9]] = .3
    actions[:, 6], actions[:, 13] = [1, 4, 2], [2, 3, 1]
    UmiEnv.exec_actions(env, actions, times, compensate_latency=True,
                        replace_from_time=cut if replace else None)
    for i in range(4):
        latency = .11 if i < 2 else .176
        assert len(calls[i]) == 3
        for n, call in enumerate(calls[i]):
            assert call['target_time'] == pytest.approx(times[n]-latency)
            if replace and n == 0:
                assert call['replace_from_time'] == pytest.approx(cut-latency)
            else:
                assert 'replace_from_time' not in call
    assert [x['pos'] for x in calls[2]] == [1, 4, 2]
    assert [x['pos'] for x in calls[3]] == [2, 3, 1]


@pytest.mark.parametrize("kind", ["arm", "gripper"])
def test_controller_api_clears_optional_boundary_in_every_waypoint(kind):
    queue = []
    controller = SimpleNamespace(input_queue=SimpleNamespace(put=queue.append))
    method = (RosTargetInterpolationController.schedule_waypoint if kind == 'arm'
              else RosGripperController.schedule_waypoint)
    target = np.zeros(6) if kind == 'arm' else 5.
    method(controller, target, 12., replace_from_time=11.)
    method(controller, target, 13.)
    assert queue[0]['replace_from_time'] == 11.
    # SharedMemoryQueue.put does not fill omitted keys. Explicitly write zero
    # even when a circular-buffer slot formerly contained a replacement.
    assert queue[1]['replace_from_time'] == 0.


def test_expired_splice_cannot_partially_enqueue_one_arm():
    def unexpected(**kwargs):
        pytest.fail('invalid boundary must be rejected before any device enqueue')
    device = SimpleNamespace(schedule_waypoint=unexpected)
    env = SimpleNamespace(is_ready=True, robots=[device, device], grippers=[device, device],
                          robot_action_latency=.11, gripper_action_latency=.176,
                          action_accumulator=None)
    with pytest.raises(ValueError, match='expired or invalid'):
        UmiEnv.exec_actions(env, np.zeros((1, 14)), [time.time()+2],
                            compensate_latency=True, replace_from_time=time.time()+.1)


def test_scalar_gripper_splice_keeps_old_prefix_and_speed_cap():
    old = PoseTrajectoryInterpolator([10., 11.], [[0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]])
    times = np.linspace(10.2, 10.4, 20)
    new = old.schedule_waypoint([25, 0, 0, 0, 0, 0], 10.6, max_pos_speed=35,
                                max_rot_speed=35, curr_time=10.2,
                                last_waypoint_time=11., replace_from_time=10.4)
    np.testing.assert_allclose(new(times), old(times), atol=1e-10)
    t = np.linspace(10.2, new.times[-1], 1000)
    speed = np.abs(np.diff(new(t)[:, 0]))/np.diff(t)
    assert speed.max() <= 35+1e-7


def test_numerically_late_boundary_uses_current_pose_without_replaying_past():
    old = PoseTrajectoryInterpolator([10., 11.], [[0, 0, 0, 0, 0, 0], [.1, 0, 0, 0, 0, 0]])
    new = old.schedule_waypoint([.05, 0, 0, 0, 0, 0], 10.8, max_pos_speed=.5,
                                curr_time=10.5, replace_from_time=10.49)
    np.testing.assert_allclose(new(10.5), old(10.5))
    assert new.times[0] == 10.5
