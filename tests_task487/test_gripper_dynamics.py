"""Joint timing across policy scheduling and the real controller interpolator."""
from dataclasses import replace

import numpy as np
import pytest
import yaml

from task487_runtime.author_sync import AuthorSyncScheduler, GRIPPER_LATENCY
from task487_runtime.gripper_dynamics import load_gripper_dynamics
from task487_runtime.scheduler import RollingScheduler, RunState, SchedulerConfig
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


SPEED = np.rad2deg(3.0 / 20.0)


def write_config(path, speed=3.0, enabled=True):
    path.write_text(yaml.safe_dump({"/gripper_can_node": {"ros__parameters": {
        "gripper_stall_guard_enable": enabled, "gripper_stall_velocity_rad_s": speed}}}))


def test_load_motor_speed_as_joint_speed_without_changing_backend(tmp_path):
    path = tmp_path / "gripper.yaml"
    write_config(path)
    before = path.read_bytes()
    dynamics = load_gripper_dynamics(path)
    assert dynamics.joint_speed_deg_s == pytest.approx(8.59436692696)
    assert dynamics.config_path == path.resolve()
    assert path.read_bytes() == before


@pytest.mark.parametrize("speed", [0, -3, float("nan"), float("inf")])
def test_invalid_can_speed_cannot_silently_use_old_35deg_limit(tmp_path, speed):
    path = tmp_path / "gripper.yaml"
    write_config(path, speed)
    with pytest.raises(ValueError, match="finite and positive"):
        load_gripper_dynamics(path)


def test_unknown_backend_mode_or_missing_config_fails_before_motion(tmp_path):
    path = tmp_path / "gripper.yaml"
    with pytest.raises(ValueError, match="Invalid gripper CAN configuration"):
        load_gripper_dynamics(path)
    write_config(path, enabled=False)
    with pytest.raises(ValueError, match="stall guard enabled"):
        load_gripper_dynamics(path)


@pytest.mark.parametrize("rate,horizon", [(12.5, 20), (25.0, 16)])
@pytest.mark.parametrize("side", [6, 13])
def test_rolling_retimes_arm_and_gripper_together(rate, horizon, side):
    s = RollingScheduler(SchedulerConfig.for_policy_rate(
        rate, horizon, max_physical_gripper_speed_deg_s=SPEED))
    current = np.zeros(14)
    # Small arm travel alone takes .1 s; a 10 deg grip transition takes >1 s.
    targets = np.zeros((2, 14))
    targets[:, 0] = [.002, .003]
    targets[:, side] = [10., 0.]
    times = s._retime(targets, np.array([1/rate, 2/rate]), current, 0.0)
    assert times[0] == pytest.approx(10/SPEED)
    assert times[1] == pytest.approx(20/SPEED)


def make_plain():
    s = AuthorSyncScheduler(SchedulerConfig.for_policy_rate(
        12.5, 20, max_physical_gripper_speed_deg_s=SPEED),
        action_horizon=20, max_pos_speed=.5, max_rot_speed=.5)
    current = np.zeros(14)
    current[[6, 13]] = 1.
    s.activate(current)
    return s, current


def test_fast_opening_keeps_replanning_and_shared_physical_horizon():
    s, current = make_plain()
    raw = np.tile(current, (20, 1))
    raw[:, 0] = np.arange(1, 21) * .001
    raw[:, 6] = np.linspace(1., 15., 20)
    raw[:, 13] = np.linspace(1., 12., 20)
    s.mark_request_started(10.)
    stats = s.merge_chunk(raw, 10., 10.1, current)
    assert s.state is RunState.ACTIVE and stats.speed_retimed > 0
    batch = s.pop_batch(current, now=10.1)
    boundary = s.replace_from_time
    targets = np.vstack([current, [a.target for a in batch]])
    times = np.r_[boundary, [a.target_time for a in batch]]
    assert times[-1] - boundary <= .71 + 1e-9
    assert np.max(np.abs(np.diff(targets[:, [6, 13]], axis=0)) / np.diff(times)[:, None]) <= SPEED + 1e-8
    assert np.max(np.linalg.norm(np.diff(targets[:, :3], axis=0), axis=1) / np.diff(times)) <= .5 + 1e-8
    # A real scalar controller should accept exactly these times, with no
    # private delay that would desynchronize it from the arm.
    for side in (6, 13):
        interpolation = PoseTrajectoryInterpolator([10.], [[current[side], 0, 0, 0, 0, 0]])
        tail = 10.
        for i, action in enumerate(batch):
            due = action.target_time - GRIPPER_LATENCY
            interpolation = interpolation.schedule_waypoint(
                [action.target[side], 0, 0, 0, 0, 0], due,
                max_pos_speed=SPEED, max_rot_speed=SPEED, curr_time=10.11,
                last_waypoint_time=tail,
                replace_from_time=boundary-GRIPPER_LATENCY if i == 0 else None)
            tail = interpolation.times[-1]
            assert tail == pytest.approx(due, abs=1e-8)
    assert not s.request_due(10.47)
    assert s.request_due(10.48)


def test_partial_long_segment_advances_all_joints_without_committing_its_tail():
    anchor = np.zeros(14)
    targets = np.zeros((2, 14))
    targets[:, 0] = [.04, .08]
    targets[:, 6] = [10, 20]
    kept, times, nominal = AuthorSyncScheduler._trim_execution_window(
        targets, np.array([2., 4.]), np.array([.08, .16]), anchor, .7)
    assert kept.shape == (1, 14)
    assert kept[0, 0] == pytest.approx(.014)
    assert kept[0, 6] == pytest.approx(3.5)
    assert times.tolist() == [.7]
    assert nominal[0] == pytest.approx(.028)


def test_gripper_travel_time_already_allows_arm_to_meet_its_speed_cap():
    s, anchor = make_plain()
    targets = np.tile(anchor, (2, 1))
    targets[:, 0] = [.03, .04]
    targets[:, 6] = [5., 9.]
    # At nominal .01 s intervals the arm alone would need >150 ms of extra
    # timing at this deliberately low cap. The gripper's required travel
    # time already accommodates it; it must not cause a spurious HOLD.
    s.config = replace(s.config, max_physical_translation_speed_m_s=.1)
    times = s._limit_local_timing(targets, np.array([.01, .02]), anchor)
    assert times[-1] == pytest.approx(8 / SPEED)
    assert np.max(np.diff(np.r_[0., targets[:, 0]]) / np.diff(np.r_[0., times])) <= .1


def test_new_speed_does_not_weaken_existing_arm_outlier_guard():
    s, current = make_plain()
    s.config = replace(s.config, max_physical_translation_speed_m_s=.001)
    raw = np.tile(current, (20, 1))
    raw[:, 0] = np.arange(1, 21) * .001
    raw[:, 6] = np.linspace(1, 15, 20)
    s.mark_request_started(10.)
    with pytest.raises(RuntimeError, match="velocity"):
        s.merge_chunk(raw, 10., 10.1, current)
    assert s.state is RunState.HOLD
