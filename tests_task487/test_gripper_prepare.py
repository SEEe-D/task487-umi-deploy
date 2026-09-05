from types import SimpleNamespace

import numpy as np
import pytest

import task487_client as client
from task487_runtime.contract import POLICY_RUNTIME_CONTRACTS


def run_prepare(monkeypatch, *, blocked=None, initial=(3.4254, 7.1082)):
    clock = SimpleNamespace(t=100.)
    positions = np.array(initial, dtype=float)
    commands = []
    latched = [True, True]
    def send(hand, goal, when):
        commands.append((hand, goal))
        if hand == blocked:
            return
        if goal > positions[hand] + 1.:
            latched[hand] = False
        if not latched[hand]:
            positions[hand] = goal
    grippers = [SimpleNamespace(schedule_waypoint=lambda q, t, hand=i: send(hand, q, t)) for i in range(2)]
    env = SimpleNamespace(grippers=grippers, get_gripper_state=lambda: [
        {'gripper_position': np.array([q])} for q in positions])
    monkeypatch.setattr(client, 'time', SimpleNamespace(time=lambda: clock.t, monotonic=lambda: clock.t,
                        sleep=lambda dt: setattr(clock, 't', clock.t + dt)))
    contract = POLICY_RUNTIME_CONTRACTS['pi05_umi_task487_wrist_only_4w_12_5_v1']
    return lambda: client.prepare_grippers_for_runtime(env, contract, timeout=.3), positions, commands, latched


def test_previous_dual_stall_latch_opens_before_closing_to_one_degree(monkeypatch):
    prepare, pos, commands, latched = run_prepare(monkeypatch)
    prepare()
    np.testing.assert_allclose([q for _, q in commands[:2]], [5.4254, 9.1082])
    assert commands[2:] == [(0, 1.), (1, 1.)]
    assert latched == [False, False]
    np.testing.assert_allclose(pos, [1., 1.])


def test_failed_open_freezes_and_never_proceeds_to_close(monkeypatch):
    prepare, pos, commands, _ = run_prepare(monkeypatch, blocked=1)
    with pytest.raises(RuntimeError, match='open-to-rearm'):
        prepare()
    assert not any(q == 1. for _, q in commands)
    np.testing.assert_allclose([q for _, q in commands[-2:]], pos)


def test_release_targets_never_exceed_verified_open_endpoints(monkeypatch):
    prepare, pos, commands, _ = run_prepare(monkeypatch, initial=(33.9, 23.9))
    with pytest.raises(RuntimeError, match='checkpoint-close'):
        prepare()
    np.testing.assert_allclose([q for _, q in commands[:2]], client.GRIPPER_SAFE_OPEN_DEG)
    # Already within opening tolerance, but insufficient travel to rearm.
    # Closure must still fail and freeze rather than report preparation ready.
    np.testing.assert_allclose(pos, [33.9, 23.9])
    np.testing.assert_allclose([q for _, q in commands[-2:]], pos)
    assert all(0 <= q <= client.GRIPPER_SAFE_OPEN_DEG[hand] for hand, q in commands)
