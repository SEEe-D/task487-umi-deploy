"""Full client loop with inert hardware/policy substitutes; no sockets or ROS."""
from types import SimpleNamespace
from dataclasses import replace
import json

import numpy as np
import pytest

import task487_client as client
from task487_runtime.contract import POLICY_RUNTIME_CONTRACTS
from task487_runtime.worker import InferenceResult


@pytest.mark.parametrize("bounded", [True, False])
@pytest.mark.parametrize("variant", ["raw", "cnc", "wrist_only"])
@pytest.mark.parametrize("compensation", [False, True])
def test_plain_client_lifecycle_cameras_requests_stop_and_budget(monkeypatch, tmp_path, caplog, bounded, variant, compensation):
    events, requests, env_kwargs = [], [], {}
    config_path = tmp_path / "gripper.yaml"
    config_path.write_text("/gripper_can_node:\n  ros__parameters:\n    gripper_stall_guard_enable: true\n    gripper_stall_velocity_rad_s: 3.0\n")
    monkeypatch.setenv("TASK487_GRIPPER_CONFIG", str(config_path))
    # main's force=True would remove pytest's capture handler.
    monkeypatch.setattr(client.logging, "basicConfig", lambda **kw: None)
    clock = SimpleNamespace(now=100.0)
    def sleep(seconds):
        clock.now += max(seconds, 0.001)
    monkeypatch.setattr(client, "time", SimpleNamespace(time=lambda: clock.now,
                        monotonic=lambda: clock.now, sleep=sleep))
    signal_handler = {}
    def signal(_kind, callback):
        old = signal_handler.get("callback", lambda *a: None)
        signal_handler["callback"] = callback
        return old
    monkeypatch.setattr(client, "signal", SimpleNamespace(SIGINT=2, signal=signal))
    monkeypatch.setattr(client, "acquire_client_lock", lambda: SimpleNamespace(close=lambda: None))
    class Manager:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(client, "SharedMemoryManager", Manager)
    class PassiveRecorder(Manager):
        def __init__(self, *args): pass
        def check(self): pass
    monkeypatch.setattr(client, "PassiveGripperTelemetry", PassiveRecorder)
    class Keys(Manager):
        tick = 0
        def poll(self):
            self.tick += 1
            if self.tick == 1: return ["d"]
            # Stop/restart once while inference is pending, once after motion.
            if not bounded and self.tick in (2, 17): return ["s"]
            if not bounded and self.tick in (3, 19): return ["d"]
            if self.tick == 25:
                signal_handler["callback"](2, None)
            return []
    monkeypatch.setattr(client, "TerminalKeys", Keys)
    monkeypatch.setattr(client, "validate_thor", lambda *a, **k: None)
    monkeypatch.setattr(client, "wait_for_thor", lambda *a, **k: None)

    class Env(Manager):
        def __init__(self, **kwargs):
            env_kwargs.update(kwargs)
            self.pose = np.zeros(14)
            self.pose[[6, 13]] = 1
            self.commands = []
            self.policy_angles = np.ones(2)
        def __exit__(self, *args): events.append(("env_exit",))
        def get_task_instruction(self): return None
        def get_robot_state(self):
            # Ideal tracking of the last commanded segment, purely in memory.
            for actions, times, refs in self.commands:
                for action, timestamp, ref in zip(actions, times, refs):
                    if timestamp <= clock.now:
                        self.pose = action.copy()
                        self.policy_angles = ref.copy()
            return [{"ActualTCPPose": self.pose[o:o+6], "TargetTCPPose": self.pose[o:o+6]}
                    for o in (0, 7)]
        def get_gripper_state(self):
            return [{"gripper_position": np.array([self.pose[o]]), "gripper_policy_position": np.array([self.policy_angles[i]])}
                    for i, o in enumerate((6, 13))]
        def get_obs(self):
            self.get_robot_state()
            obs = {"timestamp": np.array([clock.now-.08, clock.now])}
            for camera in (0, 1, 2, 3, 5):
                obs[f"camera{camera}_rgb"] = np.full((2, 224, 224, 3), camera, np.uint8)
            for arm, o in enumerate((0, 7)):
                obs[f"robot{arm}_eef_pos"] = np.tile(self.pose[o:o+3], (2, 1))
                obs[f"robot{arm}_eef_rot_axis_angle"] = np.tile(self.pose[o+3:o+6], (2, 1))
                obs[f"robot{arm}_gripper_angle"] = np.full((2, 1), self.pose[o+6])
                obs[f"robot{arm}_gripper_policy_angle"] = np.full((2, 1), self.policy_angles[arm])
            return obs
        def exec_actions(self, *, actions, timestamps, compensate_latency, time_is_new, replace_from_time, gripper_policy_angles=None):
            assert compensate_latency and time_is_new
            assert clock.now + client.GRIPPER_LATENCY < replace_from_time < timestamps[0]
            assert np.all(timestamps - client.GRIPPER_LATENCY > clock.now)
            if compensation:
                assert gripper_policy_angles.shape == (len(actions), 2)
            self.commands.append((actions.copy(), timestamps.copy(),
                                  actions[:, [6, 13]].copy() if gripper_policy_angles is None else gripper_policy_angles.copy()))
            events.append(("dispatch", len(actions)))
        def hold(self, **kwargs):
            self.commands.clear()
            events.append(("hold",))
        def go_home(self, **kwargs): pytest.fail("must never HOME automatically")
    monkeypatch.setattr(client, "UmiEnv", Env)
    contract = POLICY_RUNTIME_CONTRACTS[f"pi05_umi_task487_{variant}_4w_12_5_v1"]
    metadata = {k: v for k, v in contract.expected_metadata().items() if not k.startswith("rtc_")}
    monkeypatch.setattr(client.websocket_client_policy, "WebsocketClientPolicy",
                        lambda *a: SimpleNamespace(get_server_metadata=lambda: metadata))
    class Worker(Manager):
        busy = False
        def __init__(self, policy, action_horizon, recorder):
            self.horizon = action_horizon
            self.recorder = recorder
            self.sequence = 0
        def __exit__(self, *args): self.close()
        def submit(self, request):
            assert not self.busy
            assert "action_prefill" not in request.observation
            assert "action_prefill_len" not in request.observation
            assert "actions" not in request.observation
            assert request.observation["prompt"] == "Vegetable and Fruit Sorting"
            expected_cameras = {"cam_left_top", "cam_left_down", "cam_right_top", "cam_right_down"}
            if variant != "wrist_only": expected_cameras.add("cam_head")
            assert {k for k in request.observation if k.startswith("cam_")} == expected_cameras
            self.sequence += 1
            request = replace(request, request_id=self.sequence)
            requests.append(request)
            self.request, self.ready, self.busy = request, clock.now+.1, True
        def poll(self):
            if not self.busy or clock.now < self.ready: return None
            self.busy = False
            target = np.zeros((self.horizon, 14))
            for arm, o in enumerate((0, 7)):
                target[:, o:o+6] = self.request.tcp_bases[arm]
                target[:, o] += np.arange(1, self.horizon+1) * .001
                target[:, o+6] = 1
            return InferenceResult(self.request, target, .1, None)
        def close(self): events.append(("worker_close",))
        def record_disposition(self, result, status, **details):
            self.recorder.record("disposition", result.request.request_id,
                                 status=status, round_id=result.request.round_id, **details)
    monkeypatch.setattr(client, "InferenceWorker", Worker)
    argv = ["task487_client.py", "--execution-mode", "author-sync", "--execute", "--no-task-ui",
            "--output", str(tmp_path)]
    if not bounded: argv.append("--continuous")
    if compensation:
        argv.extend(["--sync-gripper-close-compensation-deg", "5", "--sync-gripper-open-compensation-deg", "5", "--sync-right-before-left"])
    monkeypatch.setattr(client.sys, "argv", argv)
    monkeypatch.setenv("UMI_ACTION_LOG_DIR", str(tmp_path))
    client.main()
    assert not env_kwargs["init_joints"]
    assert env_kwargs["frequency"] == 12.5 and env_kwargs["thor_resize_with_pad"]
    assert len(env_kwargs["thor_cameras"]) == (4 if variant == "wrist_only" else 5)
    assert env_kwargs["max_pos_speed"] == pytest.approx(.15 / np.sqrt(3))
    assert env_kwargs["max_rot_speed"] == pytest.approx(.35 / np.sqrt(3))
    assert env_kwargs["gripper_open_rad"] == client.GRIPPER_SAFE_OPEN_RAD
    assert env_kwargs["gripper_max_speed_dps"] == pytest.approx(np.rad2deg(3 / 20))
    # Physical HOLD happens before inference/telemetry/logging shutdown waits.
    assert events[-3:] == [("hold",), ("worker_close",), ("env_exit",)]
    assert ("worker_close",) in events
    dispatches = [e for e in events if e[0] == "dispatch"]
    assert dispatches
    records = [json.loads(line) for line in (tmp_path/"policy_io.jsonl").read_text().splitlines()]
    statuses = [r["status"] for r in records if r["event"] == "disposition"]
    assert "warmup_discarded" in statuses and "accepted" in statuses
    chunks = [json.loads(line) for line in (tmp_path/"chunk_diagnostics.jsonl").read_text().splitlines()]
    assert all(r["policy_request_id"] and r["policy_session_id"] for r in chunks)
    if compensation:
        assert all(r["gripper_handoff"]["enabled"] for r in chunks)
        assert requests[-1].diagnostics["gripper_coordinates"] == "policy_reference_plus_measured_tracking_error"
    if bounded:
        assert sum(e[1] for e in dispatches) == 5
        assert len(requests) == 2  # one plain warmup + one bounded inference
    else:
        assert len(requests) >= 3
        assert len([e for e in events if e[0] == "hold"]) >= 3
        assert "Discarded stale inference from round=1; active round=2" in caplog.text
        assert "stale_round" in statuses


def test_final_hold_on_exception_but_no_motion_in_dry_run():
    calls = []
    env = SimpleNamespace(hold=lambda **kw: calls.append(kw))
    with pytest.raises(ValueError):
        with client.physical_hold_on_exit(env, True):
            raise ValueError("synthetic client fault")
    assert calls == [{"wait": True, "timeout": 2.0}]
    with client.physical_hold_on_exit(env, False): pass
    assert len(calls) == 1
