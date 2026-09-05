"""Plain WebSocket + author-style Bezier execution, with Task487 safety guards.

No RTC prefix or old-tail blending. Bounded local timing correction keeps the
controller caps without rejecting every slightly overspeed segment. The network call
still runs in the existing single worker so stop keys remain responsive.
Based on eval_mink_new's ordinary client/stitch_arm_chunk, not its RTC client.
"""
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from task487_runtime.gripper_preload import ClosureIntent, MAX_PRELOAD_DEG, preload_right_gripper
from task487_runtime.gripper_compensation import GripperIntent, MAX_COMPENSATION_DEG, compensate_grippers
from task487_runtime.scheduler import (
    MergeStats, RollingScheduler, RunState, ScheduledAction, SchedulerConfig,
    UnsafeChunkError,
)

AUTHOR_MAX_POS_SPEED = 0.35 * math.sqrt(3.0)
AUTHOR_MAX_ROT_SPEED = math.sqrt(3.0)
DEFAULT_SYNC_MAX_POS_SPEED = 0.15
DEFAULT_SYNC_MAX_ROT_SPEED = 0.35
DEFAULT_SYNC_SPEED_SCALE = 0.5
ROBOT_LATENCY = 0.11
GRIPPER_LATENCY = 0.176
MAX_LOCAL_DELAY_S = 0.15


def stitch_targets(targets, current, previous, dt, servo_dt=0.05):
    """Cubic position / SLERP rotation; preserve every gripper knot and value.

    Unlike the source helper, never clamp the stale-prefix count to retain old
    actions. Caller has already discarded stale actions. Keep action-rate
    knots in the connection too, so a close/open transition is not skipped by
    the Cartesian smoothing. The connection ends at (c+1)*dt (not c*dt).
    """
    count = len(targets)
    if count < 3:
        raise UnsafeChunkError("Not enough future actions for a Bezier connection")
    connection = int(np.clip(math.floor(0.1 * count), 1, count - 2))
    duration = (connection + 1) * dt
    knots = np.arange(1, count + 1) * dt
    times = np.unique(np.concatenate((knots, np.linspace(
        0, duration, max(2, math.ceil(duration / servo_dt)) + 1)[1:]))).round(12)
    times = np.unique(times)
    output = np.empty((len(times), 14), dtype=np.float64)

    def unit(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-9 else np.zeros(3)

    connecting = times <= duration + 1e-10
    t = np.minimum(times[connecting] / duration, 1.0)
    for offset in (0, 7):
        start = current[offset:offset+3]
        end = targets[connection, offset:offset+3]
        history = unit(start - previous[offset:offset+3])
        future = unit(targets[connection+1, offset:offset+3] - targets[connection-1, offset:offset+3])
        fallback = unit(end - start)
        history = history if np.any(history) else fallback
        future = future if np.any(future) else fallback
        length = 0.33 * np.linalg.norm(end-start)
        p1, p2 = start + length*history, end - length*future
        u = t[:, None]
        output[connecting, offset:offset+3] = (
            (1-u)**3*start + 3*(1-u)**2*u*p1 + 3*(1-u)*u**2*p2 + u**3*end)
        output[connecting, offset+3:offset+6] = Slerp(
            [0, 1], Rotation.from_rotvec([current[offset+3:offset+6],
                                         targets[connection, offset+3:offset+6]]))(t).as_rotvec()
        # All post-connection points are original action knots, not averages
        # with an older policy chunk.
        tail_indices = np.rint(times[~connecting] / dt).astype(int) - 1
        output[~connecting, offset:offset+6] = targets[tail_indices, offset:offset+6]
        output[:, offset+6] = np.interp(
            times, np.r_[0, knots], np.r_[current[offset+6], targets[:, offset+6]])
    return output, times


class AuthorSyncScheduler(RollingScheduler):
    """Reuse only the existing state/diagnostic/physical-guard interfaces.

    All timeline construction, replacement, dispatch and inference cadence are
    separate from RollingScheduler; none of its retiming/RTC merging runs.
    """
    def __init__(self, base_config, *, action_horizon, steps_per_inference=6,
                 max_pos_speed=AUTHOR_MAX_POS_SPEED, max_rot_speed=AUTHOR_MAX_ROT_SPEED,
                 speed_scale=1.0, right_gripper_preload_deg=0.0,
                 gripper_close_compensation_deg=0.0, gripper_open_compensation_deg=0.0,
                 gripper_open_limits_deg=None, right_before_left=False):
        if not 1 <= steps_per_inference <= action_horizon - 3:
            raise ValueError("sync steps must be between 1 and action_horizon-3")
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("sync speed scale must be finite, positive and <= 1")
        if not np.isfinite(right_gripper_preload_deg) or not 0 <= right_gripper_preload_deg <= MAX_PRELOAD_DEG:
            raise ValueError(f"right gripper preload must be finite and in [0, {MAX_PRELOAD_DEG:g}] degrees")
        if right_gripper_preload_deg and base_config.max_physical_gripper_speed_deg_s is None:
            raise ValueError("right gripper preload requires the physical gripper speed")
        for value in (gripper_close_compensation_deg, gripper_open_compensation_deg):
            if not np.isfinite(value) or not 0 <= value <= MAX_COMPENSATION_DEG:
                raise ValueError(f"gripper compensation must be finite and in [0, {MAX_COMPENSATION_DEG:g}] degrees")
        compensation_enabled = bool(gripper_close_compensation_deg or gripper_open_compensation_deg)
        if compensation_enabled:
            if right_gripper_preload_deg:
                raise ValueError("gripper compensation cannot be combined with right gripper preload")
            if base_config.max_physical_gripper_speed_deg_s is None:
                raise ValueError("gripper compensation requires the physical gripper speed")
            limits = np.asarray(gripper_open_limits_deg, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits <= 0):
                raise ValueError("gripper compensation requires both physical opening limits")
        for value, maximum, name in ((max_pos_speed, AUTHOR_MAX_POS_SPEED, "position"),
                                      (max_rot_speed, AUTHOR_MAX_ROT_SPEED, "rotation")):
            if not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"sync {name} speed must be finite, positive and <= {maximum:g}")
        super().__init__(replace(
            base_config, max_physical_translation_speed_m_s=max_pos_speed,
            max_physical_rotation_speed_rad_s=max_rot_speed, handoff_blend_steps=0,
            complete_chunk_before_replan=False, commit_window_s=None))
        self.action_horizon = action_horizon
        self.steps_per_inference = steps_per_inference
        self.speed_scale = speed_scale
        self.right_gripper_preload_deg = right_gripper_preload_deg
        self._right_closure_intent = None
        self.gripper_close_compensation_deg = gripper_close_compensation_deg
        self.gripper_open_compensation_deg = gripper_open_compensation_deg
        self.gripper_open_limits_deg = None if gripper_open_limits_deg is None else tuple(gripper_open_limits_deg)
        self.compensation_enabled = compensation_enabled
        self._gripper_intents = None
        self.right_before_left = bool(right_before_left)
        if self.right_before_left and not compensation_enabled:
            raise ValueError("right-before-left requires gripper compensation")
        self._queue_policy_grippers = np.empty((0, 2))
        self._trajectory_policy_grippers = None
        self._left_released = not self.right_before_left
        self._right_closed = False
        self._right_release_goal = None
        self._release_reached_since = None
        self._left_hold = None
        self.latency_budget = max(ROBOT_LATENCY, GRIPPER_LATENCY)
        self._next_request_at = 0.0
        self._pending_since = None
        self._controller = self._previous_controller = None
        self._trajectory_times = self._trajectory_targets = None
        self.replace_from_time = None

    def activate(self, live_target=None, *, gripper_policy_live=None):
        if live_target is None:
            raise ValueError("author-sync activation requires real feedback")
        resume_gripper_episode = gripper_policy_live is not None and self._gripper_intents is not None
        super().activate(live_target)
        self._next_request_at = 0.0
        self._pending_since = None
        self._controller = np.array(live_target, dtype=np.float64, copy=True)
        self._previous_controller = self._controller.copy()
        self._right_closure_intent = ClosureIntent(float(live_target[6]))
        policy_live = np.asarray(np.asarray(live_target)[[6, 13]] if gripper_policy_live is None else gripper_policy_live, dtype=float)
        if policy_live.shape != (2,) or not np.isfinite(policy_live).all():
            raise ValueError("invalid gripper policy-coordinate feedback")
        self._initial_policy_grippers = policy_live.copy()
        # A pause must not withdraw the bias of an already held object.
        bias = np.asarray(live_target)[[6, 13]] - policy_live
        if not resume_gripper_episode:
            self._gripper_intents = tuple(GripperIntent(float(q), int(np.sign(b)) if abs(b) > .1 else 0)
                                         for q, b in zip(policy_live, bias))
        self._queue_policy_grippers = np.empty((0, 2))
        self._trajectory_policy_grippers = None
        if not resume_gripper_episode:
            self._left_released = not self.right_before_left
            self._right_closed = self._gripper_intents[0].direction < 0
            self._right_opened = self._gripper_intents[0].direction != 0
            self._right_release_goal = None
        self._release_reached_since = None
        self._left_hold = np.asarray(live_target)[7:].copy()
        self._trajectory_times = self._trajectory_targets = None
        self.replace_from_time = None

    def reset_gripper_episode(self):
        if self.state is RunState.ACTIVE:
            raise RuntimeError("gripper episode reset requires HOLD")
        self._gripper_intents = None

    def observe_controller(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (14,) or not np.isfinite(target).all():
            self.hold("Invalid controller target")
            raise UnsafeChunkError(self.reason)
        self._previous_controller = target.copy() if self._controller is None else self._controller.copy()
        self._controller = target.copy()

    def rtc_prefix_targets(self):
        return np.empty((0, 14), dtype=np.float64)

    def request_due(self, now=None):
        return (self.state is RunState.ACTIVE and not self._request_pending
                and now is not None and now + 1e-6 >= self._next_request_at)

    def mark_request_started(self, now=None):
        if not self.request_due(now):
            raise RuntimeError("Plain inference request is not due")
        self._request_pending = True
        self._pending_since = float(now)
        period = self.steps_per_inference * self.config.dt
        if self._next_request_at == 0.0:
            self._next_request_at = float(now) + period
        else:
            # Keep the author's fixed-cycle cadence. Do not accumulate a full
            # extra control tick whenever the floating-point deadline slips.
            missed = max(0, math.floor((now-self._next_request_at+1e-6) / period))
            self._next_request_at += (missed+1) * period

    def _planned_target(self, physical_time):
        """Previously submitted trajectory, indexed by physical target time."""
        if self._trajectory_times is None:
            return self._controller.copy()
        times, targets = self._trajectory_times, self._trajectory_targets
        t = np.clip(physical_time, times[0], times[-1])
        result = np.array([np.interp(t, times, targets[:, i]) for i in range(14)])
        for offset in (0, 7):
            result[offset+3:offset+6] = Slerp(
                times, Rotation.from_rotvec(targets[:, offset+3:offset+6]))(t).as_rotvec()
        return result

    def _limit_local_timing(self, targets, times, anchor):
        """Slow the shared path clock, retaining physical caps and outlier checks.

        Playback slowdown is intentional, separate from the 150 ms correction
        budget. Gripper travel remains a minimum duration, not multiplied a
        second time. Observation age, device latency and inference cadence
        always use wall time, independently of this path clock.
        """
        corrected = np.empty_like(times)
        previous, elapsed, arm_extra, nominal_previous = anchor, 0.0, 0.0, 0.0
        for i, (target, nominal) in enumerate(zip(targets, times)):
            nominal_dt = nominal - nominal_previous
            playback_dt = nominal_dt / self.speed_scale
            base_duration = max(playback_dt, self._gripper_duration(previous, target))
            duration = max(base_duration, self._minimum_duration(previous, target, include_gripper=False))
            arm_extra += duration-base_duration
            elapsed += duration
            corrected[i] = elapsed
            if arm_extra > MAX_LOCAL_DELAY_S + 1e-7:
                details = []
                for o, arm in ((0, "right"), (7, "left")):
                    speed = np.linalg.norm(target[o:o+3]-previous[o:o+3]) / playback_dt
                    rot = (Rotation.from_rotvec(previous[o+3:o+6]).inv() *
                           Rotation.from_rotvec(target[o+3:o+6])).magnitude() / playback_dt
                    details.append(f"{arm}={speed:.3f}m/s,{rot:.3f}rad/s")
                raise UnsafeChunkError(
                    f"Local velocity correction needs {arm_extra:.3f}s > "
                    f"{MAX_LOCAL_DELAY_S:.3f}s budget at segment {i}; " + "; ".join(details) +
                    f" (caps={self.config.max_physical_translation_speed_m_s:.3f}m/s,"
                    f"{self.config.max_physical_rotation_speed_rad_s:.3f}rad/s; "
                    f"playback={self.speed_scale:g}x)")
            previous, nominal_previous = target, nominal
        return corrected

    @staticmethod
    def _trim_execution_window(targets, times, nominal_times, anchor, window):
        """Cut the retimed path at a short physical horizon, including a partial segment.

        All joints share this path clock. The next inference still happens on
        its usual cadence; a slow gripper never creates seconds of committed
        open-loop motion or forces the arm to finish before the gripper.
        """
        if times[-1] <= window + 1e-9:
            return targets, times, nominal_times
        count = int(np.searchsorted(times, window, side="right"))
        if count and abs(times[count-1] - window) < 1e-9:
            return targets[:count], times[:count], nominal_times[:count]
        left = anchor if count == 0 else targets[count-1]
        left_time = 0.0 if count == 0 else times[count-1]
        left_nominal = 0.0 if count == 0 else nominal_times[count-1]
        weight = (window-left_time) / (times[count]-left_time)
        cut = left + weight*(targets[count]-left)
        for offset in (0, 7):
            cut[offset+3:offset+6] = Slerp(
                [0, 1], Rotation.from_rotvec([left[offset+3:offset+6],
                                              targets[count, offset+3:offset+6]]))(weight).as_rotvec()
        cut_nominal = left_nominal + weight*(nominal_times[count]-left_nominal)
        return (np.vstack([targets[:count], cut]), np.r_[times[:count], window],
                np.r_[nominal_times[:count], cut_nominal])

    def advance(self, now, controller_target):
        if self.state is not RunState.ACTIVE:
            return 0
        if self.right_before_left and not self._left_released and self._right_release_goal is not None:
            if controller_target[6] >= self._right_release_goal - .5:
                if self._release_reached_since is None:
                    self._release_reached_since = now
                if now - self._release_reached_since >= .16:
                    self._left_released = True
            else:
                self._release_reached_since = None
        if self._request_pending and self._pending_since is not None and now-self._pending_since > 2.0:
            self.hold("Plain inference timeout (2s)")
            raise UnsafeChunkError(self.reason)
        # These are physical target times. Do not compare them against an
        # ahead-of-time, latency-compensated robot command. Physical following
        # is independently checked by validate_physical_tracking every tick.
        count = int(np.count_nonzero(self._sent & (self._times <= now)))
        if count:
            self._queue, self._times = self._queue[count:], self._times[count:]
            self._queue_policy_grippers = self._queue_policy_grippers[count:]
            self._nominal_times, self._sent = self._nominal_times[count:], self._sent[count:]
            self._has_executed = True
        return count

    def merge_chunk(self, targets, observation_time, now, live_target):
        try:
            if self.state is not RunState.ACTIVE:
                raise UnsafeChunkError("Cannot merge a chunk while HOLD")
            targets = np.array(targets, dtype=np.float64, copy=True)
            if targets.shape != (self.action_horizon, 14):
                raise UnsafeChunkError(f"Invalid plain chunk shape {targets.shape}")
            if not np.isfinite(targets).all() or not np.isfinite([observation_time, now]).all():
                raise UnsafeChunkError("Non-finite plain chunk/timestamp")
            if observation_time > now + 0.01:
                raise UnsafeChunkError("Observation timestamp is in the future")
            stale = int(math.ceil((max(0, now-observation_time) + self.latency_budget) / self.config.dt))
            if stale > len(targets)-3:
                raise UnsafeChunkError("Plain chunk expired: fewer than 3 future actions")
            self.validate_physical_tracking(live_target, self._controller)
            if self.right_before_left and not self._left_released:
                # Replace the gated hand with a stationary physical target
                # BEFORE validation. The active hand retains all guards.
                targets[:, 7:] = self._left_hold
            self._validate(targets[stale:], self._controller)
            # Preserve the old trajectory up to a common future splice time.
            # Re-anchoring to NOW and putting that pose in the future makes
            # every new chunk brake/reverse while the old chunk is moving.
            splice_time = now + self.latency_budget + self.config.dispatch_lead_s
            anchor = self._planned_target(splice_time)
            history = (self._planned_target(splice_time-.01)
                       if self._trajectory_times is not None else self._previous_controller)
            future = targets[stale:]
            policy_anchor = (self._initial_policy_grippers if self._trajectory_policy_grippers is None else
                np.array([np.interp(splice_time, self._trajectory_times, self._trajectory_policy_grippers[:, i])
                          for i in range(2)]))
            logical_anchor = anchor.copy()
            logical_anchor[[6, 13]] = policy_anchor
            logical_future = future.copy()
            if self.right_before_left and not self._left_released:
                logical_future[:, 13] = self._initial_policy_grippers[1]
                future = logical_future
            next_intent = self._right_closure_intent
            next_gripper_intents = self._gripper_intents
            if self.right_gripper_preload_deg:
                future, next_intent = preload_right_gripper(
                    future, self._right_closure_intent, self.right_gripper_preload_deg)
            if self.compensation_enabled:
                future, next_gripper_intents = compensate_grippers(
                    future, self._gripper_intents,
                    close_deg=self.gripper_close_compensation_deg,
                    open_deg=self.gripper_open_compensation_deg,
                    open_limits_deg=self.gripper_open_limits_deg)
            stitched, relative = stitch_targets(
                future, anchor, history, self.config.dt)
            logical_knots = np.r_[0., np.arange(1, len(logical_future) + 1) * self.config.dt]
            policy_stitched = np.column_stack([np.interp(
                relative, logical_knots, np.r_[policy_anchor[i], logical_future[:, col]])
                for i, col in enumerate((6, 13))])
            execution_window = (self.steps_per_inference+1)*self.config.dt + MAX_LOCAL_DELAY_S
            keep = relative <= execution_window + 1e-10
            stitched, relative = stitched[keep], relative[keep]
            policy_grippers = policy_stitched[keep]
            self._validate(stitched, anchor)  # also checks Bezier overshoot
            corrected = self._limit_local_timing(stitched, relative, anchor)
            policy_times = np.r_[0., corrected]
            policy_values = np.vstack([policy_anchor, policy_grippers])
            stitched, corrected, relative = self._trim_execution_window(
                stitched, corrected, relative, anchor, execution_window)
            policy_grippers = np.column_stack([np.interp(corrected, policy_times, policy_values[:, i]) for i in range(2)])
            delays = corrected-relative
            self._queue = stitched
            self.replace_from_time = splice_time
            self._times = splice_time + corrected
            self._nominal_times = splice_time + relative
            self._trajectory_times = np.r_[splice_time, self._times]
            self._trajectory_targets = np.vstack([anchor, stitched])
            self._queue_policy_grippers = policy_grippers
            self._trajectory_policy_grippers = np.vstack([policy_anchor, policy_grippers])
            self._sent = np.zeros(len(stitched), dtype=bool)
            self._request_pending = False
            self._pending_since = None
            self._right_closure_intent = next_intent
            if self.right_before_left and not self._left_released:
                if next_gripper_intents[0].direction > 0:
                    self._right_opened = True
                if self._right_opened and next_gripper_intents[0].direction < 0:
                    self._right_closed = True
                if (self._right_closed and self._gripper_intents[0].direction < 0
                        and next_gripper_intents[0].direction > 0 and future[0, 6] > live_target[6] + 1.):
                    self._right_release_goal = float(future[0, 6])
                    self._release_reached_since = None
            self._gripper_intents = next_gripper_intents
            return MergeStats(len(stitched), stale, 0, stale, 0,
                              int(np.count_nonzero(delays > 1e-7)),
                              float(delays.max(initial=0.0)), 0, 0)
        except UnsafeChunkError as exc:
            self.hold(str(exc))
            raise

    def gripper_handoff_status(self):
        return {"enabled": self.right_before_left, "left_allowed": self._left_released,
                "right_closed": self._right_closed, "right_release_goal_deg": self._right_release_goal,
                "release_reached_since": self._release_reached_since,
                "confirmation": "physical opening target reached within 0.5deg for 0.16s; not visual placement confirmation"}

    def pop_batch(self, live_target, now=None, max_actions=None):
        if self.state is not RunState.ACTIVE:
            return []
        if not len(self._queue):
            if self._has_executed or not self._request_pending:
                self.hold("plain action queue exhausted")
            return []
        indices = np.flatnonzero(~self._sent)
        if max_actions is not None:
            if max_actions <= 0:
                raise ValueError("max_actions must be positive")
            indices = indices[:max_actions]
        if len(indices) and (now is None or self.replace_from_time-self.latency_budget <= now+.01):
            self.hold("plain batch expired before dispatch")
            raise UnsafeChunkError(self.reason)
        result = [ScheduledAction(self._queue[i].copy(), float(self._times[i]),
                                  float(self._nominal_times[i]),
                                  self._queue_policy_grippers[i].copy() if self.compensation_enabled else None)
                  for i in indices]
        self._sent[indices] = True
        if result:
            self._last_scheduled_target = result[-1].target.copy()
            self._last_scheduled_time = result[-1].target_time
        return result
