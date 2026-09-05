"""Reachability-aware rolling action scheduler for Task487."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from scipy.spatial.transform import Rotation


class RunState(Enum):
    HOLD = "HOLD"
    ACTIVE = "ACTIVE"


class UnsafeChunkError(RuntimeError):
    def __init__(self, message: str, action_index: int | None = None):
        super().__init__(message)
        self.action_index = action_index


@dataclass(frozen=True)
class SchedulerConfig:
    control_hz: float = 25.0
    # These defaults retain the legacy 25 Hz profile for unit-level users.
    # Production uses for_policy_rate() to derive the matching 25 or 12.5 Hz
    # durations from server metadata.
    request_every_steps: int = 5
    # Already-dispatched future actions remain while inference runs and become
    # the immutable RTC hard prefix.
    replan_remaining_steps: int = 5
    # Commit only the duration-equivalent prefix; replace its uncommitted tail
    # with the new chunk.
    commit_steps: int = 10
    # Reachability retiming breaks the equivalence between a policy step and
    # wall time. Production 12.5 Hz profiles therefore cap the immutable
    # controller prefix by actual executable duration. The remaining policy
    # horizon stays replaceable so it can provide a continuous blend reference.
    commit_window_s: float | None = None
    # Slow 4w execution must make progress through the accepted suffix. Do
    # not replace it on every short dispatch window and starve tail actions
    # (notably closing). Still dispatch only the bounded controller prefix.
    complete_chunk_before_replan: bool = False
    dispatch_lead_s: float = 0.05
    # These must match the effective (post cube_diag) limits in UmiEnv's
    # RosTargetInterpolationController instances.
    max_physical_translation_speed_m_s: float = 0.02
    max_physical_rotation_speed_rad_s: float = 0.08
    # Production reads this from the CAN node (motor rad/s / gear ratio).
    # None preserves the arm-only contract for standalone scheduler users.
    max_physical_gripper_speed_deg_s: float | None = None
    # At most one committed prefix plus one policy horizon is retained.
    max_queue_steps: int = 30
    # Cross-fade the first generated suffix samples with the replaceable,
    # already safety-checked tail of the previous chunk. RTC fixes position at
    # the prefix boundary; this additionally keeps handoff velocity/direction
    # continuous instead of letting independently sampled suffixes reverse it.
    handoff_blend_steps: int = 5
    # Task487 same-episode adjacent-action 99.9th percentiles are 29.3 mm
    # translation and 0.108 rad rotation. Keep modest numerical/model seam
    # headroom above that percentile: physical motion is independently retimed
    # to 0.02 m/s, so this remains an outlier guard rather than a speed limit.
    max_translation_step_m: float = 0.035
    max_rotation_step_rad: float = 0.12
    max_translation_tracking_error_m: float = 0.050
    max_rotation_tracking_error_rad: float = 0.30
    max_rtc_prefix_translation_error_m: float = 0.002
    max_rtc_prefix_rotation_error_rad: float = 0.02
    max_rtc_prefix_gripper_error_deg: float = 0.5
    # Optional deployment workspace guard. The policy intentionally receives
    # relative state and therefore has no kinematic knowledge of the table
    # plane. Production enables both an absolute floor and a per-ACTIVE
    # downward-excursion limit; unit-level scheduler users may leave them off.
    min_tcp_z_m: float | None = None
    max_downward_excursion_m: float | None = None

    @classmethod
    def for_policy_rate(
        cls,
        control_hz: float,
        action_horizon: int,
        *,
        min_tcp_z_m: float | None = None,
        max_downward_excursion_m: float | None = None,
        complete_chunk_before_replan: bool = False,
        max_physical_gripper_speed_deg_s: float | None = None,
    ) -> "SchedulerConfig":
        """Build a duration-equivalent scheduler for a supported policy rate."""

        control_hz = float(control_hz)
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if math.isclose(control_hz, 25.0):
            max_translation_step_m = 0.035
            max_rotation_step_rad = 0.12
        elif math.isclose(control_hz, 12.5):
            # Full-dataset 12.5 Hz adjacent-action 99.9th percentiles are
            # 39.0 mm and 0.125 rad. Keep measured headroom while rejecting
            # the 64-74 mm / 0.33-0.46 rad conversion outliers.
            max_translation_step_m = 0.045
            max_rotation_step_rad = 0.14
        else:
            raise ValueError(f"Unsupported Task487 policy rate: {control_hz:g} Hz")

        commit_steps = max(1, round(control_hz * 0.4))
        request_every_steps = max(1, math.floor(control_hz * 0.2 + 1e-9))
        replan_remaining_steps = max(0, commit_steps - request_every_steps)
        handoff_blend_steps = max(1, math.ceil(control_hz * 0.2 - 1e-9))
        return cls(
            control_hz=control_hz,
            request_every_steps=request_every_steps,
            replan_remaining_steps=replan_remaining_steps,
            commit_steps=commit_steps,
            commit_window_s=(0.4 if math.isclose(control_hz, 12.5) else None),
            max_queue_steps=commit_steps + action_horizon,
            handoff_blend_steps=handoff_blend_steps,
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
            min_tcp_z_m=min_tcp_z_m,
            max_downward_excursion_m=max_downward_excursion_m,
            complete_chunk_before_replan=complete_chunk_before_replan,
            max_physical_gripper_speed_deg_s=max_physical_gripper_speed_deg_s,
        )

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class MergeStats:
    accepted: int
    expired: int
    rtc_prefix_preserved: int
    inference_delay_steps: int
    unsafe_truncated: int
    speed_retimed: int
    max_retime_delay_s: float
    queue_truncated: int
    handoff_blended: int


@dataclass(frozen=True)
class ScheduledAction:
    target: np.ndarray
    target_time: float
    nominal_time: float
    gripper_policy_target: np.ndarray | None = None


@dataclass(frozen=True)
class SchedulerDiagnosticSnapshot:
    """Read-only copy of the rolling timeline for asynchronous diagnostics."""

    targets: np.ndarray
    target_times: np.ndarray
    nominal_times: np.ndarray
    sent: np.ndarray
    request_prefix_targets: np.ndarray
    last_scheduled_target: np.ndarray
    last_scheduled_time: float


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float((Rotation.from_rotvec(left).inv() * Rotation.from_rotvec(right)).magnitude())


class RollingScheduler:
    """Maintain one executable timeline shared with the low-level limiter.

    ``_nominal_times`` retain the policy-rate horizon solely for aligning
    overlapping chunks. ``_times`` are the speed-limited wall-clock completion
    times sent to the physical controller. Sent future points stay in the
    queue until their executable time passes, so inference cadence follows
    physical progress rather than command enqueue rate.
    """

    def __init__(self, config: SchedulerConfig = SchedulerConfig()):
        self.config = config
        if config.max_physical_translation_speed_m_s <= 0:
            raise ValueError("max physical translation speed must be positive")
        if config.max_physical_rotation_speed_rad_s <= 0:
            raise ValueError("max physical rotation speed must be positive")
        if config.max_physical_gripper_speed_deg_s is not None and (
            not np.isfinite(config.max_physical_gripper_speed_deg_s)
            or config.max_physical_gripper_speed_deg_s <= 0
        ):
            raise ValueError("max physical gripper speed must be finite and positive")
        if config.request_every_steps <= 0:
            raise ValueError("request_every_steps must be positive")
        if config.replan_remaining_steps < 0:
            raise ValueError("replan_remaining_steps must not be negative")
        if config.replan_remaining_steps > config.commit_steps:
            raise ValueError("replan_remaining_steps must not exceed commit_steps")
        if config.max_queue_steps < config.commit_steps:
            raise ValueError("max_queue_steps must cover commit_steps")
        if config.commit_window_s is not None and config.commit_window_s <= 0:
            raise ValueError("commit window must be positive")
        if config.handoff_blend_steps < 0:
            raise ValueError("handoff blend steps must not be negative")
        if config.min_tcp_z_m is not None and not np.isfinite(config.min_tcp_z_m):
            raise ValueError("minimum TCP z must be finite")
        if (
            config.max_downward_excursion_m is not None
            and config.max_downward_excursion_m <= 0
        ):
            raise ValueError("maximum downward excursion must be positive")
        self.state = RunState.HOLD
        self.reason = "startup"
        self._queue = np.empty((0, 14), dtype=np.float64)
        self._times = np.empty((0,), dtype=np.float64)
        self._nominal_times = np.empty((0,), dtype=np.float64)
        self._sent = np.empty((0,), dtype=bool)
        self._request_pending = False
        self._steps_since_request = 0
        self._has_executed = False
        self._total_completed = 0
        self._request_start_completed = 0
        self._last_requested_target_time: float | None = None
        self._request_prefix = np.empty((0, 14), dtype=np.float64)
        self._last_scheduled_target: np.ndarray | None = None
        self._last_scheduled_time: float | None = None
        self._workspace_min_z: np.ndarray | None = None

    @property
    def queued_steps(self) -> int:
        return len(self._queue)

    @property
    def committed_steps(self) -> int:
        return int(self._sent.sum())

    @property
    def request_pending(self) -> bool:
        return self._request_pending

    def diagnostic_snapshot(self) -> SchedulerDiagnosticSnapshot:
        """Return copies of all state needed to audit a chunk handoff.

        The returned arrays never alias scheduler state, so diagnostic I/O can
        safely retain them while the control loop continues to mutate its
        queue.  This method is intentionally observational and does not change
        request cadence, prefix selection, or waypoint dispatch.
        """
        last_target = (
            np.empty((0,), dtype=np.float64)
            if self._last_scheduled_target is None
            else self._last_scheduled_target.copy()
        )
        return SchedulerDiagnosticSnapshot(
            targets=self._queue.copy(),
            target_times=self._times.copy(),
            nominal_times=self._nominal_times.copy(),
            sent=self._sent.copy(),
            request_prefix_targets=self._request_prefix.copy(),
            last_scheduled_target=last_target,
            last_scheduled_time=(
                float("nan")
                if self._last_scheduled_time is None
                else float(self._last_scheduled_time)
            ),
        )

    def rtc_prefix_targets(self) -> np.ndarray:
        """Return only actions already committed to the low-level controller.

        Unsent queue entries are replaceable old-chunk tail, not RTC context.
        Prefilling them forced every replan to preserve actions 5..9 that had
        never been dispatched and made the robot repeatedly execute the much
        less stable 10..14 suffix of each newly sampled horizon.
        """
        if self.state is not RunState.ACTIVE or not len(self._queue):
            return np.empty((0, 14), dtype=np.float64)
        count = 0
        while count < len(self._sent) and self._sent[count]:
            count += 1
        return self._queue[:count].copy()

    def _clear_timeline(self) -> None:
        self._queue = np.empty((0, 14), dtype=np.float64)
        self._times = np.empty((0,), dtype=np.float64)
        self._nominal_times = np.empty((0,), dtype=np.float64)
        self._sent = np.empty((0,), dtype=bool)
        self._last_scheduled_target = None
        self._last_scheduled_time = None
        self._request_prefix = np.empty((0, 14), dtype=np.float64)

    @property
    def workspace_min_z(self) -> np.ndarray | None:
        return None if self._workspace_min_z is None else self._workspace_min_z.copy()

    def activate(self, live_target: np.ndarray | None = None) -> None:
        self.state = RunState.ACTIVE
        self.reason = ""
        self._clear_timeline()
        self._request_pending = False
        self._last_requested_target_time = None
        self._steps_since_request = self.config.request_every_steps
        self._has_executed = False
        self._total_completed = 0
        self._request_start_completed = 0
        static_floor = self.config.min_tcp_z_m
        downward_limit = self.config.max_downward_excursion_m
        if live_target is not None:
            live_target = np.asarray(live_target, dtype=np.float64)
            if live_target.shape != (14,) or not np.isfinite(live_target).all():
                raise ValueError(f"Invalid activation target: {live_target.shape}")
        if downward_limit is not None:
            if live_target is None:
                raise ValueError("live target is required for the downward-excursion guard")
            floors = live_target[[2, 9]] - downward_limit
            if static_floor is not None:
                floors = np.maximum(floors, static_floor)
            self._workspace_min_z = floors
        elif static_floor is not None:
            self._workspace_min_z = np.full(2, static_floor, dtype=np.float64)
        else:
            self._workspace_min_z = None

    def hold(self, reason: str) -> None:
        self.state = RunState.HOLD
        self.reason = reason
        self._clear_timeline()
        self._request_pending = False
        self._last_requested_target_time = None
        self._workspace_min_z = None

    def request_due(self, now: float | None = None) -> bool:
        if self.config.complete_chunk_before_replan:
            if self.state is not RunState.ACTIVE or self._request_pending:
                return False
            if not len(self._queue):
                return True
            if not self._sent.all():
                return False
            # The whole remaining suffix is now immutable and fits the RTC
            # prefix. Inference may overlap its final execution, not replace
            # the still-undispatched tail. HOLD always clears this timeline.
            if now is None:
                return len(self._queue) <= self.config.replan_remaining_steps
            return (
                float(self._times[-1]) - float(now)
                <= self.config.replan_remaining_steps * self.config.dt + 1e-9
            )
        committed = self.committed_steps
        executable_prefix_due = False
        if now is not None and committed and self.config.replan_remaining_steps:
            committed_end_time = float(self._times[committed - 1])
            already_requested = (
                self._last_requested_target_time is not None
                and committed_end_time <= self._last_requested_target_time + 1e-9
            )
            replan_lead_s = self.config.replan_remaining_steps * self.config.dt
            executable_prefix_due = (
                not already_requested
                and committed_end_time - float(now) <= replan_lead_s + 1e-9
            )
        return (
            self.state is RunState.ACTIVE
            and not self._request_pending
            and (
                executable_prefix_due
                or self._steps_since_request >= self.config.request_every_steps
                or (
                    self._steps_since_request > 0
                    and 0 < len(self._queue) <= self.config.replan_remaining_steps
                )
            )
        )

    def mark_request_started(self, now: float | None = None) -> None:
        if not self.request_due(now):
            raise RuntimeError("Inference request is not due")
        self._request_pending = True
        self._steps_since_request = 0
        self._request_start_completed = self._total_completed
        self._request_prefix = self.rtc_prefix_targets()
        committed = self.committed_steps
        self._last_requested_target_time = (
            None if not committed else float(self._times[committed - 1])
        )

    def mark_request_failed(self, reason: str) -> None:
        self.hold(reason)

    def _validate(self, targets: np.ndarray, live_target: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        live_target = np.asarray(live_target, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 14 or live_target.shape != (14,):
            raise UnsafeChunkError(f"Invalid target shapes: {targets.shape}, {live_target.shape}")
        if not np.isfinite(targets).all() or not np.isfinite(live_target).all():
            raise UnsafeChunkError("Target contains NaN or infinity")
        previous = live_target
        for index, target in enumerate(targets):
            translation_limit = (
                self.config.max_translation_tracking_error_m
                if index == 0
                else self.config.max_translation_step_m
            )
            rotation_limit = (
                self.config.max_rotation_tracking_error_rad
                if index == 0
                else self.config.max_rotation_step_rad
            )
            comparison = "tracking error" if index == 0 else "translation step"
            rotation_comparison = "rotation tracking error" if index == 0 else "rotation step"
            for arm_offset, arm_name in ((0, "right"), (7, "left")):
                arm_index = 0 if arm_offset == 0 else 1
                if (
                    self._workspace_min_z is not None
                    and target[arm_offset + 2] < self._workspace_min_z[arm_index]
                ):
                    raise UnsafeChunkError(
                        f"action[{index}] {arm_name} TCP z {target[arm_offset + 2]:.6f}m "
                        f"is below workspace floor {self._workspace_min_z[arm_index]:.6f}m",
                        action_index=index,
                    )
                translation = np.linalg.norm(
                    target[arm_offset : arm_offset + 3] - previous[arm_offset : arm_offset + 3]
                )
                rotation = _rotation_distance(
                    previous[arm_offset + 3 : arm_offset + 6],
                    target[arm_offset + 3 : arm_offset + 6],
                )
                if translation > translation_limit:
                    raise UnsafeChunkError(
                        f"action[{index}] {arm_name} {comparison} {translation:.6f}m exceeds "
                        f"{translation_limit:.6f}m",
                        action_index=index,
                    )
                if rotation > rotation_limit:
                    raise UnsafeChunkError(
                        f"action[{index}] {arm_name} {rotation_comparison} {rotation:.6f}rad exceeds "
                        f"{rotation_limit:.6f}rad",
                        action_index=index,
                    )
            previous = target

    def _minimum_duration(self, left: np.ndarray, right: np.ndarray, *, include_gripper=True) -> float:
        duration = 0.0
        for arm_offset in (0, 7):
            translation = np.linalg.norm(
                right[arm_offset : arm_offset + 3] - left[arm_offset : arm_offset + 3]
            )
            rotation = _rotation_distance(
                left[arm_offset + 3 : arm_offset + 6],
                right[arm_offset + 3 : arm_offset + 6],
            )
            duration = max(
                duration,
                translation / self.config.max_physical_translation_speed_m_s,
                rotation / self.config.max_physical_rotation_speed_rad_s,
            )
        if include_gripper:
            duration = max(duration, self._gripper_duration(left, right))
        return float(duration)

    def _gripper_duration(self, left: np.ndarray, right: np.ndarray) -> float:
        speed = self.config.max_physical_gripper_speed_deg_s
        if speed is None:
            return 0.0
        return float(np.max(np.abs(right[[6, 13]] - left[[6, 13]])) / speed)

    def _retime(
        self,
        targets: np.ndarray,
        nominal_times: np.ndarray,
        start_target: np.ndarray,
        start_time: float,
    ) -> np.ndarray:
        result = np.empty(len(targets), dtype=np.float64)
        previous_target = np.asarray(start_target, dtype=np.float64)
        previous_time = float(start_time)
        for index, (target, nominal_time) in enumerate(zip(targets, nominal_times)):
            minimum_time = previous_time + self._minimum_duration(previous_target, target)
            result[index] = max(float(nominal_time), minimum_time)
            previous_target = target
            previous_time = result[index]
        return result

    def advance(self, now: float, controller_target: np.ndarray) -> int:
        """Retire due waypoints against the low-level controller setpoint.

        RTC aligns overlapping command chunks, so progress is the setpoint
        timeline consumed by ``RosTargetInterpolationController``. The physical
        EEF can legitimately lag this timeline during Mink's anchor release;
        comparing a due command against ``ActualTCPPose`` shifts the RTC action
        index and can cause a false HOLD.
        """
        if self.state is not RunState.ACTIVE or not len(self._queue):
            return 0
        now = float(now)
        due = self._sent & (self._times <= now)
        completed = 0
        while completed < len(due) and due[completed]:
            completed += 1
        if not completed:
            return 0
        try:
            self._validate(self._queue[completed - 1 : completed], controller_target)
        except UnsafeChunkError as exc:
            self.hold(str(exc))
            raise
        self._queue = self._queue[completed:]
        self._times = self._times[completed:]
        self._nominal_times = self._nominal_times[completed:]
        self._sent = self._sent[completed:]
        self._steps_since_request += completed
        self._total_completed += completed
        self._has_executed = True
        return completed

    def validate_physical_tracking(
        self,
        actual_target: np.ndarray,
        controller_target: np.ndarray,
    ) -> None:
        """Fail closed if Marvin does not follow the emitted setpoint.

        This check is deliberately separate from ``advance``: RTC command
        progress follows the controller timeline, while hardware health must
        compare that timeline with independent physical feedback.
        """
        if self.state is not RunState.ACTIVE:
            return
        try:
            self._validate(
                np.asarray(controller_target, dtype=np.float64)[None, :],
                np.asarray(actual_target, dtype=np.float64),
            )
        except UnsafeChunkError as exc:
            self.hold(str(exc))
            raise

    def _validate_rtc_prefix(self, predicted: np.ndarray, expected: np.ndarray) -> None:
        if len(predicted) != len(expected):
            raise UnsafeChunkError(
                f"RTC prefix length mismatch: predicted={len(predicted)} expected={len(expected)}"
            )
        for index, (left, right) in enumerate(zip(predicted, expected)):
            for arm_offset, arm_name in ((0, "right"), (7, "left")):
                translation = np.linalg.norm(
                    left[arm_offset : arm_offset + 3] - right[arm_offset : arm_offset + 3]
                )
                rotation = _rotation_distance(
                    left[arm_offset + 3 : arm_offset + 6],
                    right[arm_offset + 3 : arm_offset + 6],
                )
                gripper = abs(float(left[arm_offset + 6] - right[arm_offset + 6]))
                if translation > self.config.max_rtc_prefix_translation_error_m:
                    raise UnsafeChunkError(
                        f"RTC prefix[{index}] {arm_name} translation mismatch "
                        f"{translation:.6f}m"
                    )
                if rotation > self.config.max_rtc_prefix_rotation_error_rad:
                    raise UnsafeChunkError(
                        f"RTC prefix[{index}] {arm_name} rotation mismatch {rotation:.6f}rad"
                    )
                if gripper > self.config.max_rtc_prefix_gripper_error_deg:
                    raise UnsafeChunkError(
                        f"RTC prefix[{index}] {arm_name} gripper mismatch {gripper:.3f}deg"
                    )

    @staticmethod
    def _blend_chunk_handoff(
        old_tail: np.ndarray,
        new_suffix: np.ndarray,
        blend_steps: int,
    ) -> int:
        """Fade from the old safe tail into a newly sampled RTC suffix in place."""
        count = min(max(0, int(blend_steps)), len(old_tail), len(new_suffix))
        for index in range(count):
            # Do not reach weight=1 inside the blend window: the following
            # unmodified suffix sample completes the transition naturally.
            weight = float(index + 1) / float(count + 1)
            for arm_offset in (0, 7):
                old_pose = old_tail[index, arm_offset : arm_offset + 6]
                new_pose = new_suffix[index, arm_offset : arm_offset + 6]
                new_suffix[index, arm_offset : arm_offset + 3] = (
                    (1.0 - weight) * old_pose[:3] + weight * new_pose[:3]
                )
                old_rotation = Rotation.from_rotvec(old_pose[3:6])
                new_rotation = Rotation.from_rotvec(new_pose[3:6])
                delta = old_rotation.inv() * new_rotation
                new_suffix[index, arm_offset + 3 : arm_offset + 6] = (
                    old_rotation * Rotation.from_rotvec(delta.as_rotvec() * weight)
                ).as_rotvec()
                gripper_index = arm_offset + 6
                new_suffix[index, gripper_index] = (
                    (1.0 - weight) * old_tail[index, gripper_index]
                    + weight * new_suffix[index, gripper_index]
                )
        return count

    def merge_chunk(
        self,
        targets: np.ndarray,
        observation_time: float,
        now: float,
        live_target: np.ndarray,
    ) -> MergeStats:
        if self.state is not RunState.ACTIVE or not self._request_pending:
            raise RuntimeError("Received a chunk without an active inference request")
        targets = np.asarray(targets, dtype=np.float64)
        live_target = np.asarray(live_target, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 14:
            raise UnsafeChunkError(f"Expected chunk shape (H, 14), got {targets.shape}")
        if not np.isfinite(targets).all():
            self.hold("inference chunk contains NaN or infinity")
            raise UnsafeChunkError(self.reason)
        nominal_times = observation_time + (np.arange(len(targets)) + 1) * self.config.dt
        inference_delay_steps = max(0, self._total_completed - self._request_start_completed)
        requested_prefix_len = len(self._request_prefix)

        if requested_prefix_len:
            # The model received _request_prefix as a hard action prefill. Any
            # part consumed while inference was running is dropped; the still
            # pending part must be reproduced exactly and remains immutable.
            preserved_count = max(0, requested_prefix_len - inference_delay_steps)
            preserved_count = min(preserved_count, len(self._queue))
            predicted_start = min(inference_delay_steps, requested_prefix_len)
            predicted_prefix = targets[predicted_start : predicted_start + preserved_count]
            expected_prefix = self._request_prefix[
                predicted_start : predicted_start + preserved_count
            ]
            try:
                self._validate_rtc_prefix(predicted_prefix, expected_prefix)
            except UnsafeChunkError as exc:
                self.hold(str(exc))
                raise

            old_prefix = self._queue[:preserved_count].copy()
            old_prefix_times = self._times[:preserved_count].copy()
            old_prefix_nominal = self._nominal_times[:preserved_count].copy()
            old_prefix_sent = self._sent[:preserved_count].copy()
            old_replaceable_tail = self._queue[preserved_count:].copy()
            generated_start = min(len(targets), max(requested_prefix_len, inference_delay_steps))
            new_targets = targets[generated_start:].copy()
            new_nominal = nominal_times[generated_start:].copy()
            blend_reference = old_replaceable_tail
            if not len(blend_reference) and len(old_prefix) and len(new_targets):
                # Full-chunk execution intentionally leaves no replaceable old
                # tail.  Still ease the generated suffix away from the final
                # immutable RTC target; otherwise the prefix->suffix seam gets
                # blend=0 and a harmless ~35 mm boundary step can trip the
                # outlier guard.  Repeating the anchor gives a five-step
                # position/rotation/gripper ramp without changing the prefix.
                blend_reference = np.repeat(
                    old_prefix[-1:],
                    min(self.config.handoff_blend_steps, len(new_targets)),
                    axis=0,
                )
            handoff_blended = self._blend_chunk_handoff(
                blend_reference,
                new_targets,
                self.config.handoff_blend_steps,
            )
            expired = generated_start
        else:
            valid = nominal_times > now + self.config.dispatch_lead_s
            expired = int((~valid).sum())
            old_prefix = np.empty((0, 14), dtype=np.float64)
            old_prefix_times = np.empty((0,), dtype=np.float64)
            old_prefix_nominal = np.empty((0,), dtype=np.float64)
            old_prefix_sent = np.empty((0,), dtype=bool)
            new_targets = targets[valid].copy()
            new_nominal = nominal_times[valid].copy()
            preserved_count = 0
            handoff_blended = 0

        if not len(old_prefix) and not len(new_targets):
            self.hold("inference chunk expired before arrival")
            raise UnsafeChunkError(self.reason)

        merged = np.concatenate((old_prefix, new_targets), axis=0)
        merged_nominal_times = np.concatenate((old_prefix_nominal, new_nominal))
        merged_sent = np.concatenate((old_prefix_sent, np.zeros(len(new_targets), dtype=bool)))
        expected_merged_count = len(merged)
        required_prefix = min(
            len(merged),
            # Only the RTC prefix and the refill reserve are indispensable.
            # A bad remote tail can be truncated and replenished before queue
            # exhaustion; requiring the whole generated horizon to be valid
            # would turn a harmless uncommitted far-tail outlier into HOLD.
            max(preserved_count, self.config.replan_remaining_steps),
        )
        # A preserved RTC prefix is an already validated, immutable controller
        # trajectory. The physical EEF may lag it while Mink releases/follows
        # its anchor, so rechecking that old prefix against ActualTCPPose would
        # be a false tracking failure. Still validate every adjacent action,
        # including the critical prefix->generated-suffix seam. A fresh chunk
        # with no preserved prefix remains anchored to live physical feedback.
        validation_anchor = merged[0] if preserved_count else live_target
        try:
            self._validate(merged, validation_anchor)
        except UnsafeChunkError as exc:
            if exc.action_index is None or exc.action_index < required_prefix:
                self.hold(str(exc))
                raise
            merged = merged[: exc.action_index]
            merged_nominal_times = merged_nominal_times[: exc.action_index]
            merged_sent = merged_sent[: exc.action_index]

        unsafe_truncated = expected_merged_count - len(merged)
        queue_truncated = max(0, len(merged) - self.config.max_queue_steps)
        if queue_truncated:
            merged = merged[: self.config.max_queue_steps]
            merged_nominal_times = merged_nominal_times[: self.config.max_queue_steps]
            merged_sent = merged_sent[: self.config.max_queue_steps]

        prefix_len = min(preserved_count, len(merged))
        if prefix_len:
            merged_times = old_prefix_times[:prefix_len].copy()
            start_target = merged[prefix_len - 1]
            start_time = merged_times[-1]
        else:
            if self._last_scheduled_target is None:
                start_target = live_target
                start_time = now + self.config.dispatch_lead_s
            else:
                start_target = self._last_scheduled_target
                start_time = max(
                    float(self._last_scheduled_time),
                    now + self.config.dispatch_lead_s,
                )
            merged_times = np.empty((0,), dtype=np.float64)
        if prefix_len < len(merged):
            suffix_times = self._retime(
                merged[prefix_len:],
                merged_nominal_times[prefix_len:],
                start_target,
                start_time,
            )
            merged_times = np.concatenate((merged_times, suffix_times))

        self._queue = merged
        self._times = merged_times
        self._nominal_times = merged_nominal_times
        self._sent = merged_sent
        self._request_pending = False
        self._request_prefix = np.empty((0, 14), dtype=np.float64)
        delays = np.maximum(0.0, merged_times - merged_nominal_times)
        return MergeStats(
            accepted=len(merged),
            expired=expired,
            rtc_prefix_preserved=prefix_len,
            inference_delay_steps=inference_delay_steps,
            unsafe_truncated=unsafe_truncated,
            speed_retimed=int(np.count_nonzero(delays > 1e-9)),
            max_retime_delay_s=float(delays.max(initial=0.0)),
            queue_truncated=queue_truncated,
            handoff_blended=handoff_blended,
        )

    def _retime_unsent_suffix(self, now: float, live_target: np.ndarray) -> None:
        unsent_indices = np.flatnonzero(~self._sent)
        if not len(unsent_indices):
            return
        first = int(unsent_indices[0])
        if self._times[first] > now:
            return
        if first:
            start_target = self._queue[first - 1]
            start_time = max(self._times[first - 1], now + self.config.dispatch_lead_s)
        elif self._last_scheduled_target is not None:
            start_target = self._last_scheduled_target
            start_time = max(
                float(self._last_scheduled_time),
                now + self.config.dispatch_lead_s,
            )
        else:
            start_target = live_target
            start_time = now + self.config.dispatch_lead_s
        self._times[first:] = self._retime(
            self._queue[first:],
            self._nominal_times[first:],
            start_target,
            start_time,
        )

    def pop_batch(
        self,
        live_target: np.ndarray,
        now: float | None = None,
        max_actions: int | None = None,
    ) -> list[ScheduledAction]:
        """Fill the committed low-level window in one atomic scheduler step.

        The client control loop runs at the same rate as the policy timebase.
        Dispatching only one waypoint per tick therefore cannot build the
        committed window: after the early near-stationary actions have
        completed, inference can consume the final sent action and leave an
        empty RTC prefix.  Replanning then repeatedly replaces the meaningful
        later part of each chunk.  Return every currently committable waypoint
        so ``UmiEnv.exec_actions`` can enqueue them as one batch.
        """
        if self.state is not RunState.ACTIVE:
            return []
        if len(self._queue) == 0:
            if self._request_pending:
                return []
            self.hold("action queue exhausted")
            return []
        # Everything in the RTC request prefix is already immutable. While
        # inference is running, do not preload additional old-tail points that
        # the model has never seen and therefore cannot join smoothly.
        if self._request_pending:
            return []
        if now is not None:
            self._retime_unsent_suffix(float(now), np.asarray(live_target, dtype=np.float64))
        # Never preload replaceable points. Keeping only the committed prefix
        # in the physical controller prevents a later chunk from trimming an
        # already-scheduled trajectory that was meant to remain immutable.
        commit_limit = min(self.config.commit_steps, len(self._queue))
        if self.config.commit_window_s is not None and now is not None:
            cutoff = float(now) + self.config.commit_window_s
            within_window = int(
                np.searchsorted(self._times, cutoff + 1e-12, side="right")
            )
            # A long but safety-retimed first segment must still make progress.
            physical_limit = max(1, within_window)
            # Never reinterpret an already dispatched prefix as replaceable.
            physical_limit = max(self.committed_steps, physical_limit)
            commit_limit = min(commit_limit, physical_limit)
        available = commit_limit - int(self._sent.sum())
        if max_actions is not None:
            max_actions = int(max_actions)
            if max_actions <= 0:
                raise ValueError("max_actions must be positive")
            available = min(available, max_actions)
        if available <= 0:
            return []
        unsent_indices = np.flatnonzero(~self._sent)
        if not len(unsent_indices):
            return []
        indices = unsent_indices[:available]
        actions = [
            ScheduledAction(
                self._queue[index].copy(),
                float(self._times[index]),
                float(self._nominal_times[index]),
            )
            for index in indices
        ]
        self._sent[indices] = True
        self._last_scheduled_target = actions[-1].target.copy()
        self._last_scheduled_time = actions[-1].target_time
        return actions

    def pop_next(self, live_target: np.ndarray, now: float | None = None) -> ScheduledAction | None:
        """Compatibility wrapper returning one committable waypoint."""
        actions = self.pop_batch(live_target, now=now, max_actions=1)
        return None if not actions else actions[0]
