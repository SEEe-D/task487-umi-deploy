"""Structured per-chunk diagnostics for Task487 RTC handoffs."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from task487_runtime.scheduler import MergeStats, SchedulerDiagnosticSnapshot


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _direction_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return _json_float(float(np.dot(left, right) / denominator))


def _first_motion(anchor: np.ndarray, targets: np.ndarray, arm_offset: int) -> np.ndarray:
    if not len(targets):
        return np.empty((0,), dtype=np.float64)
    return (
        np.asarray(targets[0, arm_offset : arm_offset + 3], dtype=np.float64)
        - np.asarray(anchor[arm_offset : arm_offset + 3], dtype=np.float64)
    )


def _first_rotation_deg(anchor: np.ndarray, targets: np.ndarray, arm_offset: int) -> float | None:
    if not len(targets):
        return None
    anchor_rotation = Rotation.from_rotvec(anchor[arm_offset + 3 : arm_offset + 6])
    target_rotation = Rotation.from_rotvec(targets[0, arm_offset + 3 : arm_offset + 6])
    return _json_float(np.rad2deg((anchor_rotation.inv() * target_rotation).magnitude()))


def _vector_mm(value: np.ndarray) -> list[float] | None:
    if value.shape != (3,):
        return None
    return [float(component * 1000.0) for component in value]


def _path_reversals(anchor: np.ndarray, targets: np.ndarray, arm_offset: int) -> dict[str, int]:
    if not len(targets):
        return {"compared": 0, "negative": 0, "strong": 0}
    points = np.concatenate(
        (
            anchor[None, arm_offset : arm_offset + 3],
            targets[:, arm_offset : arm_offset + 3],
        ),
        axis=0,
    )
    deltas = np.diff(points, axis=0)
    magnitudes = np.linalg.norm(deltas, axis=1)
    compared = negative = strong = 0
    for index in range(1, len(deltas)):
        if magnitudes[index - 1] < 0.001 or magnitudes[index] < 0.001:
            continue
        cosine = float(
            np.dot(deltas[index - 1], deltas[index])
            / (magnitudes[index - 1] * magnitudes[index])
        )
        compared += 1
        negative += int(cosine < 0.0)
        strong += int(cosine < -0.5)
    return {"compared": compared, "negative": negative, "strong": strong}


def _arm_summary(
    anchor: np.ndarray,
    old_tail: np.ndarray,
    raw_suffix: np.ndarray,
    blended_suffix: np.ndarray,
    arm_offset: int,
) -> dict[str, Any]:
    old_motion = _first_motion(anchor, old_tail, arm_offset)
    raw_motion = _first_motion(anchor, raw_suffix, arm_offset)
    blended_motion = _first_motion(anchor, blended_suffix, arm_offset)
    return {
        "old_first_delta_mm": _vector_mm(old_motion),
        "raw_first_delta_mm": _vector_mm(raw_motion),
        "blended_first_delta_mm": _vector_mm(blended_motion),
        "old_vs_raw_direction_cos": (
            _direction_cosine(old_motion, raw_motion)
            if old_motion.shape == raw_motion.shape == (3,)
            else None
        ),
        "old_vs_blended_direction_cos": (
            _direction_cosine(old_motion, blended_motion)
            if old_motion.shape == blended_motion.shape == (3,)
            else None
        ),
        "raw_vs_blended_direction_cos": (
            _direction_cosine(raw_motion, blended_motion)
            if raw_motion.shape == blended_motion.shape == (3,)
            else None
        ),
        "old_first_rotation_deg": _first_rotation_deg(anchor, old_tail, arm_offset),
        "raw_first_rotation_deg": _first_rotation_deg(anchor, raw_suffix, arm_offset),
        "blended_first_rotation_deg": _first_rotation_deg(
            anchor, blended_suffix, arm_offset
        ),
        "raw_path_reversals": _path_reversals(anchor, raw_suffix, arm_offset),
        "blended_path_reversals": _path_reversals(anchor, blended_suffix, arm_offset),
    }


class ChunkDiagnosticRecorder:
    """Persist exact arrays and a compact JSONL index for every policy merge."""

    def __init__(self, output_dir: str | pathlib.Path):
        self.output_dir = pathlib.Path(output_dir)
        self.chunk_dir = self.output_dir / "policy_chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "chunk_diagnostics.jsonl"
        self._sequence = 0

    def record(
        self,
        *,
        round_id: int,
        observation_time: float,
        merge_time: float,
        inference_latency_s: float,
        raw_targets: np.ndarray,
        live_target: np.ndarray,
        controller_target: np.ndarray,
        tcp_bases: np.ndarray,
        before: SchedulerDiagnosticSnapshot,
        after: SchedulerDiagnosticSnapshot,
        stats: MergeStats | None,
        error: str | None = None,
        splice_time: float | None = None,
        splice_anchor: np.ndarray | None = None,
        policy_request_id: int | None = None,
        policy_session_id: str | None = None,
        gripper_policy_targets: np.ndarray | None = None,
        gripper_handoff: dict | None = None,
    ) -> pathlib.Path:
        self._sequence += 1
        raw_targets = np.asarray(raw_targets, dtype=np.float64)
        live_target = np.asarray(live_target, dtype=np.float64)
        controller_target = np.asarray(controller_target, dtype=np.float64)
        requested_prefix_len = len(before.request_prefix_targets)
        inference_delay_steps = 0 if stats is None else stats.inference_delay_steps
        preserved_count = 0 if stats is None else stats.rtc_prefix_preserved

        if requested_prefix_len:
            generated_start = min(
                len(raw_targets), max(requested_prefix_len, inference_delay_steps)
            )
        elif stats is not None:
            generated_start = min(len(raw_targets), stats.expired)
        else:
            generated_start = 0

        raw_suffix = raw_targets[generated_start:].copy()
        old_replaceable_tail = before.targets[preserved_count:].copy()
        blended_suffix = after.targets[preserved_count:].copy()
        if splice_anchor is not None:
            anchor = np.asarray(splice_anchor, dtype=np.float64).copy()
        elif preserved_count:
            anchor = after.targets[preserved_count - 1].copy()
        elif before.last_scheduled_target.shape == (14,):
            anchor = before.last_scheduled_target.copy()
        else:
            anchor = live_target.copy()

        filename = (
            f"chunk_{self._sequence:06d}_round{int(round_id):04d}_"
            f"{float(merge_time):.6f}.npz"
        )
        array_path = self.chunk_dir / filename
        np.savez(
            array_path,
            raw_targets=raw_targets,
            raw_suffix=raw_suffix,
            request_prefix_targets=before.request_prefix_targets,
            old_replaceable_tail=old_replaceable_tail,
            blended_suffix=blended_suffix,
            live_target=live_target,
            controller_target=controller_target,
            tcp_bases=np.asarray(tcp_bases, dtype=np.float64),
            handoff_anchor=anchor,
            pre_queue_targets=before.targets,
            pre_queue_target_times=before.target_times,
            pre_queue_nominal_times=before.nominal_times,
            pre_queue_sent=before.sent,
            post_queue_targets=after.targets,
            post_queue_target_times=after.target_times,
            post_queue_nominal_times=after.nominal_times,
            post_queue_sent=after.sent,
            gripper_policy_targets=(np.empty((0, 2)) if gripper_policy_targets is None else gripper_policy_targets),
        )

        summary: dict[str, Any] = {
            "schema_version": 1,
            "sequence": self._sequence,
            "policy_request_id": policy_request_id,
            "policy_session_id": policy_session_id,
            "gripper_handoff": gripper_handoff,
            "status": "rejected" if error else "accepted",
            "error": error,
            "round_id": int(round_id),
            "observation_time": float(observation_time),
            "merge_time": float(merge_time),
            "splice_time": None if splice_time is None else float(splice_time),
            "observation_age_s": float(merge_time - observation_time),
            "inference_latency_ms": float(inference_latency_s * 1000.0),
            "array_file": str(array_path.relative_to(self.output_dir)),
            "horizon": int(len(raw_targets)),
            "requested_prefix": int(requested_prefix_len),
            "inference_delay_steps": int(inference_delay_steps),
            "preserved_prefix": int(preserved_count),
            "generated_start": int(generated_start),
            "raw_suffix_count": int(len(raw_suffix)),
            "old_replaceable_tail_count": int(len(old_replaceable_tail)),
            "blended_suffix_count": int(len(blended_suffix)),
            "accepted_count": None if stats is None else int(stats.accepted),
            "unsafe_truncated": None if stats is None else int(stats.unsafe_truncated),
            "queue_truncated": None if stats is None else int(stats.queue_truncated),
            "handoff_blended": None if stats is None else int(stats.handoff_blended),
            "max_retime_delay_s": (
                None if stats is None else float(stats.max_retime_delay_s)
            ),
            "right": _arm_summary(
                anchor, old_replaceable_tail, raw_suffix, blended_suffix, 0
            ),
            "left": _arm_summary(
                anchor, old_replaceable_tail, raw_suffix, blended_suffix, 7
            ),
        }
        with self.summary_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
        return array_path
