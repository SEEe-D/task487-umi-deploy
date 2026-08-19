#!/usr/bin/env python3
"""Read-only Task487 gripper/action audit for a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


def _vector_column(table, name: str) -> np.ndarray:
    column = table[name].combine_chunks()
    if hasattr(column, "values"):
        values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(column), -1)
    return np.asarray(column.to_pylist(), dtype=np.float64)


def _scalar_column(table, name: str) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().to_pylist())
    return values.reshape(len(values), -1)[:, 0]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    return {f"q{100 * q:g}": float(np.quantile(values, q)) for q in QUANTILES}


def _degrees_summary(values_rad: np.ndarray) -> dict[str, object]:
    degrees = np.rad2deg(np.asarray(values_rad, dtype=np.float64))
    return {
        "mean_deg": float(np.mean(degrees)),
        "std_deg": float(np.std(degrees)),
        "quantiles_deg": _quantiles(degrees),
        "fraction_below_0deg": float(np.mean(degrees < 0.0)),
        "fraction_above_35deg": float(np.mean(degrees > 35.0)),
        "fraction_at_or_below_1deg": float(np.mean(degrees <= 1.0)),
        "fraction_at_or_below_3deg": float(np.mean(degrees <= 3.0)),
        "fraction_at_or_above_30deg": float(np.mean(degrees >= 30.0)),
        "fraction_at_or_above_34deg": float(np.mean(degrees >= 34.0)),
    }


def _task_names(root: Path) -> dict[int, str]:
    path = root / "meta/tasks.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    if "task_index" not in table.column_names:
        return {}
    text_key = next(
        (
            key
            for key in ("task", "task_name", "instruction", "__index_level_0__")
            if key in table.column_names
        ),
        None,
    )
    if text_key is None:
        return {}
    indices = _scalar_column(table, "task_index")
    texts = table[text_key].combine_chunks().to_pylist()
    return {int(index): str(text) for index, text in zip(indices, texts, strict=True)}


def _episode_slices(episode_index: np.ndarray) -> list[slice]:
    if not len(episode_index):
        return []
    boundaries = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(episode_index)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _side_audit(values_rad: np.ndarray, episodes: list[slice], horizon: int) -> dict[str, object]:
    values_deg = np.rad2deg(values_rad)
    starts = np.asarray([values_deg[item.start] for item in episodes])
    ends = np.asarray([values_deg[item.stop - 1] for item in episodes])
    deltas = np.concatenate([np.diff(values_deg[item]) for item in episodes if item.stop - item.start > 1])
    ranges = np.asarray([np.ptp(values_deg[item]) for item in episodes])
    net = ends - starts

    horizon_ranges = []
    horizon_net = []
    horizon_close = []
    horizon_open = []
    for item in episodes:
        episode = values_deg[item]
        if len(episode) < horizon:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(episode, horizon)
        horizon_ranges.append(np.ptp(windows, axis=1))
        horizon_net.append(windows[:, -1] - windows[:, 0])
        horizon_close.append(windows[:, 0] - np.min(windows, axis=1))
        horizon_open.append(np.max(windows, axis=1) - windows[:, 0])
    h_range = np.concatenate(horizon_ranges) if horizon_ranges else np.empty(0)
    h_net = np.concatenate(horizon_net) if horizon_net else np.empty(0)
    h_close = np.concatenate(horizon_close) if horizon_close else np.empty(0)
    h_open = np.concatenate(horizon_open) if horizon_open else np.empty(0)

    return {
        "values": _degrees_summary(values_rad),
        "episode_start_deg": _quantiles(starts),
        "episode_end_deg": _quantiles(ends),
        "one_step_delta_deg": {
            "mean": float(np.mean(deltas)),
            "abs_quantiles": _quantiles(np.abs(deltas)),
            "fraction_abs_below_0_1deg": float(np.mean(np.abs(deltas) < 0.1)),
            "fraction_abs_above_0_25deg": float(np.mean(np.abs(deltas) > 0.25)),
            "fraction_abs_above_1deg": float(np.mean(np.abs(deltas) > 1.0)),
            "fraction_abs_above_3deg": float(np.mean(np.abs(deltas) > 3.0)),
            "fraction_opening_above_0_25deg": float(np.mean(deltas > 0.25)),
            "fraction_closing_above_0_25deg": float(np.mean(deltas < -0.25)),
        },
        "per_episode_range_deg": {
            "quantiles": _quantiles(ranges),
            "fraction_above_5deg": float(np.mean(ranges > 5.0)),
            "fraction_above_10deg": float(np.mean(ranges > 10.0)),
            "fraction_above_20deg": float(np.mean(ranges > 20.0)),
        },
        "per_episode_net_change_deg": _quantiles(net),
        f"horizon_{horizon}_steps": {
            "window_count": int(len(h_range)),
            "range_deg_quantiles": _quantiles(h_range),
            "net_change_deg_quantiles": _quantiles(h_net),
            "fraction_range_above_1deg": float(np.mean(h_range > 1.0)),
            "fraction_range_above_5deg": float(np.mean(h_range > 5.0)),
            "fraction_range_above_10deg": float(np.mean(h_range > 10.0)),
            "fraction_closes_above_5deg": float(np.mean(h_close > 5.0)),
            "fraction_opens_above_5deg": float(np.mean(h_open > 5.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    root = args.dataset.resolve()
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    states, actions, episode_indices, frame_indices, task_indices = [], [], [], [], []
    for path in files:
        schema_names = pq.read_schema(path).names
        columns = ["observation.state", "action", "episode_index", "frame_index"]
        if "task_index" in schema_names:
            columns.append("task_index")
        table = pq.read_table(path, columns=columns)
        states.append(_vector_column(table, "observation.state"))
        actions.append(_vector_column(table, "action"))
        episode_indices.append(_scalar_column(table, "episode_index").astype(np.int64))
        frame_indices.append(_scalar_column(table, "frame_index").astype(np.int64))
        if "task_index" in table.column_names:
            task_indices.append(_scalar_column(table, "task_index").astype(np.int64))

    state = np.concatenate(states)
    action = np.concatenate(actions)
    episode_index = np.concatenate(episode_indices)
    frame_index = np.concatenate(frame_indices)
    task_index = np.concatenate(task_indices) if task_indices else None
    order = np.lexsort((frame_index, episode_index))
    state, action = state[order], action[order]
    episode_index, frame_index = episode_index[order], frame_index[order]
    if task_index is not None:
        task_index = task_index[order]
    episodes = _episode_slices(episode_index)

    next_rows = np.arange(len(state) - 1)
    next_rows = next_rows[episode_index[next_rows] == episode_index[next_rows + 1]]
    next_state_error = np.abs(action[next_rows] - state[next_rows + 1])
    last_rows = np.asarray([item.stop - 1 for item in episodes], dtype=np.int64)
    terminal_error = np.abs(action[last_rows] - state[last_rows])

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    report: dict[str, object] = {
        "read_only": True,
        "dataset": str(root),
        "metadata": {
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "declared_total_episodes": info.get("total_episodes"),
            "declared_total_frames": info.get("total_frames"),
            "scanned_parquet_files": len(files),
            "scanned_rows": int(len(state)),
            "scanned_episodes": int(len(episodes)),
        },
        "layout": {"state_dim": int(state.shape[1]), "action_dim": int(action.shape[1])},
        "action_next_state_contract": {
            "compared_nonterminal_rows": int(len(next_rows)),
            "max_abs_error_all_dims": float(np.max(next_state_error)),
            "max_abs_error_right_gripper_rad": float(np.max(next_state_error[:, 9])),
            "max_abs_error_left_gripper_rad": float(np.max(next_state_error[:, 19])),
            "max_terminal_hold_error_all_dims": float(np.max(terminal_error)),
        },
        "right": _side_audit(state[:, 9], episodes, args.horizon),
        "left": _side_audit(state[:, 19], episodes, args.horizon),
    }

    if task_index is not None:
        names = _task_names(root)
        by_task = {}
        for index in np.unique(task_index):
            mask = task_index == index
            by_task[str(int(index))] = {
                "name": names.get(int(index), ""),
                "rows": int(np.sum(mask)),
                "right": _degrees_summary(state[mask, 9]),
                "left": _degrees_summary(state[mask, 19]),
            }
        report["by_task"] = by_task

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
