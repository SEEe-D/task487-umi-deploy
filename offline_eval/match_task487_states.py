#!/usr/bin/env python3
"""Match local task487 report states against a LeRobot parquet dataset."""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = json.load(open(args.report, encoding="utf-8"))
    targets = []
    for task in ("1", "2"):
        for row in report["neighbors"][task]:
            targets.append(
                {
                    "source_task": int(task),
                    "source_index": int(row["dataset_index"]),
                    "source_episode": int(row["episode_index"]),
                    "source_frame": int(row["frame_index"]),
                    "state": np.asarray(row["state"], dtype=np.float32),
                    "best_squared_l2": float("inf"),
                }
            )

    columns = ["observation.state", "index", "task_index", "episode_index", "frame_index"]
    paths = glob.glob(f"{args.dataset_root}/data/**/*.parquet", recursive=True)
    for path in paths:
        table = parquet.read_table(path, columns=columns)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        indices = np.asarray(table["index"])
        task_indices = np.asarray(table["task_index"])
        episodes = np.asarray(table["episode_index"])
        frames = np.asarray(table["frame_index"])
        for target in targets:
            candidates = np.flatnonzero(task_indices == target["source_task"])
            if candidates.size == 0:
                continue
            delta = states[candidates] - target["state"]
            squared = np.einsum("ij,ij->i", delta, delta)
            local = int(np.argmin(squared))
            distance = float(squared[local])
            if distance < target["best_squared_l2"]:
                row = int(candidates[local])
                target.update(
                    {
                        "best_squared_l2": distance,
                        "matched_index": int(indices[row]),
                        "matched_task": int(task_indices[row]),
                        "matched_episode": int(episodes[row]),
                        "matched_frame": int(frames[row]),
                        "max_abs_error": float(np.abs(delta[local]).max()),
                    }
                )

    for target in targets:
        target.pop("state")
        target["l2_error"] = float(np.sqrt(target.pop("best_squared_l2")))
    print(json.dumps({"rows": targets}, separators=(",", ":")))


if __name__ == "__main__":
    main()
