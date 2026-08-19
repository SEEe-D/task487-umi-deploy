#!/usr/bin/env python3
"""Summarize legacy and resize-with-pad-aligned task487 masks as 16x16 tokens.

This intentionally emits no source images or robot states.  It is suitable for
running next to a LeRobot dataset and copying only the compact JSON result back
to the deployment checkout.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as torch_f
from lerobot.datasets.lerobot_dataset import LeRobotDataset


GRID = 16
MIN_KEEP_TOKENS = 64


def token_keep(mask: np.ndarray) -> np.ndarray:
    """Match fixed_head_mask_to_token_keep_mask in umi_policy.py."""
    masked = np.asarray(mask) > 0.5
    height, width = masked.shape
    y_edges = np.linspace(0, height, GRID + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, GRID + 1, dtype=np.int64)
    ratios = np.empty((GRID, GRID), dtype=np.float32)
    for row in range(GRID):
        for column in range(GRID):
            ratios[row, column] = masked[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ].mean()
    keep = ratios < 0.5
    if int(keep.sum()) < MIN_KEEP_TOKENS:
        indices = np.argsort(ratios.reshape(-1), kind="stable")[:MIN_KEEP_TOKENS]
        keep.reshape(-1)[indices] = True
    return keep.reshape(-1)


def resize_with_pad_mask(mask: torch.Tensor, height: int = 224, width: int = 224) -> np.ndarray:
    """Apply the same 640x512 -> 224x224 geometry as RGB resize_with_pad."""
    if mask.ndim == 3:
        mask = mask[:3].float().mean(dim=0)
    elif mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    cur_height, cur_width = mask.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = torch_f.interpolate(
        mask[None, None],
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    return torch_f.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="task487_lerobot_combined_12_5")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    rows = []
    for index in args.indices:
        item = dataset[index]
        source = item["observation.images.fixed_head_mask"]
        source_2d = source[:3].float().mean(dim=0).cpu().numpy()
        legacy = token_keep(source_2d)
        aligned = token_keep(resize_with_pad_mask(source))
        changed = np.flatnonzero(legacy != aligned)
        rows.append(
            {
                "dataset_index": index,
                "task_index": int(item["task_index"]),
                "episode_index": int(item["episode_index"]),
                "frame_index": int(item["frame_index"]),
                "source_masked_fraction": float((source_2d > 0.5).mean()),
                "legacy_keep_count": int(legacy.sum()),
                "aligned_keep_count": int(aligned.sum()),
                "changed_token_count": int(changed.size),
                "changed_token_indices": changed.tolist(),
                "legacy_keep_bits": "".join("1" if value else "0" for value in legacy),
                "aligned_keep_bits": "".join("1" if value else "0" for value in aligned),
            }
        )
    print(json.dumps({"rows": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
