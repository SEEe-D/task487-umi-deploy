#!/usr/bin/env python3
"""Read-only same-noise A/B test for Task487 legacy versus aligned masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.policies import policy_config
from openpi.policies.umi_policy import fixed_head_mask_to_token_keep_mask
from openpi.shared import image_tools
from openpi.training import config as training_config
from task487_runtime.contract import rot6d_to_matrix


ARM_OFFSETS = {"right": 0, "left": 10}
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 20
EXECUTION_PREFIX = 5


def resize_with_pad(image: np.ndarray) -> np.ndarray:
    return np.asarray(image_tools.resize_with_pad(image, 224, 224))


def load_samples(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = []
    for meta in manifest:
        path = root / meta["file"]
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: np.array(archive[key]) for key in archive.files}
        sample.update(meta)
        sample["path"] = path
        samples.append(sample)
    return samples


def make_observation(sample: dict, mask: np.ndarray) -> dict:
    return {
        "cam_head": np.ascontiguousarray(sample["head"]),
        "cam_left_top": np.ascontiguousarray(sample["left"]),
        "cam_right_top": np.ascontiguousarray(sample["right"]),
        "fixed_head_mask": np.ascontiguousarray(mask),
        "pre_state": np.asarray(sample["pre_state"], dtype=np.float32),
        "state": np.asarray(sample["state"], dtype=np.float32),
        "prompt": sample["prompt"],
    }


def rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = rot6d_to_matrix(left)
    right_matrix = rot6d_to_matrix(right)
    relative = np.einsum("...ji,...jk->...ik", left_matrix, right_matrix)
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def rotation_magnitude_degrees(rot6d: np.ndarray) -> np.ndarray:
    return np.rad2deg(Rotation.from_matrix(rot6d_to_matrix(rot6d)).magnitude())


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    fields = (
        "translation_diff_mm",
        "rotation_diff_deg",
        "gripper_diff_deg",
        "legacy_translation_norm_mm",
        "aligned_translation_norm_mm",
        "legacy_rotation_magnitude_deg",
        "aligned_rotation_magnitude_deg",
    )
    output = {field: summarize_values([value for row in rows for value in row[field]]) for field in fields}
    for condition in ("legacy", "aligned"):
        values = np.asarray([value for row in rows for value in row[f"{condition}_gripper_deg"]])
        output[f"{condition}_gripper_deg_range"] = [float(values.min()), float(values.max())]
    return output


def compare_actions(legacy: np.ndarray, aligned: np.ndarray, *, sample: dict, seed: int) -> list[dict]:
    if legacy.shape != (ACTION_HORIZON, 20) or aligned.shape != (ACTION_HORIZON, 20):
        raise ValueError(f"Unexpected output shapes: {legacy.shape}, {aligned.shape}")
    if not np.isfinite(legacy).all() or not np.isfinite(aligned).all():
        raise ValueError("Policy returned NaN or infinity")
    rows = []
    for region, stop in (("first5", EXECUTION_PREFIX), ("full20", ACTION_HORIZON)):
        for arm, offset in ARM_OFFSETS.items():
            legacy_arm = legacy[:stop, offset : offset + 10]
            aligned_arm = aligned[:stop, offset : offset + 10]
            rows.append(
                {
                    "dataset_index": int(sample["index"]),
                    "task_index": int(sample["task_index"]),
                    "seed": seed,
                    "region": region,
                    "arm": arm,
                    "translation_diff_mm": (
                        1000.0 * np.linalg.norm(legacy_arm[:, :3] - aligned_arm[:, :3], axis=1)
                    ).tolist(),
                    "rotation_diff_deg": rotation_distance_degrees(
                        legacy_arm[:, 3:9], aligned_arm[:, 3:9]
                    ).tolist(),
                    "gripper_diff_deg": np.rad2deg(
                        np.abs(legacy_arm[:, 9] - aligned_arm[:, 9])
                    ).tolist(),
                    "legacy_translation_norm_mm": (
                        1000.0 * np.linalg.norm(legacy_arm[:, :3], axis=1)
                    ).tolist(),
                    "aligned_translation_norm_mm": (
                        1000.0 * np.linalg.norm(aligned_arm[:, :3], axis=1)
                    ).tolist(),
                    "legacy_rotation_magnitude_deg": rotation_magnitude_degrees(
                        legacy_arm[:, 3:9]
                    ).tolist(),
                    "aligned_rotation_magnitude_deg": rotation_magnitude_degrees(
                        aligned_arm[:, 3:9]
                    ).tolist(),
                    "legacy_gripper_deg": np.rad2deg(legacy_arm[:, 9]).tolist(),
                    "aligned_gripper_deg": np.rad2deg(aligned_arm[:, 9]).tolist(),
                }
            )
    return rows


def token_overlay(head: np.ndarray, keep: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = head.astype(np.float32).copy()
    masked = (~keep.reshape(16, 16)).repeat(14, axis=0).repeat(14, axis=1)
    output[masked] = 0.45 * output[masked] + 0.55 * np.asarray(color, dtype=np.float32)
    return output.clip(0, 255).astype(np.uint8)


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (223, 24), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_preview(sample: dict, aligned_mask: np.ndarray, output_dir: Path) -> dict:
    head = resize_with_pad(sample["head"])
    legacy_keep = fixed_head_mask_to_token_keep_mask(sample["mask"])
    aligned_keep = fixed_head_mask_to_token_keep_mask(aligned_mask)
    changed = legacy_keep != aligned_keep
    legacy = token_overlay(head, legacy_keep, (255, 40, 40))
    aligned = token_overlay(head, aligned_keep, (40, 160, 255))
    diff = head.copy()
    grid = changed.reshape(16, 16).repeat(14, axis=0).repeat(14, axis=1)
    diff[grid] = (0.35 * diff[grid] + 0.65 * np.array([255, 220, 20])).clip(0, 255)
    panels = np.concatenate(
        (
            labeled_panel(head, "RGB after resize_with_pad"),
            labeled_panel(legacy, f"legacy masked={int((~legacy_keep).sum())}"),
            labeled_panel(aligned, f"aligned masked={int((~aligned_keep).sum())}"),
            labeled_panel(diff, f"changed tokens={int(changed.sum())}"),
        ),
        axis=1,
    )
    path = output_dir / f"mask_ab_idx{int(sample['index'])}.jpg"
    if not cv2.imwrite(str(path), panels):
        raise RuntimeError(f"Failed to write {path}")
    return {
        "legacy_keep_count": int(legacy_keep.sum()),
        "aligned_keep_count": int(aligned_keep.sum()),
        "changed_token_count": int(changed.sum()),
        "changed_token_fraction": float(changed.mean()),
        "preview": path.name,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Task487 mask alignment offline A/B",
        "",
        "Read-only validation. No robot commands were published.",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Seeds: {report['seeds']}",
        f"Determinism control max abs error: {report['determinism_control_max_abs']:.3g}",
        "",
        "## First five action steps",
        "",
        "| Task | Arm | Translation diff mean/max (mm) | Rotation diff mean/max (deg) | Gripper diff mean/max (deg) | Legacy gripper range (deg) | Aligned gripper range (deg) |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for task in (1, 2):
        for arm in ("right", "left"):
            value = report["summary"][f"task{task}"]["first5"][arm]
            trans = value["translation_diff_mm"]
            rotation = value["rotation_diff_deg"]
            gripper = value["gripper_diff_deg"]
            lines.append(
                f"| {task} | {arm} | {trans['mean']:.2f} / {trans['max']:.2f} | "
                f"{rotation['mean']:.2f} / {rotation['max']:.2f} | "
                f"{gripper['mean']:.2f} / {gripper['max']:.2f} | "
                f"{value['legacy_gripper_deg_range'][0]:.2f} … {value['legacy_gripper_deg_range'][1]:.2f} | "
                f"{value['aligned_gripper_deg_range'][0]:.2f} … {value['aligned_gripper_deg_range'][1]:.2f} |"
            )
    lines.extend(
        (
            "",
            "## Per-sample mask change",
            "",
            "| Task | Dataset index | Changed tokens | Legacy keep | Aligned keep | Preview |",
            "|---:|---:|---:|---:|---:|:---|",
        )
    )
    for sample in report["samples"]:
        mask = sample["mask"]
        lines.append(
            f"| {sample['task_index']} | {sample['dataset_index']} | "
            f"{mask['changed_token_count']}/256 | {mask['legacy_keep_count']} | "
            f"{mask['aligned_keep_count']} | [{mask['preview']}]({mask['preview']}) |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="pi05_umi_task487_masked_12_5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.samples)
    if len(samples) != 6:
        raise ValueError(f"Expected six samples, got {len(samples)}")

    config = training_config.get_config(args.config)
    load_start = time.monotonic()
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    load_seconds = time.monotonic() - load_start

    all_rows = []
    sample_reports = []
    saved_actions = {}
    determinism_control = None
    for sample_number, sample in enumerate(samples):
        aligned_mask = resize_with_pad(sample["mask"])
        mask_stats = save_preview(sample, aligned_mask, args.output)
        legacy_obs = make_observation(sample, sample["mask"])
        aligned_obs = make_observation(sample, aligned_mask)
        inference_rows = []
        for seed in args.seeds:
            noise = np.random.default_rng(seed).standard_normal(
                (ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32
            )
            legacy_result = policy.infer(legacy_obs, noise=noise)
            aligned_result = policy.infer(aligned_obs, noise=noise)
            legacy_actions = np.asarray(legacy_result["actions"], dtype=np.float64)
            aligned_actions = np.asarray(aligned_result["actions"], dtype=np.float64)
            if sample_number == 0 and seed == args.seeds[0]:
                repeat = np.asarray(policy.infer(legacy_obs, noise=noise)["actions"], dtype=np.float64)
                determinism_control = float(np.max(np.abs(repeat - legacy_actions)))
            saved_actions[f"idx{int(sample['index'])}_seed{seed}_legacy"] = legacy_actions
            saved_actions[f"idx{int(sample['index'])}_seed{seed}_aligned"] = aligned_actions
            inference_rows.extend(compare_actions(legacy_actions, aligned_actions, sample=sample, seed=seed))
        all_rows.extend(inference_rows)
        sample_reports.append(
            {
                "dataset_index": int(sample["index"]),
                "task_index": int(sample["task_index"]),
                "episode_index": int(sample["episode_index"]),
                "frame_index": int(sample["frame_index"]),
                "mask": mask_stats,
                "first5": {
                    arm: summarize_rows(
                        [row for row in inference_rows if row["region"] == "first5" and row["arm"] == arm]
                    )
                    for arm in ARM_OFFSETS
                },
            }
        )

    summary = {}
    for task in (1, 2):
        task_rows = [row for row in all_rows if row["task_index"] == task]
        summary[f"task{task}"] = {
            region: {
                arm: summarize_rows(
                    [row for row in task_rows if row["region"] == region and row["arm"] == arm]
                )
                for arm in ARM_OFFSETS
            }
            for region in ("first5", "full20")
        }
    summary["overall"] = {
        region: {
            arm: summarize_rows([row for row in all_rows if row["region"] == region and row["arm"] == arm])
            for arm in ARM_OFFSETS
        }
        for region in ("first5", "full20")
    }

    report = {
        "read_only": True,
        "robot_commands_published": False,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "samples": sample_reports,
        "seeds": args.seeds,
        "model_load_seconds": load_seconds,
        "determinism_control_max_abs": determinism_control,
        "summary": summary,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    np.savez_compressed(args.output / "actions.npz", **saved_actions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
