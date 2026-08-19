from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def find_packaged_urdf() -> Path:
    matches = sorted((PACKAGE_ROOT / "robot_model").glob("**/urdf/*.urdf"))
    if len(matches) != 1:
        formatted = "\n".join(f"  - {path}" for path in matches) or "  (none)"
        raise RuntimeError(f"expected one packaged URDF, found {len(matches)}:\n{formatted}")
    return matches[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively calibrate camera_from_base transforms using five frames."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "camera_head_main.json",
    )
    parser.add_argument(
        "--max-faces-per-link",
        type=int,
        default=600,
        help="Visual-mesh face limit per Link. Use 0 for the original full meshes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "manual_calibration",
    )
    parser.add_argument(
        "--initial-yaml",
        type=Path,
        help="Initial camera_from_base YAML. Existing output is reused when omitted.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render and save the initial five-frame view without opening the UI.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml = output_dir / "calibrated_extrinsics.yaml"
    preview_output = output_dir / "five_frame_preview.png"

    if args.initial_yaml is not None:
        initial_yaml = args.initial_yaml.resolve()
    elif save_yaml.exists():
        initial_yaml = save_yaml
    else:
        initial_yaml = PACKAGE_ROOT / "configs" / "dual_arm_extrinsics_initial.yaml"
    if not initial_yaml.is_file():
        raise FileNotFoundError(f"initial extrinsics YAML does not exist: {initial_yaml}")

    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "calibrate_dual_arm_extrinsics.py"),
        "--dataset-root",
        str(dataset_root),
        "--urdf",
        str(find_packaged_urdf()),
        "--camera-config",
        str(args.camera_config.resolve()),
        "--initial-yaml",
        str(initial_yaml),
        "--save-yaml",
        str(save_yaml),
        "--preview-output",
        str(preview_output),
        "--episode-index",
        str(args.episode_index),
        "--frame-count",
        str(args.frame_count),
        "--max-faces-per-link",
        str(args.max_faces_per_link),
        "--sequential-confirm",
    ]
    if args.headless:
        command.append("--headless")

    print(f"dataset_root={dataset_root}")
    print(f"initial_yaml={initial_yaml}")
    print(f"save_yaml={save_yaml}")
    print(f"frames_displayed_together={args.frame_count}")
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)


if __name__ == "__main__":
    main()
