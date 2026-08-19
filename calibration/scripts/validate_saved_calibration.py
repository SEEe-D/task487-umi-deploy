from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def find_packaged_urdf() -> Path:
    matches = sorted((PACKAGE_ROOT / "robot_model").glob("**/urdf/*.urdf"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one packaged URDF, found {len(matches)}")
    return matches[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate saved dual-arm extrinsics with offline fisheye projection."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--extrinsics-yaml",
        type=Path,
        default=PACKAGE_ROOT
        / "outputs"
        / "manual_calibration"
        / "calibrated_extrinsics.yaml",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-indices", default="0")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--sequence-max-frames", type=int, default=0)
    parser.add_argument("--max-faces-per-link", type=int, default=600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "calibration_validation",
    )
    args = parser.parse_args()

    extrinsics_yaml = args.extrinsics_yaml.resolve()
    if not extrinsics_yaml.is_file():
        raise FileNotFoundError(
            f"calibrated YAML not found: {extrinsics_yaml}. Run run_manual_calibration.py first."
        )

    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "render_dual_arm_projection.py"),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--urdf",
        str(find_packaged_urdf()),
        "--camera-config",
        str(PACKAGE_ROOT / "configs" / "camera_head_main.json"),
        "--extrinsics-yaml",
        str(extrinsics_yaml),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--episode-index",
        str(args.episode_index),
        "--frame-indices",
        args.frame_indices,
        "--geometry-mode",
        "visual_mesh",
        "--max-faces-per-link",
        str(args.max_faces_per_link),
        "--mask-dilation-px",
        "2",
        "--sequence-stride",
        str(args.sequence_stride),
        "--sequence-max-frames",
        str(args.sequence_max_frames),
    ]
    if args.sequence:
        command.append("--sequence")

    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)


if __name__ == "__main__":
    main()

