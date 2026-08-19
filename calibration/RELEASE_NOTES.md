# Release 1.0.0 (2026-08-11)

- Packaged the complete high-quality dual-arm URDF and visual meshes.
- Added a portable Python entry point with an explicit `--dataset-root`.
- Displays five synchronized poses simultaneously and saves them with one confirmation.
- Persists `T_camera_left_base` and `T_camera_right_base` without overwriting the initial estimate.
- Uses the OpenCV fisheye model, FK, triangle rasterization, and Z-buffer for validation.
- Excludes `TCP_Link_L` and `TCP_Link_R` from all masks.
- Handles Unicode paths when writing previews and sequence videos on Windows.
- Verified on a real episode with five-frame calibration input and frames 0, 100, and 200 projection output.

