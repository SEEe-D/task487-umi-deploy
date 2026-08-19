# 双臂 URDF 人工外参标定 Pipeline

这是一个独立、可搬运的离线标定包。它用同步图像、双臂关节角和高质量 URDF Mesh，在 5 个不同姿态帧上同时调整：

- `T_camera_left_base`：左臂基座坐标到相机坐标的 4x4 齐次变换；
- `T_camera_right_base`：右臂基座坐标到相机坐标的 4x4 齐次变换。

相机坐标系固定为 OpenCV 定义：`+x` 向右、`+y` 向下、`+z` 向前。内部长度为米、角度为弧度，界面旋转步进以度显示。投影使用 OpenCV fisheye/equidistant 模型、三角形光栅化和 Z-buffer。`TCP_Link_L` 与 `TCP_Link_R` 被排除，不遮挡夹爪。

## 目录

- `scripts/run_manual_calibration.py`：推荐入口，五帧同时人工标定；
- `scripts/validate_saved_calibration.py`：用单帧或 episode 回放验证保存结果；
- `configs/camera_head_main.json`：1920x1536 原始 fisheye 标定及 224x224 预处理定义；
- `configs/dual_arm_extrinsics_initial.yaml`：手工测量初值；
- `configs/dual_arm_extrinsics_calibrated_reference.yaml`：当前已验证结果，仅作参考；
- `robot_model/`：原始高质量 URDF 与 STL Mesh；
- `reports/`：URDF、Mesh、相机检查报告；
- `examples/reference/`：已验证叠加图、Mask 和 episode 视频。

## 环境

在解压后的目录执行：

```powershell
uv sync
```

## 启动人工标定

```powershell
uv run python scripts/run_manual_calibration.py `
  --dataset-root "D:\path\to\task3-3" `
  --episode-index 0 `
  --frame-count 5
```

界面会同时显示 5 帧，不会逐帧跳转。调整完成后点击图像顶部的 `SAVE ALL 5`，或按 `C`/`Enter`，一次保存这 5 帧共同确认的结果并正常退出。

输出：

```text
outputs/manual_calibration/calibrated_extrinsics.yaml
outputs/manual_calibration/five_frame_preview.png
outputs/manual_calibration/calibrated_extrinsics_history/
```

再次运行时会自动以上一次保存的 YAML 为初值，方便继续微调。通过 `--initial-yaml` 可显式指定其他初值。

## 操作

- 控制窗口的 12 个滑块：分别调整左右臂 `tx/ty/tz/roll/pitch/yaw`；
- `J/K`：选择上一个/下一个参数；
- `H/L`：细调，平移 1 mm 或旋转 0.2 度；
- `U/O`：粗调，平移 10 mm 或旋转 2 度；
- `M`：切换显示模式；
- `R`：恢复本次启动值；
- `C` 或 `Enter`：确认并保存全部 5 帧；
- `Q`：退出且不覆盖结果。

默认每个 Link 最多使用 600 个三角面，仍来自原始高质量 Mesh，适合交互刷新。需要完整原始 Mesh 时可传 `--max-faces-per-link 0`，但刷新会明显变慢。

## 验证保存结果

单帧检查：

```powershell
uv run python scripts/validate_saved_calibration.py `
  --dataset-root "D:\path\to\task3-3" `
  --frame-indices "0,100,200"
```

连续 episode 回放：

```powershell
uv run python scripts/validate_saved_calibration.py `
  --dataset-root "D:\path\to\task3-3" `
  --sequence `
  --sequence-stride 5
```

验证输出位于 `outputs/calibration_validation/`，包括左右臂 Mask、合并 Mask、Link ID、Z-buffer、半透明叠加图和可选视频。

## 注意

- 输入必须是本项目检查通过的 LeRobot 风格目录，并包含头部视频与同步双臂关节状态；
- 当前内参对应中心裁剪后缩放到 `224x224` 的、仍带 fisheye 畸变的图像；
- 标定结果是 `camera_from_base`，不要在接入其他系统时擅自取反或转置；
- 没有真实场景深度时只处理机械臂自身遮挡，不处理桌面或物体对机械臂的遮挡；
- 参考外参只适用于当前安装关系，换相机或移动基座后需要重新确认。

