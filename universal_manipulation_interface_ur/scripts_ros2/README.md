## 目标

把 `universal_manipulation_interface` 的策略推理结果，作为双臂末端目标位姿下发给 `ros2_ws_wbc` 的 `ocs2_arm_controller` 执行。

链路如下：

- **反馈（来自 ros2_ws_wbc / ocs2_arm_controller）**：`/left_current_pose`、`/right_current_pose`（`geometry_msgs/PoseStamped`）
- **推理（本仓库节点）**：订阅相机 + 当前 pose，输出目标 pose
- **指令（发给 ocs2_arm_controller 的输入）**：`/left_target/stamped`、`/right_target/stamped` 或 `/dual_target/stamped`
- **控制器回显（用于可视化/调试）**：`/left_current_target`、`/right_current_target`（注意这不是控制器输入）

---

## 1) 先启动 OCS2 Arm Controller（ros2_ws_wbc）

你提供的命令：

```bash
ros2 launch ocs2_arm_controller demo.launch.py robot:=simpleai_x3 hardware:=real \
  can_interface_left:=can2 can_interface_right:=can3 can_fd:=false enable_protect_keyboard:=false
```

启动后建议确认话题存在（至少应有 feedback）：

```bash
ros2 topic list | egrep "left_current_pose|right_current_pose|left_target/stamped|right_target/stamped|dual_target/stamped"
ros2 topic info /left_current_pose
```

---

## 2) 启动 UMI 推理（conda）+ ROS2 客户端（Jazzy）

由于你的模型必须在 conda(Py3.9) 下运行，而 ROS2 Jazzy 的 rclpy 是 Py3.12，
因此采用“两进程”架构：

- **conda 推理服务端**：`scripts_ros2/umi_conda_inference_server.py`（只负责 torch 推理，不依赖 rclpy）
- **ROS2 客户端节点**：`scripts_ros2/umi_ocs2_client_node.py`（只负责相机/话题/控制，不依赖 torch）

### 2.1 终端 A：启动 conda 推理服务端（Py3.9）

```bash
conda activate umi
cd /home/simpleai/AI/universal_manipulation_interface

python3 scripts_ros2/umi_conda_inference_server.py \
  --ckpt /home/simpleai/AI/universal_manipulation_interface/model/0121-1036-baifang0120-unet-bimanual/epoch=0020-train_loss=0.017.ckpt \
  --host 127.0.0.1 \
  --port 18080 \
  --device cuda \
  --steps-per-inference 6
```

### 2.2 终端 B：启动 ROS2 客户端节点（Py3.12 + rclpy）

先 source ROS2：

```bash
source /opt/ros/jazzy/setup.bash
source /home/simpleai/ros2_ws_wbc/install/setup.bash
cd /home/simpleai/AI/universal_manipulation_interface
```

启动 client（默认 dry-run 只打印不下发）：

```bash
python3 scripts_ros2/umi_ocs2_client_node.py \
  --server-host 127.0.0.1 \
  --server-port 18080 \
  --camera0-dev 0 \
  --camera1-dev 1
```

如果你还要做 fisheye remap（`FisheyeRemapper`），建议同时指定 map（并确保相机输出分辨率与 map 的 src_w/src_h 一致）：

```bash
python3 scripts_ros2/umi_ocs2_client_node.py \
  --server-host 127.0.0.1 \
  --server-port 18080 \
  --camera0-dev 0 \
  --camera1-dev 1 \
  --fisheye-map0 /home/simpleai/AI/universal_manipulation_interface/scripts_ros2/fisheye_map.npz \
  --fisheye-map1 /home/simpleai/AI/universal_manipulation_interface/scripts_ros2/fisheye_map.npz
```

（已按当前需求移除 topic 模式：仅支持设备端口读取）

### 2.2 常用参数

- **`--target-frame-id`**：默认 `auto`（沿用 `/left_current_pose.header.frame_id`）。如果你的 TF 配置明确，也可手动指定（如 `base_link` 或 `world`）。
- **设备模式参数**：
  - `--camera0-dev` / `--camera1-dev`：如 `0`、`1` 或 `/dev/video0`
  - `--camera-width` / `--camera-height`：设备模式下尝试设置采集分辨率（两路共用）。若启用 fisheye map 且你没指定分辨率，会自动用 map 的 `src_w/src_h`。
- **鱼眼 remap（client侧集成，按你给的调用方式）**：
  - `--fisheye-map0 /path/fisheye_map0.npz`
  - `--fisheye-map1 /path/fisheye_map1.npz`
  - 节点内部会做：`frame_bgr -> remapper.remap_frame(frame_bgr) -> 转RGB -> 喂给策略`
  - 说明：ROS client 侧 remap 采用 `cv2.remap`（不依赖 torch），避免与 conda/ROS Python 版本冲突
- **相机可视化调试**：
  - `--show-cameras` 需要本机有图形界面（`$DISPLAY`/Wayland）。若你是无显示/SSH 环境，OpenCV 窗口可能不显示甚至卡住。
  - 无 GUI 调试可用：`--save-camera-debug-dir /tmp/cam_debug`（会周期性落盘两路 jpg，不依赖 start/server）
  - 可调保存频率：`--save-camera-debug-rate-hz 1`（默认 1Hz）
- **夹爪（Joint69/Joint79）**：
  - 节点会从 `/joint_states.position[69]` 与 `[79]` 读取夹爪角度（rad，范围约 `-0.61086524~0`），转换成 **0~35 的角度值(deg)** 作为策略输入（写入 `robot*_gripper_width` key）。
  - 策略输出的夹爪值按 **0~35 deg** 解释，发布时转换成 **负弧度**：`cmd_rad = -deg * pi/180`，并发布到 `/<JointName>/position_command`（`std_msgs/Float64`）。
  - 可通过参数覆盖：
    - `--gripper-position-index-left/--gripper-position-index-right`（默认 69/79）
    - `--gripper-deg-min/--gripper-deg-max`（默认 0/35）
- **`--frequency`**：控制频率（Hz），默认 10。
- **`--steps-per-inference`**：每次推理生成一段动作序列后，本节点按频率逐步发布多少步（默认 6）。
- **`--action-base`**：
  - `current`：动作相对“当前”末端 pose（与 UMI 现有 `get_real_umi_action` 一致，默认）
  - `episode_start`：动作相对 `/start` 时锁定的 start pose（符合你描述的“相对 start”语义；但前提是模型训练时动作语义一致）
- **默认发布通路（已改为 RViz 连续控制同路）**：
  - 默认会发布 `geometry_msgs/Pose` 到 `/left_target`、`/right_target`
  - 如需禁用（只走 stamped/path），加：`--no-publish-to-pose-target`

---

## 3) 用 RViz 拖拽到起始姿态，然后开始推理

1. 用 `arms_target_manager` / RViz 把机器人拖拽到你希望的起始位置（这一步是通过控制器链路完成的）。
2. 调用服务锁定起始 pose 并开始发布目标：

```bash
ros2 service call /umi_ocs2_inference/start std_srvs/srv/Trigger {}
```

说明：刚 start 后，服务端需要先累计满足模型观测 horizon 的历史帧（warmup），此时 client 会提示 `server warming up`，属于正常现象，稍等 1-2 秒即可。

停止推理：

```bash
ros2 service call /umi_ocs2_inference/stop std_srvs/srv/Trigger {}
```

---

## 4) 调试建议

### 4.1 看目标是否真的发到了控制器输入

控制器**输入**应看这些话题（不是 `*_current_target`）：

```bash
ros2 topic echo /left_target/stamped --once
ros2 topic echo /dual_target/stamped --once
```

如果你的现场系统（launch remap 或定制分支）确实把控制器输入接在 `*_current_target` 上，
本仓库推理节点也支持可选发布到该话题（谨慎使用）：

```bash
python3 scripts_ros2/umi_ocs2_inference_node.py ... --publish-to-current-target
```

另外，`arms_target_manager` 的**连续模式**默认发布的是 `geometry_msgs/Pose` 到 `left_target/right_target`
（注意无 frame_id、无 stamp）。如果你想完全复用这条通路，可开启：

```bash
python3 scripts_ros2/umi_ocs2_inference_node.py ...
```

### 4.2 看控制器是否在回发布目标与反馈

```bash
ros2 topic echo /left_current_pose --once
ros2 topic echo /left_current_target --once
```

### 4.3 若你看到 TF 警告（frame 转换失败）

- 最简单：把本节点 `--target-frame-id auto` 保持默认，确保发布 frame_id 与反馈一致；
- 或者把 `--target-frame-id` 设成 `task.info` 里 `baseFrame` 对应的 frame，并确保 TF 树里存在从发布 frame 到 baseFrame 的变换。

