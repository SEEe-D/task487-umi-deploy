# Task487 当前真机运行指令

更新于 2026-09-05。当前可选 CNC、RAW4W、Wrist4W 三组 `29999` 权重。
下面以 Wrist4W `wrist4w_seed42/29999` 模型、端口 `8000`、Marvin
双臂关节阻抗、`vegetable` 任务。请按“终端 1 → 终端 2 → 终端 3”的顺序启动，
每条命令独占一个终端。

该权重使用 UMI 12.5 Hz 契约：20 步动作块、左右腕各上/下两路共四路、head 输入禁用、
无 mask、`resize_with_pad` 图像处理。客户端会从服务端元数据切换到 12.5 Hz，并在
HOME 时将左右夹爪准备到该 checkpoint 的边界状态 1°/1°。这与旧真机权重的
25 Hz、三相机和物理全开 HOME 契约不同，不要混用。

## 终端 1：启动模型服务

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
unset TASK487_POLICY_CONFIG
CUDA_VISIBLE_DEVICES=1 bash run_task487_server.sh wrist4w 8000



unset TASK487_POLICY_CONFIG

# RAW：头部 + 四腕
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
unset TASK487_POLICY_CONFIG
CUDA_VISIBLE_DEVICES=1 bash run_task487_server.sh raw4w 8000

# CNC：头部 + 四腕
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
unset TASK487_POLICY_CONFIG
CUDA_VISIBLE_DEVICES=1 bash run_task487_server.sh cnc 8000

# Wrist-only：仅四腕
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
unset TASK487_POLICY_CONFIG
CUDA_VISIBLE_DEVICES=1 bash run_task487_server.sh wrist4w 8000
```

上述命令加载：

```text
checkpoints/pi05_umi_task487_wrist_only_12_5/wrist4w_seed42/29999
```

并监听端口 `8000`。GPU 1 用于与 GPU 0 上的 B1 服务并存；如果只运行一套模型，可指定空闲 GPU。
切换模型时先退出客户端并停止该端口的旧模型服务，再把 `wrist4w` 换成 `raw4w` 或 `cnc`。
脚本同时接受完整 checkpoint 路径并自动匹配这三组配置；残留的旧 `TASK487_POLICY_CONFIG` 会被拒绝。

| 服务别名 | 本地模型配置 | 模型使用的画面 |
|---|---|---|
| `raw4w` | `pi05_umi_task487_raw_4w_12_5` | 头部 + 左上、左下、右上、右下 |
| `cnc` | `pi05_umi_task487_cnc_4w_12_5` | 头部 + 左上、左下、右上、右下；推理不加 token mask |
| `wrist4w` | `pi05_umi_task487_wrist_only_4w_12_5` | 左上、左下、右上、右下；head 禁用 |

这些 `_4w` 名称是本地推理配置，匹配开发机的四腕训练输入；旧三相机/两腕配置保留。
详细映射与验证记录见 [TASK487_FOUR_WRIST_DEPLOY.md](TASK487_FOUR_WRIST_DEPLOY.md)。

## 终端 2：启动 Marvin/Mink 阻抗后端

```bash
cd /home/simpleai/Code/mjm/eval_mink
env \
  -u ROS_DISTRO \
  -u ROS_VERSION \
  -u ROS_PYTHON_VERSION \
  -u ROS_PACKAGE_PATH \
  -u ROS_ROOT \
  -u ROS_ETC_DIR \
  -u ROS_MASTER_URI \
  -u AMENT_PREFIX_PATH \
  -u COLCON_PREFIX_PATH \
  -u CATKIN_PREFIX_PATH \
  -u CMAKE_PREFIX_PATH \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  ROS_DOMAIN_ID=77 \
  REPLAY_BASE=start_teleop_replay.sh \
  ./start_teleop_replay_impedance.sh enable_intervention:=false
```

启动成功必须最终看到左右臂都进入关节阻抗：

```text
A/左臂 关节阻抗 OK, mode=ArmMode.IMP_JOINT
B/右臂 关节阻抗 OK, mode=ArmMode.IMP_JOINT
```

如果出现“阻抗切换失败，回退 POSITION_FOLLOW”，不要继续启动动作。

## 终端 3：启动 Task487 真机客户端

确认终端 1、终端 2 均正常后执行：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client.sh vegetable 127.0.0.1 8000 \
  --execute --continuous --show-processed-cameras
```

`--show-processed-cameras` 会显示请求中的相机画面；服务端元数据必须明确显示
`camera_order=[cam_left_top, cam_left_down, cam_right_top, cam_right_down]` 和
`head_enabled=false`。Wrist4W 预览只显示四路腕部，不订阅头部。
窗口在 HOLD、dry-run 和 ACTIVE 状态下均按控制循环持续刷新，只读取请求内容，
不会修改模型输入。若只想看原始相机画面，可改回 `--show-cameras`。

客户端启动后的按键顺序：

1. 按 `r` 执行 HOME，并将左右夹爪准备到该 UMI checkpoint 的 1°/1° 边界状态。
2. 人工确认双臂归位、夹爪接近闭合且周围安全。
3. 按 `d` 进入 ACTIVE，开始模型控制。
4. 紧急停止或暂停时按 `s`，立即回到 HOLD。

## 强制松抱闸指令

> **危险：强制松闸后机械臂会失去支撑并可能因重力坠落。必须有人扶住机械臂、
> 末端无重物且机械臂处于低姿态。该操作只用于维护/诊断，不是上述三个终端的
> 常规启动步骤。执行前必须停止 Marvin/Mink 后端，避免两个程序同时连接机器人。**

启动 Marvin 伺服工具：

```bash
cd /home/simpleai/Code/eval_benchmark/Marvin/marvin_wrapper/tests
/home/simpleai/anaconda3/envs/openpi/bin/python ./servo_utility.py
```

交互按键：

```text
3  -> 强制松开当前臂抱闸
y  -> 确认
s  -> 切换 A/B 臂
4  -> 强制抱闸
y  -> 确认
q  -> 退出
```

如需对两条臂都松闸：先对默认的 `A` 臂输入 `3`、`y`，再输入 `s` 切换到 `B` 臂，
再次输入 `3`、`y`。维护结束后，应分别用 `4`、`y` 将两条臂重新抱闸，或重新执行
机器人 enable。

强制松闸要求伺服参数处于 `166` 混合控制模式，否则命令不会生效。

## 关闭顺序

1. 在终端 3 按 `s` 回到 HOLD，然后按 `Ctrl+C` 退出客户端。
2. 在终端 2 按 `Ctrl+C`，确认日志显示双臂已进入 `DISABLE`。
3. 在终端 1 按 `Ctrl+C` 停止模型服务。
