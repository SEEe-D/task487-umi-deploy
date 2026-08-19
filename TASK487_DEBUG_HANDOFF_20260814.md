# Task487 Pi0.5 真机调试交接文档

> 快照时间：2026-08-15 18:54 CST  
> 工程目录：`/home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy`  
> 当前目标：先把 Task487 三相机蔬果分拣基线在天机/Marvin 双臂上稳定跑通，再开展 UMI 头部稳定化与 embodiment mask 研究。  
> 最新运行状态：B1 模型服务器监听 `8082`；Mink/Marvin 后端位于 tmux `marvin-impedance`，A/B 两臂均已确认进入 `IMP_JOINT`；一个 Task487 主客户端正在运行（其余同命令行 PID 是派生工作进程），但已于 18:54:24 因左臂跟踪误差 `50.142 mm > 50.000 mm` 自动进入 HOLD。最新日志为 `task487_logs/20260815_184941_2291301/client.log`。  
> 安全结论：当前不是 ACTIVE。不要放宽跟踪阈值或直接按 `d` 续跑；先查看最新日志和现场机械臂状态。本文末尾的 2026-08-15 最新章节覆盖前文过时的端口、相机、坐标、控制模式和进程快照。

## 2026-08-14 17:02 更新：机械臂不动的根因与恢复状态

- 16:50 的 RTC/action-chunk 链路实际持续输出：Mink 为 ACTIVE，Marvin bridge 以约 100 Hz push，内部目标也在变化；真机关节却一直停在 HOME。
- 根因不是 RTC 拼接。旧 Marvin 后端在 15:57 报过双臂 `err_code=15`，SafetyMonitor 执行了 `soft_stop`；后续自动恢复只清 fault，没有重新执行 `POSITION_FOLLOW`，所以软件目标在走、机械臂不走。
- 已重启 Mink/Marvin 后端。启动日志确认 A/B 两臂均切到 `position_follow`，`dual enable` 成功；后端持续报告 `state=HOLD joint_states=ok`，没有新的硬件 fault。
- 已恢复 Task487 30k RTC server，监听 `0.0.0.0:8000`；checkpoint 参数与 norm stats 加载成功。
- 当前没有 `task487_client.py`；机械臂仍处于 HOLD，不会自动 HOME 或 ACTIVE。下一轮仍由操作者启动 client 后按 `r` HOME，再按 `d` ACTIVE。
- scheduler 新增独立的“真实 EEF 对控制器 setpoint”跟踪保护。RTC action index 仍跟随控制器时间线，但若硬件再次不跟随，会在 50 mm / 0.30 rad 阈值触发 HOLD，而不会再把硬件停滞误判成正常 RTC 消费。
- 本轮测试：`28 passed`。

## 2026-08-14 17:12 更新：复位后连续运行往返抖动

- 后端日志证明抖动目标来自客户端上层，Marvin/Mink 跟踪正常：命令 age 0–1 ms，Mink `trk_err` 最低约 0.8 mm；不是电机或 IK 超调。
- 根因是 `rtc_prefix_targets()` 错把尚未发给控制器的旧 chunk 尾巴也纳入固定 10 步前缀。每 5 步重推后，实际长期执行的是新预测中较不稳定的 action 10–14，目标会周期性反向。
- 已改为只把 `_sent=True` 的连续已提交动作作为 RTC hard prefix；未发送旧尾巴立即由新 chunk 覆盖。保留 action-chunk + RTC，不增加额外滤波器。
- 预期在线日志由 `expired=10, rtc_prefix≈8–10` 改为 `expired=5, rtc_prefix≈3–5`。
- 本轮测试：`29 passed`。

## 2026-08-14 17:16 更新：RTC suffix 速度方向接缝

- 已提交 prefix 修复在线生效：日志主要变为 `expired=3–5, rtc_prefix=1–5`。
- 新暴露的 HOLD 为 prefix 后首段新 suffix 跳变 30.612 mm，超过原有 30 mm 安全阈值；没有放宽阈值。
- 新增 5 步 action-chunk handoff cross-fade：固定 RTC prefix 不变，新 suffix 的前 5 点在旧安全未发送尾巴与新预测之间按 1/6 到 5/6 淡入；平移、SO(3) 旋转和夹爪同步处理。
- 在线日志新增 `blend=5`，用于确认交叉淡入生效。
- 本轮测试：`30 passed`。

## 1. 一页结论

目前已经确认的事实：

1. 当前训练与推理是 Physical Intelligence 官方 OpenPI/Pi0.5 主干（commit `15a9616`）加受控的 UMI 双臂适配，不是 StarVLA，也不是重新实现的 VLA。
2. Task487 30k checkpoint 已训练完成并同步到本地，模型、norm stats、三相机输入、20D 双臂状态/动作和 25 Hz 数据时间契约能够加载和推理。
3. 离线评测显示 30k 权重在训练分布内明显优于 10k/20k 和零动作基线，不能把当前真机问题简单归因为“权重完全没训练好”。
4. 当前真机路径使用天机/Marvin + Mink ROS bridge，不使用 UR RTDE。日志里的 `Connect robot: None None` 在这条路径上是预期行为。
5. 三路 Thor 相机已经能够稳定收帧；最后确认的源为 `video0/right head`、`video3/left top`、`video5/right top`，接收端口分别是 5001、5002、5004。
6. 夹爪的假零点问题已经定位并修复；两夹爪在最近一次 `r` 后到达右 30.2°、左 30.3°，满足 Task487 开爪起始条件。
7. 最新且最确定的在线问题是：上层 scheduler 按模型原始 25 Hz 时间戳消费轨迹，而底层 `PoseTrajectoryInterpolator` 因 0.05 m/s、0.20 rad/s 限速把完成时间向后延长。上层不知道这个时间延长，仍继续 pop、重推理和拼接，因此计划轨迹逐渐领先真实末端，最终触发 50 mm 跟踪误差保护。
8. 模型自身还存在跨 chunk 时间不一致：每 0.2 秒重推时，同一未来时刻的新旧预测接管跳变 P95 约为右臂 4.45 mm、左臂 3.53 mm。控制端需要平滑它，但控制端不能凭空恢复完全一致的轨迹。
9. 现在不应继续改大速度、步长或安全阈值。应先让 scheduler 和底层限速插值器共享同一条“实际可执行时间轴”，然后再做单轮真机验证。

## 2. 用户真正要解决的研究问题

长期目标不是普通的视觉干扰消除，而是 UMI 数据采集机制带来的训练—部署域差异：

- UMI 训练数据的头戴相机必然随采集者头部运动；
- 头部画面必然拍到采集者手臂、肩膀和躯干；
- 真机推理时头部相机固定，画面中出现的是机器人本体；
- 模型可能利用只存在于人类示范域的视觉捷径。

计划方法：

1. 训练端稳定 UMI 头部画面；
2. 对稳定后的 RGB 与人体 mask 应用完全相同的几何变换；
3. 在 Pi0.5 最终视觉 token 层屏蔽人类本体；
4. 推理端用机器人几何投影生成机器人本体 mask，并屏蔽相同位置的 token；
5. 让训练和部署两端都尽量只保留桌面、物体、容器和目标区域。

当前 Task487 30k 是 **no-mask 基线**，`mask_enabled=False`。它只用于验证 Pi0.5 数据—模型—真机闭环，不能用于证明 mask 方法有效。

研究主说明见：`UMI_HEAD_VIEW_DOMAIN_GAP_RESEARCH_PLAN.md`。服务器 mask 管线交接包位于：

```text
/home/simpleai/Code/universal_manipulation_interface-main/gj/umi_mask_pipeline_server_handoff_20260813
```

## 3. 当前软硬件拓扑

```text
Task487 checkpoint / Pi0.5 server (localhost:8000)
                         │ WebSocket
                         ▼
task487_client.py ── 3路Thor RGB + 双臂EEF/夹爪状态
       │
       ├─ 25 Hz RollingScheduler
       ├─ ROS RosTargetController (100 Hz, 10 ms tick)
       ├─ Mink arm controller / target bridge
       └─ Marvin bridge → 天机双臂 position_follow / DualSender 1000 Hz

夹爪：can1 → gripper_can_node → ROS gripper controller
相机：Thor 192.168.2.178 → UDP video/meta → 本机 192.168.2.108
```

本机网卡快照：

| 接口 | 地址 | 用途 |
|---|---|---|
| `eno1` | `192.168.31.130/24` | 普通局域网 |
| `enx6c1ff7df8e78` | `192.168.1.165/24` | 另一设备网段 |
| `enx6c1ff7730f44` | `192.168.2.108/24` | Thor/机器人相关网段 |
| `can1` | `UP, LOWER_UP` | 双夹爪 CAN |

Thor 主机当前使用：`192.168.2.178`。

## 4. 当前宿主机进程快照

2026-08-14 16:09 查询到：

| PID | 进程 | 状态/说明 |
|---:|---|---|
| 234405 | `serve_policy.py ... pi05_umi_task487 .../29999` | 模型服务器，监听 `0.0.0.0:8000` |
| 276333 | `gripper_can_node` | 修复后独立夹爪节点 |
| 314324 | `start_teleop_replay.sh enable_intervention:=false` | Mink/Marvin 后端父进程 |
| 314440 | `mink_arm_controller.py` | 双臂控制器 |
| 314442 | `ee_pose_publisher.py` | EEF 反馈 |
| 314446 | `marvin_bridge.py` | Mink → Marvin 真机 bridge |
| 314448 | `ros_target_bridge.py` | 客户端目标 → ROS bridge |

没有发现 `task487_client.py` 进程。独立夹爪节点由 tmux session `gripper-rezero-fixed2` 启动。不要重复启动第二个 `gripper_can_node`，也不要在上述后端仍运行时再启动第二套 Mink/Marvin backend。

进程状态会变化，接手后先重新执行：

```bash
pgrep -af 'serve_policy|task487_client|gripper_can_node|mink_arm_controller|marvin_bridge|ros_target_bridge|start_teleop_replay'
ip -details link show can1
ss -ltnup | grep ':8000'
```

## 5. 训练端与 checkpoint

### 5.1 云端

最后确认的云端目录：

```text
/root/pi05_mask
```

公网 SSH 曾使用：

```bash
ssh -p 20926 root@124.174.13.117
```

不要把账号密码或私钥写入仓库。云端和本地使用同一个 `pi05_umi_task487` 适配版本；早期本地旧 config 已被同步替换。

训练命令为：

```text
scripts/train.py pi05_umi_task487 \
  --exp-name task487_3cam_nomask_30k \
  --num-train-steps 30000
```

主要训练配置：

- `Pi0Config(pi05=True, action_dim=32, action_horizon=20)`；
- 有效机器人动作/状态为 20D，模型内部补齐到 32D；
- batch size 64；
- 4 张 GPU；
- base 权重：`checkpoints/pi05_base/params`；
- warmup 1000 steps；
- peak LR `5e-5`，30k decay 到 `5e-6`；
- AdamW，gradient clip 1.0；
- EMA 0.999；
- 每 5000 steps 保存。

### 5.2 数据集

Task487 云端数据约 762 MB：

```text
/root/pi05_mask/datasets/task487
```

已核对：

- 230 episodes；
- 150,907 帧；
- 严格 25 Hz；
- 相邻时间戳约 0.04 s；
- 150,677 个同 episode 相邻对全部满足 `action[t] == state[t+1]`；
- action stride 1；
- 20 个动作对应未来 `+0.04 ... +0.80 s`；
- 状态历史对应 `[-0.04, 0.0] s`；
- 全量数据用于训练，没有独立 validation split。

三个数据集 task：

| task index | 文本 |
|---:|---|
| 0 | `Vegetable and Fruit Sorting.` |
| 1 | `Pick Up Vegetable and Place Vegetable on the Pink  Plate on the Right` |
| 2 | `Pick Up Fruit and Place Fruit on the Blue Plate on the Left` |

注意 task 1 的 `Pink` 与 `Plate` 之间是两个空格；真机 client 保留了精确文本。

当前茄子测试使用 `vegetable`，预期由右臂把蔬菜放到右侧粉色盘；茄子在数据语义中按 vegetable 处理。

### 5.3 本地 checkpoint

```text
/home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy/checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999
```

- 大小约 12 GB；
- `params/` 存在；
- checkpoint 内自带 `assets/umi_task487/norm_stats.json`；
- server 若先报告正式 assets 目录没有 norm stats、随后从 checkpoint assets 成功加载，是正常的；
- JAX 的 ROCm/TPU backend 初始化 warning 在 NVIDIA 本机上是无害信息；
- 首次加载很快不代表漏载，Orbax 日志已显示恢复 12.5 GiB 参数；
- 第一次 warmup 可能约 10 s（JIT 编译），随后推理约 90–120 ms。

## 6. OpenPI 源码状态

正式本地 OpenPI 仓库：

```text
gj/pi05-deploy/openpi-official
```

base commit：`15a9616`。

它不是完全干净的官方仓库，而是官方 Pi0.5 核心加 UMI 适配。当前 tracked 改动包括：

```text
.gitignore
pyproject.toml
scripts/compute_norm_stats.py
src/openpi/models/model.py
src/openpi/models/pi0.py
src/openpi/models_pytorch/pi0_pytorch.py
src/openpi/models_pytorch/preprocessing_pytorch.py
src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/training/data_loader_test.py
uv.lock
```

新增文件：

```text
src/openpi/policies/umi_policy.py
src/openpi/policies/umi_policy_test.py
```

受控适配内容：

- Task487 LeRobot 本地数据读取；
- 三相机 repack；
- 双臂 20D 状态/动作；
- `[t-1, t]` 状态历史；
- action sequence timestamps；
- 可选 `fixed_head_mask` → 16×16 token keep mask；
- policy metadata 契约。

保持官方核心不变的部分：Pi0.5 flow matching、PaliGemma 主体、action expert、标准 `sample_actions`、normalization/unnormalization 和 Orbax checkpoint。

不要用 `legacy/pre_task487_20260813` 下的旧 sync/RTC client 替代当前正式入口。旧 RTC 代码不是完整 model-side RTC，且旧 client 是单臂/两相机或旧键名，不符合 Task487 契约。

## 7. 精确的三相机契约

训练模型的输入顺序：

1. `cam_head` ← `observation.images.head_main`
2. `cam_left_top` ← `observation.images.left_hand_up`
3. `cam_right_top` ← `observation.images.right_hand_up`

最后一次人工逐路核对后，Thor sender 到本地接收端的映射记录为：

| 模型输入 | 物理源 | 本地 video/meta 端口 | UmiEnv observation key |
|---|---|---|---|
| `cam_head` | `/dev/video0`, right head | `5001 / 6001` | `camera4_rgb` |
| `cam_left_top` | `/dev/video3`, left top | `5002 / 6002` | `camera2_rgb` |
| `cam_right_top` | `/dev/video5`, right top | `5004 / 6004` | `camera0_rgb` |

这里“设备号”和“接收端口尾号”不是同一个编号，不能按 `5002 == video2` 推断。证据保存在：

```text
offline_eval/task487_20260814/live_hold_150549/metadata.json
```

尽管用户最后看图确认过映射，接手后若继续排查“不找茄子”，仍应先生成一次带大字标签和时间戳的三视图 montage，再由现场确认。不要只信变量名。

相机输入要求：

- RGB `uint8`；
- 当前接收图 640×512；
- policy 侧最终变换到 224×224；
- 两帧状态历史；
- camera age 默认不超过 0.25 s；
- 三路 capture skew 默认不超过 0.05 s。

`[UmiEnv] Found 0 cameras v4l_paths` 在 Thor 网络相机模式下正常，不表示没有相机。

## 8. 状态、动作和坐标系契约

每臂 10D：

```text
xyz3 + rotation6d6 + gripper1
```

双臂顺序：

```text
right_pose9 + right_gripper1 + left_pose9 + left_gripper1
```

细节：

- `rotation6d` 是旋转矩阵前两行按行展开；
- 模型状态和动作中的 gripper 单位是 rad；
- 真机 ROS gripper 对外状态/命令单位是 degree；
- client 输入模型前 degree → rad；
- 模型输出后 rad → degree；
- 训练动作原始为绝对下一状态；
- `UMIBimanualInputs` 将目标变成相对当前 EEF 的 body-frame SE(3) delta；
- 推理端 `body_actions_to_robot_targets` 再组合成 UmiEnv 的 14D 绝对目标：右 pose6+grip、左 pose6+grip。

右臂训练数据曾表达在左臂 base frame，部署端使用 `contract.py` 内硬编码的 `T_LEFT_FROM_RIGHT` 做转换。现有单测证明代数 round trip 一致，但尚不等于真实标定绝对正确。若时间轴修复后仍出现稳定的方向/位置偏差，应把这个外参和真实同姿态数据做数值核对，而不是再改 scheduler。

## 9. 当前正式真机入口

Server：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_server.sh \
  checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999 \
  8000
```

当前 server 已在运行，重复启动前先检查 8000 端口。

观察/干跑 client（不会发布模型动作，但会连接传感器/控制器）：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client.sh vegetable 127.0.0.1 8000
```

安全单轮真机 client：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

连续模式：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute --continuous
```

在当前时间轴问题修复前，不要再使用连续模式做真机测试。

按键：

| 键 | 行为 |
|---|---|
| `r` | 仅在 HOLD：Marvin HOME，完成后打开两夹爪并验证均 ≥30° |
| `d` | 从 HOLD 开始一轮模型控制；夹爪未准备好时拒绝启动 |
| `s` | 进入 HOLD、清空软件和物理轨迹 |
| `Ctrl+C` | 发 HOLD 后退出 |

`r` 是阻塞等待 HOME，日志会出现约 4–9 s 的 `Control tick overrun`。由于 `r` 只能在 HOLD 使用，这个 overrun 本身不是 ACTIVE 抖动根因。

## 10. 已经修复的问题

### 10.1 从 UR 路径切换到天机/Marvin

旧 client 尝试连接 `192.168.5.254`、`192.168.3.244` 的 UR 控制器，和当前真机不符。正式 `task487_client.py` 现在使用：

```text
robot_ip=[None, None]
robot_type="Marvin"
```

并通过本地 ROS/Mink bridge 控制。

### 10.2 启动/HOLD/HOME 状态机

- 启动停在 HOLD，不自动归位；
- `r` 明确 HOME；
- `d` 明确 ACTIVE；
- `s` 和 Ctrl+C 清空未来轨迹并 HOLD；
- 单实例文件锁 `/tmp/task487_client.lock` 防止两个 client 同时发目标；
- action queue 耗尽也会镜像到物理 HOLD。

### 10.3 过期 action

当前 scheduler 会按 observation wall time 计算每个动作时间，去掉已经过期或不能按 `time_is_new=True` 分发的 waypoint。旧同步 client 中“算了 is_new 但仍发送全部 action”的问题不再是正式入口的问题。

### 10.4 chunk 后半段被持续覆盖

HOLD 离线复核发现：夹爪全开时，Task487 右臂 action 0–9 位移很小，真正前移可能主要出现在 action 10–19。旧逻辑每五步重推并只保留五步，会一直覆盖有效后半段。

当前改为：

- 25 Hz waypoint；
- 每执行 5 步发起一次推理，约 5 Hz inference；
- `commit_steps=10`，先保留旧 chunk 的半段；
- `blend_steps=5`，在重叠区平滑接管。

### 10.5 夹爪假零点与开爪硬限位

原生 velocity auto-zero 只看力矩，夹爪仍在运动时也可能误判“捏紧”，造成错误零点。已修改：

```text
/home/simpleai/Code/mjm/eval_mink/ros2_ws/src/x3arm_can/src/x3arm/can/socket/gripper_component.cpp
```

现在要求同时满足：

```text
torque_ema > threshold AND velocity < stall_velocity
```

并在 `x3arm_can/CMakeLists.txt` 添加 `Threads::Threads` 后完成重编译。

右夹爪经人工移回小角度后重新归零，5°闭环测试正常。当前 client 保留模型 0–35° 契约，但把物理 full-open 配成 32°，留 3°硬限位余量：

```text
gripper_deg_open=35.0
gripper_open_rad=-rad(32.0)
```

最近一次 `r` 的实际反馈：右 30.2°、左 30.3°，已通过准备检查。

### 10.6 相机启动和失败关闭

早期出现过 Thor 无帧后 `get_obs()` / `end_episode()` 的二次 `AssertionError`。当前正式路径已能等待三相机恢复并正常进入 warmup；若相机 age/skew 超限，client 会 HOLD，而不是继续发动作。

## 11. 离线评测结果

完整报告：

```text
offline_eval/task487_20260814/REPORT.md
```

原始结果包括：

```text
task487_eval_10000.json
task487_eval_20000.json
task487_eval_29999.json
task487_behavior_29999.json
live_hold_150549/
```

### 11.1 训练分布内拟合

三个 task 合并：

| checkpoint | 区间 | 右臂平移误差 | 右臂零动作基线 | 左臂平移误差 | 左臂零动作基线 |
|---:|---|---:|---:|---:|---:|
| 10k | first5 | 2.452 mm | 7.559 mm | 0.830 mm | 2.464 mm |
| 20k | first5 | 1.634 mm | 7.559 mm | 0.620 mm | 2.464 mm |
| 30k | first5 | **1.299 mm** | 7.559 mm | **0.503 mm** | 2.464 mm |
| 30k | full20 | **2.395 mm** | 27.794 mm | **0.920 mm** | 9.435 mm |

30k first5 方向与真值一致率约右臂 97.9%、左臂 100%。这说明训练和基本动作语义没有完全错位。

限制：全部 230 episodes 都参与训练，所以这是 fit/contract 诊断，不是未见场景成功率。

### 11.2 随机性与跨 chunk 一致性

同一观测重复三次，30k first5 对三次均值的平移偏差：

| 手臂 | 均值 | P95 |
|---|---:|---:|
| 右臂 | 0.416 mm | 1.465 mm |
| 左臂 | 0.229 mm | 0.679 mm |

相隔 5 帧/0.2 s 的相邻观测：

| 指标 | 右臂 | 左臂 |
|---|---:|---:|
| 新旧 chunk 重叠目标误差均值 | 2.273 mm | 1.588 mm |
| 新旧 chunk 重叠目标误差 P95 | 8.735 mm | 5.282 mm |
| 五步接管点跳变均值 | 1.102 mm | 1.275 mm |
| 五步接管点跳变 P95 | 4.448 mm | 3.526 mm |

即使固定相同扩散噪声，接管 P95 改善也有限，所以不是单纯随机 seed 问题。

### 11.3 输入敏感性

模型不是在无视图像、机械复读固定轨迹：

| 输入改动 | 右臂 first5 变化均值 | 左臂 first5 变化均值 |
|---|---:|---:|
| 头图全黑 | 0.956 mm | 0.595 mm |
| 两路腕图全黑 | 1.648 mm | 1.367 mm |
| 三路图全黑 | 2.316 mm | 1.926 mm |
| 左右腕图交换 | 1.106 mm | 0.961 mm |
| vegetable/fruit 语言互换 | 0.233 mm | 0.229 mm |

结论：模型使用三路视觉，但腕图影响大于头图；语言条件较弱，模型更依赖视觉、状态及任务场景关联。

### 11.4 真机 HOLD 复核

在不发布任何机器人命令的条件下采集过 6 s、5 Hz、30 组真机观测。发现：

- 夹爪接近闭合时，模型会判断任务阶段错误并可能选择错误手臂；
- 把同一观测的夹爪状态离线改成 35°后，模型切回右臂抓取方向；
- 真机视觉与训练样本存在明显桌面材质、相机姿态、盘子和茄子布局差异。

夹爪起点现已修复，但视觉域差异仍未解决。

## 12. 当前 scheduler 与底层控制器

当前 `SchedulerConfig`：

| 参数 | 值 |
|---|---:|
| `control_hz` | 25 Hz |
| `request_every_steps` | 5（约 5 Hz 推理） |
| `commit_steps` | 10 |
| `blend_steps` | 5 |
| `dispatch_lead_s` | 0.05 s |
| 最大相邻平移 | 30 mm |
| 最大相邻旋转 | 0.12 rad |
| 最大 live 平移跟踪误差 | 50 mm |
| 最大 live 旋转跟踪误差 | 0.30 rad |

底层 RosTargetController：

- tick 100 Hz，即 10 ms；
- 实际生效平移限速 0.05 m/s，即每 tick 最大 0.5 mm；
- 实际生效旋转限速 0.20 rad/s；
- Mink/Marvin sender 1000 Hz。

“别人建议 5 Hz”需要拆开理解：

- 当前 **推理请求已经约 5 Hz**；
- waypoint 仍是训练一致的 25 Hz；
- 底层 servo 是 100/1000 Hz；
- 直接把 25 Hz 动作逐点当成 5 Hz 执行会把 `+0.04...+0.80 s` 的模型时间语义改坏；
- 如果要降发送率，应在保持物理时间含义的前提下重采样/时间缩放，而不是简单改 `control_hz=5`。

## 13. 最新真机故障与根因

最近一次日志：

```text
HOME complete; state remains HOLD
Grippers ready at right=30.2deg left=30.3deg
ACTIVE: beginning a fresh inference round
chunk accepted=16 expired=4 ... queue=16
双臂目标预装完成 → FSM 3 (ACTIVE)
chunk accepted=16 ... queue=16
chunk accepted=17 ... queue=17
chunk accepted=18 ... queue=18
HOLD: FSM 2 已发，清空旧轨迹
ERROR Live target rejected; HOLD:
action[0] right tracking error 0.052022m exceeds 0.050000m
```

同一段 raw model intent 的右臂 first5 幅度在相邻请求间约为 1.1 mm、8.3 mm、3.2 mm、0.9 mm，说明新旧预测本身也在变化。

### 13.1 已由代码证明的时间轴错位

上层 `RollingScheduler`：

1. 以 observation time + `(1..20)*0.04` 生成固定 wall-clock target time；
2. 到 nominal time 就把 waypoint pop 掉；
3. 每 5 个 nominal waypoint 请求新 chunk；
4. 用当前真实 EEF 检查队首 target 的 tracking error。

底层 `PoseTrajectoryInterpolator.schedule_waypoint()`：

```text
duration = requested_time - previous_tail
duration = max(duration,
               position_distance / max_pos_speed,
               rotation_distance / max_rot_speed)
actual_tail_time = previous_tail + duration
```

也就是说，只要模型目标按原始时间到不了，底层会把完成时间延后。`ros_target_interpolation_controller.py` 已正确把 `last_waypoint_time` 更新为插值器真实 tail，但这个真实 tail 没有回传给上层 scheduler。

结果：

```text
上层认为动作已经按 25 Hz 执行
        ↓
继续 pop / 请求 / 拼接更远的未来目标
        ↓
底层仍在限速追赶之前的目标
        ↓
队首计划逐渐领先真实 EEF
        ↓
达到 52 mm，安全检查 HOLD
```

这是当前最先要修的控制根因。放宽 50 mm 只会允许计划和真机继续分离，不是修复。

### 13.2 可能叠加的 sawtooth

底层收到一个 requested time 早于当前真实 trajectory tail 的新 waypoint 时，会从当前时刻 trim/rebuild 未来插值轨迹。连续重推理又存在几毫米接管偏差，因此重复 trim/rebuild 可能表现为往复、顿挫或方向反转。需要通过 waypoint/event log 对齐 `requested_wall_time`、`pose_interp.times[-1]`、真实 EEF 和 scheduler queue 后确认幅度。

### 13.3 为什么暂时不能判定模型不会抓

- 30k 在训练分布内拟合健康；
- 视觉敏感性存在；
- 夹爪起点刚修复；
- 当前控制仅运行约 1 s 就因时间轴错位 HOLD；
- 真机场景和训练画面仍有域差异。

所以当前证据只能说明“模型 + 在线调度 + 当前场景尚未成功”，不能单独证明 checkpoint 无效。

## 14. 下一位 agent 的建议执行顺序

### P0：保持真机安全

1. 确认没有 `task487_client.py`；
2. 确认 Mink FSM 是 HOLD；
3. 在完成离线仿真和单测前不要按 `d`；
4. 不要提高 0.05 m/s、0.20 rad/s；
5. 不要提高 50 mm tracking threshold；
6. 不要同时启动第二套 client、Mink 或 gripper node。

### P1：统一 scheduler 与物理限速时间轴

推荐先做可重复的离线实现，不直接上真机：

1. 在 scheduler 层引入与底层完全一致的速度可达性/时间缩放；
2. 对每个 model chunk 保留路径几何，但把 target timestamps 延长到 0.05 m/s、0.20 rad/s 可达；
3. scheduler 的 pop、request cadence、过期判断和 queue merge 必须基于这个实际执行时间，而不是原始 nominal time；
4. 或者让底层 controller 把 `actual_tail_time` 明确回传给上层；二者选一种，不能各自维护两条时间线；
5. 新 chunk 应相对于实时 observation/EEF 重规划，并在真实可执行时间上与 committed prefix 拼接；
6. queue 必须有明确上限，不能在长时间运行中 16→17→18→持续增长；
7. 对旋转用 SLERP/SE(3) 路径时间缩放，不要逐元素裁剪 rotvec；
8. 夹爪可以独立限速，不要让手臂时间缩放破坏绝对夹爪目标。

建议新增测试：

- 目标每 40 ms 前进 10 mm、物理上限 0.05 m/s 时，实际时间应自动延长且 scheduler 不提前 pop；
- 连续 100 次、每 5 步重推，queue 长度有界；
- 所有已发送 waypoint 的物理速度可达；
- 新旧 chunk 拼接没有平移/旋转反向尖峰；
- live feedback 落后时重新时间缩放，而不是放宽 tracking threshold；
- HOLD/恢复后旧 trajectory 不复活；
- 同一目标经时间缩放只改变速度，不改变几何终点。

### P2：先做 observation-only/HOLD 复核

修复后先：

1. 跑全部 unit tests；
2. 用记录的 `live_hold_150549` 和模型输出离线回放 scheduler；
3. 在真机 HOLD 下只收观测，不发动作；
4. 记录 raw model target、速度缩放后 target/time、底层 actual tail、真实 EEF；
5. 生成三相机带标签 montage，再由现场确认；
6. 比较当前 HOME 状态与数据集中最近邻状态。

### P3：低风险真机验证

仅在 P1/P2 通过后：

1. 使用不带 `--continuous` 的五 waypoint 单轮；
2. `r`，确认 HOME 和两夹爪 ≥30°；
3. 场景中茄子和右侧粉色盘尽量复现训练布置；
4. `d`；
5. 核对计划与真实 EEF 的误差是否保持有界；
6. 单轮平滑后才启用连续模式；
7. 连续模式先限时 2–3 s，不要直接长时间运行。

### P4：控制稳定后再判断视觉/权重

如果时间轴修复后仍不抓：

1. 比较训练三视图与真机三视图；
2. 做头图/腕图逐路遮挡和交换实验；
3. 验证 right/left camera 和 right/left state 没交叉；
4. 核对 `T_LEFT_FROM_RIGHT`；
5. 查找真机 HOME 最近邻训练帧；
6. 用相同初始物体布局做多次闭环；
7. 若输出离线就无语义，优先补当前场景数据或建立 held-out validation，不要继续调控制器。

## 15. 测试状态

2026-08-14 16:09 使用本地 `openpi` conda 环境运行：

```bash
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi
export PYTHONPYCACHEPREFIX=/tmp/pi05-handoff-pycache
export PYTHONPATH="$PWD/universal_manipulation_interface_ur:$PWD:$PWD/openpi-official/src"
python -m pytest -q tests_task487
```

结果：

```text
27 passed in 2.34s
```

测试覆盖：

- 三相机/状态/action contract；
- body-frame SE(3) round trip；
- gripper rad/degree 映射；
- scheduler 过期点、commit、blend、安全截断和 HOLD；
- worker；
- UmiEnv FSM。

尚未覆盖的关键点就是“上层 scheduler nominal time 与底层限速后的 actual tail time 一致性”。这是下一轮必须补的测试。

## 16. 重要文件索引

正式入口：

```text
run_task487_server.sh
run_task487_client.sh
task487_client.py
```

Task487 contract / worker / scheduler：

```text
task487_runtime/contract.py
task487_runtime/worker.py
task487_runtime/scheduler.py
```

真机环境和控制器：

```text
universal_manipulation_interface_ur/umi/real_world/umi_env.py
universal_manipulation_interface_ur/umi/real_world/ros_target_interpolation_controller.py
universal_manipulation_interface_ur/umi/common/pose_trajectory_interpolator.py
universal_manipulation_interface_ur/umi/real_world/ros_gripper_controller.py
```

OpenPI UMI 适配：

```text
openpi-official/src/openpi/policies/umi_policy.py
openpi-official/src/openpi/training/config.py
openpi-official/src/openpi/training/data_loader.py
openpi-official/src/openpi/models/model.py
openpi-official/src/openpi/models/pi0.py
```

测试：

```text
tests_task487/test_contract.py
tests_task487/test_gripper_contract.py
tests_task487/test_scheduler.py
tests_task487/test_umi_env_fsm.py
tests_task487/test_worker.py
```

离线证据：

```text
offline_eval/task487_20260814/REPORT.md
offline_eval/task487_20260814/live_hold_150549/
offline_eval/task487_20260814/task487_eval_29999.json
```

其他：

```text
TASK487_RUNTIME.md
UMI_HEAD_VIEW_DOMAIN_GAP_RESEARCH_PLAN.md
geometry_mask/
../dual_arm_urdf_manual_calibration_pipeline_20260811/
../umi_mask_pipeline_server_handoff_20260813/
```

## 17. 容易误判但其实正常的日志

| 日志 | 解释 |
|---|---|
| `Connect robot: None None` | Marvin ROS bridge 模式，不是 UR IP 直连，正常 |
| `Connect gripper ... can_if=None` | client 通过 ROS gripper controller，CAN 由独立节点占用，正常 |
| `[UmiEnv] Found 0 cameras v4l_paths` | 使用 Thor 网络相机，没有本地 V4L，相机随后显示 `3 cameras ready` 即正常 |
| JAX `Unable to initialize backend rocm/tpu` | NVIDIA CUDA 环境下可忽略 |
| 首次 warmup 约 10 s | 首次 JIT；后续约 90–120 ms |
| `Control tick overrun 4–9 s` 紧跟 HOME | `r` 在 HOLD 中阻塞等待归位/开爪，当前实现预期如此 |
| `expired=4` | 推理约 100 ms，25 Hz 下前几个目标已经过期，被正确删除 |

真正异常：

- queue 持续增长；
- tracking error 随运行时间增加；
- active 时重复 trim/rebuild 引起方向反转；
- camera age/skew 超限；
- gripper 未达到 ≥30°仍启动；
- 两个 client 或两个 gripper node 同时存在。

## 18. 不要做的事情

- 不要把 tracking error 从 50 mm 继续调大；
- 不要为了“不报错”删除 unsafe chunk / live target 检查；
- 不要直接把 waypoint 频率从 25 Hz 改成 5 Hz 而不做时间语义重采样；
- 不要用旧 RTC 脚本冒充当前 Task487 RTC；
- 不要在当前后端运行时再开第二套 Mink 或夹爪驱动；
- 不要把 `Connect robot: None None` 改回 UR IP；
- 不要在控制时间轴未修复时用连续真机动作来评估模型是否会抓；
- 不要把 no-mask Task487 基线的结果当作 mask 方法实验结果；
- 不要把凭据写进本文件或代码仓库。

## 19. 最终判断

当前链路不是“从头全错”，而是已经完成了训练契约、模型加载、视觉输入、Marvin 接入、夹爪起点和基本安全状态机的大部分工作。30k checkpoint 在训练分布内是健康的。

现在最需要解决的不是更多兜底，也不是放宽阈值，而是一个具体的工程契约错误：**模型/scheduler 使用 nominal 25 Hz 时间，底层限速控制器执行的是被延长的 actual time，但两者没有同步。** 修好这一点并通过有界队列和离线回放测试后，才能公平判断剩余的跨 chunk 模型不一致和真机视觉域差异。

# 2026-08-14 17:xx 整块执行 + RTC 补块

真机在加入 5 步 cross-fade 后已有抓取趋势，但仍明显往复。原因是旧调度每完成
5 个动作就重新生成/替换后缀；物理限速后的机械臂尚未沿当前 chunk 取得足够位移，
规划方向便可能再次改变。cross-fade 只能减小接缝跳变，不能消除这种高频反向重规划。

现改为：

- 一次提交完整 20 步 action chunk；
- 完成 15 步或安全队列仅剩 5 步时才发起下一次推理；
- 剩余至多 5 步作为 RTC hard prefix，新生成 suffix 接在其后；
- 单步平移离群阈值 35 mm、旋转 0.12 rad；物理限速和 50 mm tracking watchdog 保持不变；
- 若远端 tail 不安全，截取安全段，并在只剩 5 步时提前补块。

离线测试：`31 passed`，包括短安全 chunk 提前补块和连续 100 次 RTC 有界滚动。

17:32 真机在 RTC 接缝出现 `31.022 mm > 30 mm`。该值仅比数据集相邻动作
99.9 百分位 29.3 mm 高 1.7 mm，且物理限速会对它独立重定时，因此把模型离群
阈值留出余量至 35 mm；这不改变物理速度保护。随后按现场要求把臂速从
0.05 m/s、0.20 rad/s 温和降低 20% 至 0.04 m/s、0.16 rad/s。
# 2026-08-14 16:46 RTC 接缝修复

连续运行日志中第二个 chunk 已正确保持 9 步 RTC prefix，但生成 suffix
在接缝处被安全检查截断，下一轮出现
`action[4] right translation step 0.034779m > 0.030000m`。根因是最初实现
只在每个 flow step 硬锁 prefix 数值，却遗漏本地
`realtime-vla-v2-main/server/pi05rtc_infer.py` 的另一半语义：prefix token 的
AdaRMS 时间条件必须固定为 `t=0`。

已补齐：

- `openpi-official/src/openpi/models/pi0.py` 为 prefix/suffix 构造逐 token
  timestep，prefix=`0`、suffix=当前 flow timestep；
- `openpi-official/src/openpi/models/gemma.py` 支持 `[B,T,D]` 的逐 token
  AdaRMS condition；
- 保留 hard inpainting、30 mm/0.12 rad 安全拒绝和 rolling chunk 合并；
- `offline_eval/verify_task487_rtc.py` 增加 prefix→suffix 接缝检查。

验证结果：`tests_task487` 为 25 passed；30k checkpoint 相邻观测回放（旧
chunk offset=4、prefix=10）前缀误差约 `1e-6`，接缝最大为
`4.14 mm / 0.0136 rad`，明显低于安全阈值。修正后的 server 已在
`0.0.0.0:8000` 运行；旧 continuous client 及其子进程已停止，真机不会
自动动作。
# 2026-08-14 16:50 真机连续 RTC 验证通过

加载 controller-setpoint 进度判定和“已提交 RTC prefix 不再相对
`ActualTCPPose` 重验”两处修复后，真机连续运行从 16:50:30 至至少
16:50:43：共 32 次 chunk merge，32 次均为 `unsafe_tail=0`；其中 RTC
prefix 长度 10/9/8 分别出现 18/9/4 次（首 chunk 无 prefix）。期间无 chunk
接缝拒绝、无 tracking-error HOLD，推理约 90--106 ms，队列稳定在 18--20。

16:45 的 `Executed target tracking failed` 属于旧版用 `ActualTCPPose` 推进
RTC action index；16:49 的 `Chunk rejected action[0] ... 0.062591m` 属于只修了
advance、但 merge 仍把已验证旧 prefix 和滞后物理反馈重比较的中间版本。16:50
日志已证明两者均已消除。所有 action 单步、RTC 接缝及 fresh-chunk 相对真实反馈
安全阈值保持不变。

# 2026-08-14 18:xx 桌下事故端到端复核（覆盖 17:xx 整块执行结论）

右臂连续负 Z 并伸到桌下后，已停止 Task487 client。使用同一个 30k 服务、
录制 HOLD 三相机与状态做纯离线合成闭环：执行 15/提交 20/RTC 5 时，
640x512 和 224x224 两种图像输入均为 3/3 跌破安全平面；执行 5/提交 10/RTC 5
降为合计 1/6，且唯一越界位于新 chunk 的未提交 suffix。

因此 17:xx 的完整 20 步预装方案已撤销，正式参数恢复为：

- `request_every_steps=5`、`commit_steps=10`、`replan_remaining_steps=5`；
- 保留 RTC hard prefix 和 5 步 suffix handoff blend；
- 三路图像改为训练一致的 224x224，contract 拒绝其他尺寸；
- 双臂 TCP floor 为 `max(0.60m, ACTIVE起点z - 0.12m)`，越界在下发前截断/HOLD；
- 物理限速保持 0.04 m/s、0.16 rad/s。

数据、训练、臂顺序、相机顺序和 25 Hz 时间标签未发现整体错位。模型状态/action
是 EEF body-frame 相对量，本身看不到绝对桌高；当前视觉与训练数据又明显出域，
所以 workspace guard 是必需的。`T_LEFT_FROM_RIGHT` 的固定左乘在相对变换中
数值抵消（1000 组随机 SE(3) 最大误差 `7.13e-10`），不是此次持续下探根因。

完整报告：`TASK487_END_TO_END_AUDIT_20260814.md`。回归测试 `36 passed`。
尚未再次执行真机 continuous；下一步只能 observation-only 后做默认五 waypoint
单轮验证。

# 2026-08-14 19:05 左头相机根因修复（覆盖上一段相机顺序结论）

训练 `head_main` 实际来自 `cam_head_left/5000`，而旧部署使用右头 5001；同时旧部署
把 640x512 直接拉伸到 224x224。client 已切到左头 5000，并对三路图像执行训练一致
的中心方裁剪再缩放。Thor 当前由 tmux 会话 `task487-headleft` 临时恢复 5000/6000；
主配置也已改为左头启用、右头禁用，待交互式 sudo 重启服务后替代旁路。

5000 实测稳定收到左头画面，桌面、茄子和粉盘均在 224x224 模型视野内。不带
`--execute` 的两轮在线推理前十点意图仅约 0.3--1.1 mm、0--0.2 度，错误右目输入下
的 20--60 mm 无规则往复暂未复现。诊断 client 已正常退出，机械臂保持 HOLD；测试
结果为 `37 passed`。下一步只做 `--execute --max-waypoints 10` 有界真机验证。

# 2026-08-15 15:xx Task487 完整轨迹回放与 Marvin 坐标根因最终确认

本节是目前最新结论，覆盖本文早先把异常运动主要归因于 scheduler、执行频率或
checkpoint 的推断。那些因素仍可能影响连续闭环质量，但这次实验确认了一项更基础、
可单独复现的坐标契约错误。

## 1. 完整真机数据回放

从云端 Task487 数据集下载了 file-002 的 episodes 66--71 到：

```text
datasets/task487_cloud_sample
```

新增独立回放入口，不经过模型、相机、归一化或 RTC，只验证“数据绝对轨迹 → Marvin
真机”的硬件控制链：

```text
task487_trajectory_replay.py
run_task487_trajectory_replay.sh
```

入口默认回放 episode 66 全集、包含双臂姿态和夹爪、以 5 Hz 执行。按键为：

- `p`：移动到数据记录的绝对起始姿态；
- `d`：执行完整轨迹；
- `s`：立即 HOLD。

第一次完整回放时，轨迹形状和相邻运动看起来正确，但绝对起始姿态明显异常；用户
确认数据视频中的起始姿态本身是正常的。这将故障范围直接缩小到绝对姿态解码，而
不是数据录制、机械臂硬件或插值控制器。

## 2. 根因：把 UR 的工具轴矩阵用于 Marvin

本地公共契约原先沿用了 UR 版本：

```python
R_M2R_OLD = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0],
])
```

而云端已在其他 Marvin 机器跑通的 B1 部署使用：

```python
R_M2R_MARVIN = np.array([
    [0,  0, 1],
    [0, -1, 0],
    [1,  0, 0],
])
```

两种解码结果之间恰好存在固定 `90°` 旋转。旧矩阵因此产生以下现象：

- 数据视频起点正常，但回放到 Marvin 后绝对腕部姿态怪异；
- 每帧都带同一个固定偏转，所以相对轨迹仍看起来连续、形状近似正确；
- 在线推理时 `state` 姿态编码错误；
- body-frame 平移和旋转动作被映射到错误工具轴，模型意图可能表现成横移、反向纠正、
  往复或无逻辑运动。

正式修复位于：

```text
task487_runtime/contract.py
```

`R_M2R/R_R2M` 已统一为 Marvin/B1 约定。回归测试
`tests_task487/test_contract.py::test_marvin_model_tool_axis_contract_matches_task487_b1_runtime`
锁定该矩阵，防止再次从 UR 代码误复制。右臂基座变换 `T_LEFT_FROM_RIGHT` 本次没有
修改，它是另一项独立契约，不应与工具轴修复混为一谈。

## 3. 修正后完整回放证据

修正后同一 episode 66 完整回放由现场用户确认“没有问题”。量化报告：

```text
task487_logs/20260815_144300_trajectory_replay/tracking_report.json
```

关键结果：

| 指标 | 右臂 | 左臂 |
|---|---:|---:|
| 位置 RMS | 1.351 mm | 1.677 mm |
| 最大位置误差 | 7.969 mm | 11.171 mm |
| 旋转 RMS | 0.207° | 0.329° |
| 最大旋转误差 | 1.710° | 1.984° |

报告共有 9625 个反馈样本，`interrupted_by_s=false`。轨迹静态检查的最大速度为右臂
`0.072 m/s / 0.270 rad/s`、左臂 `0.091 m/s / 0.279 rad/s`，均在本次回放配置的
限制内。

这项结果可以确认：

1. Task487 记录的绝对双臂轨迹和视频是一致且可执行的；
2. Marvin + Mink + ROS target controller 能跟踪该轨迹；
3. 之前诡异起点的直接原因是错误的 UR/Marvin 工具轴矩阵；
4. 旧连续推理中“运动方向无逻辑”的一个主要根因也是同一坐标错误，而不能首先归咎
   于权重或简单地改成 5 Hz。

回放绕过了视觉模型，因此它**不能单独证明 checkpoint 已经学会分拣或抓取**；它证明
的是数据、绝对姿态解码和底层轨迹执行链现已打通。

## 4. B1 连续真机复测

B1 server 使用：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_b1_server.sh
```

默认监听 `8082`。连续客户端使用：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client.sh vegetable 127.0.0.1 8082 --execute --continuous
```

客户端必须在矩阵修改后重新启动；server 无需因这一 client-side 坐标修复而重启。
现场在重新加载修正后的公共契约后确认 B1 连续运动恢复正常，原先的坐标方向问题已
消失。随后现场进一步确认 B1 已实际完成抓取，故当前结论升级为：坐标修复后的
`B1 + Task487 三相机 + Marvin` 真机链路已获得至少一次抓取成功。这是功能性成功
验证；重复成功率、完整放置成功率和不同物体/初始位置的统计尚未完成，不能把单次
成功直接写成 benchmark 成功率。后续仍需保存每次任务视频、日志和成功/失败标签。

本次修改后的完整 `tests_task487` 结果为：

```text
44 passed
```

# 2026-08-15 夹爪重新归零与右侧开角标定

现场将两侧夹爪手动打开后，单独重启 `gripper_can_node`，启动时的
`auto_close_and_zero` 已在左右闭合机械端分别收敛并成功设置 RAM 零点。

标定时发现 `/Joint69/position_command` 和 `/Joint79/position_command` 的最终发布者
是 `target_mux`。只停止 `ros_target_bridge` 仍会由 mux 以约 10 Hz 重发旧目标，
使单次标定命令被覆盖；独立标定必须先暂停 mux，并在结束后恢复 mux 和 bridge。

右侧从旧端点 `-0.234485 rad` 逐步打开到驱动软件上限：

- `-0.610865 rad`（35°）可以到达，但稳态力矩持续约 `1.472 Nm`，不可作为运行端点；
- 内退 1°后的 `-0.59341195 rad`（34°）实际稳定在约 `-0.593384 rad`，稳态力矩约
  `0--0.105 Nm`；
- 因此正式右侧安全满开端点改为 `-0.59341195 rad`；
- 左侧继续使用独立端点 `-0.41949338 rad`。

`task487_client.py`、`task487_trajectory_replay.py` 和夹爪契约测试已同步更新，相关测试
为 `13 passed`。标定结束后 `gripper_can_node`、`target_mux` 和
`ros_target_bridge` 均已恢复，Task487 client 未运行；恢复 mux/bridge 后右侧仍保持
新端点，没有回跳到旧值。

# 2026-08-15 18:54 最新运行快照与后续调试结论

本节是当前最高优先级交接信息，覆盖前文中旧的 `8000/29999`、位置模式、右头相机、
旧夹爪端点和旧进程快照。历史分析保留用于追溯，但继续真机前应以本节为准。

## 1. 当前进程、安全状态和日志

截至 2026-08-15 18:54 CST：

- 当前模型是 B1：`stage2_b1_robot_jax/B1`，服务监听 `0.0.0.0:8082`，GPU0、
  `bfloat16`，服务端 metadata 已验证 `action_horizon=16`、RTC enabled；
- 后端在 tmux `marvin-impedance` 中运行，入口为
  `start_teleop_replay_impedance.sh` 包装不录包的 `start_teleop_replay.sh`；
- Marvin 启动日志明确确认 A/左臂和 B/右臂均为 `ArmMode.IMP_JOINT`，不是自动回退：
  `K=[8,8,8,8,6,6,6]`，`D=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]`；
- 当前只有一个主客户端。`pgrep` 会显示 6 个相同 `task487_client.py` 命令行，其中
  1 个是主进程，另外 5 个是它派生的推理/共享内存/相机工作进程，不是 6 套控制源；
- 主客户端日志目录：
  `task487_logs/20260815_184941_2291301`；
- 18:51:48 HOME 完成后夹爪报告右 `35.0°`、左 `30.1°`，随后进入 round 3；
- 18:54:24 客户端触发保护并向物理后端发送 HOLD：
  `left tracking error 0.050142m exceeds 0.050000m`；
- 当前 workspace height limit 仍按现场要求关闭；单步、物理限速、round 隔离、
  tracking guard 和 unsafe-tail 截断仍开启。

因此接手时不要看到 client 进程存在就假定机械臂仍在 ACTIVE，也不要直接按 `d`。
先看日志尾部、确认机械臂停稳，再决定是否 HOME 后重开新 round。

## 2. 当前本地推理与执行架构

```text
Thor 3路相机 (5000头左 / 5002左腕上 / 5004右腕上)
        │ UDP video + metadata
        ▼
task487_client.py / UmiEnv (25 Hz)
        ├─ 224x224 中心方裁剪，2帧图像历史
        ├─ pre_state + state，各20D
        ├─ Task UI：vegetable / fruit prompt
        ├─ 单 in-flight InferenceWorker
        └─ RollingScheduler + RTC + 安全检查
        │ WebSocket 127.0.0.1:8082
        ▼
B1 Pi0.5 JAX server (GPU0, bf16, horizon=16)
        │ 20D body-frame action chunk
        ▼
contract.py：坐标恢复 + 20D → 双臂绝对14D目标
        │ 每次下发一个按物理时间重定时的 waypoint
        ▼
左右 RosTargetInterpolationController (100 Hz)
        │ UDP :6010
        ▼
ros_target_bridge → /model/{left,right}_target
        ▼
target_mux（最终 /left_target、/right_target 唯一发布者）
        ▼
Mink IK (100 Hz, Link610/710 TCP, 腰锁定)
        ▼
marvin_bridge → DualSender 1000 Hz → 双臂 IMP_JOINT

反馈：Marvin关节 → /joint_states → FK/current_pose → UDP 6011/6012 → client
夹爪：client → UDP → mux → Joint69/79 command → gripper_can_node → can1
```

请求中的每臂状态布局为 `xyz3 + rotation6d6 + gripper1`，总计 20D；模型结果再恢复为
`右 pose6+gripper + 左 pose6+gripper` 的 14D 真机绝对目标。坐标公共契约位于
`task487_runtime/contract.py`，已使用 Marvin/B1 专用 `R_M2R`，不得再换回 UR 矩阵。

## 3. 两个权重与 action horizon

当前可用的两套服务不是同一个 horizon：

| 权重 | 服务入口/配置 | action horizon | 典型端口 |
|---|---|---:|---:|
| B1（当前） | `run_b1_server.sh` / `stage2_b1_robot_jax` | 16 | 8082 |
| Task487 30k `29999` | `run_task487_server.sh` / `pi05_umi_task487` | 20 | 8000（可改） |

客户端启动后读取 server metadata，并按服务端 horizon 动态创建 worker、queue 和 RTC
prefill；不需要为了 16/20 手工改客户端，也不能把 B1 输出强行补成普通 20 步后按
25 Hz 全执行。

Task UI 只切换数据集 prompt：

- `vegetable`：右臂把蔬菜放到右侧粉盘；
- `fruit`：左臂把水果放到左侧蓝盘。

它不切换 checkpoint。ACTIVE 中切任务已接入安全热切换：先物理 HOLD、清旧轨迹和
旧 inference round，再从实测位姿建立新 round 并恢复；旧 round 的异步结果会按
`round_id` 丢弃。权重切换仍应停止客户端并更换模型服务，不能在同一个模型进程内靠
任务窗口完成。

## 4. 当前 RTC/scheduler 基线与已回滚实验

当前正式基线参数：

- `control_hz=25`；
- 每物理完成 5 个 waypoint 触发重规划；
- `replan_remaining_steps=5`；
- `commit_steps=10`；
- RTC hard prefix 只包含已经发给低层控制器的连续 `_sent=True` 目标；
- `handoff_blend_steps=5`，平移、SO(3) 旋转和夹爪同步 cross-fade；
- 平移/旋转物理限速为 `0.02 m/s`、`0.08 rad/s`，scheduler 与低层插值器共享实际
  可执行时间轴；
- 单步保护 `35 mm / 0.12 rad`，物理跟踪保护 `50 mm / 0.30 rad`。

现场为解决“走10步退5步”和偶发掉高度，曾依次试过方向反转冻结、XY 时序融合等
额外策略。结果是往返动作没有可见改善，且一度出现下探不足、夹不起来及偶发高度
掉落。上述实验已全部回滚；当前没有 reversal guard、XY fusion 或额外旋转衰减，
保持原 RTC 基线。不要从旧对话或临时 patch 重新引入这些逻辑。

回滚后的完整测试命令（必须显式使用本仓库 PYTHONPATH，避免串到
`/home/simpleai/pi05-deploy` 旧副本）：

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi
export PYTHONPATH="$PWD/openpi-official/src:$PWD/openpi-official/packages/openpi-client/src:$PWD/universal_manipulation_interface_ur:$PWD"
PYTHONPYCACHEPREFIX=/tmp/task487-handoff-pycache python -m pytest -q tests_task487
```

2026-08-15 18:54 实测结果：`46 passed, 2 warnings`。两条 warning 是 SWIG 类型的
DeprecationWarning，不是测试失败。

## 5. “点头”现象的已确认层级

现场确认 B1 和 29999 都会出现末端“点头”，而其他机器观感不明显。当前日志已经把
故障层级分开：

- 点头对应的姿态变化在 `/left_target`、`/right_target` 中已经存在，是上层完整 6D
  目标的一部分，不是电机在稳定目标上自行振荡；
- 两次 ACTIVE 日志中，主要执行臂相对起始目标姿态的最大旋转分别约为右臂 `42°`、
  左臂 `67.6°`；
- Mink 当时条件数约 12--14，正常段 `trk_err` 约 3--8 mm，没有奇异点发散；
- `target_mux` 没有切入 VR 模式，TCP frame 两端均为 Link610/710，排除了重复控制源、
  VR 抢话题和 TCP/wrist frame 混用；
- 其他机器不明显，最可能是其部署对姿态锁定、衰减或跟踪带宽不同，不能据此断言
  两个 checkpoint 都损坏。

曾建议为旋转增量增加可配置比例做 A/B，但用户明确决定“别改了，就这样”。当前代码
没有加入姿态冻结或缩放。后续除非用户重新授权，不要为了点头擅自改 orientation
cost、旋转解码或 scheduler。

## 6. 夹爪当前正式标定

客户端保持模型侧 `0--35°` 契约，但两侧分别映射到独立的安全物理端点：

```text
right: -0.59341195 rad
left : -0.41949338 rad
```

右侧 `-0.610865 rad` 虽能到 35° 机械极限，但稳态力矩约 `1.472 Nm`，不可作为运行
端点。右侧内退 1° 后力矩恢复安全；不要为了让两侧物理弧度相同而覆盖此标定。

## 7. 位置/阻抗模式与当前启动命令

位置模式入口：

```bash
cd /home/simpleai/Code/mjm/eval_mink
./start_teleop_replay.sh
```

阻抗模式入口（当前使用，不录包）：

```bash
cd /home/simpleai/Code/mjm/eval_mink
REPLAY_BASE=start_teleop_replay.sh ./start_teleop_replay_impedance.sh
```

阻抗脚本初始化时会先短暂打印 `position_follow`，随后才执行阻抗切换。必须最终看到
以下两类日志才算成功，不能只看前面的 position_follow：

```text
A/左臂 关节阻抗 OK, mode=ArmMode.IMP_JOINT
B/右臂 关节阻抗 OK, mode=ArmMode.IMP_JOINT
```

若出现“阻抗切换失败，回退 POSITION_FOLLOW”，则实际不是阻抗模式。

从全停状态启动当前 B1 + 阻抗 + 客户端的三条命令：

```bash
# 终端1：B1模型服务
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_b1_server.sh

# 终端2：Marvin阻抗后端
cd /home/simpleai/Code/mjm/eval_mink
REPLAY_BASE=start_teleop_replay.sh ./start_teleop_replay_impedance.sh

# 终端3：Task487客户端
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client.sh vegetable 127.0.0.1 8082 --execute --continuous
```

客户端进入后仍需人工按 `r` 执行 HOME+OPEN，确认成功后按 `d` 才进入 ACTIVE；`s`
立即 HOLD。

## 8. 已知启动卡点：`ros2 daemon stop`

`start_teleop_replay.sh` 的 leftovers 清理后直接执行：

```bash
ros2 daemon stop
```

本机已多次在此永久卡住，终端只停在：

```text
[start_teleop_replay] killing leftovers...
```

此时机械臂节点尚未初始化。处理方式是先用 `ps/pgrep` 找到该次启动脚本的精确子进程
`ros2 daemon stop` PID，只终止这个子进程，让同一个父脚本继续；不要再启动第二套
后端。之前每次这样处理后脚本均能继续拉起 launch。脚本本身尚未加入 timeout，后续
可在明确维护窗口把 stop/start 包成有限超时，但当前不要在真机运行中修改或重启。

## 9. 关于训练降采样

当前 Task487 数据时间契约是严格 25 Hz：29999 的 20 步 horizon 覆盖 `0.8 s`；B1
的 16 步 horizon 在同一 25 Hz 语义下覆盖 `0.64 s`。训练时做 stride=2 时间降采样
可以减少相邻冗余、让每步位移更明显并扩大 horizon 覆盖时间，但会丢失接触和夹爪
细节，并改变 action 的真实时间尺度。

现有权重不能在推理侧简单“隔一个 action 执行一次”。若未来重新训练 12.5 Hz 版本，
必须同步修改训练 timestamps、norm stats、action horizon 解释、客户端调度周期和真机
插值时间，再做独立 A/B；降采样不是当前点头或往返动作的推理侧补丁。

# 2026-08-15 19:10 逐 chunk 诊断与 29999 切换

为定位连续执行中的动作回滚，客户端新增无控制副作用的逐 chunk 诊断记录。每次成功
合并或安全拒绝都会在本轮 `task487_logs/<run>/` 下写入：

```text
chunk_diagnostics.jsonl
policy_chunks/chunk_<sequence>_round<round>_<merge_time>.npz
```

JSONL 摘要记录左右臂旧 replaceable tail、模型 raw suffix、blend 后 suffix 的首段位移、
旋转和方向余弦，以及各路径内部的反向次数。NPZ 保留 raw targets、RTC request prefix、
blend 前后 suffix、merge 前后完整 queue、nominal/physical timestamps、sent mask、live
feedback、controller setpoint 和请求 TCP base，可将模型反向与 scheduler 接管分开复核。
诊断写入失败只记录异常，不改变安全状态机或控制输出。

相关实现：

```text
task487_runtime/diagnostics.py
task487_runtime/scheduler.py::diagnostic_snapshot
tests_task487/test_diagnostics.py
```

回归结果为 `48 passed, 2 warnings`。

B1 服务及旧客户端已安全退出，Marvin/Mink 阻抗后端未重启。Task487-only 30k 的
`29999` 已在 tmux `task487-server-29999` 中监听 `8082`，metadata 已确认 runtime 为
`pi05_umi_task487_v1`、`action_horizon=20`。新客户端位于 tmux
`task487-client-29999`，日志目录为：

```text
task487_logs/20260815_190957_3513449
```

客户端两种 JAX warmup 均已完成，三相机正常，当前停在 HOLD、夹爪 READY；仍需人工
按 `r` HOME+OPEN 后按 `d` 才会动作。

# 2026-08-15 19:24 香蕉夹持后迟迟不抬：根因与修复

用户在 29999 的第二轮实测中观察到左臂夹住香蕉后长时间不抬。对
`task487_logs/20260815_190957_3513449` 的控制反馈和逐 chunk NPZ 对齐后确认：

- 左夹爪闭合后，模型 raw suffix 在约 1 秒后的 horizon 内已经给出约 30 mm 抬升，
  因此不是 29999 权重完全没有学到抬升；
- controller target 与 ActualTCPPose 的抬升时间基本一致，20 mm 处两者只差约
  0.59 秒，因此也不是 Marvin 低层卡死；
- 旧客户端每个 25 Hz control tick 只调用一次 `pop_next()`。近静止夹持段的 waypoint
  也以约 25 Hz 完成，导致本应为 10 点的 committed window 无法提前建立；日志末段
  多次出现 `rtc_prefix=0, committed=0`，新推理持续替换尚未下发的后半段，抬升尾部
  因而发生饥饿。

修复保持正式参数 `execute 5 / preserve 5 / commit 10` 不变，只改变下发方式：

- scheduler 新增 `pop_batch()`，每次 chunk 合并后一次返回最多 10 个可提交 waypoint；
- 客户端将整批 action/timestamp 一次传给 `UmiEnv.exec_actions()`；
- inference pending 时仍不提交模型未见过的 replaceable old tail，HOLD、物理限速、
  workspace guard 和 RTC hard-prefix 规则均未放宽；
- `pop_next()` 保留为单点兼容包装。

新增回归覆盖“前 5 点近静止夹持、后段抬升”的轨迹：首次批量提交 10 点，完成 5 点
发起请求时 prefix 为 5；推理期间再完成 2 点后保留 3 点并补入 7 点，抬升目标进入
低层 committed window。完整结果为 `50 passed, 2 warnings`。

旧客户端已在 HOLD 下退出，29999 模型服务和 Marvin/Mink 阻抗后端没有重启。更新后的
客户端仍使用 tmux `task487-client-29999`，新日志目录为：

```text
task487_logs/20260815_192431_199471
```

两种 warmup、三相机及夹爪均正常，当前明确停在 HOLD。下一轮仍由现场人工按 `r`，
确认 HOME+OPEN 后再按 `d`；不要自动启动动作。
