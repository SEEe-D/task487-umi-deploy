# Task487 当前真机运行指令

当前配置：Task487 真机 `task487_3cam_nomask_30k/29999` 模型、端口 `8000`、Marvin
双臂关节阻抗、`vegetable` 任务。请按“终端 1 → 终端 2 → 终端 3”的顺序启动，
每条命令独占一个终端。

该权重使用旧真机数据契约：25 Hz、20 步动作块、三路相机、无 mask、`center_square`
图像处理，训练数据的全开标签约为 35°。当前 Marvin 的独立安全标定端点为右约 34°、
左约 24.0°，因此客户端会从服务端元数据切换到 25 Hz，并在 HOME 时准备
到这两个真实可达端点，而不会让左夹爪持续硬顶不可达的 35°。客户端也不会沿用 12.5 Hz
masked 权重的时序、`resize_with_pad` 或 1° 夹爪边界。

## 终端 1：启动模型服务

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
TASK487_POLICY_CONFIG=pi05_umi_task487 \
  bash run_task487_server.sh \
  checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999 8000
```

上述命令加载：

```text
checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999
```

并监听端口 `8000`。

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

`--show-processed-cameras` 的第一行显示实际送入策略的三路 `224×224` RGB；该无 mask
权重的第二行会明确显示 `MASK DISABLED`。窗口在 HOLD、dry-run 和 ACTIVE 状态下均按
控制循环持续刷新，只读取请求内容，不会修改模型输入。若只想看三路 RGB，可改回
`--show-cameras`。

客户端启动后的按键顺序：

1. 按 `r` 执行 HOME，并将夹爪准备到硬件安全端点（右约 34°、左约 24.0°）。
2. 人工确认双臂归位、夹爪到达上述开度且周围安全。
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
