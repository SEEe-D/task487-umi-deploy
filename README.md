# Pi0.5 UMI6 Deploy

本地推理 + 天机/Marvin 真机部署包。

> Task487 在 2026-08-19 的训练、mask、标定、离线对比和真机验证结论见
> [TASK487_TRAINING_REPORT_20260819.md](TASK487_TRAINING_REPORT_20260819.md)。

## 2026-09-05 部署更新

新增 author-sync 调度、四腕模型契约、夹爪/模型请求日志及双手 5°补偿 v2。
当前试用入口和 HOME 准备恢复见 [客户端运行说明](README_CLIENT_SYNC.md)。
补偿入口先保持左手，右手完成打开目标后再放行；这不等同于视觉确认物体落盘。
启动仍在 HOLD，按 `r` 归位并先张开解除旧夹爪锁存，再准备到 checkpoint 开度；
看到 `Grippers prepared` 后按 `d`。若仍持物，先托住，因为准备会张开。

```bash
bash run_task487_client_sync_gripper5.sh sorting 127.0.0.1 8000 --execute --continuous
```

运行该入口需现有 Marvin/Mink 后端和匹配的模型服务；外部后端安装说明见
[夹爪修复报告](reports/gripper-fix-20260905/REPORT.md)。权重、原始录像与运行日志仍在开发机。
203 项部署测试通过；新的实物释放与放置效果仍待现场复测。

## 部署架构

当前正式部署链路：

```
[GPU 机器]
run_task487_server.sh
        |
        | WebSocket :8000
        |
        v
[机器人 PC]
run_task487_client.sh
        |
        ├── Thor 相机输入
        ├── UMI observation 构造
        ├── Pi0.5 action 调度
        └── Tianji/Marvin Mink ROS bridge
                |
                v
             真机执行
```

机器人后端使用本地 Tianji/Marvin Mink ROS bridge，不使用 UR RTDE。

## 运行入口

### 1. 启动 Pi0.5 policy server（GPU机器）

```bash
bash run_task487_server.sh
```

server 负责：

- 加载 Pi0.5 checkpoint
- 接收 observation
- 执行策略推理
- 返回 action chunk

### 2. 启动真机客户端（机器人PC）

观察模式：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000
```

真实执行：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

## Observation Pipeline

输入：

- `cam_head`
- `cam_left_top`
- `cam_right_top`
- robot state
- prompt

处理流程：

```
Thor cameras
      |
      v
Observation
      |
      v
Pi0.5 inference
      |
      v
Action chunk
      |
      v
Tianji/Marvin controller
```

## Task487 Mask Runtime

启用 mask 配置时：

```
robot joint state
        |
        v
URDF forward kinematics
        |
        v
camera projection
        |
        v
robot mask
        |
        v
SigLIP token keep mask
        |
        v
visual attention filtering
```

mask 用于推理阶段屏蔽机械臂本体视觉区域，减少 embodiment gap。

## RTC Runtime

当前部署支持 action-prefill RTC：

- action chunk prefix 保留
- 异步重新推理 suffix
- handoff 时进行安全 blend
- 根据 server metadata 校验时间频率

## 测试

契约测试：

```bash
pytest -q tests_task487
```

离线验证：

```bash
python offline_eval/verify_task487_rtc.py
```

## 数据格式

输入：

- RGB images
- robot proprioception
- task prompt

输出：

```
actions: (action_horizon, 10)

xyz position
rotation 6D
gripper
```

## TODO

- [ ] 完善安全边界约束
- [ ] 增加碰撞检测
- [ ] 完善机器人状态诊断工具

## Training Information

- Model: Pi0.5
- Dataset: Task487 UMI data
- Runtime: Tianji/Marvin real robot deployment
