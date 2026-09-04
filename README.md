# Pi0.5 UMI6 Deploy

本地推理 + 天机/Marvin 真机部署包。

> Task487 在 2026-08-19 的训练、mask、标定、离线对比和真机验证结论见
> [TASK487_TRAINING_REPORT_20260819.md](TASK487_TRAINING_REPORT_20260819.md)。

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
