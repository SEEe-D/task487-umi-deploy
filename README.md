# Pi0.5 UMI6 Deploy

本地推理 + UR7e 真机部署包。

## 目录结构

```
pi05-deploy/
├── serve_umi6.sh         # GPU 机器: 启动 policy server
├── ur7e_main.py          # 机器人 PC: 主部署脚本 (10Hz policy loop)
├── ur7e_env.py           # 环境: 相机 + 夹爪 + 多进程控制器
├── ur7e_controller.py    # 多进程 RTDE 伺服控制器 (500Hz servoL)
├── infer.py              # 离线推理测试
├── checkpoint/           # -> ../pi05-ckpt-full-20000 (symlink)
│   ├── model.safetensors # 模型权重 (7GB)
│   ├── metadata.pt       # 训练元数据
│   └── assets/           # norm_stats
├── openpi-official/      # -> ../openpi-official (symlink)
└── README.md
```

## 真机部署架构

基于 UMI 官方 eval_real.py 架构:

```
[GPU 机器]                         [机器人 PC]
serve_umi6.sh                     ur7e_main.py
 └─ serve_policy.py                ├─ WebSocket → GPU policy server
      └─ WebSocket :8000  ←─────→ ├─ 10Hz policy loop:
                                   │   get_obs → infer → schedule_waypoints
                                   ├─ RTDEInterpolationController (独立进程)
                                   │   └─ 500Hz servoL + PoseTrajectoryInterpolator
                                   │       └─ 线性位置 + SLERP 旋转插值
                                   ├─ CameraCapture × 2 (cam_right + cam_head)
                                   └─ UMIGripper (串口)
```

关键设计 (来自 UMI 官方):
- **多进程 servoL**: 控制循环在独立进程以 500Hz 运行, 绕开 Python GIL
- **轨迹插值**: PoseTrajectoryInterpolator 在 waypoint 之间做平滑插值 (SLERP 旋转)
- **时间戳调度**: action chunk 的每个 action 绑定未来时间戳, 控制器平滑执行
- **延迟补偿**: 跳过推理期间已过期的 action, 只调度"未来"的
- **速度安全限制**: max_pos_speed / max_rot_speed 约束平移和旋转速度

## 环境安装

```bash
# 机器人 PC
pip install ur_rtde opencv-python scipy numpy

# GPU 机器 (或同一台机器)
cd openpi-official
pip install -e ".[dev]"
```

## 真机部署

```bash
# 1. 在 GPU 机器上启动 policy server:
bash serve_umi6.sh

# 2. 在机器人 PC 上运行:
python ur7e_main.py \
    --robot-ip 192.168.1.100 \
    --server-host <GPU_IP> \
    --server-port 8000 \
    --cam-right-id 0 \
    --cam-head-id 2 \
    --policy-freq 10 \
    --prompt "building blocks into box"

# 单机模式 (GPU 机器同时控制机器人):
python ur7e_main.py --robot-ip 192.168.1.100 --server-host localhost
```

## Task487 头部机械臂 mask

`pi05_task487` 固定使用三路图像：头部、左手 up、右手 up。当前跑通验证配置不启用 mask。
正式使用带 mask 数据训练时，将该配置的 `use_head_token_mask` 设为 `True`，双臂真机客户端会同步执行以下协议：

- UmiEnv 将两侧 `ActualQ` 插值到头部相机时间戳。
- 标定后的双臂 URDF 在完整头部画面上生成白色机械臂 mask。
- mask 转为 16×16 SigLIP token keep mask，在视觉 attention 中屏蔽机械臂区域。
- 启动时保存 `/tmp/pi05_umi_eval/geometry_mask_preview.png`，用于检查投影对齐。

使用自定义 checkpoint 启动：

```bash
bash run_pi05_client_bimanual.sh custom pi05_task487 /path/to/checkpoint "Pick Up Towel"
```

启用该配置时，缺少头部画面、任一侧 7 维关节角或几何标定文件都会直接停止推理。

## 离线推理测试

```bash
# 随机输入测试
python infer.py --test

# 真实图片推理
python infer.py \
    --cam-right right_wrist.jpg \
    --cam-head head_cam.jpg \
    --state 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1
```

## 数据格式

### 输入 (观测)
- `cam_right`: 右腕相机 RGB (256x256x3, uint8)
- `cam_head`: 头部相机 RGB (256x256x3, uint8)
- `state`: 20D float32 [right_pos(3) + right_rot_6d(6) + right_gripper(1) + left_zeros(10)]
- `prompt`: 任务描述字符串

### 输出 (动作)
- `actions`: (action_horizon, 10) — 绝对目标 EE pose
  - [:, 0:3]: 目标 xyz 位置
  - [:, 3:9]: 目标 6D 旋转
  - [:, 9]: 目标夹爪位置

## 配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy-freq` | 10 | Policy 查询频率 (Hz) |
| `--servo-freq` | 500 | servoL 控制频率 (Hz) |
| `--max-pos-speed` | 0.25 | 最大平移速度 (m/s) |
| `--max-rot-speed` | 0.16 | 最大旋转速度 (rad/s) |
| `--action-horizon` | None | 每次执行的 action 数 (None=全部) |
| `--action-latency` | 0.01 | 动作执行延迟补偿 (s) |
| `--max-steps` | 1000 | 每 episode 最大推理步数 |

## TODO

- [ ] UMIGripper: 实现真实串口通信 (目前为占位)
- [ ] HOME_POSE: 根据实际 UR7e 标定调整
- [ ] 工作空间安全边界 (workspace bounds)
- [ ] 碰撞检测 (table collision)

## 训练信息

- 模型: Pi0.5 (PaliGemma 3B + Gemma 300M action expert)
- 数据: UMI6 building-blocks-righthand (213 episodes, 全量训练)
- 训练: 20000 steps, batch_size=32, lr=2.5e-5
- Split eval MSE: 0.005610 (11 test episodes)
