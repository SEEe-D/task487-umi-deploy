# UMI + Pi0.5 完整流水线文档

> 从原始数据采集到模型训练到真机部署的全流程

---

## 目录

1. [总览](#1-总览)
2. [数据采集与格式](#2-数据采集与格式)
3. [数据转换 (zarr → LeRobot)](#3-数据转换)
4. [训练变换链 (Transform Pipeline)](#4-训练变换链)
5. [归一化统计 (Norm Stats)](#5-归一化统计)
6. [训练](#6-训练)
7. [推理服务 (Serve)](#7-推理服务)
8. [真机部署](#8-真机部署)
9. [已知坑与修复](#9-已知坑与修复)

---

## 1. 总览

```
UMI SLAM zarr.zip
       │
       ▼
 convert_zarr_to_lerobot.py     ← 绝对位姿 20D
       │
       ▼
 LeRobot v2 dataset (parquet + images)
       │
       ▼
 compute_norm_stats.py          ← q01/q99 分位数
       │
       ▼
 scripts/train.py               ← JAX 训练 (FSDP)
   │  变换链: DeltaActions → GlobalToBodyDelta → RelativeState
   │  基础权重: Pi0.5 pre-trained (ORBAX OCDBT)
       │
       ▼
 JAX checkpoint (params/)
       │
       ▼
 scripts/serve_policy.py        ← WebSocket policy server
   │  逆变换: BodyToGlobalDelta → AbsoluteActions
       │
       ▼
 ur7e_main_simple.py            ← 真机 servoL 控制
   │  坐标系恢复: body delta → R_M2R → robot TCP
       │
       ▼
 UR7e 机械臂执行
```

---

## 2. 数据采集与格式

### 2.1 UMI SLAM Pipeline 输出 (zarr.zip)

UMI 的 `07_generate_replay_buffer.py` 生成的 zarr 格式：

```
data/
  robot0_eef_pos:             (N, 3)  xyz 位置, 单位: 米, tag frame 绝对坐标
  robot0_eef_rot_axis_angle:  (N, 3)  旋转向量 (axis-angle), tag frame
  robot0_gripper_width:       (N, 1)  夹爪宽度, 单位: mm (0=闭合, ~35=全开)
  camera0_rgb:                (N, H, W, 3)  右腕相机 RGB
  camera3_rgb:                (N, H, W, 3)  头部相机 RGB (可选)
meta/
  episode_ends:               (num_episodes,)  每个 episode 的结束帧索引
```

### 2.2 20D State/Action 定义

转换后的 LeRobot 数据集使用统一的 20D 向量:

| 索引 | 字段 | 维度 | 说明 |
|------|------|------|------|
| 0-2 | position | 3 | xyz 位置 (米, 绝对) |
| 3-8 | rotation | 6 | rot6d 旋转 (旋转矩阵前两列展平) |
| 9 | gripper | 1 | 夹爪开合度 (0=闭合, ~0.389=全开35/90) |
| 10-19 | padding | 10 | 全零 (单臂不用左臂) |

**关键**: State 和 Action 完全相同，都是**绝对位姿**。相对化由训练时的 Transform 处理。

### 2.3 rot6d 表示

```python
def rotvec_to_rot6d(rotvec):
    R = Rotation.from_rotvec(rotvec).as_matrix()  # (3,3)
    return R[:, :2].T.flatten()
    # = [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
    # = 旋转矩阵第一列 + 第二列
```

恢复: Gram-Schmidt 正交化
```python
def rot6d_to_matrix(rot6d):
    r1 = rot6d[:3] / norm(rot6d[:3])          # 归一化第一列
    r2 = rot6d[3:6] - dot(r2, r1) * r1        # 正交化
    r2 = r2 / norm(r2)                          # 归一化
    r3 = cross(r1, r2)                          # 叉积得第三列
    return stack([r1, r2, r3], axis=1)         # (3,3)
```

**性质**: `rot6d_to_matrix(matrix_to_rot6d(R)) == R` (对任意旋转矩阵精确还原)

### 2.4 夹爪归一化

```
grip_normalized = gripper_width_mm / 90.0
```

- 0 = 完全闭合
- 35/90 ≈ 0.389 = 全开 (UMI 夹爪最大开度 35°)

---

## 3. 数据转换

### 3.1 zarr → LeRobot

**脚本**: `examples/umi/convert_zarr_to_lerobot.py`

```bash
cd /workspace/zt/openpi-official
export HF_LEROBOT_HOME="/workspace/zt/openpi-official/data"

uv run examples/umi/convert_zarr_to_lerobot.py \
    --zarr_path /workspace/umi/cups-0-47-crop.zarr.zip \
    --task "pick up the cup" \
    --image_size 256 \
    --fps 30
```

**逐帧处理**:
```python
pos = eef_pos[fid]                         # (3,)
rot6d = rotvec_to_rot6d(eef_rot[fid])     # (3,) → (6,)
grip = gripper[fid, 0] / 90.0             # mm → 归一化

pose_10d = [pos, rot6d, grip]              # (10,)
pose_20d = [pose_10d, zeros(10)]           # (20,) 补零
state = action = pose_20d                   # 绝对位姿, state==action
```

**输出**: `data/local/umi_zarr_single_arm_6d/` (LeRobot v2 格式, parquet + images)

### 3.2 UMI6 真机数据转换 (可选)

**脚本**: `examples/umi/convert_umi6_to_lerobot.py`

输入是 JSONL 位姿文件 + MP4 视频，转换方式类似，唯一不同是旋转用 quaternion (wxyz):

```python
def quat_wxyz_to_rot6d(q):
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # wxyz → xyzw
    return R[:, :2].T.flatten()
```

---

## 4. 训练变换链

### 4.1 三步变换 (训练时, 按顺序执行)

定义在 `src/openpi/transforms.py`, 配置在 `src/openpi/training/config.py`:

```python
# config.py: LeRobotUMI6RightOnlyDataConfig
pose_mask = make_bool_mask(9, -1, 9, -1)
# = [True]*9 + [False] + [True]*9 + [False]
#   pos(3)+rot(6): 参与delta/relative
#   gripper(1): 保持绝对值

data_transforms.push(
    inputs=[
        DeltaActions(pose_mask),       # Step 1
        GlobalToBodyDelta(),           # Step 2
        RelativeState(pose_mask),      # Step 3
    ],
    outputs=[
        BodyToGlobalDelta(),           # 逆 Step 2
        AbsoluteActions(pose_mask),    # 逆 Step 1
    ],
)
```

### 4.2 Step 1: DeltaActions

```
action[i] -= state   (对 mask=True 的维度)
```

- 位置: `delta_pos = action_pos - state_pos` (全局坐标系增量)
- 旋转: `delta_rot6d = action_rot6d - state_rot6d` (rot6d 线性减法, **不**是几何减法)
- 夹爪: 不变 (mask=False)

### 4.3 Step 2: GlobalToBodyDelta

**目的**: 把全局坐标系增量转到末端执行器 (EE) 局部坐标系

```python
R_current = rot6d_to_matrix(state[3:9])

# 位置: 全局 → 末端局部
body_delta_pos = R_current.T @ delta_pos_global

# 旋转: 恢复绝对目标 (undo DeltaActions), 计算几何相对旋转
target_rot6d = delta_rot6d + state_rot6d       # 恢复原始绝对 rot6d
R_target = rot6d_to_matrix(target_rot6d)
R_body_delta = R_current.T @ R_target           # SE(3) 相对旋转
body_delta_rot6d = matrix_to_rot6d(R_body_delta)
```

**数学本质**: 等价于 UMI 的 `inv(T_current) @ T_target`

### 4.4 Step 3: RelativeState

```
state → [0, 0, 0,  1, 0, 0, 0, 1, 0,  gripper,  0, ..., 0]
          pos=零    rot6d=identity     保留        左臂=零
```

**作用**:
- 让 state 与坐标系无关 (不需要知道 SLAM/robot frame)
- Pi0.5 单帧 obs, "相对当前帧" = identity
- 模型只需从 gripper 和图像获取信息

### 4.5 模型看到什么

经过三步变换后, 模型的输入/输出:

| | 内容 | 坐标系 |
|--|------|--------|
| **State** | `[0,0,0, identity_rot6d, gripper, zeros]` | 无 (identity) |
| **Action** (chunk, 10步) | `[body_delta_pos, body_delta_rot6d, gripper, zeros]` | 末端局部 |
| **Image** | `cam_right` 256x256 RGB | 视觉 |
| **Prompt** | `"pick up the cup"` | 文本 |

### 4.6 逆变换 (推理时)

推理时 state=identity, 逆变换简化为:

1. **BodyToGlobalDelta**: R_current=I → 无操作 (body delta = global delta)
2. **AbsoluteActions**: `action += state` → `body_delta_rot6d + identity_rot6d`

**注意**: AbsoluteActions 给 rot6d 加了 identity `[1,0,0,0,1,0]`, 部署时需要减回来 (见 [9.2](#92-旋转-identity-偏移))

---

## 5. 归一化统计

### 5.1 计算

```bash
python scripts/compute_norm_stats.py --config-name=pi05_umi_cups
```

遍历所有训练数据 (经过变换后), 计算每个维度的:
- `mean`, `std` (Welford 在线算法)
- `q01`, `q99` (直方图分位数, 5000 bins)

保存到: `assets/pi05_umi_cups/local/umi_zarr_single_arm_6d/norm_stats.json`

### 5.2 归一化方式

Pi0.5 使用 **z-score 归一化** (默认):

```
x_norm = (x - mean) / (std + 1e-6)
```

部分配置可能使用 **quantile 归一化**:

```
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1    # 映射到 [-1, 1]
```

### 5.3 Norm Stats 校验要点

变换后的 action 应呈现:
- **位置 mean ≈ 0**: body-frame delta, 均值在零附近
- **旋转 mean ≈ identity rot6d**: `[1,0,0,0,1,0]` 附近 (小角度偏转)
- **夹爪**: 绝对值, 反映数据集中的夹爪分布

---

## 6. 训练

### 6.1 配置

```python
# config.py
TrainConfig(
    name="pi05_umi_cups",
    model=Pi0Config(
        pi05=True,               # Pi0.5 架构
        action_dim=32,            # 20D 补零到 32D
        action_horizon=10,        # 预测未来 10 步
    ),
    data=LeRobotUMI6RightOnlyDataConfig(
        repo_id="local/umi_zarr_single_arm_6d",
        default_prompt="pick up the cup",
    ),
    weight_loader=CheckpointWeightLoader(
        "/workspace/zt/pi05_base_checkpoint/params"  # ORBAX OCDBT 格式!
    ),
    num_train_steps=50_000,
    batch_size=32,              # per-device
    fsdp_devices=4,             # 4 device mesh (16 GPU 时 = 4x4)
)
```

### 6.2 Pi0.5 模型架构

```
Image (256x256) → PaLiGemma (Gemma 2B vision encoder)
                        ↓
Prompt (tokenized) → Cross-attention
                        ↓
State (32D, tokenized) → Gemma 300M Action Expert
                        ↓
                   Actions (10 × 32D)
```

- **PaLiGemma**: 2B 参数, 处理图像 + 文本
- **Action Expert**: 300M 参数, 生成离散化动作 token
- **max_token_len**: 200 (Pi0.5 discrete state)

### 6.3 训练脚本

**脚本**: `train_cups_16gpu.sh`

```bash
#!/bin/sh
set -eu
cd /workspace/zt/openpi-official
export HF_LEROBOT_HOME="/workspace/zt/openpi-official/data"
export WANDB_MODE=disabled

# Step 1: 计算归一化统计
python scripts/compute_norm_stats.py --config-name=pi05_umi_cups

# Step 2: 准备 checkpoint 目录 (持久存储!)
CKPT_DIR="/workspace/zt/checkpoints_cups_v2"
mkdir -p "$CKPT_DIR"
ln -sf "$CKPT_DIR" checkpoints/pi05_umi_cups

# Step 3: JAX 训练
python scripts/train.py pi05_umi_cups \
    --exp-name=umi_cups_v2 \
    --overwrite
```

### 6.4 训练要点

| 参数 | 值 | 说明 |
|------|------|------|
| 基础权重 | Pi0.5 pre-trained | ORBAX OCDBT 格式 |
| 优化器 | AdamW | |
| EMA | decay=0.99 | JAX 独有, PyTorch 没有 |
| Batch size | 32 × 4 devices = 128 | FSDP 分片 |
| 训练步数 | 50,000 | |
| Checkpoint 间隔 | 每 10,000 步 | |
| 数据频率 | 30 fps | |
| Action horizon | 10 步 = 333ms | |

### 6.5 Checkpoint 输出

```
/workspace/zt/checkpoints_cups_v2/umi_cups_v2/
├── 9999/
│   └── params/        ← ORBAX checkpoint
├── 19999/
│   └── params/
├── 29999/
├── 39999/
└── 49999/             ← 最终权重
    ├── params/
    └── assets/        ← norm_stats 等
```

---

## 7. 推理服务

### 7.1 启动 Server

```bash
python scripts/serve_policy.py \
    --default-prompt "pick up the cup" \
    --port 8002 \
    policy:checkpoint \
    --policy.config pi05_umi_cups \
    --policy.dir /path/to/checkpoint/49999
```

### 7.2 推理数据流

```
Client 发送:
  obs = {
    "state":     [0,0,0, 1,0,0,0,1,0, gripper, 0,...,0],  # 20D identity
    "cam_right": (256, 256, 3) uint8 RGB,
    "prompt":    "pick up the cup"
  }

Server 内部:
  1. Repack → 标准字段名
  2. UMI6RightOnlyInputs → 图像映射 (right_wrist_0_rgb)
  3. DeltaActions → 无操作 (state 已是 identity, action 还没有)
  4. GlobalToBodyDelta → 无操作 (训练时处理, 推理时 state=identity)
  5. RelativeState → 无操作 (state 已是 identity)
  6. Normalize → z-score 归一化
  7. Tokenize prompt
  8. Model.sample_actions() → (1, 10, 32) 预测
  9. Unnormalize
 10. BodyToGlobalDelta → 无操作 (R_current=I)
 11. AbsoluteActions → action_rot6d += identity_rot6d  ← 关键!

Server 返回:
  {
    "actions": (10, 20D),   # chunk 10步
    "state": (20D)
  }
```

### 7.3 Server 输出解读

对于每一步 action:
- `action[0:3]`: body-frame 位置增量 (米, 相对推理时刻)
- `action[3:9]`: body-frame 旋转 rot6d **+ identity [1,0,0,0,1,0]** (AbsoluteActions 加的)
- `action[9]`: 夹爪绝对值

---

## 8. 真机部署

### 8.1 硬件

- 机械臂: UR7e (RTDE servoL 控制)
- 夹爪: CAN 总线电机夹爪 (slcand)
- 相机: USB 腕部相机 (256x256)
- 部署机: simpleai@192.168.3.86 (GPU for inference)

### 8.2 坐标系映射 R_M2R

SLAM/model frame 和 robot TCP frame 的旋转轴不同:

```python
R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]])
R_R2M = R_M2R.T
```

- 构建 state 时: `R_model = R_robot @ R_M2R` (但 state 是 identity, 不需要)
- 恢复 action 时: `R_target_robot = R_target_model @ R_R2M`

### 8.3 部署脚本

**脚本**: `ur7e_main_simple.py`

#### State 构建 (每帧)

```python
state = [0, 0, 0,  1, 0, 0, 0, 1, 0,  gripper_pos,  0, ..., 0]
#        pos=零     rot6d=identity     夹爪绝对值      左臂零
```

不需要坐标系转换, 因为 RelativeState 已经让 state 与坐标系无关。

#### Action → Robot TCP

```python
def delta_action_to_target_tcp(action, current_tcp, use_r_m2r=True):
    # 当前 TCP 旋转 → model frame
    R_robot = Rotation.from_rotvec(current_tcp[3:6]).as_matrix()
    R_current_model = R_robot @ R_M2R

    # 位置: EE 局部增量 → 机器人基座全局
    target_pos = current_tcp[:3] + R_current_model @ action[:3]

    # 旋转: 减掉 AbsoluteActions 加的 identity, 恢复纯 body delta
    body_rot6d = action[3:9] - np.array([1, 0, 0, 0, 1, 0])
    R_delta = rot6d_to_matrix(body_rot6d)

    # body delta → model frame → robot frame
    R_target_model = R_current_model @ R_delta
    R_target_robot = R_target_model @ R_R2M
    target_rotvec = Rotation.from_matrix(R_target_robot).as_rotvec()

    return [target_pos, target_rotvec]
```

#### Chunk 执行策略

```python
# 推理一次 → 得到 10 步 action chunk
actions = server.infer(obs)["actions"]  # (10, 20D)

tcp_base = current_tcp.copy()  # chunk 基准帧

# 执行 steps_per_inference 步 (默认 8), 跳过 actions[0]
for ai in range(1, 1 + steps_per_inference):
    target = delta_action_to_target_tcp(actions[ai], tcp_base)

    # 限速: 位置 max 20mm/步, 旋转 max 0.1rad/步
    clip_and_servo(target)
    sleep(1/30)  # 30Hz 控制频率
```

**关键**: 所有 action 都相对 chunk 基准帧 (推理时刻的 TCP), 不是相对每步的实时 TCP。

#### 夹爪控制

```python
# 模型输出 → 电机角度
grip_value = action[9]   # 0=闭合, ~0.389=全开
ratio = grip_value / 0.389
motor_deg = ratio * (-700)  # 0=闭合, -700=全开
```

### 8.4 启动命令

```bash
# 1. 网络路由 (机器人 IP)
sudo ip route add 192.168.3.254/32 dev enx6c1ff7bbfc25

# 2. CAN 夹爪
sudo slcand -o -c -s8 /dev/ttyACM0 can3 && sudo ip link set up can3

# 3. 运行
cd ~/pi05-deploy
python3 ur7e_main_simple.py \
    --cam-right-id 0 \
    --steps-per-inference 8 \
    --max-rot-step 0.1 \
    --max-pos-step 0.02 \
    --server-port 8002 \
    --prompt "pick up the cup"
```

### 8.5 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frequency` | 30 Hz | 控制频率 (匹配训练数据 fps) |
| `--steps-per-inference` | 2 (建议8) | 每次推理执行多少步 |
| `--max-pos-step` | 0.02 m | 单步最大位移 (20mm) |
| `--max-rot-step` | 0.1 rad | 单步最大旋转 (~5.7°) |
| `--lookahead` | 0.2 | servoL 前瞻时间 |
| `--gain` | 300 | servoL 增益 |
| `--min-z` | 0.065 m | 最低高度保护 |

### 8.6 Chunk 与频率的关系

```
训练数据: 30fps → 每帧间隔 33ms
Chunk:    10步 × 33ms = 333ms 未来预测

部署 (steps_per_inference=8):
  推理1次 → 执行 action[1]~[8] → 8步 × 33ms = 267ms
  → 重新推理 → 执行下一个 chunk 的 [1]~[8]
  → 推理频率 ≈ 30/8 ≈ 3.75 Hz

部署 (steps_per_inference=2):
  推理1次 → 执行 action[1]~[2] → 2步 × 33ms = 67ms
  → 重新推理
  → 推理频率 ≈ 15 Hz (更响应, 但延迟敏感)
```

---

## 9. 已知坑与修复

### 9.1 ORBAX_OCDBT_ENABLED=0 导致基础权重加载失败

**现象**: 训练 loss 看似正常下降, 但模型部署后几乎不动

**原因**: Pi0.5 基础 checkpoint 是 ORBAX OCDBT 格式 (目录里有 `manifest.ocdbt`)。
设置 `export ORBAX_OCDBT_ENABLED=0` 会让 orbax 无法读取, 导致模型从**随机初始化**开始训练。

**修复**: 删除 `ORBAX_OCDBT_ENABLED=0`, 确保训练脚本不包含这行。

**验证**: 训练初始 loss 应该较低 (pre-trained 的优势), 如果初始 loss 很高说明没加载到基础权重。

### 9.2 旋转 identity 偏移 (AbsoluteActions)

**现象**: 冻结旋转时位置移动正常, 开启旋转后机械臂行为异常

**原因**: 推理逆变换中 AbsoluteActions 给 rot6d 加了 identity `[1,0,0,0,1,0]`:
```
server_output[3:9] = body_delta_rot6d + [1,0,0,0,1,0]
```

如果部署脚本直接对 server 输出做 `rot6d_to_matrix()`, Gram-Schmidt 归一化会扭曲旋转角度 (大约减半):

| 真实旋转 | rot6d 直接转换结果 |
|----------|-------------------|
| 30° | 15° |
| 90° | 45° |

**修复**: 部署脚本中减掉 identity 再转换:
```python
body_rot6d = action[3:9] - np.array([1, 0, 0, 0, 1, 0])
R_delta = rot6d_to_matrix(body_rot6d)
```

### 9.3 Checkpoint 存储位置

九章云等云平台的 `/tmp` 是易失存储, 重启后消失。必须将 checkpoint 存到 `/workspace/` 等持久目录:

```bash
CKPT_DIR="/workspace/zt/checkpoints_cups_v2"
ln -sf "$CKPT_DIR" checkpoints/pi05_umi_cups
```

### 9.4 JAX vs PyTorch 训练差异

| | JAX | PyTorch |
|--|-----|---------|
| 基础权重格式 | ORBAX (OCDBT) | safetensors |
| EMA | 有 (decay=0.99) | 无 |
| 推理 | JAX native | PyTorch native |
| 其他 | 完全相同 | 完全相同 |

两种训练路径的数据管线、优化器、lr schedule、gradient clipping 完全一致。

---

## 附录: 文件索引

| 文件 | 作用 |
|------|------|
| `examples/umi/convert_zarr_to_lerobot.py` | zarr → LeRobot 数据转换 |
| `examples/umi/convert_umi6_to_lerobot.py` | UMI6 真机数据 → LeRobot |
| `src/openpi/training/config.py` | 训练配置 (模型/数据/变换) |
| `src/openpi/transforms.py` | DeltaActions/GlobalToBodyDelta/RelativeState |
| `scripts/compute_norm_stats.py` | 计算归一化统计 |
| `scripts/train.py` | JAX 训练主循环 |
| `scripts/serve_policy.py` | WebSocket 推理服务 |
| `train_cups_16gpu.sh` | 训练启动脚本 |
| `ur7e_main_simple.py` (远程) | 真机部署脚本 |
