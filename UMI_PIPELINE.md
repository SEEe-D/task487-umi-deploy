# 基于 Pi0.5 的 UMI 操作系统方法论

## 1. 方法概述

本方案基于 Physical Intelligence 的 Pi0.5 视觉-语言-动作模型 (VLA)，结合 UMI (Universal Manipulation Interface) 数据采集框架，实现从人类示教到机器人自主操作的端到端学习。

**整体流程**：人类手持 UMI 夹爪演示操作任务（SLAM 实时定位 + 腕部相机录制） → 将采集到的绝对位姿数据转换为末端执行器局部坐标系下的相对增量 → 以此微调 Pi0.5 预训练模型 → 部署到 UR7e 机械臂上，由模型根据实时相机画面和语言指令预测动作序列并执行。

**核心设计选择**：
- **表示方式**：采用 body-frame relative delta 而非绝对位姿，使模型学到"往哪个方向走"而非"去哪个坐标"
- **旋转表示**：采用 6D 连续旋转表示 (rot6d)，避免欧拉角万向锁和四元数双覆盖问题
- **坐标变换**：基于 SE(3) 几何变换（矩阵乘法）而非向量线性运算，保证旋转群结构的数学正确性

## 2. 数据采集与表示

### 2.1 UMI 数据采集

UMI 是一种低成本的人类示教数据采集方案。操作者手持带有 AprilTag 的夹爪，在工作台上执行操作任务。系统通过以下方式记录数据：

- **SLAM 定位**：GoPro 相机拍摄 AprilTag，通过视觉 SLAM 算法实时解算夹爪的 6DoF 位姿（位置 + 旋转），以标定的 tag frame 为世界坐标系
- **腕部相机**：安装在夹爪上的 RGB 相机，记录操作者视角的画面（256×256）
- **夹爪开度**：通过编码器记录夹爪的实时宽度

采集输出为 zarr 格式，包含每帧的末端位姿、夹爪宽度和相机图像，采样频率 30fps。

### 2.2 状态与动作定义

为兼容 Pi0.5 的双臂格式（UMI 原始设计支持双臂），每帧数据统一用 20D 向量表示：

| 维度 | 含义 | 说明 |
|------|------|------|
| 0-2 | 位置 (x, y, z) | 米，SLAM tag frame 绝对坐标 |
| 3-8 | 旋转 (rot6d) | 旋转矩阵前两列展平，6D 连续表示 |
| 9 | 夹爪开度 | gripper_width / 90.0，归一化到 [0, 0.389] |
| 10-19 | 左臂填充 | 全零（单臂场景不使用） |

**关键性质**：原始数据中 state 与 action 完全相同，都是该时刻末端的绝对位姿。action chunk 就是未来 N 帧（如 20 帧，对应 667ms）的绝对位姿序列。绝对位姿到相对增量的转换由训练时的变换链自动完成。

### 2.3 旋转的 6D 连续表示

采用 Zhou et al. (2019) 提出的 rot6d 表示法。传统旋转表示（欧拉角、四元数、旋转向量）在拓扑上不连续，会给神经网络学习带来困难。rot6d 通过冗余表示 + Gram-Schmidt 正交化实现连续映射。

**编码**：取旋转矩阵 R ∈ SO(3) 的前两列，转置后展平为 6 维向量

$$\text{rot6d}(R) = [R_{00}, R_{10}, R_{20}, R_{01}, R_{11}, R_{21}]$$

**解码**：通过 Gram-Schmidt 正交化从 6 维向量恢复完整的 3×3 旋转矩阵

$$r_1 = \text{normalize}(v_{0:3})$$
$$r_2 = \text{normalize}(v_{3:6} - (v_{3:6} \cdot r_1) \, r_1)$$
$$r_3 = r_1 \times r_2$$
$$R = [r_1, r_2, r_3]$$

**性质**：对任意旋转矩阵 R，解码(编码(R)) = R，round-trip 精确还原。单位矩阵 I 的 rot6d 表示为 [1, 0, 0, 0, 1, 0]。

### 2.4 夹爪归一化

UMI 夹爪的物理宽度范围为 0mm（完全闭合）到约 35mm（全开，对应夹爪电机角度 35°）。归一化方式为：

$$\text{grip} = \text{width\_mm} / 90.0$$

全开时 grip ≈ 35/90 ≈ 0.389，闭合时 grip = 0。除以 90 而非 35 是因为电机角度理论最大值为 90°。

## 3. 坐标变换

### 3.1 设计动机

原始数据是 SLAM 坐标系下的绝对位姿，直接用于训练存在两个问题：

1. **坐标系耦合**：模型需要学习特定 SLAM 坐标系下的绝对坐标值，换一个场景（不同的 tag 布局、不同的机器人安装位置）就无法使用
2. **绝对位置依赖**：模型预测"末端移动到坐标 (0.3, 0.2, 0.1)"而非"末端向前移动 5cm"，缺乏空间泛化能力

解决方案：将绝对位姿转换为末端执行器 (EE) 局部坐标系的相对增量。在 body-frame 表示下：
- 模型学到的是"从当前姿态出发，末端该往自己的哪个方向移动/旋转多少"
- 与具体的世界坐标系无关
- 部署时只需知道当前末端的位姿即可恢复动作

### 3.2 第一步：GlobalToBodyDelta — 绝对位姿 → EE 局部增量

将 action chunk 中每帧的绝对位姿，转为相对于当前帧 (state) 的 body-frame delta。

设当前帧 state 的旋转矩阵为 $R_{current}$，当前位置为 $p_{current}$。对 chunk 中每一帧（目标位置 $p_{target}$，目标旋转 $R_{target}$）：

**位置变换**：

先计算全局坐标系下的位移向量（目标位置减去当前位置），再左乘 $R_{current}^T$ 将该向量从全局坐标系投影到末端局部坐标系：

$$\Delta p_{body} = R_{current}^T \cdot (p_{target} - p_{current})$$

其中 $R_{current}^T = R_{current}^{-1}$（旋转矩阵正交性）。结果 $\Delta p_{body}$ 表示"在末端自身坐标系中，目标在我的前方/左方/上方多远"。

**旋转变换**：

旋转属于特殊正交群 SO(3)，是一个乘法群（两个旋转的"差"通过乘法计算，不能用减法）。从当前旋转到目标旋转的相对旋转为：

$$R_{body\_delta} = R_{current}^T \cdot R_{target}$$

这一步同时完成了"求相对旋转"和"表达在末端局部坐标系中"两个操作。数学上，$R_{body\_delta}$ 满足 $R_{target} = R_{current} \cdot R_{body\_delta}$，即"当前旋转 × 相对旋转 = 目标旋转"。

最后将 $R_{body\_delta}$ 编码为 rot6d 存储。

**夹爪**：不参与坐标变换，保持绝对值不变。

**整体几何含义**：位置和旋转的变换合在一起，等价于 SE(3) 齐次变换群中的 $T_{current}^{-1} \cdot T_{target}$，即"在当前末端的参考系中，目标位姿是什么"。

### 3.3 第二步：RelativeState — 状态归零

将 state 的位姿部分设为 identity（位置归零、旋转设为单位矩阵），仅保留夹爪开度：

$$\text{state} \rightarrow [0, 0, 0, \; 1, 0, 0, 0, 1, 0, \; \text{gripper}, \; 0, \ldots, 0]$$

**设计原因**：Pi0.5 是单帧观测模型（不依赖历史状态序列）。在 body-frame 表示下，"当前帧相对于当前帧"就是 identity，state 不携带任何坐标系信息。这样：
- 模型只从图像感知空间信息，从 gripper 值感知夹爪状态
- 部署时构建 state 不需要任何坐标系转换，直接发 identity + gripper 即可
- 训练数据来自 SLAM 坐标系、部署在 robot 坐标系，但 state 是同一个 identity，天然对齐

### 3.4 变换后模型看到的数据

| 字段 | 变换后的内容 | 含义 |
|------|-------------|------|
| State (20D) | [0,0,0, identity_rot6d, gripper, 0...0] | 坐标系无关的当前状态 |
| Action chunk (N×20D) | [body_Δpos, body_Δrot6d, gripper, 0...0] × N | 未来 N 步的 EE 局部增量 |
| 图像 | 腕部相机 256×256 RGB | 当前视觉观测 |
| 指令 | 自然语言文本 | 任务描述 |

变换后 action 的分布特征：位置增量均值接近零（body delta 在零附近波动），旋转增量接近 identity rot6d（小角度偏转），符合局部增量的预期分布，有利于模型学习。

## 4. 归一化

变换后的数据在送入模型前进行归一化。使用 z-score 归一化：

$$x_{norm} = \frac{x - \mu}{\sigma + \epsilon}$$

其中 $\mu$（均值）和 $\sigma$（标准差）从全量训练数据（经变换后）统计得到，同时记录 $q_{01}$ 和 $q_{99}$ 分位数用于异常值检测。这些统计量（norm_stats）保存在 checkpoint 中，推理时加载使用。

## 5. 模型与训练

### 5.1 模型架构

基于 Pi0.5 (Physical Intelligence, 2024)，包含两个核心组件：

- **视觉-语言编码器 (PaLiGemma, 2B 参数)**：接收腕部相机图像和自然语言任务指令，提取视觉-语义联合特征。PaLiGemma 是 Google 的视觉-语言基础模型，融合了 SigLIP 视觉编码器和 Gemma 语言模型。

- **动作专家 (Gemma Action Expert, 300M 参数)**：基于视觉-语义特征和当前 state，自回归生成离散化的动作 token 序列，解码为 action chunk（未来 20 步的 EE 局部增量）。采用 VQ (Vector Quantization) 离散化动作空间。

模型端到端可微，从图像像素直接到末端执行器动作，无需手工设计的感知模块（如物体检测、位姿估计等）。

### 5.2 训练策略

在 Pi0.5 大规模预训练权重基础上进行任务级微调。预训练权重包含了在多种机器人平台和任务上学到的通用操作先验，微调只需少量任务数据即可适配特定场景。

| 项目 | 设定 |
|------|------|
| 基础权重 | Pi0.5 预训练 (ORBAX OCDBT 格式) |
| 微调数据 | UMI 人类示教，30fps |
| 优化器 | AdamW |
| EMA | decay=0.99（指数移动平均，用于推理的平滑权重） |
| 归一化 | z-score（基于训练数据统计） |
| 训练规模 | 8 GPU FSDP 分片，batch_size=32/device |
| 训练步数 | 50,000 |
| Action horizon | 20 步（对应 667ms 的未来预测窗口） |
| Checkpoint 间隔 | 每 1,000 步保存 |

### 5.3 当前任务

| 任务 | 数据来源 | 数据规模 | 语言指令 |
|------|----------|----------|----------|
| 积木入盒 | UMI6 真机采集 | ~500 episodes | "building blocks into box" |
| 抓取杯子 | UMI zarr 数据 | 2,230 episodes | "pick up the cup" |

## 6. 推理与部署

### 6.1 推理流程

部署时，模型作为 WebSocket 服务运行。每个控制周期：

1. **观测构建**：采集腕部相机图像（256×256 RGB）、构建 identity state（[0,0,0, identity_rot6d, gripper, 0...0]）、设定任务指令文本
2. **模型推理**：将观测发送到 server，server 内部进行归一化 → 模型前向推理 → 反归一化 → 逆变换
3. **输出**：返回 action chunk（20 步 × 20D），每步包含 body-frame 位置增量、旋转增量和夹爪绝对值

由于推理时 state 为 identity（$R_{current} = I$），逆变换 BodyToGlobalDelta 中 $I \cdot \Delta p = \Delta p$，$I \cdot R_{delta} = R_{delta}$，等价于无操作。因此 server 直接输出纯 body-frame delta，不需要任何后处理。

### 6.2 坐标系映射

训练数据在 SLAM 坐标系下采集，部署在 UR 机器人基座坐标系下执行。两个坐标系的旋转轴约定不同（SLAM 的 x/y/z 轴方向与 robot TCP 的 x/y/z 轴方向存在固定的旋转关系）。通过正交映射矩阵 $R_{M2R}$ 桥接：

$$R_{model} = R_{robot} \cdot R_{M2R}$$

$$R_{M2R} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix}, \quad R_{R2M} = R_{M2R}^T$$

$R_{M2R}$ 将 robot frame 的旋转映射到 model frame，$R_{R2M}$ 做反向映射。这是一个固定的标定矩阵，取决于 UMI 夹爪的安装方式和 SLAM tag 的朝向约定。

### 6.3 动作恢复

将 server 返回的 body-frame delta 转换为 UR 机器人可执行的 TCP 目标位姿。设当前 TCP 位姿为 $(p_{tcp}, R_{robot})$：

**Step 1 — 当前旋转映射到模型坐标系**：

$$R_{current\_model} = R_{robot} \cdot R_{M2R}$$

**Step 2 — 位置恢复**（EE 局部增量 → 机器人基座全局坐标）：

$$p_{target} = p_{tcp} + R_{current\_model} \cdot \Delta p_{body}$$

$R_{current\_model}$ 将末端局部坐标系中的增量旋转回全局坐标系。这是训练时 $R_{current}^T \cdot \Delta p_{global}$ 的逆操作。

**Step 3 — 旋转恢复**（body delta → 模型坐标系目标 → 机器人坐标系目标）：

$$R_{delta} = \text{rot6d\_to\_matrix}(\text{action}[3{:}9])$$
$$R_{target\_model} = R_{current\_model} \cdot R_{delta}$$
$$R_{target\_robot} = R_{target\_model} \cdot R_{R2M}$$

第一行将 rot6d 解码为旋转矩阵。第二行将 body delta 右乘到当前旋转上，得到模型坐标系下的目标旋转（训练时 $R_{body\_delta} = R_{current}^T \cdot R_{target}$ 的逆操作）。第三行通过 $R_{R2M}$ 映射回机器人坐标系。

**Step 4 — 执行**：

$$\text{target\_tcp} = [p_{target}, \; \text{as\_rotvec}(R_{target\_robot})]$$

将目标位姿转为 UR 机器人 servoL 接受的格式（位置 + 旋转向量）并发送执行。

### 6.4 Chunk 执行策略

每次推理产出 20 步 action chunk，以推理时刻的 TCP 为基准帧 $tcp_{base}$。从 chunk 中按顺序取若干步执行（跳过第 0 步，因为第 0 步对应当前帧自身），控制频率 30Hz 匹配训练数据采集帧率。

**关键设计**：chunk 中所有 action 都相对同一个基准帧 $tcp_{base}$ 计算，而非逐步累积。即 action[1] 是"从基准帧出发到 t+1 的增量"，action[5] 是"从基准帧出发到 t+5 的增量"，不需要叠加前序 action。

执行完若干步后重新推理，获取新的 chunk，实现 receding horizon 控制。

### 6.5 安全限速

每步执行前施加 clipping 保护：

- **位置限速**：单步位移不超过 20mm（防止因模型预测偏差导致的大幅跳动）
- **旋转限速**：单步旋转角度可配置（当前冻结旋转 max_rot_step=0 以确保部署稳定性，待旋转校准完成后逐步开放）
- **高度保护**：末端 z 坐标不低于安全阈值（防止撞击桌面）

## 7. 总结

本方案的核心设计要点：

1. **坐标系无关的 body-frame 表示**：通过 GlobalToBodyDelta 变换将绝对位姿转为 EE 局部增量，消除了训练数据（SLAM 坐标系）与部署环境（robot 坐标系）的耦合，使模型学到的操作策略具有空间泛化能力

2. **SE(3) 几何变换的正确性**：位置变换通过"全局位移 → 左乘 $R^T$ 投影到 body frame"实现，旋转变换通过"$R_{current}^T \cdot R_{target}$"一步完成相对计算与坐标系转换。所有旋转运算均在 SO(3) 群上以矩阵乘法进行，避免了 rot6d 线性减法带来的数学错误

3. **Pi0.5 预训练微调**：利用大规模预训练的 VLA 基础模型，在少量任务级示教数据上微调即可适配具体操作任务，显著降低了数据需求

4. **端到端视觉策略**：从腕部相机图像 + 语言指令直接输出末端执行器动作序列，无需物体检测、位姿估计等中间感知步骤，简化了系统复杂度

5. **Receding horizon chunk 控制**：模型一次预测未来 20 步（667ms），实际执行若干步后重新推理，兼顾了长时预测的平滑性和实时反馈的响应性
