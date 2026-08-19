# UMI头部视角到真实机器人部署的视觉域差异研究计划

> 文档状态：研究与工程主说明  
> 更新日期：2026-08-13  
> 本地工程目录：`/home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy`  
> 云端训练目录：`/root/pi05_mask`

## 1. 项目摘要

本项目研究的问题不是普通的推理时视觉干扰，也不是单纯给机器人图像涂黑，而是UMI可穿戴数据采集机制天然造成的训练—部署视觉域差异。

UMI示范数据通过安装在人头部的相机采集。采集过程中存在两个不可避免的现象：

1. 采集者头部持续转动，使头部画面产生平移、旋转、抖动及有效视野变化。
2. 头部相机必然拍摄到采集者自己的手臂、肩膀和躯干，而且这些人体区域与示范动作高度相关。

真实机器人推理时，头部相机通常固定，画面中不存在人类采集者，出现的是机器人手臂或机器人本体。因此，同一个任务在训练和部署时具有不同的相机运动模式和不同的执行者外观。

本项目希望通过：

- 稳定UMI训练数据中的头部画面；
- 在训练端从视觉token层屏蔽人类采集者；
- 在推理端通过机器人几何投影屏蔽机器人本体；
- 使训练与部署都尽量只保留桌面、物体、容器、目标区域等任务场景信息；

从而缩小UMI人类示范到真实机器人执行之间的视觉域差异，并让Pi0.5真正有效利用头部相机提供的全局视角。

## 2. 研究问题的准确表述

### 2.1 核心问题

建议使用以下科学问题作为论文和实验设计的中心：

> UMI头部视频中不可避免的采集者本体泄漏（demonstrator embodiment leakage），是否会使VLA模型学习到仅存在于人类示范域中的视觉关联，从而削弱向真实机器人部署的迁移？通过头部画面稳定化和执行者无关的视觉token屏蔽，能否建立更一致的训练—部署头部视角，并提高真实机器人任务成功率？

### 2.2 两类域差异

#### 视角动态差异

- 训练端：头戴相机跟随人头运动。
- 推理端：机器人头部相机固定。
- 可能后果：背景、目标位置和空间关系的视觉表征不稳定。

#### 执行者本体差异

- 训练端：画面中出现人类手臂、肩膀和躯干。
- 推理端：画面中出现机器人手臂或机器人本体。
- 可能后果：模型利用人类身体区域作为动作捷径，而这些特征在部署时不存在。

### 2.3 推荐术语

- `demonstrator embodiment leakage`：示范者本体泄漏
- `human-to-robot embodiment gap`：人到机器人的本体域差异
- `wearable-to-robot visual domain gap`：可穿戴相机到机器人相机的视觉域差异
- `embodiment-invariant visual observation`：执行者无关的视觉观测
- `stabilized embodiment masking`：稳定化本体屏蔽
- `head-view domain alignment`：头部视角域对齐

## 3. 研究假设

本项目需要用实验逐条验证以下假设，而不是预设方法一定有效。

### H1：原始UMI头部画面存在可测量的部署域差异

原始头部画面的剧烈运动和人类身体区域，会降低模型向固定相机真实机器人的迁移能力。

### H2：画面稳定化能够改善全局场景表征

稳定化能够降低背景和目标位置的无关运动，使头部画面中的全局任务信息更容易被模型利用。

### H3：训练端人体token屏蔽能够减少示范者视觉捷径

屏蔽人类手臂和躯干后，模型更多依赖物体、目标区域、腕部相机和机器人状态，而不是人类外观。

### H4：推理端机器人几何token屏蔽能够进一步对齐目标域

训练端屏蔽人类、推理端屏蔽机器人，使两端头部画面都趋向于只包含任务场景。

### H5：完整方法优于Wrist-only

若完整方法只比原始头部画面好、却不优于完全不用头部画面的模型，则不能证明头部画面被有效利用。完整方法应在需要全局视野的任务上优于Wrist-only基线。

## 4. 方法总体流程

### 4.1 训练端

```text
原始UMI头部RGB视频
        │
        ├─ 估计头部运动/稳定变换
        │
        ├─ 对RGB和人体mask应用完全相同的变换
        │
        ├─ 使用与Pi0.5一致的resize_with_pad映射
        │
        ├─ 将mask映射到最终视觉token网格
        │
        └─ 在视觉attention输入中丢弃人体对应token

左腕up图像 ─┐
右腕up图像 ─┼─> Pi0.5多视角输入
双臂状态历史 ┘
```

### 4.2 推理端

```text
固定头部RGB图像
        │
        ├─ 读取实时机器人关节状态
        ├─ 根据相机标定和机器人几何模型渲染本体mask
        ├─ 使用与训练端完全一致的resize_with_pad映射
        ├─ 将mask映射到相同视觉token网格
        └─ 在视觉attention输入中丢弃机器人对应token

左腕up图像 ─┐
右腕up图像 ─┼─> Pi0.5推理 ─> 20步action chunk
双臂状态历史 ┘
```

### 4.3 Mask语义

当前约定：

- 白色/255表示需要屏蔽的像素。
- token keep mask中`True`表示保留该视觉token。
- token keep mask中`False`表示该token不参与有效视觉attention。
- 当前模型视觉token网格实现以16×16为基础，共256个视觉patch token。

## 5. Mask与RGB几何对齐的关键要求

这是正式mask训练前必须验证的最高优先级问题。

原始头部画面通常为640×512，而Pi0.5最终输入为224×224。Pi0.5使用保持宽高比的`resize_with_pad`时，RGB图像会缩放并产生padding。如果直接把原始640×512 mask均分成16×16，再把RGB单独resize/pad，mask token与RGB token的位置会错位。

正确顺序必须是：

```text
原始RGB + 原始mask
        │
        ├─ 相同的稳定化变换
        ├─ 相同的裁剪范围
        ├─ 相同的缩放比例
        ├─ 相同的padding位置
        └─ 在最终224×224坐标域生成16×16 token mask
```

必须保存并可视化以下中间结果：

- 稳定化前RGB与人体mask叠加图；
- 稳定化后RGB与人体mask叠加图；
- resize/pad后的224×224叠加图；
- 16×16 token mask上采样后的预览；
- 人体或机器人轮廓是否与被丢弃token一致。

当前no-mask Task487基线不受此问题影响，但正式mask版本开始训练前必须完成这一对齐测试。

## 6. 当前Pi0.5训练架构

### 6.1 官方基线

当前训练和部署基于Physical Intelligence官方OpenPI仓库：

```text
official base commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac
```

正式本地仓库：

```text
gj/pi05-deploy/openpi-official
```

云端训练仓库：

```text
/root/pi05_mask
```

当前不是StarVLA，也不是重新实现的VLA架构，而是官方Pi0.5主干加UMI双臂数据和部署适配。

### 6.2 保持不变的Pi0.5核心

- Pi0.5 Flow Matching训练目标；
- PaliGemma视觉语言主体；
- action expert；
- 官方denoising和`sample_actions`流程；
- normalization与unnormalization流程；
- checkpoint保存及加载格式。

### 6.3 受控适配内容

- UMI三相机与双臂20D状态/动作转换；
- LeRobot 0.4本地数据读取；
- 25Hz action时间契约；
- 前一帧和当前帧双状态历史；
- 可选视觉token mask；
- Task487训练配置与部署元数据。

主要源码：

- [`openpi-official/src/openpi/policies/umi_policy.py`](./openpi-official/src/openpi/policies/umi_policy.py)
- [`openpi-official/src/openpi/training/config.py`](./openpi-official/src/openpi/training/config.py)
- [`openpi-official/src/openpi/training/data_loader.py`](./openpi-official/src/openpi/training/data_loader.py)

## 7. Task487数据和模型契约

### 7.1 数据集事实

- 数据目录：云端`/root/pi05_mask/datasets/task487`
- 约230个episodes
- 约150,907帧
- 数据频率：严格25Hz
- 相邻有效帧时间差约0.04秒
- 已核对`action[t] == state[t+1]`
- 当前基线使用三路相机，不使用五路相机

### 7.2 三相机顺序

模型输入顺序固定为：

1. `cam_head`：头部相机
2. `cam_left_top`：左腕up相机
3. `cam_right_top`：右腕up相机

对应数据字段：

- `observation.images.head_main`
- `observation.images.left_hand_up`
- `observation.images.right_hand_up`

### 7.3 状态与动作

- 机器人有效状态维度：20D
- 机器人有效动作维度：20D
- 模型内部padding动作维度：32D
- action horizon：20
- action stride：1
- 每个action目标间隔：0.04秒
- 20个动作覆盖未来约0.04至0.80秒
- 状态历史时间：`[-0.04, 0.0]`秒

双臂布局：

```text
right_pose9 + right_gripper1 + left_pose9 + left_gripper1
```

其中：

```text
pose9 = xyz3 + rotation6d6
rotation6d = 旋转矩阵前两行，按行展开
```

夹爪单位：

- 数据和模型：弧度
- 真机Livelybot接口：角度
- 输入模型前：degree → radian
- 输出真机前：radian → degree

### 7.4 语言指令

当前数据集中包含以下任务文本：

```text
Vegetable and Fruit Sorting.
Pick Up Vegetable and Place Vegetable on the Pink  Plate on the Right
Pick Up Fruit and Place Fruit on the Blue Plate on the Left
```

注意第一条具体放置指令中`Pink`和`Plate`之间存在两个空格，部署时应保持与数据集文本一致。

## 8. 当前训练状态

当前正在进行的是no-mask基线训练，其作用是验证官方Pi0.5、Task487数据和真实机器人部署底座。

```text
config: pi05_umi_task487
experiment: task487_3cam_nomask_30k
training steps: 30,000
checkpoint interval: 5,000
GPU count: 4
mask_enabled: false
```

截至2026-08-13约10:03的状态快照：

- 训练进程PID：71749
- 进度约4.01k/30k
- 速度约2.0秒/step
- 四张GPU利用率均为100%
- 单卡显存约74.5/81.9GB
- 未发现OOM、NaN或训练报错
- 首个5k checkpoint尚未生成

该状态是时间快照，实时状态应重新读取训练日志。

## 9. 当前本地部署架构

### 9.1 正式入口

- 服务端：[`run_task487_server.sh`](./run_task487_server.sh)
- 客户端：[`run_task487_client.sh`](./run_task487_client.sh)
- 真机客户端：[`task487_client.py`](./task487_client.py)
- 调度器：[`task487_runtime/scheduler.py`](./task487_runtime/scheduler.py)
- 数据契约：[`task487_runtime/contract.py`](./task487_runtime/contract.py)
- 使用说明：[`TASK487_RUNTIME.md`](./TASK487_RUNTIME.md)

旧同步客户端、不完整RTC实验和旧任务启动脚本已归档至：

```text
legacy/pre_task487_20260813
```

### 9.2 控制时序

- 模型action waypoint频率：25Hz
- 每40ms执行一个waypoint
- 每执行5个waypoint发起一次新推理请求
- 推理请求目标频率约5Hz
- 底层RTDE伺服插值频率：300Hz
- 同一时间只允许一个推理请求在途

调度逻辑：

1. 当前chunk持续执行，不等待新推理阻塞控制线程。
2. 新chunk返回后，根据原始观测时间计算每个动作的物理目标时间。
3. 已经过期的动作前缀直接丢弃，不重复、不重新赋予新时间戳。
4. 新旧未来轨迹重叠部分进行3步平滑。
5. 平移和夹爪采用线性插值。
6. 旋转采用SLERP。
7. 队列耗尽、推理异常、传感器异常或安全检查失败时进入HOLD。

### 9.3 操作状态机

- 启动：`HOLD`
- `d`：开始一个新的推理轮次
- `s`：立即进入HOLD并清空模型调度
- `r`：仅在HOLD状态下低速返回启动时记录的HOME
- `Ctrl+C`：先下发HOLD，再关闭控制器

默认客户端不输出模型动作。真实运动必须显式添加：

```text
--execute
```

默认真实测试只执行5个waypoint后进入HOLD。连续执行必须额外添加：

```text
--continuous
```

### 9.4 首轮安全限制

- 最大平移速度：0.05m/s
- 最大旋转速度：0.20rad/s
- 相邻目标单臂平移超过10mm：整段拒绝并HOLD
- 相邻目标单臂旋转超过0.10rad：整段拒绝并HOLD
- 摄像头超时、三相机时间偏差超限：HOLD
- 首次真实评测只运行5个waypoint

## 10. 已完成验证

- 云端和本地Task487配置SHA256一致；
- 本地OpenPI环境已完成；
- LeRobot版本为0.4.0；
- JAX版本为0.5.3；
- Task487 UMI模型变换测试5项通过；
- 本地契约、调度、夹爪测试10项通过；
- Python静态编译检查通过；
- Shell启动脚本语法检查通过；
- Git diff空白检查通过；
- 官方WebSocket server/client三相机、20×20 action协议烟测通过；
- 测试过程中未向真实机器人下发模型运动指令。

## 11. 尚未完成或必须继续验证的部分

### 11.1 真实checkpoint闭环

需要等待5k checkpoint生成后完成：

1. 下载checkpoint到本地；
2. 验证服务端正常加载；
3. 验证模型输出形状、数值和时延；
4. 离线进行头部图像消融；
5. observation-only运行；
6. 低速5-waypoint真实机器人测试。

### 11.2 头部稳定化

当前Task487 no-mask基线不包含正式研究方法。正式实验前需要确认稳定化管线：

- 稳定参考坐标如何定义；
- 旋转和平移补偿方式；
- 有效视野和裁剪比例；
- 黑边如何处理；
- RGB与人体mask是否使用完全相同的变换；
- 稳定化是否会删除任务目标或引入插值伪影。

### 11.3 人体mask质量

必须抽样检查：

- 手臂、肩膀、躯干覆盖率；
- 物体和桌面误屏蔽率；
- 人体快速运动时的时序稳定性；
- mask在稳定化前后的几何一致性；
- token化后保留的场景token数量。

### 11.4 推理端机器人几何mask

当前no-mask基线没有启用几何mask。启用前需要确认：

- 当前真实机器人实际关节数和URDF一致；
- 当前几何mask代码所需的关节布局与真机状态一致；
- 相机内参与外参对应当前640×512原始图像；
- 简化几何体不会明显漏掉机器人本体；
- mask渲染、resize/pad和token映射完全对齐；
- 实际推理延迟满足25Hz调度要求。

特别注意：已有部分几何代码曾使用双臂14关节布局，而当前UMI/UR控制链读取的是每臂6个关节。正式接入前必须明确使用哪套机器人模型和关节来源，不能通过隐藏fallback绕过维度不一致。

## 12. 低成本方法有效性验证

在扩大到四个任务和数百次真机实验前，先进行可证伪的小规模试验。

### 12.1 推荐训练版本

| 编号 | 头部画面 | 训练端人体mask | 用途 |
|---|---|---:|---|
| A | 不使用 | 否 | Wrist-only基线 |
| B | 原始UMI头部画面 | 否 | 原始三相机基线 |
| C | 稳定化头部画面 | 否 | 隔离稳定化贡献 |
| D | 稳定化头部画面 | token mask | 完整训练方法 |

对D模型在推理时分别测试：

- D-no-robot-mask：不屏蔽机器人本体；
- D-robot-mask：启用机器人几何token mask。

这样无需为推理端mask单独训练新模型，即可隔离目标域机器人mask的贡献。

### 12.2 公平性要求

所有版本必须保持：

- 相同原始episodes；
- 相同训练/验证划分；
- 相同Pi0.5 base权重；
- 相同随机种子或报告多随机种子；
- 相同训练步数；
- 相同batch size和学习率；
- 相同动作、状态与语言指令；
- 除指定图像处理变量外不修改其他配置。

### 12.3 第一阶段任务和次数

先选择一个明显依赖全局头部视角的任务。每种设置进行正常机器人部署测试，不需要人为制造推理干扰。

建议pilot规模：

```text
Wrist-only                  10次
Raw head                    10次
Stabilized head             10次
Full / inference mask off   10次
Full / inference mask on    10次
合计                        50次真机episode
```

若成功率方差较大，可扩展到每种20次。

### 12.4 Go/No-Go判定

建议预先固定以下判定标准，避免观察结果后改变标准：

- 完整方法比Raw head提高至少15个百分点：值得扩展实验；
- 完整方法在正常场景下降不超过5个百分点：没有明显牺牲基础能力；
- 完整方法优于Wrist-only：证明头部视角提供了有效增益；
- Stabilized优于Raw：稳定化独立有效；
- Full优于Stabilized：人体token mask独立有效；
- D-robot-mask优于D-no-robot-mask：对称的训练/推理本体屏蔽有效。

pilot只有10次时置信区间较宽，因此15个百分点仅作为继续投入的工程门槛，不应直接作为论文最终结论。

## 13. 如何验证模型真的使用头部相机

在真实机器人运动前，可对同一批录制观测进行离线反事实推理：

1. 正常头部图像；
2. 头部图像全部置零；
3. 不同episode之间随机打乱头部图像；
4. 只保留头部图像，腕部图像置零；
5. 正常图像但关闭头部token；
6. 正常图像并启用人体/机器人token mask。

记录action chunk变化：

- 左右臂平移目标差异；
- 左右臂旋转目标差异；
- 夹爪目标差异；
- action chunk内部平滑性；
- 不同图像扰动下的输出方差。

若头部图像置零或打乱后模型输出几乎不变，说明模型没有有效使用头部相机。此时mask是否准确都不是首要问题，需要检查任务是否需要全局视野、模型多视角融合或数据中的头部图像与动作是否同步。

## 14. 完整真机实验建议

建议最终选择四个不同任务族，而不是把一个长时序任务的subtask分别计数。

### 任务1：蔬果分类放置

- 当前Task487任务族；
- 验证刚体抓取、语言条件和全局目标位置；
- 蔬菜和水果指令可作为两个类别，但论文中建议归为同一任务族。

### 任务2：拿毛巾—擦桌—放回

- 作为一个完整长时序任务；
- Pick Up Towel、Wipe Table、Place Towel属于subtask，不应声称是三个独立任务；
- 验证可变形物体、接触操作与长时序执行。

### 任务3：双臂展开或折叠毛巾

- 验证双臂协同；
- 验证可变形物体；
- UMI采集时人体区域通常更明显，适合检验本体泄漏问题。

### 任务4：大范围跨区域物体搬运

- 从桌面一侧抓取并放到另一侧目标区；
- 腕部相机难以同时观察起点和终点；
- 最适合证明头部全局视角相对于Wrist-only的价值。

### 14.1 最终评测规模

推荐强实验规模：

```text
核心比较：4任务 × 2主要方法 × 30次 = 240次
中间消融：4任务 × 3设置 × 15次   = 180次
总计约420次真机episode
```

资源受限的最低建议：

```text
核心比较：4任务 × 2主要方法 × 20次 = 160次
中间消融：4任务 × 3设置 × 10次   = 120次
总计约280次真机episode
```

## 15. 评价指标

### 15.1 主要指标

- 完整任务成功率；
- 各subtask成功率；
- 95%置信区间；
- 平均完成时间；
- 人工干预次数；
- 安全HOLD触发次数。

### 15.2 控制质量指标

- TCP路径长度；
- 平移速度和角速度；
- 加速度；
- jerk；
- 前进—后退反向次数；
- action chunk接受、过期和拒绝数量；
- 模型推理延迟和端到端观测年龄。

### 15.3 视觉与Mask指标

- 每帧mask面积比例；
- 被屏蔽token数量；
- 人体/机器人区域召回率；
- 任务物体误屏蔽率；
- 稳定化后背景关键点残差；
- 稳定化有效画面比例；
- RGB与token mask投影误差；
- 标定扰动下性能曲线。

### 15.4 统计要求

- 任务初始状态随机化但各方法使用相同分布；
- 尽量使用配对初始条件；
- 在实验前固定成功判定规则；
- 报告失败而不是只展示成功视频；
- 使用二项成功率置信区间；
- 如训练成本允许，使用多个训练随机种子；
- 对长时序任务同时报告最终成功率和阶段成功率。

## 16. 主要风险与对应诊断

### 风险1：模型本来就忽略头部相机

诊断：头部置零、打乱和Wrist-only消融。

### 风险2：人体区域包含有用动作线索

人体手臂可能帮助模型判断动作方向。屏蔽后可能降低训练效果。腕部相机和状态历史应提供局部动作信息，但必须通过Full与Stabilized对比验证。

### 风险3：稳定化损失有效视野

诊断：统计有效像素面积、裁剪比例、目标物体丢失率，并建立只稳定不mask的基线。

### 风险4：Mask与RGB token错位

诊断：逐阶段叠加可视化，并在最终224×224域检查16×16 token覆盖。

### 风险5：机器人几何标定误差

诊断：对相机外参和关节角加入受控扰动，报告mask覆盖及任务性能变化。

### 风险6：提升来自其他训练差异

诊断：所有模型使用相同数据、base权重、训练步数和超参数，只改变规定变量。

### 风险7：任务不需要头部全局视角

诊断：加入Wrist-only基线，并选择腕部相机无法同时观察起点、终点或全局目标的任务。

### 风险8：控制端问题掩盖视觉方法效果

诊断：先完成离线动作分析、固定观测回放和5-waypoint低速测试；记录调度拒绝、轨迹反向、推理延迟和HOLD原因。

## 17. 论文贡献建议

不建议将贡献写成：

> 我们给Pi0.5增加了mask并部署到机器人。

建议写成：

1. 识别并量化UMI头部视频中由相机运动和采集者本体泄漏造成的wearable-to-robot视觉域差异。
2. 提出训练端人体、推理端机器人对称的执行者无关视觉token屏蔽方法。
3. 将头部稳定化与token屏蔽结合，在保留全局任务信息的同时去除执行者外观。
4. 在刚体、可变形、长时序和双臂真实机器人任务上验证方法。
5. 通过Wrist-only、Raw、Stabilized、pixel mask和token mask消融证明提升来源。

### 17.1 可能的英文标题

> **Bridging Wearable-to-Robot Visual Gaps via Stabilized and Embodiment-Masked Head Views**

其他方向：

> **Learning Embodiment-Invariant Head-View Representations from UMI Demonstrations**

> **From Human-Worn Cameras to Robot Policies: Stabilizing and Masking Demonstrator Embodiment**

### 17.2 适合的结论边界

如果只在单台机器人和少量任务验证，不应宣称通用跨机器人泛化。可以合理声称：

- 缩小特定UMI到真实双臂机器人的视觉域差异；
- 改善头部视角的有效利用；
- 提升指定任务族中的真实机器人成功率；
- 对一定范围的标定误差和场景变化具有鲁棒性。

## 18. 投稿定位

CoRL重视机器人学习问题、创新性和被客观建立的真实机器人结论。该项目若只有四次成功演示，不足以支撑主会论文；若完成四个不同任务族、充分重复、受控基线、消融和失败分析，则具有投稿CoRL、RA-L、ICRA或IROS的潜力。

CoRL 2026投稿截止日期已过，实际应规划下一届CoRL或其他时间合适的机器人会议/期刊。

参考：

- [CoRL Call for Papers](https://www.corl.org/contributions/call-for-papers)
- [CoRL Instructions for Authors](https://www.corl.org/contributions/instruction-for-authors)
- [CoRL Review Criteria](https://www.corl.org/contributions/old_instruction-for-reviewers)

## 19. 分阶段实施路线

### 阶段0：完成no-mask基线

- 等待5k checkpoint；
- 验证服务端加载；
- 完成离线头部图像消融；
- 完成observation-only检查；
- 完成5-waypoint低速真实测试；
- 根据结果决定10k或30k checkpoint使用方式。

### 阶段1：验证稳定化和Mask数据质量

- 抽样稳定化视频；
- 抽样人体mask；
- 修复或确认resize/pad/token映射；
- 建立可视化和数值质量指标；
- 确认推理端机器人几何mask关节契约。

### 阶段2：单任务pilot

- 训练Wrist-only、Raw、Stabilized、Full版本至5k或10k；
- 完成离线反事实推理；
- 每种设置进行10至20次正常真机评测；
- 根据预注册Go/No-Go标准判断是否扩展。

### 阶段3：完整四任务实验

- 固定最终方法和基线；
- 完成四任务数据与训练；
- 完成280至420次真实机器人episode；
- 统计成功率、置信区间、控制质量和失败类型；
- 制作不超过3分钟的补充视频和项目页面。

## 20. 常用命令

### 20.1 服务端

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_server.sh /absolute/path/to/checkpoint/5000 8000
```

### 20.2 仅观察，不输出模型动作

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client.sh vegetable 127.0.0.1 8000
```

### 20.3 首次5-waypoint真实测试

必须在现场安全确认、CAN接口和机器人状态检查完成后执行：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

### 20.4 连续真实控制

仅在5-waypoint测试和日志检查通过后考虑：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute --continuous
```

### 20.5 本地测试

```bash
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi
export PYTHONPATH="$PWD/openpi-official/packages/openpi-client/src:$PWD/universal_manipulation_interface_ur:$PWD"
python -m pytest -q tests_task487
```

## 21. 当前已确定的决策

- 使用官方Pi0.5作为模型基础；
- 本地和云端使用同一份Task487配置；
- 先跑通Task487蔬果分类基线；
- 当前基线不启用mask；
- 使用头部、左腕up、右腕up三路相机；
- 数据与动作时间契约为25Hz；
- 推理请求每5个waypoint触发一次；
- 不直接照搬旧的单臂两相机伪RTC客户端；
- 第一阶段使用异步滚动调度，不修改模型端Flow Matching；
- 真实运动默认关闭，首次只执行5个waypoint；
- 正式mask训练前必须验证RGB与mask token对齐；
- 最终研究问题是UMI采集者本体泄漏和头部相机运动造成的训练—部署gap，而不是推理时偶发的人手干扰。

## 22. 完成标准

只有满足以下条件，才能认为研究链路真正闭环：

- no-mask基线checkpoint能够稳定加载和真实执行；
- 头部图像消融证明模型会使用头部相机；
- 稳定化和mask均有可视化及数值质量保证；
- 训练端人体mask与推理端机器人mask在同一token坐标系；
- 完整方法在正常机器人部署中优于Raw head；
- 完整方法在需要全局视野的任务中优于Wrist-only；
- 提升在多个任务和足够重复次数下具有统计可信度；
- 失败案例、限制和安全触发均被完整报告；
- 代码、配置、数据划分和checkpoint可以复现。

