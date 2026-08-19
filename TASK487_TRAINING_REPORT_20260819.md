# Task487 UMI 训练与真机验证日报（2026-08-19）

## 1. 报告范围

本报告记录 2026-08-19 在 Task487 双臂操作任务上的权重核验、mask 几何标定、
离线消融、Marvin 真机闭环测试、执行端修正和训练侧结论。时区为 Asia/Shanghai。

安全边界：离线分析没有发布机器人命令；真机轮次均通过 Mink FSM 的
HOME/HOLD/ACTIVE 状态机运行，操作员可随时按 `s` 进入 HOLD。本日报不把没有记录
成功数、物体位姿和误差距离的操作员观察解释为统计显著结果。

## 2. 当日结论摘要

1. 12.5 Hz 降采样没有丢失动作标签，也没有造成夹爪方向反转。
   400,010 个非终止样本全部满足 `action[t] == state[t+1]`，最大误差为 0。
2. UMI 实验权重的夹爪晚张开主要来自训练分布，而不是夹爪硬件或控制器延迟。
   UMI episode 以接近闭合状态开始；真机收到的夹爪目标也确实在运行 20--35 秒后
   才明显张开，目标与反馈基本一致。
3. “夹爪没有正对物体”不是旋转命令丢失。实验轮次中活动臂实际累计旋转约
   41--55 度，目标与反馈最大姿态误差约 1.4--2.4 度；策略发生了旋转，但方向或
   时机没有形成正确的预抓取姿态。
4. 当前 masked checkpoint 的 mask 确实进入了 Pi0.5，但作用机制较弱：完整 RGB
   先经过 SigLIP，再在 Pi0.5 prefix attention 中删除部分 head token。被遮挡区域
   已经可能通过视觉编码器的全局注意力污染保留 token。
5. masked checkpoint 训练时 mask 与 `resize_with_pad` 后的 RGB 不在同一坐标域。
   部署端现已生成对齐最终 224x224 图像的 mask，但这与旧权重训练时看到的 legacy
   token mask 不一致；修正推理 mask 无法逆转已经学到的训练分布。
6. mask 只能减弱人体/机器人外观差异，不能单独解决头戴相机运动、固定相机视角、
   夹爪协议、腕部预对准、双臂协同和机器人动力学差异。
7. 当前最有希望的路线是：训练 mask v2、训练 wrist-only 小模型、重整预抓取动作
   时序，并用少量真机成功/纠偏数据完成末端域适配。

## 3. 权重与运行契约

| 组别 | checkpoint | 频率 / horizon | 视觉输入 | HOME 夹爪协议 |
|---|---|---:|---|---:|
| 旧真机对照 | `checkpoints/pi05_umi_task487/task487_3cam_nomask_30k/29999` | 25 Hz / 20 | 三相机、无 mask、`center_square` | 右 34.0° / 左 24.0352°（物理安全端点） |
| UMI raw-head | `checkpoints/pi05_umi_task487_raw_12_5/raw_seed42/29999` | 12.5 Hz / 20 | head raw + 双 wrist、无 mask、`resize_with_pad` | 1° / 1° |
| UMI masked-head | `checkpoints/pi05_umi_task487_masked_12_5/masked_seed42/29999` | 12.5 Hz / 20 | head raw + 双 wrist、head token mask | 1° / 1° |
| UMI wrist-only | `checkpoints/pi05_umi_task487_wrist_only_12_5/wrist_only_seed42/29999` | 12.5 Hz / 20 | 双 wrist、head disabled | 1° / 1° |

三个 12.5 Hz UMI 组应使用单空格 vegetable prompt：

```text
Pick Up Vegetable and Place Vegetable on the Pink Plate on the Right
```

旧真机权重保留训练时的双空格 `Pink  Plate`。客户端现在根据服务端 runtime metadata
选择频率、horizon、图像几何、prompt 和夹爪边界，拒绝静默混用契约。

## 4. UMI 12.5 Hz 数据审计

全量数据包含 1,500 episodes、401,510 行、频率 12.5 Hz。

### 4.1 动作标签

- 400,010 个非终止样本全部满足 `action[t] == observation.state[t+1]`；
- 最大绝对误差为 0；
- terminal action 与 terminal state 完全一致；
- 因此不存在降采样导致的一帧偏移或标签损坏。

### 4.2 夹爪单位和方向

- 数据单位为弧度；
- `0 = 闭合`，数值越大表示越张开；
- 部署端直接 `rad2deg` 是正确映射，不应反转或做分位数重缩放。

### 4.3 夹爪时序分布

| 指标 | 右夹爪 | 左夹爪 |
|---|---:|---:|
| episode 内范围 >5° | 99.73% | 99.87% |
| episode 内范围 >10° | 85.93% | 97.07% |
| 20 步窗口内张开 >5° | 8.08% | 9.88% |
| 20 步窗口内闭合 >5° | 8.54% | 9.34% |
| episode 起始中位数 | 0.97° | 0.79° |
| episode 结束中位数 | 1.01° | 0.88° |

UMI 数据并非没有夹爪运动，而是大多数局部窗口没有明显张开，并且 episode 从近闭合
状态开始。纯真机样本则从约 34.9° 全开状态开始。这一协议差异会直接影响策略学到的
“何时张开”。

inactive arm 也有明显协议差异：vegetable 任务的 UMI 左夹爪有 90.0% 样本不超过
3°；fruit 任务的 UMI 右夹爪有 90.8% 样本不超过 3°。旧真机数据中的 inactive
夹爪则基本保持全开。

## 5. Mask 数据流与失效原因

### 5.1 当前语义

- 白色/255 像素表示需要屏蔽；
- 最终 SigLIP 网格为 16x16，共 256 个 head patch token；
- token 内被标记像素比例达到 50% 才删除该 token；
- mask 只作用于 head，相邻两个 wrist 视角保持全量 token。

### 5.2 训练几何错位

旧训练链先在原始 mask 尺寸上直接划分 16x16，再让 RGB 独立执行
`resize_with_pad`。640x512 RGB 缩放到 224x179 后，上下约有 22/23 像素 padding，
相当于约 1.5--1.6 个 14 像素 patch 的垂直错位。

正确链路应为：

```text
原始 RGB + 原始 mask
  -> 同一稳定化/裁剪
  -> 同一 resize_with_pad
  -> 最终 224x224 坐标域
  -> 16x16 token mask
```

部署端现在按正确链路生成 mask，但 `masked_seed42/29999` 已经在 legacy mask 上完成
训练，因此不能把推理端对齐修正视为对旧权重的完整修复。

### 5.3 Legacy / aligned 离线同噪声 A/B

使用 masked checkpoint、6 个数据样本、随机种子 42/43/44；确定性重复最大绝对误差
为 0。aligned mask 相对 legacy mask 每个样本改变 20--52/256 个 head token。

前五步动作平均差异如下：

| 任务 | 手臂 | 平移均值 / 最大 | 旋转均值 / 最大 | 夹爪均值 / 最大 |
|---:|---|---:|---:|---:|
| 1 | 右 | 0.81 / 2.32 mm | 0.14 / 0.37° | 0.03 / 0.12° |
| 1 | 左 | 0.66 / 1.71 mm | 0.10 / 0.35° | 0.01 / 0.03° |
| 2 | 右 | 0.55 / 1.38 mm | 0.06 / 0.21° | 0.02 / 0.04° |
| 2 | 左 | 0.62 / 2.21 mm | 0.08 / 0.24° | 0.02 / 0.07° |

对齐变化对动作不是严格为零，但幅度不足以解释或修复真机上的主要失败模式。

### 5.4 架构层面的弱屏蔽

当前实现顺序为：

```text
完整 RGB -> SigLIP image encoder -> image tokens -> Pi0.5 attention keep mask
```

mask 在 SigLIP 之后才参与 `image_input_mask`。视觉 Transformer 编码完整图像时，
masked patch 可以通过全局 self-attention 影响未被删除的 token。因此当前方法不是严格的
“模型看不见人体/机械臂”。

此外，真机 16:14 轮次几何 mask 标记 5,075/50,176 像素，约占 head 图的 10.1%；
离线 aligned 样本仅删除 27--69 个 head token。考虑三路视觉共有 768 个 token，信息
移除比例有限，双 wrist 和状态输入仍可主导动作。

## 6. 五姿态几何标定

### 6.1 采集与同步

2026-08-19 使用 cam_head_left 和五个双臂姿态完成标定：`start`、`left_plus`、
`left_minus`、`right_plus`、`right_minus`。相机与关节状态时间偏差分别为：

```text
10.181 ms, 7.252 ms, 15.680 ms, 6.551 ms, 6.599 ms
```

### 6.2 保存结果

标定在 15:44:43 保存并确认，五帧全部纳入。运行时配置为：

```text
T_camera_left_base translation  = [-0.013, 0.209, -0.040] m
T_camera_right_base translation = [ 0.076, 0.217, -0.014] m
```

与 2026-08-11 的右眼结果相比，左臂/右臂 camera-frame `tx` 分别变化 +22 mm 和
+57 mm；旋转不变。最终使用 equidistant fisheye、224x224 相机内参、capsule proxy、
2 px dilation，并在五姿态投影验证中得到 `status=pass`。

这解决了投影明显偏移，但不能修复训练 token mask 的历史错位，也不能改变策略的动作
时序和抓取姿态语义。

## 7. 真机执行端修正

### 7.1 12.5 Hz 调度

曾尝试把可执行 horizon 截为 0.8 秒并固定约 0.16 秒重规划；操作员反馈效果更差，
该方案已撤销。

当前 12.5 Hz 调度保持完整 20 步可替换 horizon，只把不可替换的物理 commit 窗口限制
为 0.4 秒，并在 committed prefix 剩余约 0.24 秒时预取下一次推理。同一 committed
endpoint 不会被重复请求。当前数据一致的单步保护上限为 45 mm 和 0.14 rad；25 Hz
旧真机运行保持 35 mm 和 0.12 rad。

结论：执行端可以改善时序交接和安全性，但不能把模型没有输出的“及时张开/正确正对”
动作凭空补出来。

### 7.2 夹爪安全端点

旧真机 runtime 最初要求 35°/35°，但 Marvin 实际稳定端点约为右 34.0°、左
24.0352°。16:19--16:20 两次启动因左夹爪无法达到 35°而被安全门拒绝。

客户端现使用独立安全端点 34.0°/24.0352°，保持模型度数一一映射，只在物理端点外
裁剪。16:23 的旧真机对照轮次已正常通过 READY/HOME 检查。

## 8. Masked checkpoint 真机证据

重点分析目录：`task487_logs/20260819_161437_4144777/`。

轮次：

| round | 任务 | ACTIVE 区间 | 约持续时间 |
|---:|---|---|---:|
| 1 | fruit / 左臂活动 | 16:15:00.593--16:15:33.972 | 33.4 s |
| 2 | fruit / 左臂活动 | 16:15:47.066--16:16:11.309 | 24.2 s |
| 4 | vegetable / 右臂活动 | 16:16:31.591--16:17:13.605 | 42.0 s |

### 8.1 夹爪目标与反馈

| round | 活动夹爪 | 首次 >3° | >5° | >10° | >15° | 最大值 |
|---:|---|---:|---:|---:|---:|---:|
| 1 fruit | 左 | 20.50 s | 20.73 s | 21.60 s | 23.06 s | 15.69° |
| 2 fruit | 左 | 19.99 s | 20.23 s | 21.11 s | 21.97 s | 16.29° |
| 4 vegetable | 右 | 35.54 s | 37.06 s | 38.26 s | 39.81 s | 15.66° |

控制器目标、接收 waypoint 和实际反馈基本一致。因此“夹爪晚张开”发生在策略输出，
不是 CAN 夹爪没有执行或反馈掉线。

### 8.2 腕部旋转目标与反馈

| round | 活动臂 | 目标累计旋转最大值 | 实际累计旋转最大值 | 最大跟踪误差 |
|---:|---|---:|---:|---:|
| 1 fruit | 左 | 55.85° | 55.10° | 1.64° |
| 2 fruit | 左 | 49.13° | 47.20° | 2.38° |
| 4 vegetable | 右 | 42.76° | 41.50° | 1.39° |

实际腕部在约 1.1--1.4 秒内超过 2°，约 2.7--2.8 秒内超过 5°，并持续旋转。由此可以
排除“旋转动作被执行端丢掉”；问题是策略旋转方向/时机没有形成正确抓取朝向。

### 8.3 操作员观察

- 接近物体时没有形成明显的夹爪正对姿态；
- 夹爪没有在接近阶段及时张开；
- mask 组定位仍有偏差，主观表现接近未处理 UMI；
- 操作员报告多个实验组有相似现象，但当日没有记录每组固定物体位姿下的逐次成功数，
  因此不能据此量化 raw/masked/wrist-only 的组间差异。

## 9. 旧真机对照权重

重点目录：`task487_logs/20260819_162342_509845/`。

- runtime：25 Hz、20 步、三相机、无 mask、`center_square`；
- HOME：右 34° / 左 24.0352°；
- fruit ACTIVE：16:24:03.493--16:25:12.715；
- vegetable ACTIVE：16:25:12.715--16:26:02.835；
- 共接受 123 个 policy chunks；
- 活动夹爪从开放边界开始，不存在 UMI 组先长期保持约 0--1°再张开的现象。

旧真机对照和 UMI 组使用同一套坐标变换及下层控制器。对照能够运行、UMI 组目标也被
准确跟踪，进一步说明主要差异来自训练数据/视觉域/任务协议，而不是 Mink 丢动作。

## 10. 根因分层

### 已由数据或日志直接支持

1. UMI 与旧真机的夹爪起止协议不同；
2. UMI 局部窗口内明显张开事件只占约 8--10%；
3. masked 训练和最终 RGB token 坐标存在错位；
4. mask 只删除 SigLIP 后的 head token，且不处理 wrist；
5. 真机夹爪和腕部反馈能够跟随控制目标；
6. 修改执行调度不能修复错误的动作语义。

### 高概率但仍需下一轮实验验证

1. UMI 示教把“先靠近、后张开”的人类操作习惯学进了模型；
2. 头戴相机运动与固定真机相机之间的运动域差异仍然显著；
3. 机械臂外观不是唯一域差异，因此只做 embodiment mask 收益有限；
4. 预抓取腕部朝向样本不足、分布不均或视觉条件不足，导致旋转存在但不正确。

### 当日没有证据支持

1. 夹爪弧度/角度方向反了；
2. 12.5 Hz 标签整体错一帧；
3. Mink 没有执行旋转；
4. 夹爪硬件导致 20--35 秒的张开延迟；
5. 单纯继续加大执行速度或改 action 映射能够解决定位问题。

## 11. 下一轮训练建议

### P0：先做低成本判别实验

1. 对 raw-head、masked-head、wrist-only 使用相同 seed、checkpoint step、物体位姿、
   prompt、HOME 和下层控制器；
2. 每个任务至少固定若干可复现 placement，记录 pick/place 成功、失败偏移方向与距离；
3. 若 wrist-only 不低于 raw-head，优先停用 head，避免继续投入复杂 mask；
4. 对同一离线 observation 使用相同 diffusion noise 比较三组动作，分离模型差异和采样
   噪声。

### P1：Mask v2

1. RGB 和 mask 在最终 224x224 坐标域严格同步；
2. 在进入 SigLIP 前，用数据集均值、受控噪声或 learned mask embedding 替换 masked
   patch，阻断视觉编码器内的信息泄漏；
3. SigLIP 后继续应用 token keep mask；
4. mask 边界膨胀至少一个 14 像素 patch，并重新评估 50% token 阈值；
5. 训练端人体 mask 与部署端机器人 mask 使用一致的覆盖语义和统计范围；
6. 先训练短程 pilot，完成离线敏感性和小样本真机 A/B 后再跑完整 30k。

### P1：动作数据重整

1. 重新切分或补录示教，使活动夹爪在接近物体前已经达到可抓取开度；
2. 增加“腕部先正对物体，再下降”的明确预抓取片段；
3. 提高张开 transition、预抓取旋转和抓取前关键窗口的采样权重；
4. 不要仅把夹爪 action 人工前移而保持 state/图像不变，避免制造新的状态动作不一致；
5. 明确 inactive arm 的训练协议，避免 UMI 闭合与真机全开的混合分布。

### P2：真机域适配

UMI 负责提供任务覆盖和大量演示，少量真机成功/纠偏数据负责学习固定相机、Marvin
外观、实际夹爪动力学和末端精度。这比追求“纯 UMI、零真机数据达到纯真机精度”更
现实。

## 12. 后续验收指标

下一轮必须记录：

- 每个 checkpoint、seed、task 和 placement 的成功数/总次数；
- 首次夹爪达到 3°、5°、10° 的时间和距物体阶段；
- 腕部预抓取角度、目标/反馈误差和开始旋转时间；
- 物体中心到抓取中心的像素/毫米偏差及方向；
- mask 像素占比、删除 head token 数、三路总 token 删除比例；
- raw/masked/wrist-only 的同噪声离线动作差异；
- 所有人工干预、HOLD、guard trip 和硬件异常。

在没有这些字段之前，只能报告操作员观察，不能宣称 mask 方法有效或无效。

## 13. 关键产物

- 数据审计：`offline_eval/TASK487_RAW_12_5_AUDIT_20260818.md`
- Mask A/B：`offline_eval/task487_mask_ab_20260819/results/REPORT.md`
- 五姿态标定：`offline_eval/task487_left_calibration_20260819_1540/`
- 运行时外参：`geometry_mask/configs/dual_arm_extrinsics_calibrated.yaml`
- Masked 真机证据：`task487_logs/20260819_161437_4144777/`
- 旧真机对照：`task487_logs/20260819_162342_509845/`
- 当前运行指令：`TASK487_CURRENT_RUN_COMMANDS.md`
- 当前运行契约：`task487_runtime/contract.py`
- 当前调度器：`task487_runtime/scheduler.py`

以上日志和离线产物路径指向原开发机。GitHub deploy 仓库按源码归档策略不包含原始
日志、相机帧、视频及 NPZ policy chunks；报告保留路径，便于在开发机上复核原始证据。

## 14. 代码归档

2026-08-19 已将源码分别归档到 GitHub 私有仓库：

- `SEEe-D/task487-umi-deploy`
- `SEEe-D/task487-umi-train`

归档排除了 checkpoints、datasets、视频、日志、虚拟环境、SAM 权重和凭据。部署发布
副本通过 72 项 Task487 测试；训练端 mask/model 相关测试在训练服务器环境通过。
