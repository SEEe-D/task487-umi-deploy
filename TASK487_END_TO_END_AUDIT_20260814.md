# Task487 数据—训练—部署端到端复核（2026-08-14）

## 结论

右臂伸到桌下不是单一的限速或相邻 action 阈值问题。直接原因是：模型在当前
真机视觉域中开始生成连续向下的右臂目标，而部署曾允许一次开环执行 15 个
25 Hz waypoint，并且没有累计工作空间下界。相邻步都合法、机械臂也正常跟踪时，
目标仍可一段一段累积到桌面以下。

训练数据和 30k checkpoint 没有发现维度、双臂顺序或时间标签整体错位；但数据
没有独立验证集，当前真机画面与训练画面明显出域，模型又只接收相对 EEF 状态，
不知道绝对桌高。因此“训练内拟合健康”不能保证当前场景安全闭环。

## 数据链

- 230 episodes，150,907 帧，25 Hz。
- 同 episode 相邻有效对满足 `action[t] == state[t+1]`。
- 20 维布局为 `right pose9 + gripper + left pose9 + gripper`。
- pose9 为 XYZ + 旋转矩阵前两行按行展开的 rotation6d。
- Task1 是右臂蔬菜到右侧粉盘，Task2 是左臂水果到左侧蓝盘。
- 相机键为 head、left wrist up、right wrist up。复核数据记录器后发现旧部署错误地
  把 `cam_head_right/5001` 当作训练 `head_main`；已改为训练实际使用的
  `cam_head_left/5000`。左右腕仍为 5002/5004。
- 数据记录器对所有相机执行 `center_square` 后 resize 到 224x224。部署此前把
  640x512 直接拉伸到 224x224，造成横向压缩；现已改为先左右各裁 64 像素得到
  512x512，再缩放为 224x224，与训练记录器一致。

现有真机/训练对照还显示桌布、背景、相机姿态、机器人外观和物体布置均不同。

## 训练链

- 配置：`pi05_umi_task487`，Pi0.5，action horizon 20，有效 action/state 20 维。
- 状态历史为前一帧和当前帧；训练变换把当前相对前一帧转为 body-frame 状态。
- 绝对未来 TCP action 相对当前 EEF 转为 body-frame action。
- checkpoint norm stats 对应上述相对量，不是绝对机器人工作空间。
- 30k 训练内 first5 平移误差约右 1.30 mm、左 0.50 mm，优于 10k/20k。
- 全部 episode 都参与训练，没有未见场景验证集。

关键安全含义：模型状态中没有绝对桌高。它可以在训练域靠视觉学会何时停止下降，
但部署视觉出域时，没有任何学习侧变量能保证 TCP 不穿过桌面。

## 坐标与机械臂链

- `robot0 = Marvin B = right`，`robot1 = Marvin A = left`，顺序一致。
- 右臂固定基座变换同时作用于 state 和 RTC absolute action，随后二者都转为相对
  body delta；1000 组随机 SE(3) 验证最大差 `7.13e-10`，固定左乘变换会抵消。
- 当前 `R_M2R` 将模型的主要接近轴映射为沿桌面方向；替代 Marvin 矩阵会把录制
  HOLD 的相同动作映射为竖直向上，因此没有证据支持替换。
- Mink 使用 `target_frame: tcp`（Link610/710），反馈和目标是同一个 TCP 语义。
- replay 启动脚本锁腰，未启用 chy lift；不存在额外累计 trunk Z 偏移。
- Marvin 分支有意忽略 UR 风格 flange-to-tip offset，因为 Mink 直接解 TCP。

## 部署复现

用录制 HOLD 图像、初始 TCP、同一个 30k 服务做合成闭环，始终不连接 ROS：

| 调度 | 图像 | 跌破动态 floor |
|---|---|---:|
| 执行15/提交20/RTC5 | 640x512 后补边 | 3/3 |
| 执行15/提交20/RTC5 | 224x224 方图 | 3/3 |
| 执行5/提交10/RTC5 | 640x512 后补边 | 0/3 |
| 执行5/提交10/RTC5 | 224x224 方图 | 1/3，位于未提交 suffix |

合成闭环的图像不会随虚拟机械臂移动，不能当成功率评测；但它准确证明了两点：

1. 仅改图像尺寸不能消除下探；
2. 15 步开环显著放大危险，而 5 步 RTC 加发送前 workspace 校验可以 fail-close。

## 已实施修复

- 调度改为每完成 5 步重规划、最多预装 10 步、保留 5 步 RTC hard prefix。
- 保留 5 步 suffix 平滑衔接和 25 Hz waypoint 时间语义。
- 三相机在 UmiEnv 中心方裁剪后输出 224x224；contract 对尺寸严格校验。
- TCP 高度限制按现场调试要求暂时关闭；其他异常保护仍启用。
- 平移/旋转物理限速进一步降至 0.02 m/s、0.08 rad/s。

测试：`36 passed`。离线原始结果见：

- `offline_eval/task487_20260814/closed_loop_image_geometry_audit.json`
- `offline_eval/task487_20260814/closed_loop_5step_rtc_audit.json`

## 真机下一步

18:30 日志已完成四次默认五 waypoint 单轮：物理限速与 workspace floor 生效，右臂
五点净位移分别约 0.8/2.5/1.5/4.9 mm，均自动 HOLD，未复现累计下探。但这些轮次
`rtc_prefix=0, blend=0`，尚未覆盖 RTC handoff。旧日志中 HOME 后新轮收到 154 mm
tracking error；从约 132 ms 的返回时间无法断定它是跨轮旧结果还是本轮随机采样的
离群输出。客户端现已给每个 ACTIVE round 编号并丢弃跨轮 stale result，同时记录
`round` 和 `obs_age`，后续可明确区分两者；本轮离群仍由 tracking/workspace guard
在发送前拒绝。

下一步只运行有界 RTC 验证：`--execute --max-waypoints 10`。它经过一次五 waypoint
重规划/RTC handoff 后自动 HOLD。应看到后续 chunk 的 `rtc_prefix>0` 和 `blend>0`；
若方向稳定且没有 workspace/追踪错误，再考虑显式 `--continuous`。

按现场调试要求，部署端 TCP 绝对高度与单轮下探限制现已临时关闭；物理限速、单步/
tracking error、RTC round 隔离及有界自动 HOLD 保持开启。ACTIVE 日志会明确打印
`workspace height limit DISABLED`。

当前模型仍有明显视觉域差异；如果单轮目标方向仍无语义，应补当前 Marvin 场景数据
或做域对齐训练，而不是继续调整 action 阈值或取消 workspace guard。

## 头相机根因复核

训练记录器将 `observation.images.head_main` 固定写自 `cam_head_left`；Thor 端口表为
左目 5000、右目 5001。旧部署把 5001 右目送给模型。进一步只读检查 Thor 当前
`/home/simpleai/project/thor-stream-sender/config.yaml`，发现左目被显式设置为
`enabled: false`：5000 实测 8 秒无帧，5001 正常。`/dev/video1` 左目本身报告
`Camera 0: ok`、1920x1536@30Hz，因此目前是 sender 配置禁用而非设备消失。

本地 client 已改为训练一致的 `cam_head_left/5000` 并会在无帧时拒绝启动。恢复真机
验证前，必须先在 Thor 上启用左目；为保持三路编码负载不变，建议同时禁用未使用的
右目，然后重启 root sender。未恢复 5000 前不要继续执行策略。

## 19:05 左头相机恢复与在线无执行验证

Thor 主配置现已改为启用 `cam_head_left`、禁用 `cam_head_right`，原配置备份为
`config.yaml.bak_before_task487_headleft_20260814_1902`。由于 systemd 重启需要交互式
sudo，当前先由 tmux 会话 `task487-headleft` 单独发送 `/dev/video1` 到 5000/6000；
它不操作 GPIO，继续复用主发送器的 25 Hz FSIN。5000 已实测收到 640x512 左头画面，
模型端预处理实测为中心裁剪 512x512 后缩放至 224x224，桌面、茄子和粉盘均在视野内。

随后运行不带 `--execute` 的在线诊断两轮。三路相机全部通过 age/skew 检查，模型每轮
前十点意图相对当前 TCP 为约 0.3--1.1 mm、0--0.2 度；未再出现错误右目输入时的
20--60 mm 往复跳变。诊断客户端已 Ctrl+C 正常退出，机械臂保持 HOLD。

回归测试使用本项目 `openpi-official/src`，结果为 `37 passed`。下一次真机只做
`--execute --max-waypoints 10` 有界验证，不直接恢复 continuous。
