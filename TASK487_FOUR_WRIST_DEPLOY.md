# Task487 四路腕部推理配置（2026-09-05）

开发机源代码：`/vepfs-C区/GJ/task487_umi_training/code/src/openpi/`。
核对 `training/config.py`、`policies/umi_policy.py`、`models/pi0_config.py`：
RAW 与 CNC 均为头部 + 四腕；Wrist4W 关闭头部，但模型保留置零且 image mask=False 的头部槽位。
CNC 的 `cnc_mask_v2` 是训练目标；云端推理 metadata 指定 `mask_enabled=False`。

## 输入映射

| 物理视角 | Thor 标签 | 视频/元数据端口 | UmiEnv 键 | 请求键 | 模型键 |
|---|---|---|---|---|---|
| 头部左目 | `cam_head_left` | 5000/6000 | `camera5_rgb` | `cam_head` | `base_0_rgb` |
| 左腕上 | `cam_hand_l_top` | 5002/6002 | `camera2_rgb` | `cam_left_top` | `left_wrist_0_rgb` |
| 左腕下 | `cam_hand_l_bottom` | 5003/6003 | `camera3_rgb` | `cam_left_down` | `left_wrist_1_rgb` |
| 右腕上 | `cam_hand_r_top` | 5004/6004 | `camera0_rgb` | `cam_right_top` | `right_wrist_0_rgb` |
| 右腕下 | `cam_hand_r_bottom` | 5005/6005 | `camera1_rgb` | `cam_right_down` | `right_wrist_1_rgb` |

端口取 Thor 相机列表原始下标，禁用头部右目不会重编号。
`bottom` 和旧 `btm` 都显式映射；未知标签与重复槽位直接报错。
客户端从经过严格校验的 runtime metadata 选择相机，缺帧、过期或超过 50ms 偏差会阻止执行。
RAW/CNC 订阅 5 路，Wrist4W 订阅 4 路，旧真机仍使用原三路。
预览显示实际请求内全部 RGB 画面；首次执行前可据此核对现场摄像头位置与标签。

## 配置与权重

| 别名 | 本地配置 / runtime（runtime 后缀 `_v1`） | checkpoint 相对 `checkpoints/` 路径 |
|---|---|---|
| `raw4w` | `pi05_umi_task487_raw_4w_12_5` | `pi05_umi_task487_raw_12_5/raw4w_seed42/29999` |
| `cnc` | `pi05_umi_task487_cnc_4w_12_5` | `pi05_umi_task487_cnc_12_5/cnc8gpu_seed42/29999` |
| `wrist4w` | `pi05_umi_task487_wrist_only_4w_12_5` | `pi05_umi_task487_wrist_only_12_5/wrist4w_seed42/29999` |

三组共用原 UMI 时序：12.5Hz、20 步 horizon、5 步 RTC prefix、224×224 full-FOV resize-with-pad、
HOME 夹爪 1°/1°、模型弧度与夹爪物理角度一一转换。RAW 的训练头部字段为 `head_raw`，
CNC 为 `head_fixed`；实机都使用头部左目的当前 RGB。未引入分组位姿偏移或夹爪缩放。
新配置只用于部署，没有把 CNC 训练损失实现移入本地训练框架。

启动及键盘操作见 [TASK487_CURRENT_RUN_COMMANDS.md](TASK487_CURRENT_RUN_COMMANDS.md)。

## 夹爪闭合排查与修复（2026-09-05）

当前生效方案 A（用户要求回切）：三组四腕均为正确训练指令 + 原滚动重规划，
`complete_chunk_before_replan=False`。保留相机、夹爪角度映射、限速和安全检查。
以下整块执行与重放结果是之前方案 B 的历史记录，不代表当前启动行为。
回切原因是用户报告抓取后不张开，且整块执行的实测重规划间隔约 5–12s；
本次用于区分指令修正与调度修改的影响，不宣称已经解决释放问题。回切后真机结果待测。

开发机实际训练数据 `full1500/lerobot_combined_12_5/meta/tasks.parquet` 只有一个任务：
`Vegetable and Fruit Sorting`。训练开启 `prompt_from_task=True`，三组共用这套任务索引。
先前客户端误用旧两腕数据的 vegetable/fruit 指令；现三组四腕统一为 `sorting`，
原两个 CLI 名称仅作带警告的兼容别名，UI 不再提供虚假的分类指令选择。旧真机/两腕配置不变。

日志 `20260905_104917_3910206` 第 5 轮出现模型尾段要闭合、但限速后未下发尾段
被下一次推理替换的情况。之前方案 B 仅四腕契约启用了 `complete_chunk_before_replan=True`：
保留整个已通过检查的动作块，所有剩余点已下发并接近完成时才提交下一次 RTC 请求。
控制器仍只接收至多 5 个点、通常 0.4s 的窗口（原有单个长限速段例外不变），
并非一次预载几秒轨迹。0.02m/s、0.08rad/s、工作区/反馈检查和 HOLD 清队列均未放宽。
代价是限速后的整块执行可能持续数秒，模型对新场景的重规划变慢；不会脱离位姿强制闭爪。

离线重放该日志 sequence=776 的同一条已接受路径，以记录的控制器目标为起点重新计时，
模拟理想目标跟随（不运行模型、不连接机器人）：

- 旧调度在 0.08s 请求重规划，此前仅下发 1/17 点，夹爪目标 15.25°，尚未下发闭合。
- 新调度在 9.84s 请求重规划，此前下发完整 17/17 点；首个 <3° 目标在 7.235s 到期，
  下发目标最小 0.185°。下一次 RTC 前缀仍只有 1 点。

这是调度进度验证，不是抓取成功证据；不能据此排除模型本身仍有问题。
原日志在开启阶段目标与反馈接近，数据集存在大量闭合动作，暂不支持“只因夹爪没力或单位错误”的解释。

```bash
PYTHONPATH="$PWD" /home/simpleai/anaconda3/envs/openpi/bin/python \
  offline_eval/replay_task487_closing_tail.py task487_logs/20260905_104917_3910206 --sequence 776
```

## 验证

- 文本/尾段修复后全套 87 项回归测试通过，覆盖 12.5Hz 实际循环间隔、快/慢轨迹、
  RTC 请求期间前缀部分/全部执行、5 点下发上限、暂停清队列、物理反馈保护及旧配置兼容。
- 初次迁移的 77 项 Task487 回归测试通过，包括用不同像素值核验物理槽位→请求键→模型键、
  四腕缺失/陈旧拒绝、Wrist4W 无头部、五路预览和旧三路兼容。
- Thor 5 路同时接收成功，全部为 640×512；检查时经时钟补偿的画面年龄约 6–15ms，
  通过客户端 250ms 新鲜度和 50ms 跨相机偏差检查。接收检测结束后已释放端口。
- `offline_eval/verify_task487_four_wrist.py` 加载实际权重及各自 norm stats，
  初次迁移时使用合成图像/状态分别检查普通和 RTC 推理，三组共 6 次均通过，输出为有限数值的 `20×20` 数组。
  这 6 次先前使用旧文本，只证明权重/输入结构兼容；脚本现已同步正确文本，本次未重新运行模型推理。
  该脚本无机器人连接或动作发布。
- 上述验证不代表抓取表现；新三组尚未记录本次修改后的真机试验结果。

重跑离线兼容性检查（从 deploy 目录）：

```bash
export PYTHONPATH="$PWD/openpi-official/src:$PWD"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
  openpi-official/.venv/bin/python offline_eval/verify_task487_four_wrist.py wrist_only
```
