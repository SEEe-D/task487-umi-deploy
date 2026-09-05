# 5°补偿闭环反馈与右手到左手交接修复

用户要求修复 18:58 试用中“右手夹牢但未释放，左手已开始抓取，随后 HOLD”的问题。
实现已完成；没有自动启动机器人或修改后端限矩/速度。

## 修改

1. 调度器同步生成实体夹爪轨迹与原始模型参考轨迹，共同重定时、窗口截断及前缀衔接。
   `ScheduledAction.gripper_policy_target` 经 UmiEnv 路由到对应右/左控制器，与实体 waypoint 使用同一时刻。
2. 控制器每个 tick 记录实际发送的实体目标和模型参考，将反馈转换为
   `policy_actual = policy_target + measured_actual - physical_target`。
   这消除已知执行偏置的闭环重复作用，并保留包括接触阻挡在内的跟随误差。
   使用实际裁剪/插值后的目标，不在方向切换瞬间对整个反馈直接跳加/跳减 5°。
3. 仅启用新双手补偿时，模型 pre_state/state 的夹爪两列使用该等效开度。机械臂状态与图像不变；
   原始实体角度仍作为机器人 live 状态、控制保护和单独日志的依据。缺失等效反馈时拒绝构造补偿请求。
4. 5°专用脚本加入 `--sync-right-before-left`。每轮启动先冻结左臂/夹爪，右手需经历打开、闭合、再打开，
   其实体开度达到保存的打开目标（欠开容差 0.5°）至少 0.16 秒，才放行左手。
   未达到时，左手模型动作被替换为静止目标；右手运动和实际双臂跟踪仍通过原检查。
   该门控确认开度到位，不声称已识别物体落盘，也不是完整多物体规划器。
5. 同一客户端 `s` / `d` 恢复保留补偿方向及已知交接阶段；HOME+准备完成后重置。
   新进程首次应 HOME+准备。原入口默认无补偿，新入口仍闭合/张开各 5°。

## 日志

- `gripper_*_trace.csv` 增加 `policy_target_deg`、`policy_actual_deg`、`applied_offset_deg`；原实测角度/命令不改含义。
- 实际发送模型的请求仍原样存 NPZ；diagnostics 明示使用 `policy_reference_plus_measured_tracking_error` 坐标，
  并保存两帧原始实体 `measured_gripper_degrees`。这是一种执行偏置换算，不是实体角度的新标定。
- chunk NPZ 增加与实体提交队列对齐的 `gripper_policy_targets`；JSON 增加 `gripper_handoff` 阶段。
- 启动打印 `Compensation feedback=v2 ... right-before-left=True`；右手张开到位后打印左手放行事件。

## 验证

**200 项离线测试通过**（`pytest.txt`）。包括原测试、新旧三种模型客户端生命周期、成对目标完整路由、
反馈状态选择和原始实测存档、真实插值器限速/前缀连续性/端点截断、暂停保持、
未过期预测与实际开度等待、到位持续时间、右手运动保护未弱化。

`smoke_controller.py` 在临时 localhost UDP 端口模拟桥，使用真实 `RosGripperController` 子进程、共享内存、waypoint 队列和 CSV 记录。
不连接正式 6010/6013/6014 端口、ROS 或 CAN。

| 检查 | 实体目标 | 实体实测 | 模型等效反馈 |
|---|---:|---:|---:|
| 闭合，模型参考 12.8° | 7.8° | 7.8° | 12.8° |
| 连续 8 轮读取反馈后预测保持 | 约 7.8° | 约 7.8° | 约 12.8° |
| 打开，模型参考 14.8° | 19.8° | 19.8° | 14.8° |
| 模拟物体挡在 10°，参考 12.8° | 7.8° | 10° | 15° |

子进程正常退出；共 571 个模拟命令包，真实机器人命令 0。完整结果在 `controller_smoke.json` / `controller_smoke/`。
最后一行证明没有以目标值替代真实跟随误差。

`check_policy_feedback.py` 对实际 Wrist4W 服务执行 12 次推理：使用 18:28 请求 14 的同一张持物画面，
模拟刚性物体阻挡在 7.8°，比较旧原始状态输入与新版执行偏置逆变换；各做 6 轮反馈。

- 旧方式模型输入 7.8°，实体目标持续在 **2.81–2.99°**，接触阻挡误差持续约 5°。
- 新版模型输入 12.8°，实体目标 **7.64–7.74°**，没有继续向 0°收紧。

结果在 `policy_feedback.json` / `policy_feedback_predictions.npz`。画面固定、接触为模拟、噪声未做配对控制，
这个实验验证反馈漂移问题，不能代表真实抓取、释放成功率或证明左手交接已完成实物验证。

## 运行与边界

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
bash run_task487_client_sync_gripper5.sh sorting 127.0.0.1 8000 --execute --continuous
```

先 `r` 归位准备，再 `d`，`s` HOLD。模型服务与后端不用重启。
原来的 `run_task487_client_sync.sh` 不带补偿参数仍关闭补偿。
右/左端点约 34°/24.035°、0°闭合端点、8.594°/s、双臂 0.15 m/s / 0.35 rad/s、0.71 s 窗口和 5 cm 检查保留。
大幅开合仍需要时间，可能触发其他预测接入保护；未将本次修复解释为提高保护阈值。
实体效果需新一轮日志确认。本次未重新抓取、未调用 HOME 或发送真实夹爪动作。

修改前后快照 `before/` / `after/`、SHA256 `manifest.json`、差异 `changes.patch`。
