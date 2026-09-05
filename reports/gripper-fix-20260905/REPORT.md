# 夹爪执行链修复（2026-09-05）

**17:14 已完成源码修改、离线验证和本地安装。后端与客户端尚未重启，当前进程仍运行旧版。**
本次没有发布机器人动作，没有提高 CAN 速度或力矩，没有更换模型、单位映射或夹爪端点。
修复依据见[前序诊断](../gripper-diagnosis-20260905/REPORT.md)。

## 已修复

### 堵转后误解锁

`gripper_can_node` 现在根据**实测夹爪位置**判断请求是否打开超过既有 1°死区，
不再用“额外闭合 2°”的保持目标做解锁基准。
因此，接触时实际 9.81°、请求 9.67°会保持锁存；真正请求更大开度时才解锁。
如果夹爪确实进一步压缩了物体，解锁基准也跟随新的实测位置，避免固定接触角造成无法释放。

实现与测试位于后端 `ros2_ws/src/gripper_can_node/`：
`src/stall_release.hpp`、`src/gripper_can_node.cpp`、`test/test_stall_release.cpp`、`CMakeLists.txt`。
新的驱动已编译并原子替换实际安装目录中的二进制。
原始二进制保存在 `before/gripper_can_node`。

### 夹爪与手臂的共同到达时间

- 新增 `task487_runtime/gripper_dynamics.py`，从后端实际使用的已安装 `gripper_can.yaml`
  读取电机速度。当前 3 rad/s、减速比 20，换算为 **8.594°/s**。
- `task487_client.py` 将同一个速度同时传入调度器和夹爪插值器。配置缺失、速度非法或
  不支持的控制模式会在创建机器人环境前报错，不静默退回旧 35°/s 假设。
  如后端使用其他配置路径，客户端用 `--gripper-config` 或 `TASK487_GRIPPER_CONFIG` 指向同一文件。
- `RollingScheduler` 的最短分段时间同时考虑双臂平移、旋转和左右夹爪开合，
  不再先安排手臂完成、随后由夹爪控制器私自延长时间。
- `AuthorSyncScheduler` 同样按联合约束分配时间。夹爪本身需要的行程时间不会被误判为
  手臂超速；手臂额外延时上限仍为 150 ms，平移/旋转速度、跳变和跟踪保护仍生效。
- 新客户端把重定时后的轨迹截取为短的**物理执行窗口**。12.5 Hz 默认窗口最多 0.71 s，
  边界落在段内时用同一个进度插值所有关节；窗口内完整原始夹爪节点保留。
  下一次推理仍按原有 6 步、名义 0.48 s 请求，不因夹爪慢而提交数秒不可替换的旧动作。
  窗口以外的未来动作继续由后续推理决定，不保证整个原始 20 步块逐点执行。

## 验证

- **141 项 Python 回归通过**，覆盖三组模型的客户端生命周期、共同速度配置、左右夹爪、
  12.5/25 Hz 调度、快开合、部分轨迹截取、真实低层插值器、原有保护与退出行为。
- **C++ CTest 通过**。独立测试程序验证 8 个场景：接触闭合、保持、死区噪声、主动打开、
  压缩物体后重新打开、闭合端点打开、非法指令及非法反馈。
- 原日志中 **327 条接触后 20 ms 内误解锁的闭合请求**，调用实际 C++ 修复函数回放全部通过，
  均不再解锁。这个结果说明误解锁条件已修复，不代表真实夹持成功率已测得。
- CAN 节点完整编译成功，动态库依赖均解析成功。
- 将两次 16:49 记录的相对手臂动作重新绑定到模拟 TCP、保持原始绝对夹爪动作，
  通过两个机械臂及两个夹爪的真实 `PoseTrajectoryInterpolator` 回放：
  `164935` 的 **7/7** 块通过；`164906` 的前 **6/7** 块通过，第 7 块仍触发 5 cm 跟踪保护
  （5.59 cm），未放宽该保护。
- 上述通过的块，4 个控制器的衔接前轨迹差小于 `1e-8`；底层额外延时最多约
  **0.48 微秒**（UNIX float64 时间量化），夹爪最大采样速度约 **8.59438°/s**，
  物理执行窗口不超过 0.710001 s。手臂仍在 0.5 m/s、0.5 rad/s 测试上限以内。

回放没有重新运行视觉模型：后续模型预测仍来自旧观测，不能当作新策略闭环或真机效果。
作为对照，直接使用原记录绝对 TCP 目标、完全不重绑定时，两段分别在第 5/7 块触发保护，
因为修复后的较慢轨迹已改变请求时的 TCP 基准。两类结果均保存在 `motion_replay.json`，未隐藏拒绝块。

## 安装与启用

客户端 Python 修改已在 `gj/pi05-deploy`。新后端可执行文件已安装到：

```
/home/simpleai/Code/mjm/eval_mink/ros2_ws/install/gripper_can_node/lib/gripper_can_node/gripper_can_node
```

安装使用临时文件 + 原子替换，不覆盖正在执行的 inode；安装后核对 `/proc/3086540/exe` 的 SHA256，
仍与原始二进制相同，确认没有干扰旧驱动进程。安装清单见 [binary_install.json](binary_install.json)。
CAN YAML 保持原有电机速度 3 rad/s、开/合限矩 1.05 Nm 和保护设置。
没有运行 `cmake --install` 去覆盖现场 YAML，只替换了已验证的驱动二进制。

**启用需要同时重启后端和客户端；仅重启客户端不会加载驱动修复。**
按已有流程先客户端 HOLD/退出，再退出后端并按原命令重启；模型服务无需因本次修改重启。
重启客户端后应看到：`Gripper timing: 8.594 deg/s joint (CAN 3.000 rad/s motor, gear 20)`。
驱动主动解锁事件的新日志包含 `rearmed by measured open command` 及实测角度。
启用后先用已有受控短段方式检查抓取与释放，再评价效果。

仍需真机确认物体接触、实际夹持力与脱离。原诊断中的“达到限矩但未到目标角度”可能是正常接触，
本次没有凭该现象提高力矩。此次修复也不能保证模型在每个放置时刻都会输出足够的开度。

## 证据与复现

- [Python 回归](tests.log)、[完整编译](build.log)、[C++ CTest](ctest.log)
- [327 条日志回放](stall_replay.log)、[动作回放](motion_replay.log)、[逐控制器指标](motion_replay.json)
- [原始文件清单](before_manifest.json)、[安装清单](binary_install.json)、`before/` 原始备份
- `changes.patch` 为代码改动；`after_manifest.json` 记录最终源码和二进制 SHA256。

```bash
cd /home/simpleai/Code/universal_manipulation_interface-main/gj/pi05-deploy
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:openpi-official/src:openpi-official/packages/openpi-client/src:universal_manipulation_interface_ur \
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 \
  /home/simpleai/anaconda3/envs/openpi/bin/python -m pytest -q -p no:cacheprovider tests_task487
PYTHONPATH=.:universal_manipulation_interface_ur \
  /home/simpleai/anaconda3/envs/openpi/bin/python ../gripper-fix-20260905/replay_motion.py
```

以上测试/回放均无机器人连接或动作发布。
