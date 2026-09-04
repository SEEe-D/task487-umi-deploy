# Task487 部署常用指令

当前部署架构：

```
GPU 机器
    |
    | WebSocket :8000
    |
    v
机器人 PC
    |
    ├── Thor camera
    ├── task487_client.py
    └── Tianji/Marvin Mink ROS bridge
            |
            v
          真机
```

当前后端为 **Tianji/Marvin Mink ROS bridge**，不是 UR7e/UR RTDE。

---

## 1. 启动机器人后端

在机器人 PC 上启动 Marvin/Mink 控制桥：

```bash
cd /home/simpleai/Code/mjm/eval_mink

./start_teleop_replay_impedance.sh enable_intervention:=false
```

启动成功应确认双臂进入阻抗模式。

---

## 2. 启动 Pi0.5 Policy Server

GPU 机器：

```bash
bash run_task487_server.sh
```

指定 checkpoint：

```bash
TASK487_POLICY_CONFIG=pi05_umi_task487 \
  bash run_task487_server.sh \
  /path/to/checkpoint 8000
```

server 负责：

- 加载 Pi0.5 checkpoint
- 接收 observation
- 执行 policy inference
- 返回 action chunk

---

## 3. 启动 Task487 真机客户端

观察模式：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000
```

真实执行：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

显示策略输入图像：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 \
  --execute --show-processed-cameras
```

---

## 4. 运行顺序

推荐三个终端：

### Terminal 1

启动 Pi0.5 server：

```bash
bash run_task487_server.sh
```

### Terminal 2

启动 Marvin/Mink 后端：

```bash
./start_teleop_replay_impedance.sh enable_intervention:=false
```

### Terminal 3

启动客户端：

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

---

## 5. 调试测试

RTC contract test:

```bash
pytest -q tests_task487
```

RTC offline verification:

```bash
python offline_eval/verify_task487_rtc.py
```

---

## 6. 关闭顺序

1. 客户端按 `s` 进入 HOLD，然后退出。
2. 停止 Marvin/Mink 后端。
3. 停止 Pi0.5 server。
