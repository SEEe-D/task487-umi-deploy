# OpenARM CAN - Maita Motor Driver

OpenARM CAN 是一个高性能 C++ 电机控制库，专为 **Maita ** 系列电机设计，支持高频 MIT 运控协议。本项目基于 Linux SocketCAN，提供非阻塞 IO 接口。

---

## ⚠️ 算法对接注意事项 (Critical for Algorithms)

对接算法时，请务必注意 **不同控制模式下的单位差异**：

| 控制模式 | 接口函数 | 位置单位 | 速度单位 | 力矩单位 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MIT 模式** | `mit_control_all` / `mit_control_one` | **Rad (弧度)** | **Rad/s** | **N-m** | **算法推荐模式**。高频运控使用。 |
| **位置模式** | `position_control_one` | **0.01 Degree (整数)** | N/A | N/A | 仅用于调试/归零。传入 `36000` 表示 360 度。 |
| **速度模式** | `speed_control_one` | N/A | **0.01 dps (整数)** | N/A | 仅用于调试。`dps` = deg/s。 |
| **读取状态** | `get_position()` / `get_velocity()` | **Rad (弧度)** | **Rad/s** | **N-m** | 所有 `get_` 接口统一返回标准单位 (double)。 |

---

## 📚 目录

*   [快速集成 (C++ API)](#-快速集成-c-api)
*   [环境要求](#-环境要求)
*   [编译指南](#-编译指南)
*   [详细 API 说明](#-详细-api-说明)
*   [CAN 配置](#-can-配置)

---

## � 快速集成 (C++ API)

### 1. 初始化

```cpp
#include <x3arm/can/socket/x3arm.hpp>

// 初始化 CAN 接口 (例如 "can0")
// 第二个参数 enable_fd: Maita 电机是标准 CAN 2.0，设为 false
x3arm::can::socket::X3Arm arm("can0", false);

// 初始化电机 ID 列表
// MotorType::MT4310 适用于大多数 Maita 电机
arm.init_arm_motors({MotorType::MT4310, MotorType::MT4310}, {1, 2}); 
```

### 2. MIT 运控循环 (推荐)

这是算法主要使用的接口。

```cpp
// 定义 PD 参数
x3arm::maita_motor::MITParam param;
param.kp = 10.0;
param.kd = 0.5;
param.q  = 1.57; // 目标位置 (Rad)
param.dq = 0.0;  // 目标速度 (Rad/s)
param.tau = 0.0; // 前馈力矩 (N-m)

while (running) {
    // 1. 发送控制指令 (非阻塞)
    // 这里的数组顺序对应 init_arm_motors 时的 ID 顺序 (ID 1, ID 2)
    arm.get_arm().mit_control_all({param, param});

    // 2. 接收反馈 (阻塞等待，超时时间 1ms)
    // 建议配合实时线程使用
    arm.recv_all(1000); 

    // 3. 获取当前状态 (均为标准单位)
    auto& motor1 = arm.get_arm().get_motor(0); // Index 0 -> ID 1
    double pos = motor1.get_position(); // Rad
    double vel = motor1.get_velocity(); // Rad/s
    double tor = motor1.get_torque();   // N-m
}
```

### 3. 辅助功能 (归零与启停)

```cpp
// 启用/禁用
arm.motor_off_all(); // 阻尼模式/下电

// 设置零点 (将当前位置设为 0)
// 注意：此操作写入 ROM，需重启电机生效
arm.get_arm().set_zero_one(0); // 对 Index 0 的电机设零
```

---

## 🛠 环境要求

*   **OS**: Linux (Ubuntu 20.04/22.04)
*   **Kernel**: 支持 SocketCAN
*   **Dependencies**: `cmake`, `build-essential`. 无需第三方重型库。

---

## 📥 编译指南

本项目为标准 CMake 项目。

```bash
mkdir build && cd build
cmake ..
make -j
# 生成: build/libx3arm_can.a (静态库)
# 生成: build/x3arm-can-demo (测试程序)
```

**集成到现有 CMake 项目:**

```cmake
add_subdirectory(x3arm_can)
target_link_libraries(your_algorithm_node x3arm_can)
```

---


## � 关键文件导航 (Source Map)

| 文件 (相对路径) | 核心内容 | 用途 |
| :--- | :--- | :--- |
| `x3arm/can/socket/x3arm.hpp` | `class X3Arm` | **程序入口**。负责初始化和总线通信。 |
| `x3arm/maita_motor/maita_motor_control.hpp` | `struct MITParam` | **控制参数**。定义 MIT 模式的 5 个控制量。 |
| `x3arm/maita_motor/maita_motor.hpp` | `class Motor` | **电机状态**。提供 `get_position()` / `get_torque()` 等。 |
| `x3arm/can/socket/arm_component.hpp` | `class ArmComponent` | **控制接口**。提供 `mit_control_all()` 发送函数。 |

---

## 📖 API 速查

### 1. 初始化 (`OpenArm`)

位于 `x3arm.hpp`。

*   **`OpenArm(ifname, enable_fd)`**: 构造函数。Maita 电机传 `false` (CAN 2.0)。
*   **`init_arm_motors(types, ids)`**: 注册电机。`ids` 为 CAN ID 列表 (如 `1, 2`)。
*   **`recv_all(timeout_us)`**: 阻塞接收。建议放在控制循环开头或独立线程。
*   **`get_arm()`**: 获取 `ArmComponent` 指针，用于后续控制。

### 2. 发送指令 (`ArmComponent`)

通过 `arm.get_arm()` 调用。

*   **`mit_control_all(params)`**: **(核心)** 发送 MIT 运控帧。`params` 数组长度需对应初始化 ID 数。
*   **`position_control_one(index, val)`**: 位置模式 (注意单位: 0.01度)。
*   **`speed_control_one(index, val)`**: 速度模式 (注意单位: 0.01 dps)。
*   **`set_zero_one(index)`**: 将当前位置设为零点 (写入 ROM)。

### 3. 读取状态 (`Motor`)

通过 `arm.get_arm().get_motor(index)` 获取。

*   **`get_position()`**: 返回 Rad (double)。
*   **`get_velocity()`**: 返回 Rad/s (double)。
*   **`get_torque()`**: 返回 N-m (double)。


---

## ⚙️ CAN 配置

为了保证实时性，请确保 CAN 接口配置正确。

```bash
# 推荐配置 (1Mbps)
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
# 检查是否丢包
ip -d -s link show can0
```
