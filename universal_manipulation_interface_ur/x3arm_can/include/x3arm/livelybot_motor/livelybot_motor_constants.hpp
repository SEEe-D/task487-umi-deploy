// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <iostream>
#include <string>

namespace x3arm::livelybot_motor {

// ============================================================================
// Livelybot Motor CAN ID Offsets
// ============================================================================
// MIT控制发送: 0x10000 | device_id (扩展帧)
// 查询/配置发送: 0x8000 | device_id (扩展帧)
// 普通控制发送: device_id (标准帧)
// 电机响应: device_id << 8 (扩展帧)
constexpr uint32_t MIT_SEND_OFFSET = 0x10000;
constexpr uint32_t QUERY_SEND_OFFSET = 0x8000;
constexpr uint32_t RECV_ID_SHIFT = 8; // 响应ID = device_id << 8

// ============================================================================
// MIT模式参数范围
// ============================================================================
// 位置: ±3.2768 圈 (16-bit)
constexpr double MIT_POS_MIN = -3.2768;
constexpr double MIT_POS_MAX = 3.2767;
constexpr int MIT_POS_BITS = 16;

// 速度: ±2.0 转/秒 (12-bit)
constexpr double MIT_VEL_MIN = -2.0;
constexpr double MIT_VEL_MAX = 2.0;
constexpr int MIT_VEL_BITS = 12;

// 力矩: ±10.0 Nm (12-bit)
constexpr double MIT_TQE_MIN = -10.0;
constexpr double MIT_TQE_MAX = 10.0;
constexpr int MIT_TQE_BITS = 12;

// Kp: ±400 (12-bit)
constexpr double MIT_KP_MIN = -400.0;
constexpr double MIT_KP_MAX = 400.0;
constexpr int MIT_KP_BITS = 12;

// Kd: ±100 (12-bit)
constexpr double MIT_KD_MIN = -100.0;
constexpr double MIT_KD_MAX = 100.0;
constexpr int MIT_KD_BITS = 12;

// ============================================================================
// 反馈数据参数范围 (从厂商代码中提取)
// ============================================================================
// 位置反馈: 单位 0.0001 圈 (int16)
constexpr double FEEDBACK_POS_SCALE = 0.0001; // 圈

// 速度反馈: 单位 0.00025 转/秒 (int16)
constexpr double FEEDBACK_VEL_SCALE = 0.00025; // 转/秒

// 力矩反馈: 单位 0.01 Nm (int16)
constexpr double FEEDBACK_TQE_SCALE = 0.01; // Nm

// ============================================================================
// 单位转换
// ============================================================================
constexpr double TURNS_TO_RAD = 2.0 * 3.14159265358979323846; // 圈转弧度
constexpr double RAD_TO_TURNS = 1.0 / TURNS_TO_RAD;           // 弧度转圈

// ============================================================================
// 故障代码
// ============================================================================
enum class FaultCode : uint8_t {
  NORMAL = 0,               // 正常
  DMA_STREAM_ERROR = 1,     // DMA 数据流传输错误
  DMA_FIFO_ERROR = 2,       // DMA 数据流 FIFO 错误
  UART_OVERFLOW = 3,        // UART 溢出错误
  UART_FRAME_ERROR = 4,     // UART 帧错误
  UART_NOISE_ERROR = 5,     // UART 噪声错误
  UART_BUFFER_OVERFLOW = 6, // UART 缓冲区溢出错误
  UART_PARITY_ERROR = 7,    // UART 奇偶校验错误
  // 8-31 保留
  CALIBRATION_FAULT = 32,     // 校准故障
  MOTOR_DRIVER_FAULT = 33,    // 电机驱动故障 (欠压)
  OVER_VOLTAGE = 34,          // 过压
  ENCODER_FAULT = 35,         // 编码器故障
  MOTOR_NOT_CALIBRATED = 36,  // 电机未校准
  PWM_CYCLE_LIMIT = 37,       // PWM周期过限
  OVER_TEMPERATURE = 38,      // 温度过高
  START_POS_LIMIT = 39,       // 起始位置超出限制
  UNDER_VOLTAGE = 40,         // 电压过低
  CONFIG_CHANGED = 41,        // 配置已更改
  INVALID_ANGLE = 42,         // 角度无效
  INVALID_POSITION = 43,      // 位置无效
  DRIVER_ENABLE_FAULT = 44,   // 驱动器使能故障
  STOP_POS_USAGE_ERROR = 45,  // 停止位置使用错误
  TIMING_ERROR = 46,          // 时序错误
  BEMF_FEEDFORWARD_ERROR = 47 // 反电动势前馈错误
};

// ============================================================================
// 命令标识符
// ============================================================================
// 查询状态命令
constexpr uint8_t CMD_QUERY_STATE_0 = 0x17;
constexpr uint8_t CMD_QUERY_STATE_1 = 0x01;

// 查询故障命令
constexpr uint8_t CMD_QUERY_FAULT_0 = 0x11;
constexpr uint8_t CMD_QUERY_FAULT_1 = 0x0F;

// 设置零位命令
constexpr uint8_t CMD_SET_ZERO[] = {0x40, 0x01, 0x04, 0x64, 0x20, 0x63, 0x0A};

// 保存到Flash命令
constexpr uint8_t CMD_SAVE_FLASH[] = {0x05, 0xB3, 0x02, 0x00, 0x00};

// 停止命令
constexpr uint8_t CMD_STOP[] = {0x01, 0x00, 0x00};

// 刹车命令
constexpr uint8_t CMD_BRAKE[] = {0x01, 0x00, 0x0F};

// 响应标识符
constexpr uint8_t RESP_STATE_0 = 0x27;
constexpr uint8_t RESP_STATE_1 = 0x01;
constexpr uint8_t RESP_FAULT_0 = 0x21; // 故障响应头 (第1字节)
constexpr uint8_t RESP_FAULT_1 =
    0x0F; // 故障响应头 (第2字节，假设与命令一致或是0F)
// 注: 用户描述 "can3 00000800 [3] 21 0F 21"，第2字节确实是0x0F
constexpr uint8_t RESP_VERSION_0 = 0x25;

inline std::string fault_to_string(FaultCode code) {
  switch (code) {
  case FaultCode::NORMAL:
    return "正常";
  case FaultCode::DMA_STREAM_ERROR:
    return "DMA 数据流传输错误";
  case FaultCode::DMA_FIFO_ERROR:
    return "DMA 数据流 FIFO 错误";
  case FaultCode::UART_OVERFLOW:
    return "UART 溢出错误";
  case FaultCode::UART_FRAME_ERROR:
    return "UART 帧错误";
  case FaultCode::UART_NOISE_ERROR:
    return "UART 噪声错误";
  case FaultCode::UART_BUFFER_OVERFLOW:
    return "UART 缓冲区溢出错误";
  case FaultCode::UART_PARITY_ERROR:
    return "UART 奇偶校验错误";
  case FaultCode::CALIBRATION_FAULT:
    return "校准故障";
  case FaultCode::MOTOR_DRIVER_FAULT:
    return "电机驱动故障 (欠压)";
  case FaultCode::OVER_VOLTAGE:
    return "过压";
  case FaultCode::ENCODER_FAULT:
    return "编码器故障";
  case FaultCode::MOTOR_NOT_CALIBRATED:
    return "电机未校准";
  case FaultCode::PWM_CYCLE_LIMIT:
    return "PWM周期过限";
  case FaultCode::OVER_TEMPERATURE:
    return "温度过高";
  case FaultCode::START_POS_LIMIT:
    return "起始位置超出限制";
  case FaultCode::UNDER_VOLTAGE:
    return "电压过低";
  case FaultCode::CONFIG_CHANGED:
    return "配置已更改";
  case FaultCode::INVALID_ANGLE:
    return "角度无效";
  case FaultCode::INVALID_POSITION:
    return "位置无效";
  case FaultCode::DRIVER_ENABLE_FAULT:
    return "驱动器使能故障";
  case FaultCode::STOP_POS_USAGE_ERROR:
    return "停止位置使用错误";
  case FaultCode::TIMING_ERROR:
    return "时序错误";
  case FaultCode::BEMF_FEEDFORWARD_ERROR:
    return "反电动势前馈错误";
  default:
    return "未知故障 (" + std::to_string(static_cast<int>(code)) + ")";
  }
}

} // namespace x3arm::livelybot_motor
