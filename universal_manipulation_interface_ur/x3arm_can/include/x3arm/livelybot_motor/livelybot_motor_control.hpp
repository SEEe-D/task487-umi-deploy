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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "livelybot_motor.hpp"
#include "livelybot_motor_constants.hpp"

namespace x3arm::livelybot_motor {

// ============================================================================
// MIT控制参数结构体
// ============================================================================
struct MITParam {
  double kp;  // 位置刚度
  double kd;  // 速度阻尼
  double q;   // 目标位置 (弧度)
  double dq;  // 目标速度 (弧度/秒)
  double tau; // 前馈力矩 (Nm)
};

// ============================================================================
// 状态结果结构体
// ============================================================================
struct StateResult {
  double position; // 位置 (弧度)
  double velocity; // 速度 (弧度/秒)
  double torque;   // 力矩 (Nm)
  bool valid;      // 是否有效
  int fault_code;  // 故障代码 (-1表示无效/无故障信息)
};

// ============================================================================
// CAN数据包结构体
// ============================================================================
struct CANPacket {
  uint32_t can_id;
  std::vector<uint8_t> data;
  bool is_extended; // 是否为扩展帧
};

// ============================================================================
// CAN数据包编码器
// ============================================================================
class LivelybotPacketEncoder {
public:
  /**
   * @brief 创建MIT控制命令
   * @param motor 电机对象
   * @param param MIT控制参数
   * @return CAN数据包
   */
  static CANPacket create_mit_control_command(const Motor &motor,
                                              const MITParam &param);

  /**
   * @brief 创建停止命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_stop_command(const Motor &motor);

  /**
   * @brief 创建刹车命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_brake_command(const Motor &motor);

  /**
   * @brief 创建查询状态命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_query_state_command(const Motor &motor);

  /**
   * @brief 创建设置零位命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_set_zero_command(const Motor &motor);

  /**
   * @brief 创建保存到Flash命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_save_flash_command(const Motor &motor);

  /**
   * @brief 创建查询故障命令
   * @param motor 电机对象
   * @return CAN数据包
   */
  static CANPacket create_query_fault_command(const Motor &motor);

private:
  /**
   * @brief 打包MIT控制数据
   * @param param MIT参数
   * @return 8字节数据
   */
  static std::vector<uint8_t> pack_mit_control_data(const MITParam &param);

  /**
   * @brief 将浮点数映射到无符号整数
   * @param x 输入值
   * @param x_min 最小值
   * @param x_max 最大值
   * @param bits 位数
   * @return 映射后的无符号整数
   */
  static uint16_t float_to_uint(double x, double x_min, double x_max, int bits);

  /**
   * @brief 限制值在范围内
   */
  static double limit_min_max(double x, double min_val, double max_val);
};

// ============================================================================
// CAN数据包解码器
// ============================================================================
// ============================================================================
class LivelybotPacketDecoder {
public:
  /**
   * @brief 解析状态响应数据
   * @param data CAN数据
   * @return 状态结果
   */
  static StateResult parse_state_response(const std::vector<uint8_t> &data);

  /**
   * @brief 解析故障响应数据
   * @param data CAN数据
   * @return 故障代码 (如果无效返回 -1)
   */
  static int parse_fault_response(const std::vector<uint8_t> &data);

private:
  /**
   * @brief 从数据中读取int16
   * @param data 数据
   * @param offset 偏移量
   * @return int16值
   */
  static int16_t read_int16(const std::vector<uint8_t> &data, std::size_t offset);
};

} // namespace x3arm::livelybot_motor
