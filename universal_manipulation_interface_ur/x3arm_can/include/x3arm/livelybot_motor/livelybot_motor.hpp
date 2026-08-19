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

#include "livelybot_motor_constants.hpp"

namespace x3arm::livelybot_motor {

/**
 * @brief Livelybot电机状态类
 *
 * 存储电机的当前状态（位置、速度、力矩）和配置信息。
 */
class Motor {
  friend class LivelybotCANDevice;

public:
  /**
   * @brief 构造函数
   * @param device_id 电机CAN ID
   */
  explicit Motor(uint32_t device_id);

  // ========================================================================
  // 状态获取
  // ========================================================================

  /// 获取位置 (弧度)
  double get_position() const { return state_q_; }

  /// 获取速度 (弧度/秒)
  double get_velocity() const { return state_dq_; }

  /// 获取力矩 (Nm)
  double get_torque() const { return state_tau_; }

  /// 获取故障代码
  FaultCode get_fault_code() const { return fault_code_; }

  // ========================================================================
  // 电机属性
  // ========================================================================

  /// 获取设备ID
  uint32_t get_device_id() const { return device_id_; }

  // ========================================================================
  // CAN ID计算
  // ========================================================================

  /// MIT控制发送CAN ID (0x10000 | device_id)
  uint32_t get_mit_send_can_id() const { return MIT_SEND_OFFSET | device_id_; }

  /// MIT控制接收CAN ID (device_id << 8)
  uint32_t get_mit_recv_can_id() const { return device_id_ << RECV_ID_SHIFT; }

  /// 查询/配置发送CAN ID (0x8000 | device_id)
  uint32_t get_query_send_can_id() const {
    return QUERY_SEND_OFFSET | device_id_;
  }

  /// 查询响应接收CAN ID (device_id << 8)
  uint32_t get_query_recv_can_id() const { return device_id_ << RECV_ID_SHIFT; }

  /// 普通控制发送CAN ID (device_id, 标准帧)
  uint32_t get_std_send_can_id() const { return device_id_; }

protected:
  // ========================================================================
  // 状态更新方法 (由CANDevice调用)
  // ========================================================================

  /**
   * @brief 更新电机状态
   * @param q 位置 (弧度)
   * @param dq 速度 (弧度/秒)
   * @param tau 力矩 (Nm)
   */
  void update_state(double q, double dq, double tau);

  /**
   * @brief 更新故障状态
   * @param code 故障代码
   */
  void update_fault_state(uint8_t code);

private:
  uint32_t device_id_;

  // 当前状态
  double state_q_;                          // 位置 (rad)
  double state_dq_;                         // 速度 (rad/s)
  double state_tau_;                        // 力矩 (Nm)
  FaultCode fault_code_{FaultCode::NORMAL}; // 故障代码
};

} // namespace x3arm::livelybot_motor
