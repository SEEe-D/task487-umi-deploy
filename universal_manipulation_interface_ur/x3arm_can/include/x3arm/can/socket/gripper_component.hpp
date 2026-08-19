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

#define _USE_MATH_DEFINES
#include <cmath>
#include <memory>

#include "../../canbus/can_device_collection.hpp"
#include "../../canbus/can_socket.hpp"
#include "../../livelybot_motor/livelybot_motor.hpp"
#include "../../livelybot_motor/livelybot_motor_control.hpp"
#include "../../livelybot_motor/livelybot_motor_device.hpp"

namespace x3arm::can::socket {

/**
 * @brief Livelybot夹爪组件类
 *
 * 使用Livelybot电机控制夹爪的开合。
 */
class GripperComponent {
public:
  explicit GripperComponent(canbus::CANSocket &can_socket);
  ~GripperComponent() = default;

  /**
   * @brief 初始化电机设备
   * @param device_id 电机CAN ID
   */
  void init_motor_device(uint32_t device_id);

  // ========================================================================
  // 夹爪控制接口
  // ========================================================================

  /**
   * @brief 停止电机
   */
  void stop();

  /**
   * @brief 设置当前位置为零位
   */
  void set_zero();

  /**
   * @brief 保存设置到Flash
   */
  void save_flash();

  /**
   * @brief 查询电机状态
   */
  void query_state();

  /**
   * @brief 查询电机故障
   */
  void query_fault();

  /**
   * @brief MIT控制
   * @param param MIT参数
   */
  void mit_control(const livelybot_motor::MITParam &param);

  /**
   * @brief 接收CAN数据
   * @param timeout_us 超时时间(微秒)
   */
  void recv(int timeout_us = 1000);

  // ========================================================================
  // 状态获取
  // ========================================================================

  livelybot_motor::Motor *get_motor() const { return motor_.get(); }
  canbus::CANDeviceCollection &get_device_collection() {
    return *device_collection_;
  }

private:
  /**
   * @brief 发送CAN数据包
   * @param packet CAN数据包
   */
  void send_packet(const livelybot_motor::CANPacket &packet);

  canbus::CANSocket &can_socket_;
  std::unique_ptr<canbus::CANDeviceCollection> device_collection_;
  std::unique_ptr<livelybot_motor::Motor> motor_;
  std::shared_ptr<livelybot_motor::LivelybotCANDevice> motor_device_;
};

} // namespace x3arm::can::socket
