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
#include <linux/can.h>

#include <vector>

#include "../canbus/can_device.hpp"
#include "../canbus/can_socket.hpp"
#include "livelybot_motor.hpp"
#include "livelybot_motor_control.hpp"

namespace x3arm::livelybot_motor {

/**
 * @brief Livelybot电机CAN设备类
 *
 * 处理CAN帧的接收和发送，更新电机状态。
 */
class LivelybotCANDevice : public canbus::CANDevice {
public:
  /**
   * @brief 构造函数
   * @param motor 电机对象引用
   */
  explicit LivelybotCANDevice(Motor &motor);

  /**
   * @brief 处理接收到的CAN帧
   * @param frame CAN帧
   */
  void callback(const can_frame &frame) override;

  /**
   * @brief 处理接收到的CAN-FD帧
   * @param frame CAN-FD帧
   */
  void callback(const canfd_frame &frame) override;

  /**
   * @brief 获取所有接收CAN ID
   * @return CAN ID列表
   */
  std::vector<canid_t> get_all_recv_can_ids() const override;

  /**
   * @brief 获取电机引用
   * @return 电机引用
   */
  Motor &get_motor() { return motor_; }

  /**
   * @brief 创建CAN帧
   * @param packet CAN数据包
   * @return CAN帧
   */
  can_frame create_can_frame(const CANPacket &packet);

private:
  /**
   * @brief 处理接收到的数据
   * @param data 数据
   * @param len 长度
   */
  void process_data(const uint8_t *data,std::size_t len);

  Motor &motor_;
};

} // namespace x3arm::livelybot_motor
