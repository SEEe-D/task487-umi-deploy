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

#include <functional>
#include <memory>
#include <string>

#include "../../canbus/can_device_collection.hpp"
#include "../../canbus/can_socket.hpp"
#include "arm_component.hpp"
#include "gripper_component.hpp"

namespace x3arm::can::socket {
class X3Arm {
public:
  X3Arm(const std::string &can_interface, bool enable_fd = false);
  ~X3Arm() = default;

  std::string can_interface() const noexcept { return can_interface_; }
  bool can_fd_enabled() const noexcept { return enable_fd_; }

  // Component initialization (using device_ids for multi-CAN ID support)
  void init_arm_motors(const std::vector<maita_motor::MotorType> &motor_types,
                       const std::vector<uint32_t> &device_ids);

  void init_gripper_motor(uint32_t device_id);

  // Component access
  ArmComponent &get_arm() { return *arm_; }
  GripperComponent &get_gripper() { return *gripper_; }
  canbus::CANDeviceCollection &get_master_can_device_collection() {
    return *master_can_device_collection_;
  }

  // Motor control operations
  void motor_off_all();
  void motor_stop_all();

  // Read state (0x9C)
  void read_state_one(int i);
  void read_state_all();

  // Read state1 (0x9A)
  void read_state1_one(int i);
  void read_state1_all();

  // The timeout for reading from socket, set to timeout_us.
  void recv_all(int timeout_us = 500);
  void set_callback_mode_all(maita_motor::CallbackMode callback_mode);

  // Read PID (0x30)
  void read_pid_all(maita_motor::ReadParamIndex param_index);

  void
  set_traffic_callback(std::function<void(const can_frame &, bool)> callback);

private:
  std::string can_interface_;
  bool enable_fd_;
  std::unique_ptr<canbus::CANSocket> can_socket_;
  std::unique_ptr<ArmComponent> arm_;
  std::unique_ptr<GripperComponent> gripper_;
  std::unique_ptr<canbus::CANDeviceCollection> master_can_device_collection_;
  std::vector<maita_motor::MaitaDeviceCollection *>
      sub_maita_device_collections_;
  void register_maita_device_collection(
      maita_motor::MaitaDeviceCollection &device_collection);
};

} // namespace x3arm::can::socket
