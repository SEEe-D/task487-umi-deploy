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

#include <memory>
#include <vector>

#include "../canbus/can_device_collection.hpp"
#include "maita_motor_constants.hpp"
#include "maita_motor_control.hpp"
#include "maita_motor_device.hpp"

namespace x3arm::maita_motor {

class MaitaDeviceCollection {
public:
  MaitaDeviceCollection(canbus::CANSocket &can_socket);
  virtual ~MaitaDeviceCollection() = default;

  // Motor enable/disable (Query commands: 0x80, 0x81)
  void motor_off_all();
  void motor_stop_all();
  void set_zero_one(int i);
  void system_reset_one(int i); // 0x76 软复位
  void set_zero_all();
  void set_callback_mode_all(CallbackMode callback_mode);

  // Read state (Query command: 0x9C)
  void read_state_one(int i);
  void read_state_all();

  // Read state1 (Query command: 0x9A)
  void read_state1_one(int i);
  void read_state1_all();

  // Read PID parameters (Query command: 0x30)
  void read_pid_one(int i, ReadParamIndex param_index);
  void read_pid_all(ReadParamIndex param_index);

  // MIT control operations (MIT commands: 0x400+id)
  void mit_control_one(int i, const MITParam &mit_param);
  void mit_control_all(const std::vector<MITParam> &mit_params);

  // Speed control (Query command: 0xA2)
  void speed_control_one(int i, int32_t speed_rpm_x10);
  void speed_control_all(const std::vector<int32_t> &speeds);

  // Position control (Query command: 0xA4)
  void position_control_one(int i, int32_t position_0_01deg);
  void position_control_all(const std::vector<int32_t> &positions);

  // Device collection access
  std::vector<Motor> get_motors() const;
  Motor get_motor(int i) const;
  canbus::CANDeviceCollection &get_device_collection() {
    return *device_collection_;
  }

protected:
  canbus::CANSocket &can_socket_;
  std::unique_ptr<CanPacketEncoder> can_packet_encoder_;
  std::unique_ptr<CanPacketDecoder> can_packet_decoder_;
  std::unique_ptr<canbus::CANDeviceCollection> device_collection_;

  // Helper methods for subclasses
  void send_command_to_device(std::shared_ptr<MaitaCANDevice> maita_device,
                              const CANPacket &packet);
  std::vector<std::shared_ptr<MaitaCANDevice>> get_maita_devices() const;
};
} // namespace x3arm::maita_motor
