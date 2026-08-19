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
#include <cstring>
#include <map>

#include "maita_motor_constants.hpp"

namespace x3arm::maita_motor {
class Motor {
  friend class MaitaCANDevice;
  friend class MaitaControl;

public:
  // Constructor - takes device_id for multi-CAN ID support
  Motor(MotorType motor_type, uint32_t device_id);

  // State getters
  double get_position() const { return state_q_; }
  double get_velocity() const { return state_dq_; }
  double get_torque() const { return state_tau_; }
  int get_temperature() const { return state_temperature_; }
  int get_voltage() const { return state_voltage_; }
  uint16_t get_error_code() const { return error_code_; } // Added

  // Motor property getters
  uint32_t get_device_id() const { return device_id_; }
  MotorType get_motor_type() const { return motor_type_; }

  // CAN ID getters - calculated from device_id
  // MIT control: send 0x400+id, recv 0x500+id
  // Query: send 0x140+id, recv 0x240+id
  uint32_t get_mit_send_can_id() const { return MIT_SEND_OFFSET + device_id_; }
  uint32_t get_mit_recv_can_id() const { return MIT_RECV_OFFSET + device_id_; }
  uint32_t get_query_send_can_id() const {
    return QUERY_SEND_OFFSET + device_id_;
  }
  uint32_t get_query_recv_can_id() const {
    return QUERY_RECV_OFFSET + device_id_;
  }

  // Enable status getters
  bool is_enabled() const { return enabled_; }

  // Parameter methods
  double get_param(ReadParamIndex param_index) const;

  // Static methods for motor properties
  static LimitParam get_limit_param(MotorType motor_type);

protected:
  // State update methods
  // Full update (all fields)
  void update_state(double q, double dq, double tau, int temperature,
                    int voltage);
  // MIT response: only position/velocity/torque (protocol 5.3 - no
  // temp/voltage)
  void update_motion_state(double q, double dq, double tau);
  // State2 response: with temperature but no voltage (protocol 2.15.3 - voltage
  // is in State1)
  void update_state2(double q, double dq, double tau, int temperature);

  void set_enabled(bool enabled);
  void set_temp_param(ReadParamIndex param_index, double val);

  // Motor identifiers
  uint32_t device_id_;
  MotorType motor_type_;

  // Enable status
  bool enabled_;

  // Current state
  double state_q_, state_dq_, state_tau_;
  int state_temperature_, state_voltage_;
  uint16_t error_code_; // Added for fault diagnostics

  // Parameter storage (key: ReadParamIndex)
  std::map<ReadParamIndex, double> temp_param_dict_;

  void update_error_state(uint16_t error_code); // Added
};
} // namespace x3arm::maita_motor
