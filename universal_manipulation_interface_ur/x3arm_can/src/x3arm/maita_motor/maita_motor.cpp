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

#include <stdexcept>
#include <string>
#include <x3arm/maita_motor/maita_motor.hpp>
#include <x3arm/maita_motor/maita_motor_constants.hpp>

namespace x3arm::maita_motor {

// Constructor - takes device_id for multi-CAN ID support
Motor::Motor(MotorType motor_type, uint32_t device_id)
    : device_id_(device_id), motor_type_(motor_type), enabled_(false),
      state_q_(0.0), state_dq_(0.0), state_tau_(0.0), state_temperature_(0),
      state_voltage_(0) {}

// Enable methods
void Motor::set_enabled(bool enable) { this->enabled_ = enable; }

// Parameter methods
double Motor::get_param(ReadParamIndex param_index) const {
  auto it = temp_param_dict_.find(param_index);
  return (it != temp_param_dict_.end()) ? it->second : -1;
}

void Motor::set_temp_param(ReadParamIndex param_index, double val) {
  temp_param_dict_[param_index] = val;
}

// State update methods - full update (for commands that return all data)
void Motor::update_state(double q, double dq, double tau, int temperature,
                         int voltage) {
  state_q_ = q;
  state_dq_ = dq;
  state_tau_ = tau;
  state_temperature_ = temperature;
  state_voltage_ = voltage;
}

// MIT response: only update position/velocity/torque (no temperature/voltage in
// protocol 5.3)
void Motor::update_motion_state(double q, double dq, double tau) {
  state_q_ = q;
  state_dq_ = dq;
  state_tau_ = tau;
}

// State2 response: update state without voltage (voltage is in State1 0x9A, not
// in 0x9C)
void Motor::update_state2(double q, double dq, double tau, int temperature) {
  state_q_ = q;
  state_dq_ = dq;
  state_tau_ = tau;
  state_temperature_ = temperature;
}

void Motor::update_error_state(uint16_t error_code) {
  error_code_ = error_code;
}

// Static methods
LimitParam Motor::get_limit_param(MotorType motor_type) {
  size_t index = static_cast<size_t>(motor_type);
  if (index >= MOTOR_LIMIT_PARAMS.size()) {
    throw std::invalid_argument("Invalid motor type: " +
                                std::to_string(static_cast<int>(motor_type)));
  }
  return MOTOR_LIMIT_PARAMS[index];
}

} // namespace x3arm::maita_motor
