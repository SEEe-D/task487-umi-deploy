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

#include <linux/can.h>

#include <cstdint>
#include <cstring> // for memcpy
#include <iostream>
#include <map>
#include <vector>

#include "maita_motor.hpp"
#include "maita_motor_constants.hpp"

namespace x3arm::maita_motor {
// Forward declarations
class Motor;

struct ParamResult {
  ReadParamIndex param_index;
  double value;
  bool valid;
};

// MIT response state result (0x500+id)
// Protocol: device_id, position, velocity, torque
struct MitStateResult {
  uint8_t device_id;
  double position; // rad (from 16-bit, range ±12.5rad)
  double velocity; // rad/s (from 12-bit, range ±45rad/s)
  double torque;   // N-m (from 12-bit, range ±24Nm)
  bool valid;
};

// Query response state result (0x9C - ReadState2)
// Protocol: temperature, iq(current), speed, angle
struct State2Result {
  int8_t temperature; // temperature (1C/LSB)
  double iq;          // torque current (A, from 0.01A/LSB)
  double velocity;    // speed (rad/s, from 1dps/LSB)
  double position;    // angle (rad, from 1degree/LSB)
  bool valid;
};

// Query response state result (0x9A - ReadState1)
struct State1Result {
  uint8_t error_state; // Error code (low byte of combined error)
  uint16_t error_code; // Full error code
  int8_t temperature;
  int8_t mos_temperature;
  int16_t voltage;
  bool valid;
};

struct CANPacket {
  uint32_t send_can_id;
  std::vector<uint8_t> data;
};

struct MITParam {
  double kp;
  double kd;
  double q;
  double dq;
  double tau;
};

class CanPacketEncoder {
public:
  // MIT commands (send to 0x400+id)
  static CANPacket create_mit_control_command(const Motor &motor,
                                              const MITParam &mit_param);

  // Query commands (send to 0x140+id)
  static CANPacket create_motor_off_command(const Motor &motor);
  static CANPacket create_motor_stop_command(const Motor &motor);
  static CANPacket create_system_reset_command(const Motor &motor); // 0x76
  static CANPacket create_set_zero_command(const Motor &motor);     // 0x64
  static CANPacket create_read_state_command(const Motor &motor);
  static CANPacket create_read_state1_command(const Motor &motor); // Added 0x9A
  static CANPacket create_read_pid_command(const Motor &motor,
                                           ReadParamIndex param_index);
  static CANPacket create_speed_control_command(const Motor &motor,
                                                int32_t speed_0_01dps,
                                                uint8_t max_torque_percent = 0);
  static CANPacket create_position_control_command(const Motor &motor,
                                                   int32_t position_0_01deg,
                                                   uint16_t max_speed_dps = 0);

  // Generic command builder
  static CANPacket
  create_query_command(const Motor &motor, MaitaCmd cmd,
                       const std::vector<uint8_t> &params = {});

private:
  static std::vector<uint8_t> pack_mit_control_data(MotorType motor_type,
                                                    const MITParam &mit_param);
  static double limit_min_max(double x, double min, double max);
  static uint16_t double_to_uint(double x, double x_min, double x_max,
                                 int bits);
};

class CanPacketDecoder {
public:
  // Parse MIT response (from 0x500+id)
  // DATA[0] = device_id
  // DATA[1-2] = position (16-bit), range: ±12.5rad
  // DATA[3](4-7bit) + DATA[4](4-7bit) = velocity (12-bit), range: ±45rad/s
  // DATA[4](0-3bit) + DATA[5] = torque (12-bit), range: ±24N-m
  static MitStateResult
  parse_mit_response_data(const Motor &motor, const std::vector<uint8_t> &data);

  // Parse Query response (0x9C - kReadState2)
  static State2Result parse_state2_data(const std::vector<uint8_t> &data);

  // Parse Query response (0x9A - kReadState1)
  static State1Result parse_state1_data(const std::vector<uint8_t> &data);

  // Parse PID param response (0x30)
  static ParamResult parse_pid_param_data(const std::vector<uint8_t> &data);

private:
  static double uint_to_double(uint16_t x, double min, double max, int bits);
  static float uint8s_to_float(const std::array<uint8_t, 4> &bytes);
  static int16_t bytes_to_int16(uint8_t low, uint8_t high);
  static int32_t bytes_to_int32(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3);
};

} // namespace x3arm::maita_motor