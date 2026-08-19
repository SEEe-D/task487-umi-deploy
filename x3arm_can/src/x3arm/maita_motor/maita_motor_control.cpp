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

#include <cmath>
#include <cstring>
#include <x3arm/maita_motor/maita_motor.hpp>
#include <x3arm/maita_motor/maita_motor_constants.hpp>
#include <x3arm/maita_motor/maita_motor_control.hpp>

namespace x3arm::maita_motor {

// ============================================================================
// Encoder Functions - Create CAN packets for motor commands
// ============================================================================

CANPacket
CanPacketEncoder::create_query_command(const Motor &motor, MaitaCmd cmd,
                                       const std::vector<uint8_t> &params) {
  std::vector<uint8_t> data(8, 0x00);
  data[0] = static_cast<uint8_t>(cmd);
  for (size_t i = 0; i < params.size() && i < 7; ++i) {
    data[i + 1] = params[i];
  }
  return {motor.get_query_send_can_id(), data};
}

CANPacket
CanPacketEncoder::create_mit_control_command(const Motor &motor,
                                             const MITParam &mit_param) {
  std::vector<uint8_t> data =
      pack_mit_control_data(motor.get_motor_type(), mit_param);
  return {motor.get_mit_send_can_id(), data};
}

CANPacket CanPacketEncoder::create_motor_off_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kMotorOff);
}

CANPacket CanPacketEncoder::create_motor_stop_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kMotorStop);
}

CANPacket CanPacketEncoder::create_system_reset_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kSystemReset);
}

CANPacket CanPacketEncoder::create_set_zero_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kWriteCurPosAsZero);
}

CANPacket CanPacketEncoder::create_read_state_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kReadState2);
}

CANPacket CanPacketEncoder::create_read_state1_command(const Motor &motor) {
  return create_query_command(motor, MaitaCmd::kReadState1);
}

CANPacket
CanPacketEncoder::create_read_pid_command(const Motor &motor,
                                          ReadParamIndex param_index) {
  return create_query_command(motor, MaitaCmd::kReadPid,
                              {static_cast<uint8_t>(param_index)});
}

CANPacket CanPacketEncoder::create_speed_control_command(
    const Motor &motor, int32_t speed_0_01dps, uint8_t max_torque_percent) {
  std::vector<uint8_t> data(8, 0x00);
  data[0] = static_cast<uint8_t>(MaitaCmd::kSpeedClosedLoop);
  data[1] = max_torque_percent;
  data[2] = 0x00;
  data[3] = 0x00;
  data[4] = static_cast<uint8_t>(speed_0_01dps & 0xFF);
  data[5] = static_cast<uint8_t>((speed_0_01dps >> 8) & 0xFF);
  data[6] = static_cast<uint8_t>((speed_0_01dps >> 16) & 0xFF);
  data[7] = static_cast<uint8_t>((speed_0_01dps >> 24) & 0xFF);
  return {motor.get_query_send_can_id(), data};
}

CANPacket CanPacketEncoder::create_position_control_command(
    const Motor &motor, int32_t position_0_01deg, uint16_t max_speed_dps) {
  std::vector<uint8_t> data(8, 0x00);
  data[0] = static_cast<uint8_t>(MaitaCmd::kPosClosedLoopAbs);
  data[1] = 0x00;
  data[2] = static_cast<uint8_t>(max_speed_dps & 0xFF);
  data[3] = static_cast<uint8_t>((max_speed_dps >> 8) & 0xFF);
  data[4] = static_cast<uint8_t>(position_0_01deg & 0xFF);
  data[5] = static_cast<uint8_t>((position_0_01deg >> 8) & 0xFF);
  data[6] = static_cast<uint8_t>((position_0_01deg >> 16) & 0xFF);
  data[7] = static_cast<uint8_t>((position_0_01deg >> 24) & 0xFF);
  return {motor.get_query_send_can_id(), data};
}

// ============================================================================
// Decoder Functions
// ============================================================================

// Parse MIT response (from 0x500+id)
// Protocol V4.3 Section 5.3: Reply data format (big-endian)
// DATA[0] = CANID, DATA[1-2] = position 16-bit
// DATA[3] + DATA[4] high 4 bits = velocity 12-bit
// DATA[4] low 4 bits + DATA[5] = torque 12-bit
MitStateResult
CanPacketDecoder::parse_mit_response_data(const Motor &motor,
                                          const std::vector<uint8_t> &data) {
  if (data.size() < 6) {
    return {0, 0, 0, 0, false};
  }

  uint8_t device_id = data[0];
  uint16_t p_uint = (static_cast<uint16_t>(data[1]) << 8) | data[2];
  uint16_t v_uint =
      (static_cast<uint16_t>(data[3]) << 4) | ((data[4] >> 4) & 0x0F);
  uint16_t t_uint = (static_cast<uint16_t>(data[4] & 0x0F) << 8) | data[5];

  LimitParam limits =
      MOTOR_LIMIT_PARAMS[static_cast<int>(motor.get_motor_type())];
  double position = uint_to_double(p_uint, -limits.pMax, limits.pMax, 16);
  double velocity = uint_to_double(v_uint, -limits.vMax, limits.vMax, 12);
  double torque = uint_to_double(t_uint, -limits.tMax, limits.tMax, 12);

  return {device_id, position, velocity, torque, true};
}

// Parse 0x9C (kReadState2) response
// Protocol V4.3 Section 2.15.3: temperature, iq, speed, angle (NO voltage)
State2Result
CanPacketDecoder::parse_state2_data(const std::vector<uint8_t> &data) {
  if (data.size() < 8 ||
      data[0] != static_cast<uint8_t>(MaitaCmd::kReadState2)) {
    return {0, 0, 0, 0, false};
  }

  int8_t temperature = static_cast<int8_t>(data[1]);
  int16_t iq = bytes_to_int16(data[2], data[3]);
  double torque_current = static_cast<double>(iq) * 0.01;
  int16_t speed_dps = bytes_to_int16(data[4], data[5]);
  double velocity = static_cast<double>(speed_dps) * M_PI / 180.0;
  int16_t degree = bytes_to_int16(data[6], data[7]);
  double position = static_cast<double>(degree) * M_PI / 180.0;

  return {temperature, torque_current, velocity, position, true};
}

State1Result
CanPacketDecoder::parse_state1_data(const std::vector<uint8_t> &data) {
  if (data.size() < 8 ||
      data[0] != static_cast<uint8_t>(MaitaCmd::kReadState1)) {
    return {0, 0, 0, 0, false};
  }
  // DATA[1]: Motor Temperature
  int8_t temperature = static_cast<int8_t>(data[1]);
  // DATA[2]: MOS Temperature
  int8_t mos_temperature = static_cast<int8_t>(data[2]);
  // DATA[3]: Brake info

  // DATA[4-5]: Voltage (Little-endian)
  int16_t voltage = bytes_to_int16(data[4], data[5]);
  // DATA[6-7]: Error State (Little-endian)
  uint16_t error_code = static_cast<uint16_t>(bytes_to_int16(data[6], data[7]));

  return {static_cast<uint8_t>(error_code & 0xFF),
          error_code,
          temperature,
          mos_temperature,
          voltage,
          true};
}

ParamResult
CanPacketDecoder::parse_pid_param_data(const std::vector<uint8_t> &data) {
  if (data.size() < 8 || data[0] != static_cast<uint8_t>(MaitaCmd::kReadPid)) {
    return {ReadParamIndex::kCurrentKp, 0, false};
  }

  ReadParamIndex param_index = static_cast<ReadParamIndex>(data[1]);
  uint32_t raw = data[4] | (static_cast<uint32_t>(data[5]) << 8) |
                 (static_cast<uint32_t>(data[6]) << 16) |
                 (static_cast<uint32_t>(data[7]) << 24);
  float value;
  std::memcpy(&value, &raw, sizeof(float));

  return {param_index, static_cast<double>(value), true};
}

// ============================================================================
// MIT Control Data Packing
// ============================================================================

std::vector<uint8_t>
CanPacketEncoder::pack_mit_control_data(MotorType motor_type,
                                        const MITParam &mit_param) {
  uint16_t kp_uint = double_to_uint(mit_param.kp, 0, 500, 12);
  uint16_t kd_uint = double_to_uint(mit_param.kd, 0, 5, 12);

  LimitParam limits = MOTOR_LIMIT_PARAMS[static_cast<int>(motor_type)];
  uint16_t p_des_uint =
      double_to_uint(mit_param.q, -limits.pMax, limits.pMax, 16);
  uint16_t v_des_uint =
      double_to_uint(mit_param.dq, -limits.vMax, limits.vMax, 12);
  uint16_t t_ff_uint =
      double_to_uint(mit_param.tau, -limits.tMax, limits.tMax, 12);

  return {
      static_cast<uint8_t>((p_des_uint >> 8) & 0xFF),
      static_cast<uint8_t>(p_des_uint & 0xFF),
      static_cast<uint8_t>(v_des_uint >> 4),
      static_cast<uint8_t>(((v_des_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)),
      static_cast<uint8_t>(kp_uint & 0xFF),
      static_cast<uint8_t>(kd_uint >> 4),
      static_cast<uint8_t>(((kd_uint & 0xF) << 4) | ((t_ff_uint >> 8) & 0xF)),
      static_cast<uint8_t>(t_ff_uint & 0xFF)};
}

// ============================================================================
// Utility Functions
// ============================================================================

double CanPacketEncoder::limit_min_max(double x, double min, double max) {
  return std::max(min, std::min(x, max));
}

uint16_t CanPacketEncoder::double_to_uint(double x, double x_min, double x_max,
                                          int bits) {
  x = limit_min_max(x, x_min, x_max);
  double span = x_max - x_min;
  double data_norm = (x - x_min) / span;
  return static_cast<uint16_t>(data_norm * ((1 << bits) - 1) + 0.5);
}

double CanPacketDecoder::uint_to_double(uint16_t x, double min, double max,
                                        int bits) {
  double span = max - min;
  double data_norm = static_cast<double>(x) / ((1 << bits) - 1);
  return data_norm * span + min;
}

float CanPacketDecoder::uint8s_to_float(const std::array<uint8_t, 4> &bytes) {
  float value;
  std::memcpy(&value, bytes.data(), sizeof(float));
  return value;
}

int16_t CanPacketDecoder::bytes_to_int16(uint8_t low, uint8_t high) {
  return static_cast<int16_t>(low | (high << 8));
}

int32_t CanPacketDecoder::bytes_to_int32(uint8_t b0, uint8_t b1, uint8_t b2,
                                         uint8_t b3) {
  return static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

} // namespace x3arm::maita_motor
