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

#include <x3arm/livelybot_motor/livelybot_motor_control.hpp>

#include <algorithm>
#include <cstring>

namespace x3arm::livelybot_motor {

// ============================================================================
// LivelybotPacketEncoder
// ============================================================================

CANPacket
LivelybotPacketEncoder::create_mit_control_command(const Motor &motor,
                                                   const MITParam &param) {
  CANPacket packet;
  packet.can_id = motor.get_mit_send_can_id();
  packet.data = pack_mit_control_data(param);
  packet.is_extended = true;
  return packet;
}

CANPacket
LivelybotPacketEncoder::create_query_fault_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_query_send_can_id();
  packet.data = {CMD_QUERY_FAULT_0, CMD_QUERY_FAULT_1};
  packet.is_extended = true;
  return packet;
}

CANPacket LivelybotPacketEncoder::create_stop_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_std_send_can_id();
  packet.data = std::vector<uint8_t>(CMD_STOP, CMD_STOP + sizeof(CMD_STOP));
  packet.is_extended = false;
  return packet;
}

CANPacket LivelybotPacketEncoder::create_brake_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_std_send_can_id();
  packet.data = std::vector<uint8_t>(CMD_BRAKE, CMD_BRAKE + sizeof(CMD_BRAKE));
  packet.is_extended = false;
  return packet;
}

CANPacket
LivelybotPacketEncoder::create_query_state_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_query_send_can_id();
  packet.data = {CMD_QUERY_STATE_0, CMD_QUERY_STATE_1};
  packet.is_extended = true;
  return packet;
}

CANPacket LivelybotPacketEncoder::create_set_zero_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_query_send_can_id();
  packet.data =
      std::vector<uint8_t>(CMD_SET_ZERO, CMD_SET_ZERO + sizeof(CMD_SET_ZERO));
  packet.is_extended = true;
  return packet;
}

CANPacket
LivelybotPacketEncoder::create_save_flash_command(const Motor &motor) {
  CANPacket packet;
  packet.can_id = motor.get_query_send_can_id();
  packet.data = std::vector<uint8_t>(CMD_SAVE_FLASH,
                                     CMD_SAVE_FLASH + sizeof(CMD_SAVE_FLASH));
  packet.is_extended = true;
  return packet;
}

std::vector<uint8_t>
LivelybotPacketEncoder::pack_mit_control_data(const MITParam &param) {
  // 将弧度转换为圈
  double pos_turns = param.q * RAD_TO_TURNS;
  double vel_turns = param.dq * RAD_TO_TURNS;

  // 2. Kp/Kd转换: Nm/rad -> Nm/turn
  //    因为 1 Turn = 2PI Rad, 所以 Nm/Turn = Nm/Rad * 2PI
  double kp_turns = param.kp * TURNS_TO_RAD;
  double kd_turns = param.kd * TURNS_TO_RAD;

  // 3. 厂商系数补偿 (参考 convert.c 中的 tqe_adjust 和 pid_adjust)
  //    MGENERAL 的系数 k=0.5. raw_input = val / k = val * 2.0
  double tqe_adjusted = param.tau * 2.0;
  double kp_adjusted = kp_turns * 2.0;
  double kd_adjusted = kd_turns * 2.0;

  // 4. 映射到整数 (使用厂商定义的 Turns/Unit 范围)
  uint16_t pos_raw =
      float_to_uint(pos_turns, MIT_POS_MIN, MIT_POS_MAX, MIT_POS_BITS);
  uint16_t vel_raw =
      float_to_uint(vel_turns, MIT_VEL_MIN, MIT_VEL_MAX, MIT_VEL_BITS);
  uint16_t tqe_raw =
      float_to_uint(tqe_adjusted, MIT_TQE_MIN, MIT_TQE_MAX, MIT_TQE_BITS);
  uint16_t kp_raw =
      float_to_uint(kp_adjusted, MIT_KP_MIN, MIT_KP_MAX, MIT_KP_BITS);
  uint16_t kd_raw =
      float_to_uint(kd_adjusted, MIT_KD_MIN, MIT_KD_MAX, MIT_KD_BITS);

  // 打包数据 (8字节)
  // tdata[0] = pos 低8位
  // tdata[1] = pos 高8位
  // tdata[2] = vel 低8位
  // tdata[3] = vel高4位 | tqe低4位
  // tdata[4] = tqe 高8位
  // tdata[5] = kp 低8位
  // tdata[6] = kp高4位 | kd低4位
  // tdata[7] = kd 高4位
  std::vector<uint8_t> data(8);
  data[0] = pos_raw & 0xFF;
  data[1] = (pos_raw >> 8) & 0xFF;
  data[2] = vel_raw & 0xFF;
  data[3] = ((vel_raw >> 8) & 0x0F) | ((tqe_raw & 0x0F) << 4);
  data[4] = (tqe_raw >> 4) & 0xFF;
  data[5] = kp_raw & 0xFF;
  data[6] = ((kp_raw >> 8) & 0x0F) | ((kd_raw & 0x0F) << 4);
  data[7] = (kd_raw >> 4) & 0xFF;

  return data;
}

uint16_t LivelybotPacketEncoder::float_to_uint(double x, double x_min,
                                               double x_max, int bits) {
  double x_limited = limit_min_max(x, x_min, x_max);
  double span = x_max - x_min;
  uint32_t max_val = (1 << bits) - 1;
  return static_cast<uint16_t>((x_limited - x_min) / span * max_val);
}

double LivelybotPacketEncoder::limit_min_max(double x, double min_val,
                                             double max_val) {
  return std::max(min_val, std::min(max_val, x));
}

// ============================================================================
// LivelybotPacketDecoder
// ============================================================================

StateResult
LivelybotPacketDecoder::parse_state_response(const std::vector<uint8_t> &data) {
  StateResult result;
  result.valid = false;
  result.fault_code = -1; // 默认无故障信息

  // 检查数据长度和响应标识

  // 检查数据长度和响应标识
  if (data.size() < 8) {
    return result;
  }

  if (data[0] != RESP_STATE_0 || data[1] != RESP_STATE_1) {
    return result;
  }

  // 解析位置、速度、力矩 (int16, 小端序)
  int16_t pos_raw = read_int16(data, 2);
  int16_t vel_raw = read_int16(data, 4);
  int16_t tqe_raw = read_int16(data, 6);

  // 转换为物理单位
  // 位置: 单位 0.0001 圈 -> 弧度
  result.position = pos_raw * FEEDBACK_POS_SCALE * TURNS_TO_RAD;
  // 速度: 单位 0.00025 转/秒 -> 弧度/秒
  result.velocity = vel_raw * FEEDBACK_VEL_SCALE * TURNS_TO_RAD;
  // 力矩: 单位 0.01 Nm
  // 厂商 restore 公式: val = raw * scale * k + d
  // MGENERAL: k=0.5, d=0. 所以需要 * 0.5
  result.torque = tqe_raw * FEEDBACK_TQE_SCALE * 0.5;

  result.valid = true;
  return result;
}

int LivelybotPacketDecoder::parse_fault_response(
    const std::vector<uint8_t> &data) {
  // 期望格式: [RESP_FAULT_0, RESP_FAULT_1, FAULT_CODE]
  // 即: 21 0F XX
  if (data.size() < 3) {
    return -1;
  }

  if (data[0] != RESP_FAULT_0 || data[1] != RESP_FAULT_1) {
    return -1;
  }

  return data[2];
}

int16_t LivelybotPacketDecoder::read_int16(const std::vector<uint8_t> &data,
                                           std::size_t offset) {
  if (offset + 1 >= data.size()) {
    return 0;
  }
  return static_cast<int16_t>(data[offset] | (data[offset + 1] << 8));
}

} // namespace x3arm::livelybot_motor
