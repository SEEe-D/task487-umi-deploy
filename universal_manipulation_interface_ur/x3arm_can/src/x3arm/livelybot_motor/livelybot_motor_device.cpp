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

#include <x3arm/livelybot_motor/livelybot_motor_device.hpp>

#include <cstring>

namespace x3arm::livelybot_motor {

LivelybotCANDevice::LivelybotCANDevice(Motor &motor)
    : CANDevice(motor.get_mit_send_can_id(), // send_can_id (MIT控制)
                motor.get_mit_recv_can_id(), // recv_can_id (响应)
                CAN_EFF_MASK,                // recv_can_mask (扩展帧掩码)
                false),                      // is_fd_enabled
      motor_(motor) {}

void LivelybotCANDevice::callback(const can_frame &frame) {
  process_data(frame.data, frame.can_dlc);
}

void LivelybotCANDevice::callback(const canfd_frame &frame) {
  process_data(frame.data, frame.len);
}

std::vector<canid_t> LivelybotCANDevice::get_all_recv_can_ids() const {
  // 电机响应ID: device_id << 8 (且为扩展帧)
  return {motor_.get_mit_recv_can_id() | CAN_EFF_FLAG};
}

can_frame LivelybotCANDevice::create_can_frame(const CANPacket &packet) {
  can_frame frame;
  std::memset(&frame, 0, sizeof(frame));

  if (packet.is_extended) {
    frame.can_id = packet.can_id | CAN_EFF_FLAG; // 设置扩展帧标志
  } else {
    frame.can_id = packet.can_id;
  }

  frame.can_dlc = static_cast<uint8_t>(
      std::min(packet.data.size(), static_cast<std::size_t>(8)));
  std::memcpy(frame.data, packet.data.data(), frame.can_dlc);

  return frame;
}

void LivelybotCANDevice::process_data(const uint8_t *data, std::size_t len) {
  if (len < 3) {
    return;
  }

  std::vector<uint8_t> data_vec(data, data + len);

  // 尝试解析故障
  int fault = LivelybotPacketDecoder::parse_fault_response(data_vec);
  if (fault != -1) {
    motor_.update_fault_state(static_cast<uint8_t>(fault));
    return;
  }

  if (len < 8) {
    return;
  }

  StateResult result = LivelybotPacketDecoder::parse_state_response(data_vec);

  if (result.valid) {
    motor_.update_state(result.position, result.velocity, result.torque);
  }
}

} // namespace x3arm::livelybot_motor
