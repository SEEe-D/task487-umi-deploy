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
#include <iostream>
#include <x3arm/maita_motor/maita_motor.hpp>
#include <x3arm/maita_motor/maita_motor_constants.hpp>
#include <x3arm/maita_motor/maita_motor_control.hpp>
#include <x3arm/maita_motor/maita_motor_device.hpp>

namespace x3arm::maita_motor {

MaitaCANDevice::MaitaCANDevice(Motor &motor, canid_t recv_can_mask, bool use_fd)
    : canbus::CANDevice(motor.get_mit_send_can_id(),
                        motor.get_mit_recv_can_id(), recv_can_mask, use_fd),
      motor_(motor), callback_mode_(CallbackMode::STATE), use_fd_(use_fd) {}

// Return all receive CAN IDs for this device (MIT and Query)
std::vector<canid_t> MaitaCANDevice::get_all_recv_can_ids() const {
  return {motor_.get_mit_recv_can_id(), motor_.get_query_recv_can_id()};
}

std::vector<uint8_t>
MaitaCANDevice::get_data_from_frame(const can_frame &frame) {
  return std::vector<uint8_t>(frame.data, frame.data + frame.can_dlc);
}

std::vector<uint8_t>
MaitaCANDevice::get_data_from_frame(const canfd_frame &frame) {
  return std::vector<uint8_t>(frame.data, frame.data + frame.len);
}

void MaitaCANDevice::callback(const can_frame &frame) {
  if (use_fd_) {
    std::cerr << "WARNING: WRONG CALLBACK FUNCTION" << std::endl;
    return;
  }

  std::vector<uint8_t> data = get_data_from_frame(frame);
  if (data.empty())
    return;

  // Determine which type of response based on CAN ID
  bool is_mit_response = (frame.can_id == motor_.get_mit_recv_can_id());
  bool is_query_response = (frame.can_id == motor_.get_query_recv_can_id());

  switch (callback_mode_) {
  case STATE:
    if (is_mit_response && frame.can_dlc >= 6) {
      // Parse MIT response (from 0x500+id)
      // MIT response returns torque directly in N-m
      MitStateResult result =
          CanPacketDecoder::parse_mit_response_data(motor_, data);
      if (result.valid) {
        motor_.update_motion_state(result.position, result.velocity,
                                   result.torque);
      }
    } else if (is_query_response &&
               data[0] == static_cast<uint8_t>(MaitaCmd::kReadState2)) {
      // Parse state2 response (0x9C)
      // State2 returns iq (current in A), convert to torque using Kt
      State2Result result = CanPacketDecoder::parse_state2_data(data);
      if (result.valid) {
        // Get Kt for this motor type and convert iq (A) to torque (N-m)
        double kt = MOTOR_KT[static_cast<size_t>(motor_.get_motor_type())];
        double torque = result.iq * kt; // torque = iq * Kt
        motor_.update_state2(result.position, result.velocity, torque,
                             result.temperature);
      }
    } else if (is_query_response &&
               data[0] == static_cast<uint8_t>(MaitaCmd::kReadState1)) {
      // Parse state1 response (0x9A) - Error codes
      State1Result result = CanPacketDecoder::parse_state1_data(data);
      if (result.valid) {
        motor_.update_error_state(result.error_code);
      }
    }
    break;
  case PARAM: {
    if (is_query_response &&
        data[0] == static_cast<uint8_t>(MaitaCmd::kReadPid)) {
      ParamResult result = CanPacketDecoder::parse_pid_param_data(data);
      if (result.valid) {
        motor_.set_temp_param(result.param_index, result.value);
      }
    }
    break;
  }
  case IGNORE:
    return;
  default:
    break;
  }
}

void MaitaCANDevice::callback(const canfd_frame &frame) {
  if (not use_fd_) {
    std::cerr << "WARNING: CANFD MODE NOT ENABLED" << std::endl;
    return;
  }

  // Determine which type of response based on CAN ID
  bool is_mit_response = (frame.can_id == motor_.get_mit_recv_can_id());
  bool is_query_response = (frame.can_id == motor_.get_query_recv_can_id());

  if (!is_mit_response && !is_query_response) {
    std::cerr << "WARNING: CANFD FRAME ID DOES NOT MATCH MOTOR ID" << std::endl;
    return;
  }

  std::vector<uint8_t> data = get_data_from_frame(frame);
  if (data.empty())
    return;

  if (callback_mode_ == STATE) {
    if (is_mit_response && frame.len >= 6) {
      // Parse MIT response (from 0x500+id)
      MitStateResult result =
          CanPacketDecoder::parse_mit_response_data(motor_, data);
      if (result.valid) {
        motor_.update_motion_state(result.position, result.velocity,
                                   result.torque);
      }
    } else if (is_query_response &&
               data[0] == static_cast<uint8_t>(MaitaCmd::kReadState2)) {
      // Parse state2 response (0x9C)
      // Convert iq (A) to torque (N-m) using Kt
      State2Result result = CanPacketDecoder::parse_state2_data(data);
      if (result.valid) {
        double kt = MOTOR_KT[static_cast<size_t>(motor_.get_motor_type())];
        double torque = result.iq * kt;
        motor_.update_state2(result.position, result.velocity, torque,
                             result.temperature);
      }
    } else if (is_query_response &&
               data[0] == static_cast<uint8_t>(MaitaCmd::kReadState1)) {
      // Parse state1 response (0x9A) - Error codes
      State1Result result = CanPacketDecoder::parse_state1_data(data);
      if (result.valid) {
        motor_.update_error_state(result.error_code);
      }
    }
  } else if (callback_mode_ == PARAM) {
    if (is_query_response &&
        data[0] == static_cast<uint8_t>(MaitaCmd::kReadPid)) {
      ParamResult result = CanPacketDecoder::parse_pid_param_data(data);
      if (result.valid) {
        motor_.set_temp_param(result.param_index, result.value);
      }
    }
  } else if (callback_mode_ == IGNORE) {
    return;
  }
}

can_frame MaitaCANDevice::create_can_frame(canid_t send_can_id,
                                           std::vector<uint8_t> data) {
  can_frame frame;
  std::memset(&frame, 0, sizeof(frame));
  frame.can_id = send_can_id;
  frame.can_dlc = data.size();
  std::copy(data.begin(), data.end(), frame.data);
  return frame;
}

canfd_frame MaitaCANDevice::create_canfd_frame(canid_t send_can_id,
                                               std::vector<uint8_t> data) {
  canfd_frame frame;
  frame.can_id = send_can_id;
  frame.len = data.size();
  frame.flags = CANFD_BRS;
  std::copy(data.begin(), data.end(), frame.data);
  return frame;
}

} // namespace x3arm::maita_motor
