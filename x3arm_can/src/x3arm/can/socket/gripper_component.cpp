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

#include <linux/can.h>
#include <linux/can/raw.h>

#include <iostream>
#include <x3arm/can/socket/gripper_component.hpp>

namespace x3arm::can::socket {

GripperComponent::GripperComponent(canbus::CANSocket &can_socket)
    : can_socket_(can_socket),
      device_collection_(
          std::make_unique<canbus::CANDeviceCollection>(can_socket)) {}

void GripperComponent::init_motor_device(uint32_t device_id) {
  // 创建电机
  motor_ = std::make_unique<livelybot_motor::Motor>(device_id);
  // 创建CAN设备
  motor_device_ =
      std::make_shared<livelybot_motor::LivelybotCANDevice>(*motor_);
  // 添加到设备集合
  device_collection_->add_device(motor_device_);
}

void GripperComponent::stop() {
  if (!motor_device_)
    return;

  auto packet =
      livelybot_motor::LivelybotPacketEncoder::create_stop_command(*motor_);
  send_packet(packet);
}

void GripperComponent::set_zero() {
  if (!motor_device_)
    return;

  auto packet =
      livelybot_motor::LivelybotPacketEncoder::create_set_zero_command(*motor_);
  send_packet(packet);
}

void GripperComponent::save_flash() {
  if (!motor_device_)
    return;

  auto packet =
      livelybot_motor::LivelybotPacketEncoder::create_save_flash_command(
          *motor_);
  send_packet(packet);
}

void GripperComponent::query_state() {
  if (!motor_)
    return;
  send_packet(
      livelybot_motor::LivelybotPacketEncoder::create_query_state_command(
          *motor_));
}

void GripperComponent::query_fault() {
  if (!motor_)
    return;
  send_packet(
      livelybot_motor::LivelybotPacketEncoder::create_query_fault_command(
          *motor_));
}

void GripperComponent::mit_control(const livelybot_motor::MITParam &param) {
  if (!motor_device_)
    return;

  auto packet =
      livelybot_motor::LivelybotPacketEncoder::create_mit_control_command(
          *motor_, param);
  send_packet(packet);
}

void GripperComponent::recv(int timeout_us) {
  if (can_socket_.is_data_available(timeout_us)) {
    can_frame frame;
    if (can_socket_.read_can_frame(frame)) {
      device_collection_->dispatch_frame_callback(frame);
    }
  }
}

void GripperComponent::send_packet(const livelybot_motor::CANPacket &packet) {
  can_frame frame = motor_device_->create_can_frame(packet);
  can_socket_.write_can_frame(frame);
}

} // namespace x3arm::can::socket
