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
#include <set>
#include <x3arm/maita_motor/maita_motor_device_collection.hpp>

namespace x3arm::maita_motor {

MaitaDeviceCollection::MaitaDeviceCollection(canbus::CANSocket &can_socket)
    : can_socket_(can_socket),
      can_packet_encoder_(std::make_unique<CanPacketEncoder>()),
      can_packet_decoder_(std::make_unique<CanPacketDecoder>()),
      device_collection_(
          std::make_unique<canbus::CANDeviceCollection>(can_socket_)) {}

void MaitaDeviceCollection::motor_off_all() {
  for (auto maita_device : get_maita_devices()) {
    auto &motor = maita_device->get_motor();
    CANPacket packet = CanPacketEncoder::create_motor_off_command(motor);
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::set_zero_one(int i) {
  auto maita_device = get_maita_devices().at(i);
  CANPacket packet =
      CanPacketEncoder::create_set_zero_command(maita_device->get_motor());
  send_command_to_device(maita_device, packet);
}

void MaitaDeviceCollection::system_reset_one(int i) {
  auto maita_device = get_maita_devices().at(i);
  CANPacket packet =
      CanPacketEncoder::create_system_reset_command(maita_device->get_motor());
  send_command_to_device(maita_device, packet);
}

void MaitaDeviceCollection::set_zero_all() {
  for (auto maita_device : get_maita_devices()) {
    CANPacket packet =
        CanPacketEncoder::create_set_zero_command(maita_device->get_motor());
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::motor_stop_all() {
  for (auto maita_device : get_maita_devices()) {
    CANPacket packet =
        CanPacketEncoder::create_motor_stop_command(maita_device->get_motor());
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::read_state_one(int i) {
  auto maita_device = get_maita_devices().at(i);
  auto packet =
      CanPacketEncoder::create_read_state_command(maita_device->get_motor());
  send_command_to_device(maita_device, packet);
}

void MaitaDeviceCollection::read_state_all() {
  for (auto maita_device : get_maita_devices()) {
    auto &motor = maita_device->get_motor();
    CANPacket packet = CanPacketEncoder::create_read_state_command(motor);
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::read_state1_one(int i) {
  auto maita_device = get_maita_devices().at(i);
  auto packet =
      CanPacketEncoder::create_read_state1_command(maita_device->get_motor());
  send_command_to_device(maita_device, packet);
}

void MaitaDeviceCollection::read_state1_all() {
  for (auto maita_device : get_maita_devices()) {
    auto &motor = maita_device->get_motor();
    CANPacket packet = CanPacketEncoder::create_read_state1_command(motor);
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::set_callback_mode_all(CallbackMode callback_mode) {
  for (auto maita_device : get_maita_devices()) {
    maita_device->set_callback_mode(callback_mode);
  }
}

void MaitaDeviceCollection::read_pid_one(int i, ReadParamIndex param_index) {
  CANPacket packet = CanPacketEncoder::create_read_pid_command(
      get_maita_devices()[i]->get_motor(), param_index);
  send_command_to_device(get_maita_devices()[i], packet);
}

void MaitaDeviceCollection::read_pid_all(ReadParamIndex param_index) {
  for (auto maita_device : get_maita_devices()) {
    CANPacket packet = CanPacketEncoder::create_read_pid_command(
        maita_device->get_motor(), param_index);
    send_command_to_device(maita_device, packet);
  }
}

void MaitaDeviceCollection::send_command_to_device(
    std::shared_ptr<MaitaCANDevice> maita_device, const CANPacket &packet) {
  if (can_socket_.is_canfd_enabled()) {
    canfd_frame frame =
        maita_device->create_canfd_frame(packet.send_can_id, packet.data);
    can_socket_.write_canfd_frame(frame);
  } else {
    can_frame frame =
        maita_device->create_can_frame(packet.send_can_id, packet.data);
    can_socket_.write_can_frame(frame);
  }
}

void MaitaDeviceCollection::mit_control_one(int i, const MITParam &mit_param) {
  CANPacket mit_cmd = CanPacketEncoder::create_mit_control_command(
      get_maita_devices()[i]->get_motor(), mit_param);
  send_command_to_device(get_maita_devices()[i], mit_cmd);
}

void MaitaDeviceCollection::mit_control_all(
    const std::vector<MITParam> &mit_params) {
  auto devices = get_maita_devices();
  for (size_t i = 0; i < mit_params.size() && i < devices.size(); i++) {
    CANPacket mit_cmd = CanPacketEncoder::create_mit_control_command(
        devices[i]->get_motor(), mit_params[i]);
    send_command_to_device(devices[i], mit_cmd);
  }
}

void MaitaDeviceCollection::speed_control_one(int i, int32_t speed_rpm_x10) {
  CANPacket packet = CanPacketEncoder::create_speed_control_command(
      get_maita_devices()[i]->get_motor(), speed_rpm_x10);
  send_command_to_device(get_maita_devices()[i], packet);
}

void MaitaDeviceCollection::speed_control_all(
    const std::vector<int32_t> &speeds) {
  auto devices = get_maita_devices();
  for (size_t i = 0; i < speeds.size() && i < devices.size(); i++) {
    CANPacket packet = CanPacketEncoder::create_speed_control_command(
        devices[i]->get_motor(), speeds[i]);
    send_command_to_device(devices[i], packet);
  }
}

void MaitaDeviceCollection::position_control_one(int i,
                                                 int32_t position_0_01deg) {
  CANPacket packet = CanPacketEncoder::create_position_control_command(
      get_maita_devices()[i]->get_motor(), position_0_01deg);
  send_command_to_device(get_maita_devices()[i], packet);
}

void MaitaDeviceCollection::position_control_all(
    const std::vector<int32_t> &positions) {
  auto devices = get_maita_devices();
  for (size_t i = 0; i < positions.size() && i < devices.size(); i++) {
    CANPacket packet = CanPacketEncoder::create_position_control_command(
        devices[i]->get_motor(), positions[i]);
    send_command_to_device(devices[i], packet);
  }
}

std::vector<Motor> MaitaDeviceCollection::get_motors() const {
  std::vector<Motor> motors;
  for (auto maita_device : get_maita_devices()) {
    motors.push_back(maita_device->get_motor());
  }
  return motors;
}

Motor MaitaDeviceCollection::get_motor(int i) const {
  return get_maita_devices().at(i)->get_motor();
}

// Fixed: Use set to avoid duplicate devices (same device registered with
// multiple CAN IDs)
std::vector<std::shared_ptr<MaitaCANDevice>>
MaitaDeviceCollection::get_maita_devices() const {
  std::vector<std::shared_ptr<MaitaCANDevice>> maita_devices;
  std::set<MaitaCANDevice *>
      seen; // Track seen device pointers to avoid duplicates

  for (const auto &[id, device] : device_collection_->get_devices()) {
    auto maita_device = std::dynamic_pointer_cast<MaitaCANDevice>(device);
    if (maita_device && seen.find(maita_device.get()) == seen.end()) {
      seen.insert(maita_device.get());
      maita_devices.push_back(maita_device);
    }
  }
  return maita_devices;
}

} // namespace x3arm::maita_motor
