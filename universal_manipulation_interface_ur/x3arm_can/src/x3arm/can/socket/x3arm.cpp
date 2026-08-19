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
#include <x3arm/can/socket/x3arm.hpp>

namespace x3arm::can::socket {

X3Arm::X3Arm(const std::string &can_interface, bool enable_fd)
    : can_interface_(can_interface), enable_fd_(enable_fd) {
  can_socket_ = std::make_unique<canbus::CANSocket>(can_interface_, enable_fd_);
  master_can_device_collection_ =
      std::make_unique<canbus::CANDeviceCollection>(*can_socket_);
  arm_ = std::make_unique<ArmComponent>(*can_socket_);
  gripper_ = std::make_unique<GripperComponent>(*can_socket_);
}

void X3Arm::init_arm_motors(
    const std::vector<maita_motor::MotorType> &motor_types,
    const std::vector<uint32_t> &device_ids) {
  if (motor_types.size() != device_ids.size()) {
    throw std::invalid_argument("Motor types and device IDs vectors must have "
                                "the same size, currently: " +
                                std::to_string(motor_types.size()) + ", " +
                                std::to_string(device_ids.size()));
  }
  arm_->init_motor_devices(motor_types, device_ids, enable_fd_);
  register_maita_device_collection(*arm_);
}

void X3Arm::init_gripper_motor(uint32_t device_id) {
  gripper_->init_motor_device(device_id);
  // Register livelybot device to master collection
  for (const auto &[id, device] :
       gripper_->get_device_collection().get_devices()) {
    master_can_device_collection_->add_device(device);
  }
}

void X3Arm::register_maita_device_collection(
    maita_motor::MaitaDeviceCollection &device_collection) {
  for (const auto &[id, device] :
       device_collection.get_device_collection().get_devices()) {
    master_can_device_collection_->add_device(device);
  }
  sub_maita_device_collections_.push_back(&device_collection);
}

void X3Arm::motor_off_all() {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->motor_off_all();
  }
}

void X3Arm::motor_stop_all() {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->motor_stop_all();
  }
}

void X3Arm::read_state_all() {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->read_state_all();
  }
}

void X3Arm::read_state_one(int i) {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->read_state_one(i);
  }
}

void X3Arm::read_state1_all() {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->read_state1_all();
  }
}

void X3Arm::read_state1_one(int i) {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->read_state1_one(i);
  }
}

void X3Arm::recv_all(int timeout_us) {
  // The timeout for select() is set to timeout_us (default: 500 us).
  // CAN FD
  if (enable_fd_) {
    canfd_frame response_frame;
    while (can_socket_->is_data_available(timeout_us) &&
           can_socket_->read_canfd_frame(response_frame)) {
      master_can_device_collection_->dispatch_frame_callback(response_frame);
    }
  }
  // CAN 2.0
  else {
    can_frame response_frame;
    while (can_socket_->is_data_available(timeout_us) &&
           can_socket_->read_can_frame(response_frame)) {
      master_can_device_collection_->dispatch_frame_callback(response_frame);
    }
  }
}

void X3Arm::read_pid_all(maita_motor::ReadParamIndex param_index) {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->read_pid_all(param_index);
  }
}

void X3Arm::set_callback_mode_all(maita_motor::CallbackMode callback_mode) {
  for (maita_motor::MaitaDeviceCollection *device_collection :
       sub_maita_device_collections_) {
    device_collection->set_callback_mode_all(callback_mode);
  }
}

void X3Arm::set_traffic_callback(
    std::function<void(const can_frame &, bool)> callback) {
  can_socket_->set_traffic_callback(callback);
}

} // namespace x3arm::can::socket
