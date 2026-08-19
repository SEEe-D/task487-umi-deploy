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

#include <iostream>
#include <x3arm/canbus/can_device_collection.hpp>
#include <x3arm/canbus/can_socket.hpp>

namespace x3arm::canbus {

CANDeviceCollection::CANDeviceCollection(CANSocket& can_socket) : can_socket_(can_socket) {}

CANDeviceCollection::~CANDeviceCollection() {}

void CANDeviceCollection::add_device(const std::shared_ptr<CANDevice>& device) {
    if (!device) return;

    // Register all receive CAN IDs for this device (supports multi-CAN ID devices)
    std::vector<canid_t> recv_ids = device->get_all_recv_can_ids();
    for (canid_t recv_id : recv_ids) {
        devices_[recv_id] = device;
    }
}

void CANDeviceCollection::remove_device(const std::shared_ptr<CANDevice>& device) {
    if (!device) return;

    // Remove all receive CAN IDs for this device
    std::vector<canid_t> recv_ids = device->get_all_recv_can_ids();
    for (canid_t recv_id : recv_ids) {
        auto it = devices_.find(recv_id);
        if (it != devices_.end()) {
            devices_.erase(it);
        }
    }
}

void CANDeviceCollection::dispatch_frame_callback(can_frame& frame) {
    auto it = devices_.find(frame.can_id);
    if (it != devices_.end()) {
        it->second->callback(frame);
    }
    // Note: Silently ignore frames for unknown devices (this is normal in CAN
    // networks)
}

void CANDeviceCollection::dispatch_frame_callback(canfd_frame& frame) {
    auto it = devices_.find(frame.can_id);
    if (it != devices_.end()) {
        it->second->callback(frame);
    }
    // Note: Silently ignore frames for unknown devices (this is normal in CAN
    // networks)
}

}  // namespace x3arm::canbus
