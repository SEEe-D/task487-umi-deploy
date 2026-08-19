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

#include <vector>

#include "../../canbus/can_socket.hpp"
#include "../../maita_motor/maita_motor.hpp"
#include "../../maita_motor/maita_motor_device_collection.hpp"

namespace x3arm::can::socket {

class ArmComponent : public maita_motor::MaitaDeviceCollection {
public:
    ArmComponent(canbus::CANSocket& can_socket);
    ~ArmComponent() = default;

    // Initialize with device_ids (multi-CAN ID support)
    void init_motor_devices(const std::vector<maita_motor::MotorType>& motor_types,
                            const std::vector<uint32_t>& device_ids, bool use_fd);

private:
    std::vector<maita_motor::Motor> motors_;
};

}  // namespace x3arm::can::socket
