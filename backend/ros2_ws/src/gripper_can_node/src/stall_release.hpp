#pragma once

#include <cmath>

namespace gripper_control {

// Joint convention: zero is closed, negative is open. Compare with measured
// position, never the extra-close hold setpoint: that setpoint can remain
// unreachable while gripping an object. Use the current measurement so a
// finger that has compressed the object can also deliberately reopen.
inline bool requests_stall_release(double requested_rad, double measured_rad,
                                   double opening_deadband_rad) noexcept {
  return std::isfinite(requested_rad) && std::isfinite(measured_rad) &&
         std::isfinite(opening_deadband_rad) && opening_deadband_rad >= 0.0 &&
         requested_rad < measured_rad - opening_deadband_rad;
}

}  // namespace gripper_control
