
#include <x3arm/livelybot_motor/livelybot_motor.hpp>

namespace x3arm::livelybot_motor {

Motor::Motor(uint32_t device_id)
    : device_id_(device_id), state_q_(0.0), state_dq_(0.0), state_tau_(0.0),
      fault_code_(FaultCode::NORMAL) {}

void Motor::update_state(double q, double dq, double tau) {
  state_q_ = q;
  state_dq_ = dq;
  state_tau_ = tau;
}

void Motor::update_fault_state(uint8_t code) {
  fault_code_ = static_cast<FaultCode>(code);
}

} // namespace x3arm::livelybot_motor
