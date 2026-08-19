// Copyright 2025 Enactic, Inc.
#include "x3arm/maita_motor/maita_motor_diagnostics.hpp"

namespace x3arm::maita_motor {

FaultLogger::FaultLogger(const std::string &filename, int log_interval_sec)
    : filename_(filename), log_interval_sec_(log_interval_sec) {
  log_file_.open(filename_, std::ios::app);
  if (!log_file_.is_open()) {
    std::cerr << "Failed to open log file: " << filename_ << std::endl;
  }
}

FaultLogger::~FaultLogger() {
  if (log_file_.is_open()) {
    log_file_.close();
  }
}

// Helper to format frame data
std::string format_data(const can_frame &frame) {
  std::stringstream ss;
  for (int i = 0; i < frame.can_dlc; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)frame.data[i]
       << " ";
  }
  return ss.str();
}

// Helper for conversion
static double uint_to_double(uint16_t x, double min, double max, int bits) {
  double span = max - min;
  double data_norm = static_cast<double>(x) / ((1 << bits) - 1);
  return data_norm * span + min;
}

void FaultLogger::register_motor_type(uint32_t motor_id, MotorType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  motor_type_map_[motor_id] = type;
}

void FaultLogger::log_traffic(const can_frame &frame, bool is_tx) {
  // FILTERING LOGIC --------------------------------------------------------
  uint32_t id = frame.can_id & 0x7FF;

  // 1. Filter out Query Commands (TX 0x140+) - High frequency noise
  if (is_tx && id >= QUERY_SEND_OFFSET && id <= QUERY_SEND_OFFSET + 32) {
    return; // Skip logging routine query commands
  }

  // 2. Filter out Query Responses (RX 0x240+) unless they are interesting
  if (!is_tx && id >= QUERY_RECV_OFFSET && id <= QUERY_RECV_OFFSET + 32) {
    if (frame.can_dlc > 0) {
      uint8_t cmd = frame.data[0];
      // Filter 0x9A (State1) if NO ERROR
      if (cmd == 0x9A) {
        if (frame.can_dlc >= 8) {
          uint16_t err = frame.data[6] | (frame.data[7] << 8);
          if (err == 0)
            return; // Skip if no error
        }
      }
      // Filter 0x9C (State2 - Current/Temp/etc) - Constant polling noise
      else if (cmd == 0x9C) {
        return; // Skip routine state reporting
      }
      // Filter 0x9D (State3), 0x92 (Angle), 0x90 (Encoder)
      else if (cmd == 0x9D || cmd == 0x92 || cmd == 0x90) {
        return;
      }
    }
  }
  // ------------------------------------------------------------------------

  std::lock_guard<std::mutex> lock(mutex_);

  auto now = std::chrono::system_clock::now();
  TrafficEntry entry{now, is_tx, frame};

  // Always add to buffer
  traffic_buffer_.push_back(entry);
  if (traffic_buffer_.size() > max_buffer_size_) {
    traffic_buffer_.pop_front();
  }

  // Check if we should log immediately (post-fault window)
  if (std::chrono::steady_clock::now() < monitor_end_time_) {
    if (log_file_.is_open()) {
      auto time_t = std::chrono::system_clock::to_time_t(now);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;

      std::string direction = is_tx ? "TX" : "RX";
      std::string parsed = parse_frame(frame, is_tx);

      log_file_ << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                << "." << std::setfill('0') << std::setw(3) << ms.count()
                << "] "
                << "[" << direction << "] ID:0x" << std::hex << frame.can_id
                << std::dec << " " << format_data(frame) << "| " << parsed
                << std::endl;
    }
  }
}

std::string FaultLogger::parse_frame(const can_frame &frame, bool is_tx) {
  uint32_t id = frame.can_id & 0x7FF; // Standard ID
  std::stringstream ss;

  // MIT Control (0x400 send, 0x500 recv)
  if (id >= MIT_SEND_OFFSET && id <= MIT_SEND_OFFSET + 32) {
    if (is_tx && frame.can_dlc == 8) {
      // Decode MIT CMD
      // Byte: 0-1(p), 2-3(v,kp), 4(kp), 5(kd), 6(kd,tau), 7(tau)
      uint16_t p_uint = (frame.data[0] << 8) | frame.data[1];
      uint16_t v_uint = (frame.data[2] << 4) | (frame.data[3] >> 4);
      uint16_t kp_uint = ((frame.data[3] & 0x0F) << 8) | frame.data[4];
      uint16_t kd_uint = (frame.data[5] << 4) | (frame.data[6] >> 4);
      uint16_t t_uint = ((frame.data[6] & 0x0F) << 8) | frame.data[7];

      // Get limits (Default to MTX436/Generic if not found)
      uint32_t dev_id = id - MIT_SEND_OFFSET;
      MotorType type = MotorType::MTX436;
      if (motor_type_map_.count(dev_id))
        type = motor_type_map_[dev_id];

      LimitParam limits = MOTOR_LIMIT_PARAMS[static_cast<int>(type)];

      double p = uint_to_double(p_uint, -limits.pMax, limits.pMax, 16);
      double v = uint_to_double(v_uint, -limits.vMax, limits.vMax, 12);
      double kp = uint_to_double(kp_uint, 0, 500, 12);
      double kd = uint_to_double(kd_uint, 0, 5, 12);
      double t = uint_to_double(t_uint, -limits.tMax, limits.tMax, 12);

      ss << "MIT CMD P:" << std::fixed << std::setprecision(2) << p
         << " V:" << v << " Kp:" << kp << " Kd:" << kd << " T:" << t;
      return ss.str();
    }
  }
  if (id >= MIT_RECV_OFFSET && id <= MIT_RECV_OFFSET + 32) {
    if (!is_tx && frame.can_dlc >= 6) {
      uint32_t dev_id = id - MIT_RECV_OFFSET;
      // Decode MIT RESP
      uint16_t p_uint = (frame.data[1] << 8) | frame.data[2];
      uint16_t v_uint = (frame.data[3] << 4) | (frame.data[4] >> 4);
      uint16_t t_uint = ((frame.data[4] & 0x0F) << 8) | frame.data[5];

      MotorType type = MotorType::MTX436;
      if (motor_type_map_.count(dev_id))
        type = motor_type_map_[dev_id];
      LimitParam limits = MOTOR_LIMIT_PARAMS[static_cast<int>(type)];

      double p = uint_to_double(p_uint, -limits.pMax, limits.pMax, 16);
      double v = uint_to_double(v_uint, -limits.vMax, limits.vMax, 12);
      double t = uint_to_double(t_uint, -limits.tMax, limits.tMax, 12);

      ss << "MIT FB P:" << std::fixed << std::setprecision(2) << p << " V:" << v
         << " T:" << t;
      if (frame.can_dlc >= 7)
        ss << " Td:"
           << (int)frame.data[6]; // Td? Protocol V4.3 doesn't specify byte 6
                                  // explicit for MIT? Verify later.
      return ss.str();
    }
  }

  // Query Responses (Errors only typically logged now due to filtering, plus
  // others)
  if (id >= QUERY_RECV_OFFSET && id <= QUERY_RECV_OFFSET + 32) {
    if (!is_tx && frame.can_dlc > 0) {
      uint8_t cmd = frame.data[0];
      if (cmd == 0x9A) {
        if (frame.can_dlc >= 8) {
          uint16_t err = frame.data[6] | (frame.data[7] << 8);
          if (err != 0) {
            int8_t m_temp = static_cast<int8_t>(frame.data[1]);
            int8_t mos_temp = static_cast<int8_t>(frame.data[2]);
            uint16_t voltage = frame.data[4] | (frame.data[5] << 8);
            ss << "ERROR REPORT [" << get_error_desc(err) << "] "
               << "V:" << (voltage * 0.1) << "V T:" << (int)m_temp
               << "C MOS:" << (int)mos_temp << "C";
            return ss.str();
          }
        }
      }
    }
  }

  return "";
}

void FaultLogger::log(uint32_t motor_id, uint16_t error_code) {
  if (error_code == 0)
    return;

  std::lock_guard<std::mutex> lock(mutex_);

  // Check for throttle condition
  // Key: motor_id, Value: pair<last_error_code, last_log_time>
  auto now = std::chrono::steady_clock::now();
  auto it = last_log_map_.find(motor_id);

  bool should_log = false;
  bool state_changed = false;

  if (it == last_log_map_.end()) {
    // First time seeing this motor error
    should_log = true;
    state_changed = true;
  } else {
    auto &last_entry = it->second;
    uint16_t last_code = last_entry.first;
    auto last_time = last_entry.second;

    if (last_code != error_code) {
      // Error state changed
      should_log = true;
      state_changed = true;
    } else {
      // Same error, check time interval
      auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(now - last_time)
              .count();
      if (duration >= log_interval_sec_) {
        should_log = true;
      }
    }
  }

  if (should_log) {
    if (log_file_.is_open()) {

      // TRIGGER DUMP if state changed (new fault)
      // Or if checking logic demands every log to have context?
      // User said "when fault occurs".
      if (state_changed) {
        log_file_
            << "\n========== FAULT DETECTED - DUMPING HISTORY ==========\n";
        // Dump buffer
        for (const auto &entry : traffic_buffer_) {
          auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        entry.timestamp.time_since_epoch()) %
                    1000;
          std::string direction = entry.is_tx ? "TX" : "RX";
          std::string parsed = parse_frame(entry.frame, entry.is_tx);
          log_file_ << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                    << "." << std::setfill('0') << std::setw(3) << ms.count()
                    << "] "
                    << "[" << direction << "] ID:0x" << std::hex
                    << entry.frame.can_id << std::dec << " "
                    << format_data(entry.frame) << "| " << parsed << std::endl;
        }
        log_file_ << "======================================================\n";

        // Set monitor end time to 5 seconds from now
        monitor_end_time_ = now + std::chrono::seconds(5);
      }

      auto sys_now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(sys_now);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    sys_now.time_since_epoch()) %
                1000;

      log_file_ << "["
                << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                << "." << std::setfill('0') << std::setw(3) << ms.count()
                << "] "
                << "Motor ID: " << motor_id << " Error Code: 0x" << std::hex
                << std::setw(4) << error_code << std::dec
                << " Description: " << get_error_desc(error_code) << std::endl;
      log_file_.flush();

      // Update cache
      last_log_map_[motor_id] = {error_code, now};
    }
  }
}

void FaultLogger::log_event(const std::string &source, uint32_t device_id,
                            uint32_t code, const std::string &desc) {
  if (code == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();
  const std::string key = source + ":" + std::to_string(device_id);

  auto it = last_event_log_map_.find(key);
  bool should_log = false;
  bool state_changed = false;

  if (it == last_event_log_map_.end()) {
    should_log = true;
    state_changed = true;
  } else {
    const uint32_t last_code = it->second.first;
    const auto last_time = it->second.second;
    if (last_code != code) {
      should_log = true;
      state_changed = true;
    } else {
      const auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(now - last_time)
              .count();
      if (duration >= log_interval_sec_) {
        should_log = true;
      }
    }
  }

  if (!should_log || !log_file_.is_open()) {
    return;
  }

  // For generic events we don't dump traffic history (avoid log spam); only
  // write the event line.
  (void)state_changed;

  const auto sys_now = std::chrono::system_clock::now();
  const auto time_t = std::chrono::system_clock::to_time_t(sys_now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      sys_now.time_since_epoch()) %
                  1000;

  log_file_ << "["
            << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
            << "Source: " << source << " Device ID: " << device_id
            << " Code: " << code << " Desc: " << desc << std::endl;
  log_file_.flush();

  last_event_log_map_[key] = {code, now};
}

std::string FaultLogger::get_error_desc(uint16_t error_code) {
  std::string desc;
  if (error_code & kStall)
    desc += "Stall(堵转) ";
  if (error_code & kLowVoltage)
    desc += "LowVoltage(低压) ";
  if (error_code & kOverVoltage)
    desc += "OverVoltage(过压) ";
  if (error_code & kOverCurrent)
    desc += "OverCurrent(过流) ";
  if (error_code & kPowerOverrun)
    desc += "PowerOverrun(功率超限) ";
  if (error_code & kCalibrationErr)
    desc += "CalibrationErr(标定错误) ";
  if (error_code & kOverSpeed)
    desc += "OverSpeed(超速) ";
  if (error_code & kComponentOverTemp)
    desc += "ComponentOverTemp(元件过温) ";
  if (error_code & kMotorOverTemp)
    desc += "MotorOverTemp(电机过温) ";
  if (error_code & kEncoderCalibErr)
    desc += "EncoderCalibErr(编码器校准错误) ";
  if (error_code & kEncoderDataErr)
    desc += "EncoderDataErr(编码器数据错误) ";

  if (desc.empty())
    desc = "Unknown Error";
  return desc;
}

} // namespace x3arm::maita_motor
