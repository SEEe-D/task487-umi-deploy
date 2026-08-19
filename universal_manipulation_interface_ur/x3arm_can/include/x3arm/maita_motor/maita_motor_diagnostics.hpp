// Copyright 2025 Enactic, Inc.
#pragma once

#include "maita_motor_constants.hpp"
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/can.h>
#include <map>
#include <mutex>
#include <string>

namespace x3arm::maita_motor {

// Fault bitmasks based on protocol
enum SystemErrorState : uint16_t {
  kStall = 0x0002,             // 电机堵转
  kLowVoltage = 0x0004,        // 低压
  kOverVoltage = 0x0008,       // 过压
  kOverCurrent = 0x0010,       // 相电流过流
  kPowerOverrun = 0x0040,      // 功率超限
  kCalibrationErr = 0x0080,    // 标定参数写入错误
  kOverSpeed = 0x0100,         // 超速
  kComponentOverTemp = 0x0800, // 元器件过温
  kMotorOverTemp = 0x1000,     // 电机温度过温
  kEncoderCalibErr = 0x2000,   // 编码器校准错误
  kEncoderDataErr = 0x4000     // 编码器数据错误
};

class FaultLogger {
public:
  // @param log_interval_sec: Duplicate errors for the same motor will only be
  // logged logged once every N seconds.
  FaultLogger(const std::string &filename, int log_interval_sec = 5);

  ~FaultLogger();

  void log(uint32_t motor_id, uint16_t error_code);

  // New: Log traffic for context
  void log_traffic(const can_frame &frame, bool is_tx);

  // New: Register motor type for better parsing
  void register_motor_type(uint32_t motor_id, MotorType type);

  // Generic event logging (e.g., gripper fault). Throttled using the same
  // log_interval_sec_.
  void log_event(const std::string& source, uint32_t device_id, uint32_t code,
                 const std::string& desc);

  static std::string get_error_desc(uint16_t error_code);

private:
  void flush_buffer_to_file();
  std::string parse_frame(const can_frame &frame, bool is_tx); // Helper

  std::string filename_;
  int log_interval_sec_;
  std::ofstream log_file_;
  std::mutex mutex_;

  // Cache to store last logged error state for throttling
  // Key: Motor ID
  // Value: Pair of <Last Error Code, Last Log Time>
  std::map<uint32_t, std::pair<uint16_t, std::chrono::steady_clock::time_point>>
      last_log_map_;

  // Throttle cache for generic events (key = source + ":" + device_id)
  std::map<std::string, std::pair<uint32_t, std::chrono::steady_clock::time_point>>
      last_event_log_map_;

  // Context logging
  struct TrafficEntry {
    std::chrono::system_clock::time_point timestamp;
    bool is_tx;
    can_frame frame;
  };
  std::deque<TrafficEntry> traffic_buffer_;
  size_t max_buffer_size_ = 5000; // Approx 5s @ 1kHz (tuneable)
  std::chrono::steady_clock::time_point
      monitor_end_time_; // When to stop dumping
  std::map<uint32_t, MotorType> motor_type_map_;
};

} // namespace x3arm::maita_motor
