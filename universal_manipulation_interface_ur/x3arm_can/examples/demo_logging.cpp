// Copyright 2025 Enactic, Inc.
// Interactive Demo - Dual Motor Test with Protocol Documentation
//
// ============================================================================
// MAITA Motor Protocol V4.3 - 通讯格式说明
// ============================================================================
//
// CAN ID 格式:
//   发送 (Query 命令): 0x140 + ID (1~32)  → 例如 ID=1: 0x141
//   发送 (MIT 命令):   0x400 + ID (1~32)  → 例如 ID=1: 0x401
//   回复 (Query 响应): 0x240 + ID (1~32)  → 例如 ID=1: 0x241
//   回复 (MIT 响应):   0x500 + ID (1~32)  → 例如 ID=1: 0x501
//
// 常用命令:
//   0x30 - 读取 PID 参数
//   0x64 - 写入当前位置为零点
//   0x76 - 系统复位
//   0x80 - 电机关闭
//   0x81 - 电机停止
//   0x9C - 读取电机状态2 (温度, iq, 速度, 角度)
//   0xA2 - 速度闭环控制
//   0xA4 - 绝对位置闭环控制
//
// ============================================================================

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex> // Added for safety if needed, though logger has its own
#include <sstream>
#include <thread>
#include <x3arm/can/socket/x3arm.hpp>
#include <x3arm/maita_motor/maita_motor_constants.hpp>
#include <x3arm/maita_motor/maita_motor_control.hpp>
#include <x3arm/maita_motor/maita_motor_diagnostics.hpp> // Added

using namespace x3arm::maita_motor;

std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

static std::string motor_type_to_string(MotorType t) {
  switch (t) {
  case MotorType::MTX436:
    return "MTX436";
  case MotorType::MTX660:
    return "MTX660";
  case MotorType::MTX410:
    return "MTX410";
  default:
    return "UNKNOWN";
  }
}

static std::string hex_u32(uint32_t v) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::uppercase << v << std::dec;
  return oss.str();
}

// State2 响应打印: 包含 temp (协议 2.15.3)
void print_state(const Motor &m) {
  std::cout << "id=" << m.get_device_id() << " ("
            << motor_type_to_string(m.get_motor_type()) << ") " << std::fixed
            << std::setprecision(2) << "pos=" << std::setw(8)
            << m.get_position() * 180.0 / M_PI << " deg"
            << ", vel=" << std::setw(7) << m.get_velocity() * 180.0 / M_PI
            << " dps"
            << ", torque=" << std::setw(6) << m.get_torque() << " Nm"
            << ", temp=" << m.get_temperature() << " C" << std::endl;
}

// MIT 响应打印: 不包含 temp (协议 5.3)
void print_mit_state(const Motor &m) {
  std::cout << "id=" << m.get_device_id() << " ("
            << motor_type_to_string(m.get_motor_type()) << ") " << std::fixed
            << std::setprecision(2) << "pos=" << std::setw(8)
            << m.get_position() * 180.0 / M_PI << " deg"
            << ", vel=" << std::setw(7) << m.get_velocity() * 180.0 / M_PI
            << " dps"
            << ", torque=" << std::setw(6) << m.get_torque() << " Nm"
            << std::endl;
}

// Monitor thread function (New addition)
void monitor_routine(x3arm::can::socket::X3Arm *arm, FaultLogger *logger) {
  std::cout << "[Monitor] Thread started." << std::endl;
  while (g_running) {
    // 1. Receive ALL frames from CAN bus
    // This handles both control feedback (for main thread updates) and status
    // queries Timeout set to 10ms to allow frequent checks of g_running
    arm->recv_all(10);

    // 2. Periodically query Fault Status (0x9A)
    static auto last_query_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                              last_query_time)
            .count() > 200) {

      // Ensure specific mode for fault reading if needed, but for now we assume
      // transparent Note: If Main thread sets Ignore mode, these might be
      // ignored. Ideally we should enforce STATE mode or independent parsing.
      // For this demo, we assume cooperative usage.

      arm->read_state1_all();
      last_query_time = now;

      // Check and Log errors
      auto &arm_comp = arm->get_arm();
      for (const auto &motor : arm_comp.get_motors()) {
        uint16_t err = motor.get_error_code();
        if (err != 0) {
          logger->log(motor.get_device_id(), err);
          // Optional: Print to console nicely if not interfering too much
          std::cout << "\r[Monitor] FAULT on ID " << motor.get_device_id()
                    << ": " << FaultLogger::get_error_desc(err)
                    << "            " << std::endl;
          // Reprint prompt if needed? No, too complex.
        }
      }
    }
  }
  std::cout << "[Monitor] Thread stopped." << std::endl;
}

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);

  try {
    std::string can_if = argc > 1 ? argv[1] : "can0";
    std::cout << "Init on " << can_if << "..." << std::endl;

    x3arm::can::socket::X3Arm arm(can_if, false);
    // 两电机：index=0 -> MTX660@id=0x01, index=1 -> MTX436@id=0x02
    arm.init_arm_motors({MotorType::MTX660, MotorType::MTX660,
                         MotorType::MTX436, MotorType::MTX436,
                         MotorType::MTX436, MotorType::MTX436,
                         MotorType::MTX436, MotorType::MTX410},
                        {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08});

    // 2. Initialize Fault Logger
    FaultLogger logger("motor_faults.log", 5);
    // Register motor type (assuming MTX436 for ID 1 based on other demos, or
    // generic) Actually demo default uses MTX_436.
    logger.register_motor_type(1, x3arm::maita_motor::MotorType::MTX436);

    // Wire traffic logging
    arm.set_traffic_callback([&logger](const can_frame &frame, bool is_tx) {
      logger.log_traffic(frame, is_tx);
    });

    // 3. Start Monitor Thread
    std::thread monitor_thread(monitor_routine, &arm, &logger);

    auto motors = arm.get_arm().get_motors();
    std::cout << "Motors on bus: " << motors.size() << std::endl;
    for (size_t i = 0; i < motors.size(); ++i) {
      const auto &m = motors[i];
      std::cout << "  [" << i << "] "
                << motor_type_to_string(m.get_motor_type())
                << " @ device_id=" << hex_u32(m.get_device_id()) << std::endl;
    }
    std::cout << std::endl;

    int motor_idx = 0; // 当前选择的电机 index
    std::string input;
    while (g_running) {
      motors = arm.get_arm().get_motors();
      if (motors.empty()) {
        throw std::runtime_error("No motors initialized.");
      }
      if (motor_idx < 0 || static_cast<size_t>(motor_idx) >= motors.size()) {
        motor_idx = 0;
      }
      const auto &cur_m = motors[static_cast<size_t>(motor_idx)];

      std::cout << "\n=== Dual Motor Test (Logging Active) ===" << std::endl;
      std::cout << "Current: [" << motor_idx << "] "
                << motor_type_to_string(cur_m.get_motor_type()) << " @ "
                << hex_u32(cur_m.get_device_id()) << std::endl;
      std::cout << "s. Switch Motor (cycle)" << std::endl;
      std::cout << "a. Read State ALL (0x9C)" << std::endl;
      std::cout << "1. Read State (0x9C) - current" << std::endl;
      std::cout << "2. Position Control (0xA4)" << std::endl;
      std::cout << "3. Speed Control (0xA2)" << std::endl;
      std::cout << "4. MIT Control (0x400+id)" << std::endl;
      std::cout << "5. Set Zero (0x64)" << std::endl;
      std::cout << "z. Set Zero ALL (0x64) [Write ROM]" << std::endl;
      std::cout << "6. Motor Off (0x80)" << std::endl;
      std::cout << "7. Motor Stop (0x81)" << std::endl;
      std::cout << "8. Read PID (0x30)" << std::endl;
      std::cout << "9. MIT Single Shot Test" << std::endl;
      std::cout << "0. System Reset (0x76)" << std::endl;
      std::cout << "r. System Reset ALL (0x76)" << std::endl;
      std::cout << "m. MIT Control ALL (0x400+id)" << std::endl;
      std::cout << "q. Quit" << std::endl;
      std::cout << "Command: ";
      std::getline(std::cin, input);

      if (input == "s" || input == "S") {
        motor_idx = (motor_idx + 1) % static_cast<int>(motors.size());
        continue;

      } else if (input == "a" || input == "A") {
        // ============================================================
        // 读取全部电机状态2 (0x9C)
        // ============================================================
        arm.set_callback_mode_all(CallbackMode::STATE);
        arm.read_state_all();
        // arm.recv_all(1000);  <-- REPLACED with sleep, monitor thread handles
        // recv
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        for (size_t i = 0; i < motors.size(); ++i) {
          std::cout << "[RECV] "
                    << hex_u32(arm.get_arm()
                                   .get_motor(static_cast<int>(i))
                                   .get_query_recv_can_id())
                    << ": ";
          print_state(arm.get_arm().get_motor(static_cast<int>(i)));
        }

      } else if (input == "1") {
        // ============================================================
        // 读取电机状态2 (0x9C) - 协议 2.15 节
        // ============================================================
        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0x9C (ReadState2)" << std::endl;
        arm.set_callback_mode_all(CallbackMode::STATE);
        arm.read_state_one(motor_idx);
        // arm.recv_all(1000); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        std::cout << "[RECV] " << hex_u32(cur_m.get_query_recv_can_id())
                  << ": ";
        print_state(arm.get_arm().get_motor(motor_idx));

      } else if (input == "2") {
        // ============================================================
        // 绝对位置闭环控制 (0xA4) - 协议 2.21 节
        // ============================================================
        std::cout << "Target (deg): ";
        std::getline(std::cin, input);
        int32_t target = static_cast<int32_t>(std::stod(input) * 100);

        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0xA4 (PosClosedLoop), target=" << target << " (0.01deg)"
                  << std::endl;
        arm.set_callback_mode_all(
            CallbackMode::IGNORE); // Caution: might suppress logs
        arm.get_arm().position_control_one(motor_idx, target);
        // arm.recv_all(500); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        std::cout << "Monitoring via 0x9C..." << std::endl;
        arm.set_callback_mode_all(CallbackMode::STATE);
        for (int i = 0; i < 30 && g_running; ++i) {
          arm.read_state_one(motor_idx);
          // arm.recv_all(500); <-- REPLACED
          std::this_thread::sleep_for(
              std::chrono::milliseconds(100)); // sleep instead of recv

          std::cout << "[" << i + 1 << "] ";
          print_state(arm.get_arm().get_motor(motor_idx));
        }

      } else if (input == "3") {
        // ============================================================
        // 速度闭环控制 (0xA2) - 协议 2.20 节
        // ============================================================
        std::cout << "Speed (dps): ";
        std::getline(std::cin, input);
        int32_t speed = static_cast<int32_t>(std::stod(input) * 100);

        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0xA2 (SpeedClosedLoop), speed=" << speed << " (0.01dps)"
                  << std::endl;
        arm.set_callback_mode_all(CallbackMode::IGNORE);
        arm.get_arm().speed_control_one(motor_idx, speed);
        // arm.recv_all(500); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        std::cout << "Running 3s, monitoring via 0x9C..." << std::endl;
        arm.set_callback_mode_all(CallbackMode::STATE);
        for (int i = 0; i < 30 && g_running; ++i) {
          arm.read_state_one(motor_idx);
          // arm.recv_all(500); <-- REPLACED
          std::this_thread::sleep_for(std::chrono::milliseconds(100));

          std::cout << "[" << i + 1 << "] ";
          print_state(arm.get_arm().get_motor(motor_idx));
        }
        arm.get_arm().speed_control_one(motor_idx, 0);
        // arm.recv_all(500); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "Stopped." << std::endl;

      } else if (input == "4") {
        // ============================================================
        // MIT 运动模式控制 (0x400+ID) - 协议 5 章
        // ============================================================
        std::cout << "Kp: ";
        std::getline(std::cin, input);
        double kp = std::stod(input);
        std::cout << "Kd: ";
        std::getline(std::cin, input);
        double kd = std::stod(input);
        std::cout << "Target (deg): ";
        std::getline(std::cin, input);
        double target_deg = std::stod(input);
        double target_rad = target_deg * M_PI / 180.0;

        MITParam params = {kp, kd, target_rad, 0.0, 0.0};
        arm.set_callback_mode_all(CallbackMode::STATE);

        std::cout << "[SEND] " << hex_u32(cur_m.get_mit_send_can_id())
                  << ": MIT mode, kp=" << kp << ", kd=" << kd
                  << ", target=" << target_rad << " rad" << std::endl;
        std::cout << "MIT running 3s (requires continuous sending)..."
                  << std::endl;
        for (int i = 0; i < 300 && g_running; ++i) {
          arm.get_arm().mit_control_one(motor_idx, params);
          // arm.recv_all(1000); // 1ms timeout <-- REPLACED

          if (i % 10 == 0) {
            std::cout << "[RECV] " << hex_u32(cur_m.get_mit_recv_can_id())
                      << " [" << i * 10 << "ms] ";
            print_mit_state(arm.get_arm().get_motor(
                motor_idx)); // MIT state updated by monitor thread
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        arm.get_arm().mit_control_one(motor_idx, {0, 0, 0, 0, 0});
        // arm.recv_all(50); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "Released." << std::endl;

      } else if (input == "5") {
        // ============================================================2
        // 写入当前位置为零点 (0x64) - 协议 2.10 节
        // ============================================================
        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0x64 (WriteCurPosAsZero)" << std::endl;
        std::cout << "NOTE: Requires 0x76 system reset to take effect!"
                  << std::endl;
        std::cout << "Confirm? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          arm.get_arm().set_zero_one(motor_idx);
          // arm.recv_all(500); <-- REPLACED
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          std::cout << "[RECV] " << hex_u32(cur_m.get_query_recv_can_id())
                    << ": Zero offset written to ROM." << std::endl;
        } else {
          std::cout << "Cancelled." << std::endl;
        }

      } else if (input == "z" || input == "Z") {
        // ============================================================
        // 写入当前位置为零点 (0x64) - ALL motors
        // ============================================================
        std::cout << "[SEND] ALL: 0x64 (WriteCurPosAsZero)" << std::endl;
        std::cout << "NOTE: Requires 0x76 system reset to take effect!"
                  << std::endl;
        std::cout << "WARNING: This writes zero offset to ROM for ALL motors."
                  << std::endl;
        std::cout << "Confirm? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          arm.set_callback_mode_all(CallbackMode::IGNORE);
          arm.get_arm().set_zero_all();
          // arm.recv_all(1000); <-- REPLACED
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          std::cout << "[RECV] OK: Zero offset written to ROM (all)."
                    << std::endl;
        } else {
          std::cout << "Cancelled." << std::endl;
        }

      } else if (input == "6") {
        // ============================================================
        // 电机关闭 (0x80) - 协议 2.17 节
        // ============================================================
        std::cout << "[SEND] ALL: 0x80 (MotorOff)" << std::endl;
        arm.motor_off_all();
        // arm.recv_all(500); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "[RECV] OK: Motor off (all)." << std::endl;

      } else if (input == "7") {
        // ============================================================
        // 电机停止 (0x81) - 协议 2.18 节
        // ============================================================
        std::cout << "[SEND] ALL: 0x81 (MotorStop)" << std::endl;
        arm.motor_stop_all();
        // arm.recv_all(500); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "[RECV] OK: Motor stop (all)." << std::endl;

      } else if (input == "8") {
        // ============================================================
        // 读取 PID 参数 (0x30) - 协议 2.1 节
        // ============================================================
        std::cout << "PID idx (1=Curr_Kp, 4=Vel_Kp, 7=Pos_Kp): ";
        std::getline(std::cin, input);
        int idx = std::stoi(input);
        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0x30 (ReadPid), index=" << idx << std::endl;
        arm.set_callback_mode_all(CallbackMode::PARAM);
        // 只读当前电机（避免两电机都回包时混淆）
        arm.get_arm().read_pid_one(motor_idx, static_cast<ReadParamIndex>(idx));
        // arm.recv_all(1000); <-- REPLACED
        std::this_thread::sleep_for(
            std::chrono::milliseconds(50)); // Needs more time?
        std::cout << "[RECV] " << hex_u32(cur_m.get_query_recv_can_id())
                  << ": Value = "
                  << arm.get_arm().get_motor(motor_idx).get_param(
                         static_cast<ReadParamIndex>(idx))
                  << std::endl;

      } else if (input == "9") {
        // ============================================================
        // MIT Single Shot Test - 验证电机是否需要持续发送 MIT 命令
        // ============================================================
        std::cout << "Kp: ";
        std::getline(std::cin, input);
        double kp = std::stod(input);
        std::cout << "Kd: ";
        std::getline(std::cin, input);
        double kd = std::stod(input);
        std::cout << "Target (deg): ";
        std::getline(std::cin, input);
        double target_deg = std::stod(input);
        double target_rad = target_deg * M_PI / 180.0;
        std::cout << "Target Vel (rad/s): ";
        std::getline(std::cin, input);
        double target_vel = std::stod(input);
        std::cout << "Torque FF (Nm): ";
        std::getline(std::cin, input);
        double torque_ff = std::stod(input);

        MITParam params = {kp, kd, target_rad, target_vel, torque_ff};
        arm.set_callback_mode_all(CallbackMode::STATE);

        // 详细打印发送的 MIT 参数
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[SEND] " << hex_u32(cur_m.get_mit_send_can_id())
                  << ": MIT, p_des=" << target_rad << " rad (" << target_deg
                  << " deg), v_des=" << target_vel << " rad/s, "
                  << "kp=" << kp << ", kd=" << kd << ", t_ff=" << torque_ff
                  << " Nm" << std::endl;

        arm.get_arm().mit_control_one(motor_idx, params);
        // arm.recv_all(1000); // Wait 1ms for response <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::cout << "[RECV] " << hex_u32(cur_m.get_mit_recv_can_id()) << ": ";
        print_mit_state(
            arm.get_arm().get_motor(motor_idx)); // MIT 响应不含 temp

        std::cout << "\nNow STOP MIT, query via 0x9C for 5s..." << std::endl;
        std::cout << "(If motor stops, MIT needs continuous sending)"
                  << std::endl;
        for (int i = 0; i < 100 && g_running; ++i) {
          arm.read_state_one(motor_idx); // Query via 0x9C
          // arm.recv_all(500); <-- REPLACED
          std::this_thread::sleep_for(
              std::chrono::milliseconds(50)); // quicker check
          std::cout << "[RECV] " << hex_u32(cur_m.get_query_recv_can_id())
                    << " [" << (i + 1) * 50 << "ms] ";
          print_state(
              arm.get_arm().get_motor(motor_idx)); // State2 响应包含 temp
        }
        std::cout << "Test done." << std::endl;

      } else if (input == "0") {
        // ============================================================
        // 系统复位 (0x76) - 协议 2.26 节
        // ============================================================
        std::cout << "[SEND] " << hex_u32(cur_m.get_query_send_can_id())
                  << ": 0x76 (SystemReset)" << std::endl;
        std::cout << "WARNING: Motor will restart! Confirm? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          arm.get_arm().system_reset_one(motor_idx);
          std::cout << "Reset sent. Motor restarting (no reply)..."
                    << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(2000));
          std::cout << "Motor should be ready now." << std::endl;
        } else {
          std::cout << "Cancelled." << std::endl;
        }
      } else if (input == "r" || input == "R") {
        // ============================================================
        // 系统复位 (0x76) - ALL motors
        // ============================================================
        std::cout << "[SEND] ALL: 0x76 (SystemReset)" << std::endl;
        std::cout << "WARNING: ALL motors will restart! Confirm? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          arm.set_callback_mode_all(CallbackMode::IGNORE);
          for (size_t i = 0; i < motors.size(); ++i) {
            arm.get_arm().system_reset_one(static_cast<int>(i));
            // 保守一点：避免总线瞬间拥塞
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          std::cout << "Reset sent. Motors restarting (no reply)..."
                    << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(2000));
          std::cout << "Motors should be ready now. (You can query via 0x9C)"
                    << std::endl;
        } else {
          std::cout << "Cancelled." << std::endl;
        }
      } else if (input == "m" || input == "M") {
        // ============================================================
        // MIT 运动模式控制 ALL (0x400+ID) - 协议 5 章
        // 固定参数: Kp=10, Kd=1, 所有电机相同角度循环发送, 10ms间隔
        // ============================================================
        const double kp = 10.0;
        const double kd = 1.0;
        // 角度序列，所有电机发送相同角度，循环切换
        const std::vector<double> angle_sequence_deg = {0, 10, 20, 30, 40, 90};
        size_t angle_idx = 0;

        arm.set_callback_mode_all(CallbackMode::STATE);

        std::cout << "[SEND] ALL: MIT mode, kp=" << kp << ", kd=" << kd
                  << std::endl;
        std::cout << "Angle sequence (deg): ";
        for (const auto &ang : angle_sequence_deg) {
          std::cout << ang << " ";
        }
        std::cout << std::endl;
        std::cout << "MIT running continuously (10ms interval), press Ctrl+C "
                     "to stop..."
                  << std::endl;

        int loop_count = 0;
        while (g_running) {
          double current_angle_deg = angle_sequence_deg[angle_idx];
          double target_rad = current_angle_deg * M_PI / 180.0;
          MITParam params = {kp, kd, target_rad, 0.0, 0.0};

          // 所有电机发送相同角度
          for (size_t m_idx = 0; m_idx < motors.size(); ++m_idx) {
            arm.get_arm().mit_control_one(static_cast<int>(m_idx), params);
          }
          // arm.recv_all(1000); // 1ms timeout <-- REPLACED

          if (loop_count % 100 == 0) { // 每 1 秒打印一次 (100 * 10ms)
            std::cout << "[" << loop_count * 10
                      << "ms] target=" << current_angle_deg << "deg | ";
            for (size_t m_idx = 0; m_idx < motors.size(); ++m_idx) {
              const auto &m = arm.get_arm().get_motor(static_cast<int>(m_idx));
              std::cout << "id" << m.get_device_id() << ":" << std::fixed
                        << std::setprecision(1)
                        << m.get_position() * 180.0 / M_PI << " ";
            }
            std::cout << std::endl;
          }

          ++loop_count;
          // 切换到下一个角度
          angle_idx = (angle_idx + 1) % angle_sequence_deg.size();
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // Release all motors
        for (size_t m_idx = 0; m_idx < motors.size(); ++m_idx) {
          arm.get_arm().mit_control_one(static_cast<int>(m_idx),
                                        {0, 0, 0, 0, 0});
        }
        // arm.recv_all(50); <-- REPLACED
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "Released." << std::endl;
        g_running = true; // 重置标志，允许继续使用菜单

      } else if (input == "q" || input == "Q") {
        g_running = false;
      }
    }

    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }

    arm.motor_off_all();
    std::cout << "Exit." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
