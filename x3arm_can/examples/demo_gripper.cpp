// Copyright 2025 Enactic, Inc.
// Interactive Demo - Livelybot Gripper Test
//
// ============================================================================
// Livelybot Motor Protocol - MIT控制模式
// ============================================================================
//
// CAN ID 格式 (扩展帧):
//   MIT控制发送: 0x10000 | device_id
//   查询/配置发送: 0x8000 | device_id
//   电机响应: device_id << 8
//
// MIT数据格式 (8字节):
//   pos(16bit) + vel(12bit) + tqe(12bit) + kp(12bit) + kd(12bit)
//
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <x3arm/can/socket/x3arm.hpp>
#include <x3arm/livelybot_motor/livelybot_motor_control.hpp>

using namespace x3arm::livelybot_motor;

std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

static std::string hex_u32(uint32_t v) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::uppercase << v << std::dec;
  return oss.str();
}

void print_gripper_state(const Motor *motor) {
  if (!motor) {
    std::cout << "电机未初始化" << std::endl;
    return;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "位置=" << motor->get_position() * 180.0 / M_PI << " deg"
            << ", 速度=" << motor->get_velocity() * 180.0 / M_PI << " dps"
            << ", 力矩=" << motor->get_torque() << " Nm" << std::endl;
}

void print_menu() {
  std::cout << "\n=== Livelybot 夹爪测试 ===" << std::endl;
  std::cout << "1. 查询状态 (Query)" << std::endl;
  std::cout << "2. MIT控制 (Control)" << std::endl;
  std::cout << "3. 停止电机" << std::endl;
  std::cout << "4. 设置零位" << std::endl;
  std::cout << "5. 保存到Flash" << std::endl;
  std::cout << "q. 退出" << std::endl;
  std::cout << "命令: ";
}

// ============================================================================
// Daemon mode: non-interactive, controlled via stdin/stdout line protocol.
//   SET <deg>   -> OK
//   GET          -> STATE <pos_deg> <vel_dps> <torque_nm>
//   STOP         -> OK
//   QUIT         -> OK  (then exit)
// Prints "READY" on stdout once initialisation is done.
// ============================================================================
int run_daemon_mode(const std::string &can_if, uint32_t device_id,
                    double kp, double kd, double target_vel_deg,
                    double torque, double loop_hz) {
  x3arm::can::socket::X3Arm arm(can_if, false);
  arm.init_gripper_motor(device_id);
  auto &gripper = arm.get_gripper();
  auto *motor = gripper.get_motor();
  if (!motor) {
    std::cerr << "ERR motor not initialized" << std::endl;
    return -1;
  }

  gripper.query_state();
  arm.recv_all(5000);
  std::atomic<double> target_deg{motor->get_position() * 180.0 / M_PI};
  std::atomic<double> pos_deg{target_deg.load()};
  std::atomic<double> vel_dps{0.0};
  std::atomic<double> tq_nm{0.0};
  std::atomic<bool> running{true};
  std::mutex io_mtx;

  const double vel_rad = target_vel_deg * M_PI / 180.0;
  std::thread ctrl([&]() {
    auto period = std::chrono::duration<double>(1.0 / std::max(loop_hz, 1.0));
    while (running.load() && g_running.load()) {
      const double cmd_deg = target_deg.load();
      MITParam param{kp, kd, cmd_deg * M_PI / 180.0, vel_rad, torque};
      gripper.mit_control(param);
      gripper.query_state();
      arm.recv_all(1000);

      pos_deg.store(motor->get_position() * 180.0 / M_PI);
      vel_dps.store(motor->get_velocity() * 180.0 / M_PI);
      tq_nm.store(motor->get_torque());
      std::this_thread::sleep_for(period);
    }
    gripper.stop();
  });

  std::cout << "READY" << std::endl;
  std::cout.flush();
  std::string line;
  while (running.load() && g_running.load() && std::getline(std::cin, line)) {
    if (line.rfind("SET ", 0) == 0) {
      try {
        double v = std::stod(line.substr(4));
        target_deg.store(v);
        std::lock_guard<std::mutex> lk(io_mtx);
        std::cout << "OK" << std::endl;
      } catch (...) {
        std::lock_guard<std::mutex> lk(io_mtx);
        std::cout << "ERR bad SET" << std::endl;
      }
    } else if (line == "GET") {
      std::lock_guard<std::mutex> lk(io_mtx);
      std::cout << std::fixed << std::setprecision(6) << "STATE "
                << pos_deg.load() << " "
                << vel_dps.load() << " "
                << tq_nm.load() << std::endl;
    } else if (line == "STOP") {
      gripper.stop();
      std::lock_guard<std::mutex> lk(io_mtx);
      std::cout << "OK" << std::endl;
    } else if (line == "QUIT") {
      running.store(false);
      std::lock_guard<std::mutex> lk(io_mtx);
      std::cout << "OK" << std::endl;
      break;
    } else if (line.empty()) {
      continue;
    } else {
      std::lock_guard<std::mutex> lk(io_mtx);
      std::cout << "ERR unknown command" << std::endl;
    }
    std::cout.flush();
  }

  running.store(false);
  if (ctrl.joinable()) ctrl.join();
  return 0;
}

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);

  try {
    std::string can_if = argc > 1 ? argv[1] : "can0";
    uint32_t device_id = argc > 2 ? std::stoul(argv[2]) : 8;

    // Parse optional flags
    bool daemon_mode = false;
    double kp = 8.0, kd = 0.3, target_vel_deg = 0.0, torque = 0.0,
           loop_hz = 100.0;
    for (int i = 3; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--daemon")
        daemon_mode = true;
      else if (arg == "--kp" && i + 1 < argc)
        kp = std::stod(argv[++i]);
      else if (arg == "--kd" && i + 1 < argc)
        kd = std::stod(argv[++i]);
      else if (arg == "--target-vel-deg" && i + 1 < argc)
        target_vel_deg = std::stod(argv[++i]);
      else if (arg == "--torque" && i + 1 < argc)
        torque = std::stod(argv[++i]);
      else if (arg == "--loop-hz" && i + 1 < argc)
        loop_hz = std::stod(argv[++i]);
    }

    if (daemon_mode)
      return run_daemon_mode(can_if, device_id, kp, kd, target_vel_deg, torque,
                             loop_hz);

    std::cout << "初始化 Livelybot 夹爪电机..." << std::endl;
    std::cout << "CAN接口: " << can_if << std::endl;
    std::cout << "电机ID: " << device_id << std::endl;

    x3arm::can::socket::X3Arm arm(can_if, false);
    arm.init_gripper_motor(device_id);

    auto &gripper = arm.get_gripper();
    auto *motor = gripper.get_motor();

    std::cout << "夹爪电机初始化完成!" << std::endl;
    std::cout << "  MIT发送CAN ID: " << hex_u32(motor->get_mit_send_can_id())
              << std::endl;
    std::cout << "  查询发送CAN ID: " << hex_u32(motor->get_query_send_can_id())
              << std::endl;
    std::cout << "  响应接收CAN ID: " << hex_u32(motor->get_mit_recv_can_id())
              << std::endl;

    std::string input;
    while (g_running) {
      print_menu();
      std::getline(std::cin, input);

      if (input == "1") {
        // 查询状态
        gripper.query_state();
        arm.recv_all(5000);
        std::cout << "当前状态: ";
        print_gripper_state(motor);

      } else if (input == "2") {
        // MIT控制
        std::cout << "Kp: ";
        std::getline(std::cin, input);
        double kp = std::stod(input);
        std::cout << "Kd: ";
        std::getline(std::cin, input);
        double kd = std::stod(input);
        std::cout << "目标位置 (deg): ";
        std::getline(std::cin, input);
        double target_deg = std::stod(input);
        double target_rad = target_deg * M_PI / 180.0;

        std::cout << "目标速度 (deg/s): ";
        std::getline(std::cin, input);
        double target_vel_deg = std::stod(input);
        double target_vel_rad = target_vel_deg * M_PI / 180.0;

        std::cout << "前馈力矩 (Nm): ";
        std::getline(std::cin, input);
        double torque = std::stod(input);

        MITParam param{kp, kd, target_rad, target_vel_rad, torque};

        std::cout << "MIT控制运行3秒..." << std::endl;
        for (int i = 0; i < 300 && g_running; ++i) {
          gripper.mit_control(param);
          gripper.query_state(); // 主动查询状态，确保recv_all能收到最新反馈

          // 查询故障 (每10ms查询一次)
          gripper.query_fault();

          arm.recv_all(1000);

          // 检查故障
          auto fault = motor->get_fault_code();
          if (fault != FaultCode::NORMAL) {
            std::string fault_str = fault_to_string(fault);
            std::cout << "\n[警告] 检测到故障: Code=" << static_cast<int>(fault)
                      << " (" << fault_str << ")" << std::endl;

            // 写入日志文件
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::ofstream log_file("gripper_fault_log.txt", std::ios::app);
            if (log_file.is_open()) {
              log_file << std::put_time(std::localtime(&time_t_now),
                                        "%Y-%m-%d %H:%M:%S")
                       << " - Fault Code: " << static_cast<int>(fault)
                       << ", Name: " << fault_str << std::endl;
            }

            std::cout << "执行Stop清除故障..." << std::endl;
            gripper.stop();
            // 延时一小段时间让清除生效
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }

          if (i % 30 == 0) {
            std::cout << "[" << i * 10 << "ms] ";
            print_gripper_state(motor);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        gripper.stop();
        std::cout << "完成." << std::endl;

      } else if (input == "3") {
        // 停止电机
        gripper.stop();
        std::cout << "电机已停止." << std::endl;

      } else if (input == "4") {
        // 设置零位
        std::cout << "确认设置当前位置为零位? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          gripper.set_zero();
          std::cout << "零位已设置 (RAM)." << std::endl;
          std::cout << "注意: 需要使用选项5保存到Flash并重新上电才能永久生效."
                    << std::endl;
        } else {
          std::cout << "已取消." << std::endl;
        }

      } else if (input == "5") {
        // 保存到Flash
        std::cout << "确认保存设置到Flash? (y/n): ";
        std::getline(std::cin, input);
        if (input == "y" || input == "Y") {
          gripper.save_flash();
          std::cout << "已保存到Flash. 建议重新上电." << std::endl;
        } else {
          std::cout << "已取消." << std::endl;
        }

      } else if (input == "q" || input == "Q") {
        g_running = false;
      }
    }

    gripper.stop();
    std::cout << "退出." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
