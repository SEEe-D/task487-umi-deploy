// Copyright 2026 SimpleAI
//
// 独立夹爪 CAN 节点 (mink 栈专用 / 不走 ros2_control)
// ---------------------------------------------------------------------------
// 背景: mink 主线真机走 marvin_bridge(纯 Python + 天机 SDK + fusion), 跳过
//       ros2_control / controller_manager。老栈夹爪是 ros2_control 架构
//       (simpleai_x3_hardware + adaptive_gripper_controller)。本节点把老栈
//       两段代码 1:1 搬过来, 做成一个独立 rclcpp 节点, 直接驱动 CAN1:
//         * CAN 驱动      = x3arm_can (Livelybot MIT 电机, SocketCAN)
//         * 控制律        = adaptive_gripper_controller (direct position + 开合力反馈)
//         * ROS↔电机变换  = simpleai_x3_hardware (×20 减速比 + zero_offset)
//
// 接线 (THOR 139 verified): CAN1, 左手 device_id=0x0A(10), 右手 device_id=0x09(9)。
// 嵌入式同事只负责把 can1 调通; 本节点负责控制。
//
// 命令来源 (VR teleop xr_target_node 已在发布, 无需改 VR 侧):
//   /Joint69/position_command  std_msgs/Float64  左手目标位置 (rad, ROS 关节系)
//   /Joint79/position_command  std_msgs/Float64  右手目标位置 (rad, ROS 关节系)
//   /gripper_command           arms_ros2_control_msgs/Gripper (arm_id=1左/2右, target=1开/0合)
//
// 反馈: /gripper_joint_states sensor_msgs/JointState (Joint69/79 + mimic Joint610/710)
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <arms_ros2_control_msgs/msg/gripper.hpp>

#include <x3arm/can/socket/x3arm.hpp>
#include <x3arm/livelybot_motor/livelybot_motor.hpp>
#include <x3arm/livelybot_motor/livelybot_motor_control.hpp>
#include <x3arm/livelybot_motor/livelybot_motor_constants.hpp>

#include "stall_release.hpp"

using x3arm::livelybot_motor::MITParam;
using x3arm::livelybot_motor::FaultCode;

namespace
{
// Livelybot 夹爪减速比 (copied from simpleai_x3_hardware.cpp:1455)。
// motor_rad = (ros_rad - zero_offset) * GEAR;  ros_rad = motor_rad / GEAR + zero_offset。
constexpr double kGripperGearRatio = 20.0;
}  // namespace

class GripperCanNode : public rclcpp::Node
{
public:
  GripperCanNode()
  : rclcpp::Node("gripper_can_node")
  {
    // ── 参数 ──
    can_interface_      = declare_parameter<std::string>("can_interface", "can1");
    use_can_fd_         = declare_parameter<bool>("can_fd", false);
    enable_left_        = declare_parameter<bool>("enable_left", true);
    enable_right_       = declare_parameter<bool>("enable_right", true);
    left_.device_id     = static_cast<uint32_t>(declare_parameter<int>("left_device_id", 0x0A));   // 10
    right_.device_id    = static_cast<uint32_t>(declare_parameter<int>("right_device_id", 0x09));  // 9
    left_.joint         = declare_parameter<std::string>("left_joint", "Joint69");
    right_.joint        = declare_parameter<std::string>("right_joint", "Joint79");
    left_.mimic_joint   = declare_parameter<std::string>("left_mimic_joint", "Joint610");
    right_.mimic_joint  = declare_parameter<std::string>("right_mimic_joint", "Joint710");
    kp_                 = std::max(0.0, declare_parameter<double>("gripper_kp", 15.0));
    kd_                 = std::max(0.0, declare_parameter<double>("gripper_kd", 1.0));
    left_.zero_offset   = declare_parameter<double>("left_zero_offset_rad", 0.0);
    right_.zero_offset  = declare_parameter<double>("right_zero_offset_rad", 0.0);
    // 夹爪关节限位: 0deg 全闭, -35deg 全开 (电机侧约 -700deg, ×20)
    lower_limit_        = declare_parameter<double>("lower_limit", -0.61086524);
    upper_limit_        = declare_parameter<double>("upper_limit", 0.0);
    if (lower_limit_ > upper_limit_) std::swap(lower_limit_, upper_limit_);
    // 开/合位置 (VR xr_target_node: open=-35deg, closed=0deg)
    open_rad_           = declare_parameter<double>("open_rad", -0.61086524);
    closed_rad_         = declare_parameter<double>("closed_rad", 0.0);
    // 力反馈 (adaptive_gripper_controller: 合爪触发力阈值后按 ratio 减速接近)
    use_force_feedback_ = declare_parameter<bool>("use_force_feedback", true);
    force_threshold_    = declare_parameter<double>("force_threshold", 1.5);
    force_feedback_ratio_ = declare_parameter<double>("force_feedback_ratio", 0.1);
    recv_timeout_us_    = declare_parameter<int>("recv_timeout_us", 1000);
    const double rate_hz = std::max(1.0, declare_parameter<double>("control_rate_hz", 100.0));
    publish_states_     = declare_parameter<bool>("publish_joint_states", true);
    auto_zero_on_startup_ = declare_parameter<bool>("auto_zero_on_startup", true);
    auto_zero_verbose_ = declare_parameter<bool>("auto_zero_verbose", false);

    // ── 堵转/过载保护 (遥操时夹爪夹到物体后限矩 hold; 对齐 wbc overload 参考
    //    start_mink_ec_teleop.sh + simpleai_x3_hardware::command_gripper) ──
    //   开启时遥操夹爪改走"位置-速度-最大力矩"模式 + 每周期堵转检测, 撞到物体
    //   后在当前位置基础上再向闭合方向夹 release 角并锁存 hold, 直到收到足够开
    //   的命令 (超过 hold 再开 rearm_delta) 才解除。关闭则退回原 MIT 位置控制。
    gripper_stall_guard_enable_ =
        declare_parameter<bool>("gripper_stall_guard_enable", true);
    // 夹爪关节侧行程幅度 (rad), 用于 release 上限 clamp。zkpan 系: |lower|≈0.6109。
    const double gripper_joint_span_rad = std::abs(upper_limit_ - lower_limit_);
    gripper_stall_release_rad_ = std::clamp(
        declare_parameter<double>("gripper_stall_release_rad", 0.03490658503988659),
        0.0, gripper_joint_span_rad);
    gripper_stall_max_torque_nm_ =
        std::max(0.0, declare_parameter<double>("gripper_stall_max_torque_nm", 1.05));
    gripper_release_max_torque_nm_ =
        std::max(0.0, declare_parameter<double>("gripper_release_max_torque_nm",
                                                gripper_stall_max_torque_nm_));
    // 电机侧 rad/s
    gripper_stall_velocity_rad_s_ =
        std::max(0.01, declare_parameter<double>("gripper_stall_velocity_rad_s", 1.0));
    gripper_stall_rearm_delta_rad_ =
        std::max(0.0, declare_parameter<double>("gripper_stall_rearm_delta_rad",
                                                0.03490658503988659));

    // ── 初始化两个夹爪 (同 can1 两个独立 X3Arm, 各自 socket 按 CAN ID 过滤) ──
    if (enable_left_)  init_gripper(left_,  "left");
    if (enable_right_) init_gripper(right_, "right");

    start_periodic_report(left_);
    start_periodic_report(right_);
    if (auto_zero_on_startup_) {
      auto_zero_gripper(left_, "left");
      auto_zero_gripper(right_, "right");
    }

    // ── 订阅 (VR 已发的 direct position) ──
    left_pos_sub_ = create_subscription<std_msgs::msg::Float64>(
      "/" + left_.joint + "/position_command", 10,
      [this](const std_msgs::msg::Float64::SharedPtr m) { on_position_cmd(left_, m); });
    right_pos_sub_ = create_subscription<std_msgs::msg::Float64>(
      "/" + right_.joint + "/position_command", 10,
      [this](const std_msgs::msg::Float64::SharedPtr m) { on_position_cmd(right_, m); });

    // ── 订阅 (开合 + 力反馈) ──
    gripper_cmd_sub_ = create_subscription<arms_ros2_control_msgs::msg::Gripper>(
      "/gripper_command", 10,
      [this](const arms_ros2_control_msgs::msg::Gripper::SharedPtr m) { on_gripper_cmd(m); });

    if (publish_states_) {
      js_pub_ = create_publisher<sensor_msgs::msg::JointState>("/gripper_joint_states", 10);
    }

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz),
      std::bind(&GripperCanNode::on_timer, this));

    RCLCPP_INFO(get_logger(),
      "gripper_can_node up: can=%s fd=%d left(%s id=0x%02X en=%d) right(%s id=0x%02X en=%d) "
      "kp=%.1f kd=%.1f limits=[%.3f,%.3f] open=%.3f closed=%.3f rate=%.0fHz",
      can_interface_.c_str(), use_can_fd_,
      left_.joint.c_str(), left_.device_id, enable_left_,
      right_.joint.c_str(), right_.device_id, enable_right_,
      kp_, kd_, lower_limit_, upper_limit_, open_rad_, closed_rad_, rate_hz);

    RCLCPP_INFO(get_logger(),
      "gripper stall-guard: enable=%d release=%.4frad max_tqe=%.2fNm release_tqe=%.2fNm vel=%.2frad/s rearm=%.4frad "
      "(开启时遥操夹爪走 pos-vel-maxtqe 限矩模式 + 每周期堵转检测)",
      gripper_stall_guard_enable_, gripper_stall_release_rad_,
      gripper_stall_max_torque_nm_, gripper_release_max_torque_nm_,
      gripper_stall_velocity_rad_s_,
      gripper_stall_rearm_delta_rad_);
  }

  ~GripperCanNode() override
  {
    stop_gripper(left_);
    stop_gripper(right_);
  }

private:
  struct Gripper
  {
    uint32_t device_id = 0;
    std::string joint;
    std::string mimic_joint;
    double zero_offset = 0.0;
    std::unique_ptr<x3arm::can::socket::X3Arm> arm;
    double target_rad = 0.0;     // ROS 关节系目标 (rad)
    bool direct_mode = true;     // true=direct position; false=开合(力反馈)
    bool closing = false;        // 当前开合命令是否为合爪
    bool force_triggered = false;
    bool ready = false;          // CAN init 成功
    long rx_count = 0;           // 收到的 position_command 计数 (诊断 VR 是否在发)
    double last_rx_rad = 0.0;    // 最近一条命令值
    // ── 堵转保护运行时状态 (per-gripper) ──
    bool stall_active = false;       // 已确认堵转, 锁存 hold 位置
    bool stall_tracking = false;     // 正在朝某闭合目标跟踪 (喂检测器)
    double stall_target_motor = 0.0; // 当前跟踪的电机侧目标 (rad)
    double stall_hold_ros = 0.0;     // 堵转锁存的 ROS 关节系 hold 位置 (rad)
  };

  void init_gripper(Gripper & g, const char * tag)
  {
    try {
      g.arm = std::make_unique<x3arm::can::socket::X3Arm>(can_interface_, use_can_fd_);
      g.arm->init_gripper_motor(g.device_id);
      g.arm->disable_loopback();
      g.arm->set_gripper_only_filter();
      // 读一次当前位置作为初始目标, 避免上电瞬间跳到 0
      try {
        g.arm->get_gripper().query_state();
        g.arm->recv_all(recv_timeout_us_);
        if (auto * m = g.arm->get_gripper().get_motor()) {
          g.target_rad = m->get_position() / kGripperGearRatio + g.zero_offset;
        }
      } catch (...) {}
      g.target_rad = std::clamp(g.target_rad, lower_limit_, upper_limit_);
      g.ready = true;
      RCLCPP_INFO(get_logger(), "[%s] gripper CAN init OK on %s id=0x%02X, init target=%.4f rad",
                  tag, can_interface_.c_str(), g.device_id, g.target_rad);
    } catch (const std::exception & e) {
      g.ready = false;
      RCLCPP_ERROR(get_logger(), "[%s] gripper CAN init FAILED (id=0x%02X on %s): %s",
                   tag, g.device_id, can_interface_.c_str(), e.what());
    }
  }

  void start_periodic_report(Gripper & g)
  {
    if (!g.ready || !g.arm) return;
    try {
      g.arm->get_gripper().start_periodic_report(
        x3arm::livelybot_motor::PERIODIC_REPORT_DEFAULT_MS);
      g.arm->recv_all(recv_timeout_us_);
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "[%s] start periodic report failed: %s",
                  g.joint.c_str(), e.what());
    } catch (...) {}
  }

  void auto_zero_gripper(Gripper & g, const char * tag)
  {
    if (!g.ready || !g.arm) return;
    RCLCPP_INFO(get_logger(), "[%s] 启动自动闭合标零...", tag);
    try {
      const bool ok = g.arm->get_gripper().auto_close_and_zero(
        [this, &g] { g.arm->recv_all(recv_timeout_us_); },
        [] { return rclcpp::ok(); },
        auto_zero_verbose_,
        x3arm::can::socket::CloseDriveMode::NATIVE_VELOCITY);
      g.arm->recv_all(recv_timeout_us_);
      if (ok) {
        g.target_rad = std::clamp(g.zero_offset, lower_limit_, upper_limit_);
        g.direct_mode = true;
        g.force_triggered = false;
        RCLCPP_INFO(get_logger(), "[%s] 自动闭合标零完成, target=%.4f rad",
                    tag, g.target_rad);
      } else {
        RCLCPP_WARN(get_logger(), "[%s] 自动闭合标零未完成, 保持当前目标 %.4f rad",
                    tag, g.target_rad);
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "[%s] 自动闭合标零异常: %s", tag, e.what());
    }
  }

  // direct position 模式 (VR /JointXX/position_command): 直接 clamp 到限位
  void on_position_cmd(Gripper & g, const std_msgs::msg::Float64::SharedPtr & m)
  {
    if (!m || !std::isfinite(m->data)) return;
    g.direct_mode = true;
    g.force_triggered = false;
    g.target_rad = std::clamp(m->data, lower_limit_, upper_limit_);
    g.rx_count++;
    g.last_rx_rad = m->data;
    if (g.rx_count == 1) {
      RCLCPP_INFO(get_logger(), "[%s] FIRST position_command received: %.4f rad -> target %.4f",
                  g.joint.c_str(), m->data, g.target_rad);
    }
  }

  // 开合模式 (/gripper_command): target=1 开, 0 合 (合爪走力反馈)
  void on_gripper_cmd(const arms_ros2_control_msgs::msg::Gripper::SharedPtr & m)
  {
    if (!m) return;
    Gripper * g = (m->arm_id == 1) ? &left_ : (m->arm_id == 2) ? &right_ : nullptr;
    if (!g) return;
    g->direct_mode = false;
    if (m->target == 1) {           // 开
      g->closing = false;
      g->force_triggered = false;
      g->target_rad = std::clamp(open_rad_, lower_limit_, upper_limit_);
    } else {                        // 合
      g->closing = true;
      g->target_rad = std::clamp(closed_rad_, lower_limit_, upper_limit_);
    }
  }

  void on_timer()
  {
    run_cycle(left_);
    run_cycle(right_);
    if (publish_states_) publish_joint_states();
  }

  void run_cycle(Gripper & g)
  {
    if (!g.ready || !g.arm) return;
    auto * motor = g.arm->get_gripper().get_motor();
    const double cur_ros = motor ? (motor->get_position() / kGripperGearRatio + g.zero_offset) : g.target_rad;
    const double cur_joint_torque = motor ? (motor->get_torque() * kGripperGearRatio) : 0.0;

    // 力反馈: 仅开合模式下、合爪、力超阈值时, 按 ratio 减速接近 (copied from
    // adaptive_gripper_controller::update). force_feedback_ratio=0 → 停在当前位置。
    if (!g.direct_mode && use_force_feedback_ && g.closing && !g.force_triggered &&
        std::abs(cur_joint_torque) > force_threshold_)
    {
      const double distance_to_target = g.target_rad - cur_ros;
      g.target_rad = cur_ros + distance_to_target * force_feedback_ratio_;
      g.force_triggered = true;
      RCLCPP_INFO(get_logger(),
        "[%s] force threshold %.2fNm hit (|tau|=%.2f), hold %.0f%% toward target -> %.4f",
        g.joint.c_str(), force_threshold_, std::abs(cur_joint_torque),
        force_feedback_ratio_ * 100.0, g.target_rad);
    }

    auto & gripper = g.arm->get_gripper();

    // ── 命令值 (ROS 关节系, 已 clamp 到限位) ──
    const double requested_ros = std::clamp(g.target_rad, lower_limit_, upper_limit_);

    // ── 堵转保护: 收到"足够开"的命令 → 解除锁存重新武装 ──
    //   zkpan 关节系: 闭=0(upper), 开=负(lower)。"开"= requested 更负。
    //   以实测位置为基准。hold 额外闭合 2deg，不能用它判断主动开爪，
    //   否则仍比实测位置更闭合的请求也会在下一周期解除锁存。
    if (gripper_stall_guard_enable_ && g.stall_active &&
        gripper_control::requests_stall_release(
            requested_ros, cur_ros, gripper_stall_rearm_delta_rad_))
    {
      g.stall_active = false;
      g.stall_tracking = false;
      gripper.reset_stall_detector();
      RCLCPP_INFO(get_logger(),
        "[%s] [GRIPPER-STALL] rearmed by measured open command: request=%.4f rad current=%.4f rad",
        g.joint.c_str(), requested_ros, cur_ros);
    }

    // 堵转锁存时保持 hold 位置, 否则跟随请求
    double effective_ros = (gripper_stall_guard_enable_ && g.stall_active)
        ? g.stall_hold_ros : requested_ros;
    double cmd_motor = (effective_ros - g.zero_offset) * kGripperGearRatio;
    if (!std::isfinite(cmd_motor)) cmd_motor = 0.0;

    if (!gripper_stall_guard_enable_ || !motor) {
      // ── 旧行为: MIT 位置控制 (堵转保护关闭 / 无电机时) ──
      try {
        gripper.mit_control(MITParam{kp_, kd_, cmd_motor, 0.0, 0.0});
        gripper.query_state();
        gripper.query_fault();
      } catch (...) {}
    } else {
      // ── 堵转保护: 位置-速度-最大力矩模式 + 每周期堵转检测 ──
      //   (对齐 wbc overload 参考 simpleai_x3_hardware::command_gripper)
      const double cur_ros_clamped = std::clamp(cur_ros, lower_limit_, upper_limit_);
      // closing = 命令比当前更闭 (zkpan: 更靠近 0, 即 requested 更大)
      const bool closing = requested_ros >
          cur_ros_clamped + std::max(1e-4, gripper_stall_rearm_delta_rad_ * 0.1);
      const double command_max_torque_nm =
          closing ? gripper_stall_max_torque_nm_ : gripper_release_max_torque_nm_;

      if (!g.stall_active && closing) {
        const double requested_motor = (requested_ros - g.zero_offset) * kGripperGearRatio;
        // 目标变化超过 0.02 rad 时重置检测器, 重新计窗口 (跟手时持续刷新)
        if (!g.stall_tracking ||
            std::fabs(requested_motor - g.stall_target_motor) > 0.02) {
          gripper.reset_stall_detector();
          g.stall_tracking = true;
          g.stall_target_motor = requested_motor;
        }
        if (g.stall_tracking && gripper.poll_stall(g.stall_target_motor)) {
          g.stall_active = true;
          g.stall_tracking = false;
          // hold = 当前位置再向闭合方向 (zkpan: +) 夹紧 release, clamp 到限位
          g.stall_hold_ros = std::clamp(cur_ros_clamped + gripper_stall_release_rad_,
                                        lower_limit_, upper_limit_);
          effective_ros = g.stall_hold_ros;
          cmd_motor = (effective_ros - g.zero_offset) * kGripperGearRatio;
          RCLCPP_WARN(get_logger(),
            "[%s] [GRIPPER-STALL] detected: current=%.4f rad, hold=%.4f rad, extra_close=%.4f rad",
            g.joint.c_str(), cur_ros_clamped, g.stall_hold_ros, gripper_stall_release_rad_);
        }
      } else if (!g.stall_active && !closing) {
        g.stall_tracking = false;
      }

      try {
        gripper.pos_vel_maxtqe_control(cmd_motor, gripper_stall_velocity_rad_s_,
                                       command_max_torque_nm);
        gripper.query_state();
        gripper.query_fault();
      } catch (...) {}
    }
    g.arm->recv_all(recv_timeout_us_);

    // 1Hz 自诊断: 收到命令数 / target / 下发电机值 / 电机实测位 + 力矩。
    // rx_count=0 → VR 没发命令(问题在命令源); rx_count>0 但 motor_pos 不变 → CAN/电机侧。
    {
      auto * m2 = g.arm->get_gripper().get_motor();
      const double mpos = m2 ? m2->get_position() : 0.0;
      const double mtau = m2 ? m2->get_torque() : 0.0;
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] rx=%ld last_rx=%.4f target=%.4f cmd_motor=%.3f stall=%d hold=%.4f | motor_pos=%.3f tau=%.3f",
        g.joint.c_str(), g.rx_count, g.last_rx_rad, g.target_rad, cmd_motor,
        g.stall_active ? 1 : 0, g.stall_hold_ros, mpos, mtau);
    }

    if (motor) {
      const FaultCode fault = motor->get_fault_code();
      if (fault != FaultCode::NORMAL) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "[%s] gripper fault code=%d (%s), sending stop()",
          g.joint.c_str(), static_cast<int>(fault),
          x3arm::livelybot_motor::fault_to_string(fault).c_str());
        try { g.arm->get_gripper().stop(); } catch (...) {}
      }
    }
  }

  void stop_gripper(Gripper & g)
  {
    if (!g.ready || !g.arm) return;
    try {
      g.arm->get_gripper().stop();
      g.arm->recv_all(recv_timeout_us_);
    } catch (...) {}
  }

  void publish_joint_states()
  {
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    auto add = [&](Gripper & g) {
      if (!g.ready || !g.arm) return;
      auto * m = g.arm->get_gripper().get_motor();
      if (!m) return;
      const double pos = m->get_position() / kGripperGearRatio + g.zero_offset;
      const double vel = m->get_velocity() / kGripperGearRatio;
      const double eff = m->get_torque() * kGripperGearRatio;
      js.name.push_back(g.joint);  js.position.push_back(pos);
      js.velocity.push_back(vel);  js.effort.push_back(eff);
      // mimic (仅可视化, = -master, copied from hardware read)
      if (!g.mimic_joint.empty()) {
        js.name.push_back(g.mimic_joint); js.position.push_back(-pos);
        js.velocity.push_back(-vel);      js.effort.push_back(0.0);
      }
    };
    add(left_);
    add(right_);
    if (!js.name.empty()) js_pub_->publish(js);
  }

  // ── 配置 ──
  std::string can_interface_;
  bool use_can_fd_ = false;
  bool enable_left_ = true, enable_right_ = true;
  double kp_ = 15.0, kd_ = 1.0;
  double lower_limit_ = -0.61086524, upper_limit_ = 0.0;
  double open_rad_ = -0.61086524, closed_rad_ = 0.0;
  bool use_force_feedback_ = true;
  double force_threshold_ = 1.5, force_feedback_ratio_ = 0.1;
  int recv_timeout_us_ = 1000;
  bool publish_states_ = true;
  bool auto_zero_on_startup_ = true;
  bool auto_zero_verbose_ = false;
  // ── 堵转/过载保护参数 (对齐 wbc overload 参考) ──
  bool gripper_stall_guard_enable_ = true;
  double gripper_stall_release_rad_ = 0.03490658503988659;   // 2deg, 关节侧
  double gripper_stall_max_torque_nm_ = 1.05;
  double gripper_release_max_torque_nm_ = 1.05;
  double gripper_stall_velocity_rad_s_ = 1.0;                 // 电机侧 rad/s
  double gripper_stall_rearm_delta_rad_ = 0.03490658503988659; // 2deg, 关节侧

  Gripper left_, right_;

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr left_pos_sub_, right_pos_sub_;
  rclcpp::Subscription<arms_ros2_control_msgs::msg::Gripper>::SharedPtr gripper_cmd_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GripperCanNode>());
  rclcpp::shutdown();
  return 0;
}
