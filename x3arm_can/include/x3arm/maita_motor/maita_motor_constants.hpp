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

#include <array>
#include <cstddef>
#include <cstdint>

namespace x3arm::maita_motor {

// CAN ID offset constants for Maita motor protocol
// MIT control: send 0x400 + device_id, recv 0x500 + device_id
// Query: send 0x140 + device_id, recv 0x240 + device_id
constexpr uint32_t MIT_SEND_OFFSET = 0x400;
constexpr uint32_t MIT_RECV_OFFSET = 0x500;
constexpr uint32_t QUERY_SEND_OFFSET = 0x140;
constexpr uint32_t QUERY_RECV_OFFSET = 0x240;

enum class MotorType : uint8_t {
  MTX410 = 0,
  MTX436 = 1,
  MTX660 = 2,
  MTRH32 = 3,
  COUNT = 5,
};

enum class ReadParamIndex : uint8_t {
  kCurrentKp = 0x01,
  kCurrentKi = 0x02,
  kVelocityKp = 0x04,
  kVelocityKi = 0x05,
  kPositionKp = 0x07,
  kPositionKi = 0x08,
  kPositionKd = 0x09,
};

enum class MaitaCmd : uint8_t {
  /* ---------------- 参数区读 / 写（0x20, 0x30–0x43） ---------------- */
  kFuncCtrl = 0x20,    // 功能控制 & 主动上报开关（携带索引）
  kReadPid = 0x30,     // 读参数表
  kWritePidRam = 0x31, // 写参数到 RAM（掉电失效）
  kWritePidRom = 0x32, // 写参数到 ROM（掉电保存）
  kReadPidAcc = 0x42,  // 读加速度
  kWritePidAcc = 0x43, // 写加速度

  /* ---------------- 编码器 / 位置查询（0x60–0x64, 0x90–0x94） -------- */
  kReadPosMultiTurn = 0x60,    // 多圈位置（扣零偏）
  kReadPosMultiTurnRaw = 0x61, // 多圈原始位置
  kReadPosOffset = 0x62,       // 读多圈零偏
  kWritePosOffset = 0x63,      // 写指定值为零偏
  kWriteCurPosAsZero = 0x64,   // 当前位置写零偏
  kReadPosSingleTurn = 0x90,   // 单圈编码器值
  kReadAngleMultiTurn = 0x92,  // 多圈角度（°）   (旧固件常用)
  kReadAngleSingleTurn = 0x94, // 单圈角度（°）

  /* ---------------- 状态 / 错误查询（0x9A–0x9D） -------------------- */
  kReadState1 = 0x9A, // 状态 1 & 清错误
  kReadState2 = 0x9C, // 电流、电压、温度
  kReadState3 = 0x9D, // 目标 & 实际力矩/速度/位置

  /* ---------------- 运行控制（0x70, 0x76–0x78, 0x79） --------------- */
  kReadRunMode = 0x70,  // 系统运行模式
  kSystemReset = 0x76,  // 软复位
  kBrakeRelease = 0x77, // 电磁抱闸释放
  kBrakeLock = 0x78,    // 电磁抱闸锁定

  /* ---------------- 使能 / 停止 / 关机（0x80–0x88） ------------------ */
  kMotorOff = 0x80,  // 断电关闭
  kMotorStop = 0x81, // 立即停止（保持使能）

  /* ---------------- 闭环控制（0xA1–0xA9） --------------------------- */
  kTorqueClosedLoopMit = 0xA1, // MIT 力矩环
  kSpeedClosedLoop = 0xA2,     // 速度环（rpm*10）
  kPosClosedLoopAbs = 0xA4,    // 绝对位置环
  kPosClosedLoopSingle = 0xA6, // 单圈位置环
  kPosClosedLoopInc = 0xA8,    // 增量位置（另一格式）
  kForcePosClosedLoop = 0xA9,  // 力控位置环

  /* ---------------- 信息 / 版本 / 保护（0xB1–0xB6） ---------------- */
  kReadRunTime = 0xB1,     // 系统运行时间
  kReadSwDate = 0xB2,      // 软件版本日期
  kSetCommTimeout = 0xB3,  // 通讯中断保护时间
  kSetBaudRate = 0xB4,     // 通讯波特率
  kReadModel = 0xB5,       // 电机型号
  kActiveReportCfg = 0xB6, // 主动上报参数配置

  COUNT = 35, // 枚举项总数
};

// Limit parameters structure for different motor types
struct LimitParam {
  double pMax; // Position limit (rad)
  double vMax; // Velocity limit (rad/s)
  double tMax; // Torque limit (Nm)
};
// Limit parameters for each motor type [pMax, vMax, tMax]
inline constexpr std::array<LimitParam,
                            static_cast<std::size_t>(MotorType::COUNT)>
    MOTOR_LIMIT_PARAMS = {{
        // According to Maita protocol: pMax=±12.5rad, vMax=±45rad/s,
        // tMax=±24N-m
        {12.5, 45, 24}, // MTX410
        {12.5, 45, 24}, // MTX436
        {12.5, 45, 24}, // MTX660
        {12.5, 45, 24}, // MTRH32
        {12.5, 45, 24}, // COUNT
    }};

// Torque constant Kt (N-m/A) for converting current to torque
// torque = iq * Kt
// TODO: Update with actual Kt values from motor specifications
inline constexpr std::array<double, static_cast<std::size_t>(MotorType::COUNT)>
    MOTOR_KT = {{
        0.4, // MTX410 (X4-10) Kt (N-m/A) - TODO: update with actual value
        1.2, // MTX436 (X4-36) Kt (N-m/A) - TODO: update with actual value
        1.5, // MTX660 (X6-60) Kt (N-m/A) - TODO: update with actual value
        5.7, // MTRH32 (RH-32) Kt (N-m/A) - TODO: update with actual value
        1.0, // COUNT
    }};

} // namespace x3arm::maita_motor
