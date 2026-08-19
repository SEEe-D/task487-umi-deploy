# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)
project(x3arm_can VERSION 1.1.0)

find_package(ament_cmake REQUIRED)

# For clangd/IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Testing support
include(CTest)

include(GNUInstallDirs)

# Create the main library
add_library(
  x3arm_can
  src/x3arm/can/socket/arm_component.cpp
  src/x3arm/can/socket/gripper_component.cpp
  src/x3arm/can/socket/x3arm.cpp
  src/x3arm/canbus/can_device_collection.cpp
  src/x3arm/canbus/can_socket.cpp
  src/x3arm/maita_motor/maita_motor.cpp
  src/x3arm/maita_motor/maita_motor_control.cpp
  src/x3arm/maita_motor/maita_motor_device.cpp
  src/x3arm/maita_motor/maita_motor_device_collection.cpp
  src/x3arm/maita_motor/maita_motor_diagnostics.cpp
  src/x3arm/livelybot_motor/livelybot_motor.cpp
  src/x3arm/livelybot_motor/livelybot_motor_control.cpp
  src/x3arm/livelybot_motor/livelybot_motor_device.cpp)
set_target_properties(
  x3arm_can
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR})
set(USE_FILE_SET_HEADERS FALSE)
# Meson doesn't support FILE_SET TYPE HEADERS...
# if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
#   set(USE_FILE_SET_HEADERS TRUE)
# endif()
if(USE_FILE_SET_HEADERS)
  target_sources(
    x3arm_can
    PUBLIC FILE_SET
           HEADERS
           TYPE
           HEADERS
           BASE_DIRS
           include
           FILES
           include/x3arm/can/socket/arm_component.hpp
           include/x3arm/can/socket/gripper_component.hpp
           include/x3arm/can/socket/x3arm.hpp
           include/x3arm/canbus/can_device.hpp
           include/x3arm/canbus/can_device_collection.hpp
           include/x3arm/canbus/can_socket.hpp
           include/x3arm/maita_motor/maita_motor.hpp
           include/x3arm/maita_motor/maita_motor_constants.hpp
           include/x3arm/maita_motor/maita_motor_control.hpp
           include/x3arm/maita_motor/maita_motor_device.hpp
           include/x3arm/maita_motor/maita_motor_device_collection.hpp
           include/x3arm/maita_motor/maita_motor_diagnostics.hpp)
  install(
    TARGETS x3arm_can
    EXPORT x3arm_can_export
    FILE_SET HEADERS)
else()
  target_include_directories(
    x3arm_can PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>)
  install(TARGETS x3arm_can EXPORT x3arm_can_export)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# CMake package
set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/X3ArmCAN)
install(
  EXPORT x3arm_can_export
  DESTINATION ${INSTALL_CMAKE_DIR}
  NAMESPACE X3ArmCAN::
  FILE X3ArmCANTargets.cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/X3ArmCANConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/X3ArmCANConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/X3ArmCANVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/X3ArmCANConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/X3ArmCANVersion.cmake
        DESTINATION ${INSTALL_CMAKE_DIR})

# TODO: Enable after creating x3arm-can.pc.in
# pkg-config package
# if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
#   set(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
# else()
#   set(PKG_CONFIG_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
# endif()
# if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
#   set(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
# else()
#   set(PKG_CONFIG_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
# endif()
# configure_file(x3arm-can.pc.in ${CMAKE_CURRENT_BINARY_DIR}/x3arm-can.pc
#                @ONLY)
# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/x3arm-can.pc
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# TODO: Enable after creating setup/motor_check.cpp
# add_executable(x3arm-can-motor-check setup/motor_check.cpp)
# target_link_libraries(x3arm-can-motor-check x3arm_can)
# install(TARGETS x3arm-can-motor-check DESTINATION ${CMAKE_INSTALL_BINDIR})

# TODO: Enable after creating setup/x3arm_can_diagnosis.cpp
# add_executable(x3arm-can-diagnosis setup/x3arm_can_diagnosis.cpp)
# target_link_libraries(x3arm-can-diagnosis x3arm_can)
# install(TARGETS x3arm-can-diagnosis DESTINATION ${CMAKE_INSTALL_BINDIR})

# Add motor control example executable
add_executable(x3arm-can-demo examples/demo.cpp)
target_link_libraries(x3arm-can-demo x3arm_can)
install(TARGETS x3arm-can-demo DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(x3arm-can-demo-mtrh32 examples/demo_mtrh32.cpp)
target_link_libraries(x3arm-can-demo-mtrh32 x3arm_can)
install(TARGETS x3arm-can-demo-mtrh32 DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(x3arm-can-demo-gripper examples/demo_gripper.cpp)
target_link_libraries(x3arm-can-demo-gripper x3arm_can)
install(TARGETS x3arm-can-demo-gripper DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(x3arm-can-demo-logging examples/demo_logging.cpp)
target_link_libraries(x3arm-can-demo-logging x3arm_can)
install(TARGETS x3arm-can-demo-logging DESTINATION ${CMAKE_INSTALL_BINDIR})





# TODO: Enable after creating setup scripts
# Install scripts
# install(
#   PROGRAMS setup/x3arm-can-change-baudrate
#            setup/x3arm-can-configure-socketcan
#            setup/x3arm-can-configure-socketcan-4-arms
#            setup/x3arm-can-set-zero
#            setup/x3arm-can-zero-position-calibration
#   DESTINATION ${CMAKE_INSTALL_BINDIR})

# Add tests
if(BUILD_TESTING)
  # add_subdirectory(test)
endif()

ament_package()
