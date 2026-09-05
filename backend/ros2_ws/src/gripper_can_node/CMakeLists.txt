cmake_minimum_required(VERSION 3.16)
project(gripper_can_node)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(arms_ros2_control_msgs REQUIRED)
# x3arm_can 导出的 CMake 包名是 X3ArmCAN, 目标 X3ArmCAN::x3arm_can
find_package(X3ArmCAN REQUIRED)

add_executable(gripper_can_node src/gripper_can_node.cpp)
ament_target_dependencies(gripper_can_node
  rclcpp
  std_msgs
  sensor_msgs
  arms_ros2_control_msgs)
target_link_libraries(gripper_can_node X3ArmCAN::x3arm_can)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_stall_release test/test_stall_release.cpp)
  target_include_directories(test_stall_release PRIVATE src)
  add_test(NAME stall_release COMMAND test_stall_release)
endif()

install(TARGETS gripper_can_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()
