# Marvin gripper backend source (2026-09-05)

This directory archives the current `ros2_ws/src/gripper_can_node` package from
`/home/simpleai/Code/mjm/eval_mink`. It complements the client code in this repository;
it is not a complete standalone Marvin/Mink workspace.

The stall-release fix compares an opening request with the measured gripper
position, not the additional-close hold target. It preserves the existing opening
deadband, torque limits and configured motor speed. The client HOME preparation
opens before closing to clear a previous latch.

Build this package in the existing ROS2 workspace with its `x3arm_can` dependencies.
The existing site's driver was rebuilt and installed on 2026-09-05; a Git checkout
does not update or restart that installation. See the [installation report](../reports/gripper-fix-20260905/REPORT.md).

Offline regression (no CAN or robot commands):

```bash
g++ -std=c++17 -Wall -Wextra -pedantic \
  -I backend/ros2_ws/src/gripper_can_node/src \
  backend/ros2_ws/src/gripper_can_node/test/test_stall_release.cpp \
  -o /tmp/task487-stall-release-test
/tmp/task487-stall-release-test
```

Publication validation: 8 cases passed; replay of the 327 previously recorded
closing requests also passed. Raw runtime records remain on the development machine.
The full daily source archive and internal project status are in the private
`SEEe-D/task487-umi-train` repository (`DAILY_CHANGES_20260905.md`).
