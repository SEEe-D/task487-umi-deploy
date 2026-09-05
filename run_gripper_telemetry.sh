#!/usr/bin/env bash
# Passive subscriber only. Do not source a backend launch or publish commands.
set -eo pipefail
task487_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task487_ros_env="${TASK487_ROS_ENV:-/home/simpleai/anaconda3/envs/ros_env}"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
export PATH="$task487_ros_env/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$task487_ros_env/lib"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
source "$task487_ros_env/setup.bash"
export PYTHONDONTWRITEBYTECODE=1
exec "$task487_ros_env/bin/python" -u "$task487_root/task487_runtime/gripper_telemetry.py" "$@"
