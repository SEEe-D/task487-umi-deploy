"""Read the joint-speed contract from the CAN node's installed configuration."""
from dataclasses import dataclass
import math
from pathlib import Path

import yaml


DEFAULT_GRIPPER_CONFIG = Path(
    "/home/simpleai/Code/mjm/eval_mink/ros2_ws/install/gripper_can_node/"
    "share/gripper_can_node/config/gripper_can.yaml")
# Matches kGripperGearRatio in the shared Marvin gripper CAN driver.
GRIPPER_GEAR_RATIO = 20.0


@dataclass(frozen=True)
class GripperDynamics:
    config_path: Path
    motor_speed_rad_s: float

    @property
    def joint_speed_deg_s(self) -> float:
        return math.degrees(self.motor_speed_rad_s / GRIPPER_GEAR_RATIO)


def load_gripper_dynamics(config_path: Path) -> GripperDynamics:
    path = Path(config_path).expanduser().resolve()
    try:
        params = yaml.safe_load(path.read_text())["/gripper_can_node"]["ros__parameters"]
        if params.get("gripper_stall_guard_enable") is not True:
            raise ValueError("a verified pos/vel/torque CAN speed requires stall guard enabled")
        speed = float(params["gripper_stall_velocity_rad_s"])
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("motor speed must be finite and positive")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid gripper CAN configuration {path}: {exc}") from exc
    return GripperDynamics(path, speed)
