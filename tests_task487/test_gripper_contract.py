import numpy as np
import pytest

from umi.real_world.livelybot_gripper_controller import LivelybotGripperController
from umi.real_world.ros_gripper_controller import RosGripperController
from umi.real_world.umi_env import normalize_per_gripper_value


def test_gripper_degree_motor_mapping_is_linear_and_invertible():
    controller = LivelybotGripperController.__new__(LivelybotGripperController)
    controller.gripper_deg_open = 35.0
    controller.gripper_deg_closed = 0.0
    controller.motor_deg_open = -700.0
    controller.motor_deg_closed = 0.0
    for gripper_degrees in (0.0, 7.0, 17.5, 28.0, 35.0):
        motor_degrees = controller._gripper_deg_to_motor_deg(gripper_degrees)
        assert controller._motor_deg_to_gripper_deg(motor_degrees) == pytest.approx(gripper_degrees)


@pytest.mark.parametrize("open_rad", (-0.59341195, -0.41949338))
def test_ros_gripper_mapping_preserves_joint_angle_until_safe_endpoint(open_rad):
    controller = RosGripperController.__new__(RosGripperController)
    controller.deg_open = float(np.rad2deg(abs(open_rad)))
    controller.deg_closed = 0.0
    controller.open_rad = open_rad
    controller.closed_rad = 0.0

    for opening_deg in (0.0, 1.0, controller.deg_open / 2.0, controller.deg_open):
        assert controller._deg_to_rad(opening_deg) == pytest.approx(
            -np.deg2rad(opening_deg)
        )
        assert controller._rad_to_deg(controller._deg_to_rad(opening_deg)) == pytest.approx(
            opening_deg
        )

    # Dataset commands beyond the verified physical endpoint are clipped, not
    # compressed across the entire usable range.
    assert controller._deg_to_rad(35.0) == pytest.approx(open_rad)


def test_per_gripper_calibration_accepts_scalar_or_exact_side_values():
    assert normalize_per_gripper_value(-0.5, 2, "open") == [-0.5, -0.5]
    assert normalize_per_gripper_value(
        (-0.59341195, -0.41949338), 2, "open"
    ) == [-0.59341195, -0.41949338]
    assert normalize_per_gripper_value(
        (34.0, 24.0), 2, "open_degrees"
    ) == [34.0, 24.0]

    with pytest.raises(ValueError, match="exactly 2 values"):
        normalize_per_gripper_value((-0.5,), 2, "open")
