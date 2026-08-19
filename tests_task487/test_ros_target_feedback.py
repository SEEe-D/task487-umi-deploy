import struct

import numpy as np

from umi.real_world.ros_target_interpolation_controller import (
    FEEDBACK_FMT,
    _unpack_feedback,
)


def test_pose_and_joint_feedback_round_trip():
    pose6 = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], dtype=np.float64)
    q7 = np.array([1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7], dtype=np.float64)
    packet = struct.pack(FEEDBACK_FMT, *np.concatenate((pose6, q7)))

    decoded_pose6, decoded_q7 = _unpack_feedback(packet)

    np.testing.assert_array_equal(decoded_pose6, pose6)
    np.testing.assert_array_equal(decoded_q7, q7)


def test_old_pose_only_packet_is_rejected():
    old_packet = struct.pack("<6d", *np.arange(6, dtype=np.float64))

    assert _unpack_feedback(old_packet) is None


def test_nonfinite_feedback_is_rejected():
    values = np.arange(13, dtype=np.float64)
    values[9] = np.nan
    packet = struct.pack(FEEDBACK_FMT, *values)

    assert _unpack_feedback(packet) is None
