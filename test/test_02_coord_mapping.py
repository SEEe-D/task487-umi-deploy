"""测试2: 坐标系映射验证

检查 tcp_pose_to_state_20d 和 action_to_tcp_pose 的往返一致性:
  tcp_pose → state → (假设 delta=0) → action_to_tcp_pose → 应该还原 tcp_pose
"""
import numpy as np
from scipy.spatial.transform import Rotation

# 坐标映射矩阵
R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)
R_R2M = R_M2R.T

def rotvec_to_rot6d(rotvec):
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return R[:, :2].T.flatten()

def rot6d_to_rotvec(rot6d):
    r1 = rot6d[:3].copy()
    r2 = rot6d[3:6].copy()
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    return Rotation.from_matrix(R).as_rotvec()

def tcp_pose_to_state_20d(tcp_pose, gripper_pos):
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = tcp_pose[:3]
    R_robot = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
    R_model = R_robot @ R_M2R
    state[3:9] = R_model[:, :2].T.flatten()
    state[9] = gripper_pos
    return state

def action_to_tcp_pose(action):
    pos = action[:3]
    rotvec_model = rot6d_to_rotvec(action[3:9])
    R_model = Rotation.from_rotvec(rotvec_model).as_matrix()
    R_robot = R_model @ R_R2M
    rotvec_robot = Rotation.from_matrix(R_robot).as_rotvec()
    return np.concatenate([pos, rotvec_robot])

print("=" * 70)
print("测试: 坐标映射往返一致性")
print("=" * 70)

# 测试几组 TCP pose
test_poses = [
    ("典型姿态1", [-0.074, 0.441, 0.282, -1.819, -1.877, 0.383]),
    ("典型姿态2", [-0.036, 0.083, 0.542, -0.228, 0.920, 1.796]),
    ("典型姿态3", [0.100, 0.200, 0.400, 1.000, -1.500, 0.500]),
]

for name, tcp in test_poses:
    tcp = np.array(tcp, dtype=np.float64)
    # tcp → state
    state = tcp_pose_to_state_20d(tcp, 0.5)
    # state → tcp (假装 action = state[:10], 即 delta=0 的情况)
    recovered_tcp = action_to_tcp_pose(state[:10])

    pos_err = np.linalg.norm(tcp[:3] - recovered_tcp[:3])
    # 旋转误差用角度
    R_orig = Rotation.from_rotvec(tcp[3:6]).as_matrix()
    R_recv = Rotation.from_rotvec(recovered_tcp[3:6]).as_matrix()
    rot_err = np.arccos(np.clip((np.trace(R_orig.T @ R_recv) - 1) / 2, -1, 1))

    print(f"\n--- {name} ---")
    print(f"  原始 TCP:   {[round(x,4) for x in tcp]}")
    print(f"  State[:10]: {[round(x,4) for x in state[:10]]}")
    print(f"  还原 TCP:   {[round(x,4) for x in recovered_tcp]}")
    print(f"  位置误差: {pos_err:.6f} m")
    print(f"  旋转误差: {np.degrees(rot_err):.4f} deg")
    if pos_err < 1e-4 and rot_err < 0.01:
        print(f"  ✓ 往返一致")
    else:
        print(f"  ✗ 映射有问题!")

print("\n" + "=" * 70)
print("如果所有往返一致, 说明 R_M2R/R_R2M 旋转映射正确")
print("位置是直通(不做映射), 也应该完全一致")
