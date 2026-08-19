"""测试3: 验证 delta 修正后的目标是否合理

用真实机器人 state 推理, 对比:
  A) 原始方式: target = action_to_tcp_pose(model_output)
  B) delta修正: target = action_to_tcp_pose(state[:10] + model_output[:9])
检查哪个目标更合理 (应该在当前 TCP 附近)
"""
import numpy as np
import time
from scipy.spatial.transform import Rotation
from openpi_client import websocket_client_policy as ws

R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)
R_R2M = R_M2R.T

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

# 连接 policy
policy = ws.WebsocketClientPolicy(host="localhost", port=8000)
img = np.random.randint(256, size=(256, 256, 3), dtype=np.uint8)

# 模拟几组真实 TCP
test_tcps = [
    ("位置A", [-0.074, 0.441, 0.282, -1.819, -1.877, 0.383]),
    ("位置B", [0.080, 0.150, 0.450, -0.228, 0.920, 1.796]),
    ("位置C", [0.000, 0.300, 0.600, -1.500, -1.500, 0.500]),
]

print("=" * 70)
print("测试: delta 修正 vs 直接当 absolute")
print("=" * 70)

for name, tcp_list in test_tcps:
    tcp = np.array(tcp_list, dtype=np.float64)
    state = tcp_pose_to_state_20d(tcp, 0.0)

    obs = {"cam_right": img, "state": state, "prompt": "building blocks into box"}
    r = policy.infer(obs)
    output = np.array(r["actions"])[0]  # (10,) 第一个 action

    # 方式A: 直接当 absolute
    target_A = action_to_tcp_pose(output)

    # 方式B: state + delta
    absolute_action = np.zeros(10, dtype=np.float32)
    absolute_action[:9] = state[:9] + output[:9]
    absolute_action[9] = output[9]
    target_B = action_to_tcp_pose(absolute_action)

    dist_A = np.linalg.norm(target_A[:3] - tcp[:3])
    dist_B = np.linalg.norm(target_B[:3] - tcp[:3])

    print(f"\n--- {name}: TCP={tcp[:3]} ---")
    print(f"  模型输出[:3]:     {output[:3]}")
    print(f"  方式A (absolute): target={[round(x,3) for x in target_A[:3]]}, 距当前 {dist_A:.3f}m")
    print(f"  方式B (state+δ):  target={[round(x,3) for x in target_B[:3]]}, 距当前 {dist_B:.3f}m")

    if dist_B < dist_A:
        print(f"  → 方式B更合理 (更接近当前位置)")
    else:
        print(f"  → 方式A更合理 (更接近当前位置)")

print("\n" + "=" * 70)
print("结论: 距离更小的方式是正确的解读")
print("  如果方式B总是更合理 → 模型输出是 delta, 需要加 state")
print("  如果方式A总是更合理 → 模型输出是 absolute, 直接用")
