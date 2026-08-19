"""测试4: 用真实机器人验证坐标系

1. 读取机器人真实 TCP (base frame)
2. tcp_pose → state_20d → action_to_tcp_pose 往返检查
3. 对比 state 值与训练 norm_stats 范围, 判断坐标系是否匹配
4. 推理一次, 检查 state+delta 后的目标是否合理
"""
import numpy as np
from scipy.spatial.transform import Rotation
import rtde_receive

ROBOT_IP = "192.168.3.254"

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

# Norm stats 参考
state_mean = np.array([-0.036, 0.083, 0.542, 0.218, 0.563, 0.568, 0.301, -0.624, 0.353, 0.145])
state_q01  = np.array([-0.312, -0.205, 0.274, -0.587, 0.027, -0.726, -0.898, -0.945, -0.729, 0.0])
state_q99  = np.array([ 0.297,  0.379, 0.785,  0.938, 0.985,  0.945,  0.992,  0.287,  0.991, 0.383])

dim_names = ["pos_x", "pos_y", "pos_z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]

# ========== 1. 读取真实 TCP ==========
print("=" * 70)
print("1. 读取机器人真实 TCP pose (base frame)")
print("=" * 70)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
tcp_pose = np.array(rtde_r.getActualTCPPose())
rtde_r.disconnect()

print(f"  TCP position (xyz):    {tcp_pose[:3]}")
print(f"  TCP rotation (rotvec): {tcp_pose[3:]}")

# ========== 2. 往返一致性 ==========
print("\n" + "=" * 70)
print("2. 往返一致性: TCP → state_20d → action_to_tcp_pose")
print("=" * 70)
state = tcp_pose_to_state_20d(tcp_pose, 0.0)
recovered = action_to_tcp_pose(state[:10])

pos_err = np.linalg.norm(tcp_pose[:3] - recovered[:3])
R_orig = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
R_recv = Rotation.from_rotvec(recovered[3:6]).as_matrix()
rot_err = np.arccos(np.clip((np.trace(R_orig.T @ R_recv) - 1) / 2, -1, 1))

print(f"  原始 TCP:  {[round(x,4) for x in tcp_pose]}")
print(f"  还原 TCP:  {[round(x,4) for x in recovered]}")
print(f"  位置误差: {pos_err:.6f} m")
print(f"  旋转误差: {np.degrees(rot_err):.4f} deg")
print(f"  {'✓ 往返一致' if pos_err < 1e-4 and rot_err < 0.01 else '✗ 有误差!'}")

# ========== 3. State 值与训练分布对比 ==========
print("\n" + "=" * 70)
print("3. State 值 vs 训练数据分布 (q01 ~ q99)")
print("=" * 70)
print(f"  {'维度':12s} {'当前值':>10s} {'q01':>10s} {'q99':>10s} {'mean':>10s}  状态")

all_in_range = True
for i in range(10):
    val = state[i]
    in_range = state_q01[i] <= val <= state_q99[i]
    status = "✓" if in_range else "✗ 超出范围!"
    if not in_range:
        all_in_range = False
    print(f"  {dim_names[i]:12s} {val:10.4f} {state_q01[i]:10.4f} {state_q99[i]:10.4f} {state_mean[i]:10.4f}  {status}")

if all_in_range:
    print("\n  所有维度在训练分布内 → 坐标系匹配正确")
else:
    print("\n  部分维度超出训练分布!")
    print("  可能原因: 1) 机器人姿态和训练时不同  2) 坐标系映射有误  3) 工作空间不同")

# ========== 4. 推理测试 ==========
print("\n" + "=" * 70)
print("4. 用真实 TCP 推理, 检查 delta 方向和大小")
print("=" * 70)
try:
    import cv2
    from openpi_client import websocket_client_policy as ws

    policy = ws.WebsocketClientPolicy(host="localhost", port=8000)

    # 用真实相机图像
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    cap.release()
    cv2.imwrite("/tmp/cam_check.jpg", frame)
    print(f"  相机帧已保存到 /tmp/cam_check.jpg")

    obs = {"cam_right": img, "state": state, "prompt": "building blocks into box"}
    r = policy.infer(obs)
    actions = np.array(r["actions"])

    print(f"\n  当前 state[:3] (pos): {state[:3]}")
    print(f"  模型 delta[:3]:       {actions[0,:3]}")
    print(f"  绝对目标 (state+δ):   {state[:3] + actions[0,:3]}")
    print(f"\n  Action chunk (10步) position deltas:")
    for i in range(min(10, len(actions))):
        d = actions[i, :3]
        abs_pos = state[:3] + d
        print(f"    step {i}: delta={d}, abs={abs_pos}")

    print(f"\n  如果 delta 方向合理(朝向目标物体) → 模型在正常工作")
    print(f"  如果 delta 持续单方向漂移 → 可能相机/姿态和训练不匹配")

except Exception as e:
    print(f"  推理跳过 (server未启动或相机不可用): {e}")
