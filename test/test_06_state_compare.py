"""测试6: 对比部署的 state 构建和训练数据是否一致

读取机器人真实 TCP，两种方式构建 rot6d:
  A) 有 R_M2R: R_model = R_robot @ R_M2R → rot6d
  B) 无 R_M2R: R_robot 直接 → rot6d
对比训练数据分布，看哪种匹配
"""
import numpy as np
from scipy.spatial.transform import Rotation
import rtde_receive

ROBOT_IP = "192.168.3.254"

R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)

# 训练数据统计 (从 parquet 读取)
train_stats = {
    'pos_x':  (-0.157, 0.086, -0.063),
    'pos_y':  (-0.076, 0.257, 0.061),
    'pos_z':  (0.509, 0.758, 0.602),
    'r0':     (-0.137, 0.625, 0.254),
    'r1':     (0.379, 0.895, 0.638),
    'r2':     (0.077, 0.897, 0.597),
    'r3':     (-0.590, 0.970, 0.534),
    'r4':     (-0.826, -0.226, -0.574),
    'r5':     (0.013, 0.669, 0.285),
}

dim_names = ['pos_x', 'pos_y', 'pos_z', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5']

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
tcp = np.array(rtde_r.getActualTCPPose())
rtde_r.disconnect()

print(f"TCP (base frame): pos={tcp[:3].round(4)}  rotvec={tcp[3:].round(4)}")
R_robot = Rotation.from_rotvec(tcp[3:6]).as_matrix()

# 方式A: 有 R_M2R
R_model_A = R_robot @ R_M2R
rot6d_A = R_model_A[:, :2].T.flatten()

# 方式B: 无 R_M2R
rot6d_B = R_robot[:, :2].T.flatten()

state_A = np.concatenate([tcp[:3], rot6d_A])
state_B = np.concatenate([tcp[:3], rot6d_B])

print("\n" + "=" * 75)
print(f"{'dim':6s} | {'训练 min':>10s} {'训练 max':>10s} {'训练 mean':>10s} | {'A(有R_M2R)':>10s} {'B(无R_M2R)':>10s} | {'A匹配':5s} {'B匹配':5s}")
print("-" * 75)

a_match = 0
b_match = 0
for i, name in enumerate(dim_names):
    lo, hi, mean = train_stats[name]
    va = state_A[i]
    vb = state_B[i]
    a_in = lo <= va <= hi
    b_in = lo <= vb <= hi
    if a_in: a_match += 1
    if b_in: b_match += 1
    print(f"{name:6s} | {lo:10.4f} {hi:10.4f} {mean:10.4f} | {va:10.4f} {vb:10.4f} | {'✓' if a_in else '✗':5s} {'✓' if b_in else '✗':5s}")

print("-" * 75)
print(f"匹配数: A(有R_M2R)={a_match}/9  B(无R_M2R)={b_match}/9")
print()
if a_match > b_match:
    print("结论: 方式A (有R_M2R) 更匹配训练数据 → tcp_pose_to_state_20d 正确")
elif b_match > a_match:
    print("结论: 方式B (无R_M2R) 更匹配训练数据 → tcp_pose_to_state_20d 的 R_M2R 是多余的!")
else:
    print("结论: 两种方式匹配数相同，需要更多数据判断")
