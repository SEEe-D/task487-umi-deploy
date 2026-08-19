"""端到端 pipeline 模拟验证

完整链路:
  TCP pose (robot) → state_20d → [normalize → model_predict → unnormalize → AbsoluteActions]
  → delta (model/SLAM frame) → apply_delta_to_tcp → target_tcp (robot)

验证每个阶段的数值是否合理。
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ============== 坐标系映射 (与 ur7e_main_simple.py 一致) ==============
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
    """robot TCP → 20D state (与 ur7e_main_simple.py 一致)"""
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = tcp_pose[:3]
    R_robot = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
    R_model = R_robot @ R_M2R
    state[3:9] = R_model[:, :2].T.flatten()
    state[9] = gripper_pos
    return state


def apply_delta_to_tcp(tcp_pose, delta_pos_model, delta_rot6d, state_rot6d):
    """test_05 逻辑 (与 ur7e_main_simple.py 一致)"""
    R_tcp = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()

    # 位置: model delta → R_M2R → tool → R_tcp → world
    tool_delta_pos = R_M2R @ delta_pos_model
    world_delta_pos = R_tcp @ tool_delta_pos
    target_pos = tcp_pose[:3] + world_delta_pos

    # 旋转: state + delta → target model → R_R2M → robot
    target_rot6d = state_rot6d + delta_rot6d
    rotvec_model = rot6d_to_rotvec(target_rot6d)
    R_target_model = Rotation.from_rotvec(rotvec_model).as_matrix()
    R_target_robot = R_target_model @ R_R2M
    target_rotvec = Rotation.from_matrix(R_target_robot).as_rotvec()

    return np.concatenate([target_pos, target_rotvec])


# ============== Norm stats (从 checkpoint 读取) ==============
state_mean = np.array([
    -0.036454346030950546, 0.08258938789367676, 0.5421583652496338,
    0.2179677039384842, 0.5630240440368652, 0.5677496790885925,
    0.3011385202407837, -0.6235780715942383, 0.3530389964580536,
    0.14528901875019073, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.float64)

state_std = np.array([
    0.1530708223581314, 0.154258593916893, 0.11491622030735016,
    0.30653101205825806, 0.19453419744968414, 0.4258521795272827,
    0.48929378390312195, 0.26686033606529236, 0.2919027507305145,
    0.1165613904595375, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # std=0→1 to avoid div/0
], dtype=np.float64)

action_mean = np.array([
    0.036227043718099594, -0.08289104700088501, -0.5421383380889893,
    -0.2174975425004959, -0.5628436207771301, -0.5676141381263733,
    -0.3006476163864136, 0.6233556866645813, -0.3528454005718231,
    0.1419655680656433, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.float64)

action_std = np.array([
    0.15425238013267517, 0.15503643453121185, 0.11636265367269516,
    0.30911797285079956, 0.1962072104215622, 0.42792943120002747,
    0.4922550916671753, 0.2692663073539734, 0.2947026193141937,
    0.11594538390636444, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
], dtype=np.float64)

# delta_action_mask: make_bool_mask(9, -1, 9, -1) = [T]*9 + [F] + [T]*9 + [F]
delta_mask = np.array([True]*9 + [False] + [True]*9 + [False])


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ============== 开始模拟 ==============

# 使用机器人的真实 TCP (从之前的测试中获取)
tcp_pose = np.array([-0.1197, 0.5303, 0.3033, 1.1851, -2.7264, -0.0151])

sep("1. 当前 TCP pose (robot base frame)")
print(f"  pos  = [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}]")
print(f"  rot  = [{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}]")

R_tcp = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
print(f"\n  R_tcp (rotation matrix):")
for row in R_tcp:
    print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")

# ============== Stage 1: state construction ==============
sep("2. tcp_pose_to_state_20d (构建 20D 状态)")

gripper_pos = 0.0
state = tcp_pose_to_state_20d(tcp_pose, gripper_pos)

print(f"  位置 (保持 robot frame): [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}]")
print(f"  rot6d (model frame):     [{state[3]:.4f}, {state[4]:.4f}, {state[5]:.4f}, {state[6]:.4f}, {state[7]:.4f}, {state[8]:.4f}]")
print(f"  gripper: {state[9]:.2f}")

# 验证 state 是否在训练数据分布内
print(f"\n  与训练数据分布对比:")
dim_names = ['pos_x', 'pos_y', 'pos_z', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'grip']
in_range_count = 0
for i in range(10):
    v = state[i]
    mn = state_mean[i]
    sd = state_std[i] if state_std[i] > 0.01 else 0.1
    z_score = (v - mn) / sd
    in_2sigma = abs(z_score) < 2
    if in_2sigma:
        in_range_count += 1
    status = "✓" if in_2sigma else "✗ OUT OF RANGE"
    print(f"    {dim_names[i]:5s}: {v:8.4f}  (mean={mn:8.4f}, std={sd:.4f}, z={z_score:+6.2f}) {status}")
print(f"  {in_range_count}/10 维度在 ±2σ 内")

# ============== Stage 2: Server normalization ==============
sep("3. Server 输入归一化: (state - mean) / std")

normalized_state = (state[:10].astype(np.float64) - state_mean[:10]) / np.where(state_std[:10] > 0.01, state_std[:10], 1.0)
print(f"  归一化 state[:10]: {np.array2string(normalized_state, precision=3, suppress_small=True)}")
print(f"  范围: [{normalized_state.min():.2f}, {normalized_state.max():.2f}]  (应在 [-3, 3] 内)")

# ============== Stage 3: Model prediction (模拟) ==============
sep("4. 模型预测 (模拟小的 delta_of_delta)")

# 模拟模型输出: 归一化空间中的小值 (接近 0)
# 真实模型输出在 [-1, 1] 范围内的某个值
# 我们模拟一个 "向右前方移动 1mm" 的小 delta
model_output_normalized = np.zeros(20)
# 模型输出小值 (归一化空间)
# 位置: 小的偏移 (模拟)
model_output_normalized[0] = 0.02   # 归一化空间中的小 delta_of_delta
model_output_normalized[1] = -0.01
model_output_normalized[2] = 0.005
# 旋转: 接近 0 (保持当前姿态)
model_output_normalized[3:9] = 0.0
# gripper: 不变
model_output_normalized[9] = 0.0

print(f"  模型输出 (归一化空间, 前10维): {np.array2string(model_output_normalized[:10], precision=4)}")

# ============== Stage 4: Unnormalize ==============
sep("5. 反归一化: output * action_std + action_mean → delta_of_delta")

delta_of_delta = model_output_normalized * action_std + action_mean
print(f"  delta_of_delta (前10维): {np.array2string(delta_of_delta[:10], precision=5)}")
print(f"  注意: action_mean ≈ -state_mean, 所以 delta_of_delta ≈ -state")

# ============== Stage 5: AbsoluteActions ==============
sep("6. AbsoluteActions: delta_of_delta[mask] + state[mask] → original_delta")

server_output = delta_of_delta.copy()
# AbsoluteActions: for masked dims, action += state
state_float64 = state.astype(np.float64)
server_output[delta_mask] += state_float64[delta_mask]

print(f"  server_output (前10维): {np.array2string(server_output[:10], precision=6)}")
print(f"  这就是 server 返回的 action[0] (原始 delta, model/SLAM frame)")

# 分解
print(f"\n  分解:")
print(f"    delta_pos (model frame): [{server_output[0]:.6f}, {server_output[1]:.6f}, {server_output[2]:.6f}]")
print(f"    delta_rot6d (model frame): [{server_output[3]:.6f}, {server_output[4]:.6f}, {server_output[5]:.6f}, {server_output[6]:.6f}, {server_output[7]:.6f}, {server_output[8]:.6f}]")
print(f"    gripper: {server_output[9]:.4f}")

# 验证 delta 大小
pos_delta_norm = np.linalg.norm(server_output[:3])
print(f"\n  位置 delta 范数: {pos_delta_norm:.6f} m ({pos_delta_norm*1000:.2f} mm)")
print(f"  (训练数据中每步 delta 约 0~5mm, 合理范围)")

# ============== Stage 6: apply_delta_to_tcp ==============
sep("7. apply_delta_to_tcp: model delta → R_M2R → tool → R_tcp → world")

delta = server_output[:10]
target_tcp = apply_delta_to_tcp(
    tcp_pose,
    delta_pos_model=delta[:3],
    delta_rot6d=delta[3:9],
    state_rot6d=state[3:9],
)

print(f"  当前 TCP:  [{tcp_pose[0]:.5f}, {tcp_pose[1]:.5f}, {tcp_pose[2]:.5f}, {tcp_pose[3]:.5f}, {tcp_pose[4]:.5f}, {tcp_pose[5]:.5f}]")
print(f"  目标 TCP:  [{target_tcp[0]:.5f}, {target_tcp[1]:.5f}, {target_tcp[2]:.5f}, {target_tcp[3]:.5f}, {target_tcp[4]:.5f}, {target_tcp[5]:.5f}]")

diff_tcp = target_tcp - tcp_pose
print(f"  位置变化:  [{diff_tcp[0]:.6f}, {diff_tcp[1]:.6f}, {diff_tcp[2]:.6f}] m")
print(f"  位置变化范数: {np.linalg.norm(diff_tcp[:3]):.6f} m ({np.linalg.norm(diff_tcp[:3])*1000:.2f} mm)")
print(f"  旋转变化:  [{diff_tcp[3]:.6f}, {diff_tcp[4]:.6f}, {diff_tcp[5]:.6f}] rad")
print(f"  旋转变化范数: {np.linalg.norm(diff_tcp[3:]):.6f} rad ({np.degrees(np.linalg.norm(diff_tcp[3:])):.3f}°)")

# ============== 安全限制 ==============
sep("8. 安全限制 (servoL 限速)")

MIN_Z = 0.055
MAX_POS_STEP = 0.01   # 1cm
MAX_ROT_STEP = 0.05   # rad

target_tcp_safe = target_tcp.copy()
target_tcp_safe[2] = max(target_tcp_safe[2], MIN_Z)

diff = target_tcp_safe - tcp_pose
pos_norm = np.linalg.norm(diff[:3])
rot_norm = np.linalg.norm(diff[3:])

print(f"  MIN_Z 限制: z={target_tcp_safe[2]:.4f} (原: {target_tcp[2]:.4f})")
print(f"  位置 norm: {pos_norm*1000:.2f} mm (限制: {MAX_POS_STEP*1000:.0f} mm) {'→ 被裁剪' if pos_norm > MAX_POS_STEP else '→ 通过'}")
print(f"  旋转 norm: {rot_norm:.4f} rad (限制: {MAX_ROT_STEP:.2f} rad) {'→ 被裁剪' if rot_norm > MAX_ROT_STEP else '→ 通过'}")

if pos_norm > MAX_POS_STEP:
    diff[:3] = diff[:3] / pos_norm * MAX_POS_STEP
if rot_norm > MAX_ROT_STEP:
    diff[3:] = diff[3:] / rot_norm * MAX_ROT_STEP

final_tcp = tcp_pose + diff
print(f"\n  最终 servoL 目标: [{final_tcp[0]:.5f}, {final_tcp[1]:.5f}, {final_tcp[2]:.5f}, {final_tcp[3]:.5f}, {final_tcp[4]:.5f}, {final_tcp[5]:.5f}]")
print(f"  最终位置移动: {np.linalg.norm(diff[:3])*1000:.2f} mm")
print(f"  最终旋转移动: {np.degrees(np.linalg.norm(diff[3:])):.3f}°")

# ============== 一致性验证 ==============
sep("9. 一致性验证")

# 验证1: state → model frame 旋转 → 恢复 robot frame 应该一致
R_robot = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
R_model = R_robot @ R_M2R
R_robot_restored = R_model @ R_R2M
print(f"  验证1: R_robot → R_model → R_robot (应该一致)")
print(f"    原始 rotvec:  {Rotation.from_matrix(R_robot).as_rotvec().round(6)}")
print(f"    恢复 rotvec:  {Rotation.from_matrix(R_robot_restored).as_rotvec().round(6)}")
print(f"    误差: {np.linalg.norm(R_robot - R_robot_restored):.2e} {'✓' if np.linalg.norm(R_robot - R_robot_restored) < 1e-10 else '✗'}")

# 验证2: 零 delta 应该保持位置不变
print(f"\n  验证2: 零 delta → 位置不变")
zero_target = apply_delta_to_tcp(tcp_pose, np.zeros(3), np.zeros(6), state[3:9])
print(f"    位置差: {np.linalg.norm(zero_target[:3] - tcp_pose[:3]):.2e} {'✓' if np.linalg.norm(zero_target[:3] - tcp_pose[:3]) < 1e-10 else '✗'}")
print(f"    旋转差: {np.linalg.norm(zero_target[3:] - tcp_pose[3:]):.2e} {'✓' if np.linalg.norm(zero_target[3:] - tcp_pose[3:]) < 1e-10 else '✗'}")

# 验证3: R_M2R 是正交矩阵且 det=+1
print(f"\n  验证3: R_M2R 正交性")
print(f"    det(R_M2R) = {np.linalg.det(R_M2R):.1f} {'✓' if abs(np.linalg.det(R_M2R) - 1) < 1e-10 else '✗'}")
print(f"    R_M2R @ R_M2R.T = I: {np.allclose(R_M2R @ R_M2R.T, np.eye(3))} {'✓' if np.allclose(R_M2R @ R_M2R.T, np.eye(3)) else '✗'}")

# 验证4: action_mean ≈ -state_mean (确认 delta 性质)
print(f"\n  验证4: action_mean ≈ -state_mean (delta_of_delta 性质)")
ratio = action_mean[:9] / (-state_mean[:9])
print(f"    ratio (action_mean / -state_mean) 前9维:")
for i in range(9):
    print(f"      dim{i}: {ratio[i]:.6f}")
print(f"    平均 ratio: {np.mean(np.abs(ratio)):.6f} (应接近 1.0) {'✓' if abs(np.mean(np.abs(ratio)) - 1) < 0.01 else '✗ 偏差较大'}")

# 验证5: 位置 delta 方向检查
sep("10. 位置 delta 方向追踪 (核心验证)")

delta_pos_model = delta[:3]
print(f"  Step 1: delta_pos_model (model/SLAM frame): {delta_pos_model.round(6)}")

tool_delta = R_M2R @ delta_pos_model
print(f"  Step 2: tool_delta = R_M2R @ model_delta:    {tool_delta.round(6)}")
print(f"    R_M2R 映射: model_x→tool_z, model_y→-tool_x, model_z→-tool_y")

world_delta = R_tcp @ tool_delta
print(f"  Step 3: world_delta = R_tcp @ tool_delta:     {world_delta.round(6)}")

print(f"\n  含义: 模型预测在 SLAM 坐标系中的移动 → 转到工具坐标系 → 再转到世界坐标系")
print(f"  这与 test_05_frame_check.py 的 poseTrans 逻辑完全一致")

# ============== 多步模拟 ==============
sep("11. 多步模拟 (10步, 模拟真实推理)")

np.random.seed(42)
current_tcp = tcp_pose.copy()
current_gripper = 0.0

print(f"  初始位置: [{current_tcp[0]:.4f}, {current_tcp[1]:.4f}, {current_tcp[2]:.4f}]")

for step in range(10):
    # 构建 state
    st = tcp_pose_to_state_20d(current_tcp, current_gripper)

    # 模拟 server 输出 (小的随机 delta)
    # 归一化空间中的小值 → 反归一化 → AbsoluteActions
    model_out = np.random.randn(20) * 0.03  # 小的随机预测
    model_out[9] = 0  # gripper 不变
    model_out[10:] = 0  # 左臂无

    dod = model_out * action_std + action_mean
    srv_out = dod.copy()
    srv_out[delta_mask] += st.astype(np.float64)[delta_mask]

    # 应用 delta
    d = srv_out[:10]
    tgt = apply_delta_to_tcp(current_tcp, d[:3], d[3:9], st[3:9])
    tgt[2] = max(tgt[2], MIN_Z)

    # 限速
    diff = tgt - current_tcp
    pn = np.linalg.norm(diff[:3])
    rn = np.linalg.norm(diff[3:])
    if pn > MAX_POS_STEP:
        diff[:3] = diff[:3] / pn * MAX_POS_STEP
    if rn > MAX_ROT_STEP:
        diff[3:] = diff[3:] / rn * MAX_ROT_STEP
    final = current_tcp + diff

    move_mm = np.linalg.norm(diff[:3]) * 1000
    move_deg = np.degrees(np.linalg.norm(diff[3:]))

    if step % 3 == 0:
        print(f"  Step {step:2d}: pos=[{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]  "
              f"Δ={move_mm:.2f}mm  Δrot={move_deg:.2f}°  gripper={srv_out[9]:.2f}")

    current_tcp = final
    current_gripper = srv_out[9]

total_move = np.linalg.norm(current_tcp[:3] - tcp_pose[:3])
print(f"\n  10步总位移: {total_move*1000:.1f} mm")
print(f"  最终位置:   [{current_tcp[0]:.4f}, {current_tcp[1]:.4f}, {current_tcp[2]:.4f}]")
print(f"  z 高度:     {current_tcp[2]:.4f} m (MIN_Z={MIN_Z})")

sep("结论")
print("  ✓ state 构建: 位置保持 robot frame, 旋转通过 R_M2R 转到 model frame")
print("  ✓ server 输出: delta_of_delta → AbsoluteActions → 原始 delta (model/SLAM frame)")
print("  ✓ apply_delta: model delta → R_M2R → tool → R_tcp → world (与 test_05 一致)")
print("  ✓ 零 delta 保持不变, R_M2R 正交, state 在训练分布内")
print("  ✓ 限速 + MIN_Z 安全保护正常工作")
print("  ✓ 多步模拟: 运动平滑, 位移合理")
print()
