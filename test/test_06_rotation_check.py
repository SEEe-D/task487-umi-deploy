"""测试6: 旋转坐标系验证

验证 R_M2R 旋转映射是否正确，以及旋转是围绕 TCP 还是 base。

操作:
  m      → 切换为 xyz 平移模式
  r      → 切换为 xyz 旋转模式
  x/X    → 当前模式下 x 轴 +/-
  y/Y    → 当前模式下 y 轴 +/-
  z/Z    → 当前模式下 z 轴 +/-
  h = 回起始   q = 退出
"""
import numpy as np
import rtde_control
import rtde_receive
import time
from scipy.spatial.transform import Rotation

ROBOT_IP = "192.168.3.244"
STEP_DEG = 30.0  # 每步旋转角度
STEP_POS_M = 0.02  # 每步平移 2cm

R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)
R_R2M = R_M2R.T

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c.reuploadScript()
time.sleep(0.5)

home = rtde_r.getActualTCPPose()
print("=" * 60)
print(f"起始 TCP: pos={[round(x,4) for x in home[:3]]}")
print(f"          rot={[round(x,4) for x in home[3:]]}")
print(f"平移步长: {STEP_POS_M} m")
print(f"旋转步长: {STEP_DEG}°")
print()
print("  m   → xyz平移模式")
print("  r   → xyz旋转模式")
print("  x/X → 当前模式下 x 轴 +/-")
print("  y/Y → 当前模式下 y 轴 +/-")
print("  z/Z → 当前模式下 z 轴 +/-")
print("  h = 回起始   q = 退出")
print()

axis_map = {
    'x': (0, +1), 'X': (0, -1),
    'y': (1, +1), 'Y': (1, -1),
    'z': (2, +1), 'Z': (2, -1),
}

mode = "rot"  # "rot" 或 "move"

while True:
    tcp = rtde_r.getActualTCPPose()
    R_robot = Rotation.from_rotvec(tcp[3:6]).as_matrix()
    mode_name = "xyz旋转" if mode == "rot" else "xyz平移"
    print(
        f"[{mode_name}] TCP pos: {[round(x,4) for x in tcp[:3]]}, "
        f"rot: {[round(x,4) for x in tcp[3:]]}",
        end=""
    )
    cmd = input("  > ").strip()

    if cmd == 'q':
        break
    elif cmd == 'm':
        mode = "move"
        print("  已切换到 xyz平移模式")
        continue
    elif cmd == 'r':
        mode = "rot"
        print("  已切换到 xyz旋转模式")
        continue
    elif cmd == 'h':
        print("  回起始位置...")
        rtde_c.moveL(home, 0.1, 0.1)
        time.sleep(1)
        continue
    elif cmd in axis_map:
        axis_idx, sign = axis_map[cmd]
        ax_name = ['x', 'y', 'z'][axis_idx]
        if mode == "move":
            target_pos = np.array(tcp[:3], dtype=np.float64)
            target_pos[axis_idx] += sign * STEP_POS_M
            target = list(target_pos) + list(tcp[3:6])
            print(f"  沿{ax_name}轴 {'+' if sign > 0 else '-'}{STEP_POS_M} m")
        else:
            # angle_rad = sign * np.radians(STEP_DEG)

            # axis_vec = np.zeros(3)
            # axis_vec[axis_idx] = angle_rad
            # R_delta_model = Rotation.from_rotvec(axis_vec).as_matrix()

            # R_target = R_robot @ R_delta_model
            # target_rotvec = Rotation.from_matrix(R_target).as_rotvec()

            angle_rad = sign * np.radians(STEP_DEG)

            # 世界坐标系下的旋转轴（固定 base x/y/z）
            axis_vec = np.zeros(3)
            axis_vec[axis_idx] = angle_rad
            R_delta_world = Rotation.from_rotvec(axis_vec).as_matrix()

            # 关键：左乘，表示绕世界轴旋转
            R_target = R_delta_world @ R_robot
            target_rotvec = Rotation.from_matrix(R_target).as_rotvec()

            # 修复 axis-angle π 附近符号歧义
            current_rotvec = tcp[3:6]
            if np.linalg.norm(target_rotvec - current_rotvec) > np.pi:
                R_current = Rotation.from_rotvec(current_rotvec).as_matrix()
                R_diff = R_target @ R_current.T
                diff_rotvec = Rotation.from_matrix(R_diff).as_rotvec()
                target_rotvec = current_rotvec + diff_rotvec

            target = list(tcp[:3]) + list(target_rotvec)
            print(f"  绕TCP{ax_name}轴 {'+' if sign > 0 else '-'}{STEP_DEG}°")

        rtde_c.moveL(target, 0.1, 0.1)
        time.sleep(0.5)
    else:
        print("  无效输入 (m/r x/X y/Y z/Z h q)")

rtde_c.disconnect()
rtde_r.disconnect()
print("完成")
