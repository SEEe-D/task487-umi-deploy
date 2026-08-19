"""测试6: 旋转坐标系验证

验证 R_M2R 旋转映射是否正确，以及旋转是围绕 TCP 还是 base。

操作:
  rx/RX → 绕模型 x 轴旋转 ±5°
  ry/RY → 绕模型 y 轴旋转 ±5°
  rz/RZ → 绕模型 z 轴旋转 ±5°
  h = 回起始   q = 退出
"""
import numpy as np
import rtde_control
import rtde_receive
import time
from scipy.spatial.transform import Rotation

ROBOT_IP = "192.168.3.244"
STEP_DEG = 30.0  # 每步旋转角度

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
print("旋转坐标系验证 (R_M2R)")
print("=" * 60)
print(f"起始 TCP: pos={[round(x,4) for x in home[:3]]}")
print(f"          rot={[round(x,4) for x in home[3:]]}")
print(f"步长: {STEP_DEG}°")
print()
print("  rx/RX → 绕模型x轴 +/-")
print("  ry/RY → 绕模型y轴 +/-")
print("  rz/RZ → 绕模型z轴 +/-")
print("  h = 回起始   q = 退出")
print()

axis_map = {
    'x': (0, +1), 'X': (0, -1),
    'y': (1, +1), 'Y': (1, -1),
    'z': (2, +1), 'Z': (2, -1),
}

while True:
    tcp = rtde_r.getActualTCPPose()
    R_robot = Rotation.from_rotvec(tcp[3:6]).as_matrix()
    R_model = R_robot @ R_M2R
    print(f"TCP rot: {[round(x,4) for x in tcp[3:]]}", end="")
    cmd = input("  > ").strip()

    if cmd == 'q':
        break
    elif cmd == 'h':
        print("  回起始位置...")
        rtde_c.moveL(home, 0.1, 0.1)
        time.sleep(1)
        continue
    elif cmd in axis_map:
        axis_idx, sign = axis_map[cmd]
        angle_rad = sign * np.radians(STEP_DEG)

        # 在模型坐标系下绕指定轴旋转
        axis_vec = np.zeros(3)
        axis_vec[axis_idx] = angle_rad
        R_delta_model = Rotation.from_rotvec(axis_vec).as_matrix()

        # 方法: R_target_model = R_current_model @ R_delta
        # 和部署脚本一致: R_current_model = R_robot @ R_M2R
        R_target_model = R_model @ R_delta_model
        R_target_robot = R_target_model @ R_R2M
        target_rotvec = Rotation.from_matrix(R_target_robot).as_rotvec()

        # 修复 axis-angle π 附近符号歧义
        current_rotvec = tcp[3:6]
        if np.linalg.norm(target_rotvec - current_rotvec) > np.pi:
            R_current = Rotation.from_rotvec(current_rotvec).as_matrix()
            R_diff = R_target_robot @ R_current.T
            diff_rotvec = Rotation.from_matrix(R_diff).as_rotvec()
            target_rotvec = current_rotvec + diff_rotvec

        target = list(tcp[:3]) + list(target_rotvec)
        ax_name = ['x', 'y', 'z'][axis_idx]
        print(f"  绕模型{ax_name}轴 {'+'if sign>0 else '-'}{STEP_DEG}°")
        print(f"  → R_delta_model applied via R_current_model @ R_delta → R_R2M")

        rtde_c.moveL(target, 0.1, 0.1)
        time.sleep(0.5)
    else:
        print("  无效输入 (rx/RX ry/RY rz/RZ h q)")

rtde_c.disconnect()
rtde_r.disconnect()
print("完成")
