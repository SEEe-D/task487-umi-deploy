"""测试5: 模型坐标系输入 → R_M2R转换 → TCP工具坐标系移动

  x/X  → 模型x +/-  (经R_M2R转换后: 工具z方向)
  y/Y  → 模型y +/-  (经R_M2R转换后: -工具x方向)
  z/Z  → 模型z +/-  (经R_M2R转换后: -工具y方向)
  h = 回起始   q = 退出

  R_M2R (model→tool):
    model_x → tool_z
    model_y → -tool_x
    model_z → -tool_y
"""
import numpy as np
import rtde_control
import rtde_receive
import time

ROBOT_IP = "192.168.5.254"
STEP = 0.02  # 2cm

# model → tool (TCP) 的旋转
R_M2R = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c.reuploadScript()
time.sleep(0.5)

home = rtde_r.getActualTCPPose()
print("=" * 60)
print("模型坐标系输入 → R_M2R → TCP工具坐标系移动")
print("=" * 60)
print(f"起始 TCP: {[round(x,4) for x in home]}")
print(f"步长: {STEP*100}cm")
print()
print("  x/X  → 模型x +/-  → 工具z +/-")
print("  y/Y  → 模型y +/-  → 工具-x (即工具x反向)")
print("  z/Z  → 模型z +/-  → 工具-y (即工具y反向)")
print("  h = 回起始   q = 退出")
print()

axis_map = {'x': 0, 'y': 1, 'z': 2, 'X': 0, 'Y': 1, 'Z': 2}

while True:
    tcp = rtde_r.getActualTCPPose()
    print(f"TCP(世界): x={tcp[0]:+.4f} y={tcp[1]:+.4f} z={tcp[2]:+.4f}", end="")
    cmd = input("  > ").strip()

    if cmd == 'q':
        break
    elif cmd == 'h':
        print("  回起始位置...")
        rtde_c.moveL(home, 0.1, 0.1)
        time.sleep(1)
        continue
    elif len(cmd) == 1 and cmd.lower() in 'xyz':
        axis = axis_map[cmd]
        sign = -1 if cmd.isupper() else +1
        dist = sign * STEP

        # 模型坐标系下的位移
        model_delta = np.zeros(3)
        model_delta[axis] = dist

        # R_M2R: model → tool
        tool_delta = R_M2R @ model_delta

        # poseTrans: 在工具坐标系下移动
        tool_pose_delta = [tool_delta[0], tool_delta[1], tool_delta[2], 0.0, 0.0, 0.0]
        target = rtde_c.poseTrans(tcp, tool_pose_delta)

        ax_name = ['x', 'y', 'z'][axis]
        print(f"  模型{ax_name} {'+' if dist>0 else ''}{dist*100:.0f}cm")
        print(f"  → 模型: dx={model_delta[0]:+.4f} dy={model_delta[1]:+.4f} dz={model_delta[2]:+.4f}")
        print(f"  → 工具: dx={tool_delta[0]:+.4f} dy={tool_delta[1]:+.4f} dz={tool_delta[2]:+.4f}")

        rtde_c.moveL(target, 0.1, 0.1)
        time.sleep(0.5)
    else:
        print("  无效输入 (x/X y/Y z/Z h q)")

rtde_c.disconnect()
rtde_r.disconnect()
print("完成")
