"""
测试模型坐标系与机器人坐标系之间的坐标变换。

用法：
    python scripts_real/test_coord_transform.py --robot_ip 192.168.3.254

两部分测试：
  1) 坐标轴映射测试：沿模型 +X/-X/+Y/-Y/+Z/-Z 方向移动，验证映射
  2) TCP 偏移测试：设置 setTcp 后，验证偏移和旋转是否正确

坐标映射（model -> 法兰/工具局部坐标系）：
    model +X → 法兰 +Z（工具轴方向）
    model +Y → 法兰 -X
    model +Z → 法兰 -Y
"""

import numpy as np
import time
import click
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

# 坐标变换矩阵（model -> 法兰/工具局部坐标系）
# model_x = flange_z, model_y = flange_-x, model_z = flange_-y
R_m2r = np.array([[0, -1,  0],
                   [0,  0, -1],
                   [1,  0,  0]], dtype=np.float64)

EXPECTED = {
    '+x': '法兰 +Z（工具轴方向/向前）',
    '-x': '法兰 -Z（工具轴反方向/向后）',
    '+y': '法兰 -X',
    '-y': '法兰 +X',
    '+z': '法兰 -Y',
    '-z': '法兰 +Y',
}


@click.command()
@click.option('--robot_ip', default='192.168.3.254', type=str)
@click.option('--delta', default=0.085, type=float, help='每步移动距离（米），默认 85mm')
@click.option('--speed', default=0.05, type=float, help='运动速度（m/s）')
@click.option('--acceleration', default=0.1, type=float)
@click.option('--tcp_offset_x', default=-0.016, type=float, help='TCP X offset in flange frame (meters).')
@click.option('--tcp_offset_y', default=-0.028, type=float, help='TCP Y offset in flange frame (meters).')
@click.option('--tcp_offset_z', default=0.2105, type=float, help='TCP Z offset in flange frame (meters).')
@click.option('--tcp_rot_x', default=0.0, type=float, help='TCP rotation rx in flange frame (radians).')
@click.option('--tcp_rot_y', default=-0.1745, type=float, help='TCP rotation ry in flange frame (radians).')
@click.option('--tcp_rot_z', default=0.0, type=float, help='TCP rotation rz in flange frame (radians).')
def main(robot_ip, delta, speed, acceleration,
         tcp_offset_x, tcp_offset_y, tcp_offset_z,
         tcp_rot_x, tcp_rot_y, tcp_rot_z):
    print(f"连接机器人 {robot_ip} ...")
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    print("连接成功\n")

    tcp_offset_pose = [tcp_offset_x, tcp_offset_y, tcp_offset_z,
                       tcp_rot_x, tcp_rot_y, tcp_rot_z]

    while True:
        print("\n" + "=" * 60)
        print("选择测试模式：")
        print("  1 - 坐标轴映射测试（无 TCP 偏移）")
        print("  2 - TCP 偏移测试（设置 setTcp 后平移测试）")
        print("  3 - TCP 偏移旋转测试（设置 setTcp 后绕 TCP 旋转）")
        print("  q - 退出")
        mode = input("  > ").strip().lower()
        if mode == 'q':
            break
        elif mode == '1':
            test_axis_mapping(rtde_c, rtde_r, delta, speed, acceleration)
        elif mode == '2':
            test_tcp_offset(rtde_c, rtde_r, tcp_offset_pose, delta, speed, acceleration)
        elif mode == '3':
            test_tcp_rotation(rtde_c, rtde_r, tcp_offset_pose, speed, acceleration)
        else:
            print("  无效输入")


def test_axis_mapping(rtde_c, rtde_r, delta, speed, acceleration):
    """测试模型坐标轴到法兰坐标轴的映射（不设置 TCP 偏移）"""
    # 先清除 TCP 偏移
    rtde_c.setTcp([0, 0, 0, 0, 0, 0])
    print("\n已清除 TCP 偏移 (setTcp = [0,0,0,0,0,0])")

    print("预期映射关系：")
    for k, v in EXPECTED.items():
        print(f"  model {k.upper()}  →  {v}")

    direction_map = {
        '+x': np.array([delta, 0, 0]),
        '-x': np.array([-delta, 0, 0]),
        '+y': np.array([0, delta, 0]),
        '-y': np.array([0, -delta, 0]),
        '+z': np.array([0, 0, delta]),
        '-z': np.array([0, 0, -delta]),
    }

    while True:
        print("\n输入模型方向（+x/-x/+y/-y/+z/-z），输入 b 返回：")
        cmd = input("  > ").strip().lower()
        if cmd == 'b':
            break
        if cmd not in direction_map:
            print("  无效输入")
            continue

        model_delta = direction_map[cmd]
        move_in_flange_frame(rtde_c, rtde_r, model_delta, cmd, speed, acceleration)


def test_tcp_offset(rtde_c, rtde_r, tcp_offset_pose, delta, speed, acceleration):
    """设置 TCP 偏移后测试坐标轴映射"""
    # 先读取设置前的位姿
    pose_before = np.array(rtde_r.getActualTCPPose())

    rtde_c.setTcp(tcp_offset_pose)
    print(f"\n已设置 TCP 偏移: {np.round(tcp_offset_pose, 4)}")
    print(f"  位移: x={tcp_offset_pose[0]*1000:.1f}mm, y={tcp_offset_pose[1]*1000:.1f}mm, z={tcp_offset_pose[2]*1000:.1f}mm")
    print(f"  旋转: rx={np.degrees(tcp_offset_pose[3]):.1f}°, ry={np.degrees(tcp_offset_pose[4]):.1f}°, rz={np.degrees(tcp_offset_pose[5]):.1f}°")

    time.sleep(0.2)
    pose_after = np.array(rtde_r.getActualTCPPose())
    print(f"\n  设置前 TCP 位姿: {np.round(pose_before, 4)}")
    print(f"  设置后 TCP 位姿: {np.round(pose_after, 4)}")
    print(f"  位置差:          {np.round(pose_after[:3] - pose_before[:3], 4)}")

    direction_map = {
        '+x': np.array([delta, 0, 0]),
        '-x': np.array([-delta, 0, 0]),
        '+y': np.array([0, delta, 0]),
        '-y': np.array([0, -delta, 0]),
        '+z': np.array([0, 0, delta]),
        '-z': np.array([0, 0, -delta]),
    }

    while True:
        print("\n输入模型方向（+x/-x/+y/-y/+z/-z），输入 b 返回（会清除TCP偏移）：")
        cmd = input("  > ").strip().lower()
        if cmd == 'b':
            rtde_c.setTcp([0, 0, 0, 0, 0, 0])
            print("  已清除 TCP 偏移")
            break
        if cmd not in direction_map:
            print("  无效输入")
            continue

        model_delta = direction_map[cmd]
        move_in_flange_frame(rtde_c, rtde_r, model_delta, cmd, speed, acceleration)


ROT_EXPECTED = {
    '+rx': '绕模型 X 轴（法兰 Z 轴/工具轴）正向旋转',
    '-rx': '绕模型 X 轴（法兰 Z 轴/工具轴）负向旋转',
    '+ry': '绕模型 Y 轴（法兰 -X 轴）正向旋转',
    '-ry': '绕模型 Y 轴（法兰 -X 轴）负向旋转',
    '+rz': '绕模型 Z 轴（法兰 -Y 轴）正向旋转',
    '-rz': '绕模型 Z 轴（法兰 -Y 轴）负向旋转',
}


def test_tcp_rotation(rtde_c, rtde_r, tcp_offset_pose, speed, acceleration):
    """设置 TCP 偏移后，绕 TCP 点做旋转测试"""
    pose_before = np.array(rtde_r.getActualTCPPose())

    rtde_c.setTcp(tcp_offset_pose)
    print(f"\n已设置 TCP 偏移: {np.round(tcp_offset_pose, 4)}")
    print(f"  位移: x={tcp_offset_pose[0]*1000:.1f}mm, y={tcp_offset_pose[1]*1000:.1f}mm, z={tcp_offset_pose[2]*1000:.1f}mm")
    print(f"  旋转: rx={np.degrees(tcp_offset_pose[3]):.1f}°, ry={np.degrees(tcp_offset_pose[4]):.1f}°, rz={np.degrees(tcp_offset_pose[5]):.1f}°")

    time.sleep(0.2)
    pose_after = np.array(rtde_r.getActualTCPPose())
    print(f"\n  设置后 TCP 位姿: {np.round(pose_after, 4)}")

    angle_deg = 10.0  # 每次旋转角度（度）
    angle_rad = np.radians(angle_deg)

    rot_map = {
        '+rx': np.array([angle_rad, 0, 0]),
        '-rx': np.array([-angle_rad, 0, 0]),
        '+ry': np.array([0, angle_rad, 0]),
        '-ry': np.array([0, -angle_rad, 0]),
        '+rz': np.array([0, 0, angle_rad]),
        '-rz': np.array([0, 0, -angle_rad]),
    }

    while True:
        print(f"\n绕 TCP 旋转测试（每次 {angle_deg}°）")
        print("输入旋转方向（+rx/-rx/+ry/-ry/+rz/-rz），输入 b 返回（会清除TCP偏移）：")
        cmd = input("  > ").strip().lower()
        if cmd == 'b':
            rtde_c.setTcp([0, 0, 0, 0, 0, 0])
            print("  已清除 TCP 偏移")
            break
        if cmd not in rot_map:
            print("  无效输入")
            continue

        model_rot_delta = rot_map[cmd]

        # 获取当前位姿
        tcp = np.array(rtde_r.getActualTCPPose())
        current_pos = tcp[:3]
        current_rotvec = tcp[3:]
        R_current = R.from_rotvec(current_rotvec).as_matrix()

        # 模型旋转轴 -> 法兰旋转轴
        flange_rot_delta = R_m2r @ model_rot_delta

        # 法兰局部坐标系的旋转增量 -> 世界坐标系的旋转增量
        world_rot_delta = R_current @ flange_rot_delta

        # 新的旋转 = 增量旋转 @ 当前旋转
        R_delta_world = R.from_rotvec(world_rot_delta).as_matrix()
        R_target = R_delta_world @ R_current
        target_rotvec = R.from_matrix(R_target).as_rotvec()

        # 位置不变（绕 TCP 点旋转，setTcp 后 moveL 自动保持 TCP 位置）
        target_pose = np.concatenate([current_pos, target_rotvec])

        print(f"  模型旋转:     {np.round(np.degrees(model_rot_delta), 1)}°  （{ROT_EXPECTED[cmd]}）")
        print(f"  法兰旋转:     {np.round(np.degrees(flange_rot_delta), 1)}°")
        print(f"  当前位姿:     {np.round(tcp, 4)}")
        print(f"  目标位姿:     {np.round(target_pose, 4)}")
        input("  按 Enter 执行旋转...")

        rtde_c.moveL(target_pose.tolist(), speed, acceleration)
        time.sleep(0.3)

        actual = np.array(rtde_r.getActualTCPPose())
        print(f"  实际位姿:     {np.round(actual, 4)}")
        print(f"  位置变化:     {np.round(actual[:3] - current_pos, 4)}")

        input("  观察完毕，按 Enter 返回原位...")
        rtde_c.moveL(tcp.tolist(), speed, acceleration)
        time.sleep(0.3)


def move_in_flange_frame(rtde_c, rtde_r, model_delta, cmd, speed, acceleration):
    """沿模型方向移动（通过法兰坐标系转换到世界坐标系）"""
    label = cmd.upper()

    tcp = np.array(rtde_r.getActualTCPPose())
    current_pos = tcp[:3]
    current_rotvec = tcp[3:]

    # 当前TCP旋转矩阵（工具坐标系 -> 世界坐标系）
    R_tcp2world = R.from_rotvec(current_rotvec).as_matrix()

    # 模型位移 -> 法兰坐标系位移
    flange_delta = R_m2r @ model_delta

    # 法兰坐标系位移 -> 世界坐标系位移
    world_delta = R_tcp2world @ flange_delta

    target_pos = current_pos + world_delta
    target_pose = np.concatenate([target_pos, current_rotvec])

    print(f"  模型位移:     {model_delta}  （model {label}）")
    print(f"  法兰坐标位移: {np.round(flange_delta, 4)}  （预期: {EXPECTED[cmd]}）")
    print(f"  世界坐标位移: {np.round(world_delta, 4)}")
    print(f"  当前位置:     {np.round(current_pos, 4)}")
    print(f"  目标位置:     {np.round(target_pos, 4)}")
    input("  按 Enter 执行运动...")

    rtde_c.moveL(target_pose.tolist(), speed, acceleration)
    time.sleep(0.3)

    actual = np.array(rtde_r.getActualTCPPose())
    actual_delta = actual[:3] - current_pos
    print(f"  实际位移:     {np.round(actual_delta, 4)}")

    input("  观察完毕，按 Enter 返回原位...")
    rtde_c.moveL(tcp.tolist(), speed, acceleration)
    time.sleep(0.3)


if __name__ == '__main__':
    main()
