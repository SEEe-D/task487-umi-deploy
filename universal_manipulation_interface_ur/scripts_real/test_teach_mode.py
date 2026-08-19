"""
测试 UR 机器人 teachMode（自由拖动/freedrive）功能。

用法：
    python scripts_real/test_teach_mode.py --robot_ip 192.168.3.254

按键：
    t - 切换 freedrive 模式（可自由拖动机械臂）
    q - 退出
"""

import time
import click
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


@click.command()
@click.option('--robot_ip', default='192.168.3.254', type=str)
def main(robot_ip):
    print(f"连接机器人 {robot_ip} ...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("连接成功")

    in_teach = False

    print("\n按键说明：")
    print("  t - 切换 freedrive（自由拖动）")
    print("  q - 退出")

    try:
        while True:
            # 显示当前 TCP 位姿
            tcp = np.array(rtde_r.getActualTCPPose())
            status = "FREEDRIVE" if in_teach else "LOCKED"
            print(f"\r[{status}] TCP: {np.round(tcp, 4)}", end="", flush=True)

            cmd = input("\n  > ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 't':
                if not in_teach:
                    ret = rtde_c.teachMode()
                    print(f"  teachMode() 返回: {ret}")
                    if ret:
                        in_teach = True
                        print("  ✓ FREEDRIVE 已开启，可以拖动机械臂")
                    else:
                        print("  ✗ teachMode() 调用失败！")
                else:
                    ret = rtde_c.endTeachMode()
                    print(f"  endTeachMode() 返回: {ret}")
                    in_teach = False
                    print("  ✓ FREEDRIVE 已关闭，机械臂已锁定")
            else:
                print("  无效输入，请输入 t 或 q")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        if in_teach:
            rtde_c.endTeachMode()
            print("已关闭 freedrive")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("已断开连接")


if __name__ == '__main__':
    main()
