"""UR7e RTDE 连接 + 运动测试

独立测试脚本，不依赖 policy server。
验证:
  1. RTDE 连接是否通
  2. 读取 TCP 位姿 + 关节角
  3. 多进程 servoL 控制器是否正常
  4. 执行一段简单的 TCP 轨迹 (XY 平面画方形/圆形)

安全设计:
  - 所有运动都基于当前位姿做微小偏移 (默认 ±3cm)
  - 每一步都有确认提示
  - Ctrl+C 随时停止

用法:
    # 1. 仅测试连接 (不运动)
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test connect

    # 2. 读取状态循环
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test read

    # 3. 单进程 servoL 画方形 (最简单, 排查硬件)
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test square

    # 4. 单进程 servoL 画圆形
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test circle

    # 5. 多进程控制器 + 画方形 (测试完整部署架构)
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test mp-square

    # 调整运动幅度
    python test_ur7e_rtde.py --robot-ip 192.168.1.100 --test square --amplitude 0.05
"""

import argparse
import logging
import math
import signal
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

# 全局停止标志
_stop = False


def signal_handler(sig, frame):
    global _stop
    if _stop:
        sys.exit(1)
    print("\n[!] Ctrl+C, stopping...")
    _stop = True


signal.signal(signal.SIGINT, signal_handler)


# ============== Test 1: 连接测试 ==============

def test_connect(robot_ip: str):
    """测试 RTDE 连接, 读取一次状态"""
    print(f"\n{'='*60}")
    print(f"Test: RTDE Connection")
    print(f"Robot IP: {robot_ip}")
    print(f"{'='*60}\n")

    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("[ERROR] ur_rtde not installed!")
        print("  pip install ur_rtde")
        return False

    print("[1/4] Connecting RTDEReceiveInterface...")
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print("[2/4] Connecting RTDEControlInterface...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        rtde_r.disconnect()
        return False

    print("[3/4] Reading robot state...")
    try:
        tcp_pose = rtde_r.getActualTCPPose()
        joints = rtde_r.getActualQ()
        speed = rtde_r.getActualTCPSpeed()
        robot_mode = rtde_r.getRobotMode()
        safety_mode = rtde_r.getSafetyMode()
        print(f"  Robot mode:  {robot_mode}")
        print(f"  Safety mode: {safety_mode}")
        print(f"  TCP pose:    [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}, "
              f"{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}]")
        print(f"  TCP pos:     x={tcp_pose[0]:.4f}m, y={tcp_pose[1]:.4f}m, z={tcp_pose[2]:.4f}m")
        print(f"  TCP speed:   [{speed[0]:.4f}, {speed[1]:.4f}, {speed[2]:.4f}] m/s")
        print(f"  Joints (deg): [{', '.join(f'{math.degrees(j):.1f}' for j in joints)}]")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print("[4/4] Checking control mode...")
    try:
        is_connected = rtde_c.isConnected()
        print(f"  Control connected: {is_connected}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 断开
    rtde_c.disconnect()
    rtde_r.disconnect()

    print(f"\n{'='*60}")
    print("RTDE connection test: PASSED")
    print(f"{'='*60}")
    return True


# ============== Test 2: 持续读取状态 ==============

def test_read(robot_ip: str, duration: float = 10.0):
    """持续读取并打印 TCP 状态"""
    print(f"\n{'='*60}")
    print(f"Test: Continuous State Reading ({duration}s)")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    t_start = time.time()
    count = 0

    try:
        while not _stop and (time.time() - t_start) < duration:
            tcp = rtde_r.getActualTCPPose()
            joints = rtde_r.getActualQ()
            t_elapsed = time.time() - t_start

            print(
                f"\r[{t_elapsed:6.2f}s] "
                f"TCP: x={tcp[0]:+.4f} y={tcp[1]:+.4f} z={tcp[2]:+.4f} | "
                f"rot: [{tcp[3]:+.3f}, {tcp[4]:+.3f}, {tcp[5]:+.3f}]",
                end="", flush=True,
            )
            count += 1
            time.sleep(0.05)  # 20Hz 打印
    finally:
        rtde_r.disconnect()
        print(f"\n\nRead {count} samples in {time.time() - t_start:.1f}s")


# ============== Test 3: 单进程 servoL 画方形 ==============

def test_square(robot_ip: str, amplitude: float = 0.03, speed: float = 0.3):
    """在当前位置的 XY 平面画一个方形 (单进程 servoL)

    Args:
        amplitude: 方形边长的一半 (m), 默认 3cm
        speed: 运动速度 (m/s)
    """
    print(f"\n{'='*60}")
    print(f"Test: Single-process servoL Square")
    print(f"Amplitude: ±{amplitude*100:.1f}cm in XY plane")
    print(f"{'='*60}\n")

    import rtde_control
    import rtde_receive

    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)

    # 读取当前位姿作为中心点
    center = np.array(rtde_r.getActualTCPPose())
    print(f"Center pose: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Will move ±{amplitude*100:.1f}cm around this position")

    ans = input("\nConfirm start? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        rtde_r.disconnect()
        rtde_c.disconnect()
        return

    # 方形顶点 (相对于 center, XY 平面)
    corners = [
        np.array([+amplitude, +amplitude, 0, 0, 0, 0]),
        np.array([-amplitude, +amplitude, 0, 0, 0, 0]),
        np.array([-amplitude, -amplitude, 0, 0, 0, 0]),
        np.array([+amplitude, -amplitude, 0, 0, 0, 0]),
    ]

    servo_dt = 0.002  # 500Hz
    lookahead = 0.1
    gain = 300
    n_laps = 3

    print(f"\nDrawing {n_laps} laps...")

    try:
        for lap in range(n_laps):
            if _stop:
                break
            for i, offset in enumerate(corners):
                if _stop:
                    break

                target = center + offset
                current = np.array(rtde_r.getActualTCPPose())
                distance = np.linalg.norm(target[:3] - current[:3])
                n_steps = max(int(distance / speed / servo_dt), 10)

                print(f"  Lap {lap+1}/{n_laps}, corner {i+1}/4, "
                      f"dist={distance*100:.1f}cm, steps={n_steps}")

                for step in range(n_steps):
                    if _stop:
                        break
                    alpha = (step + 1) / n_steps
                    interp = current + alpha * (target - current)
                    rtde_c.servoL(interp.tolist(), 0, 0, servo_dt, lookahead, gain)
                    time.sleep(servo_dt)

    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        print("\nStopping servo...")
        rtde_c.servoStop()
        time.sleep(0.5)

        # 回到中心
        print("Returning to center...")
        rtde_c.moveL(center.tolist(), speed, 0.5)
        time.sleep(1.0)

        rtde_c.stopScript()
        rtde_r.disconnect()
        rtde_c.disconnect()
        print("Done.")


# ============== Test 4: 单进程 servoL 画圆 ==============

def test_circle(robot_ip: str, amplitude: float = 0.03, speed: float = 0.3):
    """在当前位置的 XY 平面画圆 (单进程 servoL)"""
    print(f"\n{'='*60}")
    print(f"Test: Single-process servoL Circle")
    print(f"Radius: {amplitude*100:.1f}cm in XY plane")
    print(f"{'='*60}\n")

    import rtde_control
    import rtde_receive

    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)

    center = np.array(rtde_r.getActualTCPPose())
    print(f"Center pose: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Will draw circle with radius {amplitude*100:.1f}cm")

    ans = input("\nConfirm start? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        rtde_r.disconnect()
        rtde_c.disconnect()
        return

    servo_dt = 0.002  # 500Hz
    lookahead = 0.1
    gain = 300
    n_laps = 3
    circumference = 2 * math.pi * amplitude
    total_time_per_lap = circumference / speed
    n_steps_per_lap = int(total_time_per_lap / servo_dt)

    print(f"\nDrawing {n_laps} laps ({n_steps_per_lap} steps/lap, "
          f"{total_time_per_lap:.1f}s/lap)...")

    try:
        for lap in range(n_laps):
            if _stop:
                break
            print(f"  Lap {lap+1}/{n_laps}...")
            for step in range(n_steps_per_lap):
                if _stop:
                    break
                theta = 2 * math.pi * step / n_steps_per_lap
                target = center.copy()
                target[0] += amplitude * math.cos(theta)
                target[1] += amplitude * math.sin(theta)
                rtde_c.servoL(target.tolist(), 0, 0, servo_dt, lookahead, gain)
                time.sleep(servo_dt)

    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        print("\nStopping servo...")
        rtde_c.servoStop()
        time.sleep(0.5)

        print("Returning to center...")
        rtde_c.moveL(center.tolist(), speed, 0.5)
        time.sleep(1.0)

        rtde_c.stopScript()
        rtde_r.disconnect()
        rtde_c.disconnect()
        print("Done.")


# ============== Test 5: 多进程控制器画方形 ==============

def test_mp_square(robot_ip: str, amplitude: float = 0.03):
    """使用多进程 RTDEInterpolationController 画方形

    测试完整的部署架构: 主进程调度 waypoint, 控制器进程 500Hz servoL。
    """
    print(f"\n{'='*60}")
    print(f"Test: Multi-process Controller Square")
    print(f"Amplitude: ±{amplitude*100:.1f}cm in XY plane")
    print(f"{'='*60}\n")

    from ur7e_controller import RTDEInterpolationController

    print("Starting RTDEInterpolationController...")
    controller = RTDEInterpolationController(
        robot_ip=robot_ip,
        frequency=500.0,
        lookahead_time=0.1,
        gain=300,
        max_pos_speed=0.25,
        max_rot_speed=0.16,
    )
    controller.start(wait=True)

    state = controller.get_state()
    center = state["tcp_pose"].copy()
    print(f"Center pose: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Will move ±{amplitude*100:.1f}cm around this position")

    ans = input("\nConfirm start? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        controller.stop()
        return

    # 方形顶点
    corners = [
        center + np.array([+amplitude, +amplitude, 0, 0, 0, 0]),
        center + np.array([-amplitude, +amplitude, 0, 0, 0, 0]),
        center + np.array([-amplitude, -amplitude, 0, 0, 0, 0]),
        center + np.array([+amplitude, -amplitude, 0, 0, 0, 0]),
    ]

    move_duration = 1.0  # 每条边 1 秒
    n_laps = 3

    try:
        for lap in range(n_laps):
            if _stop:
                break
            for i, target in enumerate(corners):
                if _stop:
                    break

                t_target = time.monotonic() + move_duration
                controller.schedule_waypoint(target, t_target)

                state = controller.get_state()
                print(
                    f"  Lap {lap+1}/{n_laps}, corner {i+1}/4 → "
                    f"[{target[0]:.4f}, {target[1]:.4f}], "
                    f"current: [{state['tcp_pose'][0]:.4f}, {state['tcp_pose'][1]:.4f}]"
                )
                time.sleep(move_duration)

        # 回到中心
        if not _stop:
            print("\nReturning to center...")
            controller.servoL(center, duration=2.0)
            time.sleep(2.5)

    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        controller.stop()
        print("Done.")


# ============== Test 6: 多进程控制器画圆 ==============

def test_mp_circle(robot_ip: str, amplitude: float = 0.03):
    """使用多进程 RTDEInterpolationController 画圆

    以 10Hz 调度 waypoint (模拟 policy 频率), 控制器 500Hz 插值执行。
    """
    print(f"\n{'='*60}")
    print(f"Test: Multi-process Controller Circle (10Hz scheduling)")
    print(f"Radius: {amplitude*100:.1f}cm in XY plane")
    print(f"{'='*60}\n")

    from ur7e_controller import RTDEInterpolationController

    print("Starting RTDEInterpolationController...")
    controller = RTDEInterpolationController(
        robot_ip=robot_ip,
        frequency=500.0,
        lookahead_time=0.1,
        gain=300,
    )
    controller.start(wait=True)

    state = controller.get_state()
    center = state["tcp_pose"].copy()
    print(f"Center pose: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Will draw circle with radius {amplitude*100:.1f}cm")

    ans = input("\nConfirm start? [y/N] ").strip().lower()
    if ans != "y":
        controller.stop()
        return

    schedule_freq = 10.0  # 模拟 policy 频率
    schedule_dt = 1.0 / schedule_freq
    circle_period = 3.0  # 一圈 3 秒
    n_laps = 3
    total_steps = int(n_laps * circle_period * schedule_freq)

    print(f"\nScheduling at {schedule_freq}Hz, {n_laps} laps "
          f"({total_steps} waypoints, {circle_period}s/lap)...")

    try:
        t_start = time.monotonic()
        for step in range(total_steps):
            if _stop:
                break

            t_now = time.monotonic()
            t_target = t_start + (step + 1) * schedule_dt
            theta = 2 * math.pi * (step * schedule_dt) / circle_period

            target = center.copy()
            target[0] += amplitude * math.cos(theta)
            target[1] += amplitude * math.sin(theta)

            controller.schedule_waypoint(target, t_target)

            # 打印当前状态
            if step % 10 == 0:
                state = controller.get_state()
                actual = state["tcp_pose"]
                err = np.linalg.norm(actual[:3] - target[:3]) * 1000
                print(
                    f"  Step {step:4d}/{total_steps} | "
                    f"target: [{target[0]:.4f}, {target[1]:.4f}] | "
                    f"actual: [{actual[0]:.4f}, {actual[1]:.4f}] | "
                    f"err: {err:.1f}mm"
                )

            # 等待到下一个调度时间
            t_sleep = t_target - time.monotonic()
            if t_sleep > 0:
                time.sleep(t_sleep)

        # 回到中心
        if not _stop:
            print("\nReturning to center...")
            controller.servoL(center, duration=2.0)
            time.sleep(2.5)

    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        controller.stop()
        print("Done.")


# ============== Main ==============

TESTS = {
    "connect": ("RTDE 连接测试 (不运动)", test_connect),
    "read": ("持续读取状态", test_read),
    "square": ("单进程 servoL 画方形", test_square),
    "circle": ("单进程 servoL 画圆", test_circle),
    "mp-square": ("多进程控制器画方形", test_mp_square),
    "mp-circle": ("多进程控制器画圆 (模拟 10Hz policy)", test_mp_circle),
}


def main():
    parser = argparse.ArgumentParser(
        description="UR7e RTDE Connection & Motion Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available tests:\n" + "\n".join(
            f"  {name:12s}  {desc}" for name, (desc, _) in TESTS.items()
        ),
    )
    parser.add_argument("--robot-ip", type=str, required=True,
                        help="UR7e robot IP address")
    parser.add_argument("--test", type=str, default="connect",
                        choices=list(TESTS.keys()),
                        help="Test to run (default: connect)")
    parser.add_argument("--amplitude", type=float, default=0.03,
                        help="Motion amplitude in meters (default: 0.03 = 3cm)")
    args = parser.parse_args()

    desc, test_fn = TESTS[args.test]
    print(f"\nUR7e RTDE Test: {desc}")
    print(f"Robot IP: {args.robot_ip}")

    # 根据测试类型传参
    if args.test == "connect":
        test_fn(args.robot_ip)
    elif args.test == "read":
        test_fn(args.robot_ip)
    elif args.test in ("square", "circle"):
        test_fn(args.robot_ip, amplitude=args.amplitude)
    elif args.test in ("mp-square", "mp-circle"):
        test_fn(args.robot_ip, amplitude=args.amplitude)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )
    main()
