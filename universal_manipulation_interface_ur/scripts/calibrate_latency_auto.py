"""
自动标定机器人动作延迟（不需要 SpaceMouse）

在当前位置做小幅正弦运动，记录命令时间和实际位置，
通过互相关计算延迟。

用法:
    python scripts/calibrate_latency_auto.py --robot_ip 192.168.3.254
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency


@click.command()
@click.option('--robot_ip', default='192.168.3.254')
@click.option('--frequency', '-f', type=float, default=30)
@click.option('--duration', '-d', type=float, default=5.0, help='Duration of sine motion in seconds')
@click.option('--amplitude', '-a', type=float, default=0.02, help='Amplitude in meters')
@click.option('--sine_freq', type=float, default=0.5, help='Sine frequency in Hz')
@click.option('--tcp_offset_z', type=float, default=0.2105)
def main(robot_ip, frequency, duration, amplitude, sine_freq, tcp_offset_z):
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=0.5,
            max_rot_speed=1.0,
            tcp_offset_pose=[0, 0, tcp_offset_z, 0, 0, 0],
            get_max_k=int(125 * (duration + 5)),
            verbose=False
        ) as controller:
            print(f"Connected to {robot_ip}")
            time.sleep(1.0)

            state = controller.get_state()
            start_pose = state['ActualTCPPose'].copy()
            print(f"Start pose: {start_pose}")
            print(f"Running sine motion: amp={amplitude}m, freq={sine_freq}Hz, duration={duration}s")
            print("Press Ctrl+C to abort")

            t_target = []
            x_target = []

            t_start = time.time()
            iter_idx = 0
            n_steps = int(duration * frequency)

            for iter_idx in range(n_steps):
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_command_target = t_cycle_end + dt

                t_elapsed = t_command_target - t_start
                # z 方向正弦运动
                target_pose = start_pose.copy()
                target_pose[2] += amplitude * np.sin(2 * np.pi * sine_freq * t_elapsed)

                t_target.append(t_command_target)
                x_target.append(target_pose.copy())

                controller.schedule_waypoint(target_pose, t_command_target)
                precise_wait(t_cycle_end, time_func=time.time)

            # 等待运动结束
            print("Motion done, collecting data...")
            time.sleep(1.0)

            # 回到起始位置
            controller.schedule_waypoint(start_pose, time.time() + 1.0)
            time.sleep(1.5)

            states = controller.get_all_state()

    # 分析延迟
    t_target = np.array(t_target)
    x_target = np.array(x_target)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']

    print("\n========== 延迟标定结果 ==========")
    latencies = []
    dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for i in range(6):
        try:
            latency, info = get_latency(
                x_target[..., i], t_target,
                x_actual[..., i], t_actual)
            latencies.append(latency)
            print(f"  {dim_names[i]}: {latency:.4f}s")
        except Exception as e:
            print(f"  {dim_names[i]}: failed ({e})")

    if latencies:
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        print(f"\n  Mean:   {mean_latency:.4f}s")
        print(f"  Median: {median_latency:.4f}s")
        print(f"\n用于 eval_real_pi05.py:")
        print(f"  robot_action_latency = {median_latency:.3f}")

    # 保存图表
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for i in range(6):
            ax = axes[i // 2, i % 2]
            try:
                latency, info = get_latency(
                    x_target[..., i], t_target,
                    x_actual[..., i], t_actual)
                t_samples = info['t_samples'] - info['t_samples'][0]
                ax.plot(t_samples, info['x_target'], label='target', alpha=0.8)
                ax.plot(t_samples - latency, info['x_actual'], label=f'actual (lag={latency:.3f}s)', alpha=0.8)
                ax.legend(fontsize=8)
            except:
                pass
            ax.set_title(f'{dim_names[i]}')
            ax.set_xlabel('time (s)')

        fig.suptitle(f'Robot Action Latency Calibration (median={median_latency:.3f}s)')
        fig.tight_layout()
        out_path = '/tmp/robot_latency_calibration.png'
        plt.savefig(out_path, dpi=100)
        print(f"\n图表已保存: {out_path}")
    except Exception as e:
        print(f"图表保存失败: {e}")


if __name__ == '__main__':
    main()
