"""
Fake model -> RTDE real robot test.

Goal:
- Simulate a model that outputs 6-DoF pose (x,y,z,rx,ry,rz)
- Send those poses to RTDEInterpolationController to move a real UR robot
- Continuously publish robot state (console print, optional CSV log)

Safety:
- Keep one hand near E-stop.
- Start with small motion amplitude.
"""

import csv
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.precise_sleep import precise_wait
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController


def fake_model_predict_pose6(base_pose6: np.ndarray, t_sec: float, amp_m: float, amp_deg: float) -> np.ndarray:
    """
    A stand-in for "model output 6-DoF pose".
    Returns an absolute target pose around current base pose.
    """
    out = np.array(base_pose6, dtype=np.float64).copy()
    # Position: small smooth oscillation
    out[0] += amp_m * np.sin(2.0 * np.pi * 0.12 * t_sec)

    out[1] += amp_m * 0.5 * np.sin(2.0 * np.pi * 0.08 * t_sec + 0.7)
    # Orientation: small rx oscillation in rotvec space
    out[3] += np.deg2rad(amp_deg) * np.sin(2.0 * np.pi * 0.10 * t_sec)
    
    return out


@click.command()
@click.option("--robot_ip", default="192.168.3.254", help="UR robot IP.")
@click.option("--duration", "-d", default=30.0, type=float, help="Test duration (seconds).")
@click.option("--ctrl_hz", "-f", default=10.0, type=float, help="Control frequency (Hz).")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Command lead time (s).")
@click.option("--amp_m", default=0.01, type=float, help="Position oscillation amplitude (m).")
@click.option("--amp_deg", default=3.0, type=float, help="Rotation oscillation amplitude (deg).")
@click.option("--state_print_hz", default=5.0, type=float, help="Robot state print rate (Hz).")
@click.option("--save_csv", default="", type=str, help="Optional CSV path for state logging.")
@click.option("--init_joints", is_flag=True, default=False, help="Move to preset joint init before test.")
@click.option("--tcp_offset_x", default=0.016, type=float, help="TCP offset x (m).")
@click.option("--tcp_offset_z", default=0.2105, type=float, help="TCP offset z (m).")
@click.option("--tcp_rot_y", default=0.17453, type=float, help="TCP offset ry (rad).")
def main(
    robot_ip,
    duration,
    ctrl_hz,
    command_latency,
    amp_m,
    amp_deg,
    state_print_hz,
    save_csv,
    init_joints,
    tcp_offset_x,
    tcp_offset_z,
    tcp_rot_y,
):
    dt = 1.0 / float(ctrl_hz)
    cube_diag = np.linalg.norm([1.0, 1.0, 1.0])
    max_pos_speed = 2.0
    max_rot_speed = 6.0

    csv_file = None
    csv_writer = None
    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        csv_file = open(save_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "t_sec",
                "target_x",
                "target_y",
                "target_z",
                "target_rx",
                "target_ry",
                "target_rz",
                "actual_x",
                "actual_y",
                "actual_z",
                "actual_rx",
                "actual_ry",
                "actual_rz",
            ]
        )

    print("=== Fake model pose6 -> RTDE 真机测试 ===")
    print(f"robot_ip={robot_ip}, duration={duration}s, ctrl_hz={ctrl_hz}Hz")
    print(f"amp_m={amp_m}, amp_deg={amp_deg}, state_print_hz={state_print_hz}")
    print("请确保急停可达。按 Ctrl+C 可退出。")

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=500,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed * cube_diag,
            max_rot_speed=max_rot_speed * cube_diag,
            tcp_offset_pose=[tcp_offset_x, 0.0, tcp_offset_z, 0.0, tcp_rot_y, 0.0],
            joints_init=np.array([0, -90, -90, -90, 90, 0]) / 180.0 * np.pi if init_joints else None,
            joints_init_speed=1.05,
            verbose=False,
        ) as controller:
            state = controller.get_state()
            base_pose6 = np.array(state["ActualTCPPose"], dtype=np.float64)
            t_start_mono = time.monotonic()
            iter_idx = 0
            last_print_t = 0.0
            print_period = 1.0 / max(float(state_print_hz), 0.1)

            try:
                while True:
                    t_now_mono = time.monotonic()
                    t_elapsed = t_now_mono - t_start_mono
                    if t_elapsed >= duration:
                        break

                    t_cycle_end = t_start_mono + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # 1) "Model output": target 6-DoF pose.
                    target_pose6 = fake_model_predict_pose6(base_pose6, t_elapsed, amp_m, amp_deg)

                    # 2) Send command to RTDE interpolation controller.
                    precise_wait(t_sample, time_func=time.monotonic)
                    schedule_wall_t = t_command_target - time.monotonic() + time.time()
                    controller.schedule_waypoint(target_pose6, schedule_wall_t)

                    # 3) Publish/read robot state.
                    st_now = controller.get_state()
                    actual_pose6 = np.array(st_now["ActualTCPPose"], dtype=np.float64)

                    if (t_elapsed - last_print_t) >= print_period:
                        last_print_t = t_elapsed
                        pos_err_mm = np.linalg.norm(actual_pose6[:3] - target_pose6[:3]) * 1000.0
                        print(
                            f"t={t_elapsed:6.2f}s "
                            f"target=({target_pose6[0]: .4f},{target_pose6[1]: .4f},{target_pose6[2]: .4f}) "
                            f"actual=({actual_pose6[0]: .4f},{actual_pose6[1]: .4f},{actual_pose6[2]: .4f}) "
                            f"pos_err={pos_err_mm: .1f}mm"
                        )

                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                t_elapsed,
                                *target_pose6.tolist(),
                                *actual_pose6.tolist(),
                            ]
                        )

                    precise_wait(t_cycle_end, time_func=time.monotonic)
                    iter_idx += 1
            except KeyboardInterrupt:
                print("Interrupted by user.")

    if csv_file is not None:
        csv_file.close()
        print(f"CSV saved to: {save_csv}")
    print("Test finished.")


if __name__ == "__main__":
    main()

