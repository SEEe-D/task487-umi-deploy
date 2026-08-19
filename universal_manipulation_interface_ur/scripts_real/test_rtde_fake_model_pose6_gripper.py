"""
Fake model -> RTDE real robot + gripper test.

Goal:
- Simulate a model that outputs 6-DoF pose and gripper angle.
- Send pose to RTDEInterpolationController.
- Send gripper width to Livelybot gripper controller.
- Continuously print robot + gripper state (optional CSV log).

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
from umi.real_world.livelybot_gripper_controller import LivelybotGripperController
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController


def fake_model_predict_pose6(
    base_pose6: np.ndarray, t_sec: float, amp_m: float, amp_deg: float
) -> np.ndarray:
    """A stand-in for model output (absolute pose around current base pose)."""
    out = np.array(base_pose6, dtype=np.float64).copy()
    out[0] += amp_m * np.sin(2.0 * np.pi * 0.12 * t_sec)
    out[1] += amp_m * 0.5 * np.sin(2.0 * np.pi * 0.08 * t_sec + 0.7)
    out[3] += np.deg2rad(amp_deg) * np.sin(2.0 * np.pi * 0.10 * t_sec)
    return out


def fake_model_predict_gripper(
    t_sec: float,
    gripper_open_deg: float,
    gripper_closed_deg: float,
    wave_hz: float,
) -> float:
    """Generate smooth open-close gripper command in output degrees."""
    low = min(float(gripper_open_deg), float(gripper_closed_deg))
    high = max(float(gripper_open_deg), float(gripper_closed_deg))
    mid = 0.5 * (low + high)
    amp = 0.5 * (high - low)
    gripper_deg = mid + amp * np.sin(2.0 * np.pi * float(wave_hz) * float(t_sec))
    return float(np.clip(gripper_deg, low, high))


@click.command()
@click.option("--robot_ip", default="192.168.3.254", help="UR robot IP.")
@click.option("--duration", "-d", default=30.0, type=float, help="Test duration (seconds).")
@click.option("--ctrl_hz", "-f", default=10.0, type=float, help="Control frequency (Hz).")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Command lead time (s).")
@click.option("--amp_m", default=0.01, type=float, help="Position oscillation amplitude (m).")
@click.option("--amp_deg", default=3.0, type=float, help="Rotation oscillation amplitude (deg).")
@click.option("--state_print_hz", default=5.0, type=float, help="State print rate (Hz).")
@click.option("--save_csv", default="", type=str, help="Optional CSV path for state logging.")
@click.option("--init_joints", is_flag=True, default=False, help="Move to preset joint init before test.")
@click.option("--tcp_offset_x", default=0.016, type=float, help="TCP offset x (m).")
@click.option("--tcp_offset_z", default=0.2105, type=float, help="TCP offset z (m).")
@click.option("--tcp_rot_y", default=0.17453, type=float, help="TCP offset ry (rad).")
@click.option("--gripper_open_deg", default=35.0, type=float, help="Gripper output angle at fully open.")
@click.option("--gripper_closed_deg", default=0.0, type=float, help="Gripper output angle at fully closed.")
@click.option("--gripper_wave_hz", default=0.10, type=float, help="Open/close sine frequency (Hz).")
@click.option(
    "--gripper_executable_path",
    default="x3arm_can/build_ws/x3arm-can-demo-gripper",
    help="Path to livelybot gripper daemon executable.",
)
@click.option("--gripper_can_if", default="can3")
@click.option("--gripper_device_id", default=8, type=int)
@click.option("--gripper_kp", default=10.0, type=float)
@click.option("--gripper_kd", default=1.0, type=float)
@click.option("--gripper_target_vel_deg", default=0.0, type=float)
@click.option("--gripper_torque_nm", default=0.0, type=float)
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
    gripper_open_deg,
    gripper_closed_deg,
    gripper_wave_hz,
    gripper_executable_path,
    gripper_can_if,
    gripper_device_id,
    gripper_kp,
    gripper_kd,
    gripper_target_vel_deg,
    gripper_torque_nm,
):
    dt = 1.0 / float(ctrl_hz)
    cube_diag = np.linalg.norm([1.0, 1.0, 1.0])
    max_pos_speed = 2.0
    max_rot_speed = 6.0

    if gripper_open_deg <= gripper_closed_deg:
        raise ValueError("gripper_open_deg must be > gripper_closed_deg")
    if gripper_wave_hz < 0:
        raise ValueError("gripper_wave_hz must be >= 0")

    csv_file = None
    csv_writer = None
    if save_csv:
        save_csv_abs = os.path.abspath(save_csv)
        os.makedirs(os.path.dirname(save_csv_abs), exist_ok=True)
        csv_file = open(save_csv_abs, "w", newline="", encoding="utf-8")
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
                "target_gripper_deg",
                "actual_gripper_deg",
                "actual_gripper_vel_dps",
                "actual_gripper_force",
            ]
        )

    print("=== Fake model pose6+gripper -> RTDE 真机测试 ===")
    print(
        f"robot_ip={robot_ip}, gripper_type=livelybot, "
        f"duration={duration}s, ctrl_hz={ctrl_hz}Hz"
    )
    print(
        f"amp_m={amp_m}, amp_deg={amp_deg}, gripper_wave_hz={gripper_wave_hz}, "
        f"gripper_range=[{gripper_closed_deg:.1f}, {gripper_open_deg:.1f}]deg"
    )
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
            gripper = LivelybotGripperController(
                shm_manager=shm_manager,
                executable_path=gripper_executable_path,
                can_if=gripper_can_if,
                device_id=gripper_device_id,
                receive_latency=0.01,
                deg_open=gripper_open_deg,
                deg_closed=gripper_closed_deg,
                kp=gripper_kp,
                kd=gripper_kd,
                target_vel_deg=gripper_target_vel_deg,
                torque_nm=gripper_torque_nm,
            )

            with gripper:
                state = controller.get_state()
                base_pose6 = np.array(state["ActualTCPPose"], dtype=np.float64)
                t_start_mono = time.monotonic()
                iter_idx = 0
                last_print_t = -1e9
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
                        schedule_wall_t = t_command_target - time.monotonic() + time.time()

                        target_pose6 = fake_model_predict_pose6(
                            base_pose6, t_elapsed, amp_m, amp_deg
                        )
                        target_gripper_deg = fake_model_predict_gripper(
                            t_elapsed,
                            gripper_open_deg=gripper_open_deg,
                            gripper_closed_deg=gripper_closed_deg,
                            wave_hz=gripper_wave_hz,
                        )

                        precise_wait(t_sample, time_func=time.monotonic)
                        controller.schedule_waypoint(target_pose6, schedule_wall_t)
                        gripper.schedule_waypoint(target_gripper_deg, schedule_wall_t)

                        st_now = controller.get_state()
                        grip_now = gripper.get_state()
                        actual_pose6 = np.array(st_now["ActualTCPPose"], dtype=np.float64)
                        actual_gripper_deg = float(grip_now["gripper_position"])
                        actual_gripper_vel = float(grip_now["gripper_velocity"])
                        actual_gripper_force = float(grip_now["gripper_force"])

                        if (t_elapsed - last_print_t) >= print_period:
                            last_print_t = t_elapsed
                            pos_err_mm = np.linalg.norm(actual_pose6[:3] - target_pose6[:3]) * 1000.0
                            grip_err_deg = abs(actual_gripper_deg - target_gripper_deg)
                            print(
                                f"t={t_elapsed:6.2f}s "
                                f"pos_err={pos_err_mm: .1f}mm "
                                f"grip_target={target_gripper_deg: .2f}deg "
                                f"grip_actual={actual_gripper_deg: .2f}deg "
                                f"grip_err={grip_err_deg: .2f}deg"
                            )

                        if csv_writer is not None:
                            csv_writer.writerow(
                                [
                                    t_elapsed,
                                    *target_pose6.tolist(),
                                    *actual_pose6.tolist(),
                                    target_gripper_deg,
                                    actual_gripper_deg,
                                    actual_gripper_vel,
                                    actual_gripper_force,
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
