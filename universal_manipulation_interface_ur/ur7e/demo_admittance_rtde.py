"""
Minimal Cartesian admittance control demo for UR7e via RTDE.

Control law (translation only):
    x_dot = (F - K*x) / D
where:
    x      : displacement from start pose (m)
    x_dot  : Cartesian velocity command (m/s)
    F      : external force after bias removal (N)
    K, D   : virtual stiffness/damping

Safety notes:
- Start with small gains and low max velocity.
- Keep one hand near E-stop.
- This demo is for quick validation, not production safety logic.
"""

import os
import sys
import time

import click
import numpy as np
import scipy.spatial.transform as st

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.precise_sleep import precise_wait

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except Exception as exc:  # pragma: no cover
    RTDEControlInterface = None
    RTDEReceiveInterface = None
    IMPORT_ERR = exc
else:
    IMPORT_ERR = None


def estimate_wrench_bias(rtde_r, seconds: float, sample_hz: float) -> np.ndarray:
    dt = 1.0 / max(sample_hz, 1e-6)
    t0 = time.monotonic()
    samples = []
    idx = 0
    while True:
        now = time.monotonic()
        if now - t0 >= seconds:
            break
        samples.append(np.array(rtde_r.getActualTCPForce(), dtype=np.float64))
        idx += 1
        precise_wait(t0 + idx * dt, time_func=time.monotonic)
    if len(samples) == 0:
        return np.zeros(6, dtype=np.float64)
    return np.mean(np.stack(samples, axis=0), axis=0)


def deadband(vec: np.ndarray, threshold: float) -> np.ndarray:
    out = vec.copy()
    mask = np.abs(out) < threshold
    out[mask] = 0.0
    return out


@click.command()
@click.option("--robot_ip", default="192.168.0.9", help="UR controller IP.")
@click.option("--ctrl_hz", default=125.0, type=float, help="Control frequency (Hz).")
@click.option("--print_hz", default=10.0, type=float, help="Print frequency (Hz).")
@click.option("--duration", default=0.0, type=float, help="Run time in seconds, 0 means until Ctrl+C.")
@click.option("--zero_seconds", default=1.5, type=float, help="Force bias calibration time (s).")
@click.option("--ema_alpha", default=0.15, type=float, help="Force EMA alpha in [0,1], 0 disables.")
@click.option("--deadband_n", default=2.0, type=float, help="Force deadband (N).")
@click.option("--k_n_per_m", default=80.0, type=float, help="Virtual stiffness K (N/m).")
@click.option("--d_n_s_per_m", default=120.0, type=float, help="Virtual damping D (N*s/m).")
@click.option("--max_vel", default=0.06, type=float, help="Max Cartesian speed (m/s).")
@click.option("--max_disp", default=0.08, type=float, help="Max displacement from start pose (m).")
@click.option("--lookahead_time", default=0.1, type=float, help="UR servoL lookahead_time.")
@click.option("--gain", default=300, type=int, help="UR servoL gain [100, 2000].")
@click.option("--tool_force_frame", is_flag=True, default=False, help="Use tool frame force directly.")
@click.option(
    "--force_sign",
    default=-1,
    type=click.Choice(["-1", "1"]),
    help="Force direction sign. Use -1 if robot moves opposite to your push.",
)
def main(
    robot_ip,
    ctrl_hz,
    print_hz,
    duration,
    zero_seconds,
    ema_alpha,
    deadband_n,
    k_n_per_m,
    d_n_s_per_m,
    max_vel,
    max_disp,
    lookahead_time,
    gain,
    tool_force_frame,
    force_sign,
):
    if RTDEControlInterface is None or RTDEReceiveInterface is None:
        raise RuntimeError(
            f"Failed to import ur_rtde modules. Original error: {IMPORT_ERR}"
        )

    ctrl_hz = float(ctrl_hz)
    print_hz = float(print_hz)
    duration = float(duration)
    zero_seconds = float(zero_seconds)
    ema_alpha = float(ema_alpha)
    deadband_n = float(deadband_n)
    k_n_per_m = float(k_n_per_m)
    d_n_s_per_m = float(d_n_s_per_m)
    max_vel = float(max_vel)
    max_disp = float(max_disp)
    lookahead_time = float(lookahead_time)
    gain = int(gain)
    force_sign = int(force_sign)

    if ctrl_hz <= 0:
        raise ValueError("ctrl_hz must be > 0")
    if print_hz <= 0:
        raise ValueError("print_hz must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")
    if zero_seconds < 0:
        raise ValueError("zero_seconds must be >= 0")
    if not (0.0 <= ema_alpha <= 1.0):
        raise ValueError("ema_alpha must be in [0,1]")
    if deadband_n < 0:
        raise ValueError("deadband_n must be >= 0")
    if k_n_per_m <= 0:
        raise ValueError("k_n_per_m must be > 0")
    if d_n_s_per_m <= 0:
        raise ValueError("d_n_s_per_m must be > 0")
    if max_vel <= 0:
        raise ValueError("max_vel must be > 0")
    if max_disp <= 0:
        raise ValueError("max_disp must be > 0")
    if not (0.03 <= lookahead_time <= 0.2):
        raise ValueError("lookahead_time must be in [0.03, 0.2]")
    if not (100 <= gain <= 2000):
        raise ValueError("gain must be in [100, 2000]")
    if force_sign not in (-1, 1):
        raise ValueError("force_sign must be -1 or 1")

    dt = 1.0 / ctrl_hz
    print_period = 1.0 / print_hz
    vel = 0.5  # ignored by UR e-series servoL
    acc = 0.5  # ignored by UR e-series servoL

    print("=== UR7e Cartesian admittance demo (translation only) ===")
    print(
        f"robot_ip={robot_ip} ctrl_hz={ctrl_hz}Hz K={k_n_per_m}N/m D={d_n_s_per_m}N*s/m "
        f"max_vel={max_vel}m/s max_disp={max_disp}m force_sign={force_sign}"
    )
    print("Keep E-stop reachable. Press Ctrl+C to stop.")

    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    base_pose = np.array(rtde_r.getActualTCPPose(), dtype=np.float64)
    base_pos = base_pose[:3].copy()
    base_rotvec = base_pose[3:].copy()

    bias = np.zeros(6, dtype=np.float64)
    if zero_seconds > 0:
        print(f"Calibrating wrench bias for {zero_seconds:.2f}s ... keep robot static.")
        bias = estimate_wrench_bias(rtde_r, seconds=zero_seconds, sample_hz=min(ctrl_hz, 125.0))
        print("Bias:", np.array2string(bias, precision=4, suppress_small=True))

    disp = np.zeros(3, dtype=np.float64)
    f_filtered = None
    t_start = time.monotonic()
    last_print_t = -1e9
    iter_idx = 0

    try:
        while True:
            now = time.monotonic()
            t_elapsed = now - t_start
            if duration > 0 and t_elapsed >= duration:
                break

            actual_pose = np.array(rtde_r.getActualTCPPose(), dtype=np.float64)
            wrench_raw = np.array(rtde_r.getActualTCPForce(), dtype=np.float64)
            wrench = wrench_raw - bias

            f_tool = force_sign * wrench[:3]
            if tool_force_frame:
                f_base = f_tool
            else:
                rotm = st.Rotation.from_rotvec(actual_pose[3:]).as_matrix()
                f_base = rotm @ f_tool

            if ema_alpha > 0:
                if f_filtered is None:
                    f_filtered = f_base.copy()
                else:
                    f_filtered = ema_alpha * f_base + (1.0 - ema_alpha) * f_filtered
                f_used = f_filtered
            else:
                f_used = f_base

            f_used = deadband(f_used, deadband_n)

            v_cmd = (f_used - k_n_per_m * disp) / d_n_s_per_m
            v_norm = np.linalg.norm(v_cmd)
            if v_norm > max_vel:
                v_cmd = v_cmd / (v_norm + 1e-12) * max_vel

            disp = disp + v_cmd * dt
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                disp = disp / (disp_norm + 1e-12) * max_disp

            target_pose = np.zeros(6, dtype=np.float64)
            target_pose[:3] = base_pos + disp
            target_pose[3:] = base_rotvec

            ok = rtde_c.servoL(target_pose.tolist(), vel, acc, dt, lookahead_time, gain)
            if not ok:
                print("servoL returned False, stopping.")
                break

            if (t_elapsed - last_print_t) >= print_period:
                last_print_t = t_elapsed
                print(
                    f"t={t_elapsed:7.3f}s "
                    f"F=({f_used[0]: .2f},{f_used[1]: .2f},{f_used[2]: .2f})N "
                    f"disp=({disp[0]: .4f},{disp[1]: .4f},{disp[2]: .4f})m "
                    f"|v|={np.linalg.norm(v_cmd): .3f}m/s"
                )

            iter_idx += 1
            precise_wait(t_start + iter_idx * dt, time_func=time.monotonic)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        rtde_c.servoStop()
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Controller stopped and disconnected.")


if __name__ == "__main__":
    main()
