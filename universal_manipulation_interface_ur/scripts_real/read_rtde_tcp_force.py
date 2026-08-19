"""
Real-time UR TCP force/torque monitor via RTDE.

Features:
- Read 6D wrench from UR e-series internal force sensor:
  [Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm)
- Optional startup zero-bias calibration
- Optional exponential moving average (EMA) filtering
- Optional CSV logging
"""

import csv
import os
import sys
import time

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.precise_sleep import precise_wait

try:
    from rtde_receive import RTDEReceiveInterface
except Exception as exc:  # pragma: no cover
    RTDEReceiveInterface = None
    IMPORT_ERR = exc
else:
    IMPORT_ERR = None


def estimate_bias(rtde_r, seconds: float, sample_hz: float) -> np.ndarray:
    """Estimate force/torque zero-bias by averaging for a short duration."""
    dt = 1.0 / max(sample_hz, 1e-6)
    t_start = time.monotonic()
    samples = []
    idx = 0
    while True:
        now = time.monotonic()
        if now - t_start >= seconds:
            break
        wrench = np.array(rtde_r.getActualTCPForce(), dtype=np.float64)
        samples.append(wrench)
        idx += 1
        precise_wait(t_start + idx * dt, time_func=time.monotonic)
    if len(samples) == 0:
        return np.zeros(6, dtype=np.float64)
    return np.mean(np.stack(samples, axis=0), axis=0)


@click.command()
@click.option("--robot_ip", default="192.168.0.9", help="UR controller IP.")
@click.option("--read_hz", default=125.0, type=float, help="RTDE read frequency (Hz).")
@click.option("--print_hz", default=10.0, type=float, help="Console print frequency (Hz).")
@click.option("--ema_alpha", default=0.2, type=float, help="EMA alpha in [0,1], 0 disables filter.")
@click.option("--zero_seconds", default=1.5, type=float, help="Startup bias calibration duration (s).")
@click.option("--duration", default=0.0, type=float, help="Total run time (s), 0 means run until Ctrl+C.")
@click.option("--save_csv", default="", type=str, help="Optional CSV path for logging.")
def main(robot_ip, read_hz, print_hz, ema_alpha, zero_seconds, duration, save_csv):
    if RTDEReceiveInterface is None:
        raise RuntimeError(
            f"Failed to import rtde_receive (ur_rtde). Original error: {IMPORT_ERR}"
        )

    read_hz = float(read_hz)
    print_hz = float(print_hz)
    ema_alpha = float(ema_alpha)
    zero_seconds = float(zero_seconds)
    duration = float(duration)

    if read_hz <= 0:
        raise ValueError("read_hz must be > 0")
    if print_hz <= 0:
        raise ValueError("print_hz must be > 0")
    if not (0.0 <= ema_alpha <= 1.0):
        raise ValueError("ema_alpha must be in [0, 1]")
    if zero_seconds < 0:
        raise ValueError("zero_seconds must be >= 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")

    csv_file = None
    csv_writer = None
    if save_csv:
        save_csv_abs = os.path.abspath(save_csv)
        os.makedirs(os.path.dirname(save_csv_abs), exist_ok=True)
        csv_file = open(save_csv_abs, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["t_sec", "fx", "fy", "fz", "tx", "ty", "tz", "raw_fx", "raw_fy", "raw_fz", "raw_tx", "raw_ty", "raw_tz"]
        )

    print("=== UR RTDE TCP force monitor ===")
    print(f"robot_ip={robot_ip} read_hz={read_hz}Hz print_hz={print_hz}Hz ema_alpha={ema_alpha}")
    print("Wrench format: [Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm)")
    print("Press Ctrl+C to stop.")

    rtde_r = RTDEReceiveInterface(robot_ip)
    bias = np.zeros(6, dtype=np.float64)
    if zero_seconds > 0:
        print(f"Calibrating zero bias for {zero_seconds:.2f}s... keep robot static.")
        bias = estimate_bias(rtde_r, seconds=zero_seconds, sample_hz=min(read_hz, 125.0))
        print("Estimated bias:", np.array2string(bias, precision=4, suppress_small=True))

    dt = 1.0 / read_hz
    print_period = 1.0 / print_hz
    t_start = time.monotonic()
    last_print = -1e9
    filtered = None
    idx = 0

    try:
        while True:
            now = time.monotonic()
            t_elapsed = now - t_start
            if duration > 0 and t_elapsed >= duration:
                break

            raw = np.array(rtde_r.getActualTCPForce(), dtype=np.float64)
            corr = raw - bias

            if ema_alpha > 0:
                if filtered is None:
                    filtered = corr.copy()
                else:
                    filtered = ema_alpha * corr + (1.0 - ema_alpha) * filtered
                out = filtered
            else:
                out = corr

            if t_elapsed - last_print >= print_period:
                last_print = t_elapsed
                print(
                    f"t={t_elapsed:7.3f}s "
                    f"F=({out[0]: .3f},{out[1]: .3f},{out[2]: .3f})N "
                    f"T=({out[3]: .3f},{out[4]: .3f},{out[5]: .3f})Nm"
                )

            if csv_writer is not None:
                csv_writer.writerow([t_elapsed, *out.tolist(), *raw.tolist()])

            idx += 1
            precise_wait(t_start + idx * dt, time_func=time.monotonic)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        rtde_r.disconnect()
        if csv_file is not None:
            csv_file.close()
            print(f"CSV saved to: {save_csv}")

    print("Monitor finished.")


if __name__ == "__main__":
    main()
