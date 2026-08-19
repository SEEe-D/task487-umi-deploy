"""
标定 Livelybot CAN 夹爪执行延迟
正弦波指令 -> 读实际位置 -> 互相关算延迟
用法: python3 calibrate_gripper_latency.py --can_if can1 --gripper_id 9
"""
import click, time, subprocess, numpy as np
from scipy.interpolate import interp1d
import scipy.signal as ss

class GripperDaemon:
    def __init__(self, exe, can_if, gid, kp=10, kd=1):
        self.proc = subprocess.Popen(
            [exe, can_if, str(gid), "--daemon", "--kp", str(kp), "--kd", str(kd),
             "--target-vel-deg", "0", "--torque", "0", "--loop-hz", "100"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        r = self.proc.stdout.readline().strip()
        if r != "READY": raise RuntimeError(f"Gripper failed: {r}")
        self.proc.stdin.write("STOP\n"); self.proc.stdin.flush()
        self.proc.stdout.readline(); time.sleep(0.3)

    def set(self, deg):
        self.proc.stdin.write(f"SET {deg:.2f}\n"); self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def get(self):
        self.proc.stdin.write("GET\n"); self.proc.stdin.flush()
        l = self.proc.stdout.readline().strip()
        if l.startswith("STATE"):
            p = l.split(); return float(p[1]), float(p[2]), float(p[3])
        return 0.0, 0.0, 0.0

    def quit(self):
        try: self.proc.stdin.write("QUIT\n"); self.proc.stdin.flush(); self.proc.wait(timeout=3)
        except: self.proc.kill()

@click.command()
@click.option('--executable', default='./x3arm-can-demo-gripper')
@click.option('--can_if', default='can3')
@click.option('--gripper_id', type=int, default=8)
@click.option('--duration', type=float, default=10.0)
@click.option('--freq', type=float, default=100.0, help='Control/sample freq Hz')
@click.option('--amplitude', type=float, default=200.0, help='Sine amplitude (motor deg)')
@click.option('--offset', type=float, default=-350.0, help='Sine center (motor deg)')
@click.option('--wave_freq', type=float, default=0.5, help='Sine frequency Hz')
def main(executable, can_if, gripper_id, duration, freq, amplitude, offset, wave_freq):
    print(f"Gripper latency calibration")
    print(f"  CAN:{can_if} ID:{gripper_id} Duration:{duration}s Freq:{freq}Hz")
    print(f"  Sine: amp={amplitude} offset={offset} wave={wave_freq}Hz\n")

    g = GripperDaemon(executable, can_if, gripper_id)
    pos, _, _ = g.get()
    print(f"Current: {pos:.1f} deg")

    start_pos = offset + amplitude
    print(f"Moving to {start_pos:.1f}...")
    g.set(start_pos); time.sleep(2.0)

    dt = 1.0 / freq
    n = int(duration * freq)
    t_arr = np.arange(n) * dt
    target = offset + amplitude * np.sin(2 * np.pi * wave_freq * t_arr)

    t_cmd = np.zeros(n)
    pos_act = []; t_act = []

    print(f"Running {duration}s sine wave...")
    for i in range(n):
        t0 = time.time()
        g.set(target[i]); t_cmd[i] = time.time()
        p, _, _ = g.get(); pos_act.append(p); t_act.append(time.time())
        el = time.time() - t0
        if el < dt: time.sleep(dt - el)
        # if (i+1) % int(freq) == 0:
        print(f"  {i+1}/{n} target={target[i]:.0f} actual={p:.0f}")

    pos_act = np.array(pos_act); t_act = np.array(t_act)

    ts = max(t_cmd[0], t_act[0]); te = min(t_cmd[-1], t_act[-1])
    rdt = 1/1000; nr = int((te-ts)/rdt)
    t_s = np.arange(nr)*rdt + ts
    ft = interp1d(t_cmd, target, bounds_error=False, fill_value=(target[0], target[-1]))
    fa = interp1d(t_act, pos_act, bounds_error=False, fill_value=(pos_act[0], pos_act[-1]))
    tr = ft(t_s); ar = fa(t_s)
    m = np.mean(np.concatenate([tr, ar])); s = np.std(np.concatenate([tr, ar]))
    tn = (tr-m)/s; an = (ar-m)/s
    corr = ss.correlate(an, tn)
    lags = ss.correlation_lags(len(an), len(tn)) * rdt
    latency = lags[np.argmax(corr)]

    print(f"\n=== Result ===")
    print(f"Gripper execution latency: {latency*1000:.1f}ms ({latency:.4f}s)")
    print(f"Suggested gripper_action_latency = {latency:.3f}")

    try:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(lags*1000, corr)
        axes[0].axvline(latency*1000, color='r', ls='--', label=f'{latency*1000:.1f}ms')
        axes[0].set_xlabel('Lag(ms)'); axes[0].set_title('Cross Correlation'); axes[0].legend()
        axes[1].plot(t_cmd-t_cmd[0], target, label='target', alpha=0.8)
        axes[1].plot(t_act-t_cmd[0], pos_act, label='actual', alpha=0.8)
        axes[1].set_xlabel('Time(s)'); axes[1].set_title('Raw'); axes[1].legend()
        tr2 = t_s - t_s[0]
        axes[2].plot(tr2, tn, label='target', alpha=0.8)
        axes[2].plot(tr2-latency, an, label='actual-latency', alpha=0.8)
        axes[2].set_xlabel('Time(s)'); axes[2].set_title(f'Aligned ({latency*1000:.1f}ms)'); axes[2].legend()
        plt.tight_layout(); plt.savefig('/tmp/gripper_latency.png', dpi=150)
        print(f"Plot: /tmp/gripper_latency.png"); plt.show()
    except Exception as e: print(f"Plot: {e}")
    g.quit()

if __name__ == "__main__": main()
