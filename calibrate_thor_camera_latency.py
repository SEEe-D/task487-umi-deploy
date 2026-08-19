"""
标定Thor推流相机端到端延迟
屏幕显示QR码(含时间戳) -> Thor相机拍到 -> 推流 -> 解码 -> 读QR时间戳
延迟 = 接收时间 - QR时间戳 - QR显示延迟
用法: python3 calibrate_thor_camera_latency.py --port 5002
按 'c' 计算, 'q' 退出
"""
import sys, os, click, cv2, qrcode, time, threading
import numpy as np
from collections import deque
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)


class ThorCamRecv:
    def __init__(self, port, w=640, h=512):
        self.port, self.w, self.h = port, w, h
        self.lock = threading.Lock()
        self.frame = None
        self.recv_ts = 0.0
        self.running = False
        self.pipeline = None

    def start(self):
        self.running = True
        for dec in ["nvh264dec", "avdec_h264"]:
            ps = (
                f'udpsrc port={self.port} buffer-size=2097152 '
                f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
                f'! rtph264depay ! h264parse ! {dec} '
                f'! queue max-size-buffers=1 leaky=downstream ! videoconvert '
                f'! video/x-raw,format=BGR,width={self.w},height={self.h} '
                f'! appsink name=sink emit-signals=false sync=false drop=true max-buffers=1'
            )
            try:
                self.pipeline = Gst.parse_launch(ps)
                self.pipeline.set_state(Gst.State.PLAYING)
                self._sink = self.pipeline.get_by_name('sink')
                print(f"Recv port {self.port} ({dec})")
                break
            except:
                continue
        else:
            print("ERROR: all decoders failed")
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            sa = self._sink.emit('try-pull-sample', int(1e8))
            if sa is None:
                time.sleep(0.002)
                continue
            buf = sa.get_buffer()
            ok, mi = buf.map(Gst.MapFlags.READ)
            if ok:
                f = np.frombuffer(mi.data, dtype=np.uint8).reshape(self.h, self.w, 3).copy()
                buf.unmap(mi)
                with self.lock:
                    self.frame, self.recv_ts = f, time.time()

    def get(self):
        with self.lock:
            return self.frame, self.recv_ts

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)


@click.command()
@click.option('--port', type=int, default=5002)
@click.option('--tile_w', type=int, default=640)
@click.option('--tile_h', type=int, default=512)
@click.option('--qr_size', type=int, default=720)
def main(port, tile_w, tile_h, qr_size):
    print(f"Thor latency calibration | port={port} {tile_w}x{tile_h}")
    print("Put QR window in Thor camera view")
    print("Press 'c' to compute, 'q' to quit\n")

    r = ThorCamRecv(port, tile_w, tile_h)
    r.start()
    time.sleep(3)

    det = cv2.QRCodeDetector()
    qr_disp = deque(maxlen=500)
    qr_det = deque(maxlen=500)
    samples = []

    while True:
        t0 = time.time()
        frame, t_recv = r.get()

        if frame is not None:
            code, corners, _ = det.detectAndDecodeCurved(frame)
            c = (0, 0, 255)
            if len(code) > 0:
                try:
                    ts = float(code)
                    qr_det.append(t_recv - ts)
                    samples.append((ts, t_recv))
                    c = (0, 255, 0)
                except:
                    qr_det.append(float('nan'))
            else:
                qr_det.append(float('nan'))
            if corners is not None:
                cv2.fillPoly(frame, corners.astype(np.int32), c)
            d = frame.copy()
            al = np.nanmean(qr_det) if qr_det else 0
            dr = 1 - np.mean(np.isnan(list(qr_det))) if qr_det else 0
            cv2.putText(d, f"Lat:{al*1000:.0f}ms Det:{dr:.0%} N:{len(samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Thor Camera', d)

        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
        ts = time.time()
        qr.add_data(str(ts))
        qr.make(fit=True)
        img = np.array(qr.make_image()).astype(np.uint8) * 255
        img = np.repeat(img[:, :, None], 3, axis=-1)
        img = cv2.resize(img, (qr_size, qr_size), cv2.INTER_NEAREST)
        cv2.imshow('QR', img)
        qr_disp.append(time.time() - ts)

        fps = 1.0 / max(time.time() - t0, 0.001)
        aq = np.mean(qr_disp) if qr_disp else 0
        ae = np.nanmean(qr_det) if qr_det else 0
        print(f"\rFPS:{fps:.0f} E2E:{ae*1000:.0f}ms QRdisp:{aq*1000:.0f}ms "
              f"Corrected:{(ae-aq)*1000:.0f}ms N:{len(samples)}", end='', flush=True)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('c') and len(samples) > 10:
            break
        elif k == ord('q'):
            r.stop()
            cv2.destroyAllWindows()
            return

    print("\n\n=== Result ===")
    aq = np.mean(qr_disp)
    lats = [recv - qr_ts - aq for qr_ts, recv in samples]
    print(f"Samples: {len(samples)}, QR display latency: {aq*1000:.1f}ms")
    print(f"Thor end-to-end latency:")
    print(f"  AVG = {np.mean(lats)*1000:.1f}ms")
    print(f"  STD = {np.std(lats)*1000:.1f}ms")
    print(f"  MED = {np.median(lats)*1000:.1f}ms")
    print(f"  P95 = {np.percentile(lats,95)*1000:.1f}ms")
    print(f"\nSuggested camera_obs_latency = {np.mean(lats):.3f}")

    try:
        from matplotlib import pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot([l*1000 for l in lats], 'b.-', alpha=0.5)
        a1.axhline(np.mean(lats)*1000, color='r', ls='--',
                   label=f'avg={np.mean(lats)*1000:.1f}ms')
        a1.set_xlabel('Sample')
        a1.set_ylabel('ms')
        a1.set_title('Latency')
        a1.legend()
        a1.grid(True, alpha=0.3)
        a2.hist([l*1000 for l in lats], bins=30, edgecolor='black')
        a2.axvline(np.mean(lats)*1000, color='r', ls='--')
        a2.set_xlabel('ms')
        a2.set_title('Distribution')
        plt.tight_layout()
        plt.savefig('/tmp/thor_latency.png', dpi=150)
        print(f"Plot: /tmp/thor_latency.png")
        plt.show()
    except Exception as e:
        print(f"Plot error: {e}")

    r.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
