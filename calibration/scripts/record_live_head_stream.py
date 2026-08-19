#!/usr/bin/env python3
"""Record a live Thor head stream with local arrival timestamps.

This runs with the Ubuntu system Python because the GStreamer GI bindings are
installed there.  It is observation-only and never connects to robot control.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receiver-path",
        default=(
            "/home/simpleai/Code/universal_manipulation_interface-main/"
            "gj/pi05-deploy"
        ),
    )
    parser.add_argument("--thor-host", default="192.168.2.178")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-label", default="cam_head_left")
    parser.add_argument("--video-port", type=int, default=5000)
    parser.add_argument("--meta-port", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=90.0)
    args = parser.parse_args()

    receiver_path = str(Path(args.receiver_path).expanduser().resolve())
    if receiver_path not in sys.path:
        sys.path.insert(0, receiver_path)
    from receiver_gi import CameraReceiver, measure_clock_offset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "head_main_640x512.mp4"
    timestamps_path = args.output_dir / "head_main_timestamps.jsonl"
    ready_path = args.output_dir / "camera_ready.json"

    offset_ms = float(measure_clock_offset(args.thor_host, port=7777))
    receiver = CameraReceiver(
        camera_id=args.camera_id,
        label=args.camera_label,
        video_port=args.video_port,
        meta_port=args.meta_port,
        listen_host="0.0.0.0",
        tile_w=640,
        tile_h=512,
        clock_offset_ms=offset_ms,
    )
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 512), True
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open MP4 writer: {}".format(video_path))

    receiver.start()
    frame_count = 0
    last_total_frames = -1
    started = time.monotonic()
    try:
        deadline = started + 20.0
        while time.monotonic() < deadline:
            with receiver.lock:
                frame = None if receiver.frame is None else receiver.frame.copy()
            if frame is not None:
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("no decoded head camera frame within 20 seconds")

        ready_path.write_text(
            json.dumps(
                {
                    "ready": True,
                    "ready_time_ns": time.time_ns(),
                    "clock_offset_ms": offset_ms,
                    "video_path": str(video_path.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("CAMERA_READY {}".format(video_path), flush=True)

        with timestamps_path.open("w", encoding="utf-8") as timestamp_file:
            while time.monotonic() - started < args.duration_s:
                total_frames = int(receiver.total_frames)
                if total_frames == last_total_frames:
                    time.sleep(0.002)
                    continue
                with receiver.lock:
                    frame = None if receiver.frame is None else receiver.frame.copy()
                    arrival_ns = int(receiver._last_frame_time * 1_000_000_000.0)
                last_total_frames = total_frames
                if frame is None or frame.shape[:2] != (512, 640):
                    continue
                writer.write(frame)
                timestamp_file.write(
                    json.dumps(
                        {
                            "video_frame_index": frame_count,
                            "arrival_time_ns": arrival_ns,
                            "receiver_total_frames": total_frames,
                        }
                    )
                    + "\n"
                )
                timestamp_file.flush()
                frame_count += 1
    except KeyboardInterrupt:
        print("camera recorder interrupted after {} frames".format(frame_count), flush=True)
    finally:
        receiver.stop()
        writer.release()

    summary = {
        "frame_count": frame_count,
        "duration_s": time.monotonic() - started,
        "video_path": str(video_path.resolve()),
        "timestamps_path": str(timestamps_path.resolve()),
    }
    (args.output_dir / "camera_recording_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0 if frame_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
