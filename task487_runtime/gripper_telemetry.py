"""Read-only ROS gripper feedback recorder; no command publishers or services."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

JOINTS = {"Joint69": "left", "Joint79": "right"}
GEAR_RATIO = 20.0  # Same conversion as gripper_can_node::publish_states.
FEEDBACK_COLUMNS = ["receive_time", "receive_monotonic", "ros_stamp", "message_id", "side", "joint",
    "position_rad", "opening_deg", "velocity_rad_s", "joint_effort_nm", "motor_torque_nm",
    "last_command_rad", "last_command_receive_time", "command_age_s"]


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def feedback_rows(message, commands, receive_time, receive_monotonic, message_id):
    """Keep actual masters only; missing fields stay missing, never zero-filled."""
    stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec)*1e-9
    rows = []
    for index, name in enumerate(message.name):
        if name not in JOINTS: continue
        def field(values): return finite(values[index]) if index < len(values) else None
        position, velocity, effort = [field(v) for v in (message.position, message.velocity, message.effort)]
        command = commands.get(name)
        rows.append([receive_time, receive_monotonic, stamp, message_id, JOINTS[name], name,
            position, None if position is None else -math.degrees(position), velocity, effort,
            None if effort is None else effort/GEAR_RATIO,
            None if command is None else command[0], None if command is None else command[1],
            None if command is None else receive_time-command[1]])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0., help="0 records until terminated")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/"gripper_telemetry.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Int32
    from rcl_interfaces.msg import Log

    rclpy.init()
    node = rclpy.create_node(f"task487_gripper_recorder_{os.getpid()}", enable_rosout=False)
    # Best effort avoids imposing reliability/backpressure on control publishers.
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    stop = False
    def stop_recording(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_recording)
    signal.signal(signal.SIGINT, stop_recording)
    commands = {}; counts = {"left": 0, "right": 0}; last_seen = {}; missing_effort = dict(counts)
    message_count = 0
    started = time.time(); started_mono = time.monotonic()
    streams = []
    for name, columns in [("gripper_feedback.csv", FEEDBACK_COLUMNS),
                          ("gripper_commands.csv", ["receive_time", "receive_monotonic", "joint", "side", "command_rad", "opening_deg"])]:
        path = args.output/name
        empty = not path.exists() or path.stat().st_size == 0
        stream = path.open("a", buffering=64*1024, newline=""); streams.append(stream)
        writer = csv.writer(stream)
        if empty: writer.writerow(columns)
    feedback_writer, command_writer = map(csv.writer, streams)
    event_stream = (args.output/"gripper_ros_events.jsonl").open("a", buffering=64*1024)
    streams.append(event_stream)
    def event(kind, **data):
        event_stream.write(json.dumps(dict(event=kind, receive_time=time.time(),
            receive_monotonic=time.monotonic(), **data), allow_nan=False)+"\n")

    def state_callback(message):
        nonlocal message_count
        now = time.time(); mono = time.monotonic(); message_count += 1
        rows = feedback_rows(message, commands, now, mono, message_count)
        feedback_writer.writerows(rows)
        for row in rows:
            side = row[4]; counts[side] += 1; last_seen[side] = mono
            missing_effort[side] += int(row[9] is None)

    def command_callback(joint, message):
        now = time.time(); value = finite(message.data)
        commands[joint] = (value, now)
        command_writer.writerow([now, time.monotonic(), joint, JOINTS[joint], value,
                                 None if value is None else -math.degrees(value)])

    subscriptions = [node.create_subscription(JointState, "/gripper_joint_states", state_callback, qos)]
    for joint in JOINTS:
        subscriptions.append(node.create_subscription(Float64, f"/{joint}/position_command",
            lambda message, joint=joint: command_callback(joint, message), qos))
    subscriptions.append(node.create_subscription(Int32, "/fsm_command",
        lambda message: event("fsm_command", value=int(message.data)), qos))
    def log_callback(message):
        if message.name.rsplit("/", 1)[-1] == "gripper_can_node":
            event("backend_log", ros_stamp=message.stamp.sec+message.stamp.nanosec*1e-9,
                  level=int(message.level), name=message.name, message=message.msg)
    subscriptions.append(node.create_subscription(Log, "/rosout", log_callback, qos))
    metadata = dict(schema_version=1, started_time=started, pid=os.getpid(),
        ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"), gear_ratio=GEAR_RATIO,
        topics=["/gripper_joint_states", "/Joint69/position_command", "/Joint79/position_command", "/fsm_command", "/rosout"],
        notes=["Subscriptions only; no commands published.",
               "Effort is motor torque times gear ratio, not calibrated fingertip force.",
               "opening_deg = -degrees(ROS joint position), unclipped, for this backend convention.",
               "Command time is local callback receipt; Float64 commands have no source timestamp.",
               "ROS stamp is backend publication time, not a hardware CAN acquisition timestamp.",
               "Stall events appear in backend_log; initial stall state is unknown."])
    (args.output/"gripper_telemetry_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    def status(state):
        now_mono = time.monotonic()
        value = dict(state=state, updated_time=time.time(), started_time=started,
            message_count=message_count, rows_by_side=counts, missing_effort_by_side=missing_effort,
            age_s={side: None if side not in last_seen else now_mono-last_seen[side] for side in counts})
        path = args.output/"gripper_telemetry_status.json"
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(value, indent=2)+"\n"); tmp.replace(path)
        return value
    print(f"Read-only gripper telemetry started, domain={metadata['ros_domain_id']}, output={args.output}", flush=True)
    event("recorder_started", **metadata)
    last_flush = last_warning = 0.
    try:
        while rclpy.ok() and not stop:
            if args.parent_pid and os.getppid() != args.parent_pid: break
            if args.duration > 0 and time.monotonic()-started_mono >= args.duration: break
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if now-last_flush >= 1:
                for stream in streams: stream.flush()
                current = status("running"); last_flush = now
                if now-started_mono > 3 and now-last_warning > 5 and any(
                        age is None or age > 1 for age in current["age_s"].values()):
                    print(f"WARNING: gripper telemetry missing/stale: {current['age_s']}", flush=True)
                    last_warning = now
    finally:
        event("recorder_stopped", rows_by_side=counts)
        for stream in streams: stream.close()
        status("closed")
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        lock.close()
        print(f"Gripper telemetry saved: {counts}, missing effort={missing_effort}", flush=True)


if __name__ == "__main__": main()
