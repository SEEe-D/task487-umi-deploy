"""Manage the passive ROS subscriber in its separate Python environment."""
import logging
import os
from pathlib import Path
import subprocess
import threading


class PassiveGripperTelemetry:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.process = None
        self._warned = False
        self._stream = None
        self._pump = None

    def __enter__(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = (self.output_dir/"gripper_recorder.log").open("a")
            script = Path(__file__).resolve().parent.parent/"run_gripper_telemetry.sh"
            self.process = subprocess.Popen(["/bin/bash", str(script), "--output", str(self.output_dir),
                "--parent-pid", str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
                stdin=subprocess.DEVNULL, start_new_session=True)
            def pump():
                try:
                    for line in self.process.stdout:
                        self._stream.write(line); self._stream.flush()
                        level = logging.WARNING if "WARNING" in line else logging.INFO
                        logging.log(level, "Gripper recorder: %s", line.rstrip())
                except Exception:
                    logging.exception("Failed to relay gripper recorder output")
            self._pump = threading.Thread(target=pump, name="gripper-log-relay", daemon=True)
            self._pump.start()
            logging.info("Passive gripper recorder PID=%d; feedback/torque logs in %s",
                         self.process.pid, self.output_dir)
        except Exception:
            logging.exception("Gripper telemetry unavailable; see gripper_recorder.log")
        return self

    def check(self):
        if self.process is not None and self.process.poll() is not None and not self._warned:
            self._warned = True
            logging.error("Gripper telemetry exited rc=%s; feedback/torque recording incomplete: %s",
                          self.process.returncode, self.output_dir/"gripper_recorder.log")

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=2)
        if self._pump is not None: self._pump.join(timeout=1)
        if self.process is not None and self.process.stdout is not None:
            self.process.stdout.close()
        if self._stream is not None: self._stream.close()
