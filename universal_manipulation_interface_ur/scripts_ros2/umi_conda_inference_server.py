#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CondA-only 推理服务（不依赖 rclpy）：
 - 用 conda(py3.9) 跑 torch/hydra/diffusion_policy 模型
 - 通过本机 TCP 接收来自 ROS2 client 的观测（压缩JPEG + 低维状态）
 - 返回多步动作序列 env_action: (H, 14)

协议：length-prefixed pickle（仅建议 localhost 调试使用）
request:
  {"type": "start", "episode_start_pose6": [[6],[6]], "action_base": "current|episode_start"}
  {"type": "stop"}
  {"type": "infer",
    "cam0_jpg": bytes, "cam1_jpg": bytes, ("cam2_jpg": bytes optional),
    "left_pose6": [6], "right_pose6": [6],
    "left_grip_deg": float, "right_grip_deg": float,
    "timestamp": float}
response:
  {"ok": True, "env_action": np.ndarray(H,14)}  # grip entries are degrees per your model convention
  {"ok": False, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import queue
import socket
import struct
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# make repo importable
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dill  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action  # noqa: E402
from umi.common.pose_util import pose_to_mat, mat_to_pose, pose10d_to_mat  # noqa: E402
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)

SERVER_VERSION = "umi_conda_inference_server:v2"


class _AsyncRecorder:
    """Best-effort async writer for images/json to reduce impact on inference latency."""

    def __init__(self, run_dir: str, *, max_queue: int = 512, enable: bool = True) -> None:
        self.enable = bool(enable)
        self.run_dir = str(run_dir)
        self.max_queue = int(max_queue)
        self._q: "queue.Queue[Tuple[str, str, object]]" = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()

        # subfolders (keep minimal: images + inference only)
        self.dir_images = os.path.join(self.run_dir, "images")
        self.dir_infer = os.path.join(self.run_dir, "inference")
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_infer, exist_ok=True)

        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def close(self, timeout_s: float = 2.0) -> None:
        if not self.enable:
            return
        self._stop.set()
        try:
            self._th.join(timeout=float(timeout_s))
        except Exception:
            pass

    def _try_put(self, kind: str, path: str, payload: object) -> None:
        if not self.enable:
            return
        try:
            self._q.put_nowait((kind, path, payload))
        except queue.Full:
            # drop silently (best-effort)
            pass

    def save_image_bgr(self, *, name: str, t_ns: int, image_bgr: np.ndarray) -> str:
        fname = f"{name}_{int(t_ns)}.png"
        path = os.path.join(self.dir_images, fname)
        self._try_put("image", path, image_bgr)
        return path

    def save_json(self, *, subdir: str, name: str, t_ns: int, data: dict) -> str:
        fname = f"{name}_{int(t_ns)}.json"
        path = os.path.join(self.run_dir, subdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._try_put("json", path, data)
        return path

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                kind, path, payload = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                if kind == "image":
                    cv2.imwrite(path, payload)
                elif kind == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass


def _jsonify(x):
    """numpy -> JSON-friendly {dtype, shape, data(flat)}; scalars -> python scalar."""
    if isinstance(x, np.ndarray):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "data": x.reshape(-1).tolist()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    return x


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def _recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 8)
    (n,) = struct.unpack("!Q", header)
    payload = _recv_exact(sock, n)
    return pickle.loads(payload)


def _send_msg(sock: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class InferenceEngine:
    def __init__(
        self,
        ckpt: str,
        device: str,
        steps_per_inference: int,
        *,
        save_io: bool = True,
        io_dir: str = "/tmp/umi_ocs2_inference_io",
        io_async_max_queue: int = 512,
    ) -> None:
        if not ckpt.endswith(".ckpt"):
            ckpt = os.path.join(ckpt, "checkpoints", "latest.ckpt")
        payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]

        self.cfg = cfg
        self.shape_meta = cfg.task.shape_meta
        self.obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr
        self.steps_per_inference = int(steps_per_inference)
        # debug: throttle raw_action prints (can be large)
        self._last_raw_action_log_t = 0.0
        self._raw_action_log_period_s = 1.0

        cls = __import__("hydra").utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.num_inference_steps = 16

        self.device = torch.device(device)
        self.policy = policy.eval().to(self.device)

        # buffers
        self._obs_buf: Dict[str, List[np.ndarray]] = {}
        # IMPORTANT:
        # shape_meta['obs'] may contain many "derived" eef keys (e.g. *_wrt*, *_wrt_start) that are NOT provided by client.
        # Those keys are generated inside get_real_umi_obs_dict(), so env_obs should only include the raw keys that we provide.
        self._env_obs_horizons, self._filler_specs = self._infer_env_obs_horizons_and_fillers(self.shape_meta)

        # episode state
        self._active = False
        self._episode_start_pose: Optional[List[np.ndarray]] = None
        self._action_base = "current"

        # for bimanual relative features; you removed yaml, so default identity
        self._tx_robot1_robot0 = np.eye(4, dtype=np.float64)

        # warmup flag
        self._did_warmup = False

        # debug: throttle obs_dict (non-image) prints
        self._last_obs_dict_log_t = 0.0
        self._obs_dict_log_period_s = 1.0

        # debug: throttle raw request pose prints (before pose repr conversion)
        self._last_req_pose_log_t = 0.0
        self._req_pose_log_period_s = 1.0

        # --- IO dump (per request) ---
        self._save_io = bool(save_io)
        self._io_run_dir: Optional[str] = None
        self._recorder: Optional[_AsyncRecorder] = None
        base = str(io_dir) if io_dir is not None else "/tmp/umi_ocs2_inference_io"
        print(f"[SERVER] IO dump setting: save_io={self._save_io} io_dir={base}")
        if self._save_io:
            run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            try:
                self._io_run_dir = os.path.join(base, run_id)
                os.makedirs(self._io_run_dir, exist_ok=True)
                self._recorder = _AsyncRecorder(self._io_run_dir, max_queue=int(io_async_max_queue), enable=True)
                meta = {
                    "created_localtime": run_id,
                    "server_version": SERVER_VERSION,
                    "ckpt_path": ckpt,
                    "device": str(self.device),
                    "steps_per_inference": int(self.steps_per_inference),
                    "obs_pose_repr": str(self.obs_pose_repr),
                    "action_pose_repr": str(self.action_pose_repr),
                    "note": "saved folders: images/ + inference/. images are BGR PNG; timestamps are int(ts*1e9).",
                }
                self._recorder.save_json(subdir=".", name="meta", t_ns=int(time.time() * 1e9), data=meta)
                print(f"[SERVER] IO dump enabled. run_dir={self._io_run_dir}")
            except Exception as e:
                self._save_io = False
                self._io_run_dir = None
                self._recorder = None
                print(f"[SERVER][ERROR] IO dump init failed, disabled. err={e}")

    @staticmethod
    def _infer_env_obs_horizons_and_fillers(shape_meta: dict) -> Tuple[Dict[str, int], Dict[str, Tuple[str, Tuple[int, ...]]]]:
        """
        Returns:
          - env_obs_horizons: horizons for keys that MUST be provided in env_obs (raw keys only)
          - filler_specs: for low_dim keys (excluding eef keys) that may be required by model but not provided by client,
            fill with zeros. map: key -> (type, shape_tuple)
        """
        obs_meta = shape_meta["obs"]
        raw_pose_keys = {
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "robot1_eef_pos",
            "robot1_eef_rot_axis_angle",
        }
        env_obs_horizons: Dict[str, int] = {}
        filler_specs: Dict[str, Tuple[str, Tuple[int, ...]]] = {}

        for key, attr in obs_meta.items():
            typ = attr.get("type", "low_dim")
            h = int(attr.get("horizon", 1))
            shape = tuple(attr.get("shape", ()))

            if typ == "rgb":
                # we can provide camera0_rgb/camera1_rgb, and optionally camera2_rgb if client sends cam2_jpg
                if key in ("camera0_rgb", "camera1_rgb", "camera2_rgb"):
                    env_obs_horizons[key] = h
                continue

            # low_dim
            if key in raw_pose_keys:
                env_obs_horizons[key] = h
                continue

            # any other key containing 'eef' is derived inside get_real_umi_obs_dict -> do NOT require in env_obs
            if "eef" in key:
                continue

            # non-eef low_dim keys must exist in env_obs; if client doesn't provide them, we fill zeros.
            env_obs_horizons[key] = h
            if key not in ("timestamp",):
                # store filler spec
                filler_specs[key] = (typ, shape)

        # timestamp is always required
        env_obs_horizons["timestamp"] = max(env_obs_horizons.values()) if env_obs_horizons else 1
        return env_obs_horizons, filler_specs

    def start(self, episode_start_pose6: Optional[List[List[float]]], action_base: str) -> None:
        self._active = True
        self._action_base = action_base
        self._episode_start_pose = None
        if episode_start_pose6 is not None:
            self._episode_start_pose = [np.array(x, dtype=np.float64) for x in episode_start_pose6]
        # reset history so horizons are satisfied from this start point
        self._obs_buf = {}
        self._did_warmup = False
        self.policy.reset()

    def stop(self) -> None:
        self._active = False
        self._episode_start_pose = None

    def _push(self, key: str, x: np.ndarray, horizon: int) -> None:
        # If first time seeing this key, prefill to satisfy horizon immediately.
        if key not in self._obs_buf:
            self._obs_buf[key] = [x.copy() for _ in range(max(1, horizon))]
            return
        buf = self._obs_buf[key]
        buf.append(x)
        if len(buf) > horizon:
            del buf[0 : len(buf) - horizon]

    def _build_env_obs(self) -> Optional[Dict[str, np.ndarray]]:
        for key, h in self._env_obs_horizons.items():
            if key not in self._obs_buf or len(self._obs_buf[key]) < h:
                return None
        env_obs: Dict[str, np.ndarray] = {}
        for key, h in self._env_obs_horizons.items():
            xs = self._obs_buf[key][-h:]
            x = np.stack(xs, axis=0)
            if key == "timestamp":
                x = x.reshape(-1)
            env_obs[key] = x
        return env_obs

    def _decode_jpg_to_rgb(self, jpg: bytes) -> np.ndarray:
        arr = np.frombuffer(jpg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("failed to decode jpg")
        rgb = bgr[..., ::-1]
        return rgb.astype(np.uint8, copy=False)

    def _raw_action_to_env_action(self, raw_action: np.ndarray, env_obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self._action_base == "current":
            return get_real_umi_action(raw_action, env_obs, self.action_pose_repr)

        if self._action_base != "episode_start":
            raise RuntimeError(f"Unsupported action_base: {self._action_base}")
        if self._episode_start_pose is None:
            raise RuntimeError("episode_start_pose is None; call start first")

        n_robots = int(raw_action.shape[-1] // 10)
        if n_robots != 2:
            raise RuntimeError(f"Expected 2 robots, got {n_robots}")

        env_action_parts: List[np.ndarray] = []
        for robot_idx in range(n_robots):
            base_pose_mat = pose_to_mat(self._episode_start_pose[robot_idx].astype(np.float64))
            start = robot_idx * 10
            action_pose10d = raw_action[..., start : start + 9]
            action_grip = raw_action[..., start + 9 : start + 10]
            action_pose_mat = pose10d_to_mat(action_pose10d)
            action_abs_mat = convert_pose_mat_rep(
                action_pose_mat, base_pose_mat=base_pose_mat, pose_rep=self.action_pose_repr, backward=True
            )
            action_pose6 = mat_to_pose(action_abs_mat)
            env_action_parts.append(action_pose6)
            env_action_parts.append(action_grip)
        return np.concatenate(env_action_parts, axis=-1)

    def infer(self, req: dict) -> np.ndarray:
        if not self._active:
            raise RuntimeError("server not started (call start)")

        cam0_rgb = self._decode_jpg_to_rgb(req["cam0_jpg"])
        cam1_rgb = self._decode_jpg_to_rgb(req["cam1_jpg"])
        cam2_rgb = None
        if "camera2_rgb" in (self.shape_meta.get("obs", {}) or {}):
            if "cam2_jpg" not in req:
                raise RuntimeError("missing cam2_jpg but model expects camera2_rgb")
            cam2_rgb = self._decode_jpg_to_rgb(req["cam2_jpg"])
        left_pose6 = np.array(req["left_pose6"], dtype=np.float32)
        right_pose6 = np.array(req["right_pose6"], dtype=np.float32)
        left_grip_deg = float(req["left_grip_deg"])
        right_grip_deg = float(req["right_grip_deg"])
        ts = float(req.get("timestamp", time.time()))
        t_ns = int(ts * 1e9)

        # dump request inputs (images only), best-effort
        if self._save_io and (self._recorder is not None):
            try:
                # cv2.imwrite expects BGR
                cam0_bgr = cam0_rgb[..., ::-1]
                cam1_bgr = cam1_rgb[..., ::-1]
                _ = os.path.basename(
                    self._recorder.save_image_bgr(name="camera0", t_ns=t_ns, image_bgr=cam0_bgr)
                )
                _ = os.path.basename(
                    self._recorder.save_image_bgr(name="camera1", t_ns=t_ns, image_bgr=cam1_bgr)
                )
                if cam2_rgb is not None:
                    cam2_bgr = cam2_rgb[..., ::-1]
                    _ = os.path.basename(
                        self._recorder.save_image_bgr(name="camera2", t_ns=t_ns, image_bgr=cam2_bgr)
                    )
            except Exception:
                pass

        # Debug: print raw pose inputs carried by request (exclude images), throttled.
        # IMPORTANT: the client may intentionally swap left/right payload to match model ordering.
        now = time.time()
        if (now - self._last_req_pose_log_t) >= self._req_pose_log_period_s:
            self._last_req_pose_log_t = now
            try:
                lp = np.asarray(left_pose6, dtype=np.float32).reshape(-1)
                rp = np.asarray(right_pose6, dtype=np.float32).reshape(-1)
                print(
                    "[SERVER] req poses (raw, before obs_pose_repr conversion) | "
                    f"left_pose6={lp.tolist()} | right_pose6={rp.tolist()} | "
                    f"left_grip_deg={left_grip_deg:.4f} right_grip_deg={right_grip_deg:.4f}"
                )
                print(
                    "[SERVER] NOTE: obs_dict robot*_eef_pos may appear ~0 when obs_pose_repr='relative' "
                    "(last timestep is expressed w.r.t itself)."
                )
            except Exception as e:
                print(f"[SERVER][WARN] failed to log req poses: {e}")

        # push into buffers (assume policy expects these keys; if not present, we still keep)
        if "camera0_rgb" in self.shape_meta["obs"]:
            if "camera0_rgb" in self._env_obs_horizons:
                self._push("camera0_rgb", cam0_rgb, self._env_obs_horizons["camera0_rgb"])
        if "camera1_rgb" in self.shape_meta["obs"]:
            if "camera1_rgb" in self._env_obs_horizons:
                self._push("camera1_rgb", cam1_rgb, self._env_obs_horizons["camera1_rgb"])
        if "camera2_rgb" in (self.shape_meta.get("obs", {}) or {}):
            if "camera2_rgb" in self._env_obs_horizons and cam2_rgb is not None:
                self._push("camera2_rgb", cam2_rgb, self._env_obs_horizons["camera2_rgb"])

        # robot0=left, robot1=right
        self._push("robot0_eef_pos", left_pose6[:3], self._env_obs_horizons.get("robot0_eef_pos", 1))
        self._push("robot0_eef_rot_axis_angle", left_pose6[3:6], self._env_obs_horizons.get("robot0_eef_rot_axis_angle", 1))
        self._push("robot1_eef_pos", right_pose6[:3], self._env_obs_horizons.get("robot1_eef_pos", 1))
        self._push("robot1_eef_rot_axis_angle", right_pose6[3:6], self._env_obs_horizons.get("robot1_eef_rot_axis_angle", 1))

        for k in ("robot0_gripper_width", "robot1_gripper_width"):
            if k in self.shape_meta["obs"]:
                v = left_grip_deg if k == "robot0_gripper_width" else right_grip_deg
                if k in self._env_obs_horizons:
                    self._push(k, np.array([v], dtype=np.float32), self._env_obs_horizons[k])

        self._push("timestamp", np.array([ts], dtype=np.float64), self._env_obs_horizons["timestamp"])

        # Fill missing non-eef low_dim keys with zeros to satisfy env_obs_horizons immediately.
        for key, h in self._env_obs_horizons.items():
            if key in self._obs_buf:
                continue
            if key == "timestamp":
                continue
            spec = self._filler_specs.get(key, None)
            if spec is None:
                # likely camera keys; if missing, let env_obs be None and report warmup
                continue
            _typ, shape = spec
            # low_dim expected input is (D,) per timestep
            if len(shape) == 0:
                z = np.zeros((1,), dtype=np.float32)
            else:
                z = np.zeros(shape, dtype=np.float32)
            self._push(key, z, h)

        env_obs = self._build_env_obs()
        if env_obs is None:
            # normal warmup: not enough history yet
            raise RuntimeError("WARMING_UP: not enough history yet to satisfy obs horizons")

        allowed_obs_keys = set(self.shape_meta["obs"].keys())

        # warmup once
        if not self._did_warmup:
            with torch.no_grad():
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=env_obs,
                    shape_meta=self.shape_meta,
                    obs_pose_repr=self.obs_pose_repr,
                    tx_robot1_robot0=self._tx_robot1_robot0,
                    episode_start_pose=self._episode_start_pose,
                )
                # get_real_umi_obs_dict may generate derived keys not present in shape_meta['obs'].
                # Filter to the keys this checkpoint expects to avoid normalizer key mismatch.
                obs_dict_np = {k: v for k, v in obs_dict_np.items() if k in allowed_obs_keys}
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                _ = self.policy.predict_action(obs_dict)
            self._did_warmup = True

        with torch.no_grad():
            obs_dict_np = get_real_umi_obs_dict(
                env_obs=env_obs,
                shape_meta=self.shape_meta,
                obs_pose_repr=self.obs_pose_repr,
                tx_robot1_robot0=self._tx_robot1_robot0,
                episode_start_pose=self._episode_start_pose,
            )
            obs_dict_np = {k: v for k, v in obs_dict_np.items() if k in allowed_obs_keys}
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # Debug: print obs_dict (exclude image keys), throttled.
            now = time.time()
            if (now - self._last_obs_dict_log_t) >= self._obs_dict_log_period_s:
                self._last_obs_dict_log_t = now
                try:
                    # Determine which obs keys are rgb from shape_meta
                    rgb_keys = {
                        k for k, attr in (self.shape_meta.get("obs", {}) or {}).items()
                        if (attr or {}).get("type", "low_dim") == "rgb"
                    }
                    lines = []
                    for k in sorted(obs_dict.keys()):
                        if k in rgb_keys:
                            continue
                        v = obs_dict[k]
                        # Print full numeric values (all timesteps) with fixed decimals (no scientific notation).
                        shape = tuple(v.shape)
                        # common low_dim shape: (B, T, D) or (B, D)
                        arr = v.detach().to("cpu").float().numpy()
                        # Show only batch0 if B==1; keep all remaining dims (e.g. T x D).
                        show = arr[0] if (arr.ndim >= 1 and arr.shape[0] == 1) else arr
                        show_str = np.array2string(
                            show,
                            separator=", ",
                            max_line_width=200,
                            formatter={"float_kind": lambda x: f"{float(x):.6f}"},
                        )
                        lines.append(f"  - {k}: shape={shape} values={show_str}")
                    print("[SERVER] obs_dict (non-image)\\n" + "\\n".join(lines))
                except Exception as e:
                    print(f"[SERVER][WARN] failed to log obs_dict: {e}")

            result = self.policy.predict_action(obs_dict)
            raw_action = result["action_pred"][0].detach().to("cpu").numpy()
            # Debug: print model raw_action/env_action (throttled) for inspection.
            now = time.time()
            do_log_action = (now - self._last_raw_action_log_t) >= self._raw_action_log_period_s
            if do_log_action:
                self._last_raw_action_log_t = now
                try:
                    raw_action_str = np.array2string(
                        raw_action,
                        separator=", ",
                        max_line_width=200,
                        formatter={"float_kind": lambda x: f"{float(x):.6f}"},
                    )
                    print(f"[SERVER] raw_action shape={tuple(raw_action.shape)} values={raw_action_str}")
                except Exception as e:
                    print(f"[SERVER][WARN] failed to log raw_action: {e}")
            env_action = self._raw_action_to_env_action(raw_action, env_obs)
            if do_log_action:
                try:
                    ea = env_action
                    if not isinstance(ea, np.ndarray):
                        ea = np.asarray(ea, dtype=np.float32)
                    if ea.ndim == 1:
                        ea = ea[None, :]
                    ea_str = np.array2string(
                        ea,
                        separator=", ",
                        max_line_width=200,
                        formatter={"float_kind": lambda x: f"{float(x):.6f}"},
                    )
                    print(f"[SERVER] env_action shape={tuple(ea.shape)} values={ea_str}")
                except Exception as e:
                    print(f"[SERVER][WARN] failed to log env_action: {e}")

        if env_action.ndim == 1:
            env_action = env_action[None, :]
        H = min(self.steps_per_inference, int(env_action.shape[0]))
        env_action_ret = env_action[:H].astype(np.float32, copy=False)

        # dump model IO (chunk), best-effort
        if self._save_io and (self._recorder is not None):
            try:
                # Do NOT record image arrays inside inference json to keep files small.
                _rgb_drop = {"camera0_rgb", "camera1_rgb", "camera2_rgb"}
                payload = {
                    "t_ns": int(t_ns),
                    "timestamp_sec": float(ts),
                    "env_obs": {str(k): _jsonify(np.asarray(v)) for k, v in env_obs.items() if str(k) not in _rgb_drop},
                    "obs_dict_np": {
                        str(k): _jsonify(np.asarray(v)) for k, v in obs_dict_np.items() if str(k) not in _rgb_drop
                    },
                    "raw_action": _jsonify(np.asarray(raw_action)),
                    "env_action_full": _jsonify(np.asarray(env_action)),
                    "env_action_returned": _jsonify(np.asarray(env_action_ret)),
                    "H_returned": int(H),
                }
                self._recorder.save_json(subdir="inference", name="chunk", t_ns=t_ns, data=payload)
            except Exception:
                pass

        return env_action_ret


def _handle_client(conn: socket.socket, addr: Tuple[str, int], engine: InferenceEngine) -> None:
    try:
        while True:
            req = _recv_msg(conn)
            typ = req.get("type", "")
            if typ == "start":
                engine.start(req.get("episode_start_pose6"), req.get("action_base", "current"))
                _send_msg(conn, {"ok": True})
            elif typ == "stop":
                engine.stop()
                _send_msg(conn, {"ok": True})
            elif typ == "ping":
                _send_msg(
                    conn,
                    {
                        "ok": True,
                        "server_version": SERVER_VERSION,
                        "env_obs_horizons": getattr(engine, "_env_obs_horizons", {}),
                    },
                )
            elif typ == "infer":
                try:
                    env_action = engine.infer(req)
                    _send_msg(conn, {"ok": True, "env_action": env_action})
                except Exception as e:
                    msg = str(e)
                    if msg.startswith("WARMING_UP:"):
                        _send_msg(conn, {"ok": True, "warming_up": True, "message": msg})
                    else:
                        _send_msg(conn, {"ok": False, "error": msg})
            else:
                _send_msg(conn, {"ok": False, "error": f"unknown type: {typ}"})
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-inference", type=int, default=16)
    ap.add_argument(
        "--save-io",
        dest="save_io",
        action="store_true",
        default=True,
        help="保存每次infer的输入(相机+state)与输出(action chunk)到文件（默认开启）。",
    )
    ap.add_argument(
        "--no-save-io",
        dest="save_io",
        action="store_false",
        help="关闭保存IO（不落盘）。",
    )
    ap.add_argument(
        "--io-dir",
        default="/tmp/umi_ocs2_inference_io",
        help="IO保存根目录；会自动创建 run 子目录（如 /tmp/umi_ocs2_inference_io/20260225_123456）。",
    )
    ap.add_argument(
        "--io-async-max-queue",
        type=int,
        default=512,
        help="异步写盘队列上限（满了会丢数据，避免阻塞推理线程）。",
    )
    args = ap.parse_args()

    engine = InferenceEngine(
        ckpt=args.ckpt,
        device=args.device,
        steps_per_inference=args.steps_per_inference,
        save_io=args.save_io,
        io_dir=args.io_dir,
        io_async_max_queue=args.io_async_max_queue,
    )
    print(f"[SERVER] ready on {args.host}:{args.port}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_handle_client, args=(conn, addr, engine), daemon=True)
            t.start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()

