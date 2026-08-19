"""eval_offline.py — 训练阶段离线评估

从训练数据中抽取验证集，加载 checkpoint，计算 MSE/MAE/Cosine Sim。
不需要 model server，直接在训练机上跑。

用法:
    python eval_offline.py --config pi05_umi_cups --ckpt-dir checkpoints/pi05_umi_cups/umi_cups_v2/25000
    python eval_offline.py --config pi05_umi_cups --ckpt-dir /workspace/zt/checkpoints_cups_v2/umi_cups_v2/25000 --num-episodes 20
"""
import argparse
import numpy as np
import pathlib
import sys


def rot6d_to_matrix(rot6d):
    r1 = rot6d[:3].copy()
    r2 = rot6d[3:6].copy()
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=1)


def matrix_to_rot6d(R):
    return R[:, :2].T.flatten()


def compute_gt_body_actions(states, t, horizon=10):
    """ground truth body-frame delta (和训练 transform 一致)"""
    N = len(states)
    gt = np.zeros((horizon, 10), dtype=np.float32)
    state_t = states[t]
    R_current = rot6d_to_matrix(state_t[3:9])

    for i in range(horizon):
        fut = min(t + i, N - 1)
        act = states[fut].copy()

        # DeltaActions
        delta_pos = act[:3] - state_t[:3]
        delta_rot6d_linear = act[3:9] - state_t[3:9]

        # GlobalToBodyDelta - position
        body_pos = R_current.T @ delta_pos

        # GlobalToBodyDelta - rotation (geometric relative)
        recovered_rot6d = delta_rot6d_linear + state_t[3:9]
        R_target = rot6d_to_matrix(recovered_rot6d)
        R_rel = R_current.T @ R_target
        body_rot6d = matrix_to_rot6d(R_rel)

        gt[i, :3] = body_pos
        gt[i, 3:9] = body_rot6d
        gt[i, 9] = act[9]  # gripper absolute

    return gt


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Training config name")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory (e.g. .../25000)")
    parser.add_argument("--num-episodes", type=int, default=10, help="验证 episode 数")
    parser.add_argument("--sample-interval", type=int, default=15, help="每隔 N 帧评估一次")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    # 加载 openpi
    from openpi.training import config as train_config
    from openpi.policies import policy_config
    from openpi.shared import normalize

    cfg = train_config.get_config(args.config)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt_dir}")

    # 加载 norm_stats
    ckpt_path = pathlib.Path(args.ckpt_dir)
    asset_dirs = list((ckpt_path / "assets").iterdir()) if (ckpt_path / "assets").exists() else []
    norm_stats = None
    for ad in asset_dirs:
        for sub in ad.iterdir():
            ns_file = sub / "norm_stats.json"
            if ns_file.exists():
                norm_stats = normalize.load(sub)
                print(f"Loaded norm_stats from {sub}")
                break

    if norm_stats is None:
        print("ERROR: norm_stats not found in checkpoint")
        sys.exit(1)

    # 构建 policy runtime
    runtime = policy_config.create_runtime(
        config=cfg.model,
        data_config=cfg.data,
        norm_stats=norm_stats,
        checkpoint_dir=ckpt_path,
    )
    print("Model loaded")

    # 加载验证数据
    import pandas as pd
    import glob
    import os

    data_root = os.environ.get("HF_LEROBOT_HOME", "data")
    parquet_dir = os.path.join(data_root, cfg.data.repo_id, "data", "chunk-000")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} episodes")

    # 取最后 N 个 episode 做验证 (训练没见过最后的数据太多次)
    val_files = parquet_files[-args.num_episodes:]
    print(f"Using last {len(val_files)} episodes for validation")

    results = {
        'pos_mse': [], 'pos_mae': [], 'pos_cosine': [],
        'rot_mse': [], 'rot_mae': [], 'rot_cosine': [],
        'grip_mae': [],
    }

    from io import BytesIO
    from PIL import Image
    import cv2

    for ep_idx, pq_file in enumerate(val_files):
        df = pd.read_parquet(pq_file)
        states = np.stack(df['state'].values)
        N = len(df)

        eval_frames = list(range(0, N - args.horizon, args.sample_interval))

        for t in eval_frames:
            # Ground truth
            gt = compute_gt_body_actions(states, t, args.horizon)

            # Model input
            state_input = np.zeros(20, dtype=np.float32)
            state_input[3:9] = [1, 0, 0, 0, 1, 0]
            state_input[9] = states[t, 9]

            img_data = df.iloc[t]['cam_right']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img = np.array(Image.open(BytesIO(img_data['bytes'])))
            else:
                img = np.array(img_data)
            if img.shape[0] != 256 or img.shape[1] != 256:
                img = cv2.resize(img, (256, 256))

            # 推理
            obs = {"cam_right": img, "state": state_input, "prompt": cfg.data.default_prompt or "pick up the cup"}
            pred_result = runtime.infer(obs)
            pred = np.asarray(pred_result["actions"])[:args.horizon, :10]

            # Metrics (skip step 0)
            for step in range(1, args.horizon):
                g, p = gt[step], pred[step]

                pos_err = g[:3] - p[:3]
                results['pos_mse'].append(float(np.mean(pos_err ** 2)))
                results['pos_mae'].append(float(np.mean(np.abs(pos_err))))
                results['pos_cosine'].append(cosine_sim(g[:3], p[:3]))

                rot_err = g[3:9] - p[3:9]
                results['rot_mse'].append(float(np.mean(rot_err ** 2)))
                results['rot_mae'].append(float(np.mean(np.abs(rot_err))))
                results['rot_cosine'].append(cosine_sim(g[3:9], p[3:9]))

                results['grip_mae'].append(float(abs(g[9] - p[9])))

        print(f"  Episode {ep_idx + 1}/{len(val_files)} ({os.path.basename(pq_file)}): "
              f"{len(eval_frames)} frames, "
              f"pos_mae={np.mean(results['pos_mae'][-len(eval_frames)*9:])*1000:.1f}mm, "
              f"pos_cos={np.mean(results['pos_cosine'][-len(eval_frames)*9:]):.3f}")

    # 汇总
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {len(val_files)}, Total comparisons: {len(results['pos_mse'])}")
    print()
    print("Position (body-frame delta):")
    print(f"  RMSE: {np.mean(results['pos_mse'])**0.5*1000:.2f}mm")
    print(f"  MAE:  {np.mean(results['pos_mae'])*1000:.2f}mm")
    print(f"  Cosine Sim: {np.mean(results['pos_cosine']):.4f}")
    print()
    print("Rotation (body-frame rot6d):")
    print(f"  RMSE: {np.mean(results['rot_mse'])**0.5:.4f}")
    print(f"  MAE:  {np.mean(results['rot_mae']):.4f}")
    print(f"  Cosine Sim: {np.mean(results['rot_cosine']):.4f}")
    print()
    print(f"Gripper MAE: {np.mean(results['grip_mae']):.4f}")

    pos_cos = np.mean(results['pos_cosine'])
    pos_mae_mm = np.mean(results['pos_mae']) * 1000
    print()
    if pos_cos > 0.8 and pos_mae_mm < 15:
        print(">>> GOOD — 方向正确, 精度可接受")
    elif pos_cos > 0.5:
        print(">>> FAIR — 大致方向对, 精度待提升")
    elif pos_cos > 0.0:
        print(">>> POOR — 方向基本随机, 模型没学好")
    else:
        print(">>> BROKEN — 方向完全反, 预训练权重可能没加载")


if __name__ == "__main__":
    main()
