"""eval_model.py — 模型离线评估 (MSE / MAE / Cosine Similarity)

用训练数据的一部分做验证，对比模型预测 vs ground truth action。

用法:
    # 在部署机上 (model server 在运行)
    python eval_model.py --parquet /path/to/episode_000854.parquet --port 8001

    # 指定评估帧数 (默认每 10 帧采样一次)
    python eval_model.py --parquet data.parquet --port 8001 --sample-interval 5
"""
import argparse
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation


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


def compute_ground_truth_actions(states, t, horizon=10):
    """计算 frame t 的 ground truth body-frame delta actions (和训练时 transform 一致)

    Pipeline: DeltaActions → GlobalToBodyDelta → (RelativeState 只影响 state, 不影响 action 值)

    Args:
        states: (N, 20) 所有帧的绝对位姿
        t: 当前帧索引
        horizon: action chunk 长度

    Returns:
        gt_actions: (horizon, 10) body-frame delta actions
    """
    N = len(states)
    gt = np.zeros((horizon, 10), dtype=np.float32)

    state_t = states[t]  # 当前帧绝对位姿
    R_current = rot6d_to_matrix(state_t[3:9])

    for i in range(horizon):
        future_t = min(t + i, N - 1)
        action_abs = states[future_t].copy()

        # Step 1: DeltaActions — action -= state (对 pose 9D, gripper 不减)
        delta_pos = action_abs[:3] - state_t[:3]           # global frame position delta
        delta_rot6d = action_abs[3:9] - state_t[3:9]       # linear rot6d subtraction (DeltaActions)
        gripper = action_abs[9]                              # absolute gripper

        # Step 2: GlobalToBodyDelta
        # Position: R_current.T @ global_delta
        body_pos = R_current.T @ delta_pos

        # Rotation: recover original rot6d, then geometric relative
        recovered_rot6d = delta_rot6d + state_t[3:9]  # undo DeltaActions subtraction
        R_target = rot6d_to_matrix(recovered_rot6d)
        R_relative = R_current.T @ R_target
        body_rot6d = matrix_to_rot6d(R_relative)

        gt[i, :3] = body_pos
        gt[i, 3:9] = body_rot6d
        gt[i, 9] = gripper

    return gt


def decode_image(img_data):
    """从 parquet 的 image dict 解码图像"""
    if isinstance(img_data, dict) and 'bytes' in img_data:
        return np.array(Image.open(BytesIO(img_data['bytes'])))
    return np.array(img_data)


def cosine_sim(a, b):
    """向量余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions vs ground truth")
    parser.add_argument("--parquet", type=str, required=True, help="Path to validation parquet file")
    parser.add_argument("--port", type=int, default=8001, help="Model server port")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="每隔 N 帧评估一次 (避免太慢)")
    parser.add_argument("--horizon", type=int, default=10, help="Action chunk horizon")
    parser.add_argument("--prompt", type=str, default="pick up the cup")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    # 加载数据
    df = pd.read_parquet(args.parquet)
    states = np.stack(df['state'].values)
    N = len(df)
    print(f"Loaded {N} frames from {args.parquet}")

    # 连接 model server
    from openpi_client import websocket_client_policy as ws
    policy = ws.WebsocketClientPolicy(host=args.host, port=args.port)

    # Warm up
    dummy_img = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    dummy_state = np.zeros(20, dtype=np.float32)
    dummy_state[3:9] = [1, 0, 0, 0, 1, 0]
    policy.infer({"cam_right": dummy_img, "state": dummy_state, "prompt": args.prompt})
    print("Server connected & warmed up")

    # 评估
    eval_frames = list(range(0, N - args.horizon, args.sample_interval))
    print(f"Evaluating {len(eval_frames)} frames (interval={args.sample_interval})")

    results = {
        'pos_mse': [], 'pos_mae': [], 'pos_cosine': [],
        'rot_mse': [], 'rot_mae': [], 'rot_cosine': [],
        'grip_mae': [],
        'full_mse': [], 'full_cosine': [],
    }

    for idx, t in enumerate(eval_frames):
        # Ground truth
        gt = compute_ground_truth_actions(states, t, args.horizon)

        # Model input: identity state + gripper
        state_input = np.zeros(20, dtype=np.float32)
        state_input[3:9] = [1, 0, 0, 0, 1, 0]
        state_input[9] = states[t, 9]  # gripper from data

        # 解码图像
        img = decode_image(df.iloc[t]['cam_right'])
        if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
            import cv2
            img = cv2.resize(img, (args.image_size, args.image_size))

        # 推理
        result = policy.infer({
            "cam_right": img,
            "state": state_input,
            "prompt": args.prompt,
        })
        pred = np.asarray(result["actions"])[:args.horizon, :10]

        # 逐步对比 (跳过 step 0, 因为 step 0 ≈ 0)
        for step in range(1, args.horizon):
            gt_step = gt[step]
            pred_step = pred[step]

            # Position (dims 0:3)
            pos_err = gt_step[:3] - pred_step[:3]
            results['pos_mse'].append(float(np.mean(pos_err**2)))
            results['pos_mae'].append(float(np.mean(np.abs(pos_err))))
            results['pos_cosine'].append(cosine_sim(gt_step[:3], pred_step[:3]))

            # Rotation (dims 3:9)
            rot_err = gt_step[3:9] - pred_step[3:9]
            results['rot_mse'].append(float(np.mean(rot_err**2)))
            results['rot_mae'].append(float(np.mean(np.abs(rot_err))))
            results['rot_cosine'].append(cosine_sim(gt_step[3:9], pred_step[3:9]))

            # Gripper (dim 9)
            results['grip_mae'].append(float(abs(gt_step[9] - pred_step[9])))

            # Full action
            full_err = gt_step[:10] - pred_step[:10]
            results['full_mse'].append(float(np.mean(full_err**2)))
            results['full_cosine'].append(cosine_sim(gt_step[:10], pred_step[:10]))

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(eval_frames)}] pos_mae={np.mean(results['pos_mae'][-9:])*1000:.1f}mm "
                  f"pos_cos={np.mean(results['pos_cosine'][-9:]):.3f} "
                  f"grip_mae={np.mean(results['grip_mae'][-9:]):.3f}")

    # 汇总
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Frames evaluated: {len(eval_frames)}")
    print(f"Steps per frame: {args.horizon - 1} (skip step 0)")
    print()
    print("Position (body-frame delta):")
    print(f"  MSE:  {np.mean(results['pos_mse']):.6f} ({np.mean(results['pos_mse'])**0.5*1000:.2f}mm RMSE)")
    print(f"  MAE:  {np.mean(results['pos_mae'])*1000:.2f}mm")
    print(f"  Cosine Sim: {np.mean(results['pos_cosine']):.4f}")
    print()
    print("Rotation (body-frame rot6d):")
    print(f"  MSE:  {np.mean(results['rot_mse']):.6f}")
    print(f"  MAE:  {np.mean(results['rot_mae']):.4f}")
    print(f"  Cosine Sim: {np.mean(results['rot_cosine']):.4f}")
    print()
    print(f"Gripper MAE: {np.mean(results['grip_mae']):.4f}")
    print()
    print("Full action (10D):")
    print(f"  MSE:  {np.mean(results['full_mse']):.6f}")
    print(f"  Cosine Sim: {np.mean(results['full_cosine']):.4f}")

    # 判断模型质量
    pos_cos = np.mean(results['pos_cosine'])
    pos_mae = np.mean(results['pos_mae']) * 1000
    print()
    if pos_cos > 0.8 and pos_mae < 15:
        print(">>> 模型质量: 良好 (方向正确, 精度可接受)")
    elif pos_cos > 0.5 and pos_mae < 30:
        print(">>> 模型质量: 一般 (大致方向对, 精度不够)")
    elif pos_cos > 0.0:
        print(">>> 模型质量: 差 (方向基本随机)")
    else:
        print(">>> 模型质量: 极差 (方向完全反, 可能权重没加载)")


if __name__ == "__main__":
    main()
