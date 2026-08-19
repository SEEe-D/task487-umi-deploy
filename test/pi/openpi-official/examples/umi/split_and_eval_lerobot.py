"""
按照扩散策略 (Diffusion Policy) 的方式对 LeRobot 数据集做 train/test 分割，
并在 test set 上评估模型的 action 预测误差。

方法：
- 按 episode 级别随机划分，5% 的 episode 作为 test set (val_ratio=0.05)
- 与 UMI diffusion policy 中的 get_val_mask() 方法一致
- 生成分割后的 info 供训练使用（修改 splits 字段）
- 评估时计算 action MSE、state MSE 等指标

Usage:
    # 1. 分割数据集（生成 train/test episode 列表）
    python split_and_eval_lerobot.py split --dataset-path ~/.cache/huggingface/lerobot/local/umi6_single_arm_6d_rel

    # 2. 评估（训练完成后，对 test set 计算指标）
    python split_and_eval_lerobot.py eval --dataset-path ~/.cache/huggingface/lerobot/local/umi6_single_arm_6d_rel \
        --policy-config pi05_umi6_single_task_split --checkpoint-dir checkpoints/pi05_umi6_single_task_split/umi6_split/20000
"""

import json
import os
import pathlib
from typing import Optional

import click
import numpy as np
import pandas as pd


# ============== 辅助函数 ==============

def _load_episodes_meta(meta_dir: pathlib.Path) -> dict[int, int]:
    """从 meta/episodes/ 读取每个 episode 的帧数, 返回 {episode_index: length}"""
    import glob as _glob
    episodes_dir = meta_dir / "episodes"
    parquet_files = sorted(_glob.glob(str(episodes_dir / "**" / "*.parquet"), recursive=True))
    result = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index", "length"])
        for _, row in df.iterrows():
            result[int(row["episode_index"])] = int(row["length"])
    return result


def _load_episodes_meta_full(meta_dir: pathlib.Path) -> dict[int, dict]:
    """从 meta/episodes/ 读取完整 episode 元信息, 返回 {episode_index: {length, chunk_index, file_index}}"""
    import glob as _glob
    episodes_dir = meta_dir / "episodes"
    parquet_files = sorted(_glob.glob(str(episodes_dir / "**" / "*.parquet"), recursive=True))
    result = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index", "length", "data/chunk_index", "data/file_index"])
        for _, row in df.iterrows():
            result[int(row["episode_index"])] = {
                "length": int(row["length"]),
                "chunk_index": int(row["data/chunk_index"]),
                "file_index": int(row["data/file_index"]),
            }
    return result


# ============== 与 Diffusion Policy 一致的 val_mask 生成 ==============

def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 42) -> np.ndarray:
    """生成验证集 mask，与 diffusion_policy/common/sampler.py 中的实现一致。

    按 episode 级别随机划分，确保至少 1 个 episode 用于验证，至少 1 个用于训练。

    Args:
        n_episodes: 总 episode 数量
        val_ratio: 验证集比例 (例如 0.05 = 5%)
        seed: 随机种子，保证可复现

    Returns:
        bool array, True 表示该 episode 属于验证集
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


# ============== 分割命令 ==============

@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset-path', required=True, help='LeRobot 数据集路径')
@click.option('--val-ratio', default=0.05, type=float, help='验证集比例 (默认 0.05)')
@click.option('--seed', default=42, type=int, help='随机种子')
def split(dataset_path: str, val_ratio: float, seed: int):
    """按 episode 级别分割数据集为 train/test"""
    dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"

    # 读取 info.json
    with open(info_path, "r") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    print(f"数据集: {dataset_path}")
    print(f"总 episodes: {total_episodes}")

    # 生成 val mask
    val_mask = get_val_mask(total_episodes, val_ratio, seed)
    train_indices = np.where(~val_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()

    print(f"训练集: {len(train_indices)} episodes")
    print(f"测试集: {len(val_indices)} episodes")
    print(f"测试集 episode indices: {val_indices}")

    # 统计各集的帧数 (从 episodes 元数据读取)
    episodes_meta = _load_episodes_meta(meta_dir)
    train_frames = 0
    val_frames = 0
    for ep_idx in range(total_episodes):
        n_frames = episodes_meta.get(ep_idx, 0)
        if val_mask[ep_idx]:
            val_frames += n_frames
        else:
            train_frames += n_frames

    print(f"训练集帧数: {train_frames}")
    print(f"测试集帧数: {val_frames}")

    # 保存分割信息
    split_info = {
        "val_ratio": val_ratio,
        "seed": seed,
        "total_episodes": total_episodes,
        "train_episodes": train_indices,
        "val_episodes": val_indices,
        "train_frames": train_frames,
        "val_frames": val_frames,
    }

    split_path = meta_dir / "train_test_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n分割信息已保存到: {split_path}")

    # 修改 info.json 的 splits 字段，用于 openpi 训练时只用 train set
    # 原始: "train": "0:68"
    # 新增: 保存 train/val episode 列表供后续使用
    info["splits"]["train_original"] = info["splits"].get("train", f"0:{total_episodes}")

    # 生成 train-only 的 episode 范围（LeRobot 格式）
    # 注意: LeRobot 的 splits 字段是简单的 "start:end" 格式
    # 如果要精确过滤，需要在 data_loader 层面处理
    # 这里保存完整信息供参考
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n下一步:")
    print(f"  1. 用 split 版 config 训练 (只用 train episodes):")
    print(f"     bash scripts/finetune_umi6_split.sh")
    print(f"  2. 训练完成后，运行 eval 命令评估 test set:")
    print(f"     python split_and_eval_lerobot.py eval \\")
    print(f"       --dataset-path {dataset_path} \\")
    print(f"       --policy-config pi05_umi6_single_task_split \\")
    print(f"       --checkpoint-dir checkpoints/pi05_umi6_single_task_split/<exp_name>/<step>")


# ============== 评估命令 ==============

@cli.command()
@click.option('--dataset-path', required=True, help='LeRobot 数据集路径')
@click.option('--policy-config', default='pi05_umi6_single_task_split', help='策略配置名')
@click.option('--checkpoint-dir', required=True, help='模型 checkpoint 目录')
@click.option('--max-samples', default=0, type=int, help='最大评估样本数 (0=全部)')
@click.option('--device', default=None, help='PyTorch device (默认自动检测)')
def eval(dataset_path: str, policy_config: str, checkpoint_dir: str,
         max_samples: int, device: str):
    """在 test set 上评估模型的 action 预测误差"""
    import time
    import torch

    dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
    checkpoint_dir = pathlib.Path(os.path.expanduser(checkpoint_dir))
    meta_dir = dataset_path / "meta"

    # ---- 1. 读取分割信息 ----
    split_path = meta_dir / "train_test_split.json"
    if not split_path.exists():
        print("错误: 请先运行 split 命令生成分割信息")
        return

    with open(split_path, "r") as f:
        split_info = json.load(f)

    val_episodes = split_info["val_episodes"]
    print(f"测试集 episodes: {val_episodes}")
    print(f"测试集帧数: {split_info['val_frames']}")

    # ---- 2. 加载 openpi config 和 policy ----
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.training import data_loader as _data_loader
    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    except ModuleNotFoundError:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

    print(f"\n=== 加载配置: {policy_config} ===")
    config = _config.get_config(policy_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.model.action_horizon

    print(f"=== 加载模型: {checkpoint_dir} ===")
    policy = _policy_config.create_trained_policy(
        config,
        str(checkpoint_dir),
        repack_transforms=data_config.repack_transforms,
        pytorch_device=device,
    )
    print("模型加载成功!")

    # ---- 3. 创建 LeRobot 数据集 ----
    print(f"\n=== 加载数据集: {data_config.repo_id} ===")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    full_dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        tolerance_s=1e9,
    )
    print(f"数据集总样本数: {len(full_dataset)}")

    # ---- 4. 计算 test episode 的数据集索引 ----
    print("计算 test episode 索引...")
    episodes_meta = _load_episodes_meta(meta_dir)
    with open(meta_dir / "info.json", "r") as f:
        info = json.load(f)
    total_episodes = info["total_episodes"]

    # 构建 episode -> (start_idx, end_idx) 映射
    cumulative = 0
    episode_ranges = {}
    for ep_idx in range(total_episodes):
        n_frames = episodes_meta.get(ep_idx, 0)
        episode_ranges[ep_idx] = (cumulative, cumulative + n_frames)
        cumulative += n_frames

    # 收集 test indices (跳过 episode 末尾 action_horizon-1 帧，避免不完整序列)
    test_indices = []
    for ep_idx in val_episodes:
        start, end = episode_ranges[ep_idx]
        safe_end = max(start + 1, end - action_horizon + 1)
        for i in range(start, safe_end):
            test_indices.append(i)

    print(f"Test 样本数: {len(test_indices)}")

    if max_samples > 0 and len(test_indices) > max_samples:
        step = max(1, len(test_indices) // max_samples)
        test_indices = test_indices[::step][:max_samples]
        print(f"下采样到: {len(test_indices)} 样本")

    # ---- 5. 逐样本推理评估 ----
    print(f"\n=== 开始评估 ({len(test_indices)} 样本) ===")
    all_mse = []
    all_per_dim_mse = []
    timings = []

    for i, idx in enumerate(test_indices):
        item = full_dataset[idx]

        # 转换 torch.Tensor → numpy (policy.infer 期望 numpy)
        obs = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                obs[key] = value.numpy()
            elif hasattr(value, 'convert'):  # PIL Image
                obs[key] = np.array(value)
            else:
                obs[key] = value

        # 添加 prompt（policy.infer 需要）
        if "prompt" not in obs:
            obs["prompt"] = "building blocks into box"

        # ground truth actions: (action_horizon, 20)
        gt_actions = np.asarray(obs.get("action", obs.get("actions", None)))

        # 模型推理
        t0 = time.time()
        with torch.no_grad():
            result = policy.infer(obs)
        dt = time.time() - t0
        timings.append(dt)

        # 预测 actions: (action_horizon, 20) — 经过 UmiOutputs6D 截断
        pred_actions = np.asarray(result["actions"])

        # 对齐维度后计算 MSE
        min_h = min(pred_actions.shape[0], gt_actions.shape[0])
        min_d = min(pred_actions.shape[1], gt_actions.shape[1])
        pred = pred_actions[:min_h, :min_d]
        gt = gt_actions[:min_h, :min_d]

        mse = np.mean((pred - gt) ** 2)
        per_dim = np.mean((pred - gt) ** 2, axis=0)

        all_mse.append(mse)
        all_per_dim_mse.append(per_dim)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_indices)}] MSE={mse:.6f}, 推理耗时={dt:.3f}s")

    # ---- 6. 汇总结果 ----
    all_mse = np.array(all_mse)
    all_per_dim_mse = np.stack(all_per_dim_mse)
    mean_per_dim = np.mean(all_per_dim_mse, axis=0)

    print(f"\n{'='*60}")
    print(f"评估结果 ({len(test_indices)} 样本, {len(val_episodes)} test episodes)")
    print(f"{'='*60}")
    print(f"Overall Action MSE:  {np.mean(all_mse):.6f}")
    print(f"Median  Action MSE:  {np.median(all_mse):.6f}")
    print(f"Std     Action MSE:  {np.std(all_mse):.6f}")
    print(f"Min     Action MSE:  {np.min(all_mse):.6f}")
    print(f"Max     Action MSE:  {np.max(all_mse):.6f}")

    # 20D 维度标签: [right(10), left(10)]
    dim_labels = [
        "R_dx", "R_dy", "R_dz",
        "R_r1", "R_r2", "R_r3", "R_r4", "R_r5", "R_r6",
        "R_grip",
        "L_dx", "L_dy", "L_dz",
        "L_r1", "L_r2", "L_r3", "L_r4", "L_r5", "L_r6",
        "L_grip",
    ]

    print(f"\n各维度 MSE:")
    for j in range(min(len(dim_labels), len(mean_per_dim))):
        print(f"  {dim_labels[j]:8s}: {mean_per_dim[j]:.6f}")

    # 按组汇总
    if len(mean_per_dim) >= 20:
        groups = {
            "Right pos (dx,dy,dz)":  mean_per_dim[0:3],
            "Right rot_6d":          mean_per_dim[3:9],
            "Right gripper":         mean_per_dim[9:10],
            "Left pos (dx,dy,dz)":   mean_per_dim[10:13],
            "Left rot_6d":           mean_per_dim[13:19],
            "Left gripper":          mean_per_dim[19:20],
        }
        print(f"\n分组 MSE:")
        for name, vals in groups.items():
            print(f"  {name:25s}: {np.mean(vals):.6f}")

    # 推理速度
    print(f"\n推理速度:")
    print(f"  平均: {np.mean(timings):.3f}s/sample")
    print(f"  中位: {np.median(timings):.3f}s/sample")
    print(f"  总计: {np.sum(timings):.1f}s")

    # 保存结果到文件
    results = {
        "checkpoint": str(checkpoint_dir),
        "config": policy_config,
        "test_episodes": val_episodes,
        "n_samples": len(test_indices),
        "overall_mse": float(np.mean(all_mse)),
        "median_mse": float(np.median(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "per_dim_mse": {dim_labels[j]: float(mean_per_dim[j])
                        for j in range(min(len(dim_labels), len(mean_per_dim)))},
        "avg_inference_time_s": float(np.mean(timings)),
    }

    results_path = checkpoint_dir / "eval_results.json"
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {results_path}")
    except Exception as e:
        print(f"\n保存结果失败 ({e}), 打印 JSON:")
        print(json.dumps(results, indent=2))


# ============== 检查数据集命令 ==============

@cli.command()
@click.option('--dataset-path', required=True, help='LeRobot 数据集路径')
@click.option('--episode', default=0, type=int, help='要检查的 episode index')
def check(dataset_path: str, episode: int):
    """检查数据集中某个 episode 的数据"""
    dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
    meta_dir = dataset_path / "meta"

    # 从 episodes 元数据找到对应的 data file
    episodes_meta_raw = _load_episodes_meta_full(meta_dir)
    if episode not in episodes_meta_raw:
        print(f"Episode {episode} 不存在")
        return

    ep_info = episodes_meta_raw[episode]
    chunk_idx = ep_info["chunk_index"]
    file_idx = ep_info["file_index"]
    data_dir = dataset_path / "data" / f"chunk-{chunk_idx:03d}"
    parquet_path = data_dir / f"file-{file_idx:03d}.parquet"
    if not parquet_path.exists():
        print(f"数据文件不存在: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    # 过滤到指定 episode
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode]
    print(f"Episode {episode}: {len(df)} frames")
    print(f"Columns: {df.columns.tolist()}")

    # 检查 state 和 action
    for col in df.columns:
        vals = df[col].values
        if hasattr(vals[0], '__len__'):
            arr = np.array([list(v) for v in vals])
            print(f"\n{col}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"  min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
        else:
            print(f"\n{col}: dtype={vals.dtype}, min={vals.min()}, max={vals.max()}")


if __name__ == "__main__":
    cli()
