"""生成 LeRobot v3.0 -> 新版所需的 tasks.jsonl 和 episodes.jsonl"""
import pandas as pd
import json
import sys

base = sys.argv[1] if len(sys.argv) > 1 else "data/local/umi6_single_arm_6d_rel/meta"

# 1. 生成 tasks.jsonl
tasks_parquet = f"{base}/tasks.parquet"
try:
    tdf = pd.read_parquet(tasks_parquet)
    with open(f"{base}/tasks.jsonl", "w") as f:
        for task_name, row in tdf.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": str(task_name)}) + "\n")
    print(f"tasks.jsonl: {len(tdf)} tasks")
except Exception as e:
    # fallback
    with open(f"{base}/tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "building blocks into box"}) + "\n")
    print(f"tasks.jsonl: fallback (1 task), error was: {e}")

# 2. 生成 episodes.jsonl
import glob
parquet_files = sorted(glob.glob(f"{base}/episodes/**/*.parquet", recursive=True))
if not parquet_files:
    print("ERROR: no episodes parquet found")
    sys.exit(1)

all_rows = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    all_rows.append(df)

df = pd.concat(all_rows, ignore_index=True)

with open(f"{base}/episodes.jsonl", "w") as f:
    for _, row in df.iterrows():
        tasks_val = row.get("tasks", ["building blocks into box"])
        if isinstance(tasks_val, str):
            tasks_val = [tasks_val]
        elif not isinstance(tasks_val, list):
            tasks_val = [str(tasks_val)]
        ep = {
            "episode_index": int(row["episode_index"]),
            "tasks": tasks_val,
            "length": int(row["length"]),
        }
        f.write(json.dumps(ep) + "\n")

print(f"episodes.jsonl: {len(df)} episodes")
print("Done!")
