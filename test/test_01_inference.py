"""测试1: 推理输出是 delta 还是 absolute

发送不同 state, 观察输出是否跟随 state 变化:
- 如果输出 ≈ state 附近 → absolute
- 如果输出 ≈ 0 附近且不随 state 变化 → delta
"""
import numpy as np
import time
from openpi_client import websocket_client_policy as ws

policy = ws.WebsocketClientPolicy(host="localhost", port=8000)

# 固定随机图像
np.random.seed(42)
img = np.random.randint(256, size=(256, 256, 3), dtype=np.uint8)

# norm_stats 参考值
state_mean = np.array([-0.036, 0.083, 0.542, 0.218, 0.563, 0.568, 0.301, -0.624, 0.353, 0.145,
                        0,0,0,0,0,0,0,0,0,0], dtype=np.float32)

print("=" * 70)
print("测试: 模型输出是 delta 还是 absolute?")
print("=" * 70)

tests = [
    ("零 state",        np.zeros(20, dtype=np.float32)),
    ("训练均值 state",   state_mean.copy()),
    ("偏移 state (x+0.2, y-0.1, z+0.1)", state_mean + np.array([0.2, -0.1, 0.1] + [0]*17, dtype=np.float32)),
]

results = []
for name, state in tests:
    obs = {"cam_right": img, "state": state, "prompt": "building blocks into box"}
    t0 = time.time()
    r = policy.infer(obs)
    dt = time.time() - t0
    actions = np.array(r["actions"])

    print(f"\n--- {name} ---")
    print(f"  输入 state[:3] = {state[:3]}")
    print(f"  输出 action[0][:3] = {actions[0,:3]}  (推理 {dt:.3f}s)")
    print(f"  差值 (output - state[:3]) = {actions[0,:3] - state[:3]}")
    results.append((name, state[:3].copy(), actions[0,:3].copy()))

print("\n" + "=" * 70)
print("分析:")
for name, s, a in results:
    print(f"  {name:25s}: output={a}, diff_from_state={a - s}")

print("\n结论:")
print("  如果 output 数值很小(接近0)且不随 state 变化 → 输出是 DELTA")
print("  如果 output ≈ state (差值很小) → 输出是 ABSOLUTE")
print("  如果 diff_from_state 三组都接近 → 输出是 DELTA (模型预测固定delta)")
