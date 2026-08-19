"""测试连接 Pi0.5 server (双相机输入)"""
import numpy as np
import sys
sys.path.insert(0, '/home/simpleai/pi05-deploy')

from openpi_client import websocket_client_policy as ws

policy = ws.WebsocketClientPolicy(host="192.168.3.148", port=8081)

# 构造测试输入
obs = {
    "cam_right_top": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
    "cam_right_btm": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
    "state": np.zeros(20, dtype=np.float32),
    "prompt": "Pick up the blocks, and place it into the box.",
}
obs["state"][3:9] = [1, 0, 0, 0, 1, 0]

print("Connecting to 192.168.3.148:8081...")
result = policy.infer(obs)
actions = np.asarray(result["actions"])
print(f"OK! actions shape: {actions.shape}")
print(f"First action: {actions[0][:10].round(4)}")
