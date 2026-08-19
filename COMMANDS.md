# 部署常用指令

## 1. 初始化

### CAN 总线
sudo slcand -o -c -s8 /dev/ttyACM0 can3 && sudo ip link set up can3

### 机器人路由
sudo ip route add 192.168.3.254/32 dev enx6c1ff7bbfc25

## 2. 启动模型 Server

### BB 31k (积木)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate openpi && cd ~/pi05-deploy/openpi-official && python scripts/serve_policy.py --default-prompt 'building blocks into box' --port 8000 policy:checkpoint --policy.config pi05_umi6_bb_h20 --policy.dir ~/pi05-deploy/checkpoint_bb_h20_31k/31000

### BB 24k
source ~/anaconda3/etc/profile.d/conda.sh && conda activate openpi && cd ~/pi05-deploy/openpi-official && python scripts/serve_policy.py --default-prompt 'building blocks into box' --port 8000 policy:checkpoint --policy.config pi05_umi6_bb_h20 --policy.dir ~/pi05-deploy/checkpoint_bb_h20_24k/24000

## 3. 运行

### 16步 + 旋转0.05 (推荐)
cd ~/pi05-deploy && python3 ur7e_main_simple.py --cam-right-id 0 --steps-per-inference 16 --max-rot-step 0.05 --server-port 8000 --prompt 'building blocks into box'

### 16步 + 冻结旋转
cd ~/pi05-deploy && python3 ur7e_main_simple.py --cam-right-id 0 --steps-per-inference 16 --max-rot-step 0.0 --server-port 8000 --prompt 'building blocks into box'

### 8步
cd ~/pi05-deploy && python3 ur7e_main_simple.py --cam-right-id 0 --steps-per-inference 8 --max-rot-step 0.05 --server-port 8000 --prompt 'building blocks into box'

## 4. 夹爪

### 打开
cd ~/pi05-deploy && echo -e 'SET -700.00\nQUIT' | ./x3arm-can-demo-gripper can3 8 --daemon --kp 10.0 --kd 1.0 --loop-hz 100

### 关闭
cd ~/pi05-deploy && echo -e 'SET 0.00\nQUIT' | ./x3arm-can-demo-gripper can3 8 --daemon --kp 10.0 --kd 1.0 --loop-hz 100
