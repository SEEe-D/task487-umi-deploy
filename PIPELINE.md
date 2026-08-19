DeltaActions：算出"从首帧到目标帧的位移量"
  delta = pos[40] - pos[35] = [+0.015, +0.019, +0.028]
  这个位移是用 tag 全局坐标系的 xyz 描述的。

  GlobalToBodyDelta：把这个位移量换一种描述方式
  delta_local = R_current.T @ [+0.015, +0.019, +0.028] = [+0.022, +0.008, -0.014]
  同样的物理位移，改用 夹爪自己的 xyz 描述。

  打个比方：你从 A 走到 B。

  - DeltaActions 告诉你："往北走 3 米，往东走 4 米"（用地图方向描述）
  - GlobalToBodyDelta 告诉你："往前走 5 米"（用你面朝的方向描述）

  物理移动一样，坐标系不同。"往北 3 米往东 4 米"在不同地图上方向不同（tag 地图 vs robot 地图）。但"往前走
   5 米"在任何地图上都一样——因为它是相对你自己的朝向。

  所以 DeltaActions 算的是"移动了多少"，GlobalToBodyDelta 把"多少"从地图方向转成夹爪自身方向。

❯ 那你直接GlobalToBodyDelta再计算移动多少不行吗 这个有先后要求吗

● 数学上结果一样：

  先 DeltaActions 再 GlobalToBodyDelta:
    R.T @ (pos[40] - pos[35])

  先 GlobalToBodyDelta 再 DeltaActions:
    R.T @ pos[40] - R.T @ pos[35]

  同一个结果。但当前实现有先后要求：GlobalToBodyDelta 只改 action 的位置，不改 state。如果先跑
  GlobalToBodyDelta：

  action_pos → R.T @ action_pos (body frame 绝对位置)
  state_pos → 还是 global 绝对位置
  DeltaActions: body_action_pos -= global_state_pos → 混了两个坐标系，错了

  先跑 DeltaActions：
  action_pos → action_pos - state_pos (global frame delta)
  GlobalToBodyDelta: R.T @ global_delta → body frame delta ✓

  当然也可以合成一步（R.T @ (action - state)），但 DeltaActions 同时还处理旋转的 rot6d
  减法，不能跳过。所以分两步更清晰。