# Task487 Pi0.5 deployment

This is the formal three-camera Task487 path. It now defaults to the newly
trained raw-head checkpoint and cloud training contract
`pi05_umi_task487_raw_12_5`, with 12.5 Hz policy waypoints. The client preloads
five actions, then starts the next asynchronous inference after two physically
completed actions or whenever only three safe actions remain. Those already
committed actions form the hard RTC prefix, and the generated suffix refills the
timeline.

Raw-head, masked-head, and wrist-only operator outcomes are tracked in
`TASK487_HEAD_INPUT_ABLATION_LOG.md`. Keep per-group behavioral corrections out
of the deployment until the ablation comparison is complete.
Chunk handoff now uses model-side action-prefill RTC: the committed robot-target
prefix is converted back through the UMI 20D action contract, normalized by the
regular policy transforms, and hard-clamped during every Pi0.5 flow step. The
generated suffix therefore attends to the exact old-chunk prefix. When inference
returns, waypoints completed during inference are removed and the generated suffix
is appended after the still-pending hard prefix. A three-action handoff blend is
applied only to the generated suffix. It does not enable geometry masking.

The server advertises:

```text
rtc_enabled=True
rtc_mode=action_prefill_hard_inpainting_v1
rtc_prefix_steps=5
```

`rtc_prefix_steps=5` is the server capability/warmup contract; the rolling
client normally sends three or fewer pending actions at a chunk boundary.

Adjacent policy targets use 45 mm translation and 0.14 rad rotation outlier
guards, chosen above the new 12.5 Hz dataset's 99.9th percentiles. These are
separate from the 0.02 m/s and 0.08 rad/s physical speed limits: accepted targets
are retimed to those physical speeds before dispatch.

The first three generated suffix actions cross-fade against the safe replaceable
old tail; normal online logs should show `blend=3` at RTC handoffs.

The 20-step horizon now spans 1.6 seconds, with the first target at +0.08 s and
state history at `[-0.08, 0.0]` seconds. The client derives all scheduler timing
from the validated server metadata and rejects mismatched 25/12.5 Hz contracts.

RGB preprocessing is selected by the server runtime contract. The raw 12.5 Hz
runtime preserves the full 640x512 Thor field of view using `resize_with_pad`:
it resizes to 224x179 and adds 22 black rows above and 23 below, matching raw
UMI training. The older pure-real and B1 runtimes retain their center-square
crop, which removes 64 pixels from each horizontal edge before resizing. The
contract rejects any non-224x224 policy input.

The absolute TCP floor and per-ACTIVE downward-excursion guard are temporarily
disabled for diagnosis. Step/tracking guards and bounded-round HOLD remain active.

The client refuses a server without this metadata. Startup performs both a plain
and an RTC-prefill warmup while HOLD, so the first live handoff does not trigger a
new JAX compile.

Exact prompts are also runtime-specific. Raw 12.5 Hz:

- `vegetable`: `Pick Up Vegetable and Place Vegetable on the Pink Plate on the Right`
- `fruit`: `Pick Up Fruit and Place Fruit on the Blue Plate on the Left`

Older pure-real/B1:

- `vegetable`: `Pick Up Vegetable and Place Vegetable on the Pink  Plate on the Right`
- `fruit`: `Pick Up Fruit and Place Fruit on the Blue Plate on the Left`

The client starts in `HOLD` and defaults to dry-run plus a five-waypoint single
round. For real execution, `r` returns to HOME and prepares the checkpoint's
demonstration boundary state: raw 12.5 Hz commands 1/1 degrees (closed), while
older pure-real/B1 commands 35/35 degrees (open). `d` is refused until both
grippers are within the runtime-specific tolerance of that target. Clear hands
and objects from the grippers before pressing `r` with the raw runtime. Then `d`
starts a fresh round, `s` enters HOLD and clears model scheduling, and Ctrl+C
commands HOLD before stopping.

Server (new raw 12.5 Hz checkpoint is the default):

```bash
bash run_task487_server.sh
```

To select a checkpoint or roll back deliberately, pass its path and port and set
the matching config, for example:

```bash
TASK487_POLICY_CONFIG=pi05_umi_task487 \
  bash run_task487_server.sh /absolute/path/to/old/checkpoint 8000
```

Observation-only client (connects sensors/controllers but emits no policy targets):

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000
```

Show the exact three 224x224 RGB images placed in every Pi0.5 request.  The
window refreshes only when a real policy request is submitted, so it is not a
separate unsynchronized camera monitor.  On the local desktop, use display 1:

```bash
export DISPLAY=:1
bash run_task487_client.sh vegetable 127.0.0.1 8000 --show-cameras
```

The labeled header is outside the image pixels.  The displayed panels remain
the exact `cam_head`, `cam_left_top`, and `cam_right_top` inputs before numeric
normalization.

First approved five-waypoint real-robot round:

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute
```

The first RTC handoff should be tested with a bounded five-waypoint round. This
executes one two-waypoint replan/handoff and then automatically enters HOLD.
After startup, press `r`, wait for `Grippers ready`, check the scene, and only
then press `d`:

```bash
bash run_task487_client.sh vegetable 127.0.0.1 8000 --execute --max-waypoints 5
```

Unbounded command output still requires the explicit `--continuous` option.
Every operator-started round carries a round id; asynchronous inference from a
previous HOLD/HOME round is discarded instead of being merged into a new round.

For the current diagnostic run, the absolute/activation-relative TCP height
guard is disabled in `task487_client.py`. Translation/rotation speed limits,
step/tracking checks, bounded-round HOLD, and RTC round isolation remain active.

The arm backend is the local Tianji/Marvin Mink ROS bridge (not UR RTDE). The
client uses no robot IP addresses and communicates with the bridge locally. The
verified Thor mapping is training `head_main`=`cam_head_left` on 5000,
left-up=`cam_hand_l_top` on 5002, and right-up=`cam_hand_r_top` on 5004, with
metadata on 6000/6002/6004. Port 5001 is the unused `head_main_stereo_right`
view. The Thor host defaults to `192.168.2.178`.

The main Thor config has been corrected to enable the left head camera and
disable the unused right head camera. Until an interactive-sudo service restart,
tmux session `task487-headleft` supplies 5000/6000 as a temporary sidecar. On
Thor, replace the sidecar with the corrected main service after sudo
authentication succeeds:

```bash
sudo -v && tmux kill-session -t task487-headleft && sudo systemctl restart thor-stream-sender.service
```

Contract and scheduler tests (no robot connection):

```bash
source /home/simpleai/anaconda3/etc/profile.d/conda.sh
conda activate openpi
PYTHONPATH="$PWD/openpi-official/src:$PWD/openpi-official/packages/openpi-client/src:$PWD/universal_manipulation_interface_ur:$PWD" \
  python -m pytest -q tests_task487
```

Read-only RTC checkpoint verification using the recorded HOLD observation:

```bash
python offline_eval/verify_task487_rtc.py \
  --capture offline_eval/task487_20260814/live_hold_150549 \
  --host 127.0.0.1 --port 8000 --prefix-steps 5
```

Synthetic closed-loop audit (read-only, never connects to ROS):

```bash
python offline_eval/audit_task487_closed_loop.py \
  --capture offline_eval/task487_20260814/live_hold_150549 \
  --host 127.0.0.1 --port 8000 --execute-steps 2 --chunks 12 --repeats 3 \
  --output offline_eval/task487_20260814/closed_loop_2step_rtc_audit.json
```
