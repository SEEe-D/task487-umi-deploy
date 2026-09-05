# Task487 head-input ablation experiment log

This file records operator-visible outcomes separately from deployment audits.
Do not introduce a per-group TCP offset, inactive-arm lock, prompt change, or
other behavioral correction during the head-input ablation: those changes
would confound the comparison.

## Experiment matrix

2026-09-05 subsequent rollback (current scheme A): at the operator's request,
all three four-wrist contracts restore the original rolling replanning
(`complete_chunk_before_replan=False`), retaining the corrected sorting prompt.
The operator reported failure to open after grasping under full-chunk execution
(scheme B), whose observed replan intervals were about 5–12 seconds. This is a
controlled scheduling comparison, not a demonstrated release fix. Cameras,
gripper scaling, speed limits and safety checks are unchanged. No post-rollback
hardware outcome has been measured. Keep scheme A and B trial results separate.

2026-09-05 follow-up: operator reports that the new four-wrist runs often fail to
close enough to pick. Read-only audit found an incorrect legacy task prompt and
replanning that replaced speed-retimed closing tails. All three four-wrist groups
now use the verified training prompt `Vegetable and Fruit Sorting` and preserve
accepted chunk tails until dispatched. These are common deployment corrections,
not per-group tuning. Treat post-fix runs as a new comparison; do not pool their
outcomes with earlier runs. Legacy R0/W0 configurations are unchanged. Offline
replay shows tail dispatch progress, not a successful physical grasp. Post-fix
hardware outcomes remain unmeasured; details are in the four-wrist deployment note.

2026-09-05: the newly downloaded `cnc8gpu_seed42`, `raw4w_seed42`, and `wrist4w_seed42`
checkpoints use four wrist views. Deployment support and read-only verification are
recorded in [TASK487_FOUR_WRIST_DEPLOY.md](TASK487_FOUR_WRIST_DEPLOY.md).
The R0/W0 observations below concern the older two-wrist experiments; no new grasp
outcome has been recorded for the four-wrist checkpoints in this update.

| Run | Group | Head input | Checkpoint | Status | Primary observation |
|---|---|---|---|---|---|
| R0 | raw-head control | Original `observation.images.head_raw`, no mask | `pi05_umi_task487_raw_12_5/raw_seed42/29999` | Run observed | Both arms move; grippers open/close normally; grasp is offset |
| M0 | masked-head | TBD | TBD | Pending | — |
| W0 | wrist-only | Disabled; left/right wrist only | `pi05_umi_task487_wrist_only_12_5/wrist_only_seed42/29999` | Server ready; execution pending | Metadata verified; no robot trial recorded yet |

## R0 — raw-head control — 2026-08-18

### Purpose

Establish the unmodified original-head-image control. A repeatable grasp
localization error in this group, followed by an improvement in masked-head or
wrist-only under otherwise identical conditions, is evidence that the original
head image is harmful. The raw group must therefore remain uncorrected while
the ablation is being measured.

### Runtime contract

- Server config: `pi05_umi_task487_raw_12_5`
- Checkpoint: `checkpoints/pi05_umi_task487_raw_12_5/raw_seed42/29999`
- Server runtime: `pi05_umi_task487_12_5_v1`
- Control frequency / horizon: 12.5 Hz / 20 targets
- RTC: enabled, hard action-prefill, prefix capability 5
- Cameras: head + left upper wrist + right upper wrist
- Head preprocessing: full-FOV `resize_with_pad` to 224x224
- Head feature used in training: `observation.images.head_raw`
- Head mask: disabled
- Vegetable prompt: single-space `Pink Plate` form
- Boundary gripper target: right 1 degree, left 1 degree
- Gripper action conversion: direct radians to degrees; no inversion or
  quantile rescaling
- Client mode: real command, continuous
- Intervention: disabled (`enable_intervention:=false`)
- Lower stack: joint-impedance wrapper was used according to the launch
  terminal (`ARM_JOINT_IMPEDANCE=1`), delegating to
  `start_teleop_replay.sh`
- Known safety configuration: workspace height limit disabled; ordinary
  step/tracking guards and HOLD remain active

### Artifacts

- Client run directory:
  `task487_logs/20260818_154706_1119580/`
- Client console log:
  `task487_logs/20260818_154706_1119580/client.log`
- Exact policy chunks:
  `task487_logs/20260818_154706_1119580/policy_chunks/`
- Chunk summaries:
  `task487_logs/20260818_154706_1119580/chunk_diagnostics.jsonl`
- Per-gripper traces and waypoint records are in the same run directory.
- Dataset/deployment audit:
  `offline_eval/TASK487_RAW_12_5_AUDIT_20260818.md`
- All completed Task487 runs preceding R0 were consolidated on 2026-08-18
  into the verified archive
  `task487_logs/archive/task487_completed_runs_through_20260818_153733.tar.gz`.
  The active R0 directory was deliberately excluded from that archive.

### Timeline

- 15:47:07: client connected; runtime validated as raw 12.5 Hz with
  `resize_with_pad` and 1/1-degree gripper boundary targets.
- 15:47:11: plain and RTC-prefill warmups completed; client ready in HOLD.
- 15:47:31–15:47:37: HOME completed and grippers prepared.
- 15:47:39: vegetable round entered ACTIVE.
- 15:49:16: task switched during ACTIVE from vegetable to fruit; the client
  commanded physical HOLD, discarded the old inference, and started a fresh
  fruit round.
- 15:49:43: operator pressed HOLD.
- 15:49:45–15:49:50: HOME completed; client remained in HOLD and prepared the
  raw boundary gripper state.

### Operator observation

- Both arms move together.
- Grippers open and close normally.
- The grasp is offset from the object.
- Offset direction: not recorded.
- Approximate offset distance: not recorded.
- Repeatability across placements: not recorded.
- Per-task trial count and success count: not recorded.

The missing items above are unknown, not zero.

### Read-only evidence collected after the run

The double-arm motion is present in both the pure UMI data and the model output;
it is not a duplicated actuator command.

Pure UMI dataset, vegetable task (right arm is primary):

- 98.23% of adjacent-frame samples move both arms by more than 0.5 mm.
- In 98.95% of 20-step windows, both arms move by more than 5 mm.
- Median 20-step translation is 148.5 mm right and 66.9 mm left.
- The median cosine between simultaneous left/right step directions is 0.835.

Pure UMI dataset, fruit task (left arm is primary):

- 98.13% of adjacent-frame samples move both arms by more than 0.5 mm.
- In 99.14% of 20-step windows, both arms move by more than 5 mm.
- Median 20-step translation is 69.0 mm right and 143.2 mm left.
- The median cosine between simultaneous left/right step directions is 0.822.

R0 model chunks available at the time of analysis (211 chunks, first five raw
targets per chunk, vegetable and fruit phases combined):

- Median displacement from live pose: 7.01 mm right, 6.41 mm left.
- 52.13% of samples move both arms by more than 5 mm.
- Left/right displacement-norm correlation: 0.73.
- Median direction cosine: 0.73.
- The maximum difference between the two arm displacement vectors is 37.26 mm,
  proving that the deployment is not copying one arm target to the other.

### Interpretation boundary

- Normal gripper opening/closing supports the direct gripper mapping and does
  not explain the grasp localization error.
- Double-arm motion is a learned property of the pure UMI action data. It is a
  separate outcome from whether the raw head image helps or harms grasp
  localization.
- R0 alone does not establish causality. The harmful-head conclusion requires
  masked-head and/or wrist-only trials with the same scene, prompt, lower stack,
  checkpoint step, HOME convention, and scoring rule.
- Do not add a TCP compensation or freeze the nominally inactive arm in R0
  before completing the comparison.

## Fields required for each next run

- Group and exact checkpoint path
- Git/code snapshot or a note that it is unchanged from R0
- Task and object placement identifier
- Trial count, successful picks, successful placements
- Grasp offset direction and distance for every failed pick
- Whether the error is repeatable after resetting the same placement
- Whether one or both arms moved unexpectedly
- Gripper behavior
- Start/end time and diagnostic run-directory path
- Any operator intervention, collision, guard trip, or manual gripper command
