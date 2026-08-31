# VLA G1 MuJoCo

A LIBERO-style manipulation playground for a Unitree G1 humanoid in MuJoCo:
scripted-expert task environments, model-free control, a demonstration data
pipeline, and zero-shot OpenVLA policy inference.

## Quick reference

```bash
# evaluate the scripted expert (1 = pick&place, 2 = pick by type, 3 = sliding door)
python scripts/run_expert.py 1 --seeds 30            # headless success rate
mjpython scripts/run_expert.py 1 --visual --seed 0   # watch one selected seed

# collect demonstration episodes
python scripts/collect_data.py 1 --episodes 10 --version lerobot_v0

# optional lightweight NPZ backend (same causal delta actions)
python scripts/collect_data_npz.py 1 --episodes 10 --version npz_v0

# check the dataset: stats + montages + trajectory + full-batch replay
python scripts/check_data.py --dataset task1_pick_place --version lerobot_v0

# interactive task viewer
mjpython scripts/demo_tasks.py 2
mjpython scripts/demo_tasks.py 3                 # sliding cabinet door

# OpenVLA: short closed-loop smoke rollout
python scripts/eval_openvla.py --task 1 --seeds 100 --max-actions 120

# OpenVLA: held-out evaluation with results in a named run directory
python scripts/eval_openvla.py --task 1 --seeds 0-4 --max-actions 120 \
  --run-name openvla_seeds_0_4

# validate, then LoRA-fine-tune OpenVLA on the Task 1 LeRobot dataset
python scripts/train_openvla_lora.py --task 1 --dry-run

# Equivalent explicit dataset selection (--dataset-path overrides --version)
python scripts/train_openvla_lora.py --task 1 \
  --dataset-path data/task1_pick_place/lerobot_v0 --dry-run
python scripts/train_openvla_lora.py --task 1 --epochs 5 \
  --batch-size 1 --grad-accumulation 8 --image-aug

# evaluate the merged fine-tuned checkpoint on held-out seeds
python scripts/eval_openvla.py --task 1 --seeds 100-109 \
  --model outputs/openvla_finetune/task1_pick_place/lora_v0 \
  --unnorm-key g1_task1 --action-units meters \
  --run-name task1_lora_seeds_100_109
```

Task argument conventions: `run_expert` / `collect_data` take a numeric id
(`1`/`2`/`3`); `check_data` reads a dataset folder — `--dataset` is the
first-level dir name under `data/` (i.e. the task's registered name,
`task1_pick_place` / `task2_pick_by_type`), `--version` the second level:
`data/<task>/<version>/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # macOS
pip install lerobot==0.4.4
pip install peft==0.11.1                 # OpenVLA LoRA fine-tuning
```

- Scripts without a viewer window run with plain `python`.
- Scripts using `mujoco.viewer` (e.g. `demo_tasks.py`) must run under
  `mjpython` on macOS.

## Workflow

```
scripts/run_expert.py    evaluate the scripted expert (control quality)
scripts/collect_data.py  collect demonstration episodes (task 3)
scripts/check_data.py    inspect + replay-verify the dataset
scripts/collect_data_npz.py / check_data_npz.py  lightweight NumPy workflow
scripts/demo_tasks.py    interactive task viewer (mjpython)
scripts/check_openvla.py inspect one OpenVLA prediction without executing it
scripts/eval_openvla.py  run and score zero-shot OpenVLA rollouts
scripts/train_openvla_lora.py  LoRA fine-tune OpenVLA from LeRobot v3 data
```

Task difficulty increases from single-object pick-and-place (Task 1), through
language-conditioned object selection (Task 2), to opening a constrained
articulated sliding cabinet door through sustained handle contact (Task 3).
Task 3 randomizes the complete window/frame assembly by up to 2.5 cm in x and
2.0 cm in y, in addition to its initial gap, colors, and language variant.

### 1. Evaluate the expert

```bash
python scripts/run_expert.py 1 --seeds 30            # headless, success rate
mjpython scripts/run_expert.py 1 --visual --seed 0   # watch one selected seed
```

All three tasks are selected through `make_expert(env, task)` in
`control/expert.py`. The pick/place and sliding-door state machines emit a
common 20 Hz action, `[delta_x, delta_y, delta_z, gripper_close]`, in world-frame
meters; `OnlineActionExecutor` integrates the translation command and tracks the
resulting absolute target with the existing IK/Cartesian servo. End-effector
orientation remains the fixed downward grasp orientation at this stage.

### 2. Collect demonstration data

```bash
python scripts/collect_data.py 1 --episodes 10 --version lerobot_v0
python scripts/collect_data.py 2 --episodes 10 --version lerobot_v0
python scripts/collect_data.py 2 --seeds 12,18,21 --version lerobot_seeded
```

Flags:

| flag | default | meaning |
|---|---|---|
| `--episodes N` | 50 | collect until N **successful** episodes (failures retry with new seeds) |
| `--version V` | v0 | dataset dir `data/<task>/V/`; must be a new/empty directory |
| `--seed0 N` | 0 | first seed, incremented per try |
| `--seeds a,b,c` | — | explicit seed list (overrides `--seed0`) |
| `--repo-id ID` | `local/<task>` | LeRobot repository id stored in metadata |

Output per dataset version:

```
data/task1_pick_place/lerobot_v0/
├── data/chunk-000/file-000.parquet
├── videos/
│   ├── observation.images.overhead/chunk-000/file-000.mp4
│   └── observation.images.wrist/chunk-000/file-000.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    └── episodes/chunk-000/file-000.parquet
```

Notes:

- Collection writes the official LeRobot v3 format directly. Each row stores
  `observation_t` together with the command executed at that instant:
  `[delta_x_m, delta_y_m, delta_z_m, gripper_close]`. Failed attempts are
  discarded. Use a new version name for every collection run.
- Instructions are color-free for task1 (`pick up the cube and place it on the
  pad`) and type-only for task2 objects (`pick up the cylinder and place it on
  the blue pad`); colors are still randomized visually.

For quick NumPy debugging, the alternative backend stores one NPZ and one MP4
per episode while preserving the exact same action and temporal semantics:

```bash
python scripts/collect_data_npz.py 1 --episodes 10 --version npz_v0
python scripts/check_data_npz.py --dataset task1_pick_place --version npz_v0
```

### 3. Check the dataset

`--dataset`/`--version` point at the dataset folder `data/<dataset>/<version>/`
(task name = first-level dir under `data/`).

```bash
python scripts/check_data.py --dataset task1_pick_place --version lerobot_v0
```

One command, full batch:

- summary stats + per-episode replay lines
  `"instruction" | episode_XXXX | success | inspect/replay_XXXX.mp4`
- `inspect/montage_episode_XXXX.png` — keyframe strip per episode
  (one row per recorded camera, time left to right). Task 1/2 record overhead
  and wrist cameras; Task 3 records only the shoulder-view `door_cam`.
- `inspect/trajectory.png` — cumulative commanded EE displacement + gripper command
- `inspect/replay_episode_XXXX.mp4` — re-execution from recorded delta actions

Flags: `--replay all` (default; `0` skips, `N` replays first N) and
`--no-video` to skip replay videos.

### 4. Watch tasks interactively

```bash
mjpython scripts/demo_tasks.py 2          # cycle random variants of task 2
```

### 5. Evaluate OpenVLA zero-shot

OpenVLA uses the overhead RGB image and language instruction only. Its LIBERO
7D delta action is mapped to a bounded Cartesian target while the existing IK
controller keeps the gripper orientation vertically downward.

```bash
# One real MuJoCo frame -> one model action, without moving the robot
python scripts/check_openvla.py --seed 100

# Short closed-loop safety smoke test
python scripts/eval_openvla.py --task 1 --seeds 100 --max-actions 10

# Held-out evaluation
python scripts/eval_openvla.py --task 1 --seeds 100-119 --max-actions 120 \
  --run-name heldout_seeds_100_119
```

`--task 1`, `--task 2`, and `--task 3` select pick-and-place,
pick-by-type, and sliding-door evaluation respectively. The default is Task 1.

Each invocation writes its summary, episode CSV, per-step JSON logs and videos
under a separate `data/openvla_eval/<run_name>/` directory. If `--run-name` is
omitted, a name such as
`task1_20260830_142155_seeds_100-119_scale_0p02` is generated from the task,
timestamp,
seeds and action scale. Existing run directories are never overwritten; choose
a new `--run-name` when repeating an experiment.

```text
data/openvla_eval/
└── heldout_seeds_100_119/
    ├── summary.json       # aggregate success/lift/reach, IK, clipping, latency
    ├── episodes.csv       # one summary row per seed
    ├── seed_100.json      # raw/model-adapted actions and state per timestep
    ├── seed_100.mp4       # overhead + wrist rollout video
    ├── seed_101.json
    └── seed_101.mp4
```

The original LIBERO checkpoint does not output an absolute target in meters. It
outputs a normalized LIBERO controller command
`[dx, dy, dz, droll, dpitch, dyaw, gripper]`. With the default
`--action-units normalized`, the adapter applies:

```text
delta_xyz_m = 0.02 * [dx, dy, dz]
target_xyz  = current_ee_xyz + delta_xyz_m
```

Thus a translation component of `+1.0` requests `+2 cm` along that axis. The
length of the combined 3D step is capped at `2.5 cm`, and the resulting target
is clipped to the G1's tested reachable workspace. Model-predicted rotation is
ignored because the existing IK controller keeps the gripper vertically
downward throughout this task. Finally, OpenVLA uses `gripper=1` for open and
`gripper=0` for closed, while this environment uses `1=close`; the adapter
therefore inverts and binarizes the gripper command.

### 6. LoRA fine-tune OpenVLA on Task 1

The fine-tuning loader reads the causal LeRobot transitions directly. It maps
the stored action to `[dx_m, dy_m, dz_m, 0, 0, 0, gripper_open]`, normalizes it
to action tokens using the dataset q01/q99 values, and writes those statistics
into the checkpoint as `g1_task1`. Consequently `predict_action()` from the
fine-tuned model returns translation in meters, not a LIBERO controller unit.

```bash
# Fast validation: real first video frame + instruction + action-token labels
python scripts/train_openvla_lora.py --task 1 --dry-run

# LoRA training on MPS; output is a merged, directly loadable checkpoint
python scripts/train_openvla_lora.py --task 1 --epochs 5 \
  --batch-size 1 --grad-accumulation 8 --image-aug

# Held-out evaluation; meters is required for this fine-tuned checkpoint
python scripts/eval_openvla.py --task 1 --seeds 100-109 \
  --model outputs/openvla_finetune/task1_pick_place/lora_v0 \
  --unnorm-key g1_task1 --action-units meters \
  --run-name task1_lora_seeds_100_109
```

Training output defaults to
`outputs/openvla_finetune/task1_pick_place/lora_v0/`. Use `--adapter-only` only
when a small PEFT adapter is desired; the default merges LoRA into a checkpoint
that `eval_openvla.py` can load directly.

Per-optimizer-step loss, unclipped gradient norm, dynamic loss scale, and
learning rate are written to `train_metrics.csv`. The same metrics are printed
every five optimizer steps by default (`--log-every N` changes the interval).
Training aborts immediately on a non-finite loss/gradient, and all trainable
parameters are checked again before saving.

`--val-ratio 0.1` (the default) holds out complete episodes rather than random
frames; for 50 episodes this produces 45 train and 5 validation episodes.
Validation uses no image augmentation and writes `validation_metrics.csv`.
It runs at each epoch end by default; `--eval-every-steps N` additionally runs
it every N optimizer steps. Use `--val-ratio 0` to train on every episode and
disable validation.

`--save-every-steps N` saves intermediate PEFT adapters under
`<output>/checkpoints/step_XXXXXX/`; `--save-every-epoch` similarly saves
`epoch_XXX/`. Intermediate checkpoints are adapter-only to avoid repeatedly
writing a full 15 GB model (a rank-32 adapter is still roughly 670 MB). The
final output keeps the normal behavior: merged model by default, adapter-only
with `--adapter-only`.

## Current dataset versions

| dataset | version | notes |
|---|---|---|
| task1_pick_place | v2 | 50 episodes, 10D proprio, color-free instructions |
| task2_pick_by_type | v2 | 10 episodes, 10D proprio, type-only object instructions |

`v2` above is the legacy NPZ data. New collections should use a fresh name such
as `lerobot_v0`; they are written directly as LeRobot v3 and are not mixed with
the legacy datasets.

Observation proprioception is 10D: waist yaw, seven right-arm joints, and two
finger joints. Dataset versions recorded before v2 used a legacy 9D state that
omitted waist yaw.

## Project layout

```
envs/      task environments (robot, scene, tasks/) — physics & observations
control/   IK / servo / executor / OpenVLA action adapter — how to move
mujoco_datasets/  spec.py / recorder.py — data contract & recording
policies/  learned-policy wrappers (OpenVLA)
scripts/   thin CLI wrappers (collect / check / evaluate / demo)
data/      datasets, videos, previews (gitignored)
```

Delta-expert validation (30 randomized seeds): task1 30/30, task2 30/30,
task3 30/30. The deterministic smoke suite also checks that `act()` is
side-effect-free and each translation command respects the 15 mm step limit.
