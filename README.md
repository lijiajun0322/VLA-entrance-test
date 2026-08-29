# VLA G1 MuJoCo

A LIBERO-style manipulation playground for a Unitree G1 humanoid in MuJoCo:
scripted-expert task environments, model-free control, a demonstration data
pipeline, and (planned) VLA policy inference.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # macOS
```

- Scripts without a viewer window run with plain `python`.
- Scripts using `mujoco.viewer` (e.g. `demo_tasks.py`) must run under
  `mjpython` on macOS.

## Workflow

```
scripts/run_expert.py    evaluate the scripted expert (control quality)
scripts/collect_data.py  collect demonstration episodes (task 3)
scripts/check_data.py    inspect + replay-verify the dataset
scripts/demo_tasks.py    interactive task viewer (mjpython)
```

### 1. Evaluate the expert

The first positional argument is the task id: `1` = pick & place,
`2` = pick by type, `3` = (placeholder).

```bash
python scripts/run_expert.py 1 --seeds 30            # headless, success rate
mjpython scripts/run_expert.py 1 --visual            # watch it execute
```

### 2. Collect demonstration data

First positional argument is the task id (`1`/`2`/`3`), same as above.

```bash
python scripts/collect_data.py 1 --episodes 10 --version v1
python scripts/collect_data.py 2 --episodes 50 --version v0
python scripts/collect_data.py 2 --seeds 12,18,21 --version v0   # explicit seeds
```

Flags:

| flag | default | meaning |
|---|---|---|
| `--episodes N` | 50 | collect until N **successful** episodes (failures retry with new seeds) |
| `--version V` | v0 | dataset dir `data/<task>/V/`; if it exists, **appends** with continued episode ids (never overwrites) |
| `--seed0 N` | 0 | first seed, incremented per try |
| `--seeds a,b,c` | — | explicit seed list (overrides `--seed0`) |

Output per dataset version:

```
data/task1_pick_place/v1/
├── episodes/episode_0000.npz   # per-frame obs + action labels (training data)
├── videos/episode_0000.mp4     # expert execution video (dual cam + phase caption)
├── failures/                   # failed episodes (npz + video pairs); only created if any
├── manifest.jsonl              # one JSON line per episode
└── norm_stats.json             # proprio/action normalization statistics
```

Notes:

- Bump the version number whenever the collection config changes; appending to
  an existing version is meant for same-config batch extension only
  (`norm_stats.json` covers the last run's episodes).
- Instructions are color-free for task1 (`pick up the cube and place it on the
  pad`) and type-only for task2 objects (`pick up the cylinder and place it on
  the blue pad`); colors are still randomized visually.

### 3. Check the dataset

`--task` takes the task **name** (`task1_pick_place` / `task2_pick_by_type`),
not the numeric id.

```bash
python scripts/check_data.py --task task1_pick_place --version v1
```

One command, full batch:

- summary stats + per-episode replay lines
  `"instruction" | episode_XXXX | success | inspect/replay_XXXX.mp4`
- `inspect/montage_episode_XXXX.png` — keyframe strip per episode
  (top: overhead cam, bottom: wrist cam, time left to right)
- `inspect/trajectory.png` — ee trajectories + gripper opening
- `inspect/replay_episode_XXXX.mp4` — re-execution from the recorded action
  labels (deployment path preview for task 4; compare against `videos/` to see
  open-loop following error)

Flags: `--replay all` (default; `0` skips, `N` replays first N) and
`--no-video` to skip replay videos.

### 4. Watch tasks interactively

```bash
mjpython scripts/demo_tasks.py 2          # cycle random variants of task 2
```

## Current dataset versions

| dataset | version | notes |
|---|---|---|
| task1_pick_place | v1 | 10 episodes, color-free instructions |
| task2_pick_by_type | v0 | 10 episodes, type-only object instructions |

## Project layout

```
envs/      task environments (robot, scene, tasks/) — physics & observations
control/   ik.py / servo.py / expert.py / executor.py — how to move
datasets/  spec.py / recorder.py — data contract & recording
scripts/   thin CLI wrappers (collect / check / evaluate / demo)
data/      datasets, videos, previews (gitignored)
```

Expert status: task1 30/30, task2 ~200/200 success, all phases < 1s,
no visible stalls.
