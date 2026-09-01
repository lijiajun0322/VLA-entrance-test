"""Collect lightweight NPZ+MP4 demonstrations with causal delta actions.

    python scripts/collect_data_npz.py 1 --episodes 10 --version npz_v0

For the training-oriented official LeRobot v3 backend, use collect_data.py.
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.expert import make_expert
from envs import ALL_TASKS, G1TaskEnv
from mujoco_datasets.npz_recorder import NpzEpisodeRecorder


def collect(task_id: int, episodes: int, version: str, seed0: int, seeds=None,
            grasp_wait_ticks: int = 0, release_wait_ticks: int = 0):
    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    out_dir = Path(__file__).resolve().parent.parent / "data" / task.name / version
    recorder = NpzEpisodeRecorder(out_dir, task.name)
    pending = list(seeds) if seeds else itertools.count(seed0)

    start_time, wins, tries = time.time(), 0, 0
    for seed in pending:
        if wins >= episodes:
            break
        env.reset(seed)
        recorder.start(env, seed)
        expert = make_expert(env, task, grasp_wait_ticks, release_wait_ticks)
        while not expert.done:
            phase = expert._phase()
            action = expert.act()
            recorder.add_frame(env, action.as_array(), phase)
            expert.executor.step_delta(action)
            expert.update()
        success = bool(env.is_success())
        path = recorder.end(success)
        wins += success
        tries += 1
        print(f"[{tries:3d}] seed={seed} {'success' if success else 'FAIL'} "
              f"frames={recorder.last_n_frames if path else 0:3d} "
              f"done {wins}/{episodes}  {env.instruction}")
    recorder.write_stats()

    elapsed = time.time() - start_time
    print(f"\nNPZ dataset: {out_dir}")
    print(f"episodes {wins} success ({tries} tries), {elapsed / 60:.1f} min total, "
          f"{elapsed / tries:.1f} s/episode")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int, choices=(1, 2, 3))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--version", default="npz_v0")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated explicit seeds (overrides --seed0)")
    parser.add_argument("--grasp-wait-ticks", type=int, default=0)
    parser.add_argument("--release-wait-ticks", type=int, default=0)
    args = parser.parse_args()
    explicit = [int(value) for value in args.seeds.split(",")] if args.seeds else None
    collect(args.task, args.episodes, args.version, args.seed0, explicit,
            args.grasp_wait_ticks, args.release_wait_ticks)
