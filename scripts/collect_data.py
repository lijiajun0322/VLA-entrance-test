"""数据采集：脚本专家直接写 LeRobot v3 数据集。

    python scripts/collect_data.py 1 --episodes 5              # 采 5 个成功回合
    python scripts/collect_data.py 2 --seeds 12,18,21          # 显式指定 seed

存储到 data/<task_name>/<version>/：官方 LeRobot v3 Parquet + MP4 + metadata。
目录必须为空；追加采集请使用新的 --version。失败回合不会写入数据集。
采样规格：20Hz，action = world-frame delta xyz (m) + gripper_close。
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import ALL_TASKS, G1TaskEnv
from control.expert import make_expert
from mujoco_datasets.recorder import EpisodeRecorder


def collect(task_id: int, episodes: int, version: str, seed0: int, seeds=None,
            repo_id: str | None = None, grasp_wait_ticks: int = 0,
            release_wait_ticks: int = 0):
    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    out_dir = Path(__file__).resolve().parent.parent / "data" / task.name / version
    rec = EpisodeRecorder(out_dir, task.name, repo_id=repo_id)
    # 显式 seed 列表优先（可手工挑回合构成，如垫子距离分层）；否则从 seed0 顺延
    pending = list(seeds) if seeds else itertools.count(seed0)

    t0, seed, wins, tries = time.time(), None, 0, 0
    for seed in pending:
        if wins >= episodes:
            break
        env.reset(seed)
        rec.start(env, seed=seed)
        expert = make_expert(env, task, grasp_wait_ticks, release_wait_ticks)
        while not expert.done:
            # Causal alignment: store (observation_t, action_t), then execute.
            phase = expert._phase()
            action = expert.act()
            rec.add_frame(env, action.as_array(), phase=phase)
            expert.executor.step_delta(action)
            expert.update()
        ok = env.is_success()
        n_frames = rec.end(ok)
        wins += ok
        tries += 1
        print(f"[{tries:3d}] seed={seed} {'success' if ok else 'FAIL'} "
              f"frames={n_frames:3d} "
              f"done {wins}/{episodes}  {env.instruction}")
    rec.finalize()

    dt = time.time() - t0
    print(f"\ndataset: {out_dir}")
    print(f"episodes {wins} success ({tries} tries), {dt / 60:.1f} min total, "
          f"{dt / tries:.1f} s/episode")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=(1, 2, 3))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--version", default="v0")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的显式 seed 列表（优先于 --seed0 顺延）")
    ap.add_argument("--repo-id", default=None,
                    help="LeRobot repo id metadata（默认 local/<task_name>）")
    ap.add_argument("--grasp-wait-ticks", type=int, default=0)
    ap.add_argument("--release-wait-ticks", type=int, default=0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    collect(args.task, args.episodes, args.version, args.seed0, seeds,
            args.repo_id, args.grasp_wait_ticks, args.release_wait_ticks)
