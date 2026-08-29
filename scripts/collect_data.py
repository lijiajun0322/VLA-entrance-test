"""数据采集：脚本专家跑回合，逐帧录制 (观测, 指令, 动作) 数据集。

    python scripts/collect_data.py 1 --episodes 5              # 采 5 个成功回合
    python scripts/collect_data.py 2 --seeds 12,18,21          # 显式指定 seed

存储到 data/<task_name>/<version>/（--version 可改，默认 v0；目录已存在则
接续编号追加，不覆盖）：
  episodes/ 每回合一个 npz、videos/ 同名 mp4、manifest.jsonl 索引、
  norm_stats.json 归一化统计；失败回合 npz+视频成对归档到 failures/
  （无失败不建该目录）。
采样规格见 datasets/spec.py：20Hz，动作 = ee_xyz + 夹爪开度（4 维）。
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import ALL_TASKS, G1TaskEnv
from control.expert import PickPlaceExpert
from datasets.recorder import EpisodeRecorder
from datasets.spec import ACTION_SPEC


def collect(task_id: int, episodes: int, version: str, seed0: int, seeds=None):
    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    out_dir = Path(__file__).resolve().parent.parent / "data" / task.name / version
    rec = EpisodeRecorder(out_dir, task.name)
    every = ACTION_SPEC.sample_every          # 每 25 步采一帧 (20Hz)

    # 显式 seed 列表优先（可手工挑回合构成，如垫子距离分层）；否则从 seed0 顺延
    pending = list(seeds) if seeds else itertools.count(seed0)

    t0, seed, wins, tries = time.time(), None, 0, 0
    for seed in pending:
        if wins >= episodes:
            break
        env.reset(seed)
        rec.start(env, seed=seed)
        expert = PickPlaceExpert(env, task.expert_spec())
        i = 0
        while not expert.done:
            expert.step()
            if i % every == 0:
                rec.sample(env, phase=expert._phase())
            i += 1
        rec.sample(env, phase="done")         # 末帧（成功判定的静止状态）
        ok = env.is_success()
        path = rec.end(ok)
        wins += ok
        tries += 1
        print(f"[{tries:3d}] seed={seed} {'success' if ok else 'FAIL'} "
              f"frames={rec._manifest[-1]['n_frames']:3d} "
              f"done {wins}/{episodes}  {env.instruction}")
    rec.write_stats()

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
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    collect(args.task, args.episodes, args.version, args.seed0, seeds)
