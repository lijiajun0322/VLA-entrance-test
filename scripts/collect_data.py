"""数据采集：脚本专家跑回合，逐帧录制 (观测, 指令, 动作) 数据集。

    python scripts/collect_data.py 1 --episodes 50            # 采 50 个成功回合
    python scripts/collect_data.py 1 --episodes 2 --dry       # 试采 2 回合验证管线

存储到 data/<task_name>/v0/（--version 可改），失败回合归档到 failures/。
采样规格见 datasets/spec.py：20Hz，动作 = ee_xyz + 夹爪开度（4 维）。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import ALL_TASKS, G1TaskEnv
from control.expert import PickPlaceExpert
from datasets.recorder import EpisodeRecorder
from datasets.spec import ACTION_SPEC


def collect(task_id: int, episodes: int, version: str, seed0: int):
    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    out_dir = Path(__file__).resolve().parent.parent / "data" / task.name / version
    rec = EpisodeRecorder(out_dir, task.name)
    every = ACTION_SPEC.sample_every          # 每 25 步采一帧 (20Hz)

    t0, seed, wins, tries = time.time(), seed0, 0, 0
    while wins < episodes:
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
        print(f"[{tries:3d}] seed={seed} {'成功' if ok else '失败'} "
              f"帧数={rec._manifest[-1]['n_frames']:3d} "
              f"累计成功 {wins}/{episodes}  {env.instruction}")
        seed += 1
    rec.write_stats()

    dt = time.time() - t0
    print(f"\n数据集: {out_dir}")
    print(f"成功回合 {wins}（尝试 {tries}），总时长 {dt / 60:.1f} min，"
          f"平均 {dt / tries:.1f} s/回合")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=(1, 2, 3))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--version", default="v0")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    collect(args.task, args.episodes, args.version, args.seed0)
