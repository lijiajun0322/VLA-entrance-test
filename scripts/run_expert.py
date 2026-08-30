"""脚本专家运行器：headless 评测成功率 / 可视化单次演示。

    python scripts/run_expert.py 1 --seeds 20          # 任务1跑20个种子出成功率
    mjpython scripts/run_expert.py 1 --visual          # 可视化看一次
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import ALL_TASKS, G1TaskEnv
from envs.scene import configure_viewer_camera
from control.expert import make_expert


def run_headless(task_id: int, seeds: int):
    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    wins = 0
    for seed in range(seeds):
        env.reset(seed)
        expert = make_expert(env, task)
        expert.run()
        ok = env.is_success()
        wins += ok
        print(f"seed {seed:3d}: {'成功' if ok else '失败':4s} "
              f"(停于 {expert.PHASES[expert.phase_i]})  {env.instruction}")
    print(f"\n[{task.name}] 成功率: {wins}/{seeds} = {wins / seeds:.0%}")


def run_visual(task_id: int, seed: int = 0):
    import mujoco
    import mujoco.viewer

    task = ALL_TASKS[task_id - 1]
    env = G1TaskEnv(task)
    env.reset(seed)
    expert = make_expert(env, task)
    print(f"[{task.name}] {env.instruction}")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_viewer_camera(viewer, env.model, task.name)
        while viewer.is_running() and not expert.done:
            expert.step()
            viewer.sync()
            time.sleep(env.model.opt.timestep)
        # 多转 1 秒确认物体静止
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < 1.0:
            env.step()
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    print("成功!" if env.is_success() else "失败")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=(1, 2, 3))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--visual", action="store_true")
    args = ap.parse_args()
    if args.visual:
        run_visual(args.task)
    else:
        run_headless(args.task, args.seeds)
