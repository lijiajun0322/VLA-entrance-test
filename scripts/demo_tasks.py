"""可视化演示三个任务 + 随机化效果。

macOS 上必须用 mjpython 运行（mujoco.viewer 需要）:

    mjpython scripts/demo_tasks.py            # 依次展示三个任务
    mjpython scripts/demo_tasks.py 1          # 只看任务 1
    mjpython scripts/demo_tasks.py 2 --n 8    # 任务 2 换 8 个随机变体

每 4 秒自动换一个随机 variant，终端同步打印该回合的指令文本。
按 Ctrl+C 或关窗口退出。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from envs import ALL_TASKS, G1TaskEnv

VARIANT_SECONDS = 4.0


def main():
    which = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n_variants = 5
    if "--n" in sys.argv:
        n_variants = int(sys.argv[sys.argv.index("--n") + 1])

    tasks = ALL_TASKS if which is None else [ALL_TASKS[which - 1]]
    for task in tasks:
        env = G1TaskEnv(task)
        print(f"\n=== {task.name} ===")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for variant in range(n_variants):
                env.reset()
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                print(f"  variant {variant + 1}: {env.instruction}")
                t0 = time.time()
                while viewer.is_running() and time.time() - t0 < VARIANT_SECONDS:
                    env.step()
                    viewer.sync()
                    time.sleep(env.model.opt.timestep)
                if not viewer.is_running():
                    return
        print(f"  ({task.name} 演示结束)")


if __name__ == "__main__":
    main()
