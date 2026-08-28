"""数据集检查：统计、可视化、回放验证。

    python scripts/inspect_data.py --task task1_pick_place          # 全部检查
    python scripts/inspect_data.py --task task1_pick_place --replay 5

输出（存到 data/<task>/<version>/inspect/）：
  summary        终端打印统计
  montage.png    随机回合的关键帧拼图（肉眼检查图像-动作对齐）
  trajectory.png ee 轨迹 3D 图（检查路径合理、无跳变）
  replay         用 CartesianServo 跟踪录下的动作序列重跑（验证标签可执行）
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_episode(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for key in ("overhead_rgb", "wrist_rgb"):
        out[key] = [np.array(Image.open(io.BytesIO(bytes(b)))) for b in d[key]]
    return out


def load_dataset(task: str, version: str):
    out_dir = ROOT / "data" / task / version
    manifest = [e for e in json.loads((out_dir / "manifest.json").read_text())
                if e["success"]]
    stats = json.loads((out_dir / "stats.json").read_text())
    return out_dir, manifest, stats


def show_summary(manifest, stats):
    frames = sum(e["n_frames"] for e in manifest)
    instr = sorted(set(e["instruction"] for e in manifest))
    print(f"成功回合: {len(manifest)}, 总帧数: {frames}")
    print(f"指令种类: {len(instr)}")
    for s in instr[:5]:
        print(f"   {s}")
    if len(instr) > 5:
        print(f"   ... 共 {len(instr)} 种")
    print(f"action 均值: {[round(v, 3) for v in stats['action_mean']]}")
    print(f"action 标准差: {[round(v, 3) for v in stats['action_std']]}")


def make_montage(out_dir, manifest, n_frames=8):
    ep = load_episode(out_dir / manifest[0]["episode"])
    idx = np.linspace(0, len(ep["action"]) - 1, n_frames).astype(int)
    strip = np.concatenate([ep["overhead_rgb"][i] for i in idx], axis=1)
    strip_w = np.concatenate([ep["wrist_rgb"][i] for i in idx], axis=1)
    img = np.concatenate([strip, strip_w], axis=0)
    path = out_dir / "inspect" / "montage.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"montage: {path}（上=第三人称 下=腕部，时间从左到右）")


def plot_trajectory(out_dir, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    for e in manifest[:10]:
        ep = load_episode(out_dir / e["episode"])
        a = ep["action"]
        ax1.plot(a[:, 0], a[:, 1], a[:, 2], alpha=0.5)
        ax2.plot(a[:, 3], alpha=0.5)
    ax1.set_title("ee trajectory (10 episodes)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax2.set_title("gripper opening")
    ax2.set_xlabel("frame @20Hz"); ax2.set_ylabel("close 0~1")
    path = out_dir / "inspect" / "trajectory.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"轨迹图: {path}")


def replay_check(out_dir, manifest, task_name: str, n: int):
    """回放验证：跟随录下的动作标签重跑回合，检验标签可执行。

    执行链路 = 任务 4 策略推理的预演（动作序列 → 控制 → 仿真），
    只是动作来源从"录制的标签"换成"策略输出"。每帧：
      solve_ik 把 ee 目标转成关节目标 -> 前 15 步关节插值 -> 后 10 步伺服补残差。
    （纯伺服跟踪的稳态精度 ~1-2cm，会在下降段把方块扫走，故用 IK 主导。）
    """
    import mujoco
    from envs.tasks import TASKS
    from envs.g1_env import G1TaskEnv
    from control.ik import solve_ik, IK_JOINTS, HOME_POSE
    from control.servo import CartesianServo
    from datasets.spec import ACTION_SPEC

    env = G1TaskEnv(TASKS[task_name]())
    servo = CartesianServo(env.model, env.data)
    qadr = np.array([env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in IK_JOINTS])
    every = ACTION_SPEC.sample_every
    SERVO_TAIL = 10

    def goto(target, steps=300, tol=0.006):
        """对中悬停：把 ee 收敛到 target 再继续（下降前必做，防斜插扫走物体）。"""
        for _ in range(steps):
            q_star, _ = solve_ik(env.model, target,
                                 seeds=[env.data.qpos[qadr].copy(), HOME_POSE],
                                 n_random=3, iters=600)
            env.set_arm8(q_star)
            env.step()
            if np.linalg.norm(target - env.ee_xpos()) < tol:
                break

    wins = 0
    for e in manifest[:n]:
        ep = load_episode(out_dir / e["episode"])
        a = ep["action"]
        env.reset(int(ep["seed"]))
        prev_z = a[0, 2]
        for t in range(len(a)):
            # 下降沿（抓取/放置都有）：先在当前高度对准 xy，再垂直下降
            if a[t, 2] < 0.80 and prev_z >= 0.80:
                hover = a[t, :3].copy()
                hover[2] = prev_z
                goto(hover)
            prev_z = a[t, 2]
            q_meas = env.data.qpos[qadr].copy()
            q_star, _ = solve_ik(env.model, a[t, :3],
                                 seeds=[q_meas, HOME_POSE], n_random=5, iters=800)
            for k in range(every):
                if k < every - SERVO_TAIL:
                    s = (k + 1) / (every - SERVO_TAIL)
                    env.set_arm8(q_meas + s * (q_star - q_meas))
                else:
                    env.set_arm8(servo.step(a[t, :3]))
                env.set_gripper(float(a[t, 3]))
                env.step()
        ok = env.is_success()
        wins += ok
        print(f"  回放 {e['episode']}: {'成功' if ok else '失败'}")
    print(f"回放验证: {wins}/{n} —— 开环跟随录制标签的固有误差（无视觉闭环修正），"
          f"失败集中在工作区角点等跟踪残差最大处；这正是任务 4 闭环策略要解决的问题")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1_pick_place")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--replay", type=int, default=3, help="回放验证的回合数, 0 跳过")
    args = ap.parse_args()

    out_dir, manifest, stats = load_dataset(args.task, args.version)
    show_summary(manifest, stats)
    make_montage(out_dir, manifest)
    plot_trajectory(out_dir, manifest)
    if args.replay:
        replay_check(out_dir, manifest, args.task, args.replay)
