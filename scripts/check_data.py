"""数据集检查 + 回放验证（原 inspect_data / record_video 合并）。

    python scripts/check_data.py --task task1_pick_place --version v1   # 全套检查+全量回放+回放视频
    python scripts/check_data.py --task task1_pick_place --replay 5     # 只回放前 5 个
    python scripts/check_data.py --task task1_pick_place --replay 0 --no-video  # 只看统计/图

检查内容：
  summary        终端打印统计（回合/帧数/指令/action 分布）
  montage.png    随机回合的关键帧拼图（肉眼检查图像-动作对齐）
  trajectory.png ee 轨迹 3D 图（检查路径合理、无跳变）
  replay         用 control/executor 跟随录下的动作序列重跑（任务 4 部署
                 链路的预演：动作 -> IK -> 伺服 -> 仿真），验证标签可执行；
                 --video 同时把回放过程录成 MP4，与 videos/ 里的专家执行
                 视频对照，两者差异 = 开环跟随误差的可视化

输出存到 data/<task>/<version>/inspect/。
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
    for key in (k for k in d.files if k.endswith("_rgb")):
        out[key] = [np.array(Image.open(io.BytesIO(bytes(b)))) for b in d[key]]
    return out


def load_dataset(task: str, version: str):
    out_dir = ROOT / "data" / task / version
    manifest = [json.loads(line) for line in
                (out_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    manifest = [e for e in manifest if e["success"]]
    stats = json.loads((out_dir / "norm_stats.json").read_text())
    return out_dir, manifest, stats


def show_summary(manifest, stats):
    frames = sum(e["n_frames"] for e in manifest)
    print(f"episodes: {len(manifest)}, frames: {frames}")
    print(f"action mean: {[round(v, 3) for v in stats['action_mean']]}")
    print(f"action std:  {[round(v, 3) for v in stats['action_std']]}")


def make_montages(out_dir, manifest, n_frames=8):
    """每个 episode 一张关键帧拼图：inspect/montage_episode_XXXX.png"""
    (out_dir / "inspect").mkdir(exist_ok=True)
    for e in manifest:
        ep = load_episode(out_dir / e["episode"])
        idx = np.linspace(0, len(ep["action"]) - 1, n_frames).astype(int)
        image_keys = [k for k in ep if k.endswith("_rgb")]
        rows = [np.concatenate([ep[key][i] for i in idx], axis=1)
                for key in image_keys]
        img = np.concatenate(rows, axis=0)
        stem = Path(e["episode"]).stem
        Image.fromarray(img).save(out_dir / "inspect" / f"montage_{stem}.png")
    print(f"montage: inspect/montage_episode_*.png x{len(manifest)} "
          f"(one row per camera, time left to right)")


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
    print(f"trajectory: {path}")


def replay_check(out_dir, manifest, task_name: str, n: int, video: bool):
    """回放验证：跟随录下的动作标签重跑回合，检验标签可执行。

    执行链路 = 任务 4 策略推理的预演（动作序列 -> 控制 -> 仿真），
    只是动作来源从"录制的标签"换成"策略输出"。引擎在 control/executor.py。
    """
    import mujoco
    import imageio.v2 as imageio
    from envs.tasks import TASKS
    from envs.g1_env import G1TaskEnv
    from control.executor import execute_actions
    from mujoco_datasets.spec import ACTION_SPEC

    env = G1TaskEnv(TASKS[task_name]())
    renderer = None
    cam_ids = {name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
               for name in env.obs_camera_map.values()} if video else None
    if video:
        renderer = mujoco.Renderer(env.model, 224, 224)
        (out_dir / "inspect").mkdir(exist_ok=True)

    def compose(instruction, t) -> np.ndarray:
        from PIL import ImageDraw
        frames = []
        for cam in env.obs_camera_map.values():
            renderer.update_scene(env.data, camera=cam_ids[cam])
            frames.append(renderer.render())
        img = Image.fromarray(frames[0])
        strip = Image.new("RGB", (img.width, 22), (0, 0, 0))
        ImageDraw.Draw(strip).text((6, 5), f"[replay] {instruction[:60]} | {t}",
                                   fill=(255, 255, 255))
        img.paste(strip, (0, 0))
        frames[0] = np.array(img)
        return np.concatenate(frames, axis=1)

    wins = 0
    for e in manifest[:n]:
        ep = load_episode(out_dir / e["episode"])
        env.reset(int(ep["seed"]))
        writer = None
        if video:
            stem = Path(e["episode"]).stem
            writer = imageio.get_writer(out_dir / "inspect" / f"replay_{stem}.mp4",
                                        fps=ACTION_SPEC.control_hz,
                                        macro_block_size=1)

        def on_frame(t, writer=writer, e=e):
            if writer is not None:
                writer.append_data(compose(e["instruction"], t))

        ok = execute_actions(env, ep["action"],
                             on_frame=on_frame if video else None)
        if writer is not None:
            writer.close()
        wins += ok
        stem = Path(e["episode"]).stem
        video_rel = f"inspect/replay_{stem}.mp4" if video else "-"
        print(f'  "{e["instruction"]}" | {stem} | '
              f'{"success" if ok else "FAIL"} | {video_rel}')
    print(f"replay verification: {wins}/{n} passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1_pick_place")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--replay", default="all",
                    help="回放验证范围：默认整批全部回放，0 跳过，数字回放前 N 个")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="不录回放视频（默认录，存 inspect/replay_*.mp4）")
    args = ap.parse_args()

    out_dir, manifest, stats = load_dataset(args.task, args.version)
    show_summary(manifest, stats)
    make_montages(out_dir, manifest)
    plot_trajectory(out_dir, manifest)
    if args.replay:
        n = len(manifest) if args.replay == "all" else int(args.replay)
        replay_check(out_dir, manifest, args.task, n, args.video)
