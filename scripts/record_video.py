"""把专家执行 / 数据回放的完整过程录成视频（双相机并排 + 字幕）。

    python scripts/record_video.py 1 --mode expert --seed 3                    # 专家执行
    python scripts/record_video.py 1 --mode replay --dataset v2 --episode 0   # 数据回放
    python scripts/record_video.py 1 --mode expert --cam overhead             # 只录第三人称

输出 MP4 到 data/<task>_videos/（20fps，与数据采样同频）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from envs import ALL_TASKS, G1TaskEnv
from control.expert import PickPlaceExpert
from control.executor import execute_actions
from datasets.spec import ACTION_SPEC

ROOT = Path(__file__).resolve().parent.parent


def compose(renderer, env, cam_ids, cams, caption) -> np.ndarray:
    """渲染一帧：各相机横排拼接，左起第一路顶部叠加字幕。"""
    imgs = []
    for i, cam in enumerate(cams):
        renderer.update_scene(env.data, camera=cam_ids[cam])
        rgb = renderer.render()
        if i == 0 and caption:
            img = Image.fromarray(rgb)
            strip = Image.new("RGB", (img.width, 22), (0, 0, 0))
            ImageDraw.Draw(strip).text((6, 5), caption[:70], fill=(255, 255, 255))
            img.paste(strip, (0, 0))
            rgb = np.array(img)
        imgs.append(rgb)
    return np.concatenate(imgs, axis=1)


def record_expert(env, task, seed, cams, out):
    expert = PickPlaceExpert(env, task.expert_spec())
    renderer = mujoco.Renderer(env.model, 224, 224)
    cam_ids = {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, n)
               for n in ("overhead_cam", "wrist_cam")}
    every = ACTION_SPEC.sample_every
    with imageio.get_writer(out, fps=ACTION_SPEC.control_hz, macro_block_size=1) as w:
        i = 0
        while not expert.done:
            expert.step()
            if i % every == 0:
                w.append_data(compose(renderer, env, cam_ids, cams,
                                      f"[expert] {expert._phase()}"))
            i += 1
        w.append_data(compose(renderer, env, cam_ids, cams,
                              f"[expert] done success={env.is_success()}"))
    print(f"已保存: {out}")


def record_replay(env, task_name, dataset, episode, cams, out):
    man_path = ROOT / "data" / task_name / dataset / "manifest.json"
    manifest = json.loads(man_path.read_text())
    e = manifest[episode]
    d = np.load(man_path.parent / e["episode"], allow_pickle=True)
    env.reset(int(d["seed"]))
    renderer = mujoco.Renderer(env.model, 224, 224)
    cam_ids = {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, n)
               for n in ("overhead_cam", "wrist_cam")}
    caption = f"[replay] {e['instruction']}"
    with imageio.get_writer(out, fps=ACTION_SPEC.control_hz, macro_block_size=1) as w:
        def on_frame(t):
            w.append_data(compose(renderer, env, cam_ids, cams,
                                  f"{caption} | frame {t}"))
        ok = execute_actions(env, d["action"], on_frame=on_frame)
        w.append_data(compose(renderer, env, cam_ids, cams,
                              f"{caption} | done success={ok}"))
    print(f"已保存: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int, choices=(1, 2, 3))
    ap.add_argument("--mode", choices=("expert", "replay"), default="expert")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--episode", type=int, default=0, help="replay 模式的回合号")
    ap.add_argument("--cam", choices=("both", "overhead", "wrist"), default="both")
    args = ap.parse_args()

    cams = {"both": ("overhead_cam", "wrist_cam"),
            "overhead": ("overhead_cam",), "wrist": ("wrist_cam",)}[args.cam]
    task = ALL_TASKS[args.task - 1]
    env = G1TaskEnv(task)
    out_dir = ROOT / "data" / f"{task.name}_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "expert":
        env.reset(args.seed)
        record_expert(env, task, args.seed, cams,
                      str(out_dir / f"{task.name}_expert_seed{args.seed}.mp4"))
    else:
        record_replay(env, task.name, args.dataset, args.episode, cams,
                      str(out_dir / f"{task.name}_{args.dataset}_replay_ep{args.episode}.mp4"))


if __name__ == "__main__":
    main()
