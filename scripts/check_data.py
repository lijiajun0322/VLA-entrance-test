"""Inspect and replay-check a directly recorded LeRobot v3 dataset.

    python scripts/check_data.py --task task1_pick_place --version lerobot_v0
    python scripts/check_data.py --task task1_pick_place --version lerobot_v0 --replay 1

Outputs montages, delta/cumulative trajectories, and optional replay videos to
``data/<task>/<version>/inspect``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
)


def load_dataset(task: str, version: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = ROOT / "data" / task / version
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"not a LeRobot v3 dataset: {root}")
    repo_id = f"local/{task}"
    # torchcodec is installed as a LeRobot dependency but may not find shared
    # Homebrew FFmpeg dylibs on macOS; PyAV is already available and reliable.
    return root, LeRobotDataset(repo_id, root=root, video_backend="pyav")


def episode_indices(ds) -> list[np.ndarray]:
    ep = np.asarray(ds.hf_dataset["episode_index"], dtype=int)
    return [np.flatnonzero(ep == i) for i in range(ds.meta.total_episodes)]


def _numpy_image(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape[0] in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    return value[..., :3]


def show_summary(ds, groups):
    actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    timestamps = np.asarray(ds.hf_dataset["timestamp"], dtype=np.float32)
    delta_norm = np.linalg.norm(actions[:, :3], axis=1)
    bad_dt = sum(
        np.count_nonzero(np.abs(np.diff(timestamps[idx]) - 1 / ds.fps) > 1e-4)
        for idx in groups
    )
    print(f"format: LeRobot {ds.meta.info['codebase_version']}")
    print(f"episodes: {ds.meta.total_episodes}, frames: {ds.meta.total_frames}, fps: {ds.fps}")
    print(f"robot: {ds.meta.robot_type}, cameras: {ds.meta.camera_keys}")
    print(f"tasks: {list(ds.meta.tasks.index)}")
    print(f"action mean: {np.round(actions.mean(0), 4).tolist()}")
    print(f"action std:  {np.round(actions.std(0), 4).tolist()}")
    print(f"delta norm max: {delta_norm.max():.4f} m, timestamp errors: {bad_dt}")
    if delta_norm.max() > 0.0151:
        raise AssertionError("translation command exceeds 1.5 cm action limit")
    if bad_dt:
        raise AssertionError("timestamps are not uniformly aligned at dataset fps")


def make_montages(root: Path, ds, groups, n_frames=8):
    inspect_dir = root / "inspect"
    inspect_dir.mkdir(exist_ok=True)
    for ep_i, idx in enumerate(groups):
        selected = idx[np.linspace(0, len(idx) - 1, n_frames).astype(int)]
        rows = []
        for key in ds.meta.camera_keys:
            rows.append(np.concatenate([_numpy_image(ds[int(i)][key]) for i in selected], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(
            inspect_dir / f"montage_episode_{ep_i:06d}.png"
        )
    print(f"montages: inspect/montage_episode_*.png x{len(groups)}")


def plot_trajectory(root: Path, ds, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    for idx in groups[:10]:
        a = actions[idx]
        xyz = np.cumsum(a[:, :3], axis=0)
        ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.6)
        ax2.plot(a[:, 3], alpha=0.6)
    ax1.set_title("cumulative commanded EE displacement")
    ax1.set_xlabel("Δx sum (m)"); ax1.set_ylabel("Δy sum (m)"); ax1.set_zlabel("Δz sum (m)")
    ax2.set_title("gripper command")
    ax2.set_xlabel("frame @20Hz"); ax2.set_ylabel("close 0..1")
    path = root / "inspect" / "trajectory.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"trajectory: {path}")


def replay_check(root: Path, ds, groups, task_name: str, n: int, video: bool):
    import imageio.v2 as imageio
    import mujoco
    from control.actions import DeltaEEAction
    from control.executor import OnlineActionExecutor
    from envs.g1_env import G1TaskEnv
    from envs.tasks import TASKS

    env = G1TaskEnv(TASKS[task_name]())
    renderer = mujoco.Renderer(env.model, 224, 224) if video else None
    actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    seeds = np.asarray(ds.hf_dataset["seed"], dtype=np.int64).reshape(-1)
    inspect_dir = root / "inspect"
    inspect_dir.mkdir(exist_ok=True)
    wins = 0
    for ep_i, idx in enumerate(groups[:n]):
        env.reset(int(seeds[idx[0]]))
        executor = OnlineActionExecutor(env)
        writer = None
        if video:
            writer = imageio.get_writer(inspect_dir / f"replay_episode_{ep_i:06d}.mp4",
                                         fps=ds.fps, macro_block_size=1)
        for a in actions[idx]:
            executor.step_delta(DeltaEEAction(a[:3], float(a[3])))
            if writer is not None:
                frames = []
                for cam in env.obs_camera_map.values():
                    renderer.update_scene(env.data, camera=cam)
                    frames.append(renderer.render().copy())
                writer.append_data(np.concatenate(frames, axis=1))
        if writer is not None:
            writer.close()
        ok = bool(env.is_success())
        wins += ok
        print(f"  episode {ep_i:06d}, seed={seeds[idx[0]]}: {'success' if ok else 'FAIL'}")
    print(f"delta-action replay: {wins}/{min(n, len(groups))} passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1_pick_place")
    ap.add_argument("--version", default="lerobot_v0")
    ap.add_argument("--replay", default="all",
                    help="all / 0 / first N episodes")
    ap.add_argument("--no-video", dest="video", action="store_false")
    args = ap.parse_args()

    root, dataset = load_dataset(args.task, args.version)
    groups = episode_indices(dataset)
    show_summary(dataset, groups)
    make_montages(root, dataset, groups)
    plot_trajectory(root, dataset, groups)
    n = len(groups) if args.replay == "all" else int(args.replay)
    if n:
        replay_check(root, dataset, groups, args.task, n, args.video)
