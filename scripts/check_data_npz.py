"""Inspect and delta-replay the lightweight NPZ dataset backend.

    python scripts/check_data_npz.py --dataset task1_pick_place --version npz_v0
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        episode = {key: data[key] for key in data.files}
    for key in [key for key in episode if key.endswith("_rgb")]:
        episode[key] = [np.asarray(Image.open(io.BytesIO(bytes(value))))
                        for value in episode[key]]
    return episode


def load_dataset(task: str, version: str):
    root = ROOT / "data" / task / version
    manifest = [json.loads(line) for line in
                (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    return root, [entry for entry in manifest if entry["success"]]


def inspect_dataset(root: Path, manifest: list[dict], n_frames=8):
    inspect_dir = root / "inspect"
    inspect_dir.mkdir(exist_ok=True)
    all_actions = []
    for entry in manifest:
        episode = load_episode(root / entry["episode"])
        actions = episode["action"].astype(np.float32)
        all_actions.append(actions)
        if str(episode["alignment"]) != "observation_t+action_t_before_execution":
            raise AssertionError(f"legacy/misaligned episode: {entry['episode']}")
        if np.linalg.norm(actions[:, :3], axis=1).max() > 0.0151:
            raise AssertionError(f"delta limit exceeded: {entry['episode']}")
        selected = np.linspace(0, len(actions) - 1, n_frames).astype(int)
        image_keys = [key for key in episode if key.endswith("_rgb")]
        rows = [np.concatenate([episode[key][i] for i in selected], axis=1)
                for key in image_keys]
        Image.fromarray(np.concatenate(rows, axis=0)).save(
            inspect_dir / f"montage_{Path(entry['episode']).stem}.png"
        )

    actions = np.concatenate(all_actions)
    print(f"episodes: {len(manifest)}, frames: {len(actions)}")
    print(f"action mean: {np.round(actions.mean(0), 4).tolist()}")
    print(f"action std:  {np.round(actions.std(0), 4).tolist()}")
    print(f"delta norm max: {np.linalg.norm(actions[:, :3], axis=1).max():.4f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    for entry in manifest[:10]:
        action = load_episode(root / entry["episode"])["action"]
        xyz = np.cumsum(action[:, :3], axis=0)
        ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.6)
        ax2.plot(action[:, 3], alpha=0.6)
    ax1.set_title("cumulative commanded EE displacement")
    ax2.set_title("gripper command")
    fig.savefig(inspect_dir / "trajectory.png", dpi=120, bbox_inches="tight")


def replay(root: Path, manifest: list[dict], task_name: str, n: int):
    from control.actions import DeltaEEAction
    from control.executor import OnlineActionExecutor
    from envs.g1_env import G1TaskEnv
    from envs.tasks import TASKS

    env = G1TaskEnv(TASKS[task_name]())
    wins = 0
    for entry in manifest[:n]:
        episode = load_episode(root / entry["episode"])
        env.reset(int(episode["seed"]))
        executor = OnlineActionExecutor(env)
        for action in episode["action"]:
            executor.step_delta(DeltaEEAction(action[:3], float(action[3])))
        success = bool(env.is_success())
        wins += success
        print(f"  {Path(entry['episode']).stem}, seed={int(episode['seed'])}: "
              f"{'success' if success else 'FAIL'}")
    print(f"delta-action replay: {wins}/{min(n, len(manifest))} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="task1_pick_place",
                        help="first-level dataset directory under data/")
    parser.add_argument("--version", default="npz_v0")
    parser.add_argument("--replay", default="all", help="all / 0 / first N episodes")
    args = parser.parse_args()
    dataset_root, entries = load_dataset(args.dataset, args.version)
    inspect_dataset(dataset_root, entries)
    count = len(entries) if args.replay == "all" else int(args.replay)
    if count:
        replay(dataset_root, entries, args.dataset, count)
