"""Lightweight NPZ+MP4 recorder with the same causal delta-action contract.

Unlike :mod:`mujoco_datasets.recorder` (official LeRobot v3), this backend keeps
one compressed NPZ and one preview MP4 per episode. It is useful for debugging
with NumPy, but both backends store ``observation_t`` with the command
``action_t = [delta_xyz_m, gripper_close]`` before that command is executed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .spec import ACTION_SPEC, OBS_SPEC

JPEG_QUALITY = 90


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _compose(images: list[np.ndarray], caption: str) -> np.ndarray:
    images = [image.copy() for image in images]
    img = Image.fromarray(images[0])
    strip = Image.new("RGB", (img.width, 22), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 5), caption[:70], fill=(255, 255, 255))
    img.paste(strip, (0, 0))
    images[0] = np.asarray(img)
    return np.concatenate(images, axis=1)


class NpzEpisodeRecorder:
    def __init__(self, out_dir: str | Path, task_name: str):
        self.out_dir = Path(out_dir)
        if (self.out_dir / "meta" / "info.json").exists():
            raise ValueError(f"refusing to mix NPZ with LeRobot dataset: {self.out_dir}")
        for sub in ("episodes", "videos"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        manifest_path = self.out_dir / "manifest.jsonl"
        if manifest_path.exists():
            self._manifest = [json.loads(line) for line in
                              manifest_path.read_text().splitlines() if line.strip()]
            ids = [int(Path(e["episode"]).stem.split("_")[-1])
                   for e in self._manifest]
            self._ep_id = max(ids) + 1 if ids else 0
        else:
            self._manifest, self._ep_id = [], 0

    def start(self, env, seed: int):
        self._frames = []
        self._seed = int(seed)
        self._instruction = env.instruction
        self._task_info = env.task_info()
        self._camera_map = dict(env.obs_camera_map)
        self._image_keys = list(self._camera_map)
        self._video_path = self.out_dir / "videos" / f"episode_{self._ep_id:06d}.mp4"
        self._writer = imageio.get_writer(
            self._video_path, fps=ACTION_SPEC.control_hz, macro_block_size=1
        )

    def add_frame(self, env, action: np.ndarray, phase: str):
        """Capture observation_t and action_t before action execution."""
        obs = env.get_obs()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_SPEC.dim,):
            raise ValueError(f"invalid action shape: {action.shape}")
        frame = {
            "proprio": obs["proprio"].astype(np.float32),
            "action": action.copy(),
            "ctrl": env.data.ctrl[np.concatenate([
                env.arm8_act_ids, env.grip_act_ids])].astype(np.float32),
            "phase": phase,
            "sim_time": np.float32(env.data.time),
        }
        for key in self._image_keys:
            frame[key] = _jpeg(obs[key])
        self._frames.append(frame)
        self._writer.append_data(_compose(
            [obs[key] for key in self._image_keys], f"[{phase}] {self._instruction}"
        ))

    def end(self, success: bool) -> Path | None:
        self._writer.close()
        if not self._frames:
            self._video_path.unlink(missing_ok=True)
            return None
        n = len(self._frames)
        self.last_n_frames = n
        data = {
            "instruction": self._instruction,
            "task_info": json.dumps(self._task_info),
            "task": self.task_name,
            "success": bool(success),
            "n_frames": n,
            "seed": self._seed,
            "action_semantics": "world_delta_xyz_m+gripper_close",
            "alignment": "observation_t+action_t_before_execution",
        }
        for key in self._image_keys:
            data[key] = np.array([f[key] for f in self._frames], dtype=object)
        for key in ("proprio", "action", "ctrl", "sim_time"):
            data[key] = np.stack([f[key] for f in self._frames])
        data["phase"] = np.array([f["phase"] for f in self._frames])

        stem = f"episode_{self._ep_id:06d}"
        if success:
            path = self.out_dir / "episodes" / f"{stem}.npz"
            ep_rel, video_rel = f"episodes/{stem}.npz", f"videos/{stem}.mp4"
        else:
            failure_dir = self.out_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            path = failure_dir / f"{stem}.npz"
            ep_rel, video_rel = f"failures/{stem}.npz", f"failures/{stem}.mp4"
            self._video_path.rename(failure_dir / f"{stem}.mp4")
        np.savez_compressed(path, **data)

        entry = {
            "episode": ep_rel,
            "video": video_rel,
            "n_frames": n,
            "success": bool(success),
            "instruction": self._instruction,
            "task": self.task_name,
            "seed": self._seed,
            "cameras": self._camera_map,
        }
        self._manifest.append(entry)
        with (self.out_dir / "manifest.jsonl").open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._ep_id += 1
        self._frames = []
        return path

    def write_stats(self):
        successful = [entry for entry in self._manifest if entry["success"]]
        if not successful:
            return
        # Re-read all successful episodes so stats remain correct after append.
        proprio_parts, action_parts = [], []
        for entry in successful:
            with np.load(self.out_dir / entry["episode"], allow_pickle=True) as episode:
                proprio_parts.append(episode["proprio"])
                action_parts.append(episode["action"])
        proprio = np.concatenate(proprio_parts)
        action = np.concatenate(action_parts)
        stats = {
            "proprio_mean": proprio.mean(0).tolist(),
            "proprio_std": proprio.std(0).tolist(),
            "action_mean": action.mean(0).tolist(),
            "action_std": action.std(0).tolist(),
            "n_episodes": len(successful),
            "n_frames": int(len(proprio)),
            "obs_spec": {
                "cameras": list(self._camera_map.values()),
                "image_keys": self._image_keys,
                "img_size": OBS_SPEC.img_size,
                "proprio_dim": OBS_SPEC.proprio_dim,
            },
            "action_spec": {
                "names": list(ACTION_SPEC.names),
                "control_hz": ACTION_SPEC.control_hz,
                "alignment": "observation_t+action_t_before_execution",
            },
        }
        (self.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
