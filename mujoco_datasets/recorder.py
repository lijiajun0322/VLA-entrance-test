"""Direct LeRobot v3 recorder for MuJoCo expert demonstrations.

Each stored row is one causal transition label:
    observation_t, instruction_t, action_t -> observation_{t+1}

Images are encoded as LeRobot MP4 video features and low-dimensional fields are
written to Parquet by the official ``LeRobotDataset`` writer. Failed episodes
are discarded before they enter the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spec import ACTION_SPEC, OBS_SPEC, PROPRIO_NAMES


def _camera_feature(image_key: str) -> str:
    name = image_key.removesuffix("_rgb")
    return f"observation.images.{name}"


def lerobot_features(image_keys: list[str]) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (OBS_SPEC.proprio_dim,),
            "names": list(PROPRIO_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_SPEC.dim,),
            "names": list(ACTION_SPEC.names),
        },
        "seed": {"dtype": "int64", "shape": (1,), "names": ["seed"]},
        "sim_time": {"dtype": "float32", "shape": (1,), "names": ["seconds"]},
        "phase": {"dtype": "string", "shape": (1,), "names": None},
    }
    for key in image_keys:
        features[_camera_feature(key)] = {
            "dtype": "video",
            "shape": (OBS_SPEC.img_size, OBS_SPEC.img_size, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotEpisodeRecorder:
    """Buffer one episode, then commit it with the official LeRobot v3 writer."""

    def __init__(self, out_dir: str | Path, task_name: str,
                 repo_id: str | None = None):
        self.out_dir = Path(out_dir)
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            raise FileExistsError(
                f"LeRobot output directory is not empty: {self.out_dir}. "
                "Choose a new --version; v3 append/resume is not enabled here."
            )
        # LeRobotDataset.create requires the dataset root itself not to exist.
        if self.out_dir.exists():
            self.out_dir.rmdir()
        self.task_name = task_name
        self.repo_id = repo_id or f"local/{task_name}"
        self.dataset = None
        self._frames: list[dict] = []
        self.saved_episodes = 0

    def _create(self, env):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._image_keys = list(env.obs_camera_map)
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.out_dir,
            fps=ACTION_SPEC.control_hz,
            robot_type="unitree_g1_mujoco",
            features=lerobot_features(self._image_keys),
            use_videos=True,
            image_writer_threads=max(2, len(self._image_keys) * 2),
            batch_encoding_size=1,
        )

    def start(self, env, seed: int):
        if self.dataset is None:
            self._create(env)
        elif list(env.obs_camera_map) != self._image_keys:
            raise ValueError("camera schema changed within one LeRobot dataset")
        self._frames = []
        self._seed = int(seed)
        self._instruction = env.instruction

    def add_frame(self, env, action: np.ndarray, phase: str):
        """Capture observation_t together with the command action_t."""
        obs = env.get_obs()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_SPEC.dim,):
            raise ValueError(f"invalid action shape: {action.shape}")
        frame = {
            "observation.state": obs["proprio"].astype(np.float32),
            "action": action,
            "seed": np.array([self._seed], dtype=np.int64),
            "sim_time": np.array([env.data.time], dtype=np.float32),
            "phase": phase,
            "task": self._instruction,
        }
        for key in self._image_keys:
            frame[_camera_feature(key)] = obs[key].copy()
        self._frames.append(frame)

    def end(self, success: bool) -> int:
        """Commit a successful episode; discard a failed one."""
        n_frames = len(self._frames)
        if success:
            for frame in self._frames:
                self.dataset.add_frame(frame)
            # Parallel multi-camera encoding is counterproductive on macOS.
            self.dataset.save_episode(parallel_encoding=False)
            self.saved_episodes += 1
        self._frames = []
        return n_frames

    def finalize(self):
        if self.dataset is not None:
            self.dataset.finalize()


# Backward-compatible import name; storage semantics are now LeRobot v3.
EpisodeRecorder = LeRobotEpisodeRecorder
