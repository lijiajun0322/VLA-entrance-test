"""OpenVLA 单图像策略包装：MuJoCo RGB + 指令 -> LIBERO 7D action。"""

import os
import time
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
DEFAULT_UNNORM_KEY = "libero_spatial"


def _adapter_norm_stats(adapter_path: Path) -> dict:
    """Load saved action stats, with a fallback for older project checkpoints."""
    stats_path = adapter_path / "action_norm_stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text())

    # Checkpoints written before action_norm_stats.json was introduced still
    # contain enough information to reproduce the episode split exactly.
    config_path = adapter_path / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter has no action stats or training config: {adapter_path}"
        )
    cfg = json.loads(config_path.read_text())
    dataset_root = Path(cfg["dataset"])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task_name = dataset_root.parent.name
    dataset = LeRobotDataset(
        f"local/{task_name}", root=dataset_root, video_backend="pyav"
    )
    episode_ids = np.asarray([
        int(value) for value in dataset.hf_dataset["episode_index"]
    ])
    episodes = np.unique(episode_ids)
    val_ratio = float(cfg.get("val_ratio", 0.0))
    if val_ratio > 0:
        shuffled = np.random.default_rng(int(cfg.get("seed", 0))).permutation(episodes)
        n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_ratio)))
        train_mask = ~np.isin(episode_ids, shuffled[:n_val])
    else:
        train_mask = np.ones(len(episode_ids), dtype=bool)
    action_column = dataset.hf_dataset["action"]
    actions = np.stack([
        np.asarray(action_column[i], dtype=np.float32)
        for i in np.flatnonzero(train_mask)
    ])
    q01, q99 = np.quantile(actions[:, :3], (0.01, 0.99), axis=0)
    key = cfg.get("unnorm_key", f"g1_task{cfg['task']}")
    return {
        key: {"action": {
            "q01": [*q01.tolist(), -1.0, -1.0, -1.0, 0.0],
            "q99": [*q99.tolist(), 1.0, 1.0, 1.0, 1.0],
            "mask": [True, True, True, False, False, False, True],
        }}
    }


class OpenVLAPolicy:
    """加载一次 checkpoint，之后逐帧输出 7D LIBERO action。"""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 unnorm_key: str = DEFAULT_UNNORM_KEY,
                 adapter_path: str | Path | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model_id = model_id
        self.unnorm_key = unnorm_key

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        if adapter_path is not None:
            from peft import PeftModel

            adapter_path = Path(adapter_path).expanduser().resolve()
            if not adapter_path.exists():
                raise FileNotFoundError(adapter_path)
            print(f"loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path
            ).merge_and_unload()
            stats = _adapter_norm_stats(adapter_path)
            self.model.norm_stats.update(stats)
            self.model.config.norm_stats = self.model.norm_stats
            self.model_id = f"{model_id} + {adapter_path}"
        self.model.eval()
        if unnorm_key not in self.model.norm_stats:
            raise ValueError(f"unnorm key {unnorm_key!r} not in {list(self.model.norm_stats)}")

    @staticmethod
    def prompt(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def _sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()

    def predict(self, rgb: np.ndarray, instruction: str) -> tuple[np.ndarray, float]:
        """返回 (action7, inference_seconds)。rgb 为 HWC uint8。"""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HWC RGB image, got {rgb.shape}")
        image = Image.fromarray(rgb.astype(np.uint8, copy=False))
        batch = self.processor(self.prompt(instruction), image).to(
            self.device, dtype=self.dtype
        )
        # OpenVLA remote code 会补一个 prompt token，却未同步扩展 attention_mask。
        # 单样本无 padding 时 mask 可安全省略。
        batch.pop("attention_mask", None)

        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            action = self.model.predict_action(
                **batch, unnorm_key=self.unnorm_key, do_sample=False
            )
        self._sync()
        latency = time.perf_counter() - t0

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid OpenVLA action: shape={action.shape}, value={action}")
        return action, latency
