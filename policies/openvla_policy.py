"""OpenVLA 单图像策略包装：MuJoCo RGB + 指令 -> LIBERO 7D action。"""

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
DEFAULT_UNNORM_KEY = "libero_spatial"


class OpenVLAPolicy:
    """加载一次 checkpoint，之后逐帧输出 7D LIBERO action。"""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 unnorm_key: str = DEFAULT_UNNORM_KEY):
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
        ).to(self.device).eval()
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
