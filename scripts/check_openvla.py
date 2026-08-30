"""用一帧真实 MuJoCo 观测检查 OpenVLA 推理接口（不执行动作）。

    python scripts/check_openvla.py --seed 100

输出模型的 7D LIBERO action、加载/推理耗时，并把实际输入图像保存到
data/openvla_probe/，用于确认相机和 prompt 无误。
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 少数 PyTorch 算子没有 MPS kernel 时退回 CPU；必须在 import torch 前设置。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from envs import ALL_TASKS, G1TaskEnv

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（默认优先 MPS）")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    )
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    # 只取 Task 1 reset 后的 overhead 图像；不向模型提供 task_info/物体坐标。
    env = G1TaskEnv(ALL_TASKS[0])
    obs = env.reset(args.seed)
    image = Image.fromarray(obs["overhead_rgb"])
    instruction = obs["instruction"]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    out_dir = ROOT / "data" / "openvla_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"task1_seed{args.seed}_overhead.png"
    image.save(image_path)

    print(f"device:      {device}")
    print(f"dtype:       {dtype}")
    print(f"model:       {args.model}")
    print(f"instruction: {instruction}")
    print(f"prompt:      {prompt!r}")
    print(f"image:       {image_path} ({image.size[0]}x{image.size[1]})")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    sync(device)
    print(f"load time:   {time.perf_counter() - t0:.2f}s")
    print(f"unnorm keys: {list(model.norm_stats)}")

    batch = processor(prompt, image).to(device, dtype=dtype)
    # OpenVLA predict_action() 会在 prompt 末尾补训练时使用的空格 token，
    # 但旧版 remote code 没有同步扩展 attention_mask。单样本无 padding，
    # 因此删除 mask 等价且可避免新 PyTorch 对 281/280 长度的严格检查失败。
    batch.pop("attention_mask", None)
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = model.predict_action(
            **batch,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )
    sync(device)
    latency = time.perf_counter() - t0

    action = np.asarray(action, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid OpenVLA action: shape={action.shape}, value={action}")

    names = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
    print(f"inference:   {latency:.3f}s")
    print("action7:")
    for name, value in zip(names, action):
        print(f"  {name:8s} {value:+.6f}")
    print("PASS: real MuJoCo image -> finite 7D OpenVLA action")


if __name__ == "__main__":
    main()
