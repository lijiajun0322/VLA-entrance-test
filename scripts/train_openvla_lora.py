"""LoRA fine-tune OpenVLA directly from this project's LeRobot v3 data.

Examples:
    # Validate data/action-token encoding without loading the 7B model.
    python scripts/train_openvla_lora.py --task 1 --dry-run

    # Or explicitly select any compatible LeRobot v3 directory.
    python scripts/train_openvla_lora.py --task 1 \
        --dataset-path data/task1_pick_place/lerobot_v0 --dry-run

    # MPS training on the Task 1 demonstrations.
    python scripts/train_openvla_lora.py --task 1 --epochs 5 \
        --batch-size 1 --grad-accumulation 8 --image-aug
        

python scripts/train_openvla_lora.py \
    --task 1 \
    --dataset-path data/task1_pick_place/lerobot_no_wait_v2 \
    --epochs 1 \
    --batch-size 16 \
    --grad-accumulation 1 \
    --learning-rate 5e-4 \
    --lora-rank 32 \
    --val-ratio 0.1 \
    --eval-every-steps 100 \
    --log-every 5 \
    --save-every-epoch \
    --output-dir outputs/openvla_finetune/task1_pick_place/lora_v1

The stored 4D command is converted to a physical-unit 7D command:
    [dx_m, dy_m, dz_m, 0, 0, 0, 1-gripper_close]
The training labels are quantile-normalized to action tokens. The q01/q99
statistics are stored under ``g1_task<N>`` so predict_action returns meters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from envs import ALL_TASKS

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
IGNORE_INDEX = -100


class ActionTokenizer:
    """Official OpenVLA 256-bin action tokenization."""

    def __init__(self, tokenizer, bins: int = 256):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.bins = np.linspace(-1.0, 1.0, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = tokenizer.vocab_size - (bins + 1)

    def __call__(self, action: np.ndarray) -> str:
        action = np.clip(np.asarray(action), -1.0, 1.0)
        discretized = np.digitize(action, self.bins)
        return self.tokenizer.decode(
            (self.tokenizer.vocab_size - discretized).tolist()
        )

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        indices = self.tokenizer.vocab_size - np.asarray(token_ids) - 1
        indices = np.clip(indices, 0, len(self.bin_centers) - 1)
        return self.bin_centers[indices]


def action4_to_openvla(action4) -> np.ndarray:
    """Map this project's causal command to physical-unit OpenVLA 7D."""
    action4 = np.asarray(action4, dtype=np.float32)
    if action4.shape != (4,):
        raise ValueError(f"expected a 4D action, got {action4.shape}")
    action7 = np.zeros(7, dtype=np.float32)
    action7[:3] = action4[:3]
    action7[6] = 1.0 - action4[3]  # OpenVLA: 1=open; this project: 1=close
    return action7


def compute_action_stats(dataset, indices: list[int]) -> dict:
    """Build physical 7D q01/q99 stats from training frames only."""
    column = dataset.dataset.hf_dataset["action"]
    actions = np.stack([np.asarray(column[i], dtype=np.float32) for i in indices])
    q01_4, q99_4 = np.quantile(actions, (0.01, 0.99), axis=0)
    q01 = np.array([q01_4[0], q01_4[1], q01_4[2], -1, -1, -1, 0],
                   dtype=np.float32)
    q99 = np.array([q99_4[0], q99_4[1], q99_4[2], 1, 1, 1, 1],
                   dtype=np.float32)
    # Rotation is fixed and ignored by the executor, so it is not unnormalized.
    mask = np.array([True, True, True, False, False, False, True])
    return {"q01": q01, "q99": q99, "mask": mask}


def split_by_episode(dataset, val_ratio: float, seed: int):
    """Return frame indices with whole episodes assigned to one split."""
    episode_column = dataset.dataset.hf_dataset["episode_index"]
    episode_ids = np.asarray([int(value) for value in episode_column])
    episodes = np.unique(episode_ids)
    if val_ratio == 0:
        return list(range(len(dataset))), [], episodes.tolist(), []
    if len(episodes) < 2:
        raise ValueError("validation requires at least two episodes")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_ratio)))
    val_episodes = set(shuffled[:n_val].tolist())
    train_indices = np.flatnonzero(
        ~np.isin(episode_ids, list(val_episodes))
    ).tolist()
    val_indices = np.flatnonzero(
        np.isin(episode_ids, list(val_episodes))
    ).tolist()
    train_episodes = sorted(set(episodes.tolist()) - val_episodes)
    return train_indices, val_indices, train_episodes, sorted(val_episodes)


def normalize_action(action7: np.ndarray, stats: dict) -> np.ndarray:
    low, high, mask = stats["q01"], stats["q99"], stats["mask"]
    normalized = action7.copy()
    normalized[mask] = 2.0 * (action7[mask] - low[mask]) / (
        high[mask] - low[mask]
    ) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


class TaskLeRobotDataset(Dataset):
    def __init__(self, root: Path, task_name: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(
            f"local/{task_name}", root=root, video_backend="pyav"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return {
            "image": row["observation.images.overhead"],
            "instruction": row["task"],
            "action": row["action"],
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
        }


class OpenVLACollator:
    def __init__(self, processor, action_stats: dict, image_aug: bool):
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.action_stats = action_stats
        self.image_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.10, hue=0.02),
        ]) if image_aug else None

    @staticmethod
    def prompt(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def __call__(self, examples: list[dict]) -> dict:
        images, texts, actions = [], [], []
        eos = self.processor.tokenizer.eos_token or ""
        for example in examples:
            tensor = example["image"]
            if self.image_aug is not None:
                tensor = self.image_aug(tensor)
            array = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(array))
            action7 = action4_to_openvla(example["action"])
            normalized = normalize_action(action7, self.action_stats)
            actions.append(action7)
            response = self.action_tokenizer(normalized)
            texts.append(f"{self.prompt(example['instruction'])} {response}{eos}")

        batch = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = IGNORE_INDEX
        # Exactly seven action tokens plus the stop token receive loss.
        for i, length in enumerate(batch["attention_mask"].sum(dim=1).tolist()):
            labels[i, : int(length) - 8] = IGNORE_INDEX
        batch["labels"] = labels
        batch["actions"] = torch.from_numpy(np.stack(actions))
        return batch


@dataclass
class TrainConfig:
    task: int
    dataset: str
    model: str
    output_dir: str
    epochs: int
    batch_size: int
    grad_accumulation: int
    learning_rate: float
    lora_rank: int
    image_aug: bool
    val_ratio: float
    eval_every_steps: int
    seed: int


def choose_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_action_norm_stats(model, key: str, stats: dict) -> None:
    """Store physical-unit statistics used by predict_action unnormalization."""
    model.norm_stats[key] = {
        "action": {
            "q01": stats["q01"].tolist(),
            "q99": stats["q99"].tolist(),
            "mask": stats["mask"].tolist(),
        }
    }
    model.config.norm_stats = model.norm_stats


def assert_trainable_parameters_finite(model) -> None:
    """Refuse to save an adapter contaminated by NaN/Inf."""
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite trainable parameter: {name}")


@torch.no_grad()
def validate(model, loader, device: torch.device, dtype: torch.dtype) -> float:
    """Compute mean held-out cross-entropy without augmentation or gradients."""
    model.eval()
    loss_sum, sample_count = 0.0, 0
    for batch in loader:
        batch.pop("actions")
        batch = {key: value.to(device) for key, value in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(dtype)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=device.type == "mps"):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss: {loss.item()}")
        batch_size = batch["input_ids"].shape[0]
        loss_sum += float(loss.cpu()) * batch_size
        sample_count += batch_size
    model.train()
    return loss_sum / sample_count


def save_checkpoint(model, processor, out_dir: Path, cfg: TrainConfig,
                    norm_key: str, action_stats: dict,
                    metadata: dict | None = None) -> None:
    started = time.perf_counter()
    tqdm.write(f"[checkpoint] start LoRA adapter: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_trainable_parameters_finite(model)
    set_action_norm_stats(model, norm_key, action_stats)
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    (out_dir / "training_config.json").write_text(
        json.dumps({**asdict(cfg), "unnorm_key": norm_key,
                    "merged": False, **(metadata or {})}, indent=2)
    )
    (out_dir / "action_norm_stats.json").write_text(json.dumps({
        norm_key: {"action": {
            "q01": action_stats["q01"].tolist(),
            "q99": action_stats["q99"].tolist(),
            "mask": action_stats["mask"].tolist(),
        }}
    }, indent=2))

    size_bytes = sum(path.stat().st_size for path in out_dir.rglob("*")
                     if path.is_file())
    elapsed = time.perf_counter() - started
    tqdm.write(
        f"[checkpoint] done LoRA adapter: {out_dir} "
        f"({size_bytes / 1024**2:.1f} MiB, {elapsed:.1f}s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--version", default="lerobot_v0")
    ap.add_argument("--dataset-path", type=Path, default=None,
                    help="explicit LeRobot v3 directory; overrides --version")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--overwrite-output", action="store_true",
                    help="allow replacing files in an existing output directory")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="optional optimizer-step limit for a short training smoke test")
    ap.add_argument("--log-every", type=int, default=5,
                    help="print training metrics every N optimizer steps")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="fraction of episodes held out for validation; 0 disables")
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="run validation every N optimizer steps; 0 means epoch end")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accumulation", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--image-aug", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--save-every-steps", type=int, default=0,
                    help="save an adapter checkpoint every N optimizer steps; 0 disables")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate one encoded batch without loading model weights")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.log_every <= 0:
        ap.error("--log-every must be positive")
    if args.save_every_steps < 0:
        ap.error("--save-every-steps must be non-negative")
    if not 0.0 <= args.val_ratio < 1.0:
        ap.error("--val-ratio must be in [0, 1)")
    if args.eval_every_steps < 0:
        ap.error("--eval-every-steps must be non-negative")
    if args.val_ratio == 0 and args.eval_every_steps > 0:
        ap.error("--eval-every-steps requires --val-ratio > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_name = ALL_TASKS[args.task - 1].name
    dataset_root = args.dataset_path or (ROOT / "data" / task_name / args.version)
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "outputs" / "openvla_finetune" / task_name / "lora_v0"
    )
    output_dir = output_dir.expanduser().resolve()
    if (not args.dry_run and output_dir.exists() and any(output_dir.iterdir())
            and not args.overwrite_output):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose --output-dir "
            "with a new run name, or explicitly pass --overwrite-output"
        )
    cfg = TrainConfig(
        task=args.task, dataset=str(dataset_root),
        model=args.model, output_dir=str(output_dir), epochs=args.epochs,
        batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
        learning_rate=args.learning_rate, lora_rank=args.lora_rank,
        image_aug=args.image_aug, val_ratio=args.val_ratio,
        eval_every_steps=args.eval_every_steps, seed=args.seed,
    )

    print(f"dataset: {dataset_root}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dataset = TaskLeRobotDataset(dataset_root, task_name)
    train_indices, val_indices, train_episodes, val_episodes = split_by_episode(
        dataset, args.val_ratio, args.seed
    )
    action_stats = compute_action_stats(dataset, train_indices)
    train_collator = OpenVLACollator(processor, action_stats, args.image_aug)
    val_collator = OpenVLACollator(processor, action_stats, image_aug=False)
    loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size,
        shuffle=not args.dry_run, num_workers=args.num_workers,
        collate_fn=train_collator,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=val_collator,
        )

    first = next(iter(loader))
    supervised = (first["labels"] != IGNORE_INDEX).sum(dim=1)
    print(f"episodes: train={len(train_episodes)} {train_episodes}, "
          f"val={len(val_episodes)} {val_episodes}")
    print(f"frames: train={len(train_indices)}, val={len(val_indices)}, "
          f"batch shape: {tuple(first['input_ids'].shape)}")
    print(f"first physical action7: {first['actions'][0].tolist()}")
    print(f"q01 xyz: {action_stats['q01'][:3].tolist()}")
    print(f"q99 xyz: {action_stats['q99'][:3].tolist()}")
    print(f"supervised tokens/sample: {supervised.tolist()} (expected 8)")
    if not torch.all(supervised == 8):
        raise RuntimeError("action label mask must contain 7 action tokens + EOS")
    if args.dry_run:
        print("PASS: LeRobot frame -> OpenVLA training batch")
        return

    from peft import LoraConfig, get_peft_model

    device = choose_device(args.cpu)
    if device.type == "cpu" and not args.cpu:
        raise RuntimeError("MPS is unavailable; pass --cpu only for debugging")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    print(f"loading {args.model} on {device} ({dtype})")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=min(args.lora_rank, 16),
        lora_dropout=0.0, target_modules="all-linear",
        init_lora_weights="gaussian",
    ))
    # MPS trains the frozen backbone in FP16, but AdamW states for FP16 LoRA
    # parameters are numerically fragile (notably eps=1e-8 underflows). Keep all
    # trainable parameters and optimizer states in FP32; PEFT casts internally.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )
    scaler = torch.amp.GradScaler(
        device.type, init_scale=1024.0, enabled=device.type == "mps"
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accumulation)
    print(f"optimizer steps: {steps_per_epoch}/epoch, {steps_per_epoch * args.epochs} total")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    stop = False
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.csv"
    val_metrics_path = output_dir / "validation_metrics.csv"
    with (metrics_path.open("w", newline="") as metrics_file,
          val_metrics_path.open("w", newline="") as val_metrics_file):
        metrics = csv.DictWriter(
            metrics_file,
            fieldnames=("epoch", "global_step", "loss", "grad_norm", "scale",
                        "learning_rate"),
        )
        metrics.writeheader()
        val_metrics = csv.DictWriter(
            val_metrics_file,
            fieldnames=("epoch", "global_step", "val_loss", "val_frames"),
        )
        val_metrics.writeheader()
        last_eval_step = -1

        def run_validation(current_epoch: int) -> None:
            nonlocal last_eval_step
            if val_loader is None:
                return
            val_loss = validate(model, val_loader, device, dtype)
            val_metrics.writerow({
                "epoch": current_epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                "val_frames": len(val_indices),
            })
            val_metrics_file.flush()
            last_eval_step = global_step
            tqdm.write(
                f"[val] epoch={current_epoch} step={global_step} "
                f"loss={val_loss:.6f} frames={len(val_indices)}"
            )

        for epoch in range(1, args.epochs + 1):
            progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
            running_loss = 0.0
            accumulation_loss = 0.0
            accumulation_count = 0
            for micro_step, batch in enumerate(progress, 1):
                batch.pop("actions")
                batch = {k: v.to(device) for k, v in batch.items()}
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=device.type == "mps"):
                    output = model(**batch)
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss before backward: epoch={epoch}, "
                        f"micro_step={micro_step}, loss={loss.item()}"
                    )
                scaler.scale(loss / args.grad_accumulation).backward()
                loss_value = float(loss.detach().cpu())
                running_loss += loss_value
                accumulation_loss += loss_value
                accumulation_count += 1
                do_step = (micro_step % args.grad_accumulation == 0
                           or micro_step == len(loader))
                if do_step:
                    step_loss = accumulation_loss / accumulation_count
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics.writerow({
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": step_loss,
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "scale": scaler.get_scale(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })
                    metrics_file.flush()
                    if global_step == 1 or global_step % args.log_every == 0:
                        tqdm.write(
                            f"[train] epoch={epoch} step={global_step} "
                            f"loss={step_loss:.6f} "
                            f"grad_norm={float(grad_norm.detach().cpu()):.4f} "
                            f"scale={scaler.get_scale():.1f} "
                            f"lr={optimizer.param_groups[0]['lr']:.2e}"
                        )
                    if (args.save_every_steps > 0
                            and global_step % args.save_every_steps == 0):
                        checkpoint_dir = (
                            output_dir / "checkpoints" / f"step_{global_step:06d}"
                        )
                        save_checkpoint(
                            model, processor, checkpoint_dir, cfg,
                            f"g1_task{args.task}", action_stats,
                            metadata={"epoch": epoch, "global_step": global_step},
                        )
                        tqdm.write(f"[checkpoint] saved adapter: {checkpoint_dir}")
                    if (val_loader is not None and args.eval_every_steps > 0
                            and global_step % args.eval_every_steps == 0):
                        run_validation(epoch)
                    accumulation_loss = 0.0
                    accumulation_count = 0
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                progress.set_postfix(loss=f"{running_loss / micro_step:.4f}",
                                     step=global_step)
                if stop:
                    break

            # With interval evaluation, still validate the final state of each
            # epoch unless that exact optimizer step was already evaluated.
            if val_loader is not None and last_eval_step != global_step:
                run_validation(epoch)
            if args.save_every_epoch and epoch != args.epochs:
                epoch_dir = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
                save_checkpoint(
                    model, processor, epoch_dir, cfg,
                    f"g1_task{args.task}", action_stats,
                    metadata={"epoch": epoch, "global_step": global_step},
                )
                tqdm.write(f"[checkpoint] saved adapter: {epoch_dir}")
            if stop:
                break

    save_checkpoint(model, processor, output_dir, cfg, f"g1_task{args.task}",
                    action_stats,
                    metadata={"epoch": epoch, "global_step": global_step})
    print(f"saved: {output_dir}")
    print(f"metrics: {metrics_path}")
    if val_loader is not None:
        print(f"validation metrics: {val_metrics_path}")
    print(f"eval unnorm key: g1_task{args.task}")


if __name__ == "__main__":
    main()
