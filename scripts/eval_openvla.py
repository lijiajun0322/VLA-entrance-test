"""OpenVLA zero-shot 闭环评测：真实模型 action -> G1 MuJoCo。

先跑短 smoke：
    python scripts/eval_openvla.py --task 1 --seeds 100 --max-actions 10

正式评测：
    python scripts/eval_openvla.py --task 1 --seeds 100-119 --max-actions 120


python scripts/eval_openvla.py \
  --task 1 \
  --seeds 500-504 \
  --max-actions 120 \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --unnorm-key libero_spatial \
  --action-units normalized \
  --translation-scale 0.02 \
  --run-name task1_base_seeds_500_504

python scripts/eval_openvla.py \
  --task 1 \
  --seeds 500-504 \
  --max-actions 120 \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --adapter outputs/openvla_finetune/task1_pick_place/lora_v1 \
  --unnorm-key g1_task1 \
  --action-units meters \
  --run-name task1_lora_seeds_500_504

"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import numpy as np

from control.executor import OnlineActionExecutor
from control.openvla_action_adapter import OpenVLAActionAdapter
from envs import ALL_TASKS, G1TaskEnv
from envs import scene as sb
from policies import OpenVLAPolicy
from policies.openvla_policy import DEFAULT_MODEL, DEFAULT_UNNORM_KEY

ROOT = Path(__file__).resolve().parent.parent


def parse_seeds(value: str) -> list[int]:
    if "-" in value:
        lo, hi = (int(x) for x in value.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in value.split(",")]


def make_run_name(requested: str | None, task_id: int, seeds_arg: str,
                  translation_scale: float) -> str:
    """生成安全目录名；手工名称也不允许路径穿越。"""
    if requested:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", requested).strip("._")
        if not name:
            raise ValueError("--run-name must contain at least one letter or digit")
        return name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds = re.sub(r"[^0-9,-]+", "_", seeds_arg)
    scale = f"{translation_scale:g}".replace(".", "p")
    return f"task{task_id}_{timestamp}_seeds_{seeds}_scale_{scale}"


def run_episode(env, policy, adapter, seed: int, max_actions: int, out_dir: Path):
    obs = env.reset(seed)
    spec = env.task.expert_spec()
    is_door = spec.get("kind") == "sliding_door"
    executor = OnlineActionExecutor(env)
    executor.reset()
    log = []
    video_path = out_dir / f"seed_{seed}.mp4"
    writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
    success = False
    ever_lifted = False
    ever_reached_target = False
    max_task_progress = 0.0

    try:
        for t in range(max_actions):
            raw7, latency = policy.predict(obs["overhead_rgb"], obs["instruction"])
            adapted = adapter.convert(raw7, env.ee_xpos())
            ik_ok = executor.step(adapted.action4)
            obs = env.get_obs()
            images = [obs["overhead_rgb"]]
            if "wrist_rgb" in obs:
                images.append(obs["wrist_rgb"])
            writer.append_data(np.concatenate(images, axis=1))
            success = env.is_success()
            if is_door:
                door_qpos = env.joint_qpos(spec["joint_name"])
                handle = env.site_xpos(spec["handle_site"])
                ever_reached_target |= bool(
                    np.linalg.norm(env.ee_xpos() - handle) < 0.06
                )
                progress = door_qpos / spec["success_qpos"]
                max_task_progress = max(max_task_progress, progress)
                task_state = {
                    "door_qpos": door_qpos,
                    "door_progress": progress,
                    "handle_xyz": handle.tolist(),
                }
            else:
                obj = env.obj_xpos(spec["obj_name"])
                target = env.obj_xpos(spec["pad_name"])
                ever_lifted |= bool(obj[2] > sb.TABLE_TOP_Z + 0.08)
                ever_reached_target |= bool(
                    np.linalg.norm(obj[:2] - target[:2]) < 0.06
                )
                max_task_progress = max(
                    max_task_progress,
                    float(np.linalg.norm(obj[:2] - target[:2]) < 0.06),
                )
                task_state = {
                    "object_name": spec["obj_name"],
                    "target_name": spec["pad_name"],
                    "object_xyz": obj.tolist(),
                    "target_xyz": target.tolist(),
                }
            row = {
                "t": t,
                "raw_action7": raw7.tolist(),
                "action4": adapted.action4.tolist(),
                "raw_delta_m": adapted.raw_delta_m.tolist(),
                "applied_delta_m": adapted.applied_delta_m.tolist(),
                "clipped": adapted.clipped,
                "ik_converged": bool(ik_ok),
                "ee_xyz": env.ee_xpos().tolist(),
                "task_state": task_state,
                "inference_s": latency,
                "success": success,
            }
            log.append(row)
            print(f"seed={seed} t={t:03d} a7={np.round(raw7, 3)} "
                  f"target={np.round(adapted.action4, 3)} "
                  f"ik={'ok' if ik_ok else 'FAIL'} {latency:.2f}s")
            if success:
                break
    finally:
        writer.close()

    latencies = np.array([row["inference_s"] for row in log], dtype=float)
    clipped = sum(row["clipped"] for row in log)
    episode = {
        "seed": seed,
        "task": env.task.name,
        "task_kind": "sliding_door" if is_door else "pick_place",
        "instruction": env.instruction,
        "success": success,
        "ever_lifted": ever_lifted,
        "ever_reached_target": ever_reached_target,
        "max_task_progress": max_task_progress,
        "n_actions": len(log),
        "ik_failures": executor.ik_failures,
        "ik_failure_rate": executor.ik_failures / len(log) if log else 0.0,
        "clipped_actions": clipped,
        "clipping_rate": clipped / len(log) if log else 0.0,
        "mean_inference_s": float(latencies.mean()) if len(log) else 0.0,
        "p95_inference_s": float(np.percentile(latencies, 95)) if len(log) else 0.0,
        "video": str(video_path.relative_to(ROOT)),
        "steps": log,
    }
    log_path = out_dir / f"seed_{seed}.json"
    log_path.write_text(json.dumps(episode, indent=2))
    episode["log"] = str(log_path.relative_to(ROOT))
    return episode


def summarize(episodes: list[dict], policy, args, elapsed_s: float) -> dict:
    """汇总一批 rollout；所有 rate 均以实际 action/episode 数为分母。"""
    n = len(episodes)
    total_actions = sum(e["n_actions"] for e in episodes)
    latencies = np.array([
        step["inference_s"] for e in episodes for step in e["steps"]
    ], dtype=float)
    successes = sum(e["success"] for e in episodes)
    lifted = sum(e["ever_lifted"] for e in episodes)
    reached = sum(e["ever_reached_target"] for e in episodes)
    ik_failures = sum(e["ik_failures"] for e in episodes)
    clipped = sum(e["clipped_actions"] for e in episodes)
    return {
        "model": policy.model_id,
        "device": str(policy.device),
        "task_id": args.task,
        "task": episodes[0]["task"],
        "seeds": [e["seed"] for e in episodes],
        "max_actions": args.max_actions,
        "action_units": args.action_units,
        "translation_scale": args.translation_scale,
        "n_episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "episodes_lifted": lifted,
        "lift_rate": lifted / n,
        "episodes_reached_target": reached,
        "reach_target_rate": reached / n,
        "total_actions": total_actions,
        "mean_actions_per_episode": total_actions / n,
        "ik_failures": ik_failures,
        "ik_failure_rate": ik_failures / total_actions if total_actions else 0.0,
        "clipped_actions": clipped,
        "clipping_rate": clipped / total_actions if total_actions else 0.0,
        "inference_s": {
            "mean": float(latencies.mean()) if len(latencies) else 0.0,
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            "max": float(latencies.max()) if len(latencies) else 0.0,
        },
        "wall_time_s": elapsed_s,
    }


def write_episode_csv(path: Path, episodes: list[dict]) -> None:
    fields = [
        "seed", "task", "task_kind", "success", "ever_lifted",
        "ever_reached_target", "max_task_progress", "n_actions",
        "ik_failures", "ik_failure_rate", "clipped_actions", "clipping_rate",
        "mean_inference_s", "p95_inference_s", "video", "log",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({key: episode[key] for key in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=(1, 2, 3), default=1,
                    help="评测任务编号：1=pick-place, 2=pick-by-type, 3=sliding-door")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Hugging Face model id, local merged checkpoint, or adapter base")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT adapter checkpoint; merged in memory before eval")
    ap.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY,
                    help="model action normalization key; fine-tuned Task 1 uses g1_task1")
    ap.add_argument("--seeds", default="100")
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--translation-scale", type=float, default=0.02)
    ap.add_argument("--action-units", choices=("normalized", "meters"),
                    default="normalized",
                    help="original LIBERO model: normalized; project fine-tune: meters")
    ap.add_argument("--run-name", default=None,
                    help="输出子目录名；默认由时间、seeds 和 action scale 生成")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    eval_root = ROOT / "data" / "openvla_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    out_dir = eval_root / make_run_name(
        args.run_name, args.task, args.seeds, args.translation_scale
    )
    try:
        out_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"evaluation run directory already exists: {out_dir}; "
            "choose a different --run-name"
        ) from exc
    print(f"output dir: {out_dir}")
    env = G1TaskEnv(ALL_TASKS[args.task - 1])
    policy = OpenVLAPolicy(
        model_id=args.model, device="cpu" if args.cpu else None,
        unnorm_key=args.unnorm_key, adapter_path=args.adapter,
    )
    adapter = OpenVLAActionAdapter(
        translation_scale=args.translation_scale,
        action_units=args.action_units,
    )

    episodes = []
    t0 = time.perf_counter()
    for seed in seeds:
        episode = run_episode(
            env, policy, adapter, seed, args.max_actions, out_dir
        )
        episodes.append(episode)
        print(f"[{seed}] {'SUCCESS' if episode['success'] else 'FAIL'} "
              f"lifted={episode['ever_lifted']} "
              f"reached_target={episode['ever_reached_target']} "
              f"actions={episode['n_actions']} ik_failures={episode['ik_failures']} "
              f"video={episode['video']}")

    summary = summarize(episodes, policy, args, time.perf_counter() - t0)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "episodes.csv"
    summary_path.write_text(json.dumps({
        **summary,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "episodes": [{k: v for k, v in e.items() if k != "steps"} for e in episodes],
    }, indent=2))
    write_episode_csv(csv_path, episodes)

    print("\n=== OpenVLA evaluation ===")
    print(f"success:       {summary['successes']}/{summary['n_episodes']} "
          f"= {summary['success_rate']:.1%}")
    if episodes[0]["task_kind"] == "sliding_door":
        print(f"reached handle:{summary['episodes_reached_target']}/{summary['n_episodes']} "
              f"= {summary['reach_target_rate']:.1%}")
    else:
        print(f"lifted object: {summary['episodes_lifted']}/{summary['n_episodes']} "
              f"= {summary['lift_rate']:.1%}")
        print(f"reached pad:   {summary['episodes_reached_target']}/{summary['n_episodes']} "
              f"= {summary['reach_target_rate']:.1%}")
    print(f"IK failures:   {summary['ik_failures']}/{summary['total_actions']} "
          f"= {summary['ik_failure_rate']:.1%}")
    print(f"action clips:  {summary['clipped_actions']}/{summary['total_actions']} "
          f"= {summary['clipping_rate']:.1%}")
    print(f"inference:     mean={summary['inference_s']['mean']:.3f}s "
          f"p50={summary['inference_s']['p50']:.3f}s "
          f"p95={summary['inference_s']['p95']:.3f}s")
    print(f"summary:       {summary_path}")
    print(f"episodes CSV:  {csv_path}")


if __name__ == "__main__":
    main()
