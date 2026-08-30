"""回合录制器：把专家跑出的轨迹存成可训练的数据集。

每帧记录三块内容（规格见 mujoco_datasets/spec.py）：
  观测   任务相机 RGB（Task 1/2 双相机，Task 3 单相机）+ proprio(10)
  标签   action = [ee_x, y, z, gripper] (4) —— VLA 的监督信号
  备份   ctrl (10 = 腰+右臂8 + 双指2, 关节目标) —— 可离线转成其他动作空间
  元数据 phase、上帝视角 task_info（分析用，绝不进模型）

存储布局（一回合三件套，npz + mp4 同名对应）：
  out_dir/episodes/episode_0000.npz   逐帧数据
  out_dir/videos/episode_0000.mp4     一个或多个相机横排视频（20fps）
  out_dir/manifest.jsonl              索引，一回合一行
  out_dir/norm_stats.json             proprio/action 的均值方差（训练归一化用）
  out_dir/failures/                   失败回合（npz+视频成对归档；无失败不建此目录）
"""

import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from envs.robot import GRIPPER_CLOSED
from .spec import ACTION_SPEC, OBS_SPEC

JPEG_QUALITY = 90


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def sample_action(env) -> np.ndarray:
    """当前帧的动作标签：ee 位置 (3) + 夹爪开度 0~1 (1)。

    ee 用实测位置（真实轨迹，无接触塌缩问题）；夹爪用【指令值】而非实测——
    夹住物体时实测开度被物体挡住（如指令1.0实测0.55），录实测回放时就是
    零夹持力，物体会在提升段滑落。标签必须记录"意图"。
    """
    gripper = float(np.clip(env.data.ctrl[env.grip_act_ids[0]] / GRIPPER_CLOSED, 0.0, 1.0))
    return np.concatenate([env.ee_xpos(), [gripper]])


def _compose(images: list[np.ndarray], caption: str) -> np.ndarray:
    """一个或多个相机横排 + 左上角阶段字幕。"""
    img = Image.fromarray(images[0])
    strip = Image.new("RGB", (img.width, 22), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 5), caption[:70], fill=(255, 255, 255))
    img.paste(strip, (0, 0))
    images[0] = np.array(img)
    return np.concatenate(images, axis=1)


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path, task_name: str):
        self.out_dir = Path(out_dir)
        for sub in ("episodes", "videos"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        # 已有 manifest 则接续编号与索引：同一版本目录追加采集不会覆盖旧文件
        man = self.out_dir / "manifest.jsonl"
        if man.exists():
            self._manifest = [json.loads(l) for l in
                              man.read_text().splitlines() if l.strip()]
            ids = [int(Path(e["episode"]).stem.split("_")[-1])
                   for e in self._manifest]
            self._ep_id = max(ids) + 1 if ids else 0
            print(f"[recorder] appending to existing dataset, "
                  f"resuming at episode_{self._ep_id:04d}")
        else:
            self._manifest = []
            self._ep_id = 0
        self._proprio_all, self._action_all = [], []

    # ---- 一回合的生命周期 ----

    def start(self, env, seed: int | None = None):
        """reset 之后调用。seed 存进 manifest，回放验证据此复现初始布局。"""
        self._frames = []
        self._seed = seed
        self._instruction = env.instruction
        self._task_info = env.task_info()
        self._camera_map = dict(env.obs_camera_map)
        self._image_keys = list(self._camera_map)
        self._video_path = self.out_dir / "videos" / f"episode_{self._ep_id:04d}.mp4"
        self._writer = imageio.get_writer(self._video_path,
                                          fps=ACTION_SPEC.control_hz,
                                          macro_block_size=1)

    def sample(self, env, phase: str = ""):
        """专家每走 sample_every 步调一次（建议在 collect_data 里计数）。"""
        obs = env.get_obs()
        frame = {
            "proprio": obs["proprio"].astype(np.float32),
            "action": sample_action(env).astype(np.float32),
            "ctrl": env.data.ctrl[np.concatenate([
                env.arm8_act_ids, env.grip_act_ids])].astype(np.float32),
            "phase": phase,
            "sim_t": float(env.data.time),
        }
        for key in self._image_keys:
            frame[key] = _jpeg(obs[key])
        self._frames.append(frame)
        self._writer.append_data(_compose(
            [obs[key] for key in self._image_keys], f"[{phase}] {self._instruction}"))

    def end(self, success: bool) -> Path | None:
        """回合结束。成功入正集（episodes/ + videos/）；
        失败成对归档到 failures/（npz + 视频）；无失败不建 failures/ 目录。"""
        self._writer.close()
        if not self._frames:
            self._video_path.unlink(missing_ok=True)
            return None
        n = len(self._frames)
        data = {
            "instruction": self._instruction,
            "task_info": json.dumps(self._task_info),
            "task": self.task_name,
            "success": success,
            "n_frames": n,
            "seed": self._seed if self._seed is not None else -1,
        }
        for key in self._image_keys:
            data[key] = np.array([f[key] for f in self._frames], dtype=object)
        for key in ("proprio", "action", "ctrl", "sim_t"):
            data[key] = np.stack([f[key] for f in self._frames])
        data["phase"] = np.array([f["phase"] for f in self._frames])

        stem = f"episode_{self._ep_id:04d}"
        if success:
            path = self.out_dir / "episodes" / f"{stem}.npz"
            ep_rel, vid_rel = f"episodes/{stem}.npz", f"videos/{stem}.mp4"
        else:
            fail_dir = self.out_dir / "failures"
            fail_dir.mkdir(exist_ok=True)          # 只有真失败才建目录
            path = fail_dir / f"{stem}.npz"
            ep_rel, vid_rel = f"failures/{stem}.npz", f"failures/{stem}.mp4"
            self._video_path.rename(fail_dir / f"{stem}.mp4")
        np.savez_compressed(path, **data)

        entry = {
            "episode": ep_rel, "video": vid_rel,
            "n_frames": n, "success": success,
            "instruction": self._instruction, "task": self.task_name,
            "seed": self._seed,
            "cameras": self._camera_map,
        }
        self._manifest.append(entry)
        with open(self.out_dir / "manifest.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if success:
            self._proprio_all.append(data["proprio"])
            self._action_all.append(data["action"])
        self._ep_id += 1
        return path

    # ---- 汇总 ----

    def write_stats(self):
        """成功回合的 proprio/action 归一化统计（训练时直接用）。"""
        if not self._proprio_all:
            return
        p = np.concatenate(self._proprio_all)
        a = np.concatenate(self._action_all)
        stats = {
            "proprio_mean": p.mean(0).tolist(), "proprio_std": p.std(0).tolist(),
            "action_mean": a.mean(0).tolist(), "action_std": a.std(0).tolist(),
            "n_episodes": len(self._proprio_all),
            "n_frames": int(p.shape[0]),
            "obs_spec": {"cameras": list(self._camera_map.values()),
                         "image_keys": self._image_keys,
                         "img_size": OBS_SPEC.img_size,
                         "proprio_dim": OBS_SPEC.proprio_dim},
            "action_spec": {"names": list(ACTION_SPEC.names),
                            "control_hz": ACTION_SPEC.control_hz},
        }
        (self.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=1))
