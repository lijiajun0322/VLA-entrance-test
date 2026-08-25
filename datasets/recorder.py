"""回合录制器：把专家跑出的轨迹存成可训练的数据集。

每帧记录三块内容（规格见 data/spec.py）：
  观测   overhead_rgb / wrist_rgb (JPEG 压缩)、proprio (9)
  标签   action = [ee_x, ee_y, ee_z, gripper] (4) —— VLA 的监督信号
  备份   ctrl (10 = 腰+右臂8 + 双指2, 关节目标) —— 可离线转成其他动作空间
  元数据 phase、上帝视角 task_info（分析用，绝不进模型）

存储布局：
  out_dir/episode_0000.npz   每回合一个文件
  out_dir/failures/          失败回合（单独归档，报告用）
  out_dir/manifest.json      索引
  out_dir/stats.json         proprio/action 的均值方差（训练归一化用）
"""

import io
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from envs.robot import GRIPPER_CLOSED
from .spec import ACTION_SPEC, OBS_SPEC

JPEG_QUALITY = 90


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def sample_action(env) -> np.ndarray:
    """当前帧的动作标签：ee 位置 (3) + 夹爪开度 0~1 (1)。"""
    adr = env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left_joint")]
    gripper = float(np.clip(env.data.qpos[adr] / GRIPPER_CLOSED, 0.0, 1.0))
    return np.concatenate([env.ee_xpos(), [gripper]])


class EpisodeRecorder:
    def __init__(self, out_dir: str | Path, task_name: str):
        self.out_dir = Path(out_dir)
        (self.out_dir / "failures").mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self._manifest = []
        self._proprio_all, self._action_all = [], []
        self._ep_id = 0

    # ---- 一回合的生命周期 ----

    def start(self, env):
        """reset 之后调用。"""
        self._frames = []
        self._instruction = env.instruction
        self._task_info = env.task_info()

    def sample(self, env, phase: str = ""):
        """专家每走 sample_every 步调一次（建议在 collect_data 里计数）。"""
        obs = env.get_obs()
        self._frames.append({
            "overhead_rgb": _jpeg(obs["overhead_rgb"]),
            "wrist_rgb": _jpeg(obs["wrist_rgb"]),
            "proprio": obs["proprio"].astype(np.float32),
            "action": sample_action(env).astype(np.float32),
            "ctrl": env.data.ctrl[np.concatenate([
                env.arm8_act_ids, env.grip_act_ids])].astype(np.float32),
            "phase": phase,
            "sim_t": float(env.data.time),
        })

    def end(self, success: bool) -> Path | None:
        """回合结束。成功才入正集；失败归档到 failures/。"""
        if not self._frames:
            return None
        n = len(self._frames)
        data = {
            "instruction": self._instruction,
            "task_info": json.dumps(self._task_info),
            "task": self.task_name,
            "success": success,
            "n_frames": n,
        }
        for key in ("overhead_rgb", "wrist_rgb"):
            data[key] = np.array([f[key] for f in self._frames], dtype=object)
        for key in ("proprio", "action", "ctrl", "sim_t"):
            data[key] = np.stack([f[key] for f in self._frames])
        data["phase"] = np.array([f["phase"] for f in self._frames])

        sub = "" if success else "failures/"
        path = self.out_dir / f"{sub}episode_{self._ep_id:04d}.npz"
        np.savez_compressed(path, **data)

        self._manifest.append({
            "episode": path.name, "n_frames": n, "success": success,
            "instruction": self._instruction, "task": self.task_name,
        })
        if success:
            self._proprio_all.append(data["proprio"])
            self._action_all.append(data["action"])
        self._ep_id += 1
        self._write_manifest()
        return path

    # ---- 汇总 ----

    def _write_manifest(self):
        (self.out_dir / "manifest.json").write_text(
            json.dumps(self._manifest, indent=1, ensure_ascii=False))

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
            "obs_spec": {"cameras": list(OBS_SPEC.cameras),
                         "img_size": OBS_SPEC.img_size,
                         "proprio_dim": OBS_SPEC.proprio_dim},
            "action_spec": {"names": list(ACTION_SPEC.names),
                            "control_hz": ACTION_SPEC.control_hz},
        }
        (self.out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
