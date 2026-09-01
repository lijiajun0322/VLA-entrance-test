"""把 OpenVLA 的 LIBERO 7D delta action 安全映射为本项目 4D action。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptedAction:
    action4: np.ndarray
    raw_delta_m: np.ndarray
    applied_delta_m: np.ndarray
    clipped: bool


class OpenVLAActionAdapter:
    """7D [dxyz, drot, gripper] -> [absolute xyz, close01]。

    ``action_units=normalized`` 用于原始 LIBERO checkpoint：translation_scale
    将无量纲 controller command 转为米。``action_units=meters`` 用于本项目微调
    checkpoint：predict_action 已用训练集统计量还原出米制 delta，不再缩放。
    orientation 对当前竖直顶抓任务由 IK 固定，不执行模型 drot。
    OpenVLA/LIBERO gripper 约定 1=open、0=close，需转为本项目 1=close。
    """

    def __init__(self, translation_scale: float = 0.02,
                 action_units: str = "normalized",
                 max_delta_m: float = 0.025,
                 xyz_low=(0.24, -0.28, 0.755),
                 xyz_high=(0.36, -0.03, 0.97)):
        self.translation_scale = float(translation_scale)
        if action_units not in {"normalized", "meters"}:
            raise ValueError(f"invalid action_units: {action_units}")
        self.action_units = action_units
        self.max_delta_m = float(max_delta_m)
        self.xyz_low = np.asarray(xyz_low, dtype=float)
        self.xyz_high = np.asarray(xyz_high, dtype=float)

    def convert(self, action7: np.ndarray, current_xyz: np.ndarray) -> AdaptedAction:
        action7 = np.asarray(action7, dtype=float)
        current_xyz = np.asarray(current_xyz, dtype=float)
        if action7.shape != (7,) or not np.isfinite(action7).all():
            raise ValueError(f"invalid action7: {action7}")
        if current_xyz.shape != (3,) or not np.isfinite(current_xyz).all():
            raise ValueError(f"invalid current_xyz: {current_xyz}")

        scale = self.translation_scale if self.action_units == "normalized" else 1.0
        raw_delta = action7[:3] * scale
        applied = raw_delta.copy()
        norm = float(np.linalg.norm(applied))
        if norm > self.max_delta_m:
            applied *= self.max_delta_m / norm
        target_unclipped = current_xyz + applied
        target = np.clip(target_unclipped, self.xyz_low, self.xyz_high)
        applied = target - current_xyz
        clipped = not np.allclose(applied, raw_delta, atol=1e-9)

        # 官方 LIBERO evaluation 中 OpenVLA gripper 需反转；这里二值化后转为 close01。
        close01 = float(action7[6] < 0.5)
        return AdaptedAction(
            action4=np.concatenate([target, [close01]]).astype(np.float32),
            raw_delta_m=raw_delta.astype(np.float32),
            applied_delta_m=applied.astype(np.float32),
            clipped=clipped,
        )
