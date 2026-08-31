"""策略/专家共用的高层动作类型。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaEEAction:
    """20Hz 世界坐标系末端增量动作；平移单位为米，1 表示闭合夹爪。"""

    delta_xyz_m: np.ndarray
    gripper_close: float

    def __post_init__(self):
        delta = np.asarray(self.delta_xyz_m, dtype=np.float32)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"invalid delta_xyz_m: {delta}")
        close = float(self.gripper_close)
        if not np.isfinite(close) or not 0.0 <= close <= 1.0:
            raise ValueError(f"invalid gripper_close: {close}")
        object.__setattr__(self, "delta_xyz_m", delta)
        object.__setattr__(self, "gripper_close", close)

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.delta_xyz_m, [self.gripper_close]]).astype(
            np.float32
        )

