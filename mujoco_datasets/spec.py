"""观测/动作规格：env、数据录制、（将来的）策略三方的唯一契约。

改相机、改图像尺寸、换动作空间，只需要改这里。
"""

from dataclasses import dataclass

SIM_DT = 0.002           # 仿真步长 (s)


@dataclass(frozen=True)
class ObsSpec:
    cameras: tuple               # 相机名（= get_obs 里图像字段的来源）
    img_size: int                # 正方形图像边长
    proprio_dim: int             # 本体感知维度（腰yaw1 + 右臂7 + 双指2）


@dataclass(frozen=True)
class ActionSpec:
    names: tuple                 # 动作维度名
    control_hz: int              # 数据采样/策略输出频率
    @property
    def dim(self) -> int:
        return len(self.names)
    @property
    def sample_every(self) -> int:
        """每多少个仿真步采一帧。"""
        return round(1.0 / (SIM_DT * self.control_hz))


OBS_SPEC = ObsSpec(
    cameras=("overhead_cam", "wrist_cam"),
    img_size=224,
    proprio_dim=10,
)

PROPRIO_NAMES = (
    "waist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "finger_left", "finger_right",
)

DEFAULT_CAMERA_MAP = {
    "overhead_rgb": "overhead_cam",
    "wrist_rgb": "wrist_cam",
}
TASK_CAMERA_MAPS = {
    # Task 3 只使用远距离机器人肩后主视角，不录腕部相机。
    "task3_open_sliding_door": {"overhead_rgb": "door_cam"},
}


def camera_map_for_task(task_name: str) -> dict:
    return dict(TASK_CAMERA_MAPS.get(task_name, DEFAULT_CAMERA_MAP))

# 动作标签 = 世界坐标系 EE translation delta (m) + 夹爪闭合指令。
# observation_t 与 action_t 在执行前配对；executor 再将 delta 积分成 absolute
# EE target。夹爪全程保持固定向下姿态，因此 action 不包含 orientation。
ACTION_SPEC = ActionSpec(
    names=("delta_x_m", "delta_y_m", "delta_z_m", "gripper_close"),
    control_hz=20,
)
