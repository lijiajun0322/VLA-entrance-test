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

# 动作标签 = ee_site 世界坐标(3) + 夹爪开度(1)。
# 夹爪全程竖直朝下（姿态恒定），故 4 维即完整动作；推理时用
# control.CartesianServo 跟踪 ee 目标即可驱动仿真（无需完整 IK）。
ACTION_SPEC = ActionSpec(
    names=("ee_x", "ee_y", "ee_z", "gripper"),
    control_hz=20,
)
