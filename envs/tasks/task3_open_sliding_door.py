"""任务 3（最难）：抓住把手，将柜门横向滑开。

相比 Task 1/2 的自由物体抓放，本任务要求与带约束的 articulated object 持续
接触：夹爪必须抓稳竖直把手，并沿门的 slide joint 方向连续拉动。每回合还会
在已验证的 IK 可达盒内整体随机平移窗户与框架。
"""

import numpy as np

from envs import scene as sb
from .base import Task


class Task3OpenSlidingDoor(Task):
    name = "task3_open_sliding_door"
    DOOR = "sliding_door"
    JOINT = "sliding_door_joint"
    HANDLE_SITE = "sliding_door_handle_site"
    SUCCESS_QPOS = 0.12        # 15cm 窗扇至少拉开 12cm（约 80%）
    EXPERT_TARGET_QPOS = 0.135 # 基本叠到固定窗扇，给限位留 1.5cm 余量
    DOOR_BASE_POS = np.array([0.365, -0.205, 0.83])
    POS_JITTER_X = 0.025
    POS_JITTER_Y = 0.020

    def setup(self, spec):
        sb.add_sliding_door(spec, self.DOOR)
        self.panel_color = "blue"
        self.handle_color = "orange"

    def randomize(self, env, rng):
        # 外框 geom 使用 frame-local 坐标，活动扇使用独立 slide body；对两个
        # worldbody 子节点施加相同平移，使整套窗结构保持刚性对齐。
        self.position_offset = np.array([
            rng.uniform(-self.POS_JITTER_X, self.POS_JITTER_X),
            rng.uniform(-self.POS_JITTER_Y, self.POS_JITTER_Y),
            0.0,
        ])
        env.set_body_base_pos(f"{self.DOOR}_frame", self.position_offset)
        env.set_body_base_pos(self.DOOR, self.DOOR_BASE_POS + self.position_offset)
        env.set_joint_qpos(self.JOINT, float(rng.uniform(0.0, 0.008)))
        colors = list(sb.COLORS)
        self.panel_color = colors[rng.integers(len(colors))]
        self.handle_color = colors[rng.integers(len(colors))]
        while self.handle_color == self.panel_color:
            self.handle_color = colors[rng.integers(len(colors))]
        panel_rgba = list(sb.COLORS[self.panel_color])
        panel_rgba[3] = 0.28
        env.set_geom_rgba(f"{self.DOOR}_panel", panel_rgba)
        env.set_geom_rgba(f"{self.DOOR}_handle", sb.COLORS[self.handle_color])

    def instruction(self, env, rng):
        variants = (
            "slide the cabinet door open",
            "open the sliding cabinet door",
            "pull the sliding door open",
        )
        return variants[rng.integers(len(variants))]

    def is_success(self, env) -> bool:
        return bool(env.joint_qpos(self.JOINT) >= self.SUCCESS_QPOS
                    and abs(env.joint_qvel(self.JOINT)) < 0.03)

    def expert_spec(self) -> dict:
        return {
            "kind": "sliding_door",
            "joint_name": self.JOINT,
            "handle_site": self.HANDLE_SITE,
            "slide_axis": [0, 1, 0],
            # 拉 14cm：覆盖约 2mm 的接触跟踪误差，同时离 15cm 限位留足余量。
            "open_distance": 0.14,
            "grasp_z_offset": 0.025,
            "success_qpos": self.EXPERT_TARGET_QPOS,
        }

    def info(self, env) -> dict:
        return {
            "panel_color": self.panel_color,
            "handle_color": self.handle_color,
            "door_qpos": env.joint_qpos(self.JOINT),
            "door_qvel": env.joint_qvel(self.JOINT),
            "handle_pos": env.site_xpos(self.HANDLE_SITE).tolist(),
            "door_base_pos": (self.DOOR_BASE_POS + self.position_offset).tolist(),
            "position_offset": self.position_offset.tolist(),
        }
