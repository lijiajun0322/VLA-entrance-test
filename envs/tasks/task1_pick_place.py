"""任务 1（最简单）：抓起桌上的方块，放到目标垫上。

指令例: "pick up the red cube and place it on the green pad"
随机化: 方块/垫位置、方块与垫的颜色组合 —— 防 VLA 对单一固定布局过拟合。
"""

import numpy as np

from envs import scene as sb
from .base import Task


class Task1PickPlace(Task):
    name = "task1_pick_place"

    def setup(self, spec):
        sb.add_object(spec, "cube", "cube", size=0.025)
        sb.add_pad(spec, "target_pad", rgba=[0, 0, 0, 1], half=0.06)
        self.cube_color = "red"
        self.pad_color = "green"

    def randomize(self, env, rng):
        # 方块与目标垫互不重叠
        (cx, cy), (px, py) = sb.sample_workspace(rng, 2, min_dist=0.15)
        env.set_obj_pose("cube", sb.table_pos(cx, cy, 0.05))
        env.set_pad_pos("target_pad", [px, py, sb.TABLE_TOP_Z + 0.005])
        # 颜色：方块和垫颜色不同
        colors = list(sb.COLORS)
        self.cube_color = colors[rng.integers(len(colors))]
        self.pad_color = colors[rng.integers(len(colors))]
        while self.pad_color == self.cube_color:
            self.pad_color = colors[rng.integers(len(colors))]
        env.set_geom_rgba("cube_geom", sb.COLORS[self.cube_color])
        env.set_geom_rgba("target_pad_geom", sb.COLORS[self.pad_color])

    def instruction(self, env, rng):
        return (f"pick up the {self.cube_color} cube "
                f"and place it on the {self.pad_color} pad")

    def is_success(self, env) -> bool:
        cube, pad = env.obj_xpos("cube"), env.obj_xpos("target_pad")
        on_pad = (np.hypot(*(cube[:2] - pad[:2])) < 0.06            # 在垫子范围内
                  and abs(cube[2] - sb.TABLE_TOP_Z) < 0.08          # 落在桌面高度
                  and np.linalg.norm(env.obj_vel("cube")) < 0.05)   # 已静止
        return bool(on_pad)

    def expert_spec(self) -> dict:
        return {"obj_name": "cube", "pad_name": "target_pad", "place_h": 0.045}

    def info(self, env) -> dict:
        return {
            "cube_color": self.cube_color,
            "pad_color": self.pad_color,
            "cube_pos": env.obj_xpos("cube").tolist(),
            "pad_pos": env.obj_xpos("target_pad").tolist(),
        }
