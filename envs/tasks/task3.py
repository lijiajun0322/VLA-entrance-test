"""任务 3（最难，占位名：具体玩法后续可能调整）：多物体交互 ——
把方块抓起来叠放到竖直圆柱顶上。

相比任务 1/2 的"放到平面上"，这里要求精确的 3D 对位（方块要落在圆柱顶面
且不掉下来），并涉及两个物体间的接触交互。

指令例: "pick up the red cube and stack it on top of the blue cylinder"
随机化: 方块/圆柱位置、颜色组合。
"""

import numpy as np

from envs import scene as sb
from .base import Task


class Task3(Task):
    name = "task3"

    def setup(self, spec):
        sb.add_object(spec, "cube", "cube", size=0.025)
        sb.add_object(spec, "pole", "cylinder", size=0.022)   # 半径 0.022
        self.cube_color = "red"
        self.pole_color = "blue"

    def randomize(self, env, rng):
        (cx, cy), (px, py) = sb.sample_workspace(rng, 2, min_dist=0.16)
        env.set_obj_pose("cube", sb.table_pos(cx, cy, 0.05))
        # 圆柱竖直立在桌上：初始姿态绕 x 转 90°，从高处落下落稳
        env.set_obj_pose("pole", sb.table_pos(px, py, 0.05),
                         quat=(0.7071, 0.7071, 0, 0))
        colors = list(sb.COLORS)
        self.cube_color = colors[rng.integers(len(colors))]
        self.pole_color = colors[rng.integers(len(colors))]
        while self.pole_color == self.cube_color:
            self.pole_color = colors[rng.integers(len(colors))]
        env.set_geom_rgba("cube_geom", sb.COLORS[self.cube_color])
        env.set_geom_rgba("pole_geom", sb.COLORS[self.pole_color])

    def instruction(self, env, rng):
        return (f"pick up the {self.cube_color} cube "
                f"and stack it on top of the {self.pole_color} cylinder")

    def is_success(self, env) -> bool:
        cube, pole = env.obj_xpos("cube"), env.obj_xpos("pole")
        pole_half_h = 0.022
        stacked = (np.hypot(*(cube[:2] - pole[:2])) < 0.02             # 对位
                   and pole[2] > sb.TABLE_TOP_Z + 0.01                 # 圆柱立着
                   and cube[2] > pole[2] + pole_half_h                 # 方块在顶上
                   and np.linalg.norm(env.obj_vel("cube")) < 0.05)     # 静止
        return bool(stacked)

    def expert_spec(self) -> dict:
        # 叠放的"垫"是圆柱顶面：放置高度 = 圆柱顶 + 方块半高 - 少许余量
        return {"obj_name": "cube", "pad_name": "pole", "place_h": 0.075}

    def info(self, env) -> dict:
        return {
            "cube_color": self.cube_color,
            "pole_color": self.pole_color,
            "cube_pos": env.obj_xpos("cube").tolist(),
            "pole_pos": env.obj_xpos("pole").tolist(),
        }
