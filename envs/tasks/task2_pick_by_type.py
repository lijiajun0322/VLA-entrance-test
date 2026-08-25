"""任务 2（中等）：桌上有 3 种不同形状/颜色的物体和 3 个目标垫，
按指令把指定物体放到指定垫上（形状 + 颜色双重指代，且有干扰选项）。

指令例: "pick up the blue cylinder and place it on the yellow pad"
随机化: 三个物体和三个垫的位置、颜色分配、指令中目标物体/垫的组合。
"""

import numpy as np

from envs import scene as sb
from .base import Task


class Task2PickByType(Task):
    name = "task2_pick_by_type"
    OBJECTS = [("obj_cube", "cube"), ("obj_cylinder", "cylinder"), ("obj_ball", "ball")]

    def setup(self, spec):
        for name, shape in self.OBJECTS:
            sb.add_object(spec, name, shape, size=0.022)
        for i in range(3):
            sb.add_pad(spec, f"pad{i}", rgba=[0, 0, 0, 1], half=0.055)
        self.obj_colors = {}
        self.pad_colors = {}
        self.target_obj = "obj_cube"
        self.target_pad = "pad0"

    def randomize(self, env, rng):
        pts = sb.sample_workspace(rng, 6, min_dist=0.12)
        colors = list(sb.COLORS)
        perm = rng.permutation(len(colors))
        # 物体 3 色 + 垫 3 色，全部不同（颜色既是外观也是指令词）
        for (name, _), (x, y), ci in zip(self.OBJECTS, pts[:3], perm[:3]):
            env.set_obj_pose(name, sb.table_pos(x, y, 0.05))
            self.obj_colors[name] = colors[ci]
            env.set_geom_rgba(f"{name}_geom", sb.COLORS[colors[ci]])
        for i, (x, y), ci in zip(range(3), pts[3:], perm[3:]):
            env.set_pad_pos(f"pad{i}", [x, y, sb.TABLE_TOP_Z + 0.005])
            self.pad_colors[f"pad{i}"] = colors[ci]
            env.set_geom_rgba(f"pad{i}_geom", sb.COLORS[colors[ci]])
        # 本回合的目标：某个物体 -> 某个垫
        self.target_obj = self.OBJECTS[rng.integers(3)][0]
        self.target_pad = f"pad{rng.integers(3)}"

    def instruction(self, env, rng):
        c = self.obj_colors[self.target_obj]
        shape = self.target_obj.split("_")[1]
        return (f"pick up the {c} {shape} "
                f"and place it on the {self.pad_colors[self.target_pad]} pad")

    def is_success(self, env) -> bool:
        obj, pad = env.obj_xpos(self.target_obj), env.obj_xpos(self.target_pad)
        ok = (np.hypot(*(obj[:2] - pad[:2])) < 0.055
              and abs(obj[2] - sb.TABLE_TOP_Z) < 0.08
              and np.linalg.norm(env.obj_vel(self.target_obj)) < 0.05)
        return bool(ok)

    def expert_spec(self) -> dict:
        # 目标由 randomize 决定，专家在 reset 之后构造
        return {"obj_name": self.target_obj, "pad_name": self.target_pad,
                "place_h": 0.04}

    def info(self, env) -> dict:
        return {
            "obj_colors": dict(self.obj_colors),
            "pad_colors": dict(self.pad_colors),
            "target_obj": self.target_obj,
            "target_pad": self.target_pad,
            "obj_pos": {n: env.obj_xpos(n).tolist() for n, _ in self.OBJECTS},
            "pad_pos": {f"pad{i}": env.obj_xpos(f"pad{i}").tolist() for i in range(3)},
        }
