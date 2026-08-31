"""任务 2（中等）：桌上有 3 种不同形状的物体和 3 个目标垫，
按指令把指定物体放到指定垫上（物体按形状指代，垫按颜色指代，有干扰选项）。

指令例: "pick up the cylinder and place it on the yellow pad"
随机化: 三个物体和三个垫的位置、颜色分配、指令中目标物体/垫的组合。
"""

import numpy as np

from envs import scene as sb
from .base import Task


class Task2PickByType(Task):
    name = "task2_pick_by_type"
    # 三种形状：cube / cylinder / T 积木（体积 85/67/65 cm³ 同量级；
    # 球类点接触易滑已弃用）
    OBJECTS = [("obj_cube", "cube"), ("obj_cylinder", "cylinder"), ("obj_tblock", "t")]
    SHAPE_WORDS = {"cube": "cube", "cylinder": "cylinder", "t": "T block"}

    # 布局间距：AABB（切比雪夫距离）判定 + 分类额外间距（CLEAR_*）。
    # 夹爪全开跨距 8.4cm、指板外沿 ±4.8cm：物-物净距不足时夹爪会被邻物挡住
    # 或一把夹起两个。额外保留 4cm AABB 净距，让手指和腕部下降时也有余量。
    PAD_HALF = 0.03
    MARGIN = 0.005             # 补落稳的毫米级漂移
    CLEAR_PAD_PAD = 0.010      # 垫-垫：防放置歧义
    CLEAR_PAD_OBJ = 0.005      # pad 无碰撞；保留视觉净空即可
    CLEAR_OBJ_OBJ = 0.040
    # 各物体的平面半包围（cube/T 轴对齐平躺、圆柱竖立）
    OBJ_HALF = {"cube": (0.021, 0.021), "cylinder": (0.022, 0.022),
                "t": (0.023, 0.026)}
    # 生成位姿（quat, 半高）：直接以落稳定姿态贴桌面生成（+2mm 间隙），
    # 物体不再下坠弹跳滚移，落稳位置即采样位置。
    # 圆柱竖立（r=h/2 矮胖不会倒）：手指沿 y 横跨直径夹持，稳定不滑
    SPAWN = {"cube": ((1, 0, 0, 0), 0.021),
             "cylinder": ((1, 0, 0, 0), 0.022),
             "t": ((1, 0, 0, 0), 0.022)}
    # 抓取时 ee 相对物体中心的 z 补偿（按物体半高定，让指板咬在物体中部）。
    # T 不能更低（如 0.015）：低 1.4cm 的目标会把手臂构型带进位置雅可比
    # 奇异带（最小奇异值 ~0.02），descend 的残差恰好落在不可达方向上，
    # 伺服完全推不动，磨到 8s 超时
    GRASP_Z = {"cube": 0.022, "cylinder": 0.022, "t": 0.022}

    def setup(self, spec):
        for name, shape in self.OBJECTS:
            size = 0.021 if shape == "cube" else 0.022
            sb.add_object(spec, name, shape, size=size)
        for i in range(3):
            sb.add_pad(spec, f"pad{i}", rgba=[0, 0, 0, 1], half=self.PAD_HALF)
        self.obj_colors = {}
        self.pad_colors = {}
        self.target_obj = "obj_cube"
        self.target_pad = "pad0"

    @staticmethod
    def _aabb_free(p, hp, pts, hqs, margin):
        """p/hp 为候选点及其平面半包围，检查与已放点 (q, hq) 的 AABB 是否分离。"""
        return all(abs(p[0] - q[0]) > hp[0] + hq[0] + margin
                   or abs(p[1] - q[1]) > hp[1] + hq[1] + margin
                   for q, hq in zip(pts, hqs))

    def _sample_layout(self, rng):
        """采 3 垫 + 3 物体的位置：AABB 互不重叠，失败整局重试。"""
        X, Y = sb.WORKSPACE["x"], sb.WORKSPACE["y"]
        # 垫中心须留在桌面内（桌 x∈[0.18,0.50], y∈[-0.39,0.13]）
        PX = (max(X[0], sb.TABLE_CENTER_X - sb.TABLE_SIZE[0] + self.PAD_HALF),
              min(X[1], sb.TABLE_CENTER_X + sb.TABLE_SIZE[0] - self.PAD_HALF))
        PY = (max(Y[0], sb.TABLE_CENTER_Y - sb.TABLE_SIZE[1] + self.PAD_HALF),
              min(Y[1], sb.TABLE_CENTER_Y + sb.TABLE_SIZE[1] - self.PAD_HALF))
        ph = (self.PAD_HALF, self.PAD_HALF)
        shapes = [s for _, s in self.OBJECTS]
        for _ in range(1000):                    # 整局重试（工作区收窄后成功率降低）
            pads, objs = [], []
            for _ in range(2000):
                p = (rng.uniform(*PX), rng.uniform(*PY))
                if self._aabb_free(p, ph, pads, [ph] * len(pads),
                                   self.MARGIN + self.CLEAR_PAD_PAD):
                    pads.append(p)
                if len(pads) == 3:
                    break
            if len(pads) < 3:
                continue
            for _ in range(2000):
                i = len(objs)
                h = self.OBJ_HALF[shapes[i]]
                if (self._aabb_free(p := (rng.uniform(*X), rng.uniform(*Y)), h,
                                    pads, [ph] * 3,
                                    self.MARGIN + self.CLEAR_PAD_OBJ)
                        and self._aabb_free(p, h, objs,
                                            [self.OBJ_HALF[shapes[j]] for j in range(i)],
                                            self.MARGIN + self.CLEAR_OBJ_OBJ)):
                    objs.append(p)
                if len(objs) == 3:
                    return pads, objs
        raise RuntimeError("task2 布局采样不收敛")

    def randomize(self, env, rng):
        self.instruction_variant = int(rng.random() >= 0.5)
        self.target_obj = self.OBJECTS[rng.integers(3)][0]
        self.target_pad = f"pad{rng.integers(3)}"
        colors = list(sb.COLORS)
        perm = rng.permutation(len(colors))
        pads, objs = self._sample_layout(rng)
        # 物体 3 色 + 垫 3 色，全部不同（颜色既是外观也是指令词）
        for (name, shape), (x, y), ci in zip(self.OBJECTS, objs, perm[:3]):
            quat, hz = self.SPAWN[shape]
            env.set_obj_pose(name, sb.table_pos(x, y, hz + 0.002), quat)
            self.obj_colors[name] = colors[ci]
            rgba = sb.COLORS[colors[ci]]
            env.set_geom_rgba(f"{name}_geom", rgba)
            if shape == "t":                    # T 积木两块板都要上色
                env.set_geom_rgba(f"{name}_geom_stem", rgba)
        for i, (x, y), ci in zip(range(3), pads, perm[3:]):
            env.set_pad_pos(f"pad{i}", [x, y, sb.TABLE_TOP_Z + 0.005])
            self.pad_colors[f"pad{i}"] = colors[ci]
            env.set_geom_rgba(f"pad{i}_geom", sb.COLORS[colors[ci]])

    def instruction(self, env, rng):
        # 物体按形状唯一（各一种），指令只说类型不说颜色；垫子只能靠颜色区分，
        # 颜色保留。物体颜色仅作视觉随机化
        shape = dict((n, s) for n, s in self.OBJECTS)[self.target_obj]
        word = self.SHAPE_WORDS[shape]
        pad = self.pad_colors[self.target_pad]
        if self.instruction_variant == 0:
            return f"pick up the {word} and place it on the {pad} pad"
        return f"move the {word} onto the {pad} pad"

    def is_success(self, env) -> bool:
        obj, pad = env.obj_xpos(self.target_obj), env.obj_xpos(self.target_pad)
        ok = (np.hypot(*(obj[:2] - pad[:2])) < self.PAD_HALF
              and abs(obj[2] - sb.TABLE_TOP_Z) < 0.08
              and np.linalg.norm(env.obj_vel(self.target_obj)) < 0.05)
        return bool(ok)

    def expert_spec(self) -> dict:
        # 目标由 randomize 决定，专家在 reset 之后构造。
        # place_h 0.035（目标 z≈0.76）：太低（≤0.03）贴近奇异带，伺服残差
        # 卡在 8mm 容差线上磨几秒；太高（≥0.04）圆柱落下弹跳会滚出 3cm 容差
        return {"obj_name": self.target_obj, "pad_name": self.target_pad,
                "grasp_z": self.GRASP_Z[dict(self.OBJECTS)[self.target_obj]],
                "place_h": 0.035}

    def info(self, env) -> dict:
        return {
            "obj_colors": dict(self.obj_colors),
            "pad_colors": dict(self.pad_colors),
            "target_obj": self.target_obj,
            "target_pad": self.target_pad,
            "obj_pos": {n: env.obj_xpos(n).tolist() for n, _ in self.OBJECTS},
            "pad_pos": {f"pad{i}": env.obj_xpos(f"pad{i}").tolist() for i in range(3)},
        }
