"""脚本专家（无模型控制）：pick-and-place 的通用状态机。

不经过任何学习/策略，直接用仿真里的上帝视角信息生成动作：
    HOME -> 物体上方 -> 下降 -> 夹取 -> 抬升 -> 目标上方 -> 下降 -> 松爪 -> 收尾

控制方式：每个路点先离线解 IK（control.ik.solve_ik，多随机重启避局部极小），
再把关节目标从当前值平滑插值下发给位置伺服；插值到位后用 CartesianServo
闭环修掉模型-机构残差。本模块只做"决策"，控制细节全部委托给 control 包。
"""

import numpy as np

import mujoco

from .ik import IK_JOINTS, HOME_POSE, solve_ik
from .servo import CartesianServo

TOL_POS = 0.015          # 路点到达判定阈值 (m)
SETTLE_STEPS = 10        # 到位后保持步数（等伺服与物体稳定）
PHASE_TIMEOUT = 4000     # 单阶段超时步数 (0.002s * 4000 = 8s)
WAIT_GRASP = 200         # 夹爪闭合等待（实测闭合 20ms，其余为挤压稳定余量）
WAIT_RELEASE = 150       # 松爪等待（实测方块落稳 146ms）
INTERP_SPEED = 0.004     # 插值速度: 每步最多走的关节弧长 (rad)

APPROACH_H = 0.12        # 物体/目标上方悬停高度
LIFT_H = 0.15            # 夹起/放下后抬升高度
GRASP_Z_OFF = 0.022      # 抓取时 ee_site 相对物体中心的 z 补偿。60° 斜置安装后
                          # 腕部会下探，0.006 时腕 mesh 压到方块顶(顶住手臂 → 8s 超时)；
                          # 0.022 让腕底离方块顶 ~1.5cm，36mm 指板仍可靠咬住上半部


class PickPlaceExpert:
    """通用抓放专家。构造参数 spec 来自 Task.expert_spec()：
    {"obj_name": 抓取物体, "pad_name": 放置目标, "place_h": 放置离目标面高度}"""

    PHASES = ["above_obj", "descend", "grasp", "lift",
              "above_pad", "place", "release", "retract", "done"]

    def __init__(self, env, spec: dict):
        self.env = env
        self.obj_name = spec["obj_name"]
        self.pad_name = spec["pad_name"]
        self.place_h = spec.get("place_h", 0.045)
        self.grasp_z = spec.get("grasp_z", GRASP_Z_OFF)   # 按物体半高让指板咬在中部
        self.servo = CartesianServo(env.model, env.data)
        self.phase_i = 0
        self.phase_step = 0
        self.stable = 0
        self._q_from = self._q_measured()
        self._q_star = np.array(HOME_POSE, float)
        self._q_hover = np.array(HOME_POSE, float)   # above_obj 悬停构型(lift 原路返回)
        self._interp_len = 1
        self._target = None
        self._rng = np.random.default_rng(0)
        self._enter_phase()          # 初始化第一阶段（step()/run() 可直接调用）

    # ---- 基础 ----

    def _q_measured(self) -> np.ndarray:
        m, d = self.env.model, self.env.data
        return np.array([d.qpos[m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in IK_JOINTS])

    def _phase(self) -> str:
        return self.PHASES[self.phase_i]

    @property
    def done(self) -> bool:
        return self._phase() == "done"

    # ---- 路点目标 ----

    def _waypoint(self):
        env = self.env
        obj = env.obj_xpos(self.obj_name)
        pad = env.obj_xpos(self.pad_name)
        ph = self._phase()
        if ph == "above_obj":
            return obj + [0, 0, APPROACH_H]
        if ph == "descend":
            return obj + [0, 0, self.grasp_z]
        if ph == "lift":
            return obj + [0, 0, LIFT_H]
        if ph == "above_pad":
            return pad + [0, 0, APPROACH_H]
        if ph == "place":
            return pad + [0, 0, self.place_h]
        if ph == "retract":
            return pad + [0, 0, APPROACH_H]   # 收尾抬到常规悬停高度即可
        return None

    def _gripper_cmd(self) -> float:
        ph = self._phase()
        return 1.0 if ph in ("grasp", "lift", "above_pad", "place") else 0.0

    def _enter_phase(self):
        self.phase_step, self.stable = 0, 0
        ph = self._phase()
        if ph == "done":
            return
        if ph in ("grasp", "release"):        # 等待阶段：保持当前关节目标
            self._target = None
            return
        self._target = self._waypoint()       # 全部用 IK 解（HOME_POSE 只作初值）
        self._q_from = self._q_measured()

        if ph == "lift":
            # lift 不重新解 IK：直接原路返回 above_obj 的悬停构型。
            # 抓取构型(腰~0.8)与悬停构型(腰~0)都是已知好解，上升沿下降来路
            # 逆转，腰平滑转回；独立解 lift 目标会跳到第三枝(腰+2.0)，导致
            # 后续 above_pad 大弧扫摆（解流形在此区域呈螺旋，实测无连续捷径）。
            self._q_path = [self._q_from, self._q_hover]
            self._q_star = self._q_hover
            dist = float(np.linalg.norm(self._q_hover - self._q_from))
            self._interp_len = int(np.clip(dist / INTERP_SPEED, 120, 900))
        elif ph in ("descend", "place"):
            # 竖直段用小步链：沿直线路径每 2cm 一段、每段 IK 只用前一段解作种子，
            # 连续跟踪解流形。竖直接近在抓取高度必须扭腰（几何性质），一步大跨
            # 的局部解会跳到远处枝（如腰 +0.7→+2.0），插值轨迹扫大弧；
            # 2cm 小步下腰在下降/上升中平滑转移（纯 IK 已验证）。
            ee0 = self.env.ee_xpos()
            qs = [self._q_from]
            n = max(2, int(np.linalg.norm(self._target - ee0) / 0.02))
            for i in range(1, n + 1):
                tgt = ee0 + (i / n) * (self._target - ee0)
                q, _ = solve_ik(self.env.model, tgt, seeds=[qs[-1]],
                                n_random=0, rng=self._rng)
                qs.append(q)
            self._q_path = qs
            self._q_star = qs[-1]
            total = sum(float(np.linalg.norm(qs[i + 1] - qs[i]))
                        for i in range(len(qs) - 1))
            self._interp_len = int(np.clip(total / INTERP_SPEED, 120, 900))
        else:
            q_star, ok = solve_ik(self.env.model, self._target,
                                  seeds=[self._q_measured(), HOME_POSE],
                                  rng=self._rng)
            if ph == "above_obj":
                self._q_hover = q_star        # 存给 lift 原路返回用
            self._q_path = [self._q_from, q_star]
            self._q_star = q_star
            dist = float(np.linalg.norm(q_star - self._q_from))
            self._interp_len = int(np.clip(dist / INTERP_SPEED, 120, 800))

    def _path_interp(self, s: float) -> np.ndarray:
        """按路径长度比例在 q_path 折线上取关节插值点。"""
        qs = self._q_path
        seg = [float(np.linalg.norm(qs[i + 1] - qs[i])) for i in range(len(qs) - 1)]
        total = sum(seg) or 1.0
        d, acc = s * total, 0.0
        for i, L in enumerate(seg):
            if acc + L >= d or i == len(seg) - 1:
                t = (d - acc) / L if L > 0 else 0.0
                return qs[i] + t * (qs[i + 1] - qs[i])
            acc += L
        return qs[-1]

    def _advance(self):
        self.phase_i += 1
        self._enter_phase()

    # ---- 主循环 ----

    def step(self) -> str:
        """推进一步，返回当前阶段名。"""
        env, ph = self.env, self._phase()
        self.phase_step += 1

        if ph == "done":
            env.step()
            return ph

        if ph in ("grasp", "release"):        # 定长等待：只动夹爪
            env.set_gripper(self._gripper_cmd())
            env.step()
            if self.phase_step > (WAIT_GRASP if ph == "grasp" else WAIT_RELEASE):
                self._advance()
            return ph

        # 沿关节折线插值下发位置目标（竖直段为小步链）；到位后伺服闭环修正
        s = min(1.0, self.phase_step / self._interp_len)
        if s < 1.0:
            env.set_arm8(self._path_interp(s))
        else:
            env.set_arm8(self.servo.step(self._target))
        env.set_gripper(self._gripper_cmd())
        env.step()

        perr = np.linalg.norm(self._target - env.ee_xpos())
        # 抓取下降段要求更准（1.5cm 容差下手指容易蹭掉半宽 2.5cm 的方块）
        tol = 0.008 if ph in ("descend", "place") else TOL_POS
        self.stable = self.stable + 1 if (s >= 1 and perr < tol) else 0
        if self.stable > SETTLE_STEPS:
            self._advance()
        elif self.phase_step > PHASE_TIMEOUT:
            self._advance()                   # 超时兜底前进
        return ph

    def run(self, max_steps: int = 20000) -> str:
        """一口气跑完整条流程（可视化时逐步 step 以便同步渲染）。"""
        for _ in range(max_steps):
            self.step()
            if self.done:
                break
        return self._phase()
