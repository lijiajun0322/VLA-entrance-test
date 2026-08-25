"""脚本专家（无模型控制）：pick-and-place 的通用状态机。

不经过任何学习/策略，直接用仿真里的上帝视角信息生成动作：
    HOME -> 物体上方 -> 下降 -> 夹取 -> 抬升 -> 目标上方 -> 下降 -> 松爪 -> 收尾

控制方式：每个路点先离线解 IK（control.ik.solve_ik，多随机重启避局部极小），
再把关节目标从当前值平滑插值下发给位置伺服；插值到位后用 CartesianServo
闭环修掉模型-机构残差。本模块只做"决策"，控制细节全部委托给 control 包。
"""

import numpy as np

import mujoco

from envs import scene as sb
from .ik import IK_JOINTS, HOME_POSE, solve_ik
from .servo import CartesianServo

TOL_POS = 0.015          # 路点到达判定阈值 (m)
SETTLE_STEPS = 60        # 到位后保持步数（等伺服与物体稳定）
PHASE_TIMEOUT = 4000     # 单阶段超时步数 (0.002s * 4000 = 8s)
WAIT_GRASP = 400         # 夹爪闭合等待 (~0.8s)
WAIT_RELEASE = 300       # 松爪等待 (~0.6s)
INTERP_SPEED = 0.004     # 插值速度: 每步最多走的关节弧长 (rad)

APPROACH_H = 0.12        # 物体/目标上方悬停高度
LIFT_H = 0.15            # 夹起/放下后抬升高度
GRASP_Z_OFF = 0.012      # 抓取时 ee_site 相对物体中心的 z 补偿（防指尖戳桌面）


class PickPlaceExpert:
    """通用抓放专家。构造参数 spec 来自 Task.expert_spec()：
    {"obj_name": 抓取物体, "pad_name": 放置目标, "place_h": 放置离目标面高度}"""

    PHASES = ["home", "above_obj", "descend", "grasp", "lift",
              "above_pad", "place", "release", "retract", "done"]

    def __init__(self, env, spec: dict):
        self.env = env
        self.obj_name = spec["obj_name"]
        self.pad_name = spec["pad_name"]
        self.place_h = spec.get("place_h", 0.045)
        self.servo = CartesianServo(env.model, env.data)
        self.phase_i = 0
        self.phase_step = 0
        self.stable = 0
        self._q_from = self._q_measured()
        self._q_star = np.array(HOME_POSE, float)
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
        if ph == "home":
            return np.array([sb.TABLE_CENTER_X, sb.TABLE_CENTER_Y,
                             sb.TABLE_TOP_Z + APPROACH_H])
        if ph == "above_obj":
            return obj + [0, 0, APPROACH_H]
        if ph == "descend":
            return obj + [0, 0, GRASP_Z_OFF]
        if ph == "lift":
            return obj + [0, 0, LIFT_H]
        if ph == "above_pad":
            return pad + [0, 0, APPROACH_H]
        if ph == "place":
            return pad + [0, 0, self.place_h]
        if ph in ("release", "retract"):
            return pad + [0, 0, APPROACH_H + LIFT_H]
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
        self._target = self._waypoint()       # 含 home：全部用 IK 解（HOME_POSE 只作初值）
        q_star, ok = solve_ik(self.env.model, self._target,
                              seeds=[self._q_measured(), HOME_POSE],
                              rng=self._rng)
        self._q_from = self._q_measured()
        self._q_star = q_star
        dist = float(np.linalg.norm(q_star - self._q_from))
        self._interp_len = int(np.clip(dist / INTERP_SPEED, 120, 800))

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

        # 关节空间线性插值下发位置目标；插值到位后用局部伺服闭环修正
        # （腿/腰伺服的静态下垂让真实 FK 比 IK 模型低 ~2cm，需要实测状态上的微调）
        s = min(1.0, self.phase_step / self._interp_len)
        if s < 1.0:
            env.set_arm8(self._q_from + s * (self._q_star - self._q_from))
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
