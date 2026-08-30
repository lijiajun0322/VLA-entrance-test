"""横向滑动门 scripted expert。"""

import numpy as np
import mujoco

from .ik import IK_JOINTS, HOME_POSE, solve_ik
from .servo import CartesianServo

TOL_POS = 0.010
SETTLE_STEPS = 15
WAIT_GRASP = 250
WAIT_RELEASE = 150
PHASE_TIMEOUT = 2500
INTERP_SPEED = 0.003
APPROACH_H = 0.10
SLIDE_SPEED = 0.05  # m/s，slide 阶段按笛卡尔直线匀速推进


def _trapezoid_position(u: float, blend: float = 0.10) -> float:
    """短加减速、长匀速的归一化位置曲线。"""
    vmax = 1.0 / (1.0 - blend)
    if u < blend:
        return vmax * u * u / (2.0 * blend)
    if u <= 1.0 - blend:
        return vmax * (u - blend / 2.0)
    remaining = 1.0 - u
    return 1.0 - vmax * remaining * remaining / (2.0 * blend)


class OpenSlidingDoorExpert:
    PHASES = ["above_handle", "descend", "grasp", "slide",
              "release", "retract", "done"]

    def __init__(self, env, spec: dict):
        self.env, self.spec = env, spec
        self.joint_name = spec["joint_name"]
        self.handle_site = spec["handle_site"]
        self.slide_axis = np.asarray(spec["slide_axis"], dtype=float)
        self.open_distance = float(spec["open_distance"])
        self.grasp_z_offset = float(spec.get("grasp_z_offset", 0.0))
        self.success_qpos = float(spec["success_qpos"])
        self.servo = CartesianServo(env.model, env.data)
        self.phase_i = self.phase_step = self.stable = 0
        self._q_star = self._q_measured()
        self._q_from = self._q_star.copy()
        self._interp_len = 1
        self._target = None
        self._slide_start = None
        self._enter_phase()

    def _q_measured(self):
        m, d = self.env.model, self.env.data
        return np.array([d.qpos[m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in IK_JOINTS])

    def _phase(self):
        return self.PHASES[self.phase_i]

    @property
    def done(self):
        return self._phase() == "done"

    def _gripper_cmd(self):
        return 1.0 if self._phase() in ("grasp", "slide") else 0.0

    def _enter_phase(self):
        self.phase_step = self.stable = 0
        ph = self._phase()
        if ph == "done":
            return
        if ph in ("grasp", "release"):
            self._target = None
            return
        handle = self.env.site_xpos(self.handle_site)
        if ph == "above_handle":
            self._target = handle + [0, 0, APPROACH_H]
        elif ph == "descend":
            # ee_site 停在把手中心会让掌板撞上门板；夹住把手上段即可可靠横拉。
            self._target = handle + [0, 0, self.grasp_z_offset]
        elif ph == "slide":
            # 从实测 EE 位姿开始生成直线轨迹，不用关节空间端点插值。
            self._slide_start = self.env.ee_xpos()
            self._target = self._slide_start + self.slide_axis * self.open_distance
        elif ph == "retract":
            self._target = self.env.ee_xpos() + [0, 0, APPROACH_H]
        self._q_from = self._q_measured()
        self._q_star, _ = solve_ik(
            self.env.model, self._target,
            seeds=[self._q_from, HOME_POSE], n_random=20, iters=2000,
        )
        if ph == "slide":
            dt = self.env.model.opt.timestep
            self._interp_len = max(1, int(self.open_distance / (SLIDE_SPEED * dt)))
        else:
            dist = float(np.linalg.norm(self._q_star - self._q_from))
            self._interp_len = int(np.clip(dist / INTERP_SPEED, 100, 900))

    def _advance(self):
        self.phase_i += 1
        self._enter_phase()

    def step(self):
        ph = self._phase()
        self.phase_step += 1
        if ph == "done":
            self.env.step()
            return ph
        if ph in ("grasp", "release"):
            self.env.set_gripper(self._gripper_cmd())
            self.env.step()
            wait = WAIT_GRASP if ph == "grasp" else WAIT_RELEASE
            if self.phase_step > wait:
                self._advance()
            return ph
        u = min(1.0, self.phase_step / self._interp_len)
        if ph == "slide":
            # 笛卡尔空间短加减速 + 长匀速，确保夹爪沿滑轨方向直线拉动。
            s = _trapezoid_position(u)
            target = self._slide_start + self.slide_axis * (self.open_distance * s)
            self.env.set_arm8(self.servo.step(target, dq_max=0.015))
        else:
            # 接近和撤离仍用柔和的 quintic 起停。
            s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
            if u < 1.0:
                self.env.set_arm8(self._q_from + s * (self._q_star - self._q_from))
            else:
                self.env.set_arm8(self.servo.step(self._target))
        self.env.set_gripper(self._gripper_cmd())
        self.env.step()
        reached = (self.env.joint_qpos(self.joint_name) >= self.success_qpos
                   if ph == "slide" else
                   np.linalg.norm(self._target - self.env.ee_xpos()) < TOL_POS)
        self.stable = self.stable + 1 if u >= 1.0 and reached else 0
        if self.stable > SETTLE_STEPS or self.phase_step > PHASE_TIMEOUT:
            self._advance()
        return ph

    def run(self, max_steps=15000):
        for _ in range(max_steps):
            self.step()
            if self.done:
                break
        return self._phase()
