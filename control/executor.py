"""动作序列执行器：在仿真里执行 (ee_xyz + gripper) 动作序列。

这是"策略输出 → 机器人动作"的部署链路（任务 4 直接复用）：
  每帧动作 -> solve_ik(竖直朝下约束) -> 前 15 步关节插值 -> 后 10 步伺服补残差
外加两个鲁棒性设计（回放验证踩坑的产出）：
  - 下降沿对中悬停：z 首次降到 0.80 以下时，先在当前高度对准 xy 再垂直下降
  - 静止段冻结：相邻帧目标几乎不变时保持关节目标不动（不折腾）
"""

import numpy as np

import mujoco

from .ik import solve_ik, IK_JOINTS, HOME_POSE
from .servo import CartesianServo
from mujoco_datasets.spec import ACTION_SPEC

SERVO_TAIL = 10       # 每帧最后 N 步用笛卡尔伺服补残差
HOVER_Z = 0.80        # 下降沿检测阈值（工作段高度以下 = 接近物体）


class OnlineActionExecutor:
    """在线策略逐帧执行器：每次 step 一个 absolute ee_xyz + gripper action。"""

    def __init__(self, env):
        self.env = env
        self.model, self.data = env.model, env.data
        self.servo = CartesianServo(self.model, self.data)
        self.qadr = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in IK_JOINTS])
        self.n_steps = 0
        self.ik_failures = 0

    def reset(self):
        self.n_steps = 0
        self.ik_failures = 0

    def step(self, action4: np.ndarray) -> bool:
        """执行一个 20Hz action frame；返回本帧 IK 是否收敛。"""
        action4 = np.asarray(action4, dtype=float)
        if action4.shape != (4,) or not np.isfinite(action4).all():
            raise ValueError(f"invalid action4: {action4}")
        target = action4[:3]
        q_meas = self.data.qpos[self.qadr].copy()
        q_star, converged = solve_ik(
            self.model, target, seeds=[q_meas, HOME_POSE], n_random=5, iters=800
        )
        if not converged:
            self.ik_failures += 1

        every = ACTION_SPEC.sample_every
        for k in range(every):
            if k < every - SERVO_TAIL:
                s = (k + 1) / (every - SERVO_TAIL)
                self.env.set_arm8(q_meas + s * (q_star - q_meas))
            else:
                self.env.set_arm8(self.servo.step(target))
            self.env.set_gripper(float(np.clip(action4[3], 0.0, 1.0)))
            self.env.step()
        self.n_steps += 1
        return converged


def execute_actions(env, actions, on_frame=None) -> bool:
    """在 env 里执行完整动作序列（需先 reset 到对应初始状态）。

    on_frame(t): 每执行完一帧（20Hz）回调一次，用于录视频/记录。
    返回 is_success()。
    """
    m, d = env.model, env.data
    servo = CartesianServo(m, d)
    qadr = np.array([m.jnt_qposadr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in IK_JOINTS])
    every = ACTION_SPEC.sample_every

    def goto(target, steps=300, tol=0.006):
        """对中悬停：把 ee 收敛到 target 再继续。"""
        for _ in range(steps):
            q_star, _ = solve_ik(m, target, seeds=[d.qpos[qadr].copy(), HOME_POSE],
                                 n_random=3, iters=600)
            env.set_arm8(q_star)
            env.step()
            if np.linalg.norm(target - env.ee_xpos()) < tol:
                break

    prev_z = actions[0, 2]
    for t in range(len(actions)):
        # 下降沿（抓取/放置都有）：先在当前高度对准 xy，再垂直下降
        if actions[t, 2] < HOVER_Z and prev_z >= HOVER_Z:
            hover = actions[t, :3].copy()
            hover[2] = prev_z
            goto(hover)
        prev_z = actions[t, 2]

        q_meas = d.qpos[qadr].copy()
        q_star, _ = solve_ik(m, actions[t, :3],
                             seeds=[q_meas, HOME_POSE], n_random=5, iters=800)
        for k in range(every):
            if k < every - SERVO_TAIL:
                s = (k + 1) / (every - SERVO_TAIL)
                env.set_arm8(q_meas + s * (q_star - q_meas))
            else:
                env.set_arm8(servo.step(actions[t, :3]))
            env.set_gripper(float(actions[t, 3]))
            env.step()
        if on_frame is not None:
            on_frame(t)
    return env.is_success()
