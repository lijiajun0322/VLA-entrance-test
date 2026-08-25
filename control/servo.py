"""笛卡尔闭环伺服：让 ee_site 平滑跟踪一个目标位置。

每步在仿真实测状态上算 ee 的位置/姿态残差，经雅可比伪逆转成
关节增量，输出下一拍的位置伺服目标。用途有三处：
1. 脚本专家到位后的残差修正（腿/腰伺服静态下垂 ~2cm）
2. 数据管线回放检查（跟踪录下的 ee 序列，验证标签可执行）
3. 任务 4 策略推理（VLA 输出 ee 目标 -> 本伺服驱动仿真）
"""

import numpy as np

import mujoco

from .ik import IK_JOINTS, GRIPPER_DOWN


class CartesianServo:
    """用法: env.set_arm8(servo.step(target_pos))"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.dof_ids = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in IK_JOINTS])
        self.qadr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in IK_JOINTS])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def errors(self, target_pos: np.ndarray):
        """位置误差与姿态误差（夹爪保持竖直朝下）。"""
        site = self.data.site(self.site_id)
        perr = target_pos - site.xpos
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, (GRIPPER_DOWN @ site.xmat.reshape(3, 3).T).reshape(9))
        if q[0] < 0:
            q = -q
        ang = 2 * np.arccos(np.clip(q[0], -1, 1))
        oerr = np.zeros(3) if ang < 1e-6 else q[1:] / np.sin(ang / 2) * ang
        return perr, oerr

    def step(self, target_pos: np.ndarray, kori: float = 0.3,
             dq_max: float = 0.03) -> np.ndarray:
        """返回这拍应下发的 8 关节位置目标（腰yaw + 右臂）。"""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        perr, oerr = self.errors(target_pos)
        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site_id)
        J = np.vstack([self._jacp[:, self.dof_ids], kori * self._jacr[:, self.dof_ids]])
        err = np.concatenate([perr, kori * oerr])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        n = np.linalg.norm(dq)
        if n > dq_max:
            dq *= dq_max / n
        return d.qpos[self.qadr] + dq
