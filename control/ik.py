"""离线逆运动学：给定 ee_site 的目标位姿，解腰yaw+右臂 8 关节角。

方法：阻尼最小二乘(DLS)迭代 + 严格关节限位 + 多随机重启。
重启是为了跳出局部极小（如腕关节顶限位的构型）；限位内解不出时
返回误差最小的候选。解算在独立的 MjData 上进行，不污染仿真状态。
"""

import numpy as np

import mujoco

# IK 控制链关节（顺序即 8 维动作向量的顺序）
IK_JOINTS = ["waist_yaw_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
             "right_shoulder_yaw_joint", "right_elbow_joint",
             "right_wrist_roll_joint", "right_wrist_pitch_joint",
             "right_wrist_yaw_joint"]

# 初始/收尾姿态：高悬停 + 夹爪竖直朝下（与抓取姿态一致，全程无姿态过渡）。
# ee≈(0.30,-0.13,0.94)，指尖 0.90m 高于桌上物体 0.77m；腕 pitch 余量 0.83rad，
# 在 60° 斜置夹爪下所有关节工作在中位区（不贴限位）。
# 该关节解由 solve_ik 对上述悬停点求得，换安装角/场景常量后需重新求解。
HOME_POSE = [-0.411, -1.022, -0.086, 0.537, 0.489, 0.857, 0.781, -0.434]

# 期望的 ee_site 姿态：夹爪接近轴(+x)竖直向下，y 轴保持世界 +y
GRIPPER_DOWN = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).astype(float)


def _errors(data, site_id, target_pos):
    """ee_site 相对目标的位置误差 + 姿态误差（轴角向量）。"""
    site = data.site(site_id)
    perr = target_pos - site.xpos
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, (GRIPPER_DOWN @ site.xmat.reshape(3, 3).T).reshape(9))
    if q[0] < 0:
        q = -q
    ang = 2 * np.arccos(np.clip(q[0], -1, 1))
    oerr = np.zeros(3) if ang < 1e-6 else q[1:] / np.sin(ang / 2) * ang
    return perr, oerr


def solve_ik(model: mujoco.MjModel, target_pos: np.ndarray,
             seeds, n_random: int = 30, iters: int = 3000,
             tol_p: float = 0.006, tol_o: float = 0.08,
             rng: np.random.Generator | None = None):
    """解一路点的 IK（严格关节限位 + 多随机重启避免局部极小）。

    seeds: 优先尝试的初值（如当前关节角、HOME_POSE），失败再随机重启。
    返回 (q8, converged)；未收敛时返回误差最小的那个解。
    """
    rng = rng or np.random.default_rng(0)
    data = mujoco.MjData(model)                     # 独立 scratch，不动仿真状态
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in IK_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    dof = np.array([model.jnt_dofadr[j] for j in jids])
    ranges = np.array([model.jnt_range[j] for j in jids])
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def cost(q):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        perr, oerr = _errors(data, sid, target_pos)
        return np.linalg.norm(perr) + 0.1 * np.linalg.norm(oerr), perr, oerr

    best_q, best_cost = None, np.inf
    q0_list = [np.asarray(s, float) for s in seeds] + \
              [rng.uniform(ranges[:, 0], ranges[:, 1]) for _ in range(n_random)]
    for q in q0_list:
        q = np.clip(q, ranges[:, 0], ranges[:, 1])
        for _ in range(iters):
            c, perr, oerr = cost(q)
            if c < best_cost:
                best_q, best_cost = q.copy(), c
            if np.linalg.norm(perr) < tol_p and np.linalg.norm(oerr) < tol_o:
                return q.copy(), True
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J = np.vstack([jacp[:, dof], 0.5 * jacr[:, dof]])
            err = np.concatenate([perr, 0.5 * oerr])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), ranges[:, 0], ranges[:, 1])
    return best_q, False
