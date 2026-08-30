"""G1 桌面操作环境：机器人 + 场景 + 随机化 reset + 观测 + 成功判定。

职责边界：本模块只管"物理世界与观测"。怎么动（IK/伺服/专家）在 control/，
怎么录（数据管线）在 data/。任务内容由 envs/tasks/ 的 Task 子类定义。

设计对齐 LIBERO：每次 reset 一个随机 variant，指令文本与场景一致；
get_obs() 返回的观测**不含任何上帝视角信息**（物体真实坐标只进
task.info() 元数据），防止策略学到作弊输入。
"""

from typing import Optional

import numpy as np

import mujoco

from .robot import build_g1_arm_spec, RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS, LEFT_ARM_POSE
from . import scene as sb
from .tasks.base import Task
from control.ik import IK_JOINTS, HOME_POSE
from mujoco_datasets.spec import OBS_SPEC, camera_map_for_task

SETTLE_STEPS = 300      # reset 后让物体落稳的步数 (0.002s * 300 = 0.6s)


class G1TaskEnv:
    """G1 桌面操作环境。"""

    def __init__(self, task: Task):
        self.task = task
        spec = build_g1_arm_spec()
        sb.add_world(spec)
        task.setup(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # 执行器 id（顺序 = RIGHT_ARM_JOINTS）
        self.arm_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_{j}")
             for j in RIGHT_ARM_JOINTS])
        # IK 控制链执行器 id（腰 yaw + 右臂，顺序同 control.ik.IK_JOINTS）
        self.arm8_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                               f"hold_{j}" if j == "waist_yaw_joint" else f"arm_{j}")
             for j in IK_JOINTS])
        # 夹爪两指执行器（按名查，不依赖 id 相邻）
        self.grip_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
             for n in ("gripper_left", "gripper_right")])
        # 左臂收纳姿态的 qpos 地址与保持执行器 id（reset 时摆出画面外）
        self._left_qadr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in LEFT_ARM_JOINTS])
        self._left_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hold_{j}")
             for j in LEFT_ARM_JOINTS])
        # 本体感知索引：完整 IK 控制链（腰 yaw + 右臂）+ 手指关节。
        # 顺序与策略控制链保持一致，避免相同右臂状态对应多个不可观测的 EE 位姿。
        proprio_joints = IK_JOINTS + ["finger_left_joint", "finger_right_joint"]
        self.proprio_adr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in proprio_joints])
        # Task 1/2 为 overhead+wrist；Task 3 只保留 door_cam 主视角。
        self.obs_camera_map = camera_map_for_task(task.name)
        self.cam_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, n)
                        for n in self.obs_camera_map.values()}
        self._renderer = mujoco.Renderer(self.model, OBS_SPEC.img_size, OBS_SPEC.img_size)
        # 重力补偿作用的 dof：所有非 freejoint（机器人铰链/滑动关节）。
        # 物体的 freejoint 绝不能补——否则物体失去重力飘起来。
        self._robot_dofs = np.array(
            [i for i in range(self.model.nv)
             if self.model.jnt_type[self.model.dof_jntid[i]] != mujoco.mjtJoint.mjJNT_FREE])
        self.np_rng = np.random.default_rng()
        self.instruction = ""
        self.reset()

    # ---------------- 随机化与回合管理 ----------------

    def reset(self, seed: Optional[int] = None) -> dict:
        """重置一回合。seed=None 时续用内部随机流（每回合都不同）。"""
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0                      # 站姿 + 夹爪全开
        # 手臂摆到 HOME（前抬悬在桌面上方），避免自然下垂时手指戳进桌面
        for i, j in enumerate(("waist_yaw_joint",) + tuple(RIGHT_ARM_JOINTS)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr] = HOME_POSE[i]
        self.data.ctrl[self.arm8_act_ids] = HOME_POSE
        # 左臂收到髋侧收纳位，避免悬在桌面上方挡 overhead_cam 视线
        self.data.qpos[self._left_qadr] = LEFT_ARM_POSE
        self.data.ctrl[self._left_act_ids] = LEFT_ARM_POSE
        self.task.randomize(self, self.np_rng)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):              # 让物体落稳
            mujoco.mj_step(self.model, self.data)
        self.instruction = self.task.instruction(self, self.np_rng)
        return self.get_obs()

    # ---------------- 控制接口（脚本专家 / 策略推理用） ----------------

    def set_arm(self, qpos7):
        """右臂 7 关节目标位置。"""
        self.data.ctrl[self.arm_act_ids] = qpos7

    def set_arm8(self, qpos8):
        """腰yaw + 右臂 7 关节位置目标（IK 控制链，顺序同 control.ik.IK_JOINTS）。"""
        self.data.ctrl[self.arm8_act_ids] = qpos8

    def set_gripper(self, close01: float):
        """夹爪开合: 0=全开, 1=全闭。"""
        from .robot import GRIPPER_CLOSED
        self.data.ctrl[self.grip_act_ids] = GRIPPER_CLOSED * close01

    def step(self, n: int = 1):
        """推进一步仿真（带重力补偿：机器人的 qfrc_bias 前馈抵消）。

        没有补偿时位置伺服靠稳态误差扛重力（kp·Δq = τ_重力），关节永远垂
        ~1-2cm 且逐级累积；补偿后伺服只负责运动跟踪，静态下垂趋近零。
        qfrc_bias 是上一步 forward 的值（滞后一步，500Hz 下可忽略）。
        """
        for _ in range(n):
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self._robot_dofs] = \
                self.data.qfrc_bias[self._robot_dofs]
            mujoco.mj_step(self.model, self.data)

    # ---------------- 上帝视角信息（专家决策 / 成功判定 / 元数据用，绝不进观测） ----------------

    def obj_xpos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def obj_vel(self, name: str) -> np.ndarray:
        bid = self.data.body(name).id
        return self.data.cvel[bid][3:6].copy()   # 线速度（世界系）

    def site_xpos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def ee_xpos(self) -> np.ndarray:
        return self.data.site("ee_site").xpos.copy()

    def set_obj_pose(self, name: str, pos, quat=(1, 0, 0, 0)):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = quat

    def set_pad_pos(self, name: str, pos):
        """目标垫（mocap body）直接设位置。"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.data.mocap_pos[self.model.body_mocapid[bid]] = pos

    def joint_qpos(self, name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    def joint_qvel(self, name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qvel[self.model.jnt_dofadr[jid]])

    def set_joint_qpos(self, name: str, value: float):
        """任务 reset 专用：设置标量 hinge/slide joint 的初始位置。"""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self.data.qpos[self.model.jnt_qposadr[jid]] = value

    def set_body_base_pos(self, name: str, pos):
        """任务 reset 专用：设置直接挂在 worldbody 下的模型基座位置。"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.model.body_pos[bid] = pos

    def set_geom_rgba(self, geom_name: str, rgba):
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        self.model.geom_rgba[gid] = rgba

    # ---------------- 观测 ----------------

    def _render(self, cam: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self.cam_ids[cam])
        return self._renderer.render().copy()

    def get_obs(self) -> dict:
        """VLA 观测；图像字段由任务的 obs_camera_map 决定。"""
        obs = {
            "proprio": self.data.qpos[self.proprio_adr].copy(),
            "instruction": self.instruction,
        }
        for image_key, camera_name in self.obs_camera_map.items():
            obs[image_key] = self._render(camera_name)
        return obs

    def is_success(self) -> bool:
        return self.task.is_success(self)

    def task_info(self) -> dict:
        """上帝视角元数据（数据录制用，不进模型）。"""
        return self.task.info(self)
