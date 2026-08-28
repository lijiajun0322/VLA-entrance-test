"""G1 机器人改造（MjSpec 程序化建模）。

宇树 g1_29dof.xml 原样是"力矩电机 + 自由浮动基座"的人形，直接做操作任务需要
先解决两件事：站立平衡（新手过难）和没有夹爪（末端是固定橡胶手）。

本模块的做法：
1. 删除 freejoint —— 骨盆固定在世界坐标 (0, 0, 0.787)，双脚正好踩地（qpos0 站姿）
2. 删掉全部 29 个力矩电机，换成位置伺服（position actuator）：
   - 腿(12) + 腰(3) + 左臂(7) 共 22 个 -> 永远锁定在 0（保持站姿）
   - 右臂(7) -> 留给上层控制器发目标角度
3. 在右腕 (right_wrist_yaw_link) 上挂一个两指平行夹爪（palm + 左右两指 slide joint），
   两个手指各由一个位置执行器驱动，开合方向相反（左指 +y 关向内，右指 -y 关向内）
4. 在夹爪中心加 ee_site，作为后续 IK 的末端目标点
"""

from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
G1_XML = PROJECT_ROOT / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml"

# qpos0 站姿下脚底相对骨盆下垂 0.787m，骨盆抬高后双脚贴地
PELVIS_HEIGHT = 0.787

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# 左臂收纳姿态：肩前抬 + 外展 + 肘弯，左手落到髋侧 (x≈-0.03, y≈0.30, z≈0.77)，
# 完全移出 overhead_cam 视野（默认 0 位时左手悬在桌面上方会挡视线）
LEFT_ARM_JOINTS = [j.replace("right_", "left_") for j in RIGHT_ARM_JOINTS]
LEFT_ARM_POSE = [0.6, 0.35, 0.0, 1.2, 0.0, 0.0, 0.0]

# 锁定站姿的关节：腿 + 腰 + 左臂（名字都是 XXX_joint 形式）
HOLD_JOINTS = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")]
    + [j.replace("right_", "left_") for j in RIGHT_ARM_JOINTS]
)

# 位置伺服增益：(kp, kv)，按关节类别取值。
# kp 偏大是为了抑制重力下的静态下垂（kp=60 时肩部下垂 ~0.09rad，末端误差 ~6cm）
_GAINS = {
    "leg": (400.0, 18.0),
    "waist": (300.0, 12.0),
    "arm": (400.0, 14.0),
    "wrist": (150.0, 5.0),
    "finger": (250.0, 8.0),
}

# 夹爪几何：手指开度范围（关节 qpos 0=张开，0.04=闭合）。
# 全开跨距 2*(0.045-0.003)=8.4cm，能容纳 5cm 方块的对角线(7.07cm)——
# 姿态略有倾斜时也能包住物体
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.040


def _add_position_actuator(spec, name: str, joint_name: str, kp: float, kv: float,
                            ctrlrange, forcerange):
    """通用位置执行器：gainprm=[kp], biasprm=[0,-kp,-kv]（即 <position kp kv/>）"""
    spec.add_actuator(
        name=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=joint_name,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[kp] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -kp, -kv] + [0.0] * 7,
        ctrllimited=True,
        ctrlrange=list(ctrlrange),
        forcelimited=True,
        forcerange=list(forcerange),
    )


def _add_gripper(spec):
    """在右腕挂两指平行夹爪，返回 None（元素命名固定，运行时按名查 id）。"""
    wrist = spec.body("right_wrist_yaw_link")

    # 删掉右腕原装的固定橡胶手视觉件（否则会和新掌板穿模；左手的保留）
    for g in list(wrist.geoms):
        if g.meshname == "right_rubber_hand":
            spec.delete(g)

    # 夹爪斜置 60° 安装：让"夹爪竖直向下"时腕关节工作在中位区（余量 ~90%）。
    # 沿小臂安装（旧行为）时竖直接近要求腕折 ~90°，贴着关节限位工作，
    # 伺服一修正就顶死抖动；90° 安装则远-低角点臂展不够。60° 是实测折中。
    # quat = 绕 y 轴 +60°: 掌 +x -> 腕架 (cos60°, 0, -sin60°)
    a = np.deg2rad(60)
    palm = wrist.add_body(name="gripper_palm", pos=[0.03, 0, 0],
                          quat=[np.cos(a/2), 0, np.sin(a/2), 0])
    # 掌板：x 加长让它与腕链(到 x≈0.026)搭接，补上删掉橡胶手后留下的空档；
    # 颜色与机身（0.7 灰）一致，视觉上像原厂手腕的延伸
    palm.add_geom(
        name="gripper_palm_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.012, 0.036, 0.040],
        rgba=[0.72, 0.72, 0.74, 1],
        friction=[1.2, 0.005, 0.0001],
    )
    # 末端目标点：夹爪指间夹持中心（后续 IK 用）。
    # 透明 + 极小 = 不渲染（site 默认显示为小球），坐标照样可读。
    palm.add_site(name="ee_site", pos=[0.040, 0, 0],
                  size=[0.0001] * 3, rgba=[0, 0, 0, 0])

    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = palm.add_body(name=f"finger_{side}", pos=[0.008, sign * 0.045, 0])
        finger.add_joint(
            name=f"finger_{side}_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, -sign, 0],  # q 增大 = 向掌中心收拢（左指在 +y 侧，轴向 -y）
            range=[GRIPPER_OPEN, GRIPPER_CLOSED],
            damping=4.0,          # 高阻尼 = 柔和闭合，避免把物体挤射出去
            frictionloss=0.8,     # 静摩擦死区：吃掉夹持力-接触柔顺的 ±1mm 极限环
                                   # （夹持力 ~5N，0.8N 死区不影响抓取）
        )
        # 指身沿接近方向 (+x) 前伸的长条板，厚方向为 y（闭合方向）。
        # z 向加高到 36mm：夹 5cm 方块时接触高度余量足，ee 高度偏差 ±1cm 也能夹住
        finger.add_geom(
            name=f"finger_{side}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.026, 0.003, 0.018],
            pos=[0.026, sign * 0.003, 0],
            rgba=[0.72, 0.72, 0.74, 1],
            friction=[1.5, 0.005, 0.0001],
            solref=[0.004, 1.0],   # 硬接触：消除夹持-接触柔顺的极限环抖动
        )
        # 指尖一小段深色胶面（高摩擦，夹物体就靠它）
        finger.add_geom(
            name=f"finger_{side}_tip_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.008, 0.003, 0.014],
            pos=[0.056, sign * 0.003, 0],
            rgba=[0.2, 0.2, 0.22, 1],
            friction=[1.8, 0.005, 0.0001],
            solref=[0.004, 1.0],
        )

    j = spec.joint("finger_left_joint")
    _add_position_actuator(spec, "gripper_left", "finger_left_joint",
                           *_GAINS["finger"], ctrlrange=j.range,
                           forcerange=[-30, 30])
    _add_position_actuator(spec, "gripper_right", "finger_right_joint",
                           *_GAINS["finger"], ctrlrange=j.range,
                           forcerange=[-30, 30])


def build_g1_arm_spec() -> "mujoco.MjSpec":
    """加载宇树 G1 并完成上述全部改造，返回未编译的 MjSpec（场景还要往里加东西）。"""
    spec = mujoco.MjSpec.from_file(str(G1_XML))

    # 1) 固定基座：删自由关节，骨盆放到站立高度
    spec.delete(spec.joint("floating_base_joint"))
    spec.body("pelvis").pos = [0, 0, PELVIS_HEIGHT]

    # 2) 删掉原有 29 个力矩电机
    for act in list(spec.actuators):
        spec.delete(act)

    # 3) 锁定关节的位置保持执行器（腿/腰/左臂），目标永远是 0
    for name in HOLD_JOINTS:
        j = spec.joint(name)
        kind = ("leg" if any(k in name for k in ("hip", "knee", "ankle"))
                else "waist" if "waist" in name else "arm")
        _add_position_actuator(spec, f"hold_{name}", name, *_GAINS[kind],
                               ctrlrange=j.range, forcerange=[-88, 88])

    # 4) 右臂位置执行器（可控）
    for name in RIGHT_ARM_JOINTS:
        j = spec.joint(name)
        kind = "wrist" if "wrist" in name else "arm"
        _add_position_actuator(spec, f"arm_{name}", name, *_GAINS[kind],
                               ctrlrange=j.range, forcerange=[-25, 25])

    # 5) 夹爪
    _add_gripper(spec)
    return spec


def build_g1_arm_model() -> mujoco.MjModel:
    """一步到位：改造 + 编译。任务环境从 build_g1_arm_spec() 出发加场景。"""
    return build_g1_arm_spec().compile()


if __name__ == "__main__":
    model = build_g1_arm_model()
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    assert model.nq == 29 + 2, "29 个关节 + 2 个手指"
    assert model.nu == 22 + 7 + 2, "22 保持 + 7 右臂 + 2 手指"
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ee = data.site("ee_site").xpos
    print(f"ee_site 世界坐标: {[round(v, 3) for v in ee]}")
    for name in ("left_ankle_roll_link", "gripper_palm"):
        print(f"{name}: {[round(v, 3) for v in data.body(name).xpos]}")
