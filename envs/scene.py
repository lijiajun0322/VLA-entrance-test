"""通用场景搭建：地板、桌子、灯光、相机、任务物体。

坐标系约定：G1 站在原点面向 +x，桌面高度 TABLE_TOP_Z，桌心在 (TABLE_CENTER_X, 0)。
右臂肩部约在 y=-0.15 一侧，物体工作区 WORKSPACE 在桌面上、右臂可达范围内。
"""

import numpy as np

import mujoco

# ---- 场景常量（单位: 米） ----
TABLE_TOP_Z = 0.72          # 桌面高度
TABLE_CENTER_X = 0.34       # 桌子中心 x（右臂+腰yaw 实测可达范围内）
TABLE_CENTER_Y = -0.13
TABLE_SIZE = (0.16, 0.26)   # 桌面半尺寸 (x, y)
# 右臂工作区（x, y 范围）——经关节限位约束的 8 自由度 IK 网格实测可达
# （夹爪竖直朝下、多随机重启全局解；比无约束可达区小一圈，保证每点都能解）
WORKSPACE = dict(
    x=(0.26, 0.38),
    y=(-0.26, 0.00),
)

# 任务物体颜色池（LIBERO 风格的高饱和纯色，指令文本与之绑定）
COLORS = {
    "red":    [0.8, 0.1, 0.1, 1],
    "green":  [0.1, 0.7, 0.15, 1],
    "blue":   [0.15, 0.25, 0.85, 1],
    "yellow": [0.9, 0.8, 0.1, 1],
    "purple": [0.6, 0.15, 0.75, 1],
    "orange": [0.95, 0.5, 0.1, 1],
}


def quat_look_at(cam_pos, look_at) -> list:
    """生成让相机从 cam_pos 看向 look_at 的 xyaxes（相机 -z 朝向目标）。"""
    pos, look = np.asarray(cam_pos, float), np.asarray(look_at, float)
    z = pos - look
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, z)) > 0.99:      # 近乎垂直俯视时换一个 up
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return list(x) + list(y)


def add_world(spec: "mujoco.MjSpec"):
    """地板 + 桌子 + 灯光 + 第三人称相机，加到 worldbody 上。"""
    wb = spec.worldbody

    # 地板
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05], rgba=[0.35, 0.38, 0.42, 1])
    # 桌子：一整块实心箱体（省去桌腿，碰撞更稳）
    h = TABLE_TOP_Z / 2
    wb.add_body(name="table", pos=[TABLE_CENTER_X, TABLE_CENTER_Y, h]).add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_SIZE[0], TABLE_SIZE[1], h], rgba=[0.55, 0.45, 0.32, 1],
    )

    wb.add_light(pos=[0.5, 0, 2.2], dir=[0, 0, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(pos=[1.5, 1.0, 1.8], dir=[-0.5, -0.4, -0.8],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    # 第三人称俯视相机：从机器人后上方看向桌心
    cam_pos = [0.05, 0.62, 1.45]
    look_at = [TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z]
    wb.add_camera(name="overhead_cam", pos=cam_pos,
                  xyaxes=quat_look_at(cam_pos, look_at), fovy=45)

    # 腕部相机：装在夹爪掌上，朝夹持方向 (+x) 看
    palm = spec.body("gripper_palm")
    palm.add_camera(name="wrist_cam", pos=[0.01, 0.06, 0.05],
                    xyaxes=[0, 1, 0, 0, 0, 1], fovy=75)


def add_object(spec: "mujoco.MjSpec", name: str, shape: str,
               size: float, rgba=(0.5, 0.5, 0.5, 1), pos=(0.45, 0, TABLE_TOP_Z + 0.05)):
    """加一个带 freejoint 的任务物体。shape: cube / cylinder / ball。size 为半边长/半径。
    rgba 只是初始色，任务每回合会随机覆盖。"""
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    gtype = {"cube": mujoco.mjtGeom.mjGEOM_BOX,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
             "ball": mujoco.mjtGeom.mjGEOM_SPHERE}[shape]
    body.add_geom(name=f"{name}_geom", type=gtype, size=[size, size, size],
                  rgba=list(rgba), mass=0.1, friction=[1.0, 0.005, 0.0001])
    return body


def add_pad(spec: "mujoco.MjSpec", name: str, rgba,
            pos=(0.45, -0.15, TABLE_TOP_Z), half=0.06):
    """目标垫：贴在桌面上的薄方块。mocap body，reset 时直接设位置；
    无碰撞（contype=0），纯视觉标记 + 成功判定锚点。"""
    body = spec.worldbody.add_body(name=name, pos=list(pos), mocap=True)
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half, half, 0.004], rgba=list(rgba),
                  contype=0, conaffinity=0, mass=0)   # 无碰撞，纯标记
    body.add_site(name=f"{name}_site", size=[0.0001] * 3,
                  rgba=[0, 0, 0, 0])  # 透明，不渲染
    return body


def table_pos(x: float, y: float, clearance: float = 0.03) -> list:
    """工作区坐标 -> 桌面上方悬浮点（reset 时物体从这里落下更稳）。"""
    return [x, y, TABLE_TOP_Z + clearance]


def sample_workspace(rng: np.random.Generator, n: int, min_dist: float = 0.14) -> list:
    """在工作区里采 n 个互不重叠的 (x, y)。拒绝采样，采不动时逐步放宽距离。"""
    for dist in (min_dist, min_dist * 0.85, min_dist * 0.7, min_dist * 0.55):
        pts, tries = [], 0
        while len(pts) < n and tries < 2000:
            tries += 1
            p = (rng.uniform(*WORKSPACE["x"]), rng.uniform(*WORKSPACE["y"]))
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > dist for q in pts):
                pts.append(p)
        if len(pts) == n:
            return pts
    raise RuntimeError(f"workspace too small for {n} points")
