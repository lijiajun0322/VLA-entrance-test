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
# 右臂工作区（x, y 范围）——z=0.764（抓取/放置高度，最低最紧）IK 网格实测可达
# 的保守内缩矩形（x≥0.35,y≤-0.25 远-低角差 6~9mm 不可达）。
# 保持窄条核心区：扩大到近身区（x<0.26）后转移距离变长，插值峰值速度超出
# 电机力矩上限（饱和欠账）+ 近身构型自碰撞，动作出现可见停滞
# 右臂工作区（x, y 范围）——z=0.764（抓取/放置高度，最低最紧）IK 网格实测可达
# 的保守内缩矩形（x≥0.35,y≤-0.25 远-低角差 6~9mm 不可达）。
# y 上界 -0.05：y>-0.05 的近身角（x=0.26,y≈0）构型接近位置雅可比奇异带，
# 转移/放置经过时 ee 明显降速（实测降速点全部落在 y∈[-0.06,0]）
WORKSPACE = dict(
    x=(0.26, 0.34),
    y=(-0.26, -0.05),
)

# 第三人称俯视相机（世界坐标）：从桌前上方看向工作区中心，覆盖整张桌面
OVERHEAD_CAM_POS = [0.95, 0.20, 1.25]
OVERHEAD_CAM_LOOK_AT = [0.32, -0.13, 0.80]
OVERHEAD_CAM_FOVY = 45
# 腕部相机（掌板局部坐标）：装在掌上沿接近方向看手指和接触区
WRIST_CAM_POS = [0.00, 0.00, 0.08]
WRIST_CAM_LOOK_AT = [0.15, 0.00, 0.00]
WRIST_CAM_FOVY = 75

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

    # 第三人称俯视相机
    wb.add_camera(name="overhead_cam", pos=OVERHEAD_CAM_POS,
                  xyaxes=quat_look_at(OVERHEAD_CAM_POS, OVERHEAD_CAM_LOOK_AT),
                  fovy=OVERHEAD_CAM_FOVY)
    # Task 3 数据相机：与交互 viewer 的初始肩后视角一致。
    door_cam_pos = [0.0, 0.25, 1.40]
    door_cam_look_at = [0.38, -0.14, 0.83]
    wb.add_camera(name="door_cam", pos=door_cam_pos,
                  xyaxes=quat_look_at(door_cam_pos, door_cam_look_at), fovy=45)

    # 腕部相机：装在掌板上，沿掌 +x（接近方向）看手指和工作区。
    # 注意 MuJoCo 相机沿自身 -z 看：xyaxes 的 x=(0,-1,0), y=(0,0,1)
    # => z = x×y = (-1,0,0)，视线 = -z = (+1,0,0) = 掌坐标系 +x ✓
    palm = spec.body("gripper_palm")
    palm.add_camera(name="wrist_cam", pos=WRIST_CAM_POS,
                    xyaxes=quat_look_at(WRIST_CAM_POS, WRIST_CAM_LOOK_AT),
                    fovy=WRIST_CAM_FOVY)


def add_object(spec: "mujoco.MjSpec", name: str, shape: str,
               size: float, rgba=(0.5, 0.5, 0.5, 1), pos=(0.45, 0, TABLE_TOP_Z + 0.05)):
    """加一个带 freejoint 的任务物体。shape: cube / cylinder / ball / t。
    size 为半边长/半径（t 形状忽略 size，用固定两块板拼装）。
    rgba 只是初始色，任务每回合会随机覆盖。"""
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_joint")
    kw = dict(rgba=list(rgba), friction=[1.0, 0.005, 0.0001],
              solref=[0.004, 1.0])   # 硬接触：避免夹爪闭合时的接触柔顺抖动
    if shape == "t":
        # T 字积木（平躺放置，初始位姿即稳定位姿）：
        # 横梁 4.6×2.4×4.4cm + 竖梁 2.4×3.4×4.4cm（重叠 0.8cm），体积 76cm³
        # （介于圆柱 67 与方块 85 之间）。厚度 4.4cm 与方块同高——抓取目标
        # z 与方块一致（0.764），避免薄 T 的低目标把手臂构型带进位置雅可比
        # 奇异带（最小奇异值 ~0.02，descend 残差卡 1cm 磨到超时）。
        # 两块板相对 body 原点对称摆放，原点 = T 几何中心，落在竖梁上
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.023, 0.012, 0.022], pos=[0, -0.014, 0],
                      mass=0.06, **kw)
        body.add_geom(name=f"{name}_geom_stem", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.012, 0.017, 0.022], pos=[0, 0.011, 0],
                      mass=0.04, **kw)
        return body
    gtype = {"cube": mujoco.mjtGeom.mjGEOM_BOX,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
             "ball": mujoco.mjtGeom.mjGEOM_SPHERE}[shape]
    body.add_geom(name=f"{name}_geom", type=gtype, size=[size, size, size],
                  mass=0.1, **kw)
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


def add_sliding_door(spec: "mujoco.MjSpec", name: str = "sliding_door"):
    """添加一扇沿世界 +y 横向滑动的双扇窗（Meta-World window-open 风格）。"""
    frame = spec.worldbody.add_body(name=f"{name}_frame")
    frame_rgba = [0.30, 0.22, 0.20, 1]
    for suffix, pos, size in (
        # 外框在后轨，活动扇在 x=0.365 的前轨；两者 x 方向留 6mm 净空。
        ("left",  [0.395, -0.295, 0.83], [0.012, 0.015, 0.105]),
        ("right", [0.395,  0.015, 0.83], [0.012, 0.015, 0.105]),
        ("top",   [0.395, -0.140, 0.94], [0.012, 0.170, 0.015]),
        ("base",  [0.395, -0.140, 0.735], [0.012, 0.170, 0.010]),
    ):
        frame.add_geom(name=f"{name}_frame_{suffix}",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=pos, size=size, rgba=frame_rgba,
                       friction=[0.8, 0.005, 0.0001])

    # 右侧固定窗扇位于后轨。半透明窗格仅用于视觉，不遮住打开后的左侧洞口。
    # 与 Meta-World 的 window-open 一样，活动窗扇滑向固定窗扇并与其重叠。
    fixed_y = -0.0625
    frame.add_geom(name=f"{name}_fixed_glass",
                   type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.397, fixed_y, 0.83], size=[0.004, 0.0675, 0.080],
                   rgba=[0.45, 0.62, 0.68, 0.22],
                   contype=0, conaffinity=0, mass=0)
    for suffix, pos, size in (
        ("fixed_left",   [0.397, -0.130, 0.83], [0.008, 0.007, 0.090]),
        ("fixed_right",  [0.397,  0.005, 0.83], [0.008, 0.007, 0.090]),
        ("fixed_top",    [0.397, fixed_y, 0.92], [0.008, 0.0745, 0.008]),
        ("fixed_bottom", [0.397, fixed_y, 0.74], [0.008, 0.0745, 0.008]),
    ):
        frame.add_geom(name=f"{name}_{suffix}",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=pos, size=size, rgba=frame_rgba,
                       contype=0, conaffinity=0, mass=0)

    # q=0 时活动窗扇覆盖左侧开口；q 增大时向 +y 滑到固定窗扇前方。
    # 两扇各宽约 15cm，因此需要 15cm 行程才能基本露出整个左侧洞口。
    door = spec.worldbody.add_body(name=name, pos=[0.365, -0.205, 0.83])
    door.add_joint(name=f"{name}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE,
                   axis=[0, 1, 0], range=[0.0, 0.15],
                   # 低静摩擦避免夹爪蓄力后门扇突然跳动；阻尼负责平顺停车。
                   damping=1.5, frictionloss=0.15, armature=0.01)
    door.add_geom(name=f"{name}_panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                  # 碰撞窗板比可见窗框略窄：否则 q>13cm 时右缘会顶住外框立柱。
                  size=[0.012, 0.068, 0.080], mass=0.8,
                  rgba=[0.25, 0.45, 0.75, 0.28],
                  friction=[0.9, 0.005, 0.0001])
    # 活动窗扇边框随 slide joint 一起移动；无碰撞，避免改变已验证的抓取动力学。
    for suffix, pos, size in (
        ("moving_left",   [0.0, -0.075, 0.0], [0.014, 0.007, 0.090]),
        ("moving_right",  [0.0,  0.075, 0.0], [0.014, 0.007, 0.090]),
        ("moving_top",    [0.0,  0.0, 0.09], [0.014, 0.082, 0.008]),
        ("moving_bottom", [0.0,  0.0, -0.09], [0.014, 0.082, 0.008]),
    ):
        door.add_geom(name=f"{name}_{suffix}",
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, size=size, rgba=frame_rgba,
                      contype=0, conaffinity=0, mass=0)
    # 把手向机器人方向(-x)突出；留足掌板净空，避免拉到后段时掌部磨窗板。
    door.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[-0.055, 0.0, 0.0], size=[0.012, 0.040, 0.0],
                  mass=0.08, rgba=[0.92, 0.55, 0.12, 1],
                  friction=[1.8, 0.005, 0.0001], solref=[0.004, 1.0])
    door.add_site(name=f"{name}_handle_site", pos=[-0.055, 0.0, 0.0],
                  size=[0.0001] * 3, rgba=[0, 0, 0, 0])
    return door


def configure_viewer_camera(viewer, model, task_name: str):
    """设置交互 viewer 的初始自由视角，设置后仍可用鼠标旋转/平移/缩放。"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    if task_name == "task3_open_sliding_door":
        # 机器人左肩上方看向窗户；比原视角更远，完整显示窗框和右臂。
        viewer.cam.lookat[:] = [0.38, -0.14, 0.83]
        viewer.cam.distance = 0.79
        viewer.cam.azimuth = -45.7
        viewer.cam.elevation = -46.3
    else:
        # 精确复现数据采集 overhead_cam 的斜侧俯视构图。
        viewer.cam.lookat[:] = OVERHEAD_CAM_LOOK_AT
        viewer.cam.distance = 0.842
        viewer.cam.azimuth = -152.4
        viewer.cam.elevation = -32.3


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
