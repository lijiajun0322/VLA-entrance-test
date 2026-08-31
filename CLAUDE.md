# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 环境设置

本项目使用 Python 虚拟环境，位于项目根目录的 `.venv/`。

在运行任何 Python 命令之前，必须先激活虚拟环境：

```bash
cd /Users/lijiajun/workspace/vla_g1_mujoco
source .venv/bin/activate
```

或者在单条命令中直接使用虚拟环境的解释器：

```bash
.venv/bin/python scripts/run_expert.py 1 --seeds 10
.venv/bin/pip install <package>
```

## macOS 注意事项

- 在 macOS 上，凡是使用 `mujoco.viewer`（如 `launch` / `launch_passive`）的脚本，
  必须用 `mjpython` 运行，不能用普通 `python`：

```bash
mjpython scripts/demo_tasks.py 1
```

- 不涉及可视化窗口的脚本（加载模型、跑物理仿真、训练数据生成等）用普通 `python` 即可。

## 项目结构

四层纵切：**envs/**（物理世界与观测）/ **control/**（怎么动）/ **data/**（怎么录）/
**scripts/**（薄壳接线）。观测/动作规格在 data/spec.py 单点定义，
任务 4 的学习策略将来原位替换 control/expert.py 的脚本专家。

- `envs/` — LIBERO 风格任务环境（VLA 入门测试任务一）
  - `robot.py` — MjSpec 程序化改造 G1：固定基座（删 freejoint，骨盆 z=0.787）、
    全部力矩电机换成位置伺服、右腕挂两指平行夹爪（`ee_site` 为末端点）
  - `scene.py` — 地板/桌子(高0.72m)/灯光/相机(`overhead_cam`,`wrist_cam`,
    task3 单独使用的 `door_cam`)、
    物体与目标垫(mocap)工具函数、工作区采样（工作区经关节限位约束 IK 实测可达）
  - `g1_env.py` — `G1TaskEnv`：reset(seed) 随机化、get_obs()（按 data/spec.py 规格）、
    set_arm8/set_gripper 控制接口、is_success、task_info（上帝视角元数据）
  - `tasks/` — 任务定义（base.py 为 Task ABC，含 expert_spec()/info() 接口）
    `task1_pick_place.py` / `task2_pick_by_type.py` / `task3_open_sliding_door.py`
    （抓放 → 按类型挑选 → 横向滑开带约束柜门），各自带文本指令与随机化
- `control/` — 控制层（任务二"无模型控制"）
  - `ik.py` — `solve_ik()`：限位约束 + 多随机重启的离线 IK（腰yaw+右臂 8 自由度，
    夹爪竖直朝下）；`HOME_POSE` 初始姿态
  - `servo.py` — `CartesianServo`：笛卡尔闭环伺服（专家收尾修正 / 数据回放检查 /
    任务4策略推理三处复用）
  - `expert.py` — 统一的 20Hz delta-EE scripted experts；按
    `Task.expert_spec().kind` 选择抓放或横拉门状态机
  - `openvla_action_adapter.py` — OpenVLA LIBERO 7D delta action 到安全 4D
    absolute ee_xyz + gripper action 的映射；模型旋转输出不用于当前竖直顶抓任务
- `policies/` — 任务四学习策略接入
  - `openvla_policy.py` — OpenVLA LIBERO-Spatial checkpoint 的 MPS/CPU 推理包装
- `mujoco_datasets/` — 数据层（任务三，Python 包；避免与 Hugging Face
  `datasets` 包同名）
  - `spec.py` — `OBS_SPEC`/`ACTION_SPEC`：观测（task1/2 双相机、task3 单相机，
    224 RGB + 10 维 proprio：腰yaw+右臂7+双指2）、
    动作标签（world-frame delta xyz + gripper_close, 4 维 @20Hz）的唯一契约
  - `recorder.py` — 官方 `LeRobotDataset` v3 writer；采集时直接写 Parquet +
    camera MP4 + metadata，严格记录 observation_t/action_t；失败回合丢弃
  - `npz_recorder.py` — 可选轻量后端；逐回合 NPZ+MP4，可追加，动作和时序契约
    与 LeRobot 后端完全相同
- `data/` — 录制数据与预览图的存放处（gitignore，不入库）：
  `data/<task_name>/v0/` 为数据集，`data/preview/` 为预览图，
  `data/<task_name>_videos/` 为各任务演示视频
- `tests/smoke_test.py` — 冒烟安全网：构建/reset/专家成功率/录制 round-trip
- `scripts/run_expert.py` — `python scripts/run_expert.py 1 --seeds 20` headless 评测；
  `mjpython scripts/run_expert.py 1 --visual` 可视化演示专家执行
- `scripts/collect_data.py` — `python scripts/collect_data.py 1 --episodes 50` 采集数据集
  （`--seeds` 可显式指定 seed 列表控制回合构成）
- `scripts/collect_data_npz.py` / `check_data_npz.py` — 轻量 NPZ 采集与检查
- `scripts/check_data.py` — 加载 LeRobot v3，检查 schema/timestamp/delta 限幅，
  生成关键帧与累计轨迹，并用 delta executor 回放验证
- `scripts/demo_tasks.py` — `mjpython scripts/demo_tasks.py [任务号]` 可视化看任务和随机化
- `data/preview/` — 各任务相机预览图
- `unitree_mujoco/` — 宇树官方 MuJoCo repo（clone 而来）；G1 模型在
  `unitree_mujoco/unitree_robots/g1/`，带地板场景用 `scene_29dof.xml`（29 自由度含双臂）
  或 `scene_23dof.xml`（无臂）
- `assets/` — 预留的资源目录
- `.venv/` — Python 虚拟环境（勿提交、勿修改）
