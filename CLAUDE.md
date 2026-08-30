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
  - `scene.py` — 地板/桌子(高0.72m)/灯光/相机(`overhead_cam`,`wrist_cam`)、
    物体与目标垫(mocap)工具函数、工作区采样（工作区经关节限位约束 IK 实测可达）
  - `g1_env.py` — `G1TaskEnv`：reset(seed) 随机化、get_obs()（按 data/spec.py 规格）、
    set_arm8/set_gripper 控制接口、is_success、task_info（上帝视角元数据）
  - `tasks/` — 任务定义（base.py 为 Task ABC，含 expert_spec()/info() 接口）
    `task1_pick_place.py` / `task2_pick_by_type.py` / `task3.py`
    （抓放 → 按形状颜色挑选 → 叠放[占位，玩法待定]），各自带文本指令与随机化
- `control/` — 控制层（任务二"无模型控制"）
  - `ik.py` — `solve_ik()`：限位约束 + 多随机重启的离线 IK（腰yaw+右臂 8 自由度，
    夹爪竖直朝下）；`HOME_POSE` 初始姿态
  - `servo.py` — `CartesianServo`：笛卡尔闭环伺服（专家收尾修正 / 数据回放检查 /
    任务4策略推理三处复用）
  - `expert.py` — `PickPlaceExpert` 脚本专家：路点 IK + 关节插值 + 伺服修正；
    task1 成功率 30/30；构造参数来自 `Task.expert_spec()`
- `mujoco_datasets/` — 数据层（任务三，Python 包；避免与 Hugging Face
  `datasets` 包同名）
  - `spec.py` — `OBS_SPEC`/`ACTION_SPEC`：观测（双相机 224 RGB + 10 维 proprio：
    腰yaw+右臂7+双指2）、
    动作标签（ee_xyz + gripper, 4 维 @20Hz）的唯一契约
  - `recorder.py` — `EpisodeRecorder`：逐帧录 obs/action/ctrl/元数据，npz + manifest
    + 归一化统计；失败回合归档到 failures/
- `data/` — 录制数据与预览图的存放处（gitignore，不入库）：
  `data/<task_name>/v0/` 为数据集，`data/preview/` 为预览图，
  `data/<task_name>_videos/` 为各任务演示视频
- `tests/smoke_test.py` — 冒烟安全网：构建/reset/专家成功率/录制 round-trip
- `scripts/run_expert.py` — `python scripts/run_expert.py 1 --seeds 20` headless 评测；
  `mjpython scripts/run_expert.py 1 --visual` 可视化演示专家执行
- `scripts/collect_data.py` — `python scripts/collect_data.py 1 --episodes 50` 采集数据集
  （`--seeds` 可显式指定 seed 列表控制回合构成）
- `scripts/check_data.py` — 数据集检查 + 回放验证（原 inspect_data/record_video
  合并）：统计/关键帧拼图/ee轨迹图，`--replay N` 用 control/executor 跟随动作
  标签重跑验证可执行性（任务 4 部署链路预演），`--video` 同步录回放 MP4 到
  `inspect/`，与 `videos/` 的专家执行对照
- `scripts/demo_tasks.py` — `mjpython scripts/demo_tasks.py [任务号]` 可视化看任务和随机化
- `data/preview/` — 各任务相机预览图
- `unitree_mujoco/` — 宇树官方 MuJoCo repo（clone 而来）；G1 模型在
  `unitree_mujoco/unitree_robots/g1/`，带地板场景用 `scene_29dof.xml`（29 自由度含双臂）
  或 `scene_23dof.xml`（无臂）
- `assets/` — 预留的资源目录
- `.venv/` — Python 虚拟环境（勿提交、勿修改）
