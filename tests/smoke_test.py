"""冒烟测试：重构/改动的安全网。

    python tests/smoke_test.py            # 全部
    python tests/smoke_test.py expert     # 只跑专家项

覆盖：模型构建、任务 reset、观测规格、脚本专家成功率、录制器 round-trip。
每项独立可跑，失败即非零退出。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_build_and_reset():
    """三个任务都能构建 + 多 seed reset 后物体落稳。"""
    from envs import ALL_TASKS, G1TaskEnv
    from mujoco_datasets.spec import OBS_SPEC

    for task in ALL_TASKS:
        env = G1TaskEnv(task)
        for seed in range(3):
            obs = env.reset(seed)
            image_keys = {k for k in obs if k.endswith("_rgb")}
            assert image_keys == set(env.obs_camera_map)
            for key in image_keys:
                assert obs[key].shape == (OBS_SPEC.img_size,) * 2 + (3,)
            assert obs["proprio"].shape == (OBS_SPEC.proprio_dim,)
            assert len(obs["instruction"]) > 10
            info = env.task_info()
            assert isinstance(info, dict) and info
        print(f"  [ok] {task.name}: 构建+reset")


def test_expert_task1(rate_threshold=0.8):
    """task1 脚本专家成功率 >= 80%（基线 100%，留余量防物理抖动）。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import PickPlaceExpert

    env = G1TaskEnv(ALL_TASKS[0])
    wins = 0
    n = 10
    for seed in range(n):
        env.reset(seed)
        expert = PickPlaceExpert(env, env.task.expert_spec())
        expert.run()
        wins += env.is_success()
    rate = wins / n
    assert rate >= rate_threshold, f"专家成功率 {rate:.0%} < {rate_threshold:.0%}"
    print(f"  [ok] task1 专家成功率 {wins}/{n}")


def test_delta_expert_contract():
    """三个 expert 的 act() 都是无副作用、有限幅的 4D delta command。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import MAX_DELTA_M, make_expert

    for task in ALL_TASKS:
        env = G1TaskEnv(task)
        env.reset(0)
        expert = make_expert(env, task)
        time_before = float(env.data.time)
        qpos_before = env.data.qpos.copy()
        action = expert.act()
        assert env.data.time == time_before
        np.testing.assert_array_equal(env.data.qpos, qpos_before)
        assert action.as_array().shape == (4,)
        assert np.linalg.norm(action.delta_xyz_m) <= MAX_DELTA_M + 1e-7
    print("  [ok] 三任务 delta expert act() 契约")


def test_expert_task2(rate_threshold=0.8):
    """task2 delta expert 在物体/目标随机化下成功率 >= 80%。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import make_expert

    task = ALL_TASKS[1]
    env = G1TaskEnv(task)
    wins, n = 0, 10
    for seed in range(n):
        env.reset(seed)
        expert = make_expert(env, task)
        expert.run()
        wins += env.is_success()
    rate = wins / n
    assert rate >= rate_threshold, f"task2 专家成功率 {rate:.0%} < {rate_threshold:.0%}"
    print(f"  [ok] task2 专家成功率 {wins}/{n}")


def test_expert_task3(rate_threshold=0.8):
    """task3 横拉门专家成功率 >= 80%（基线 100%）。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import make_expert

    task = ALL_TASKS[2]
    env = G1TaskEnv(task)
    wins, n = 0, 5
    for seed in range(n):
        env.reset(seed)
        expert = make_expert(env, task)
        expert.run()
        wins += env.is_success()
    rate = wins / n
    assert rate >= rate_threshold, f"横拉门专家成功率 {rate:.0%} < {rate_threshold:.0%}"
    print(f"  [ok] task3 横拉门专家成功率 {wins}/{n}")


def test_recorder_roundtrip():
    """直接录制一个 LeRobot v3 回合 -> 读回 -> schema/时序自洽。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import PickPlaceExpert
    from mujoco_datasets.recorder import EpisodeRecorder
    from mujoco_datasets.spec import ACTION_SPEC, OBS_SPEC
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with tempfile.TemporaryDirectory() as tmp:
        rec = EpisodeRecorder(tmp, "task1_pick_place", repo_id="local/smoke")
        env = G1TaskEnv(ALL_TASKS[0])
        env.reset(0)
        rec.start(env, seed=0)
        expert = PickPlaceExpert(env, env.task.expert_spec())
        while not expert.done:
            phase = expert._phase()
            action = expert.act()
            before = env.ee_xpos()
            rec.add_frame(env, action.as_array(), phase)
            expert.executor.step_delta(action)
            expert.update()
            # Recorded translation is the command issued from this observation.
            assert np.linalg.norm(action.delta_xyz_m) <= 0.015 + 1e-7
            assert np.isfinite(before).all()
        n_frames = rec.end(env.is_success())
        rec.finalize()

        ds = LeRobotDataset("local/smoke", root=tmp, video_backend="pyav")
        assert ds.meta.total_episodes == 1
        assert ds.meta.total_frames == n_frames == len(ds)
        assert ds.meta.fps == ACTION_SPEC.control_hz
        assert ds.meta.features["action"]["shape"] == (ACTION_SPEC.dim,)
        assert ds.meta.features["observation.state"]["shape"] == (OBS_SPEC.proprio_dim,)
        assert len(ds.meta.camera_keys) == 2
        sample = ds.hf_dataset[0]
        assert np.asarray(sample["action"]).shape == (4,)
        assert np.asarray(sample["observation.state"]).shape == (10,)
        assert (Path(tmp) / "meta" / "info.json").exists()
        assert list((Path(tmp) / "data").rglob("*.parquet"))
        assert len(list((Path(tmp) / "videos").rglob("*.mp4"))) == 2
        print(f"  [ok] LeRobot v3 round-trip ({n_frames} 帧 + 双相机视频)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {"build": test_build_and_reset,
             "action": test_delta_expert_contract,
             "expert": test_expert_task1,
             "expert2": test_expert_task2,
             "door": test_expert_task3,
             "recorder": test_recorder_roundtrip}
    names = tests if which == "all" else {which: tests[which]}
    print("冒烟测试:")
    for name, fn in names.items():
        fn()
    print("全部通过 ✓")
