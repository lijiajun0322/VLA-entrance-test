"""冒烟测试：重构/改动的安全网。

    python tests/smoke_test.py            # 全部
    python tests/smoke_test.py expert     # 只跑专家项

覆盖：模型构建、任务 reset、观测规格、脚本专家成功率、录制器 round-trip。
每项独立可跑，失败即非零退出。
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image


def test_build_and_reset():
    """三个任务都能构建 + 多 seed reset 后物体落稳。"""
    from envs import ALL_TASKS, G1TaskEnv
    from datasets.spec import OBS_SPEC

    for task in ALL_TASKS:
        env = G1TaskEnv(task)
        for seed in range(3):
            obs = env.reset(seed)
            assert obs["overhead_rgb"].shape == (OBS_SPEC.img_size,) * 2 + (3,)
            assert obs["wrist_rgb"].shape == (OBS_SPEC.img_size,) * 2 + (3,)
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


def test_recorder_roundtrip():
    """录制一回合 -> 读回 -> 字段形状/内容自洽。"""
    from envs import ALL_TASKS, G1TaskEnv
    from control.expert import PickPlaceExpert
    from datasets.recorder import EpisodeRecorder
    from datasets.spec import ACTION_SPEC

    with tempfile.TemporaryDirectory() as tmp:
        rec = EpisodeRecorder(tmp, "task1_pick_place")
        env = G1TaskEnv(ALL_TASKS[0])
        env.reset(0)
        rec.start(env)
        expert = PickPlaceExpert(env, env.task.expert_spec())
        every = ACTION_SPEC.sample_every
        i = 0
        while not expert.done:
            expert.step()
            if i % every == 0:
                rec.sample(env, phase=expert._phase())
            i += 1
        path = rec.end(env.is_success())
        rec.write_stats()
        assert path is not None and path.exists()

        d = np.load(path, allow_pickle=True)
        assert d["action"].shape[1] == ACTION_SPEC.dim
        assert d["proprio"].shape[1] == 9 and d["ctrl"].shape[1] == 10  # 8关节+2指
        assert d["n_frames"] == len(d["action"])
        # 图像可解码且尺寸正确
        img = Image.open(io_bytes(d["overhead_rgb"][0]))
        assert img.size == (224, 224)
        # action 与 task_info 对齐（上帝视角元数据存在且可解析）
        assert json.loads(str(d["task_info"]))
        stats = json.loads((Path(tmp) / "stats.json").read_text())
        assert stats["n_episodes"] == 1
        print(f"  [ok] 录制 round-trip ({int(d['n_frames'])} 帧)")


def io_bytes(b):
    import io
    return io.BytesIO(bytes(b))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {"build": test_build_and_reset,
             "expert": test_expert_task1,
             "recorder": test_recorder_roundtrip}
    names = tests if which == "all" else {which: tests[which]}
    print("冒烟测试:")
    for name, fn in names.items():
        fn()
    print("全部通过 ✓")
