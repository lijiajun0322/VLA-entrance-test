"""三个任务共用的 20Hz scripted experts。

Expert 只负责从当前仿真状态生成世界系 EE delta command；统一的
OnlineActionExecutor 将 delta 转成 absolute EE target，再经 IK/servo 执行。
orientation 不进入 action，始终由 IK/servo 保持 GRIPPER_DOWN。
"""

from __future__ import annotations

import numpy as np

from .actions import DeltaEEAction
from .executor import OnlineActionExecutor

MAX_DELTA_M = 0.015
TOL_POS = 0.008
SETTLE_TICKS = 2
PHASE_TIMEOUT_TICKS = 180
WAIT_GRASP_TICKS = 10
WAIT_RELEASE_TICKS = 6

APPROACH_H = 0.12
LIFT_H = 0.15
GRASP_Z_OFF = 0.022


def _clipped_delta(current: np.ndarray, target: np.ndarray,
                   max_delta_m: float = MAX_DELTA_M) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm > max_delta_m:
        delta *= max_delta_m / norm
    return delta.astype(np.float32)


class _DeltaExpertBase:
    """act() 不推进仿真；step() 是 act→execute→update 的便利封装。"""

    PHASES: list[str]

    def __init__(self, env):
        self.env = env
        self.executor = OnlineActionExecutor(env)
        self.phase_i = 0
        self.phase_tick = 0
        self.stable_ticks = 0
        self._target: np.ndarray | None = None

    def _phase(self) -> str:
        return self.PHASES[self.phase_i]

    @property
    def done(self) -> bool:
        return self._phase() == "done"

    def _advance(self):
        self.phase_i += 1
        self._enter_phase()

    def _enter_phase(self):
        self.phase_tick = 0
        self.stable_ticks = 0
        self._target = self._waypoint()

    def _waypoint(self) -> np.ndarray | None:
        raise NotImplementedError

    def _gripper_close(self) -> float:
        raise NotImplementedError

    def _movement_reached(self) -> bool:
        return bool(self._target is not None and
                    np.linalg.norm(self._target - self.env.ee_xpos()) < TOL_POS)

    def _is_wait_phase(self) -> bool:
        return self._phase() in ("grasp", "release")

    def act(self) -> DeltaEEAction:
        """返回即将执行的 20Hz command；不会修改 env 时间或状态。"""
        if self.done or self._is_wait_phase():
            delta = np.zeros(3, dtype=np.float32)
        else:
            delta = _clipped_delta(self.env.ee_xpos(), self._target)
        return DeltaEEAction(delta, self._gripper_close())

    def update(self):
        """在 command 执行后，根据实测状态推进状态机。"""
        if self.done:
            return
        self.phase_tick += 1
        ph = self._phase()
        if self._is_wait_phase():
            wait = WAIT_GRASP_TICKS if ph == "grasp" else WAIT_RELEASE_TICKS
            if self.phase_tick >= wait:
                self._advance()
            return

        reached = self._movement_reached()
        self.stable_ticks = self.stable_ticks + 1 if reached else 0
        if self.stable_ticks >= SETTLE_TICKS:
            self._advance()
        elif self.phase_tick >= PHASE_TIMEOUT_TICKS:
            self._advance()

    def step(self, on_substep=None) -> str:
        """生成并执行一个 20Hz delta command，返回执行前的 phase。"""
        ph = self._phase()
        if self.done:
            return ph
        action = self.act()
        self.executor.step_delta(action, on_substep=on_substep)
        self.update()
        return ph

    def run(self, max_actions: int = 1200) -> str:
        for _ in range(max_actions):
            if self.done:
                break
            self.step()
        return self._phase()


class PickPlaceExpert(_DeltaExpertBase):
    """Task 1/2 共用：用 20Hz delta xyz 完成抓取和放置。"""

    PHASES = ["above_obj", "descend", "grasp", "lift",
              "above_pad", "place", "release", "retract", "done"]

    def __init__(self, env, spec: dict):
        self.obj_name = spec["obj_name"]
        self.pad_name = spec["pad_name"]
        self.place_h = float(spec.get("place_h", 0.045))
        self.grasp_z = float(spec.get("grasp_z", GRASP_Z_OFF))
        super().__init__(env)
        self._enter_phase()

    def _waypoint(self):
        ph = self._phase()
        if ph in ("grasp", "release", "done"):
            return None
        obj = self.env.obj_xpos(self.obj_name)
        pad = self.env.obj_xpos(self.pad_name)
        if ph == "above_obj":
            return obj + [0, 0, APPROACH_H]
        if ph == "descend":
            return obj + [0, 0, self.grasp_z]
        if ph == "lift":
            return self.env.ee_xpos() + [0, 0, LIFT_H - self.grasp_z]
        if ph == "above_pad":
            return pad + [0, 0, APPROACH_H]
        if ph == "place":
            return pad + [0, 0, self.place_h]
        if ph == "retract":
            return self.env.ee_xpos() + [0, 0, APPROACH_H - self.place_h]
        raise RuntimeError(f"unknown phase: {ph}")

    def _gripper_close(self) -> float:
        return 1.0 if self._phase() in (
            "grasp", "lift", "above_pad", "place"
        ) else 0.0


class OpenSlidingDoorExpert(_DeltaExpertBase):
    """Task 3：接近、夹住把手并沿 slide joint 方向横拉。"""

    PHASES = ["above_handle", "descend", "grasp", "slide",
              "release", "retract", "done"]

    def __init__(self, env, spec: dict):
        self.joint_name = spec["joint_name"]
        self.handle_site = spec["handle_site"]
        self.slide_axis = np.asarray(spec["slide_axis"], dtype=float)
        self.open_distance = float(spec["open_distance"])
        self.grasp_z_offset = float(spec.get("grasp_z_offset", 0.0))
        self.success_qpos = float(spec["success_qpos"])
        super().__init__(env)
        self._enter_phase()

    def _waypoint(self):
        ph = self._phase()
        if ph in ("grasp", "release", "done"):
            return None
        handle = self.env.site_xpos(self.handle_site)
        if ph == "above_handle":
            return handle + [0, 0, 0.10]
        if ph == "descend":
            return handle + [0, 0, self.grasp_z_offset]
        if ph == "slide":
            return self.env.ee_xpos() + self.slide_axis * self.open_distance
        if ph == "retract":
            return self.env.ee_xpos() + [0, 0, 0.10]
        raise RuntimeError(f"unknown phase: {ph}")

    def _gripper_close(self) -> float:
        return 1.0 if self._phase() in ("grasp", "slide") else 0.0

    def _movement_reached(self) -> bool:
        if self._phase() == "slide":
            return self.env.joint_qpos(self.joint_name) >= self.success_qpos
        return super()._movement_reached()


def make_expert(env, task):
    """由 task.expert_spec().kind 选择同文件内的 expert。"""
    spec = task.expert_spec()
    if spec.get("kind") == "sliding_door":
        return OpenSlidingDoorExpert(env, spec)
    return PickPlaceExpert(env, spec)
