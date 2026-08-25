"""任务注册表：新增任务在此登记即可被所有 scripts 使用。"""

from .base import Task
from .task1_pick_place import Task1PickPlace
from .task2_pick_by_type import Task2PickByType
from .task3 import Task3

TASKS = {
    Task1PickPlace.name: Task1PickPlace,
    Task2PickByType.name: Task2PickByType,
    Task3.name: Task3,
}

# 按任务号索引（scripts/run_expert.py 1/2/3 用）
ALL_TASKS = [Task1PickPlace(), Task2PickByType(), Task3()]

__all__ = ["Task", "TASKS", "ALL_TASKS"]
