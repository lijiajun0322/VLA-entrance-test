"""任务定义基类。

一个 Task = 一类仿真任务的全部"世界规则"：场景里有什么物体、每回合
怎么随机化、指令文本怎么生成、怎样算成功、脚本专家需要哪些参数。

子类必须实现的四个钩子与两个对外接口：
    setup(spec)              往 MjSpec 里加物体/目标垫
    randomize(env, rng)      每回合随机化
    instruction(env, rng)    本回合指令文本（须与随机化后的场景一致）
    is_success(env)          上帝视角成功判定
    expert_spec() -> dict    脚本专家参数 {"obj_name","pad_name","place_h",...}
    info(env) -> dict        上帝视角元数据（数据录制用，不进模型）
"""


class Task:
    name: str = "task"

    def setup(self, spec):
        """往 spec 里加任务物体/目标垫，并在 self 上记录元数据。"""

    def randomize(self, env, rng):
        """每回合随机化：物体位姿、颜色分配、干扰物等。"""

    def instruction(self, env, rng) -> str:
        """返回本回合的自然语言指令。"""
        return ""

    def is_success(self, env) -> bool:
        """上帝视角成功判定。"""
        return False

    def expert_spec(self) -> dict:
        """统一脚本专家工厂（control.make_expert）的构造参数。
        默认抓 cube 放 target_pad；子类按需覆盖。"""
        return {"obj_name": "cube", "pad_name": "target_pad", "place_h": 0.045}

    def info(self, env) -> dict:
        """上帝视角元数据 dict（录制数据时随帧/随回合保存，供分析用）。"""
        return {}
