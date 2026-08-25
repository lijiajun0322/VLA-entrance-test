import time
from pathlib import Path

import mujoco
import mujoco.viewer

# Unitree G1: 29 自由度（含双臂），带地板和灯光；无臂版换成 scene_23dof.xml
G1_XML = (
    Path(__file__).parent
    / "unitree_mujoco"
    / "unitree_robots"
    / "g1"
    / "scene_29dof.xml"
)

model = mujoco.MjModel.from_xml_path(str(G1_XML))
data = mujoco.MjData(model)

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        