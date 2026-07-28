import mujoco
import mujoco.viewer

# Load the UR5e model
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)

# Open interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
