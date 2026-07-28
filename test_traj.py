import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mmo_700.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in ARM_JOINTS]
q_ids = [model.jnt_qposadr[jid] for jid in jids]

base_q_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")]
data.qpos[base_q_adr] = 1.3
mujoco.mj_forward(model, data)

start_q = [0.0, -2.0, 1.57, -1.0, -1.57, 0.0]
target_q = [-0.26495, -1.3478, 2.0920, -2.2699, -1.5829, -0.2645]

print("Interpolating trajectory Z heights:")
min_z = 100
for i in range(11):
    alpha = i / 10.0
    for j, qi in enumerate(q_ids):
        data.qpos[qi] = start_q[j] + alpha * (target_q[j] - start_q[j])
    mujoco.mj_forward(model, data)
    z = data.site_xpos[pinch_id][2]
    min_z = min(min_z, z)
    print(f"alpha {alpha:.1f} -> Z = {z:.4f}")
print("Min Z:", min_z)
