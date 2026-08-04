#!/usr/bin/env python3
from pathlib import Path
import mujoco
import numpy as np

# Resolve repo root and model path
REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_DIR / "robots" / "mmo_700.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in ARM_JOINTS]
q_ids = [model.jnt_qposadr[jid] for jid in jids]
dof_ids = [model.jnt_dofadr[jid] for jid in jids]

# Base x to 1.3m (as if driven)
# Assuming base freejoint starts at 0, 0, -0.01
base_q_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
]
data.qpos[base_q_adr] = 1.3
mujoco.mj_forward(model, data)

perceived_box_pos = data.xpos[box_id].copy()
perceived_box_pos[2] -= 0.02
print("Target Pos:", perceived_box_pos)

target_pos = perceived_box_pos.copy()
target_pos[2] += 0.005
R_target = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

d_ik = mujoco.MjData(model)
d_ik.qpos[:] = data.qpos[:]
for i, qi in enumerate(q_ids):
    d_ik.qpos[qi] = [0.0, -2.0, 1.57, -1.0, -1.57, 0.0][i]

for it in range(3000):
    mujoco.mj_forward(model, d_ik)
    pos_err = target_pos - d_ik.site_xpos[pinch_id]
    R_cur = d_ik.site_xmat[pinch_id].reshape(3, 3)
    R_err = R_target @ R_cur.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if abs(angle) > 1e-6:
        ax = np.array(
            [
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]
        ) / (2 * np.sin(angle))
        rot_err = angle * ax
    else:
        rot_err = np.zeros(3)

    err6 = np.concatenate([pos_err, 0.3 * rot_err])
    if np.linalg.norm(pos_err) < 0.012 and np.linalg.norm(rot_err) < 0.15:
        print(f"Converged in {it} iterations")
        break

    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, d_ik, Jp, Jr, pinch_id)
    Jp = Jp[:, dof_ids]
    Jr = Jr[:, dof_ids]
    J6 = np.vstack([Jp, Jr])
    lam = 0.03
    dq = J6.T @ np.linalg.solve(J6 @ J6.T + lam**2 * np.eye(6), err6)
    for i, (qi, jid) in enumerate(zip(q_ids, jids)):
        lo, hi = model.jnt_range[jid]
        d_ik.qpos[qi] = np.clip(d_ik.qpos[qi] + 0.05 * dq[i], lo, hi)

pos_final = d_ik.site_xpos[pinch_id]
print(f"Position error: {np.linalg.norm(target_pos - pos_final):.4f} m")
print(f"Final qpos: {[d_ik.qpos[qi] for qi in q_ids]}")
