#!/usr/bin/env python3
from pathlib import Path
import mujoco
import numpy as np

# Resolve repo root and model path
REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_DIR / "robots" / "mmo_700.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

# Move robot closer to box to test perception
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")

# In mmo_700.xml, base is a free joint or similar
WHEEL_FL = 8
WHEEL_FR = 9
WHEEL_BL = 10
WHEEL_BR = 11

for _ in range(500):
    data.ctrl[WHEEL_FL] = 4.0
    data.ctrl[WHEEL_FR] = 4.0
    data.ctrl[WHEEL_BL] = 4.0
    data.ctrl[WHEEL_BR] = 4.0
    mujoco.mj_step(model, data)

# Prepare arm reach pose
data.ctrl[0:6] = [0.0, -2.0, 1.57, -1.0, -1.57, 0.0]
for _ in range(500):
    mujoco.mj_step(model, data)

mujoco.mj_forward(model, data)
print("Robot X:", data.xpos[base_id][0])
print("Box X:", data.xpos[box_id][0])

renderer = mujoco.Renderer(model, 480, 640)

cx = 640 / 2.0
cy = 480 / 2.0

for cam_name in ["pan_tilt_cam", "wrist_cam"]:
    print(f"\n--- {cam_name} ---")
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render()

    mask = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        print("No red pixels found!")
    else:
        med_y = int(np.median(ys))
        med_x = int(np.median(xs))
        d = depth[med_y, med_x]

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = model.cam_fovy[cam_id]
        f = 0.5 * 480 / np.tan(np.deg2rad(fovy) / 2)

        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

        x_cam = (med_x - cx) * d / f
        y_cam = (cy - med_y) * d / f
        z_cam = -d
        pt_cam = np.array([x_cam, y_cam, z_cam])
        pt_world = cam_pos + cam_mat @ pt_cam

        print("Estimated box pos:", pt_world)
        print("Ground truth box pos:", data.xpos[box_id])
        print("Error:", np.linalg.norm(pt_world - data.xpos[box_id]))
