#!/usr/bin/env python3
"""
MMO-700 Kinematic Whole Body Control
====================================
Demonstrates kinematic WBC using the pseudo-inverse of the full Jacobian
(Base X, Y, Yaw + Arm joints) to track a circular trajectory in the air.
"""
import time
from pathlib import Path
import mujoco
from mujoco import viewer
import numpy as np
import math

# Absolute path to the repository root directory (one level up from scripts/)
REPO_DIR = Path(__file__).resolve().parent.parent

# Path to the XML model inside the robots folder
MODEL_PATH = REPO_DIR / "robots" / "mmo_700_wbc.xml"

# WBC Joints
WBC_JOINTS = [
    "base_x", "base_y", "base_yaw",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

# Velocity Actuators
VEL_ACTUATORS = [
    "base_x_vel", "base_y_vel", "base_yaw_vel",
    "shoulder_pan_vel", "shoulder_lift_vel", "elbow_vel",
    "wrist_1_vel", "wrist_2_vel", "wrist_3_vel"
]

def get_rotation_error(R_target, R_current):
    """Computes the rotation error vector (axis-angle representation)."""
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if abs(angle) > 1e-6:
        ax = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2 * np.sin(angle))
        return angle * ax
    return np.zeros(3)

def main():
    print(f"Loading Kinematic WBC model from: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
        # Give the arm a decent starting configuration
        arm_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in WBC_JOINTS[3:]]
        arm_qpos_adr = [model.jnt_qposadr[jid] for jid in arm_jids]
        init_q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        for adr, q in zip(arm_qpos_adr, init_q):
            data.qpos[adr] = q
        mujoco.mj_forward(model, data)

    pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    # Get DoF addresses for WBC joints
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in WBC_JOINTS]
    dof_ids = [model.jnt_dofadr[jid] for jid in jids]

    # Get actuator IDs for velocity control
    vel_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in VEL_ACTUATORS]

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    state = "REACHING"
    state_ticks = 0
    lift_target = None

    # Target Orientation (pointing down)
    R_target = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=float)

    # Disable all other actuators to prevent interference (e.g., motor actuators)
    for i in range(model.nu):
        if i not in vel_act_ids:
            data.ctrl[i] = 0.0

    print("Starting Kinematic WBC...")

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:]  = [1.5, 0.0, 0.8]
        v.cam.distance   = 3.0
        v.cam.elevation  = -20

        t = 0.0
        while v.is_running():
            t += model.opt.timestep
            state_ticks += 1

            box_pos = data.xpos[box_id]
            current_pos = data.site_xpos[pinch_id]
            current_R = data.site_xmat[pinch_id].reshape(3, 3)

            if state == "REACHING":
                target_pos = box_pos.copy()
                target_pos[2] += 0.005
                target_vel = np.zeros(3)
                data.ctrl[gripper_act_id] = 0.0

                if np.linalg.norm(target_pos - current_pos) < 0.015:
                    state = "GRASPING"
                    state_ticks = 0
                    print("Reached box. Grasping...")
            elif state == "GRASPING":
                target_pos = box_pos.copy()
                target_pos[2] += 0.005
                target_vel = np.zeros(3)
                data.ctrl[gripper_act_id] = 255.0

                if state_ticks > 500:
                    state = "LIFTING"
                    state_ticks = 0
                    lift_target = box_pos.copy()
                    lift_target[2] += 0.20
                    print("Grasped. Lifting...")
            elif state == "LIFTING":
                target_pos = lift_target
                target_vel = np.zeros(3)
                data.ctrl[gripper_act_id] = 255.0

            # Current end-effector state
            current_pos = data.site_xpos[pinch_id]
            current_R = data.site_xmat[pinch_id].reshape(3, 3)

            # 2. Compute Cartesian Error (Position + Orientation)
            pos_err = target_pos - current_pos
            rot_err = get_rotation_error(R_target, current_R)

            # Combine into a 6D desired twist (Proportional feedback + Feedforward)
            Kp_pos = 5.0
            Kp_rot = 5.0
            v_des = np.concatenate([
                target_vel + Kp_pos * pos_err,
                Kp_rot * rot_err
            ])

            # 3. Compute Full System Jacobian (9 DoFs: Base + Arm)
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)

            J_full = np.vstack([Jp, Jr])

            # Extract only the columns corresponding to our WBC joints
            J_wbc = J_full[:, dof_ids]

            # 4. Compute Joint Velocities using Damped Pseudo-Inverse
            lam = 0.05  # Damping factor for singularities
            J_pinv = J_wbc.T @ np.linalg.solve(J_wbc @ J_wbc.T + lam**2 * np.eye(6), np.eye(6))
            dq_wbc = J_pinv @ v_des

            # 5. Command Velocity Actuators
            for i, act_id in enumerate(vel_act_ids):
                data.ctrl[act_id] = dq_wbc[i]

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
