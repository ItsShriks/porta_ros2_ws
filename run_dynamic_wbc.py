#!/usr/bin/env python3
"""
MMO-700 Dynamic Whole Body Control (Operational Space Control)
==============================================================
Demonstrates torque-based WBC using the Task-Space Inertia Matrix
(Lambda) and Coriolis/Gravity compensation to track a trajectory.
"""
import time
from pathlib import Path
import mujoco
from mujoco import viewer
import numpy as np
import math

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH  = SCRIPT_DIR / "mmo_700_wbc.xml"

# WBC Joints
WBC_JOINTS = [
    "base_x", "base_y", "base_yaw",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

# Motor Actuators
MOTOR_ACTUATORS = [
    "base_x_motor", "base_y_motor", "base_yaw_motor",
    "shoulder_pan_motor", "shoulder_lift_motor", "elbow_motor",
    "wrist_1_motor", "wrist_2_motor", "wrist_3_motor"
]

def get_rotation_error(R_target, R_current):
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
    print(f"Loading Dynamic WBC model from: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
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
    
    # Get actuator IDs for motor control
    motor_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in MOTOR_ACTUATORS]

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

    # Disable all other actuators to prevent interference (e.g., velocity actuators)
    for i in range(model.nu):
        if i not in motor_act_ids:
            data.ctrl[i] = 0.0

    print("Starting Dynamic WBC...")
    
    # Memory allocation for M
    M = np.zeros((model.nv, model.nv))
    
    # Pre-calculate damping matrix for null-space posture
    # We want to keep joints near their initial position
    q_home = data.qpos.copy()
    
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
            
            # 1. Target kinematics
            if state == "REACHING":
                target_pos = box_pos.copy()
                target_pos[2] += 0.005
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                data.ctrl[gripper_act_id] = 0.0
                
                if np.linalg.norm(target_pos - current_pos) < 0.015:
                    state = "GRASPING"
                    state_ticks = 0
                    print("Reached box. Grasping...")
            elif state == "GRASPING":
                target_pos = box_pos.copy()
                target_pos[2] += 0.005
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
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
                target_acc = np.zeros(3)
                data.ctrl[gripper_act_id] = 255.0
            
            # 2. Current state
            current_pos = data.site_xpos[pinch_id]
            current_R = data.site_xmat[pinch_id].reshape(3, 3)
            
            # 3. Compute Full System Jacobian (6 x nv)
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)
            J_full = np.vstack([Jp, Jr])
            
            current_vel = J_full @ data.qvel
            
            # 4. Cartesian Errors
            pos_err = target_pos - current_pos
            rot_err = get_rotation_error(R_target, current_R)
            
            vel_err = np.concatenate([target_vel, np.zeros(3)]) - current_vel
            
            # 5. Desired Cartesian Acceleration (PD Control)
            Kp = 150.0
            Kd = 25.0
            
            err_6d = np.concatenate([pos_err, rot_err])
            a_des = np.concatenate([target_acc, np.zeros(3)]) + Kp * err_6d + Kd * vel_err
            
            # 6. Mass Matrix and Bias Forces
            mujoco.mj_fullM(model, data, M)
            # Add small regularizer to M to prevent singularity with passive joints
            M_reg = M + np.eye(model.nv) * 1e-4 
            M_inv = np.linalg.inv(M_reg)
            
            h = data.qfrc_bias
            
            # 7. Task-Space Inertia Matrix (Lambda)
            lam_damping = 0.01
            Lambda_inv = J_full @ M_inv @ J_full.T + lam_damping**2 * np.eye(6)
            Lambda = np.linalg.inv(Lambda_inv)
            
            # 8. Compute Task Torques
            # tau_task = J^T * Lambda * a_des
            tau_task = J_full.T @ Lambda @ a_des
            
            # 9. Null-Space Posture Control (Keep joints near home position)
            # tau_null = N^T * tau_0
            # N = I - J^T * (J^T)^+ = I - J^T * Lambda * J * M^-1
            J_bar = M_inv @ J_full.T @ Lambda
            N_T = np.eye(model.nv) - J_full.T @ J_bar.T
            
            # Simple PD on joint posture
            Kp_joint = 10.0
            Kd_joint = 2.0
            q_err = q_home - data.qpos
            # Approximate q_err for free joints (size mismatch if q_home includes quaternions)
            # For simplicity, we just apply it to our 9 DoFs
            tau_0 = np.zeros(model.nv)
            for i, dof in enumerate(dof_ids):
                # Ensure we handle qpos mapping correctly (our 9 joints are 1D so qposadr == dofadr mapping is 1-to-1)
                qpos_idx = model.jnt_qposadr[jids[i]]
                tau_0[dof] = Kp_joint * (q_home[qpos_idx] - data.qpos[qpos_idx]) - Kd_joint * data.qvel[dof]
                
            tau_null = N_T @ tau_0
            
            # 10. Total Command Torques
            tau_total = tau_task + tau_null + h
            
            # 11. Command Motor Actuators
            for i, act_id in enumerate(motor_act_ids):
                dof = dof_ids[i]
                data.ctrl[act_id] = tau_total[dof]

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
