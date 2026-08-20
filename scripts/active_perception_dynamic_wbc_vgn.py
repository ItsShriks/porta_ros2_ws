#!/usr/bin/env python3
"""
active_perception_dynamic_wbc_vgn.py
=====================================
MMO-700 + UR5e  •  mmo_700_wbc.xml

Full pipeline
─────────────
  DRIVE          — base slides forward; SICK S300 front_lidar controls stop
                   dynamically:
                     > LIDAR_SLOW_DIST  m  → DRIVE_SPEED
                     > LIDAR_STOP_DIST  m  → SLOW_SPEED
                     ≤ LIDAR_STOP_DIST  m  → stop (→ BACKUP check or SETTLE_ARM)
                   Emergency halt fires at < LIDAR_EMERGENCY_DIST in any state.

  BACKUP         — if the robot stopped with front_lidar < LIDAR_EMERGENCY_DIST
                   (too close), it reverses until front_lidar > LIDAR_STOP_DIST,
                   then transitions to SETTLE_ARM.

  SETTLE_ARM     — arm velocity-servos to forward-looking wide-FOV scan seed pose.

  COARSE_DETECT  — shoulder_pan sweep; wrist_cam detects red box + depth-unproject
                   → coarse 3-D position used to centre the TSDF.

  VGN_SCAN       — arm sweeps N_VIEWS poses; wrist_cam depth → TSDF → VGN ConvNet
                   predicts best 6-DOF grasp pose.
                   Graceful fallback: if torch/open3d unavailable, uses coarse pos.

  REACHING       — Operational Space Control (torques) to VGN grasp position
                   with VGN grasp orientation.

  GRASPING       — close Robotiq 2F-85; OSC holds position.

  LIFTING        — OSC drives pinch site +0.20 m in Z.

  DONE           — hold final pose.

LiDAR reference (steve_essentials/neo_sick_s300-2/launch/s300_1.yaml):
  scan_intervals : [[-1.48, 1.48]] rad  (±85°)
  frame_id       : lidar_1_link (front LiDAR)
  MuJoCo sensors : "front_lidar"  "rear_lidar"

VGN reference:
  Breyer et al., "Volumetric Grasping Network: Real-time 6 DOF Grasp Detection
  in Clutter", CoRL 2020.  Model: vgn/data/models/vgn_conv.pth
  Conda env: vgn_env  (torch 2.5.1+cu121, open3d 0.19.0)

OSC controller:
  τ = Jᵀ Λ (Kp·e + Kd·ė)  +  Nᵀ τ_0  +  h
  where Λ = (J M⁻¹ Jᵀ)⁻¹,  Nᵀ = I − Jᵀ J̄ᵀ,  h = Coriolis/gravity

Run:
  cd /home/shrikar/porta_ros2_ws
  python3 scripts/active_perception_dynamic_wbc_vgn.py [--headless]
"""

import sys
import time
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np

REPO_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_DIR / "robots" / "mmo_700_wbc.xml"
sys.path.insert(0, str(REPO_DIR / "scripts"))

from perception_core import (
    PerceptionConfig, IMG_H, IMG_W,
    render_rgbd, detect_red_object,
    unproject_pixel_to_world,
)
from vgn_mujoco_bridge import VGNBridge, scan_arm_configs, read_lidar, VGN_OK

# ── Actuator / joint names ─────────────────────────────────────────────────────
MOTOR_ACTUATORS = [
    "base_x_motor",       "base_y_motor",       "base_yaw_motor",
    "shoulder_pan_motor", "shoulder_lift_motor", "elbow_motor",
    "wrist_1_motor",      "wrist_2_motor",       "wrist_3_motor",
]
VEL_ACTUATORS = [
    "base_x_vel",       "base_y_vel",       "base_yaw_vel",
    "shoulder_pan_vel", "shoulder_lift_vel", "elbow_vel",
    "wrist_1_vel",      "wrist_2_vel",      "wrist_3_vel",
]
WBC_JOINTS = [
    "base_x",             "base_y",             "base_yaw",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# ── Driving parameters ─────────────────────────────────────────────────────────
DRIVE_SPEED   = 0.5    # m/s — fast approach
SLOW_SPEED    = 0.15   # m/s — near obstacle
BACKUP_SPEED  = 0.10   # m/s — reverse when too close

# ── LiDAR thresholds (SICK S300, ±85°, 40 ms cycle) ──────────────────────────
LIDAR_SLOW_DIST      = 0.60   # m — switch to SLOW_SPEED
LIDAR_STOP_DIST      = 0.10   # m — stop base cleanly when 10 cm from table skirt
LIDAR_EMERGENCY_DIST = 0.04   # m — hard emergency halt

# Hard position stop (base_x) — safety net
# Base at x=1.20m puts arm shoulder at x=1.405m, giving 64.5 cm reach to box at x=2.05m (optimal UR5e reach envelope)
ARM_STOP_X = 1.20    # m — halt base for optimal arm reach

# ── Arm poses ─────────────────────────────────────────────────────────────────
# Retracted pose for safe driving
DRIVE_POSE  = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
# Forward-looking wide-FOV scan seed
SCAN_SEED_Q = [0.0, -1.8, 1.4, -1.2, -1.57, 0.0]
SETTLE_STEPS = 1200   # sim steps to reach SCAN_SEED_Q

# ── VGN scan parameters ───────────────────────────────────────────────────────
VGN_N_VIEWS      = 4     # 4 centered TSDF depth views for superfast scan
VGN_SETTLE_STEPS = 300   # sim steps to hold each pan angle before capture
VGN_MODEL_PATH   = REPO_DIR / "vgn" / "data" / "models" / "vgn_conv.pth"

# ── Perception config ─────────────────────────────────────────────────────────
CFG = PerceptionConfig(cam_name="wrist_cam")

# ── OSC controller gains ──────────────────────────────────────────────────────
KP_CART  = 150.0;  KD_CART  = 25.0
KP_JOINT = 10.0;   KD_JOINT = 2.0
LAM_DAMP = 0.01

# Top-down fallback rotation (used when VGN is unavailable)
R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rot_err(R_t: np.ndarray, R_c: np.ndarray) -> np.ndarray:
    Re  = R_t @ R_c.T
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0))
    if abs(ang) > 1e-6:
        ax = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]]) / (2*np.sin(ang))
        return ang * ax
    return np.zeros(3)


def osc_torques(model, data, M, pinch_id, dof_ids, jids, tgt_pos, tgt_vel,
                q_home, R_target=None):
    if R_target is None:
        R_target = R_TOPDOWN
    Jp = np.zeros((3, model.nv));  Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)
    J6 = np.vstack([Jp, Jr])
    e6 = np.concatenate([
        tgt_pos - data.site_xpos[pinch_id],
        rot_err(R_target, data.site_xmat[pinch_id].reshape(3, 3)),
    ])
    v_err  = np.concatenate([tgt_vel, np.zeros(3)]) - J6 @ data.qvel
    a_des  = KP_CART * e6 + KD_CART * v_err
    mujoco.mj_fullM(model, data, M)
    M_inv  = np.linalg.inv(M + np.eye(model.nv) * 1e-4)
    Lambda = np.linalg.inv(J6 @ M_inv @ J6.T + LAM_DAMP**2 * np.eye(6))
    tau_t  = J6.T @ Lambda @ a_des
    J_bar  = M_inv @ J6.T @ Lambda
    N_T    = np.eye(model.nv) - J6.T @ J_bar.T
    tau_0  = np.zeros(model.nv)
    for i, dof in enumerate(dof_ids):
        qa = model.jnt_qposadr[jids[i]]
        tau_0[dof] = KP_JOINT * (q_home[qa] - data.qpos[qa]) - KD_JOINT * data.qvel[dof]
    return tau_t + N_T @ tau_0 + data.qfrc_bias


def kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel, R_target=None):
    if R_target is None:
        R_target = R_TOPDOWN
    arm_dofs = dof_ids[3:]
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)
    J6 = np.vstack([Jp[:, arm_dofs], Jr[:, arm_dofs]])

    p_err = tgt_pos - data.site_xpos[pinch_id]
    r_err = rot_err(R_target, data.site_xmat[pinch_id].reshape(3, 3))
    v_des = np.concatenate([8.0 * p_err, 4.0 * r_err])

    dq = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-3 * np.eye(6), v_des)
    return dq


def zero_vel(data, ids):
    for i in ids: data.ctrl[i] = 0.0

def zero_motor(data, ids):
    for i in ids: data.ctrl[i] = 0.0


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_PATH}")
    model    = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, IMG_H, IMG_W)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # ── ID look-ups ───────────────────────────────────────────────────────────
    pinch_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     "pinch")
    box_id         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "target_box")
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    jids          = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    j) for j in WBC_JOINTS]
    dof_ids       = [model.jnt_dofadr[jid]  for jid in jids]
    qpos_ids      = [model.jnt_qposadr[jid] for jid in jids]
    motor_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in MOTOR_ACTUATORS]
    vel_act_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in VEL_ACTUATORS]

    base_x_qpos         = qpos_ids[0]
    base_x_vel_id       = vel_act_ids[0]
    base_y_vel_id       = vel_act_ids[1]
    arm_qpos            = qpos_ids[3:]
    shoulder_pan_vel_id  = vel_act_ids[3]
    shoulder_lift_vel_id = vel_act_ids[4]
    shoulder_pan_qpos    = qpos_ids[3]

    # ── Initialize Dynamic Red Ball on Table ──────────────────────────────────
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
    if box_joint_id >= 0:
        qpos_adr = model.jnt_qposadr[box_joint_id]
        qvel_adr = model.jnt_dofadr[box_joint_id]
        # Red ball sits on table surface (Z=0.825m) rolling along +Y at 0.02 m/s
        r = 0.025; vy = 0.02; wx = -vy / r
        data.qpos[qpos_adr:qpos_adr+3] = [2.05, -0.15, 0.825]
        data.qvel[qvel_adr:qvel_adr+6] = [0.0, vy, 0.0, wx, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    # ── VGN bridge ────────────────────────────────────────────────────────────
    vgn_bridge = VGNBridge(model_path=VGN_MODEL_PATH)

    # ── State machine variables ────────────────────────────────────────────────
    state              = "DRIVE"
    ticks              = 0
    scan_dir           = 1
    scan_tmr           = 0
    perceived_box_pos  = None
    grasp_R_target     = R_TOPDOWN.copy()
    tgt_pos            = None
    tgt_vel            = np.zeros(3)
    lift_target        = None
    q_home             = None
    M                  = np.zeros((model.nv, model.nv))

    # VGN scan sweep
    vgn_pan_targets    = scan_arm_configs(VGN_N_VIEWS)
    vgn_view_idx       = 0
    vgn_settle_counter = 0

    q_init = data.qpos.copy()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  MMO-700  Dynamic WBC + LiDAR Avoidance + VGN Grasping")
    print("═" * 65)
    print(f"  Controller  : Operational Space Control (torque)")
    print(f"  LiDAR       : SICK S300  slow={LIDAR_SLOW_DIST} m  "
          f"stop={LIDAR_STOP_DIST} m  emergency={LIDAR_EMERGENCY_DIST} m")
    print(f"  VGN         : available={vgn_bridge.available}  "
          f"model={VGN_MODEL_PATH.name}")
    if not vgn_bridge.available:
        print("  ⚠  VGN fallback: depth-unproject for grasp position, "
              "top-down rotation")
    print(f"  Target box  : pos=[2.05, 0.08, 0.82] m  (offset from table centre)")
    print("═" * 65 + "\n")

    headless = "--headless" in sys.argv
    if headless:
        print("Running in headless mode …")
        v = None
    else:
        v = viewer.launch_passive(model, data)
        v.cam.lookat[:] = [1.0, 0.0, 0.8]
        v.cam.distance  = 4.5
        v.cam.elevation = -15

    try:
        while (v.is_running() if v else True):

            # ── LiDAR reads (every step) ─────────────────────────────────────
            front_lidar = read_lidar(model, data, "front_lidar")
            rear_lidar  = read_lidar(model, data, "rear_lidar")

            # ── Global emergency halt (terminal warning during driving) ───────
            if state == "DRIVE" and front_lidar < LIDAR_EMERGENCY_DIST:
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0
                if ticks % 200 == 0:
                    print(f"  [⚠ LIDAR EMERGENCY] front={front_lidar:.3f} m < "
                          f"{LIDAR_EMERGENCY_DIST} m — base halted!")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 1 — DRIVE  (LiDAR-reactive two-speed approach)
            # ══════════════════════════════════════════════════════════════════
            if state == "DRIVE":
                zero_motor(data, motor_act_ids)
                base_x = data.qpos[base_x_qpos]

                if front_lidar > LIDAR_SLOW_DIST:
                    speed = DRIVE_SPEED
                elif front_lidar > LIDAR_STOP_DIST:
                    speed = SLOW_SPEED
                else:
                    speed = 0.0

                data.ctrl[base_x_vel_id] = speed
                data.ctrl[base_y_vel_id] = 0.0

                # Hold arm in retracted drive pose
                for i, (qi, vid) in enumerate(zip(arm_qpos, vel_act_ids[3:])):
                    data.ctrl[vid] = 5.0 * (DRIVE_POSE[i] - data.qpos[qi])

                ticks += 1
                if ticks % 200 == 0:
                    fl_str = f"{front_lidar:.3f}" if front_lidar < 99 else "inf"
                    print(f"  [DRIVE] base_x={base_x:.3f} m  "
                          f"front_lidar={fl_str} m  speed={speed:.2f} m/s")

                # Stop when EITHER LiDAR detects obstacle OR position limit reached
                lidar_triggered = (front_lidar <= LIDAR_STOP_DIST)
                pos_triggered   = (base_x >= ARM_STOP_X)
                if lidar_triggered or pos_triggered:
                    data.ctrl[base_x_vel_id] = 0.0
                    trigger = (f"LiDAR front={front_lidar:.3f} m"
                               if lidar_triggered else
                               f"position base_x={base_x:.3f} m >= ARM_STOP_X={ARM_STOP_X}")
                    print(f"\n[DRIVE → ?]  Stop trigger: {trigger}")

                    # If already dangerously close, back up first
                    if front_lidar < LIDAR_EMERGENCY_DIST:
                        print(f"  → Too close ({front_lidar:.3f} m < "
                              f"{LIDAR_EMERGENCY_DIST} m) — BACKUP first\n")
                        state = "BACKUP"
                    else:
                        print("  → SETTLE_ARM\n")
                        state = "SETTLE_ARM"
                    ticks = 0

            # ══════════════════════════════════════════════════════════════════
            # BACKUP — reverse until safe clearance, then SETTLE_ARM
            # ══════════════════════════════════════════════════════════════════
            elif state == "BACKUP":
                zero_motor(data, motor_act_ids)

                # Hold arm pose during backup
                for i, (qi, vid) in enumerate(zip(arm_qpos, vel_act_ids[3:])):
                    data.ctrl[vid] = 5.0 * (DRIVE_POSE[i] - data.qpos[qi])

                # Reverse until BOTH: lidar clear AND base_x safely below ARM_STOP_X
                # Uses max() so we back up at least until ARM_STOP_X - 0.1 m
                target_retreat_x = ARM_STOP_X - 0.10   # back up past the safe stop pos
                lidar_clear = (front_lidar >= LIDAR_STOP_DIST)
                pos_clear   = (base_x <= target_retreat_x)

                if not (lidar_clear and pos_clear):
                    # Still too close — keep reversing
                    data.ctrl[base_x_vel_id] = -BACKUP_SPEED
                    data.ctrl[base_y_vel_id] = 0.0
                    ticks += 1
                    if ticks % 100 == 0:
                        fl_str = f"{front_lidar:.3f}" if front_lidar < 99 else "inf"
                        print(f"  [BACKUP] front_lidar={fl_str} m  "
                              f"base_x={base_x:.3f} m  "
                              f"reversing at {BACKUP_SPEED:.2f} m/s …")
                else:
                    # Clear on both conditions — proceed
                    data.ctrl[base_x_vel_id] = 0.0
                    data.ctrl[base_y_vel_id] = 0.0
                    fl_str = f"{front_lidar:.3f}" if front_lidar < 99 else "inf"
                    print(f"[BACKUP → SETTLE_ARM]  front_lidar={fl_str} m  "
                          f"base_x={base_x:.3f} m  — safe clearance reached\n")
                    state = "SETTLE_ARM"
                    ticks = 0


            # ══════════════════════════════════════════════════════════════════
            # PHASE 2 — SETTLE ARM to forward-looking scan seed pose
            # ══════════════════════════════════════════════════════════════════
            elif state == "SETTLE_ARM":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0
                for i, (qi, vid) in enumerate(zip(arm_qpos, vel_act_ids[3:])):
                    data.ctrl[vid] = 5.0 * (SCAN_SEED_Q[i] - data.qpos[qi])
                ticks += 1
                if ticks >= SETTLE_STEPS:
                    state    = "COARSE_DETECT"
                    ticks    = 0
                    scan_dir = 1
                    scan_tmr = 0
                    print(f"[SETTLE_ARM → COARSE_DETECT]  "
                          f"Starting shoulder_pan sweep "
                          f"[{CFG.scan_start:.1f} → {CFG.scan_end:.1f}] rad …")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 3 — COARSE DETECT + depth-unproject localisation
            # ══════════════════════════════════════════════════════════════════
            elif state == "COARSE_DETECT":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0

                # Hold non-scanning arm joints
                for i in [2, 3, 4, 5]:
                    data.ctrl[vel_act_ids[3 + i]] = \
                        5.0 * (SCAN_SEED_Q[i] - data.qpos[arm_qpos[i]])
                data.ctrl[shoulder_lift_vel_id] = \
                    5.0 * (SCAN_SEED_Q[1] - data.qpos[arm_qpos[1]])

                # Sweep shoulder_pan
                cur_pan = data.qpos[shoulder_pan_qpos]
                if cur_pan >= CFG.scan_end:    scan_dir = -1
                elif cur_pan <= CFG.scan_start: scan_dir =  1
                data.ctrl[shoulder_pan_vel_id] = scan_dir * CFG.scan_vel

                scan_tmr += 1
                if scan_tmr % CFG.render_every == 0:
                    mujoco.mj_forward(model, data)
                    rgb, depth = render_rgbd(renderer, model, data, CFG.cam_name)
                    found, cx, cy, _ = detect_red_object(rgb, CFG)

                    if found:
                        pt, raw_d = unproject_pixel_to_world(
                            model, data, CFG.cam_name, cx, cy, depth)
                        if pt is not None:
                            coarse_pos = pt.copy()
                            coarse_pos[2] -= CFG.box_half_height
                            gt  = data.xpos[box_id].copy()
                            err = np.linalg.norm(coarse_pos - gt)
                            print(f"\n[COARSE_DETECT] ✓ box found "
                                  f"pixel=({cx:.0f},{cy:.0f})  depth={raw_d:.3f} m  "
                                  f"pan={cur_pan:.3f} rad")
                            print(f"  Coarse pos  : {np.round(coarse_pos, 4)}")
                            print(f"  Ground truth: {np.round(gt, 4)}")
                            print(f"  3-D error   : {err*100:.2f} cm\n")
                            perceived_box_pos = coarse_pos.copy()

                            if vgn_bridge.available:
                                vgn_bridge.init_tsdf(perceived_box_pos)
                                vgn_view_idx       = 0
                                vgn_settle_counter = 0
                                state = "VGN_SCAN"
                                print(f"[COARSE_DETECT → VGN_SCAN]  "
                                      f"Collecting {VGN_N_VIEWS} TSDF views …")
                            else:
                                q_home = data.qpos.copy()
                                state  = "REACHING"
                                ticks  = 0
                                print("[COARSE_DETECT → REACHING]  "
                                      "(VGN unavailable — depth-unproject position, "
                                      "top-down rotation)")
                        else:
                            if scan_tmr % (CFG.render_every * 20) == 0:
                                print(f"  [COARSE_DETECT] box visible but depth "
                                      f"invalid (d={raw_d:.4f} m)")
                    else:
                        if scan_tmr % (CFG.render_every * 20) == 0:
                            print(f"  [COARSE_DETECT] pan={cur_pan:.3f} rad — searching …")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 4 — VGN SCAN  (collect depth views → TSDF → inference)
            # ══════════════════════════════════════════════════════════════════
            elif state == "VGN_SCAN":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0

                if vgn_view_idx < VGN_N_VIEWS:
                    target_pan = vgn_pan_targets[vgn_view_idx]

                    # Servo shoulder_pan to target; hold other joints
                    data.ctrl[shoulder_pan_vel_id] = \
                        3.0 * (target_pan - data.qpos[shoulder_pan_qpos])
                    for i in [1, 2, 3, 4, 5]:
                        data.ctrl[vel_act_ids[3 + i]] = \
                            5.0 * (SCAN_SEED_Q[i] - data.qpos[arm_qpos[i]])

                    if abs(data.qpos[shoulder_pan_qpos] - target_pan) < 0.05:
                        vgn_settle_counter += 1
                    else:
                        vgn_settle_counter = 0

                    if vgn_settle_counter >= VGN_SETTLE_STEPS:
                        mujoco.mj_forward(model, data)
                        n = vgn_bridge.collect_view(
                            model, data, renderer, CFG.cam_name, perceived_box_pos)
                        print(f"  [VGN_SCAN] View {n}/{VGN_N_VIEWS}  "
                              f"pan={data.qpos[shoulder_pan_qpos]:.3f} rad")
                        vgn_view_idx      += 1
                        vgn_settle_counter = 0
                else:
                    # All views captured — run inference
                    print("  [VGN_SCAN] Running VGN inference …")
                    found_grasp = vgn_bridge.run_inference()

                    if found_grasp:
                        result = vgn_bridge.best_grasp_world()
                        if result is not None:
                            pos_w, rot_w, width_m, score = result
                            gt  = data.xpos[box_id].copy()
                            err = np.linalg.norm(pos_w - gt)
                            print(f"\n[VGN] Best grasp:")
                            print(f"  Position (world) : {np.round(pos_w, 4)}")
                            print(f"  Rotation matrix  :\n{np.round(rot_w, 3)}")
                            print(f"  Width            : {width_m*100:.1f} cm")
                            print(f"  Quality score    : {score:.3f}")
                            print(f"  Ground truth pos : {np.round(gt, 4)}")
                            print(f"  Position error   : {err*100:.2f} cm\n")
                            perceived_box_pos = pos_w.copy()
                            grasp_R_target    = rot_w.copy()
                        else:
                            print("  [VGN] best_grasp_world() returned None "
                                  "— using coarse position, top-down rotation")
                    else:
                        print("  [VGN] No grasp above threshold "
                              "— using coarse depth-unproject position, top-down rotation")

                    q_home = data.qpos.copy()
                    for i, qid in enumerate(arm_qpos):
                        q_home[qid] = [0.0, -1.0, 1.5, -2.05, -1.57, 0.0][i]
                    state  = "REACHING"
                    ticks  = 0
                    print(f"[VGN_SCAN → REACHING]  "
                          f"target={np.round(perceived_box_pos, 4)}\n")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 5 — REACHING  (Dynamic tracking & Kinematic WBC intercept)
            # ══════════════════════════════════════════════════════════════════
            elif state == "REACHING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 0.0
                # Real-time visual tracking of dynamic rolling ball
                perceived_box_pos = data.xpos[box_id].copy()
                tgt_pos = perceived_box_pos.copy()
                tgt_pos[2] += 0.020   # 2.0 cm above ball centre
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel, R_target=grasp_R_target)
                for i, vid in enumerate(vel_act_ids[3:]):
                    data.ctrl[vid] = dq[i]
                dist = np.linalg.norm(tgt_pos - data.site_xpos[pinch_id])
                ticks += 1
                if ticks % 200 == 0:
                    print(f"  [REACHING] ball={np.round(perceived_box_pos,3)} dist={dist*100:.1f} cm")
                if dist < 0.025 or ticks > 2500:   # 2.5 cm threshold — ensures end-effector surrounds target sphere
                    state = "GRASPING"
                    ticks = 0
                    print(f"Intercepted dynamic target (dist={dist*100:.1f} cm) → GRASPING")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 6 — GRASPING  (Lockstep tracking while closing fingers)
            # ══════════════════════════════════════════════════════════════════
            elif state == "GRASPING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 255.0
                perceived_box_pos = data.xpos[box_id].copy()
                tgt_pos = perceived_box_pos.copy()
                tgt_pos[2] += 0.020
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel, R_target=grasp_R_target)
                for i, vid in enumerate(vel_act_ids[3:]):
                    data.ctrl[vid] = dq[i]
                ticks += 1
                if ticks > 600:
                    lift_target = data.site_xpos[pinch_id].copy()
                    lift_target[2] += 0.25
                    state = "LIFTING"
                    ticks = 0
                    print("Grasped dynamic target → LIFTING")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 7 — LIFTING
            # ══════════════════════════════════════════════════════════════════
            elif state == "LIFTING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 255.0
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, lift_target, tgt_vel, R_target=grasp_R_target)
                for i, vid in enumerate(vel_act_ids[3:]):
                    data.ctrl[vid] = dq[i]
                ticks += 1
                if ticks == 600:
                    box_z = data.xpos[box_id][2]
                    status = "LIFTED ✓" if box_z > 0.90 else "low — check grasp"
                    print(f"\n✅ Done.  Dynamic Object Z = {box_z:.3f} m  [{status}]")
                    state = "DONE"

            # ══════════════════════════════════════════════════════════════════
            # DONE
            # ══════════════════════════════════════════════════════════════════
            elif state == "DONE":
                if headless:
                    break

            mujoco.mj_step(model, data)
            if v:
                v.sync()
            time.sleep(model.opt.timestep)

    finally:
        if v:
            v.close()


if __name__ == "__main__":
    main()
