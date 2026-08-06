#!/usr/bin/env python3
"""
Active Perception + Kinematic Whole-Body Control (Pseudo-Inverse Jacobian)
==========================================================================
MMO-700 + UR5e  •  mmo_700_wbc.xml

Full pipeline (all phases visible in the viewer):

  DRIVE         — base_x_vel drives the robot forward to arm-reach distance
  SETTLE_ARM    — arm velocity-servos to wrist-cam wide-FOV scan seed pose
  WRIST_SCAN    — shoulder_pan sweeps ±0.8 rad; wrist_cam detects red box
  WRIST_SERVO   — pixel-error servo centres box in wrist_cam frame
  LOCALISE      — depth un-project centroid → 3-D world position
  REACHING      — Kinematic WBC (damped pseudo-inverse) → perceived_box_pos
  GRASPING      — close Robotiq 2F-85; WBC holds position
  LIFTING       — WBC drives pinch site +0.20 m in Z
  DONE          — hold

NOTE: mmo_700_wbc.xml has no pan-tilt actuators, so the pan-tilt camera
      points downward at default joint angles.  wrist_cam is used for all
      object localisation.

Kinematic WBC:
  q̇_wbc = J†_wbc · v_des    (9 DOFs: base x/y/yaw + arm×6)
  J†_wbc = Jᵀ(JJᵀ + λ²I)⁻¹  (damped right pseudo-inverse)
  v_des  = Kp_pos·e_pos + Kp_rot·e_rot

Run:
  cd /Users/shrikar/porta_ros2_ws
  mjpython scripts/active_perception_kinematic_wbc.py
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

# ── Actuator / joint name lists ────────────────────────────────────────────────
VEL_ACTUATORS = [
    "base_x_vel",      "base_y_vel",      "base_yaw_vel",
    "shoulder_pan_vel","shoulder_lift_vel","elbow_vel",
    "wrist_1_vel",     "wrist_2_vel",     "wrist_3_vel",
]
MOTOR_ACTUATORS = [
    "base_x_motor",      "base_y_motor",      "base_yaw_motor",
    "shoulder_pan_motor","shoulder_lift_motor","elbow_motor",
    "wrist_1_motor",     "wrist_2_motor",     "wrist_3_motor",
]
WBC_JOINTS = [
    "base_x",             "base_y",             "base_yaw",
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# ── Navigation geometry ────────────────────────────────────────────────────────
TABLE_X     = 2.0
ARM_STOP    = 1.3
DRIVE_SPEED = 0.5
SLOW_SPEED  = 0.15
SLOW_START  = 0.9

# ── Arm seed pose for wrist-cam scan ─────────────────────────────────────────
SETTLE_STEPS = 1200  # move to scan pose after reaching table
SCAN_SEED_Q  = [0.0, -1.8, 1.4, -1.2, -1.57, 0.0]

# ── Perception ─────────────────────────────────────────────────────────────────
CFG = PerceptionConfig(cam_name="wrist_cam")

# ── Kinematic WBC gains ────────────────────────────────────────────────────────
KP_POS = 5.0;  KP_ROT = 5.0;  LAM = 0.05
R_TARGET = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rot_err(R_t, R_c):
    Re  = R_t @ R_c.T
    ang = np.arccos(np.clip((np.trace(Re)-1)/2, -1, 1))
    if abs(ang) > 1e-6:
        ax = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]]) / (2*np.sin(ang))
        return ang * ax
    return np.zeros(3)


def kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel):
    """9-DOF damped pseudo-inverse WBC → dq_wbc (9,)."""
    pos_e = tgt_pos - data.site_xpos[pinch_id]
    rot_e = rot_err(R_TARGET, data.site_xmat[pinch_id].reshape(3,3))
    v_des = np.concatenate([tgt_vel + KP_POS * pos_e, KP_ROT * rot_e])
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)
    Jw     = np.vstack([Jp, Jr])[:, dof_ids]
    J_pinv = Jw.T @ np.linalg.solve(Jw @ Jw.T + LAM**2 * np.eye(6), np.eye(6))
    return J_pinv @ v_des


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

    # ── Resolve IDs ─────────────────────────────────────────────────────────
    pinch_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     "pinch")
    box_id          = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "target_box")
    gripper_act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    jids          = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    j) for j in WBC_JOINTS]
    dof_ids       = [model.jnt_dofadr[jid]  for jid in jids]
    qpos_ids      = [model.jnt_qposadr[jid] for jid in jids]
    vel_act_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in VEL_ACTUATORS]
    motor_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in MOTOR_ACTUATORS]

    base_x_qpos          = qpos_ids[0]
    base_x_vel_id        = vel_act_ids[0]
    base_y_vel_id        = vel_act_ids[1]
    arm_qpos             = qpos_ids[3:]
    shoulder_pan_vel_id  = vel_act_ids[3]
    shoulder_lift_vel_id = vel_act_ids[4]
    shoulder_pan_qpos    = qpos_ids[3]

    # ── State vars ───────────────────────────────────────────────────────────
    state             = "DRIVE"
    ticks             = 0
    scan_dir          = 1
    scan_tmr          = 0
    servo_tmr         = 0
    perceived_box_pos = None
    tgt_pos           = None
    tgt_vel           = np.zeros(3)
    lift_target       = None
    
    # Capture initial pose to hold during drive
    q_init = data.qpos.copy()

    print("\n=== Kinematic WBC + Active Perception ===")
    print("  Controller  : Pseudo-Inverse Jacobian WBC (velocity-based, 9 DOFs)")
    print("  Detection   : wrist_cam (D405, fovy=69°)  scan + visual servo + depth")
    print("  Base joints : base_x/base_y slide (WBC holonomic model)")
    print(f"  Table at    : X={TABLE_X} m   Stop at: X={ARM_STOP} m\n")

    headless = "--headless" in sys.argv
    if headless:
        print("Running in headless mode...")
        v = None
    else:
        v = viewer.launch_passive(model, data)
        v.cam.lookat[:] = [1.0, 0.0, 0.8]
        v.cam.distance  = 4.5
        v.cam.elevation = -15

    try:
        while (v.is_running() if v else True):

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 1 — DRIVE BASE TO TABLE (Hold arm at initial pose)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if state == "DRIVE":
                zero_motor(data, motor_act_ids)
                base_x = data.qpos[base_x_qpos]

                # ── Base drive: two-speed approach ──
                if base_x < SLOW_START:
                    speed = DRIVE_SPEED
                elif base_x < ARM_STOP:
                    speed = SLOW_SPEED
                else:
                    speed = 0.0
                data.ctrl[base_x_vel_id] = speed
                data.ctrl[base_y_vel_id] = 0.0

                # ── Arm: actively hold at initial pose to prevent falling ──
                for i, (qi, vid) in enumerate(zip(arm_qpos, vel_act_ids[3:])):
                    data.ctrl[vid] = 5.0 * (q_init[qi] - data.qpos[qi])

                ticks += 1
                if ticks % 200 == 0:
                    dist_to_table = TABLE_X - base_x
                    print(f"  [DRIVE] base_x={base_x:.3f} m  "
                          f"dist_to_table={dist_to_table:.3f} m  "
                          f"speed={speed:.2f} m/s")

                if base_x >= ARM_STOP:
                    data.ctrl[base_x_vel_id] = 0.0
                    print(f"\n[DRIVE → SETTLE_ARM]  Base stopped at x={base_x:.3f} m "
                          f"(arm reach ≈ {TABLE_X - base_x:.2f} m to box)\n")
                    state = "SETTLE_ARM"
                    ticks = 0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 2 — SETTLE ARM TO SCAN SEED POSE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif state == "SETTLE_ARM":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0
                # Move arm to wrist-scan seed pose
                for i, (qi, vid) in enumerate(zip(arm_qpos, vel_act_ids[3:])):
                    data.ctrl[vid] = 5.0 * (SCAN_SEED_Q[i] - data.qpos[qi])
                ticks += 1
                if ticks >= SETTLE_STEPS:
                    state    = "WRIST_SCAN"
                    ticks    = 0
                    scan_dir = 1
                    scan_tmr = 0
                    print(f"[SETTLE_ARM → WRIST_SCAN]  "
                          f"Starting shoulder_pan sweep [{CFG.scan_start:.1f} → {CFG.scan_end:.1f}] rad …")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 3 — WRIST CAM SCAN + SERVO
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif state == "WRIST_SCAN":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0

                # ── Actively hold non-scanning joints ──
                for i in [2, 3, 4, 5]:
                    qi = arm_qpos[i]; vid = vel_act_ids[3+i]
                    data.ctrl[vid] = 5.0 * (SCAN_SEED_Q[i] - data.qpos[qi])
                data.ctrl[shoulder_lift_vel_id] = 5.0 * (SCAN_SEED_Q[1] - data.qpos[arm_qpos[1]])

                current_pan = data.qpos[shoulder_pan_qpos]
                if current_pan >= CFG.scan_end:    scan_dir = -1
                elif current_pan <= CFG.scan_start: scan_dir =  1
                data.ctrl[shoulder_pan_vel_id]  = scan_dir * CFG.scan_vel

                scan_tmr += 1
                if scan_tmr % CFG.render_every == 0:
                    mujoco.mj_forward(model, data)
                    rgb, _ = render_rgbd(renderer, model, data, CFG.cam_name)
                    found, cx, cy, _ = detect_red_object(rgb, CFG)
                    if found:
                        print(f"  [WRIST_SCAN] Box detected!  "
                              f"pixel=({cx:.1f},{cy:.1f})  pan={current_pan:.3f} rad")
                        state    = "WRIST_SERVO"
                        ticks    = 0
                        servo_tmr = CFG.servo_render_every - 1
                    elif scan_tmr % (CFG.render_every * 20) == 0:
                        print(f"  [WRIST_SCAN] pan={current_pan:.3f} rad — searching …")

            elif state == "WRIST_SERVO":
                zero_motor(data, motor_act_ids)
                data.ctrl[base_x_vel_id] = 0.0
                data.ctrl[base_y_vel_id] = 0.0

                # ── Actively hold non-scanning joints ──
                for i in [2, 3, 4, 5]:
                    qi = arm_qpos[i]; vid = vel_act_ids[3+i]
                    data.ctrl[vid] = 5.0 * (SCAN_SEED_Q[i] - data.qpos[qi])

                servo_tmr += 1
                ticks     += 1
                if servo_tmr % CFG.servo_render_every == 0:
                    mujoco.mj_forward(model, data)
                    rgb, _ = render_rgbd(renderer, model, data, CFG.cam_name)
                    found, cx, cy, _ = detect_red_object(rgb, CFG)

                    if not found:
                        print("  [WRIST_SERVO] Object lost → WRIST_SCAN")
                        state    = "WRIST_SCAN"
                        ticks    = 0
                        scan_tmr = 0
                    else:
                        err_x  = cx - IMG_W / 2.0
                        err_y  = cy - IMG_H / 2.0
                        px_err = np.hypot(err_x, err_y)
                        if ticks % 60 == 0:
                            print(f"  [WRIST_SERVO] centroid=({cx:.1f},{cy:.1f})"
                                  f"  err={px_err:.1f} px")
                        if px_err < CFG.servo_pix_thr:
                            print(f"  [WRIST_SERVO] Converged ({px_err:.1f} px) → LOCALISE")
                            state = "LOCALISE"
                            ticks = 0
                        else:
                            # Camera is physically mounted sideways on the UR5e wrist!
                            # Pan controls vertical image axis (cy), Lift controls horizontal (cx)
                            data.ctrl[shoulder_pan_vel_id]  = -CFG.servo_gain_pan  * err_y
                            data.ctrl[shoulder_lift_vel_id] = -CFG.servo_gain_lift * err_x

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 4 — LOCALISE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif state == "LOCALISE":
                for i in vel_act_ids: data.ctrl[i] = 0.0
                zero_motor(data, motor_act_ids)
                mujoco.mj_forward(model, data)
                rgb, depth = render_rgbd(renderer, model, data, CFG.cam_name)
                found, cx, cy, mask = detect_red_object(rgb, CFG)

                print("\n" + "═" * 58)
                print("  LOCALISE — wrist_cam depth un-projection")
                print("═" * 58)

                gt = data.xpos[box_id].copy()
                if not found:
                    print("  WARNING: object not visible — using ground truth")
                    perceived_box_pos = gt.copy()
                else:
                    pt, raw_d = unproject_pixel_to_world(
                        model, data, CFG.cam_name, cx, cy, depth)
                    if pt is None:
                        print(f"  WARNING: invalid depth ({raw_d:.4f} m) — using ground truth")
                        perceived_box_pos = gt.copy()
                    else:
                        perceived_box_pos = pt.copy()
                        perceived_box_pos[2] -= CFG.box_half_height
                        err3d = np.linalg.norm(perceived_box_pos - gt)
                        print(f"  Pixel          : ({cx:.1f}, {cy:.1f})")
                        print(f"  Raw depth      : {raw_d:.4f} m")
                        print(f"  Perceived pos  : {np.round(perceived_box_pos, 4)}")
                        print(f"  Ground truth   : {np.round(gt, 4)}")
                        print(f"  3-D error      : {err3d*100:.2f} cm")

                        if not headless:
                            try:
                                import matplotlib, matplotlib.pyplot as plt
                                matplotlib.use("TkAgg")
                                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                                fig.suptitle("Active Perception — wrist_cam (Kinematic WBC)", fontsize=12)
                                axes[0].imshow(rgb); axes[0].set_title("RGB")
                                axes[0].plot(cx, cy, "g+", ms=20, mew=3)
                                axes[0].axhline(IMG_H/2, color="cyan", lw=0.8, ls="--")
                                axes[0].axvline(IMG_W/2, color="cyan", lw=0.8, ls="--")
                                axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Red Mask")
                                axes[1].plot(cx, cy, "r+", ms=20, mew=3)
                                dv = depth.copy(); dv[dv<=0] = np.nan
                                im = axes[2].imshow(dv, cmap="plasma"); axes[2].set_title("Depth (m)")
                                fig.colorbar(im, ax=axes[2], shrink=0.8)
                                plt.tight_layout()
                                plt.show(block=False)
                                plt.pause(0.5)
                            except ImportError:
                                pass

                print("═" * 58)
                print(f"\nHandoff → Kinematic WBC.  target={np.round(perceived_box_pos,4)}\n")
                state = "REACHING"
                ticks = 0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 5 — KINEMATIC WBC GRASP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif state == "REACHING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 0.0
                tgt_pos = perceived_box_pos.copy(); tgt_pos[2] += 0.005
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel)
                for i, aid in enumerate(vel_act_ids): data.ctrl[aid] = dq[i]
                dist = np.linalg.norm(tgt_pos - data.site_xpos[pinch_id])
                if dist < 0.015:
                    state = "GRASPING"; ticks = 0
                    print(f"Reached target (dist={dist*100:.1f} cm) → GRASPING")

            elif state == "GRASPING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 255.0
                tgt_pos = perceived_box_pos.copy(); tgt_pos[2] += 0.005
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel)
                for i, aid in enumerate(vel_act_ids): data.ctrl[aid] = dq[i]
                ticks += 1
                if ticks > 500:
                    lift_target = perceived_box_pos.copy()
                    lift_target[2] += 0.20
                    state = "LIFTING"; ticks = 0
                    print("Grasped → LIFTING")

            elif state == "LIFTING":
                zero_motor(data, motor_act_ids)
                data.ctrl[gripper_act_id] = 255.0
                dq = kinematic_wbc(model, data, pinch_id, dof_ids, lift_target, tgt_vel)
                for i, aid in enumerate(vel_act_ids): data.ctrl[aid] = dq[i]
                ticks += 1
                if ticks == 1000:
                    print(f"\n✅ Done.  Box Z = {data.xpos[box_id][2]:.3f} m")
                    state = "DONE"

            elif state == "DONE":
                if headless: break
                pass

            mujoco.mj_step(model, data)
            if v: v.sync()
            time.sleep(model.opt.timestep)

    finally:
        if v:
            v.close()


if __name__ == "__main__":
    main()
