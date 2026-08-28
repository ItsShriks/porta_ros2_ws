#!/usr/bin/env python3
"""
figure8_wbc.py
==============
MMO-700 + UR5e  •  mmo_700_wbc.xml

Whole-Body Control Demonstration: Drawing a Figure-8 (Lemniscate)
──────────────────────────────────────────────────────────────────
References:
  [1] "Whole-Body MPC for Mobile Manipulation" (paper.pdf, Zhao et al.)
       Unified task-space + time-dimension weighted WBC for simultaneous
       base and EE trajectory tracking.
  [2] "Omnidirectional Wheel-Legged WBC" (2509.14010v1.pdf)
       Uses the Lissajous lemniscate as benchmark trajectory for validating
       unified whole-body motion control of base + manipulator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 — HORIZONTAL Figure-8 ON TABLE SURFACE  (Whole-Body)
─────────────────────────────────────────────────────────────
  Step A — APPROACH: Base drives forward in +X to ARM_STOP_X (≈1.0 m).
            Arm stays in retracted drive pose.

  Step B — TABLE FIGURE-8: EE traces a lemniscate ON the table top
            surface (world Z = 0.82 m = table top + 2 cm clearance).
            The figure-8 is inscribed in the table XY footprint.

            Whole-body coordination:
              • base_y slides laterally (±BASE_Y_ASSIST m) to extend
                the lateral range of the figure-8 beyond pure arm reach.
              • arm IK (kinematic_wbc) tracks the full 3-D EE target
                (x, y, Z_table) simultaneously, compensating for base Y motion.
            → Both base and arm contribute to the lemniscate path.

  Lemniscate (table XY plane, constant Z):
      x_ee(t) = TABLE_X + A·sin(ω·t)·cos(ω·t)
      y_ee(t) = TABLE_Y + A·sin(ω·t)
      z_ee    = TABLE_Z_TOP + EE_CLEARANCE   (constant)
      y_base(t) = BASE_Y_ASSIST·sin(ω·t)    (base lateral assist)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 2 — VERTICAL Figure-8 IN FRONTAL PLANE  (Whole-Body)
───────────────────────────────────────────────────────────
  The figure-8 stands upright in the robot's frontal (YZ) plane —
  as seen from in front of the robot, facing +X (where the front LiDAR is).

  Whole-body coordination (as requested):
    • base_y slides LATERALLY to provide the Y (side-to-side) component.
    • arm moves in X (forward reach) AND Z (height) simultaneously.
    • Combined: base Y-sweep + arm XZ-motion traces the YZ lemniscate.

  Lemniscate (frontal YZ plane):
      y(t) = cy + A·sin(ω·t)·cos(ω·t)   → base_y tracks this
      z(t) = cz + A·sin(ω·t)             → arm Z
      x(t) = cx (fixed depth)             → arm X (constant reach)
      arm WBC tracks full (x, y, z) EE — arm handles X and Z directly,
      Y is aided by base, arm compensates the residual.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kinematic WBC (arm):
  v_des = Kp_pos·(p_tgt − p_ee) + ṗ_tgt
  dq    = J†_arm · v_des   (damped pseudoinverse, arm DOFs only)

Run:
  cd /home/shrikar/porta_ros2_ws
  python3 scripts/figure8_wbc.py [--headless] [--phase1_only] [--phase2_only]
                                  [--loops N] [--scale S] [--speed S]
"""

import argparse
import time
from collections import deque
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_DIR / "robots" / "mmo_700_wbc.xml"

# ── Actuator / joint names ────────────────────────────────────────────────────
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

# ── Table geometry (from mmo_700_wbc.xml) ────────────────────────────────────
# <body name="table" pos="2.0 0 0">
#   <geom name="table_top" size="0.40 0.35 0.025" pos="0 0 0.775"/>
# table top surface world Z = 0.775 + 0.025 = 0.80 m
TABLE_X       = 2.0        # table centre world X [m]
TABLE_Y       = 0.0        # table centre world Y [m]
TABLE_Z_TOP   = 0.80       # table top surface world Z [m]
EE_CLEARANCE  = 0.025      # EE height above table surface [m]
EE_TABLE_Z    = TABLE_Z_TOP + EE_CLEARANCE   # = 0.825 m

# ── Phase 1: Table figure-8 + orbital base motion ────────────────────────────
# EE lemniscate on table top (asymmetric: smaller X-depth, larger Y-lateral)
H8_AMP_X      = 0.10       # m — half-stroke in X (depth on table)
H8_AMP_Y      = 0.18       # m — half-stroke in Y (lateral on table)
H8_OMEGA      = 0.40       # rad/s — T ≈ 15.7 s per full figure-8
ARM_VEL_CLAMP = 3.0        # rad/s — max arm joint velocity clamp (singularity guard)
H8_AMPLITUDE  = max(H8_AMP_X, H8_AMP_Y)  # used for banner display

# Base ORBITAL motion around the table (whole-body aspect of Phase 1):
#   The base traces a smooth arc of ±ORBIT_PHI_MAX around the table,
#   always yawing to face the table centre. The arm re-solves IK every
#   step to maintain the EE on the table surface despite base movement.
ORBIT_CX        = TABLE_X          # orbit centre = table centre X [m]
ORBIT_CY        = TABLE_Y          # orbit centre = table centre Y [m]
ORBIT_R         = 0.85             # orbit radius from table centre [m]
ORBIT_PHI_MAX   = np.pi / 3.0     # ±60° arc sweep (front + both sides)
ORBIT_OMEGA_FAC = 0.5              # orbit sweeps at OMEGA_FAC × h8_w
# Approach target: orbit start position (directly in front of table)
ORBIT_X_START   = ORBIT_CX - ORBIT_R   # = TABLE_X − ORBIT_R = 1.15 m
ORBIT_Y_START   = 0.0
DRIVE_SPEED     = 0.35             # m/s — base approach speed

# ── Phase 2: Vertical frontal figure-8 parameters ────────────────────────────
V8_AMPLITUDE  = 0.22       # m — half-height/half-width of each lobe
V8_OMEGA      = 0.45       # rad/s — T ≈ 14.0 s per full figure-8
V8_CZ         = 1.05       # m — vertical centre (world Z) of the figure-8
BASE_Y_AMP_V8 = 0.15       # m — base Y amplitude for the lateral YZ component

# ── Arm poses ─────────────────────────────────────────────────────────────────
# Retracted pose for driving (safe)
DRIVE_POSE    = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
# Home pose for table reach (arm extended forward, wrist down)
ARM_HOME_Q    = [0.0, -1.0,    1.5,    -2.05,   -1.57,   0.0]
# Phase 2 start: arm raised, shoulder pan neutral
ARM_REACH_Q   = [0.0, -1.2,    1.2,    -1.6,    -1.57,   0.0]
SETTLE_STEPS  = 1500

# ── Kinematic IK gains ────────────────────────────────────────────────────────
KP_IK_POS = 30.0
KP_IK_ROT = 8.0

# ── Orientation targets ───────────────────────────────────────────────────────
# Top-down: wrist points straight down — used for table phase (EE traces on table)
R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
# Forward: wrist points in robot +X — used for vertical frontal phase
R_FORWARD = np.array([[0, 0, 1], [0,  1, 0], [-1, 0,  0]], dtype=float)


# ── Trajectory tracing visualization ────────────────────────────────────────
N_PATH_DOTS   = 200    # reference path dots drawn before simulation starts
N_TRAIL_DOTS  = 400    # max live EE trail dots kept in scene
TRAIL_EVERY   = 3     # add a trail dot every N sim steps
# Colours (RGBA)
COL_PATH_P1   = np.array([0.15, 0.85, 0.15, 0.55])   # green  — P1 reference path
COL_PATH_P2   = np.array([0.20, 0.60, 1.00, 0.55])   # blue   — P2 reference path
COL_TRAIL     = np.array([1.00, 0.30, 0.10, 0.90])   # red/orange — live EE trail
COL_TGT       = np.array([1.00, 1.00, 0.00, 0.80])   # yellow — current EE target
PATH_DOT_R    = 0.010   # radius of reference path spheres [m]
TRAIL_DOT_R   = 0.007   # radius of live EE trail spheres [m]
TGT_DOT_R     = 0.015   # radius of current target marker [m]


def _add_sphere(scn, pos, radius, rgba):
    """Append one sphere to a MjvScene user scene. Returns True on success."""
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return True


def draw_reference_path(v, path_pts, rgba, dot_r=PATH_DOT_R):
    """
    Pre-draw the complete figure-8 reference path as small spheres in
    v.user_scn.  Returns the ngeom count AFTER the path is drawn, so
    the caller can reset to it when refreshing the trail.
    """
    if v is None:
        return 0
    with v.lock():
        v.user_scn.ngeom = 0     # clear everything first
        for pt in path_pts:
            _add_sphere(v.user_scn, pt, dot_r, rgba)
        return v.user_scn.ngeom  # = number of path dots added


def sample_path(traj_fn, args_tuple, n=N_PATH_DOTS):
    """
    Sample n evenly-spaced points along exactly one full figure-8 period.
    traj_fn(t, *args_tuple) must return (pos, vel[, ...]) where pos is (3,).
    """
    omega = args_tuple[-1]
    T     = 2 * np.pi / omega
    pts   = []
    for i in range(n + 1):           # +1 so last point closes the loop
        t       = i / n * T
        result  = traj_fn(t, *args_tuple)
        pos     = result[0]          # first element is always pos (3,)
        pts.append(pos.copy())
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory generators
# ══════════════════════════════════════════════════════════════════════════════

def lemniscate_table(t, cx, cy, Ax, Ay, omega):
    """
    Asymmetric Lissajous figure-8 in the horizontal XY plane at constant Z.

    Uses separate amplitudes for X (depth) and Y (lateral) because the arm
    is near its reach limit in X so the X stroke must be smaller:
        x(t) = cx + Ax·sin(ω·t)·cos(ω·t)   depth on table (small ±Ax/2)
        y(t) = cy + Ay·sin(ω·t)             lateral on table (larger ±Ay)
        z    = EE_TABLE_Z                    constant

    Returns pos (3,), vel (3,)
    """
    s  = np.sin(omega * t)
    c  = np.cos(omega * t)
    x  = cx + Ax * s * c           # depth oscillation
    y  = cy + Ay * s               # lateral oscillation
    dx = Ax * omega * (c*c - s*s)
    dy = Ay * omega * c
    return np.array([x, y, EE_TABLE_Z]), np.array([dx, dy, 0.0])


def lemniscate_frontal(t, cx, cy, cz, A, omega):
    """
    Lissajous figure-8 in the frontal YZ plane (vertical, seen from front of robot).

    The figure-8 stands upright:
        y(t) = cy + A·sin(ω·t)·cos(ω·t)   lateral (base slides in Y)
        z(t) = cz + A·sin(ω·t)             vertical (arm moves in Z)
        x(t) = cx                           constant forward reach (arm in X)

    Returns pos (3,), vel (3,),
            y_base_target (float)   — desired base Y position for the Y component
    """
    s  = np.sin(omega * t)
    c  = np.cos(omega * t)
    y  = cy + A * s * c            # = cy + (A/2)·sin(2ωt) — lateral
    z  = cz + A * s                # vertical
    dy = A * omega * (c*c - s*s)
    dz = A * omega * c
    return (np.array([cx, y, z]),
            np.array([0.0, dy, dz]),
            y)                     # y_base_target = the full Y position


# ══════════════════════════════════════════════════════════════════════════════
# Controller helpers
# ══════════════════════════════════════════════════════════════════════════════

def rot_err(R_t, R_c):
    """Rotation error as axis-angle vector."""
    Re  = R_t @ R_c.T
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0))
    if abs(ang) > 1e-6:
        ax = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]]) / (2*np.sin(ang))
        return ang * ax
    return np.zeros(3)


def kinematic_wbc(model, data, pinch_id, dof_ids, tgt_pos, tgt_vel,
                  R_target=None, kp_pos=None, kp_rot=None):
    """
    Kinematic Whole-Body Control for the 6-DOF arm.

    Velocity-resolved IK (Jacobian pseudoinverse):
        v_des = Kp_pos·(p_tgt − p_ee) + ṗ_tgt   [position]
                Kp_rot·e_rot                       [orientation]
        dq    = J†_arm · v_des

    Only the arm DOFs [3:] of the 9-DOF system are computed.
    The base is commanded separately via velocity actuators.
    """
    if R_target is None:
        R_target = R_TOPDOWN
    if kp_pos is None:
        kp_pos = KP_IK_POS
    if kp_rot is None:
        kp_rot = KP_IK_ROT

    arm_dofs = dof_ids[3:]
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, pinch_id)
    J6 = np.vstack([Jp[:, arm_dofs], Jr[:, arm_dofs]])

    p_err = tgt_pos - data.site_xpos[pinch_id]
    r_err = rot_err(R_target, data.site_xmat[pinch_id].reshape(3, 3))
    v_des = np.concatenate([kp_pos * p_err + tgt_vel,
                            kp_rot * r_err])
    dq = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-4 * np.eye(6), v_des)
    return dq   # shape (6,)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MMO-700 Figure-8 WBC: table surface + vertical frontal planes")
    parser.add_argument("--headless",    action="store_true",
                        help="Run without viewer window")
    parser.add_argument("--phase1_only", action="store_true",
                        help="Run only Phase 1: horizontal figure-8 on table top")
    parser.add_argument("--phase2_only", action="store_true",
                        help="Run only Phase 2: vertical frontal figure-8 (base Y + arm XZ)")
    parser.add_argument("--loops",  type=int,   default=3,
                        help="Number of complete figure-8 loops per phase (default: 3)")
    parser.add_argument("--scale",  type=float, default=1.0,
                        help="Scale factor for lemniscate amplitude (default: 1.0)")
    parser.add_argument("--speed",  type=float, default=1.0,
                        help="Speed multiplier for lemniscate frequency (default: 1.0)")
    parser.add_argument("--no_trace", action="store_true",
                        help="Disable trajectory path preview and live EE trail")
    args = parser.parse_args()
    do_trace = not args.headless and not args.no_trace

    h8_A = H8_AMPLITUDE * args.scale
    h8_w = H8_OMEGA     * args.speed
    v8_A = V8_AMPLITUDE * args.scale
    v8_w = V8_OMEGA     * args.speed
    h8_period = 2 * np.pi / h8_w
    v8_period = 2 * np.pi / v8_w

    run_p1 = not args.phase2_only
    run_p2 = not args.phase1_only

    print(f"\n{'═'*72}")
    print(f"  MMO-700 + UR5e  —  Figure-8 Whole-Body Control Demonstration")
    print(f"{'═'*72}")
    print(f"  Model   : {MODEL_PATH.name}")
    print(f"  Loops   : {args.loops} per phase")
    print(f"  Phase 1 : Table surface figure-8  Ax={H8_AMP_X*args.scale:.3f}m  Ay={H8_AMP_Y*args.scale:.3f}m  ω={h8_w:.3f}rad/s  T={h8_period:.1f}s")
    print(f"            Base ORBITS ±{int(np.degrees(ORBIT_PHI_MAX))}° around table (R={ORBIT_R}m), arm traces EE on table Z={EE_TABLE_Z:.3f}m")
    print(f"  Phase 2 : Vertical frontal figure-8  A={v8_A:.3f}m  ω={v8_w:.3f}rad/s  T={v8_period:.1f}s")
    print(f"            Base moves in Y, arm moves in X+Z simultaneously")
    print(f"{'═'*72}\n")

    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # ── ID look-ups ────────────────────────────────────────────────────────────
    pinch_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     "pinch")
    jids          = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    j) for j in WBC_JOINTS]
    dof_ids       = [model.jnt_dofadr[jid]  for jid in jids]
    qpos_ids      = [model.jnt_qposadr[jid] for jid in jids]
    motor_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in MOTOR_ACTUATORS]
    vel_act_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in VEL_ACTUATORS]

    base_x_qpos     = qpos_ids[0]
    base_y_qpos     = qpos_ids[1]
    base_yaw_qpos   = qpos_ids[2]
    arm_qpos        = qpos_ids[3:]
    base_x_vel_id   = vel_act_ids[0]
    base_y_vel_id   = vel_act_ids[1]
    base_yaw_vel_id = vel_act_ids[2]
    arm_vel_ids     = vel_act_ids[3:]

    dt = model.opt.timestep

    # ── Viewer ─────────────────────────────────────────────────────────────────
    if args.headless:
        print("Running in headless mode …\n")
        v = None
    else:
        v = viewer.launch_passive(model, data)
        v.cam.lookat[:] = [1.0, 0.0, 0.9]
        v.cam.distance  = 5.5
        v.cam.elevation = -20
        v.cam.azimuth   = 30

    # ── Visualization setup (sites + trajectory tracing) ─────────────────────
    if v is not None:
        # MuJoCo 3.x: site visibility is controlled via opt.sitegroup[]
        # (a 6-element group bitmask), NOT the removed mjVIS_SITE flag.
        # Enable all 6 site groups so the 'pinch' sphere is always rendered.
        v.opt.sitegroup[:] = 1
        # Show contact points for visual richness
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    # Pre-compute reference paths (pure maths — no simulation steps needed)
    h8_Ax_s = H8_AMP_X * args.scale
    h8_Ay_s = H8_AMP_Y * args.scale
    v8_A_s  = V8_AMPLITUDE * args.scale
    _p1_path_args = (TABLE_X, TABLE_Y, h8_Ax_s, h8_Ay_s, h8_w)
    # Phase 2 path centre is resolved at runtime (after settle); pre-compute
    # a placeholder at origin — it is redrawn with correct centre before P2.
    _p2_path_args_placeholder = (1.0, 0.0, V8_CZ, v8_A_s, v8_w)

    p1_ref_path = sample_path(lemniscate_table,  _p1_path_args)
    p2_ref_path = sample_path(lemniscate_frontal, _p2_path_args_placeholder)

    # ngeom watermark = number of path dots committed to user_scn
    n_path_p1 = 0
    n_path_p2 = 0
    if do_trace:
        n_path_p1 = draw_reference_path(v, p1_ref_path, COL_PATH_P1)
        print(f"  [VIZ] Phase-1 reference path drawn ({n_path_p1} dots, green)")
        print(f"  [VIZ] Site visibility ON — pinch sphere visible in viewer\n")

    # Deque holds recent EE world positions for the live trail
    ee_trail: deque = deque(maxlen=N_TRAIL_DOTS)

    def is_running():
        return v.is_running() if v else True

    def step_sim():
        mujoco.mj_step(model, data)
        if v:
            v.sync()
        time.sleep(dt * 0.3)

    def zero_motors():
        for i in motor_act_ids:
            data.ctrl[i] = 0.0

    def set_arm_vel(dq_arm, clamp=ARM_VEL_CLAMP):
        for i, vid in enumerate(arm_vel_ids):
            data.ctrl[vid] = float(np.clip(dq_arm[i], -clamp, clamp))

    def set_base_vel(vx, vy, vyaw=0.0):
        data.ctrl[base_x_vel_id]    = vx
        data.ctrl[base_y_vel_id]    = vy
        data.ctrl[base_yaw_vel_id]  = vyaw

    def stop_base():
        set_base_vel(0.0, 0.0, 0.0)

    def settle_arm_to(target_q, n_steps, label=""):
        if label:
            print(f"[SETTLE] {label} …")
        for _ in range(n_steps):
            if not is_running():
                break
            zero_motors()
            stop_base()
            for i, (qi, vid) in enumerate(zip(arm_qpos, arm_vel_ids)):
                data.ctrl[vid] = 5.0 * (target_q[i] - data.qpos[qi])
            step_sim()
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[pinch_id].copy()
        print(f"  EE after settle : {np.round(ee, 3)}\n")
        return ee

    # ══════════════════════════════════════════════════════════════════════════
    # INITIAL SETTLE — arm to drive/retract pose
    # ══════════════════════════════════════════════════════════════════════════
    settle_arm_to(DRIVE_POSE, SETTLE_STEPS, "Arm to retracted drive pose")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — TABLE SURFACE Figure-8  (base approach + whole-body WBC)
    # ══════════════════════════════════════════════════════════════════════════
    if run_p1:
        print(f"{'─'*72}")
        print(f"  PHASE 1 — Table Surface Figure-8  (orbital whole-body WBC)")
        print(f"  Step A: Base approaches orbit start at X≈{ORBIT_X_START:.2f}m")
        print(f"          (orbit centre: table at X={ORBIT_CX}m, orbit radius {ORBIT_R}m)")
        print(f"  Step B: Base ORBITS ±{np.degrees(ORBIT_PHI_MAX):.0f}° around the table")
        print(f"          sweeping: front → right side → front → left side → …")
        print(f"          Base yaw always faces table centre.")
        print(f"          Arm IK re-solves every step — EE traces figure-8 ON table")
        print(f"          surface (Z={EE_TABLE_Z}m) regardless of base orbit position.")
        print(f"  Loops: {args.loops}   h8_period: {h8_period:.1f}s   EE amp Ax={H8_AMP_X*args.scale:.3f}m Ay={H8_AMP_Y*args.scale:.3f}m")
        print(f"{'─'*72}\n")

        # ── Step A: Drive base forward to ORBIT_X_START ──────────────────────────
        print(f"  [APPROACH] Driving to orbit start ({ORBIT_X_START:.2f}, {ORBIT_Y_START:.2f}) …")
        while is_running():
            bx = data.qpos[base_x_qpos]
            if bx >= ORBIT_X_START:
                break
            zero_motors()
            set_base_vel(DRIVE_SPEED, 0.0, 0.0)
            for i, (qi, vid) in enumerate(zip(arm_qpos, arm_vel_ids)):
                data.ctrl[vid] = 5.0 * (DRIVE_POSE[i] - data.qpos[qi])
            step_sim()

        stop_base()
        mujoco.mj_forward(model, data)
        print(f"  [APPROACH] Done — base at ({data.qpos[base_x_qpos]:.3f}, {data.qpos[base_y_qpos]:.3f}) m\n")

        # ── Settle arm to forward-reach pose (wrist pointing down to table) ───
        settle_arm_to(ARM_HOME_Q, SETTLE_STEPS, "Arm to forward-reach pose for table")

        mujoco.mj_forward(model, data)

        # Figure-8 on table surface (world frame) — fixed lemniscate target
        ee_now = data.site_xpos[pinch_id]
        print(f"  EE after settle                   : {np.round(ee_now, 3)}")
        print(f"  Table surface lemniscate centre   : ({TABLE_X:.3f}, {TABLE_Y:.3f}, {EE_TABLE_Z:.3f})")
        print(f"  Orbit: R={ORBIT_R}m  phi_max=±{np.degrees(ORBIT_PHI_MAX):.0f}°  "
              f"omega_orb={ORBIT_OMEGA_FAC*h8_w:.3f} rad/s\n")

        # ── Draw orbit arc in viewer (small white spheres on the floor) ────────
        if do_trace and v is not None:
            orbit_arc_pts = []
            for k in range(81):   # 80 steps = smooth arc
                phi_k = ORBIT_PHI_MAX * np.sin(np.linspace(-np.pi/2, np.pi/2, 81)[k])
                ox = ORBIT_CX + ORBIT_R * np.cos(phi_k + np.pi)
                oy = ORBIT_CY + ORBIT_R * np.sin(phi_k + np.pi)
                orbit_arc_pts.append(np.array([ox, oy, 0.03]))  # 3 cm above floor
            # Append orbit arc to current user_scn (after figure-8 path dots)
            with v.lock():
                for pt in orbit_arc_pts:
                    _add_sphere(v.user_scn, pt, 0.018,
                                np.array([0.9, 0.9, 0.9, 0.50], dtype=np.float32))  # white
                n_path_p1 = v.user_scn.ngeom   # update watermark to include orbit dots
            print(f"  [VIZ] Orbit arc drawn (white dots on floor, {len(orbit_arc_pts)} dots)\n")

        total_dur = args.loops * h8_period
        sim_t = 0.0
        step  = 0
        print_every = max(1, int(0.5 / dt))
        ee_errs = []
        h8_Ax = H8_AMP_X * args.scale
        h8_Ay = H8_AMP_Y * args.scale
        omega_orbit = ORBIT_OMEGA_FAC * h8_w   # orbit angular frequency

        while sim_t <= total_dur and is_running():
            zero_motors()

            # ── Lemniscate target on table surface ────────────────────────────
            tgt_pos, tgt_vel = lemniscate_table(sim_t, TABLE_X, TABLE_Y, h8_Ax, h8_Ay, h8_w)

            # ── Base ORBITAL motion: sinusoidal arc ±phi_max around table ────
            # phi(t) oscillates ±ORBIT_PHI_MAX using a sinusoid → smooth sweep:
            #   right-of-table → front → left-of-table → front → …
            phi     = ORBIT_PHI_MAX * np.sin(omega_orbit * sim_t)
            dphi_dt = ORBIT_PHI_MAX * omega_orbit * np.cos(omega_orbit * sim_t)

            base_x_des = ORBIT_CX + ORBIT_R * np.cos(phi + np.pi)
            base_y_des = ORBIT_CY + ORBIT_R * np.sin(phi + np.pi)
            yaw_des    = phi   # arctan2(table_centre - base) = phi analytically

            # World-frame feedforward + P-correction
            vx_ff = -ORBIT_R * np.sin(phi + np.pi) * dphi_dt
            vy_ff =  ORBIT_R * np.cos(phi + np.pi) * dphi_dt

            bx   = data.qpos[base_x_qpos]
            by   = data.qpos[base_y_qpos]
            byaw = data.qpos[base_yaw_qpos]

            ramp = min(1.0, sim_t / 2.0)
            vx   = ramp * (vx_ff + 3.0 * (base_x_des - bx))
            vy   = ramp * (vy_ff + 3.0 * (base_y_des - by))

            yaw_err = (yaw_des - byaw + np.pi) % (2*np.pi) - np.pi
            vyaw    = ramp * 4.0 * yaw_err
            set_base_vel(vx, vy, vyaw)

            # ── Arm WBC: IK re-solved from current base position every step ──
            # EE target (table surface lemniscate) is fixed in WORLD frame.
            # As the base orbits around the table the arm continuously adapts
            # its configuration to keep the EE on the table surface.
            dq_arm = kinematic_wbc(
                model, data, pinch_id, dof_ids,
                tgt_pos, tgt_vel * ramp,
                R_target=R_TOPDOWN,
                kp_pos=32.0, kp_rot=8.0)
            set_arm_vel(dq_arm)

            # ── Logging ───────────────────────────────────────────────────────
            if step % print_every == 0:
                ee  = data.site_xpos[pinch_id]
                err = np.linalg.norm(ee - tgt_pos)
                ee_errs.append(err)
                loop_no = int(sim_t / h8_period) + 1
                print(f"  [P1 t={sim_t:5.1f}s loop={loop_no}/{args.loops}]  "
                      f"phi={np.degrees(phi):+.1f}°  base=({bx:+.3f},{by:+.3f})  "
                      f"yaw={np.degrees(byaw):+.1f}°  "
                      f"EE=({ee[0]:+.3f},{ee[1]:+.3f})  err={err*100:.1f}cm")

            # ── Live EE trail (user_scn) ──────────────────────────────────
            if do_trace and step % TRAIL_EVERY == 0 and v is not None:
                ee_trail.append(data.site_xpos[pinch_id].copy())
                with v.lock():
                    # Reset to path-dots-only watermark, then redraw trail
                    v.user_scn.ngeom = n_path_p1
                    # Current EE target (yellow)
                    _add_sphere(v.user_scn, tgt_pos, TGT_DOT_R, COL_TGT)
                    # Historical trail (red-orange, older = more transparent)
                    n = len(ee_trail)
                    for k, pt in enumerate(ee_trail):
                        alpha = 0.3 + 0.6 * (k / max(n - 1, 1))
                        rgba  = np.array([COL_TRAIL[0], COL_TRAIL[1],
                                          COL_TRAIL[2], alpha], dtype=np.float32)
                        _add_sphere(v.user_scn, pt, TRAIL_DOT_R, rgba)

            step_sim()
            sim_t += dt
            step  += 1

        stop_base()
        set_arm_vel(np.zeros(6))
        mujoco.mj_forward(model, data)

        print(f"\n  ✓ Phase 1 complete.")
        if ee_errs:
            print(f"    Mean EE tracking error : {np.mean(ee_errs)*100:.1f} cm")
            print(f"    Max  EE tracking error : {np.max(ee_errs)*100:.1f} cm\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TRANSITION — settle arm to raised forward pose for Phase 2
    # ══════════════════════════════════════════════════════════════════════════
    if run_p1 and run_p2:
        settle_arm_to(ARM_REACH_Q, int(2.0 / dt),
                      "Transitioning arm for Phase 2 (vertical frontal figure-8)")
        ee_trail.clear()   # wipe Phase 1 trail before Phase 2

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — VERTICAL FRONTAL Figure-8 in YZ plane
    #           Base moves in Y (lateral), Arm moves in X + Z simultaneously
    # ══════════════════════════════════════════════════════════════════════════
    if run_p2:
        print(f"{'─'*72}")
        print(f"  PHASE 2 — Vertical Frontal Figure-8  (YZ plane, whole-body)")
        print(f"  The figure-8 stands upright in front of the robot (+X direction,")
        print(f"  where the front LiDAR is) and is traced in the YZ plane.")
        print(f"  ┌───────────────────────────────────────────────────────────┐")
        print(f"  │  base_y  ← provides lateral (Y) sweep of the lemniscate  │")
        print(f"  │  arm X   ← holds constant forward reach depth             │")
        print(f"  │  arm Z   ← drives vertical (Z) component of the 8        │")
        print(f"  │  arm Y   ← compensates Y residual not covered by base     │")
        print(f"  └───────────────────────────────────────────────────────────┘")
        print(f"  Loops: {args.loops}   Period: {v8_period:.1f}s   Amplitude: {v8_A:.3f}m")
        print(f"{'─'*72}\n")

        # If phase2_only, settle arm first
        if args.phase2_only:
            settle_arm_to(ARM_REACH_Q, SETTLE_STEPS,
                          "Arm to forward-reach pose for Phase 2")

        mujoco.mj_forward(model, data)
        ee_p2_init = data.site_xpos[pinch_id].copy()

        # Centre of the frontal lemniscate:
        #   x = current EE X (fixed forward reach depth)
        #   y = current EE Y ≈ base Y (centred laterally)
        #   z = V8_CZ (vertical centre)
        v8_cx = float(ee_p2_init[0])   # fixed X depth (arm holds this)
        v8_cy_world = float(ee_p2_init[1])   # lateral centre in world Y
        # The base Y at start of Phase 2
        base_y_start = float(data.qpos[base_y_qpos])

        print(f"  EE start position  : {np.round(ee_p2_init, 3)}")
        print(f"  Base Y at start    : {base_y_start:.3f} m")
        print(f"  Lemniscate centre  : x={v8_cx:.3f}  y={v8_cy_world:.3f}  z={V8_CZ:.3f}")
        print(f"  Lemniscate in YZ   : y ∈ [{v8_cy_world-v8_A:.3f}, {v8_cy_world+v8_A:.3f}]  "
              f"z ∈ [{V8_CZ-v8_A:.3f}, {V8_CZ+v8_A:.3f}]\n")

        # ── Re-draw Phase 2 reference path with correct centre ────────────────
        if do_trace:
            p2_ref_path = sample_path(
                lemniscate_frontal, (v8_cx, v8_cy_world, V8_CZ, v8_A_s, v8_w))
            n_path_p2 = draw_reference_path(v, p2_ref_path, COL_PATH_P2)
            print(f"  [VIZ] Phase-2 reference path drawn ({n_path_p2} dots, blue)\n")

        # ── Position camera for frontal YZ view (side-on) ─────────────────────
        if v is not None:
            v.cam.lookat[:] = [v8_cx, v8_cy_world, V8_CZ]
            v.cam.distance  = 3.5
            v.cam.elevation = 0      # straight horizontal — best for vertical 8
            v.cam.azimuth   = 90     # looking in from +Y direction

        # Pre-compute start point for smooth ramp-in
        tgt_start, _, _ = lemniscate_frontal(0.0, v8_cx, v8_cy_world, V8_CZ, v8_A, v8_w)

        total_dur = args.loops * v8_period
        sim_t = 0.0
        step  = 0
        print_every = max(1, int(0.5 / dt))
        ee_errs = []
        ee_y_log, ee_z_log = [], []
        v8_A_scaled = v8_A   # already scaled above

        while sim_t <= total_dur and is_running():
            zero_motors()

            # ── Lemniscate target in frontal YZ plane ─────────────────────────
            tgt_pos, tgt_vel, y_base_world = lemniscate_frontal(
                sim_t, v8_cx, v8_cy_world, V8_CZ, v8_A_scaled, v8_w)

            # 1.5-second ramp-in from current EE to trajectory start
            ramp = min(1.0, sim_t / 1.5)
            if ramp < 1.0:
                tgt_pos = (1.0 - ramp) * ee_p2_init + ramp * tgt_start
                tgt_vel = tgt_vel * ramp

            # ── Base Y: tracks the Y component of the lemniscate ──────────────
            # base_y drives to y_base_world (the full Y lemniscate position).
            # The base provides most of the lateral Y motion.
            # The arm WBC compensates the Y residual (base Y lag + arm offset).
            by = data.qpos[base_y_qpos]
            # Feedforward velocity + P-correction
            vy_ff = tgt_vel[1]                         # dy/dt of lemniscate
            vy    = ramp * (vy_ff + 4.0 * (y_base_world - by))

            # Base X: hold current X position (do NOT move forward/backward)
            bx = data.qpos[base_x_qpos]
            vx = 2.0 * (bx - bx)   # = 0, but written explicitly
            # If phase2_only, base may be at 0; hold it there
            vx = 2.0 * (float(data.qpos[base_x_qpos]) * 0.0 - 0.0)   # zero
            vx = 0.0

            set_base_vel(vx, vy, 0.0)

            # ── Arm WBC: tracks full (x, y, z) EE target ─────────────────────
            # Arm handles:
            #   X = v8_cx  (constant forward depth — arm in X)
            #   Y = y_base_world (residual after base; arm in Y to compensate)
            #   Z = cz + A·sin(ωt) (vertical height — arm in Z)
            # This is the "arm moves in X and Z simultaneously" as requested.
            dq_arm = kinematic_wbc(
                model, data, pinch_id, dof_ids,
                tgt_pos, tgt_vel * ramp,
                R_target=R_FORWARD,
                kp_pos=32.0, kp_rot=8.0)
            set_arm_vel(dq_arm)

            # ── Logging ───────────────────────────────────────────────────────
            ee  = data.site_xpos[pinch_id]
            ee_y_log.append(float(ee[1]))
            ee_z_log.append(float(ee[2]))

            if step % print_every == 0:
                err = np.linalg.norm(ee - tgt_pos)
                ee_errs.append(err)
                loop_no = int(sim_t / v8_period) + 1
                print(f"  [P2 t={sim_t:5.1f}s loop={loop_no}/{args.loops}]  "
                      f"EE=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})  "
                      f"tgt=({tgt_pos[0]:+.3f},{tgt_pos[1]:+.3f},{tgt_pos[2]:+.3f})  "
                      f"err={err*100:.1f}cm  base_y={by:+.3f}m")

            # ── Live EE trail (user_scn) ──────────────────────────────────
            if do_trace and step % TRAIL_EVERY == 0 and v is not None:
                ee_trail.append(data.site_xpos[pinch_id].copy())
                with v.lock():
                    v.user_scn.ngeom = n_path_p2
                    # Current EE target (yellow)
                    _add_sphere(v.user_scn, tgt_pos, TGT_DOT_R, COL_TGT)
                    # Historical trail (red-orange)
                    n = len(ee_trail)
                    for k, pt in enumerate(ee_trail):
                        alpha = 0.3 + 0.6 * (k / max(n - 1, 1))
                        rgba  = np.array([COL_TRAIL[0], COL_TRAIL[1],
                                          COL_TRAIL[2], alpha], dtype=np.float32)
                        _add_sphere(v.user_scn, pt, TRAIL_DOT_R, rgba)

            step_sim()
            sim_t += dt
            step  += 1

        stop_base()
        set_arm_vel(np.zeros(6))

        print(f"\n  ✓ Phase 2 complete.")
        if ee_errs:
            print(f"    Mean EE tracking error : {np.mean(ee_errs)*100:.1f} cm")
        if ee_y_log:
            print(f"    EE Y range (lateral)   : [{min(ee_y_log):+.3f}, {max(ee_y_log):+.3f}] m  "
                  f"(target ±{v8_A_scaled:.3f} m)")
            print(f"    EE Z range (vertical)  : [{min(ee_z_log):+.3f}, {max(ee_z_log):+.3f}] m  "
                  f"(target ±{v8_A_scaled:.3f} m around Z={V8_CZ:.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  ✅  Figure-8 WBC Demonstration Complete!")
    print(f"  [1] WBMPC-style: base + arm unified task tracking")
    print(f"  [2] Lemniscate benchmark: horizontal (table) + vertical (frontal YZ)")
    print(f"{'═'*72}\n")

    if v:
        print("Viewer open — close window to exit.")
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(dt)
        v.close()


if __name__ == "__main__":
    main()
