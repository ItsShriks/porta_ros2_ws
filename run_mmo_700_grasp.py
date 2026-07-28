#!/usr/bin/env python3
"""
MMO-700 Grasping Scene — UR5e Edition
======================================
Launches the MMO-700 robot in MuJoCo with:
  - Official UR5e arm (menagerie kinematics, collision capsules)
  - Robotiq 2F-85 gripper
  - Pan-tilt camera + wrist D405 camera
  - Base IMU, LiDAR rangefinders, wheel and arm joint sensors

Sequence:
  1. Drive toward the table (base stops safely before contact).
  2. Perform Inverse Kinematics to reach the red box.
  3. Close the gripper to grasp the box.
  4. Lift the arm to pick up the box.

Sensor telemetry is printed to the terminal throughout.
"""
import time
from pathlib import Path
import mujoco
from mujoco import viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH  = SCRIPT_DIR / "mmo_700.xml"

# ---------- Wheel actuator indices (actuator list order) ----------
WHEEL_FL = 8
WHEEL_FR = 9
WHEEL_BL = 10
WHEEL_BR = 11

# ---------- Driving parameters ----------
APPROACH_SPEED = 4.0   # rad/s  — forward wheel spin
# Stop when robot base-X ≥ (table_x - STOP_DISTANCE).
# With STOP_DISTANCE=0.7 the robot stops at x≈1.3;
# arm base reaches ~x=1.5 — comfortably within UR5e's ~0.85m reach of the box.
STOP_DISTANCE  = 0.7

# ---------- UR5e arm joint names (menagerie convention) ----------
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ---------- Sensor name → sensordata index helpers ----------
def get_sensor_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

def read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid  = get_sensor_id(model, name)
    adr  = model.sensor_adr[sid]
    dim  = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim].copy()


def print_telemetry(model: mujoco.MjModel, data: mujoco.MjData, step: int) -> None:
    """Print sensor telemetry once every 500 steps (~1 s)."""
    if step % 500 != 0:
        return
    accel  = read_sensor(model, data, "base_accel")
    gyro   = read_sensor(model, data, "base_gyro")
    fl_rng = read_sensor(model, data, "front_lidar")[0]
    rl_rng = read_sensor(model, data, "rear_lidar")[0]
    whl_v  = [read_sensor(model, data, f"wheel_{w}_vel_sensor")[0]
               for w in ("fl", "fr", "bl", "br")]
    q_arm  = [read_sensor(model, data, f"{j.split('_joint')[0]}_pos"
                          if "_joint" in j else j + "_pos")[0]
               for j in ["shoulder_pan", "shoulder_lift", "elbow",
                          "wrist_1", "wrist_2", "wrist_3"]]
    print(f"\n--- Telemetry  (step {step}) ---")
    print(f"  IMU accel [m/s²]:  {accel}")
    print(f"  IMU gyro  [rad/s]: {gyro}")
    print(f"  LiDAR front={fl_rng:.3f} m   rear={rl_rng:.3f} m")
    print(f"  Wheels [rad/s]:    FL={whl_v[0]:.2f}  FR={whl_v[1]:.2f}  "
          f"BL={whl_v[2]:.2f}  BR={whl_v[3]:.2f}")
    q_str = "  ".join(f"{q:.3f}" for q in q_arm)
    print(f"  Arm joints [rad]:  {q_str}")


def main() -> None:
    print(f"Loading MuJoCo model from: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    # Load home keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # ---------- Body / site / actuator look-ups ----------
    base_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "base_link")
    box_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "target_box")
    pinch_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     "pinch")
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    # UR5e joint → qpos / dof address maps
    jids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in ARM_JOINTS]
    q_ids   = [model.jnt_qposadr[jid] for jid in jids]
    dof_ids = [model.jnt_dofadr[jid]  for jid in jids]

    table_x = 2.0
    state   = "DRIVING"
    ik_iterations = 0
    target_q      = None
    start_q       = None
    target_lift_q = None
    start_lift_q  = None
    step_count    = 0
    perceived_box_pos = None

    # Retracted arm pose (safe for driving)
    data.ctrl[0:6] = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    print("\n=== MMO-700 + UR5e  grasping scene ===")
    print("  Sensors: IMU (accel/gyro), LiDAR×2, wheel encoders, arm joint pos/vel")
    print("  Cameras: pan_tilt_cam, wrist_cam\n")

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:]  = [1.5, 0.0, 0.8]
        v.cam.distance   = 3.0
        v.cam.elevation  = -20

        while v.is_running():
            mujoco.mj_step(model, data)
            step_count += 1

            # Print sensor telemetry once per second
            print_telemetry(model, data, step_count)

            # ────────────────────────────────────────────
            #  STATE MACHINE
            # ────────────────────────────────────────────
            if state == "DRIVING":
                # Keep spinning wheels
                data.ctrl[WHEEL_FL] = APPROACH_SPEED
                data.ctrl[WHEEL_FR] = APPROACH_SPEED
                data.ctrl[WHEEL_BL] = APPROACH_SPEED
                data.ctrl[WHEEL_BR] = APPROACH_SPEED

                robot_x = data.xpos[base_id][0]
                if robot_x >= (table_x - STOP_DISTANCE):
                    # Stop wheels
                    data.ctrl[WHEEL_FL] = 0.0
                    data.ctrl[WHEEL_FR] = 0.0
                    data.ctrl[WHEEL_BL] = 0.0
                    data.ctrl[WHEEL_BR] = 0.0
                    state = "PREPARE_REACH"
                    print(f"\nRobot stopped at x={robot_x:.3f} m. Preparing arm for top-down reach...")
                    # Pre-reach pose: elbow-up overhead config for top-down approach
                    # shoulder_pan=0 (forward), shoulder_lift=-2.0 (arm back/up),
                    # elbow=1.57 (elbow bent up), wrist_1=-1.0, wrist_2=-1.57, wrist_3=0
                    data.ctrl[0:6] = [0.0, -2.0, 1.57, -1.0, -1.57, 0.0]
                    data.ctrl[gripper_act_id] = 0.0   # gripper open
                    ik_iterations = 0

            elif state == "PREPARE_REACH":
                ik_iterations += 1
                if ik_iterations > 1000:
                    state = "REACHING"
                    print("\nArm at pre-reach pose. Performing active perception...")
                    
                    # Update scene with true physics state so renderer can see it
                    mujoco.mj_forward(model, data)
                    
                    # Render RGB and Depth from wrist_cam
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera="wrist_cam")
                    rgb = renderer.render()
                    
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera="wrist_cam")
                    depth = renderer.render()
                    
                    # Segment red box
                    mask = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
                    ys, xs = np.where(mask)
                    if len(ys) == 0:
                        print("  WARNING: No red pixels found! Falling back to ground truth.")
                        perceived_box_pos = data.xpos[box_id].copy()
                    else:
                        med_y = int(np.median(ys))
                        med_x = int(np.median(xs))
                        d = depth[med_y, med_x]
                        
                        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
                        fovy = model.cam_fovy[cam_id]
                        f = 0.5 * 480 / np.tan(np.deg2rad(fovy) / 2)
                        cx, cy = 640 / 2.0, 480 / 2.0
                        
                        # Project pixel to 3D in camera frame
                        x_cam = (med_x - cx) * d / f
                        y_cam = (cy - med_y) * d / f
                        z_cam = -d
                        pt_cam = np.array([x_cam, y_cam, z_cam])
                        
                        # Transform to world frame
                        cam_pos = data.cam_xpos[cam_id]
                        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
                        surface_pt_world = cam_pos + cam_mat @ pt_cam
                        
                        # Depth hits the top surface (z approx 0.84), subtract half box height (0.02)
                        perceived_box_pos = surface_pt_world.copy()
                        perceived_box_pos[2] -= 0.02
                        
                        print(f"  Perceived box center: {perceived_box_pos}")
                        print(f"  Ground truth center:  {data.xpos[box_id]}")
                        err = np.linalg.norm(perceived_box_pos - data.xpos[box_id])
                        print(f"  Perception error:     {err:.4f} m")

                        # --- Visualization of Perception ---
                        try:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                            
                            ax[0].imshow(rgb)
                            ax[0].set_title("Wrist Cam: RGB")
                            ax[0].plot(med_x, med_y, 'g+', markersize=20, markeredgewidth=3)
                            
                            ax[1].imshow(mask, cmap='gray')
                            ax[1].set_title("Red Mask & Perceived Center")
                            ax[1].plot(med_x, med_y, 'g+', markersize=20, markeredgewidth=3)
                            
                            plt.show(block=False)
                            plt.pause(0.1) # Allow the window to render
                        except ImportError:
                            print("  (matplotlib not installed, skipping perception visualization)")

                    print("Computing IK to reach perceived target...")
                    ik_iterations = 0

            elif state == "REACHING":
                # One-shot 6-DOF IK (position + orientation) in a shadow copy of data.
                # We want the gripper to approach from ABOVE the box.
                d_ik = mujoco.MjData(model)
                d_ik.qpos[:] = data.qpos[:]
                # Seed IK from the elbow-up overhead pose
                for i, qi in enumerate(q_ids):
                    d_ik.qpos[qi] = [0.0, -2.0, 1.57, -1.0, -1.57, 0.0][i]

                # Target: 0.5 cm above the perceived box center so the fingers close around the box
                target_pos = perceived_box_pos.copy()
                target_pos[2] = perceived_box_pos[2] + 0.005   # 0.5 cm above box center

                # Desired gripper orientation: Z-axis of pinch site pointing DOWN (-Z world)
                # The pinch site's z-axis should align with world -Z.
                # target_rot_mat: columns are [x_site, y_site, z_site] in world frame
                # For top-down: z_site = [0, 0, -1], pick x_site = [1, 0, 0]
                R_target = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                ], dtype=float)  # gripper points down

                for _ in range(3000):
                    mujoco.mj_forward(model, d_ik)
                    # --- Position error ---
                    pos_err = target_pos - d_ik.site_xpos[pinch_id]

                    # --- Orientation error (rotation vector) ---
                    R_cur = d_ik.site_xmat[pinch_id].reshape(3, 3)
                    R_err = R_target @ R_cur.T           # error rotation matrix
                    # Convert to axis-angle (rotation vector)
                    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
                    if abs(angle) > 1e-6:
                        ax = np.array([
                            R_err[2, 1] - R_err[1, 2],
                            R_err[0, 2] - R_err[2, 0],
                            R_err[1, 0] - R_err[0, 1]
                        ]) / (2 * np.sin(angle))
                        rot_err = angle * ax
                    else:
                        rot_err = np.zeros(3)

                    err6 = np.concatenate([pos_err, 0.3 * rot_err])  # weight orientation less

                    if np.linalg.norm(pos_err) < 0.012 and np.linalg.norm(rot_err) < 0.15:
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
                print(f"  IK converged. Pinch at {pos_final}, target {target_pos}")
                print(f"  Position error: {np.linalg.norm(target_pos - pos_final):.4f} m")
                target_q = [d_ik.qpos[qi] for qi in q_ids]
                start_q  = [data.ctrl[i]  for i in range(6)]
                state = "EXECUTE_REACH"
                ik_iterations = 0
                print(f"  IK target joints: {[f'{q:.3f}' for q in target_q]}")

            elif state == "EXECUTE_REACH":
                ik_iterations += 1
                progress = min(1.0, ik_iterations / 1000.0)
                for i in range(6):
                    data.ctrl[i] = start_q[i] + progress * (target_q[i] - start_q[i])
                if ik_iterations >= 1000:
                    state = "WAIT_BEFORE_GRASP"
                    print("Reached target. Waiting before grasping...")
                    ik_iterations = 0

            elif state == "WAIT_BEFORE_GRASP":
                ik_iterations += 1
                if ik_iterations >= 500:  # wait 1 second (500 steps * 0.002s = 1s)
                    state = "GRASPING"
                    print("Closing gripper...")
                    ik_iterations = 0

            elif state == "GRASPING":
                data.ctrl[gripper_act_id] = 255.0
                ik_iterations += 1
                if ik_iterations > 300:
                    state = "COMPUTE_LIFT"
                    print("Gripper closed. Computing lift IK...")
                    ik_iterations = 0

            elif state == "COMPUTE_LIFT":
                d_ik = mujoco.MjData(model)
                d_ik.qpos[:] = data.qpos[:]
                # Seed from current configuration for lift
                target_pos      = data.site_xpos[pinch_id].copy()
                target_pos[2]  += 0.15   # lift 15 cm

                for _ in range(2000):
                    mujoco.mj_forward(model, d_ik)
                    pos_err = target_pos - d_ik.site_xpos[pinch_id]
                    if np.linalg.norm(pos_err) < 0.01:
                        break
                    J = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, d_ik, J, None, pinch_id)
                    J   = J[:, dof_ids]
                    dq  = J.T @ np.linalg.solve(J @ J.T + 0.03**2 * np.eye(3), pos_err)
                    for i, (qi, jid) in enumerate(zip(q_ids, jids)):
                        lo, hi = model.jnt_range[jid]
                        d_ik.qpos[qi] = np.clip(d_ik.qpos[qi] + 0.04 * dq[i], lo, hi)

                target_lift_q = [d_ik.qpos[qi] for qi in q_ids]
                start_lift_q  = [data.ctrl[i]  for i in range(6)]
                state = "EXECUTE_LIFT"
                ik_iterations = 0

            elif state == "EXECUTE_LIFT":
                ik_iterations += 1
                progress = min(1.0, ik_iterations / 1000.0)
                for i in range(6):
                    data.ctrl[i] = start_lift_q[i] + progress * (target_lift_q[i] - start_lift_q[i])
                if ik_iterations >= 1000:
                    state = "DONE"
                    print(f"✅ Task complete. Box Z = {data.xpos[box_id][2]:.3f} m")

            v.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
