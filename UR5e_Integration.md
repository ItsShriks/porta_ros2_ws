# Implementation Plan: UR5e Integration, Collision Fixes, Base Cameras & Sensors for MMO-700

Upgrade the MMO-700 simulation scene in `run_mmo_700_grasp.py` and `mmo_700.xml` by replacing the custom arm with the official UR5e model from `mujoco_menagerie/universal_robots_ur5e/ur5e.xml`, fixing base and arm collisions with the table, and adding camera sensors and state telemetry to the robot base.

## User Review Required

> [!IMPORTANT]
> - **Arm Model Replacement**: Replacing the old custom UR5 arm definition with the official `mujoco_menagerie/universal_robots_ur5e/ur5e.xml` standard model changes joint names (`ur5eshoulder_pan_joint` $\rightarrow$ `shoulder_pan_joint`), actuator control gains, and link kinematics to match official Universal Robots specifications.
> - **Collision Bitmask Adjustment**: `base_link_collision` was set to `contype="2" conaffinity="2"` while the table used `contype="1" conaffinity="1"`, causing physics collision to be ignored. Changing base and arm geoms to `contype="1" conaffinity="1"` enables physical solid collisions.
> - **Safe Driving Distance**: `STOP_DISTANCE` is adjusted to `0.95m` (stopping at $x \approx 1.05\text{m}$) so that the robot front bumper ($x \approx 1.53\text{m}$) stops before contacting the table front edge ($x = 1.60\text{m}$).

## Open Questions

None at present. The requirements are clear and mapped to specific MuJoCo XML and Python script updates.

---

## Proposed Changes

### MuJoCo XML Model

#### [MODIFY] [mmo_700.xml](file:///home/shrikar/Master_Thesis_Sim/mmo_700.xml)

- **UR5e Arm Integration**:
  - Add menagerie default classes (`ur5e`, `size3`, `size3_limited`, `size1`, `visual`, `collision`, `eef_collision`).
  - Import UR5e mesh assets (`base_0.obj`, `base_1.obj`, `shoulder_*.obj`, `upperarm_*.obj`, `forearm_*.obj`, `wrist*.obj`) from `mujoco_menagerie/universal_robots_ur5e/assets/`.
  - Replace `ur5ebase_link` subtree with menagerie UR5e link chain and collision capsules on every arm link.
  - Mount Robotiq 2F-85 gripper and D405 camera to UR5e `wrist_3_link` / `attachment_site`.
  - Update actuators to menagerie affine position controllers (`shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3`).

- **Collision Detection Fixes**:
  - Update `base_link_collision` and base component geoms to `contype="1" conaffinity="1"`.
  - Ensure table geoms use `contype="1" conaffinity="1"` and valid solver parameters (`solimp`, `solref`).
  - Enable collision capsules on all arm links so the arm physically cannot penetrate the table top or legs.

- **Cameras & Sensors Addition**:
  - Add base cameras: `base_front_cam` (front facing camera on base) and `pan_tilt_cam` (pan-tilt tower camera).
  - Add sites on `base_link`: `base_imu_site`, `lidar_front_site`, `lidar_rear_site`.
  - Add `<sensor>` section with:
    - Base IMU accelerometer (`base_accel`) & gyroscope (`base_gyro`).
    - Front & rear LiDAR rangefinders (`front_lidar`, `rear_lidar`).
    - Wheel velocity sensors (`wheel_fl_vel_sensor`, etc.).
    - Arm joint position & velocity sensors (`shoulder_pan_pos`, `shoulder_pan_vel`, etc.).

---

### Python Control & Demo Scripts

#### [MODIFY] [run_mmo_700_grasp.py](file:///home/shrikar/Master_Thesis_Sim/run_mmo_700_grasp.py)

- **Joint & Actuator Mapping**: Update joint list and actuator indices to target menagerie UR5e names (`shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`).
- **Safe Driving & Table Stop**: Set `STOP_DISTANCE = 0.95` so the robot stops with a safe gap before the table front edge.
- **IK & Trajectory Smoothing**:
  - Update IK calculations to use menagerie UR5e link transformations and pinch site location.
  - Add pre-reach height elevation to ensure arm trajectory approaches the target box from above, avoiding table collisions.
- **Base Sensor & Camera Telemetry**:
  - Read sensor data (`data.sensordata`) during runtime loop and log IMU (accel/gyro), LiDAR distance to table, wheel speeds, and joint positions.
  - Add offscreen camera rendering capability (`mujoco.Renderer`) for `base_front_cam` and `wrist_cam`.

#### [MODIFY] [run_mmo_700_scene.py](file:///home/shrikar/Master_Thesis_Sim/run_mmo_700_scene.py)

- Align `STOP_DISTANCE` and keyframe initialization with updated collision bitmasks and UR5e model.

---

## Verification Plan

### Automated Tests
- Run XML validation with MuJoCo parser:
  `python3 -c "import mujoco; model = mujoco.MjModel.from_xml_path('mmo_700.xml'); print('XML syntax & mesh check passed! nbody:', model.nbody, 'nsensor:', model.nsensor)"`
- Run the full grasping pipeline:
  `python3 run_mmo_700_grasp.py`
  - Verify that:
    1. Base drives forward and stops safely at $x \approx 1.05\text{m}$ without contacting or penetrating the table.
    2. Arm (UR5e) reaches for the box without penetrating the table top.
    3. Gripper grasps and lifts the box cleanly.
    4. Base IMU, LiDAR rangefinder, wheel velocities, and camera telemetry are printed and processed without errors.

### Manual Verification
- Launch viewer window during `run_mmo_700_grasp.py` and inspect:
  - UR5e arm visual and collision geometry.
  - Absence of any base-table or arm-table clipping/collisions.
  - Camera view overlay / sensor printouts.
