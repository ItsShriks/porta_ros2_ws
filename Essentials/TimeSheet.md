# Timesheet Log

**Developer / Author:** Shrikar Nakhye (<nakhyeshrikar@icloud.com>)  
**Project:** Master Thesis Simulation (`Master_Thesis_Sim` / `porta_ros2_ws`)  
**Commit Date:** July 28, 2026  

---

## 1. Commit Log & Work Summary

| Commit Hash | Author | Date & Time | Commit Message | Key Changes / Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| `fa7d21f` | Shrikar Nakhye | 2026-07-28 15:56:06 | Initial commit | Repository initialization, standard `.gitignore` setup for ROS/Catkin/Python/Eclipse, initial `README.md`. |
| `17816115` | ItsShriks | 2026-07-28 16:03:07 | Initial Commit | Integrated full MMO-700 simulation environment, UR5e arm integration plan, and Whole Body Control (WBC) specifications. |

---

## 2. Activity Breakdown

### 2.1 Repository Setup & Infrastructure (`fa7d21f`)
- Configured repository ignore rules (`.gitignore`) covering ROS build spaces (`devel/`, `logs/`, `build/`, `bin/`, `lib/`), dynamic reconfigure, IDE workspace settings (QtCreator, Eclipse, Emacs), and Catkin build artifacts.
- Created base workspace README.

### 2.2 Simulation Environment & Control Architecture (`17816115`)
- **MuJoCo Simulation Framework:**
  - Added standalone simulation scripts: `run_mmo_700_scene.py` (table approach) and `run_mmo_700_grasp.py` (autonomous grasp & lift pipeline).
  - Maintained robot models: `mmo_700.xml` (MJCF model) and `mmo_700.urdf` (URDF description) with 3D mesh assets.
- **UR5e Arm & Sensor Integration Plan (`UR5e_Integration.md`):**
  - Replaced custom arm definitions with standard MuJoCo Menagerie UR5e model specifications.
  - Adjusted collision bitmasks (`contype="1"`, `conaffinity="1"`) and established safe stopping distances ($x pprox 1.05	ext{m}$).
  - Configured sensor telemetry integration (Base IMU, dual LiDAR rangefinders, wheel encoders, wrist and pan-tilt cameras).
- **Whole Body Control Architecture (`WBC.md`):**
  - Designed dual control strategy: Kinematic WBC (1st order differential kinematics with pseudo-inverse Jacobian) and Dynamic WBC (Operational Space Control / Impedance Control with Mass Matrix and Null-space projection).
  - Created blueprint for planar base movement model (`mmo_700_wbc.xml`).

---

## 3. Work Estimates & Task Summary

| Work Package | Task / Activity | Hours Logged | Status |
| :--- | :--- | :---: | :---: |
| **WP 1** | Workspace & Git Repository Setup | 1.0 | Completed |
| **WP 2** | MMO-700 Simulation Environment & Pipeline | 3.5 | Completed |
| **WP 3** | UR5e Model Integration & Collision Tuning | 2.0 | Completed |
| **WP 4** | Whole Body Control (WBC) Architecture Design | 1.5 | Completed |
| **Total** | | **8.0** | |
