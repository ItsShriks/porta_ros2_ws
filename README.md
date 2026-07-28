# Master Thesis Simulation: MMO-700 MuJoCo Autonomous Grasping & Approach

This folder contains a standalone MuJoCo simulation environment for the **MMO-700** mobile manipulator (Neobotix MPO-700 omnidirectional base + Universal Robots UR5e arm + Robotiq 2F-85 gripper).

---

## 📁 Repository Structure

```
Master_Thesis_Sim/
├── run_mmo_700_scene.py    # Simulation scene: MMO-700 table approach
├── run_mmo_700_grasp.py    # Full autonomous grasp pipeline (Drive -> Reach IK -> Grasp -> Lift)
├── mmo_700.xml             # Main MuJoCo MJCF model file
├── mmo_700.urdf            # Complete robot URDF description
├── mmo_700/                # 3D meshes (STL/DAE) and URDF macro definitions
│   ├── meshes/             # Visual and collision mesh assets (Base, Arm, Gripper, Cameras)
│   └── urdf/               # Xacro components
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🚀 Requirements & Quickstart

### Prerequisites
- Python 3.8+
- `mujoco` (>= 3.0.0)
- `numpy`

### Installation

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Simulations

### 1. MMO-700 Table Approach Scene
Drives the robot forward towards a wooden table with a target red box using omnidirectional wheel velocity control.

```bash
python3 run_mmo_700_scene.py
```

### 2. Autonomous Grasping & Lifting Scene
Launches the full autonomous manipulation sequence:
1. **Drive**: Robot approaches the table and stops at a designated safety distance.
2. **Pre-reach**: Arm transitions to pre-grasp pose.
3. **Reach IK**: Numerical Inverse Kinematics computes joint targets to position the Robotiq 2F-85 pinch site at the target box.
4. **Grasp**: Actuates gripper fingers to clamp the red box.
5. **Lift**: Executes differential IK to lift the box 15 cm above the table.

```bash
python3 run_mmo_700_grasp.py
```

---

## 🔧 Model Features
- **Kinematics**: 4 Caster wheel assemblies, 6-DoF UR5e arm, Robotiq 2F-85 parallel gripper.
- **Sensors**: Pan-tilt Dynamixel mount, Intel RealSense D405 wrist camera, SICK S300 safety lidars.
- **Physics**: Real-time passive viewer, custom friction & contact parameters for realistic grasping.
