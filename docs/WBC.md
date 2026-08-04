# Implementation Plan: Whole Body Control for MMO-700

This plan outlines the steps to create two distinct scripts for Whole Body Control (WBC) on the MMO-700 mobile manipulator: one using a Kinematic approach (velocity/position) and one using a Dynamic approach (Operational Space Control with torques).

## Open Questions
> [!IMPORTANT]
> The robot currently drives using simulated contact physics on its wheels (with a floating base `freejoint`). For standard WBC in a simulator, it is strongly recommended to replace the `freejoint` and wheel contacts with an idealized **planar joint** (X, Y, Yaw) at the base. This provides a perfect, continuous Jacobian from the base to the end-effector. 
> 
> **Are you okay with me creating a modified XML (`mmo_700_wbc.xml`) that uses planar joints for the base and `<motor>` actuators for dynamic control?** (The original `mmo_700.xml` will remain untouched).

## Proposed Changes

### Modified Simulation Environment
I will create a new simulation environment tailored for Whole Body Control, bypassing the complex wheel-ground friction physics which usually require separate non-holonomic mobile base controllers.

#### [NEW] [mmo_700_wbc.xml](file:///home/shrikar/Master_Thesis_Sim/mmo_700_wbc.xml)
- Replace `<freejoint name="floating_base">` with three joints: `<slide axis="1 0 0">` (X), `<slide axis="0 1 0">` (Y), and `<hinge axis="0 0 1">` (Yaw).
- Add `<motor>` (torque) actuators for the base X, Y, Yaw joints and the 6 UR5e arm joints (required for Dynamic WBC).
- Add `<velocity>` actuators for the same joints (required for Kinematic WBC).

### Kinematic WBC Script
#### [NEW] [run_kinematic_wbc.py](file:///home/shrikar/Master_Thesis_Sim/run_kinematic_wbc.py)
- **Control Strategy:** First-order differential kinematics.
- **Implementation:**
  - Compute the full 9-DoF Jacobian (3 Base + 6 Arm) using `mujoco.mj_jacSite`.
  - Define a Cartesian trajectory for the end-effector (e.g., tracking a moving point or drawing a shape while the base moves).
  - Compute desired joint velocities using the pseudo-inverse: $\dot{q} = J^{\dagger} v_{des}$.
  - Command the `<velocity>` actuators on the base and arm to achieve these velocities.

### Dynamic WBC Script
#### [NEW] [run_dynamic_wbc.py](file:///home/shrikar/Master_Thesis_Sim/run_dynamic_wbc.py)
- **Control Strategy:** Operational Space Control (OSC) / Impedance Control.
- **Implementation:**
  - Compute the Mass Matrix ($M$) and Coriolis/Gravity forces ($h$) using MuJoCo's dynamic functions.
  - Compute the full 9-DoF Jacobian ($J$) and its time derivative.
  - Compute the Task-Space Inertia Matrix: $\Lambda = (J M^{-1} J^T)^{-1}$.
  - Formulate a PD controller in Cartesian space to get desired acceleration $a_{des}$.
  - Compute the command torques: $\tau = J^T \Lambda a_{des} + h$.
  - Optionally add a null-space projection term to maintain a preferred posture (e.g., keeping the arm close to its home configuration while the base moves to reach the target).
  - Command the `<motor>` actuators with the computed torques.

## Verification Plan
### Automated Tests
- N/A (Scripts are visual and interactive).

### Manual Verification
- Run `python run_kinematic_wbc.py` and observe the base and arm moving simultaneously to track a trajectory.
- Run `python run_dynamic_wbc.py` and verify that the robot exhibits compliant, torque-controlled behavior while tracking the same trajectory, demonstrating successful dynamic decoupling.
