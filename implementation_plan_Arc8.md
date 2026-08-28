# Figure-8 Whole-Body Control Demonstration

## Goal

Create a new MuJoCo simulation script where the MMO-700 robot (mobile base + UR5e arm) draws a figure-8 (∞) lemniscate in **both the horizontal plane** (base moves, arm follows) **and the vertical plane** (base is stationary, arm traces the 8 in the frontal/sagittal plane). This serves as a WBC demonstration showcasing coordinated base + manipulator control.

## Background from Papers

**paper.pdf (WBMPC — Zhao et al., Tsinghua):**
- Proposes Whole-Body Model Predictive Control (WBMPC) for mobile manipulators
- Key idea: task-space commander specifies EE + base trajectories, unified kinematic WBC generates joint-space commands
- OSC-style formulation: minimizes weighted tracking errors over a prediction horizon
- Demonstrates coordinated mobile base + arm for door-opening

**2509.14010v1.pdf (Wheel-legged platform, omnidirectional WBC):**
- Unified whole-body motion control framework: base motion + arm manipulation in one optimization
- Uses a **lemniscate (figure-8) trajectory as a key demonstration task** for validating WBC capability
- The trajectory is parameterized as: `x(t) = a·sin(ωt)`, `y(t) = a·sin(ωt)·cos(ωt)` (Lissajous form)
- Validates both locomotion and manipulation simultaneously

## Design Decisions

The script will run **two sequential phases**:

### Phase 1 — HORIZONTAL Figure-8 (Base Loco-manipulation)
- The mobile base (base_x, base_y, base_yaw) traces a lemniscate in the XY plane
- The arm EE is held at a fixed height offset from the base (WBC tracks a point rigidly above the base)
- Both base velocities AND arm joint velocities are commanded simultaneously via kinematic WBC
- This demonstrates that the arm stays globally stable while the base moves

### Phase 2 — VERTICAL Figure-8 (Arm Manipulation)
- Base stops; arm end-effector traces a lemniscate in the XZ (vertical) plane in front of the robot
- The arm alone draws the full figure-8 using the kinematic Jacobian-based WBC from the existing code
- Orientation held constant (top-down or forward-facing)

### Trajectory Parameterization (Lemniscate of Bernoulli)
Using the standard Lissajous figure-8:
```
horizontal:  x(t) = cx + A·sin(ω·t)
             y(t) = cy + A·sin(ω·t)·cos(ω·t)
             z(t) = fixed EE height

vertical:    x(t) = cx + R·sin(ω·t)·cos(ω·t)   (depth oscillation)
             y(t) = fixed
             z(t) = cz + R·sin(ω·t)
```

### Controller
Reuses the existing `kinematic_wbc()` IK function (Jacobian pseudo-inverse, velocity-resolved):
- For horizontal phase: base_vel actuators get velocity commands; arm vel actuators get Jacobian commands
- For vertical phase: only arm vel actuators

## Proposed Changes

### [NEW] scripts/figure8_wbc.py
A standalone self-contained script. No external dependencies beyond what already exists (`mujoco`, `numpy`).

**Key sections:**
1. Constants / parameters (figure-8 size, frequency, phase durations)
2. Lemniscate trajectory generator function
3. Horizontal phase loop: base + arm coordinated
4. Vertical phase loop: arm-only
5. Trace visualization via MuJoCo `mjv_addGeoms` (optional, sphere markers)

## Verification Plan

### Automated
- Script runs without error in headless mode

### Manual / Visual
- Viewer shows base tracing a figure-8 on the floor
- End-effector follows the figure-8 in the air (horizontal phase)
- Arm tip traces a vertical figure-8 in front of the robot (vertical phase)
- Console logs position errors at each phase

> [!NOTE]
> The script is self-contained and uses the same `mmo_700_wbc.xml` model already used by other scripts. No new XML changes are needed.
