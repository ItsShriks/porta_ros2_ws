# Whole-Body Control (WBC) Literature Summary & Implementation Guide
**Target Workspace:** `~/Master_Thesis/porta_ros2_ws/`  
**Robot Platform:** Neobotix MMO-700 (Omnidirectional Base + UR5e 6-DoF Arm)  
**Simulation / Perception:** MuJoCo & Volumetric Grasping Network (VGN)

---

## 1. Executive Overview

This document summarizes the core methodologies, mathematical formulations, and control architectures from the three Whole-Body Control (WBC) and Mobile Manipulation references, along with blueprints for integrating them into `porta_ros2_ws`.

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │               WHOLE-BODY CONTROL STACK                 │
                                    └────────────────────────────────────────────────────────┘
                                                                │
                 ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
                 ▼                                              ▼                                              ▼
    [1] Theoretical Foundation                     [2] Real-Time Horizon Control                 [3] Dynamic Multi-Contact
      hr11_locomotion_3.pdf                                  paper.pdf                            jyegCr-2509.14010v1.pdf
 ─────────────────────────────────              ───────────────────────────────────            ──────────────────────────────
 • Hierarchical QP (HQP)                        • Whole-Body MPC (WBMPC)                       • Full-body dynamic modeling
 • Null-space task projections                  • Task priority transitions over time          • Contact-wrench cone (CWC)
 • Inequality constraints (bounds/limits)       • Translational manipulability maximization    • DDP / FDDP with warm-starting
 • Real-time reactive execution                 • RTI-SQP (acados/CasADi) @ 100 Hz             • Unified omnidirectional kinematics
```

---

## 2. In-Depth Paper Summaries

### Document 1: `hr11_locomotion_3.pdf`
* **Topic:** *Realizing Motion Plans: Whole-Body Control*
* **Type:** Lecture / Foundational Technical Reference

#### Core Problem & Insight
Motion planners and model predictive controllers (MPC) generate high-level Center of Mass (CoM), base paths, and end-effector trajectories, but often simplify or ignore low-level hardware constraints (torque saturation, joint limits, instantaneous contact forces, and self-collisions). A high-frequency ($500\text{--}1000\text{ Hz}$) stabilizing controller is necessary to map these plans onto the robot while enforcing physical limits.

#### Key Mathematical Formulations
1. **Linear Task Space Representation:**
   Each task $i$ is formulated as a linear equality constraint:
   $$A_i \mathbf{x} = \mathbf{b}_i$$
   where $\mathbf{x}$ is the control decision variable (generalized joint accelerations $\mathbf{\ddot{q}}$ or actuator torques $\boldsymbol{\tau}$).

2. **Hierarchical Task Resolution (Lexicographic / Null-Space):**
   Tasks are ranked by strict priority ($i=1$ being highest, e.g., contact/balance > end-effector tracking > posture regularization):
   $$\mathbf{x}^* = \arg\min_{\mathbf{x}} \|A_2 \mathbf{x} - \mathbf{b}_2\|^2 \quad \text{s.t.} \quad A_1 \mathbf{x} = \mathbf{b}_1, \quad C \mathbf{x} \le \mathbf{d}$$
   Alternatively solved iteratively via null-space projection matrices:
   $$N_i = I - A_i^\dagger A_i$$

3. **Sequential Quadratic Programming (HQP):**
   Formulates a cascade of QPs where lower-priority objectives optimize over the residual degrees of freedom without degrading higher-priority task performance.

---

### Document 2: `paper.pdf`
* **Title:** *Whole-Body Model Predictive Control for Mobile Manipulation with Task Priority Transition*
* **Authors:** Yushi Wang, Ruoqu Chen, Mingguo Zhao (Tsinghua University)

#### Core Problem & Insight
Standard mobile manipulation often controls the mobile base and the arm independently or solves WBC instantaneously at a single time step. This causes poor coordination, velocity spikes during task switching (e.g., approach $\to$ grasp $\to$ hold $\to$ push), and arm singularity entrapment. This paper introduces **Whole-Body MPC (WBMPC)** with **transitional task priorities over a predictive time horizon**.

#### Key Mathematical Formulations
1. **Unified State & Control Vectors:**
   * State: $\mathbf{x} = [\mathbf{x}_{\text{base}}, \mathbf{x}_{\text{arm}}]^T = [x, y, \theta, q_1, \dots, q_n]^T$
   * Control input: $\mathbf{u} = [\dot{q}_l, \dot{q}_r, \dot{q}_1, \dots, \dot{q}_n]^T$ (or $[\dot{x}_b, \dot{y}_b, \dot{\theta}_b, \dot{\mathbf{q}}_{\text{arm}}]^T$ for omnidirectional bases).

2. **Time-Varying Priority Weight Matrix:**
   Rather than static task weights, a matrix $\mathbf{W} = \{w_{i,k}\}$ is optimized over tasks $i$ and horizon stages $k \in \{0, \dots, N-1\}$. When switching task priorities at time $t_s$, the weights transition progressively over the prediction window, reducing transition velocity spikes by **53%**:
   $$\ell_{t,k} = \sum_{i} w_{i,k} \|\mathbf{e}_{i,k}\|_{\mathbf{Q}_i}^2$$

3. **Translational Manipulability Cost:**
   To actively repel the arm from kinematic singularities during base locomotion, the cost function directly penalizes low manipulability:
   $$\ell_m(\mathbf{q}) = \frac{1}{m(\mathbf{q})}, \quad m(\mathbf{q}) = \sqrt{\det(J_t(\mathbf{q}) J_t(\mathbf{q})^T)}$$
   *(Note: Uses only translational Jacobian $J_t$ to ensure dimensional consistency).*

4. **Solving Strategy:**
   Discretization $\Delta t = 0.1\text{ s}$, horizon $N=20$ ($T=2.0\text{ s}$). Solved via **Real-Time Iteration SQP (RTI-SQP)** using `acados` / `CasADi` in **$4.3\text{ ms}$ at $100\text{ Hz}$**, commanding a $1000\text{ Hz}$ low-level joint PD controller.

---

### Document 3: `jyegCr-2509.14010v1.pdf`
* **Title:** *Whole-body Motion Control of an Omnidirectional Wheel-Legged Mobile Manipulator via Contact-Aware Dynamic Optimization*
* **Authors:** Zong Chen, Shaoyang Li, Ben Liu, Min Li, Zhouping Yin, Yiqun Li (HUST, arXiv:2509.14010)

#### Core Problem & Insight
Handling floating-base dynamics subject to simultaneous wheel rolling constraints (line contact) and arm-environment manipulation forces (point contact) is challenging. This paper presents a contact-aware dynamic optimization framework with a unified 4-wheel independent steering/driving (4WIS-4WID) kinematic model that eliminates discontinuous mode switching.

#### Key Mathematical Formulations
1. **Full-Body Coupled Contact Dynamics:**
   $$\begin{bmatrix} \mathbf{M}(\mathbf{q}) & -\mathbf{J}_c^T(\mathbf{q}) \\ \mathbf{J}_c(\mathbf{q}) & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{\ddot{q}} \\ \mathbf{f}_c \end{bmatrix} = \begin{bmatrix} \mathbf{S}^T \mathbf{u} - \mathbf{h}(\mathbf{q}, \mathbf{\dot{q}}) \\ -\mathbf{\dot{J}}_c(\mathbf{q}, \mathbf{\dot{q}})\mathbf{\dot{q}} \end{bmatrix}$$

2. **Degenerate Contact-Wrench Cone (CWC):**
   * **Manipulator:** Standard 3D friction cone point contact ($|f_x| \le \mu f_z, |f_y| \le \mu f_z, f_z \ge 0$).
   * **Wheels:** Degenerate line-contact model enforcing friction cone and normal/lateral moments $(\tau_x, \tau_z)$, but leaving rolling moment $\tau_y$ unconstrained:
     $$\mathbf{A}_c \mathbf{f}_c \le \mathbf{b}_c$$
     Converted into a smooth penalty function $\alpha(\mathbf{r}) = \frac{1}{2} \mathbf{r}^T \mathbf{r}$ for $r > 0$ to ensure solver convergence.

3. **Unified Omnidirectional Kinematic Steering:**
   Computes continuous steering angles and wheel speeds analytically without discrete mode switching:
   $$\delta_i^{\text{opt}} = \text{atan2}(v_{b,y} + \omega_b x_i,\; v_{b,x} - \omega_b y_i)$$
   with $\pm \pi$ modulo optimization and wheel velocity inversion ($v_i := -v_i$) if $|\Delta \delta_i| > \frac{\pi}{2}$ to prevent extreme steering motions.

4. **Warm-Started DDP / FDDP:**
   Uses Feasibility-Driven Differential Dynamic Programming (FDDP). Warm-starting with previous trajectory rollouts reduces compute time from $150\text{ ms}$ down to **$3\text{--}13\text{ ms}$**.

---

## 3. Comparison Matrix

| Feature | `hr11_locomotion_3.pdf` | `paper.pdf` (Wang et al.) | `jyegCr-2509.14010v1.pdf` (Chen et al.) |
| :--- | :--- | :--- | :--- |
| **Control Level** | Reactive HQP / OSC ($500\text{--}1000\text{ Hz}$) | Predictive Whole-Body MPC ($100\text{ Hz}$) | Full-body DDP/FDDP ($500\text{ Hz}$) |
| **Priority Handling** | Strict null-space / lexicographic QP | Time-horizon weight matrix $\mathbf{W}(t)$ | Cost-weighted penalty terms in DDP |
| **Singularity Treatment** | Damped Least Squares (DLS) | Translational manipulability cost $\frac{1}{m(\mathbf{q})}$ | Dynamic inertia-aware joint bounds |
| **Contact Modeling** | Rigid contact constraints | Kinematic collision avoidance (ESDF) | Line contact (degenerated wrench cone) |
| **Primary Advantage** | Computationally lightweight & robust | Smooth multi-phase task transitions | Accurate dynamic contact handling |

---

## 4. Implementation Mapping for `porta_ros2_ws`

Here is how to map these principles directly to your Master's thesis repository:

```
porta_ros2_ws/
├── scripts/
│   ├── run_kinematic_wbc.py              <── Apply Null-Space Prioritization (hr11)
│   ├── run_dynamic_wbc.py                <── Apply Operational Space Control (OSC) (jyegCr)
│   ├── active_perception_dynamic_wbc.py  <── Apply Time-Varying Priority Transitions (Wang et al.)
│   └── active_perception_dynamic_wbc_vgn.py <── Manipulability Maximization during Grasp Planning
└── robots/
    └── mmo_700_wbc.xml                   <── Planar Base Actuation + Dynamic UR5e Model
```

### 1. `run_kinematic_wbc.py` (Null-Space Hierarchy)
Upgrade simple pseudo-inverse IK ($\mathbf{\dot{q}} = \mathbf{J}^\dagger \mathbf{v}_{\text{des}}$) to a two-level hierarchical controller:
$$\mathbf{\dot{q}} = \mathbf{J}_{ee}^\dagger \mathbf{v}_{ee} + (\mathbf{I} - \mathbf{J}_{ee}^\dagger \mathbf{J}_{ee}) \left( \mathbf{K}_p (\mathbf{q}_{\text{nominal}} - \mathbf{q}) + \alpha \nabla_{\mathbf{q}} m(\mathbf{q}) \right)$$
* **Primary Task ($J_{ee}$):** 6-DoF End-Effector Cartesian target tracking.
* **Secondary Task (Null-Space):** Arm posture regularization towards home configuration + gradient ascent on Yoshikawa manipulability $m(\mathbf{q})$.

### 2. `run_dynamic_wbc.py` (Operational Space Dynamics)
* Compute Mass Matrix $\mathbf{M}(\mathbf{q})$ via `mj_fullM` and bias forces $\mathbf{h}$ via `data.qfrc_bias`.
* Compute Task-Space Inertia Matrix: $\boldsymbol{\Lambda}(\mathbf{q}) = (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T)^{-1}$.
* Compute Dynamically Consistent Generalized Inverse: $\mathbf{J}^{T\#} = \boldsymbol{\Lambda} \mathbf{J} \mathbf{M}^{-1}$.
* Compute Torque Command:
  $$\boldsymbol{\tau} = \mathbf{J}^T \boldsymbol{\Lambda} \left( \mathbf{\ddot{x}}_{\text{des}} + \mathbf{K}_d (\mathbf{\dot{x}}_{\text{des}} - \mathbf{\dot{x}}) + \mathbf{K}_p (\mathbf{x}_{\text{des}} - \mathbf{x}) \right) + \mathbf{h} + (\mathbf{I} - \mathbf{J}^T \mathbf{J}^{T\#}) \boldsymbol{\tau}_{\text{posture}}$$

### 3. `active_perception_dynamic_wbc_vgn.py` (Phase Transitions)
Structure the state machine using smooth task priority weights:
* **Phase 1: Scene Exploration & VGN Point Cloud Capture**  
  $\mathbf{W}_{\text{base}} \gg 0$, Camera gaze vector prioritized towards clutter bounding box.
* **Phase 2: Mobile Pre-Grasp Approach**  
  Progressively shift weight from base positioning to arm reachability while keeping $m(\mathbf{q})$ maximized.
* **Phase 3: 6-DoF VGN Grasp Execution**  
  Lock base movement ($\mathbf{W}_{\text{base\_vel}} \gg 0$) and give $100\%$ control authority to the UR5e end-effector trajectory to execute the grasp cleanly.
