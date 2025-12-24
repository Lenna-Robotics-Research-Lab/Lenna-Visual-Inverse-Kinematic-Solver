
# Lenna Visual Inverse Kinematic Solver

A specialized 6-DOF kinematic engine providing high-accuracy Inverse Kinematics (IK) and 3D path visualization. This repository implements a numerical solver for serial-link manipulators using standard Denavit-Hartenberg (D-H) conventions.

## 🛠 Technical Specifications

The solver is pre-configured for a standard 6-joint industrial-style arm with the following D-H configuration:

* **Degrees of Freedom**: 6 (Base, Shoulder, Elbow, and 3-axis Wrist).
* **IK Methodology**: Numerical optimization using the `L-BFGS-B` algorithm via `scipy.optimize`.
* **Trajectory Planning**: Linear joint-space interpolation.
* **Rotation Logic**: Standard XYZ Roll-Pitch-Yaw (RPY) extraction from 3x3 rotation matrices.

## 📂 Core Scripts

### 1. `IK_static_visuals.py`

The base simulator designed for command-line pose testing.

* Supports relative movement (`rel dx dy dz`) and absolute pose commands (`abs x y z R P Y`).
* Generates a static 3D plot of the start and end configurations.

### 2. `IK_live_visuals.py`

The high-accuracy implementation for real-time tracking.

* **Multi-start Optimization**: Iterates through multiple initial configurations to avoid local minima and ensure accuracy within a 5mm tolerance.
* **Live Animation**: Renders a dynamic Matplotlib window showing the robot's motion and the end-effector path.
* **Visual Debugging**: Real-time magenta labels tracking the end-effector  coordinates.

## 💻 Usage

```bash
# Run the live high-accuracy simulator
python IK_live_visuals.py

```

**Commands:**

* `abs 0.5 0.2 0.4 0 0.1 0`: Move end-effector to an absolute coordinate.
* `rel 0.1 0 -0.1`: Move the end-effector relative to its current position (+10cm X, -10cm Z).

