# Simulation

This directory contains all resources related to musculoskeletal simulation used in the project. The simulation pipeline is built around **OpenSim 4.5** and is responsible for generating biomechanical variables from experimental motion capture data.

The outputs produced here serve as the physics-based reference for the deep learning and residual learning framework.

---

## Simulation Workflow

```
Motion Capture (.trc)
        │
        ▼
Model Scaling
        │
        ▼
Inverse Kinematics (IK)
        │
        ▼
Inverse Dynamics (ID)
        │
        ▼
Static Optimization (SO)
        │
        ▼
Muscle Analysis
        │
        ▼
Joint Torques • Muscle Activations • Muscle Forces
```

---

## Directory Structure

```
simulation/

├── models/             # OpenSim musculoskeletal models (.osim)
├── markers/            # Marker trajectory files (.trc)
├── forces/             # Ground reaction or external force files (.mot/.sto)
├── scaled_models/      # Subject-specific scaled models
├── ik/                 # Inverse Kinematics outputs
├── id/                 # Inverse Dynamics outputs
├── so/                 # Static Optimization results
├── analysis/           # Muscle analysis outputs
├── setup_files/        # XML configuration files
└── scripts/            # Automation scripts
```

---

## Simulation Components

### Musculoskeletal Model

The simulations use an OpenSim upper-limb musculoskeletal model that represents the shoulder, elbow, and associated muscle groups involved in biceps curl movements.

---

### Motion Capture

Experimental marker trajectories recorded using the **Vicon Nexus Motion Capture System** are imported into OpenSim to reconstruct upper-limb kinematics.

---

### Inverse Kinematics (IK)

Computes joint angles by fitting the musculoskeletal model to the recorded marker trajectories.

**Output**

- Shoulder joint angles
- Elbow joint angles

---

### Inverse Dynamics (ID)

Uses joint kinematics and external loads to estimate net joint moments.

**Output**

- Shoulder torque
- Elbow torque

---

### Static Optimization (SO)

Estimates muscle activations by distributing joint torques across individual muscles while minimizing overall muscular effort.

**Output**

- Muscle activations
- Muscle forces

---

### Muscle Analysis

Generates additional biomechanical variables for further analysis and machine learning.

Typical outputs include:

- Muscle lengths
- Muscle velocities
- Moment arms
- Force generation

---

## Generated Outputs

The simulation pipeline produces biomechanical variables including:

- Joint Angles
- Joint Torques
- Muscle Activations
- Muscle Forces
- Muscle Lengths
- Biomechanical States

These outputs are synchronized with experimentally recorded EMG signals to create the multimodal dataset used for residual learning and uncertainty-aware prediction.

---

## Software

- OpenSim 4.5
- Python
- NumPy
- Pandas

---


## Simulation Preview

<p align="center">
  <img src="media/video/simulation.gif" width="850">
</p>


## Notes

This directory contains only the simulation assets and outputs. Experimental data acquisition, preprocessing, and deep learning models are organized in their respective directories within the repository.
