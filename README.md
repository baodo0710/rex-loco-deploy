# Rex — Quadruped Locomotion in Isaac Lab

<div align="center">
    
![Rex Quadruped Render](docs/images/rex_hero.png)\
**A large-scale reinforcement learning framework for servo-actuated SpotMicro-class quadruped locomotion with systematic sim-to-sim validation**

[![Isaac Lab](https://img.shields.io/badge/Built%20on-Isaac%20Lab-orange)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

</div>

---

## Abstract

Rex is a reinforcement-learning framework for a 12-DOF, MG996R-actuated SpotMicro-class quadruped, built on NVIDIA Isaac Lab. The project adapts a velocity-tracking locomotion pipeline to a small, low-cost servo-actuated platform and examines how platform-scale kinematic constraints affect training, actuation modeling, and deployment.

The project originated as a from-scratch Pygame kinematic simulator running on a Raspberry Pi 4B, used to validate 12-DOF forward kinematics, joint-limit constraints, and gait sequencing before scaling to GPU-accelerated reinforcement-learning training. In Isaac Lab, Rex uses flat-terrain velocity-tracking environments with deterministic evaluation variants, domain randomization, observation corruption, and an MG996R-inspired actuation configuration.

A central motivation is the mismatch between general-purpose rough-terrain locomotion settings and the Rex platform’s limited leg stroke, ground clearance, and actuator envelope. The project therefore treats terrain feasibility as a platform-dependent design problem rather than assuming that policies trained for larger quadrupeds transfer directly to a small servo-actuated system.

Trained policies are exported through ONNX and evaluated in PyBullet to investigate policy-export, observation-interface, and physics-engine differences prior to hardware deployment. The repository also includes an STM32-based deployment interface for the physical robot, including servo calibration, joint-limit clipping, communication watchdog behavior, and a safe-pose recovery routine.

---

## Origins: Pygame Simulation on Raspberry Pi

Before scaling Rex into NVIDIA Isaac Lab's GPU-accelerated simulation framework, the project originated as a minimal, from-scratch kinematic simulator executing on a **Raspberry Pi 4B**. Using **Pygame**, the complete 12-DOF quadruped was modeled and visualized **joint by joint, frame by frame**, a ground-up methodology for validating forward kinematics, joint-limit constraints, and gait sequencing without reliance on GPU acceleration or a commercial physics engine.

<div align="center">
    
![Pygame Quadruped Simulation on Raspberry Pi 4B](docs/images/rex_pygame_origin.png)\
**Early embedded prototype:** Joint-angle visualization and real-time servo control logic executing natively on Raspberry Pi 4B

</div>

This stage served as the architectural and kinematic foundation for the project: validating the 12-DOF leg kinematics, establishing joint coordinate frames and Denavit-Hartenberg parameters, and constructing the real-time control loop that would subsequently scale into the full reinforcement learning pipeline in Isaac Lab.

---

## Overview

Rex is a custom manager-based reinforcement-learning task extension for [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). It adapts Isaac Lab’s velocity-tracking locomotion pipeline to a servo-actuated SpotMicro-class quadruped represented by a customized `spot2.usd` asset.

The platform is actuated by twelve MG996R servos. Rather than treating the actuators as ideal torque sources, Rex uses Isaac Lab’s DC-motor actuator model to represent a nominal MG996R-inspired torque-speed saturation envelope.

The repository supports:

- Large-scale parallel policy training in Isaac Lab.
- Deterministic policy evaluation.
- Domain randomization and observation corruption.
- ONNX policy export.
- PyBullet-based cross-simulator inference.
- STM32-based low-level hardware deployment.

The project is intended as a simulation-to-cross-simulator validation pipeline preceding physical deployment.
<!-- TODO: Add side-by-side sim vs. real photo here -->
<div align="center">

![Simulation vs Reality](docs/images/sim_vs_real.png)\
*Left: Isaac Sim high-fidelity simulation | Right: Target physical hardware platform*

![IsaacLab](docs/videos/isaaclab.gif)\
*Isaac Sim training platform*

</div>

---

## Hardware & Electronics

### Custom CAD Model

<!-- TODO: Add CAD model renders/screenshots here -->
<div align="center">

![CAD Model](docs/images/cad_model.png)\
*Custom CAD design of the SpotMicro-class chassis*

</div>

### Electronics & Wiring

Wiring.md provides a detailed wiring pinout.
<div align="center">

![Electronics Wiring](docs/images/electronics_wiring.png)\
*Wiring diagram: STM32 controller, MG996R servos, power distribution, and sensors*

</div>

---

## Environments

Six Gym-registered tasks provide a complete training-to-deployment lifecycle spanning velocity tracking, postural stability, and systematic evaluation:

| Task ID | Config Class | Description |
|---|---|---|
| `Isaac-Velocity-Flat-Rex-v0` | `RexFlatEnvCfg` | Velocity tracking on flat terrain with full domain randomization. Trains robust forward locomotion with randomized friction, mass, and external perturbations across 4096 parallel environments. |
| `Isaac-Velocity-Flat-Rex-Play-v0` | `RexFlatEnvCfg_PLAY` | Flat terrain evaluation variant with all domain randomization disabled. Use for policy validation, benchmarking, and sim-to-real transfer. Deterministic behavior for reproducible gait analysis. |
| `Isaac-Velocity-Rough-Rex-v0` | `RexRoughEnvCfg` | Velocity tracking on procedurally generated rough terrain with height-field noise, slope variation, and obstacle gaps. Domain randomization includes terrain geometry, friction, and push recovery. |
| `Isaac-Velocity-Rough-Rex-Play-v0` | `RexRoughEnvCfg_PLAY` | Rough terrain evaluation with fixed terrain seeds and no randomization. Validates generalization to unseen rough terrain layouts and measures robustness under consistent conditions. |
| `Isaac-Stand-Rex-v0` | `RexStandEnvCfg` | Static balancing task with zero velocity command. Trains postural stability with randomized CoM shifts, external pushes, and joint configuration noise. Foundation for standing recovery behaviors. |
| `Isaac-Stand-Rex-Play-v0` | `RexStandEnvCfg_PLAY` | Standing evaluation with deterministic conditions. Measures static stability margin, sway amplitude, and disturbance rejection without training noise. |

<!-- TODO: Add terrain comparison image -->
<div align="center">

![Terrain Types](docs/images/terrain_comparison.png)\
*Flat (top) vs. procedurally generated rough terrain (bottom)*

</div>

---

## System Architecture
The overall Rex training, simulation, validation, and hardware deployment
pipeline is shown below.

<div align="center">

![Rex System Architecture](docs/images/diagram.png)\
*Overall system architecture: RL training, Isaac Lab integration, sim-to-sim
validation, and hardware deployment*

</div>

### Repository Structure

```
rex/
├── __init__.py                  # Gym environment registration (6 tasks)
├── rex_velocity_env_cfg.py      # Base LocomotionVelocityRoughEnvCfg
│                                 #   → scene, rewards, events, terrain
├── quad_ws_articulation.py      # QUAD_WS_CFG
│                                 #   → robot articulation + MG996R actuator model
├── flat_env_cfg.py              # Flat-terrain training & play configs
├── rough_env_cfg.py             # Rough-terrain training & play configs
├── stand_env_cfg.py             # Standing / balance task configs
└── agents/
    ├── rsl_rl_ppo_cfg.py        # RSL-RL PPO runner configurations
    ├── rl_games_flat_ppo_cfg.yaml
    ├── rl_games_rough_ppo_cfg.yaml
    ├── skrl_flat_ppo_cfg.yaml
    └── skrl_rough_ppo_cfg.yaml
```

---

## Actuation Model

Rex uses Isaac Lab’s `DCMotorCfg` to represent the MG996R servo actuators as DC-motor-like joints with a torque-speed saturation envelope.

The MG996R is specified by TowerPro to provide approximately 1.1 N·m of stall torque at 6 V and 0.15 s per 60° of no-load motion at 6 V. The Isaac Lab configuration uses a 1.1 N·m effort limit and a 6.0 rad/s no-load velocity limit.

Isaac Lab’s DC-motor actuator model applies a linear torque-speed saturation curve: available joint effort decreases from the stall-torque limit at zero velocity to zero at the no-load velocity limit. This prevents the policy from receiving unrealistic torque at high joint velocities and creates a more representative actuation envelope for a low-cost servo platform.

### Actuator Parameters

| Parameter | Value | Interpretation |
|---|---:|---|
| `saturation_effort` | 1.1 N·m | Nominal MG996R stall torque at 6 V |
| `effort_limit` | 1.1 N·m | Upper bound on computed joint effort |
| `velocity_limit` | 6.0 rad/s | Conservative no-load joint-speed estimate |
| `stiffness` | 25 N·m/rad | PD position gain used by the actuator model |
| `damping` | 0.5 N·m·s/rad | PD velocity damping |
| `armature` | 0.01 kg·m² | Added joint-space inertia for numerical stability |

The PD gains and armature value are simulation and control parameters selected for stable training within the modeled actuation envelope. They are not intrinsic MG996R datasheet specifications.

---

## Domain Randomization

The flat-terrain training configuration randomizes physical, actuation, initial-state, and disturbance parameters to improve robustness to model mismatch and deployment variation.

| Randomization | Range / setting | Mode | Motivation |
|---|---:|---|---|
| Rigid-body mass | 0.8–1.2× nominal | Reset | Payload, battery, and assembly variation |
| Base center of mass | ±3.5 cm in x, y, z | Reset | Assembly tolerance and payload placement |
| Static/dynamic friction | 0.65–1.35× nominal | Reset | Surface and contact variation |
| Joint friction | 0.85–1.15× nominal | Reset | Gear wear and temperature effects |
| Joint armature | 0.85–1.15× nominal | Reset | Unmodeled inertia and actuator variation |
| Actuator stiffness | 0.85–1.15× nominal | Reset | Servo and control-model mismatch |
| Actuator damping | 0.85–1.15× nominal | Reset | Servo and control-model mismatch |
| Initial joint position | ±0.04 rad offset | Reset | Reset and assembly variation |
| Initial joint velocity | ±0.1 rad/s | Reset | Initial-state variation |
| External velocity perturbation | X: ±0.15 m/s; Y: ±0.1 m/s; yaw: ±0.15 rad/s | Interval, 10–15 s | Disturbances and interaction |
| Action delay | 1–2 simulation steps | Reset | Communication and control latency |

Action delay is included when supported by the installed Isaac Lab version. It introduces a one- to two-step delay between policy actions and environment execution.

The randomization ranges are engineering choices intended to cover plausible variation in a low-cost servo platform. They are not yet derived from a systematic hardware-characterization campaign.

---

## Observation Noise

Proprioceptive observations are corrupted during training to reduce reliance on idealized sensor measurements.

| Observation | Noise model | Range |
|---|---|---:|
| Base linear velocity | Additive uniform | ±0.2 m/s |
| Base angular velocity | Additive uniform | ±0.15 rad/s |
| Projected gravity | Additive uniform | ±0.1 |
| Joint position | Additive uniform | ±0.03 rad |
| Joint velocity | Not corrupted in the canonical flat configuration | — |

---

## Reward Design

The flat-terrain reward combines velocity tracking, gait timing, posture, smoothness, energy, and contact penalties. The weights were selected to prioritize command tracking while discouraging hopping, excessive joint acceleration, actuator effort, and base contact.

| Reward term | Weight | Purpose |
|---|---:|---|
| `track_lin_vel_xy_exp` | 20.0 | Track commanded XY velocity |
| `track_ang_vel_z_exp` | 8.0 | Track commanded yaw rate |
| `feet_air_time` | 40.0 | Encourage swing-phase timing |
| `base_height_l2` | -3.0 | Maintain 0.17 m target height |
| `flat_orientation_l2` | -15.0 | Discourage base tilt |
| `lin_vel_z_l2` | -3.0 | Discourage vertical bouncing |
| `dof_acc_l2` | -2.5e-7 | Penalize joint acceleration |
| `action_rate_l2` | -0.01 | Penalize abrupt actions |
| `joint_vel_l2` | -0.1 | Penalize excessive joint speed |
| `dof_torques_l2` | -1.0e-5 | Penalize actuator effort |
| `survival` | 0.0 | Disabled in the canonical flat configuration |

The reward configuration is empirical and platform-specific. It has not been subjected to a systematic hyperparameter sweep.

---

## Training Configuration

Policies are trained using RL-Games PPO with an actor-critic network.

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| RL framework | RL-Games |
| Network | Actor-critic MLP |
| Hidden units | 512, 256, 128 |
| Activation | ELU |
| Seed | 42 |
| Maximum epochs | 1500 |
| Horizon length | 24 |
| Mini-batch size | 24576 |
| Mini-epochs | 5 |
| Learning rate | 1e-3 |
| Learning-rate schedule | Adaptive |
| Target KL divergence | 0.01 |
| PPO clipping parameter | 0.2 |
| Discount factor | 0.99 |
| GAE parameter | 0.95 |
| Entropy coefficient | 0.005 |
| Gradient norm | 1.0 |
| Reward scale | 0.6 |
| Mixed precision | Enabled |
| Value normalization | Enabled |

The provided configuration was adapted from an Isaac Lab ANYmal C rough-terrain configuration. It is not an unmodified ANYmal C policy or training setup.

---

## Quick Start

### Prerequisites

- [Isaac Sim](https://developer.nvidia.com/isaac-sim) ≥ 4.0
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) ≥ 1.0
- Python 3.10+

### Installation

The reinforcement learning training pipeline, reward-shaping strategy, and environment configuration extend the existing ANYmal locomotion framework in Isaac Lab. The ANYmal reference provides the foundational velocity-tracking locomotion architecture, while domain-randomization methodology, and curriculum design are built from scratch, systematically adapted for a smaller, servo-actuated quadruped platform with discrete actuator dynamics. Place the package under:

```bash
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/rex/
```

Or register it as an external extension per the [official docs](https://isaac-sim.github.io/IsaacLab/).

> **⚠️ Asset Path:** Update the `usd_path` in `quad_ws_articulation.py` to point to your local `spot2.usd` before training.

### Training

**Flat terrain (RSL-RL):**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Rex-v0 --headless
```

**Rough terrain (RSL-RL):**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Rex-v0 --headless
```

---

## Results

<div align="center">

![Walking Result](docs/videos/Flatwalking.gif)\
*Baseline flat-terrain walking policy without domain randomization*\
![DomainRandomization](docs/videos/rl-video-step-0.gif)\
*Final Flat Terrain with domain randomization*

</div>

### Terrain-Scale Mismatch and Platform Kinematic Constraints

A critical prerequisite for rough-terrain training on the Rex platform is the recognition that **standard Isaac Lab rough terrain is kinematically incompatible with the SpotMicro form factor**. The SpotMicro is a small-scale quadruped with limited ground clearance, short leg stroke, and MG996R servo actuators that exhibit bounded torque and positional accuracy. Standard procedurally generated rough terrain — designed for full-size platforms such as ANYmal or Unitree Go2 — features obstacle gaps, height-field amplitudes, and slope gradients that exceed the reachable workspace and collision-free envelope of the 12-DOF SpotMicro leg kinematics.

Preliminary evaluation confirmed that the default rough-terrain configuration produces **frequent self-collision between the chassis and terrain features**, **kinematic singularities in the leg Jacobian during stance**, and **saturated actuator commands** that destabilize the policy before any meaningful learning signal can accumulate. The root cause is not policy failure but a **mismatch between terrain geometry and the robot's physical scale**: terrain perturbations on the order of the robot's hip height cannot be treated as traversable obstacles for a platform with centimeter-scale leg stroke.

Consequently, rough-terrain training for Rex requires **customized terrain generation** with parameters scaled to the SpotMicro kinematic envelope. Specifically:

- **Height-field amplitude** must be bounded to a fraction of the leg's maximum vertical stroke, ensuring the base remains within the statically reachable workspace without requiring extreme joint configurations.
- **Obstacle gap spacing** must not exceed the maximum reachable stride length derived from the forward kinematics, preventing the policy from being asked to span unachievable distances.
- **Slope gradients** must respect the friction-limited tipping margin of the small chassis, whose center of mass sits close to the support polygon boundary under even modest inclines.

This constraint is methodological, not merely engineering: training against terrain that the hardware cannot physically traverse would produce a **distributional mismatch between training and deployment**, causing the policy to learn recovery behaviors that are kinematically unrealizable on the physical platform. The customized rough-terrain curriculum was validated against the embedded Pygame kinematic model and the CAD-derived reachable workspace before integration into the Isaac Lab training pipeline.

### Joint Mapping

Isaac Lab's articulation joint ordering does not follow a per-leg (hip-elbow-knee-toe) grouping. Instead, joints are indexed **by joint type across all four legs before moving to the next type** — all four elbow joints first, then all four knee joints, then all four toe joints. This ordering must be respected when mapping actuator commands, observations, and the DCMotorCfg actuator model to the physical MG996R servos, or joint commands will be applied to the wrong physical joint.

---

## Sim-to-Sim Validation

Before physical hardware deployment, trained policies undergo systematic **sim-to-sim transfer validation** from Isaac Lab to PyBullet.

### Validation Pipeline

1. **Train in Isaac Lab** — Export the trained policy checkpoint (`.pt`) after convergence on flat or rough terrain.
2. **Convert to ONNX** — Use Isaac Lab's export utility to convert the PyTorch policy into ONNX format for framework-agnostic inference.
3. **Load in PyBullet** — Instantiate a PyBullet simulation of the SpotMicro URDF with matched joint limits, inertial properties, and actuator dynamics on the Jetson Orin Nano embedded platform.
4. **Run inference** — Feed identical velocity commands and quantitatively compare base trajectories, foot contact patterns, and stability margins between Isaac Lab and PyBullet.

### Methodological Purpose

Sim-to-sim validation isolates **physics-engine discrepancies** from **sim-to-real gaps**. If the policy fails in PyBullet while performing reliably in Isaac Lab, the failure mode lies in physics parameterization (contact friction, stiffness, integration timestep) rather than in the policy's learned representation. This step provides a necessary precondition for sim-to-real deployment by ensuring the policy is sufficiently robust to survive transitions across divergent physics engines — a proxy for the perturbations encountered on physical hardware.

<div align="center">

![S2S](docs/videos/strafe2.gif)\
![Sim](docs/videos/strafe.gif)\
![Sim1](docs/videos/walk.gif)\
*Sim-to-sim transfer results under varying velocity commands*

</div>

---

## License

Licensed under the Apache License 2.0. See the `LICENSE` file for the full text.

---

## ⚠️ Disclaimer

This repository interfaces with physical hardware (servo-actuated legged systems). Trained policies and control code may produce unexpected or dynamically unsafe motion. Use at your own risk — validate thoroughly in simulation first, maintain physical clearance during hardware trials, and employ appropriate safety measures (emergency stop, supervision, protective equipment) before operating on real hardware. The author(s) assume no liability for damage, injury, or loss resulting from use of this code or any hardware built to match it.

---

## References

NVIDIA, "Isaac Sim," NVIDIA Developer, 2024. Accessed: Aug. 7, 2026. [Online]. Available: https://developer.nvidia.com/isaac-sim\

NVIDIA, "Isaac Lab," NVIDIA Developer, 2024. Accessed: Aug. 7, 2026. [Online]. Available: https://developer.nvidia.com/isaac/lab\

C. Schwarke et al., "RSL-RL: A Learning Library for Robotics Research," arXiv preprint arXiv:2509.10771, Sep. 2025. Accessed: Aug. 7, 2026. [Online]. Available: https://github.com/leggedrobotics/rsl_rl\

SpotMicroAI, "SpotMicroAI Documentation," SpotMicroAI, 2026. Accessed: Aug. 7, 2026. [Online]. Available: https://spotmicroai.readthedocs.io/en/latest/\

NVIDIA, "Isaac Lab — Locomotion Velocity Tracking," Isaac Lab Documentation. Accessed: Aug. 7, 2026. [Online]. Available: https://isaac-sim.github.io/IsaacLab/\

robot mania, How to Train a Custom Quadruped Robot to Walk Using Isaac Lab. (Mar. 9, 2025). Accessed: Aug. 7, 2026. [Online Video]. Available: https://www.youtube.com/watch?v=z62oU4hM1xM\

O. Omotuyi, D. Hoeller, and T. Burnham, "Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab," NVIDIA Technical Blog, Jun. 17, 2024. Accessed: Aug. 7, 2026. [Online]. Available: https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/\

M. Kim, J.-S. Kim, and J.-H. Park, "Automated Hyperparameter Tuning in Reinforcement Learning for Quadrupedal Robot Locomotion," Electronics, vol. 13, no. 1, p. 116, 2023. https://doi.org/10.3390/electronics13010116.\
