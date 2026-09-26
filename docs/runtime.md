# What is the runtime?

The runtime is the part of the project a simulation user installs and runs.
It combines a MuJoCo model, a Python controller and example environments. It
does not contain manufacturing drawings or require a CAD conversion pipeline.

| Component | Responsibility |
| --- | --- |
| `assets/fastgripper/` | Simplified exterior geometry, rigid-body inertias, articulated linkages, collision shapes and effective transmission |
| `assets/fastgripper_v1_1/` | Separately versioned v1.1 model with rigid removable stock tips and a simulation interface catalog |
| `fastgripper_sim.models` / `variant_demo` | Select a hardware revision and available fingertip configuration explicitly |
| `fastgripper_sim.controller` | Position, speed and torque-limit commands; motor enable and firmware-style stall observation |
| `fastgripper_sim.assets` | Find bundled assets in a checkout or wheel and verify their hashes |
| `fastgripper_sim.demo` / `viewer` | Standalone contact-driven grasp, lift and release |
| `fastgripper_sim.env` | Gymnasium task for gripper-policy learning |
| `fastgripper_sim.arm_demo` | Mounted UR5e and UR3e examples driven by Python |
| `fastgripper_sim.ursim` / `ursim_demo` | Optional connection between a simulated mechanism and the URCap/PolyScope workflow |

The package is named `fastgripper-sim`; Python imports use `fastgripper_sim`.
The repository is named `fastgripper-mujoco`.
The default, bundled policy and mounted examples remain v1.0. See
[model selection](models.md) for the standalone v1.1 stock-tip demonstration.

## What moves the object?

MuJoCo advances the joints and solves gravity, joint constraints and object
contacts. The controller supplies motor commands. A grasped block remains a
free body and can slip or drop; a successful grasp is not implemented by
welding the block to a finger.

In the standalone and learning examples, a prescribed external fixture raises
the gripper. In the mounted examples, bounded arm actuators move the gripper.
The learning example trains gripper commands and does not learn arm motion.

## What is approximated?

The model uses simplified exterior and collision geometry. Its transmission
represents the effective relationship between motor and linkages; gear-tooth
contact is not resolved. The gripper's collision masks suppress self contacts,
so the included envelopes support external obstacle screening, not a complete
mechanical interference check.

Masses, motor response, friction, backlash, travel endpoints and contact
compliance are provisional simulation parameters. The controller captures part
of the firmware behavior, with idealized inner velocity feedback and a native
torque cap. It does not reproduce CAN timing, electronics, physical homing or
every fault mode. A motor-stall observation is not a payload-presence sensor.

The bundled UR examples use nominal robot geometry and provisional attachment
transforms. The direct MuJoCo arm examples use bounded joint servos; the optional
URSim connection exercises controller programs but does not feed MuJoCo contact
forces back into URSim's controller.

Use this runtime to develop scenes, test software interfaces and compare
policies under stated simulation assumptions. Hardware transfer needs measured
parameters, the actual attachment and validation on the physical system.
