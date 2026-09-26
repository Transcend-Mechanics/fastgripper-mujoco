# Mounted UR5e and UR3e grasp examples

These examples generate their arm references directly in Python. To run actual
PolyScope and FastGripper URCap programs instead, see [the URSim guide](ursim.md).

Run from this package's checkout after `uv sync --locked`:

```sh
uv run --locked python -m fastgripper_sim.arm_demo --arm ur5e --output outputs/ur5e
uv run --locked python -m fastgripper_sim.arm_demo --arm ur3e --output outputs/ur3e
uv run --locked python -m fastgripper_sim.arm_demo --arm ur5e --output outputs/ur5e --render
uv run --locked python -m fastgripper_sim.arm_demo --arm ur3e --output outputs/ur3e --render
uv run --locked python -m fastgripper_sim.arm_demo --arm ur5e --output outputs/ur5e-varied --randomize --seed 8
```

Each command runs headlessly and writes telemetry JSON. Rendering also writes a
GIF and a PNG of the raised hold. `--randomize` changes the object's initial
planar position within a small seeded range and plans the matching approach;
this is a scripted baseline, not perception or a learned arm policy.

The 20-second cycle settles, approaches the block, closes the fingers, lifts
100 mm, holds, lowers, opens and retracts. The block has a free joint and moves
through contact and gravity. The arm's six joints move through bounded servos.
Inverse kinematics uses separate scratch state to prepare the reference
trajectory; live joint positions are assigned only during reset. The trajectory
controller adds reference velocity and gravity feedforward through those same
bounded actuators. No object weld, moving fixture or object pose playback is used.

The success check requires more than 80 mm object rise and bilateral contact
during the assessed raised hold, then return to table height with no finger
contact after release, with no unexpected active contact pairs in the audit.
Telemetry includes both finger normal forces, motor
torque, sampled arm actuator torques, actuator force limits, joint planning
ranges, attachment tracking error and linkage closure error. Those contact
normals are simulator measurements, not sensors exposed by the physical
gripper's bridge.

An additional contact audit samples all active contact pairs at 1 ms control
boundaries, recording peak normal force and penetration. The expected pairs
are each finger with the block, and the table with the block; other pairs remain
visible in the report. Fixed wrist/adapter/chassis bodies share a rigid tree
and use MuJoCo's standard same-body collision filtering.

## Arm sources and limits

The UR5e uses the MuJoCo Menagerie model, commit
`8161bba264d7fa7c99ca301e91e7fb44737676ad`, with its original BSD-3-Clause license,
README and file hashes retained in `assets/ur5e/`. Its files are unchanged.
The upstream package derives from Universal Robots' public description and
provides simplified collisions and joint servos.

[Pinned Menagerie UR5e model](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/universal_robots_ur5e)

The UR3e is independently converted from Universal Robots' official ROS 2
description at commit `89bbe795f38a7ab00fb66fe8831dfff79dc99edf`. The original
BSD-3-Clause license, configuration files, source hashes and conversion details
are retained in `assets/ur3e/`. Its UR3e transforms, masses, centers of mass,
inertia tensors and effort limits come from those configuration files. DAE
visual surfaces and material colors are converted to OBJ; the upstream collision
STLs are retained and MuJoCo uses their convex hulls. The adjacent base/shoulder
pair is explicitly excluded because its stock collision hulls overlap at the
pivot. Other collision pairs remain enabled. The conversion uses nominal
kinematics rather than calibration for an individual robot.

[Pinned official UR3e description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/89bbe795f38a7ab00fb66fe8831dfff79dc99edf)

| Demo parameter | UR5e | UR3e |
| --- | --- | --- |
| Nominal object X from base | 0.50 m | 0.26 m |
| Table top | 0.12 m | 0.06 m |
| Attachment height during grasp / raised hold | 0.38 / 0.48 m | 0.32 / 0.42 m |
| Joint actuator caps, shoulder pan through wrist 3 | 150, 150, 150, 28, 28, 28 N m | 54, 54, 28, 9, 9, 9 N m |

The UR3e is not a scaled UR5e. Its continuous wrist-3 joint remains unlimited
in the physics model; the example uses a finite +/-2 pi planning/control window.
Both demos use ideal bounded joint servos, not Universal Robots' proprietary
controller. URSim and real robot interfaces are not involved.

## Provisional attachment

The runtime composes each arm with the same gripper `mechanism.xml` and
`body.xml`. The gripper mounts beneath `attachment_site`, preserving that site's
position and rotation relative to `wrist_3_link`. The chassis mass is included
once; the fixture's free joint and mocap weld are absent.

For UR5e, the reference is Menagerie's `attachment_site`. For UR3e, that site
aliases the official **tool0** frame, whose +Z points outward. The ROS description
also defines a differently rotated frame named `flange`; use the tool0-to-flange
rotation when attaching the same transform to that frame. Telemetry identifies
the reference frame explicitly. The `flange_position_error_m` fields measure
tracking at `attachment_site`.

The attachment-to-gripper transform is **provisional** for both arms:

```text
translation, metres:  [0.0279919658976, -0.00243382114975, 0.11]
quaternion, wxyz:    [1, 0, 0, 0]
```

It centers the simplified gripper below the attachment with an authored spacer
envelope 30 mm long and 64 mm in diameter. The spacer has an assumed 0.10 kg
mass with uniform-cylinder inertia. Replace this transform, envelope and inertia
with the actual adapter before interpreting installation clearance or loading.
The gripper base's +Z points toward its fingers; the demo's desired attachment
orientation points +Z downward. The approach heights are authored for the
40 x 24 x 30 mm, 50 g block and the opening-dependent finger geometry; they are
not universal TCP calibrations.

Gripper friction, inertias and motor parameters remain provisional. The examples
use a 50 microsecond physics timestep with batched 1 ms control updates; this
retains the small timestep required by the gripper motor model.
