# Add FastGripper to a simulation

The reusable model and Python controller can be used without the packaged task.
Keep three responsibilities distinct in your environment: compose the scene,
map actions into gripper commands, and advance physics.

## Load and command one gripper

This complete example runs the fixed-base gripper. Use `scene.xml` instead of
`model.xml` to load the bundled table, block and lift fixture as well.

```python
import mujoco
from fastgripper_sim.assets import asset_path
from fastgripper_sim.controller import GripperController

model = mujoco.MjModel.from_xml_path(str(asset_path("model.xml")))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)
mujoco.mj_forward(model, data)

gripper = GripperController(model, data)
gripper.set_enabled(True)
gripper.command(position_pct=100, speed_pct=100, force_pct=30)

for _ in range(round(4.0 / model.opt.timestep)):
    gripper.update(model.opt.timestep)
    mujoco.mj_step(model, data)

print(gripper.snapshot()["aperture_m"])
```

`update()` supplies controls but does not advance time. Call it before stepping.
Create one model, data and controller instance per independent environment:
the controller changes actuator parameters in the model.

| Command or measurement | Meaning |
| --- | --- |
| `position_pct=0` / `100` | Nominal open / closed calibration endpoints |
| `speed_pct` | Percentage of modeled maximum motor speed; zero selects the firmware-style 100% default |
| `force_pct` | Percentage of modeled motor torque cap; zero selects the 50% default |
| `set_enabled(False)` | Disable active motor output |
| `stop()` | Stop seeking; the controller remains enabled if it was enabled |
| `snapshot()["aperture_m"]` | Distance between the two contact-surface sites, in metres |
| `motor_position_rad` / `motor_torque_nm` | Motor-side angle and torque, not jaw angle or fingertip force |

FORCE is a motor torque limit, not a force request in newtons. Raw actuator
`ctrl` is not position percent. `set_torque()` is a separate, torque-capped
validation mode; use the position command interface when reproducing the
firmware-style behavior.

Commands are events. Reissuing the same command resets seeking and the stall
observer. Hold a command across policy steps until its requested values change.
After an environment reset, reset physics, call `mj_forward()`, then call
`controller.reset()` and `set_enabled(True)` before issuing new commands.

## Compose with another robot

The reusable asset is split into two MJCF fragments:

1. Include `mechanism.xml` once at the top level. It defines scoped defaults,
   meshes, link constraints, transmission, actuator and sensors.
2. Under your robot's attachment body, add a body named `gripper_mount` with
   `childclass="fastgripper"`, then include `body.xml` there. This adds the
   chassis inertial once and the moving mechanism.

The body fragment has no world objects, free joint, fixture weld or keyframes.
The schematic structure is:

```xml
<mujoco>
  <!-- Supply your compiler, physics options and robot definitions. -->
  <include file="gripper/mechanism.xml" />
  <worldbody>
    <body name="robot_attachment">
      <body name="gripper_mount" childclass="fastgripper">
        <include file="gripper/body.xml" />
      </body>
    </body>
  </worldbody>
</mujoco>
```

Resolve the fragment paths through `asset_path()` or copy the complete asset
directory into your scene project while retaining its license. Resolve its
mesh paths relative to that asset directory. Keep the supplied `50e-6` second
timestep and `implicitfast` integrator until you have demonstrated convergence
for a change. Do not copy the demo's keyframe vector into a larger robot model:
initialize the gripper's joints by name while preserving your robot's state.
The working [UR composition code](../fastgripper_sim/arm_demo.py) demonstrates
asset path resolution, fragment composition and consistent joint initialization.

The base frame opens along X, uses hinge axes along Y and points toward the
fingers along +Z. `gripper_base` marks the model origin, not a measured flange
interface. Apply your actual adapter transform and inertial to the attachment.
The instantaneous jaw TCP is the midpoint of
`finger_positive_contact_surface` and `finger_negative_contact_surface`; its
height changes with opening. `jaw_tcp_reference` is only the reference-pose TCP.

The current controller expects the shipped names, including `worm_joint` and
`worm_motor`. Multiple grippers in one scene require namespacing both the MJCF
and controller lookups; automatic prefix selection is not supplied in this
release.

Gripper collision shapes use bit 1, objects bit 2 and environment bit 4. Check
the masks in [the asset guide](../assets/fastgripper/README.md) when composing
your scene. Audit intentional adapter overlaps and unexpected contacts before
using the model for a new task.

## Add policy actions

Use your environment's arm actions for the robot and send only the gripper's
action fields to `GripperController`. A policy can run at a lower rate than
physics; hold its command while performing the intervening physics steps.
The bundled [learning task](../learning.md) uses 20 Hz policy actions and
1 ms control updates with 20 native 50 microsecond steps per update.

`GripLiftEnv` is a small task example, not a universal wrapper for any robot
scene. It expects its fixture, free block, named contacts and observations.
For an existing training framework, reuse the asset and controller, then define
your own reset, actions, observations, rewards and success conditions. Simulator
contact forces and object pose are available for analysis but are privileged
measurements if your intended hardware cannot observe them.

UR3e and UR5e training can use the [direct MuJoCo models](../arm.md). UR commands
are unnecessary for that path; [URSim](../ursim.md) is an optional way to test
PolyScope programs and the URCap.
