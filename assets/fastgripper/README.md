# FastGripper simulation assets

`model.xml` is a reusable fixed-base gripper. `scene.xml` adds a free fixture,
table and freely simulated 50 g block for a grasp/lift/release demonstration.
Both have an `open` keyframe. They require no CAD, private meshes, or FreeCAD.
All physical values remain provisional. This is not a hardware payload rating.

## Composition and frames

Include `mechanism.xml` once at the top level of a robot MJCF. Inside its
attachment body named `gripper_mount`, include `body.xml`. The body fragment
contains the chassis inertial exactly once, then the moving mechanism. It has
no freejoint, world objects, fixture weld, or keyframes. Defaults are scoped to
the `fastgripper` class. Set `childclass="fastgripper"` on the attachment body.
Use a unique namespace when composing more than one gripper.

The base frame retains the modeled gripper coordinates: opening along X,
hinge axes along Y, fingers toward +Z, motor behind the base toward -Z.
`gripper_base` marks the origin; it is not a measured robot flange interface.
The actual mount adapter transform is not supplied. `jaw_tcp_reference` marks
the midpoint of the contact bands at driver angle zero. The instantaneous TCP
is the midpoint of `finger_positive_contact_surface` and
`finger_negative_contact_surface`; its Z coordinate changes with aperture.
Its orientation follows the base. Do not use a constant TCP for every opening.

## Controls and physics

`worm_joint` and `worm_motor` use motor-side radians and N m with unit gear.
Raw `ctrl` is a shaft velocity target in rad/s. POS/SPEED/FORCE percentages,
enable/disable and firmware-style stall detection require the separate runtime
controller. Position 0 means open and 100 means closed. Contact force is
simulation ground truth, not a claim that the real gripper measures it.

Keep the supplied 50 microsecond physics timestep and implicit integrator until
convergence has been demonstrated for changes. The controller outer loop is
100 Hz; the reference Python runner updates it at 1 kHz. Do not continually
reissue identical position commands: each new command resets the stall observer.
The drive includes a torque cap, finite friction, and backlash. The model is
not an infinitely strong position clamp and the object is never welded to it.

## Geometry and contacts

Exterior meshes are small authored solids, not concatenated source triangles.
There are no tooth forms, bolt holes, lamination boundaries or motor internals.
Body inertials, articulation dimensions and transmission parameters are still
explicit. This package therefore does not conceal the mechanism's behavior.

Visual geometry has contacts disabled. Separate approximate housing, motor,
bar, pivot and upper-finger envelopes contact external objects. The original
distal gripping boxes preserve their contact planes, friction and compliance;
the bare metal fingers have no added pads or pad mass. Envelope intersections
within the assembly do not create self contacts: gripper uses `contype=1`,
`conaffinity=6`; object uses bit 2 and environment bit 4. An arm must exclude
intentional wrist/adapter overlaps and verify its own collision configuration.
These envelopes support obstacle screening, not exact CAD clearance analysis.

`manifest.json` hashes every payload file, excluding the manifest itself.
Use the recorded MuJoCo version for regression comparisons. The Simulation and Accessory License in LICENSE permits commercial
simulation and compatible accessories while reserving core mechanical-design
rights. See the root LICENSE for the separate software and model scopes.
