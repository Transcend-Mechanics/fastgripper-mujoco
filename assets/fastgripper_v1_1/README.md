# FastGripper hardware v1.1 simulation assets

This is the stock bare-metal fingertip configuration. `model.xml` is a fixed-base
gripper, and `scene.xml` adds a table, free block and movable test fixture. Both
have an `open` keyframe. `fingertips.json` describes the selected `stock-metal`
configuration, fixed tip attachment frames, separate masses/inertias and contacts.

## Simulation assembly

Include `mechanism.xml` at the top level of a host MJCF and `body.xml` inside its
attachment body named `gripper_mount`, with `childclass="fastgripper"`. The body
fragment has no freejoint, fixture weld, world objects or keyframes. Defaults
are scoped to `fastgripper`. Namespace all identifiers for multiple grippers.
Keep the 50 microsecond physics timestep until convergence is checked again.

Opening is along X, hinge axes along Y and fingers point toward +Z. `gripper_base`
is the simulation root frame, not a measured robot flange. `jaw_tcp_reference`
is the contact-center midpoint at driver angle zero. The instantaneous TCP is
the midpoint of the two `finger_*_contact_surface` sites; it moves with aperture.

`worm_motor` drives `worm_joint` with motor-side radians, rad/s and N m. Raw
control is shaft velocity with a native torque cap. The runtime controller maps
POS/SPEED/FORCE percentages and stall behavior; position 0 is open and 100 is
closed. The nominal commanded aperture is about 3–104 mm; the geometric
envelope is 0–171.5 mm. These conservative demonstration endpoints are not
measured physical stops or a validation of gear engagement across the full
geometric envelope. Contact forces are simulated
ground truth, not a claim that hardware measures fingertip force.

## Detachable fingertips

The lower `finger_positive_x` / `finger_negative_x` bodies are the linkage
carriers. Their fixed children `fingertip_positive` / `fingertip_negative` hold
the removable stock tips, with independent visual meshes, inertias, contact
boxes and surface sites. No new moving joint is introduced at the screwed seam.

To prototype another tip, retain the carriers and update each child tip's
geometry, inertia and contacts together. Recompute travel calibration/TCP and
rerun grasp, release and clearance tests. This version has one catalog choice,
`stock-metal`; it does not offer a validated flexible fingertip or a runtime
hot-swap API. Simulation attachment coordinates do not specify bolt patterns,
threads, tolerances, tightening torque or a manufacturing-ready interface.

## Disclosure and physical scope

Exterior meshes retain the linkage silhouettes, worm helix, both toothed wheels,
and the bent gear housing. Assembly bores are filled through their depth before
meshing; no hidden bore walls or internal shells are retained. The motor casing
uses solid sections at the assembled exterior dimensions, without motor internals.
The worm and coupling rotate with the motor shaft; the two wheels rotate with
their driven links. Functional wheel seats remain open around the gears.

These gear surfaces are visual-only. The motor drives the existing effective
20:1 transmission with provisional backlash and friction. Gear teeth do not
generate contact forces, and their appearance does not validate engagement,
load-dependent efficiency or self-locking. Collision proxies are unchanged.
Articulation dimensions and explicit physical parameters remain visible. The
source CAD stays outside this bundle. Approximate housing, motor, link and
carrier collision envelopes screen external obstacles; exact stock gripping
boxes retain the source physics contact planes and friction. Self-contact is
disabled with bit masks (gripper 1, object 2, environment 4). This model does not
validate internal fit, manufacturing clearances, gear contact or hardware stops.

Parameters are provisional and not hardware-calibrated. Do not treat the
demonstration as a payload rating. A real arm adapter and measured calibration
remain separate work. `manifest.json` hashes every other payload file. The Simulation and Accessory License in LICENSE covers this model.
See the root LICENSE for the separate software and model scopes.
