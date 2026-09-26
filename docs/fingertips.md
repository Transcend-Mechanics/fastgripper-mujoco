# Custom fingertips and attachments

FastGripper is intended to support customer-designed fingertips and attachments.
The licensing split reserves the gripper's mechanical design while
allowing compatible accessories. The repository's [license overview](../LICENSE)
and [asset/accessory terms](../LICENSE-ASSETS) define the scope. The accessory
grant lets an author retain rights in their own design.

## What this release includes

The package includes the v1.0 baseline and an explicitly selectable v1.1 model.
Both offer `stock-metal` fingertips. In v1.1 the lower carriers and removable
tips are separate bodies, with each tip fixed to its carrier. Each has its own
geometry and inertia. Select a
bundled configuration before creating a model:

```python
from fastgripper_sim.models import available_tips, load_model

print(available_tips("v1.1"))  # ('stock-metal',)
model = load_model(hardware="v1.1", tip="stock-metal")
```

See [hardware and fingertip selection](models.md) for the demo commands.
This selector chooses a complete bundled model; it is not a mesh plugin or a
runtime hot-swap API. The TPU Fin Ray concept and a mechanical mounting drawing
are not included.

The visual outlines and v1.1 attachment frames are simulation geometry. They do not define
bolt patterns, mating tolerances, approved fasteners or a manufacturing
interface. Request the interface specification for your hardware revision
before making a physical accessory; do not infer mounting dimensions from a
render or the simplified mesh.

## Customize a v1.1 rigid tip

Customer tips currently require manual MJCF editing. Start with a separate
versioned copy of the v1.1 asset folder so relative mesh paths remain valid:

```python
from pathlib import Path
import shutil
from fastgripper_sim.models import model_path

source = model_path(hardware="v1.1").parent
shutil.copytree(source, Path("custom_models/my-rigid-tip-v0.1"))
```

Edit `body.xml` in that copy. Keep the two carrier bodies and their linkage
joints intact. Their removable tips are the following fixed children:

| Side | Carrier to retain | Child tip to edit | Surface site to retain/update |
| --- | --- | --- | --- |
| Positive | `finger_positive_x` | `fingertip_positive` | `finger_positive_contact_surface` |
| Negative | `finger_negative_x` | `fingertip_negative` | `finger_negative_contact_surface` |

`fingertips.json` records each child's current transform, visual mesh, inertia,
contact box and contact site in that tip's frame. Keep its fixed transform when
the new tip uses that same attachment datum; do not add a moving joint at the
screw-fastened seam. Replacing only a visual mesh changes appearance, not
physical contact or mass. Update these items together:

| Item | What must agree |
| --- | --- |
| Attachment transform | The replacement geometry's frame relative to the carrier, in metres and radians |
| Visual geometry | The shape shown in rendered observations |
| Collision geometry | The gripping face and clearance shape actually used by the solver |
| Mass and inertia | Each tip's own mass, center of mass and inertia; retain the carrier's independent properties |
| Contact properties | Friction and compliance for the intended material and object |
| Contact-surface sites | Jaw opening, instantaneous TCP and controller telemetry |
| Travel and task setup | The replacement's closure limit, opening clearance and pregrasp position |

1. Add your visual mesh files and their `<mesh>` entries in `mechanism.xml`.
   Replace the visual geoms inside each child tip in `body.xml`.
2. Replace each child's contact/envelope geoms. Retain the named gripping geoms
   `finger_positive_contact` and `finger_negative_contact` so contact telemetry
   remains meaningful. Preserve collision masks: gripper `1`, object `2`,
   environment `4`.
3. Replace the child's `<inertial>` using the new tip's mass and center/inertia
   in the child frame. Do not add the old tip mass to the new mass or include
   the carrier again. Keep geometry masses at zero where explicit inertials
   are used, following the supplied model.
4. Move the `finger_*_contact_surface` sites to the replacement's actual inner
   gripping surfaces. Update `jaw_tcp_reference` for the zero-angle reference
   pose; instantaneous TCP is the midpoint of the two contact sites.
5. Recompute closure and opening for the new shape, then update motor command
   endpoints, joint limits and related range metadata in `mechanism.xml` and
   `body.xml`. Update the `open` keyframes in both `model.xml` and `scene.xml`
   so reset poses agree with the new travel. Review the scene's object
   placement and fixture clearance.
   The stock demo's approximately 3.0–104.1 mm command aperture is different
   from its 0–171.5 mm geometric envelope. Do not copy either range blindly.
6. Give your configuration a new tip ID and asset version. Update the copied
   catalog and file hashes to describe it, and retain the original licenses
   and a modification notice. The stock selector will not discover an external
   folder automatically; run the customized scene by path:

```sh
uv run --locked python -m fastgripper_sim.demo \
  --model custom_models/my-rigid-tip-v0.1/scene.xml --output outputs/my-rigid-tip
uv run --locked python -m fastgripper_sim.viewer \
  --model custom_models/my-rigid-tip-v0.1/scene.xml --demo
```

Path-based demos compile the supplied XML but do not verify a custom manifest
or add catalog provenance. Record your custom asset version, hashes and
material assumptions alongside their reports.

Use simple contact shapes or an appropriate convex decomposition for hollow
or concave tips. Ordinary mesh collision uses a convex hull, so a detailed
visual mesh may otherwise produce contact across an intended opening. Preserve
the model's collision-mask conventions when adding shapes.

Before training, check open/closed clearance, bilateral contact with your test
object, mass and inertia, held-grasp stability, slip and release. A changed tip
also changes the task seen by a policy; the bundled checkpoint and reference
reports do not validate that new configuration.

The original v1.0 bodies combine their carrier/finger properties. When editing
that baseline, its `finger_*_solid_visual` geoms are visual-only, while
`finger_*_contact` and `finger_*_upper_envelope` supply contacts. Recompute the
combined inertia for those bodies; the separate-child instructions above apply
specifically to v1.1.

## Compliant tips

A rigid-body model of a TPU tip shows mounting and motion but cannot predict
its bending, contact adaptation or screw retention. A future compliant model
needs a chosen deformation representation and measured material/contact
parameters. Label rigid previews accordingly and validate a printed prototype
before treating its simulated grip as a prediction.

The v1.1 catalog now records attachment frames, compatible hardware, separate
inertias, visual/contact geometry and provisional material status for
`stock-metal`. Adding a compliant variant requires a deformation
model and validation work; the current selector makes no TPU elasticity claim.
