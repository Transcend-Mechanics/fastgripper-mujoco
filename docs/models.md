# Select hardware and fingertips

The package includes an explicit model selector. Existing examples, the learning
environment and the bundled policy continue to use hardware **v1.0** by default.

| Hardware | Asset folder | Fingertip choice | Scope |
| --- | --- | --- | --- |
| `v1.0` | `assets/fastgripper` | `stock-metal` | Original rigid-finger baseline and bundled policy |
| `v1.1` | `assets/fastgripper_v1_1` | `stock-metal` | Separate rigid carriers and screw-fastened child tips; provisional physics |

Software 0.4.0 and the v1.1 asset version 0.3.0 are separate version numbers.
The v1.1 visuals retain the exterior linkage silhouettes as full-depth solid
profiles. Assembly holes, countersinks, internal cavities and sheet interfaces
are omitted. The worm, both toothed wheels, bent housing and motor casing are
visible. The gears follow the existing shaft/linkage joints; tooth-contact
forces are not simulated. Collision geometry and physical parameters retain
the prior model's effective 20:1 transmission with provisional losses/backlash.
Only `stock-metal` is selectable in this package. The selector does not include
the TPU Fin Ray concept, import arbitrary customer geometry or swap parts in a
running simulation.

## Run the selected model

```sh
uv run --locked python -m fastgripper_sim.models --list
uv run --locked python -m fastgripper_sim.models --verify
uv run --locked python -m fastgripper_sim.variant_demo --hardware v1.1 --tip stock-metal
uv run --locked python -m fastgripper_sim.variant_demo --hardware v1.1 --tip stock-metal --render
uv run --locked python -m fastgripper_sim.variant_demo --hardware v1.1 --tip stock-metal --viewer
```

`--list` reads the verified asset catalogs. `--verify` also compiles every
listed gripper and fixture scene. The headless demo writes
`outputs/grasp-v1.1-stock-metal.json`; `--render` adds a GIF and PNG with the
same prefix. Change the prefix with `--output outputs/my-trial`. Reports include
the hardware revision, tip ID, asset version, manifest hash and selected XML
hash under `model_selection`.

`--viewer` opens the existing interactive viewer with the chosen scene and a
repeating grasp cycle. It delegates to `mjpython` on macOS through the viewer's
existing launcher. This interactive mode does not produce the JSON report.
`--viewer` and `--render` are alternative modes.

## Select from Python

```python
import mujoco
from fastgripper_sim.models import available_tips, load_model, model_info, model_path

print(available_tips("v1.1"))  # ('stock-metal',)

# Fixed-base gripper without the table, object or external lift fixture.
path = model_path(hardware="v1.1", tip="stock-metal")
model = load_model(hardware="v1.1", tip="stock-metal")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)

# The bundled contact/grasp scene is an explicit choice.
scene = load_model(hardware="v1.1", tip="stock-metal", scene=True)
print(model_info(hardware="v1.1", tip="stock-metal", scene=True))
```

`model_path()` returns `model.xml`; `scene=True` returns `scene.xml`. Both the
path helper and loader verify the selected asset manifest. Unknown hardware or
tip IDs raise `ValueError` with the supported choices. Calling these functions
without arguments selects the unchanged v1.0 baseline.

For a customer's robot scene, use the selected asset's `mechanism.xml` and
`body.xml` fragments as described in the [scene integration guide](integration.md)
and the [v1.1 asset guide](../assets/fastgripper_v1_1/README.md). Loading a
different model does not automatically adapt an existing Gymnasium environment,
robot mount or policy to its changed mass and contact geometry.

## v1.1 interpretation

Each carrier remains a moving four-bar link. Its removable fingertip is a fixed
child body, so it has separate visual, inertia and contact definitions without
an additional joint. The [fingertip catalog](../assets/fastgripper_v1_1/fingertips.json)
records both attachment frames, body names and contact sites. It describes
simulation attachments; no manufacturing interface drawing is released.

The stock demo commands conservative motor endpoints of **-3.0 to 10.4 rad**,
corresponding nominally to about **104.1 mm open and 3.0 mm closed**. The CAD
kinematic envelope of **0–171.5 mm** is recorded separately. The demo does not
exercise that full envelope, and neither range establishes measured hardware
stops. Actual aperture under load can differ because contacts, backlash and
constraints affect the result. Use the two `finger_*_contact_surface` sites
for instantaneous opening.

Mass, friction, motor behavior and attachment rigidity remain provisional.
The unresolved physical motor/mount fit is not validated by the simplified
public collision envelopes. Grasping the demo block is a simulation check,
not a payload rating or proof that the physical v1.1 assembly fits together.

The bundled trained checkpoint was trained and checked on v1.0. No v1.1 policy
transfer is claimed. For another tip, follow the
[manual rigid-tip customization workflow](fingertips.md#customize-a-v11-rigid-tip).
