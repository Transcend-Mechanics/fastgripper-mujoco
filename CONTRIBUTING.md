# Contributing

Bug reports, documentation improvements, integrations and simulation tests are
welcome. Include the package and MuJoCo versions, operating system, selected
model, reproduction command and relevant error or telemetry. A small scene
that demonstrates the issue is especially useful.

From a checkout:

```sh
uv sync --locked --extra learn
uv run --locked python -m fastgripper_sim.assets
uv run --locked python -m fastgripper_sim.demo
uv run --locked --extra learn python -m pytest
```

Rendering is optional for headless checks. For a controller or physics change,
also run the affected examples and report the effect on sustained grasp,
release, constraint error and warnings. Keep code, model and hardware version
changes explicit. Preserve third-party attribution and describe any changed
simulation assumptions.

Do not submit proprietary manufacturing files or third-party assets you lack
permission to redistribute. For compatible fingertips, distinguish your own
accessory geometry from the gripper interface and include the intended hardware
revision, units, inertial/contact assumptions and validation status. Read
[LICENSE](LICENSE) before contributing code or assets; different components
can have different terms.
