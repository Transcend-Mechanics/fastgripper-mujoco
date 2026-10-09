# Versions and reproducibility

Software, simulation assets and physical hardware have separate versions.

| Version | Identifies |
| --- | --- |
| Python package `0.4.0` | Upgrades v1.1 visual fidelity with solid exterior linkage profiles |
| v1.0 asset `0.1.1` | License/provenance metadata update; the baseline physical model is unchanged |
| v1.1 asset `0.3.0` | Restored worm, wheels and housing; solid linkage profiles; unchanged transmission physics |
| Hardware `v1.0` / `v1.1` | The physical gripper revision represented by the selected model |

The default remains the tested **hardware v1.0** runtime. Select **v1.1**
explicitly using the [model selection API](models.md). The existing trained
policy, UR demonstrations and URSim example still use v1.0; the v1.1 stock-tip
addition has its own standalone grasp checks. A software release number must
not be read as the hardware revision. Read the exact
package version from `pyproject.toml` and the installed `fastgripper_sim.__version__`;
read the asset version from the selected asset directory's `manifest.json`.

The retained training and evaluation reports describe their original runtime
and asset versions. They remain historical evidence for the unchanged physics;
the metadata update is not a claim of retraining or a new evaluation. Consult
the release's compatibility record before comparing the new manifest hash
with those original reports.

## Reproduce a result

Record the repository tag or commit, package version, asset manifest hash,
MuJoCo version, dependency lock, task configuration, random seeds and any local
scene changes. Keep the complete model directory: hashing only a top-level
XML file does not cover included fragments or meshes.

```sh
uv run --locked python -m fastgripper_sim.assets
uv run --locked python -m fastgripper_sim.demo
```

The first command rejects missing, modified and undeclared model payload files.
The learning examples also retain configuration and provenance with their
checkpoints and reports. A new simulation-asset release requires a new
validation result even if the Python API is unchanged.

## Release contents

A public release should contain a source archive and an installable wheel,
the license overview and applicable notices, model manifests, a changelog and
the results of the headless acceptance checks. The wheel contains the models
and example artifacts used by the library. Installation instructions do not
assume that a PyPI publication already exists.

Model or contact changes can invalidate a retained policy's assumptions.
Retain earlier versioned assets and clearly state which model a checkpoint
was trained and evaluated with. Future hardware and fingertip variants should
be selectable alongside the baseline rather than silently replacing its
geometry under an existing version.
