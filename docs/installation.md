# Installation

Use Python 3.12 for the locked, tested development environment. The package
declares support for Python 3.12–3.14. The core dependencies are MuJoCo, NumPy
and Pillow; reinforcement learning adds Gymnasium, Stable-Baselines3 and
Matplotlib.

## Run a repository checkout

From the checkout root:

```sh
uv sync --locked
uv run --locked python -m fastgripper_sim.assets
uv run --locked python -m fastgripper_sim.demo
```

Keep `uv.lock` when reproducing results. The asset verification command checks
the declared file list and every payload hash for the gripper and both arms.

## Install into another project

Install the local checkout with pip:

```sh
python -m pip install /path/to/fastgripper-mujoco
```

Or build and install a wheel:

```sh
uv build --wheel
python -m pip install dist/fastgripper_sim-*.whl
```

The wheel includes model assets and retained example artifacts. The library's
`asset_path()` and `example_path()` helpers find them after installation;
applications should not assume assets live beside their own working directory.
These commands do not require a package to have been published to PyPI.

For learning from a checkout:

```sh
uv sync --locked --extra learn
uv run --locked --extra learn python -m fastgripper_sim.evaluate
```

For learning in an existing pip environment, use:

```sh
python -m pip install '/path/to/fastgripper-mujoco[learn]'
```

## Graphics

The demo, arm demo, training and evaluation run headlessly by default. Rendering
or opening a native viewer requires a working MuJoCo graphics backend and a
desktop session. Use the packaged viewer command on macOS so it can select
`mjpython`; run it from a normal desktop terminal if a restricted launcher
cannot initialize AppKit.

Keep rendering out of throughput measurements. A saved GIF plays independently
of MuJoCo, and closing the native viewer stops its simulation process.

Docker and a separately obtained FastGripper URCap bundle are needed only for
[the optional URSim workflow](../ursim.md).
