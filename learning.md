# Learn to grip, lift, and hold

This is a small continuous reinforcement-learning example using the exported
FastGripper MuJoCo model. A SAC policy controls the gripper while an external
fixture performs a prescribed 50 mm vertical lift. It is a gripper task, not
arm motion planning: no UR robot, camera perception, reach, or pick-and-place
policy is involved.

The bundled `examples/grip_lift/policy.zip` is a real trained Stable-Baselines3
SAC checkpoint. `training.json` records the seed, hyperparameters, package
versions, task configuration, public asset-manifest/checkpoint hashes, and
training time. The reports also hash the public environment/controller/fixture
source used by the run. `training.csv` and
`learning_curve.png` show stochastic training episodes. `evaluation.json`
compares deterministic policy execution with a strong scripted grasp and two
failure controls on separate, matched seeds. These files contain simulator
results under provisional physical parameters; they are not hardware ratings.

## Run the bundled policy

From this runtime repository:

```sh
uv sync --locked --extra learn
uv run --locked --extra learn python -m fastgripper_sim.evaluate --episodes 16
uv run --locked --extra learn python -m fastgripper_sim.replay --seed 10000
```

The default checkpoint is resolved from the checkout or installed package.
The new report goes into `outputs/grip_lift/evaluation.json`; the bundled
reference report stays unchanged. Evaluation uses seeds 10000 through 10015
by default. The same seeds, masses, and friction coefficients are used for
every controller.
Replay writes a GIF, final PNG, and per-step JSON to `outputs/grip_lift/rollout.*`.
Its fixture trajectory is prescribed; the learned policy supplies the gripper
commands visible in that rollout.

## Train another policy

```sh
uv run --locked --extra learn python -m fastgripper_sim.train \
  --steps 6000 --seed 7 --output outputs/my_run
uv run --locked --extra learn python -m fastgripper_sim.evaluate \
  --checkpoint outputs/my_run/policy.zip \
  --output outputs/my_run/evaluation.json
```

Training requires a new or empty output directory. SAC runs on the CPU with
one Torch thread, a two-layer 64-unit MLP, a replay buffer, and actual gradient
updates. It does not use scripted demonstrations, preloaded weights, or
rewards substituted for physical outcomes. A few thousand steps take minutes
on the development machine; the precise elapsed time is in `training.json`.

## Task contract

Each episode lasts 2.2 simulated seconds: 0.8 seconds to grasp, a 0.6-second
smooth fixture lift, and 0.8 seconds at the raised position. Reset initializes
the mechanism in a consistent, approximately 49 mm pregrasp opening around a
40 mm-wide block resting on the table. Only reset assigns joint state.
During the episode MuJoCo advances the motor, linkage, contacts, and free
block; the task moves only the external fixture's mocap target.
The task phase is observed, and completion terminates this finite-horizon
episode. It is not an external timeout that allows value bootstrapping past
the final success reward.

The policy acts at 20 Hz through two normalized actions:

| Action | -1 | +1 |
|---|---|---|
| Target | Hold the pregrasp angle | Request the closed endpoint |
| FORCE | 5% of the provisional motor torque cap | Request 45% (44% after quantization) |

Targets are quantized into 2% increments of the pregrasp-to-closed interval;
FORCE uses two percentage point increments, then is bounded to the requested
range. SPEED stays at 100%. An
unchanged command stays active without resetting firmware stall detection.
A changed command genuinely retargets the controller and restarts seeking,
matching the bridge's command semantics. The policy never directly supplies
joint positions, contact forces, or a block trajectory.

The existing controller retains its 100 Hz position-to-velocity loop,
feedback-relative velocity clamp, and stall handling. Physics retains the
validated **50 microsecond timestep**: each 1 ms controller update batches
20 native MuJoCo steps. Neither training nor evaluation coarsens that timestep.

Block mass is sampled uniformly from 30–70 g and the finger-contact sliding
friction coefficient from 0.25–0.45 at each reset. The block's physical inertia is recalculated for its
mass. Shape, width, alignment, and the fixture path stay fixed. Evaluation
samples different seeds within these same ranges; it does not establish
performance on unfamiliar object shapes or outside these ranges.

Success is assessed at 20 Hz and requires the block to remain at least 45 mm above its initial height
for at least 0.5 seconds during the raised hold, with bilateral finger
contact and vertical speed below 25 mm/s. A brief toss does not meet that
condition. Reward favors lift and contact, adds a terminal sustained-hold
bonus, and modestly penalizes both the requested FORCE cap and the measured
mean squared actuator torque. Actual squared motor torque integrated over
time is also reported separately, using 1 ms samples. This is an actuator-effort
proxy in N²m²s, not electrical energy or mechanical work in joules. The model's finite
passive worm friction can retain a grasp after the controller stops seeking;
a low holding motor torque is not a measurement of low contact force.

The observation vector is intentionally privileged simulator state: task
phase, aperture, motor angle/velocity/torque, block height/vertical velocity,
fixture lift, individual contact normal forces, sampled mass/friction,
held command, and seeking state. Its named order is recorded in the report
and exported as `fastgripper_sim.env.OBSERVATIONS`. A real gripper does not
provide all of these measurements. A deployable policy would need a revised
observation contract and calibration, along with hardware validation.

## Interpreting the result

The retained seed-7 run trained for 6,000 policy steps (136 completed episodes,
5,800 optimizer updates) in approximately four minutes. Evaluation on seeds
10000–10015 produced:

| Controller | Successful holds | Mean requested FORCE | Mean actuator effort (N²m²s) | Mean return |
|---|---:|---:|---:|---:|
| Learned SAC | 16/16 | 19.28% | 0.13939 | 58.45 |
| Scripted grasp | 16/16 | 30% | 0.04584 | 58.55 |
| Leave jaws open | 0/16 | 30% | Approximately zero | -1.56 |
| Insufficient torque | 0/16 | 5% | 0.00095 | -0.05 |

Both successful controllers raised the block approximately 49.95 mm. The
learned controller requested a lower cap but consumed more actuator effort;
it **did not outperform the scripted baseline** on effort or reward. This is
a functioning learning example, not an efficiency improvement over the
handwritten controller. Exact per-seed measurements are in `evaluation.json`.

The scripted 30% FORCE controller is deliberately a strong baseline for this
small, aligned task. The learning example demonstrates environment stepping,
reward construction, real policy optimization, saved weights, and evaluation;
it need not outperform that baseline to be useful. The open-jaw and 5% FORCE
controls check that success depends on actually grasping the free block.
Compare held-out success and commanded/actual effort together. Training
curves include exploration noise and should not be read as deterministic
policy evaluation or evidence of transfer to hardware.
