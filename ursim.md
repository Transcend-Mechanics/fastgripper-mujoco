# URSim, the FastGripper URCap, and MuJoCo

This integration runs a normal PolyScope program with the FastGripper URCap
against a simulated gripper and object. URSim generates the arm trajectory;
MuJoCo advances the arm, motor, linkages, and contact dynamics. Gripper commands
and status pass through the existing URCap daemon and register interface.

```text
PolyScope + FastGripper URCap
  ├─ arm target joints ── read-only RTDE ──► bounded MuJoCo arm actuators
  └─ gripper nodes ── XML-RPC daemon ◄────► MuJoCo gripper bridge
                                          motor, linkage, and free object
```

The gripper connection is bidirectional. Its position, torque, and contact
observer come from the simulated mechanism. The arm connection is one-way:
MuJoCo contact forces do not feed back into URSim's servo or safety controller.
This exercises the URCap software path and physical grasp together; it does
not qualify a physical installation or reproduce the UR controller's dynamics.

## Run a complete program

From this checkout, install Python 3.12, uv and Docker Desktop/Engine with
`host-gateway` support. Start Docker before running the example. Obtain a built
FastGripper URCap bundle that supports the explicit `mujoco:` transport; the
URCap and its Java build toolchain are separate from this simulation package.
Supply the bundle's path explicitly:

```sh
uv sync --locked

uv run --locked python -m fastgripper_sim.ursim_demo \
  --arm ur5e \
  --urcap /path/to/fastgripper.urcap \
  --output outputs/ursim-ur5e

uv run --locked python -m fastgripper_sim.ursim_demo \
  --arm ur3e \
  --urcap /path/to/fastgripper.urcap \
  --output outputs/ursim-ur3e
```

Use a new or empty output directory for each run. No sibling checkout is needed.
The bundled Docker image is pinned by digest to
URSim 5.25.2 and runs as `linux/amd64`, including on Apple Silicon through
Docker's emulation. A first run may need to download the image; booting
PolyScope takes longer than starting a direct MuJoCo example. The runner maps
`host.docker.internal` to Docker's host gateway for the gripper connection.

The runner creates an isolated container, checks its simulation mode, moves
the simulated arm to its initial pose, starts the physics bridge, and activates
the simulated gripper. It then loads a generated `.urp` containing ordinary
arm moves and actual contributed Gripper and Grip Check nodes:

1. Open, approach the block, and close.
2. Check contact, lift 100 mm, hold, and check again.
3. Lower, open, and retract.

The free block moves through gravity and contact. Live arm joint positions are
initialized once before activation; subsequent motion uses bounded actuators.
The success check requires sustained bilateral contact with more than 80 mm
of object rise, then return to the table and release. Unexpected contact pairs,
lost RTDE references, clock overruns, and MuJoCo warnings fail acceptance.

The runner writes `session.json` with acceptance results, physics samples,
contact audit, bridge metadata, daemon service history, and program/URCap/image
provenance. A failed run writes diagnostics when the physics bridge has started.
The generated program and per-phase scripts are retained under `container/programs/`.

Add `--control-checks` to exercise interruptions after the physical task passes.
A real URCap Close node closes empty jaws at the raised pose, pauses during
motion, and resumes to completion. Further checks stop a moving gripper, then
disconnect the physics service's RTDE stream, probe its motion gate, and
reconnect. Empty closure must report arrival at the endpoint without object
detection; reconnection must leave the interrupted command stopped. Results go
into `control-checks.json`, with a summary in `session.json`. These deliberate
interruptions are recorded separately from the normal grasp acceptance.

## View the pendant and the physics

While a run is active, open the matching pendant in a browser:

| Arm | Pendant | Dashboard | Secondary | RTDE | Physics bridge |
| --- | --- | --- | --- | --- | --- |
| UR5e | <http://localhost:6280/vnc.html> | 32999 | 33002 | 33004 | 55431 |
| UR3e | <http://localhost:6380/vnc.html> | 33999 | 34002 | 34004 | 55432 |

The pendant shows the PolyScope program and URCap. To view the gripper's contact
motion, render the recorded MuJoCo states after execution:

```sh
uv run --locked python -m fastgripper_sim.ursim \
  --render-recording outputs/ursim-ur5e/session.json \
  --output outputs/ursim-ur5e/demo
```

This writes `demo.gif` and a raised-hold `demo.png`. Rendering needs an available
MuJoCo graphics backend. The live acceptance run is headless; rendering its
recording afterward keeps graphics work out of the control loop. `--render`
on `ursim_demo` performs that same rendering after the program finishes.

By default the runner removes its container on exit. `--keep-container` leaves
it available for pendant inspection, but the Python physics server still stops.
The runner refuses to reuse an existing `fg-mujoco-ur5e` or `fg-mujoco-ur3e`
container. After inspection, remove the corresponding example container before
starting another run:

```sh
docker rm -f fg-mujoco-ur5e
```

The equivalent Makefile shortcut creates a dated output directory:

```sh
make ursim ARM=ur5e URCAP=/path/to/fastgripper.urcap
make ursim ARM=ur3e URCAP=/path/to/fastgripper.urcap URSIM_ARGS=--keep-container
make ursim ARM=ur5e URCAP=/path/to/fastgripper.urcap URSIM_ARGS=--control-checks
make ursim-render RECORDING=outputs/ursim-ur5e/session.json
```

Set `URCAP` to the bundle path and `URSIM_OUTPUT` to select an explicit new
output directory. `ursim-render` defaults to
`outputs/ursim-<arm>-replay/demo.*`; set `ARM` or `RENDER_OUTPUT` as needed.

## Connect an existing simulator

For a separately configured URSim instance, start only the physics service:

```sh
uv run --locked python -m fastgripper_sim.ursim \
  --arm ur5e --rtde-host 127.0.0.1 --rtde-port 33004 \
  --listen 0.0.0.0 --port 55431 --output outputs/custom-ursim/session.json
```

The arm must start at a suitable pose for the packaged scene. This command
waits for fresh RTDE data, initializes the MuJoCo arm to `actual_q`, then tracks
`target_q` with joint position/rate bounds and actuator torque limits. It does
not position URSim, load a program, or activate the gripper.
The stream uses the timestamp and joint fields documented in
[Universal Robots' RTDE guide](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html).

In the URCap's Installation transport field, use
`mujoco:host.docker.internal:55431` for URSim in Docker Desktop, or substitute
the physics host reachable from your simulator. Use slave ID 9. The daemon
must report `simulated: true` and `simulation_backend: "mujoco"`. `sim:` selects
the separate deterministic register simulator and does not connect to physics.

The transport carries raw Modbus RTU frames, including CRC, over a TCP stream;
it is not standard Modbus TCP with an MBAP header. The physics listener binds
to loopback by default. `--listen 0.0.0.0`, as used by the Docker runner,
permits the container to reach the host bridge and exposes that port on the
host's interfaces. URSim's published ports are restricted to host loopback.

Physics follows the UR controller timestamp: one simulated second per controller
second, even when Docker emulation runs slower than wall time. During an RTDE
outage, wall-time stepping keeps gravity and contacts active while the arm holds
its current simulated position and the gripper command is stopped. Reconnection
rebases the clock without resetting object or joint state. A simulation clock
overrun latches a stop for the rest of the run. The report records these events;
a completed motion alone is insufficient for a passing run.

Program pause and stop retain the URCap daemon's existing behavior: its program
watcher stops a moving gripper, and a resumed waiting node can reissue the
interrupted move. The physics bridge separately handles stale RTDE references
and clock faults; reconnecting its RTDE stream does not replay old gripper
commands. The report records PolyScope runtime-state transitions. This models
application behavior, not a hardware safety stop.

## Model and interface boundaries

The physics bridge advertises firmware protocol **1.0** and implements its
ordinary command/status registers, activation, motion, stop, disable,
auto-release, and a communication watchdog. The watchdog uses simulation time.
Activation accepts the model's nominal open/closed calibration; it does not
perform physical homing. Temperature is unavailable and reads zero. Firmware
1.1 re-grasp, endpoint calibration writes, persistence, and reboot are unsupported.
Operational speed/gain/torque-cap changes are ephemeral.

Contact status uses the motor-stall observer rather than exposing MuJoCo contact
forces as a hardware sensor. Separate telemetry records actual simulated finger
normal forces for assessing the test. Requested FORCE remains a motor torque
limit, not fingertip force in newtons. A latched contact result is not a payload
sensor, and the bridge's protocol-1.0 behavior does not detect every later slip.

The integration scene uses the nominal UR controller base and DH geometry for
each arm. It applies its frame corrections in memory, leaving the packaged
arm assets and the earlier direct-MuJoCo examples unchanged. Individual robot
calibration, the pending mechanical adapter, fingertip geometry, mass, friction,
and motor/transmission parameters still need measurement. The current mount
is the provisional spacer described in [the arm guide](arm.md).

Use this path to rehearse PolyScope programs and the URCap. For reinforcement
learning, [the direct Python environment](learning.md) avoids Docker and network
timing and provides explicit reset, step, observation, and reward functions.
Passing this example does not establish policy transfer to hardware.
