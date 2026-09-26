# Universal Robots UR3e — converted official description

This is a UR3e model built from the official [Universal Robots ROS2 description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/89bbe795f38a7ab00fb66fe8831dfff79dc99edf), pinned to `89bbe795f38a7ab00fb66fe8831dfff79dc99edf`. It is not a scaled UR5e. Original BSD-3-Clause terms are in LICENSE; the upstream README records its licensing scope. The conversion introduces no additional restriction on the upstream assets.

The four UR3e configuration files supply all six joint origins, local axes,
link masses, centers of mass, full rotated inertia tensors, position/effort
limits and visual/collision offsets. Their parsed content is retained in
`source_parameters.json`; upstream bytes and generated files are hashed in
`manifest.json`. COLLADA scene transforms, normals and diffuse colors are
preserved in material-separated OBJ surface files; non-surface wire lines are
omitted. Official collision STL files are
unchanged. MuJoCo uses their convex hulls, which may close nonconvex recesses.
One explicit base/shoulder collision exclusion removes the stock adjacent
hulls' approximately 0.18 mm overlap at their invariant pivot interface.
The static base is welded to world, so ordinary parent filtering does not
exclude that pair. No nonadjacent arm collision pair is disabled.

`ur3e.xml` is composable; `scene.xml` supplies a floor and lighting. Both load
without ROS or a converter. The `home` keyframe is a convenient authored pose.
Joint and actuator names match the UR5e example's interface. Kinematics follow
the official URDF frame convention, including the base rotation of pi about Z;
the local joint axes are all Z. `attachment_site` aliases official `tool0` at
the wrist_3_link origin; +Z points out of the tool. The separately named
`flange` site retains the different ROS-Industrial flange orientation.

The UR3e upper-arm/forearm lengths are 0.24355/0.2132 m. Published description
effort caps are 54/54/28/9/9/9 N m. Wrist 3 is continuous: no physical joint
stop is added; the demo uses an authored +/-2 pi target/planning window.
The other position limits follow the official configuration, including its
deliberate +/-pi elbow planning restriction. Velocity limits are recorded in
the custom numeric `ur3e_max_joint_velocity_rad_s`; they are not hard solver
constraints. A trajectory generator must respect them.

The native position-servo gains (2000/400 for the first three axes, 500/100
for wrists) are authored simulation settings, not a UR controller emulator.
Joint damping/friction follow the official zero defaults; no unmeasured rotor
inertia is added. The upstream description flags the base mass as potentially
incorrect. Generic kinematics are not per-robot factory calibration.

`validation.json` records forward-kinematic comparison against independently
composed official frame transforms at 25 joint configurations, full-inertia
reconstruction, a short live dynamics check, and independent visual/STL bound
agreement. No proprietary gripper geometry is included in this arm package.
