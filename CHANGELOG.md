# Changelog

## 0.4.0 — restored mechanism and housing preview (unreleased)

- Rebuilds v1.1 linkage and fingertip silhouettes as closed, full-depth solids,
  with recognizable linkage colors.
- Removes bores, countersinks, internal cavities and sheet interfaces from the
  linkage and fingertip profiles.
- Restores the worm helix, both toothed wheels, bent gear housing and motor casing.
  Gear shaft bores and housing mounting holes are filled throughout their depth;
  functional wheel seats remain open. Source CAD and original meshes stay private.
- Gear visuals follow their existing joints. Forces still use the effective
  20:1 transmission with provisional backlash/friction, without tooth contacts.
- Advances v1.1 assets to 0.3.0; joints, inertia, contacts and collision proxies
  retain the previous physics. v1.0 and its retained policy are unchanged.

## 0.3.0 — initial public release

- Adds separately selectable hardware v1.1 assets and a stock-metal fingertip catalog.
- Separates each rigid screw-on tip from its carrier, including mass, inertia,
  attachment frame and contact geometry.
- Adds a motor-driven gravity/contact grasp demonstration and model selection API.
- Retains v1.0 as the default and preserves its existing policy and arm examples.
- Uses independently authored simplified exterior geometry; source CAD and
  detailed manufacturing meshes remain excluded.
- v1.1 parameters, software travel endpoints and mechanical fit remain provisional.
- Publishes MIT software and documentation with FastGripper Simulation and
  Accessory License version 1.0 for the separately licensed model assets.

## 0.2.1 — historical local candidate (never published)

- Standalone repository, documentation, public regression tests and release workflow.
- Retains the tested 0.2.0 runtime physics, controller, training environment,
  policy checkpoint and historical results.
- Asset metadata 0.1.1 records the proposed licensing/documentation change;
  all simulation XML and geometry are identical to asset 0.1.0.
- Records that licensing-only transition without rewriting historical
  training reports or claiming retraining.
- Hardware baseline: v1.0. This earlier candidate did not include v1.1.
- Licensing draft: MIT software and separately scoped simulation/accessory terms.

Version 0.3.0 is the first public release; 0.2.1 remained a local preparation
candidate. See release-readiness.json for the recorded licensing decision.
