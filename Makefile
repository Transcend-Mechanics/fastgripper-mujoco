UV ?= uv
ARM ?= ur5e
URCAP ?=
URSIM_OUTPUT ?= outputs/ursim-$(ARM)-$(shell date -u +%Y%m%dT%H%M%SZ)
URSIM_ARGS ?=
RENDER_OUTPUT ?= outputs/ursim-$(ARM)-replay/demo
.PHONY: setup demo demo-v1.1 viewer render learn-setup arm train evaluate replay verify test test-learn build ursim ursim-render

setup:
	$(UV) sync --locked

demo:
	$(UV) run --locked python -m fastgripper_sim.demo

demo-v1.1:
	$(UV) run --locked python -m fastgripper_sim.variant_demo --hardware v1.1 --tip stock-metal

viewer:
	$(UV) run --locked python -m fastgripper_sim.viewer --demo

render:
	$(UV) run --locked python -m fastgripper_sim.demo --render

learn-setup:
	$(UV) sync --locked --extra learn

arm:
	$(UV) run --locked python -m fastgripper_sim.arm_demo --arm $(ARM)

ursim:
	$(if $(URCAP),,$(error Set URCAP to the built .urcap file from your FastGripper hardware integration checkout))
	$(UV) run --locked python -m fastgripper_sim.ursim_demo --arm $(ARM) --urcap "$(URCAP)" --output "$(URSIM_OUTPUT)" $(URSIM_ARGS)

ursim-render:
	$(if $(RECORDING),,$(error Set RECORDING to the completed session.json path))
	$(UV) run --locked python -m fastgripper_sim.ursim --render-recording "$(RECORDING)" --output "$(RENDER_OUTPUT)"

train:
	$(UV) run --locked --extra learn python -m fastgripper_sim.train --output outputs/grip_lift

evaluate:
	$(UV) run --locked --extra learn python -m fastgripper_sim.evaluate

replay:
	$(UV) run --locked --extra learn python -m fastgripper_sim.replay

verify:
	$(UV) run --locked python -m fastgripper_sim.assets
	$(UV) run --locked python -m fastgripper_sim.models --verify
	$(UV) run --locked python scripts/verify_release_provenance.py

test:
	$(UV) run --locked pytest -q tests --ignore=tests/test_learning.py

test-learn:
	$(UV) run --locked --extra learn pytest -q tests/test_learning.py

build:
	$(UV) build
	$(UV) run --locked python scripts/checksums.py
