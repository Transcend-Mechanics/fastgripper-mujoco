"""Run with an installed wheel, outside the source checkout (python -I)."""
from pathlib import Path
import sys

import mujoco

import fastgripper_sim
from fastgripper_sim.assets import asset_path, example_path, verify_assets
from fastgripper_sim.arm_demo import make_arm_model
from fastgripper_sim.demo import run_demo


package = Path(fastgripper_sim.__file__).resolve()
if not package.is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit(f"Expected an installed wheel inside the test environment, got {package}")
for name in ("fastgripper", "fastgripper_v1_1", "ur5e", "ur3e"):
    verify_assets(name)
    if not asset_path("manifest.json", model=name).is_relative_to(package.parent):
        raise SystemExit("Assets were resolved outside the installed wheel")
for arm in ("ur5e", "ur3e"):
    model = make_arm_model(arm=arm)
    assert model.nu == 7 and model.nmocap == 0
assert mujoco.MjModel.from_xml_path(str(asset_path("model.xml"))).nu == 1
assert example_path("policy.zip").is_relative_to(package.parent)
result = run_demo(fps=10)
if not result["success"] or result["warning_count"]:
    raise SystemExit("Installed-wheel grasp/lift/release failed")
from fastgripper_sim.models import model_path

v11 = model_path(hardware="v1.1", tip="stock-metal", scene=True)
if not v11.is_relative_to(package.parent):
    raise SystemExit("v1.1 resolved outside the installed wheel")
result = run_demo(model_path=v11, fps=10)
if not result["success"] or result["warning_count"]:
    raise SystemExit("Installed-wheel v1.1 grasp/lift/release failed")
print(f"Installed fastgripper-sim {fastgripper_sim.__version__}: both gripper revisions, arms and physical grasps passed")
