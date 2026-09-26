"""The public package must compile using only its bundled assets."""
import hashlib
import json
from pathlib import Path

import mujoco
import pytest

from fastgripper_sim.assets import asset_path, example_path, verify_assets


@pytest.mark.parametrize("name", ["fastgripper", "fastgripper_v1_1", "ur5e", "ur3e"])
def test_asset_manifest_and_model_are_self_contained(name):
    manifest = verify_assets(name)
    assert manifest["files"]
    path = asset_path("model.xml" if name.startswith("fastgripper") else f"{name}.xml", model=name)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq > 0
    assert model.nu > 0
    if name.startswith("fastgripper"):
        assert (model.nq, model.nv, model.nu, model.nmocap) == (7, 7, 1, 0)
        assert manifest["geometry"]["source_meshes_copied"] is False


def test_scene_has_a_free_object():
    model = mujoco.MjModel.from_xml_path(str(asset_path()))
    assert model.joint("block_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("test_block").mocapid[0] == -1
    block = model.body("test_block").id
    for index, kind in enumerate(model.eq_type):
        if kind in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            for object_id in (model.eq_obj1id[index], model.eq_obj2id[index]):
                body = (model.site_bodyid[object_id]
                        if model.eq_objtype[index] == mujoco.mjtObj.mjOBJ_SITE else object_id)
                assert body != block


@pytest.mark.parametrize("name", ["../scene.xml", "/tmp/scene.xml"])
def test_asset_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        asset_path(name)
    with pytest.raises(ValueError):
        example_path(name)


def test_retained_policy_matches_recorded_weights_and_physics_code():
    import fastgripper_sim

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_hash = digest(example_path("policy.zip"))
    package = Path(fastgripper_sim.__file__).parent
    sources = {name: digest(package / name)
               for name in ("controller.py", "env.py", "fixture.py")}
    # The original reports retain their original asset-manifest hash. Packaging
    # and license-only changes are documented separately; physics stays fixed.
    reports = [json.loads(example_path(f"{name}.json").read_text())
               for name in ("training", "evaluation")]
    for report in reports:
        assert report["checkpoint_sha256"] == checkpoint_hash
        assert report["runtime_source_sha256"] == sources
        assert report["model_xml_sha256"] == digest(asset_path())
    assert reports[0]["environment"] == reports[1]["environment"]
