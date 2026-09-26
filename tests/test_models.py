"""Version selection must preserve the baseline and identify every demo run."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from fastgripper_sim.assets import asset_path
from fastgripper_sim import models, variant_demo


def test_default_selection_preserves_v10():
    assert models.model_path() == asset_path("model.xml")
    assert models.model_path(scene=True) == asset_path("scene.xml")
    assert models.model_info()["hardware_revision"] == "v1.0"


def test_tip_choices_come_from_v11_catalog():
    catalog = json.loads(asset_path("fingertips.json", model="fastgripper_v1_1").read_text())
    assert set(models.available_tips("v1.1")) == set(catalog["choices"])
    assert models.available_tips() == ("stock-metal",)
    assert models.available_tips("v1.1") == ("stock-metal",)


@pytest.mark.parametrize("hardware,directory", [
    ("v1.0", "fastgripper"), ("v1.1", "fastgripper_v1_1"),
])
@pytest.mark.parametrize("scene", [False, True])
def test_selected_models_compile(hardware, directory, scene):
    filename = "scene.xml" if scene else "model.xml"
    assert models.model_path(hardware=hardware, scene=scene) == asset_path(filename, model=directory)
    model = models.load_model(hardware=hardware, scene=scene)
    assert model.nu == 1
    assert model.nq == (21 if scene else 7)
    assert models.model_info(hardware=hardware, scene=scene)["entrypoint"] == filename
    if hardware == "v1.1":
        for side in ("positive", "negative"):
            tip = model.body(f"fingertip_{side}")
            assert tip.parentid[0] == model.body(f"finger_{side}_x").id
            assert tip.jntnum[0] == 0
            assert tip.mass[0] > 0
    if scene:
        assert model.joint("block_free").type[0] == mujoco.mjtJoint.mjJNT_FREE


@pytest.mark.parametrize("hardware,tip,match", [
    ("v9.9", "stock-metal", "Unknown hardware"),
    ("../fastgripper", "stock-metal", "Unknown hardware"),
    ("v1.1", "finray", "Unknown fingertip"),
    ("v1.0", "../stock-metal", "Unknown fingertip"),
])
def test_unknown_selection_is_explicit(hardware, tip, match):
    with pytest.raises(ValueError, match=match):
        models.model_path(hardware=hardware, tip=tip)


def test_catalog_choice_cannot_silently_load_the_stock_model(monkeypatch):
    directory, manifest, catalog = models._catalog("v1.1")
    catalog["choices"]["unbuilt-tip"] = {"behavior": "rigid"}
    monkeypatch.setattr(models, "_catalog", lambda _: (directory, manifest, catalog))
    assert "unbuilt-tip" in models.available_tips("v1.1")
    with pytest.raises(ValueError, match="no bundled model entrypoints"):
        models.model_path(hardware="v1.1", tip="unbuilt-tip")


def test_catalog_entrypoint_cannot_escape_assets(monkeypatch):
    directory, manifest, catalog = models._catalog("v1.1")
    catalog["choices"]["stock-metal"]["entrypoints"] = {"gripper": "../../private.xml"}
    monkeypatch.setattr(models, "_catalog", lambda _: (directory, manifest, catalog))
    with pytest.raises(ValueError, match="relative paths"):
        models.model_path(hardware="v1.1")


def test_variant_report_records_the_actual_model_and_tip(tmp_path, monkeypatch):
    calls = []

    def fake_demo(**kwargs):
        calls.append(kwargs)
        return {"success": True, "samples": []}

    monkeypatch.setattr(variant_demo, "run_demo", fake_demo)
    output = tmp_path / "trial"
    result = variant_demo.run_variant_demo(hardware="v1.1", output=output)
    assert calls[0]["model_path"] == models.model_path(hardware="v1.1", scene=True)
    assert calls[0]["output"] is None
    assert result["hardware_revision"] == "v1.1"
    assert result["tip"] == "stock-metal"
    info = result["model_selection"]
    assert info["asset_directory"] == "fastgripper_v1_1"
    assert info["asset_version"] == "0.1.0"
    assert info["entrypoint"] == "scene.xml"
    assert len(info["manifest_sha256"]) == len(info["model_xml_sha256"]) == 64
    assert json.loads(output.with_suffix(".json").read_text()) == result
    assert str(Path.cwd()) not in json.dumps(result)


@pytest.mark.parametrize("render", [False, True])
def test_dotted_hardware_output_prefixes_stay_distinct(tmp_path, monkeypatch, render):
    temporary_prefixes = []

    def fake_demo(**kwargs):
        prefix = kwargs["output"]
        marker = kwargs["model_path"].parent.name.encode()
        if kwargs["render"]:
            assert prefix is not None and "." not in prefix.name
            temporary_prefixes.append(prefix)
            # Match the unchanged baseline writer's suffix-replacement behavior.
            for suffix in (".gif", ".png"):
                prefix.with_suffix(suffix).write_bytes(marker)
        else:
            assert prefix is None
        return {"success": True, "samples": []}

    monkeypatch.setattr(variant_demo, "run_demo", fake_demo)
    for hardware, directory in (("v1.0", "fastgripper"), ("v1.1", "fastgripper_v1_1")):
        prefix = tmp_path / "new-directory" / f"grasp-{hardware}-stock-metal"
        variant_demo.run_variant_demo(hardware=hardware, output=prefix, render=render)
        report = json.loads(Path(str(prefix) + ".json").read_text())
        assert report["hardware_revision"] == hardware
        assert report["model_selection"]["asset_directory"] == directory
        if render:
            for suffix in (".gif", ".png"):
                assert Path(str(prefix) + suffix).read_bytes() == directory.encode()
    assert not (tmp_path / "new-directory/grasp-v1.json").exists()
    assert len(list((tmp_path / "new-directory").iterdir())) == (6 if render else 2)
    assert all(not prefix.parent.exists() for prefix in temporary_prefixes)


def test_render_requires_an_output_prefix():
    with pytest.raises(ValueError, match="output prefix"):
        variant_demo.run_variant_demo(render=True)


def test_viewer_delegates_selected_scene_without_changing_baseline(monkeypatch):
    calls = []
    monkeypatch.setattr(variant_demo.subprocess, "run",
                        lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0)))
    with pytest.raises(SystemExit) as exited:
        variant_demo.main(["--hardware", "v1.1", "--tip", "stock-metal", "--viewer"])
    assert exited.value.code == 0
    command = calls[0]
    assert command[1:3] == ["-m", "fastgripper_sim.viewer"]
    assert command[command.index("--model") + 1] == str(models.model_path(hardware="v1.1", scene=True))
    assert "--demo" in command


def test_cli_verifies_all_models(capsys):
    models.main(["--verify"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified_and_compiled"
    assert {(row["hardware_revision"], row["tip"]) for row in result["models"]} == {
        ("v1.0", "stock-metal"), ("v1.1", "stock-metal")}
