"""Select a bundled hardware/fingertip configuration before creating a model.

The legacy runtime still defaults to hardware v1.0. This module adds explicit
selection without changing the controller or the bundled policy environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .assets import asset_path, verify_assets


_HARDWARE_ASSETS = {"v1.0": "fastgripper", "v1.1": "fastgripper_v1_1"}


def available_hardware() -> tuple[str, ...]:
    """Hardware revisions supported by this package's selection API."""
    return tuple(_HARDWARE_ASSETS)


def _catalog(hardware: str) -> tuple[str, dict, dict]:
    try:
        directory = _HARDWARE_ASSETS[hardware]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown hardware revision {hardware!r}; choose from "
            f"{', '.join(available_hardware())}"
        ) from None
    manifest = verify_assets(directory)
    recorded_hardware = manifest.get("hardware_revision", "v1.0")
    if recorded_hardware != hardware:
        raise ValueError("Asset manifest hardware revision does not match selection")
    entrypoints = manifest.get("entrypoints", {})
    catalog_file = entrypoints.get("fingertip_catalog")
    if catalog_file is None:
        # The frozen v1.0 asset predates a separate carrier/tip catalog.
        if hardware != "v1.0":
            raise ValueError("Selected hardware has no fingertip catalog")
        catalog = {"schema_version": 1, "hardware_revision": hardware,
                   "selected": "stock-metal", "choices": {
                       "stock-metal": {"label": "Stock bare metal fingers",
                                       "behavior": "rigid",
                                       "entrypoints": entrypoints}}}
    else:
        if not isinstance(catalog_file, str):
            raise ValueError("Fingertip catalog entrypoint must be a filename")
        catalog = json.loads(asset_path(catalog_file, model=directory).read_text())
        if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
            raise ValueError("Unsupported fingertip catalog schema")
        if catalog.get("hardware_revision") != hardware:
            raise ValueError("Fingertip catalog hardware revision does not match selection")
    choices = catalog.get("choices")
    if (not isinstance(choices, dict) or not choices
            or any(not isinstance(key, str) or not key or not isinstance(value, dict)
                   for key, value in choices.items())
            or catalog.get("selected") not in choices):
        raise ValueError("Fingertip catalog has invalid choices or selected tip")
    return directory, manifest, catalog


def available_tips(hardware: str = "v1.0") -> tuple[str, ...]:
    """Return tip IDs from the verified catalog (stock-metal for legacy v1.0)."""
    _, _, catalog = _catalog(hardware)
    return tuple(catalog["choices"])


def _selection(hardware: str, tip: str, scene: bool) -> tuple[Path, dict]:
    directory, manifest, catalog = _catalog(hardware)
    try:
        choice = catalog["choices"][tip]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown fingertip {tip!r} for hardware {hardware}; choose from "
            f"{', '.join(catalog['choices'])}"
        ) from None
    entrypoints = choice.get("entrypoints")
    if entrypoints is None and tip == catalog["selected"]:
        entrypoints = manifest.get("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ValueError(f"Fingertip {tip!r} has no bundled model entrypoints")
    filename = entrypoints.get("demonstration" if scene else "gripper")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Fingertip {tip!r} has no {'scene' if scene else 'gripper'} entrypoint")
    path = asset_path(filename, model=directory)
    digest = lambda item: hashlib.sha256(item.read_bytes()).hexdigest()
    metadata = {
        "hardware_revision": hardware,
        "tip": tip,
        "asset_version": manifest["asset_version"],
        "asset_directory": directory,
        "entrypoint": filename,
        "manifest_sha256": digest(asset_path("manifest.json", model=directory)),
        "model_xml_sha256": digest(path),
        "behavior": choice.get("behavior", "rigid"),
        "calibration": choice.get("calibration", "provisional"),
    }
    return path, metadata


def model_path(*, hardware: str = "v1.0", tip: str = "stock-metal",
               scene: bool = False) -> Path:
    """Return a verified model.xml, or the bundled fixture scene if scene=True.

    Selection loads a complete authored configuration. It is not a hot-swap
    operation on an existing MjModel and does not load arbitrary mesh paths.
    """
    return _selection(hardware, tip, scene)[0]


def model_info(*, hardware: str = "v1.0", tip: str = "stock-metal",
               scene: bool = False) -> dict:
    """Return version/hash provenance without embedding a machine-local path."""
    return _selection(hardware, tip, scene)[1]


def load_model(*, hardware: str = "v1.0", tip: str = "stock-metal",
               scene: bool = False):
    """Compile the selected bundled model with MuJoCo."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(model_path(
        hardware=hardware, tip=tip, scene=scene)))


def main(argv: list[str] | None = None) -> None:
    """List verified catalogs, optionally compiling every bundled configuration."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", help="list hardware and tip choices")
    modes.add_argument("--verify", action="store_true",
                       help="verify hashes and compile gripper/scene for every choice")
    args = parser.parse_args(argv)
    catalog = []
    try:
        for hardware in available_hardware():
            for tip in available_tips(hardware):
                info = model_info(hardware=hardware, tip=tip)
                if args.verify:
                    load_model(hardware=hardware, tip=tip)
                    load_model(hardware=hardware, tip=tip, scene=True)
                catalog.append(info)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Model verification failed: {error}\n")
    print(json.dumps({"status": "verified_and_compiled" if args.verify else "catalog_verified",
                      "models": catalog}, indent=2))


if __name__ == "__main__":
    main()
