"""Locate runtime assets in a checkout or an installed wheel."""
from pathlib import Path
import hashlib
import json


def asset_path(name: str = "scene.xml", *, model: str = "fastgripper") -> Path:
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Asset names must be relative paths within the model")
    if not model or Path(model).name != model or model in (".", ".."):
        raise ValueError("Invalid model name")
    package = Path(__file__).resolve().parent
    for root in (package / "assets", package.parent / "assets"):
        candidate = root / model / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing bundled asset: {model}/{name}")


def example_path(name: str, *, example: str = "grip_lift") -> Path:
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Example filenames must stay within the example")
    if not example or Path(example).name != example or example in (".", ".."):
        raise ValueError("Invalid example name")
    package = Path(__file__).resolve().parent
    for root in (package / "examples", package.parent / "examples"):
        candidate = root / example / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing bundled example: {example}/{name}")


def verify_assets(model: str = "fastgripper") -> dict:
    """Check the manifest's payload hashes and reject undeclared files/links."""
    directory = asset_path("manifest.json", model=model).parent
    manifest = json.loads((directory / "manifest.json").read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Asset manifest has no files")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Asset directory contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Asset file list does not match its manifest")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset manifest contains an escaping path")
        path = directory / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset hash mismatch: {name}")
    return manifest


if __name__ == "__main__":
    for name in ("fastgripper", "ur5e", "ur3e"):
        manifest = verify_assets(name)
        print(f"{name}: verified {len(manifest['files'])} payload files")
