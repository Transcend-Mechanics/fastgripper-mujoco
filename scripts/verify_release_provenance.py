#!/usr/bin/env python3
"""Check that public packaging preserved the retained policy's physics inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    directory = root / "examples/grip_lift"
    report = json.loads((directory / "public-release-compatibility.json").read_text())
    old = report["original_asset_manifest"]
    current_path = root / "assets/fastgripper/manifest.json"
    current = json.loads(current_path.read_text())
    old_raw = report["original_asset_manifest_text"].encode()
    if json.loads(old_raw) != old or hashlib.sha256(old_raw).hexdigest() != report["original_asset_manifest_sha256"]:
        raise ValueError("Original asset manifest witness is inconsistent")
    if sha(current_path) != report["public_asset_manifest_sha256"]:
        raise ValueError("Public asset manifest no longer matches the compatibility record")
    allowed = {"LICENSE", "README.md"}
    if set(old["files"]) != set(current["files"]):
        raise ValueError("Public packaging added or removed model payload files")
    for name, expected in old["files"].items():
        if name not in allowed:
            if current["files"][name] != expected or sha(root / "assets/fastgripper" / name) != expected:
                raise ValueError(f"Physics input changed: {name}")
    old_metadata = {k: v for k, v in old.items() if k not in {"asset_version", "files", "limitations", "license"}}
    new_metadata = {k: v for k, v in current.items() if k not in {"asset_version", "files", "limitations", "license"}}
    if old_metadata != new_metadata:
        raise ValueError("Model interface or geometry metadata changed")
    for name, expected in report["retained_artifact_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Retained learning artifact changed: {name}")
    for name in ("training", "evaluation"):
        artifact = json.loads((directory / f"{name}.json").read_text())
        if artifact["asset_manifest_sha256"] != report["original_asset_manifest_sha256"]:
            raise ValueError(f"{name} did not use the original asset manifest")
        if artifact["model_xml_sha256"] != sha(root / artifact["model_reference"]):
            raise ValueError(f"{name} model changed")
        if artifact["checkpoint_sha256"] != sha(directory / "policy.zip"):
            raise ValueError(f"{name} checkpoint changed")
        for filename, expected in artifact["runtime_source_sha256"].items():
            if sha(root / "fastgripper_sim" / filename) != expected:
                raise ValueError(f"{name} runtime physics source changed: {filename}")
    print("Verified retained v1.0 policy, reports, physics and control sources; v1.0 changes are licensing/documentation metadata only. No policy-transfer claim is made for v1.1.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().root)
