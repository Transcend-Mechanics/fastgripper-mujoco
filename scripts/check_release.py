"""Check the maintainer's recorded license decision and immutable release tag.

This is a publication check, not a requirement for local use, tests or builds.
Changed or unapproved license documents cannot silently inherit approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib


LICENSE_FILES = {
    "LICENSE", "LICENSE-CODE", "LICENSE-ASSETS",
    "assets/fastgripper/LICENSE", "assets/fastgripper_v1_1/LICENSE", "docs/licensing.md",
}


def validate_readiness(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    record = json.loads((root / "release-readiness.json").read_text())
    if (record.get("schema_version") != 1 or record.get("status") != "approved"
            or record.get("license_choices_confirmed") is not True):
        raise ValueError("Publication is pending the owner's confirmation of the license terms")
    if record.get("package_version") != version:
        raise ValueError("Release decision does not match the package version")
    hashes = record.get("license_files_sha256", {})
    if set(hashes) != LICENSE_FILES:
        raise ValueError("Release decision must cover every license document")
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"License document changed since confirmation: {name}")
    return version


def validate_release(root: Path, tag: str) -> str:
    version = validate_readiness(root)
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?", tag):
        raise ValueError("Release tag must have the form v<package version>")
    if tag != f"v{version}":
        raise ValueError(f"Expected release tag v{version}")
    def resolve(ref):
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", ref], cwd=root, text=True).strip()
    if resolve(f"refs/tags/{tag}^{{commit}}") != resolve("HEAD"):
        raise ValueError("Release tag does not point to the tested checkout")
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    try:
        version = validate_release(Path(__file__).resolve().parents[1], args.tag)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Release v{version}: owner-confirmed license documents and tag verified")


if __name__ == "__main__":
    main()
