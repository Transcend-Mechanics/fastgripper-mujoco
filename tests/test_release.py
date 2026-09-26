"""A publication decision cannot silently cover changed license documents."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py")
check_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_release)


@pytest.fixture
def candidate(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fastgripper-sim"\nversion = "0.2.1"\n')
    hashes = {}
    for name in check_release.LICENSE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Confirmed test terms\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"schema_version": 1, "status": "approved", "license_choices_confirmed": True,
              "package_version": "0.2.1", "license_files_sha256": hashes}
    (tmp_path / "release-readiness.json").write_text(json.dumps(record))
    return tmp_path


def test_confirmed_terms_match(candidate):
    assert check_release.validate_readiness(candidate) == "0.2.1"


@pytest.mark.parametrize("change", ["draft", "decision", "version", "missing", "tamper"])
def test_unconfirmed_or_changed_terms_block_publication(candidate, change):
    path = candidate / "release-readiness.json"
    record = json.loads(path.read_text())
    if change == "draft":
        record["status"] = "draft"
    elif change == "decision":
        record["license_choices_confirmed"] = False
    elif change == "version":
        record["package_version"] = "0.1.0"
    elif change == "missing":
        record["license_files_sha256"].pop("LICENSE-CODE")
    else:
        (candidate / "LICENSE-ASSETS").write_text("Changed terms\n")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        check_release.validate_readiness(candidate)


def test_release_tag_must_match_version_and_tested_commit(candidate):
    def git(*args):
        subprocess.run(["git", *args], cwd=candidate, check=True, capture_output=True)
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "test")
    git("tag", "v0.2.1")
    assert check_release.validate_release(candidate, "v0.2.1") == "0.2.1"
    with pytest.raises(ValueError, match="Expected release tag"):
        check_release.validate_release(candidate, "v0.2.0")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "later")
    with pytest.raises(ValueError, match="tested checkout"):
        check_release.validate_release(candidate, "v0.2.1")
