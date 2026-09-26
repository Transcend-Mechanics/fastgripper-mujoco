"""Write SHA256SUMS for exactly one wheel and one source distribution."""
from pathlib import Path
import hashlib


directory = Path(__file__).resolve().parents[1] / "dist"
wheels = list(directory.glob("*.whl"))
sources = list(directory.glob("*.tar.gz"))
if len(wheels) != 1 or len(sources) != 1:
    raise SystemExit("Expected exactly one wheel and one source distribution; start with an empty dist/")
payloads = sorted(wheels + sources)
# uv creates this local build-directory marker; it is not a release payload.
unknown = {path.name for path in directory.iterdir()} - {path.name for path in payloads} - {"SHA256SUMS", ".gitignore"}
if unknown:
    raise SystemExit(f"Unexpected release files: {sorted(unknown)}")
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in payloads]
(directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
