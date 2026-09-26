"""Run a selected hardware/fingertip configuration with the existing grasp demo.

python -m fastgripper_sim.variant_demo --hardware v1.1 --tip stock-metal
python -m fastgripper_sim.variant_demo --hardware v1.1 --render
python -m fastgripper_sim.variant_demo --hardware v1.1 --viewer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .demo import run_demo
from .fixture import percent
from .models import available_hardware, model_info, model_path


def run_variant_demo(*, hardware: str = "v1.0", tip: str = "stock-metal",
                     output: Path | None = None, render: bool = False,
                     force: float = 30.0, fps: int = 20) -> dict:
    """Run contact physics and attach hardware, tip and asset provenance."""
    if render and output is None:
        raise ValueError("Rendering requires an output prefix")
    output = Path(output) if output is not None else None
    path = model_path(hardware=hardware, tip=tip, scene=True)
    provenance = model_info(hardware=hardware, tip=tip, scene=True)
    if render:
        # The baseline writer uses Path.with_suffix(). Render with a plain
        # temporary prefix, then append extensions to the full caller prefix
        # so hardware IDs such as v1.0 and v1.1 cannot overwrite one another.
        with tempfile.TemporaryDirectory(prefix="fastgripper-render-") as temporary:
            prefix = Path(temporary) / "preview"
            result = run_demo(output=prefix, render=True, model_path=path,
                              force=force, fps=fps)
            output.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".gif", ".png"):
                shutil.copyfile(prefix.with_suffix(suffix), Path(str(output) + suffix))
    else:
        result = run_demo(output=None, render=False, model_path=path,
                          force=force, fps=fps)
    result["hardware_revision"] = hardware
    result["tip"] = tip
    result["model_selection"] = provenance
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(output) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", default="v1.0", choices=available_hardware())
    parser.add_argument("--tip", default="stock-metal")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--render", action="store_true", help="also save a GIF and PNG")
    modes.add_argument("--viewer", action="store_true", help="open the repeating native demo")
    parser.add_argument("--output", type=Path, help="output prefix; default: outputs/grasp-HARDWARE-TIP")
    parser.add_argument("--force", type=percent, default=30.0)
    args = parser.parse_args(argv)
    try:
        path = model_path(hardware=args.hardware, tip=args.tip, scene=True)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.viewer:
        print(f"Opening hardware {args.hardware}, tip {args.tip}; provisional simulation.", flush=True)
        completed = subprocess.run([
            sys.executable, "-m", "fastgripper_sim.viewer", "--model", str(path),
            "--demo", "--force", str(args.force),
        ], check=False)
        raise SystemExit(completed.returncode)
    output = args.output or Path(f"outputs/grasp-{args.hardware}-{args.tip}")
    result = run_variant_demo(hardware=args.hardware, tip=args.tip, output=output,
                              render=args.render, force=args.force)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    if not result["success"]:
        raise SystemExit("Grasp/lift/release validation failed; see the telemetry")


if __name__ == "__main__":
    main()
