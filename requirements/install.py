#!/usr/bin/env python3
"""Create flex_env and install FLEX with its dependencies."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV_NAME = "flex_env"


def run(command, dry_run=False):
    """Print and optionally run a command."""
    print("[install] $", " ".join(shlex.quote(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def conda_path():
    """Return a working Conda executable."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError(
            "Anaconda/Miniconda was not found. Install it and expose 'conda' on PATH."
        )
    subprocess.run([conda, "--version"], check=True)
    return conda


def environment_exists(conda):
    output = subprocess.check_output(
        [conda, "env", "list", "--json"], text=True
    )
    return any(Path(path).name == ENV_NAME for path in json.loads(output)["envs"])


def cuda_version():
    """Read the installed CUDA version without requiring PyTorch."""
    for command in (["nvidia-smi"], ["nvcc", "--version"]):
        if not shutil.which(command[0]):
            continue
        try:
            output = subprocess.check_output(
                command, stderr=subprocess.STDOUT, text=True
            )
        except subprocess.CalledProcessError:
            continue
        match = re.search(
            r"(?:CUDA Version:|release)\s*(\d+)\.(\d+)", output, re.IGNORECASE
        )
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def select_profile(requested):
    if requested:
        return requested
    if platform.system() == "Darwin":
        return "cpu"
    version = cuda_version()
    if not version:
        return "cpu"
    return "cuda124" if version >= (12, 4) else "cuda121"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create flex_env and install FLEX dependencies."
    )
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--cpu", dest="profile", action="store_const", const="cpu")
    profile.add_argument(
        "--cuda121", dest="profile", action="store_const", const="cuda121"
    )
    profile.add_argument(
        "--cuda124", dest="profile", action="store_const", const="cuda124"
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        conda = conda_path()
        profile = select_profile(args.profile)
        print("[install] Environment:", ENV_NAME)
        print("[install] Profile:", profile)

        if not environment_exists(conda):
            run(
                [conda, "create", "--yes", "--name", ENV_NAME, "python=3.10", "pip"],
                args.dry_run,
            )

        conda_run = [conda, "run", "--no-capture-output", "--name", ENV_NAME]
        run(
            conda_run + ["python", "-m", "pip", "install", "--upgrade", "pip"],
            args.dry_run,
        )

        requirement_files = [HERE / f"{profile}.txt", HERE / "base.txt"]
        if args.dev:
            requirement_files.append(HERE / "dev.txt")
        for requirement_file in requirement_files:
            run(
                conda_run
                + ["python", "-m", "pip", "install", "-r", str(requirement_file)],
                args.dry_run,
            )

        run(
            conda_run + ["python", "-m", "pip", "install", "-e", str(ROOT)],
            args.dry_run,
        )
        run(
            conda_run
            + [
                "python",
                "-c",
                "import flex, torch, transformers; "
                "print(f'FLEX ready: torch={torch.__version__}')",
            ],
            args.dry_run,
        )
    except (
        OSError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"[install] ERROR: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[install] Dry run complete.")
    else:
        print(f"[install] Done. Run: conda activate {ENV_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
