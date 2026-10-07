#!/usr/bin/env python3
"""Prepare released v2 with the matched benchmark-only API adaptation."""

import argparse
import io
from pathlib import Path
import subprocess
import tarfile


BASELINE = "c20465ba22d078e0fb115fb6a99fa77027992c90"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    destination = arguments.output.resolve()
    if destination.exists():
        parser.error("output must be a new directory")
    patch = root / "benches" / "v2.patch"
    patch_text = patch.read_text()
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        path = line[4:].split("\t", 1)[0]
        if path == "/dev/null":
            continue
        path = path.removeprefix("a/").removeprefix("b/")
        parts = Path(path).parts
        if (
            Path(path).is_absolute()
            or ".." in parts
            or not (
                path.startswith("benches/")
                or path.startswith("external/")
                or path in {"Cargo.toml", "Cargo.lock", "crate.just", "justfile"}
            )
        ):
            raise ValueError("baseline patch touches a non-benchmark path")
    archive = subprocess.check_output(["git", "archive", BASELINE], cwd=root)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")
    subprocess.run(
        ["git", "apply", "--unsafe-paths", f"--directory={destination}", str(patch)],
        cwd="/tmp",
        check=True,
    )
    for name in ("crate.just", "justfile"):
        (destination / name).write_bytes((root / name).read_bytes())
    print(f"BASELINE\trevision={BASELINE}\toutput={destination}")


if __name__ == "__main__":
    main()
