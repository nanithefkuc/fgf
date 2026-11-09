"""Observe compiler-mapped kernel functions for each checked public-operation row."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys

import coverage_map


FORMAT = "fgf-route-diagnostic/1"
LIMITATION = (
    "Diagnostic instrumented build only, not the timed artifact. Compiler-mapped "
    "function counters identify observed helpers, not a complete branch trace. "
    "An empty row is unmapped, never evidence of scalar execution."
)


def observed_routes(functions: list[dict], startup: list[dict]) -> list[dict]:
    baseline = {(row["path"], row["file"]): row["count"] for row in startup}
    routes = []
    for row in functions:
        if not row["path"].startswith("fgf::kernel::"):
            continue
        if row["file"] != "src/kernel.rs" and not row["file"].startswith("src/kernel/"):
            continue
        count = row["count"] - baseline.get((row["path"], row["file"]), 0)
        if count > 0:
            routes.append({"path": row["path"], "file": row["file"], "count": count})
    return sorted(routes, key=lambda row: (row["path"], row["file"]))


def probe(
    root: Path,
    binary: Path,
    tier: str,
    toolchain: str,
    scratch: Path,
    cell_ids: list[str] | None = None,
    jobs: int = 0,
) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    llvm_cov, profdata = coverage_map.llvm_tools(toolchain)
    inventory = coverage_map.run_inventory(root, binary, tier, scratch)
    known = {cell["id"] for cell in inventory}
    selected = cell_ids if cell_ids is not None else [cell["id"] for cell in inventory]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= known:
        raise coverage_map.CoverageMapError("route diagnostic needs unique, registered cells")
    filters = []
    for index, cell in enumerate(selected):
        path = scratch / f"filter-{index:04d}.txt"
        path.write_text(cell + "\n", encoding="utf-8")
        filters.append(path)
    spec = {
        "cells": selected, "filters": filters, "binary": binary, "root": root,
        "scratch": scratch, "tier": tier, "llvm_cov": llvm_cov,
        "profdata": profdata, "keep_exports": False, "entries": [], "sites": [],
    }
    startup = json.loads(coverage_map._export_profile(
        spec, scratch / "inventory.profraw", scratch / "inventory.profdata", "inventory",
    ))
    startup_functions = coverage_map.analyze_export(startup, root, [], [])["functions"]
    rows = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs or min(16, os.cpu_count() or 1),
        initializer=coverage_map._worker_init, initargs=(spec,),
    ) as pool:
        futures = [pool.submit(coverage_map._run_one_cell, index) for index in range(len(selected))]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            routes = observed_routes(result["evidence"]["functions"], startup_functions)
            rows[result["cell"]] = {
                "status": "observed" if routes else "unmapped", "functions": routes,
            }
    return {
        "format": FORMAT, "detail": "per-row", "status": "ok",
        "tier": tier, "toolchain": toolchain, "binary": str(binary),
        "limitation": LIMITATION, "rows": dict(sorted(rows.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--tier", choices=coverage_map.TIER_ORDER)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--bench", default="operations")
    parser.add_argument("--features", default="all")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--cells", help="newline-separated registered cell IDs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-dir", default="target/route-probe")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--binary", help="reuse an instrumented diagnostic artifact")
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve()
    out = Path(arguments.out).resolve()
    target = Path(arguments.target_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    try:
        if arguments.binary:
            binary = Path(arguments.binary).resolve()
        else:
            binary = coverage_map.build_bench(
                root, arguments.toolchain, arguments.bench, target / "build",
                out.with_suffix(".build.log"),
                features=[value for value in arguments.features.split(",") if value],
                no_default_features=arguments.no_default_features,
            )
        if arguments.build_only:
            out.write_text(json.dumps({"binary": str(binary)}) + "\n", encoding="utf-8")
            print("route-probe: built", f"binary={binary}")
            return 0
        if arguments.tier is None:
            parser.error("--tier is required unless --build-only is used")
        cells = None
        if arguments.cells:
            cells = [line.strip() for line in Path(arguments.cells).read_text().splitlines() if line.strip()]
        report = probe(root, binary, arguments.tier, arguments.toolchain, target / arguments.tier, cells, arguments.jobs)
    except (coverage_map.CoverageMapError, OSError, ValueError) as error:
        print("route-probe: failed", f"error={str(error)!r}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("route-probe: complete", f"rows={len(report['rows'])}", f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
