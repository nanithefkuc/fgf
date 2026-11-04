#!/usr/bin/env python3
"""Stable public-surface checks over rustdoc JSON and `cargo public-api` output.

Subcommands:

- `paths JSON --mode report|enforce`: the one-path check. Walks every public
  export edge from the crate root, following each ancestry path separately,
  and lists every item reachable through more than one stable path. Paths
  rooted at `<crate>::internals` are not stable and are skipped. Glob
  re-exports outside `internals` are violations. In `enforce` mode any
  violation exits with status 1; `report` mode always exits 0.
- `project`: reads `cargo public-api` output on stdin and writes the stable
  projection to stdout. A line that names a path under `<crate>::internals`
  is dropped, and so are the associated items listed under a dropped impl
  block (they follow it and are qualified by its self type), which covers
  impls of unstable traits for stable or foreign types. A stable signature
  that names an unstable type is caught in the configurations without
  `internals`, where it renders under its private path.
- `same CONFIG FILE...`: requires the inventories of one feature
  configuration, one per target, to be identical, and names the first
  differing target otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

UNSTABLE_ROOT = "internals"
EXPECTED_FORMAT = 61


def load(path: Path) -> dict:
    doc = json.loads(path.read_text())
    version = doc.get("format_version")
    if version != EXPECTED_FORMAT:
        sys.exit(
            f"surface: {path}: rustdoc JSON format {version}, expected {EXPECTED_FORMAT}; "
            "regenerate with the pinned API_NIGHTLY"
        )
    return doc


def kind(item: dict) -> str:
    return next(iter(item["inner"]))


def export_paths(doc: dict) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Return (items, modules, globs): every stable path per item.

    Items are keyed by rustdoc id, or by source path for an external item
    rustdoc records without an id. Modules are tracked separately so totals
    count non-module items, while a module with two paths is still a
    violation.
    """
    index = doc["index"]
    root = str(doc["root"])
    crate = index[root]["name"]
    unstable = f"{crate}::{UNSTABLE_ROOT}"
    items: dict[str, list[str]] = defaultdict(list)
    modules: dict[str, list[str]] = defaultdict(list)
    globs: list[str] = []

    def walk(module_id: str, prefix: str, ancestry: frozenset[str]) -> None:
        for child in index[module_id]["inner"]["module"]["items"]:
            entry = index.get(str(child))
            if entry is None or entry.get("visibility") != "public":
                continue
            k = kind(entry)
            if k == "use":
                use = entry["inner"]["use"]
                target_id = None if use.get("id") is None else str(use["id"])
                target = index.get(target_id) if target_id is not None else None
                is_module = target is not None and kind(target) == "module"
                if use.get("is_glob"):
                    globs.append(f"{prefix}::{use['source']}::*")
                    if is_module and target_id not in ancestry:
                        walk(target_id, prefix, ancestry | {target_id})
                    continue
                path = f"{prefix}::{use['name']}"
                if path == unstable:
                    continue
                if is_module:
                    modules[target_id].append(path)
                    if target_id not in ancestry:
                        walk(target_id, path, ancestry | {target_id})
                    continue
                key = target_id if target_id is not None else f"external:{use['source']}"
                items[key].append(path)
            elif k == "module":
                path = f"{prefix}::{entry['name']}"
                if path == unstable:
                    continue
                cid = str(child)
                modules[cid].append(path)
                if cid not in ancestry:
                    walk(cid, path, ancestry | {cid})
            else:
                items[str(child)].append(f"{prefix}::{entry['name']}")

    walk(root, crate, frozenset({root}))
    return items, modules, globs


def cmd_paths(args: argparse.Namespace) -> int:
    doc = load(Path(args.json))
    index = doc["index"]
    items, modules, globs = export_paths(doc)
    owned = {k: v for k, v in items.items() if k in index}
    external = {k: v for k, v in items.items() if k not in index}
    print(
        f"surface: {len(owned)} crate-owned items through {sum(map(len, owned.values()))} paths; "
        f"{len(external)} external items through {sum(map(len, external.values()))} paths"
    )
    violations = 0
    for label, table in (("item", items), ("module", modules)):
        for paths in sorted((sorted(v) for v in table.values() if len(v) > 1), key=lambda v: v[0]):
            violations += 1
            print(f"duplicate {label} ({len(paths)} paths): {' | '.join(paths)}")
    for glob in sorted(globs):
        violations += 1
        print(f"glob re-export: {glob}")
    print(f"surface: {violations} one-path violation(s)")
    return 1 if violations and args.mode == "enforce" else 0


DECLARATION = re.compile(
    r'^(?:pub\s+)?(?:(?:const|unsafe|async|extern\s+"[^"]*"|fn|struct|enum|union|trait|type|mod|use|static|mut)\s+)*'
)


def impl_self_type(line: str) -> str:
    """Return the self type of an `impl` line as `cargo public-api` renders it."""
    rest = line[len("impl"):]
    if rest.startswith("<"):
        depth = 0
        for i, ch in enumerate(rest):
            depth += (ch == "<") - (ch == ">")
            if depth == 0:
                rest = rest[i + 1:]
                break
    rest = rest.strip()
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif depth == 0 and rest.startswith(" for ", i):
            rest = rest[i + len(" for "):]
            break
    return rest.split(" where ")[0].strip()


def cmd_project(args: argparse.Namespace) -> int:
    unstable = f"{args.crate}::{UNSTABLE_ROOT}"
    dropped_items: str | None = None
    for line in sys.stdin:
        text = line.rstrip("\n")
        if text.startswith("impl"):
            dropped_items = f"{impl_self_type(text)}::" if unstable in text else None
            if dropped_items is None:
                sys.stdout.write(line)
            continue
        if dropped_items is not None:
            if DECLARATION.sub("", text, count=1).startswith(dropped_items):
                continue
            dropped_items = None
        if unstable not in text:
            sys.stdout.write(line)
    return 0


def cmd_same(args: argparse.Namespace) -> int:
    files = [Path(f) for f in args.files]
    reference = files[0].read_text()
    for other in files[1:]:
        if other.read_text() != reference:
            print(
                f"surface: configuration {args.config}: {other.name} differs from {files[0].name}; "
                "the stable surface must be identical on every supported target",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("paths")
    p.add_argument("json")
    p.add_argument("--mode", choices=("report", "enforce"), required=True)
    p.set_defaults(run=cmd_paths)
    p = sub.add_parser("project")
    p.add_argument("--crate", default="fgf")
    p.set_defaults(run=cmd_project)
    p = sub.add_parser("same")
    p.add_argument("config")
    p.add_argument("files", nargs="+")
    p.set_defaults(run=cmd_same)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
