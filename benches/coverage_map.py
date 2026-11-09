#!/usr/bin/env python3
"""Instrumented coverage map for the public operations bench.

Builds ``benches/operations.rs`` with LLVM source coverage on the pinned
nightly toolchain, runs ``--check`` once per cell under
``SIMD_BACKEND=<tier>``, and merges the per-cell profiles into one map:

- every kernel-registry entry at the requested tier, with the cells that
  executed it and the evidence (a function-level count on the entry wrapper,
  or the nearest mapped caller region when the pinned compiler emits no
  mapping for the ``#[archmage::arcane]`` wrapper);
- every routing threshold and tuning parameter from the threshold inventory,
  with its comparison branch taken both ways (true and false counts and the
  cells responsible), or the route value exercised when it is a selector
  rather than a comparison;
- unreachable items with reasons, and every remaining unexecuted item as a
  gap.

The tool exits non-zero when a required entry is unexecuted or a mapped
threshold branch is one-sided, unless ``--exceptions`` records the item with
a reason.

Macro call paths bind fixed comma-separated captures to their invocation
arguments. Attribution uses invocation-specific LLVM expansion records or
function regions whose v0 impl-parent path identifies exactly one invocation.
Pooled definition counts never credit an expansion. Unsupported symbol paths,
repeated captures, and ambiguous invocations produce ``macro-unattributed``
gaps.

Crate-internal shared helpers inherit the weakest direct caller tier only when
their name is unique and every non-import reference is a call with a known
tier. Stable-reachable public functions stay unresolved. Arm and short-circuit
guard hints require token-proven enclosure; unknown references stay required.

Usage::

    python3 benches/coverage_map.py --tier v3_gfni_crypto \
        --inventory /path/to/inventory.json --scan /path/to/scan.py
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

FORMAT = "fgf-coverage-map/1"
TIER_RANK = {
    "scalar": 0,
    "v2": 1,
    "v3": 2,
    "v3_gfni_crypto": 3,
    "v4x": 4,
}
TIER_ORDER = tuple(sorted(TIER_RANK, key=TIER_RANK.get))
COVERAGE_RUSTFLAGS = (
    "-Cinstrument-coverage",
    "-Zcoverage-options=branch",
    "-Cdebuginfo=1",
    # Keep call sites out of line: the proc-macro-generated `#[arcane]`
    # wrappers carry no coverage mapping on the pinned toolchain, so the
    # mapped caller region is the fallback entry evidence.
    "-Zinline-llvm=no",
)
DEFAULT_EXCEPTIONS = "benches/coverage_exceptions.json"
ANALYSIS_FORMAT = "fgf-coverage-evidence/3"


class CoverageMapError(Exception):
    """A fail-closed coverage-map condition with its evidence."""


# ---------------------------------------------------------------------------
# Synthetic-llvm-cov analysis (pure functions; the unit tests drive these)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class MacroCallSite:
    """A definition position bound to one concrete macro invocation."""

    file: str
    line: int
    column: int
    invocation_file: str
    invocation_line: int
    invocation_column: int
    invocation_module: str = ""


@dataclasses.dataclass(frozen=True)
class EntryTarget:
    entry: str
    tier: str
    file: str
    start_line: int
    end_line: int
    publicly_reachable: bool
    call_sites: tuple[tuple[str, int, int] | MacroCallSite, ...] = ()


@dataclasses.dataclass(frozen=True)
class SiteTarget:
    site_id: int
    item_key: str
    file: str
    line: int
    column: int
    role: str  # "branch" or "use"
    condition_line: int | None = None
    condition_column: int | None = None
    condition_end_line: int | None = None
    condition_end_column: int | None = None
    description: str = ""


@dataclasses.dataclass
class CoverageIndex:
    """Per-file regions, segments, and branches from one llvm-cov export."""

    regions: dict[str, list[tuple[int, int, int, int, int]]]
    branches: dict[str, list[tuple[int, int, int, int, int, int, int, int, int]]]
    segments: dict[str, list[tuple[int, int, int, bool, bool, bool]]] = dataclasses.field(
        default_factory=dict
    )
    expansions: dict[
        tuple[str, int, int], dict[str, list[tuple[int, int, int, int, int]]]
    ] = dataclasses.field(default_factory=dict)
    functions: dict[
        str, list[tuple[str | None, list[tuple[int, int, int, int, int]]]]
    ] = dataclasses.field(default_factory=dict)
    named_functions: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self._segment_keys = {
            file: [(segment[0], segment[1]) for segment in values]
            for file, values in self.segments.items()
        }


def relative_source(path: str, root: Path) -> str | None:
    """The crate-relative source path for an absolute coverage filename."""

    root_text = str(root.resolve())
    if path == root_text:
        return None
    prefix = root_text + os.sep
    if path.startswith(prefix):
        return path[len(prefix) :]
    return None


def _v0_disambiguator(name: str, offset: int) -> int:
    if name[offset] == "s":
        end = name.index("_", offset)
        if not re.fullmatch(r"[0-9a-zA-Z]*", name[offset + 1 : end]):
            raise ValueError
        return end + 1
    return offset


def _v0_identifier(name: str, offset: int) -> tuple[str, int]:
    offset = _v0_disambiguator(name, offset)
    match = re.match(r"[0-9]+", name[offset:])
    if match is None:
        raise ValueError
    length = int(match.group())
    start = offset + len(match.group())
    if name[start] == "_":
        start += 1
    value = name[start : start + length]
    if len(value) != length or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError
    return value, start + length


def _v0_path(name: str, offset: int, namespaces: str) -> tuple[list[str], int]:
    tag = name[offset]
    if tag == "C":
        value, end = _v0_identifier(name, offset + 1)
        return [value], end
    if tag == "N" and name[offset + 1] in namespaces:
        parent, end = _v0_path(name, offset + 2, namespaces)
        value, end = _v0_identifier(name, end)
        return parent + [value], end
    raise ValueError


@functools.lru_cache(maxsize=32768)
def impl_parent(name: str) -> str | None:
    """Decode only a v0 impl's lexical parent, which demanglers may omit.

The supported prefix grammar is generic/nested paths leading to an impl,
with a crate-root/nested module parent. Other encodings fail closed.
"""
    def origin(offset):
        tag = name[offset]
        if tag in {"M", "X"}:
            parent, _ = _v0_path(name, _v0_disambiguator(name, offset + 1), "t")
            return "::".join(parent)
        if tag == "I":
            return origin(offset + 1)
        if tag == "N" and name[offset + 1].isascii() and name[offset + 1].isalpha():
            return origin(offset + 2)
        raise ValueError

    if not name.startswith("_R"):
        return None
    try:
        return origin(2)
    except (ValueError, IndexError, RecursionError):
        return None


@functools.lru_cache(maxsize=32768)
def source_function_path(name: str) -> str | None:
    """Decode a named free function's path; unsupported v0 encodings fail closed."""
    if not name.startswith("_R"):
        return None
    try:
        offset = 2
        while name[offset] == "I":
            offset += 1
        if name[offset:offset + 2] != "Nv":
            return None
        path, _ = _v0_path(name, offset, "tv")
        return "::".join(path)
    except (ValueError, IndexError, RecursionError):
        return None



def build_coverage_index(export: dict, root: Path) -> CoverageIndex:
    """Index one ``llvm-cov export -format=text`` document by source file."""

    data = export.get("data")
    if not isinstance(data, list) or not data:
        raise CoverageMapError("llvm-cov export has no data record")
    record = data[0]
    regions: dict[str, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    branches: dict[str, list[tuple[int, int, int, int, int, int, int, int, int]]] = defaultdict(list)
    functions = defaultdict(list)
    named_functions: dict[tuple[str, str], int] = defaultdict(int)
    for function in record.get("functions", []):
        names = [relative_source(name, root) for name in function.get("filenames", [])]
        function_regions = defaultdict(list)
        for region in function.get("regions", []):
            if len(region) < 8 or region[7] != 0:
                continue
            file_id = region[5]
            if not isinstance(file_id, int) or not 0 <= file_id < len(names):
                continue
            relative = names[file_id]
            if relative is None:
                continue
            regions[relative].append(tuple(region[:5]))
            if len(region) >= 8 and region[7] == 0:
                function_regions[relative].append(tuple(region[:5]))
        parent = impl_parent(function.get("name", ""))
        for relative, values in function_regions.items():
            functions[relative].append((parent, values))
        path = source_function_path(function.get("name", ""))
        count = function.get("count", 0)
        if path is not None and isinstance(count, int) and count > 0:
            for relative in function_regions:
                named_functions[(path, relative)] += count
    segments: dict[str, list[tuple[int, int, int, bool, bool, bool]]] = defaultdict(list)
    expansions = {}
    for file_record in record.get("files", []):
        relative = relative_source(file_record.get("filename", ""), root)
        if relative is None:
            continue
        for branch in file_record.get("branches", []):
            if len(branch) < 6:
                continue
            branches[relative].append(tuple(branch[:9]))
        for segment in file_record.get("segments", []):
            if len(segment) < 6:
                continue
            segments[relative].append(
                (segment[0], segment[1], segment[2], bool(segment[3]), bool(segment[4]), bool(segment[5]))
            )
        for expansion in file_record.get("expansions", []):
            names = [relative_source(name, root) for name in expansion.get("filenames", [])]
            source = expansion.get("source_region", [])
            if len(source) < 8 or not 0 <= source[5] < len(names):
                continue
            invocation_file = names[source[5]]
            if invocation_file is None:
                continue
            key = (invocation_file, source[0], source[1])
            definition_regions = expansions.setdefault(key, defaultdict(list))
            for region in expansion.get("target_regions", []):
                # Only code regions count; expansion/gap/skipped records do not
                # prove that the call expression executes.
                if len(region) < 8 or region[7] != 0 or not 0 <= region[5] < len(names):
                    continue
                definition_file = names[region[5]]
                if definition_file is not None:
                    definition_regions[definition_file].append(tuple(region[:5]))
    for values in segments.values():
        values.sort(key=lambda segment: (segment[0], segment[1]))
    return CoverageIndex(
        regions=dict(regions), branches=dict(branches), segments=dict(segments),
        expansions=expansions, functions=dict(functions),
        named_functions=dict(named_functions),
    )


def _contains(region: tuple[int, int, int, int, int], line: int, column: int) -> bool:
    line_start, column_start, line_end, column_end, _ = region
    if line < line_start or line > line_end:
        return False
    if line == line_start and column < column_start:
        return False
    if line == line_end and column > column_end:
        return False
    return True


def position_evidence(
    index: CoverageIndex, file: str, line: int, column: int
) -> tuple[bool, int]:
    """Whether a mapped region contains the position, and its count."""

    segments = index.segments.get(file)
    if segments:
        keys = index._segment_keys[file]
        position = bisect.bisect_right(keys, (line, column)) - 1
        if position < 0:
            return False, 0
        segment = segments[position]
        if not segment[3]:
            return False, 0
        return True, segment[2]
    hits = [region for region in index.regions.get(file, ()) if _contains(region, line, column)]
    if not hits:
        return False, 0
    smallest = min(hits, key=lambda region: (region[2] - region[0], region[3] - region[1]))
    return True, smallest[4]


def macro_call_evidence(
    index: CoverageIndex, site: MacroCallSite, peers: set[MacroCallSite]
) -> tuple[bool, int]:
    """Execution at a definition position within exactly one invocation."""

    regions = index.expansions.get(
        (site.invocation_file, site.invocation_line, site.invocation_column), {}
    ).get(site.file, ())
    hits = [region for region in regions if _contains(region, site.line, site.column)]
    if not hits and site.invocation_module:
        invocations = {
            (peer.invocation_file, peer.invocation_line, peer.invocation_column)
            for peer in peers if peer.invocation_module == site.invocation_module
        }
        if len(invocations) != 1:
            return False, 0
        # Keep each function's region counts separate. In particular, a live
        # function does not make its untaken branch's call execute.
        counts = []
        for parent, function_regions in index.functions.get(site.file, ()):
            if parent != site.invocation_module:
                continue
            regions = [
                region for region in function_regions
                if _contains(region, site.line, site.column)
            ]
            if regions:
                size = lambda region: (region[2] - region[0], region[3] - region[1])
                smallest = min(map(size, regions))
                counts.append(max(r[4] for r in regions if size(r) == smallest))
        return (True, max(counts)) if counts else (False, 0)
    if not hits:
        return False, 0
    size = lambda region: (region[2] - region[0], region[3] - region[1])
    smallest = min(map(size, hits))
    return True, max(region[4] for region in hits if size(region) == smallest)


def wrapper_evidence(
    index: CoverageIndex, target: EntryTarget
) -> tuple[bool, int]:
    """Function-level evidence on the entry's definition span."""

    hits = [
        region
        for region in index.regions.get(target.file, ())
        if region[0] <= target.end_line and region[2] >= target.start_line
    ]
    if not hits:
        return False, 0
    return True, max(region[4] for region in hits)


def branch_evidence(
    index: CoverageIndex, target: SiteTarget
) -> tuple[bool, int, int, bool]:
    """Counts for the one branch record matching a comparison's span.

    Returns ``(mapped, true, false, ambiguous)``. Multiple instantiations of
    the same branch span are summed; two or more distinct spans overlapping
    the comparison are ambiguous, never a reason to borrow a neighbour's
    counts.
    """

    condition_line = target.condition_line if target.condition_line is not None else target.line
    condition_column = (
        target.condition_column if target.condition_column is not None else target.column
    )
    condition_end_line = (
        target.condition_end_line if target.condition_end_line is not None else condition_line
    )
    condition_end_column = (
        target.condition_end_column
        if target.condition_end_column is not None
        else condition_column
    )
    records = index.branches.get(target.file, ())
    if not records:
        return False, 0, 0, False

    def span(record):
        return (record[0], record[1], record[2], record[3])

    def totals(group):
        return sum(record[4] for record in group), sum(record[5] for record in group)

    exact = [
        record
        for record in records
        if span(record) == (condition_line, condition_column, condition_end_line, condition_end_column)
    ]
    if exact:
        true_count, false_count = totals(exact)
        return True, true_count, false_count, False

    overlapping = []
    for record in records:
        if not (record[0] <= condition_line <= record[2]):
            continue
        if condition_end_line == condition_line == record[0] == record[2]:
            if record[1] > condition_end_column or record[3] < condition_column:
                continue
        overlapping.append(record)
    if not overlapping:
        return False, 0, 0, False
    groups = defaultdict(list)
    for record in overlapping:
        groups[span(record)].append(record)
    if len(groups) > 1:
        return True, 0, 0, True
    group = next(iter(groups.values()))
    true_count, false_count = totals(group)
    return True, true_count, false_count, False


def analyze_export(
    export: dict,
    root: Path,
    entries: list[EntryTarget],
    sites: list[SiteTarget],
) -> dict:
    """Compact per-cell evidence for every target in one export."""

    index = build_coverage_index(export, root)
    macro_peers = defaultdict(set)
    for target in entries:
        for site in target.call_sites:
            if isinstance(site, MacroCallSite):
                macro_peers[(site.file, site.line, site.column)].add(site)
    entry_evidence = {}
    for target in entries:
        mapped, count = wrapper_evidence(index, target)
        # Per-call-site counts stay aligned with target.call_sites: a shared
        # kernel must not pool one caller's execution into another caller.
        call_counts: list[int | None] = [None] * len(target.call_sites)
        macro_unattributed = []
        if count <= 0:
            for position, site in enumerate(target.call_sites):
                if isinstance(site, MacroCallSite):
                    call_mapped, call_count = macro_call_evidence(
                        index, site, macro_peers[(site.file, site.line, site.column)]
                    )
                    if not call_mapped:
                        macro_unattributed.append(dataclasses.asdict(site))
                else:
                    call_mapped, call_count = position_evidence(index, *site)
                if call_mapped:
                    call_counts[position] = call_count
        entry_evidence[target.entry] = {
            "wrapper_mapped": mapped,
            "wrapper_count": count,
            "call_counts": call_counts,
            "macro_unattributed": macro_unattributed,
        }
    site_evidence = {}
    for target in sites:
        if target.role == "branch":
            mapped, true_count, false_count, ambiguous = branch_evidence(index, target)
            condition_line = target.condition_line if target.condition_line is not None else target.line
            condition_column = (
                target.condition_column
                if target.condition_column is not None
                else target.column
            )
            site_mapped, _ = position_evidence(index, target.file, condition_line, condition_column)
            site_evidence[str(target.site_id)] = {
                "mapped": mapped,
                "site_mapped": site_mapped,
                "true": true_count,
                "false": false_count,
                "ambiguous": ambiguous,
            }
        else:
            mapped, count = position_evidence(index, target.file, target.line, target.column)
            site_evidence[str(target.site_id)] = {"mapped": mapped, "count": count}
    return {
        "format": ANALYSIS_FORMAT,
        "entries": entry_evidence,
        "sites": site_evidence,
        "functions": [
            {"path": path, "file": file, "count": count}
            for (path, file), count in sorted(index.named_functions.items())
        ],
        "files": len(index.regions),
        "branches": sum(len(value) for value in index.branches.values()),
    }


def operation_function_cells(
    cell_evidence: list[tuple[str, dict]], cells: list[dict],
    startup_functions: list[dict] | None = None,
) -> dict[tuple[str, str], set[str]]:
    """Witness operation-owned counters beyond the inventory startup footprint."""
    operations = {
        cell["id"]: cell["operation"] + ("_with" if cell.get("prep") == "prepared" else "")
        for cell in cells
    }
    startup = {
        (item["path"], item["file"]): item["count"]
        for item in startup_functions or ()
    }
    root_counts: dict[tuple[str, str], int] = defaultdict(int)
    for file, model in _SOURCE_MODELS.items():
        if file.startswith("src/ops/"):
            for function in _source_functions(file, model):
                root_counts[(function.module + "::" + function.name, file)] += 1
    witnesses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cell, evidence in cell_evidence:
        active = [
            function for function in evidence.get("functions", ())
            if function["count"] > startup.get((function["path"], function["file"]), 0)
        ]
        if not any(
            root_counts.get((function["path"], function["file"])) == 1
            and function["path"].rpartition("::")[2] == operations.get(cell)
            for function in active
        ):
            continue
        for function in active:
            witnesses[(function["path"], function["file"])].add(cell)
    return dict(witnesses)


# ---------------------------------------------------------------------------
# Registry, inventory, and exception loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CoverageMapError(f"{path}: file not found")
    except json.JSONDecodeError as error:
        raise CoverageMapError(f"{path}: invalid JSON: {error}")


def load_registry(path: Path) -> list[dict]:
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise CoverageMapError(f"{path}: kernel registry has no entries array")
    entries = []
    for entry in data["entries"]:
        if not isinstance(entry, dict) or "entry" not in entry or "tier" not in entry:
            raise CoverageMapError(f"{path}: malformed registry entry {entry!r}")
        if entry["tier"] not in TIER_RANK:
            raise CoverageMapError(f"{path}: unknown tier {entry['tier']!r}")
        entries.append(entry)
    return entries


def load_private_spans(path: Path) -> dict[str, dict]:
    """Definition path -> rustdoc span from the private rustdoc document."""

    data = load_json(path)
    if not isinstance(data, dict) or "paths" not in data or "index" not in data:
        raise CoverageMapError(f"{path}: private rustdoc document is missing paths/index")
    spans: dict[str, dict] = {}
    for item_id, entry in data["paths"].items():
        item = data["index"].get(str(item_id))
        if not isinstance(item, dict) or "span" not in item:
            continue
        path_text = "::".join(entry.get("path", ()))
        if path_text:
            spans[path_text] = item["span"]
    return spans


def load_inventory(path: Path) -> list[dict]:
    data = load_json(path)
    if not isinstance(data, list):
        raise CoverageMapError(f"{path}: inventory root is not an array")
    return data


EXCEPTION_CLASSES = {
    "unreachable",
    "unmapped_arcane",
    "unmapped_branch",
}


def load_exceptions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, list):
        exceptions = data
    elif isinstance(data, dict):
        exceptions = data.get("exceptions", [])
    else:
        raise CoverageMapError(f"{path}: exceptions must be an array or an object")
    result: dict[str, dict] = {}
    for entry in exceptions:
        if not isinstance(entry, dict):
            raise CoverageMapError(f"{path}: exception entry is not an object: {entry!r}")
        key = entry.get("key", entry.get("item"))
        if not isinstance(key, str) or not key:
            raise CoverageMapError(f"{path}: exception needs a `key`: {entry!r}")
        item_class = entry.get("class")
        if item_class not in EXCEPTION_CLASSES:
            raise CoverageMapError(
                f"{path}: {key}: class must be one of {sorted(EXCEPTION_CLASSES)}"
            )
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise CoverageMapError(f"{path}: {key}: exception needs a `reason`")
        tiers = entry.get("tiers")
        if tiers is not None and (
            not isinstance(tiers, list)
            or not tiers
            or any(tier not in TIER_RANK for tier in tiers)
        ):
            raise CoverageMapError(f"{path}: {key}: `tiers` must be a non-empty tier list")
        straddling = entry.get("straddling_cells")
        if item_class in {"unmapped_arcane", "unmapped_branch"}:
            groups = [straddling]
            if isinstance(straddling, dict):
                if tiers is None or set(straddling) != set(tiers):
                    raise CoverageMapError(
                        f"{path}: {key}: tier witness sets must match `tiers` exactly"
                    )
                groups = list(straddling.values())
            if not groups or any(
                not isinstance(group, list)
                or not group
                or not all(isinstance(cell, str) and cell for cell in group)
                for group in groups
            ):
                raise CoverageMapError(
                    f"{path}: {key}: {item_class} needs non-empty `straddling_cells`"
                )
        elif straddling is not None:
            raise CoverageMapError(
                f"{path}: {key}: `straddling_cells` is only valid for "
                "unmapped_arcane and unmapped_branch"
            )
        via = entry.get("via_entry")
        if via is not None and (
            item_class != "unmapped_arcane"
            or not isinstance(via, dict)
            or not via
            or tiers is None
            or not set(via).issubset(tiers)
        ):
            raise CoverageMapError(
                f"{path}: {key}: `via_entry` needs scoped unmapped_arcane caller entries"
            )
        for proof in via.values() if via is not None else ():
            if (
                not isinstance(proof, dict)
                or set(proof) != {"entry", "sources"}
                or not isinstance(proof["entry"], str)
                or not proof["entry"]
                or not isinstance(proof["sources"], dict)
                or not proof["sources"]
                or any(
                    not isinstance(file, str) or not file
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    for file, digest in proof["sources"].items()
                )
            ):
                raise CoverageMapError(
                    f"{path}: {key}: `via_entry` needs an entry and source SHA-256 fingerprints"
                )
        if key in result:
            raise CoverageMapError(f"{path}: duplicate exception for {key}")
        result[key] = dict(entry, key=key)
    return result


def _declared_cells(entry: dict, tier: str) -> list[str]:
    """Select the common or tier-specific witness set."""
    cells = entry.get("straddling_cells") or []
    if isinstance(cells, dict):
        if tier not in cells:
            raise CoverageMapError(
                f"exceptions: {entry['key']}: no straddling cells for tier {tier}"
            )
        return cells[tier]
    return cells


# ---------------------------------------------------------------------------
# Source scanning: entry call sites and threshold positions
# ---------------------------------------------------------------------------


def load_scan_module(path: Path):
    spec = importlib.util.spec_from_file_location("fgf_threshold_scan", path)
    if spec is None or spec.loader is None:
        raise CoverageMapError(f"{path}: cannot import the threshold scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_sources(root: Path, scan_module) -> dict[str, object]:
    models = {}
    # Caller inference needs every library caller, not just threshold owners.
    # The scanner strips cfg(test) bodies; standalone test modules stay excluded.
    files = {
        path.relative_to(root).as_posix()
        for path in (root / "src").rglob("*.rs")
        if not (set(path.relative_to(root / "src").parts) & {"tests", "tests.rs"})
    }
    for relative in sorted(files):
        source = (root / relative).read_text(encoding="utf-8")
        models[relative] = scan_module.SourceModel(relative, source)
    return models


def _in_use_item(tokens: list, index: int) -> bool:
    j = index - 1
    while j >= 0 and index - j <= 120:
        text = tokens[j].text
        if text == "use":
            return True
        if text == ";":
            return False
        j -= 1
    return False


def _call_path(tokens: list, index: int) -> tuple[list[str], int] | None:
    """The path segments ending at a called function name, and its start."""

    if index + 1 >= len(tokens):
        return None
    follower = tokens[index + 1].text
    if follower not in {"(", "::"}:
        return None
    if follower == "::":
        if index + 2 >= len(tokens) or tokens[index + 2].text != "<":
            return None
    if index > 0 and tokens[index - 1].text in {"fn", "."}:
        return None
    start = index - 1 if index > 0 and tokens[index - 1].text == "$" else index
    segments = [("$" if start != index else "") + tokens[index].text]
    j = start
    while j >= 2 and tokens[j - 1].text == "::" and tokens[j - 2].kind == "ident":
        ident = j - 2
        start = ident - 1 if ident > 0 and tokens[ident - 1].text == "$" else ident
        segments.insert(0, ("$" if start != ident else "") + tokens[ident].text)
        j = start
    return segments, tokens[start].start


def _delimiter_end(tokens: list, start: int) -> int:
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index in range(start, len(tokens)):
        text = tokens[index].text
        if text in pairs:
            stack.append(pairs[text])
        elif stack and text == stack[-1]:
            stack.pop()
            if not stack:
                return index
    raise CoverageMapError("unclosed macro delimiter")


def _macro_arguments(tokens: list) -> list[list]:
    """Comma-separated arguments, retaining nested groups and type arguments."""

    arguments = []
    start = 0
    depth = 0
    for index, token in enumerate(tokens):
        text = token.text
        if text in {"(", "[", "{", "<"}:
            depth += 1
        elif text in {")", "]", "}", ">", ">>"}:
            depth -= 2 if text == ">>" else 1
        elif text == "," and depth == 0:
            arguments.append(tokens[start:index])
            start = index + 1
    if start < len(tokens):
        arguments.append(tokens[start:])
    return arguments


def invocation_module(file: str, tokens: list, index: int) -> str:
    """Lexical module path for an ordinary crate source file invocation."""

    path = Path(file).with_suffix("")
    if not path.parts or path.parts[0] != "src":
        return ""
    parts = ["fgf", *path.parts[1:]]
    if parts[-1] in {"lib", "mod"}:
        parts.pop()
    for start in range(index - 2):
        if (
            tokens[start].text == "mod"
            and tokens[start + 1].kind == "ident"
            and tokens[start + 2].text == "{"
            and _delimiter_end(tokens, start + 2) > index
        ):
            parts.append(tokens[start + 1].text)
    return "::".join(parts)


def _macro_calls(models: dict[str, object]) -> tuple[dict, dict]:
    """Resolve fixed comma-separated macro captures; reject ambiguous rules.

    Repetitions other than optional trailing commas and optional final captures
    are not resolved. Definition positions never fall back to pooled counts.
    """

    definitions = defaultdict(list)
    ranges = defaultdict(list)
    capture = r"\$(\w+):(ident|ty|path|literal|expr|tt)"
    optional = rf"\$\(\s*,\s*{capture}\s*\)\?"
    grammar = rf"(?:{capture}|{optional}|,|\$\(\s*,\s*\)\?|\s)+"
    for relative, model in models.items():
        tokens = model.tokens
        for index, token in enumerate(tokens[:-3]):
            if token.text != "macro_rules" or tokens[index + 1].text != "!":
                continue
            end = _delimiter_end(tokens, index + 3)
            ranges[relative].append((index, end))
            pattern_start = index + 4
            if tokens[pattern_start].text not in {"(", "[", "{"}:
                continue
            pattern_end = _delimiter_end(tokens, pattern_start)
            if tokens[pattern_end + 1].text != "=>":
                continue
            body_start = pattern_end + 2
            body_end = _delimiter_end(tokens, body_start)
            if any(t.text != ";" for t in tokens[body_end + 1 : end]):
                continue
            pattern = " ".join(t.text for t in tokens[pattern_start + 1 : pattern_end])
            pattern = re.sub(r"\s*([$:])\s*", r"\1", pattern)
            pattern = re.sub(r"\)\s*\?", ")?", pattern)
            if re.fullmatch(grammar, pattern) is None:
                continue
            parameters = re.findall(capture, pattern)
            if len({name for name, _ in parameters}) != len(parameters):
                continue
            optional_names = {match[0] for match in re.findall(optional, pattern)}
            definitions[tokens[index + 2].text].append(
                (relative, body_start, body_end, parameters, optional_names)
            )
    calls = defaultdict(list)
    invocation_modules = defaultdict(set)
    for invocation_file, model in models.items():
        tokens = model.tokens
        for index, token in enumerate(tokens[:-2]):
            if token.text not in definitions or tokens[index + 1].text != "!":
                continue
            if any(start <= index <= end for start, end in ranges[invocation_file]):
                continue
            choices = definitions[token.text]
            if len(choices) != 1 or tokens[index + 2].text not in {"(", "[", "{"}:
                continue
            relative, body_start, body_end, parameters, optional_names = choices[0]
            end = _delimiter_end(tokens, index + 2)
            module = invocation_module(invocation_file, tokens, index)
            invocation_modules[(relative, body_start, body_end, module)].add(
                (invocation_file, token.start)
            )
            arguments = _macro_arguments(tokens[index + 3 : end])
            if not len(parameters) - len(optional_names) <= len(arguments) <= len(parameters):
                continue
            bindings = {
                name: "".join(t.text for t in argument)
                for (name, _), argument in zip(parameters, arguments)
            }
            definition = models[relative]
            for body_index in range(body_start + 1, body_end):
                if definition.tokens[body_index].kind != "ident":
                    continue
                call = _call_path(definition.tokens, body_index)
                if call is None or _in_use_item(definition.tokens, body_index):
                    continue
                segments, start = call
                resolved = []
                for segment in segments:
                    value = bindings.get(segment[1:]) if segment.startswith("$") else segment
                    if segment == "$crate":
                        value = "crate"
                    if value is None or not re.fullmatch(r"\w+(?:::\w+)*", value):
                        break
                    resolved.extend(value.split("::"))
                else:
                    site = MacroCallSite(
                        relative, *scan_line_column(definition.source, start),
                        invocation_file, *scan_line_column(model.source, token.start),
                        module,
                    )
                    calls[resolved[-1]].append((relative, resolved, site))
    for values in calls.values():
        for index, (relative, resolved, site) in enumerate(values):
            ambiguous = any(
                key[0] == relative and key[3] == site.invocation_module and len(positions) != 1
                for key, positions in invocation_modules.items()
            )
            if ambiguous:
                values[index] = (
                    relative, resolved, dataclasses.replace(site, invocation_module="")
                )
    return calls, ranges


def discover_call_sites(
    entries: list[dict], models: dict[str, object]
) -> dict[str, tuple[tuple[str, int, int] | MacroCallSite, ...]]:
    """Candidate call sites, binding macro paths to concrete invocations."""

    name_counts = Counter(entry["entry"].rsplit("::", 1)[-1] for entry in entries)
    macro_calls, macro_ranges = _macro_calls(models)
    occurrences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for relative, model in models.items():
        for index, token in enumerate(model.tokens):
            if token.kind == "ident":
                occurrences[token.text].append((relative, index))
    sites = defaultdict(list)
    for entry in entries:
        path = entry["entry"]
        parts = path.split("::")
        name = parts[-1]
        family = parts[3:] if len(parts) > 3 else parts
        family = [part for part in family if part != name]
        model_cache: dict[str, object] = {}
        for relative, index in occurrences.get(name, ()):
            model = model_cache.get(relative)
            if model is None:
                model = models[relative]
                model_cache[relative] = model
            tokens = model.tokens
            if any(start <= index <= end for start, end in macro_ranges[relative]):
                continue
            if index in model.attribute_tokens or index in model.generic_tokens:
                continue
            call = _call_path(tokens, index)
            if call is None:
                continue
            if _in_use_item(tokens, index):
                continue
            segments, start = call
            if name_counts[name] > 1:
                if len(segments) > 1:
                    if not any(segment in family for segment in segments[:-1]):
                        continue
                elif not any(part in relative for part in family[:2]):
                    continue
            line, column = scan_line_column(model.source, start)
            sites[path].append((relative, line, column))
        for relative, segments, site in macro_calls.get(name, ()):
            suffix = segments[segments.index("x86") :] if "x86" in segments else segments
            if len(segments) == 1:
                # A bare name needs lexical correspondence, not registry uniqueness.
                if not site.invocation_module:
                    continue
                suffix = [*site.invocation_module.split("::"), *segments]
            # Family facades can omit private ISA modules from the call path.
            definition_parts = iter(parts)
            if not all(segment in definition_parts for segment in suffix):
                continue
            sites[path].append(site)
    result = {}
    for path, values in sites.items():
        unique = sorted(set(values), key=lambda site: (
            site.file, site.line, site.column, site.invocation_file,
            site.invocation_line, site.invocation_column,
        ) if isinstance(site, MacroCallSite) else (*site, "", 0, 0))
        result[path] = tuple(unique)
    return result


def scan_line_column(source: str, offset: int) -> tuple[int, int]:
    return source.count("\n", 0, offset) + 1, offset - source.rfind("\n", 0, offset)


def build_entry_targets(
    registry: list[dict],
    spans: dict[str, dict],
    call_sites: dict[str, tuple[tuple[str, int, int] | MacroCallSite, ...]],
    private_allowlist: set[str],
) -> list[EntryTarget]:
    targets = []
    missing = []
    for entry in registry:
        span = spans.get(entry["entry"])
        if span is None:
            missing.append(entry["entry"])
            continue
        targets.append(
            EntryTarget(
                entry=entry["entry"],
                tier=entry["tier"],
                file=span["filename"],
                start_line=span["begin"][0],
                end_line=span["end"][0],
                publicly_reachable=entry["entry"] not in private_allowlist,
                call_sites=call_sites.get(entry["entry"], ()),
            )
        )
    if missing:
        raise CoverageMapError(
            "the private rustdoc document has no span for: " + ", ".join(missing[:8])
        )
    return sorted(targets, key=lambda target: target.entry)


def build_site_targets(
    inventory: list[dict], positions: dict[str, dict]
) -> tuple[list[SiteTarget], list[dict]]:
    """Branch and use targets for every routing/tuning inventory item."""

    targets: list[SiteTarget] = []
    items: list[dict] = []
    next_id = 0

    def add(item_key: str, file: str, line: int, column: int, role: str, **kwargs) -> int:
        nonlocal next_id
        target = SiteTarget(
            site_id=next_id,
            item_key=item_key,
            file=file,
            line=line,
            column=column,
            role=role,
            **kwargs,
        )
        targets.append(target)
        next_id += 1
        return target.site_id

    for item in inventory:
        if item.get("class") not in {"routing_threshold", "tuning_parameter"}:
            continue
        key = item["key"]
        position = positions.get(key)
        site_ids: list[int] = []
        if position is not None:
            condition = position.get("condition")
            uses = position.get("uses") or []
            is_comparison = "::cmp:" in key
            is_expression = (
                "::match:" in key
                or "::return:" in key
                or "::tail:" in key
                or "::let " in key
            )
            if is_comparison:
                site_ids.append(
                    add(
                        key,
                        position["file"],
                        position["line"],
                        position["column"],
                        "branch",
                        condition_line=condition["line"] if condition else None,
                        condition_column=condition["column"] if condition else None,
                        condition_end_line=condition["end_line"] if condition else None,
                        condition_end_column=condition["end_column"] if condition else None,
                        description="comparison",
                    )
                )
            elif uses:
                for use in uses:
                    if use["kind"] == "compare":
                        site_ids.append(
                            add(
                                key,
                                use["file"],
                                use["line"],
                                use["column"],
                                "branch",
                                condition_line=use["condition_line"],
                                condition_column=use["condition_column"],
                                condition_end_line=use["condition_end_line"],
                                condition_end_column=use["condition_end_column"],
                                description=f"use {use['operator']}",
                            )
                        )
                    else:
                        site_ids.append(
                            add(
                                key,
                                use["file"],
                                use["line"],
                                use["column"],
                                "use",
                                description="use",
                            )
                        )
            elif is_expression:
                site_ids.append(
                    add(
                        key,
                        position["file"],
                        position["line"],
                        position["column"],
                        "use",
                        description="selector value",
                    )
                )
        items.append(
            {
                "key": key,
                "class": item["class"],
                "value": item.get("value"),
                "unit": item.get("unit"),
                "axis": item.get("axis"),
                "routes": item.get("routes"),
                "comparison": item.get("comparison"),
                "site_ids": site_ids,
                "mapped": position is not None,
            }
        )
    return targets, items


# ---------------------------------------------------------------------------
# Summaries and rules
# ---------------------------------------------------------------------------


def summarize_entries(
    targets: list[EntryTarget],
    cell_evidence: list[tuple[str, dict]],
    requested_tier: str,
    exceptions: dict[str, dict],
) -> list[dict]:
    requested_rank = TIER_RANK[requested_tier]
    rows = []
    for target in targets:
        executed_cells = []
        wrapper_cells = []
        site_cells = [set() for _ in target.call_sites]
        call_cells = []
        macro_unattributed = {}
        total = 0
        for cell, evidence in cell_evidence:
            record = evidence["entries"].get(target.entry)
            if record is None:
                continue
            count = record.get("wrapper_count", 0)
            counts = record.get("call_counts") or []
            call_count = max(
                (value for value in counts if isinstance(value, int)), default=0
            )
            for position, value in enumerate(counts):
                if isinstance(value, int) and value > 0 and position < len(site_cells):
                    site_cells[position].add(cell)
            for site in record.get("macro_unattributed", []):
                macro_unattributed[tuple(site.values())] = site
            if count > 0:
                wrapper_cells.append(cell)
            if call_count > 0:
                call_cells.append(cell)
            if count > 0 or call_count > 0:
                executed_cells.append(cell)
                total += max(count, call_count)
        required = TIER_RANK[target.tier] == requested_rank and target.publicly_reachable
        if executed_cells:
            status = "executed"
            evidence_kind = "wrapper" if wrapper_cells else "call_site"
        elif required:
            status = "unexecuted"
            evidence_kind = "none"
        else:
            status = "not_required"
            evidence_kind = "none"
        rows.append(
            {
                "entry": target.entry,
                "tier": target.tier,
                "rank": TIER_RANK[target.tier],
                "required": required,
                "publicly_reachable": target.publicly_reachable,
                "status": status,
                "evidence": evidence_kind,
                "count": total,
                "executed_by": executed_cells,
                "executed_by_call_site": call_cells,
                "call_site_cells": [sorted(cells) for cells in site_cells],
                "definition": {"file": target.file, "line": target.start_line},
                "reason": "macro-unattributed: no per-expansion call mapping"
                if macro_unattributed and not executed_cells else None,
                "macro_unattributed": list(macro_unattributed.values()),
                "call_sites": [
                    dataclasses.asdict(site) if isinstance(site, MacroCallSite) else
                    {"file": site[0], "line": site[1], "column": site[2]}
                    for site in target.call_sites
                ],
            }
        )
    return rows


_SOURCE_MODELS: dict[str, object] = {}
_SOURCE_TIER_SPANS: dict[str, list[tuple[int, int, str]]] = {}
_CALLER_TIERS: dict[tuple[str, int, int], str] = {}
_STRATEGY_VARIANTS: dict[str, set[str]] = {}
_VARIANT_TIERS = {
    "V4x": "v4x", "Avx512": "v4x",
    "V3GfniCrypto": "v3_gfni_crypto", "Gfni": "v3_gfni_crypto",
    "V3": "v3", "Avx2": "v3",
    "V2": "v2", "Ssse3": "v2", "Sse42": "v2", "Sse2": "v2",
    "Scalar": "scalar",
}
_SOURCE_GENERATION = 0


def configure_sources() -> None:
    global _SOURCE_GENERATION
    _SOURCE_GENERATION += 1
    _SOURCE_MODELS.clear()
    _SOURCE_TIER_SPANS.clear()
    _CALLER_TIERS.clear()
    _STRATEGY_VARIANTS.clear()


def _pattern_tier(tokens: list) -> str | None:
    """The weakest tier of a fully recognized pattern, excluding its guard."""
    pattern = "".join(token.text for token in tokens)
    variant = (
        r"(?:\w+::)*(?:Backend|X86Proof|\w*Strategy)::"
        r"(V4x|V3GfniCrypto|V3|V2|Avx512|Gfni|Avx2|Ssse3|Sse42|Sse2|Scalar)"
        r"(?:\(\w+\))?"
    )
    if re.fullmatch(rf"{variant}(?:\|{variant})*", pattern) is None:
        return None
    return min((_VARIANT_TIERS[name] for name in re.findall(variant, pattern)), key=TIER_RANK.get)


def _closed_strategy_variants(models: dict[str, object]) -> dict[str, set[str]]:
    """Unique fieldless strategy enums whose complete variant set is recognized."""
    choices = defaultdict(list)
    for model in models.values():
        tokens = model.tokens
        for index, token in enumerate(tokens[:-2]):
            if token.text == "type" and tokens[index + 1].text.endswith("Strategy"):
                choices[tokens[index + 1].text].append(set())
            if token.text != "enum" or not tokens[index + 1].text.endswith("Strategy"):
                continue
            name = tokens[index + 1].text
            variants = set()
            if tokens[index + 2].text == "{":
                closing = _delimiter_end(tokens, index + 2)
                cursor = index + 3
                while cursor < closing:
                    if tokens[cursor].text == "#" and tokens[cursor + 1].text == "[":
                        cursor = _delimiter_end(tokens, cursor + 1) + 1
                        continue
                    if (
                        tokens[cursor].text not in _VARIANT_TIERS
                        or tokens[cursor + 1].text not in {",", "}"}
                    ):
                        variants.clear()
                        break
                    variants.add(tokens[cursor].text)
                    cursor += 2
            choices[name].append(variants)
    return {
        name: values[0] for name, values in choices.items()
        if len(values) == 1 and values[0]
    }


def _strategy_guard_tier(tokens: list, *, truth: bool) -> str | None:
    """Capability floor of a simple strategy equality or closed-set inequality."""
    comparison = [i for i, token in enumerate(tokens) if token.text in {"==", "!="}]
    if len(comparison) != 1:
        return None
    index = comparison[0]
    lhs = "".join(token.text for token in tokens[:index])
    rhs = "".join(token.text for token in tokens[index + 1:])
    if re.fullmatch(r"(?:\w+::)*\w+(?:\(\))?", lhs) is None:
        return None
    pattern = re.fullmatch(r"(\w+Strategy)::(\w+)", rhs)
    if pattern is None:
        return None
    name, variant = pattern.groups()
    variants = _STRATEGY_VARIANTS.get(name)
    if variants is None or variant not in variants:
        return None
    equal = (tokens[index].text == "==") == truth
    if equal:
        return _pattern_tier(tokens[index + 1:])
    remaining = variants - {variant}
    return min((_VARIANT_TIERS[value] for value in remaining), key=TIER_RANK.get) if remaining else None


def _strategy_context_explicit(tokens: list) -> bool:
    """Refuse aliases and glob imports in the source context."""
    for index, token in enumerate(tokens[:-2]):
        if token.text == "use":
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            if any(part.text in {"as", "*"} for part in tokens[index + 1:end]):
                return False
    return True


def _strategy_if_spans(
    model, index: int, groups: dict[int, int], *, context_explicit: bool
) -> list[tuple[int, int, str]]:
    """Short-circuit RHSs and direct continuation after an immutable guard returns."""
    if not context_explicit:
        return []
    tokens = model.tokens
    declaration = model.scope_decl(index)
    if (
        tokens[declaration].text != "fn" or tokens[declaration + 2].text == "<"
        or any(scope.startswith(("impl ", "trait ", "macro_rules! ")) for scope in model.scopes[index])
    ):
        return []
    opening = index + 1
    while opening < len(tokens) and tokens[opening].text not in {"{", ";", "=>", "}"}:
        opening = groups.get(opening, opening) + 1
    if opening not in groups or tokens[opening].text != "{":
        return []
    closing = groups[opening]
    condition = tokens[index + 1:opening]
    if any(token.text == "||" for token in condition):
        return []
    conjunction = next((i for i, token in enumerate(condition) if token.text == "&&"), None)
    if conjunction is not None:
        tier = _strategy_guard_tier(condition[:conjunction], truth=True)
        if tier is None:
            return []
        spans = [(condition[conjunction].start, tokens[opening].start, tier)]
        if not any(token.text == "fn" for token in tokens[opening + 1:closing]):
            spans.append((tokens[opening].start, tokens[closing].end, tier))
        return spans
    if index > 0 and tokens[index - 1].text == "else":
        return []
    tier = _strategy_guard_tier(condition, truth=False)
    body = tokens[opening + 1:closing]
    if (
        tier is None or len(body) < 2 or body[0].text != "return"
        or body[-1].text != ";" or sum(token.text == ";" for token in body) != 1
        or any(token.text in {"{", "}"} for token in body)
        or (closing + 1 < len(tokens) and tokens[closing + 1].text == "else")
    ):
        return []
    # Continuation facts attach only to a local immutable value, never a
    # second call to a getter whose result could change.
    if len(condition) < 3 or condition[0].kind != "ident" or condition[1].text not in {"==", "!="}:
        return []
    name = condition[0].text
    parents = [left for left, right in groups.items() if tokens[left].text == "{" and left < index < right]
    if not parents:
        return []
    parent = max(parents)
    immutable = False
    cursor = parent + 1
    while cursor < index:
        if tokens[cursor].text == "let":
            end = cursor + 1
            while end < index and tokens[end].text not in {"=", ";"}:
                end += 1
            if any(token.text == name for token in tokens[cursor + 1:end]):
                immutable = end == cursor + 2 and tokens[cursor + 1].text == name
        cursor = groups.get(cursor, cursor) + 1
    if not immutable:
        return []
    spans = []
    start = tokens[closing].end
    cursor = closing + 1
    while cursor < groups[parent]:
        if tokens[cursor].text == "let":
            end = cursor + 1
            while end < groups[parent] and tokens[end].text not in {"=", ";"}:
                end += 1
            if any(token.text == name for token in tokens[cursor + 1:end]):
                break
        if tokens[cursor].text == "{":
            spans.append((start, tokens[cursor].end, tier))
            cursor = groups[cursor]
            start = tokens[cursor].end
        cursor += 1
    spans.append((start, tokens[cursor].start, tier))
    return spans


def _tier_spans(model) -> list[tuple[int, int, str]]:
    """Strategy-constrained expressions and blocks over the scanned Rust tokens."""
    tokens = model.tokens
    context_explicit = _strategy_context_explicit(tokens)
    groups = {}
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index, token in enumerate(tokens):
        if token.text in pairs:
            stack.append(index)
        elif token.text in pairs.values():
            if not stack or pairs[tokens[stack[-1]].text] != token.text:
                return []
            groups[stack.pop()] = index
    if stack:
        return []

    def seek(start, stops, limit):
        while start < limit:
            if tokens[start].text in stops:
                return start
            start = groups.get(start, start) + 1
        return limit

    spans = []
    for index, token in enumerate(tokens):
        if token.text == "match":
            opening = seek(index + 1, {"{", ";", "=>", "}"}, len(tokens))
            if opening not in groups or tokens[opening].text != "{":
                continue
            closing = groups[opening]
            start = opening + 1
            while start < closing:
                # Attributes belong to an arm, not to its tier pattern.
                while tokens[start].text == "#" and start + 1 in groups:
                    start = groups[start + 1] + 1
                arrow = seek(start, {"=>", ","}, closing)
                if arrow == closing or tokens[arrow].text != "=>":
                    break
                guard = seek(start, {"if"}, arrow)
                tier = _pattern_tier(tokens[start:guard])
                body = arrow + 1
                if body >= closing:
                    break
                if tokens[body].text == "{":
                    end = groups[body] + 1
                else:
                    end = seek(body, {",", "=>"}, closing)
                    if end < closing and tokens[end].text == "=>":
                        break  # Unsupported block-like expression without a comma.
                comma = end < closing and tokens[end].text == ","
                if tier is not None:
                    right = tokens[end].end if comma else tokens[end - 1].end
                    spans.append((tokens[start].start, right, tier))
                start = end + comma
        elif token.text == "if" and index + 1 < len(tokens) and tokens[index + 1].text == "let":
            equal = seek(index + 2, {"=", "{"}, len(tokens))
            if equal == len(tokens) or tokens[equal].text != "=":
                continue
            tier = _pattern_tier(tokens[index + 2:equal])
            # Only a plain path or path call proves which brace is the body.
            # A block scrutinee executes before its pattern has matched.
            opening = equal + 1
            if (
                opening == len(tokens) or tokens[opening].kind != "ident"
                or tokens[opening].text in {"unsafe", "async", "const", "try", "loop"}
            ):
                continue
            opening += 1
            while (
                opening + 1 < len(tokens) and tokens[opening].text == "::"
                and tokens[opening + 1].kind == "ident"
            ):
                opening += 2
            if opening in groups and tokens[opening].text == "(":
                opening = groups[opening] + 1
            if tier is not None and opening in groups and tokens[opening].text == "{":
                spans.append((tokens[opening].start, tokens[groups[opening]].end, tier))
        elif token.text == "if":
            spans.extend(_strategy_if_spans(
                model, index, groups, context_explicit=context_explicit,
            ))
    return spans


def _arm_tier_near(file: str, line: int) -> str | None:
    model = _SOURCE_MODELS.get(file)
    if model is None:
        return None
    if file not in _SOURCE_TIER_SPANS:
        _SOURCE_TIER_SPANS[file] = _tier_spans(model)
    lines = model.source.splitlines(keepends=True)
    if not 1 <= line <= len(lines):
        return None
    start = sum(map(len, lines[:line - 1]))
    end = start + len(lines[line - 1].rstrip())
    start += len(lines[line - 1]) - len(lines[line - 1].lstrip())
    # A line shared with another arm or a portable tail cannot prove enclosure.
    tiers = [
        tier for left, right, tier in _SOURCE_TIER_SPANS[file]
        if left <= start < end <= right
    ]
    return max(tiers, key=TIER_RANK.get) if tiers else None


def site_tier_hint(file: str, line: int | None = None) -> str | None:
    """Tier of a threshold site from its kernel module or dispatch arm."""

    if "avx512" in file:
        tier = "v4x"
    elif "gfni" in file:
        tier = "v3_gfni_crypto"
    elif "avx2" in file:
        tier = "v3"
    elif any(name in file for name in ("ssse3", "sse2", "sse42")):
        tier = "v2"
    else:
        tier = None
    if tier is None and line is not None:
        tier = _arm_tier_near(file, line)
        if tier is None:
            for (source, start, end), hint in _CALLER_TIERS.items():
                if source == file and start <= line <= end:
                    return hint
    return tier


def _item_visibility(tokens: list, index: int) -> str:
    """Classify an item's visibility without treating restricted pub as public."""
    start = index
    while start > 0 and tokens[start - 1].text not in {";", "{", "}"}:
        start -= 1
    for position in range(start, index):
        if tokens[position].text == "pub":
            return "restricted" if tokens[position + 1].text == "(" else "public"
    return "private"


def _public_boundaries(models: dict[str, object]) -> tuple[dict[str, str], set[str]]:
    """Module visibilities and possible non-internals public export names."""
    declarations = {}
    exports = set()
    for source, model in models.items():
        for position, token in enumerate(model.tokens[:-1]):
            if token.text == "mod" and model.tokens[position + 1].kind == "ident":
                parent = invocation_module(source, model.tokens, position)
                path = parent + "::" + model.tokens[position + 1].text
                visibility = _item_visibility(model.tokens, position)
                if declarations.get(path) != "public":
                    declarations[path] = visibility
            if token.text != "use":
                continue
            owner = invocation_module(source, model.tokens, position).split("::")
            if owner[:2] == ["fgf", "internals"]:
                continue
            end = position + 1
            while end < len(model.tokens) and model.tokens[end].text != ";":
                end += 1
            public = _item_visibility(model.tokens, position) == "public"
            # A private module alias can itself be re-exported under a new name.
            renamed = any(token.text == "as" for token in model.tokens[position + 1:end])
            if not public and not renamed:
                continue
            if any(token.text == "internals" for token in model.tokens[position + 1:end]):
                # An alias of the facade can expose its entire export graph.
                exports.add("*")
            for leaf in range(position + 1, end):
                token = model.tokens[leaf]
                if token.text in {"*", "self"}:
                    exports.add("*")  # Module-self exports need full path resolution.
                elif token.kind == "ident" and model.tokens[leaf + 1].text in {",", "}", ";", "as"}:
                    exports.add(token.text)
    return declarations, exports


def _internal_public_function(file: str, index: int, models: dict[str, object],
                              boundaries: tuple[dict[str, str], set[str]]) -> bool:
    """Prove a private boundary with no possible stable re-export.

    Re-exports mentioning the function or an ancestor, and globs, block proof
    even from private modules. Missing declarations never prove privacy.
    Facade aliases and any public cfg alternative also block proof.
    """
    declarations, exports = boundaries
    module = invocation_module(file, models[file].tokens, index)
    ancestors = module.split("::")
    if not module or ancestors[0] != "fgf":
        return False
    paths = ["::".join(ancestors[:end]) for end in range(2, len(ancestors) + 1)]
    if not all(path in declarations for path in paths):
        return False
    if not any(declarations[path] != "public" for path in paths):
        return False
    names = set(ancestors[1:]) | {models[file].tokens[index + 1].text}
    return "*" not in exports and not names & exports


def _caller_tier(file: str, model, index: int) -> str | None:
    """Tier of a call, including an exact equality's short-circuit RHS."""
    line, _ = scan_line_column(model.source, model.tokens[index].start)
    tier = site_tier_hint(file, line)
    if tier is not None:
        return tier
    # Recognize only `if tier == Strategy::Variant && ... call(...)`.
    # No disjunction, block, or statement may intervene before the call.
    start = index - 1
    while start >= 0 and model.tokens[start].text not in {"if", "{", "}", ";", "||"}:
        start -= 1
    if start < 0 or model.tokens[start].text != "if":
        return None
    prefix = model.tokens[start + 1:index]
    if len(prefix) < 6 or prefix[0].kind != "ident" or prefix[1].text != "==":
        return None
    conjunction = next((i for i, token in enumerate(prefix) if token.text == "&&"), None)
    if conjunction is None:
        return None
    return _pattern_tier(prefix[2:conjunction])


def configure_caller_tiers(models: dict[str, object]) -> None:
    """Hint enumerable helpers only when every reference has a known tier.

    Hints propagate through complete helper chains. Imports do not execute a
    helper; indirect uses, duplicate names, macros, stable-reachable functions,
    and unknown-tier callers stay unresolved.
    """

    global _SOURCE_GENERATION
    _SOURCE_GENERATION += 1
    _SOURCE_MODELS.clear()
    _SOURCE_MODELS.update(models)
    _SOURCE_TIER_SPANS.clear()
    _CALLER_TIERS.clear()
    declarations = defaultdict(list)
    references = defaultdict(list)
    _STRATEGY_VARIANTS.clear()
    _STRATEGY_VARIANTS.update(_closed_strategy_variants(models))
    boundaries = _public_boundaries(models)
    for file, model in models.items():
        ends = {}
        for index, token in enumerate(model.tokens):
            if any(scope.startswith("fn ") for scope in model.scopes[index]):
                ends[model.scope_decl(index)] = token.end
            if token.kind == "ident":
                references[token.text].append((file, index))
        for index, token in enumerate(model.tokens[:-1]):
            if token.text != "fn" or model.tokens[index + 1].kind != "ident":
                continue
            name = model.tokens[index + 1].text
            visibility = _item_visibility(model.tokens, index)
            enumerable = visibility != "public" or _internal_public_function(
                file, index, models, boundaries
            )
            enumerable = enumerable and not any(
                scope.startswith(("impl ", "trait ", "macro_rules! "))
                for scope in model.scopes[index]
            )
            end = ends.get(index)
            declarations[name].append((file, index, end, enumerable))
    while True:
        inferred = {}
        for name, choices in declarations.items():
            if len(choices) != 1:
                continue
            file, declaration, end, enumerable = choices[0]
            if not enumerable or end is None:
                continue
            model = models[file]
            start_line, _ = scan_line_column(model.source, model.tokens[declaration].start)
            end_line, _ = scan_line_column(model.source, end)
            key = (file, start_line, end_line)
            if key in _CALLER_TIERS:
                continue
            tiers = []
            for caller_file, index in references[name]:
                if caller_file == file and index == declaration + 1:
                    continue
                model = models[caller_file]
                if any(scope.startswith("macro_rules! ") for scope in model.scopes[index]):
                    break
                if _in_use_item(model.tokens, index):
                    import_end = index
                    while import_end < len(model.tokens) and model.tokens[import_end].text != ";":
                        import_end += 1
                    if any(t.text == "as" for t in model.tokens[index:import_end]):
                        break
                    continue
                if _call_path(model.tokens, index) is None:
                    break
                hint = _caller_tier(caller_file, model, index)
                if hint is None:
                    break
                tiers.append(hint)
            else:
                if tiers:
                    inferred[key] = min(tiers, key=TIER_RANK.get)
        if not inferred:
            break
        _CALLER_TIERS.update(inferred)


def summarize_thresholds(
    items: list[dict],
    targets: list[SiteTarget],
    cell_evidence: list[tuple[str, dict]],
    exceptions: dict[str, dict],
    requested_tier: str,
    reach: _Reachability | None = None,
) -> list[dict]:
    by_id = {target.site_id: target for target in targets}
    site_state: dict[int, dict] = {}
    for target in targets:
        site_state[target.site_id] = {
            "target": target,
            "mapped": False,
            "site_mapped": False,
            "ambiguous": False,
            "true": 0,
            "false": 0,
            "executed": 0,
            "true_cells": set(),
            "false_cells": set(),
            "executed_cells": set(),
        }
    for cell, evidence in cell_evidence:
        for key, record in evidence["sites"].items():
            state = site_state.get(int(key))
            if state is None:
                continue
            if record.get("site_mapped"):
                state["site_mapped"] = True
            if record.get("ambiguous"):
                state["ambiguous"] = True
            if not record.get("mapped"):
                continue
            state["mapped"] = True
            if state["target"].role == "branch":
                true_count = record.get("true", 0)
                false_count = record.get("false", 0)
                state["true"] += true_count
                state["false"] += false_count
                if true_count > 0:
                    state["true_cells"].add(cell)
                if false_count > 0:
                    state["false_cells"].add(cell)
            else:
                count = record.get("count", 0)
                state["executed"] += count
                if count > 0:
                    state["executed_cells"].add(cell)

    rows = []
    for item in items:
        exception = exceptions.get(item["key"])
        sites = []
        statuses = []
        for site_id in item["site_ids"]:
            state = site_state[site_id]
            target = state["target"]
            hint = site_tier_hint(target.file, target.line)
            observed = bool(state["executed"] or state["true"] or state["false"])
            if not observed and reach is not None:
                observed = bool(_serving_cells(
                    item["key"],
                    {"sites": [{"file": target.file, "line": target.line,
                                "column": target.column}]},
                    reach,
                    measured_only=hint is not None
                    and TIER_RANK[hint] > TIER_RANK[requested_tier],
                ))
            site_required = (
                hint is None or TIER_RANK[hint] == TIER_RANK[requested_tier] or observed
            )
            if target.role == "branch":
                if not site_required:
                    status = "tier_not_required"
                elif state["ambiguous"]:
                    status = "ambiguous"
                elif not state["mapped"] and state["site_mapped"]:
                    status = "unexecuted"
                elif not state["mapped"]:
                    status = "unmapped"
                elif state["true"] == 0 and state["false"] == 0:
                    status = "unexecuted"
                elif state["true"] == 0 or state["false"] == 0:
                    status = "one_sided"
                else:
                    status = "both_ways"
                sites.append(
                    {
                        "site_id": site_id,
                        "file": target.file,
                        "line": target.line,
                        "column": target.column,
                        "role": target.role,
                        "tier_hint": hint,
                        "required": site_required,
                        "description": target.description,
                        "condition": {
                            "line": target.condition_line,
                            "column": target.condition_column,
                        },
                        "mapped": state["mapped"],
                        "ambiguous": state["ambiguous"],
                        "true": state["true"],
                        "false": state["false"],
                        "true_cells": sorted(state["true_cells"]),
                        "false_cells": sorted(state["false_cells"]),
                        "status": status,
                    }
                )
            else:
                if not site_required:
                    status = "tier_not_required"
                elif not state["mapped"]:
                    status = "unmapped"
                elif state["executed"] == 0:
                    status = "unexecuted"
                else:
                    status = "exercised"
                sites.append(
                    {
                        "site_id": site_id,
                        "file": target.file,
                        "line": target.line,
                        "column": target.column,
                        "role": target.role,
                        "tier_hint": hint,
                        "required": site_required,
                        "description": target.description,
                        "mapped": state["mapped"],
                        "count": state["executed"],
                        "executed_cells": sorted(state["executed_cells"]),
                        "status": status,
                    }
                )
            statuses.append(status)

        relevant = [status for status in statuses if status != "tier_not_required"]
        if not sites:
            status = "unmapped"
            reason = "no source position or use site"
        elif not relevant:
            status = "tier_not_required"
            reason = "no observed execution at this tier; each site's own tier session measures it"
        elif "ambiguous" in statuses:
            status = "ambiguous"
            reason = "two or more branch records overlap the comparison; attribution is ambiguous"
        elif "one_sided" in relevant:
            status = "one_sided"
            reason = "a mapped comparison branch is taken one way only"
        elif "unexecuted" in statuses:
            status = "unexecuted"
            failing = [
                site
                for site in sites
                if site["status"] == "unexecuted" and site.get("required", True)
            ]
            above = [
                site for site in failing
                if site.get("tier_hint") is not None
                and TIER_RANK.get(site["tier_hint"], 0) > TIER_RANK[requested_tier]
            ]
            if len(above) == len(failing):
                reason = (
                    "mapped site tier "
                    + "/".join(sorted({site["tier_hint"] for site in above}))
                    + " is above the requested tier; its own tier session measures it"
                )
            else:
                reason = "a mapped site never executed"
        elif "unmapped" in relevant and all(status == "unmapped" for status in relevant):
            status = "unmapped"
            reason = "the pinned compiler emitted no mapping for any site"
        elif "unmapped" in relevant:
            status = "partial_mapping"
            reason = "some sites have no compiler mapping; every mapped site passed"
        elif any(status == "both_ways" for status in relevant):
            status = "both_ways"
            reason = None
        else:
            status = "exercised"
            reason = None

        required = any(site.get("required", True) for site in sites) if sites else True
        rows.append(
            {
                "key": item["key"],
                "class": item["class"],
                "required": required,
                "value": item["value"],
                "unit": item["unit"],
                "axis": item["axis"],
                "routes": item["routes"],
                "comparison": item.get("comparison"),
                "status": status,
                "reason": reason,
                "sites": sites,
            }
        )
    return rows


_COUNT_AXIS_FIELDS = {
    "source count": "sources",
    "sources per register group": "sources",
    "cohort fill": "rows",
    "row group width": "rows",
    "remaining row count": "rows",
    "output row count": "rows",
    "outputs per delayed tile": "rows",
    "terms per resolved tile": "sources",
    "terms per destination block": "sources",
    "terms per resolve chunk": "sources",
    "term count": "sources",
    "remainder alignment exponent": "gather_remainder_alignment",
}


def _axis_field(unit: str, axis: str) -> str | None:
    """The bench cell geometry field a threshold's unit and axis label read.

    Byte, row, term, and source axes map to the cell geometry they name.
    Count axes resolve through the scanner's axis label because the bare
    unit does not identify the geometry. Unsupported axes return `None`;
    their items are validated for applicability only.
    """
    if unit == "bytes":
        return "row_len"
    if unit == "elements":
        return "elements"
    if unit == "rows":
        return "rows"
    if unit in {"terms", "sources"}:
        return "sources"
    if unit == "count":
        return _COUNT_AXIS_FIELDS.get(axis.strip())
    return None


def _axis_value(field: str, cell: dict) -> int | None:
    if field == "elements":
        width, length = cell.get("element_bytes"), cell.get("row_len")
        if (
            isinstance(width, int) and width > 0 and isinstance(length, int)
            and length % width == 0
        ):
            return length // width
        return None
    if field == "gather_remainder_alignment":
        length, sources = cell.get("row_len"), cell.get("sources")
        # The fused GFNI gather evaluates this exponent only after its
        # short-row and source-count guards, before any complete tile.
        if (
            isinstance(length, int) and 0 < length < 128
            and isinstance(sources, int) and sources > 2
        ):
            return (length & -length).bit_length() - 1
        return None
    value = cell.get(field)
    return value if isinstance(value, int) else None


# A scatter cohort four rows wide leaves `((rows - 1) % 4) + 1` rows in
# its final group, so `filled == 4` reads that remainder, not the row count.
_COHORT_WIDTH = 4


def _axis_quantity(axis: str, field: str, value: int) -> int:
    if axis == "cohort fill" and field == "rows":
        return ((value - 1) % _COHORT_WIDTH) + 1
    return value


def _satisfies(value: int, op: str, constant: int) -> bool:
    """The scanner's low-side cutoff predicate: `value <op> constant`."""
    if op == "==":
        return value == constant
    if op == "<":
        return value < constant
    if op == "<=":
        return value <= constant
    raise CoverageMapError(f"unsupported source predicate {op!r}")


# ---------------------------------------------------------------------------
# Source reachability: the cells that can provably execute a helper body
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Function:
    """One `fn` body with its lexical module identity."""

    file: str
    name: str
    decl: int
    open: int
    close: int
    module: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.file, self.decl)


@dataclasses.dataclass
class _Reachability:
    """Call graph over the scanned sources, seeded with entry executions.

    A registry entry's measured cells witness its own body, and each
    measured call site witnesses the body containing it. Witnesses flow
    along direct calls from callers to callees, but never into a registry
    entry, whose cells stay exactly what the session measured. Name
    resolution is conservative: ambiguous names, aliases, globs, method
    calls, and unresolvable paths produce no edge, and an unresolved helper
    receives no cells rather than a same-family credit.
    """

    functions: dict[str, list[_Function]]
    owner: dict[tuple[str, int], _Function]
    nested: set[tuple[str, int]]
    by_module: dict[tuple[str, str], list[_Function]]
    imports: dict[tuple[str, str], dict[str, str]]
    callees: dict[tuple[str, int], set[tuple[str, int]]]
    witness: dict[tuple[str, int], set[str]]
    entry_spans: dict[str, list[tuple[int, int, str]]]
    entry_cells: dict[str, set[str]]
    line_offsets: dict[str, list[int]]
    token_starts: dict[str, list[int]]
    direct_witness: dict[tuple[str, int], set[str]] = dataclasses.field(default_factory=dict)
    entry_calls: dict[tuple[str, int], set[tuple[str, int]]] = dataclasses.field(default_factory=dict)

    def owner_at(self, file: str, line: int, column: int = 1) -> _Function | None:
        offsets = self.line_offsets.get(file)
        starts = self.token_starts.get(file)
        if offsets is None or starts is None or not 1 <= line <= len(offsets):
            return None
        offset = offsets[line - 1] + max(column - 1, 0)
        index = bisect.bisect_right(starts, offset) - 1
        if index < 0:
            return None
        return self.owner.get((file, index))


def _source_functions(file: str, model) -> list[_Function]:
    """Top-level and nested `fn` bodies of one file, in body order."""
    tokens = model.tokens
    functions = []
    for index, token in enumerate(tokens[:-1]):
        if token.text != "fn" or tokens[index + 1].kind != "ident":
            continue
        if index in model.attribute_tokens or index in model.generic_tokens:
            continue
        if any(scope.startswith("macro_rules!") for scope in model.scopes[index]):
            continue
        position = index + 2
        while position < len(tokens) and position in model.generic_tokens:
            position += 1  # the generic parameter list
        if position >= len(tokens) or tokens[position].text != "(":
            continue
        depth = 0
        close = None
        for scan in range(position, len(tokens)):
            text = tokens[scan].text
            if text in {"(", "[", "{"}:
                depth += 1
            elif text in {")", "]", "}"}:
                depth -= 1
                if depth == 0:
                    close = scan
                    break
        if close is None:
            continue
        open_index = None
        scan = close + 1
        while scan < len(tokens):
            text = tokens[scan].text
            if text == "[":
                try:
                    scan = _delimiter_end(tokens, scan)
                except CoverageMapError:
                    break
            elif text == "{":
                open_index = scan
                break
            elif text == ";":
                break
            scan += 1
        if open_index is None:
            continue
        try:
            body_close = _delimiter_end(tokens, open_index)
        except CoverageMapError:
            continue
        functions.append(
            _Function(
                file=file,
                name=tokens[index + 1].text,
                decl=index,
                open=open_index,
                close=body_close,
                module=invocation_module(file, tokens, index),
            )
        )
    return sorted(functions, key=lambda function: function.open)


def _owner_map(file: str, functions: list[_Function]) -> dict[int, _Function]:
    """Innermost `fn` per token index, signatures included.

    A function owns its declaration and signature tokens, so a call site on
    a default-argument expression attributes to that function. Nested
    bodies belong to the nested function.
    """
    owner: dict[int, _Function] = {}
    stack: list[_Function] = []
    position = 0
    order = sorted(functions, key=lambda function: function.decl)
    for index in range(_token_count_floor(functions)):
        while position < len(order) and order[position].decl == index:
            stack.append(order[position])
            position += 1
        while stack and index > stack[-1].close:
            stack.pop()
        if stack:
            owner[index] = stack[-1]
    return owner


def _nested_functions(functions: list[_Function]) -> set[tuple[str, int]]:
    """Keys of functions declared inside another function's body."""
    nested: set[tuple[str, int]] = set()
    for outer in functions:
        for inner in functions:
            if inner is not outer and outer.decl < inner.decl < outer.close:
                nested.add(inner.key)
    return nested


def _token_count_floor(functions: list[_Function]) -> int:
    return max((function.close for function in functions), default=-1) + 1


def _absolute_module(module: str, segments: list[str]) -> str | None:
    """Normalize keyword-led or crate-led segments; None when unsupported."""
    parts = module.split("::") if module else []
    index = 0
    while index < len(segments) and segments[index] == "super":
        if len(parts) < 2:
            return None
        parts = parts[:-1]
        index += 1
    remainder = segments[index:]
    if index == 0 and remainder:
        lead = remainder[0]
        if lead == "crate" or lead == "fgf":
            parts = ["fgf"]
            remainder = remainder[1:]
        elif lead == "self":
            remainder = remainder[1:]
        elif not module.startswith("fgf::"):
            return None
    if not remainder:
        return None
    return "::".join([*parts, *remainder])


def _use_leaves(
    module: str, tokens: list, start: int, end: int
) -> dict[str, str]:
    """Imported local names of one `use` item, or an empty mapping.

    Aliases (`as`), globs (`*`), extern-crate paths, and absolute `::`
    paths are unsupported and contribute nothing, so names reachable only
    through them stay unresolved.
    """
    result: dict[str, str] = {}

    def walk(position: int, base: list[str], stop: int) -> tuple[int, bool]:
        segments: list[str] = []
        while position < stop:
            text = tokens[position].text
            if text in {",", "}"}:
                break
            if text == "{":
                try:
                    close = _delimiter_end(tokens, position)
                except CoverageMapError:
                    return position, False
                if close >= stop:
                    return position, False
                inner = position + 1
                while inner < close:
                    inner, ok = walk(inner, base + segments, close)
                    if not ok:
                        return inner, False
                    if inner < close:
                        if tokens[inner].text != ",":
                            return inner, False
                        inner += 1
                return close + 1, True
            if text == "::":
                position += 1
                continue
            if tokens[position].kind != "ident":
                return position, False
            if position + 1 < stop and tokens[position + 1].text == "as":
                return position, False
            segments.append(text)
            position += 1
        if segments:
            absolute = _absolute_module(module, base + segments[:-1])
            if absolute is not None:
                result.setdefault(segments[-1], absolute)
        return position, True

    walk(start, [], end)
    return result


def _source_imports(file: str, model) -> dict[tuple[str, str], dict[str, str]]:
    """Imported names per (file, lexical module)."""
    tokens = model.tokens
    imports: dict[tuple[str, str], dict[str, str]] = {}
    for index, token in enumerate(tokens):
        if token.text != "use" or token.kind != "ident":
            continue
        if index in model.attribute_tokens or index in model.generic_tokens:
            continue
        if any(scope.startswith("macro_rules!") for scope in model.scopes[index]):
            continue
        end = index + 1
        while end < len(tokens) and tokens[end].text != ";":
            end += 1
        body = tokens[index + 1 : end]
        if any(part.text in {"as", "*"} for part in body):
            continue
        module = invocation_module(file, tokens, index)
        leaves = _use_leaves(module, tokens, index + 1, end)
        if leaves:
            imports.setdefault((file, module), {}).update(leaves)
    return imports


def _resolve_call(
    reach: _Reachability, file: str, module: str, segments: list[str]
) -> _Function | None:
    """The single function a call path names, or None when it is not provable."""
    if not segments or any(segment.startswith("$") for segment in segments):
        return None
    name = segments[-1]
    candidates: list[_Function] = []
    if len(segments) == 1:
        candidates.extend(reach.by_module.get((module, name), ()))
        imported = reach.imports.get((file, module), {}).get(name)
        if imported is not None:
            candidates.extend(reach.by_module.get((imported, name), ()))
    else:
        lead = segments[0]
        if lead in {"crate", "super", "self", "fgf"}:
            absolute = _absolute_module(module, segments)
            if absolute is not None:
                parts = absolute.split("::")
                for cut in range(len(parts) - 1, 0, -1):
                    found = reach.by_module.get(("::".join(parts[:cut]), name), ())
                    if found:
                        candidates.extend(found)
                        break
        else:
            # A plain relative path: every ancestor module of the caller is
            # a candidate scope for the leading segments.
            parts = module.split("::") if module else []
            for keep in range(len(parts), 0, -1):
                scope = "::".join([*parts[:keep], *segments[:-1]])
                found = reach.by_module.get((scope, name), ())
                if found:
                    candidates.extend(found)
                    break
    unique = {candidate.key: candidate for candidate in candidates}
    return next(iter(unique.values())) if len(unique) == 1 else None


_REACH_CACHE: tuple[tuple, _Reachability] | None = None


def reachability_index(
    targets: list, entry_rows: list[dict],
    function_cells: dict[tuple[str, str], set[str]] | None = None,
) -> _Reachability:
    """Build (once per distinct source/target/evidence set) the call graph."""
    global _REACH_CACHE
    function_cells = function_cells or {}
    signature = (
        id(_SOURCE_MODELS),
        _SOURCE_GENERATION,
        tuple(
            (target.entry, target.tier, target.file, target.start_line,
             target.end_line, target.publicly_reachable, len(target.call_sites))
            for target in targets
        ),
        tuple(
            (
                row["entry"],
                tuple(row.get("executed_by") or ()),
                tuple(tuple(cells) for cells in row.get("call_site_cells") or ()),
            )
            for row in entry_rows
        ),
        tuple(
            (path, file, tuple(sorted(cells)))
            for (path, file), cells in sorted(function_cells.items())
        ),
    )
    cached = _REACH_CACHE
    if cached is not None and cached[0] == signature:
        return cached[1]
    functions = {
        file: _source_functions(file, model)
        for file, model in _SOURCE_MODELS.items()
    }
    owner = {}
    nested: set[tuple[str, int]] = set()
    for file, table in functions.items():
        owner.update(
            ((file, index), function)
            for index, function in _owner_map(file, table).items()
        )
        nested.update((file, key) for key in _nested_functions(table))
    by_module: dict[tuple[str, str], list[_Function]] = {}
    for table in functions.values():
        for function in table:
            by_module.setdefault((function.module, function.name), []).append(function)
    imports: dict[tuple[str, str], dict[str, str]] = {}
    for file, model in _SOURCE_MODELS.items():
        imports.update(_source_imports(file, model))
    reach = _Reachability(
        functions=functions,
        owner=owner,
        nested=nested,
        by_module=by_module,
        imports=imports,
        callees={},
        witness={},
        entry_spans={},
        entry_cells={},
        line_offsets={},
        token_starts={},
    )
    for file, model in _SOURCE_MODELS.items():
        reach.line_offsets[file] = _line_offsets(model.source)
        reach.token_starts[file] = [token.start for token in model.tokens]
    _link_calls(reach, targets)
    _seed_entry_witnesses(reach, targets, entry_rows)
    _propagate(reach)
    for (path, file), cells in function_cells.items():
        module, separator, name = path.rpartition("::")
        candidates = reach.by_module.get((module, name), ()) if separator else ()
        if len(candidates) == 1 and candidates[0].file == file:
            reach.witness.setdefault(candidates[0].key, set()).update(cells)
            reach.direct_witness.setdefault(candidates[0].key, set()).update(cells)
    _REACH_CACHE = (signature, reach)
    return reach


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _link_calls(reach: _Reachability, targets: list) -> None:
    """Direct-call edges between resolved function bodies.

    Witnesses flow along direct calls from callers to callees, including
    calls inside match arms and branches: a helper behind a peel or dispatch
    arm is credited by the cells that ran its caller. Registry entries are
    excluded from that propagation — their cells are measured directly, and
    crediting them from a caller would pool one dispatch arm's execution
    into a sibling arm that never ran at the session's tier.
    """
    entry_bodies = {
        body.key
        for body in (_entry_function(reach, target) for target in targets)
        if body is not None
    }
    for file, model in _SOURCE_MODELS.items():
        tokens = model.tokens
        table = reach.functions.get(file) or []
        owners = {
            index: function
            for index, function in _owner_map(file, table).items()
        }
        for index, token in enumerate(tokens):
            if token.kind != "ident":
                continue
            function = owners.get(index)
            if function is None or index <= function.open or index > function.close:
                continue
            if index in model.attribute_tokens or index in model.generic_tokens:
                continue
            if _in_use_item(tokens, index):
                continue
            call = _call_path(tokens, index)
            if call is None:
                continue
            segments, _ = call
            callee = _resolve_call(reach, file, function.module, segments)
            if callee is None:
                continue
            if callee.key in entry_bodies:
                reach.entry_calls.setdefault(function.key, set()).add(callee.key)
                continue
            reach.callees.setdefault(function.key, set()).add(callee.key)


def _seed_entry_witnesses(
    reach: _Reachability, targets: list, entry_rows: list[dict]
) -> None:
    """Entry executions witness their body and each call site's caller."""
    rows = {row["entry"]: row for row in entry_rows}
    spans: dict[str, list[tuple[int, int, str]]] = {}
    for target in targets:
        spans.setdefault(target.file, []).append(
            (target.start_line, target.end_line, target.entry)
        )
    reach.entry_spans = spans
    for target in targets:
        row = rows.get(target.entry)
        if row is None:
            continue
        cells = set(row.get("executed_by") or ())
        if not cells:
            continue
        reach.entry_cells[target.entry] = cells
        body = _entry_function(reach, target)
        if body is not None:
            reach.witness.setdefault(body.key, set()).update(cells)
            reach.direct_witness.setdefault(body.key, set()).update(cells)
        per_site = row.get("call_site_cells") or []
        for position, site in enumerate(target.call_sites):
            if position >= len(per_site):
                continue
            site_cells = set(per_site[position])
            if not site_cells:
                continue
            if isinstance(site, MacroCallSite):
                where = (site.invocation_file, site.invocation_line, site.invocation_column)
            else:
                where = (site[0], site[1], site[2])
            caller = reach.owner_at(*where)
            if caller is not None:
                reach.witness.setdefault(caller.key, set()).update(site_cells)
                reach.direct_witness.setdefault(caller.key, set()).update(site_cells)


def _entry_function(
    reach: _Reachability, target
) -> _Function | None:
    """The top-level function one registry entry span names, if unique.

    The kernel registry binds each arcane entry to its `#[archmage::arcane]`
    attribute line, one line above the `fn` keyword; other spans cover the
    body. Both shapes must name exactly one top-level function, or the
    entry's cells stay unseeded rather than land on a neighbour.
    """
    table = reach.functions.get(target.file) or []
    matches = []
    for function in table:
        if function.key in reach.nested:
            continue
        start, end = _function_lines(reach, target.file, function)
        if start == target.end_line + 1 or (
            target.start_line <= start and end <= target.end_line
        ):
            matches.append(function)
    return matches[0] if len(matches) == 1 else None


def _function_lines(
    reach: _Reachability, file: str, function: _Function
) -> tuple[int, int]:
    """The 1-based first and last lines of one function body."""
    model = _SOURCE_MODELS[file]
    offsets = reach.line_offsets[file]
    start = max(1, bisect.bisect_right(offsets, model.tokens[function.decl].start))
    end = max(1, bisect.bisect_right(offsets, model.tokens[function.close].end - 1))
    return start, end


def _propagate(reach: _Reachability) -> None:
    """Serving-route fixpoint over resolved direct-call edges.

    Caller witnesses identify potential serving routes, not branch coverage;
    exception validation separately checks each cell's operation and geometry.
    Reverse credit comes only from a measured entry call site, never blanket
    execution of a callee shared by several callers.
    """
    frontier = set(reach.witness)
    while frontier:
        key = frontier.pop()
        cells = reach.witness.get(key, set())
        if not cells:
            continue
        for callee in reach.callees.get(key, set()):
            known = reach.witness.setdefault(callee, set())
            if not cells <= known:
                known.update(cells)
                frontier.add(callee)


def _serving_cells(
    item_key: str, row: dict, reach: _Reachability, *, measured_only: bool = False
) -> set[str]:
    """The executed bench cells that demonstrably reach this item's code.

    The evidence is this session's registry attribution and the direct-call
    graph: entries whose definition span contains one of the item's sites,
    plus every function whose body contains a site and that the entries'
    witnesses reach through resolved direct calls. An empty set means the
    map contains no measured tie between the item and any bench cell; an
    unresolved helper therefore constrains every declared cell instead of
    borrowing a same-family credit.

    `measured_only` excludes inferred callees when judging whether stronger-tier
    code actually ran below its declared tier.
    """
    sites = [site for site in row.get("sites", []) if isinstance(site.get("file"), str)]
    served: set[str] = set()
    for site in sites:
        line = site.get("line")
        if not isinstance(line, int):
            continue
        if measured_only and _arm_tier_near(site["file"], line) is not None:
            # A whole-function counter does not prove which dispatch arm ran.
            # Positive site counters are handled before serving-route inference.
            continue
        for start, end, entry in reach.entry_spans.get(site["file"], ()):
            if start <= line <= end:
                served.update(reach.entry_cells.get(entry, ()))
        function = reach.owner_at(site["file"], line, site.get("column") or 1)
        if function is not None:
            witnesses = reach.direct_witness if measured_only else reach.witness
            served.update(witnesses.get(function.key, set()))
    return served


def _serving_via_entry(
    key: str, row: dict, proof: dict, targets: list, reach: _Reachability
) -> set[str]:
    """Witness an opaque nested entry through an explicitly declared direct caller.

    This is a manual source-proof exception, not execution credit for the
    callee. The declaration's reason establishes the branch and argument
    geometry; compiler evidence must still prove the named parent entry ran.
    """
    entry = proof["entry"]
    matches = [target for target in targets if target.entry == entry and target.publicly_reachable]
    caller = _entry_function(reach, matches[0]) if len(matches) == 1 else None
    owners = {
        owner.key: owner
        for site in row.get("sites", [])
        if isinstance(site.get("file"), str) and isinstance(site.get("line"), int)
        for owner in [reach.owner_at(site["file"], site["line"], site.get("column") or 1)]
        if owner is not None
    }
    matched = set(owners).intersection(reach.entry_calls.get(caller.key, ())) if caller else set()
    if not matched:
        raise CoverageMapError(
            f"exceptions: {key}: via_entry {entry} is not a direct source caller "
            "of an owning registry entry"
        )
    required_sources = {caller.file} | {owners[key].file for key in matched}
    if not required_sources.issubset(proof["sources"]):
        raise CoverageMapError(
            f"exceptions: {key}: via_entry fingerprints must bind the caller and owning sources"
        )
    for file, expected in proof["sources"].items():
        model = _SOURCE_MODELS.get(file)
        if model is None or hashlib.sha256(model.source.encode("utf-8")).hexdigest() != expected:
            raise CoverageMapError(
                f"exceptions: {key}: via_entry source fingerprint mismatch for {file}"
            )
    return set(reach.entry_cells.get(entry, ()))


def _requires_sides(key: str, item_class: str, row: dict) -> bool:
    """Whether a declaration must cover both sides of the item's boundary.

    Comparison keys and unmapped branches are boundaries by construction;
    routing thresholds select between two routes by a comparison. Tuning
    parameters without an inline comparison are schedule selectors whose
    evidence is the cells that run them at their actual values.
    """
    return "::cmp:" in key or item_class == "unmapped_branch" or (
        row.get("class") == "routing_threshold" or row.get("comparison") is not None
    )


def _schedule_floor(row: dict) -> tuple[str, int] | None:
    """The `(geometry field, schedule width)` a never-compared parameter needs.

    A tuning parameter judged at its used schedule values requires at least
    one declared cell whose geometry executes a complete scheduled group or
    block. Fields with no bench geometry (delay depths, lane-group widths,
    alignment exponents) impose no floor.
    """
    if row.get("class") != "tuning_parameter" or "::cmp:" in row["key"]:
        return None
    value = row.get("value")
    if not isinstance(value, int):
        return None
    field = _axis_field(row.get("unit") or "other", row.get("axis") or "")
    if field is None:
        return None
    return field, value


def straddling_cells(item: dict, cells: list[dict], allowed: set[str] | None = None) -> list[str]:
    """Cells on both sides of an item's boundary, or the first cells otherwise."""

    axis_field = _axis_field(item.get("unit", "other"), item.get("axis", ""))
    if axis_field is None:
        pairs = [
            (cell["id"], None) for cell in cells if isinstance(cell.get("id"), str)
        ]
    else:
        pairs = [
            (cell["id"], _axis_value(axis_field, cell))
            for cell in cells
            if isinstance(cell.get("id"), str)
        ]
    if allowed is not None:
        pairs = [pair for pair in pairs if pair[0] in allowed]
    numeric = [(cell, value) for cell, value in pairs if isinstance(value, int)]
    comparison = item.get("comparison")
    if comparison is not None and numeric:
        axis = item.get("axis", "")
        constant = item.get("value")
        true_side = [
            pair
            for pair in numeric
            if isinstance(constant, int)
            and _satisfies(_axis_quantity(axis, axis_field, pair[1]), comparison, constant)
        ]
        false_side = [
            pair
            for pair in numeric
            if not isinstance(constant, int)
            or not _satisfies(_axis_quantity(axis, axis_field, pair[1]), comparison, constant)
        ]
        chosen = []
        if true_side:
            chosen.append(min(true_side, key=lambda pair: abs(pair[1] - constant))[0])
        if false_side:
            chosen.append(min(false_side, key=lambda pair: abs(pair[1] - constant))[0])
        if chosen:
            return chosen
    cutoff = item.get("value")
    if axis_field is not None and isinstance(cutoff, int):
        below = [entry for entry in numeric if entry[1] < cutoff]
        above = [entry for entry in numeric if entry[1] > cutoff]
        chosen = []
        if below:
            chosen.append(min(below, key=lambda entry: entry[1])[0])
        if above:
            chosen.append(max(above, key=lambda entry: entry[1])[0])
        if chosen:
            return chosen
    return [cell for cell, _ in pairs[:2]]


def validate_exceptions(
    exceptions: dict[str, dict],
    targets: list,
    entry_rows: list[dict],
    thresholds: list[dict],
    cells: list[dict],
    requested_tier: str,
    function_cells: dict[tuple[str, str], set[str]] | None = None,
) -> list[dict]:
    """Fail on stale keys, wrong classes, unproven cells, and one-sided bounds.

    Every declared cell must exist, run a bench operation that actually
    varies the item's axis, and carry measured execution of the entry or
    helper that owns the item's code at this tier. Boundary items must cover
    both sides of their source predicate or cutoff through one operation; a
    never-compared tuning parameter must run its complete schedule in at
    least one declared cell.
    """

    cell_ids = {cell["id"] for cell in cells}
    cell_by_id = {cell["id"]: cell for cell in cells}
    threshold_by_key = {row["key"]: row for row in thresholds}
    reach = reachability_index(targets, entry_rows, function_cells)
    diagnostics: list[dict] = []
    for key, entry in exceptions.items():
        scope = entry.get("tiers")
        if scope and requested_tier not in scope:
            continue
        item_class = entry["class"]
        if item_class == "unreachable":
            if key not in threshold_by_key:
                raise CoverageMapError(
                    f"exceptions: unknown unreachable item {key}"
                )
            continue
        row = threshold_by_key.get(key)
        if row is None:
            raise CoverageMapError(f"exceptions: unknown threshold item {key}")
        if item_class == "unmapped_arcane":
            if row["status"] not in {"unmapped", "partial_mapping"}:
                diagnostics.append(
                    {
                        "key": key,
                        "class": item_class,
                        "status": row["status"],
                        "reason": "stale at this tier: item is not unmapped",
                    }
                )
                continue
        elif item_class == "unmapped_branch":
            if row["status"] != "unexecuted":
                diagnostics.append(
                    {
                        "key": key,
                        "class": item_class,
                        "status": row["status"],
                        "reason": "stale at this tier: no missing branch record",
                    }
                )
                continue
        else:
            raise CoverageMapError(f"exceptions: {key}: unknown class {item_class}")
        ids = _declared_cells(entry, requested_tier)
        missing = [cell for cell in ids if cell not in cell_ids]
        if missing:
            raise CoverageMapError(
                f"exceptions: {key}: unknown bench cells {missing[:4]}"
            )
        served = _serving_cells(key, row, reach)
        via = (entry.get("via_entry") or {}).get(requested_tier)
        if via is not None:
            served = _serving_via_entry(key, row, via, targets, reach)
        outside = [cell for cell in ids if cell not in served]
        if outside:
            raise CoverageMapError(
                f"exceptions: {key}: cells {outside[:4]} have no measured "
                "execution of the entry or helper that owns this item at "
                "this tier"
            )
        axis_field = _axis_field(row.get("unit") or "other", row.get("axis") or "")
        if axis_field is not None:
            for cell_id in ids:
                if _axis_value(axis_field, cell_by_id[cell_id]) is None:
                    raise CoverageMapError(
                        f"exceptions: {key}: cell {cell_id} has no valid "
                        f"{axis_field} geometry"
                    )
                operation = cell_by_id[cell_id]["operation"]
                values = {
                    _axis_value(axis_field, cell)
                    for cell in cells
                    if cell.get("operation") == operation
                }
                values.discard(None)
                if len(values) < 2:
                    raise CoverageMapError(
                        f"exceptions: {key}: cell {cell_id} runs operation "
                        f"{operation}, which never varies {axis_field} in the "
                        "bench inventory and cannot carry a "
                        f"{axis_field} boundary"
                    )
        values = [
            _axis_value(axis_field, cell_by_id[cell_id]) for cell_id in ids
        ] if axis_field is not None else []
        floor = _schedule_floor(row)
        if floor is not None:
            field, width = floor
            complete = [
                cell_id
                for cell_id in ids
                if isinstance(_axis_value(field, cell_by_id[cell_id]), int)
                and _axis_value(field, cell_by_id[cell_id]) >= width
            ]
            if not complete:
                raise CoverageMapError(
                    f"exceptions: {key}: no declared cell executes a complete "
                    f"{width}-{row.get('unit')} {row.get('axis')} schedule; "
                    "the partial tiles do not excuse a missing full block"
                )
        if _requires_sides(key, item_class, row):
            if axis_field is None:
                raise CoverageMapError(
                    f"exceptions: {key}: boundary item has no bench axis for "
                    f"unit {row.get('unit')!r} axis {row.get('axis')!r}"
                )
            comparison = row.get("comparison")
            if comparison is not None and comparison not in {"<", "<=", "=="}:
                raise CoverageMapError(
                    f"exceptions: {key}: comparison {comparison!r} is not a "
                    "source predicate the scanner produces"
                )
            if comparison is not None:
                row_axis = row.get("axis") or ""
                constant = row.get("value")
                if not isinstance(constant, int):
                    raise CoverageMapError(
                        f"exceptions: {key}: comparison item has no integer "
                        "cutoff value"
                    )
                covered = any(
                    isinstance(value, int)
                    and _satisfies(
                        _axis_quantity(row_axis, axis_field, value),
                        comparison,
                        constant,
                    )
                    for value in values
                )
                uncovered = any(
                    isinstance(value, int)
                    and not _satisfies(
                        _axis_quantity(row_axis, axis_field, value),
                        comparison,
                        constant,
                    )
                    for value in values
                )
                boundary = (
                    f"`{comparison} {constant}` on {row_axis or axis_field}"
                )
            else:
                cutoff = row.get("value")
                covered = any(
                    isinstance(value, int) and value < cutoff for value in values
                )
                uncovered = any(
                    isinstance(value, int) and value > cutoff for value in values
                )
                boundary = f"the {cutoff}-{row.get('unit')} cutoff"
            operations = sorted({cell_by_id[cell_id]["operation"] for cell_id in ids})
            if len(operations) > 1:
                raise CoverageMapError(
                    f"exceptions: {key}: straddling_cells mix operations "
                    f"{operations}; both sides of a boundary run one operation"
                )
            if not covered or not uncovered:
                raise CoverageMapError(
                    f"exceptions: {key}: straddling_cells do not cover both "
                    f"sides of {boundary}"
                )
    return diagnostics


def apply_exceptions(
    exceptions: dict[str, dict], thresholds: list[dict], requested_tier: str
) -> None:
    for row in thresholds:
        entry = exceptions.get(row["key"])
        if entry is None:
            continue
        scope = entry.get("tiers")
        if scope and requested_tier not in scope:
            continue
        if entry["class"] == "unreachable":
            row["status"] = "excused"
            row["reason"] = entry["reason"]
            row["exception_class"] = entry["class"]
        elif entry["class"] == "unmapped_arcane" and row["status"] in {
            "unmapped",
            "partial_mapping",
        }:
            row["status"] = "excused"
            row["reason"] = entry["reason"]
            row["exception_class"] = entry["class"]
        elif entry["class"] == "unmapped_branch" and row["status"] == "unexecuted":
            row["status"] = "excused"
            row["reason"] = entry["reason"]
            row["exception_class"] = entry["class"]


def failures(
    entries: list[dict], thresholds: list[dict]
) -> list[dict]:
    """The rule: unexecuted required entries and one-sided threshold branches."""

    result = []
    for row in entries:
        if row["status"] == "unexecuted":
            result.append(
                {
                    "kind": "entry",
                    "item": row["entry"],
                    "reason": row.get("reason") or
                    "publicly reachable entry at the requested tier executed in no cell",
                }
            )
    for row in thresholds:
        if row["status"] in {"unmapped", "partial_mapping", "ambiguous"}:
            result.append(
                {
                    "kind": "threshold",
                    "item": row["key"],
                    "reason": row["reason"],
                }
            )
        elif row.get("required", True) and row["status"] in {"one_sided", "unexecuted"}:
            result.append(
                {
                    "kind": "threshold",
                    "item": row["key"],
                    "reason": row["reason"],
                }
            )
    return result


def mapping_limitations(
    entries: list[dict], thresholds: list[dict], requested_tier: str
) -> list[dict]:
    """Reachable items the map could not measure, kept apart from failures."""

    result = []
    for row in entries:
        if row["status"] != "not_required" or row["evidence"] != "none":
            continue
        if row["rank"] > TIER_RANK[requested_tier]:
            continue
        if not row["publicly_reachable"]:
            reason = "private helper not reachable through internals"
        else:
            reason = "lower-tier entry not reached at this tier; its own tier session covers it"
        result.append({"kind": "entry", "item": row["entry"], "reason": reason})
    for row in thresholds:
        if row["status"] in {"unmapped", "partial_mapping"}:
            result.append(
                {
                    "kind": "threshold",
                    "item": row["key"],
                    "reason": row["reason"],
                }
            )
        elif row["status"] == "tier_not_required":
            hints = sorted(
                {
                    site["tier_hint"]
                    for site in row["sites"]
                    if site.get("tier_hint")
                }
            )
            result.append(
                {
                    "kind": "threshold",
                    "item": row["key"],
                    "reason": (
                        "site tier " + "/".join(hints) + " is not the requested tier; "
                        "that tier's session measures it" if hints else "tier not required"
                    ),
                }
            )
        elif not row.get("required", True) and row["status"] in {"one_sided", "unexecuted"}:
            hints = sorted(
                {
                    site["tier_hint"]
                    for site in row["sites"]
                    if site.get("tier_hint")
                }
            )
            result.append(
                {
                    "kind": "threshold",
                    "item": row["key"],
                    "reason": (
                        "site tier " + "/".join(hints) + " is not the requested tier; "
                        "that tier's session measures it"
                    ),
                }
            )
    return result


# ---------------------------------------------------------------------------
# Build and per-cell instrumentation
# ---------------------------------------------------------------------------


def llvm_tools(toolchain: str) -> tuple[Path, Path]:
    try:
        sysroot = subprocess.check_output(
            ["rustc", f"+{toolchain}", "--print", "sysroot"], text=True
        ).strip()
        version = subprocess.check_output(
            ["rustc", f"+{toolchain}", "-vV"], text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CoverageMapError(f"cannot query toolchain {toolchain}: {error}")
    host = None
    for line in version.splitlines():
        if line.startswith("host: "):
            host = line.removeprefix("host: ").strip()
    if host is None:
        raise CoverageMapError(f"rustc +{toolchain} -vV has no host line")
    bindir = Path(sysroot) / "lib" / "rustlib" / host / "bin"
    candidates = [
        (bindir / "llvm-cov", bindir / "llvm-profdata"),
        (Path(shutil.which("llvm-cov") or ""), Path(shutil.which("llvm-profdata") or "")),
    ]
    for cov, profdata in candidates:
        if cov.is_file() and profdata.is_file():
            return cov, profdata
    raise CoverageMapError(
        "llvm-cov and llvm-profdata are missing; install the llvm-tools component "
        f"for {toolchain} or put them in PATH"
    )


def build_bench(
    root: Path,
    toolchain: str,
    bench: str,
    target_dir: Path,
    log_path: Path,
    *,
    features: list[str] | None = None,
    no_default_features: bool = False,
) -> Path:
    env = os.environ.copy()
    env.pop("SIMD_BACKEND", None)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    # Instrumented build scripts write profiles too; keep them out of the
    # crate root.
    env["LLVM_PROFILE_FILE"] = str(target_dir / "build-profiles" / "%p-%m.profraw")
    previous = env.get("RUSTFLAGS", "").strip()
    env["RUSTFLAGS"] = " ".join(
        ([previous] if previous else []) + list(COVERAGE_RUSTFLAGS)
    )
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["CARGO_PROFILE_BENCH_OPT_LEVEL"] = "2"
    env["CARGO_PROFILE_BENCH_LTO"] = "off"
    env["CARGO_PROFILE_BENCH_CODEGEN_UNITS"] = "16"
    env["CARGO_PROFILE_BENCH_DEBUG"] = "1"
    command = [
        "cargo",
        f"+{toolchain}",
        "bench",
        "--locked",
        "--bench",
        bench,
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    if features is None or "all" in features:
        command.append("--all-features")
    elif features:
        command.extend(["--features", ",".join(features)])
    if no_default_features:
        command.append("--no-default-features")
    process = subprocess.run(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.write_text(
        "command: " + " ".join(command) + "\n"
        + "RUSTFLAGS=" + env["RUSTFLAGS"] + "\n"
        + process.stdout
        + process.stderr,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise CoverageMapError(f"instrumented bench build failed; see {log_path}")
    executables = set()
    for line in process.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("reason") != "compiler-artifact":
            continue
        target = record.get("target", {})
        if target.get("name") == bench and record.get("executable"):
            executables.add(record["executable"])
    if len(executables) != 1:
        raise CoverageMapError(
            f"expected one instrumented {bench} executable, found {len(executables)}; see {log_path}"
        )
    return Path(executables.pop())


def base_run_environment(tier: str, profile: Path | None) -> dict:
    env = os.environ.copy()
    env.pop("RUSTFLAGS", None)
    env["SIMD_BACKEND"] = tier
    env["FGF_REQUIRE_TIER"] = tier
    if profile is not None:
        env["LLVM_PROFILE_FILE"] = str(profile)
    else:
        # Uninstrumented runs of an instrumented binary discard their profile
        # instead of writing `default_*.profraw` into the working directory.
        env["LLVM_PROFILE_FILE"] = os.devnull
    return env


def parse_bench_header(stdout: str) -> tuple[str, list[tuple[str, str]]]:
    backend = None
    fields = []
    for line in stdout.splitlines():
        if line.startswith("BACKEND\t"):
            backend = line.split("\t", 1)[1]
        elif line.startswith("FIELD_BACKEND\t"):
            parts = line.split("\t")
            if len(parts) >= 3:
                fields.append((parts[1], parts[2]))
    if backend is None:
        raise CoverageMapError("bench output has no BACKEND line")
    return backend, fields


def run_inventory(
    root: Path, binary: Path, tier: str, scratch: Path
) -> list[dict]:
    profile = scratch / "inventory.profraw"
    process = subprocess.run(
        [str(binary), "--inventory"],
        cwd=root,
        env=base_run_environment(tier, profile),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise CoverageMapError("bench --inventory failed")
    backend, _ = parse_bench_header(process.stdout)
    if backend != tier:
        raise CoverageMapError(
            f"process backend is {backend!r}, not the requested {tier!r}"
        )
    cells = []
    widths: dict[str, int] = {}
    for line in process.stdout.splitlines():
        if line.startswith("FIELD_ELEMENT_BYTES\t"):
            parts = line.split("\t")
            try:
                width = int(parts[2]) if len(parts) == 3 else 0
            except ValueError:
                width = 0
            if width <= 0 or parts[1] in widths:
                raise CoverageMapError("malformed or duplicate FIELD_ELEMENT_BYTES record")
            widths[parts[1]] = width
        if line.startswith("CELL\t"):
            try:
                cells.append(json.loads(line.split("\t", 1)[1]))
            except json.JSONDecodeError as error:
                raise CoverageMapError(f"malformed CELL record: {error}")
    if not cells:
        raise CoverageMapError("bench --inventory returned no cells")
    for cell in cells:
        if cell["field"] in widths:
            cell["element_bytes"] = widths[cell["field"]]
    return cells


_WORKER: dict = {}


def _worker_init(spec: dict) -> None:
    _WORKER.clear()
    _WORKER.update(spec)


def _export_profile(spec: dict, profile: Path, merged: Path, label: str) -> str:
    merge = subprocess.run(
        [str(spec["profdata"]), "merge", "-sparse", "-o", str(merged), str(profile)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if merge.returncode != 0:
        raise CoverageMapError(
            f"{label}: llvm-profdata merge failed: {merge.stderr.strip()}"
        )
    exported = subprocess.run(
        [str(spec["llvm_cov"]), "export", "--format=text",
         f"--instr-profile={merged}", str(spec["binary"])],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if exported.returncode != 0:
        raise CoverageMapError(
            f"{label}: llvm-cov export failed: {exported.stderr.strip()}"
        )
    return exported.stdout


def _run_one_cell(index: int) -> dict:
    spec = _WORKER
    cell_id = spec["cells"][index]
    root: Path = spec["root"]
    scratch: Path = spec["scratch"]
    profile = scratch / f"cell-{index:04d}.profraw"
    command = [
        str(spec["binary"]),
        "--check",
        "--cells",
        str(spec["filters"][index]),
    ]
    process = subprocess.run(
        command,
        cwd=root,
        env=base_run_environment(spec["tier"], profile),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise CoverageMapError(
            f"cell {cell_id}: bench --check failed: {process.stderr.strip()}"
        )
    backend, _ = parse_bench_header(process.stdout)
    if backend != spec["tier"]:
        raise CoverageMapError(
            f"cell {cell_id}: process backend {backend!r}, not {spec['tier']!r}"
        )
    output_lines = [line for line in process.stdout.splitlines() if line.startswith("OUTPUT\t")]
    if len(output_lines) != 1 or output_lines[0].split("\t")[1] != cell_id:
        raise CoverageMapError(f"cell {cell_id}: expected exactly one OUTPUT record")
    merged = scratch / f"cell-{index:04d}.profdata"
    exported = scratch / f"cell-{index:04d}.json"
    exported_text = _export_profile(spec, profile, merged, f"cell {cell_id}")
    if spec["keep_exports"]:
        exported.write_text(exported_text, encoding="utf-8")
    try:
        document = json.loads(exported_text)
    except json.JSONDecodeError as error:
        raise CoverageMapError(f"cell {cell_id}: llvm-cov export JSON invalid: {error}")
    evidence = analyze_export(document, root, spec["entries"], spec["sites"])
    if not spec["keep_exports"]:
        profile.unlink(missing_ok=True)
        merged.unlink(missing_ok=True)
    return {"cell": cell_id, "evidence": evidence}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _short(value: str, limit: int = 44) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _cell_list(cells: list[str], limit: int = 4) -> str:
    if not cells:
        return "-"
    shown = ", ".join(f"`{_short(cell, 34)}`" for cell in cells[:limit])
    if len(cells) > limit:
        shown += f" +{len(cells) - limit} more"
    return shown


def render_markdown(report: dict) -> str:
    lines = [
        f"# Coverage map: {report['tier']}",
        "",
        f"- host: `{report['host']}`",
        f"- toolchain: `{report['toolchain']}`",
        f"- cells: {report['cells']}",
        f"- runtime: {report['runtime_seconds']:.1f} s",
        f"- instrumented binary: `{report['binary']}`",
        f"- required entries executed: {report.get('required_entries_executed', '?')}/{report.get('required_entries', '?')}",
        f"- entries executed at tiers at or below the request: "
        f"{sum(1 for row in report['entries'] if row['status'] == 'executed')}/"
        f"{sum(1 for row in report['entries'] if row.get('rank', 99) <= report.get('requested_tier_rank', 0))}",
        "",
        "Every cell ran `--check` in its own process with its own profile, so a",
        "region is attributed to exactly the cells that reached it. Entry evidence",
        "is a function-level count on the entry wrapper. The pinned compiler emits",
        "no coverage mapping for bodies generated by `#[archmage::arcane]` and",
        "`#[archmage::rite]`; for those entries the map uses the nearest mapped",
        "caller region and labels the evidence `call_site`. Threshold branches",
        "inside such bodies have no compiler mapping and are reported as",
        "`unmapped` mapping limitations, never as passes.",
        "",
        "## Kernel registry entries",
        "",
        "| entry | tier | required | status | evidence | cells | count |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in report["entries"]:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                _short(row["entry"], 80),
                row["tier"],
                "yes" if row["required"] else "no",
                row["status"],
                row["evidence"],
                _cell_list(row["executed_by"]),
                row["count"],
            )
        )
    lines += [
        "",
        "## Thresholds and tuning parameters",
        "",
        "| item | class | value | status | sites | taken true / false | cells |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in report["thresholds"]:
        true_count = sum(site.get("true", 0) for site in row["sites"])
        false_count = sum(site.get("false", 0) for site in row["sites"])
        cells = sorted(
            {
                cell
                for site in row["sites"]
                for cell in site.get("true_cells", []) + site.get("false_cells", []) + site.get("executed_cells", [])
            }
        )
        lines.append(
            "| `{}` | {} | {} {} | {} | {} | {} / {} | {} |".format(
                _short(row["key"], 72),
                row["class"],
                row["value"],
                row["unit"],
                row["status"],
                len(row["sites"]),
                true_count,
                false_count,
                _cell_list(cells),
            )
        )
    failing_sites = [
        (row, site)
        for row in report["thresholds"]
        for site in row["sites"]
        if site.get("status") not in {"both_ways", "exercised"}
    ]
    lines += ["", "### One-sided or unexecuted sites", ""]
    if failing_sites:
        lines.append("| item | site | role | tier | status | true | false | executed |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | --- |")
        for row, site in failing_sites:
            lines.append(
                "| `{}` | {}:{}:{} | {} | {} | {} | {} | {} | {} |".format(
                    _short(row["key"], 64),
                    site.get("file", "-"),
                    site.get("line", "-"),
                    site.get("column", "-"),
                    site.get("role", "-"),
                    site.get("tier_hint") or "-",
                    site.get("status", "-"),
                    site.get("true", "-"),
                    site.get("false", "-"),
                    _cell_list(
                        site.get("true_cells", [])
                        + site.get("false_cells", [])
                        + site.get("executed_cells", [])
                    ),
                )
            )
    else:
        lines.append("- none")
    lines += ["", "## Unreachable items", ""]
    unreachable = report.get("unreachable") or []
    if unreachable:
        for item in unreachable:
            lines.append(f"- `{item['item']}` — {item['reason']}")
    else:
        lines.append("- none")
    lines += ["", "## Exceptions", ""]
    exceptions = [
        entry for entry in (report.get("exceptions") or [])
        if not entry.get("tiers") or report["tier"] in entry["tiers"]
    ]
    if exceptions:
        for entry in exceptions:
            cells = _declared_cells(entry, report["tier"])
            suffix = f" (straddling cells: {_cell_list(cells, 6)})" if cells else ""
            via = (entry.get("via_entry") or {}).get(report["tier"])
            if via is not None:
                suffix += f" (fingerprinted manual source proof via `{via['entry']}`)"
            lines.append(
                f"- `{_short(entry['key'], 90)}` [{entry['class']}] — {entry['reason']}{suffix}"
            )
    else:
        lines.append("- none")
    lines += ["", "## Mapping limitations", ""]
    limitations = report.get("mapping_limitations") or []
    if limitations:
        for item in limitations:
            lines.append(f"- `{_short(item['item'], 90)}` — {item['reason']}")
    else:
        lines.append("- none")
    lines += ["", "## Gaps", ""]
    gaps = report.get("gaps") or []
    if gaps:
        for item in gaps:
            lines.append(f"- `{_short(item['item'], 90)}` ({item['kind']}) — {item['reason']}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Method",
        "",
        f"- build: `cargo +{report['toolchain']} bench --all-features --bench {report['bench']}` "
        "with `-Cinstrument-coverage -Zcoverage-options=branch`; LTO off, "
        "`-Zinline-llvm=no`, bench profile debug info.",
        "- per-cell profile: one process per cell, `LLVM_PROFILE_FILE` per cell, "
        "`llvm-profdata merge -sparse`, `llvm-cov export -format=text`.",
        "- attribution: an entry executes in a cell when its wrapper span has a "
        "region with a positive count, or (fallback) a mapped region containing one "
        "of its call sites has a positive count. A branch site is taken both ways "
        "when the summed true and false counts across all instantiations are both "
        "positive. A selector/value site is exercised when its mapped region count "
        "is positive.",
        "- exit: non-zero for an unexecuted required entry, a one-sided required "
        "branch, or an unmapped/ambiguous threshold item. Unmapped exceptions "
        "require a reason and straddling cells; ambiguous attribution cannot be excused.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def aggregate_reports(
    inputs: list[Path],
    exceptions_path: Path,
    out_json: Path,
    out_markdown: Path,
) -> int:
    """Merge per-tier reports and decide the cross-host acceptance rule."""

    reports = [load_json(path) for path in inputs]
    if not all(isinstance(report, dict) for report in reports):
        raise CoverageMapError("aggregate inputs are not coverage-map reports")
    sessions = [
        {"host": report.get("host"), "tier": report.get("tier"), "path": str(path)}
        for report, path in zip(reports, inputs)
    ]
    entry_executed = {}
    entry_tier = {}
    entry_required_tier_seen = set()
    threshold_rows: dict[str, list[dict]] = defaultdict(list)
    exception_applied = set()
    for report in reports:
        tier = report.get("tier")
        for row in report.get("entries", []):
            entry_tier[row["entry"]] = TIER_RANK[row["tier"]]
            if row["tier"] == tier:
                entry_required_tier_seen.add(row["entry"])
                if row["status"] == "executed":
                    entry_executed[row["entry"]] = entry_executed.get(row["entry"], True)
                else:
                    entry_executed[row["entry"]] = False
        for row in report.get("thresholds", []):
            threshold_rows[row["key"]].append({"tier": tier, **row})
            if row.get("status") == "excused":
                exception_applied.add(row["key"])
    gaps = []
    for entry, rank in entry_tier.items():
        if entry not in entry_required_tier_seen:
            gaps.append(
                {"kind": "entry", "item": entry, "reason": "no session at the entry's own tier"}
            )
        elif not entry_executed.get(entry):
            gaps.append(
                {"kind": "entry", "item": entry, "reason": "not executed at its own tier"}
            )
    threshold_summary = {}
    for key, rows in threshold_rows.items():
        required = [row for row in rows if row["status"] != "tier_not_required"]
        if not required:
            threshold_summary[key] = "gap"
            gaps.append({
                "kind": "threshold", "item": key,
                "reason": "no session measures this threshold at a required tier",
            })
        elif all(row["status"] in {"both_ways", "exercised", "excused"} for row in required):
            threshold_summary[key] = (
                "excused" if any(row["status"] == "excused" for row in required) else "passed"
            )
        else:
            threshold_summary[key] = "gap"
            gaps.append(
                {
                    "kind": "threshold",
                    "item": key,
                    "reason": "not both ways at a tier where it is required: "
                    + ", ".join(f"{row['tier']}={row['status']}" for row in required[:6]),
                }
            )
    exceptions = load_exceptions(exceptions_path)
    stale_exceptions = []
    valid_exceptions = []
    for key, entry in exceptions.items():
        scope = entry.get("tiers")
        applies = scope is None or any(session["tier"] in scope for session in sessions)
        if not applies:
            stale_exceptions.append(key)
        elif entry["class"] == "unreachable":
            valid_exceptions.append(key)
        elif key in exception_applied:
            valid_exceptions.append(key)
        else:
            stale_exceptions.append(key)
            gaps.append(
                {
                    "kind": "exception",
                    "item": key,
                    "reason": "exception never applies at any recorded tier",
                }
            )
    report = {
        "format": FORMAT + "-aggregate",
        "sessions": sessions,
        "entries": {
            entry: {
                "tier": TIER_ORDER[rank],
                "executed_at_own_tier": bool(entry_executed.get(entry)),
            }
            for entry, rank in sorted(entry_tier.items())
        },
        "thresholds": threshold_summary,
        "exceptions": {
            "valid": sorted(valid_exceptions),
            "stale": sorted(stale_exceptions),
        },
        "gaps": gaps,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Aggregate coverage map", "", f"- sessions: {len(sessions)}"]
    for session in sessions:
        lines.append(f"  - `{session['host']}` / `{session['tier']}`")
    lines += ["", "## Exceptions", ""]
    lines.append(f"- valid: {len(valid_exceptions)}")
    lines.append(f"- stale: {len(stale_exceptions)}")
    lines += ["", "## Gaps", ""]
    for gap in gaps:
        lines.append(f"- `{gap['item']}` ({gap['kind']}) — {gap['reason']}")
    if not gaps:
        lines.append("- none")
    out_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    entry_gaps = sum(1 for gap in gaps if gap["kind"] == "entry")
    threshold_gaps = sum(1 for gap in gaps if gap["kind"] == "threshold")
    exception_gaps = sum(1 for gap in gaps if gap["kind"] == "exception")
    print(
        f"coverage-map aggregate: {len(sessions)} sessions; "
        f"{len(entry_tier) - entry_gaps}/{len(entry_tier)} entries executed at own tier; "
        f"{len(threshold_summary) - threshold_gaps}/{len(threshold_summary)} threshold items passed; "
        f"{len(valid_exceptions)} valid, {len(stale_exceptions)} stale exceptions; "
        f"{len(gaps)} gap(s)"
    )
    return 1 if gaps else 0


def run(arguments: argparse.Namespace) -> int:
    started = time.monotonic()
    if (
        arguments.tier is None
        or arguments.inventory is None
        or arguments.scan is None
        or arguments.toolchain is None
    ):
        raise CoverageMapError("--tier, --inventory, --scan, and --toolchain are required")
    root = Path(arguments.root).resolve()
    target_dir = Path(arguments.target_dir)
    scratch = target_dir / f"{arguments.tier}-scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    scan_module = load_scan_module(Path(arguments.scan))
    inventory_items = load_inventory(Path(arguments.inventory))
    generated_items, positions = scan_module.scan_with_positions(root)
    generated_bytes = scan_module.serialize_json(generated_items).encode("utf-8")
    recorded_bytes = Path(arguments.inventory).read_bytes()
    if hashlib.sha256(generated_bytes).digest() != hashlib.sha256(recorded_bytes).digest():
        raise CoverageMapError(
            "the threshold inventory does not match the current source tree; "
            "rerun the scanner before generating the coverage map"
        )
    host = re.sub(r"[^A-Za-z0-9._-]+", "_", socket.gethostname())
    if arguments.out is None:
        out_json = Path("target/coverage-map") / f"{host}-{arguments.tier}.json"
    else:
        out_json = Path(arguments.out)
    if arguments.summary is None:
        out_markdown = out_json.with_suffix(".md")
    else:
        out_markdown = Path(arguments.summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)

    configure_sources()
    registry = load_registry(Path(arguments.registry))
    private_doc = Path(arguments.registry).with_name(arguments.private_doc)
    spans = load_private_spans(private_doc)
    private_allowlist = set(
        load_json(Path(arguments.registry)).get("private_allowlist", [])  # type: ignore[union-attr]
    )
    models = scan_sources(root, scan_module)
    configure_caller_tiers(models)
    call_sites = discover_call_sites(registry, models)
    entries = build_entry_targets(registry, spans, call_sites, private_allowlist)
    sites, threshold_items = build_site_targets(inventory_items, positions)
    exceptions = load_exceptions(Path(arguments.exceptions))

    llvm_cov, profdata = llvm_tools(arguments.toolchain)
    binary = build_bench(
        root,
        arguments.toolchain,
        arguments.bench,
        target_dir / "build",
        out_json.with_suffix(".build.log"),
    )
    cells = run_inventory(root, binary, arguments.tier, scratch)
    cell_ids = [cell["id"] for cell in cells]
    filters = []
    for index, cell_id in enumerate(cell_ids):
        path = scratch / f"filter-{index:04d}.txt"
        path.write_text(cell_id + "\n", encoding="utf-8")
        filters.append(path)

    spec = {
        "cells": cell_ids,
        "filters": filters,
        "binary": binary,
        "root": root,
        "scratch": scratch,
        "tier": arguments.tier,
        "llvm_cov": llvm_cov,
        "profdata": profdata,
        "keep_exports": arguments.keep,
        "entries": entries,
        "sites": sites,
    }
    startup_text = _export_profile(
        spec, scratch / "inventory.profraw", scratch / "inventory.profdata", "inventory",
    )
    try:
        startup_document = json.loads(startup_text)
    except json.JSONDecodeError as error:
        raise CoverageMapError(f"inventory: llvm-cov export JSON invalid: {error}")
    startup_functions = analyze_export(startup_document, root, [], [])["functions"]
    jobs = arguments.jobs or min(32, os.cpu_count() or 1)
    cell_evidence: list[tuple[str, dict]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs, initializer=_worker_init, initargs=(spec,)
    ) as pool:
        futures = [pool.submit(_run_one_cell, index) for index in range(len(cells))]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cell_evidence.append((result["cell"], result["evidence"]))
    cell_evidence.sort(key=lambda pair: pair[0])

    entry_rows = summarize_entries(entries, cell_evidence, arguments.tier, exceptions)
    function_cells = operation_function_cells(cell_evidence, cells, startup_functions)
    reach = reachability_index(entries, entry_rows, function_cells)
    threshold_rows = summarize_thresholds(
        threshold_items, sites, cell_evidence, exceptions, arguments.tier, reach
    )
    diagnostics = validate_exceptions(
        exceptions, entries, entry_rows, threshold_rows, cells, arguments.tier,
        function_cells,
    )
    if arguments.suggest_exceptions is not None:
        existing = {
            entry["key"]: entry
            for entry in exceptions.values()
            if entry["class"] == "unreachable"
        }
        proposals = dict(existing)
        for row in threshold_rows:
            if row["key"] in exceptions:
                continue
            if row["status"] not in {"unmapped", "partial_mapping"}:
                continue
            cells_for_item = straddling_cells(
                row,
                cells,
                allowed=_serving_cells(row["key"], row, reach),
            )
            if not cells_for_item:
                continue
            proposals[row["key"]] = {
                "key": row["key"],
                "class": "unmapped_arcane",
                "reason": row["reason"],
                "straddling_cells": cells_for_item,
            }
        Path(arguments.suggest_exceptions).write_text(
            json.dumps(
                {"exceptions": [proposals[key] for key in sorted(proposals)]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"coverage-map: proposed {len(proposals)} exceptions in {arguments.suggest_exceptions}")
    apply_exceptions(exceptions, threshold_rows, arguments.tier)
    unreachable = [
        {"item": key, "reason": entry["reason"]}
        for key, entry in exceptions.items()
        if entry["class"] == "unreachable"
    ]
    for row in entry_rows:
        if row["status"] == "unexecuted" and row["evidence"] == "none" and not row["reason"]:
            row["reason"] = (
                "executed in no cell at this tier: no wrapper-level mapping and "
                "no mapped call site (the pinned compiler emits no mapping for "
                "#[archmage::arcane] bodies; a missing bench panel is also possible)"
            )
    gaps = failures(entry_rows, threshold_rows)
    limitations = mapping_limitations(entry_rows, threshold_rows, arguments.tier)
    runtime = time.monotonic() - started
    required = [row for row in entry_rows if row["required"]]
    executed = [row for row in required if row["status"] == "executed"]
    report_required_total = len(required)
    report_required_executed = len(executed)

    report = {
        "format": FORMAT,
        "host": host,
        "tier": arguments.tier,
        "toolchain": arguments.toolchain,
        "bench": arguments.bench,
        "binary": str(binary),
        "llvm_cov": str(llvm_cov),
        "llvm_profdata": str(profdata),
        "rustflags": list(COVERAGE_RUSTFLAGS),
        "inventory": str(Path(arguments.inventory)),
        "scan": str(Path(arguments.scan)),
        "cells": len(cells),
        "startup_functions": startup_functions,
        "runtime_seconds": runtime,
        "entries": entry_rows,
        "thresholds": threshold_rows,
        "unreachable": unreachable,
        "exceptions": [exceptions[key] for key in sorted(exceptions)],
        "exception_diagnostics": diagnostics,
        "mapping_limitations": limitations,
        "gaps": gaps,
        "required_entries": report_required_total,
        "required_entries_executed": report_required_executed,
        "requested_tier_rank": TIER_RANK[arguments.tier],
        "inventory_sha256": hashlib.sha256(recorded_bytes).hexdigest(),
        "scan_sha256": hashlib.sha256(Path(arguments.scan).read_bytes()).hexdigest(),
    }
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_markdown.write_text(render_markdown(report), encoding="utf-8")

    one_sided = [
        row
        for row in threshold_rows
        if row["status"] == "one_sided" and row.get("required", True)
    ]
    unexecuted = [
        row
        for row in threshold_rows
        if row["status"] == "unexecuted" and row.get("required", True)
    ]
    unmapped_failures = [
        row
        for row in threshold_rows
        if row["status"] in {"unmapped", "partial_mapping", "ambiguous"}
    ]
    print(
        f"coverage-map: {executed.__len__()}/{required.__len__()} required entries executed; "
        f"{len(one_sided)} one-sided, {len(unexecuted)} unexecuted, "
        f"{len(unmapped_failures)} unmapped/ambiguous threshold items; "
        f"{len(gaps)} gap(s); {runtime:.1f} s"
    )
    for gap in gaps:
        print(f"coverage-map gap: {gap['kind']} {gap['item']}: {gap['reason']}")
    print(f"coverage-map: wrote {out_json} and {out_markdown}")
    return 1 if gaps else 0




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="crate root (default: cwd)")
    parser.add_argument("--tier", default=None, choices=TIER_ORDER)
    parser.add_argument("--registry", default="target/kernel-registry/registry.json")
    parser.add_argument(
        "--private-doc",
        default="private.json",
        help="private rustdoc document beside the registry",
    )
    parser.add_argument("--inventory", default=None, help="threshold inventory.json")
    parser.add_argument("--scan", default=None, help="threshold scan.py")
    parser.add_argument("--aggregate", action="store_true", help="merge per-tier reports")
    parser.add_argument("--inputs", nargs="+", default=[], help="per-tier report JSONs")
    parser.add_argument("--exceptions", default=DEFAULT_EXCEPTIONS)
    parser.add_argument(
        "--out",
        default=None,
        help="JSON output (default: target/coverage-map/<host>-<tier>.json)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="markdown output (default: beside --out)",
    )
    parser.add_argument("--toolchain", default=None, help="pinned nightly toolchain")
    parser.add_argument("--bench", default="operations")
    parser.add_argument("--target-dir", default="target/coverage-map")
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="per-cell workers (default: CPU count)",
    )
    parser.add_argument("--keep", action="store_true", help="keep per-cell exports")
    parser.add_argument(
        "--suggest-exceptions",
        default=None,
        help="write proposed exception entries for unmapped items; ambiguity remains a failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.aggregate:
            if not arguments.inputs:
                raise CoverageMapError("--aggregate needs --inputs")
            out_json = Path(arguments.out or "target/coverage-map/aggregate.json")
            out_markdown = Path(arguments.summary or out_json.with_suffix(".md"))
            return aggregate_reports(
                [Path(path) for path in arguments.inputs],
                Path(arguments.exceptions),
                out_json,
                out_markdown,
            )
        return run(arguments)
    except CoverageMapError as error:
        print(f"coverage-map: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())