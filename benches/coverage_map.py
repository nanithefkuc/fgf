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

Usage::

    python3 benches/coverage_map.py --tier v3_gfni_crypto \
        --inventory /path/to/inventory.json --scan /path/to/scan.py
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import dataclasses
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
ANALYSIS_FORMAT = "fgf-coverage-evidence/1"


class CoverageMapError(Exception):
    """A fail-closed coverage-map condition with its evidence."""


# ---------------------------------------------------------------------------
# Synthetic-llvm-cov analysis (pure functions; the unit tests drive these)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EntryTarget:
    entry: str
    tier: str
    file: str
    start_line: int
    end_line: int
    publicly_reachable: bool
    call_sites: tuple[tuple[str, int, int], ...] = ()


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


def build_coverage_index(export: dict, root: Path) -> CoverageIndex:
    """Index one ``llvm-cov export -format=text`` document by source file."""

    data = export.get("data")
    if not isinstance(data, list) or not data:
        raise CoverageMapError("llvm-cov export has no data record")
    record = data[0]
    regions: dict[str, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    branches: dict[str, list[tuple[int, int, int, int, int, int, int, int, int]]] = defaultdict(list)
    for function in record.get("functions", []):
        names = [relative_source(name, root) for name in function.get("filenames", [])]
        for region in function.get("regions", []):
            if len(region) < 6:
                continue
            file_id = region[5]
            if not isinstance(file_id, int) or not 0 <= file_id < len(names):
                continue
            relative = names[file_id]
            if relative is None:
                continue
            regions[relative].append(tuple(region[:5]))
    segments: dict[str, list[tuple[int, int, int, bool, bool, bool]]] = defaultdict(list)
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
    for values in segments.values():
        values.sort(key=lambda segment: (segment[0], segment[1]))
    return CoverageIndex(
        regions=dict(regions), branches=dict(branches), segments=dict(segments)
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
    entry_evidence = {}
    for target in entries:
        mapped, count = wrapper_evidence(index, target)
        call_counts = []
        if count <= 0:
            for file, line, column in target.call_sites:
                call_mapped, call_count = position_evidence(index, file, line, column)
                if call_mapped:
                    call_counts.append(call_count)
        entry_evidence[target.entry] = {
            "wrapper_mapped": mapped,
            "wrapper_count": count,
            "call_counts": call_counts,
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
        "files": len(index.regions),
        "branches": sum(len(value) for value in index.branches.values()),
    }


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
    "ambiguous",
}
KNOWN_UNREACHABLE = set()


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
            if (
                not isinstance(straddling, list)
                or not straddling
                or not all(isinstance(cell, str) and cell for cell in straddling)
            ):
                raise CoverageMapError(
                    f"{path}: {key}: {item_class} needs a non-empty `straddling_cells`"
                )
        elif straddling is not None:
            raise CoverageMapError(
                f"{path}: {key}: `straddling_cells` is only valid for "
                "unmapped_arcane and unmapped_branch"
            )
        if key in result:
            raise CoverageMapError(f"{path}: duplicate exception for {key}")
        result[key] = dict(entry, key=key)
    return result


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
    for relative in scan_module.collect_files(root):
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
    segments = [tokens[index].text]
    start = index
    j = index
    while j >= 2 and tokens[j - 1].text == "::" and tokens[j - 2].kind == "ident":
        segments.insert(0, tokens[j - 2].text)
        j -= 2
        start = j
    return segments, tokens[start].start


def discover_call_sites(
    entries: list[dict], models: dict[str, object]
) -> dict[str, tuple[tuple[str, int, int], ...]]:
    """Candidate call sites for each registry entry, by source token search."""

    name_counts = Counter(entry["entry"].rsplit("::", 1)[-1] for entry in entries)
    occurrences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for relative, model in models.items():
        for index, token in enumerate(model.tokens):
            if token.kind == "ident":
                occurrences[token.text].append((relative, index))
    sites: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
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
    result = {}
    for path, values in sites.items():
        unique = sorted(set(values))
        result[path] = tuple(unique[:12])
    return result


def scan_line_column(source: str, offset: int) -> tuple[int, int]:
    return source.count("\n", 0, offset) + 1, offset - source.rfind("\n", 0, offset)


def build_entry_targets(
    registry: list[dict],
    spans: dict[str, dict],
    call_sites: dict[str, tuple[tuple[str, int, int], ...]],
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
        call_cells = []
        total = 0
        for cell, evidence in cell_evidence:
            record = evidence["entries"].get(target.entry)
            if record is None:
                continue
            count = record.get("wrapper_count", 0)
            call_count = max(record.get("call_counts") or [0])
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
                "definition": {"file": target.file, "line": target.start_line},
                "reason": None,
                "call_sites": [
                    {"file": file, "line": line, "column": column}
                    for file, line, column in target.call_sites
                ],
            }
        )
    return rows


_SOURCE_ROOT: Path | None = None
_SOURCE_LINES: dict[str, list[str]] = {}


def configure_sources(root: Path) -> None:
    global _SOURCE_ROOT
    _SOURCE_ROOT = root
    _SOURCE_LINES.clear()


def _lines(file: str) -> list[str]:
    if file not in _SOURCE_LINES:
        root = _SOURCE_ROOT or Path(".")
        try:
            _SOURCE_LINES[file] = (root / file).read_text(encoding="utf-8").splitlines()
        except OSError:
            _SOURCE_LINES[file] = []
    return _SOURCE_LINES[file]


def _arm_tier_near(file: str, line: int) -> str | None:
    lines = _lines(file)
    for index in range(min(line, len(lines)) - 1, max(0, line - 120) - 1, -1):
        text = lines[index]
        if not any(
            token in text
            for token in ("Backend::", "X86Proof::V4x", "V3GfniCrypto")
        ):
            if " fn " in text or text.lstrip().startswith("fn "):
                above = lines[max(0, index - 6) : index + 1]
                if any("simd512" in line for line in above):
                    return "v4x"
                break
            continue
        if "V4x" in text and ("V3GfniCrypto" in text or "Backend::V3" in text):
            return "v3"
        if "V4x" in text:
            return "v4x"
        if "V3GfniCrypto" in text:
            return "v3_gfni_crypto"
        if "Backend::V3" in text:
            return "v3"
        if "Backend::V2" in text:
            return "v2"
    return None


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
    return tier


def summarize_thresholds(
    items: list[dict],
    targets: list[SiteTarget],
    cell_evidence: list[tuple[str, dict]],
    exceptions: dict[str, dict],
    requested_tier: str,
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
            site_required = hint is None or TIER_RANK[hint] <= TIER_RANK[requested_tier]
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
            reason = "every site's code runs only at a stronger tier"
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
                "status": status,
                "reason": reason,
                "sites": sites,
            }
        )
    return rows


def _axis_value(unit: str, cell: dict) -> int | None:
    if unit == "bytes":
        value = cell.get("row_len")
        return value if isinstance(value, int) else cell.get("bytes")
    if unit == "rows":
        return cell.get("rows") if isinstance(cell.get("rows"), int) else None
    if unit in {"terms", "sources"}:
        return cell.get("sources") if isinstance(cell.get("sources"), int) else None
    return None


def straddling_cells(item: dict, cells: list[dict]) -> list[str]:
    """Cells on both sides of an item's cutoff, or the first cells otherwise."""

    unit = item.get("unit", "other")
    cutoff = item.get("value")
    pairs = [
        (cell["id"], _axis_value(unit, cell))
        for cell in cells
        if isinstance(cell.get("id"), str)
    ]
    numeric = [(cell, value) for cell, value in pairs if isinstance(value, int)]
    if unit in {"bytes", "rows", "terms", "sources"} and isinstance(cutoff, int):
        below = [entry for entry in numeric if entry[1] < cutoff]
        above = [entry for entry in numeric if entry[1] > cutoff]
        chosen = []
        if below:
            chosen.append(min(below, key=lambda entry: entry[1])[0])
        if above:
            chosen.append(max(above, key=lambda entry: entry[1])[0])
        return chosen
    return [cell for cell, _ in pairs[:2]]


def validate_exceptions(
    exceptions: dict[str, dict],
    entries: list[dict],
    thresholds: list[dict],
    cells: list[dict],
    requested_tier: str,
) -> list[dict]:
    """Fail on stale keys, wrong classes, and invalid straddling declarations."""

    cell_ids = {cell["id"] for cell in cells}
    threshold_by_key = {row["key"]: row for row in thresholds}
    diagnostics: list[dict] = []
    for key, entry in exceptions.items():
        scope = entry.get("tiers")
        if scope and requested_tier not in scope:
            continue
        item_class = entry["class"]
        if item_class == "unreachable":
            if key not in KNOWN_UNREACHABLE and key not in threshold_by_key:
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
            ids = entry["straddling_cells"]
            missing = [cell for cell in ids if cell not in cell_ids]
            if missing:
                raise CoverageMapError(
                    f"exceptions: {key}: unknown bench cells {missing[:4]}"
                )
            if row["unit"] in {"bytes", "rows", "terms", "sources"} and isinstance(
                row["value"], int
            ):
                values = [
                    _axis_value(row["unit"], cell)
                    for cell in cells
                    if cell["id"] in set(ids)
                ]
                numeric = [value for value in values if isinstance(value, int)]
                cutoff = row["value"]
                if not any(value < cutoff for value in numeric) or not any(
                    value > cutoff for value in numeric
                ):
                    raise CoverageMapError(
                        f"exceptions: {key}: straddling_cells do not cover both sides "
                        f"of the {cutoff}-{row['unit']} cutoff"
                    )
            continue
        if item_class == "unmapped_branch":
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
            ids = entry["straddling_cells"]
            missing = [cell for cell in ids if cell not in cell_ids]
            if missing:
                raise CoverageMapError(
                    f"exceptions: {key}: unknown bench cells {missing[:4]}"
                )
            if row["unit"] in {"bytes", "rows", "terms", "sources"} and isinstance(
                row["value"], int
            ):
                values = [
                    _axis_value(row["unit"], cell)
                    for cell in cells
                    if cell["id"] in set(ids)
                ]
                numeric = [value for value in values if isinstance(value, int)]
                cutoff = row["value"]
                if not any(value < cutoff for value in numeric) or not any(
                    value > cutoff for value in numeric
                ):
                    raise CoverageMapError(
                        f"exceptions: {key}: straddling_cells do not cover both sides "
                        f"of the {cutoff}-{row['unit']} cutoff"
                    )
            continue
        raise CoverageMapError(f"exceptions: {key}: unknown class {item_class}")
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
        elif entry["class"] == "ambiguous" and row["status"] == "ambiguous":
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
                    "reason": "publicly reachable entry at the requested tier executed in no cell",
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
                        "site tier " + "/".join(hints) + " is stronger than the requested tier; "
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
) -> Path:
    env = os.environ.copy()
    env.pop("SIMD_BACKEND", None)
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
        "--all-features",
        "--bench",
        bench,
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    started = time.monotonic()
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
    _ = started
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
    for line in process.stdout.splitlines():
        if line.startswith("CELL\t"):
            try:
                cells.append(json.loads(line.split("\t", 1)[1]))
            except json.JSONDecodeError as error:
                raise CoverageMapError(f"malformed CELL record: {error}")
    if not cells:
        raise CoverageMapError("bench --inventory returned no cells")
    return cells


_WORKER: dict = {}


def _worker_init(spec: dict) -> None:
    _WORKER.clear()
    _WORKER.update(spec)


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
    merge = subprocess.run(
        [str(spec["profdata"]), "merge", "-sparse", "-o", str(merged), str(profile)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if merge.returncode != 0:
        raise CoverageMapError(
            f"cell {cell_id}: llvm-profdata merge failed: {merge.stderr.strip()}"
        )
    export = subprocess.run(
        [
            str(spec["llvm_cov"]),
            "export",
            "--format=text",
            f"--instr-profile={merged}",
            str(spec["binary"]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if export.returncode != 0:
        raise CoverageMapError(
            f"cell {cell_id}: llvm-cov export failed: {export.stderr.strip()}"
        )
    if spec["keep_exports"]:
        exported.write_text(export.stdout, encoding="utf-8")
    try:
        document = json.loads(export.stdout)
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
    exceptions = report.get("exceptions") or []
    if exceptions:
        for entry in exceptions:
            cells = entry.get("straddling_cells") or []
            suffix = f" (straddling cells: {_cell_list(cells, 6)})" if cells else ""
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
        "branch, or an unmapped/ambiguous threshold item, unless the exceptions "
        "file records it with a reason (unmapped_arcane also declares straddling cells).",
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
                    entry_executed[row["entry"]] = True
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
            threshold_summary[key] = "tier_not_required"
        elif any(row["status"] in {"both_ways", "exercised"} for row in required):
            threshold_summary[key] = "passed"
        elif any(row["status"] == "excused" for row in required):
            threshold_summary[key] = "excused"
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
    positions = {key: value for key, value in positions.items()}
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

    configure_sources(root)
    registry = load_registry(Path(arguments.registry))
    private_doc = Path(arguments.registry).with_name(arguments.private_doc)
    spans = load_private_spans(private_doc)
    private_allowlist = set(
        load_json(Path(arguments.registry)).get("private_allowlist", [])  # type: ignore[union-attr]
    )
    models = scan_sources(root, scan_module)
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
    threshold_rows = summarize_thresholds(
        threshold_items, sites, cell_evidence, exceptions, arguments.tier
    )
    diagnostics = validate_exceptions(
        exceptions, entry_rows, threshold_rows, cells, arguments.tier
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
            if row["status"] not in {"unmapped", "partial_mapping", "ambiguous"}:
                continue
            cells_for_item = straddling_cells(row, cells)
            if not cells_for_item:
                continue
            proposals[row["key"]] = {
                "key": row["key"],
                "class": "ambiguous" if row["status"] == "ambiguous" else "unmapped_arcane",
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
        if row["status"] == "unexecuted" and row["evidence"] == "none":
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
        help="write proposed exception entries for unmapped/ambiguous items",
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