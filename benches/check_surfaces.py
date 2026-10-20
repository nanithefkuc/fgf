#!/usr/bin/env python3
"""Validate surfaces benchmark inventories against the declared public API.

    python3 benches/check_surfaces.py --manifest benches/surface_manifest.json \
        --rustdoc target/doc/fgf.json [--records RECORDS] [--check-log LOG]...

The surfaces campaign registers every measured case twice: once as a
``SURFACE_INVENTORY`` line without timing and once as a ``SURFACE`` or
``GF16_SURFACE`` record with timing. This tool is the inventory-side gate. It
fails closed on a case whose ``api`` paths do not resolve against the crate's
declared stable public API (plus the manifest's explicit additions), on a
compiler-declared callable that no case covers, on duplicate case strings or
duplicate identities, and on a timed round whose records disagree with its
inventory.

The public callable universe comes from pinned nightly rustdoc JSON, including
macro-generated methods and reachable re-exports. The supplemental manifest
names only foreign/operator spellings and non-callable observations; it never
defines the required fgf-owned callable set.

``--records`` validates the campaign record tree: every
``round-{1..5}-v3-surfaces.log`` under each available host directory must
parse, hold identical inventories across rounds and hosts, match the
manifest-declared process backend and pinned core, and cover the manifest's
required APIs. ``--check-log`` validates one captured ``--check`` /
``--presentations --check`` stdout: the full inventory with no timing
records and the harness completion marker. Required-API coverage spans the
union of all given inputs. Every failure exits nonzero with a diagnostic;
nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from publish import (
    PublishError,
    check_positive,
    fail,
    is_noise,
    normalize_backend,
)

# Sections in their fixed order inside one log; the surfaces section is
# implicit from the first record, the presentations section starts at its
# process header.
SURFACE_SECTIONS = ("surfaces", "gf16-presentations")
SURFACE_SCOPES = ("setup", "steady", "query", "end_to_end")
PRESENTATION_SCOPES = ("native", "end_to_end", "setup")
CONTROL_PHASES = ("entry", "exit")
# Dispatch tags a current artifact may report for an fgf arm or surface record.
KNOWN_BACKENDS = frozenset({"v2", "v3", "v3_gfni_crypto", "v4x", "neon_aes",
                            "neon", "wasm128", "scalar"})

# Surfaces-section payload: the inventory omits the backend; the record adds
# ns, backend, and the sample count. `control`/`phase` are always present.
SURFACE_INVENTORY_KEYS = ("api", "bytes", "case", "control", "destinations",
                          "field", "lanes", "operation", "phase", "scope",
                          "sources")
SURFACE_RECORD_KEYS = SURFACE_INVENTORY_KEYS + ("backend", "ns", "samples")
SURFACE_OPTIONAL_KEYS = ("p10_ns", "p90_ns")
# Presentations payloads carry arm and backend in both forms.
GF16_INVENTORY_KEYS = ("api", "arm", "backend", "bytes", "case",
                       "destinations", "field", "lanes", "operation", "scope",
                       "sources")
GF16_RECORD_KEYS = GF16_INVENTORY_KEYS + ("ns", "p10_ns", "p90_ns", "samples")
# Timed presentation records additionally carry `batch_reps`: the positive
# integer of public calls inside one timed batch. It is timing metadata, not
# case identity, so it stays out of every key tuple above.
GF16_TIMED_KEYS = ("batch_reps",)
GF16_FGF_ARMS = ("fgf", "fgf-control")
# One BENCHMARK_MODE marker follows each process header. Campaign rounds run
# full; smoke and check declare themselves; the surfaces harness also prints
# inventory-only runs.
BENCHMARK_MODES = ("full", "smoke", "check")
SURFACES_ONLY_MODES = ("inventory",)

SURFACES_BANNER_RE = re.compile(r"fgf surfaces benchmark — backend: (\S+)")
SURFACES_PROCESS_RECORD_RE = re.compile(
    r"FIELD_BACKEND\tprocess\t(\S+)\tnative_mul=(true|false)"
    r"\tblocked_rows=(true|false)")
SURFACES_PANEL_RE = re.compile(
    r"^.+\s[0-9.]+(ns|µs|ms)\s+(setup|steady|query|end_to_end)$")
PRESENTATIONS_HEADER_RE = re.compile(
    r"^GF16 presentations; fgf=(\S+) process_backend=(\S+); "
    r"rss=([\d.]+) AVX2; rse=([\d.]+) scalar GF16; "
    r"GF-Complete=([\d.]+) (\S+) SSSE3 split-4/16 polynomial=0x1100b; "
    r"Leopard-RS 1\.x \(pre-Leopard2\)=([0-9a-f]{40}) AVX2$")
PRESENTATIONS_COMPLETE_RE = re.compile(
    r"^PRESENTATIONS complete: (\d+) presentations, (\d+) measurements"
    r"( validated without timing)?\.$")
PRESENTATIONS_PROSE_RE = re.compile(
    r"^Records: |^\S.*; backend=\S+; "
    r"(canonical tower encoding|exact caller-image map validated)$"
    r"|^affinity=\S* scheduler=\S*$")
HOST_RECORD_RE = re.compile(
    r"HOST\tcpus_allowed_list=([0-9,-]+)\tsched_policy=(\d+) (\S+)")
# Map work: surfaces cases reserve the `maps/` case prefix (composite
# `S->T` presentation fields, scalar caller loops); the presentations'
# `end_to_end` conversion caller loops page under v3/maps.md too.
MAP_CASE_PREFIX = "maps/"


def is_surfaces_map_case(obj: dict) -> bool:
    return obj["case"].startswith(MAP_CASE_PREFIX) or any(
        canonical_api(path).startswith(("Embedding::", "Isomorphism::"))
        for path in obj["api"]
    )


def is_presentation_map_record(obj: dict) -> bool:
    """Presentations scope end_to_end is exactly the public scalar caller
    loops over the validated exact maps."""
    return obj["scope"] == "end_to_end"


def identity_keys(section: str, timed: bool) -> tuple:
    if section == "gf16-presentations":
        return GF16_RECORD_KEYS if timed else GF16_INVENTORY_KEYS
    return SURFACE_RECORD_KEYS if timed else SURFACE_INVENTORY_KEYS


def workload_keys(section: str) -> tuple:
    if section == "gf16-presentations":
        return tuple(key for key in GF16_INVENTORY_KEYS if key != "backend")
    return SURFACE_INVENTORY_KEYS


def workload_of(obj: dict, section: str) -> tuple:
    """Case identity without backend or timing; comparable across hosts."""
    return tuple(obj[key] if key != "api" else tuple(obj[key])
                 for key in workload_keys(section))


def presentation_schedule(obj: dict) -> str:
    """Coefficient schedule from the frozen presentation case grammar
    ``gf16-presentations/<field>/<operation>/<schedule>/<shape>/<bytes>/<scope>/<arm>``."""
    parts = obj["case"].split("/")
    if len(parts) < 4:
        fail(f"presentation case {obj['case']!r} does not follow the frozen "
             f"gf16-presentations case grammar")
    return parts[3]


def williams_period(arms: int) -> int:
    """Orders in one balanced Latin-square cycle: n for even n, 2n for odd."""
    return arms if arms % 2 == 0 else 2 * arms


def check_balanced_samples(samples: dict[str, int], full: bool, error) -> None:
    """All arms of one interleaved case ran the same whole number of balanced
    cycles; full runs keep at least 30 samples per arm."""
    counts = set(samples.values())
    if len(counts) != 1:
        error(f"arms report unequal sample counts {samples!r}")
        return
    count = counts.pop()
    period = williams_period(len(samples))
    if count % period:
        error(f"{count} samples per arm is not a whole number of "
              f"{period}-order balanced cycles")
    if full and count < 30:
        error(f"{count} samples per arm is below the 30-sample full-mode floor")


def group_of(obj: dict) -> tuple:
    """Group matching arms while preserving schedule and every case parameter."""
    keys = tuple(key for key in GF16_INVENTORY_KEYS
                 if key not in ("api", "arm", "backend", "case"))
    return tuple(obj[key] for key in keys) + (obj["case"].rsplit("/", 1)[0],)


def identity_digest(entries: list[dict]) -> str:
    """Stable digest of one inventory's canonical identity payloads."""
    canonical = json.dumps(
        sorted(json.dumps(entry, sort_keys=True, ensure_ascii=False)
               for entry in entries),
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSON object validation
# ---------------------------------------------------------------------------

def _string(obj: dict, key: str, where: str) -> str:
    value = obj[key]
    if not isinstance(value, str) or not value or value.strip() != value:
        fail(f"{where}: {key} must be a nonempty trimmed string, got {value!r}")
    if any(character in value for character in "\t\n\r"):
        fail(f"{where}: {key} must not carry tabs or newlines: {value!r}")
    return value


def _count(obj: dict, key: str, where: str) -> int:
    value = obj[key]
    if type(value) is not int or value < 0:
        fail(f"{where}: {key} must be a nonnegative integer, got {value!r}")
    return value


def _number(obj: dict, key: str, where: str) -> float:
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{where}: {key} must be a JSON number, got {value!r}")
    check_positive(float(value), f"{where}: {key}")
    return float(value)


def _validate_timing(obj: dict, where: str) -> None:
    obj["ns"] = _number(obj, "ns", where)
    if _count(obj, "samples", where) < 1:
        fail(f"{where}: samples must be at least one")
    for key in ("p10_ns", "p90_ns"):
        if key in obj:
            obj[key] = _number(obj, key, where)
    if ("p10_ns" in obj) != ("p90_ns" in obj):
        fail(f"{where}: p10_ns and p90_ns must be reported together")
    if "p10_ns" in obj and not obj["p10_ns"] <= obj["ns"] <= obj["p90_ns"]:
        fail(f"{where}: ns lies outside its reported decile band")


def surface_object(raw: dict, *, section: str, timed: bool, where: str) -> dict:
    """Validate one SURFACE_INVENTORY / SURFACE / GF16_SURFACE JSON object."""
    if not isinstance(raw, dict):
        fail(f"{where}: inventory and record payloads must be JSON objects")
    present = set(raw)
    required = set(identity_keys(section, timed))
    optional = set(SURFACE_OPTIONAL_KEYS) if timed else set()
    if section == "gf16-presentations" and timed:
        required.update(GF16_TIMED_KEYS)
    missing = sorted(required - present)
    unknown = sorted(present - required - optional)
    if missing:
        fail(f"{where}: missing keys {missing}")
    if unknown:
        fail(f"{where}: unsupported keys {unknown}")

    _string(raw, "case", where)
    api = raw["api"]
    if not isinstance(api, list) or not api or not all(
            isinstance(item, str) and item and item.strip() == item
            for item in api):
        fail(f"{where}: api must be a nonempty array of trimmed strings")
    _string(raw, "field", where)
    _string(raw, "operation", where)
    scopes = PRESENTATION_SCOPES if section == "gf16-presentations" else SURFACE_SCOPES
    if raw["scope"] not in scopes:
        fail(f"{where}: scope {raw['scope']!r} is not one of {scopes}")
    for key in ("bytes", "lanes", "sources", "destinations"):
        _count(raw, key, where)
    if section == "surfaces":
        if not isinstance(raw["control"], bool):
            fail(f"{where}: control must be a boolean")
        phase = raw["phase"]
        if (phase is None) == raw["control"]:
            fail(f"{where}: phase is non-null exactly for control cases")
        if raw["control"]:
            if phase not in CONTROL_PHASES:
                fail(f"{where}: phase must be one of {CONTROL_PHASES}")
    else:
        arm = _string(raw, "arm", where)
        backend = normalize_backend(_string(raw, "backend", where))
        raw["backend"] = backend
        if arm not in ("fgf", "fgf-control") and backend == "":
            fail(f"{where}: native arm {arm!r} needs a backend")
        if arm in GF16_FGF_ARMS and backend not in KNOWN_BACKENDS:
            fail(f"{where}: arm {arm!r} reports unsupported backend {backend!r}")
    if timed:
        if section == "gf16-presentations":
            reps = raw["batch_reps"]
            if type(reps) is not int or reps < 1:
                fail(f"{where}: batch_reps must be a positive integer, got {reps!r}")
        if section == "surfaces":
            backend = normalize_backend(_string(raw, "backend", where))
            raw["backend"] = backend
            if backend not in KNOWN_BACKENDS:
                fail(f"{where}: unsupported backend {backend!r}")
        _validate_timing(raw, where)
    return raw


# ---------------------------------------------------------------------------
# Public API universe
# ---------------------------------------------------------------------------

from api_inventory import canonical_api, declared_callables


class Universe:
    """Compiler-declared callables and explicit supplemental observations."""

    def __init__(self, names: frozenset, literal: frozenset, sha256: str) -> None:
        self.names = names
        self.literal = literal
        self.sha256 = sha256

    def resolve(self, path: str) -> None:
        if path not in self.literal and canonical_api(path) not in self.names:
            fail(f"api path {path!r} names no reachable stable public callable")


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing surfaces manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"unreadable surfaces manifest {path}: {exc}")
    if not isinstance(manifest, dict):
        fail(f"{path}: the manifest must be a JSON object")
    if set(manifest) != {"items"}:
        fail(f"{path}: the supplemental manifest only accepts an items array")
    items = manifest["items"]
    if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
        fail(f"{path}: items must be nonempty strings")
    if len(set(items)) != len(items):
        fail(f"{path}: duplicate supplemental API observations")
    return manifest


def public_universe(rustdoc: Path, manifest: dict) -> Universe:
    """Derive the stable callable set from compiler visibility and ownership."""
    try:
        payload = rustdoc.read_bytes()
        document = json.loads(payload)
        names = frozenset(declared_callables(document))
    except (OSError, ValueError, KeyError) as exc:
        fail(f"unreadable compiler API inventory {rustdoc}: {exc}")
    literal = frozenset(manifest["items"])
    redundant = sorted(path for path in literal if canonical_api(path) in names)
    if redundant:
        fail(f"supplemental manifest duplicates compiler-declared APIs: {redundant}")
    return Universe(names, literal, hashlib.sha256(payload).hexdigest())


def validate_case_api(obj: dict, universe: Universe, where: str) -> None:
    if obj.get("control") is True:
        return  # the byte-XOR control is not a public API call
    for path in obj["api"]:
        try:
            universe.resolve(path)
        except PublishError as exc:
            fail(f"{where}: case {obj['case']!r}: {exc}")


def validate_required_coverage(cases: list[dict], manifest: dict,
                               universe: Universe) -> None:
    covered: set[str] = set()
    for entry in cases:
        if entry.get("control") is True:
            continue
        covered.update(canonical_api(path) for path in entry["api"] if path not in universe.literal)
    missing = sorted(universe.names - covered)
    if missing:
        fail(f"compiler-declared public APIs carry no surfaces case: {missing}")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

class SurfacesLog:
    """Parse one surfaces-family log: campaign round or --check capture.

    The log opens in the implicit surfaces section; the presentations section
    starts at its ``GF16 presentations;`` process header.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.section = "surfaces"
        self.inventories: dict[str, list[dict]] = {section: []
                                                   for section in SURFACE_SECTIONS}
        self.records: dict[str, list[dict]] = {section: []
                                               for section in SURFACE_SECTIONS}
        self.banners: list[dict] = []
        self.modes: dict[str, str] = {}
        self.process_evidence: dict[str, set[str]] = {section: set()
                                                      for section in SURFACE_SECTIONS}
        self.host_placements: list[dict] = []
        self.presentations_header: dict | None = None
        self.presentations_complete: tuple[int, int, bool] | None = None
        self.surfaces_complete: int | None = None
        self.timer_floor: dict | None = None
        self.check_lines: list[str] = []
        self.checked_cases: list[str] = []
        self.checked_complete: int | None = None
        self._banner_fields: dict[str, list[str]] = {section: []
                                                     for section in SURFACE_SECTIONS}

    def error(self, message: str) -> None:
        fail(f"{self.path.name}: {message}")

    def parse(self, text: str) -> None:
        for lineno, raw in enumerate(text.splitlines(), 1):
            try:
                self.line(raw)
            except PublishError:
                raise
            except Exception as exc:
                fail(f"{self.path.name}:{lineno}: unsupported parser input "
                     f"{raw!r}: {exc}")
        self.finish()

    def line(self, raw: str) -> None:
        if raw.startswith("SURFACE_CHECKED\t"):
            self.checked_cases.append(raw.split("\t", 1)[1])
            return
        if raw.startswith("SURFACES_CHECK_COMPLETE\tcases="):
            if self.checked_complete is not None:
                self.error("duplicate check completion marker")
            try:
                self.checked_complete = int(raw.split("=", 1)[1])
            except ValueError:
                self.error(f"malformed check completion: {raw!r}")
            return
        if raw.startswith("SURFACES\t"):
            match = re.fullmatch(r"SURFACES\tcases=(\d+)\telapsed_s=([0-9.]+)", raw)
            if match is None or self.surfaces_complete is not None:
                self.error(f"malformed or duplicate timing completion: {raw!r}")
            check_positive(float(match.group(2)), "surfaces elapsed seconds")
            self.surfaces_complete = int(match.group(1))
            return
        if raw.startswith("TIMER_FLOOR\t"):
            if self.section != "surfaces" or self.timer_floor is not None:
                self.error("misplaced or duplicate timer-floor measurement")
            value = json.loads(raw.split("\t", 1)[1])
            if not isinstance(value, dict) or set(value) != {"ns", "samples"}:
                self.error("malformed timer-floor measurement")
            if type(value["ns"]) not in (int, float):
                self.error("timer-floor latency must be a JSON number")
            check_positive(float(value["ns"]), "timer-floor nanoseconds")
            if type(value["samples"]) is not int or value["samples"] < 1:
                self.error("invalid timer-floor sample count")
            self.timer_floor = value
            return
        for prefix, handler in (
            ("BENCHMARK_MODE\t", self.mode_line),
            ("FIELD_BACKEND\t", self.banner_line),
            ("SURFACE_INVENTORY\t", lambda line: self.json_line(line, "inventory")),
            ("SURFACE\t", lambda line: self.json_line(line, "record")),
            ("GF16_SURFACE\t", lambda line: self.json_line(line, "record")),
            ("CHECK ", self.check_line),
            ("HOST\t", self.host_line),
        ):
            if raw.startswith(prefix):
                handler(raw)
                return
        self.context_line(raw)

    def mode_line(self, line: str) -> None:
        """One mode marker per section, immediately after its process header."""
        parts = line.split("\t")
        if len(parts) != 2 or parts[1] not in BENCHMARK_MODES + SURFACES_ONLY_MODES:
            self.error(f"malformed benchmark mode marker: {line!r}")
        if parts[1] in SURFACES_ONLY_MODES and self.section != "surfaces":
            self.error(f"mode {parts[1]!r} is a surfaces mode: {line!r}")
        if self.section in self.modes:
            self.error(f"duplicate benchmark mode marker for {self.section!r}")
        if not self.process_evidence[self.section]:
            self.error(f"benchmark mode marker before the {self.section!r} "
                       f"process header")
        self.modes[self.section] = parts[1]

    def banner_line(self, line: str) -> None:
        process = SURFACES_PROCESS_RECORD_RE.fullmatch(line)
        if process is not None:
            if self.section != "surfaces":
                self.error(f"process capability record outside the surfaces "
                           f"section: {line!r}")
            self.process_evidence["surfaces"].add(
                normalize_backend(process.group(1)))
            return
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
            self.error(f"malformed FIELD_BACKEND record: {line!r}")
        label, backend = parts[1].strip(), normalize_backend(parts[2].strip())
        if label in self._banner_fields[self.section]:
            self.error(f"duplicate FIELD_BACKEND banner for {label!r}")
        self._banner_fields[self.section].append(label)
        self.banners.append({"section": self.section, "label": label,
                             "backend": backend})

    def json_line(self, line: str, kind: str) -> None:
        tag, _, payload = line.partition("\t")
        if not payload.strip():
            self.error(f"malformed {kind}: {line!r}")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            self.error(f"malformed {kind} JSON: {exc}")
        if kind == "record" and tag == "SURFACE" and self.section != "surfaces":
            self.error(f"SURFACE record outside the surfaces section: {line!r}")
        if tag == "GF16_SURFACE" or (kind == "inventory"
                                     and self.section == "gf16-presentations"):
            if self.section != "gf16-presentations":
                self.error(f"{tag or 'GF16 inventory'} outside the "
                           f"presentations section: {line!r}")
            section = "gf16-presentations"
        else:
            section = "surfaces"
        record = surface_object(raw, section=section, timed=kind == "record",
                                where=f"{self.path.name}: {kind}")
        if kind == "inventory":
            self.inventories[section].append(record)
        else:
            self.records[section].append(record)

    def check_line(self, line: str) -> None:
        from publish import GF16_CHECK_LINES
        if line not in GF16_CHECK_LINES:
            self.error(f"altered surfaces CHECK statement: {line!r}")
        self.check_lines.append(line)

    def host_line(self, line: str) -> None:
        match = HOST_RECORD_RE.fullmatch(line)
        if match is None:
            self.error(f"malformed HOST record: {line!r}")
        self.host_placements.append({
            "section": self.section, "cpus": match.group(1),
            "policy": int(match.group(2)), "policy_name": match.group(3)})

    def context_line(self, raw: str) -> None:
        stripped = raw.strip()
        if stripped in {"UNIT\tsurfaces", "UNIT\tpresentations"}:
            return
        if not stripped or is_noise(raw):
            return
        match = SURFACES_BANNER_RE.fullmatch(stripped)
        if match:
            if self.process_evidence["surfaces"]:
                self.error("duplicate surfaces process header")
            self.section = "surfaces"
            self.process_evidence["surfaces"].add(normalize_backend(match.group(1)))
            return
        header = PRESENTATIONS_HEADER_RE.match(stripped)
        if header:
            if self.presentations_header is not None:
                self.error("duplicate presentations process header")
            self.section = "gf16-presentations"
            self.presentations_header = {
                "fgf": header.group(1),
                "backend": normalize_backend(header.group(2)),
            }
            self.process_evidence["gf16-presentations"].add(
                self.presentations_header["backend"])
            return
        if self.section == "surfaces":
            if SURFACES_PANEL_RE.match(stripped):
                return
            self.error(f"unsupported line in the surfaces section: "
                       f"{stripped!r}")
            return
        if PRESENTATIONS_COMPLETE_RE.match(stripped):
            match = PRESENTATIONS_COMPLETE_RE.match(stripped)
            self.presentations_complete = (
                int(match.group(1)), int(match.group(2)),
                match.group(3) is not None)
            return
        if PRESENTATIONS_PROSE_RE.match(stripped):
            return
        self.error(f"unsupported line in the presentations section: "
                   f"{stripped!r}")

    def finish(self) -> None:
        if not any(self.inventories[section] or self.records[section]
                   or self.banners for section in SURFACE_SECTIONS):
            self.error("log holds no surfaces content")
        present = set(self.sections)
        for section in SURFACE_SECTIONS:
            if section in present and section not in self.modes:
                self.error(f"section {section!r} lacks its benchmark mode "
                           f"marker")
            if section in self.modes and section not in present:
                self.error(f"benchmark mode marker without {section!r} content")
        for section in SURFACE_SECTIONS:
            if (self.records[section] or self._banners_of(section)) \
                    and not self.inventories[section]:
                self.error(f"section {section!r} holds no inventory entries")
            if self.inventories[section] \
                    and not self.process_evidence[section]:
                self.error(f"section {section!r} lacks its process backend "
                           f"evidence")
            if len(self.process_evidence[section]) > 1:
                self.error(f"section {section!r} reports conflicting process "
                           f"backends {sorted(self.process_evidence[section])}")
        if self.inventories["gf16-presentations"]:
            if self.presentations_header is None:
                self.error("the presentations section lacks its process header")
            if self.presentations_complete is None:
                self.error("the presentations section lacks its completion "
                           "marker")
            registered, measured, _ = self.presentations_complete
            if registered < 1:
                self.error("malformed presentations completion counts")
            if measured != len(self.inventories["gf16-presentations"]):
                self.error(f"the presentations completion marker counts "
                           f"{measured} measurements against "
                           f"{len(self.inventories['gf16-presentations'])} "
                           f"inventory entries")

    @property
    def sections(self) -> list[str]:
        return [section for section in SURFACE_SECTIONS
                if self.inventories[section] or self.records[section]
                or self._banners_of(section)]

    def process_backends(self) -> dict[str, str]:
        return {section: next(iter(evidence))
                for section, evidence in self.process_evidence.items()
                if evidence}

    def require_full(self) -> None:
        """Campaign rounds measure every present section in full mode; smoke
        and check runs parse structurally but are never campaign inputs."""
        for section in self.sections:
            mode = self.modes.get(section)
            if mode != "full":
                self.error(f"section {section!r} ran in {mode!r} mode; "
                           f"campaign rounds require full mode")

    def _banners_of(self, section: str) -> dict[str, str]:
        return {banner["label"]: banner["backend"] for banner in self.banners
                if banner["section"] == section}

    # -- per-log invariants shared by the checker and the publisher ----------

    def validate_batch_reps(self, section: str) -> None:
        """Timed presentation records carry the public-call count of one timed
        batch: native fgf and fgf-control arms calibrate identical batches,
        and cycling16 native batches hold whole coefficient cycles. Native
        groups also prove the balanced stop rule ran: every arm reports the
        same sample count, a whole number of Williams cycles, and at least
        30 samples in full mode."""
        groups: dict[tuple, dict[str, int]] = {}
        native_samples: dict[tuple, dict[str, int]] = {}
        for record in self.records[section]:
            group = group_of(record)
            groups.setdefault(group, {})[record["arm"]] = record["batch_reps"]
            if record["scope"] == "native":
                native_samples.setdefault(group, {})[record["arm"]] = record["samples"]
            if (record["scope"] == "native"
                    and presentation_schedule(record) == "cycling16"
                    and record["batch_reps"] % 16):
                self.error(f"presentation group {group!r}: cycling16 "
                           f"batch_reps {record['batch_reps']!r} is not a "
                           f"multiple of 16")
        for group, arms in groups.items():
            if ("fgf" in arms and "fgf-control" in arms
                    and arms["fgf"] != arms["fgf-control"]):
                self.error(f"presentation group {group!r}: fgf and fgf-control "
                           f"batch_reps differ ({arms['fgf']} vs "
                           f"{arms['fgf-control']})")
        full = self.modes.get(section) == "full"
        for group, arms in native_samples.items():
            # Single-arm native records (the scalar chain) use the
            # single-body sampler, not the interleaved balanced design.
            if len(arms) > 1:
                check_balanced_samples(arms, full, lambda message, group=group:
                                       self.error(f"presentation group {group!r}: {message}"))

    def validate_section(self, section: str, timed: bool = True) -> None:
        """Structural checks run on inventories so that --check captures
        validate identically; backend rules and record bijections run only
        on timed campaign logs."""
        inventories = self.inventories[section]
        records = self.records[section]
        if not timed and records:
            self.error(f"{self.path.name}: a --check capture carries no "
                       f"timing records")
        seen_cases: set[str] = set()
        seen_workloads: dict[tuple, str] = {}
        for entry in inventories:
            where = f"{self.path.name}: case {entry['case']!r}"
            if entry["case"] in seen_cases:
                self.error(f"{where}: duplicate case string")
            seen_cases.add(entry["case"])
            workload = workload_of(entry, section)
            if workload in seen_workloads:
                self.error(f"{where}: identity duplicates case "
                           f"{seen_workloads[workload]!r}")
            seen_workloads[workload] = entry["case"]
        banners = self._banners_of(section)
        if section == "surfaces":
            control_invs = [entry for entry in inventories
                            if entry.get("control")]
            if sorted(entry["phase"] for entry in control_invs) \
                    != sorted(CONTROL_PHASES):
                self.error(f"{self.path.name}: the surfaces inventory must "
                           f"register one entry and one exit control")
            if len({entry["case"] for entry in control_invs}) != 2:
                self.error(f"{self.path.name}: entry and exit controls must "
                           f"carry distinct case strings")
            plain_fields = {entry["field"] for entry in inventories
                            if not entry.get("control")
                            and not is_surfaces_map_case(entry)
                            and entry["field"] not in {"process", "u64", "gf2-packed-bits"}}
            if plain_fields != set(banners):
                self.error(f"{self.path.name}: FIELD_BACKEND banners "
                           f"{sorted(banners)} do not cover exactly the measured "
                           f"fields {sorted(plain_fields)}; map cases reserve "
                           f"composite fields outside the banner list")
        else:
            groups: dict[tuple, set[str]] = {}
            expected_arms: dict[tuple, set[str]] = {}
            fgf_backends: dict[tuple, set[str]] = {}
            for entry in inventories:
                groups.setdefault(group_of(entry), set()).add(entry["arm"])
                if entry["scope"] == "native" and entry["bytes"] > 0:
                    expected_arms[group_of(entry)] = {
                        "fgf", "fgf-control", "gf-complete",
                    }
                    if entry["operation"] == "mul_assign":
                        expected_arms[group_of(entry)].add("reed-solomon-simd")
                    else:
                        expected_arms[group_of(entry)].update({
                            "reed-solomon-erasure", "leopard",
                        })
                else:
                    expected_arms[group_of(entry)] = {"fgf"}
                if entry["arm"] in GF16_FGF_ARMS:
                    fgf_backends.setdefault(group_of(entry), set()).add(
                        entry["backend"])
            for group, arms in groups.items():
                if arms != expected_arms[group]:
                    self.error(f"presentation group {group!r} has arms "
                               f"{sorted(arms)}, expected "
                               f"{sorted(expected_arms[group])}")
            for group, backends in fgf_backends.items():
                if len(backends) != 1:
                    self.error(f"{self.path.name}: presentation group "
                               f"{group[0]!r} reports conflicting fgf backends "
                               f"{sorted(backends)}")
        if not timed:
            return
        identities = [workload_of(record, section) for record in records]
        if len(identities) != len(set(identities)):
            self.error(f"duplicate timing records in {section}")
        if sum(placement["section"] == section for placement in self.host_placements) != 1:
            self.error(f"{section} must report exactly one host placement")
        if section == "surfaces":
            if self.surfaces_complete != len(inventories):
                self.error("surfaces timings lack an exact completion marker")
            if self.timer_floor is None:
                self.error("surfaces timings lack their timer-floor control")
        process = next(iter(self.process_evidence[section]), None)
        for record in records:
            if section == "surfaces":
                case = record["case"]
                if (record.get("control") or is_surfaces_map_case(record)
                        or case.startswith(("scalar/", "convert/", "details-math/", "query/description/"))):
                    expected = "scalar"
                elif case.startswith("bits/"):
                    if record["operation"] in {"xor-assign", "xor-range", "xor-range-prepared", "xor-gather"}:
                        expected = process
                    elif record["operation"] == "weight" and process == "v4x":
                        expected = "v4x"
                    else:
                        expected = "scalar"
                elif case in {"query/backend/tiers", "query/backend/parse"}:
                    expected = "scalar"
                elif record["field"] == "process":
                    expected = process
                else:
                    expected = banners.get(record["field"])
                if record["backend"] != expected:
                    self.error(f"{self.path.name}: case {case!r} reports backend "
                               f"{record['backend']!r} against expected {expected!r}")
        if section == "gf16-presentations":
            inventory_ids = {workload_of(entry, section)
                             for entry in inventories}
            record_ids = {workload_of(record, section) for record in records}
            full_ids = {(record["case"], record["arm"], record["backend"])
                        for record in records}
            if len(full_ids) != len(records):
                self.error(f"{self.path.name}: duplicate presentation record "
                           f"identity")
            for entry in inventories:
                marker = (entry["case"], entry["arm"], entry["backend"])
                if marker not in full_ids:
                    self.error(f"{self.path.name}: presentation case "
                               f"{entry['case']!r} ({entry['arm']!r}) has no "
                               f"record with its registered backend")
            self.validate_batch_reps(section)
        else:
            inventory_ids = {workload_of(entry, section)
                             for entry in inventories}
            record_ids = {workload_of(record, section) for record in records}
        case_at = workload_keys(section).index("case")
        if inventory_ids != record_ids:
            dropped = sorted(identity[case_at]
                             for identity in inventory_ids - record_ids)
            stray = sorted(identity[case_at]
                           for identity in record_ids - inventory_ids)
            self.error(f"{self.path.name}: inventory and records disagree in "
                       f"{section!r}; inventories without records {dropped[:4]}, "
                       f"records without inventories {stray[:4]}")
        if section == "surfaces":
            controls = [record for record in records if record.get("control")]
            phases = sorted(control["phase"] for control in controls)
            if phases != sorted(CONTROL_PHASES):
                self.error(f"{self.path.name}: the surfaces control must bracket "
                           f"the process with one entry and one exit, got {phases}")
            if len({control["case"] for control in controls}) != 2:
                self.error(f"{self.path.name}: entry and exit control records "
                           f"must carry distinct case strings")

    def validate_check_capture(self) -> None:
        """A --check capture: the full inventory, no timing records."""
        for section in self.sections:
            if self.modes.get(section) != "check":
                self.error(f"section {section!r} must run in check mode for a "
                           f"check capture, got {self.modes.get(section)!r}")
            self.validate_section(section, timed=False)
        if self.inventories["surfaces"]:
            expected = {entry["case"] for entry in self.inventories["surfaces"]}
            if (self.checked_complete != len(expected)
                    or len(self.checked_cases) != len(expected)
                    or set(self.checked_cases) != expected):
                self.error("check completion must prove execution of every registered case")
        if self.inventories["gf16-presentations"] and not self.presentations_complete[2]:
            self.error("presentations completion must identify --check execution")


# ---------------------------------------------------------------------------
# Campaign record-tree validation
# ---------------------------------------------------------------------------

def campaign_round_logs(records: Path, hosts: list[str]) -> dict:
    trees = {}
    for host in hosts:
        logs = []
        for rnd in range(1, 6):
            path = records / host / f"round-{rnd}-v3-surfaces.log"
            if not path.is_file():
                fail(f"missing surfaces round log {path}")
            logs.append((rnd, path))
        trees[host] = logs
    return trees


def parse_campaign_logs(records: Path, hosts: list[str]) -> dict:
    """Parse every available host's five surfaces round logs and check the
    process backend evidence against the campaign manifest."""
    metadata_path = records / "metadata.json"
    if not metadata_path.is_file():
        fail(f"missing campaign manifest: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"unreadable campaign manifest {metadata_path}: {exc}")
    declared = metadata.get("hosts")
    if not isinstance(declared, dict):
        fail(f"{metadata_path}: hosts must be an object")
    trees = campaign_round_logs(records, hosts)
    parsed: dict[str, list[tuple[int, SurfacesLog]]] = {}
    for host in hosts:
        details = declared.get(host)
        if not isinstance(details, dict):
            fail(f"{metadata_path}: host {host!r} is not declared")
        requested = details.get("requested_backend")
        rounds = []
        for rnd, path in trees[host]:
            parser = SurfacesLog(path)
            parser.parse(path.read_text(encoding="utf-8", errors="replace"))
            parser.require_full()
            if sorted(parser.sections) != sorted(SURFACE_SECTIONS):
                fail(f"{path.name}: a campaign round log must hold both "
                     f"{sorted(SURFACE_SECTIONS)} sections")
            for section, backend in parser.process_backends().items():
                if backend != requested:
                    fail(f"{path.name}: {section} process backend {backend!r} "
                         f"does not match the requested {requested!r}")
            for placement in parser.host_placements:
                if (placement["cpus"] != str(details.get("core"))
                        or placement["policy"] != 5):
                    fail(f"{path.name}: a surfaces-family process is not pinned to "
                         f"the declared core under SCHED_IDLE: {placement}")
            for section in parser.sections:
                parser.validate_section(section)
            rounds.append((rnd, parser))
        parsed[host] = rounds
    return parsed


def validate_campaign(records: Path, manifest: dict, universe: Universe,
                      hosts: list[str] | None = None) -> dict:
    """Validate every available host's surfaces round logs.

    Returns the per-host inventory digests and the merged unique case list.
    """
    metadata_path = records / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"unreadable campaign manifest {metadata_path}: {exc}")
    if metadata.get("rustdoc_json_sha256") != universe.sha256:
        fail("campaign rustdoc JSON digest does not match its compiler-derived API inventory")
    declared = metadata.get("hosts")
    if not isinstance(declared, dict):
        fail(f"{metadata_path}: hosts must be an object")
    available = []
    for host, details in declared.items():
        if not isinstance(details, dict) or type(details.get("available", True)) is not bool:
            fail(f"{metadata_path}: malformed host availability for {host!r}")
        if details.get("available", True):
            available.append(host)
    if not available:
        fail(f"{metadata_path}: no available measurement hosts")
    if hosts is not None and set(hosts) != set(available):
        fail("requested hosts differ from the available campaign hosts")
    hosts = available
    present = sorted(path.name for path in records.iterdir() if path.is_dir())
    if present != sorted(hosts):
        fail(f"record host directories {present} differ from available hosts {sorted(hosts)}")
    parsed = parse_campaign_logs(records, hosts)
    digests: dict[str, dict[str, str]] = {}
    workloads: dict[str, dict[str, frozenset]] = {}
    cases: list[dict] = []
    for host, rounds in parsed.items():
        round_digests = []
        section_sets: dict[str, set] = {section: set()
                                        for section in SURFACE_SECTIONS}
        for rnd, parser in rounds:
            round_digests.append({
                section: identity_digest(parser.inventories[section])
                for section in parser.sections})
            for section in parser.sections:
                section_sets[section].update(
                    workload_of(entry, section)
                    for entry in parser.inventories[section])
                validate_required_and_api(parser.inventories[section], manifest,
                                          universe, parser.path.name)
                cases.extend(parser.inventories[section])
        for index, digest_entry in enumerate(round_digests[1:], 2):
            if digest_entry != round_digests[0]:
                fail(f"{host}: surfaces inventories changed at round {index}")
        digests[host] = round_digests[0]
        workloads[host] = {section: frozenset(section_sets[section])
                           for section in SURFACE_SECTIONS}
    reference = workloads[hosts[0]]
    for host in hosts[1:]:
        if workloads[host] != reference:
            fail(f"{host}: surfaces workloads differ from {hosts[0]}")
    unique = _unique_cases(cases)
    validate_required_coverage(unique, manifest, universe)
    return {"hosts": digests, "cases": unique}


def _unique_cases(cases: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for entry in cases:
        marker = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            seen.add(marker)
            unique.append(entry)
    return unique


def validate_required_and_api(entries: list[dict], manifest: dict,
                              universe: Universe, where: str) -> None:
    for entry in entries:
        validate_case_api(entry, universe, where)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "surface_manifest.json",
                        help="supplemental foreign/operator and non-callable observations")
    parser.add_argument("--rustdoc", type=Path,
                        default=Path(__file__).resolve().parent.parent / "target/doc/fgf.json",
                        help="pinned nightly rustdoc JSON for the measured source")
    parser.add_argument("--records", type=Path, default=None,
                        help="campaign records root with per-host directories")
    parser.add_argument("--check-log", type=Path, action="append", default=[],
                        help="captured --check or --presentations --check stdout; "
                             "repeatable")
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        universe = public_universe(arguments.rustdoc, manifest)
        union: list[dict] = []
        inputs = 0
        if arguments.records is not None:
            result = validate_campaign(arguments.records, manifest, universe)
            union.extend(result["cases"])
            inputs += 1
        for path in arguments.check_log:
            log = SurfacesLog(path)
            log.parse(path.read_text(encoding="utf-8", errors="replace"))
            log.validate_check_capture()
            for section in log.sections:
                validate_required_and_api(log.inventories[section], manifest,
                                          universe, path.name)
                union.extend(log.inventories[section])
            inputs += 1
        if inputs == 0:
            parser.error("nothing to validate: pass --records or --check-log")
        validate_required_coverage(_unique_cases(union), manifest, universe)
    except PublishError as exc:
        print(f"check-surfaces: {exc}", file=sys.stderr)
        return 1
    print(f"check-surfaces: {inputs} input tree(s) validated against the "
          f"declared public API")
    return 0


if __name__ == "__main__":
    sys.exit(main())
