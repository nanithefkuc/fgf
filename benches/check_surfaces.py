#!/usr/bin/env python3
"""Validate surfaces benchmark check captures against the declared public API.

    python3 benches/check_surfaces.py --manifest benches/surface_manifest.json \
        --rustdoc target/doc/fgf.json [--check-log LOG]...

The surfaces harness registers every measured case once as a
``SURFACE_INVENTORY`` line. A ``--check`` run executes every registered
case's correctness oracle without timing and prints one ``SURFACE_CHECKED``
line per case plus a completion marker. This tool is that capture's gate. It
fails closed on a case whose ``api`` paths do not resolve against the crate's
declared stable public API (plus the manifest's explicit additions), on a
compiler-declared callable that no case covers, on duplicate case strings or
duplicate identities, and on a capture whose mode or completion markers do
not prove a full ``--check`` execution.

The public callable universe comes from pinned nightly rustdoc JSON, including
macro-generated methods and reachable re-exports. The supplemental manifest
names only foreign/operator spellings and non-callable observations; it never
defines the required fgf-owned callable set.

``--check-log`` validates one captured ``--check`` /
``--presentations --check`` stdout. Required-API coverage spans the union of
all given logs. Every failure exits nonzero with a diagnostic; nothing is
silently dropped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class CheckError(Exception):
    """A fail-closed condition: the input does not match the checker contract."""


def fail(message: str) -> None:
    raise CheckError(message)


def normalize_backend(name: str) -> str:
    """Normalize enum Debug spellings to Backend.name() tags."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


# Invocation chatter that the recipes echo into captured logs. These lines
# carry no measurement content; any other unrecognized line fails closed.
NOISE_PREFIXES = (
    "cargo ", "taskset ", "chrt ", "echo ", "printf ",
    "warning:", "Compiling ", "Finished ", "Running ", "Blocking ", "Downloading ",
    "Downloaded ", "pinned to core", "── ",
)


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return stripped.startswith(NOISE_PREFIXES) or stripped.split(" ")[0] in ("just", "cargo")


# The frozen CHECK statements the native GF16 harness prints after its
# differential validation passes; an altered statement fails closed.
GF16_CHECK_LINES = frozenset({
    "CHECK field maps: all 65536 values, all 256 basis products; native scalar oracles; "
    "all-value regions with zero/unit/full-field coefficients; repeated calls; legal tile "
    "boundaries and offset alignment passed.",
    "CHECK every measured geometry and coefficient schedule passed exact differential validation.",
})

# Sections in their fixed order inside one log; the surfaces section is
# implicit from the first record, the presentations section starts at its
# process header.
SURFACE_SECTIONS = ("surfaces", "gf16-presentations")
SURFACE_SCOPES = ("setup", "steady", "query", "end_to_end")
PRESENTATION_SCOPES = ("native", "end_to_end", "setup")
CONTROL_PHASES = ("entry", "exit")
# Dispatch tags a current artifact may report for an fgf arm or inventory
# entry.
KNOWN_BACKENDS = frozenset({"v2", "v3", "v3_gfni_crypto", "v4x", "scalar"})

# Surfaces-section payload: no backend, no timing; `control`/`phase` are
# always present.
SURFACE_INVENTORY_KEYS = ("api", "bytes", "case", "control", "destinations",
                          "field", "lanes", "operation", "phase", "scope",
                          "sources")
# Presentations payloads carry arm and backend.
GF16_INVENTORY_KEYS = ("api", "arm", "backend", "bytes", "case",
                       "destinations", "field", "lanes", "operation", "scope",
                       "sources")
GF16_FGF_ARMS = ("fgf", "fgf-control")
# One BENCHMARK_MODE marker follows each process header. A check capture
# declares check; the other markers parse so a non-check log fails on its
# mode rather than as an unrecognized line.
BENCHMARK_MODES = ("full", "smoke", "check")
SURFACES_ONLY_MODES = ("inventory",)

SURFACES_BANNER_RE = re.compile(r"fgf surfaces benchmark — backend: (\S+)")
SURFACES_PROCESS_RECORD_RE = re.compile(
    r"FIELD_BACKEND\tprocess\t(\S+)\tnative_mul=(true|false)"
    r"\tblocked_rows=(true|false)")
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
# `end_to_end` conversion caller loops are map work too.
MAP_CASE_PREFIX = "maps/"


def is_surfaces_map_case(obj: dict) -> bool:
    return obj["case"].startswith(MAP_CASE_PREFIX) or any(
        canonical_api(path).startswith(("Embedding::", "Isomorphism::"))
        for path in obj["api"]
    )


def identity_keys(section: str) -> tuple:
    if section == "gf16-presentations":
        return GF16_INVENTORY_KEYS
    return SURFACE_INVENTORY_KEYS


def workload_keys(section: str) -> tuple:
    if section == "gf16-presentations":
        return tuple(key for key in GF16_INVENTORY_KEYS if key != "backend")
    return SURFACE_INVENTORY_KEYS


def workload_of(obj: dict, section: str) -> tuple:
    """Case identity without backend or timing; comparable across hosts."""
    return tuple(obj[key] if key != "api" else tuple(obj[key])
                 for key in workload_keys(section))


def group_of(obj: dict) -> tuple:
    """Group matching arms while preserving schedule and every case parameter."""
    keys = tuple(key for key in GF16_INVENTORY_KEYS
                 if key not in ("api", "arm", "backend", "case"))
    return tuple(obj[key] for key in keys) + (obj["case"].rsplit("/", 1)[0],)


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


def surface_object(raw: dict, *, section: str, where: str) -> dict:
    """Validate one SURFACE_INVENTORY JSON object."""
    if not isinstance(raw, dict):
        fail(f"{where}: inventory payloads must be JSON objects")
    present = set(raw)
    required = set(identity_keys(section))
    missing = sorted(required - present)
    unknown = sorted(present - required)
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
    return raw


# ---------------------------------------------------------------------------
# Public API universe
# ---------------------------------------------------------------------------

from api_inventory import canonical_api, declared_callables


class Universe:
    """Compiler-declared callables and explicit supplemental observations."""

    def __init__(self, names: frozenset, literal: frozenset) -> None:
        self.names = names
        self.literal = literal

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
        document = json.loads(rustdoc.read_bytes())
        names = frozenset(declared_callables(document))
    except (OSError, ValueError, KeyError) as exc:
        fail(f"unreadable compiler API inventory {rustdoc}: {exc}")
    literal = frozenset(manifest["items"])
    redundant = sorted(path for path in literal if canonical_api(path) in names)
    if redundant:
        fail(f"supplemental manifest duplicates compiler-declared APIs: {redundant}")
    return Universe(names, literal)


def validate_case_api(obj: dict, universe: Universe, where: str) -> None:
    if obj.get("control") is True:
        return  # the byte-XOR control is not a public API call
    for path in obj["api"]:
        try:
            universe.resolve(path)
        except CheckError as exc:
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
    """Parse one surfaces-family ``--check`` capture.

    The log opens in the implicit surfaces section; the presentations section
    starts at its ``GF16 presentations;`` process header.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.section = "surfaces"
        self.inventories: dict[str, list[dict]] = {section: []
                                                   for section in SURFACE_SECTIONS}
        self.banners: list[dict] = []
        self.modes: dict[str, str] = {}
        self.process_evidence: dict[str, set[str]] = {section: set()
                                                      for section in SURFACE_SECTIONS}
        self.presentations_header: dict | None = None
        self.presentations_complete: tuple[int, int, bool] | None = None
        self.checked_cases: list[str] = []
        self.checked_complete: int | None = None
        self.check_lines: list[str] = []
        self._banner_fields: dict[str, list[str]] = {section: []
                                                     for section in SURFACE_SECTIONS}

    def error(self, message: str) -> None:
        fail(f"{self.path.name}: {message}")

    def parse(self, text: str) -> None:
        for lineno, raw in enumerate(text.splitlines(), 1):
            try:
                self.line(raw)
            except CheckError:
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
        for prefix, handler in (
            ("BENCHMARK_MODE\t", self.mode_line),
            ("FIELD_BACKEND\t", self.banner_line),
            ("SURFACE_INVENTORY\t", self.inventory_line),
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

    def inventory_line(self, line: str) -> None:
        _, _, payload = line.partition("\t")
        if not payload.strip():
            self.error(f"malformed inventory: {line!r}")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            self.error(f"malformed inventory JSON: {exc}")
        section = ("gf16-presentations" if self.section == "gf16-presentations"
                   else "surfaces")
        record = surface_object(raw, section=section,
                                where=f"{self.path.name}: inventory")
        self.inventories[section].append(record)

    def check_line(self, line: str) -> None:
        if line not in GF16_CHECK_LINES:
            self.error(f"altered surfaces CHECK statement: {line!r}")
        self.check_lines.append(line)

    def host_line(self, line: str) -> None:
        if HOST_RECORD_RE.fullmatch(line) is None:
            self.error(f"malformed HOST record: {line!r}")

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
        if not any(self.inventories[section] or self.banners
                   for section in SURFACE_SECTIONS):
            self.error("log holds no surfaces content")
        present = set(self.sections)
        for section in SURFACE_SECTIONS:
            if section in present and section not in self.modes:
                self.error(f"section {section!r} lacks its benchmark mode "
                           f"marker")
            if section in self.modes and section not in present:
                self.error(f"benchmark mode marker without {section!r} content")
        for section in SURFACE_SECTIONS:
            if self._banners_of(section) and not self.inventories[section]:
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
                if self.inventories[section] or self._banners_of(section)]

    def _banners_of(self, section: str) -> dict[str, str]:
        return {banner["label"]: banner["backend"] for banner in self.banners
                if banner["section"] == section}

    # -- per-log invariants ----------------------------------------------------

    def validate_section(self, section: str) -> None:
        """Structural checks over one section's registered inventories."""
        inventories = self.inventories[section]
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

    def validate_check_capture(self) -> None:
        """A --check capture: the full inventory, no timing records."""
        for section in self.sections:
            if self.modes.get(section) != "check":
                self.error(f"section {section!r} must run in check mode for a "
                           f"check capture, got {self.modes.get(section)!r}")
            self.validate_section(section)
        if self.inventories["surfaces"]:
            expected = {entry["case"] for entry in self.inventories["surfaces"]}
            if (self.checked_complete != len(expected)
                    or len(self.checked_cases) != len(expected)
                    or set(self.checked_cases) != expected):
                self.error("check completion must prove execution of every registered case")
        if self.inventories["gf16-presentations"] and not self.presentations_complete[2]:
            self.error("presentations completion must identify --check execution")


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
    parser.add_argument("--check-log", type=Path, action="append", default=[],
                        help="captured --check or --presentations --check stdout; "
                             "repeatable")
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        universe = public_universe(arguments.rustdoc, manifest)
        union: list[dict] = []
        inputs = 0
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
            parser.error("nothing to validate: pass --check-log")
        validate_required_coverage(_unique_cases(union), manifest, universe)
    except CheckError as exc:
        print(f"check-surfaces: {exc}", file=sys.stderr)
        return 1
    print(f"check-surfaces: {inputs} input log(s) validated against the "
          f"declared public API")
    return 0


if __name__ == "__main__":
    sys.exit(main())
