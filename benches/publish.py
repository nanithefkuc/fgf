#!/usr/bin/env python3
"""Generate version snapshots and paired comparisons from raw paired-run records.

    python3 benches/publish.py --records PATH --output PATH [--verify]

The records directory holds one subdirectory per private host identifier,
each produced by ``just bench-paired``: ``complete.txt``,
``environment.txt``, ``order.tsv``, and
``round-{1..5}-{v2|v3}-{gf|gdl|m31|gf2}.log``. ``PATH/metadata.json`` is the
parent-supplied campaign manifest; it must declare the host list, and may
carry source revisions, toolchain facts, and per-host backend expectations.

Generation fails closed: a missing or incomplete run, a nonfinite or
nonpositive value, a malformed or duplicate workload key, or a row set that
changes across rounds is an error, never a silently dropped measurement.

Outputs, under ``--output``: ``v2/<family>.md``, ``v3/<family>.md``, and
``comparison/<family>.md`` for the six family pages. The raw-record side
gains ``evidence.json`` with every normalized case, per-host per-round
samples, aggregates, backend banners, controls, availability, and the page,
table, row, and column of every published cell. ``--verify`` regenerates the
expected bytes and compares them against the files on disk without writing.

No previously published number is an input; the generator reads only the raw
records and the metadata manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

GIB = 1024.0**3
ROUNDS = (1, 2, 3, 4, 5)
VERSIONS = ("v2", "v3")
FAMILIES = ("gf", "gdl", "m31", "gf2")
# Capture-family identifiers preserve the immutable raw record filenames.
PAGE_FAMILIES = ("gf8", "gf16", "wide-binary", "gf1", "mersenne31", "goldilocks")
ROUND_FAMILY = {
    "gf8": "gf", "gf16": "gf", "wide-binary": "gf", "gf1": "gf2",
    "mersenne31": "m31", "goldilocks": "gdl",
}
FAMILY_UNITS = {
    "gf": ("self", "trio", "gf16"),
    "gdl": ("self", "prime"),
    "m31": ("self", "prime"),
    "gf2": ("self",),
}
NATIVE_GF16_ARMS = ("gf-complete", "reed-solomon-erasure", "reed-solomon-simd", "leopard")
GF16_ARM_DISPLAY = {
    "fgf": "`fgf`",
    "gf-complete": "GF-Complete",
    "reed-solomon-erasure": "`reed-solomon-erasure`",
    "reed-solomon-simd": "`reed-solomon-simd`",
    "leopard": "Leopard-RS 1.x",
}

# Canonical field labels shared by both adapted measurement harnesses.
FIELDS = (
    "gf8b", "gf8d", "gf16b", "gf16d", "gf32b", "gf64b",
    "fp16", "fp32", "fp64", "gf1", "m31", "qm31", "gld",
)
FIELD_FAMILY = {
    "gf8b": "gf8", "gf8d": "gf8",
    "gf16b": "gf16", "gf16d": "gf16",
    "gf32b": "wide-binary", "gf64b": "wide-binary",
    "fp16": "wide-binary", "fp32": "wide-binary", "fp64": "wide-binary",
    "gf1": "gf1",
    "m31": "mersenne31", "qm31": "mersenne31",
    "gld": "goldilocks",
}
PAGE_TITLES = {
    "gf8": "Byte fields: `gf8b`, `gf8d`",
    "gf16": "Degree-16 fields: `gf16b`, `gf16d`",
    "wide-binary": "Wider binary fields: `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64`",
    "gf1": "Bit-packed GF(2): `gf1`",
    "mersenne31": "Mersenne fields: `m31`, `qm31`",
    "goldilocks": "Goldilocks: `gld`",
}
CANONICAL_FIELD_RE = r"(gf8b|gf8d|gf16b|gf16d|gf32b|gf64b|fp16|fp32|fp64|m31|qm31|gld)"

# Fields each campaign family may produce. Anything else inside a family
# round log is unsupported parser input.
FAMILY_FIELDS = {
    "gf": {"gf8b", "gf8d", "gf16b", "gf16d", "gf32b", "gf64b", "fp16", "fp32", "fp64"},
    "gdl": {"gld"},
    "m31": {"m31", "qm31"},
    "gf2": {"gf1", "gf8b"},
}

THROUGHPUT = "throughput"
LATENCY = "latency"
SCALAR_METRIC = "scalar"
ROWPAIR = "rowpair"
METRIC_UNIT = {THROUGHPUT: "GiB/s", LATENCY: "ns/op", SCALAR_METRIC: "ns/op",
               ROWPAIR: "ns/row pair"}
# ns values arrive with nine decimals; derived cells inherit that resolution.
SOURCE_DECIMALS = 9

TRIO_CONTROL_LABEL = "control: mul_into vs itself"
PRIME_CONTROL_LABEL = "control: mul_into vs itself"
KERNEL_CONTROL_LABELS = ("control byte xor start", "control byte xor end")

GF16_HEADER_RE = re.compile(
    r"^GF16 native field comparison; fgf=(\S+) backend=(\S+) field_backend=(\S+); "
    r"rss=([\d.]+) AVX2; rse=([\d.]+) scalar GF16; "
    r"GF-Complete=([\d.]+) (\S+) SSSE3 split-4/16 polynomial=0x1100b; "
    r"Leopard-RS 1\.x \(pre-Leopard2\)=(\S+) AVX2$"
)
TRIO_ISAL_RE = re.compile(r"^ISA-L (\S+) \(runtime dispatch\)")
TRIO_KP_RE = re.compile(r"^klauspost/reedsolomon (\S+) via ([^,]+), GOMAXPROCS (\d+)")
PRIME_P3_RE = re.compile(r"^Plonky3 (\S+), built")
PRIME_HEADING_RE = re.compile(
    r"^(Mersenne31|Goldilocks|QuadMersenne31) — (\d+) B regions, (\d+) B lanes$")
PRIME_FIELD_OF = {"Mersenne31": "m31", "Goldilocks": "gld", "QuadMersenne31": "qm31"}
PRIME_FAMILY_CONTROL_FIELD = {"gdl": "gld", "m31": "m31"}
GF16_CHECK_LINES = frozenset({
    "CHECK field maps: all 65536 values, all 256 basis products; native scalar oracles; "
    "all-value regions with zero/unit/full-field coefficients; repeated calls; legal tile "
    "boundaries and offset alignment passed.",
    "CHECK every measured geometry and coefficient schedule passed exact differential validation.",
})


class PublishError(Exception):
    """A fail-closed condition: the records do not match the input contract."""


def fail(message: str) -> None:
    raise PublishError(message)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _round_half_up(value: float, decimals: int) -> Decimal:
    return Decimal(repr(float(value))).quantize(
        Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _int_digits(value: float) -> int:
    if value >= 1.0:
        return len(str(int(value)))
    return 1  # the single leading zero


def check_positive(value: float, what: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        fail(f"{what} must be finite and positive, got {value!r}")


def table_digits(values: list[float]) -> int:
    """Choose a total digit count retaining three significant figures."""
    positive = [v for v in values if v is not None and math.isfinite(v) and v > 0.0]
    if not positive:
        fail("digit count requested for a table with no positive values")
    smallest = min(positive)
    exponent = math.floor(math.log10(smallest))
    rounded = float(_round_half_up(smallest, max(0, 2 - exponent)))
    digits = _int_digits(rounded) + max(0, 2 - math.floor(math.log10(rounded)))
    return max(digits, max(
        _int_digits(float(_round_half_up(v, max(0, digits - _int_digits(v)))))
        for v in positive))


def _format_digits(value: float, digits: int, cap: int | None = None) -> str:
    integer_digits = _int_digits(value)
    decimals = max(0, digits - integer_digits)
    if cap is not None:
        decimals = min(decimals, cap)
    rounded = _round_half_up(value, decimals)
    if _int_digits(float(rounded)) > integer_digits:
        decimals = max(0, digits - _int_digits(float(rounded)))
        if cap is not None:
            decimals = min(decimals, cap)
        rounded = _round_half_up(value, decimals)
    if rounded <= 0:
        fail(f"published value {value!r} is below the supported output precision")
    return f"{rounded:.{decimals}f}"


def fmt_value(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    check_positive(value, "published value")
    return _format_digits(value, digits, SOURCE_DECIMALS)


def fmt_ratio(value: float | None) -> str:
    """Ratios print five digits."""
    if value is None:
        return "-"
    check_positive(value, "published ratio")
    return _format_digits(value, 5)


def fmt_pair(a: float | None, b: float | None, digits: int) -> str:
    if a is None and b is None:
        return "-"
    return f"{fmt_value(a, digits)} / {fmt_value(b, digits)}"


def fmt_ratio_pair(a: float | None, b: float | None) -> str:
    if a is None and b is None:
        return "-"
    return f"{fmt_ratio(a)} / {fmt_ratio(b)}"


def fmt_bytes(num: int) -> str:
    for threshold, unit in ((1024**4, "TiB"), (1024**3, "GiB"),
                            (1024**2, "MiB"), (1024, "KiB")):
        if num >= threshold and num % threshold == 0:
            return f"{num // threshold} {unit}"
    return f"{num} B"


def parse_size(text: str) -> int:
    m = re.fullmatch(r"(\d+) (KiB|MiB|GiB|TiB)", text)
    if m:
        scale = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}[m.group(2)]
        return int(m.group(1)) * scale
    fail(f"unparseable size token: {text!r}")


def gib_per_sec(logical_bytes: int | None, ns: float) -> float:
    if logical_bytes is None or logical_bytes <= 0:
        fail("throughput requested for a case with no logical byte count")
    return logical_bytes / ns * 1e9 / GIB


def median(values: list[float]) -> float:
    if not values:
        fail("median of an empty sample list")
    for value in values:
        check_positive(value, "sample value")
    return statistics.median(values)

def normalize_backend(name: str) -> str:
    """Normalize enum Debug spellings to Backend.name() tags."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()



def normalize_label(label: str) -> str:
    return " ".join(label.split())


def field_from_tokens(label: str) -> str | None:
    for token in label.split():
        if token in FIELD_FAMILY:
            return token
    return None


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------

@dataclass
class Rec:
    """One normalized measurement record from one round log."""
    kind: str          # sample | scalar | trio | prime | result | control16
    suite: str         # kernels | compare | trio | prime | gf16
    panel: str
    heading: str
    label: str         # normalized, field suffix stripped
    raw_label: str
    field: str
    metric: str
    logical_bytes: int | None
    ns: float | None
    geometry: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    public: bool = True
    disposition: str = "public"


@dataclass
class VersionSeries:
    rounds: dict[int, float] = field(default_factory=dict)

    def add(self, rnd: int, value: float) -> None:
        if rnd in self.rounds:
            fail(f"duplicate round {rnd} for one case")
        check_positive(value, "per-run value")
        self.rounds[rnd] = value

    def agg(self) -> float:
        return median([self.rounds[r] for r in sorted(self.rounds)])


@dataclass
class CaseData:
    """One workload case across every host, version, and round."""
    first: Rec
    values: dict = field(default_factory=dict)      # (host, version) -> VersionSeries
    byte_counts: dict = field(default_factory=dict)  # (host, version, round) -> bytes

    def add(self, host: str, version: str, rnd: int, rec: Rec, where: str) -> None:
        slot = (host, version)
        if slot in self.values and rnd in self.values[slot].rounds:
            fail(f"{where}: duplicate workload key in one round: {rec.raw_label!r}")
        if self.first.logical_bytes != rec.logical_bytes:
            fail(f"{where}: logical byte count for {rec.raw_label!r} changed across records")
        self.values.setdefault(slot, VersionSeries()).add(rnd, rec.ns)
        self.byte_counts[slot + (rnd,)] = rec.logical_bytes


def case_key(rec: Rec) -> tuple:
    geometry = tuple(sorted((k, _jsonable(v)) for k, v in rec.geometry.items()))
    return (rec.kind, rec.suite, rec.panel, rec.label, rec.field, rec.metric, geometry)


def _jsonable(value):
    if isinstance(value, list):
        return tuple(_jsonable(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

KERNELS_PANEL_RES = [
    (re.compile(r"^preparation crossover — one-shot vs prepared, by row length:$"), "crossover"),
    (re.compile(r"^custom scalar-routed tower gf16d — GF\(2\^16\)/0x11D, A=2 B=0x80:$"), "crossover"),
    (re.compile(r"^small-row GF\(2\^16\) multi-row shapes — preparation-dominated:$"), "smallrow"),
    (re.compile(r"^add_assign_rows — (.+):$"), "addrows"),
    (re.compile(r"^large destinations — mul_into store policy:$"), "largedst"),
    (re.compile(r"^destination alignment — multi-row scatter:$"), "destalign"),
    (re.compile(r"^blocked vs AXPY — direct GF\(2\^16\) kernel calls \(dispatch bypassed\):$"), "blocked"),
    (re.compile(r"^network-size GF\(2\^8\) payloads:$"), "network"),
    (re.compile(r"^bit-packed GF\(2\) — packed vs byte-per-element control:$"), "gf2bits"),
    (re.compile(r"^scatter/matrix — (\d+) rows x (\d+) KiB:$"), "scattermatrix"),
    (re.compile(r"^Tier 3 field cost — (\d+) bytes:$"), "tier3"),
]
KERNELS_GEOM_RES = [
    (re.compile(r"^  row (\d+) B:$"), "row_bytes"),
    (re.compile(r"^  row (\d+) B, (\d+) rows, (\d+) sources:$"), "smallrow"),
    (re.compile(r"^  payload (\d+) B:$"), "payload"),
    (re.compile(r"^  (\d+ [KMGT]iB) buffers \((\d+) elements\):$"), "bitsbuf"),
    (re.compile(r"^  (\d+) row pairs, 16-byte rows, bits 61\.\.69:$"), "rowpairs"),
    (re.compile(r"^buffer (\d+ [KMGT]iB):$"), "buffer"),
]
ADDROWS_SUMMARY_RE = re.compile(r"^\s+rows/flat: .*\((\d+) rows x (\d+) B\)$")
COMPARE_PANEL_RES = [
    (re.compile(r"^(\d+) B x (\d+) sources overwrite dot product \(raw\)$"), "dotraw"),
    (re.compile(r"^(\d+) B x (\d+) sources overwrite dot product$"), "dotprep"),
    (re.compile(r"^(\d+) B x (\d+) sources -> (\d+) rows encode$"), "encode"),
    (re.compile(r"^gf8d mul_add 16 KiB destination offset sweep \(dst/src shift mod 64\)$"), "offsweep"),
    (re.compile(r"^gf8d peel floor sweep \(row bytes, base offset mod 64\)$"), "peelsweep"),
]
TRIO_HEADING_RES = [
    (re.compile(r"^(\d+) B single source$"), "trio-single"),
    (re.compile(r"^(\d+) B x (\d+) sources -> 1 row$"), "trio-gather"),
    (re.compile(r"^(\d+) B x (\d+) sources -> (\d+) rows$"), "trio-encode"),
]
GF16_LABEL_SECTIONS = {
    "mul_assign": "In-place scaling",
    "mul_into": "Single-row multiplication",
    "mul_add": "Single-row multiplication",
    "mul_into_gather_with": "Multi-row multiplication",
    "mul_add_gather_with": "Multi-row multiplication",
    "mul_into_matrix_with": "Multi-row multiplication",
    "mul_add_matrix_with": "Multi-row multiplication",
}
GF16_OP_DISPLAY = {
    "mul_assign": "dst = c * dst",
    "mul_into": "dst = c * src",
    "mul_add": "dst ^= c * src",
    "mul_into_gather_with": "dst = Σ c[i] * src[i]",
    "mul_add_gather_with": "dst ^= Σ c[i] * src[i]",
    "mul_into_matrix_with": "dst = Σ c[i][r] * src[i]",
    "mul_add_matrix_with": "dst ^= Σ c[i][r] * src[i]",
}
GF16_ARMS_OK = frozenset({"fgf", "fgf-control", *NATIVE_GF16_ARMS})

# Invocation chatter that the recipes echo into round logs. These lines carry
# no measurement content; any other unrecognized line fails closed.
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


# The human timing rows and summary ratios the harnesses print beside the
# machine records. Their values are lower-precision duplicates of SAMPLE,
# TRIO, and PRIME records, so they are tolerated but never parsed.
HUMAN_ROW_RES = [
    re.compile(r"^\s*\S.*GiB/s"),
    re.compile(r"^\s*\S.*Mops/s"),
    re.compile(r"^\s*per call: one-shot .*\)$"),
    re.compile(r"^\s*(one-shot/prepared|ssse3 blocked/AXPY|gfni  blocked/AXPY|"
               r"copy\+scale/fused): \S"),
]


def is_human_row(line: str) -> bool:
    return any(pattern.match(line) for pattern in HUMAN_ROW_RES)


class LogParser:
    """Parse one round log into records, banners, controls, and harness state."""

    def __init__(self, family: str, path: Path) -> None:
        self.family = family
        self.path = path
        self.records: list[Rec] = []
        self.banners: list[dict] = []
        self.process_backends: list[dict] = []
        self.host_placement: list[dict] = []
        self.unit_sequence: list[str] = []
        self.trio_header: dict = {}
        self.prime_header: dict = {}
        self.gf16_header: dict = {}
        self.controls: list[dict] = []
        self.check_lines: list[str] = []
        self.unit: str | None = None
        self.suite: str | None = None
        self.panel: str | None = None
        self.heading = ""
        self.geom: dict = {}
        self.addrows_field: str | None = None
        self.addrows_pending: list[Rec] = []
        self.trio_control_seen = False
        self.prime_headings_seen = 0

    def error(self, message: str) -> None:
        fail(f"{self.path.name}: {message}")

    # -- record routing ----------------------------------------------------------

    def parse(self, text: str) -> None:
        for lineno, raw in enumerate(text.splitlines(), 1):
            try:
                self.line(raw)
            except PublishError:
                raise
            except Exception as exc:
                fail(f"{self.path.name}:{lineno}: unsupported parser input {raw!r}: {exc}")
        if self.addrows_pending:
            self.error("add_assign_rows panel ended without its trailing geometry summary")
        if sorted(self.unit_sequence) != sorted(FAMILY_UNITS[self.family]):
            self.error(f"unit sequence {self.unit_sequence} does not match "
                       f"{FAMILY_UNITS[self.family]}")

    def line(self, raw: str) -> None:
        for prefix, handler in (
            ("UNIT\t", self.unit_line),
            ("HOST\t", self.host_line),
            ("FIELD_BACKEND\t", self.banner_line),
            ("SAMPLE\t", self.sample_line),
            ("SCALAR\t", self.scalar_line),
            ("TRIO\t", self.trio_line),
            ("PRIME\t", self.prime_line),
            ("RESULT\t", self.result_line),
            ("CONTROL\t", self.control_line),
            ("CHECK ", self.check_line),
        ):
            if raw.startswith(prefix):
                handler(raw)
                return
        self.context_line(raw)

    def unit_line(self, line: str) -> None:
        unit = line.split("\t", 1)[1].strip()
        if unit not in FAMILY_UNITS[self.family]:
            self.error(f"unexpected unit {unit!r}")
        if unit in self.unit_sequence:
            self.error(f"unit {unit!r} appears twice")
        if self.addrows_pending:
            self.error("add_assign_rows panel ended without its trailing geometry summary")
        self.unit = unit
        self.unit_sequence.append(unit)
        self.suite = "kernels" if unit == "self" else unit
        self.panel = None
        self.heading = ""
        self.geom = {}
        self.addrows_field = None
        self.trio_control_seen = False
        self.prime_headings_seen = 0

    def host_line(self, line: str) -> None:
        match = re.fullmatch(
            r"HOST\tcpus_allowed_list=([0-9,-]+)\tsched_policy=(\d+) (\S+)", line)
        if self.unit != "self" or match is None:
            self.error(f"malformed self HOST record: {line!r}")
        self.host_placement.append({
            "suite": self.suite, "cpus": match.group(1),
            "policy": int(match.group(2)), "policy_name": match.group(3),
        })

    def banner_line(self, line: str) -> None:
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
            self.error(f"malformed FIELD_BACKEND record: {line!r}")
        self.banners.append({"unit": self.unit, "suite": self.suite, "label": parts[1].strip(),
                             "backend": parts[2].strip()})

    def sample_line(self, line: str) -> None:
        parts = line.split("\t")
        if len(parts) != 4:
            self.error(f"malformed SAMPLE record: {line!r}")
        label, raw_bytes, raw_ns = parts[1], parts[2], parts[3]
        try:
            logical_bytes = int(raw_bytes)
            ns = float(raw_ns)
        except ValueError:
            self.error(f"malformed SAMPLE record: {line!r}")
        if logical_bytes < 0:
            self.error(f"negative SAMPLE byte count: {line!r}")
        check_positive(ns, "SAMPLE ns value")
        if label.strip() in KERNEL_CONTROL_LABELS:
            self.controls.append({
                "kind": "byte-xor",
                "position": "start" if label.strip() == KERNEL_CONTROL_LABELS[0] else "end",
                "unit": self.unit, "suite": self.suite,
                "logical_bytes": logical_bytes, "ns": ns,
            })
            return
        self.self_record("sample", label, logical_bytes, ns)

    def scalar_line(self, line: str) -> None:
        parts = line.split("\t")
        if len(parts) != 3:
            self.error(f"malformed SCALAR record: {line!r}")
        label, raw_ns = parts[1], parts[2]
        try:
            ns = float(raw_ns)
        except ValueError:
            self.error(f"malformed SCALAR record: {line!r}")
        check_positive(ns, "SCALAR ns value")
        if label.strip() in KERNEL_CONTROL_LABELS:
            self.controls.append({
                "kind": "byte-xor",
                "position": "start" if label.strip() == KERNEL_CONTROL_LABELS[0] else "end",
                "unit": self.unit, "suite": self.suite, "logical_bytes": None, "ns": ns,
            })
            return
        self.self_record("scalar", label, None, ns)

    def self_record(self, kind: str, raw_label: str, logical_bytes: int | None,
                    ns: float) -> None:
        if self.suite not in ("kernels", "compare"):
            self.error(f"{kind.upper()} record outside the self suite: {raw_label!r}")
        if kind == "scalar":
            build_regions(self, kind, raw_label, normalize_label(raw_label), None, ns)
            return
        if self.panel is None:
            self.error(f"{kind.upper()} record before any panel heading: {raw_label!r}")
        builder = SELF_BUILDERS.get((self.suite, self.panel))
        if builder is None:
            self.error(f"no record grammar for suite={self.suite} panel={self.panel}: "
                       f"{raw_label!r}")
        builder(self, kind, raw_label, normalize_label(raw_label), logical_bytes, ns)

    # -- competitor records --------------------------------------------------------

    def trio_line(self, line: str) -> None:
        if self.unit != "trio":
            self.error(f"TRIO record outside the trio unit: {line!r}")
        parts = line.split("\t")
        if len(parts) != 10:
            self.error(f"malformed TRIO record: {line!r}")
        label = normalize_label(parts[1])
        try:
            logical_bytes = int(parts[2])
            fgf, isal, kp = (float(p) for p in parts[3:6])
            lo_isal, hi_isal, lo_kp, hi_kp = (float(p) for p in parts[6:10])
        except ValueError:
            self.error(f"malformed TRIO record: {line!r}")
        for name, value in (("fgf", fgf), ("isal", isal), ("klauspost", kp)):
            check_positive(value, f"TRIO {name} median")
        for name, lo, hi in (("isal", lo_isal, hi_isal), ("klauspost", lo_kp, hi_kp)):
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or lo > hi:
                self.error(f"malformed TRIO {name} ratio band: {line!r}")
        if logical_bytes <= 0:
            self.error(f"nonpositive TRIO byte count: {line!r}")
        if label == TRIO_CONTROL_LABEL:
            if self.trio_control_seen:
                self.error("duplicate trio control row")
            if self.panel is not None:
                self.error("the trio control row must precede every panel heading")
            self.trio_control_seen = True
            self.controls.append({"kind": "trio", "label": label,
                                  "logical_bytes": logical_bytes,
                                  "fgf": fgf, "isal": isal, "klauspost": kp})
            return
        if not self.trio_control_seen:
            self.error("the trio control row is missing before the first measured row")
        if self.panel is None:
            self.error(f"TRIO record before any panel heading: {parts[1]!r}")
        self.records.append(Rec(
            kind="trio", suite="trio", panel=self.panel, heading=self.heading,
            label=label, raw_label=parts[1], field="gf8d", metric=THROUGHPUT,
            logical_bytes=logical_bytes, ns=fgf, geometry=dict(self.geom),
            extra={"isal": isal, "klauspost": kp,
                   "isal_p10": lo_isal, "isal_p90": hi_isal,
                   "klauspost_p10": lo_kp, "klauspost_p90": hi_kp}))

    def prime_line(self, line: str) -> None:
        if self.unit != "prime":
            self.error(f"PRIME record outside the prime unit: {line!r}")
        parts = line.split("\t")
        if len(parts) != 8:
            self.error(f"malformed PRIME record: {line!r}")
        label = normalize_label(parts[1])
        kind = parts[2]
        try:
            logical_bytes = int(parts[3])
            fgf, p3, lo, hi = (float(p) for p in parts[4:8])
        except ValueError:
            self.error(f"malformed PRIME record: {line!r}")
        if kind not in ("rate", "scalar"):
            self.error(f"unknown PRIME row kind {kind!r}: {line!r}")
        for name, value in (("fgf", fgf), ("p3", p3)):
            check_positive(value, f"PRIME {name} median")
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or lo > hi:
            self.error(f"malformed PRIME ratio band: {line!r}")
        if kind == "rate" and logical_bytes <= 0:
            self.error(f"nonpositive PRIME byte count: {line!r}")
        if kind == "scalar" and logical_bytes != 0:
            self.error(f"scalar PRIME row carries a byte count: {line!r}")
        if label == PRIME_CONTROL_LABEL:
            if self.prime_headings_seen == 0:
                prime_field = PRIME_FAMILY_CONTROL_FIELD[self.family]
            else:
                prime_field = self.geom.get("field")
                if prime_field is None:
                    self.error(f"PRIME control row after a non-field heading: {line!r}")
            self.controls.append({"kind": "prime", "label": label, "field": prime_field,
                                  "logical_bytes": logical_bytes, "fgf": fgf, "p3": p3})
            return
        if self.panel is None:
            self.error(f"PRIME record before any field heading: {parts[1]!r}")
        self.records.append(Rec(
            kind="prime", suite="prime", panel=self.panel, heading=self.heading,
            label=label, raw_label=parts[1], field=self.geom["field"],
            metric=THROUGHPUT if kind == "rate" else SCALAR_METRIC,
            logical_bytes=logical_bytes or None, ns=fgf, geometry=dict(self.geom),
            extra={"p3": p3, "p10": lo, "p90": hi}))

    def result_line(self, line: str) -> None:
        if self.unit != "gf16":
            self.error(f"RESULT record outside the gf16 unit: {line!r}")
        parts = line.split("\t")
        if len(parts) != 12:
            self.error(f"malformed RESULT record: {line!r}")
        label = normalize_label(parts[1])
        try:
            sources, destinations, row_bytes = int(parts[2]), int(parts[3]), int(parts[4])
            ns, gib, p10, p90 = (float(p) for p in parts[7:11])
            sample_count = int(parts[11])
        except ValueError:
            self.error(f"malformed RESULT record: {line!r}")
        arm, schedule = parts[6], parts[5]
        if label not in GF16_LABEL_SECTIONS:
            self.error(f"unsupported gf16 harness label: {parts[1]!r}")
        if arm not in GF16_ARMS_OK:
            self.error(f"unknown gf16 harness arm: {arm!r}")
        if schedule not in ("fixed", "cycling16"):
            self.error(f"unknown gf16 coefficient schedule: {schedule!r}")
        for name, value in (("ns", ns), ("gib", gib), ("p10", p10), ("p90", p90)):
            check_positive(value, f"gf16 RESULT {name}")
        if not p10 <= ns <= p90:
            self.error(f"gf16 RESULT median outside its decile band: {line!r}")
        if sample_count < 1:
            self.error(f"gf16 RESULT sample count below one: {line!r}")
        self.records.append(Rec(
            kind="result", suite="gf16", panel=GF16_LABEL_SECTIONS[label], heading="",
            label=label, raw_label=parts[1], field="gf16b", metric=THROUGHPUT,
            logical_bytes=row_bytes * sources, ns=ns,
            geometry={"sources": sources, "destinations": destinations,
                      "row_bytes": row_bytes, "schedule": schedule},
            extra={"arm": arm, "gib": gib, "p10": p10, "p90": p90,
                   "sample_count": sample_count}))

    def control_line(self, line: str) -> None:
        if self.unit != "gf16":
            self.error(f"CONTROL record outside the gf16 unit: {line!r}")
        parts = line.split("\t")
        if len(parts) != 7:
            self.error(f"malformed CONTROL record: {line!r}")
        label = normalize_label(parts[1])
        try:
            sources, destinations, row_bytes = int(parts[2]), int(parts[3]), int(parts[4])
            ratio = float(parts[6])
        except ValueError:
            self.error(f"malformed CONTROL record: {line!r}")
        if label not in GF16_LABEL_SECTIONS:
            self.error(f"unsupported gf16 control label: {parts[1]!r}")
        if parts[5] not in ("fixed", "cycling16"):
            self.error(f"unknown gf16 coefficient schedule: {parts[5]!r}")
        check_positive(ratio, "gf16 CONTROL ratio")
        self.records.append(Rec(
            kind="control16", suite="gf16", panel=GF16_LABEL_SECTIONS[label], heading="",
            label=label, raw_label=parts[1], field="gf16b", metric=THROUGHPUT,
            logical_bytes=row_bytes * sources, ns=ratio,
            geometry={"sources": sources, "destinations": destinations,
                      "row_bytes": row_bytes, "schedule": parts[5]},
            extra={"ratio": ratio}, public=False, disposition="control"))

    def check_line(self, line: str) -> None:
        if self.unit != "gf16":
            self.error(f"CHECK statement outside the gf16 unit: {line!r}")
        if line not in GF16_CHECK_LINES:
            self.error(f"altered gf16 CHECK statement: {line!r}")
        self.check_lines.append(line)

    # -- context lines ---------------------------------------------------------------

    def context_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        if self.unit == "trio":
            self._trio_context(stripped)
        elif self.unit == "prime":
            self._prime_context(stripped)
        elif self.unit == "gf16":
            self._gf16_context(stripped)
        elif self.unit == "self":
            self._self_context(line)
        elif is_noise(line):
            return  # invocation echo before the first unit marker
        else:
            self.error(f"unsupported line outside any unit: {line!r}")

    def _trio_context(self, stripped: str) -> None:
        m = re.fullmatch(r"fgf — process backend (\S+), .+ backend (\S+)", stripped)
        if m:
            self.process_backends.append({"unit": "trio", "suite": "trio", "backend": m.group(1)})
            self.trio_header.update({"backend": m.group(1), "field_backend": m.group(2)})
            return
        for pattern, panel in TRIO_HEADING_RES:
            m = pattern.match(stripped)
            if m:
                self.panel = panel
                self.heading = stripped
                self.geom = {"row_bytes": int(m.group(1))}
                if panel == "trio-gather":
                    self.geom["sources"], self.geom["destinations"] = int(m.group(2)), 1
                elif panel == "trio-encode":
                    self.geom["sources"] = int(m.group(2))
                    self.geom["destinations"] = int(m.group(3))
                return
        m = TRIO_ISAL_RE.match(stripped)
        if m:
            self.trio_header["isal"] = m.group(1)
            return
        m = TRIO_KP_RE.match(stripped)
        if m:
            self.trio_header.update({"klauspost": m.group(1), "go": m.group(2),
                                     "gomaxprocs": int(m.group(3))})
            return
        if stripped.startswith(("fgf —", "[schedule:", "[ratios:", "shape")):
            return
        if is_human_row(stripped) or is_noise(stripped):
            return
        self.error(f"unsupported trio line: {stripped!r}")

    def _prime_context(self, stripped: str) -> None:
        m = re.fullmatch(
            r"fgf — process backend (\S+), m31 (\S+), gld (\S+), qm31 (\S+)", stripped)
        if m:
            self.process_backends.append({"unit": "prime", "suite": "prime", "backend": m.group(1)})
            self.prime_header.update({
                "backend": m.group(1), "m31_backend": m.group(2),
                "gld_backend": m.group(3), "qm31_backend": m.group(4),
            })
            return
        m = PRIME_HEADING_RE.match(stripped)
        if m:
            self.panel = f"prime-{PRIME_FIELD_OF[m.group(1)]}"
            self.heading = stripped
            self.geom = {"field": PRIME_FIELD_OF[m.group(1)],
                         "region_bytes": int(m.group(2)), "lane_bytes": int(m.group(3))}
            self.prime_headings_seen += 1
            return
        m = PRIME_P3_RE.match(stripped)
        if m:
            self.prime_header["p3"] = m.group(1)
            widths = re.search(r"packing widths: m31 (\d+), gld (\d+), extension (\d+)", stripped)
            if widths is None:
                self.error(f"missing native Plonky3 packing widths: {stripped!r}")
            self.prime_header["packing_widths"] = {
                label: int(width) for label, width in zip(("m31", "gld", "qm31"), widths.groups())
            }
            return
        if stripped.startswith(("fgf —", "ratio =", "[low-high]", "shape",
                                "[Plonky3 packing")):
            return
        if is_human_row(stripped) or is_noise(stripped):
            return
        self.error(f"unsupported prime line: {stripped!r}")

    def _gf16_context(self, stripped: str) -> None:
        if stripped in GF16_CHECK_LINES:
            return
        m = GF16_HEADER_RE.match(stripped)
        if m:
            self.gf16_header = {
                "fgf": m.group(1), "backend": normalize_backend(m.group(2)),
                "field_backend": normalize_backend(m.group(3)),
                "reed_solomon_simd": m.group(4), "reed_solomon_erasure": m.group(5),
                "gf_complete": m.group(6), "gf_complete_revision": m.group(7),
                "leopard": m.group(8),
            }
            self.process_backends.append({"unit": "gf16", "suite": "gf16", "backend": normalize_backend(m.group(2))})
            return
        if stripped.startswith(("affinity=", "Native layouts;")):
            return
        if is_noise(stripped):
            return
        self.error(f"unsupported gf16 line: {stripped!r}")

    def _self_context(self, line: str) -> None:
        stripped = line.strip()
        if stripped == "control byte xor — drift check:":
            return
        m = re.fullmatch(r"fgf kernel benchmark — backend: (\S+)", stripped)
        if m:
            self.suite = "kernels"
            self.panel = None
            self.heading = ""
            self.geom = {}
            self.process_backends.append({"unit": "self", "suite": "kernels", "backend": m.group(1)})
            return
        m = re.fullmatch(r"fgf — backend: (\S+)", stripped)
        if m:
            self.suite = "compare"
            self.panel = "regions"
            self.heading = ""
            self.geom = {}
            self.process_backends.append({"unit": "self", "suite": "compare", "backend": m.group(1)})
            return
        if self.suite == "kernels":
            for pattern, panel in KERNELS_PANEL_RES:
                m = pattern.match(line)
                if m:
                    self._kernel_panel(panel, m, line)
                    return
            for pattern, kind in KERNELS_GEOM_RES:
                m = pattern.match(line)
                if m:
                    self._kernel_geometry(kind, m, line)
                    return
            m = ADDROWS_SUMMARY_RE.match(line)
            if m:
                self._addrows_summary(m)
                return
            if stripped.startswith("(override with SIMD_BACKEND"):
                return
        elif self.suite == "compare":
            for pattern, panel in COMPARE_PANEL_RES:
                match = pattern.match(stripped)
                if match:
                    if self.addrows_pending:
                        self.error("add_assign_rows panel ended without its trailing summary")
                    self.panel = panel
                    self.heading = stripped
                    self.geom = {}
                    if panel in ("dotraw", "dotprep", "encode"):
                        self.geom = {
                            "row_bytes": int(match.group(1)),
                            "sources": int(match.group(2)),
                            "destinations": int(match.group(3)) if panel == "encode" else 1,
                            "alignment": 64,
                        }
                    return
        if stripped.startswith(("(override with SIMD_BACKEND", "pinned to core")):
            return
        if is_human_row(stripped):
            return
        if stripped.startswith(("just", "cargo", "taskset", "chrt", "warning:",
                                "Compiling", "Finished", "Running", "Blocking",
                                "Downloading", "Downloaded", "pinned to core", "── ")):
            return
        self.error(f"unsupported line in the self suite: {line!r}")

    def _kernel_panel(self, panel: str, m: re.Match, line: str) -> None:
        if self.addrows_pending:
            self.error("add_assign_rows panel ended without its trailing geometry summary")
        self.panel = panel
        self.heading = line.strip()
        self.geom = {}
        if panel == "addrows":
            name = normalize_label(m.group(1))
            field = field_from_tokens(name)
            if field is None:
                self.error(f"cannot resolve the field of the add_assign_rows panel: {name!r}")
            self.addrows_field = field
        elif panel == "scattermatrix":
            self.geom = {"rows": int(m.group(1)), "row_kib": int(m.group(2))}
        elif panel == "tier3":
            self.geom = {"bytes": int(m.group(1))}

    def _kernel_geometry(self, kind: str, m: re.Match, line: str) -> None:
        if kind == "buffer":
            self.panel = "buffers"
        elif kind == "rowpairs":
            self.panel = "gf2short"
        if self.panel is None:
            self.error(f"geometry heading before any panel: {line!r}")
        self.heading = line.strip()
        if kind == "row_bytes":
            self.geom = {"row_bytes": int(m.group(1))}
        elif kind == "smallrow":
            self.geom = {"row_bytes": int(m.group(1)), "rows": int(m.group(2)),
                         "sources": int(m.group(3))}
        elif kind == "payload":
            self.geom = {"payload_bytes": int(m.group(1))}
        elif kind == "bitsbuf":
            self.geom = {"buffer": m.group(1), "buffer_bytes": parse_size(m.group(1)),
                         "bits": int(m.group(2))}
        elif kind == "rowpairs":
            self.geom = {"row_pairs": int(m.group(1)), "row_bytes": 16, "bits": [61, 69]}
        elif kind == "buffer":
            self.geom = {"buffer": m.group(1), "buffer_bytes": parse_size(m.group(1))}

    def _addrows_summary(self, m: re.Match) -> None:
        if self.panel != "addrows" or not self.addrows_pending:
            self.error("row-count summary outside an add_assign_rows panel")
        rows, row_len = int(m.group(1)), int(m.group(2))
        if len(self.addrows_pending) != 3:
            self.error(f"add_assign_rows trio holds {len(self.addrows_pending)} samples, "
                       f"expected 3")
        labels = sorted(rec.label for rec in self.addrows_pending)
        if labels != ["add_assign", "add_assign per row", "add_assign_rows"]:
            self.error(f"unexpected add_assign_rows trio composition: {labels}")
        byte_counts = {rec.logical_bytes for rec in self.addrows_pending}
        if len(byte_counts) != 1:
            self.error("the add_assign_rows trio reports different logical byte counts")
        for rec in self.addrows_pending:
            rec.geometry = {"rows": rows, "row_bytes": row_len}
            self.records.append(rec)
        self.addrows_pending = []


# ---------------------------------------------------------------------------
# Self-record builders: strict per-panel label grammars.
# ---------------------------------------------------------------------------

def _push(parser: LogParser, rec: Rec) -> None:
    if rec.field not in FAMILY_FIELDS[parser.family]:
        parser.error(f"field {rec.field!r} is not part of family {parser.family!r}: "
                     f"{rec.raw_label!r}")
    parser.records.append(rec)


def _push_self(parser: LogParser, kind: str, raw_label: str, label: str, field_name: str,
               panel: str, metric: str, nbytes: int | None, ns: float,
               geometry: dict, disposition: str = "public") -> None:
    stripped = label
    if stripped.endswith(f" {field_name}"):
        stripped = stripped[: -len(field_name) - 1]
    _push(parser, Rec(
        kind=kind, suite=parser.suite, panel=panel, heading=parser.heading,
        label=stripped, raw_label=raw_label, field=field_name, metric=metric,
        logical_bytes=nbytes, ns=ns, geometry=geometry, public=True,
        disposition=disposition))


def _field_of(parser: LogParser, label: str) -> str:
    found = field_from_tokens(label)
    if found is None:
        parser.error(f"cannot resolve the field of record: {label!r}")
    return found


def build_crossover(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    m = re.fullmatch(r"mul_add (one-shot|prepared) " + CANONICAL_FIELD_RE, label)
    if not m:
        parser.error(f"unsupported crossover label: {raw!r}")
    op = "mul_add" if m.group(1) == "one-shot" else "mul_add_with"
    _push_self(parser, kind, raw, op, f, "crossover", LATENCY, nbytes, ns, dict(parser.geom))


def build_smallrow(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    for prefix, op in (("scatter", "mul_add_scatter"), ("gather", "mul_add_gather"),
                       ("matrix", "mul_add_matrix"), ("mul_into fused", "mul_into"),
                       ("mul_into copy+scale", "mul_into copy+scale")):
        if label == f"{prefix} {f}":
            _push_self(parser, kind, raw, op, f, "smallrow", THROUGHPUT, nbytes, ns,
                       dict(parser.geom))
            return
    parser.error(f"unsupported small-row label: {raw!r}")


def build_addrows(parser, kind, raw, label, nbytes, ns):
    if parser.addrows_field is None:
        parser.error(f"add_assign_rows sample before the panel heading: {raw!r}")
    op_map = {"flat add_assign": "add_assign", "add_assign_rows": "add_assign_rows",
              "add_assign per row": "add_assign per row"}
    if label not in op_map:
        parser.error(f"unsupported add_assign_rows label: {raw!r}")
    disposition = ("row-wise addition" if label == "add_assign_rows"
                   else "public call composition")
    parser.addrows_pending.append(Rec(
        kind=kind, suite="kernels", panel="addrows", heading=parser.heading,
        label=op_map[label], raw_label=raw, field=parser.addrows_field, metric=THROUGHPUT,
        logical_bytes=nbytes, ns=ns, geometry={}, public=True, disposition=disposition))


def build_largedst(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    m = re.fullmatch(r"(\d+) MiB (mul_into|mul_into\+read|mul_add) " + CANONICAL_FIELD_RE,
                     label)
    if not m:
        parser.error(f"unsupported large-destination label: {raw!r}")
    op = {"mul_into": "mul_into", "mul_into+read": "mul_into + read",
          "mul_add": "mul_add"}[m.group(2)]
    geometry = {"buffer_bytes": int(m.group(1)) * 1024 * 1024}
    _push_self(parser, kind, raw, op, f, "largedst", THROUGHPUT, nbytes, ns, geometry)


def build_destalign(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    m = re.fullmatch(r"(\d+) KiB rows, skew (\d+) " + CANONICAL_FIELD_RE, label)
    if not m:
        parser.error(f"unsupported destination-alignment label: {raw!r}")
    geometry = {"row_bytes": int(m.group(1)) * 1024, "skew": int(m.group(2))}
    _push_self(parser, kind, raw, "mul_add_scatter", f, "destalign", THROUGHPUT, nbytes,
               ns, geometry)


def build_blocked(parser, kind, raw, label, nbytes, ns):
    # Direct `internals` kernel calls: retained in the evidence, never published.
    m = re.fullmatch(r"gather (blocked|AXPY) (ssse3|gfni)", label)
    if not m:
        parser.error(f"unsupported blocked-vs-AXPY label: {raw!r}")
    _push(parser, Rec(
        kind=kind, suite="kernels", panel="blocked", heading=parser.heading,
        label=f"gather {m.group(1)} ({m.group(2)})", raw_label=raw, field="gf16b",
        metric=THROUGHPUT, logical_bytes=nbytes, ns=ns, geometry=dict(parser.geom),
        public=False, disposition="direct kernel"))


def build_network(parser, kind, raw, label, nbytes, ns):
    # Network labels carry no field suffix; they measure gf8b by definition.
    m = re.fullmatch(r"(xor|mul_add|mul_assign|scatter \((\d+) rows\))", label)
    if not m:
        parser.error(f"unsupported network label: {raw!r}")
    if m.group(1) == "xor":
        op, geometry = "add_assign", dict(parser.geom)
    elif m.group(1) in ("mul_add", "mul_assign"):
        op, geometry = m.group(1), dict(parser.geom)
    else:
        op, geometry = "mul_add_scatter", {**parser.geom, "rows": int(m.group(2))}
    _push_self(parser, kind, raw, op, "gf8b", "network", THROUGHPUT, nbytes, ns, geometry)


def build_gf2bits(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(
        r"bits::(xor_assign|and_into|weight|dot_product|xor_range 5/8|XorRange 5/8) packed",
        label)
    if m:
        _push_self(parser, kind, raw, "bits::" + m.group(1), "gf1", "gf2bits", THROUGHPUT,
                   nbytes, ns, dict(parser.geom))
        return
    m = re.fullmatch(r"ops::add_assign (" + CANONICAL_FIELD_RE + r") control", label)
    if m:
        _push_self(parser, kind, raw, "add_assign", m.group(1), "gf2control", THROUGHPUT,
                   nbytes, ns, dict(parser.geom))
        return
    parser.error(f"unsupported bit-packed GF(2) label: {raw!r}")


def build_gf2short(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(r"bits::(xor_range|XorRange) 8-bit row", label)
    if not m:
        parser.error(f"unsupported GF(2) short-range label: {raw!r}")
    pairs = parser.geom.get("row_pairs", 0)
    if pairs <= 0:
        parser.error("GF(2) short-range record without its row-pair geometry")
    _push_self(parser, kind, raw, "bits::" + m.group(1), "gf1", "gf2short", ROWPAIR,
               nbytes, ns / pairs, dict(parser.geom))


BUFFER_OPS = {
    "xor": "add_assign", "add_assign": "add_assign",
    "add_assign_scalar": "add_assign_scalar", "sub_assign_scalar": "sub_assign_scalar",
    "mul_add": "mul_add", "mul_add prepared": "mul_add_with",
    "mul_into": "mul_into", "mul_assign": "mul_assign",
    "elementwise": "mul_elementwise", "elementwise_assign": "mul_elementwise_assign",
}


def build_buffers(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    if not label.endswith(f" {f}"):
        parser.error(f"unsupported buffer-panel label: {raw!r}")
    op_text = label[: -len(f) - 1]
    if op_text not in BUFFER_OPS:
        parser.error(f"unsupported buffer-panel operation: {raw!r}")
    _push_self(parser, kind, raw, BUFFER_OPS[op_text], f, "buffers", THROUGHPUT, nbytes,
               ns, dict(parser.geom))


SCATTERMATRIX_OPS = {
    "scatter": "mul_add_scatter",
    "scatter prepared": "mul_add_scatter_with",
    "matrix (selected)": "mul_add_matrix",
    "matrix prepared": "mul_add_matrix_with",
    "gather (selected)": "mul_add_gather",
    "gather prepared": "mul_add_gather_with",
    "scatter (unblocked)": None,
    "matrix (unblocked AXPY)": None,
    "gather (unblocked)": None,
}


def build_scattermatrix(parser, kind, raw, label, nbytes, ns):
    f = _field_of(parser, label)
    for prefix in SCATTERMATRIX_OPS:
        if label == f"{prefix} {f}":
            break
    else:
        parser.error(f"unsupported scatter/matrix label: {raw!r}")
    prefix = label[: -(len(f) + 1)]
    op = SCATTERMATRIX_OPS[prefix]
    if op is None:
        _push(parser, Rec(
            kind=kind, suite="kernels", panel="scattermatrix", heading=parser.heading,
            label=prefix, raw_label=raw, field=f, metric=THROUGHPUT,
            logical_bytes=nbytes, ns=ns, geometry=dict(parser.geom), public=False,
            disposition="unblocked composition"))
        return
    _push_self(parser, kind, raw, op, f, "scattermatrix", THROUGHPUT, nbytes, ns,
               dict(parser.geom))


def build_tier3(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(
        r"mul_add (?:polynomial tower|canonical Fan-Paar) (" + CANONICAL_FIELD_RE + r")",
        label)
    if not m:
        parser.error(f"unsupported Tier 3 label: {raw!r}")
    _push_self(parser, kind, raw, "mul_add", m.group(1), "tier3", THROUGHPUT, nbytes, ns,
               dict(parser.geom))


def build_regions(parser, kind, raw, label, nbytes, ns):
    if kind == "scalar":
        m = re.fullmatch(CANONICAL_FIELD_RE + r" scalar mul", label)
        if not m:
            parser.error(f"unsupported scalar label: {raw!r}")
        _push_self(parser, kind, raw, "scalar mul", m.group(1), "scalar", SCALAR_METRIC,
                   None, ns, {})
        return
    m = re.fullmatch(CANONICAL_FIELD_RE + r" "
                     r"(mul_add|mul_into|elementwise|elementwise assign|"
                     r"mul_into 4MiB) \(dst [^)]+\)", label)
    if m:
        op = {"elementwise": "mul_elementwise", "elementwise assign": "mul_elementwise_assign",
              "mul_into 4MiB": "mul_into"}.get(m.group(2), m.group(2))
        _push_self(parser, kind, raw, op, m.group(1), "regions", THROUGHPUT, nbytes, ns,
                   {"row_bytes": nbytes})
        return
    m = re.fullmatch(CANONICAL_FIELD_RE + r" assign (\d+)B", label)
    if m:
        _push_self(parser, kind, raw, "mul_assign", m.group(1), "assign-sweep", THROUGHPUT,
                   nbytes, ns, {"buffer_bytes": int(m.group(2))})
        return
    m = re.fullmatch(r"scatter 4x16K (raw|prepared)", label)
    if m:
        op = "mul_add_scatter" if m.group(1) == "raw" else "mul_add_scatter_with"
        _push_self(parser, kind, raw, op, "gf8d", "destalign-compare", THROUGHPUT, nbytes,
                   ns, {"rows": 4, "row_bytes": 16 * 1024})
        return
    parser.error(f"unsupported compare region label: {raw!r}")


def build_dot(parser, kind, raw, label, nbytes, ns):
    match = re.fullmatch(
        r"fgf (gf8b|gf8d) (mul_into_gather|mul_add_gather_with|mul_into_gather_with)", label)
    if match is None:
        parser.error(f"unsupported overwrite dot-product label: {raw!r}")
    _push_self(parser, kind, raw, match.group(2), match.group(1), "dot", THROUGHPUT,
               nbytes, ns, dict(parser.geom))


def build_encode(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(r"fgf (gf8b|gf8d) (.+)", label)
    if not m:
        parser.error(f"unsupported encode label: {raw!r}")
    f, op = m.group(1), m.group(2)
    op_map = {
        "fill + mul_add_matrix": "fill(0) + mul_add_matrix",
        "mul_into_matrix": "mul_into_matrix",
        "mul_into_matrix_with": "mul_into_matrix_with",
        "mul_add_matrix_with": "mul_add_matrix_with",
        "mul_add_matrix_at": "mul_add_matrix_at",
    }
    if op not in op_map:
        parser.error(f"unsupported encode operation: {raw!r}")
    _push_self(parser, kind, raw, op_map[op], f, "encode", THROUGHPUT, nbytes, ns,
               dict(parser.geom))


def build_offsweep(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(r"gf8d mul_add off=(\d+)", label)
    if m:
        _push_self(parser, kind, raw, "mul_add", "gf8d", "offsweep", THROUGHPUT, nbytes,
                   ns, {"offset": int(m.group(1)), "row_bytes": nbytes})
        return
    build_regions(parser, kind, raw, label, nbytes, ns)


def build_peelsweep(parser, kind, raw, label, nbytes, ns):
    m = re.fullmatch(r"scatter 4x(\d+) off=(\d+)", label)
    if m:
        _push_self(parser, kind, raw, "mul_add_scatter", "gf8d", "peel", THROUGHPUT,
                   nbytes, ns, {"row_bytes": int(m.group(1)), "offset": int(m.group(2)),
                                "rows": 4})
        return
    m = re.fullmatch(r"gather 16x(\d+) off=(\d+)", label)
    if m:
        _push_self(parser, kind, raw, "mul_add_gather", "gf8d", "peel", THROUGHPUT,
                   nbytes, ns, {"row_bytes": int(m.group(1)), "offset": int(m.group(2)),
                                "sources": 16})
        return
    m = re.fullmatch(r"matrix 10->4x(\d+) off=(\d+)", label)
    if m:
        _push_self(parser, kind, raw, "mul_add_matrix", "gf8d", "peel", THROUGHPUT,
                   nbytes, ns, {"row_bytes": int(m.group(1)), "offset": int(m.group(2)),
                                "sources": 10, "rows": 4})
        return
    parser.error(f"unsupported peel-floor label: {raw!r}")


SELF_BUILDERS = {
    ("kernels", "crossover"): build_crossover,
    ("kernels", "smallrow"): build_smallrow,
    ("kernels", "addrows"): build_addrows,
    ("kernels", "largedst"): build_largedst,
    ("kernels", "destalign"): build_destalign,
    ("kernels", "blocked"): build_blocked,
    ("kernels", "network"): build_network,
    ("kernels", "gf2bits"): build_gf2bits,
    ("kernels", "gf2short"): build_gf2short,
    ("kernels", "buffers"): build_buffers,
    ("kernels", "scattermatrix"): build_scattermatrix,
    ("kernels", "tier3"): build_tier3,
    ("compare", "regions"): build_regions,
    ("compare", "dotraw"): build_dot,
    ("compare", "dotprep"): build_dot,
    ("compare", "encode"): build_encode,
    ("compare", "offsweep"): build_offsweep,
    ("compare", "peelsweep"): build_peelsweep,
}


# ---------------------------------------------------------------------------
# Campaign loading, validation, and aggregation
# ---------------------------------------------------------------------------

class Campaign:
    def __init__(self, records: Path) -> None:
        self.records_root = records
        self.metadata = self._load_metadata(records / "metadata.json")
        self.hosts: list[str] = list(self.metadata["hosts"])
        self.host_labels = dict(zip(self.hosts, (
            "Willow Cove (i5-1135G7)", "Golden Cove (i7-12700K)"), strict=True))
        self.cases: dict[tuple, CaseData] = {}
        self.trio: dict[tuple, dict] = {}
        self.prime: dict[tuple, dict] = {}
        self.gf16: dict[tuple, dict] = {}
        self.gf16_controls: dict[tuple, dict] = {}
        self.log_evidence: list[dict] = []
        self.order_rows: dict[str, list] = {}
        self.environments: dict[str, dict] = {}
        self.aggregates: dict[tuple, float] = {}

    # -- loading -----------------------------------------------------------------

    def _load_metadata(self, path: Path) -> dict:
        if not path.is_file():
            fail(f"missing metadata manifest: {path}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"unreadable metadata manifest {path}: {exc}")
        if not isinstance(metadata, dict) or "hosts" not in metadata:
            fail("metadata manifest must be a JSON object declaring 'hosts'")
        hosts = metadata["hosts"]
        if (not isinstance(hosts, dict)
                or [details.get("cpu") if isinstance(details, dict) else None
                    for details in hosts.values()]
                != ["Intel Core i5-1135G7", "Intel Core i7-12700K"]):
            fail("metadata hosts must declare Willow Cove then Golden Cove CPUs")
        if metadata.get("host_order") != list(hosts) or metadata.get("rounds") != 5:
            fail("metadata must fix the agreed host order and five complete rounds")
        for host, details in hosts.items():
            if (not isinstance(details, dict)
                    or not isinstance(details.get("core"), int)
                    or details["core"] < 0
                    or not isinstance(details.get("requested_backend"), str)):
                fail(f"metadata host {host!r} needs a core and requested_backend")
        return metadata

    def load(self) -> None:
        present = sorted(p.name for p in self.records_root.iterdir() if p.is_dir())
        if sorted(self.hosts) != present:
            fail(f"record host directories {present} do not match declared hosts "
                 f"{sorted(self.hosts)}")
        for host in self.hosts:
            self._load_host(host)
        self._validate_row_sets()
        self._aggregate()

    def _load_host(self, host: str) -> None:
        root = self.records_root / host
        complete = root / "complete.txt"
        environment = root / "environment.txt"
        order = root / "order.tsv"
        for path in (complete, environment, order):
            if not path.is_file():
                fail(f"missing {path}")
        if complete.read_text(encoding="utf-8").strip() != "CAMPAIGN_COMPLETE":
            fail(f"{complete} does not read CAMPAIGN_COMPLETE; the campaign is incomplete")
        text = environment.read_text(encoding="utf-8", errors="replace")
        self.environments[host] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        self.order_rows[host] = self._parse_order(order)
        for rnd in ROUNDS:
            for version in VERSIONS:
                for family in FAMILIES:
                    path = root / f"round-{rnd}-{version}-{family}.log"
                    if not path.is_file():
                        fail(f"missing round log {path}")
                    self._parse_log(host, rnd, version, family, path)

    def _parse_order(self, path: Path) -> list[dict]:
        rows = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                fail(f"{path}:{lineno}: an order row has three columns: {line!r}")
            rnd, family, version = parts
            if (rnd not in {str(r) for r in ROUNDS} or family not in FAMILIES
                    or version not in VERSIONS):
                fail(f"{path}:{lineno}: malformed order row: {line!r}")
            rows.append({"round": int(rnd), "family": family, "version": version})
        keys = [(row["round"], row["family"], row["version"]) for row in rows]
        if len(keys) != len(set(keys)):
            fail(f"{path}: duplicate order rows")
        expected = {(rnd, family, version)
                    for rnd in ROUNDS for family in FAMILIES for version in VERSIONS}
        if set(keys) != expected:
            fail(f"{path}: order rows do not cover every round, family, and version "
                 f"exactly once")
        return rows

    def _parse_log(self, host: str, rnd: int, version: str, family: str, path: Path) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            fail(f"empty round log: {path}")
        parser = LogParser(family, path)
        parser.parse(text)
        self._check_semantics(parser, host, version, path)
        self.log_evidence.append({
            "host": host, "round": rnd, "version": version, "family": family,
            "file": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "units": parser.unit_sequence, "banners": parser.banners,
            "process_backends": parser.process_backends,
            "host_placement": parser.host_placement,
            "trio_header": parser.trio_header, "prime_header": parser.prime_header,
            "gf16_header": parser.gf16_header, "controls": parser.controls,
        })
        self._collect(parser, host, version, rnd)

    def _check_semantics(self, parser: LogParser, host: str, version: str,
                         path: Path) -> None:
        if not parser.banners:
            fail(f"{path}: no FIELD_BACKEND banner; the run is incomplete")
        if "gf16" in parser.unit_sequence and set(parser.check_lines) != GF16_CHECK_LINES:
            fail(f"{path}: the gf16 harness CHECK statements are missing or altered")
        details = self.metadata["hosts"][host]
        expected = details["requested_backend"]
        suites = {"kernels", "compare"} if parser.family == "gf" else {"kernels"}
        expected_processes = suites | (set(parser.unit_sequence) - {"self"})
        reported = [banner["suite"] for banner in parser.process_backends]
        if sorted(reported) != sorted(expected_processes):
            fail(f"{path}: missing or duplicate process backend banners: {reported}")
        for banner in parser.process_backends:
            if banner["backend"] != expected:
                fail(f"{path}: resolved process backend {banner['backend']!r} "
                     f"does not match requested {expected!r}")
        if sorted(item["suite"] for item in parser.host_placement) != sorted(suites):
            fail(f"{path}: missing or duplicate self process placement records")
        for placement in parser.host_placement:
            if placement["cpus"] != str(details["core"]) or placement["policy"] != 5:
                fail(f"{path}: process is not core-pinned under SCHED_IDLE: {placement}")
        for suite in suites:
            positions = [control["position"] for control in parser.controls
                         if control["kind"] == "byte-xor" and control["suite"] == suite]
            if sorted(positions) != ["end", "start"]:
                fail(f"{path}: {suite} must have exactly one entry and exit byte-XOR control")
            fields = [banner["label"] for banner in parser.banners if banner["suite"] == suite]
            expected_fields = set(FIELDS) - {"gf1", "gf16d"}
            if version == "v3" and suite == "kernels":
                expected_fields.add("gf16d")
            if sorted(fields) != sorted(expected_fields):
                fail(f"{path}: missing or duplicate {suite} field backend banners: {fields}")
        for banner in parser.banners:
            if banner["label"] == "gf16d" and banner["backend"] != "scalar":
                fail(f"{path}: custom gf16d fixture must report its scalar route")
        if "trio" in parser.unit_sequence and not any(c["kind"] == "trio"
                                                      for c in parser.controls):
            fail(f"{path}: the trio harness control row is missing")
        if "prime" in parser.unit_sequence and not any(c["kind"] == "prime"
                                                       for c in parser.controls):
            fail(f"{path}: the prime harness control row is missing")
        if "gf16" in parser.unit_sequence:
            arms = {r.extra["arm"] for r in parser.records if r.kind == "result"}
            missing = [arm for arm in NATIVE_GF16_ARMS if arm not in arms]
            if missing:
                fail(f"{path}: native gf16 arms absent from the run: {missing}")

    def _collect(self, parser: LogParser, host: str, version: str, rnd: int) -> None:
        where = parser.path.name
        for rec in parser.records:
            key = case_key(rec)
            if rec.kind in ("trio", "prime") or (
                    rec.kind == "result" and rec.extra["arm"] == "fgf"):
                self.cases.setdefault(key, CaseData(first=rec)).add(
                    host, version, rnd, rec, where)
            if rec.kind in ("sample", "scalar"):
                self.cases.setdefault(key, CaseData(first=rec)).add(
                    host, version, rnd, rec, where)
            elif rec.kind == "trio":
                entry = self.trio.setdefault(key, {
                    "meta": rec,
                    "arms": {"fgf": {}, "isal": {}, "klauspost": {}},
                    "bands": {},
                })
                values = {"fgf": rec.ns, "isal": rec.extra["isal"],
                          "klauspost": rec.extra["klauspost"]}
                for arm, value in values.items():
                    series = entry["arms"][arm].setdefault((host, version), {})
                    if rnd in series:
                        fail(f"{where}: duplicate trio case in one round: {rec.raw_label!r}")
                    series[rnd] = value
                entry["bands"][(host, version, rnd)] = {
                    arm: (rec.extra[f"{arm}_p10"], rec.extra[f"{arm}_p90"])
                    for arm in ("isal", "klauspost")}
            elif rec.kind == "prime":
                entry = self.prime.setdefault(key, {
                    "meta": rec, "arms": {"fgf": {}, "p3": {}}, "bands": {},
                })
                for arm, value in (("fgf", rec.ns), ("p3", rec.extra["p3"])):
                    series = entry["arms"][arm].setdefault((host, version), {})
                    if rnd in series:
                        fail(f"{where}: duplicate prime case in one round: "
                             f"{rec.raw_label!r}")
                    series[rnd] = value
                entry["bands"][(host, version, rnd)] = (rec.extra["p10"], rec.extra["p90"])
            elif rec.kind == "result":
                arms = self.gf16.setdefault(key, {"meta": rec, "arms": {}})
                series = arms["arms"].setdefault((host, version, rec.extra["arm"]), {})
                if rnd in series:
                    fail(f"{where}: duplicate gf16 case in one round: {rec.raw_label!r}")
                series[rnd] = rec.ns
            elif rec.kind == "control16":
                controls = self.gf16_controls.setdefault(
                    key, {"meta": rec, "ratios": {}})
                ratios = controls["ratios"].setdefault((host, version), {})
                if rnd in ratios:
                    fail(f"{where}: duplicate gf16 control in one round: {rec.raw_label!r}")
                ratios[rnd] = rec.extra["ratio"]

    def _log_case_keys(self, entry: dict) -> set:
        """Re-derive the case keys of one log by reparsing it."""
        parser = LogParser(entry["family"], Path(entry["file"]))
        parser.parse(Path(entry["file"]).read_text(encoding="utf-8", errors="replace"))
        keys = set()
        for rec in parser.records:
            key = case_key(rec)
            if rec.kind == "result":
                keys.add(key + (rec.extra["arm"],))
            else:
                keys.add(key)
        return keys

    def _validate_row_sets(self) -> None:
        """Row sets must be identical across rounds within one host, version,
        and family."""
        grouped: dict[tuple, dict[int, set]] = {}
        for entry in self.log_evidence:
            log_id = (entry["host"], entry["version"], entry["family"])
            keys = grouped.setdefault(log_id, {}).setdefault(entry["round"], set())
            keys.update(self._log_case_keys(entry))
        for (host, version, family), rounds in sorted(grouped.items(), key=str):
            reference = None
            for rnd in ROUNDS:
                if rnd not in rounds:
                    fail(f"{host} {version} {family}: round {rnd} produced no records")
                if reference is None:
                    reference = rounds[rnd]
                elif rounds[rnd] != reference:
                    diff = sorted(str(k) for k in set(rounds[rnd]) ^ set(reference))
                    fail(f"{host} {version} {family}: the row set changed at round {rnd}: "
                         f"{diff[:4]}")
        for version in VERSIONS:
            for family in FAMILIES:
                reference = grouped[(self.hosts[0], version, family)][ROUNDS[0]]
                for host in self.hosts[1:]:
                    if grouped[(host, version, family)][ROUNDS[0]] != reference:
                        fail(f"{version} {family}: host workload sets differ")
        for family in FAMILIES:
            released = grouped[(self.hosts[0], "v2", family)][ROUNDS[0]]
            current = grouped[(self.hosts[0], "v3", family)][ROUNDS[0]]
            matched = {key for key in current if key[4] != "gf16d"}
            if released != matched:
                fail(f"{family}: version workload sets differ beyond custom gf16d")

    def _aggregate(self) -> None:
        for key, data in self.cases.items():
            for slot, series in data.values.items():
                self.aggregates[(key, slot)] = series.agg()

    # -- lookups ---------------------------------------------------------------------

    def value(self, key: tuple, host: str, version: str, metric: str,
              logical_bytes: int | None) -> float | None:
        ns = self.aggregates.get((key, (host, version)))
        if ns is None:
            return None
        return ns if metric != THROUGHPUT else gib_per_sec(logical_bytes, ns)

    def _series_median(self, series: dict, host: str, version: str) -> float | None:
        runs = series.get((host, version))
        if not runs:
            return None
        return median([runs[r] for r in sorted(runs)])

    def trio_value(self, key: tuple, arm: str, host: str, version: str) -> float | None:
        entry = self.trio.get(key)
        if entry is None:
            return None
        return self._series_median(entry["arms"][arm], host, version)

    def prime_value(self, key: tuple, arm: str, host: str, version: str) -> float | None:
        entry = self.prime.get(key)
        if entry is None:
            return None
        return self._series_median(entry["arms"][arm], host, version)

    def gf16_value(self, key: tuple, arm: str, host: str, version: str) -> float | None:
        entry = self.gf16.get(key)
        if entry is None:
            return None
        runs = entry["arms"].get((host, version, arm))
        return median([runs[r] for r in sorted(runs)]) if runs else None

    def backend_of(self, host: str, version: str, family: str) -> str | None:
        backends = set()
        for entry in self.log_evidence:
            if (entry["host"], entry["version"], entry["family"]) == (host, version, ROUND_FAMILY[family]):
                backends.update(b["backend"] for b in entry["process_backends"])
        if not backends:
            return None
        if len(backends) > 1:
            fail(f"{host} {version} {family}: inconsistent backend banners: "
                 f"{sorted(backends)}")
        return next(iter(backends))
    def field_backend_of(self, host: str, version: str, label: str) -> str | None:
        backends = {banner["backend"] for entry in self.log_evidence
                    if (entry["host"], entry["version"]) == (host, version)
                    for banner in entry["banners"] if banner["label"] == label}
        if len(backends) > 1:
            fail(f"{host} {version} {label}: inconsistent field backend banners: {sorted(backends)}")
        return next(iter(backends), None)


    def harness_versions(self, kind: str, host: str, version: str) -> dict:
        for entry in self.log_evidence:
            if (entry["host"], entry["version"]) == (host, version):
                header = entry.get(f"{kind}_header") or {}
                if header:
                    return header
        return {}


# ---------------------------------------------------------------------------
# Page tables
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    key: tuple
    metric: str
    logical_bytes: int | None


@dataclass
class Row:
    labels: list[str]
    field: str
    cells: dict = field(default_factory=dict)  # column name -> {version: Cell | None}


@dataclass
class Table:
    id: str
    metric: str
    label_cols: list[str]
    value_cols: list[str]
    rows: list[Row] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)
    rendered: str | None = None   # competitor tables render directly


class CellIndex:
    """Page/table/row/column references for every published cell."""

    def __init__(self) -> None:
        self.refs: list[dict] = []

    def record(self, page: str, table: str, row: str, column: str,
               key: tuple, version: str) -> None:
        self.refs.append({"page": page, "table": table, "row": row, "column": column,
                          "case": list(key),
                          "version": version})


def page_fields(family: str) -> list[str]:
    return [f for f in FIELDS if FIELD_FAMILY[f] == family]


def _geom(key: tuple) -> dict:
    return dict(key[6])


def render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


BUFFER_OP_ORDER = [
    "add_assign", "add_assign_scalar", "sub_assign_scalar", "mul_add",
    "mul_add_with", "mul_into", "mul_assign", "mul_elementwise",
    "mul_elementwise_assign",
]
LARGEDST_OP_ORDER = ["mul_into", "mul_into + read", "mul_add"]
PEEL_OP_ORDER = ["mul_add_scatter", "mul_add_gather", "mul_add_matrix"]


def cell_versions(campaign: Campaign, key: tuple) -> dict:
    """Which versions hold data for this case; a cell exists per version."""
    data = campaign.cases[key]
    present = {}
    for version in VERSIONS:
        slots = [slot for slot in data.values if slot[1] == version]
        present[version] = Cell(key, key[5], data.first.logical_bytes) if slots else None
    return present


class Renderer:
    def __init__(self, campaign: Campaign) -> None:
        self.campaign = campaign
        self.cells = CellIndex()
        self.pages: dict[str, str] = {}

    # -- self-table cell formatting ------------------------------------------------

    def host_cell(self, cell: Cell, host: str, version: str) -> float | None:
        return self.campaign.value(cell.key, host, version, cell.metric,
                                   cell.logical_bytes)

    def _snapshot_pair(self, page: str, table: Table, row: Row, column: str,
                       cell: Cell, version: str, digits: int, recorded: set) -> str:
        values = [self.host_cell(cell, host, version) for host in self.campaign.hosts]
        if all(v is None for v in values):
            return "-"
        marker = (page, table.id, " | ".join(row.labels), column, version)
        if marker not in recorded:
            recorded.add(marker)
            self.cells.record(page, table.id, " | ".join(row.labels), column,
                              cell.key, version)
        return fmt_pair(values[0], values[1], digits)

    def render_table(self, page: str, table: Table, version: str | None) -> str:
        recorded: set = set()
        header = list(table.label_cols)
        if version is not None:
            header += [f"{c} ({METRIC_UNIT[table.metric]})" for c in table.value_cols]
            all_values = []
            for row in table.rows:
                for column in table.value_cols:
                    cell = row.cells.get(column, {}).get(version)
                    if cell is not None:
                        all_values += [v for v in (self.host_cell(cell, h, version)
                                                   for h in self.campaign.hosts)
                                       if v is not None]
            digits = table_digits(all_values) if all_values else 3
            lines = []
            for row in table.rows:
                if not any(row.cells.get(column, {}).get(version) is not None
                           for column in table.value_cols):
                    continue
                line = list(row.labels)
                for column in table.value_cols:
                    cell = row.cells.get(column, {}).get(version)
                    if cell is None:
                        line.append("-")
                    else:
                        line.append(self._snapshot_pair(page, table, row, column, cell,
                                                        version, digits, recorded))
                lines.append(line)
            return render_markdown_table(header, lines)
        return self._render_comparison_table(page, table)

    def _render_comparison_table(self, page: str, table: Table) -> str:
        multiple_cases = len(table.value_cols) > 1
        header = list(table.label_cols) + (["Case"] if multiple_cases else [])
        header += ["v3 speed factor"]
        lines = []
        for row in table.rows:
            # Each geometry/operation column becomes its own matched comparison row.
            for column in table.value_cols:
                pair = row.cells.get(column, {})
                c2, c3 = pair.get("v2"), pair.get("v3")
                if c2 is None and c3 is None:
                    continue
                labels = list(row.labels) + ([column] if multiple_cases else [])
                ratios = []
                for index, host in enumerate(self.campaign.hosts):
                    a = self.host_cell(c2, host, "v2") if c2 is not None else None
                    b = self.host_cell(c3, host, "v3") if c3 is not None else None
                    if a is None or b is None:
                        ratios.append(None)
                    elif table.metric == THROUGHPUT:
                        ratios.append(b / a)
                    else:
                        ratios.append(a / b)
                lines.append(labels + [fmt_ratio_pair(ratios[0], ratios[1])])
                if c2 is not None and c3 is not None:
                    self.cells.record(page, table.id, " | ".join(labels), "v3 speed factor",
                                      c3.key, "v3/v2" if table.metric == THROUGHPUT else "v2/v3")
        return render_markdown_table(header, lines)

    # -- self-table construction ------------------------------------------------------

    @staticmethod
    def _finalize(table: Table) -> Table:
        fields = {row.field for row in table.rows}
        if len(fields) > 1 and "Field" not in table.label_cols:
            table.label_cols = ["Field", *table.label_cols]
            table.rows = [Row([row.field, *row.labels], row.field, row.cells)
                          for row in table.rows]
        elif len(fields) == 1 and "Field" in table.label_cols:
            table.label_cols = [c for c in table.label_cols if c != "Field"]
            table.rows = [Row(row.labels[1:], row.field, row.cells) for row in table.rows]
        return table

    def _assemble(self, table_id: str, metric: str, label_cols: list[str],
                  value_cols: list[str], entries: list[tuple],
                  prose: list[str] | None = None) -> Table | None:
        """entries: (row_labels, row_field, column, key)."""
        rows: dict[tuple, Row] = {}
        for row_labels, row_field, column, key in entries:
            row = rows.setdefault((row_field, tuple(row_labels)), Row(list(row_labels), row_field))
            for version, cell in cell_versions(self.campaign, key).items():
                if cell is None:
                    continue
                slot = row.cells.setdefault(column, {})
                if slot.get(version) is not None:
                    fail(f"duplicate cell for {table_id} row {row_labels} column {column}")
                slot[version] = cell
        if not rows:
            return None
        table = Table(table_id, metric, list(label_cols), list(value_cols),
                      [rows[labels] for labels in sorted(rows, key=str)], prose or [])
        return self._finalize(table)

    # -- section builders ----------------------------------------------------------------

    def build_sections(self, family: str) -> list[tuple[str, list[Table]]]:
        builders = (
            ("Scalar multiplication", self.section_scalar),
            ("Packed operations", self.section_buffers),
            ("Aligned single-row operations", self.section_regions),
            ("One-shot and prepared multiplication", self.section_crossover),
            ("Scatter, gather, and matrix", self.section_scattermatrix),
            ("Short multi-row operations", self.section_smallrow),
            ("Aligned gather", self.section_dot),
            ("Overwrite matrix", self.section_encode),
            ("Row-wise addition", self.section_rowwise),
            ("Row-wise addition compositions", self.section_rowwise_comp),
            ("Network-size payloads", self.section_network),
            ("Fixed-offset multi-row operations", self.section_peel),
            ("Fixed-offset single-row multiplication", self.section_offsweep),
            ("Scatter destination alignment", self.section_destalign),
            ("In-place scaling by buffer size", self.section_assign_sweep),
            ("Byte-field density control", self.section_gf2control),
            ("Large destinations", self.section_largedst),
            ("Buffer operations", self.section_gf2bits),
            ("Short ranges", self.section_gf2short),
            ("Single-row multiplication", self.section_tier3),
        )
        sections = []
        for title, builder in builders:
            tables = [t for t in builder(family) if t is not None and t.rows]
            if tables:
                sections.append((title, tables))
        return sections

    def native_comparison_sections(self, family: str) -> list[tuple[str, list[Table]]]:
        sections = []
        for metric, title in (
                (SCALAR_METRIC, "Indexed scalar multiplication and accumulation"),
                (THROUGHPUT, "Native-layout public operations")):
            entries = []
            for key, data in self.campaign.cases.items():
                if key[1] not in ("trio", "prime", "gf16") or FIELD_FAMILY[key[4]] != family:
                    continue
                if key[5] != metric:
                    continue
                rec = data.first
                geometry = dict(rec.geometry)
                geometry["alignment"] = "page"
                if metric == THROUGHPUT:
                    geometry["logical_bytes"] = rec.logical_bytes
                    geometry["preparation"] = "prepared" if rec.label.endswith("_with") else "one-shot"
                else:
                    geometry["scope"] = "indexed acc += a * b"
                operation = "acc += a * b" if metric == SCALAR_METRIC else rec.label
                description = "; ".join(f"{name}={value}" for name, value in sorted(geometry.items()))
                entries.append(([operation, description], rec.field, "Result", key))
            table = self._assemble("native-" + metric, metric, ["Operation", "Geometry"],
                                   ["Result"], entries,
                                   ["The fgf arm's public calls from the native-layout harness; "
                                    "only version arms are compared. Layout conversion remains "
                                    "outside timing."])
            if table is not None:
                sections.append((title, [table]))
        return sections

    def _panel_keys(self, family: str, panel: str) -> list[tuple]:
        return [key for key, data in self.campaign.cases.items()
                if data.first.public and key[1] in ("kernels", "compare")
                and key[2] == panel and FIELD_FAMILY.get(key[4]) == family]

    def section_scalar(self, family: str) -> list[Table]:
        entries = sorted(
            ([key[4], "scalar mul"], key[4], "scalar mul", key)
            for key in self._panel_keys(family, "scalar"))
        table = self._assemble("scalar", SCALAR_METRIC, ["Field", "Operation"],
                               ["scalar mul"], entries,
                               ["Latency of the dependent multiply chain, per scalar "
                                "operation."])
        return [table] if table else []

    def section_buffers(self, family: str) -> list[Table]:
        tables = []
        for field_name in page_fields(family):
            keys = [k for k in self._panel_keys(family, "buffers") if k[4] == field_name]
            if not keys:
                continue
            sizes = sorted({_geom(k).get("buffer_bytes") for k in keys})
            ops = [op for op in BUFFER_OP_ORDER if any(k[3] == op for k in keys)]
            ops += sorted({k[3] for k in keys} - set(BUFFER_OP_ORDER))
            entries = [([op], field_name, fmt_bytes(_geom(key)["buffer_bytes"]), key)
                       for op in ops for key in keys if key[3] == op]
            tables.append(self._assemble(
                f"buffers-{field_name}", THROUGHPUT, ["Operation"],
                [fmt_bytes(s) for s in sizes], entries,
                [f"`{field_name}`, ordinary `Vec<u8>` buffers."]))
        return tables

    def section_regions(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "regions")
        sizes = sorted({_geom(k).get("row_bytes") for k in keys})
        entries = [([key[4], key[3]], key[4], fmt_bytes(_geom(key)["row_bytes"]), key)
                   for key in keys]
        table = self._assemble("regions", THROUGHPUT, ["Field", "Operation"],
                               [fmt_bytes(s) for s in sizes], entries,
                               ["64-byte-aligned buffers."])
        return [table] if table else []

    def section_crossover(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "crossover")
        lengths = sorted({_geom(k).get("row_bytes") for k in keys})
        entries = [([key[4], fmt_bytes(_geom(key)["row_bytes"])], key[4], key[3], key)
                   for key in keys]
        table = self._assemble("crossover", LATENCY, ["Field", "Row bytes"],
                               ["mul_add", "mul_add_with"], entries,
                               ["Latency per call. The one-shot form derives the backend "
                                "coefficient on every call; the prepared form reuses "
                                "coefficients built outside the timed region."])
        return [table] if table else []

    def section_scattermatrix(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "scattermatrix")
        counts = sorted({_geom(k).get("rows") for k in keys})
        entries = [([key[4], key[3]], key[4], f"{_geom(key)['rows']} rows", key)
                   for key in keys]
        table = self._assemble("scattermatrix", THROUGHPUT, ["Field", "Operation"],
                               [f"{n} rows" for n in counts], entries,
                               ["64 KiB rows, ordinary `Vec<u8>` buffers. The column "
                                "count is destinations for scatter and matrix, sources "
                                "for gather."])
        return [table] if table else []

    def section_smallrow(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "smallrow")
        lengths = sorted({_geom(k).get("row_bytes") for k in keys})
        entries = [([key[4], key[3]], key[4], fmt_bytes(_geom(key)["row_bytes"]), key)
                   for key in keys]
        table = self._assemble("smallrow", THROUGHPUT, ["Field", "Operation"],
                               [fmt_bytes(n) for n in lengths], entries,
                               ["Sixteen rows, eight sources; coefficient sets include "
                                "zeros and ones."])
        return [table] if table else []

    def section_dot(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "dot")
        lengths = sorted({_geom(k).get("row_bytes") for k in keys})
        entries = [([key[4], key[3]], key[4], fmt_bytes(_geom(key)["row_bytes"]), key)
                   for key in keys]
        table = self._assemble("dot", THROUGHPUT, ["Field", "Operation"],
                               [fmt_bytes(n) for n in lengths], entries,
                               ["Sixteen sources into one row, 64-byte-aligned buffers. "
                                "`mul_into_gather` is the raw one-shot overwrite form."])
        return [table] if table else []

    def section_encode(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "encode")
        dests = sorted({_geom(k).get("destinations") for k in keys})
        entries = [([key[4], key[3]], key[4],
                    f"{_geom(key)['destinations']} destinations", key) for key in keys]
        table = self._assemble("encode", THROUGHPUT, ["Field", "Operation"],
                               [f"{n} destinations" for n in dests], entries,
                               ["Ten sources, 64 KiB rows, 64-byte-aligned buffers. The "
                                "`fill(0)` cases include clearing the destination; "
                                "`mul_add_matrix_at` uses contiguous row offsets."])
        return [table] if table else []

    def section_rowwise(self, family: str) -> list[Table]:
        keys = [k for k in self._panel_keys(family, "addrows")
                if k[3] == "add_assign_rows"]
        entries = [([fmt_bytes(_geom(k)["row_bytes"]), str(_geom(k)["rows"])], k[4],
                    "add_assign_rows", k) for k in keys]
        table = self._assemble("rowwise", THROUGHPUT, ["Row bytes", "Rows"],
                               ["add_assign_rows"], entries,
                               ["`add_assign_rows` interleaves the row streams; flat "
                                "`add_assign` and the literal per-row loop appear under "
                                "compositions."])
        return [table] if table else []

    def section_rowwise_comp(self, family: str) -> list[Table]:
        keys = [k for k in self._panel_keys(family, "addrows")
                if k[3] != "add_assign_rows"]
        entries = [([key[3], fmt_bytes(_geom(key)["row_bytes"]),
                     str(_geom(key)["rows"])], key[4], "Throughput", key)
                   for key in keys]
        table = self._assemble("rowwise-comp", THROUGHPUT,
                               ["Operation", "Row bytes", "Rows"], ["Throughput"], entries,
                               ["Public call compositions: flat `add_assign` and the "
                                "literal per-row loop over the same geometry."])
        return [table] if table else []

    def section_network(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "network")
        lengths = sorted({_geom(k).get("payload_bytes") for k in keys})
        entries = []
        for key in keys:
            label = (f"{key[3]} ({_geom(key)['rows']} rows)"
                     if key[3] == "mul_add_scatter" else key[3])
            entries.append(([label], "gf8b", fmt_bytes(_geom(key)["payload_bytes"]), key))
        table = self._assemble("network", THROUGHPUT, ["Operation"],
                               [fmt_bytes(n) for n in lengths], entries,
                               ["gf8b, ordinary `Vec<u8>` buffers; scatter rows use the "
                                "payload length as the row pitch."])
        return [table] if table else []

    def section_peel(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "peel")
        entries = [([fmt_bytes(_geom(k)["row_bytes"]), str(_geom(k)["offset"])], k[4],
                    k[3], k) for k in keys]
        table = self._assemble("peel", THROUGHPUT, ["Row bytes", "Base offset mod 64"],
                               PEEL_OP_ORDER, entries,
                               ["gf8d; every source and destination base carries the "
                                "stated offset modulo 64. Scatter has four destinations, "
                                "gather sixteen sources, matrix ten sources and four "
                                "destinations."])
        return [table] if table else []

    def section_offsweep(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "offsweep")
        entries = [([str(_geom(k)["offset"])], k[4], "mul_add", k) for k in keys]
        table = self._assemble("offsweep", THROUGHPUT, ["Base offset mod 64"],
                               ["mul_add"], entries,
                               ["gf8d `mul_add`, 16 KiB; source and destination start at "
                                "the stated offset from a 64-byte boundary."])
        return [table] if table else []

    def section_destalign(self, family: str) -> list[Table]:
        tables = []
        keys = self._panel_keys(family, "destalign")
        if keys:
            entries = [([fmt_bytes(_geom(k)["row_bytes"]), str(_geom(k)["skew"])], k[4],
                        "mul_add_scatter", k) for k in keys]
            tables.append(self._assemble(
                "destalign", THROUGHPUT, ["Row size", "Destination offset mod 64"],
                ["mul_add_scatter"], entries,
                ["Eight destinations, ordinary `Vec<u8>` source; the destination starts "
                 "at the stated offset from a 64-byte boundary."]))
        keys = self._panel_keys(family, "destalign-compare")
        if keys:
            entries = [([key[3]], key[4], "Throughput", key) for key in keys]
            tables.append(self._assemble(
                "destalign-compare", THROUGHPUT, ["Operation"], ["Throughput"], entries,
                ["gf8d, one source and four 16 KiB destination rows, 64-byte-aligned "
                 "buffers; raw coefficients against prepared coefficients through the "
                 "plan."]))
        return tables

    def section_assign_sweep(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "assign-sweep")
        entries = [([k[4], fmt_bytes(_geom(k)["buffer_bytes"])], k[4], "mul_assign", k)
                   for k in keys]
        table = self._assemble("assign-sweep", THROUGHPUT, ["Field", "Buffer size"],
                               ["mul_assign"], entries,
                               ["`mul_assign`, 64-byte-aligned buffers."])
        return [table] if table else []

    def section_gf2control(self, family: str) -> list[Table]:
        entries = [([key[4], fmt_bytes(_geom(key)["buffer_bytes"])], key[4], "add_assign", key)
                   for key in self._panel_keys(family, "gf2control")]
        table = self._assemble("gf2control", THROUGHPUT, ["Field", "Buffer bytes"], ["add_assign"],
                               entries,
                               ["`gf8b` byte-per-element density control measured alongside "
                                "the bit-packed buffers."])
        return [table] if table else []

    def section_largedst(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "largedst")
        ops = [op for op in LARGEDST_OP_ORDER if any(k[3] == op for k in keys)]
        entries = [([key[4], fmt_bytes(_geom(key)["buffer_bytes"])], key[4], key[3], key)
                   for key in keys]
        table = self._assemble("largedst", THROUGHPUT, ["Field", "Buffer size"], ops,
                               entries,
                               ["Ordinary `Vec<u8>` buffers. The read-back row XOR-folds "
                                "every written 64-bit word after `mul_into`."])
        return [table] if table else []

    def section_gf2bits(self, family: str) -> list[Table]:
        keys = self._panel_keys(family, "gf2bits")
        sizes = sorted({_geom(k).get("buffer_bytes") for k in keys})
        entries = [([key[3]], key[4], fmt_bytes(_geom(key)["buffer_bytes"]), key)
                   for key in keys]
        table = self._assemble("gf2bits", THROUGHPUT, ["Operation"],
                               [fmt_bytes(s) for s in sizes], entries,
                               ["Ordinary `Vec<u8>` buffers. Range XOR updates bits "
                                "[N/4, 7N/8); `bits::XorRange` is the prepared geometry "
                                "that `bits::xor_range_with` applies."])
        return [table] if table else []

    def section_gf2short(self, family: str) -> list[Table]:
        entries = [([key[3]], key[4], "Latency", key)
                   for key in self._panel_keys(family, "gf2short")]
        table = self._assemble("gf2short", ROWPAIR, ["Operation"], ["Latency"], entries,
                               ["16-byte rows, bits [61, 69); latency per row pair, not "
                                "per batch."])
        return [table] if table else []

    def section_tier3(self, family: str) -> list[Table]:
        entries = [([key[4], "mul_add"], key[4], "mul_add", key)
                   for key in self._panel_keys(family, "tier3")]
        table = self._assemble("tier3", THROUGHPUT, ["Field", "Operation"], ["mul_add"],
                               entries,
                               ["256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte "
                                "volume means one information volume: the wider fields "
                                "carry the same payload per call."])
        return [table] if table else []

    # -- competitor tables (snapshot pages only) ---------------------------------------

    def _competitor_table(self, page: str, table_id: str, label_cols: list[str],
                          columns: list[str], row_specs: list[tuple], metric: str,
                          version: str) -> str:
        """row_specs: (labels, case_key, {column: [host values in ns or None]}),
        with a converter applied per metric."""
        recorded: set = set()
        all_values = []

        def convert(value: float | None, logical_bytes: int | None) -> float | None:
            if value is None:
                return None
            return value if metric != THROUGHPUT else gib_per_sec(logical_bytes, value)

        prepared = []
        for labels, key, cells, logical_bytes in row_specs:
            row_cells = {}
            for column in columns:
                converted = [convert(v, logical_bytes) for v in cells[column]]
                row_cells[column] = converted
                all_values += [v for v in converted if v is not None]
            prepared.append((labels, key, row_cells))
        digits = table_digits(all_values) if all_values else 3
        header = list(label_cols) + [f"{c} ({METRIC_UNIT[metric]})" for c in columns]
        lines = []
        for labels, key, row_cells in prepared:
            line = list(labels)
            for column in columns:
                converted = row_cells[column]
                if all(v is None for v in converted):
                    line.append("-")
                    continue
                marker = (page, table_id, " | ".join(labels), column, version)
                if marker not in recorded:
                    recorded.add(marker)
                    self.cells.record(page, table_id, " | ".join(labels), column,
                                      key, version)
                line.append(fmt_pair(converted[0], converted[1], digits))
            lines.append(line)
        return render_markdown_table(header, lines)

    def trio_rendered(self, page: str, version: str) -> str | None:
        campaign = self.campaign
        arms = ("fgf", "isal", "klauspost")
        columns = ("`fgf`", "Intel ISA-L", "klauspost/reedsolomon")
        order = {"trio-single": 0, "trio-gather": 1, "trio-encode": 2}
        specs = []
        for key, entry in campaign.trio.items():
            meta = entry["meta"]
            values = {column: [campaign.trio_value(key, arm, host, version)
                               for host in campaign.hosts]
                      for column, arm in zip(columns, arms)}
            if all(v is None for converted in values.values() for v in converted):
                continue
            geom = meta.geometry
            labels = [meta.label,
                      f"{geom.get('sources', 1)} × {geom.get('destinations', 1)}",
                      fmt_bytes(geom["row_bytes"])]
            specs.append((order.get(meta.panel, 9), geom.get("row_bytes", 0),
                          geom.get("destinations", 0), meta.label,
                          (labels, key, values, meta.logical_bytes)))
        if not specs:
            return None
        specs.sort(key=lambda item: item[:4])
        table = self._competitor_table(
            page, "trio", ["Operation", "Sources × destinations", "Row size"],
            list(columns), [item[4] for item in specs], THROUGHPUT, version)
        return table

    def prime_rendered(self, page: str, version: str) -> list[tuple[str, str]]:
        campaign = self.campaign
        groups: dict[str, list] = {"Scalar": [], "Packed": []}
        for key, entry in campaign.prime.items():
            meta = entry["meta"]
            present = any(campaign.prime_value(key, arm, host, version) is not None
                          for arm in ("fgf", "p3") for host in campaign.hosts)
            if present:
                groups["Scalar" if meta.metric == SCALAR_METRIC else "Packed"].append(
                    (key, meta))
        results = []
        for title in ("Scalar", "Packed"):
            items = sorted(groups[title], key=lambda pair: (pair[1].field, pair[1].label))
            if not items:
                continue
            metric = SCALAR_METRIC if title == "Scalar" else THROUGHPUT
            specs = []
            for key, meta in items:
                values = {"`fgf`": [campaign.prime_value(key, "fgf", host, version)
                                    for host in campaign.hosts],
                          "Plonky3": [campaign.prime_value(key, "p3", host, version)
                                      for host in campaign.hosts]}
                specs.append(([meta.field, meta.label], key, values, meta.logical_bytes))
            rendered = self._competitor_table(page, f"prime-{title.lower()}",
                                              ["Field", "Operation"],
                                              ["`fgf`", "Plonky3"], specs, metric, version)
            results.append((title, rendered))
        return results

    def gf16_rendered(self, page: str, version: str) -> list[tuple[str, str]]:
        campaign = self.campaign
        groups: dict[tuple, list] = {}
        for key, entry in campaign.gf16.items():
            meta = entry["meta"]
            arms = tuple(a for a in ("fgf", *NATIVE_GF16_ARMS)
                         if any(entry["arms"].get((host, version, a))
                                for host in campaign.hosts))
            if not arms:
                continue
            groups.setdefault((meta.panel, arms), []).append((key, meta))
        results = []
        for (panel, arms), items in sorted(groups.items(), key=str):
            items.sort(key=lambda pair: (pair[1].geometry["sources"],
                                         pair[1].geometry["destinations"],
                                         pair[1].geometry["row_bytes"],
                                         pair[1].geometry["schedule"], pair[1].label))
            schedules = {pair[1].geometry["schedule"] for pair in items}
            label_cols = ["Operation", "Sources × destinations", "Row size"]
            schedule_col = schedules == {"fixed", "cycling16"}
            if schedule_col:
                label_cols.append("Coefficients")
            columns = [GF16_ARM_DISPLAY[a] for a in arms]
            specs = []
            for key, meta in items:
                geom = meta.geometry
                labels = [GF16_OP_DISPLAY[meta.label],
                          f"{geom['sources']} × {geom['destinations']}",
                          fmt_bytes(geom["row_bytes"])]
                if schedule_col:
                    labels.append("Fixed" if geom["schedule"] == "fixed" else "Cycling")
                values = {column: [campaign.gf16_value(key, arm, host, version)
                                   for host in campaign.hosts]
                          for column, arm in zip(columns, arms)}
                specs.append((labels, key, values, meta.logical_bytes))
            table_id = f"gf16-{panel.lower().replace(' ', '-')}"
            rendered = self._competitor_table(page, table_id, label_cols, columns, specs,
                                              THROUGHPUT, version)
            results.append((panel, rendered))
        controls = self.gf16_controls_rendered(page, version)
        if controls is not None:
            results.append(("Controls", controls))
        return results

    def gf16_controls_rendered(self, page: str, version: str) -> str | None:
        campaign = self.campaign
        rows = []
        for key, entry in campaign.gf16_controls.items():
            meta = entry["meta"]
            host_stats = []
            for host in campaign.hosts:
                ratios = entry["ratios"].get((host, version))
                if not ratios:
                    host_stats.append(None)
                    continue
                values = [ratios[r] for r in sorted(ratios)]
                host_stats.append((median(values), min(values), max(values)))
            if all(stat is None for stat in host_stats):
                continue
            geom = meta.geometry
            labels = [GF16_OP_DISPLAY[meta.label],
                      f"{geom['sources']} × {geom['destinations']}",
                      fmt_bytes(geom["row_bytes"]),
                      "Fixed" if geom["schedule"] == "fixed" else "Cycling"]
            medians = [None if s is None else s[0] for s in host_stats]
            ranges = ["-" if s is None else f"{fmt_ratio(s[1])}–{fmt_ratio(s[2])}"
                      for s in host_stats]
            rows.append((labels, key, medians, ranges))
        if not rows:
            return None
        rows.sort(key=lambda item: str(item[0]))
        header = ["Operation", "Sources × destinations", "Row size", "Coefficients",
                  f"Median ratio ({campaign.host_labels[campaign.hosts[0]]} / "
                  f"{campaign.host_labels[campaign.hosts[1]]})",
                  f"Run range ({campaign.host_labels[campaign.hosts[0]]} / "
                  f"{campaign.host_labels[campaign.hosts[1]]})"]
        lines = []
        for labels, key, medians, ranges in rows:
            lines.append(labels + [fmt_ratio_pair(medians[0], medians[1]),
                                   f"{ranges[0]} / {ranges[1]}"])
            self.cells.record(page, "gf16-controls", " | ".join(labels), "Median ratio",
                              key, version)
        return render_markdown_table(header, lines)

    # -- page assembly -------------------------------------------------------------------

    def render_snapshot(self, family: str, version: str) -> str:
        campaign = self.campaign
        hosts = [campaign.host_labels[host] for host in campaign.hosts]
        page = f"{version}/{family}.md"
        other = "v3" if version == "v2" else "v2"
        lines = [
            f"# {PAGE_TITLES[family]} ({version})",
            "",
            f"Paired-run snapshot for {version}. Every result cell reports "
            f"{hosts[0]} / {hosts[1]}; shared hosts, toolchain, sampling, and number "
            f"format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: "
            f"[labels.md](../labels.md). The paired comparison lives at "
            f"[v2 versus v3](../comparison/{family}.md); the other snapshot at "
            f"[{other}](../{other}/{family}.md).",
            "",
            "## Setup",
            "",
        ]
        setup = self.setup_rows(family, version=version)
        lines.append(render_markdown_table(["Setting", "Value"], setup))
        lines += ["", "## Self-timings", ""]
        for title, tables in self.build_sections(family):
            lines.append(f"### {title}")
            lines.append("")
            for table in tables:
                for prose in table.prose:
                    lines.append(prose)
                    lines.append("")
                lines.append(self.render_table(page, table, version))
                lines.append("")
        lines.append("## Competitors")
        lines.append("")
        competitor = self.render_competitors(page, family, version)
        lines.extend(competitor)
        lines += ["", "## Caveats", ""]
        lines.extend(self.caveat_lines(family, comparison=False))
        return "\n".join(lines).rstrip() + "\n"

    def render_comparison(self, family: str) -> str:
        campaign = self.campaign
        hosts = [campaign.host_labels[host] for host in campaign.hosts]
        page = f"comparison/{family}.md"
        lines = [
            f"# {PAGE_TITLES[family]}: v2 versus v3",
            "",
            f"Same-session comparison of the v2 line and v3. Every result cell reports "
            f"{hosts[0]} / {hosts[1]}; hosts and shared method: "
            f"[BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: "
            f"[labels.md](../labels.md). Snapshots: [v2](../v2/{family}.md), "
            f"[v3](../v3/{family}.md).",
            "",
            "## Setup",
            "",
        ]
        lines.append(render_markdown_table(["Setting", "Value"],
                                           self.setup_rows(family, version=None)))
        lines.append("")
        sections = self.build_sections(family) + self.native_comparison_sections(family)
        for field_name in page_fields(family):
            field_lines = []
            for title, tables in sections:
                for table in tables:
                    rows = [row for row in table.rows if row.field == field_name]
                    if not rows:
                        continue
                    has_field_col = "Field" in table.label_cols
                    if has_field_col:
                        rows = [Row(row.labels[1:], row.field, row.cells) for row in rows]
                    sub = Table(f"{field_name}-{table.id}", table.metric,
                                [c for c in table.label_cols if c != "Field"],
                                table.value_cols, rows, table.prose)
                    field_lines.append(f"### {title}")
                    field_lines.append("")
                    for prose in sub.prose:
                        field_lines.append(prose)
                        field_lines.append("")
                    field_lines.append(self.render_table(page, sub, None))
                    field_lines.append("")
            if field_lines:
                lines.append(f"## {field_name}")
                lines.append("")
                lines.extend(field_lines)
        lines.append("## Caveats")
        lines.append("")
        lines.extend(self.caveat_lines(family, comparison=True))
        return "\n".join(lines).rstrip() + "\n"

    def setup_rows(self, family: str, version: str | None) -> list[list[str]]:
        campaign = self.campaign
        rows = [["Campaign", "`just bench-paired` (canonical family `gf1`)" if family == "gf1"
                 else "`just bench-paired` (family round `{}`)".format(ROUND_FAMILY[family])]]
        if version is None:
            for ver in VERSIONS:
                revision = self._revision(ver)
                if revision is not None:
                    rows.append([ver, revision])
        else:
            revision = self._revision(version)
            if revision is not None:
                rows.append(["Revision", revision])
        toolchain = campaign.metadata.get("toolchain")
        if toolchain is not None:
            rows.append(["Toolchain", self._meta_text(toolchain)])
        backends = {}
        for host in campaign.hosts:
            for ver in ([version] if version else VERSIONS):
                backends[(host, ver)] = campaign.backend_of(host, ver, family)
        if version is not None:
            cells = [backends.get((host, version)) or "-" for host in campaign.hosts]
            rows.append(["Resolved process backend", "`" + "` / `".join(cells) + "`"])
        else:
            per_version = []
            for ver in VERSIONS:
                cells = [backends.get((host, ver)) or "-" for host in campaign.hosts]
                per_version.append(f"{ver}: `" + "` / `".join(cells) + "`")
            rows.append(["Resolved process backend", "; ".join(per_version)])
        field_routes = []
        for ver in ([version] if version else VERSIONS):
            for label in page_fields(family):
                routes = [campaign.field_backend_of(host, ver, label) for host in campaign.hosts]
                if any(route is not None for route in routes):
                    prefix = f"{ver} " if version is None else ""
                    field_routes.append(f"{prefix}`{label}`: " + " / ".join(
                        f"`{route}`" if route is not None else "-" for route in routes))
        if field_routes:
            rows.append(["Resolved field backends", "; ".join(field_routes)])
        rows.append(["Protocol", "Five rounds per host; each round runs both versions in "
                                 "a shuffled order; competitor arms interleave inside "
                                 "their harness process."])
        rows.append(["Aggregation", "Median of five per-run medians."])
        if version is None:
            rows.append(["Ratio", "v3/v2 for throughput, v2/v3 for latency; above one "
                                  "means v3 is faster. Computed from unrounded "
                                  "host-local aggregates."])
        else:
            rows.append(["Throughput", "Logical bytes divided by the aggregate latency "
                                       "per call; numerators are stated per panel."])
        competitor = self._competitor_versions(family, version)
        if competitor:
            rows.append(["Competitors", competitor])
        return rows

    def _competitor_versions(self, family: str, version: str | None) -> str | None:
        kind = {"gf8": "trio", "gf16": "gf16"}.get(family, "prime" if family in
                                                   ("mersenne31", "goldilocks") else None)
        if kind is None:
            return None
        texts = []
        for host in self.campaign.hosts:
            for ver in ([version] if version else VERSIONS):
                header = self.campaign.harness_versions(kind, host, ver)
                if not header:
                    continue
                if kind == "trio":
                    text = (f"Intel ISA-L {header.get('isal', '-')}; "
                            f"klauspost/reedsolomon {header.get('klauspost', '-')} "
                            f"via Go {header.get('go', '-')}")
                elif kind == "gf16":
                    text = (f"GF-Complete {header.get('gf_complete', '-')} "
                            f"{header.get('gf_complete_revision', '-')}; "
                            f"reed-solomon-erasure {header.get('reed_solomon_erasure', '-')}; "
                            f"reed-solomon-simd {header.get('reed_solomon_simd', '-')}; "
                            f"Leopard-RS 1.x {header.get('leopard', '-')}")
                else:
                    text = (f"Plonky3 {header.get('p3', '-')}; native packing widths "
                            f"{header.get('packing_widths', '-')}")
                if version is None:
                    texts.append(f"{self.campaign.host_labels[host]} {ver}: {text}")
                else:
                    texts.append(f"{self.campaign.host_labels[host]}: {text}")
        return "; ".join(texts) or None

    def _revision(self, version: str) -> str | None:
        revision = self.campaign.metadata.get(f"{version}_revision")
        return f"`{revision}`" if isinstance(revision, str) else None

    @staticmethod
    def _meta_text(value) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def render_competitors(self, page: str, family: str, version: str) -> list[str]:
        lines = []
        if family == "gf8":
            rendered = self.trio_rendered(page, version)
            if rendered is None:
                lines.append("No matched competitor measurements are available.")
                return lines
            lines.append("Polynomial `0x11D`, page-aligned buffers, and matched coding "
                         "coefficients. ISA-L uses its field kernels; klauspost uses "
                         "`Encode` with a supplied coding matrix, or `EncodeIdx` for "
                         "accumulation. Outputs are validated before timing.")
            lines.append("")
            lines.append(rendered)
            return lines
        if family == "gf16":
            rendered = self.gf16_rendered(page, version)
            if not rendered:
                lines.append("No matched competitor measurements are available.")
                return lines
            lines.append("Each arm runs its native field primitive; no codec operation "
                         "substitutes for a field primitive. Tables split by the "
                         "primitives each library exposes, so every column is populated. "
                         "\"Fixed\" reuses one coefficient set; \"Cycling\" rotates "
                         "through sixteen prepared sets.")
            lines.append("")
            lines.extend([
                "| Arm | Native representation | Execution |",
                "| --- | --- | --- |",
                "| `gf16b` | AES-rooted quadratic tower; adjacent component bytes | Resolved field route |",
                "| GF-Complete | Polynomial `0x1100B`; little-endian words | SSSE3 split-4/16 |",
                "| reed-solomon-erasure | RS-rooted quadratic extension; big-endian components | Scalar GF16 |",
                "| reed-solomon-simd | Cantor coordinates; tile-local low/high planes | AVX2 |",
                "| Leopard-RS 1.x | Cantor coordinates; tile-local low/high planes | AVX2 |",
                "",
                "Field maps and coefficient/layout conversions are validated and "
                "excluded from timing. Native encodings are not renamed `gf16d`.",
                "",
            ])
            for title, table in rendered:
                lines.append(f"### {title}")
                lines.append("")
                if title == "Controls":
                    lines.append("Control ratios divide duplicate-`fgf` latency by "
                                 "primary-`fgf` latency. The median ratio and range are "
                                 "across the five runs.")
                    lines.append("")
                lines.append(table)
                lines.append("")
            return lines
        if family in ("mersenne31", "goldilocks"):
            rendered = self.prime_rendered(page, version)
            if not rendered:
                lines.append("No matched competitor measurements are available.")
                return lines
            lines.append("Page-aligned regions on each library's native layout; "
                         "conversions are excluded. Packed throughput counts two region "
                         "byte extents per call. Scalar rows measure indexed products "
                         "with accumulation. Arms alternate within one process, and outputs "
                         "are validated before timing.")
            lines.append("")
            for title, table in rendered:
                lines.append(f"### {title}")
                lines.append("")
                lines.append(table)
                lines.append("")
            return lines
        lines.append("No matched competitor measurements are available.")
        return lines

    def caveat_lines(self, family: str, comparison: bool) -> list[str]:
        lines = []
        if comparison:
            lines.append("- Ratios are computed from unrounded host-local aggregates, "
                         "never from the displayed cells.")
            lines.append("- Differences comparable to control variability are "
                         "unresolved, not proof of a faster implementation.")
        else:
            lines.append("- The unchanged byte-XOR control bracketed every self-suite "
                         "process; its values are retained in the raw evidence.")
        if family == "gf":
            lines.append("- The trio harness opens with an all-`fgf` control row; "
                         "control values are retained in the raw evidence and excluded "
                         "from the table.")
        if family == "gf16":
            lines.append("- Duplicate `fgf` arms measure scheduling and interleaving "
                         "bias; control ratios are tabulated under Controls.")
        if family == "mersenne31" or family == "goldilocks":
            lines.append("- The prime harness control row precedes the first field "
                         "heading; control values are retained in the raw evidence.")
        if family == "gf16" and self._has_gf16d():
            if comparison:
                lines.append("- `gf16d` is measured only on v3: the v2 release has no "
                             "public counterpart, so its comparison rows carry no "
                             "invented v2 value.")
            else:
                lines.append("- `gf16d` rows appear only on the v3 page: the v2 release "
                             "has no public counterpart.")
        return lines

    def _has_gf16d(self) -> bool:
        return any(key[4] == "gf16d" for key in self.campaign.cases
                   if key[1] in ("kernels", "compare"))


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def _case_payload(campaign: Campaign, key: tuple, data: CaseData) -> dict:
    values = {}
    for (host, version), series in sorted(data.values.items()):
        values.setdefault(host, {})[version] = {
            "rounds": {str(r): series.rounds[r] for r in sorted(series.rounds)},
            "median": campaign.aggregates.get((key, (host, version))),
        }
    availability = {version: {host: bool(data.values.get((host, version))
                                         and data.values[(host, version)].rounds)
                              for host in campaign.hosts} for version in VERSIONS}
    return {
        "case": [json.loads(json.dumps(part, default=list)) for part in key],
        "kind": key[0], "suite": key[1], "panel": key[2], "label": key[3],
        "field": key[4], "metric": key[5],
        "geometry": dict(_geom(key)),
        "logical_bytes": data.first.logical_bytes,
        "public": data.first.public,
        "disposition": data.first.disposition,
        "raw_label": data.first.raw_label,
        "availability": availability,
        "values": values,
    }


def _competitor_payload(campaign: Campaign, store: dict, arms: tuple) -> list[dict]:
    payload = []
    for key, entry in store.items():
        meta = entry["meta"]
        item = {
            "case": [json.loads(json.dumps(part, default=list)) for part in key],
            "panel": key[2], "label": key[3], "field": key[4], "metric": key[5],
            "geometry": dict(_geom(key)),
            "logical_bytes": meta.logical_bytes,
        }
        values = {}
        for arm in arms:
            for (host, version), runs in sorted(entry["arms"][arm].items()):
                values.setdefault(host, {}).setdefault(version, {})[arm] = {
                    "rounds": {str(r): runs[r] for r in sorted(runs)},
                    "median": campaign._series_median(entry["arms"][arm], host, version),
                }
        item["values"] = values
        if entry.get("bands"):
            item["bands"] = {f"{host}|{version}|{round}": band
                             for (host, version, round), band
                             in sorted(entry["bands"].items(), key=str)}
        payload.append(item)
    return payload


def _gf16_payload(campaign: Campaign) -> list[dict]:
    """The gf16 store keys its series by (host, version, arm) in one dict."""
    payload = []
    for key, entry in campaign.gf16.items():
        meta = entry["meta"]
        item = {
            "case": [json.loads(json.dumps(part, default=list)) for part in key],
            "panel": key[2], "label": key[3], "field": key[4], "metric": key[5],
            "geometry": dict(_geom(key)),
            "logical_bytes": meta.logical_bytes,
        }
        values = {}
        for (host, version, arm), runs in sorted(entry["arms"].items(), key=str):
            values.setdefault(host, {}).setdefault(version, {})[arm] = {
                "rounds": {str(r): runs[r] for r in sorted(runs)},
                "median": campaign.gf16_value(key, arm, host, version),
            }
        item["values"] = values
        payload.append(item)
    return payload


def build_evidence(campaign: Campaign, renderer: Renderer) -> bytes:
    payload = {
        "generator": "benches/publish.py",
        "records_root": str(campaign.records_root),
        "metadata": campaign.metadata,
        "hosts": [{"name": host,
                   "environment_sha256": campaign.environments[host]["sha256"],
                   "environment_text": campaign.environments[host]["text"]}
                  for host in campaign.hosts],
        "order": campaign.order_rows,
        "logs": campaign.log_evidence,
        "cases": [_case_payload(campaign, key, data)
                  for key, data in sorted(campaign.cases.items(), key=str)],
        "trio_cases": _competitor_payload(campaign, campaign.trio,
                                          ("fgf", "isal", "klauspost")),
        "prime_cases": _competitor_payload(campaign, campaign.prime, ("fgf", "p3")),
        "gf16_cases": _gf16_payload(campaign),
        "gf16_control_arms": {
            str(key): {f"{h}|{v}": {str(r): ratio for r, ratio in sorted(ratios.items())}
                       for (h, v), ratios in entry["ratios"].items()}
            for key, entry in campaign.gf16_controls.items()},
        "controls_by_log": [
            {"host": entry["host"], "round": entry["round"],
             "version": entry["version"], "family": entry["family"],
             "controls": entry["controls"]}
            for entry in campaign.log_evidence],
        "cells": renderer.cells.refs,
        "warnings": [],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True,
                        help="raw paired-run records directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="render root for v2/, v3/, and comparison/ pages")
    parser.add_argument("--verify", action="store_true",
                        help="regenerate and compare bytes without overwriting")
    arguments = parser.parse_args(argv)

    campaign = Campaign(arguments.records)
    try:
        campaign.load()
    except PublishError as exc:
        print(f"publish: {exc}", file=sys.stderr)
        return 1
    renderer = Renderer(campaign)
    pages: dict[str, bytes] = {}
    try:
        for version in VERSIONS:
            for family in PAGE_FAMILIES:
                pages[f"{version}/{family}.md"] = renderer.render_snapshot(
                    family, version).encode("utf-8")
        for family in PAGE_FAMILIES:
            pages[f"comparison/{family}.md"] = renderer.render_comparison(
                family).encode("utf-8")
    except PublishError as exc:
        print(f"publish: {exc}", file=sys.stderr)
        return 1
    evidence = build_evidence(campaign, renderer)

    if arguments.verify:
        mismatches = []
        for relative, content in sorted(pages.items()):
            path = arguments.output / relative
            if not path.is_file():
                mismatches.append(f"missing {path}")
            elif path.read_bytes() != content:
                mismatches.append(f"changed {path}")
        evidence_path = arguments.records / "evidence.json"
        if not evidence_path.is_file():
            mismatches.append(f"missing {evidence_path}")
        elif evidence_path.read_bytes() != evidence:
            mismatches.append(f"changed {evidence_path}")
        if mismatches:
            for line in mismatches:
                print(f"publish: verify: {line}", file=sys.stderr)
            return 2
        print(f"verified {len(pages)} pages and evidence.json")
        return 0

    for relative, content in sorted(pages.items()):
        path = arguments.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (arguments.records / "evidence.json").write_bytes(evidence)
    print(f"wrote {len(pages)} pages under {arguments.output} and "
          f"{arguments.records / 'evidence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
