#!/usr/bin/env python3
"""Render the row-composition comparison page from paired campaign summaries.

    python3 benches/compositions_publish.py --records DIR --output PATH [--verify]

``DIR`` holds ``meta.json`` and, per host, ``paired-<host>.json`` and
``pilot-<host>.json`` for the adopted campaign plus ``boundary-<host>.json``
for the campaign that set the routing boundaries. Factors are base latency
over candidate latency from unrounded per-cell aggregates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish import fmt_ratio_pair, render_markdown_table  # noqa: E402

PAGE = "v3/compositions.md"
HOSTS = ("willow-cove", "golden-cove")
PRIMARY = {"willow-cove": "v4x", "golden-cove": "v3_gfni_crypto"}
LANES = (8, 64, 512, 4096)
PATTERNS = ("general", "mixed", "one", "zero")
TIERS = (("scalar", "Scalar"), ("v2", "SSE4.2"), ("primary", "AVX2 and wider"))

FIELDS = (
    ("qm31", "QuadMersenne31"),
    ("m31", "Mersenne31"),
    ("gld", "Goldilocks"),
    ("gf101", "residue GF(101)"),
    ("gf127", "residue GF(127)"),
    ("gf131", "residue GF(131)"),
    ("gf65521", "residue GF(65521)"),
    ("gf65537", "residue GF(65537)"),
    ("gf1e9p7", "residue GF(1000000007)"),
    ("gf1280000003", "residue GF(1280000003)"),
    ("gf4294967291", "residue GF(4294967291)"),
    ("gfmaxprime", "residue GF(2^64 - 59)"),
    ("m31r", "residue at the Mersenne31 modulus"),
    ("gldr", "residue at the Goldilocks modulus"),
    ("gf2", "residue GF(2)"),
    ("m31quad", "Mersenne31 components, t^2 + 2"),
    ("gldquad", "Goldilocks components, t^2 + 7"),
    ("aes8quad", "AES GF(2^8) components, degree 2"),
    ("aes8nquad", "normal-basis AES GF(2^8) components, degree 2"),
    ("cantor-pair", "Cantor-basis GF(2^8) components, degree 2"),
    ("degree64", "AES GF(2^8) components, degree 8"),
    ("packed8-base", "Fan-Paar GF(2^16) over packed GF(2^8) components"),
    ("packed-base", "Rijndael GF(2^32) over packed GF(2^16) components"),
    ("packed32-base", "Rijndael GF(2^64) over packed GF(2^32) components"),
    ("gf81", "GF(81) over GF(9) components"),
    ("maxprimequad", "GF(2^64 - 59) components, degree 2"),
    ("m31cube", "Mersenne31 components, degree 3"),
    ("over-quad", "QuadMersenne31 components, degree 2"),
    ("residue3-pair", "residue GF(3) components, degree 2"),
    ("residue131-pair", "residue GF(131) components, degree 2"),
    ("residue257-pair", "residue GF(257) components, degree 2"),
    ("residue65537-pair", "residue GF(65537) components, degree 2"),
    ("residue7-cube", "residue GF(7) components, degree 3"),
    ("residue3-quartic", "residue GF(3) components, degree 4"),
    ("residue-degree8", "degree 2 over residue GF(3) degree-4 components"),
    ("residue2-pair", "residue GF(2) components, degree 2"),
)
COMPONENTS = {label for label, _ in FIELDS[15:]}
SECTIONS = (
    ("Scatter", ("mul_add_scatter", "mul_into_scatter", "mul_add_scatter_with")),
    ("Gather", ("mul_add_gather", "mul_into_gather", "mul_add_gather_with",
                "mul_into_gather_with")),
    ("Contiguous matrix", ("mul_add_matrix", "mul_into_matrix", "mul_add_matrix_with",
                           "mul_into_matrix_with")),
    ("Offset matrix", ("mul_add_matrix_at", "mul_into_matrix_at",
                       "mul_add_matrix_at_with", "mul_into_matrix_at_with")),
    ("Single row", ("mul_add", "mul_add_with")),
    ("Row addition", ("add_assign", "sub_assign", "add_assign_rows",
                      "sub_assign_rows", "add_gather_offsets")),
)
MULTI_ROW = {"mul_add_scatter", "mul_into_scatter", "mul_add_gather", "mul_into_gather",
             "mul_add_matrix", "mul_into_matrix", "mul_add_matrix_at",
             "mul_into_matrix_at"}
ADDITIVE = {"add_assign", "sub_assign", "add_assign_rows", "sub_assign_rows",
            "add_gather_offsets"}


class PublishError(Exception):
    pass


def load(path: Path) -> dict:
    if not path.is_file():
        raise PublishError(f"missing {path}")
    return json.loads(path.read_text())


def cells(config: dict) -> dict:
    """Factors keyed by (field, operation, pattern, lanes) for one config."""
    out = {}
    for cell in config["cells"]:
        parts = cell["label"].split("/")
        if len(parts) != 5 or "_control-" in parts[2]:
            continue
        out[(parts[1], parts[2], parts[3], int(parts[4][len("lanes-"):]))] = (
            float(cell["base_over_candidate"]), cell["classification"])
    return out


def config(summary: dict, features: str, tier: str, host: str) -> dict:
    request = PRIMARY[host] if tier == "primary" else tier
    found = [c for c in summary["configs"]
             if c["features"] == features and c["request"] == request]
    if len(found) != 1 or found[0]["status"] == "rejected":
        raise PublishError(f"{host}: missing or rejected config {features}/{request}")
    return found[0]


def factor_table(per_host: dict, operation: str, patterns: tuple) -> str:
    header = ["Field"] + (["Pattern"] if len(patterns) > 1 else []) + [
        f"{lanes} lanes (factor)" for lanes in LANES]
    rows = []
    for label, _ in FIELDS:
        for pattern in patterns:
            values = [tuple(per_host[host].get((label, operation, pattern, lanes),
                                               (None, None))[0] for host in HOSTS)
                      for lanes in LANES]
            if all(a is None and b is None for a, b in values):
                continue
            rows.append([label] + ([pattern] if len(patterns) > 1 else []) + [
                fmt_ratio_pair(a, b) for a, b in values])
    if not rows:
        return ""
    return render_markdown_table(header, rows)


def mechanism(field: str, operation: str, pattern: str) -> str | None:
    if field == "residue2-pair":
        return None
    if operation in MULTI_ROW and pattern in ("general", "mixed"):
        return "Per-row multiplication dispatch"
    if (operation in MULTI_ROW and pattern == "one") or operation in ADDITIVE:
        if field in COMPONENTS:
            return "Component addition through base-field dispatch"
        if field == "m31r":
            return "Mersenne31-modulus residue addition"
        if field == "gldr":
            return "Goldilocks-modulus residue addition"
    return None


def boundary_tables(boundary: dict, layouts: dict) -> list[str]:
    """Smallest factor per mechanism, row length, host, and tier."""
    worst: dict = defaultdict(lambda: defaultdict(dict))
    for host in HOSTS:
        for tier, _ in TIERS:
            for features in ("default", "all"):
                for (field, operation, pattern, lanes), (factor, _) in cells(
                        config(boundary[host], features, tier, host)).items():
                    name = mechanism(field, operation, pattern)
                    if name is None:
                        continue
                    slot = worst[name][lanes * layouts[field]]
                    slot[(host, tier)] = min(factor, slot.get((host, tier), factor))
    out = []
    for name in ("Per-row multiplication dispatch",
                 "Component addition through base-field dispatch",
                 "Mersenne31-modulus residue addition",
                 "Goldilocks-modulus residue addition"):
        if name not in worst:
            continue
        header = ["Row bytes"] + [f"{title} smallest factor" for _, title in TIERS]
        rows = [[str(size)] + [fmt_ratio_pair(*(worst[name][size].get((host, tier))
                                                for host in HOSTS))
                               for tier, _ in TIERS]
                for size in sorted(worst[name])]
        out.append(f"### {name}\n")
        out.append(render_markdown_table(header, rows) + "\n")
    return out


def outcome_table(summaries: dict) -> str:
    header = ["Features", "Request", "Win", "Tie", "Slower", "Unresolved"]
    rows = []
    for features in ("default", "all"):
        for tier, title in TIERS:
            counts = {host: Counter(c for _, c in cells(
                config(summaries[host], features, tier, host)).values()) for host in HOSTS}
            rows.append([features, title] + [
                " / ".join(str(counts[host][kind]) for host in HOSTS)
                for kind in ("win", "tie", "slower", "unresolved")])
    return render_markdown_table(header, rows)


def render(records: Path) -> str:
    meta = load(records / "meta.json")
    paired = {host: load(records / f"paired-{host}.json") for host in HOSTS}
    pilots = {host: load(records / f"pilot-{host}.json") for host in HOSTS}
    boundary = {host: load(records / f"boundary-{host}.json") for host in HOSTS}
    for host in HOSTS:
        if paired[host]["status"] == "rejected" or pilots[host]["status"] != "passing":
            raise PublishError(f"{host}: campaign or pilot not accepted")
    layouts = meta["field_layouts"]
    if set(layouts) != {label for label, _ in FIELDS}:
        raise PublishError("field layout inventory differs from the field definitions")
    per_host = {host: cells(config(paired[host], "all", "primary", host)) for host in HOSTS}

    out = [
        "# Row compositions (v3)\n",
        "Every numeric cell uses **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)** "
        "order; unavailable measurements are `-`. Shared context: "
        "[BENCHMARKS.md](../../BENCHMARKS.md).\n",
        "Each factor is base latency divided by candidate latency for the same "
        "public call, field, coefficient pattern, and geometry. **Above one, the "
        "candidate is faster.** Base and candidate are separately built artifacts "
        "of identical benchmark sources.\n",
        "## Find a result\n",
        "- [Setup](#setup)\n- [Outcomes](#outcomes)\n"
        + "".join(f"- [{title}](#{title.lower().replace(' ', '-')})\n"
                  for title, _ in SECTIONS)
        + "- [Routing boundaries](#routing-boundaries)\n"
        "- [Field definitions](#field-definitions)\n"
        "- [Measurement details](#measurement-details)\n",
        "## Setup\n",
        render_markdown_table(["Setting", "Value"], [
            ["Base revision", f"`{meta['base_revision']}`"],
            ["Candidate revision", f"`{meta['candidate_revision']}`"],
            ["Toolchain", meta["toolchain"]],
            ["Geometry", f"{meta['rows']} destination rows, {meta['sources']} sources"],
            ["Main tables", "all features; `v4x` / `v3_gfni_crypto` requests"],
            ["Protocol", "Five interleaved base/candidate pairs per host, feature set, "
             "and request; shuffled pair order."],
            ["Aggregation", "Median of five per-run medians; factors from unrounded "
             "aggregates."],
        ]) + "\n",
        "## Outcomes\n",
        "Cell counts across every field, operation, pattern, and geometry. "
        "A cell is unresolved when its field geometry is resolution-limited or its "
        "duplicate controls moved between artifacts, or when fewer than four of "
        "five paired rounds agree in direction.\n",
        outcome_table(paired) + "\n",
    ]
    for title, operations in SECTIONS:
        out.append(f"## {title}\n")
        for operation in operations:
            general = factor_table(per_host, operation, ("general",))
            others = factor_table(per_host, operation, PATTERNS[1:])
            if not general and not others:
                continue
            out.append(f"### {operation}\n")
            if general:
                out.append("General coefficients.\n\n" + general + "\n")
            if others:
                out.append("<details>\n<summary>Mixed, unit, and zero coefficients"
                           "</summary>\n\n" + others + "\n\n</details>\n")
    out.append("## Routing boundaries\n")
    out.append(
        f"Revision `{meta['boundary_revision']}` routed every row length. Each table "
        "lists the smallest factor at each measured row length across fields, "
        "operations, feature sets, and coefficient patterns that reach that route. "
        "The candidate's routing thresholds cite these tables.\n")
    out.extend(boundary_tables(boundary, layouts))
    out.append("## Field definitions\n")
    out.append(render_markdown_table(
        ["Label", "Field", "Lane bytes"],
        [[label, text, str(layouts[label])] for label, text in FIELDS]) + "\n")
    out.append("## Measurement details\n")
    out.append("<details>\n<summary>Hosts, controls, and classification</summary>\n\n"
               + "".join(f"- {line}\n" for line in meta["caveats"])
               + "\n</details>\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        content = render(arguments.records).encode("utf-8")
    except PublishError as error:
        print(f"compositions_publish: {error}", file=sys.stderr)
        return 1
    path = arguments.output / PAGE
    if arguments.verify:
        if not path.is_file() or path.read_bytes() != content:
            print(f"compositions_publish: verify: changed {path}", file=sys.stderr)
            return 2
        print(f"verified {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
