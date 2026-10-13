"""Render residue snapshots and matched backend comparisons from one campaign."""

from __future__ import annotations

import json

from publish import (
    FAMILY_VERSIONS,
    RESIDUE_ELEMENTWISE_LANES,
    RESIDUE_ELEMENTWISE_ROWS,
    RESIDUE_MATRIX_LANES,
    RESIDUE_MATRIX_SOURCES,
    RESIDUE_ROW_OF,
    RESIDUE_SINGLE_LANES,
    RESIDUE_SINGLE_ROWS,
    fmt_pair,
    fmt_ratio_pair,
    gib_per_sec,
    render_markdown_table,
    table_digits,
)

SNAPSHOT_PAGE = "v3/residue.md"
COMPARISON_PAGE = "comparison/residue.md"
MATRIX_DISPLAY_ROWS = (
    "word65537", "word67m", "word268m", "word536m",
    "word858m", "word1e9p7", "word1280000003", "wordmax",
)
SECTIONS = (
    ("Prepared single-row multiplication", "single",
     ("mul_into_with", "mul_add_with", "mul_assign_with")),
    ("One-shot single-row multiplication", "single", ("mul_into",)),
    ("Elementwise multiplication", "elementwise",
     ("mul_elementwise", "mul_elementwise_assign")),
    ("Matrix multiplication", "matrix",
     ("mul_into_matrix_with", "mul_add_matrix_with",
      "mul_into_gather_with", "mul_add_gather_with",
      "mul_into_matrix", "mul_add_matrix")),
)
DISPLAY_OPERATION = {
    "mul_into_gather_with": "Four prepared mul_into_gather_with calls",
    "mul_add_gather_with": "Four prepared mul_add_gather_with calls",
    "mul_into_matrix": "Prepare CoeffMatrix + mul_into_matrix_with",
    "mul_add_matrix": "Prepare CoeffMatrix + mul_add_matrix_with",
}
NUMERATORS = {
    "single": "Numerator: row bytes per call.",
    "elementwise": "Numerator: one operand's bytes per call.",
    "matrix": "Four outputs; numerator: row bytes × sources × outputs. "
              "Each lane occupies four bytes.",
}
CAVEATS = (
    "- Backend requests use one current artifact, not separate released residue implementations.",
    "- Short rows can take a narrower or scalar path. Exact operation routes and field caps stay in the evidence.",
    "- mul_assign_with includes restoring the staged destination inside the timed call.",
    "- Prepared matrix application excludes construction; prepare-and-apply panels include CoeffMatrix allocation and construction.",
    "- Gather panels compose four prepared output gathers at the same matrix geometry.",
    "- Direct internals timings are excluded from public tables.",
    "- Controls and per-round variability stay in the raw evidence; differences within that variability do not establish a win.",
)


def _index(campaign):
    return {
        (case.first.panel, case.first.label, case.first.field,
         tuple(sorted(case.first.geometry.items()))): (key, case.first)
        for key, case in campaign.residue_cases.items() if case.first.public
    }


def _geometry_rows(panel):
    if panel == "single":
        return [(row, None) for row in RESIDUE_SINGLE_ROWS], RESIDUE_SINGLE_LANES
    if panel == "elementwise":
        return [(row, None) for row in RESIDUE_ELEMENTWISE_ROWS], RESIDUE_ELEMENTWISE_LANES
    return ([(row, sources) for row in MATRIX_DISPLAY_ROWS
             for sources in RESIDUE_MATRIX_SOURCES], RESIDUE_MATRIX_LANES)


def _table(campaign, index, cells, page, panel, op, arm="v3", baseline=None):
    rows, lengths = _geometry_rows(panel)
    prepared = []
    values = []
    for row, sources in rows:
        label = RESIDUE_ROW_OF[row]
        rendered = []
        for lanes in lengths:
            geometry = {"lanes": lanes}
            if sources is not None:
                geometry.update(sources=sources, accumulate=op.startswith("mul_add"))
            key, rec = index[(panel, op, label, tuple(sorted(geometry.items())))]
            host_values = []
            for host in campaign.hosts:
                current = campaign.residue_value(key, host, arm)
                other = campaign.residue_value(key, host, baseline) if baseline else None
                if current is None or (baseline and other is None):
                    host_values.append(None)
                elif baseline:
                    host_values.append(other / current)
                else:
                    host_values.append(gib_per_sec(rec.logical_bytes, current))
            rendered.append((key, host_values))
            values.extend(value for value in host_values if value is not None)
        prepared.append(([label] + ([str(sources)] if sources else []), rendered))
    digits = table_digits(values) if values and baseline is None else 5
    unit = "×" if baseline else "GiB/s"
    header = ["Field"] + (["Sources"] if panel == "matrix" else [])
    header += [f"{lanes} lanes ({unit})" for lanes in lengths]
    output = []
    for labels, rendered in prepared:
        line = list(labels)
        for lanes, (key, pair) in zip(lengths, rendered, strict=True):
            line.append(fmt_ratio_pair(*pair) if baseline else fmt_pair(*pair, digits))
            cells.record(page, f"residue-{op}-{baseline or arm}", " | ".join(labels),
                         f"{lanes} lanes", key, f"{baseline}/{arm}" if baseline else arm)
        output.append(line)
    return render_markdown_table(header, output)


def _setup(campaign):
    rows = [["Revision", f"`{campaign.metadata['v3_revision']}`"],
            ["Toolchain", str(campaign.metadata.get("toolchain", "-"))],
            ["Aggregation", "Median of five per-run placement medians."],
            ["Placements", "Offsets 0, 16, 32 within each run."],
            ["Order", "Routed, AVX2, scalar requests rotate within each round."]]
    for arm in FAMILY_VERSIONS["prime"]:
        reports = [campaign.residue_backend(host, arm) or "-" for host in campaign.hosts]
        rows.append([f"Resolved process backend ({arm})", " / ".join(reports)])
    return render_markdown_table(["Setting", "Value"], rows)


def _intro(campaign):
    labels = " / ".join(campaign.host_labels[host] for host in campaign.hosts)
    return (f"Every numeric cell uses **{labels}** order; unavailable measurements "
            "are `-`. Shared context and reproduction: "
            "[BENCHMARKS.md](../../BENCHMARKS.md). "
            "[Field labels](../labels.md) fix modulus and lane representation.")


def _details(summary, body):
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def render_snapshot(campaign, renderer):
    index = _index(campaign)
    lines = ["# Generic prime residues (v3)", "", _intro(campaign), "",
             "Current public calls under the routed request. Backend factors live "
             "in the [comparison](../comparison/residue.md); generic residues "
             "have no released-v2 counterpart.", "", "## Setup", "", _setup(campaign),
             "", "## Self-timings", ""]
    for title, panel, ops in SECTIONS:
        lines += [f"### {title}", "", NUMERATORS[panel], ""]
        for op in ops:
            table = _table(campaign, index, renderer.cells, SNAPSHOT_PAGE, panel, op)
            lines += [_details(DISPLAY_OPERATION.get(op, op), table), ""]
    lines += ["## Competitors", "", "No matched competitor measurements are available.",
              "", "## Caveats", "", *CAVEATS]
    return "\n".join(lines).rstrip() + "\n"


def render_comparison(campaign, renderer):
    index = _index(campaign)
    lines = ["# Generic prime residues: backend comparisons", "", _intro(campaign), "",
             "Factors divide AVX2 or scalar latency by current routed latency on "
             "the same workload. Above one means the routed call finished sooner; "
             "below one remains a regression. These are backend comparisons, "
             "not released-v2 comparisons. [Absolute values](../v3/residue.md).",
             "", "## Setup", "", _setup(campaign), "", "## Factors", ""]
    for title, panel, ops in SECTIONS:
        lines += [f"### {title}", ""]
        for op in ops:
            body = []
            for arm, name in (("avx2", "AVX2 / current routed"),
                              ("scalar", "Scalar / current routed")):
                body += [f"#### {name}", "", _table(
                    campaign, index, renderer.cells, COMPARISON_PAGE, panel, op,
                    baseline=arm), ""]
            lines += [_details(DISPLAY_OPERATION.get(op, op), "\n".join(body).rstrip()), ""]
    lines += ["## Caveats", "", *CAVEATS]
    return "\n".join(lines).rstrip() + "\n"


def residue_payload(campaign):
    cases = []
    for key, case in sorted(campaign.residue_cases.items(), key=str):
        rec = case.first
        values = {}
        for (host, arm), series in sorted(case.values.items()):
            values.setdefault(host, {})[arm] = {
                "rounds": {str(r): series.rounds[r] for r in sorted(series.rounds)},
                "median": campaign.residue_value(key, host, arm),
            }
        cases.append({
            "case": [json.loads(json.dumps(part, default=list)) for part in key],
            "panel": rec.panel, "label": rec.label, "raw_label": rec.raw_label,
            "row": rec.field, "metric": rec.metric, "geometry": dict(rec.geometry),
            "logical_bytes": rec.logical_bytes, "public": rec.public,
            "disposition": rec.disposition, "values": values,
            "availability": {arm: {host: (host, arm) in case.values
                                    for host in campaign.hosts}
                             for arm in FAMILY_VERSIONS["prime"]},
        })
    return {"cases": cases,
            "state": {f"{host}|{arm}": state for (host, arm), state
                      in sorted(campaign.residue_state.items(), key=str)}}
