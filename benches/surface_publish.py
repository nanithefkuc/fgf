"""Render the current-only public-API surfaces family from one campaign.

The surfaces family is measured only on the current artifact: five rounds per
host, one ``round-N-v3-surfaces.log`` per round holding a surfaces section
(the public-API artifact) and a ``gf16-presentations`` section (the GF16
harness under ``--presentations``). This module collects the parsed logs onto
the campaign, validates them fail-closed, renders ``v3/public-api.md`` (the
surfaces cases), ``v3/maps.md`` (surfaces ``map `` operations plus the
presentations' ``end_to_end`` conversion caller loops), and
``v3/gf16-presentations.md`` (native and preparation panels), and builds the
surfaces side of ``evidence.json``. New public-API surfaces have no
released-v2 counterpart, so the pages carry no version factors.
"""

from __future__ import annotations

import json
import re

from publish import (
    GF16_ARM_DISPLAY,
    GF16_ARMS_OK,
    LATENCY,
    METRIC_UNIT,
    PublishError,
    ROUNDS,
    THROUGHPUT,
    fail,
    fmt_bytes,
    fmt_pair,
    fmt_ratio,
    fmt_ratio_pair,
    gib_per_sec,
    median,
    render_markdown_table,
    table_digits,
)

from check_surfaces import (
    GF16_FGF_ARMS,
    SURFACE_SECTIONS,
    group_of,
    identity_digest,
    is_presentation_map_record,
    is_surfaces_map_case,
    load_manifest,
    public_universe,
    validate_case_api,
    validate_required_coverage,
    workload_of,
)

PUBLIC_API_PAGE = "v3/public-api.md"
MAPS_PAGE = "v3/maps.md"
PRESENTATIONS_PAGE = "v3/gf16-presentations.md"
# Display order: measured steady state first, query latency, preparation,
# then compositions.
SCOPE_ORDER = ("steady", "query", "setup", "end_to_end")
SCOPE_TITLES = {
    "steady": "Steady-state operations",
    "query": "Query and scalar latency",
    "setup": "Preparation and setup",
    "end_to_end": "End-to-end compositions",
}
SCOPE_PROSE = {
    "steady": "Repeated public calls on prebuilt fixtures. Explicit prepared "
              "coefficients stay outside timing; work internal to the named "
              "public call remains inside it.",
    "query": "Black-boxed scalar and getter calls; these report per-call "
             "latency, never a throughput.",
    "setup": "Construction and preparation costs excluded from steady-state "
             "panels; owned results are destroyed before the next iteration.",
    "end_to_end": "Public compositions from construction through consumption, "
                  "timed as one call.",
}
PRESENTATION_SCOPE_ORDER = ("native", "setup")
PRESENTATION_SCOPE_TITLES = {
    "native": "Native presentations",
    "setup": "Preparation and setup",
}
MAPS_PROSE = (
    "Map panels time public scalar calls and caller loops over validated "
    "exact isomorphisms; no bulk-map API exists. Map fixtures are validated "
    "outside the timed region, and map records report the scalar route.")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class SurfaceEntry:
    """One surfaces-section case across hosts and rounds."""

    def __init__(self, first: dict) -> None:
        self.first = first
        self.values: dict[str, dict[int, float]] = {}
        self.backends: dict[str, str] = {}

    def add(self, host: str, rnd: int, record: dict, where: str) -> None:
        series = self.values.setdefault(host, {})
        if rnd in series:
            fail(f"{where}: duplicate surfaces case in one round: "
                 f"{self.first['case']!r}")
        series[rnd] = record["ns"]
        if self.backends.setdefault(host, record["backend"]) != record["backend"]:
            fail(f"{where}: backend changed between rounds for "
                 f"{self.first['case']!r}")

    @property
    def is_control(self) -> bool:
        return self.first.get("control") is True

    @property
    def is_map(self) -> bool:
        return is_surfaces_map_case(self.first)

    @property
    def metric(self) -> str:
        return THROUGHPUT if self.first["bytes"] > 0 else LATENCY


class PresentationEntry:
    """One gf16-presentations case group (identity without arm) per arm."""

    def __init__(self, first: dict) -> None:
        self.first = first
        self.arms: dict[str, dict[str, dict[int, float]]] = {}
        self.backends: dict[str, dict[str, str]] = {}

    def add(self, host: str, rnd: int, record: dict, where: str) -> None:
        if record["arm"] == "fgf":
            # The fgf arm's registered case labels the whole group in pages.
            self.first = record
        series = self.arms.setdefault(record["arm"], {}).setdefault(host, {})
        if rnd in series:
            fail(f"{where}: duplicate presentation case in one round: "
                 f"{record['case']!r} ({record['arm']!r})")
        series[rnd] = record["ns"]
        reported = self.backends.setdefault(record["arm"], {})
        if reported.setdefault(host, record["backend"]) != record["backend"]:
            fail(f"{where}: backend changed between rounds for "
                 f"{record['case']!r} ({record['arm']!r})")

    @property
    def is_map(self) -> bool:
        return is_presentation_map_record(self.first)

    @property
    def metric(self) -> str:
        return THROUGHPUT if self.first["bytes"] > 0 else LATENCY

    def fgf_backend(self) -> str | None:
        backends = {host: backend for arm in GF16_FGF_ARMS
                    for host, backend in self.backends.get(arm, {}).items()}
        return next(iter(backends.values()), None)


def collect_log(campaign, parser, host: str, rnd: int, digest: str) -> None:
    """Fold one parsed surfaces round log onto the campaign."""
    details = campaign.metadata["hosts"][host]
    where = f"{host} round {rnd}"
    backends = parser.process_backends()
    for section in parser.sections:
        if backends.get(section) != details["requested_backend"]:
            fail(f"{where}: {section} process backend {backends.get(section)!r} "
                 f"does not match the requested "
                 f"{details['requested_backend']!r}")
    for placement in parser.host_placements:
        if placement["cpus"] != str(details["core"]) or placement["policy"] != 5:
            fail(f"{where}: a surfaces-family process is not core-pinned under "
                 f"SCHED_IDLE: {placement}")
    state = campaign.surface_states.setdefault(host, {"digests": {}})
    state["digests"][rnd] = {
        section: identity_digest(parser.inventories[section])
        for section in parser.sections}
    for section in parser.sections:
        records = {workload_of(record, section): record
                   for record in parser.records[section]}
        for entry in parser.inventories[section]:
            record = records[workload_of(entry, section)]
            if section == "surfaces":
                key = workload_of(entry, section)
                case = campaign.surface_entries.get(key)
                if case is None:
                    case = campaign.surface_entries[key] = SurfaceEntry(entry)
                case.add(host, rnd, record, where)
            else:
                key = group_of(entry)
                case = campaign.presentation_entries.get(key)
                if case is None:
                    case = campaign.presentation_entries[key] = \
                        PresentationEntry(entry)
                case.add(host, rnd, record, where)
    campaign.surface_logs.append({
        "host": host, "round": rnd, "sha256": digest,
        "process_backends": backends,
        "presentations_header": parser.presentations_header,
        "banners": parser.banners,
        "host_placements": parser.host_placements,
        "timer_floor": parser.timer_floor,
        "surfaces_complete": parser.surfaces_complete,
        "inventory_digests": state["digests"][rnd],
        "controls": [{"case": record["case"], "phase": record["phase"],
                      "ns": record["ns"]}
                     for record in parser.records["surfaces"]
                     if record.get("control")],
    })


def validate(campaign) -> None:
    """Campaign-level surfaces validation and aggregation."""
    manifest = load_manifest(campaign.surface_manifest)
    universe = public_universe(campaign.surface_rustdoc, manifest)
    if campaign.metadata.get("rustdoc_json_sha256") != universe.sha256:
        fail("campaign rustdoc JSON digest does not match its compiler-derived API inventory")
    inventory_cases = []
    for host, rnd, parser in campaign.surface_round_parsers:
        for section in parser.sections:
            for entry in parser.inventories[section]:
                validate_case_api(entry, universe, f"{host} round {rnd}: {section}")
    for host, state in sorted(campaign.surface_states.items()):
        digests = state["digests"]
        if sorted(digests) != list(ROUNDS):
            fail(f"{host}: surfaces rounds {sorted(digests)} are incomplete")
        reference = digests[ROUNDS[0]]
        for rnd in ROUNDS[1:]:
            if digests[rnd] != reference:
                fail(f"{host}: surfaces inventories changed at round {rnd}")
    for key, entry in sorted(campaign.surface_entries.items(), key=str):
        for host, series in sorted(entry.values.items()):
            if sorted(series) != list(ROUNDS):
                fail(f"{host}: surfaces case {entry.first['case']!r} is "
                     f"missing rounds")
            campaign.surface_aggregates[(key, host)] = median(
                [series[rnd] for rnd in ROUNDS])
        if not entry.is_control:
            inventory_cases.append(entry.first)
    controls = [entry for entry in campaign.surface_entries.values()
                if entry.is_control]
    if len(controls) != 2:
        fail(f"the surfaces family must register exactly one entry and one "
             f"exit control, found {len(controls)}")
    for key, entry in sorted(campaign.presentation_entries.items(), key=str):
        for arm, arms in sorted(entry.arms.items()):
            if arm not in GF16_ARMS_OK:
                fail(f"presentation case {entry.first['case']!r} reports "
                     f"unknown arm {arm!r}")
            if set(arms) != set(campaign.available_hosts):
                fail(f"presentation {entry.first['case']!r} ({arm!r}) lacks "
                     f"available-host measurements")
            for host, series in sorted(arms.items()):
                if sorted(series) != list(ROUNDS):
                    fail(f"{host}: presentation case {entry.first['case']!r} "
                         f"({arm!r}) is missing rounds")
                campaign.presentation_aggregates[(key, arm, host)] = median(
                    [series[rnd] for rnd in ROUNDS])
        if "fgf" not in entry.arms:
            fail(f"presentation group {entry.first['case']!r} lacks its fgf "
                 f"arm across hosts")
        inventory_cases.append(entry.first)
    _validate_workload_sets(campaign)
    _validate_page_coverage(campaign)
    unique = {json.dumps(entry, sort_keys=True, ensure_ascii=False): entry
              for entry in inventory_cases}
    validate_required_coverage(list(unique.values()), manifest, universe)
    _aggregate_control_ratios(campaign)


def _validate_workload_sets(campaign) -> None:
    """Inventories must match across hosts for both sections."""
    hosts = list(campaign.available_hosts)
    surfaces = _workload_keys(campaign.surface_entries, hosts,
                              lambda entry: entry.values)
    presentations = _workload_keys(
        campaign.presentation_entries, hosts,
        lambda entry: {host for arms in entry.arms.values() for host in arms},
    )
    reference = hosts[0]
    for host in hosts[1:]:
        if surfaces[host] != surfaces[reference]:
            fail(f"{host}: surfaces workloads differ from {reference}")
        if presentations[host] != presentations[reference]:
            fail(f"{host}: presentation workloads differ from {reference}")


def _workload_keys(entries: dict, hosts: list[str], present) -> dict:
    return {host: frozenset(key for key, entry in entries.items()
                            if host in present(entry))
            for host in hosts}


def _validate_page_coverage(campaign) -> None:
    """Every page must hold at least one case; nothing may be silently
    dropped between them."""
    public = [entry for entry in campaign.surface_entries.values()
              if not entry.is_control and not entry.is_map]
    surface_maps = [entry for entry in campaign.surface_entries.values()
                    if not entry.is_control and entry.is_map]
    presentation_maps = [entry for entry in campaign.presentation_entries.values()
                         if entry.is_map]
    presentation_panels = [entry for entry in
                           campaign.presentation_entries.values()
                           if not entry.is_map]
    if not public:
        fail("the surfaces family registered no public-API operations; the "
             "public-API page would be empty")
    if not surface_maps and not presentation_maps:
        fail("no map operations were registered; the maps page would be empty")
    if not presentation_panels:
        fail("the presentations family registered no native or preparation "
             "cases; the presentations page would be empty")


def _aggregate_control_ratios(campaign) -> None:
    for key, entry in campaign.presentation_entries.items():
        fgf = entry.arms.get("fgf", {})
        control = entry.arms.get("fgf-control", {})
        if not control:
            continue  # only native groups carry the duplicate-fgf arm
        for host in campaign.available_hosts:
            if host not in fgf or host not in control:
                fail(f"{host}: presentation case {entry.first['case']!r} lacks "
                     f"its duplicate-fgf control arm")
            ratios = [control[host][rnd] / fgf[host][rnd] for rnd in ROUNDS]
            campaign.control_ratios[(key, host)] = (
                median(ratios), min(ratios), max(ratios))


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def surface_value(campaign, key: tuple, host: str) -> float | None:
    return campaign.surface_aggregates.get((key, host))


def surface_display(campaign, entry: SurfaceEntry, host: str) -> float | None:
    ns = surface_value(campaign, workload_of(entry.first, "surfaces"), host)
    if ns is None:
        return None
    if entry.metric == THROUGHPUT:
        return gib_per_sec(entry.first["bytes"], ns)
    return ns


def presentation_value(campaign, key: tuple, arm: str, host: str) -> float | None:
    return campaign.presentation_aggregates.get((key, arm, host))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _intro(campaign, *, nested=False) -> str:
    labels = " / ".join(campaign.host_labels[host] for host in campaign.hosts)
    parent = "../" if nested else ""
    return (f"Every result cell reports **{labels}**. Throughput: higher is better; "
            "latency: lower is better. A dash means no measurement.\n\n"
            f"[All benchmarks]({parent}../../BENCHMARKS.md) · "
            f"[Field definitions]({parent}../labels.md)")


def _setup_rows(campaign) -> list[list[str]]:
    rows = [["Campaign", "`just bench-paired` (current-only family `surfaces`)"]]
    revision = campaign.metadata.get("v3_revision")
    if isinstance(revision, str):
        rows.append(["Revision", f"`{revision}` plus the measured working-tree "
                                 f"archive; campaign evidence fixes its digest."])
    toolchain = campaign.metadata.get("toolchain")
    if toolchain is not None:
        rows.append(["Toolchain", str(toolchain)])
    process = []
    for host in campaign.hosts:
        if host not in campaign.available_hosts:
            process.append("-")
            continue
        reported = {log["process_backends"][section]
                    for log in campaign.surface_logs
                    if log["host"] == host
                    for section in SURFACE_SECTIONS}
        if len(reported) != 1:
            fail(f"{host}: inconsistent surfaces process backends: {sorted(reported)}")
        process.append(f"`{reported.pop()}`")
    rows.append(["Resolved process backend", " / ".join(process)])
    field_routes = {}
    for host in campaign.hosts:
        for log in campaign.surface_logs:
            if log["host"] == host and log["round"] == ROUNDS[0]:
                for banner in log["banners"]:
                    field_routes.setdefault(banner["label"], {})[host] = banner["backend"]
    for label, routes in sorted(field_routes.items()):
        rows.append([f"`{label}` backend", " / ".join(
            f"`{routes[host]}`" if host in routes else "-" for host in campaign.hosts)])
    rows.append(["Protocol", "Five rounds per host; each round runs the "
                             "public-API surfaces artifact and the GF16 "
                             "presentations harness into one log; inventories "
                             "must match across rounds and hosts."])
    rows.append(["Aggregation", "Median of five per-run medians."])
    rows.append(["Metrics", "Cases with a logical byte numerator report "
                            "GiB/s; scalar, query, and getter cases report "
                            "per-call latency."])
    rows.append(["Timer floor", "Each public-API process measures a black-boxed "
                                "u64 loop control; per-run latency and sample "
                                "count remain in the campaign evidence. "
                                "Query timings include loop overhead; values "
                                "near this floor do not distinguish work from "
                                "measurement overhead. The floor-limited-round "
                                "count identifies medians at or below each "
                                "run's control, in the declared host order."])
    rows.append(["Validation", "Fixture correctness is proven by `--check` "
                               "captures validated with "
                               "`benches/check_surfaces.py`; campaign logs "
                               "carry measurement records only."])
    return rows



def _floor_limited_rounds(campaign, entry: SurfaceEntry, host: str) -> list[int]:
    if entry.metric != LATENCY:
        return []
    floors = {log["round"]: log["timer_floor"]["ns"]
              for log in campaign.surface_logs if log["host"] == host}
    return [rnd for rnd, ns in sorted(entry.values.get(host, {}).items())
            if ns <= floors[rnd]]


GEOMETRY_COLUMNS = (
    ("bytes", "Counted bytes"), ("lanes", "Elements"),
    ("sources", "Sources"), ("destinations", "Destinations"),
)
TASK_TITLES = {
    "buffer-addition": "Buffer addition and subtraction",
    "buffer-products": "Elementwise buffer products",
    "row-multiplication": "Single-row multiplication",
    "multirow": "Scatter, gather, and matrix",
    "scalar-arithmetic": "Scalar arithmetic",
    "scalar-inherent": "Inherent scalar arithmetic",
    "encodings": "Encoding and coordinate conversion",
    "queries": "Queries and prepared-state access",
    "preparation": "Preparation and construction",
    "field-math": "Field-specific arithmetic",
    "bits": "Bit-packed operations",
}


def _details(summary, body):
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def _geometry_value(first, key):
    value = first[key]
    return fmt_bytes(value) if key == "bytes" else f"{value:,}"


def _geometry_layout(entries):
    varying, fixed = [], []
    for key, label in GEOMETRY_COLUMNS:
        values = {entry.first[key] for entry in entries}
        if len(values) > 1:
            varying.append((key, label))
        elif key != "bytes" or next(iter(values)) != 0:
            fixed.append(f"**{label}:** {_geometry_value(entries[0].first, key)}")
    return varying, " · ".join(fixed)


def _calls(entries):
    groups = {}
    for entry in entries:
        groups.setdefault(tuple(entry.first["api"]), set()).add(entry.first["field"])
    rows = [[", ".join(sorted(fields)),
             ", ".join(f"`{path}`" for path in api)]
            for api, fields in sorted(groups.items())]
    if len(rows) == 1:
        body = rows[0][1]
    else:
        body = render_markdown_table(["Fields", "Public callable paths"], rows)
    return _details("Exact public calls", body)


def _recipe(entry):
    return entry.first["case"].rsplit("/", 1)[0]


def _operation_title(entry):
    operation, separator, qualifier = entry.first["operation"].partition(" (")
    title = operation.replace("-", "_")
    prefix = entry.first["case"].split("/", 1)[0]
    if prefix in ("bulk", "spread", "details-maps", "bits"):
        primary = entry.first["api"][0]
        namespace, _, function = primary.rpartition("::")
        if namespace in ("fgf::ops", "fgf::bits"):
            title = function
            if not separator and any(word in operation.split("-")
                                     for word in ("empty", "single", "zero", "unit")):
                title += f" ({operation.replace('-', ' ')})"
    if separator:
        title += separator + qualifier
    if entry.first["case"].startswith("scalar/inherent/"):
        title += " (inherent)"
    variant = _recipe(entry).rsplit("/", 1)[-1]
    if variant in ("short", "nonresident"):
        title += f" ({'short buffers' if variant == 'short' else 'nonresident buffers'})"
    return title


def _surface_table(campaign, renderer, page: str, table_id: str,
                   entries: list[SurfaceEntry], metric: str) -> str:
    entries = sorted(entries, key=lambda entry: (entry.first["field"], entry.first["case"]))
    geometry, fixed = _geometry_layout(entries)
    values = [[surface_display(campaign, entry, host) for host in campaign.hosts]
              for entry in entries]
    digits = table_digits([v for pair in values for v in pair if v is not None])
    floors = [[len(_floor_limited_rounds(campaign, entry, host))
               if host in campaign.available_hosts else None for host in campaign.hosts]
              for entry in entries]
    show_floor = metric == LATENCY and any(count for pair in floors for count in pair)
    field_label = "Map pair (S → T)" if any("->" in e.first["field"] for e in entries) else "Field"
    header = [field_label, *[label for _, label in geometry],
              f"{'Throughput' if metric == THROUGHPUT else 'Latency'} ({METRIC_UNIT[metric]})"]
    if show_floor:
        header.append("Floor-limited rounds")
    lines = []
    for entry, pair, floor in zip(entries, values, floors, strict=True):
        labels = [entry.first["field"].replace("->", " → "),
                  *[_geometry_value(entry.first, key) for key, _ in geometry]]
        line = labels + [fmt_pair(*pair, digits)]
        if show_floor:
            line.append(" / ".join("-" if count is None else f"{count}/{len(ROUNDS)}"
                                   for count in floor))
        lines.append(line)
        renderer.cells.record(page, table_id, " | ".join(labels), "Value",
                              workload_of(entry.first, "surfaces"), "v3")
    parts = [fixed, render_markdown_table(header, lines)]
    if metric == LATENCY and not show_floor:
        parts.append("No measured rounds are at or below the timer-floor control.")
    parts.append(_calls(entries))
    return "\n\n".join(part for part in parts if part)

def _table_anchor(scope, recipe, operation, metric):
    return re.sub(r"[^a-z0-9]+", "-", f"{scope}-{recipe}-{operation}-{metric}".lower()).strip("-")



def _surface_scope_sections(campaign, renderer, page, entries):
    lines = []
    for scope in SCOPE_ORDER:
        scope_entries = [e for e in entries if e.first["scope"] == scope]
        if not scope_entries:
            continue
        lines += [f"## {SCOPE_TITLES[scope]}", "", SCOPE_PROSE[scope], ""]
        groups = {}
        for entry in scope_entries:
            groups.setdefault((_recipe(entry), _operation_title(entry), entry.metric), []).append(entry)
        for (recipe, operation, metric), group in sorted(groups.items()):
            table_id = f"{scope}-{recipe}-{operation}-{metric}"
            anchor = _table_anchor(scope, recipe, operation, metric)
            lines += [f'<a id="{anchor}"></a>', "",
                      f"### {_operation_title(group[0])}", "",
                      _surface_table(campaign, renderer, page, table_id, group, metric), ""]
    return lines


def _task(entry):
    prefix = entry.first["case"].split("/", 1)[0]
    operation = entry.first["operation"]
    if prefix == "bulk":
        if operation.startswith(("add", "sub")):
            return "buffer-addition"
        if operation.startswith("elementwise"):
            return "buffer-products"
        return "multirow" if operation.startswith("matrix") else "row-multiplication"
    if prefix == "scalar":
        return "scalar-inherent" if "/inherent/" in entry.first["case"] else "scalar-arithmetic"
    return {"spread": "multirow", "details-maps": "multirow", "details-math": "field-math",
            "convert": "encodings", "query": "queries", "setup": "preparation",
            "bits": "bits"}[prefix]


def _navigation(entries):
    rows = []
    for scope in SCOPE_ORDER:
        scope_entries = [e for e in entries if e.first["scope"] == scope]
        if scope_entries:
            title = SCOPE_TITLES[scope]
            rows.append(f"- [{title}](#{title.lower().replace(' ', '-')})")
            operations = {(_recipe(e), _operation_title(e), e.metric) for e in scope_entries}
            for recipe, operation, metric in sorted(operations):
                anchor = _table_anchor(scope, recipe, operation, metric)
                rows.append(f"  - [{operation}](#{anchor})")
    return ["## Find a result", "", *rows, ""]


def render_public_api(campaign, renderer) -> str:
    entries = [e for e in campaign.surface_entries.values() if not e.is_control and not e.is_map]
    if not entries:
        fail(f"{PUBLIC_API_PAGE}: no measured public operations")
    groups = {}
    for entry in entries:
        groups.setdefault(_task(entry), []).append(entry)
    lines = ["# Public operations", "", _intro(campaign), "",
             "Choose the work first, then find the field in its operation table. "
             "Each measured case appears once. Preparation and compositions remain "
             "separate from steady-state calls.", "",
             "[Field maps](maps.md) · [GF16 presentations](gf16-presentations.md)", ""]
    rows = []
    for task, title in TASK_TITLES.items():
        group = groups.get(task)
        if not group:
            continue
        page = f"v3/public-api/{task}.md"
        operations = sorted({_operation_title(e) for e in group})
        rows.append([f"[{title}](public-api/{task}.md)",
                     str(len(operations)), str(len(group))])
        body = [f"# {title}", "", _intro(campaign, nested=True), "",
                "[Public operations index](../public-api.md)", ""]
        body += _navigation(group)
        body += _surface_scope_sections(campaign, renderer, page, group)
        body += ["## Measurement details", "",
                 "[Campaign, backend routes, timer floor, and validation]"
                 "(../public-api.md#measurement-details)", ""]
        renderer.pages[page] = "\n".join(body).rstrip() + "\n"
    lines += [render_markdown_table(["Task", "Operations", "Cases"], rows), "",
              "## Reading geometry", "",
              "**Counted bytes** is the throughput numerator, not necessarily one "
              "row's size. **Elements**, **Sources**, and **Destinations** are the "
              "registered fixture counts. Shared counts appear above a table; only "
              "varying counts occupy columns. **S → T** names a prepared map's field "
              "pair, not every method's value direction.", "",
              "## Measurement details", "",
              _details("Setup, resolved field backends, timer floor, and validation",
                       render_markdown_table(["Setting", "Value"], _setup_rows(campaign))), "",
              "Timer-floor counts remain beside affected latency rows. A count of "
              "five means all five rounds are at or below the loop control; such "
              "figures do not distinguish call cost from measurement overhead.", "",
              "Raw case identifiers, exact per-case API paths, controls, and every "
              "round remain in the campaign evidence."]
    return "\n".join(lines).rstrip() + "\n"


def render_maps(campaign, renderer) -> str:
    surface_maps = [e for e in campaign.surface_entries.values() if not e.is_control and e.is_map]
    presentation_maps = [e for e in campaign.presentation_entries.values() if e.is_map]
    if not surface_maps and not presentation_maps:
        fail(f"{MAPS_PAGE}: no measured map operations")
    lines = ["# Field maps", "", _intro(campaign), "", MAPS_PROSE, "",
             "[Public operations](public-api.md) · [GF16 presentations](gf16-presentations.md)", ""]
    lines += ["Pairs identify **S → T** in the prepared map. For embeddings, "
              "`restrict`, `trace`, and `norm` run **T → S**; `frobenius` runs "
              "**T → T**.", ""]
    lines += _navigation(surface_maps)
    lines += ["- [Conversion caller loops](#conversion-caller-loops)", ""]
    lines += _surface_scope_sections(campaign, renderer, MAPS_PAGE, surface_maps)
    if presentation_maps:
        lines += ["## Conversion caller loops", "",
                  "Public per-element `Isomorphism` calls over 64 KiB. Packing "
                  "inside or outside the timed body is labelled separately.", ""]
        for field_name in sorted({e.first["field"] for e in presentation_maps}):
            group = [e for e in presentation_maps if e.first["field"] == field_name]
            lines += [f"### {field_name}", "", _presentation_table(
                campaign, renderer, MAPS_PAGE, f"maps-{field_name}", group, THROUGHPUT), ""]
    lines += ["## Measurement details", "",
              "[Campaign, backend routes, and timer-floor interpretation]"
              "(public-api.md#measurement-details)", "",
              "Map calls report the scalar route. A process backend is not "
              "evidence of SIMD execution for these calls."]
    return "\n".join(lines).rstrip() + "\n"


def render_presentations(campaign, renderer) -> str:
    panels = [e for e in campaign.presentation_entries.values() if not e.is_map]
    if not panels:
        fail(f"{PRESENTATIONS_PAGE}: no measured GF16 presentations")
    fields = sorted({e.first["field"] for e in panels})
    lines = ["# GF16 presentations", "", _intro(campaign), "",
             "Each library uses its native encoding. Field maps and fixtures are "
             "validated outside timing. **Fixed** reuses coefficients; **Cycling** "
             "rotates sixteen prepared sets. Conversion caller loops live on the "
             "[maps page](maps.md).", "", "## Find a presentation", ""]
    lines += [f"- [{field}](#{field})" for field in fields]
    for field_name in fields:
        field_entries = [e for e in panels if e.first["field"] == field_name]
        lines += ["", f"## {field_name}", ""]
        for scope in PRESENTATION_SCOPE_ORDER:
            scope_entries = [e for e in field_entries if e.first["scope"] == scope]
            if not scope_entries:
                continue
            lines += [f"### {PRESENTATION_SCOPE_TITLES[scope]}", ""]
            for metric in (THROUGHPUT, LATENCY):
                metric_entries = [e for e in scope_entries if e.metric == metric]
                if metric_entries:
                    table_id = f"presentations-{field_name}-{scope}-{metric}"
                    lines += [_presentation_table(
                        campaign, renderer, PRESENTATIONS_PAGE, table_id, metric_entries, metric), ""]
        controls = _presentation_controls(campaign, renderer, field_name, field_entries)
        if controls is not None:
            lines += [_details("Duplicate-fgf control ratios",
                               "Ratios divide duplicate-fgf latency by primary-fgf latency; "
                               "medians and ranges span five runs.\n\n" + controls), ""]
    lines += ["## Measurement details", "",
              "[Campaign, backend routes, and validation](public-api.md#measurement-details)", "",
              "Differences comparable to duplicate-fgf control variability do not "
              "establish a performance difference. Counted bytes are the throughput "
              "numerator, not necessarily one row's size."]
    return "\n".join(lines).rstrip() + "\n"


def _schedule(entry):
    return "Cycling" if entry.first["case"].split("/")[3] == "cycling16" else "Fixed"


def _presentation_operation(entry):
    return {
        "isomorphism_convert_packing_included": "Convert (packing timed)",
        "isomorphism_convert_packing_outside": "Convert (packing outside timing)",
    }.get(entry.first["operation"], entry.first["operation"])


def _presentation_table(campaign, renderer, page, table_id, entries, metric):
    # Share geometry above a table instead of crowding each native-library row.
    groups = {}
    for entry in entries:
        arms = tuple(a for a in ("fgf", *sorted(GF16_ARMS_OK - {"fgf", "fgf-control"}))
                     if a in entry.arms)
        shape = tuple(entry.first[key] for key, _ in GEOMETRY_COLUMNS)
        groups.setdefault((arms, shape), []).append(entry)
    tables = []
    for (arms, shape), group in sorted(groups.items()):
        _, fixed = _geometry_layout(group)
        subtable_id = f"{table_id}-{','.join(arms)}-{'-'.join(map(str, shape))}"
        schedules = {_schedule(e) for e in group}
        show_schedule = len(schedules) > 1
        rows, values = [], []
        for entry in sorted(group, key=lambda e: (
                e.first["operation"], _schedule(e),
                e.first["sources"], e.first["destinations"], e.first["bytes"])):
            key = group_of(entry.first)
            pairs = []
            for arm in arms:
                pair = [presentation_value(campaign, key, arm, h) for h in campaign.hosts]
                if metric == THROUGHPUT:
                    pair = [gib_per_sec(entry.first["bytes"], v) if v is not None else None
                            for v in pair]
                pairs.append(pair)
                values.extend(v for v in pair if v is not None)
            rows.append((entry, key, pairs))
        digits = table_digits(values) if values else 3
        header = ["Operation"] + (["Coefficients"] if show_schedule else [])
        header += [f"{GF16_ARM_DISPLAY[a]} ({METRIC_UNIT[metric]})" for a in arms]
        lines = []
        for entry, key, pairs in rows:
            labels = [_presentation_operation(entry)]
            if show_schedule:
                labels.append(_schedule(entry))
            lines.append(labels + [fmt_pair(*pair, digits) for pair in pairs])
            for arm, pair in zip(arms, pairs, strict=True):
                if any(v is not None for v in pair):
                    renderer.cells.record(page, subtable_id, " | ".join(labels),
                                          GF16_ARM_DISPLAY[arm], key + (arm,), "v3")
        if not show_schedule:
            fixed += f" · **Coefficients:** {next(iter(schedules))}"
        routes = [", ".join(sorted({entry.backends["fgf"][host] for entry in group
                                   if host in entry.backends.get("fgf", {})})) or "-"
                  for host in campaign.hosts]
        fixed += " · **fgf routes:** " + " / ".join(routes)
        tables += [fixed, render_markdown_table(header, lines), _calls(group)]
    return "\n\n".join(tables)


def _presentation_controls(campaign, renderer, field_name: str,
                           entries: list) -> str | None:
    page = PRESENTATIONS_PAGE
    rows = []
    for entry in sorted(entries, key=lambda entry: (
            entry.first["operation"], entry.first["case"])):
        key = group_of(entry.first)
        stats = [campaign.control_ratios.get((key, host))
                 for host in campaign.hosts]
        if all(stat is None for stat in stats):
            continue
        rows.append((entry, key, stats))
    if not rows:
        return None
    geometry, fixed = _geometry_layout([entry for entry, _, _ in rows])
    header = ["Operation", "Coefficients", *[label for _, label in geometry],
              "Median ratio", "Run range"]
    lines = []
    for entry, key, stats in rows:
        labels = [_presentation_operation(entry), _schedule(entry),
                  *[_geometry_value(entry.first, k) for k, _ in geometry]]
        medians = [None if stat is None else stat[0] for stat in stats]
        ranges = ["-" if stat is None
                  else f"{fmt_ratio(stat[1])}–{fmt_ratio(stat[2])}"
                  for stat in stats]
        lines.append(labels + [fmt_ratio_pair(medians[0], medians[1]),
                               f"{ranges[0]} / {ranges[1]}"])
        renderer.cells.record(page, f"presentations-controls-{field_name}",
                              " | ".join(labels), "Median ratio", key,
                              "fgf-control/fgf")
    return fixed + "\n\n" + render_markdown_table(header, lines)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def surface_payload(campaign) -> dict:
    """The surfaces side of evidence.json: cases, presentations, controls."""
    cases = []
    for key, entry in sorted(campaign.surface_entries.items(), key=str):
        values = {}
        for host, series in sorted(entry.values.items()):
            values.setdefault(host, {})["rounds"] = {
                str(rnd): series[rnd] for rnd in sorted(series)}
            values[host]["median"] = campaign.surface_aggregates.get((key, host))
            values[host]["floor_limited_rounds"] = _floor_limited_rounds(campaign, entry, host)
        cases.append({
            "case": [json.loads(json.dumps(part, default=list))
                     for part in key],
            "identity": dict(entry.first),
            "control": entry.is_control, "map": entry.is_map,
            "metric": entry.metric,
            "backends": dict(entry.backends),
            "availability": {host: host in entry.values
                             for host in campaign.hosts},
            "values": values,
        })
    presentations = []
    for key, entry in sorted(campaign.presentation_entries.items(), key=str):
        values = {}
        backends = {}
        for arm, arms in sorted(entry.arms.items()):
            for host, series in sorted(arms.items()):
                values.setdefault(host, {}).setdefault(arm, {
                    "rounds": {str(rnd): series[rnd]
                               for rnd in sorted(series)},
                    "median": campaign.presentation_aggregates.get(
                        (key, arm, host))})
                backends.setdefault(arm, {})[host] = entry.backends[arm][host]
        presentations.append({
            "case": [json.loads(json.dumps(part, default=list))
                     for part in key],
            "identity": dict(entry.first),
            "map": entry.is_map,
            "arms": sorted(entry.arms),
            "backends": backends,
            "values": values,
        })
    ratios = {f"{host}|{key[0]}": {"median": stat[0], "min": stat[1],
                                   "max": stat[2]}
              for (key, host), stat in sorted(campaign.control_ratios.items(),
                                              key=str)}
    return {
        "cases": cases,
        "presentation_cases": presentations,
        "control_ratios": ratios,
        "inventory_digests": {host: {str(rnd): digests}
                              for host, state in
                              sorted(campaign.surface_states.items())
                              for rnd, digests in sorted(state["digests"].items())},
        "logs": campaign.surface_logs,
    }
