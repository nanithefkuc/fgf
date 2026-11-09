"""Tests for the instrumented coverage map."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_map


ROOT = Path("/repo")


def export(functions, files=None):
    return {
        "data": [
            {
                "functions": functions,
                "files": files or [],
                "totals": {},
            }
        ],
        "type": "llvm.coverage.json.export",
        "version": "3.1.0",
    }


def function(name, filename, regions, count=0, branches=None):
    return {
        "name": name,
        "filenames": [filename],
        "regions": regions,
        "branches": branches or [],
        "count": count,
        "mcdc_records": [],
    }


def branch_file(filename, branches, segments=None):
    return {
        "filename": filename,
        "branches": branches,
        "segments": segments or [],
        "expansions": [],
        "mcdc_records": [],
        "summary": {},
    }


def entry_target(entry="fgf::kernel::x86::demo::f", tier="v3", file="src/kernel.rs",
                 start=10, end=10, call_sites=()):
    return coverage_map.EntryTarget(
        entry=entry,
        tier=tier,
        file=file,
        start_line=start,
        end_line=end,
        publicly_reachable=True,
        call_sites=call_sites,
    )


def branch_site(site_id=0, key="src/kernel.rs::cmp:len<32:right"):
    return coverage_map.SiteTarget(
        site_id=site_id,
        item_key=key,
        file="src/kernel.rs",
        line=30,
        column=17,
        role="branch",
        condition_line=30,
        condition_column=5,
        condition_end_line=30,
        condition_end_column=22,
    )


def use_site(site_id=0, key="src/kernel.rs::TILE"):
    return coverage_map.SiteTarget(
        site_id=site_id,
        item_key=key,
        file="src/kernel.rs",
        line=40,
        column=8,
        role="use",
    )


def source_model(text):
    tokens = [
        SimpleNamespace(text=match.group(), start=match.start(), end=match.end(),
                        kind="ident" if match.group()[0].isalpha() else "punct")
        for match in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*|::|=>|&&|\|\||==|!=|[^\s]", text)
    ]
    declarations = [0] * len(tokens)
    scopes = [()] * len(tokens)
    for index, token in enumerate(tokens):
        if token.text in {"macro_rules", "impl", "trait"}:
            opening = next(i for i in range(index, len(tokens)) if tokens[i].text == "{")
            closing = coverage_map._delimiter_end(tokens, opening)
            label = "macro_rules! body" if token.text == "macro_rules" else token.text + " body"
            for position in range(opening, closing + 1):
                scopes[position] += (label,)
    for index, token in enumerate(tokens):
        if token.text == "fn":
            opening = next(i for i in range(index, len(tokens)) if tokens[i].text == "{")
            closing = coverage_map._delimiter_end(tokens, opening)
            declarations[opening : closing + 1] = [index] * (closing + 1 - opening)
            for position in range(opening, closing + 1):
                scopes[position] += (f"fn {tokens[index + 1].text}",)
    return SimpleNamespace(
        source=text, tokens=tokens, attribute_tokens=set(), generic_tokens=set(),
        scope_decl=lambda index: declarations[index], scopes=scopes,
    )


A_SRC = (
    "pub fn tile_kernel(rows: &mut [u8]) {\n"
    "    let cut = rows.len() >= TILE;\n"
    "}\n"
    "const TILE: usize = 4;\n"
)

A_XOR_SRC = (
    "pub fn xor_kernel(dst: &mut [u8]) {\n"
    "    let cut = dst.len() < CUTOFF;\n"
    "}\n"
    "const CUTOFF: usize = 64;\n"
)


def entry_row(target, cells):
    return {
        "entry": target.entry,
        "tier": target.tier,
        "rank": coverage_map.TIER_RANK[target.tier],
        "required": True,
        "publicly_reachable": target.publicly_reachable,
        "status": "executed",
        "evidence": "wrapper",
        "count": len(cells),
        "executed_by": list(cells),
        "executed_by_call_site": list(cells),
        "call_site_cells": [],
        "definition": {"file": target.file, "line": target.start_line},
        "reason": None,
        "macro_unattributed": [],
        "call_sites": [],
    }


class AttributionTests(unittest.TestCase):
    def test_wrapper_region_executes_the_entry(self) -> None:
        document = export(
            [function("_f", "/repo/src/kernel.rs", [[10, 1, 10, 40, 3, 0, 0, 0]], count=3)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [entry_target()], [])
        row = coverage_map.summarize_entries(
            [entry_target()], [("cell-a", evidence)], "v3", {}
        )[0]
        self.assertEqual(row["status"], "executed")
        self.assertEqual(row["evidence"], "wrapper")
        self.assertEqual(row["executed_by"], ["cell-a"])
        self.assertEqual(row["count"], 3)

    def test_non_code_regions_cannot_execute_an_entry(self) -> None:
        target = entry_target()
        for kind in (1, 3):
            with self.subTest(kind=kind):
                document = export([
                    function("_f", "/repo/src/kernel.rs", [
                        [10, 1, 10, 40, 0, 0, 0, 0],
                        [10, 1, 10, 40, 9, 0, 0, kind],
                    ]),
                ])
                evidence = coverage_map.analyze_export(document, ROOT, [target], [])
                rows = coverage_map.summarize_entries(
                    [target], [("cell", evidence)], "v3", {},
                )
                self.assertEqual(coverage_map.failures(rows, [])[0]["kind"], "entry")

    def test_unmapped_wrapper_falls_back_to_a_mapped_call_site(self) -> None:
        target = entry_target(call_sites=(("src/kernel.rs", 20, 5),))
        document = export(
            [function("_caller", "/repo/src/kernel.rs", [[18, 1, 22, 2, 4, 0, 0, 0]], count=4)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [target], [])
        row = coverage_map.summarize_entries([target], [("cell-b", evidence)], "v3", {})[0]
        self.assertEqual(row["status"], "executed")
        self.assertEqual(row["evidence"], "call_site")
        self.assertEqual(row["executed_by_call_site"], ["cell-b"])

    def test_entry_without_mapping_or_call_site_is_unexecuted(self) -> None:
        target = entry_target(call_sites=(("src/kernel.rs", 20, 5),))
        document = export([])
        evidence = coverage_map.analyze_export(document, ROOT, [target], [])
        row = coverage_map.summarize_entries([target], [("cell-c", evidence)], "v3", {})[0]
        self.assertEqual(row["status"], "unexecuted")
        self.assertEqual(coverage_map.failures([row], [])[0]["kind"], "entry")

    def test_entries_above_the_requested_tier_are_not_required(self) -> None:
        target = entry_target(tier="v4x")
        document = export([])
        evidence = coverage_map.analyze_export(document, ROOT, [target], [])
        row = coverage_map.summarize_entries([target], [("cell", evidence)], "v3", {})[0]
        self.assertFalse(row["required"])
        self.assertEqual(row["status"], "not_required")
        self.assertEqual(coverage_map.failures([row], []), [])

    def test_unmapped_arcane_exception_needs_valid_straddling_cells(self) -> None:
        coverage_map.configure_caller_tiers(
            {"src/a.rs": source_model(A_SRC)}
        )
        try:
            target = entry_target(
                entry="fgf::kernel::a::tile_kernel", tier="v3",
                file="src/a.rs", start=0, end=0,
            )
            row = {
                "key": "src/a.rs::TILE",
                "class": "tuning_parameter",
                "required": True,
                "value": 8,
                "unit": "rows",
                "axis": "row group width",
                "routes": "",
                "comparison": None,
                "status": "unmapped",
                "reason": "no mapping",
                "sites": [{"file": "src/a.rs", "line": 2, "column": 14}],
            }
            cells = [
                {"id": "small", "operation": "mul_into", "row_len": 64,
                 "rows": 2, "sources": 1},
                {"id": "large", "operation": "mul_into", "row_len": 64,
                 "rows": 16, "sources": 1},
            ]
            rows = [entry_row(target, ["small", "large"])]
            exceptions = {
                row["key"]: {
                    "key": row["key"],
                    "class": "unmapped_arcane",
                    "reason": "arcane body",
                    "straddling_cells": ["small", "large"],
                }
            }
            coverage_map.validate_exceptions(
                exceptions, [target], rows, [row], cells, "v3"
            )
            coverage_map.apply_exceptions(exceptions, [row], "v3")
            self.assertEqual(row["status"], "excused")

            row["status"] = "unmapped"
            bad = {
                row["key"]: {
                    "key": row["key"],
                    "class": "unmapped_arcane",
                    "reason": "arcane body",
                    "straddling_cells": ["small"],
                }
            }
            with self.assertRaises(coverage_map.CoverageMapError):
                coverage_map.validate_exceptions(
                    bad, [target], rows, [row], cells, "v3"
                )
        finally:
            coverage_map.configure_sources()

    def test_unmapped_branch_exception_requires_unexecuted_and_straddling(self) -> None:
        coverage_map.configure_caller_tiers(
            {"src/a.rs": source_model(A_XOR_SRC)}
        )
        try:
            target = entry_target(
                entry="fgf::kernel::a::xor_kernel", tier="v3",
                file="src/a.rs", start=0, end=0,
            )
            row = {
                "key": "src/a.rs::CUTOFF",
                "class": "routing_threshold",
                "required": True,
                "value": 64,
                "unit": "bytes",
                "axis": "destination length",
                "routes": "",
                "comparison": "<",
                "status": "unexecuted",
                "reason": "no branch record",
                "sites": [{"file": "src/a.rs", "line": 2, "column": 14}],
            }
            cells = [
                {"id": "small", "operation": "add_assign", "row_len": 32,
                 "rows": 1, "sources": 1},
                {"id": "large", "operation": "add_assign", "row_len": 128,
                 "rows": 1, "sources": 1},
            ]
            rows = [entry_row(target, ["small", "large"])]
            exceptions = {
                row["key"]: {
                    "key": row["key"],
                    "class": "unmapped_branch",
                    "reason": "the compiler emitted no branch record",
                    "straddling_cells": ["small", "large"],
                }
            }
            self.assertEqual(
                coverage_map.validate_exceptions(
                    exceptions, [target], rows, [row], cells, "v3"
                ),
                [],
            )
            coverage_map.apply_exceptions(exceptions, [row], "v3")
            self.assertEqual(row["status"], "excused")

            row["status"] = "both_ways"
            diagnostics = coverage_map.validate_exceptions(
                exceptions, [target], rows, [row], cells, "v3"
            )
            self.assertEqual(len(diagnostics), 1)
            self.assertIn("stale", diagnostics[0]["reason"])
        finally:
            coverage_map.configure_sources()

    def test_stale_exception_keys_fail(self) -> None:
        cell = {"id": "c", "operation": "add_assign", "row_len": 64, "rows": 1,
                "sources": 1, "bytes": 64}
        exceptions = {
            "unknown": {
                "key": "unknown",
                "class": "unmapped_arcane",
                "reason": "stale",
                "straddling_cells": ["c"],
            }
        }
        with self.assertRaises(coverage_map.CoverageMapError):
            coverage_map.validate_exceptions(exceptions, [], [], [], [cell], "v3")


class PerSiteEvidenceTests(unittest.TestCase):
    def test_call_counts_stay_aligned_per_call_site(self) -> None:
        target = entry_target(
            entry="fgf::kernel::x86::demo::f", tier="v3", file="src/kernel.rs",
            start=10, end=10,
            call_sites=(("src/callers.rs", 5, 5), ("src/callers.rs", 9, 9)),
        )
        document = export([])
        first = coverage_map.analyze_export(document, ROOT, [target], [])
        self.assertEqual(first["entries"][target.entry]["call_counts"], [None, None])

    def test_summarize_entries_keeps_per_site_cells(self) -> None:
        target = entry_target(
            entry="fgf::kernel::x86::demo::f", tier="v3", file="src/kernel.rs",
            start=10, end=10,
            call_sites=(("src/callers.rs", 5, 5), ("src/callers.rs", 9, 9)),
        )
        evidence = {
            "format": coverage_map.ANALYSIS_FORMAT,
            "entries": {
                target.entry: {
                    "wrapper_mapped": False,
                    "wrapper_count": 0,
                    "call_counts": [7, None],
                    "macro_unattributed": [],
                }
            },
            "sites": {},
        }
        other = {
            "format": coverage_map.ANALYSIS_FORMAT,
            "entries": {
                target.entry: {
                    "wrapper_mapped": False,
                    "wrapper_count": 0,
                    "call_counts": [None, 5],
                    "macro_unattributed": [],
                }
            },
            "sites": {},
        }
        row = coverage_map.summarize_entries(
            [target], [("cell-a", evidence), ("cell-b", other)], "v3", {}
        )[0]
        self.assertEqual(row["executed_by"], ["cell-a", "cell-b"])
        self.assertEqual(row["executed_by_call_site"], ["cell-a", "cell-b"])
        self.assertEqual(row["call_site_cells"], [["cell-a"], ["cell-b"]])
        self.assertEqual(row["count"], 12)


GF8_GFNI = "src/kernel/x86/gf8/gfni.rs"
GF8_DISPATCH = "src/kernel/gf8.rs"

GF8_GFNI_SRC = (
    "/// Overwrite rows with a scaled source.\n"
    "pub fn mul_into_gfni(rows: &mut [u8]) {\n"
    "    into_body(rows);\n"
    "}\n"
    "/// Fold many sources into many rows.\n"
    "pub fn mul_add_matrix_gfni(rows: &mut [u8], cols: usize) {\n"
    "    matrix_body(rows, cols);\n"
    "}\n"
    "fn into_body(rows: &mut [u8]) {\n"
    "    let peel = rows.len() < PEEL_FLOOR;\n"
    "}\n"
    "fn matrix_body(rows: &mut [u8], cols: usize) {\n"
    "    let blocked = cols > MATRIX_FLOOR;\n"
    "}\n"
    "const PEEL_FLOOR: usize = 512;\n"
    "const MATRIX_FLOOR: usize = 4096;\n"
)

GF16_GFNI = "src/kernel/x86/gf16/gfni.rs"

GF16_GFNI_SRC = (
    "/// Overwrite rows with a scaled source.\n"
    "pub fn mul_into_gfni16(rows: &mut [u8]) {\n"
    "    into_body16(rows);\n"
    "}\n"
    "fn into_body16(rows: &mut [u8]) {\n"
    "    let peel = rows.len() < PEEL_FLOOR;\n"
    "}\n"
    "const PEEL_FLOOR: usize = 512;\n"
)

GF8_DISPATCH_SRC = (
    "use crate::kernel::x86::gf8::gfni::mul_into_gfni;\n"
    "pub fn dispatch_a(rows: &mut [u8]) {\n"
    "    mul_into_gfni(rows);\n"
    "}\n"
    "pub fn dispatch_b(rows: &mut [u8]) {\n"
    "    mul_into_gfni(rows);\n"
    "}\n"
)

GF8_ALIAS_SRC = (
    "use crate::kernel::x86::gf8::gfni::into_body as renamed;\n"
    "pub fn mul_alias_gfni(rows: &mut [u8]) {\n"
    "    renamed(rows);\n"
    "}\n"
)

GF8_GLOB_SRC = (
    "use crate::kernel::x86::gf8::gfni::*;\n"
    "pub fn mul_glob_gfni(rows: &mut [u8]) {\n"
    "    into_body(rows);\n"
    "}\n"
)

GF8_AMBIG_SRC = (
    "fn work_marker() -> usize {\n"
    "    let first = 1;\n"
    "    first\n"
    "}\n"
    "fn work_marker() -> usize {\n"
    "    let second = 2;\n"
    "    second\n"
    "}\n"
    "pub fn mul_ambig_gfni() -> usize {\n"
    "    work_marker()\n"
    "}\n"
)


def peel_row(sites=None):
    return {
        "key": f"{GF8_GFNI}::PEEL_FLOOR",
        "class": "tuning_parameter",
        "required": True,
        "value": 512,
        "unit": "bytes",
        "axis": "destination length",
        "routes": "",
        "comparison": "<",
        "status": "unmapped",
        "reason": "no mapping",
        "sites": sites
        or [{"file": GF8_GFNI, "line": 10, "column": 16}],
    }


def gf8_cells():
    return [
        {"id": "gf8-lo", "operation": "mul_into", "row_len": 384,
         "rows": 1, "sources": 1},
        {"id": "gf8-hi", "operation": "mul_into", "row_len": 768,
         "rows": 1, "sources": 1},
    ]


def gf8_world(files=None):
    world = {
        GF8_GFNI: GF8_GFNI_SRC,
        GF8_DISPATCH: GF8_DISPATCH_SRC,
    }
    world.update(files or {})
    coverage_map.configure_caller_tiers(
        {name: source_model(text) for name, text in world.items()}
    )


def gf8_targets(extra=()):
    into = entry_target(
        entry="fgf::kernel::x86::gf8::gfni::mul_into_gfni",
        tier="v3_gfni_crypto", file=GF8_GFNI, start=1, end=1,
    )
    matrix = entry_target(
        entry="fgf::kernel::x86::gf8::gfni::mul_add_matrix_gfni",
        tier="v3_gfni_crypto", file=GF8_GFNI, start=5, end=5,
    )
    return [into, matrix, *extra]


def gf8_rows(into_cells=("gf8-lo", "gf8-hi"), matrix_cells=("gf8-mx",)):
    into = gf8_targets()[0]
    matrix = gf8_targets()[1]
    return [entry_row(into, list(into_cells)), entry_row(matrix, list(matrix_cells))]


class ReachabilityTests(unittest.TestCase):
    def tearDown(self) -> None:
        coverage_map.configure_sources()

    def validate(self, cells, row=None, targets=None, rows=None):
        row = row or peel_row()
        targets = gf8_targets() if targets is None else targets
        rows = gf8_rows() if rows is None else rows
        exceptions = {
            row["key"]: {
                "key": row["key"],
                "class": "unmapped_arcane",
                "reason": "arcane body",
                "straddling_cells": [cell["id"] for cell in cells],
            }
        }
        return coverage_map.validate_exceptions(
            exceptions, targets, rows, [row], cells, "v3_gfni_crypto"
        )

    def test_a_direct_helper_route_with_the_required_geometry_is_accepted(self) -> None:
        gf8_world()
        self.validate(gf8_cells())
        reach = coverage_map.reachability_index(gf8_targets(), gf8_rows())
        served = coverage_map._serving_cells(peel_row()["key"], peel_row(), reach)
        self.assertEqual(served, {"gf8-lo", "gf8-hi"})

    def test_scoped_opaque_call_proofs_do_not_credit_the_callee(self) -> None:
        file = "src/kernel/x86/demo.rs"
        source = (
            "const CUTOFF: usize = 512;\n"
            "pub fn parent(length: usize) { child(length); }\n"
            "\n"
            "pub fn child(length: usize) {\n"
            "    if length >= CUTOFF {}\n"
            "}\n"
            "\n"
            "pub fn unrelated(_: usize) {}\n"
        )
        coverage_map.configure_caller_tiers({file: source_model(source)})
        parent = entry_target(
            entry="fgf::kernel::x86::demo::parent", tier="v4x",
            file=file, start=2, end=2,
        )
        child = entry_target(
            entry="fgf::kernel::x86::demo::child", tier="v3",
            file=file, start=4, end=6,
        )
        unrelated = entry_target(
            entry="fgf::kernel::x86::demo::unrelated", tier="v4x",
            file=file, start=8, end=8,
        )
        targets = [parent, child, unrelated]
        cells = [
            {"id": name, "operation": "mul_add_scatter", "row_len": length,
             "rows": 3, "sources": 1}
            for name, length in (
                ("lo", 384), ("hi", 768), ("other-lo", 384), ("other-hi", 768)
            )
        ]
        row = peel_row(sites=[{"file": file, "line": 5, "column": 8}])
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        proof = {"entry": parent.entry, "sources": {file: digest}}
        declaration = {
            "key": row["key"], "class": "unmapped_arcane",
            "reason": "the parent passes its full row to the opaque child",
            "tiers": ["v3", "v4x"],
            "straddling_cells": {
                "v3": ["other-lo", "other-hi"], "v4x": ["lo", "hi"],
            },
            "via_entry": {"v4x": proof},
        }

        def validate(declared, rows, tier="v4x"):
            return coverage_map.validate_exceptions(
                {row["key"]: declared}, targets, rows, [row], cells, tier,
            )

        rows = [
            entry_row(parent, ["lo", "hi"]), entry_row(child, []),
            entry_row(unrelated, ["lo", "hi"]),
        ]
        self.assertEqual(validate(declaration, rows), [])
        self.assertEqual(validate(declaration, [
            entry_row(parent, []), entry_row(child, ["other-lo", "other-hi"]),
            entry_row(unrelated, []),
        ], "v3"), [])
        rejected = [
            dict(declaration, via_entry={}),
            dict(declaration, via_entry={"v4x": dict(proof, entry=unrelated.entry)}),
            dict(declaration, via_entry={"v4x": dict(proof, entry="fgf::kernel::missing")}),
            dict(declaration, via_entry={"v4x": dict(proof, sources={file: "0" * 64})}),
            dict(declaration, via_entry={"v4x": dict(
                proof, sources={"src/kernel/x86/other.rs": digest}
            )}),
            dict(declaration, straddling_cells={
                "v3": ["other-lo", "other-hi"], "v4x": ["other-lo", "other-hi"],
            }),
        ]
        for declared in rejected:
            with self.subTest(declaration=declared):
                with self.assertRaises(coverage_map.CoverageMapError):
                    validate(declared, rows)
        for observed in ([], ["lo"]):
            with self.subTest(observed=observed):
                with self.assertRaises(coverage_map.CoverageMapError):
                    validate(declaration, [
                        entry_row(parent, observed), entry_row(child, []),
                        entry_row(unrelated, ["lo", "hi"]),
                    ])

    def test_cells_of_another_field_are_rejected(self) -> None:
        gf8_world({GF16_GFNI: GF16_GFNI_SRC})
        gf16_entry = entry_target(
            entry="fgf::kernel::x86::gf16::gfni::mul_into_gfni16",
            tier="v3_gfni_crypto", file=GF16_GFNI, start=1, end=1,
        )
        targets = gf8_targets(extra=[gf16_entry])
        rows = gf8_rows() + [entry_row(gf16_entry, ["gf16-a"])]
        cells = [
            {"id": "gf16-a", "operation": "mul_into", "row_len": 384,
             "rows": 1, "sources": 1},
            {"id": "gf16-b", "operation": "mul_into", "row_len": 768,
             "rows": 1, "sources": 1},
        ]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(cells, targets=targets, rows=rows)

    def test_cells_of_another_operation_in_the_family_are_rejected(self) -> None:
        gf8_world()
        cells = [
            {"id": "gf8-mx", "operation": "mul_add_matrix", "row_len": 384,
             "rows": 1, "sources": 1},
            {"id": "gf8-my", "operation": "mul_add_matrix", "row_len": 768,
             "rows": 1, "sources": 1},
        ]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(cells)

    def test_per_call_site_cells_credit_only_the_running_caller(self) -> None:
        gf8_world()
        target = gf8_targets()[0]
        sited = entry_target(
            entry=target.entry, tier=target.tier, file=target.file,
            start=target.start_line, end=target.end_line,
            call_sites=((GF8_DISPATCH, 3, 5), (GF8_DISPATCH, 6, 5)),
        )
        rows = [entry_row(sited, ["gf8-lo", "gf8-hi"])]
        rows[0]["call_site_cells"] = [["gf8-lo"], ["gf8-hi"]]
        reach = coverage_map.reachability_index([sited], rows)
        row = peel_row(sites=[{"file": GF8_DISPATCH, "line": 6, "column": 5}])
        served = coverage_map._serving_cells(row["key"], row, reach)
        self.assertEqual(served, {"gf8-hi"})
        other = peel_row(sites=[{"file": GF8_DISPATCH, "line": 3, "column": 5}])
        self.assertEqual(
            coverage_map._serving_cells(other["key"], other, reach), {"gf8-lo"}
        )

    def test_macro_call_sites_credit_their_invocation_body(self) -> None:
        gf8_world()
        target = gf8_targets()[0]
        site = coverage_map.MacroCallSite(
            file=GF8_GFNI, line=2, column=14,
            invocation_file=GF8_DISPATCH, invocation_line=5, invocation_column=5,
            invocation_module="fgf::kernel::gf8",
        )
        sited = entry_target(
            entry=target.entry, tier=target.tier, file=target.file,
            start=target.start_line, end=target.end_line, call_sites=(site,),
        )
        rows = [entry_row(sited, ["gf8-lo", "gf8-hi"])]
        rows[0]["call_site_cells"] = [["gf8-lo", "gf8-hi"]]
        reach = coverage_map.reachability_index([sited], rows)
        row = peel_row(sites=[{"file": GF8_DISPATCH, "line": 5, "column": 5}])
        self.assertIn("gf8-lo", coverage_map._serving_cells(row["key"], row, reach))

    def test_ambiguous_helper_names_receive_no_credit(self) -> None:
        ambiguous = "src/kernel/x86/gf8/ambig.rs"
        gf8_world({ambiguous: GF8_AMBIG_SRC})
        entry = entry_target(
            entry="fgf::kernel::x86::gf8::ambig::mul_ambig_gfni",
            tier="v3_gfni_crypto", file=ambiguous, start=9, end=9,
        )
        targets = gf8_targets(extra=[entry])
        rows = gf8_rows() + [entry_row(entry, ["gf8-am"])]
        module = "fgf::kernel::x86::gf8::ambig"
        model = coverage_map._SOURCE_MODELS[ambiguous]
        reach = coverage_map.reachability_index(targets, rows)
        self.assertIsNone(
            coverage_map._resolve_call(reach, ambiguous, module, ["work_marker"])
        )
        row = peel_row(sites=[{"file": ambiguous, "line": 6, "column": 9}])
        served = coverage_map._serving_cells(row["key"], row, reach)
        cells = [
            {"id": "gf8-am", "operation": "mul_into", "row_len": 384,
             "rows": 1, "sources": 1},
            {"id": "gf8-an", "operation": "mul_into", "row_len": 768,
             "rows": 1, "sources": 1},
        ]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(cells, row=row, targets=targets, rows=rows)

    def test_aliased_imports_receive_no_credit(self) -> None:
        alias = "src/kernel/x86/gf8/alias.rs"
        gf8_world({alias: GF8_ALIAS_SRC})
        entry = entry_target(
            entry="fgf::kernel::x86::gf8::alias::mul_alias_gfni",
            tier="v3_gfni_crypto", file=alias, start=1, end=1,
        )
        targets = gf8_targets(extra=[entry])
        rows = gf8_rows() + [entry_row(entry, ["gf8-al"])]
        reach = coverage_map.reachability_index(targets, rows)
        served = coverage_map._serving_cells(peel_row()["key"], peel_row(), reach)
        self.assertNotIn("gf8-al", served)
        cells = [
            {"id": "gf8-al", "operation": "mul_into", "row_len": 384,
             "rows": 1, "sources": 1},
            {"id": "gf8-alo", "operation": "mul_into", "row_len": 768,
             "rows": 1, "sources": 1},
        ]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(cells, targets=targets, rows=rows)

    def test_glob_imports_receive_no_credit(self) -> None:
        globs = "src/kernel/x86/gf8/globs.rs"
        gf8_world({globs: GF8_GLOB_SRC})
        entry = entry_target(
            entry="fgf::kernel::x86::gf8::globs::mul_glob_gfni",
            tier="v3_gfni_crypto", file=globs, start=1, end=1,
        )
        targets = gf8_targets(extra=[entry])
        rows = gf8_rows() + [entry_row(entry, ["gf8-gl"])]
        reach = coverage_map.reachability_index(targets, rows)
        served = coverage_map._serving_cells(peel_row()["key"], peel_row(), reach)
        self.assertNotIn("gf8-gl", served)

    def test_unproven_serving_is_reported_not_excused(self) -> None:
        gf8_world()
        row = peel_row(sites=[{"file": "src/kernel/nowhere.rs", "line": 4, "column": 1}])
        reach = coverage_map.reachability_index(gf8_targets(), gf8_rows())
        self.assertEqual(
            coverage_map._serving_cells(row["key"], row, reach), set()
        )


class MacroAttributionTests(unittest.TestCase):
    def targets(self, dynamic_name=False, private_module=False, modules=False):
        call = "x86::fan_paar::$entry()" if dynamic_name else "x86::$family::mul_add_gfni()"
        source = (
            "macro_rules! dispatch {\n"
            "    ($family:ident, $entry:ident $(,)?) => {\n"
            f"        fn apply() {{ {call}; }}\n"
            "    };\n"
            "}\n"
        )
        invocations = (
            "dispatch!(gf32, mul_add_gfni32,);\n"
            "dispatch!(gf64, mul_add_gfni64);\n"
        )
        if modules:
            invocations = (
                "mod gf32 { dispatch!(gf32, mul_add_gfni32); }\n"
                "mod gf64 { dispatch!(gf64, mul_add_gfni64); }\n"
            )

        model = source_model

        names = (
            ["fgf::kernel::x86::fan_paar::mul_add_gfni32",
             "fgf::kernel::x86::fan_paar::mul_add_gfni64"] if dynamic_name else
            [f"fgf::kernel::x86::{family}::mul_add_gfni" for family in ("gf32", "gf64")]
        )
        if private_module:
            names = [name.replace("::mul_add", "::gfni::mul_add") for name in names]
        calls = coverage_map.discover_call_sites(
            [{"entry": name} for name in names],
            {"src/definition.rs": model(source), "src/invocations.rs": model(invocations)},
        )
        return [
            entry_target(entry=name, file=f"src/{index}.rs", call_sites=calls.get(name, ()))
            for index, name in enumerate(names)
        ]

    def document(self, targets, include_expansions=True):
        records = branch_file("/repo/src/invocations.rs", [])
        if include_expansions:
            for target, count in zip(targets, (7, 0)):
                site, = target.call_sites
                records["expansions"].append({
                    "filenames": ["/repo/src/invocations.rs", "/repo/src/definition.rs"],
                    "source_region": [
                        site.invocation_line, site.invocation_column,
                        site.invocation_line, 40, count, 0, 1, 1,
                    ],
                    "target_regions": [
                        [site.line, site.column, site.line, site.column + 30, count, 1, 0, 0],
                        # A positive region in the invocation file is not definition evidence.
                        [site.line, site.column, site.line, site.column + 30, 99, 0, 0, 0],
                    ],
                })
        # Pooled macro-body counts must never credit either concrete expansion.
        return export([
            function("_pooled", "/repo/src/definition.rs", [[3, 1, 3, 100, 99, 0, 0, 0]])
        ], [records])

    def test_only_the_executed_expansion_credits_its_concrete_entry(self):
        targets = self.targets()
        evidence = coverage_map.analyze_export(self.document(targets), ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("gf32-cell", evidence)], "v3", {})
        self.assertEqual([row["status"] for row in rows], ["executed", "unexecuted"])
        self.assertEqual([row["count"] for row in rows], [7, 0])
        self.assertEqual(rows[1]["executed_by"], [])

    def test_metavariable_function_names_resolve_without_cross_credit(self):
        targets = self.targets(dynamic_name=True)
        evidence = coverage_map.analyze_export(self.document(targets), ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [7, 0])

    def test_family_facades_can_omit_private_isa_modules(self):
        targets = self.targets(private_module=True)
        evidence = coverage_map.analyze_export(self.document(targets), ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [7, 0])

    def test_absent_expansion_records_report_a_gap_not_pooled_execution(self):
        targets = self.targets()
        evidence = coverage_map.analyze_export(
            self.document(targets, include_expansions=False), ROOT, targets, []
        )
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [0, 0])
        gaps = coverage_map.failures(rows, [])
        self.assertEqual(len(gaps), 2)
        self.assertTrue(all("macro-unattributed" in gap["reason"] for gap in gaps))

    def test_another_invocations_records_cannot_credit_an_unmapped_site(self):
        targets = self.targets()
        document = self.document(targets)
        document["data"][0]["files"][0]["expansions"] = (
            document["data"][0]["files"][0]["expansions"][:1]
        )
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual(rows[1]["status"], "unexecuted")
        self.assertIn("macro-unattributed", rows[1]["reason"])

    def function_document(self, targets, parents=("gf32", "gf64"), counts=(7, 0)):
        functions = []
        for target, parent, count in zip(targets, parents, counts):
            site, = target.call_sites
            # A generic trait-impl method; its impl-parent is absent from the
            # usual demangled spelling, but present in the v0 symbol.
            symbol = (
                "_RINvXs3_NtNtC3fgf11invocations"
                f"{len(parent)}{parent}C4SelfC5Trait5applymE"
            )
            functions.append(function(
                symbol, "/repo/src/definition.rs",
                [
                    [site.line, 1, site.line, 100, 99, 0, 0, 0],
                    [site.line, site.column, site.line, site.column + 30, count, 0, 0, 0],
                ],
                count=99,
            ))
        return export(functions)

    def test_function_records_credit_only_the_executed_invocation(self):
        targets = self.targets(modules=True)
        document = self.function_document(targets)
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("gf32-cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [7, 0])
        self.assertEqual(rows[0]["executed_by"], ["gf32-cell"])
        self.assertEqual(rows[1]["executed_by"], [])
        self.assertIsNone(rows[1]["reason"])

    def test_unmatched_function_record_remains_macro_unattributed(self):
        targets = self.targets(modules=True)
        document = self.function_document(targets, parents=("unrelated", "gf64"))
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual(rows[0]["count"], 0)
        self.assertIn("macro-unattributed", rows[0]["reason"])

    def test_same_module_invocations_cannot_borrow_function_counts(self):
        targets = self.targets()
        document = self.function_document(targets)
        # Both expansions have the same lexical parent, so even a matching
        # live function cannot identify either invocation.
        document["data"][0]["functions"][0]["name"] = (
            "_RNvXNtC3fgf11invocationsC4SelfC5Trait5apply"
        )
        evidence = coverage_map.analyze_export(document, ROOT, targets[:1], [])
        rows = coverage_map.summarize_entries(targets[:1], [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [0])
        self.assertTrue(all("macro-unattributed" in row["reason"] for row in rows))


    def test_macro_peers_independently_reject_same_module_function_counts(self):
        targets = self.targets(modules=True)
        first, second = (target.call_sites[0] for target in targets)
        second = coverage_map.dataclasses.replace(
            second, invocation_module=first.invocation_module,
        )
        targets[1] = coverage_map.dataclasses.replace(
            targets[1], call_sites=(second,),
        )
        document = self.function_document(targets, parents=("gf32", "gf32"))
        single = coverage_map.analyze_export(document, ROOT, targets[:1], [])
        self.assertEqual(single["entries"][targets[0].entry]["call_counts"], [7])
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [0, 0])
        self.assertTrue(all("macro-unattributed" in row["reason"] for row in rows))

    def test_repeated_capture_names_cannot_credit_the_last_binding(self):
        source = (
            "macro_rules! dispatch {\n"
            "    ($family:ident, $family:ident) => {\n"
            "        fn apply() { x86::$family::mul_add_gfni(); }\n"
            "    };\n"
            "}\n"
        )
        names = [f"fgf::kernel::x86::{family}::mul_add_gfni" for family in ("gf32", "gf64")]
        calls = coverage_map.discover_call_sites(
            [{"entry": name} for name in names],
            {
                "src/definition.rs": source_model(source),
                "src/invocations.rs": source_model("dispatch!(gf32, gf64);"),
            },
        )
        targets = [entry_target(entry=name, call_sites=calls.get(name, ())) for name in names]
        records = branch_file("/repo/src/invocations.rs", [])
        line, column = coverage_map.scan_line_column(source, source.index("x86::"))
        records["expansions"].append({
            "filenames": ["/repo/src/invocations.rs", "/repo/src/definition.rs"],
            "source_region": [1, 1, 1, 30, 7, 0, 1, 1],
            "target_regions": [[line, column, line, column + 40, 7, 1, 0, 0]],
        })
        document = export([function(
            "_pooled", "/repo/src/definition.rs", [[3, 1, 3, 100, 99, 0, 0, 0]]
        )], [records])
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [0, 0])
        self.assertEqual(len(coverage_map.failures(rows, [])), 2)

    def test_unique_macro_names_require_path_correspondence(self):
        name = "fgf::kernel::x86::gf32::unique"
        for call, invocation_file, accepted in (
            ("unrelated::unique()", "src/invocations.rs", False),
            ("gf32::unique()", "src/invocations.rs", True),
            ("unique()", "src/invocations.rs", False),
            ("unique()", "src/kernel/x86/gf32.rs", True),
        ):
            with self.subTest(call=call, invocation_file=invocation_file):
                source = (
                    "macro_rules! dispatch { ($unused:ident) => {\n"
                    f"fn apply() {{ {call}; }}\n"
                    "}; }\n"
                )
                calls = coverage_map.discover_call_sites(
                    [{"entry": name}],
                    {
                        "src/definition.rs": source_model(source),
                        invocation_file: source_model("dispatch!(unused);"),
                    },
                )
                target = entry_target(entry=name, call_sites=calls.get(name, ()))
                records = branch_file(str(ROOT / invocation_file), [])
                line, column = coverage_map.scan_line_column(source, source.index(call))
                records["expansions"].append({
                    "filenames": [str(ROOT / invocation_file), "/repo/src/definition.rs"],
                    "source_region": [1, 1, 1, 30, 7, 0, 1, 1],
                    "target_regions": [[line, column, line, column + len(call), 7, 1, 0, 0]],
                })
                evidence = coverage_map.analyze_export(export([], [records]), ROOT, [target], [])
                row, = coverage_map.summarize_entries([target], [("cell", evidence)], "v3", {})
                self.assertEqual(row["count"], 7 if accepted else 0)

    def test_non_code_function_regions_do_not_prove_a_macro_call(self):
        targets = self.targets(modules=True)
        document = self.function_document(targets)
        for record in document["data"][0]["functions"]:
            for region in record["regions"]:
                region[7] = 3
        evidence = coverage_map.analyze_export(document, ROOT, targets, [])
        rows = coverage_map.summarize_entries(targets, [("cell", evidence)], "v3", {})
        self.assertEqual([row["count"] for row in rows], [0, 0])
        self.assertTrue(all("macro-unattributed" in row["reason"] for row in rows))

    def test_unsupported_symbol_paths_fail_closed(self):
        for symbol in ("_ZN3fgf4demoE", "_RIB0_E", "_RNvX", "_RNvXCbroken"):
            with self.subTest(symbol=symbol):
                self.assertIsNone(coverage_map.impl_parent(symbol))


class TierHintTests(unittest.TestCase):
    def hint(self, arm):
        source = (
            f"fn dispatch() {{\n    match strategy {{\n{arm}\n"
            "            if len >= CUTOFF {}\n        }\n    }\n}\n"
        )
        return self.source_hints(source, (3 + len(arm.splitlines()),))[0]

    def test_strategy_variants_and_legacy_proofs(self):
        for variant, tier in [
            ("QuadMersenneStrategy::Avx512", "v4x"),
            ("Gf8Strategy::Gfni", "v3_gfni_crypto"),
            ("Tower16Strategy::Avx2", "v3"),
            ("PrimeStrategy::Sse42", "v2"),
            ("Gf8Strategy::Ssse3", "v2"),
            ("ResidueStrategy::Sse2", "v2"),
            ("WideTowerStrategy::Scalar", "scalar"),
            ("Backend::V4x", "v4x"),
            ("Backend::V3GfniCrypto", "v3_gfni_crypto"),
            ("Backend::V3", "v3"),
            ("Backend::V2", "v2"),
            ("X86Proof::V4x(_)", "v4x"),
        ]:
            with self.subTest(variant=variant):
                self.assertEqual(self.hint(f"        {variant} => {{"), tier)

    def test_unexecuted_strategy_guard_is_required_only_at_its_tier(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy {\n"
            "        QuadMersenneStrategy::Avx512 => {\n"
            "            if len >= CUTOFF {}\n"
            "        }\n    }\n}\n"
        )
        site = coverage_map.SiteTarget(
            site_id=0, item_key="src/routing.rs::CUTOFF",
            file="dispatch.rs", line=4, column=23, role="branch",
            condition_line=4, condition_column=16,
            condition_end_line=4, condition_end_column=29,
        )
        item = {
            "key": site.item_key, "class": "routing_threshold", "value": 32,
            "unit": "bytes", "axis": "row length", "routes": "scalar <-> vector",
            "site_ids": [0], "mapped": True,
        }
        document = export([function(
            "_dispatch", "/repo/dispatch.rs", [[4, 1, 4, 40, 0, 0, 0, 0]]
        )])
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dispatch.rs").write_text(source, encoding="utf-8")
            coverage_map.configure_sources()
            coverage_map.configure_caller_tiers({"dispatch.rs": source_model(source)})
            try:
                for tier in ("v3", "v3_gfni_crypto"):
                    row, = coverage_map.summarize_thresholds(
                        [item], [site], [("cell", evidence)], {}, tier
                    )
                    self.assertEqual(row["status"], "tier_not_required")
                    self.assertEqual(row["sites"][0]["tier_hint"], "v4x")
                    self.assertEqual(coverage_map.failures([], [row]), [])
                row, = coverage_map.summarize_thresholds(
                    [item], [site], [("cell", evidence)], {}, "v4x"
                )
                self.assertEqual(row["status"], "unexecuted")
                self.assertEqual(len(coverage_map.failures([], [row])), 1)
            finally:
                coverage_map.configure_sources()

    def test_multitier_arms_keep_the_weakest_tier(self):
        for arm in [
            "        Gf8Strategy::Avx512 | Gf8Strategy::Gfni => {",
            "        Gf8Strategy::Avx512\n        | Gf8Strategy::Gfni => {",
            "        Gf8Strategy::Avx512 |\n          Gf8Strategy::Gfni => {",
            "        Gf8Strategy::Gfni |\n          Gf8Strategy::Avx512 => {",
            "        Gf8Strategy::Gfni\n        | Gf8Strategy::Avx512 => {",
        ]:
            with self.subTest(arm=arm):
                self.assertEqual(self.hint(arm), "v3_gfni_crypto")
        self.assertEqual(self.hint("        Backend::V4x | Backend::V3GfniCrypto => {"),
                         "v3_gfni_crypto")

    def source_hints(self, source, lines):
        coverage_map.configure_sources()
        coverage_map.configure_caller_tiers({"dispatch.rs": source_model(source)})
        try:
            return [coverage_map.site_tier_hint("dispatch.rs", line) for line in lines]
        finally:
            coverage_map.configure_sources()

    def test_non_scalar_guard_uses_the_complete_declared_variant_set(self):
        for variants, expected in (
            ("Scalar, Avx2, Avx512", "v3"),
            ("Scalar, Sse2, Avx2", "v2"),
            ("Scalar, Avx2, Unknown", None),
        ):
            source = (
                f"enum DemoStrategy {{ {variants} }}\n"
                "fn dispatch() {\n"
                "    if selected() != DemoStrategy::Scalar\n"
                "        && len >= CUTOFF\n"
                "    {\n"
                "        work();\n"
                "    }\n"
                "}\n"
            )
            with self.subTest(variants=variants):
                self.assertEqual(self.source_hints(source, (3, 4, 6)), [None, expected, expected])
        self.assertEqual(
            self.source_hints(
                "fn dispatch() {\n"
                "    if selected() != MissingStrategy::Scalar\n"
                "        && len >= CUTOFF {}\n"
                "}\n",
                (3,),
            ),
            [None],
        )

    def test_return_guard_propagates_through_a_complete_helper_chain(self):
        source = (
            "enum DemoStrategy { Scalar, Avx2, Avx512 }\n"
            "fn dispatch() {\n"
            "    let tier = selected();\n"
            "    if tier == DemoStrategy::Scalar { return; }\n"
            "    first();\n"
            "}\n"
            "fn first() { second(); }\n"
            "fn second() { let size = CUTOFF; }\n"
        )
        self.assertEqual(self.source_hints(source, (5, 7, 8)), ["v3", "v3", "v3"])
        self.assertEqual(
            self.source_hints(source + "fn ordinary() { second(); }\n", (8,)),
            [None],
        )
        self.assertEqual(
            self.source_hints(source + "fn unrelated<T>() {}\n", (5, 7, 8)),
            ["v3", "v3", "v3"],
        )
        for altered in (
            source.replace("let tier =", "let mut tier ="),
            source.replace("    first();", "    let tier = DemoStrategy::Scalar;\n    first();"),
            source.replace("return;", "work();"),
            source.replace("if tier ==", "if selected() =="),
            source.replace("if tier ==", "if other() { work(); } else if tier =="),
        ):
            with self.subTest(source=altered):
                self.assertIsNone(self.source_hints(altered, (len(altered.splitlines()),))[0])
        conditional = source.replace(
            "    first();", "    if first() {\n        work();\n    }",
        ).replace("fn first() { second(); }", "fn first() -> bool { second(); true }")
        self.assertEqual(
            self.source_hints(conditional, (5, len(conditional.splitlines()))),
            ["v3", "v3"],
        )

    def test_guard_proofs_do_not_cross_disjunctions_or_nested_functions(self):
        source = (
            "enum DemoStrategy { Scalar, Avx2, Avx512 }\n"
            "fn dispatch() {\n"
            "    if selected() != DemoStrategy::Scalar\n"
            "        && len >= CUTOFF || other\n"
            "    {\n"
            "        work();\n"
            "    }\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (4, 6)), [None, None])
        nested = source.replace(" || other", "").replace("        work();", "        fn nested() { work(); }")
        self.assertEqual(self.source_hints(nested, (6,)), [None])
        plain = source.replace(" || other", "")
        for ambiguous, lines in (
            ("use external::*;\n" + plain, (5, 7)),
            ("use external::Other as DemoStrategy;\n" + plain, (5, 7)),
            (plain.replace("fn dispatch()", "fn dispatch<DemoStrategy>()"), (4, 6)),
            (plain.replace("DemoStrategy::Scalar", "external::DemoStrategy::Scalar"), (4, 6)),
            (plain.replace("fn dispatch() {", "fn dispatch() {\n    type DemoStrategy = Other;"), (5, 7)),
        ):
            with self.subTest(source=ambiguous):
                self.assertEqual(self.source_hints(ambiguous, lines), [None, None])

    def test_closed_matches_do_not_hint_portable_tails_or_other_functions(self):
        for pattern in ("QuadMersenneStrategy::Avx2", "Backend::V3", "X86Proof::V3(_)"):
            with self.subTest(pattern=pattern):
                source = (
                    "fn dispatch() {\n"
                    "    match strategy() {\n"
                    f"        {pattern} => {{\n"
                    "            if len > CUTOFF { vector(); }\n"
                    "        }\n"
                    "        _ => {}\n"
                    "    }\n"
                    "    if len > CUTOFF { portable(); }\n"
                    "}\n"
                    "fn other() {\n"
                    "    if len > CUTOFF { portable(); }\n"
                    "}\n"
                )
                self.assertEqual(self.source_hints(source, (4, 8, 11)), ["v3", None, None])

    def test_expression_arms_stop_at_their_comma_and_match_close(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy() {\n"
            "        X86Proof::V3(token) =>\n"
            "            vector(token, CUTOFF),\n"
            "        _ => portable(CUTOFF),\n"
            "    }\n"
            "    portable(CUTOFF);\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (4, 5, 7)), ["v3", None, None])

    def test_guard_and_nested_block_are_enclosed_by_the_arm(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy() {\n"
            "        QuadMersenneStrategy::Avx512\n"
            "            if len > CUTOFF => {\n"
            "                if more { nested(CUTOFF); }\n"
            "            }\n"
            "        _ => {}\n"
            "    }\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (4, 5)), ["v4x", "v4x"])

    def test_portable_tail_prevents_private_helper_tier_inheritance(self):
        self.assertIsNone(self.caller_hint({
            "src/portable.rs": (
                "fn portable() {\n"
                "    match strategy() {\n"
                "        QuadMersenneStrategy::Avx2 => { vector(); }\n"
                "        _ => {}\n"
                "    }\n"
                "    blocked();\n"
                "}\n"
            ),
        }))

    def test_long_arm_keeps_its_hint_without_leaking_past_the_match(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy() {\n"
            "        Backend::V3 => {\n"
            + "            work();\n" * 130
            + "            vector(CUTOFF);\n"
            "        }\n"
            "        _ => {}\n"
            "    }\n"
            "    portable(CUTOFF);\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (134, 138)), ["v3", None])

    def test_shared_arm_and_portable_tail_line_does_not_prove_enclosure(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy() {\n"
            "        Backend::V3 => { vector(CUTOFF); }\n"
            "        _ => {}\n"
            "    } portable(CUTOFF);\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (3, 5)), ["v3", None])

    def test_unknown_pattern_alternatives_remain_required(self):
        source = (
            "fn dispatch() {\n"
            "    match strategy() {\n"
            "        Backend::V3 | Backend::Unknown => { use_cutoff(CUTOFF); }\n"
            "        _ => {}\n"
            "    }\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (3,)), [None])

    def test_if_let_proof_does_not_hint_its_else_or_tail(self):
        source = (
            "fn dispatch() {\n"
            "    if let super::X86Proof::V4x(token) = proof() {\n"
            "        vector(CUTOFF);\n"
            "    } else {\n"
            "        portable(CUTOFF);\n"
            "    }\n"
            "    portable(CUTOFF);\n"
            "}\n"
        )
        self.assertEqual(self.source_hints(source, (3, 5, 7)), ["v4x", None, None])

    def test_if_let_block_scrutinee_remains_required(self):
        for prefix in ("", "unsafe "):
            with self.subTest(prefix=prefix):
                source = (
                    "fn dispatch() {\n"
                    f"    if let X86Proof::V4x(token) = {prefix}{{\n"
                    "        portable(CUTOFF);\n"
                    "        proof()\n"
                    "    } {\n"
                    "        vector(CUTOFF);\n"
                    "    }\n"
                    "}\n"
                )
                self.assertEqual(self.source_hints(source, (3, 6)), [None, None])

    def caller_hint(self, extra=None):
        sources = {
            "src/shared.rs": "fn blocked() {\n    rows > CUTOFF\n}\n",
            "src/gfni.rs": "fn matrix() {\n    blocked();\n}\n",
            "src/avx512.rs": "fn matrix() {\n    blocked();\n}\n",
        }
        sources.update(extra or {})
        coverage_map.configure_sources()
        try:
            coverage_map.configure_caller_tiers({
                name: source_model(text) for name, text in sources.items()
            })
            return coverage_map.site_tier_hint("src/shared.rs", 2)
        finally:
            coverage_map.configure_sources()

    def test_private_helper_inherits_the_weakest_callers_tier(self):
        self.assertEqual(self.caller_hint(), "v3_gfni_crypto")
        self.assertEqual(
            self.caller_hint({"src/avx2.rs": "fn matrix() {\n    blocked();\n}\n"}),
            "v3",
        )

    def test_unresolved_or_indirect_helper_callers_prevent_a_tier_hint(self):
        for text in (
            "fn portable() {\n    blocked();\n}\n",
            "fn portable() {\n    let pointer = blocked;\n}\n",
            "fn blocked() {\n    other();\n}\n",
            "use shared::blocked as renamed;\nfn portable() {\n    renamed();\n}\n",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.caller_hint({"src/portable.rs": text}))

    def test_public_helpers_cannot_be_restricted_to_observed_callers(self):
        self.assertIsNone(self.caller_hint({
            "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
        }))

    def test_restricted_helpers_inherit_the_weakest_caller(self):
        for visibility in ("pub(crate)", "pub(super)", "pub(in crate::shared)"):
            with self.subTest(visibility=visibility):
                self.assertEqual(self.caller_hint({
                    "src/shared.rs": f"{visibility} fn blocked() {{\n    rows > CUTOFF\n}}\n",
                }), "v3_gfni_crypto")

    def test_public_helper_inside_a_private_module_is_enumerable(self):
        for declaration in ("mod shared;", "pub(crate) mod shared;"):
            with self.subTest(declaration=declaration):
                self.assertEqual(self.caller_hint({
                    "src/lib.rs": declaration,
                    "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                }), "v3_gfni_crypto")

    def test_public_helper_inside_a_private_inline_module_is_enumerable(self):
        self.assertEqual(self.caller_hint({
            "src/lib.rs": "pub mod shared;",
            "src/shared.rs": "mod hidden { pub fn blocked() {\n    rows > CUTOFF\n} }\n",
        }), "v3_gfni_crypto")

    def test_exporting_a_sibling_does_not_export_the_private_module(self):
        self.assertEqual(self.caller_hint({
            "src/lib.rs": "mod shared; pub use shared::{unrelated};",
            "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
        }), "v3_gfni_crypto")

    def test_internals_only_exports_do_not_make_a_helper_stable(self):
        for facade in (
            "pub use crate::shared::blocked;",
            "pub mod nested { pub use crate::shared::*; }",
        ):
            with self.subTest(facade=facade):
                self.assertEqual(self.caller_hint({
                    "src/lib.rs": "mod shared; pub mod internals;",
                    "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                    "src/internals.rs": facade,
                }), "v3_gfni_crypto")

    def test_a_private_module_alias_cannot_hide_a_stable_path(self):
        for imported in ("shared::hidden", "shared::{hidden as alias}"):
            use = f"use {imported} as alias;" if imported.endswith("hidden") else f"use {imported};"
            self.assertIsNone(self.caller_hint({
                "src/lib.rs": f"mod shared; {use} pub use alias as exposed;",
                "src/shared.rs": "pub mod hidden { pub fn blocked() {\n    rows > CUTOFF\n} }\n",
            }))

    def test_stable_public_paths_and_reexports_stay_required(self):
        for facade in (
            "pub mod shared;",
            "mod shared; pub use shared::blocked;",
            "mod shared; pub use shared::{blocked as exposed};",
            "mod shared; pub use shared::*;",
            "mod shared; pub use shared as exposed;",
            "mod shared; pub mod api { pub use crate::shared::blocked; }",
            "mod shared; pub mod api { pub use crate::shared::*; }",
            "mod shared; pub mod internals; pub use internals::*;",
        ):
            with self.subTest(facade=facade):
                self.assertIsNone(self.caller_hint({
                    "src/lib.rs": facade,
                    "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                    "src/internals.rs": "pub use crate::shared::blocked;",
                }))

    def test_reexporting_the_internals_facade_cannot_hide_a_stable_path(self):
        for facade in (
            "pub use internals as exposed;",
            "pub use internals::nested as exposed;",
            "use internals as alias; pub use alias as exposed;",
        ):
            with self.subTest(facade=facade):
                self.assertIsNone(self.caller_hint({
                    "src/lib.rs": f"mod shared; pub mod internals; {facade}",
                    "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                    "src/internals.rs": (
                        "pub use crate::shared::blocked;\n"
                        "pub mod nested { pub use crate::shared::blocked; }\n"
                    ),
                }))

    def test_any_public_module_declaration_prevents_privacy_proof(self):
        for declarations in (
            '#[cfg(feature = "expose")] pub mod shared;\n'
            '#[cfg(not(feature = "expose"))] mod shared;',
            '#[cfg(not(feature = "expose"))] mod shared;\n'
            '#[cfg(feature = "expose")] pub mod shared;',
        ):
            with self.subTest(declarations=declarations):
                self.assertIsNone(self.caller_hint({
                    "src/lib.rs": declarations,
                    "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                }))

    def test_restricted_and_internal_public_helpers_preserve_refusals(self):
        for visibility in ("pub(crate)", "pub(super)", "pub(in crate::shared)", "pub"):
            for caller in (
                "fn portable() {\n    blocked();\n}\n",
                "fn portable() {\n    let pointer = blocked;\n}\n",
                "use shared::blocked as renamed;\nfn portable() {\n    renamed();\n}\n",
                "macro_rules! dispatch { () => { blocked(); } }\n",
                "fn blocked() {\n    other();\n}\n",
            ):
                for file in ("src/portable.rs", "src/avx512.rs"):
                    if file.endswith("avx512.rs") and caller.startswith("fn portable() {\n    blocked();"):
                        continue  # An ordinary call in an ISA module has a known tier.
                    with self.subTest(visibility=visibility, caller=caller, file=file):
                        self.assertIsNone(self.caller_hint({
                            "src/lib.rs": "mod shared;",
                            "src/shared.rs": f"{visibility} fn blocked() {{\n    rows > CUTOFF\n}}\n",
                            file: caller,
                        }))

    def test_short_circuit_selector_call_inherits_its_equality_gate(self):
        self.assertEqual(self.caller_hint({
            "src/gfni.rs": "fn other() {}",
            "src/avx512.rs": "fn other() {}",
            "src/dispatch.rs": (
                "fn dispatch() {\n"
                "    if tier == ResidueStrategy::Avx512 && len >= blocked() {\n"
                "        vector();\n    }\n}\n"
            ),
        }), "v4x")

    def test_short_circuit_selector_unknown_and_unguarded_calls_stay_required(self):
        for condition in (
            "tier == ResidueStrategy::Avx512 || blocked()",
            "tier != ResidueStrategy::Avx512 && blocked()",
            "tier == ResidueStrategy::Unknown && blocked()",
            "blocked() && tier == ResidueStrategy::Avx512",
            "tier == ResidueStrategy::Avx512 && vector() {} blocked(); if portable()",
        ):
            with self.subTest(condition=condition):
                self.assertIsNone(self.caller_hint({
                    "src/dispatch.rs": f"fn dispatch() {{\n    if {condition} {{}}\n}}\n",
                }))

    def test_source_scan_includes_callers_outside_threshold_subtrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "src/lib.rs": "mod shared; mod field;",
                "src/shared.rs": "pub fn blocked() {\n    rows > CUTOFF\n}\n",
                "src/gfni.rs": "fn matrix() {\n    blocked();\n}\n",
                "src/field/scalar.rs": "fn portable() {\n    blocked();\n}\n",
                "src/tests.rs": "fn ignored() { blocked(); }",
            }
            for name, text in sources.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            scanner = SimpleNamespace(SourceModel=lambda relative, text: source_model(text))
            models = coverage_map.scan_sources(root, scanner)
            coverage_map.configure_sources()
            try:
                coverage_map.configure_caller_tiers(models)
                self.assertIsNone(coverage_map.site_tier_hint("src/shared.rs", 2))
            finally:
                coverage_map.configure_sources()


class BranchTests(unittest.TestCase):
    def document(self, true_count, false_count):
        return export(
            [],
            [
                branch_file(
                    "/repo/src/kernel.rs",
                    [[30, 5, 30, 22, true_count, false_count, 0, 0, 4]],
                )
            ],
        )

    def item(self, site):
        return {
            "key": site.item_key,
            "class": "routing_threshold",
            "value": 32,
            "unit": "bytes",
            "axis": "destination length",
            "routes": "scalar <-> vector",
            "site_ids": [site.site_id],
            "mapped": True,
        }

    def test_both_directions_are_detected(self) -> None:
        site = branch_site()
        evidence = coverage_map.analyze_export(self.document(2, 3), ROOT, [], [site])
        row = coverage_map.summarize_thresholds(
            [self.item(site)], [site], [("cell-a", evidence)], {}, "v3"
        )[0]
        self.assertEqual(row["status"], "both_ways")
        self.assertEqual(row["sites"][0]["true"], 2)
        self.assertEqual(row["sites"][0]["false"], 3)
        self.assertEqual(row["sites"][0]["true_cells"], ["cell-a"])
        self.assertEqual(row["sites"][0]["false_cells"], ["cell-a"])

    def test_one_sided_branch_is_a_failure(self) -> None:
        site = branch_site()
        evidence = coverage_map.analyze_export(self.document(2, 0), ROOT, [], [site])
        row = coverage_map.summarize_thresholds(
            [self.item(site)], [site], [("cell-a", evidence)], {}, "v3"
        )[0]
        self.assertEqual(row["status"], "one_sided")
        failure = coverage_map.failures([], [row])[0]
        self.assertEqual(failure["kind"], "threshold")

    def test_unrelated_branch_on_the_same_line_is_not_attributed(self) -> None:
        site = branch_site()
        document = export(
            [],
            [
                branch_file(
                    "/repo/src/kernel.rs",
                    [[30, 30, 30, 40, 7, 9, 0, 0, 4]],
                )
            ],
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        item = {
            "key": site.item_key,
            "class": "routing_threshold",
            "value": 32,
            "unit": "bytes",
            "axis": "destination length",
            "routes": "scalar <-> vector",
            "site_ids": [site.site_id],
            "mapped": True,
        }
        row = coverage_map.summarize_thresholds(
            [item], [site], [("cell", evidence)], {}, "v3"
        )[0]
        self.assertEqual(row["status"], "unmapped")

    def test_overlapping_branch_on_the_condition_line_is_attributed(self) -> None:
        site = branch_site()
        document = export(
            [],
            [
                branch_file(
                    "/repo/src/kernel.rs",
                    [[30, 8, 30, 25, 2, 3, 0, 0, 4]],
                )
            ],
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        self.assertTrue(evidence["sites"]["0"]["mapped"])
        self.assertEqual((evidence["sites"]["0"]["true"], evidence["sites"]["0"]["false"]), (2, 3))

    def test_branch_in_a_mapped_function_without_a_record_is_unexecuted(self) -> None:
        site = branch_site()
        document = export(
            [function("_f", "/repo/src/kernel.rs", [[30, 1, 32, 2, 0, 0, 0, 0]], count=0)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        item = {
            "key": site.item_key,
            "class": "routing_threshold",
            "value": 32,
            "unit": "bytes",
            "axis": "destination length",
            "routes": "scalar <-> vector",
            "site_ids": [site.site_id],
            "mapped": True,
        }
        row = coverage_map.summarize_thresholds(
            [item], [site], [("cell", evidence)], {}, "v3"
        )[0]
        self.assertEqual(row["status"], "unexecuted")
        self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")

    def test_unmapped_branch_is_a_mapping_limitation_not_a_pass(self) -> None:
        site = branch_site()
        evidence = coverage_map.analyze_export(self.document(0, 0), ROOT, [], [])
        evidence = coverage_map.analyze_export(export([]), ROOT, [], [site])
        row = coverage_map.summarize_thresholds(
            [self.item(site)], [site], [("cell-a", evidence)], {}, "v3"
        )[0]
        self.assertEqual(row["status"], "unmapped")
        self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")

    def test_above_tier_branch_is_not_required_at_this_session(self) -> None:
        site = coverage_map.SiteTarget(
            site_id=0,
            item_key="src/kernel/x86/gf8/avx512/matrix.rs::MATRIX_PEEL_MIN",
            file="src/kernel/x86/gf8/avx512/matrix.rs",
            line=30,
            column=20,
            role="branch",
            condition_line=30,
            condition_column=8,
            condition_end_line=30,
            condition_end_column=25,
        )
        document = export(
            [function("_f", "/repo/src/kernel/x86/gf8/avx512/matrix.rs", [[30, 1, 32, 2, 0, 0, 0, 0]], count=0)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        item = {
            "key": site.item_key,
            "class": "routing_threshold",
            "value": 8192,
            "unit": "bytes",
            "axis": "row length",
            "routes": "unpeeled <-> peeled",
            "site_ids": [site.site_id],
            "mapped": True,
        }
        row = coverage_map.summarize_thresholds(
            [item], [site], [("cell", evidence)], {}, "v3_gfni_crypto"
        )[0]
        self.assertEqual(row["status"], "tier_not_required")
        self.assertFalse(row["required"])
        self.assertEqual(coverage_map.failures([], [row]), [])

        required_row = coverage_map.summarize_thresholds(
            [item], [site], [("cell", evidence)], {}, "v4x"
        )[0]
        self.assertTrue(required_row["required"])
        self.assertEqual(coverage_map.failures([], [required_row])[0]["kind"], "threshold")

    def lower_tier_site(self, role="use", line=2):
        return coverage_map.SiteTarget(
            site_id=0,
            item_key="src/kernel/x86/gf16/ssse3.rs::BOUND",
            file="src/kernel/x86/gf16/ssse3.rs",
            line=line, column=15, role=role,
            condition_line=line, condition_column=5,
            condition_end_line=line, condition_end_column=30,
        )

    def test_unobserved_lower_tier_code_is_deferred_to_its_own_session(self) -> None:
        site = self.lower_tier_site()
        evidence = coverage_map.analyze_export(export([]), ROOT, [], [site])
        for requested in ("v3", "v3_gfni_crypto", "v4x"):
            with self.subTest(requested=requested):
                row = coverage_map.summarize_thresholds(
                    [self.item(site)], [site], [("cell", evidence)], {}, requested,
                )[0]
                self.assertFalse(row["required"])
                self.assertEqual(row["status"], "tier_not_required")
                self.assertEqual(coverage_map.failures([], [row]), [])
        own_row = coverage_map.summarize_thresholds(
            [self.item(site)], [site], [("cell", evidence)], {}, "v2",
        )[0]
        self.assertEqual(coverage_map.failures([], [own_row])[0]["kind"], "threshold")

    def test_observed_lower_tier_branch_remains_required(self) -> None:
        site = self.lower_tier_site(role="branch")
        document = export([], [
            branch_file("/repo/" + site.file, [[2, 5, 2, 30, 1, 0, 0, 0, 4]]),
        ])
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        row = coverage_map.summarize_thresholds(
            [self.item(site)], [site], [("cell", evidence)], {}, "v4x",
        )[0]
        self.assertTrue(row["required"])
        self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")

    def test_observed_lower_tier_wrapper_keeps_unmapped_body_required(self) -> None:
        site = self.lower_tier_site(line=3)
        target = entry_target(
            entry="fgf::kernel::x86::gf16::ssse3::f", tier="v2",
            file=site.file, start=1, end=1,
        )
        document = export([
            function("_f", "/repo/" + site.file, [[1, 1, 1, 12, 1, 0, 0, 0]], count=1),
        ])
        evidence = coverage_map.analyze_export(document, ROOT, [target], [site])
        coverage_map.configure_caller_tiers({
            site.file: source_model("#[archmage::arcane]\npub fn f() {\n    let x = BOUND;\n}\n"),
        })
        try:
            entries = coverage_map.summarize_entries(
                [target], [("cell", evidence)], "v4x", {},
            )
            reach = coverage_map.reachability_index([target], entries)
            row = coverage_map.summarize_thresholds(
                [self.item(site)], [site], [("cell", evidence)], {}, "v4x", reach,
            )[0]
            self.assertTrue(row["required"])
            self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")
        finally:
            coverage_map.configure_sources()

    def test_potential_stronger_callee_does_not_prove_execution(self) -> None:
        caller_file = "src/kernel/x86/demo/avx2.rs"
        helper_file = "src/kernel/x86/demo/avx512.rs"
        coverage_map.configure_caller_tiers({
            caller_file: source_model(
                "pub fn caller() {\n    if selected { super::avx512::helper(); }\n}\n"
            ),
            helper_file: source_model(
                "pub fn helper() {\n    let size = BOUND;\n}\n"
            ),
        })
        try:
            target = entry_target(
                entry="fgf::kernel::x86::demo::avx2::caller",
                file=caller_file, start=1, end=3,
            )
            site = coverage_map.SiteTarget(
                site_id=0, item_key=helper_file + "::BOUND",
                file=helper_file, line=2, column=16, role="use",
            )
            evidence = coverage_map.analyze_export(export([]), ROOT, [], [site])
            entries = [entry_row(target, ["cell"])]
            reach = coverage_map.reachability_index([target], entries)
            self.assertEqual(
                coverage_map._serving_cells(site.item_key, {
                    "sites": [{"file": helper_file, "line": 2, "column": 16}],
                }, reach),
                {"cell"},
            )
            row = coverage_map.summarize_thresholds(
                [self.item(site)], [site], [("cell", evidence)], {}, "v3", reach,
            )[0]
            self.assertFalse(row["required"])
            measured = coverage_map.reachability_index(
                [target], entries,
                {("fgf::kernel::x86::demo::avx512::helper", helper_file): {"cell"}},
            )
            row = coverage_map.summarize_thresholds(
                [self.item(site)], [site], [("cell", evidence)], {}, "v3", measured,
            )[0]
            self.assertTrue(row["required"])
            self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")
        finally:
            coverage_map.configure_sources()

    def test_function_execution_does_not_witness_a_stronger_dispatch_arm(self) -> None:
        file = "src/kernel/routing.rs"
        coverage_map.configure_caller_tiers({
            file: source_model(
                "pub fn dispatch() {\n"
                "    match strategy {\n"
                "        DemoStrategy::Avx512 => {\n"
                "            let size = BOUND;\n"
                "        },\n"
                "        _ => {},\n"
                "    }\n"
                "}\n"
            ),
        })
        try:
            site = coverage_map.SiteTarget(
                site_id=0, item_key=file + "::BOUND", file=file,
                line=4, column=24, role="use",
            )
            reach = coverage_map.reachability_index(
                [], [], {("fgf::kernel::routing::dispatch", file): {"cell"}},
            )
            for count in (0, 1):
                evidence = coverage_map.analyze_export(export([
                    function("_f", "/repo/" + file, [[4, 1, 4, 35, count, 0, 0, 0]]),
                ]), ROOT, [], [site])
                row = coverage_map.summarize_thresholds(
                    [self.item(site)], [site], [("cell", evidence)], {}, "v3", reach,
                )[0]
                self.assertEqual(row["required"], bool(count))
        finally:
            coverage_map.configure_sources()

    def test_selector_value_is_exercised_by_a_mapped_region(self) -> None:
        site = use_site()
        document = export(
            [function("_arm", "/repo/src/kernel.rs", [[39, 5, 41, 9, 1, 0, 0, 0]], count=1)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        item = {
            "key": site.item_key,
            "class": "tuning_parameter",
            "value": 8,
            "unit": "terms",
            "axis": "terms per tile",
            "routes": "tile width",
            "site_ids": [site.site_id],
            "mapped": True,
        }
        row = coverage_map.summarize_thresholds([item], [site], [("cell-a", evidence)], {}, "v3")[0]
        self.assertEqual(row["status"], "exercised")
        self.assertEqual(row["sites"][0]["executed_cells"], ["cell-a"])

    def test_unexecuted_selector_value_is_a_failure(self) -> None:
        site = use_site()
        document = export(
            [function("_arm", "/repo/src/kernel.rs", [[39, 5, 41, 9, 0, 0, 0, 0]], count=0)]
        )
        evidence = coverage_map.analyze_export(document, ROOT, [], [site])
        item = {
            "key": site.item_key,
            "class": "tuning_parameter",
            "value": 8,
            "unit": "terms",
            "axis": "terms per tile",
            "routes": "tile width",
            "site_ids": [site.site_id],
            "mapped": True,
        }
        row = coverage_map.summarize_thresholds([item], [site], [("cell-a", evidence)], {}, "v3")[0]
        self.assertEqual(row["status"], "unexecuted")
        self.assertEqual(coverage_map.failures([], [row])[0]["kind"], "threshold")


class IndexTests(unittest.TestCase):
    def test_segments_resolve_position_counts(self) -> None:
        document = export(
            [],
            [
                branch_file(
                    "/repo/src/kernel.rs",
                    [],
                    segments=[
                        [10, 1, 0, True, True, False],
                        [10, 5, 4, True, True, False],
                        [12, 1, 0, True, True, False],
                    ],
                )
            ],
        )
        index = coverage_map.build_coverage_index(document, ROOT)
        self.assertEqual(coverage_map.position_evidence(index, "src/kernel.rs", 10, 3), (True, 0))
        self.assertEqual(coverage_map.position_evidence(index, "src/kernel.rs", 10, 6), (True, 4))
        self.assertEqual(coverage_map.position_evidence(index, "src/kernel.rs", 12, 2), (True, 0))
        self.assertEqual(coverage_map.position_evidence(index, "src/kernel.rs", 9, 2), (False, 0))

    def test_relative_source_ignores_external_files(self) -> None:
        self.assertEqual(
            coverage_map.relative_source("/repo/src/kernel.rs", ROOT), "src/kernel.rs"
        )
        self.assertIsNone(
            coverage_map.relative_source("/cargo/registry/archmage/src/lib.rs", ROOT)
        )


class SiteTargetTests(unittest.TestCase):
    def test_const_uses_become_branch_and_use_sites(self) -> None:
        inventory = [
            {
                "key": "src/a.rs::ROW_MIN",
                "class": "routing_threshold",
                "value": 32,
                "unit": "bytes",
                "axis": "destination length",
                "routes": "scalar <-> vector",
            }
        ]
        positions = {
            "src/a.rs::ROW_MIN": {
                "file": "src/a.rs",
                "line": 1,
                "column": 1,
                "uses": [
                    {
                        "file": "src/a.rs",
                        "line": 4,
                        "column": 12,
                        "kind": "compare",
                        "operator": "<",
                        "condition_line": 4,
                        "condition_column": 5,
                        "condition_end_line": 4,
                        "condition_end_column": 19,
                    },
                    {"file": "src/a.rs", "line": 8, "column": 12, "kind": "use"},
                ],
            }
        }
        targets, items = coverage_map.build_site_targets(inventory, positions)
        self.assertEqual([target.role for target in targets], ["branch", "use"])
        self.assertEqual(items[0]["site_ids"], [0, 1])
        self.assertEqual(items[0]["status"] if "status" in items[0] else None, None)

    def test_comparison_item_uses_its_condition_span(self) -> None:
        inventory = [
            {
                "key": "src/a.rs::fn f::cmp:len<32:right",
                "class": "routing_threshold",
                "value": 32,
                "unit": "bytes",
                "axis": "destination length",
                "routes": "scalar <-> vector",
            }
        ]
        positions = {
            "src/a.rs::fn f::cmp:len<32:right": {
                "file": "src/a.rs",
                "line": 4,
                "column": 12,
                "condition": {"line": 4, "column": 5, "end_line": 4, "end_column": 19},
            }
        }
        targets, items = coverage_map.build_site_targets(inventory, positions)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].role, "branch")
        self.assertEqual(targets[0].condition_column, 5)
        self.assertEqual(items[0]["site_ids"], [0])

    def test_selector_item_is_a_use_site(self) -> None:
        inventory = [
            {
                "key": "src/a.rs::fn f::match:bytes:1=>8:value",
                "class": "tuning_parameter",
                "value": 8,
                "unit": "count",
                "axis": "row length",
                "routes": "scalar <-> vector",
            }
        ]
        positions = {
            "src/a.rs::fn f::match:bytes:1=>8:value": {
                "file": "src/a.rs",
                "line": 6,
                "column": 14,
            }
        }
        targets, _ = coverage_map.build_site_targets(inventory, positions)
        self.assertEqual(targets[0].role, "use")

    def test_const_without_uses_has_no_sites(self) -> None:
        inventory = [
            {
                "key": "src/a.rs::LONE",
                "class": "tuning_parameter",
                "value": 8,
                "unit": "count",
                "axis": "",
                "routes": "",
            }
        ]
        positions = {"src/a.rs::LONE": {"file": "src/a.rs", "line": 1, "column": 1, "uses": []}}
        targets, items = coverage_map.build_site_targets(inventory, positions)
        self.assertEqual(targets, [])
        self.assertEqual(items[0]["site_ids"], [])


class ExceptionTests(unittest.TestCase):
    def test_ambiguous_attribution_cannot_be_waived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            for cells in (None, ["below", "above"]):
                entry = {
                    "key": "src/kernel.rs::BOUND", "class": "ambiguous",
                    "reason": "overlapping compiler spans",
                }
                if cells is not None:
                    entry["straddling_cells"] = cells
                path.write_text(json.dumps({"exceptions": [entry]}), encoding="utf-8")
                with self.subTest(cells=cells):
                    with self.assertRaises(coverage_map.CoverageMapError):
                        coverage_map.load_exceptions(path)

    def test_exceptions_require_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            path.write_text(json.dumps({"exceptions": [{"item": "x"}]}), encoding="utf-8")
            with self.assertRaises(coverage_map.CoverageMapError):
                coverage_map.load_exceptions(path)

    def test_tier_witness_sets_and_opaque_callers_require_complete_scopes(self) -> None:
        entry = {
            "key": "src/kernel.rs::BOUND", "class": "unmapped_arcane",
            "reason": "opaque nested body", "tiers": ["v3", "v4x"],
            "straddling_cells": {"v3": ["a", "b"], "v4x": ["c", "d"]},
            "via_entry": {"v4x": "fgf::kernel::parent"},
        }
        invalid = [
            dict(entry, tiers=None),
            dict(entry, straddling_cells={"v3": ["a", "b"]}),
            dict(entry, straddling_cells={"v3": [], "v4x": ["c", "d"]}),
            dict(entry, straddling_cells={"v3": ["a", "b"], "scalar": ["c", "d"]}),
            dict(entry, via_entry="fgf::kernel::parent"),
            dict(entry, via_entry={"scalar": "fgf::kernel::parent"}),
            dict(entry, via_entry={"v4x": ""}),
            dict(entry, via_entry={}),
            dict(entry, **{"class": "unmapped_branch"}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            for declared in invalid:
                path.write_text(json.dumps({"exceptions": [declared]}), encoding="utf-8")
                with self.subTest(declaration=declared):
                    with self.assertRaises(coverage_map.CoverageMapError):
                        coverage_map.load_exceptions(path)

    def test_missing_file_is_an_empty_exception_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                coverage_map.load_exceptions(Path(temporary) / "absent.json"), {}
            )

    def test_duplicate_exceptions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            path.write_text(
                json.dumps(
                    {
                        "exceptions": [
                            {"key": "x", "class": "unreachable", "reason": "one"},
                            {"key": "x", "class": "unreachable", "reason": "two"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(coverage_map.CoverageMapError):
                coverage_map.load_exceptions(path)


BOUNDARY_SRC = (
    "/// Delayed tile kernel.\n"
    "pub fn tile_kernel(rows: &mut [u8], srcs: usize) {\n"
    "    let cut = rows.len() >= TILE && srcs >= SOURCES_FLOOR;\n"
    "}\n"
    "const TILE: usize = 4;\n"
    "const SOURCES_FLOOR: usize = 8;\n"
)

BOUNDARY_FILE = "src/kernel/a.rs"
BOUNDARY_ENTRY = "fgf::kernel::a::tile_kernel"


def boundary_world() -> tuple:
    coverage_map.configure_caller_tiers(
        {BOUNDARY_FILE: source_model(BOUNDARY_SRC)}
    )
    target = entry_target(
        entry=BOUNDARY_ENTRY, tier="v3", file=BOUNDARY_FILE, start=1, end=1,
    )
    return target


def boundary_row(key=f"{BOUNDARY_FILE}::TILE", item_class="tuning_parameter",
                 value=4, unit="rows", axis="outputs per delayed tile",
                 comparison=None, status="unmapped"):
    return {
        "key": key,
        "class": item_class,
        "required": True,
        "value": value,
        "unit": unit,
        "axis": axis,
        "routes": "",
        "comparison": comparison,
        "status": status,
        "reason": "no mapping",
        "sites": [{"file": BOUNDARY_FILE, "line": 3, "column": 14}],
    }


def boundary_cells(specs):
    cells = []
    for cell_id, rows, sources, operation in specs:
        cells.append(
            {"id": cell_id, "operation": operation, "row_len": 256,
             "rows": rows, "sources": sources}
        )
    return cells


class OrdinaryFunctionWitnessTests(unittest.TestCase):
    file = "src/ops/recip.rs"
    symbol = "_RINvNtNtCsabc_3fgf3ops5recip8inv_intoKh0_E"
    source = (
        "pub fn inv_into(dst: &mut [u8]) {\n"
        "    if dst[0] != 0 { stage(dst); }\n"
        "}\n"
        "fn stage(dst: &mut [u8]) {\n"
        "    let chunk = CHUNK_ELEMS;\n"
        "}\n"
        "const CHUNK_ELEMS: usize = 64;\n"
    )

    def setUp(self) -> None:
        coverage_map.configure_caller_tiers({self.file: source_model(self.source)})

    def tearDown(self) -> None:
        coverage_map.configure_sources()

    def witnesses(self, *, symbol=None, file=None, count=1, cells=("lo", "hi"),
                  startup_count=0, helper_count=None):
        document = export([
            function(
                self.symbol if symbol is None else symbol,
                "/repo/" + (self.file if file is None else file),
                [[1, 1, 3, 2, count, 0, 0, 0]], count=count,
            ),
            function(
                "_RNvNtNtCsabc_3fgf3ops5recip5stage",
                "/repo/" + (self.file if file is None else file),
                [[4, 1, 6, 2, count if helper_count is None else helper_count, 0, 0, 0]],
                count=count if helper_count is None else helper_count,
            ),
        ])
        evidence = coverage_map.analyze_export(document, ROOT, [], [])
        return coverage_map.operation_function_cells(
            [(cell, evidence) for cell in cells],
            [{"id": cell, "operation": "inv_into"} for cell in cells],
            [{**item, "count": startup_count} for item in evidence["functions"]],
        )

    def validate(self, witnesses, lengths=(128, 256), element_bytes=1):
        row = boundary_row(
            key=f"{self.file}::CHUNK_ELEMS", value=64,
            unit="elements", axis="elements per batch",
        )
        row["sites"] = [{"file": self.file, "line": 5, "column": 17}]
        exceptions = {
            row["key"]: {
                "class": "unmapped_arcane",
                "reason": "helper has no mapping",
                "straddling_cells": ["lo", "hi"],
            },
        }
        cells = [
            {"id": cell, "operation": "inv_into", "row_len": length,
             "rows": 1, "sources": 1, "element_bytes": element_bytes}
            for cell, length in zip(("lo", "hi"), lengths)
        ]
        return coverage_map.validate_exceptions(
            exceptions, [], [], [row], cells, "scalar", witnesses,
        )

    def test_a_measured_scalar_function_witnesses_its_direct_helper(self) -> None:
        self.assertEqual(self.validate(self.witnesses()), [])

    def test_zero_function_count_cannot_witness_a_helper(self) -> None:
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(self.witnesses(count=0))

    def test_startup_only_execution_cannot_witness_a_schedule(self) -> None:
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(self.witnesses(count=1, startup_count=1))
        self.assertEqual(self.validate(self.witnesses(count=2, startup_count=1)), [])

    def test_an_unexecuted_or_startup_only_helper_does_not_inherit_root_credit(self) -> None:
        for kwargs in ({"helper_count": 0},
                       {"count": 2, "startup_count": 1, "helper_count": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(coverage_map.CoverageMapError):
                    self.validate(self.witnesses(**kwargs))

    def test_element_schedules_use_compiler_element_width_not_byte_length(self) -> None:
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(self.witnesses(), lengths=(32, 64), element_bytes=2)
        self.assertEqual(
            self.validate(self.witnesses(), lengths=(64, 128), element_bytes=2), [],
        )

    def test_another_function_or_file_cannot_witness_the_operation(self) -> None:
        for change in (
            {"symbol": self.symbol.replace("8inv_into", "10inv_assign")},
            {"symbol": self.symbol.replace("8inv_into", "17batch_invert_into")},
            {"file": "src/ops/another.rs"},
            {"symbol": "_Rmalformed"},
            {"symbol": "_RNvMCsabc_3fgf3ops5recip8inv_into"},
        ):
            with self.subTest(change=change):
                with self.assertRaises(coverage_map.CoverageMapError):
                    self.validate(self.witnesses(**change))

    def test_witnesses_from_a_previous_cell_set_are_not_reused(self) -> None:
        self.assertEqual(self.validate(self.witnesses()), [])
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(self.witnesses(cells=("lo",)))

    def test_ambiguous_source_functions_do_not_receive_a_witness(self) -> None:
        coverage_map.configure_caller_tiers({
            self.file: source_model(self.source + "pub fn inv_into() {}\n"),
        })
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(self.witnesses())


class BoundaryRuleTests(unittest.TestCase):
    def test_compared_tuning_parameters_require_both_predicate_outcomes(self) -> None:
        target = boundary_world()
        ids = ["r3", "r4", "r5", "r8"]
        rows = [entry_row(target, ids)]
        row = boundary_row(comparison="<")
        above_only = boundary_cells(
            [("r5", 5, 1, "mul_into"), ("r8", 8, 1, "mul_into")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, above_only, target, rows)
        both_sides = boundary_cells(
            [("r3", 3, 1, "mul_into"), ("r4", 4, 1, "mul_into")],
        )
        self.assertEqual(self.validate(row, both_sides, target, rows), [])

    def tearDown(self) -> None:
        coverage_map.configure_sources()

    def validate(self, row, cells, target, rows):
        exceptions = {
            row["key"]: {
                "key": row["key"],
                "class": "unmapped_arcane",
                "reason": "arcane body",
                "straddling_cells": [cell["id"] for cell in cells],
            }
        }
        return coverage_map.validate_exceptions(
            exceptions, [target], rows, [row], cells, "v3"
        )

    def test_gather_remainder_alignment_requires_reached_guard_and_both_sides(self) -> None:
        target = boundary_world()
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::cmp:remainder.trailing_zeros()>=5:right",
            value=5, unit="count", axis="remainder alignment exponent", comparison="<",
        )
        cells = boundary_cells([
            ("a4", 1, 3, "mul_add_gather"), ("a5", 1, 3, "mul_add_gather"),
        ])
        cells[0]["row_len"], cells[1]["row_len"] = 48, 32
        rows = [entry_row(target, ["a4", "a5", "invalid"])]
        self.assertEqual(self.validate(row, cells, target, rows), [])
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, cells[:1], target, rows)
        for length, sources in [(4096, 3), (48, 2), (0, 3)]:
            invalid = {
                "id": "invalid", "operation": "mul_add_gather", "row_len": length,
                "rows": 1, "sources": sources,
            }
            with self.subTest(length=length, sources=sources):
                with self.assertRaises(coverage_map.CoverageMapError):
                    self.validate(row, cells + [invalid], target, rows)

    def test_a_schedule_floor_needs_one_complete_block(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["t2", "t4"])]
        partial = boundary_cells(
            [("t2", 2, 1, "mul_into")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(boundary_row(), partial, target, rows)
        complete = boundary_cells(
            [("t2", 2, 1, "mul_into"), ("t4", 4, 1, "mul_into")],
        )
        self.assertEqual(self.validate(boundary_row(), complete, target, rows), [])

    def test_partial_tiles_never_excuse_a_missing_full_block(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["b4", "b8"])]
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::TERM_BLOCK", value=16,
            unit="terms", axis="terms per destination block",
        )
        cells = boundary_cells(
            [("b4", 8, 4, "mul_add_matrix"), ("b8", 8, 8, "mul_add_matrix")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, cells, target, rows)

    def test_one_sided_source_count_boundaries_are_rejected(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["s9", "s10"])]
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::DELAYED_MIN_SOURCES",
            item_class="routing_threshold", value=8,
            unit="count", axis="source count", comparison="<",
        )
        above = boundary_cells(
            [("s9", 4, 9, "mul_add_matrix"), ("s10", 4, 10, "mul_add_matrix")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, above, target, rows)
        both = boundary_cells(
            [("s9", 4, 9, "mul_add_matrix"), ("s7", 4, 7, "mul_add_matrix")],
        )
        rows = [entry_row(target, ["s9", "s7"])]
        self.assertEqual(self.validate(row, both, target, rows), [])

    def test_numeric_left_comparisons_use_the_scanner_predicate(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["n1", "n8"])]
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::fn tile::cmp:2<=nrows:left",
            item_class="tuning_parameter", value=2,
            unit="rows", axis="remaining row count", comparison="<",
        )
        cells = boundary_cells(
            [("n1", 1, 1, "mul_add_matrix"), ("n8", 8, 1, "mul_add_matrix")],
        )
        self.assertEqual(self.validate(row, cells, target, rows), [])
        high = boundary_cells(
            [("n8", 8, 1, "mul_add_matrix"), ("n9", 9, 1, "mul_add_matrix")],
        )
        rows = [entry_row(target, ["n8", "n9"])]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, high, target, rows)

    def test_equality_endpoints_need_both_predicate_sides(self) -> None:
        target = boundary_world()
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::fn scatter::cmp:filled==4:right",
            item_class="tuning_parameter", value=4,
            unit="count", axis="cohort fill", comparison="==",
        )
        straddling = boundary_cells(
            [("f4", 4, 1, "mul_add_scatter"), ("f6", 6, 1, "mul_add_scatter")],
        )
        rows = [entry_row(target, ["f4", "f6"])]
        self.assertEqual(self.validate(row, straddling, target, rows), [])
        unequal = boundary_cells(
            [("f5", 5, 1, "mul_add_scatter"), ("f6", 6, 1, "mul_add_scatter")],
        )
        rows = [entry_row(target, ["f5", "f6"])]
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, unequal, target, rows)

    def test_boundary_items_without_a_bench_axis_fail_closed(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["l1", "l2"])]
        row = boundary_row(
            key=f"{BOUNDARY_FILE}::SCREEN_LANES",
            item_class="unmapped_branch", value=4,
            unit="count", axis="lanes per screen group", status="unexecuted",
        )
        cells = boundary_cells(
            [("l1", 1, 1, "mul_add"), ("l2", 2, 1, "mul_add")],
        )
        exceptions = {
            row["key"]: {
                "key": row["key"],
                "class": "unmapped_branch",
                "reason": "no branch record",
                "straddling_cells": ["l1", "l2"],
            }
        }
        with self.assertRaises(coverage_map.CoverageMapError):
            coverage_map.validate_exceptions(
                exceptions, [target], rows, [row], cells, "v3"
            )

    def test_a_comparison_without_a_cutoff_value_fails(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["c1", "c2"])]
        row = boundary_row(
            item_class="routing_threshold", value=None,
            unit="rows", axis="output row count", comparison="<=",
        )
        cells = boundary_cells(
            [("c1", 1, 1, "mul_into"), ("c2", 2, 1, "mul_into")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, cells, target, rows)

    def test_straddling_cells_mixing_operations_are_rejected(self) -> None:
        target = boundary_world()
        rows = [entry_row(target, ["o1", "o2"])]
        row = boundary_row(
            item_class="routing_threshold", value=4,
            unit="rows", axis="output row count", comparison="<",
        )
        cells = boundary_cells(
            [("o1", 1, 1, "mul_into"), ("o2", 8, 1, "mul_add_matrix")],
        )
        with self.assertRaises(coverage_map.CoverageMapError):
            self.validate(row, cells, target, rows)

    def test_straddling_selection_follows_the_comparison(self) -> None:
        item = boundary_row(
            item_class="routing_threshold", value=4,
            unit="rows", axis="output row count", comparison="<",
        )
        cells = boundary_cells(
            [("c1", 1, 1, "mul_into"), ("c3", 3, 1, "mul_into"),
             ("c5", 5, 1, "mul_into"), ("c9", 9, 1, "mul_into")],
        )
        picked = coverage_map.straddling_cells(item, cells)
        self.assertEqual(picked, ["c3", "c5"])


class AggregateTests(unittest.TestCase):
    def test_a_passing_tier_cannot_hide_a_failed_required_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exceptions = root / "exceptions.json"
            exceptions.write_text(json.dumps({"exceptions": []}), encoding="utf-8")
            key = "src/kernel/routing.rs::BOUND"
            inputs = [root / "first.json", root / "second.json"]
            for failure in ("one_sided", "unexecuted", "unmapped", "partial_mapping", "ambiguous"):
                for passing in ("both_ways", "exercised", "excused"):
                    for path, host, status in zip(inputs, ("first", "second"), (passing, failure)):
                        path.write_text(json.dumps({
                            "host": host, "tier": "v3", "entries": [],
                            "thresholds": [{"key": key, "status": status}],
                        }), encoding="utf-8")
                    with self.subTest(passing=passing, failure=failure):
                        out = root / "aggregate.json"
                        self.assertEqual(
                            coverage_map.aggregate_reports(inputs, exceptions, out, root / "aggregate.md"), 1,
                        )
                        self.assertEqual(json.loads(out.read_text())["thresholds"][key], "gap")

    def test_entry_execution_is_required_on_each_host_at_its_own_tier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exceptions = root / "exceptions.json"
            exceptions.write_text(json.dumps({"exceptions": []}), encoding="utf-8")
            inputs = [root / "first.json", root / "second.json"]
            for statuses in (("executed", "unexecuted"), ("unexecuted", "executed")):
                for path, host, status in zip(inputs, ("first", "second"), statuses):
                    path.write_text(json.dumps({
                        "host": host, "tier": "v3", "thresholds": [],
                        "entries": [{"entry": "kernel", "tier": "v3", "status": status}],
                    }), encoding="utf-8")
                self.assertEqual(
                    coverage_map.aggregate_reports(inputs, exceptions, root / "aggregate.json", root / "aggregate.md"), 1,
                )

    def test_a_threshold_deferred_in_every_session_is_not_covered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exceptions = root / "exceptions.json"
            exceptions.write_text(json.dumps({"exceptions": []}), encoding="utf-8")
            key = "src/kernel/x86/gf16.rs::TERM_TILE"
            inputs = []
            for tier in ("v3", "v4x"):
                path = root / f"{tier}.json"
                path.write_text(json.dumps({
                    "host": "host", "tier": tier, "entries": [],
                    "thresholds": [{"key": key, "status": "tier_not_required"}],
                }), encoding="utf-8")
                inputs.append(path)
            out = root / "aggregate.json"
            summary = root / "aggregate.md"
            self.assertEqual(coverage_map.aggregate_reports(inputs, exceptions, out, summary), 1)
            gaps = json.loads(out.read_text(encoding="utf-8"))["gaps"]
            self.assertTrue(any(gap["kind"] == "threshold" and gap["item"] == key for gap in gaps))
            own = root / "v2.json"
            own.write_text(json.dumps({
                "host": "host", "tier": "v2", "entries": [],
                "thresholds": [{"key": key, "status": "both_ways"}],
            }), encoding="utf-8")
            self.assertEqual(
                coverage_map.aggregate_reports(inputs + [own], exceptions, out, summary), 0,
            )


if __name__ == "__main__":
    unittest.main()