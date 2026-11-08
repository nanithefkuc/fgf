"""Tests for the instrumented coverage map."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
        row = {
            "key": "src/a.rs::TILE",
            "class": "tuning_parameter",
            "required": True,
            "value": 8,
            "unit": "rows",
            "status": "unmapped",
            "reason": "no mapping",
            "sites": [],
        }
        cells = [
            {"id": "small", "row_len": 64, "rows": 2, "sources": 1, "bytes": 128},
            {"id": "large", "row_len": 64, "rows": 16, "sources": 1, "bytes": 1024},
        ]
        exceptions = {
            row["key"]: {
                "key": row["key"],
                "class": "unmapped_arcane",
                "reason": "arcane body",
                "straddling_cells": ["small", "large"],
            }
        }
        coverage_map.validate_exceptions(exceptions, [], [row], cells, "v3")
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
            coverage_map.validate_exceptions(bad, [], [row], cells, "v3")

    def test_unmapped_branch_exception_requires_unexecuted_and_straddling(self) -> None:
        row = {
            "key": "src/a.rs::CUTOFF",
            "class": "routing_threshold",
            "required": True,
            "value": 64,
            "unit": "bytes",
            "status": "unexecuted",
            "reason": "no branch record",
            "sites": [],
        }
        cells = [
            {"id": "small", "row_len": 32, "rows": 1, "sources": 1, "bytes": 32},
            {"id": "large", "row_len": 128, "rows": 1, "sources": 1, "bytes": 128},
        ]
        exceptions = {
            row["key"]: {
                "key": row["key"],
                "class": "unmapped_branch",
                "reason": "the compiler emitted no branch record",
                "straddling_cells": ["small", "large"],
            }
        }
        self.assertEqual(
            coverage_map.validate_exceptions(exceptions, [], [row], cells, "v3"), []
        )
        coverage_map.apply_exceptions(exceptions, [row], "v3")
        self.assertEqual(row["status"], "excused")

        row["status"] = "both_ways"
        diagnostics = coverage_map.validate_exceptions(
            exceptions, [], [row], cells, "v3"
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("stale", diagnostics[0]["reason"])

    def test_stale_exception_keys_fail(self) -> None:
        cell = {"id": "c", "row_len": 64, "rows": 1, "sources": 1, "bytes": 64}
        exceptions = {
            "unknown": {
                "key": "unknown",
                "class": "unmapped_arcane",
                "reason": "stale",
                "straddling_cells": ["c"],
            }
        }
        with self.assertRaises(coverage_map.CoverageMapError):
            coverage_map.validate_exceptions(exceptions, [], [], [cell], "v3")


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
        limitation = coverage_map.mapping_limitations([], [row], "v3_gfni_crypto")
        self.assertTrue(any("stronger than the requested tier" in entry["reason"] for entry in limitation))

        required_row = coverage_map.summarize_thresholds(
            [item], [site], [("cell", evidence)], {}, "v4x"
        )[0]
        self.assertTrue(required_row["required"])
        self.assertEqual(coverage_map.failures([], [required_row])[0]["kind"], "threshold")

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
    def test_exceptions_require_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exceptions.json"
            path.write_text(json.dumps({"exceptions": [{"item": "x"}]}), encoding="utf-8")
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
                            {"item": "x", "reason": "one"},
                            {"item": "x", "reason": "two"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(coverage_map.CoverageMapError):
                coverage_map.load_exceptions(path)


class ReportTests(unittest.TestCase):
    def test_markdown_reports_gaps_and_the_mapping_note(self) -> None:
        report = {
            "tier": "v3_gfni_crypto",
            "host": "host",
            "toolchain": "nightly",
            "cells": 2,
            "runtime_seconds": 1.5,
            "binary": "/tmp/binary",
            "bench": "operations",
            "entries": [
                {
                    "entry": "e",
                    "tier": "v3",
                    "required": True,
                    "status": "unexecuted",
                    "evidence": "none",
                    "executed_by": [],
                    "count": 0,
                }
            ],
            "thresholds": [
                {
                    "key": "k",
                    "class": "routing_threshold",
                    "value": 1,
                    "unit": "bytes",
                    "status": "one_sided",
                    "sites": [
                        {
                            "true": 1,
                            "false": 0,
                            "true_cells": ["cell"],
                            "false_cells": [],
                            "executed_cells": [],
                        }
                    ],
                }
            ],
            "unreachable": [{"item": "u", "reason": "cfg(test)"}],
            "mapping_limitations": [],
            "gaps": [{"kind": "entry", "item": "e", "reason": "no cells"}],
        }
        markdown = coverage_map.render_markdown(report)
        self.assertIn("## Gaps", markdown)
        self.assertIn("#[archmage::arcane]", markdown)
        self.assertIn("cfg(test)", markdown)


if __name__ == "__main__":
    unittest.main()