"""Behavioral checks for standalone evidence and the two harness protocols."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_baseline as baseline
import bench_pair as pair

CELL = "gf8-aes/mul_add/unprepared/65536x1x1/aligned"


def descriptive_session(session_id="20261108T000000Z-willow", *, bench="operations"):
    metadata = {"id": CELL, "field": "gf8-aes", "operation": "mul_add",
                "prep": "unprepared", "row_len": 65536, "rows": 1,
                "sources": 1, "alignment": "aligned", "bytes": 65536}
    key = CELL
    if bench == "surfaces":
        key = "scalar/mul/gf2p8-aes"
        metadata = {"case": key, "api": ["fgf::FieldElem::mul"], "field": "gf2p8-aes",
                    "operation": "mul", "scope": "steady", "bytes": 0,
                    "lanes": 1, "sources": 2, "destinations": 1,
                    "control": False, "phase": None}
    session = {
        "method_version": baseline.METHOD_VERSION, "kind": "descriptive", "acceptance": False,
        "session_id": session_id, "created_utc": "2026-11-08T00:00:00Z",
        "host": {"hostname": "willow", "cpu_model": "Example CPU", "os": "Linux",
                 "arch": "x86_64", "toolchain_id": "1.98.1/example", "toolchain": "rustc 1.98.1",
                 "measured_utc": "2026-11-08T00:00:00Z", "campaign_lock": "/tmp/bench-records/.campaign.lock",
                 "governor": "performance", "pinned_core": 0},
        "snapshot": {"bench": bench, "features": ["simd512"], "rustflags": pair.DEFAULT_ALIGN_FLAGS,
                     "no_default_features": False, "runner_matches_snapshot": True,
                     "digest": "c" * 64, "resolved": "d" * 40, "source": "HEAD",
                     "toolchain": "rustc 1.98.1"},
        "arms": {"base": {"kind": "library", "repo": ".", "sha": "a" * 40, "metadata_seed": None}},
        "config": {"runs": 3, "rounds": 1},
        "inventory": {"entries": {key: metadata}, "order": [key], "digest": baseline.digest([metadata])},
        "scope": {"kind": "full", "rule": None, "cells": [key], "digest": baseline.digest([key]),
                  "rationale": None, "declaration_digest": None},
        "tiers": [{"requested": "v4x", "reported": "v4x", "field_backends": {metadata["field"]: "v4x"}, "ok": True}],
        "checks": {"v4x": {"full": True, "cells": [key], "inventory_digest": baseline.digest([metadata])}},
        "runs": [{"run": index, "tier": "v4x", "backend": "v4x",
                  "field_backends": {metadata["field"]: "v4x"}, "control": {"start_ns": 1.0, "end_ns": 1.0},
                  "cells": {key: {"summary_ns": value, "rounds_ns": [value]}}}
                 for index, value in enumerate([9.0, 12.0, 10.0])],
        "rejections": [], "rejected": False,
    }
    session["rows"] = baseline.build_rows(session)
    return session


def surface_metadata(key, *, scope="steady", control=False, phase=None):
    return {"case": key, "api": ["fgf::FieldElem::mul"], "field": "gf2p8-aes",
            "operation": "mul", "scope": scope, "bytes": 0, "lanes": 1,
            "sources": 2, "destinations": 1, "control": control, "phase": phase}


def surface_stream(entries, timings=(), *, checked=False):
    lines = ["fgf surfaces benchmark — backend: v3", "FIELD_BACKEND\tprocess\tv3",
             "FIELD_BACKEND\tgf2p8-aes\tv3"]
    lines += ["SURFACE_INVENTORY\t" + json.dumps(item) for item in entries]
    if checked:
        lines += ["SURFACE_CHECKED\t" + item["case"] for item in entries]
        lines += [f"SURFACES_CHECK_COMPLETE\tcases={len(entries)}"]
    lines += ["SURFACE\t" + json.dumps({**item, "ns": 10.0, "backend": "scalar", "samples": 5})
              for item in timings]
    return "\n".join(lines)


class DescriptiveEvidenceTests(unittest.TestCase):
    def test_summary_and_range_are_derived_from_standalone_processes(self):
        session = descriptive_session()
        baseline.validate_session(session)
        row = session["rows"][0]
        self.assertEqual((row["summary_ns"], row["min_ns"], row["max_ns"]), (10.0, 9.0, 12.0))
        row["min_ns"] = 10.0
        with self.assertRaises(ValueError):
            baseline.validate_session(session)

    def test_descriptive_evidence_cannot_claim_pair_acceptance(self):
        for mutation in (lambda s: s["arms"].update(candidate=s["arms"]["base"]),
                         lambda s: s.update(decisions={}),
                         lambda s: s.update(acceptance=True)):
            session = descriptive_session()
            mutation(session)
            with self.assertRaises(ValueError):
                baseline.validate_session(session)

    def test_missing_duplicate_and_wrong_backend_processes_are_rejected(self):
        for mutation in (lambda s: s["runs"].pop(),
                         lambda s: s["runs"].append(copy.deepcopy(s["runs"][0])),
                         lambda s: s["runs"][0].update(backend="scalar"),
                         lambda s: s["runs"][0]["cells"].clear(),
                         lambda s: s["rows"].clear()):
            session = descriptive_session()
            mutation(session)
            with self.assertRaises(ValueError):
                baseline.validate_session(session)

    def test_nonfinite_and_incomplete_measurements_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            session = descriptive_session()
            session["runs"][0]["cells"][CELL]["rounds_ns"] = [value]
            with self.assertRaises(ValueError):
                baseline.validate_session(session)
        session = descriptive_session()
        session["checks"]["v4x"]["full"] = False
        with self.assertRaises(ValueError):
            baseline.validate_session(session)

    def test_default_surface_scope_excludes_queries_and_controls_not_arithmetic(self):
        entries = [surface_metadata("scalar/mul/a"), surface_metadata("query/a", scope="query"),
                   surface_metadata("control/a", control=True, phase="entry")]
        scope = baseline.scope_for({item["case"]: item for item in entries},
                                   [item["case"] for item in entries], "surfaces", None)
        self.assertEqual(scope["cells"], ["scalar/mul/a"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            for wanted in ([], ["absent"], ["scalar/mul/a"] * 2, ["control/a"]):
                path.write_text(json.dumps(wanted))
                with self.assertRaises(ValueError):
                    baseline.scope_for({item["case"]: item for item in entries},
                                       [item["case"] for item in entries], "surfaces", str(path))

    def test_surface_adapter_requires_exact_timing_scope_and_both_controls(self):
        entry = surface_metadata("control/entry", control=True, phase="entry")
        case = surface_metadata("scalar/mul/a")
        other = surface_metadata("scalar/mul/b")
        exit_case = surface_metadata("control/exit", control=True, phase="exit")
        entries = [entry, case, other, exit_case]
        inventory = {item["case"]: item for item in entries}
        parsed, cells, _ = baseline.Surfaces.timing(surface_stream(entries, [entry, case, exit_case]),
                                                  inventory, [case["case"]], 1)
        self.assertEqual(set(cells), {case["case"]})
        for timed in ([entry, case], [entry, case, other, exit_case], [entry, case, case, exit_case]):
            with self.assertRaises(ValueError):
                baseline.Surfaces.timing(surface_stream(entries, timed), inventory, [case["case"]], 1)
        baseline.Surfaces.check(surface_stream(entries, checked=True), inventory, list(inventory))
        with self.assertRaises(ValueError):
            baseline.Surfaces.check(surface_stream(entries[:-1], checked=True), inventory, list(inventory))


if __name__ == "__main__":
    unittest.main()
