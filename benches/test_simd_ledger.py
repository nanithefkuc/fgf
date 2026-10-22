#!/usr/bin/env python3
"""Tests for the SIMD route ledger and its Python tooling.

The only test suite for ``simd_ledger.py`` is this stdlib ``unittest`` file.
Negative cases assert the exception type (``ValidationError``) or the stable
reject code, never the wording of a message. Each rejection test fails for
its named reason only: inputs are valid in every other respect.
"""

from __future__ import annotations

import json as json_mod
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simd_ledger as sl
from simd_ledger import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isa():
    return {
        "lane_bytes": {"v2": 16, "v3": 32, "v3_gfni_crypto": 32, "v4x": 64},
        "process": {
            "default": {"v4x": "v3_gfni_crypto", "v3_gfni_crypto": "v3_gfni_crypto",
                        "v3": "v3", "v2": "v2", "scalar": "scalar"},
            "all":     {"v4x": "v4x", "v3_gfni_crypto": "v3_gfni_crypto",
                        "v3": "v3", "v2": "v2", "scalar": "scalar"},
        }
    }


def _tiny_ledger() -> dict:
    """A small valid v2-schema ledger without explicit obligations."""
    return {
        "isa": _isa(),
        "_cell_default": {"status": "unverified", "required": False, "perf": "unmeasured"},
        "callables": {
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
            "ops::pack": {"kind": "bulk", "family": "ops/pack"},
        },
        "rules": {
            "gf8-aes": {
                "group": "binary_flat",
                "degree": 8,
                "presentation": "Polynomial",
                "elem_bytes": 1,
                "fixtures": [
                    {"id": "aes", "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>",
                     "elem_bytes": 1, "reference": "elem-arith"},
                ],
            },
            "mersenne31": {
                "group": "specialized_prime",
                "elem_bytes": 4,
                "fixtures": [
                    {"id": "m31", "type": "crate::field::Mersenne31",
                     "elem_bytes": 4, "reference": "elem-arith"},
                ],
            },
            "packed-padded": {
                "group": "packed", "non_bulk": True, "elem_bytes": 1,
                "fixtures": [{"id": "ppad", "type": "crate::field::PackedPadded<u8>",
                              "elem_bytes": 1, "reference": "elem-arith"}],
            },
        },
        "applicability": {
            "ops-row-add": {"families": ["ops/row"], "rules": ["gf8-aes", "mersenne31"], "exclusions": {}},
            "ops-pack": {"families": ["ops/pack"], "rules": ["gf8-aes", "mersenne31", "packed-padded"],
                         "exclusions": {"packed-padded": "non-bulk"}},
        },
        "work_rules": {
            "row-add": {
                "substage": "Add", "unit": "Elements", "basis": "Source", "multiplicity": "Rows",
                "iterations": "1", "loss_bytes": "2 * (lane - 1)", "justification": "test",
            },
        },
        "geometries": {},
        "coefficient_patterns": {},
        "evidence": {},
    }


def _accept_ledger() -> dict:
    """Minimal v2-schema ledger with one required obligation for accept audits."""
    return {
        "isa": _isa(),
        "_cell_default": {"status": "unverified", "required": False, "perf": "unmeasured"},
        "callables": {
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
        },
        "rules": {
            "gf8-aes": {
                "group": "binary_flat", "degree": 8, "presentation": "Polynomial", "elem_bytes": 1,
                "fixtures": [
                    {"id": "aes", "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>",
                     "elem_bytes": 1, "reference": "elem-arith"},
                ],
            },
        },
        "applicability": {
            "row": {"families": ["ops/row"], "rules": ["gf8-aes"], "exclusions": {}},
        },
        "obligations": {
            "ops::mul_add@gf8-aes": {
                "cells": {
                    "default/v3": {"status": "unverified", "required": True, "min_serving": "v3",
                                   "work": [], "process": "v3", "field": "v3", "perf": "unmeasured"}
                }
            }
        },
        "work_rules": {},
        "geometries": {},
        "coefficient_patterns": {},
        "evidence": {},
    }


def _good_receipt(obligation="ops::mul_add@gf8-aes", case_id="c1"):
    return {
        "id": case_id, "obligation": obligation,
        "source_id": "src-1", "host_id": "host-1", "features": "default",
        "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass",
    }


def _good_done(cases=1):
    return {"source_id": "src-1", "host_id": "host-1", "features": "default",
            "request": "v3", "cases": cases}


def _json(obj):
    return json_mod.dumps(obj)




def _full_tiny() -> dict:
    """Tiny ledger with one vector-eligible geometry and single-slot patterns."""
    import copy
    ledger = {
        "isa": {
            "lane_bytes": {"v2": 16, "v3": 32, "v3_gfni_crypto": 32, "v4x": 64},
            "process": {
                "default": {"v4x": "v3_gfni_crypto", "v3_gfni_crypto": "v3_gfni_crypto",
                            "v3": "v3", "v2": "v2", "scalar": "scalar"},
                "all": {"v4x": "v4x", "v3_gfni_crypto": "v3_gfni_crypto",
                        "v3": "v3", "v2": "v2", "scalar": "scalar"},
            },
        },
        "_cell_default": {"status": "unverified", "required": False, "perf": "unmeasured"},
        "callables": {
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
        },
        "rules": {
            "gf8-aes": {
                "group": "binary_flat", "characteristic": 2, "degree": 8,
                "presentation": "Polynomial", "elem_bytes": 1,
                "fixtures": [
                    {"id": "aes",
                     "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>",
                     "elem_bytes": 1, "reference": "elem-arith"},
                ],
            },
            "mersenne31": {
                "group": "specialized_prime", "elem_bytes": 4,
                "fixtures": [
                    {"id": "m31", "type": "crate::field::Mersenne31",
                     "elem_bytes": 4, "reference": "elem-arith"},
                ],
            },
        },
        "applicability": {
            "row": {"families": ["ops/row"], "rules": ["gf8-aes", "mersenne31"],
                    "exclusions": {}},
        },
        "work_rules": {
            "row-add": {
                "substage": "Add", "unit": "Elements", "basis": "Source",
                "multiplicity": "Rows", "iterations": "1",
                "loss_bytes": "2 * (lane - 1)", "justification": "test",
            },
            "row-mul-add": {
                "substage": "MulAdd", "unit": "Elements", "basis": "Source",
                "multiplicity": "ActivePairs", "iterations": "1",
                "loss_bytes": "2 * (lane - 1)", "justification": "test",
            },
        },
        "geometries": {
            "row-64": {"family": "single-row", "rows": 1, "sources": 1,
                       "len": 64, "misalign": 1, "pitch_pad": 0},
        },
        "coefficient_patterns": {
            "general-1": {"family": "single-general", "slots": ["General"]},
            "zero-1": {"family": "single-zero", "slots": ["Zero"]},
            "one-1": {"family": "single-one", "slots": ["One"]},
        },
        "evidence": {},
    }
    return ledger


def _required_tiny(features="default", request="v3", status="direct"):
    ledger = _full_tiny()
    ledger["obligations"] = {
        "ops::mul_add@gf8-aes": {
            "cells": {
                f"{features}/{request}": {
                    "status": status, "required": True, "min_serving": request,
                    "work": ["row-mul-add"], "process": request, "field": request,
                    "perf": "unmeasured",
                }
            }
        }
    }
    return ledger


# ---------------------------------------------------------------------------
# Pipeline structure: one expansion shared by every consumer
# ---------------------------------------------------------------------------

class TestOnePipeline(unittest.TestCase):
    def test_obligations_merges_overrides(self):
        ledger = _full_tiny()
        ledger["obligations"] = {
            "ops::mul_add@gf8-aes": {
                "cells": {
                    "default/v3": {"status": "direct", "required": True,
                                   "min_serving": "v3", "work": ["row-mul-add"],
                                   "process": "v3", "field": "v3", "perf": "unmeasured"}
                }
            }
        }
        obs = sl.obligations(ledger)
        self.assertTrue(obs["ops::mul_add@gf8-aes"]["cells"]["default/v3"]["required"])

    def test_cases_derive_from_obligations(self):
        ledger = _full_tiny()
        obs = sl.obligations(ledger)
        derived = sl.cases(ledger, obs)
        self.assertGreater(len(derived), 0)
        for case in derived:
            self.assertIn(case["obligation"], obs)

    def test_generate_and_accept_share_universe(self):
        ledger = _required_tiny()
        obs = sl.obligations(ledger)
        derived = sl.cases(ledger, obs)
        generated_ids = {c["id"] for c in derived}
        required = {c["id"] for c in derived
                    if c["cells"].get("default/v3", {}).get("required")}
        self.assertEqual(required, {c for c in generated_ids if c.startswith("ops::mul_add@gf8-aes#")})
        self.assertGreater(len(required), 0)

    def test_no_second_expansion_path(self):
        # Mutating the ledger between generate and accept changes both
        # consumers identically: one pipeline, no divergent re-derivation.
        ledger = _required_tiny()
        before = {c["id"] for c in sl.cases(ledger, sl.obligations(ledger))}
        ledger["geometries"]["row-64"]["len"] = 128
        after = {c["id"] for c in sl.cases(ledger, sl.obligations(ledger))}
        self.assertEqual(before, after)
        text_before = sl._generate_cases(_required_tiny())
        self.assertIn("len: 64", text_before)
        text_after = sl._generate_cases(ledger)
        self.assertIn("len: 128", text_after)


# ---------------------------------------------------------------------------
# Behavior 1: accept binds receipt to case obligation and case cell
# ---------------------------------------------------------------------------

class TestAcceptBindsCase(unittest.TestCase):
    def _audit(self, receipts, done, ledger, request="v3", features="default"):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log"
            lines = []
            for r in receipts:
                lines.append(f"ROUTE_RECEIPT\t{json_mod.dumps(r)}")
            for line in done:
                lines.append(line)
            p.write_text("\n".join(lines))
            try:
                sl.cmd_accept(p, features, request, "src-1", "host-1", ledger)
            except ValidationError as e:
                return e.code
            return None

    def _done(self, cases):
        return [f"ROUTE_DONE\t{json_mod.dumps({'source_id': 'src-1', 'host_id': 'host-1', 'features': 'default', 'request': 'v3', 'cases': cases})}"]

    def test_unknown_obligation_rejected(self):
        # Review probe: a required case id carrying an unknown obligation.
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "unknown-obligation",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger),
                         "receipt_obligation_mismatch")

    def test_wrong_case_obligation_rejected(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::add_assign@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass"}
        code = self._audit([rec], self._done(1), ledger)
        self.assertEqual(code, "receipt_obligation_mismatch")

    def test_process_from_case_cell_not_receipt(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v2", "field": "v3", "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger), "receipt_process_mismatch")

    def test_field_from_case_cell_not_receipt(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v2", "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger), "receipt_field_mismatch")

    def test_unknown_case_id_rejected(self):
        ledger = _required_tiny()
        rec = {"id": "no-such-case", "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger), "missing_required")


# ---------------------------------------------------------------------------
# Behavior 2: check validates the merged effective expansion
# ---------------------------------------------------------------------------

class TestEffectiveExpansion(unittest.TestCase):
    def test_invalid_status_in_override_rejected(self):
        ledger = _full_tiny()
        ledger["obligations"] = {"ops::mul_add@gf8-aes": {"cells": {
            "default/v3": {"status": "bogus", "required": False, "min_serving": "v3",
                           "work": [], "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_string_required_rejected(self):
        ledger = _full_tiny()
        ledger["obligations"] = {"ops::mul_add@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": "yes", "min_serving": "v3",
                           "work": [], "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_unknown_work_rule_rejected(self):
        ledger = _full_tiny()
        ledger["obligations"] = {"ops::mul_add@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": False, "min_serving": "v3",
                           "work": ["no-such-rule"], "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_invalid_backend_rejected(self):
        ledger = _full_tiny()
        ledger["obligations"] = {"ops::mul_add@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": False, "min_serving": "bogus",
                           "work": [], "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_negative_geometry_rejected(self):
        ledger = _full_tiny()
        ledger["geometries"]["row-64"]["len"] = -4
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_zero_geometry_rejected(self):
        ledger = _full_tiny()
        ledger["geometries"]["row-64"]["len"] = 0
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_unknown_coefficient_variant_rejected(self):
        ledger = _full_tiny()
        ledger["coefficient_patterns"]["general-1"]["slots"] = ["Super"]
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_zero_fixture_width_rejected(self):
        ledger = _full_tiny()
        ledger["rules"]["gf8-aes"]["fixtures"][0]["elem_bytes"] = 0
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_unknown_reference_rejected(self):
        ledger = _full_tiny()
        ledger["rules"]["gf8-aes"]["fixtures"][0]["reference"] = "oracle-x"
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_unknown_callable_in_rule_rejected(self):
        ledger = _full_tiny()
        ledger["rules"]["gf8-aes"]["callables"] = ["ops::no_such_fn"]
        ledger["applicability"]["row"]["rules"] = ["gf8-aes"]
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Behavior 3: check --full resolves every citation
# ---------------------------------------------------------------------------

class TestFullCitations(unittest.TestCase):
    def _inventories(self, d, *, test="t: test\n", case="ROUTE_CASE_LIST\tx\n",
                     surface='SURFACE_INVENTORY\t{"case": "b"}\n'):
        from pathlib import Path
        p = Path(d)
        (p / "t.txt").write_text(test)
        (p / "c.txt").write_text(case)
        (p / "s.txt").write_text(surface)
        (p / "r.json").write_text("{}")
        return p / "r.json", p / "t.txt", p / "c.txt", p / "s.txt"

    def _ledger_with_refs(self, *, cases=None, tests=None, bench=None):
        ledger = _full_tiny()
        ledger["references"] = {
            "cases": cases or [],
            "tests": tests or [],
            "bench": bench or [],
        }
        return ledger

    def test_dangling_case_reference_rejected(self):
        ledger = self._ledger_with_refs(cases=["missing-case"])
        with tempfile.TemporaryDirectory() as d:
            rustdoc, tl, cl, si = self._inventories(d)
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, rustdoc, tl, cl, si, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_dangling_test_reference_rejected(self):
        ledger = self._ledger_with_refs(tests=["missing::test"])
        with tempfile.TemporaryDirectory() as d:
            rustdoc, tl, cl, si = self._inventories(d)
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, rustdoc, tl, cl, si, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_dangling_bench_reference_rejected(self):
        ledger = self._ledger_with_refs(bench=["missing-bench"])
        with tempfile.TemporaryDirectory() as d:
            rustdoc, tl, cl, si = self._inventories(d)
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, rustdoc, tl, cl, si, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_generated_case_reference_resolves(self):
        ledger = self._ledger_with_refs()
        obs = sl.obligations(ledger)
        derived = sl.cases(ledger, obs)
        ledger["references"]["cases"] = [derived[0]["id"]]
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = Path(d)
            (p / "t.txt").write_text("t: test\n")
            (p / "c.txt").write_text(f"ROUTE_CASE_LIST\t{derived[0]['id']}\n")
            (p / "s.txt").write_text('SURFACE_INVENTORY\t{"case": "b"}\n')
            (p / "r.json").write_text("{}")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                self.assertEqual(sl.cmd_check(ledger, False, p / "r.json", p / "t.txt",
                                              p / "c.txt", p / "s.txt", None), 0)
            finally:
                sl.KNOWN_CALLABLES = old

    def _evidence_ledger(self, receipts=None, completion=None, ev_id="ev1"):
        ledger = _full_tiny()
        ledger["evidence"] = {
            ev_id: {
                "file": "route.log",
                "receipts": receipts if receipts is not None else [
                    {"id": "c1", "features": "default", "request": "v3", "source_id": "src-1"},
                ],
                "completion": completion if completion is not None else {
                    "source_id": "src-1", "host_id": "host-1", "features": "default",
                    "request": "v3", "cases": 1,
                },
            }
        }
        return ledger

    def _record(self, d, receipts=None, done=True, done_body=None):
        from pathlib import Path
        p = Path(d) / "route.log"
        lines = []
        for r in receipts if receipts is not None else [
            {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v3",
             "source_id": "src-1", "host_id": "host-1", "result": "pass"},
        ]:
            lines.append(f"ROUTE_RECEIPT\t{r if isinstance(r, str) else json_mod.dumps(r)}")
        if done:
            body = done_body if done_body is not None else {
                "source_id": "src-1", "host_id": "host-1", "features": "default",
                "request": "v3", "cases": 1,
            }
            lines.append(f"ROUTE_DONE\t{body if isinstance(body, str) else json_mod.dumps(body)}")
        p.write_text("\n".join(lines))
        return p

    def test_empty_completion_rejected(self):
        # An empty completion object against a complete record: the cited
        # source identity differs from the record, failing for that reason.
        ledger = self._evidence_ledger(
            receipts=[{"id": "c1", "features": "default", "request": "v3",
                       "source_id": "other"}],
            completion={"source_id": "other"})
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d))
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_duplicate_receipt_ids_rejected(self):
        ledger = self._evidence_ledger()
        rec = {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v3",
               "source_id": "src-1", "host_id": "host-1", "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d), receipts=[rec, rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 2})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_failed_receipt_rejected(self):
        ledger = self._evidence_ledger()
        rec = {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v3",
               "source_id": "src-1", "host_id": "host-1", "result": "fail"}
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d), receipts=[rec])
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_host_mismatch_rejected(self):
        ledger = self._evidence_ledger()
        rec = {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v3",
               "source_id": "src-1", "host_id": "other", "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d), receipts=[rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 1})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_request_mismatch_rejected(self):
        ledger = self._evidence_ledger()
        rec = {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v2",
               "source_id": "src-1", "host_id": "host-1", "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d), receipts=[rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 1})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_count_mismatch_rejected(self):
        ledger = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            self._record(Path(d), done_body={"source_id": "src-1", "host_id": "host-1",
                                             "features": "default", "request": "v3", "cases": 2})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))

    def test_duplicate_done_rejected(self):
        ledger = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = self._record(Path(d))
            with open(p, "a") as f:
                f.write("\nROUTE_DONE\t" + json_mod.dumps(
                    {"source_id": "src-1", "host_id": "host-1",
                     "features": "default", "request": "v3", "cases": 1}))
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, [], Path(d))


# ---------------------------------------------------------------------------
# Behavior 5: pattern selection per actual fixture and callable shape
# ---------------------------------------------------------------------------

class TestPatternSelection(unittest.TestCase):
    def test_gf2_drops_general_per_fixture(self):
        ledger = _full_tiny()
        ledger["rules"]["gf2"] = {
            "group": "residue", "characteristic": 2, "degree": 1, "elem_bytes": 1,
            "fixtures": [{"id": "g2",
                          "type": "crate::field::FiniteField<2, 1, crate::field::Residue>",
                          "elem_bytes": 1, "reference": "elem-arith"}],
        }
        ledger["applicability"]["row2"] = {"families": ["ops/row"], "rules": ["gf2"],
                                           "exclusions": {}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        gf2_cases = [c for c in derived if c["rule_id"] == "gf2"]
        self.assertGreater(len(gf2_cases), 0)
        for case in gf2_cases:
            self.assertNotIn("General", case["pattern"].get("slots", []))

    def test_non_gf2_keeps_general(self):
        ledger = _full_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        aes_general = [c for c in derived
                       if c["rule_id"] == "gf8-aes" and "General" in c["pattern"].get("slots", [])]
        self.assertGreater(len(aes_general), 0)

    def test_coefficient_free_case_has_empty_slots(self):
        ledger = _full_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        add_cases = [c for c in derived if c["callable"] == "ops::add_assign"]
        self.assertGreater(len(add_cases), 0)
        for case in add_cases:
            self.assertEqual(case["pattern"].get("slots", []), [])

    def test_gather_slots_equal_sources(self):
        ledger = _full_tiny()
        ledger["callables"]["ops::mul_add_gather"] = {"kind": "bulk", "family": "ops/gather"}
        ledger["applicability"]["gather"] = {"families": ["ops/gather"],
                                             "rules": ["gf8-aes"], "exclusions": {}}
        ledger["geometries"]["gather-4"] = {"family": "gather", "rows": 4, "sources": 4,
                                            "len": 64, "misalign": 1, "pitch_pad": 0}
        ledger["coefficient_patterns"]["general-4"] = {"family": "multi-all-general",
                                                       "slots": ["General"] * 4}
        derived = sl.cases(ledger, sl.obligations(ledger))
        gather = [c for c in derived if c["callable"] == "ops::mul_add_gather"]
        self.assertGreater(len(gather), 0)
        for case in gather:
            self.assertEqual(len(case["pattern"]["slots"]), 4)

    def test_scatter_slots_equal_rows(self):
        ledger = _full_tiny()
        ledger["callables"]["ops::mul_add_scatter"] = {"kind": "bulk", "family": "ops/scatter"}
        ledger["applicability"]["scatter"] = {"families": ["ops/scatter"],
                                              "rules": ["gf8-aes"], "exclusions": {}}
        ledger["geometries"]["scatter-4"] = {"family": "scatter", "rows": 4, "sources": 1,
                                             "len": 64, "misalign": 1, "pitch_pad": 0}
        ledger["coefficient_patterns"]["general-4"] = {"family": "multi-all-general",
                                                       "slots": ["General"] * 4}
        derived = sl.cases(ledger, sl.obligations(ledger))
        scatter = [c for c in derived if c["callable"] == "ops::mul_add_scatter"]
        self.assertGreater(len(scatter), 0)
        for case in scatter:
            self.assertEqual(len(case["pattern"]["slots"]), 4)

    def test_no_valid_pattern_fails_naming_obligation(self):
        ledger = _full_tiny()
        ledger["rules"]["gf2"] = {
            "group": "residue", "characteristic": 2, "degree": 1, "elem_bytes": 1,
            "fixtures": [{"id": "g2",
                          "type": "crate::field::FiniteField<2, 1, crate::field::Residue>",
                          "elem_bytes": 1, "reference": "elem-arith"}],
        }
        ledger["applicability"]["row2"] = {"families": ["ops/row"], "rules": ["gf2"],
                                           "exclusions": {}}
        ledger["coefficient_patterns"] = {"general-1": {"family": "single-general",
                                                        "slots": ["General"]}}
        with self.assertRaises(ValidationError):
            sl.cases(ledger, sl.obligations(ledger))

    def test_no_coefficient_case_never_empty_for_taker(self):
        ledger = _full_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        for case in derived:
            if sl._callable_takes_coefficient(case["callable"]):
                self.assertGreater(len(case["pattern"].get("slots", [])), 0)


# ---------------------------------------------------------------------------
# Behavior 7: budgets per fixture width
# ---------------------------------------------------------------------------

class TestBudgets(unittest.TestCase):
    def test_m31_lane_loss_at_v3(self):
        wr = {"substage": "MulAdd", "unit": "Elements", "basis": "Source",
              "multiplicity": "Rows", "iterations": "1",
              "loss_bytes": "2 * (lane - 1)", "justification": "t"}
        lane = sl._lane_in_units("Elements", 32, 4)
        loss = sl._loss_in_units("Elements", 62, 4)
        self.assertEqual(lane, 8)
        self.assertEqual(loss, 16)

    def test_target_basis_uses_target_width(self):
        wr = {"substage": "Linear", "unit": "Elements", "basis": "Target",
              "multiplicity": "One", "iterations": "1",
              "loss_bytes": "2 * (lane - 1)", "substage": "Linear", "justification": "t"}
        geo = {"rows": 1, "sources": 1, "len": 64}
        src_w, tgt_w = {"elem_bytes": 1}, {"elem_bytes": 2}
        budget = sl._work_budget(wr, geo, src_w, tgt_w, "v3", _isa(),
                                 {"slots": []}, {}, {})
        self.assertEqual(budget, 1)

    def test_iterations_degree_minus_one(self):
        wr = {"iterations": "degree - 1", "unit": "Elements", "basis": "Source",
              "multiplicity": "One", "loss_bytes": "1", "substage": "MulAdd"}
        self.assertEqual(sl._work_iterations(wr, {"degree": 8, "relative_degree": 1,
                                                  "power": 1, "words": 1,
                                                  "elem_bytes": 1}), 7)

    def test_unknown_iteration_name_rejected(self):
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["iterations"] = "bogus + 1"
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_zero_iterations_rejected_not_clamped(self):
        wr = {"iterations": "0", "unit": "Elements", "basis": "Source",
              "multiplicity": "One", "loss_bytes": "1", "substage": "MulAdd"}
        self.assertEqual(sl._work_iterations(wr, {"degree": 1, "relative_degree": 1,
                                                  "power": 1, "words": 1,
                                                  "elem_bytes": 1}), 0)
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["iterations"] = "0"
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": True, "min_serving": "v3",
                           "work": ["row-add"], "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cases(ledger, sl.obligations(ledger))

    def test_zero_loss_stays_zero(self):
        self.assertEqual(sl._loss_in_units("Elements", 0, 4), 0)

    def test_active_rows_counts_nontrivial(self):
        geo = {"rows": 3, "sources": 4}
        cp = {"slots": ["General", "General", "Zero", "One",
                        "Zero", "Zero", "Zero", "Zero",
                        "One", "General", "Zero", "Zero"]}
        self.assertEqual(sl._active_rows_count(geo, cp), 2)

    def test_all_zero_active_rows_not_required_failure(self):
        ledger = _full_tiny()
        ledger["work_rules"]["row-zero"] = {
            "substage": "MulAdd", "unit": "Elements", "basis": "Source",
            "multiplicity": "ActiveRows", "iterations": "1",
            "loss_bytes": "1", "justification": "t",
        }
        ledger["callables"]["ops::mul_add_matrix"] = {"kind": "bulk", "family": "ops/matrix"}
        ledger["applicability"]["matrix"] = {"families": ["ops/matrix"],
                                             "rules": ["gf8-aes"], "exclusions": {}}
        ledger["geometries"]["matrix-1x2"] = {"family": "matrix", "rows": 1, "sources": 2,
                                              "len": 64, "misalign": 1, "pitch_pad": 0}
        ledger["coefficient_patterns"]["zero-2"] = {"family": "mixed",
                                                    "slots": ["Zero", "Zero"]}
        ledger["obligations"] = {"ops::mul_add_matrix@gf8-aes": {"cells": {
            "default/v3": {"status": "composition", "required": True, "min_serving": "v3",
                           "work": ["row-zero"], "process": "v3", "field": "v3"}}}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        self.assertGreater(len(derived), 0)


# ---------------------------------------------------------------------------
# Behavior 6 is covered by test_generate_and_accept_share_universe above.
# Behavior 4: generated code compiles against the declared interface
# ---------------------------------------------------------------------------

class TestGeneratedCompiles(unittest.TestCase):
    def test_generated_case_count_matches_accept_universe(self):
        ledger = _required_tiny()
        text = sl._generate_cases(ledger)
        generated_ids = set()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Case {"):
                first = line.split('id: "')[1]
                generated_ids.add(first.split('"')[0])
        obs = sl.obligations(ledger)
        required = {c["id"] for c in sl.cases(ledger, obs)
                    if c["cells"].get("default/v3", {}).get("required")}
        self.assertEqual(generated_ids & required, required)

    @unittest.skipUnless(shutil.which("cargo"), "cargo not on PATH")
    def test_generated_compiles_against_interface(self):
        # The generated table references the declared interface: Case/Cell/Work
        # shapes, Backend/Features/Request enums, run paths grouped in family
        # adapter modules, and MapSpec for map cases. A temp crate wires stub
        # adapters with the same shapes and checks the table type-resolves.
        ledger = _required_tiny()
        table = sl._generate_cases(ledger)
        mods: dict[str, set[str]] = {}
        types: set[str] = set()
        for line in table.splitlines():
            line = line.strip()
            if line.startswith("const RUN_"):
                path = line.split("=", 1)[1].strip().rstrip(";")
                assert path.startswith("super::"), path
                mod, _, rest = path[len("super::"):].partition("::")
                fn, _, targs = rest.rpartition("::<")
                mods.setdefault(mod, set()).add(fn)
                types.add(targs)
        self.assertIn("rows", mods)
        self.assertIn("mul_add", mods["rows"])
        for t in types:
            self.assertIn("crate::field::", t)
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            crate = Path(d) / "gencheck"
            (crate / "src").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "gencheck"\nversion = "0.0.0"\nedition = "2021"\n')
            stub_lines = []
            for mod, fns in sorted(mods.items()):
                for fn in sorted(fns):
                    stub_lines.append(
                        f"fn {mod}_{fn}<T>(_: &Case, _: &Cell) -> Outcome "
                        "{ unimplemented!() }")
            norm = table
            for mod, fns in sorted(mods.items()):
                for fn in sorted(fns):
                    norm = norm.replace(f"super::{mod}::{fn}::",
                                        f"{mod}_{fn}::")
            body_lines = []
            skip_use = False
            skip_mod = False
            for line in norm.splitlines():
                if line.startswith("mod rows") or line.startswith("mod maps"):
                    skip_mod = True
                    continue
                if skip_mod:
                    if line.strip() == "}":
                        skip_mod = False
                    continue
                if line.startswith("use crate::kernel"):
                    skip_use = not line.rstrip().endswith(";")
                    continue
                if skip_use:
                    skip_use = not line.rstrip().endswith(";")
                    continue
                body_lines.append(line)
            src_text = (
                "pub enum Backend { Scalar, V2, V3, V3GfniCrypto, V4x, }\n"
                "pub struct Case { pub id: &'static str, pub obligation: &'static str,\n"
                " pub run: fn(&Case, &Cell) -> Outcome, pub geometry: Geometry,\n"
                " pub coeffs: &'static [Coeff], pub map: Option<MapSpec>,\n"
                " pub cells: &'static [Cell], }\n"
                "pub struct MapSpec { pub power: u32, pub images: &'static [u64], }\n"
                "pub struct Geometry { pub rows: usize, pub sources: usize,\n"
                " pub len: usize, pub misalign: usize, pub pitch_pad: usize, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Coeff { General, Zero, One, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Features { Default, All, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Request { V4x, V3GfniCrypto, V3, V2, Scalar, }\n"
                "pub struct Cell { pub features: Features, pub request: Request,\n"
                " pub process: Backend, pub field: Backend, pub required: bool,\n"
                " pub min_serving: Backend, pub work: &'static [Work], }\n"
                "#[derive(Clone, Copy)]\n"
                "pub struct Work { pub substage: Substage,\n"
                " pub scope: Option<(Scope, Pass)>,\n"
                " pub unit: Unit, pub basis: Basis, pub multiplicity: Multiplicity,\n"
                " pub lane: usize, pub loss: usize, pub iterations: usize, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Basis { Source, Target, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Multiplicity { One, Rows, ActiveRows, ActivePairs, UnitAddPairs, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Scope { Inverse, Division, Frobenius, Norm, Trace,\n"
                " Restriction, Membership, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Pass { Main, Prefix, Backward, Output, Conjugate,\n"
                " Accumulate, Project, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Substage { Add, Sub, Mul, MulAdd, MulElementwise, Inverse, Dot,\n"
                " Linear, Pack, Unpack, BitAnd, BitAndNot, BitXor, BitWeight, BitParity, }\n"
                "#[derive(Clone, Copy)]\n"
                "pub enum Unit { Elements, Bytes, Bits, Words, }\n"
                "pub struct Outcome;\n"
                "pub mod field {\n"
                " pub struct Mersenne31;\n"
                " pub struct Polynomial<const P: u64>;\n"
                " pub struct FiniteField<const C: u64, const D: usize, R>(pub core::marker::PhantomData<R>);\n"
                "}\n"
                + "\n".join(stub_lines) + "\n"
                + "\n".join(body_lines) + "\nfn main() {}\n")
            (crate / "src" / "main.rs").write_text(src_text)
            proc = subprocess.run(["cargo", "check", "--offline"],
                                  cwd=str(crate), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])

    def test_adapter_paths_follow_family_grouping(self):
        adapter, fn, _ = sl._adapter_for(
            "ops::mul_add_gather",
            {"type": "crate::field::Mersenne31"}, {})
        self.assertEqual(adapter, "fans")
        self.assertEqual(fn, "mul_add_gather")
        adapter, fn, _ = sl._adapter_for(
            "ops::mul_elementwise",
            {"type": "crate::field::Mersenne31"}, {})
        self.assertEqual(adapter, "elementwise")
        adapter, fn, types = sl._adapter_for(
            "map::embed_into_with",
            {"source": "crate::field::A", "target": "crate::field::B"}, {})
        self.assertEqual(adapter, "maps")
        self.assertIn(",", types)

    def test_map_cases_carry_map_spec(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx",
                          "source": "crate::field::A", "target": "crate::field::B"}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        text = sl._generate_cases(ledger)
        self.assertIn("map: Some(", text)
        self.assertIn("MapSpec", text)


# ---------------------------------------------------------------------------
# Remaining regression coverage: expression, ISA, expansion, budgets
# ---------------------------------------------------------------------------

class TestExpressions(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(sl._evaluate_expr("2 * (lane - 1)", {"lane": 32}), 62)
        self.assertEqual(sl._evaluate_expr("degree - 1", {"degree": 8}), 7)
        self.assertEqual(sl._evaluate_expr("lane // 2", {"lane": 32}), 16)

    def test_unknown_variable_rejected(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("bogus + 1", {"lane": 32})

    def test_negative_result_rejected(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("1 - lane", {"lane": 32})


class TestExpansion(unittest.TestCase):
    def test_default_v4x_downgrade(self):
        ledger = _full_tiny()
        obs = sl.obligations(ledger)
        first = next(iter(sorted(obs)))
        self.assertIn("default/v4x", obs[first]["cells"])
        self.assertEqual(obs[first]["cells"]["default/v4x"]["process"], "v3_gfni_crypto")
        self.assertEqual(obs[first]["cells"]["all/v4x"]["process"], "v4x")

    def test_callable_filter_honoured(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["callables"]["map::trace_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with", "map::trace_into_with"],
            "fixtures": [{"id": "mx",
                          "source": "crate::field::A", "target": "crate::field::B"}],
        }
        ledger["applicability"]["mapx"] = {
            "families": ["map"], "rules": ["map-x"],
            "callable_filter": ["map::embed_into_with"],
            "exclusions": {},
        }
        obs = sl.obligations(ledger)
        self.assertIn("map::embed_into_with@map-x", obs)
        self.assertNotIn("map::trace_into_with@map-x", obs)

    def test_rule_callables_honoured(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["callables"]["map::trace_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx",
                          "source": "crate::field::A", "target": "crate::field::B"}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        obs = sl.obligations(ledger)
        self.assertIn("map::embed_into_with@map-x", obs)
        self.assertNotIn("map::trace_into_with@map-x", obs)

    def test_overlapping_applicability_rejected(self):
        ledger = _full_tiny()
        ledger["applicability"]["row-second"] = {
            "families": ["ops/row"], "rules": ["gf8-aes"], "exclusions": {},
        }
        with self.assertRaises(ValidationError):
            sl.obligations(ledger)

    def test_orphan_bulk_family_rejected(self):
        ledger = _full_tiny()
        ledger["applicability"] = {"row": {"families": ["ops/row"],
                                           "rules": ["gf8-aes"], "exclusions": {}}}
        ledger["callables"]["ops::pack"] = {"kind": "bulk", "family": "ops/pack"}
        with self.assertRaises(ValidationError):
            sl.obligations(ledger)


class TestCheckModes(unittest.TestCase):
    def test_schema_only_counts(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(sl.cmd_check(_full_tiny(), True, None, None, None, None, None), 0)
        self.assertIn("schema valid", buf.getvalue())

    def test_full_requires_inventories(self):
        with self.assertRaises(ValidationError):
            sl.cmd_check(_full_tiny(), False, None, None, None, None, None)

    def test_malformed_surface_rejected(self):
        ledger = _full_tiny()
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = Path(d)
            (p / "t.txt").write_text("t: test\n")
            (p / "c.txt").write_text("ROUTE_CASE_LIST\tx\n")
            (p / "s.txt").write_text("not json\n")
            (p / "r.json").write_text("{}")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, p / "r.json", p / "t.txt",
                                 p / "c.txt", p / "s.txt", None)
            finally:
                sl.KNOWN_CALLABLES = old


class TestAcceptRejects(unittest.TestCase):
    def _audit(self, receipts, done, ledger, request="v3", features="default"):
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = Path(d) / "log"
            lines = []
            for r in receipts:
                lines.append(f"ROUTE_RECEIPT\t{json_mod.dumps(r)}")
            for line in done:
                lines.append(line)
            p.write_text("\n".join(lines))
            try:
                sl.cmd_accept(p, features, request, "src-1", "host-1", ledger)
            except ValidationError as e:
                return e.code
            return None

    def _good(self, ledger, mutate=None):
        derived = sl.cases(ledger, sl.obligations(ledger))
        required = sorted(c["id"] for c in derived
                          if c["cells"].get("default/v3", {}).get("required"))
        recs = []
        for cid in required:
            rec = {"id": cid, "obligation": cid.split("#")[0],
                   "source_id": "src-1", "host_id": "host-1", "features": "default",
                   "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass"}
            recs.append(rec)
        if mutate is not None:
            mutate(recs)
        done = [f"ROUTE_DONE\t{json_mod.dumps({'source_id': 'src-1', 'host_id': 'host-1', 'features': 'default', 'request': 'v3', 'cases': len(recs)})}"]
        return recs, done

    def test_accept_passes(self):
        ledger = _required_tiny()
        recs, done = self._good(ledger)
        self.assertIsNone(self._audit(recs, done, ledger))

    def test_empty_log(self):
        ledger = _required_tiny()
        done = [f"ROUTE_DONE\t{json_mod.dumps({'source_id': 'src-1', 'host_id': 'host-1', 'features': 'default', 'request': 'v3', 'cases': 0})}"]
        self.assertEqual(self._audit([], done, ledger), "empty_log")

    def test_fail_receipt(self):
        ledger = _required_tiny()
        def mutate(recs):
            recs[0]["result"] = "fail"
        recs, done = self._good(ledger, mutate)
        self.assertEqual(self._audit(recs, done, ledger), "fail_receipt")

    def test_skip_receipt(self):
        ledger = _required_tiny()
        def mutate(recs):
            recs[0]["result"] = "skip"
        recs, done = self._good(ledger, mutate)
        self.assertEqual(self._audit(recs, done, ledger), "skip_receipt")

    def test_passing_prefix_rejected(self):
        ledger = _required_tiny()
        def mutate(recs):
            recs[0]["result"] = "passing"
        recs, done = self._good(ledger, mutate)
        self.assertEqual(self._audit(recs, done, ledger), "unknown_result")

    def test_non_string_result_rejected(self):
        ledger = _required_tiny()
        def mutate(recs):
            recs[0]["result"] = True
        recs, done = self._good(ledger, mutate)
        self.assertEqual(self._audit(recs, done, ledger), "receipt_no_result")

    def test_duplicate_receipt(self):
        ledger = _required_tiny()
        recs, done = self._good(ledger)
        dup = recs + [recs[0]]
        done = [f"ROUTE_DONE\t{json_mod.dumps({'source_id': 'src-1', 'host_id': 'host-1', 'features': 'default', 'request': 'v3', 'cases': len(dup)})}"]
        self.assertEqual(self._audit(dup, done, ledger), "duplicate_receipt")

    def test_count_mismatch(self):
        ledger = _required_tiny()
        recs, _done = self._good(ledger)
        done = [f"ROUTE_DONE\t{json_mod.dumps({'source_id': 'src-1', 'host_id': 'host-1', 'features': 'default', 'request': 'v3', 'cases': len(recs) + 1})}"]
        self.assertEqual(self._audit(recs, done, ledger), "count_mismatch")

    def test_request_raw_empty(self):
        ledger = _required_tiny()
        def mutate(recs):
            recs[0]["request_raw"] = ""
        recs, done = self._good(ledger, mutate)
        self.assertEqual(self._audit(recs, done, ledger), "receipt_request_mismatch")


class TestGenerate(unittest.TestCase):
    def test_generate_roundtrip_byte_identity(self):
        ledger = _full_tiny()
        content = sl._generate_cases(ledger)
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = Path(d) / "cases.rs"
            p.write_text(content)
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)
            self.assertEqual(sl.cmd_generate(p, True, ledger), 0)
            bad = p.read_text().replace("generated", "generatXd", 1)
            p.write_text(bad)
            with self.assertRaises(ValidationError):
                sl.cmd_generate(p, True, ledger)

    @unittest.skipUnless(shutil.which("rustfmt"), "rustfmt not on PATH")
    def test_generate_rustfmt_clean(self):
        ledger = _full_tiny()
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p = Path(d) / "cases.rs"
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)

    def test_first_line_marker(self):
        self.assertTrue(sl._generate_cases(_full_tiny()).splitlines()[0].startswith("// @generated"))

    def test_work_uses_lane_loss_iterations(self):
        ledger = _required_tiny()
        text = sl._generate_cases(ledger)
        self.assertIn("lane:", text)
        self.assertIn("loss:", text)
        self.assertIn("iterations:", text)
        self.assertNotIn("lane_bytes:", text)
        self.assertNotIn("loss_bytes:", text)


if __name__ == "__main__":
    unittest.main()
