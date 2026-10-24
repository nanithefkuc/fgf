#!/usr/bin/env python3
"""Tests for the SIMD route ledger and its Python tooling.

The only test suite for ``simd_ledger.py`` is this stdlib ``unittest`` file.
Negative cases assert the exception type (``ValidationError``) or the stable
reject code, never the wording of a message. Each rejection test fails for
its named reason only: inputs are valid in every other respect.
"""

from __future__ import annotations

import json as json_mod
import re
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
        "request_raw": "v3", "process": "v3", "field": "v3",
        "required": True, "result": "pass",
    }


def _good_done(cases=1):
    return {"source_id": "src-1", "host_id": "host-1", "features": "default",
            "request": "v3", "cases": cases}


def _json(obj):
    return json_mod.dumps(obj)


def _case_ids_from_table(text):
    """Case ids present in a rendered route table."""
    return set(re.findall(r'Case \{ id: "([^"]+)"', text))


_TOTAL_DEGREE = re.compile(
    r"FiniteField\s*<\s*(?:0[xX][0-9A-Fa-f_]+|\d+)\s*,\s*(\d+)")
_EXTENSION_STEP = re.compile(r"Extension\s*<[^,]+,\s*(\d+)\s*>")


def _degree_from_type(text):
    """Total degree named by a FiniteField type expression (hex-safe)."""
    match = _TOTAL_DEGREE.search(text or "")
    return int(match.group(1)) if match else None


def _step_from_type(text):
    """Extension step degree named by an Extension type expression."""
    match = _EXTENSION_STEP.search(text or "")
    return int(match.group(1)) if match else None




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
                    {"id": "aes", "degree": 8, "relative_degree": 1,
                     "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>",
                     "elem_bytes": 1, "reference": "elem-arith"},
                ],
            },
            "mersenne31": {
                "group": "specialized_prime", "elem_bytes": 4,
                "fixtures": [
                    {"id": "m31", "degree": 1, "relative_degree": 1,
                     "type": "crate::field::Mersenne31",
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
        # The generated table's ids and the accept required universe both
        # come from cases(): drive generate to a file, then feed accept a
        # receipt for every generated id.
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        universe = {c["id"] for c in derived}
        required = {c["id"] for c in derived
                    if c["cells"].get("default/v3", {}).get("required")}
        self.assertGreater(len(required), 0)
        if not shutil.which("rustfmt"):
            self.skipTest("rustfmt not on PATH")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.rs"
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)
            generated_ids = _case_ids_from_table(p.read_text())
        self.assertEqual(generated_ids, universe)
        recs = [{"id": cid, "obligation": cid.split("#")[0],
                 "source_id": "src-1", "host_id": "host-1", "features": "default",
                 "request_raw": "v3", "process": "v3", "field": "v3",
                 "required": True, "result": "pass"}
                for cid in sorted(required)]
        done = [f"ROUTE_DONE\t{json_mod.dumps(_good_done(len(recs)))}"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log"
            p.write_text("\n".join(
                [f"ROUTE_RECEIPT\t{json_mod.dumps(r)}" for r in recs] + done))
            self.assertEqual(sl.cmd_accept(p, "default", "v3", "src-1", "host-1",
                                           ledger), 0)

    def test_no_second_expansion_path(self):
        # Mutating the ledger changes generation and acceptance together:
        # both consume the same expansion.
        ledger = _required_tiny()
        before = {c["id"] for c in sl.cases(ledger, sl.obligations(ledger))}
        ledger["geometries"]["row-64"]["len"] = 128
        after = {c["id"] for c in sl.cases(ledger, sl.obligations(ledger))}
        self.assertEqual(before, after)
        lengths = {c["geometry"]["len"]
                   for c in sl.cases(ledger, sl.obligations(ledger))}
        self.assertEqual(lengths, {128})


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
               "request_raw": "v3", "process": "v3", "field": "v3",
               "required": True, "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger),
                         "receipt_obligation_mismatch")

    def test_wrong_case_obligation_rejected(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::add_assign@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3",
               "required": True, "result": "pass"}
        code = self._audit([rec], self._done(1), ledger)
        self.assertEqual(code, "receipt_obligation_mismatch")

    def test_process_from_case_cell_not_receipt(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v2", "field": "v3",
               "required": True, "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger), "receipt_process_mismatch")

    def test_field_from_case_cell_not_receipt(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = sorted(c["id"] for c in derived if c["id"].startswith("ops::mul_add@gf8-aes#"))[0]
        rec = {"id": cid, "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v2",
               "required": True, "result": "pass"}
        self.assertEqual(self._audit([rec], self._done(1), ledger), "receipt_field_mismatch")

    def test_unknown_case_id_rejected(self):
        ledger = _required_tiny()
        rec = {"id": "no-such-case", "obligation": "ops::mul_add@gf8-aes",
               "source_id": "src-1", "host_id": "host-1", "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3",
               "required": True, "result": "pass"}
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

    def _case_id(self):
        ledger = _required_tiny()
        derived = sl.cases(ledger, sl.obligations(ledger))
        case = next(c for c in derived
                    if c["cells"].get("default/v3", {}).get("required"))
        return ledger, derived, case

    def _evidence_ledger(self, receipts=None, completion=None, ev_id="ev1"):
        ledger, derived, case = self._case_id()
        cid = case["id"]
        cited = {"id": cid, "features": "default", "request": "v3",
                 "source_id": "src-1", "host_id": "host-1"}
        ledger["evidence"] = {
            ev_id: {
                "file": "route.log",
                "receipts": receipts if receipts is not None else [cited],
                "completion": completion if completion is not None else {
                    "source_id": "src-1", "host_id": "host-1", "features": "default",
                    "request": "v3", "cases": 1,
                },
            }
        }
        return ledger, derived, cid

    def _record(self, d, case_id, receipts=None, done=True, done_body=None):
        from pathlib import Path
        p = Path(d) / "route.log"
        lines = []
        for r in receipts if receipts is not None else [
            {"id": case_id, "obligation": case_id.split("#")[0],
             "features": "default", "request_raw": "v3",
             "process": "v3", "field": "v3", "required": True,
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

    def test_complete_evidence_accepted(self):
        ledger, derived, cid = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid)
            sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_empty_completion_rejected(self):
        ledger, derived, cid = self._evidence_ledger(completion={})
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid)
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_empty_receipt_citations_rejected(self):
        ledger, derived, cid = self._evidence_ledger(receipts=[])
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid)
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_duplicate_receipt_ids_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec, rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 2})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_failed_receipt_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "source_id": "src-1", "host_id": "host-1",
               "result": "fail"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec])
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_host_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "source_id": "src-1", "host_id": "other",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 1})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_request_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v2", "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 1})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_evidence_process_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "process": "scalar", "field": "v3",
               "required": True, "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec])
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_evidence_field_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "process": "v3", "field": "scalar",
               "required": True, "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec])
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_evidence_required_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "process": "v3", "field": "v3",
               "required": False, "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec])
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_generated_case_must_appear_in_runner_list(self):
        ledger = self._ledger_with_refs()
        derived = sl.cases(ledger, sl.obligations(ledger))
        cid = derived[0]["id"]
        ledger["references"]["cases"] = [cid]
        with tempfile.TemporaryDirectory() as d:
            rustdoc, tl, cl, si = self._inventories(d,
                                                     case="ROUTE_CASE_LIST\tother\n")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, rustdoc, tl, cl, si, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_evidence_cited_case_must_appear_in_runner_list(self):
        ledger, derived, cid = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            rustdoc, tl, cl, si = self._inventories(
                d, case="ROUTE_CASE_LIST\tother\n")
            self._record(Path(d), cid)
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::mul_add", "ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, rustdoc, tl, cl, si, Path(d))
            finally:
                sl.KNOWN_CALLABLES = old

    def test_uncited_receipt_identity_conflict_rejected(self):
        # Every receipt in the record must match the completion identity,
        # not just the cited subset.
        ledger, derived, cid = self._evidence_ledger()
        other_id = next(c["id"] for c in derived if c["id"] != cid)
        other = {"id": other_id, "obligation": other_id.split("#")[0],
                 "features": "default", "request_raw": "v3",
                 "source_id": "src-2", "host_id": "host-1", "result": "pass"}
        rec = {"id": cid, "obligation": cid.split("#")[0], "features": "default",
               "request_raw": "v3", "source_id": "src-1", "host_id": "host-1",
               "result": "pass"}
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid, receipts=[rec, other],
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 2})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_citation_identity_required(self):
        ledger, derived, cid = self._evidence_ledger()
        ledger["evidence"]["ev1"]["receipts"] = [
            {"id": cid, "features": "default", "request": "v3", "source_id": "src-1"}]
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid)
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_cited_receipt_not_generated_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        ledger["evidence"]["ev1"]["receipts"] = [
            {"id": "not-a-generated-case", "features": "default",
             "request": "v3", "source_id": "src-1", "host_id": "host-1"}]
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), "not-a-generated-case")
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_count_mismatch_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            self._record(Path(d), cid,
                         done_body={"source_id": "src-1", "host_id": "host-1",
                                    "features": "default", "request": "v3", "cases": 2})
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))

    def test_duplicate_done_rejected(self):
        ledger, derived, cid = self._evidence_ledger()
        with tempfile.TemporaryDirectory() as d:
            p = self._record(Path(d), cid)
            with open(p, "a") as f:
                f.write("\nROUTE_DONE\t" + json_mod.dumps(
                    {"source_id": "src-1", "host_id": "host-1",
                     "features": "default", "request": "v3", "cases": 1}))
            with self.assertRaises(ValidationError):
                sl._check_evidence_citations(ledger, {}, derived, Path(d))


# ---------------------------------------------------------------------------
# Behavior 5: pattern selection per actual fixture and callable shape
# ---------------------------------------------------------------------------

class TestPatternSelection(unittest.TestCase):
    def test_gf2_drops_general_per_fixture(self):
        ledger = _full_tiny()
        ledger["rules"]["gf2"] = {
            "group": "residue", "characteristic": 2, "degree": 1, "elem_bytes": 1,
            "fixtures": [{"id": "g2", "degree": 1, "relative_degree": 1,
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

    def test_fixture_metadata_uses_declared_shape(self):
        # A fixture's declared shape wins over its rule fallback, even when
        # its type text carries no parseable shape.
        ledger = _full_tiny()
        ledger["rules"]["gf2-alias"] = {
            "group": "residue", "characteristic": 3, "degree": 8, "elem_bytes": 1,
            "fixtures": [{"id": "g2alias", "type": "crate::field::Alias",
                          "characteristic": 2, "degree": 1,
                          "relative_degree": 1, "elem_bytes": 1,
                          "reference": "elem-arith"}],
        }
        ledger["applicability"]["row-alias"] = {
            "families": ["ops/row"], "rules": ["gf2-alias"], "exclusions": {}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        alias_cases = [c for c in derived if c["rule_id"] == "gf2-alias"]
        self.assertGreater(len(alias_cases), 0)
        for case in alias_cases:
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
            "fixtures": [{"id": "g2", "degree": 1, "relative_degree": 1,
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
        # 100 source elements of one byte become 100 target elements of eight
        # bytes: a byte-unit budget covers 800 target bytes.
        wr = {"substage": "Linear", "unit": "Bytes", "basis": "Target",
              "multiplicity": "One", "iterations": "1",
              "loss_bytes": "0", "justification": "t"}
        geo = {"rows": 1, "sources": 1, "len": 100}
        src_w, tgt_w = {"elem_bytes": 1}, {"elem_bytes": 8}
        budget = sl._work_budget(wr, geo, src_w, tgt_w, "v3", _isa(),
                                 {"slots": []},
                                 {"degree": 1, "relative_degree": 1}, {})
        self.assertEqual(budget, 800)

    def test_bits_units_do_not_inflate_length(self):
        # A bits geometry is already in bits; only lane and loss convert.
        self.assertEqual(sl._length_in_units("Bits", 4096, 8), 4096)
        wr = {"substage": "BitXor", "unit": "Bits", "basis": "Source",
              "multiplicity": "One", "iterations": "1",
              "loss_bytes": "2*(lane-1)", "justification": "t"}
        geo = {"rows": 1, "sources": 1, "len": 4096}
        width = {"elem_bytes": 1}
        self.assertEqual(
            sl._work_budget(wr, geo, width, width, "v3", _isa(),
                            {"slots": []},
                            {"degree": 1, "relative_degree": 1}, {}), 3600)

    def test_extra_loss_rejected(self):
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["extra_loss"] = "crate::kernel::NONEXISTENT"
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_geometries_exceed_lane_loss_floors(self):
        """Every geometry exceeds its work rule's own lane-loss formula.

        This is not a live tuning-threshold check; it keeps the resolved
        budget strictly positive at the widest lane and element width.
        """
        ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))
        for geo in ledger["geometries"].values():
            bits_geometry = geo.get("family") in ("bits", "bits-gather")
            for wr in ledger["work_rules"].values():
                if (wr["unit"] == "Bits") != bits_geometry:
                    continue
                loss_bytes = sl._compute_loss_bytes(
                    wr, _isa()["lane_bytes"]["v4x"], {"elem_bytes": 8})
                length = sl._length_in_units(wr["unit"], geo["len"], 8)
                loss = sl._loss_in_units(wr["unit"], loss_bytes, 8)
                self.assertGreater(length, loss, (geo, wr))

    def test_canonical_multirow_geometry_is_vector_eligible(self):
        ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))
        multirow = [geo for geo in ledger["geometries"].values()
                    if geo.get("family") == "multi-row"]
        self.assertTrue(multirow)
        for geo in multirow:
            self.assertGreaterEqual(geo["len"], 4096)

    def test_map_relative_degree_derived_from_endpoints(self):
        fixture = {
            "id": "mx", "degree": 2, "relative_degree": 4,
            "source": "crate::field::FiniteField<2, 2, crate::field::Polynomial<7>>",
            "target": "crate::field::FiniteField<2, 8, crate::field::Polynomial<0x11D>>",
            "elem_bytes": 1, "target_elem_bytes": 1,
        }
        meta = sl._fixture_metadata(fixture, {"group": "map"})
        self.assertEqual(meta["relative_degree"], 4)

    def test_map_target_budget_uses_declared_target_width(self):
        fixture = {
            "id": "mx", "degree": 1, "relative_degree": 64,
            "source": "crate::field::FiniteField<2, 1, crate::field::Polynomial<3>>",
            "target": "crate::field::FiniteField<2, 64, crate::field::Extension<crate::field::Rijndael64, 2>>",
            "elem_bytes": 1, "target_elem_bytes": 8,
        }
        meta = sl._fixture_metadata(fixture, {"group": "map_bits"})
        wr = {"substage": "Pack", "unit": "Bytes", "basis": "Target",
              "multiplicity": "One", "iterations": "1",
              "loss_bytes": "0", "justification": "t"}
        geo = {"rows": 1, "sources": 1, "len": 100}
        self.assertEqual(
            sl._resolve_work(wr, geo, meta, "v3", _isa(), {"slots": []})["budget"],
            800)

    def test_fixture_degree_controls_iterations(self):
        fixture = {"id": "d16", "degree": 16, "relative_degree": 1, "elem_bytes": 1,
                   "type": "crate::field::FiniteField<2, 8, crate::field::Polynomial<0x11B>>",
                   "reference": "elem-arith"}
        rule = {"degree": 8}
        wr = {"iterations": "degree - 1", "unit": "Elements", "basis": "Source",
              "multiplicity": "One", "loss_bytes": "0", "substage": "MulAdd"}
        meta = sl._fixture_metadata(fixture, rule)
        self.assertEqual(sl._work_iterations(wr, meta), 15)

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

    def test_active_rows_coefficient_free_counts_sources_for_gather(self):
        geo = {"rows": 3, "sources": 5}
        self.assertEqual(
            sl._multiplicity_count("ActiveRows", geo, {"slots": []}, gather=True), 5)
        self.assertEqual(
            sl._multiplicity_count("ActiveRows", geo, {"slots": []}), 3)
        # Gather coefficients select sources, not destination rows.
        slots = {"slots": ["General", "Zero", "One", "Zero", "General"]}
        self.assertEqual(
            sl._multiplicity_count("ActiveRows", geo, slots, gather=True), 3)

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

    def test_all_zero_active_rows_not_required_cell(self):
        # The same zero-multiplicity composition must not fail when the cell
        # is not required.
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
            "default/v3": {"status": "composition", "required": False, "min_serving": "v3",
                           "work": ["row-zero"], "process": "v3", "field": "v3"}}}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        zero_cases = [c for c in derived if c["callable"] == "ops::mul_add_matrix"]
        self.assertGreater(len(zero_cases), 0)

    def test_zero_budget_status_still_validates_iterations(self):
        # A zero-budget cell may carry work rules; their iteration formula is
        # still validated, so an impossible rule is a diagnostic.
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["iterations"] = "0"
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {"cells": {
            "default/v3": {"status": "zero-budget", "required": False,
                           "min_serving": "v3", "work": ["row-add"],
                           "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cases(ledger, sl.obligations(ledger))

    def test_runtime_expression_error_is_diagnostic(self):
        # A syntactically valid expression that fails at evaluation is a
        # validation diagnostic, never an uncaught exception.
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["iterations"] = "1 // (degree - degree)"
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": True,
                           "min_serving": "v3", "work": ["row-add"],
                           "process": "v3", "field": "v3"}}}}
        with self.assertRaises(ValidationError):
            sl.cases(ledger, sl.obligations(ledger))


# ---------------------------------------------------------------------------
# Behavior 6 is covered by test_generate_and_accept_share_universe above.
# Behavior 4: generated code compiles against the declared interface
# ---------------------------------------------------------------------------

_ROUTE_TYPES = """\
use crate::kernel::Backend;

pub(super) struct Case {
    pub id: &'static str,
    pub obligation: &'static str,
    pub run: fn(&Case, &Cell) -> Outcome,
    pub geometry: Geometry,
    pub coeffs: &'static [Coeff],
    pub map: Option<MapSpec>,
    pub cells: &'static [Cell],
}
/// Construction data for prepared maps: each caller image is the target
/// element's `FieldBuffer` little-endian byte encoding (empty slice list for
/// structural construction).
pub(super) struct MapSpec { pub power: u32, pub images: &'static [&'static [u8]] }
pub(super) struct Geometry { pub rows: usize, pub sources: usize, pub len: usize /* elements */, pub misalign: usize /* bytes */, pub pitch_pad: usize /* elements */ }
pub(super) enum Coeff { General, Zero, One }
pub(super) enum Features { Default, All }
pub(super) enum Request { V4x, V3GfniCrypto, V3, V2, Scalar }
pub(super) struct Cell {
    pub features: Features, pub request: Request,
    pub process: Backend, pub field: Backend,
    pub required: bool, pub min_serving: Backend,
    pub work: &'static [Work],
}
pub(super) struct Work {
    pub substage: Substage,
    pub scope: Option<(Scope, Pass)>,
    pub unit: Unit,
    pub basis: Basis,
    pub multiplicity: Multiplicity,
    pub lane: usize,
    pub loss: usize,
    pub iterations: usize,
}
pub(super) enum Basis { Source, Target }
pub(super) enum Multiplicity {
    One, Rows, ActiveRows, ActivePairs, UnitAddPairs,
}
pub(super) enum Scope { Inverse, Division, Frobenius, Norm, Trace, Restriction, Membership }
pub(super) enum Pass { Main, Prefix, Backward, Output, Conjugate, Accumulate, Project }
pub(super) enum Substage {
    Add, Sub, Mul, MulAdd, MulElementwise, Inverse, Dot,
    Linear, Pack, Unpack, BitAnd, BitAndNot, BitXor, BitWeight, BitParity,
}
pub(super) enum Unit { Elements, Bytes, Bits, Words }
pub struct Outcome;
"""

_FIELD_TYPES = """\
pub struct Mersenne31;
pub struct Polynomial<const P: u64>;
pub struct Extension<R, const N: usize>(pub core::marker::PhantomData<R>);
pub struct Rijndael16;
pub struct FiniteField<const C: u64, const D: usize, R>(pub core::marker::PhantomData<R>);
"""


class TestGeneratedCompiles(unittest.TestCase):
    def _compile_ledger(self):
        ledger = _required_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "construction": "images",
            "arm": "injection", "arm_citation": "src/x.rs:1", "power": 3,
            "images": [[2, 0]],
            "callables": ["map::embed_into_with"],
            "fixtures": [{
                "id": "mx",
                "degree": 8,
                "relative_degree": 2,
                "source": "crate::field::FiniteField<2, 8, crate::field::Polynomial<0x11B>>",
                "target": "crate::field::FiniteField<2, 16, crate::field::Extension<crate::field::Rijndael16, 2>>",
                "elem_bytes": 1,
                "target_elem_bytes": 2,
            }],
        }
        ledger["applicability"]["mapx"] = {
            "families": ["map"], "rules": ["map-x"], "exclusions": {}}
        return ledger

    @unittest.skipUnless(shutil.which("cargo"), "cargo not on PATH")
    def test_generated_compiles_and_parent_reads_cases(self):
        # The unmodified generated table is compiled under cfg(test) with the
        # simd table feature enabled, and the parent module reads both tables.
        # Work is not Copy: the table must build &'static [Work] from consts.
        ledger = self._compile_ledger()
        table = sl._generate_cases(ledger)

        # Adapter modules and runner names come from the route interface,
        # never from the generated text.
        modules = (
            "pub mod rows {\n"
            "    pub fn ops_mul_add<T>(_: &super::Case, _: &super::Cell)"
            " -> super::Outcome { unimplemented!() }\n"
            "    pub fn ops_add_assign<T>(_: &super::Case, _: &super::Cell)"
            " -> super::Outcome { unimplemented!() }\n"
            "}\n"
            "pub mod maps {\n"
            "    pub fn map_embed_into_with<T, U>(_: &super::Case, _: &super::Cell)"
            " -> super::Outcome { unimplemented!() }\n"
            "}"
        )

        gate = "#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\"))]"
        src = (
            "pub mod kernel {\n"
            "    pub enum Backend { Scalar, V2, V3, V3GfniCrypto, V4x }\n"
            "    pub mod tests {\n"
            "        pub mod routes {\n"
            + "\n".join("            " + line if line else ""
                        for line in _ROUTE_TYPES.splitlines()) + "\n"
            + "\n".join("            " + line if line else ""
                        for line in modules.splitlines()) + "\n"
            "            pub mod cases {\n"
            + "\n".join("                " + line if line else ""
                        for line in table.splitlines()) + "\n"
            "            }\n"
            f"            {gate}\n"
            "            #[test]\n"
            "            fn parent_reads_cases() {\n"
            "                assert!(!cases::CASES.is_empty());\n"
            "                #[cfg(feature = \"simd512\")]\n"
            "                let _ = cases::CASES_SIMD512.len();\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "pub mod field {\n"
            + "\n".join("    " + line if line else ""
                        for line in _FIELD_TYPES.splitlines()) + "\n"
            "}\n"
            "fn main() {}\n"
        )
        with tempfile.TemporaryDirectory() as d:
            crate = Path(d) / "gencheck"
            (crate / "src").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "gencheck"\nversion = "0.0.0"\nedition = "2021"\n'
                "[features]\nsimd = []\nsimd512 = []\n")
            (crate / "src" / "main.rs").write_text(src)
            for features in ("simd", "simd,simd512"):
                proc = subprocess.run(
                    ["cargo", "test", "--offline", "--no-run",
                     "--features", features],
                    cwd=str(crate), capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0,
                                 f"{features}: {proc.stderr[-4000:]}")

    def test_map_spec_uses_declared_construction_data(self):
        rule = {"group": "map", "power": 3, "images": [[17, 0], [18, 0]]}
        self.assertEqual(sl._map_spec(rule, {}), (3, [[17, 0], [18, 0]]))
        self.assertEqual(sl._map_spec(rule, {"images": [[19, 0]]}),
                         (3, [[19, 0]]))


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
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A", "target": "crate::field::B",
                          "elem_bytes": 1, "target_elem_bytes": 1}],
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
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A", "target": "crate::field::B",
                          "elem_bytes": 1, "target_elem_bytes": 1}],
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

    def test_override_outside_applicability_rejected(self):
        # Applicability is the only authority for obligation membership: an
        # explicit entry cannot manufacture a rule/callable pair.
        ledger = _full_tiny()
        ledger["applicability"] = {
            "row": {"families": ["ops/row"], "rules": ["gf8-aes"], "exclusions": {}}}
        ledger["obligations"] = {"ops::mul_add@mersenne31": {"cells": {}}}
        with self.assertRaises(ValidationError):
            sl.obligations(ledger)

    def test_map_widths_validated(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A",
                          "target": "crate::field::B",
                          "elem_bytes": -1, "target_elem_bytes": 2}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_map_target_width_validated(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A",
                          "target": "crate::field::B",
                          "elem_bytes": 1, "target_elem_bytes": 0}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_map_fixture_requires_explicit_widths(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "arm": "injection",
            "arm_citation": "src/x.rs:1",
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A",
                          "target": "crate::field::B"}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_map_images_must_be_byte_encodings(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "construction": "images",
            "arm": "injection", "arm_citation": "src/x.rs:1",
            "images": [65536],
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A",
                          "target": "crate::field::B",
                          "elem_bytes": 2, "target_elem_bytes": 4}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_map_image_width_must_match_target(self):
        ledger = _full_tiny()
        ledger["callables"]["map::embed_into_with"] = {"kind": "bulk", "family": "map"}
        ledger["rules"]["map-x"] = {
            "group": "map", "kind": "embedding", "construction": "images",
            "arm": "injection", "arm_citation": "src/x.rs:1",
            "images": [[1, 2]],
            "callables": ["map::embed_into_with"],
            "fixtures": [{"id": "mx", "degree": 1, "relative_degree": 1,
                          "source": "crate::field::A",
                          "target": "crate::field::B",
                          "elem_bytes": 2, "target_elem_bytes": 4}],
        }
        ledger["applicability"]["mapx"] = {"families": ["map"], "rules": ["map-x"],
                                           "exclusions": {}}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_needs_spec_key_rejected(self):
        ledger = _full_tiny()
        ledger["rules"]["gf8-aes"]["fixtures"][0]["needs_spec"] = {"degree": 2}
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)


class TestCanonicalLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))

    def test_schema_check_passes(self):
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sl.cmd_check(self.ledger, True, None, None, None,
                                          None, None), 0)

    def test_component_elementwise_and_pack_obligations(self):
        obs = sl.obligations(self.ledger)
        self.assertIn("ops::mul_elementwise@component-quad-mersenne31", obs)
        self.assertIn("ops::unpack@component-quad-mersenne31", obs)

    def test_component_base_classes_cover_u64_and_cantor(self):
        by_base = {rule.get("base_rule") for rule in self.ledger["rules"].values()
                   if rule.get("group") == "component"}
        self.assertIn("residue-u64", by_base)
        self.assertIn("gf8-cantor", by_base)

    def test_isomorphism_injection_rule_present(self):
        rules = [rule for rule in self.ledger["rules"].values()
                 if rule.get("group") == "map"
                 and rule.get("kind") == "isomorphism"
                 and rule.get("arm") == "injection"]
        self.assertTrue(rules)
        for rule in rules:
            for fixture in rule["fixtures"]:
                self.assertEqual(fixture["source"], fixture["target"])


    def test_multirow_row_add_budget_positive(self):
        ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))
        oid = "ops::add_assign_rows@gf8-poly-aes"
        ledger["obligations"] = {oid: {"cells": {"default/v3": {
            "status": "direct", "required": True, "min_serving": "v3",
            "work": ["row-add"], "process": "v3", "field": "v3",
            "perf": "unmeasured"}}}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        cases = [case for case in derived if case["obligation"] == oid]
        self.assertTrue(cases)
        for case in cases:
            for item in case["cell_work"]["default/v3"]:
                self.assertGreater(item["budget"], 0)

    def test_bit_gather_uses_gather_geometry_and_per_source_budget(self):
        ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))
        oid = "bits::xor_gather@bits-gf2"
        ledger["obligations"] = {oid: {"cells": {"default/v3": {
            "status": "direct", "required": True, "min_serving": "v3",
            "work": ["bits-xor-gather"], "process": "v3", "field": "v3",
            "perf": "unmeasured"}}}}
        derived = sl.cases(ledger, sl.obligations(ledger))
        selected = [case for case in derived if case["obligation"] == oid]
        self.assertEqual(len(selected), 1)
        case = selected[0]
        self.assertEqual(case["geometry"]["family"], "bits-gather")
        self.assertEqual(case["geometry"]["rows"], 8)
        self.assertEqual(case["geometry"]["sources"], 8)
        item = case["cell_work"]["default/v3"][0]
        self.assertEqual(item["multiplicity_count"], 8)
        # Eight selected sources at 3,600 bits each under the V3 lane.
        self.assertEqual(item["budget"], 28_800)

    def test_structural_map_width_class_checked(self):
        ledger = sl.load(Path(__file__).with_name("simd_ledger.json"))
        ledger["rules"]["map-bits-width8"]["target_width_bytes"] = 7
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, True, None, None, None, None, None)

    def test_every_fixture_declares_degree_metadata(self):
        for rid, rule in self.ledger["rules"].items():
            for fixture in rule["fixtures"]:
                for key in ("degree", "relative_degree"):
                    value = fixture.get(key)
                    self.assertIsInstance(value, int, (rid, fixture["id"], key))
                    self.assertGreater(value, 0, (rid, fixture["id"], key))

    def test_component_fixture_degrees_match_declared_data(self):
        for rid, rule in self.ledger["rules"].items():
            if rule.get("group") != "component":
                continue
            for fixture in rule["fixtures"]:
                meta = sl._fixture_metadata(fixture, rule)
                text = fixture.get("type") or fixture.get("source") or ""
                expected = _degree_from_type(text)
                if expected is None:
                    expected = rule["degree"]
                self.assertEqual(meta["degree"], expected, fixture["id"])
                step = _step_from_type(text)
                if step is None:
                    step = rule["degree"]
                self.assertEqual(meta["relative_degree"], step, fixture["id"])
        for rid in ("component-mersenne31-custom", "component-goldilocks"):
            fixture = self.ledger["rules"][rid]["fixtures"][0]
            self.assertEqual(
                sl._fixture_metadata(fixture, self.ledger["rules"][rid])["degree"],
                2, rid)

    def test_no_duplicate_spec_key(self):
        for rule in self.ledger["rules"].values():
            for fixture in rule.get("fixtures", []):
                self.assertNotIn("needs_spec", fixture)


class TestCheckModes(unittest.TestCase):
    def test_schema_only_passes(self):
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sl.cmd_check(_full_tiny(), True, None, None, None, None, None), 0)

    def test_full_requires_inventories(self):
        with self.assertRaises(ValidationError):
            sl.cmd_check(_full_tiny(), False, None, None, None, None, None)

    def test_expression_error_exits_without_traceback(self):
        ledger = _full_tiny()
        ledger["work_rules"]["row-add"]["iterations"] = "1 // (degree - degree)"
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {"cells": {
            "default/v3": {"status": "direct", "required": True,
                           "min_serving": "v3", "work": ["row-add"],
                           "process": "v3", "field": "v3"}}}}
        script = Path(__file__).with_name("simd_ledger.py")
        with tempfile.TemporaryDirectory() as d:
            ledger_path = Path(d) / "ledger.json"
            ledger_path.write_text(json_mod.dumps(ledger))
            proc = subprocess.run(
                [sys.executable, str(script), "--ledger", str(ledger_path),
                 "check", "--schema-only"],
                capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

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
                   "request_raw": "v3", "process": "v3", "field": "v3",
                   "required": True, "result": "pass"}
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
    @unittest.skipUnless(shutil.which("rustfmt"), "rustfmt not on PATH")
    def test_generate_roundtrip_byte_identity(self):
        ledger = _required_tiny()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.rs"
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)
            self.assertEqual(sl.cmd_generate(p, True, ledger), 0)
            p.write_text(p.read_text().replace("generated", "generatXd", 1))
            with self.assertRaises(ValidationError):
                sl.cmd_generate(p, True, ledger)


class TestMapCompleteness(unittest.TestCase):
    def ledger(self):
        return sl.load(Path(__file__).with_name("simd_ledger.json"))

    def test_omitted_callable_requires_reason(self):
        ledger = self.ledger()
        ledger["rules"]["map-embed-residue-column"]["callables"].remove(
            "map::trace_into_with")
        with self.assertRaises(ValidationError):
            sl._validate_applicability(ledger, ledger["rules"])

    def test_duplicate_arm_exclusion_is_accepted(self):
        ledger = self.ledger()
        sl._validate_applicability(ledger, ledger["rules"])
        ledger["rules"]["map-embed-injection-images"]["callable_exclusions"][
            "map::trace_into_with"] = ""
        with self.assertRaises(ValidationError):
            sl._validate_applicability(ledger, ledger["rules"])

    def test_characteristic_requirement_cannot_disappear(self):
        ledger = self.ledger()
        for rid in ledger["applicability"]["map-lane-embedding"]["rules"]:
            fixtures = ledger["rules"][rid]["fixtures"]
            ledger["rules"][rid]["fixtures"] = [
                f for f in fixtures
                if sl._type_characteristic(f.get("source")) == 2
            ]
        with self.assertRaises(ValidationError):
            sl._validate_applicability(ledger, ledger["rules"])

    def test_norm_counts_conjugates_and_products_at_relative_degree(self):
        ledger = self.ledger()
        cases = [c for c in sl.cases(ledger)
                 if c["callable"] == "map::norm_into_with"]
        for case in cases:
            work = case["cell_work"]["all/v3"]
            stages = {(w["substage"], tuple(w["scope"])): w for w in work}
            conjugate = stages[("Linear", ("Norm", "Conjugate"))]
            product = stages[("Mul", ("Norm", "Accumulate"))]
            projection = stages[("Linear", ("Norm", "Project"))]
            repetitions = case["fixture"]["relative_degree"] - 1
            self.assertEqual(conjugate["iterations"], repetitions)
            self.assertEqual(product["iterations"], repetitions)
            self.assertEqual(projection["iterations"], 1)
            self.assertGreater(product["budget"], 0)


if __name__ == "__main__":
    unittest.main()
