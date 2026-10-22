#!/usr/bin/env python3
"""Tests for the SIMD route ledger and its Python tooling.

The only test suite for ``simd_ledger.py`` is this stdlib ``unittest`` file.
Negative cases assert the exception type (``ValidationError``) or the stable
reject code, never the wording of a message.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simd_ledger as sl
from simd_ledger import ValidationError

CANONICAL_LEDGER_PATH = Path(__file__).resolve().parent / "simd_ledger.json"


def _lane_loss():
    return {
        "lane_bytes": {"scalar": 1, "v2": 16, "v3": 32, "v3_gfni_crypto": 32, "v4x": 64},
        "loss_bytes": {"scalar": 0, "v2": 30, "v3": 62, "v3_gfni_crypto": 62, "v4x": 126},
        "loss_formula": "ceil(2*(lb-1))",
    }


def _tiny_ledger() -> dict:
    """A small valid ledger without GF(2) fixtures or explicit obligations."""
    return {
        "_cell_default": {"status": "unverified", "required": False, "perf": "unmeasured"},
        "callables": {
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
            "ops::pack": {"kind": "bulk", "family": "ops/pack"},
        },
        "rules": {
            "gf8-aes": {
                "group": "binary_flat",
                "fixtures": [{"id": "aes", "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>"}],
            },
            "mersenne31": {
                "group": "specialized_prime",
                "fixtures": [{"id": "m31", "type": "crate::field::Mersenne31"}],
            },
            "packed-padded": {
                "group": "packed", "non_bulk": True,
                "fixtures": [{"id": "ppad", "type": "crate::field::PackedPadded<u8>"}],
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
                **_lane_loss(),
                "justification": "test",
            },
        },
        "geometries": {},
        "coefficient_patterns": {},
        "evidence": {},
    }


def _accept_ledger() -> dict:
    """Minimal ledger with one required obligation for accept audits."""
    return {
        "_cell_default": {"status": "unverified", "required": False, "perf": "unmeasured"},
        "callables": {
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
        },
        "rules": {
            "gf8-aes": {"group": "binary_flat",
                        "fixtures": [{"id": "aes", "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>"}]},
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


def _good_receipt(obligation="ops::mul_add@gf8-aes"):
    return {
        "id": "c1", "obligation": obligation,
        "source_id": "src-1", "host_id": "host-1", "features": "default",
        "request_raw": "v3", "process": "v3", "field": "v3", "result": "pass",
    }


def _good_done(cases=1):
    return {"source_id": "src-1", "host_id": "host-1", "features": "default",
            "request": "v3", "cases": cases}


class TestExpansion(unittest.TestCase):
    """Expansion count and deviation checks from a tiny synthetic ledger."""

    def test_expansion_count(self):
        ledger = _tiny_ledger()
        rules = sl._validate_rules(ledger)
        expanded = sl._expand_obligations(ledger, rules)
        wrules = sl._validate_work_rules(ledger)
        sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})
        self.assertEqual(len(expanded), 6)
        self.assertEqual(sl._count_cells(expanded), 54)

    def test_deviation_outside_expansion_rejected(self):
        ledger = _tiny_ledger()
        ledger["obligations"] = {"ops::add_assign@nonexistent-rule": {"cells": {}}}
        rules = sl._validate_rules(ledger)
        expanded = sl._expand_obligations(ledger, rules)
        with self.assertRaises(ValidationError):
            sl._validate_explicit_obligations(ledger, expanded)

    def test_duplicate_stage_key_rejected(self):
        ledger = _tiny_ledger()
        ledger["work_rules"]["wa"] = {
            "substage": "MulAdd", "scope": None, "unit": "Elements",
            "basis": "Source", "multiplicity": "Rows", **_lane_loss(), "justification": "t",
        }
        ledger["work_rules"]["wb"] = {
            "substage": "MulAdd", "scope": None, "unit": "Elements",
            "basis": "Source", "multiplicity": "Rows", **_lane_loss(), "justification": "t",
        }
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v2": {"status": "unverified", "required": False, "min_serving": "v2",
                                     "work": ["wa", "wb"], "process": "v2", "field": "v2", "perf": "unmeasured"}}}}
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})


class TestCallableClassification(unittest.TestCase):
    """Unclassified and stale callables against a known callable set."""

    def _check(self, known, callables, *args):
        old = sl.KNOWN_CALLABLES
        sl.KNOWN_CALLABLES = set(known)
        try:
            ledger = {"callables": callables}
            sl._validate_callables(ledger, *args)
        finally:
            sl.KNOWN_CALLABLES = old

    def test_unclassified_callable_rejected(self):
        with self.assertRaises(ValidationError):
            self._check(
                ["ops::mul_add", "ops::add_assign"],
                {"ops::mul_add": {"kind": "bulk", "family": "ops/row"}},
                Path("dummy.json"),
            )

    def test_stale_callable_rejected(self):
        with self.assertRaises(ValidationError):
            self._check(
                ["ops::mul_add"],
                {"ops::mul_add": {"kind": "bulk", "family": "ops/row"},
                 "ops::gone": {"kind": "bulk", "family": "ops/row"}},
                Path("dummy.json"),
            )


class TestApplicability(unittest.TestCase):
    """Applicability rule-reference and exclusion validation."""

    def test_undefined_rule_rejected(self):
        ledger = _tiny_ledger()
        apps = dict(ledger["applicability"])
        apps["ops-row-add"] = {"families": ["ops/row"],
                               "rules": ["gf8-aes", "nonexistent-rule"], "exclusions": {}}
        bad = dict(ledger, applicability=apps)
        with self.assertRaises(ValidationError):
            sl._validate_applicability(bad, ledger["rules"])

    def test_exclusion_without_reason_rejected(self):
        ledger = _tiny_ledger()
        apps = dict(ledger["applicability"])
        apps["ops-row-add"] = {"families": ["ops/row"],
                               "rules": ["gf8-aes"], "exclusions": {"gf8-aes": ""}}
        bad = dict(ledger, applicability=apps)
        with self.assertRaises(ValidationError):
            sl._validate_applicability(bad, ledger["rules"])


class TestBudgetAndCoefficients(unittest.TestCase):
    """Generator rejections for non-positive budgets and General-on-GF(2)."""

    def test_non_positive_budget_rejected(self):
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-60": {"rows": 1, "sources": 1, "len": 60, "misalign": 1, "pitch_pad": 0}}
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v3": {"status": "unverified", "required": False, "min_serving": "v3",
                                     "work": ["row-add"], "process": "v3", "field": "v3", "perf": "unmeasured"}}}}
        with self.assertRaises(ValidationError):
            sl._validate_route_budgets(ledger)

    def test_positive_budget_accepted(self):
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v3": {"status": "unverified", "required": False, "min_serving": "v3",
                                     "work": ["row-add"], "process": "v3", "field": "v3", "perf": "unmeasured"}}}}
        sl._validate_route_budgets(ledger)  # must not raise

    def test_general_coefficient_on_gf2_rejected(self):
        ledger = _tiny_ledger()
        ledger["rules"]["residue-gf2"] = {
            "group": "residue", "modulus_class": "gf2_characteristic", "degree": 1,
            "fixtures": [{"id": "rgf2", "type": "crate::field::FiniteField<2, 1, crate::field::Residue>"}],
        }
        ledger["applicability"]["row-mul"] = {"families": ["ops/row"], "rules": ["residue-gf2"], "exclusions": {}}
        ledger["coefficient_patterns"]["general-1"] = {"slots": ["General"]}
        with self.assertRaises(ValidationError):
            sl._validate_gf2_general(ledger)

    def test_zero_one_coefficient_on_gf2_accepted(self):
        ledger = _tiny_ledger()
        ledger["rules"]["residue-gf2"] = {
            "group": "residue", "modulus_class": "gf2_characteristic", "degree": 1,
            "fixtures": [{"id": "rgf2", "type": "crate::field::FiniteField<2, 1, crate::field::Residue>"}],
        }
        ledger["applicability"]["row-mul"] = {"families": ["ops/row"], "rules": ["residue-gf2"], "exclusions": {}}
        ledger["coefficient_patterns"]["zero-1"] = {"slots": ["Zero"]}
        sl._validate_gf2_general(ledger)  # must not raise

    def test_width_generic_lane_loss_conversion(self):
        """2-byte and 8-byte fixtures get different lane/loss from the same rule.
        lane = floor(lane_bytes / elem_bytes), loss = ceil(loss_bytes / elem_bytes)."""
        rule_2b = {"elem_bytes": 2, "group": "packed"}
        rule_8b = {"elem_bytes": 8, "group": "packed"}
        # v2: lane_bytes=16, loss_bytes=30
        self.assertEqual(sl._lane_in_units("Elements", 16, rule_2b), 8)   # 16//2
        self.assertEqual(sl._loss_in_units("Elements", 30, rule_2b), 15)  # ceil(30/2)
        self.assertEqual(sl._lane_in_units("Elements", 16, rule_8b), 2)   # 16//8
        self.assertEqual(sl._loss_in_units("Elements", 30, rule_8b), 4)   # ceil(30/8)
        # Same rule, different budgets
        wr2b = {"lane_bytes": {"v2": 16}, "loss_bytes": {"v2": 30}, "loss_formula": "ceil(2*(lb-1))",
                "unit": "Elements", "basis": "Source", "multiplicity": "One"}
        wr8b = dict(wr2b)
        geo = {"rows": 1, "sources": 1, "len": 64}
        budget_2b = sl._work_budget(wr2b, geo, rule_2b, "v2")
        budget_8b = sl._work_budget(wr8b, geo, rule_8b, "v2")
        # 2-byte: len_in_units=64 elements, lane=8, loss=15 -> mult=1, budget=1*1*(64-15)=49
        # 8-byte: len_in_units=64 elements, lane=2, loss=4  -> budget=1*1*(64-4)=60
        self.assertEqual(budget_2b, 49)
        self.assertEqual(budget_8b, 60)
        self.assertNotEqual(budget_2b, budget_8b)

    def test_gf2_fixture_detection(self):
        self.assertTrue(sl._fixture_is_gf2({"type": "crate::field::FiniteField<2, 1, crate::field::Residue>"}))
        self.assertFalse(sl._fixture_is_gf2({"type": "crate::field::FiniteField<2, 8, crate::field::Polynomial<0x11B>>"}))


class TestCheckFull(unittest.TestCase):
    """Check --full inventory requirements."""

    def _tmp_files(self, d, *, test="t", case="c", surface="s"):
        def w(name, content):
            p = d / name
            p.write_text(content)
            return p
        return w("test.txt", test), w("case.txt", case), w("surface.txt", surface)

    def test_missing_inventory_rejected(self):
        ledger = _tiny_ledger()
        with self.assertRaises(ValidationError):
            sl.cmd_check(ledger, False, None, None, None, None, None)

    def test_empty_rustdoc_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list, case_list, surface = self._tmp_files(dpath)
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = set()
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(_tiny_ledger(), False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_empty_test_list_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list, case_list, surface = self._tmp_files(dpath, test="")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(_tiny_ledger(), False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_empty_case_list_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list, case_list, surface = self._tmp_files(dpath, case="")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(_tiny_ledger(), False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_empty_surface_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list, case_list, surface = self._tmp_files(dpath, surface="")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(_tiny_ledger(), False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old


class TestEvidenceRecords(unittest.TestCase):
    """Evidence record verification against cited receipts and completion."""

    def _evidence_ledger(self, receipts=None, completion=None):
        ledger = _tiny_ledger()
        ledger["evidence"] = {
            "ev1": {
                "file": "route.log",
                "receipts": receipts if receipts is not None else [
                    {"id": "c1", "features": "default", "tier": "v3", "source_id": "src-1"},
                ],
                "completion": completion if completion is not None else {
                    "source_id": "src-1", "host_id": "host-1", "features": "default", "cases": 1,
                },
            }
        }
        return ledger

    def _record(self, d, receipts=None, done=True, done_body=None):
        p = d / "route.log"
        lines = []
        for r in receipts if receipts is not None else [
            {"id": "c1", "obligation": "o", "features": "default", "request_raw": "v3", "source_id": "src-1"},
        ]:
            lines.append(f"ROUTE_RECEIPT\t{r if isinstance(r, str) else _json(r)}")
        if done:
            body = done_body if done_body is not None else {
                "source_id": "src-1", "host_id": "host-1", "features": "default", "request": "v3", "cases": 1,
            }
            lines.append(f"ROUTE_DONE\t{body if isinstance(body, str) else _json(body)}")
        p.write_text("\n".join(lines))
        return p

    def test_records_missing_rejected(self):
        with self.assertRaises(ValidationError):
            sl._verify_evidence_records(self._evidence_ledger(), None)

    def test_cited_record_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(self._evidence_ledger(), Path(d))

    def test_missing_route_done_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, done=False)
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(self._evidence_ledger(), dpath)

    def test_receipt_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, receipts=[{"id": "c1", "obligation": "o", "features": "default",
                                           "request_raw": "v3", "source_id": "src-1"}])
            ledger = self._evidence_ledger(receipts=[{"id": "c2", "features": "default", "tier": "v3", "source_id": "src-1"}])
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(ledger, dpath)

    def test_receipt_features_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, receipts=[{"id": "c1", "obligation": "o", "features": "default",
                                           "request_raw": "v3", "source_id": "src-1"}])
            ledger = self._evidence_ledger(receipts=[{"id": "c1", "features": "all", "tier": "v3", "source_id": "src-1"}])
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(ledger, dpath)

    def test_receipt_tier_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, receipts=[{"id": "c1", "obligation": "o", "features": "default",
                                           "request_raw": "v3", "source_id": "src-1"}])
            ledger = self._evidence_ledger(receipts=[{"id": "c1", "features": "default", "tier": "v2", "source_id": "src-1"}])
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(ledger, dpath)

    def test_receipt_source_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, receipts=[{"id": "c1", "obligation": "o", "features": "default",
                                           "request_raw": "v3", "source_id": "src-1"}])
            ledger = self._evidence_ledger(receipts=[{"id": "c1", "features": "default", "tier": "v3", "source_id": "src-2"}])
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(ledger, dpath)

    def test_evidence_verification_passes(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath)
            sl._verify_evidence_records(self._evidence_ledger(), dpath)  # must not raise


def _json(obj):
    import json
    return json.dumps(obj)


class TestAccept(unittest.TestCase):
    """Every accept reject code, plus one passing audit."""

    def _audit(self, receipts, done, ledger=None, request="v3", features="default",
               source="src-1", host="host-1"):
        ledger = ledger if ledger is not None else _accept_ledger()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log"
            lines = []
            for r in receipts:
                lines.append(f"ROUTE_RECEIPT\t{_json(r)}")
            for line in done:
                lines.append(line)
            p.write_text("\n".join(lines))
            try:
                sl.cmd_accept(p, features, request, source, host, ledger)
            except ValidationError as e:
                return e.code
            return None

    def test_accept_passes(self):
        self.assertIsNone(self._audit([_good_receipt()], [f"ROUTE_DONE\t{_json(_good_done())}"]))

    def test_missing_log(self):
        ledger = _accept_ledger()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValidationError) as ctx:
                sl.cmd_accept(Path(d) / "nope", "default", "v3", "src-1", "host-1", ledger)
            self.assertEqual(ctx.exception.code, "missing_log")

    def _reject(self, receipts, done, code, ledger=None):
        self.assertEqual(self._audit(receipts, done, ledger=ledger), code)

    def _done_line(self, body=None):
        body = body if body is not None else _good_done()
        return f"ROUTE_DONE\t{_json(body)}"

    def test_malformed_receipt(self):
        self.assertEqual(self._audit([], ["ROUTE_RECEIPT\tnot json"]), "malformed_receipt")

    def test_duplicate_done(self):
        self._reject([_good_receipt()], [self._done_line(), self._done_line()], "duplicate_done")

    def test_malformed_done(self):
        self._reject([_good_receipt()], ["ROUTE_DONE\tnot json"], "malformed_done")

    def test_missing_done(self):
        self._reject([_good_receipt()], [], "missing_done")

    def test_source_mismatch(self):
        done = dict(_good_done(), source_id="other")
        self._reject([_good_receipt()], [self._done_line(done)], "source_mismatch")

    def test_host_mismatch(self):
        done = dict(_good_done(), host_id="other")
        self._reject([_good_receipt()], [self._done_line(done)], "host_mismatch")

    def test_feature_mismatch(self):
        done = dict(_good_done(), features="all")
        self._reject([_good_receipt()], [self._done_line(done)], "feature_mismatch")

    def test_request_mismatch(self):
        done = dict(_good_done(), request="scalar")
        self._reject([_good_receipt()], [self._done_line(done)], "request_mismatch")

    def test_empty_log(self):
        self._reject([], [self._done_line()], "empty_log")

    def test_count_mismatch(self):
        done = dict(_good_done(), cases=2)
        self._reject([_good_receipt()], [self._done_line(done)], "count_mismatch")

    def test_no_required_obligations(self):
        done = dict(_good_done(1), request="v2")
        rec = dict(_good_receipt(), request_raw="v2", process="v2", field="v2")
        self.assertEqual(self._audit([rec], [self._done_line(done)], request="v2"), "no_required_obligations")

    def test_duplicate_receipt(self):
        rec = _good_receipt()
        done = dict(_good_done(), cases=2)
        self._reject([rec, rec], [self._done_line(done)], "duplicate_receipt")

    def test_receipt_source_mismatch(self):
        self._reject([dict(_good_receipt(), source_id="other")], [self._done_line()], "receipt_source_mismatch")

    def test_receipt_host_mismatch(self):
        self._reject([dict(_good_receipt(), host_id="other")], [self._done_line()], "receipt_host_mismatch")

    def test_receipt_feature_mismatch(self):
        self._reject([dict(_good_receipt(), features="all")], [self._done_line()], "receipt_feature_mismatch")

    def test_receipt_request_mismatch(self):
        self._reject([dict(_good_receipt(), request_raw="v2")], [self._done_line()], "receipt_request_mismatch")

    def test_receipt_no_result(self):
        rec = _good_receipt()
        del rec["result"]
        self._reject([rec], [self._done_line()], "receipt_no_result")

    def test_fail_receipt(self):
        self._reject([dict(_good_receipt(), result="fail: output_mismatch")], [self._done_line()], "fail_receipt")

    def test_skip_receipt(self):
        self._reject([dict(_good_receipt(), result="skip: unsupported_tier")], [self._done_line()], "skip_receipt")

    def test_unknown_result(self):
        self._reject([dict(_good_receipt(), result="weird")], [self._done_line()], "unknown_result")

    def test_missing_required(self):
        rec = dict(_good_receipt(), obligation="ops::add_assign@gf8-aes")
        self._reject([rec], [self._done_line()], "missing_required")

    def test_receipt_process_mismatch(self):
        self._reject([dict(_good_receipt(), process="v2")], [self._done_line()], "receipt_process_mismatch")

    def test_receipt_field_mismatch(self):
        self._reject([dict(_good_receipt(), field="v2")], [self._done_line()], "receipt_field_mismatch")


class TestGenerate(unittest.TestCase):
    """Generator emit/byte-identity and rustfmt cleanliness."""

    def test_generate_check_one_byte_difference(self):
        ledger = _tiny_ledger()
        content = sl._generate_cases(ledger)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.rs"
            p.write_text(content)
            self.assertEqual(sl.cmd_generate(p, True, ledger), 0)
            bad = content.replace("generated", "generatXd", 1)
            self.assertNotEqual(bad, content)
            p.write_text(bad)
            with self.assertRaises(ValidationError):
                sl.cmd_generate(p, True, ledger)

    @unittest.skipUnless(shutil.which("rustfmt"), "rustfmt not on PATH")
    def test_generate_output_rustfmt_clean(self):
        ledger = _tiny_ledger()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.rs"
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)
            self.assertTrue(p.is_file())


class TestCanonicalLedger(unittest.TestCase):
    """The canonical ledger passes schema-only validation from the CLI."""

    def test_schema_only_cli_passes(self):
        self.assertEqual(sl.main(["check", "--schema-only"]), 0)

    def test_ledger_default_is_next_to_script(self):
        self.assertEqual(sl.Path(__file__).with_name("simd_ledger.json"), CANONICAL_LEDGER_PATH)

    def test_test_subcommand_removed(self):
        self.assertFalse(hasattr(sl, "run_tests"))
        with self.assertRaises(SystemExit):
            sl.main(["test"])


if __name__ == "__main__":
    unittest.main()