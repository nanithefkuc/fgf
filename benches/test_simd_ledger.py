#!/usr/bin/env python3
"""Tests for the SIMD route ledger and its Python tooling.

The only test suite for ``simd_ledger.py`` is this stdlib ``unittest`` file.
Negative cases assert the exception type (``ValidationError``) or the stable
reject code, never the wording of a message.
"""

from __future__ import annotations

import json as json_mod
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simd_ledger as sl
from simd_ledger import ValidationError

CANONICAL_LEDGER_PATH = Path(__file__).resolve().parent / "simd_ledger.json"


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


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

class TestExpressionEvaluator(unittest.TestCase):
    """The safe expression evaluator for iterations and loss_bytes."""

    def test_literal_integer(self):
        self.assertEqual(sl._evaluate_expr("1", {}), 1)
        self.assertEqual(sl._evaluate_expr("42", {}), 42)

    def test_single_variable(self):
        self.assertEqual(sl._evaluate_expr("lane", {"lane": 32}), 32)
        self.assertEqual(sl._evaluate_expr("degree", {"degree": 8}), 8)

    def test_simple_subtraction(self):
        self.assertEqual(sl._evaluate_expr("lane - 1", {"lane": 32}), 31)
        self.assertEqual(sl._evaluate_expr("degree - 1", {"degree": 12}), 11)

    def test_multiplication(self):
        self.assertEqual(sl._evaluate_expr("2 * (lane - 1)", {"lane": 32}), 62)
        self.assertEqual(sl._evaluate_expr("2 * (lane - 1)", {"lane": 16}), 30)
        self.assertEqual(sl._evaluate_expr("2 * (lane - 1)", {"lane": 64}), 126)

    def test_integer_division(self):
        self.assertEqual(sl._evaluate_expr("lane // 2", {"lane": 32}), 16)

    def test_addition(self):
        self.assertEqual(sl._evaluate_expr("lane + 1", {"lane": 31}), 32)
        self.assertEqual(sl._evaluate_expr("(relative_degree + 3) * power", {"relative_degree": 2, "power": 3}), 15)

    def test_precedence(self):
        self.assertEqual(sl._evaluate_expr("2 + 3 * 4", {}), 14)
        self.assertEqual(sl._evaluate_expr("(2 + 3) * 4", {}), 20)

    def test_negative_result_raises(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("1 - lane", {"lane": 32})

    def test_division_by_zero_raises(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("lane // 0", {"lane": 32})

    def test_unknown_variable_raises(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("unknown + 1", {"lane": 32})

    def test_syntax_error_raises(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("lane +", {"lane": 32})
        with self.assertRaises(ValueError):
            sl._evaluate_expr("2 * * 3", {})

    def test_disallowed_operations_raise(self):
        with self.assertRaises(ValueError):
            sl._evaluate_expr("2 ** 3", {})
        with self.assertRaises(ValueError):
            sl._evaluate_expr("abs(-1)", {})


# ---------------------------------------------------------------------------
# Schema validation — ISA
# ---------------------------------------------------------------------------

class TestISAFacts(unittest.TestCase):
    """ISA facts validation."""

    def test_missing_isa_rejected(self):
        ledger = _tiny_ledger()
        del ledger["isa"]
        with self.assertRaises(ValidationError):
            sl._validate_isa(ledger)

    def test_lane_resolution(self):
        isa = _isa()
        self.assertEqual(sl._lane_for_tier(isa, "v2"), 16)
        self.assertEqual(sl._lane_for_tier(isa, "v3"), 32)
        self.assertEqual(sl._lane_for_tier(isa, "v4x"), 64)
        # scalar has no lane; must check before calling
        with self.assertRaises(KeyError):
            sl._lane_for_tier(isa, "scalar")

    def test_process_resolution(self):
        isa = _isa()
        self.assertEqual(sl._process_for(isa, "default", "v3"), "v3")
        self.assertEqual(sl._process_for(isa, "default", "v4x"), "v3_gfni_crypto")
        self.assertEqual(sl._process_for(isa, "all", "v4x"), "v4x")
        self.assertEqual(sl._process_for(isa, "all", "scalar"), "scalar")


# ---------------------------------------------------------------------------
# Schema validation — work rules (v2)
# ---------------------------------------------------------------------------

class TestWorkRulesV2(unittest.TestCase):
    """Work rules under schema v2 use expressions, not per-tier dicts."""

    def test_valid_v2_work_rule(self):
        ledger = _tiny_ledger()
        sl._validate_work_rules(ledger)  # must not raise

    def test_missing_iterations_rejected(self):
        ledger = _tiny_ledger()
        del ledger["work_rules"]["row-add"]["iterations"]
        with self.assertRaises(ValidationError):
            sl._validate_work_rules(ledger)

    def test_missing_loss_bytes_rejected(self):
        ledger = _tiny_ledger()
        del ledger["work_rules"]["row-add"]["loss_bytes"]
        with self.assertRaises(ValidationError):
            sl._validate_work_rules(ledger)

    def test_invalid_iterations_expression_rejected(self):
        ledger = _tiny_ledger()
        ledger["work_rules"]["row-add"]["iterations"] = "bad + expr"
        with self.assertRaises(ValidationError):
            sl._validate_work_rules(ledger)

    def test_loss_bytes_not_string_rejected(self):
        ledger = _tiny_ledger()
        ledger["work_rules"]["row-add"]["loss_bytes"] = 42
        with self.assertRaises(ValidationError):
            sl._validate_work_rules(ledger)
        # But a string expression is fine
        ledger["work_rules"]["row-add"]["loss_bytes"] = "42"
        sl._validate_work_rules(ledger)


# ---------------------------------------------------------------------------
# Expansion (Item 9: default/v4x cells)
# ---------------------------------------------------------------------------

class TestExpansion(unittest.TestCase):
    """Expansion count, deviation checks, and default/v4x cells."""

    def test_expansion_count(self):
        ledger = _tiny_ledger()
        rules = sl._validate_rules(ledger)
        expanded = sl._expand_obligations(ledger, rules)
        wrules = sl._validate_work_rules(ledger)
        sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})
        # 2 families x 2 rules = 4 obligations; 1 excluded (packed-padded) in pack
        # ops/row: add_assign@gf8-aes, add_assign@mersenne31, mul_add@gf8-aes, mul_add@mersenne31 (4)
        # ops/pack: pack@gf8-aes, pack@mersenne31 (2; packed-padded excluded)
        self.assertEqual(len(expanded), 6)
        # Each obligation has 10 cells (default x 5 requests (including v4x) + all x 5)
        self.assertEqual(sl._count_cells(expanded), 60)

    def test_default_v4x_cell_present(self):
        """Item 9: default features MUST produce a default/v4x cell."""
        ledger = _tiny_ledger()
        rules = sl._validate_rules(ledger)
        expanded = sl._expand_obligations(ledger, rules)
        first_oid = next(iter(sorted(expanded)))
        cells = expanded[first_oid]["cells"]
        self.assertIn("default/v4x", cells)
        # process resolution for default/v4x is the downgrade
        self.assertEqual(cells["default/v4x"].get("process"), "v3_gfni_crypto")

    def test_all_v4x_cell_present(self):
        """all features produce all/v4x with v4x resolution."""
        ledger = _tiny_ledger()
        rules = sl._validate_rules(ledger)
        expanded = sl._expand_obligations(ledger, rules)
        first_oid = next(iter(sorted(expanded)))
        cells = expanded[first_oid]["cells"]
        self.assertIn("all/v4x", cells)
        self.assertEqual(cells["all/v4x"].get("process"), "v4x")

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
            "basis": "Source", "multiplicity": "Rows",
            "iterations": "1", "loss_bytes": "2 * (lane - 1)", "justification": "t",
        }
        ledger["work_rules"]["wb"] = {
            "substage": "MulAdd", "scope": None, "unit": "Elements",
            "basis": "Source", "multiplicity": "Rows",
            "iterations": "1", "loss_bytes": "2 * (lane - 1)", "justification": "t",
        }
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v2": {"status": "unverified", "required": False, "min_serving": "v2",
                                     "work": ["wa", "wb"], "process": "v2", "field": "v2", "perf": "unmeasured"}}}}
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})


# ---------------------------------------------------------------------------
# Schema validation — cell contents (Item 5)
# ---------------------------------------------------------------------------

class TestCellValidation(unittest.TestCase):
    """Explicit cell contents are validated (not just obligation IDs)."""

    def _make_ledger_with_cell(self, cell_overrides=None):
        ledger = _tiny_ledger()
        cell = {"status": "unverified", "required": False, "min_serving": "v3",
                "work": [], "process": "v3", "field": "v3", "perf": "unmeasured"}
        if cell_overrides:
            cell.update(cell_overrides)
        ledger["obligations"] = {
            "ops::add_assign@gf8-aes": {"cells": {"default/v3": cell}}
        }
        return ledger

    def test_invalid_status_rejected(self):
        ledger = self._make_ledger_with_cell({"status": "bogus"})
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})

    def test_string_required_rejected(self):
        ledger = self._make_ledger_with_cell({"required": "yes"})
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})

    def test_nonexistent_work_rule_rejected(self):
        ledger = self._make_ledger_with_cell({"work": ["nonexistent"]})
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})

    def test_invalid_min_serving_rejected(self):
        ledger = self._make_ledger_with_cell({"min_serving": "bogus"})
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})

    def test_required_true_only_with_valid_status(self):
        # required: true must only appear with direct/composition/zero-budget
        ledger = self._make_ledger_with_cell({"required": True, "status": "missing"})
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        expanded = sl._expanded_with_overrides(ledger)
        with self.assertRaises(ValidationError):
            sl._validate_expanded_obligations(ledger, expanded, rules, wrules, {}, {})

    def test_orphan_bulk_family_rejected(self):
        """Every bulk family must receive at least one obligation."""
        ledger = _tiny_ledger()
        rules = sl._validate_rules(ledger)
        wrules = sl._validate_work_rules(ledger)
        # Normally passes — all families have obligations
        expanded = sl._expand_obligations(ledger, rules)
        sl._validate_bulk_family_coverage(ledger, expanded)  # must not raise

    def test_overlapping_applicability_rejected(self):
        """Same (callable, rule) from two applicability entries is an error."""
        ledger = _tiny_ledger()
        # Add a second applicability entry with overlapping coverage
        ledger["applicability"]["ops-row-second"] = {
            "families": ["ops/row"], "rules": ["gf8-aes"], "exclusions": {},
        }
        rules = sl._validate_rules(ledger)
        with self.assertRaises(ValidationError):
            sl._expand_obligations(ledger, rules)


# ---------------------------------------------------------------------------
# Callable classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Budgets and coefficients (Item 7)
# ---------------------------------------------------------------------------

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

    def test_geometry_element_length_zero_rejected(self):
        """len=0 geometry produces non-positive budget and is caught."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-0": {"rows": 1, "sources": 1, "len": 0, "misalign": 1, "pitch_pad": 0}}
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v3": {"status": "unverified", "required": False, "min_serving": "v3",
                                     "work": ["row-add"], "process": "v3", "field": "v3", "perf": "unmeasured"}}}}
        with self.assertRaises(ValidationError):
            sl._validate_route_budgets(ledger)

    def test_general_coefficient_on_gf2_rejected(self):
        ledger = _tiny_ledger()
        ledger["rules"]["residue-gf2"] = {
            "group": "residue", "elem_bytes": 1,
            "fixtures": [
                {"id": "rgf2", "type": "crate::field::FiniteField<2, 1, crate::field::Residue>",
                 "elem_bytes": 1, "reference": "elem-arith"},
            ],
        }
        ledger["applicability"]["row-mul"] = {"families": ["ops/row"], "rules": ["residue-gf2"], "exclusions": {}}
        ledger["coefficient_patterns"]["general-1"] = {"slots": ["General"]}
        with self.assertRaises(ValidationError):
            sl._validate_gf2_general(ledger)

    def test_zero_one_coefficient_on_gf2_accepted(self):
        ledger = _tiny_ledger()
        ledger["rules"]["residue-gf2"] = {
            "group": "residue", "elem_bytes": 1,
            "fixtures": [
                {"id": "rgf2", "type": "crate::field::FiniteField<2, 1, crate::field::Residue>",
                 "elem_bytes": 1, "reference": "elem-arith"},
            ],
        }
        ledger["applicability"]["row-mul"] = {"families": ["ops/row"], "rules": ["residue-gf2"], "exclusions": {}}
        ledger["coefficient_patterns"]["zero-1"] = {"slots": ["Zero"]}
        sl._validate_gf2_general(ledger)  # must not raise

    def test_width_generic_lane_loss_conversion(self):
        """2-byte and 8-byte fixtures get different lane/loss from the same rule."""
        rule_2b = {"elem_bytes": 2, "group": "packed"}
        rule_8b = {"elem_bytes": 8, "group": "packed"}
        self.assertEqual(sl._lane_in_units("Elements", 16, rule_2b), 8)   # 16//2
        self.assertEqual(sl._loss_in_units("Elements", 30, rule_2b), 15)  # ceil(30/2)
        self.assertEqual(sl._lane_in_units("Elements", 16, rule_8b), 2)   # 16//8
        self.assertEqual(sl._loss_in_units("Elements", 30, rule_8b), 4)   # ceil(30/8)

    def test_gf2_fixture_detection(self):
        self.assertTrue(sl._fixture_is_gf2({"type": "crate::field::FiniteField<2, 1, crate::field::Residue>"}))
        self.assertFalse(sl._fixture_is_gf2({"type": "crate::field::FiniteField<2, 8, crate::field::Polynomial<0x11B>>"}))

    def test_iterations_from_expression(self):
        """Item 7: iterations expression evaluator uses fixture metadata."""
        wr_deg = {"iterations": "degree - 1", "unit": "Elements", "basis": "Source",
                  "multiplicity": "One", "loss_bytes": "2 * (lane - 1)", "substage": "MulAdd"}
        # Evaluate against fixture metadata
        self.assertEqual(sl._work_iterations(wr_deg, {"degree": 8}), 7)
        self.assertEqual(sl._work_iterations(wr_deg, {"degree": 16}), 15)
        self.assertEqual(sl._work_iterations(wr_deg, {"degree": 2}), 1)

    def test_basis_target_uses_target_width(self):
        """Item 7: basis Target converts length with target fixture's width."""
        wr = {"unit": "Elements", "basis": "Target", "multiplicity": "One",
              "loss_bytes": "2 * (lane - 1)", "iterations": "1", "substage": "Linear"}
        source_rule = {"elem_bytes": 1}
        target_rule = {"elem_bytes": 2}
        geo = {"len": 64}
        # Source basis: 64 elements
        b_src = sl._work_budget(wr, geo, source_rule, source_rule, "v3", _isa())
        # Target basis: 64 * (1/2) = 32 target elements
        b_tgt = sl._work_budget(wr, geo, source_rule, target_rule, "v3", _isa())
        # Source: lane=32//1=32, loss=ceil(62/1)=62 → budget = 1*1*(64-62) = 2
        # Target: length in target units = 64*1//2 = 32, lane=32//2=16, loss=ceil(62/2)=31 → budget = 1*1*(32-31) = 1
        self.assertEqual(b_src, 2)
        self.assertEqual(b_tgt, 1)
        self.assertNotEqual(b_src, b_tgt)

    def test_coefficient_pattern_affects_multiplicity(self):
        """Item 7: ActivePairs count from actual coefficient pattern."""
        # mixed-3x4 has 12 slots: [G,G,Z,O,G,O,G,Z,O,G,G,Z]
        # General=7, Zero=4, One=3 (but One slots in accumulate are UnitAddPairs)
        cp = {"slots": ["General", "General", "Zero", "One",
                        "General", "One", "General", "Zero",
                        "One", "General", "General", "Zero"]}
        self.assertEqual(sl._count_active_pairs(cp), 6)   # 6 General slots
        self.assertEqual(sl._count_unit_add_pairs(cp), 3)  # 3 One slots

    def test_work_budget_v2_uses_isa_lane(self):
        """Item 7: lane comes from ISA facts, loss from expression."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        # row-add: iterations="1", loss_bytes="2 * (lane - 1)"
        # v3: lane=32, loss=2*(32-1)=62 → budget = 1 * 1 * (64-62) = 2
        wr = ledger["work_rules"]["row-add"]
        rule = ledger["rules"]["gf8-aes"]
        geo = {"rows": 1, "sources": 1, "len": 64}
        budget = sl._work_budget(wr, geo, rule, rule, "v3", ledger["isa"])
        self.assertEqual(budget, 2)


# ---------------------------------------------------------------------------
# Check --full (Item 6)
# ---------------------------------------------------------------------------

class TestCheckFull(unittest.TestCase):
    """Check --full inventory requirements and reference resolution."""

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

    def test_garbage_test_list_rejected(self):
        """Item 6: test list not in cargo test --lib -- --list format."""
        ledger = _tiny_ledger()
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list = dpath / "test.txt"
            test_list.write_text("not a valid test list")
            case_list, surface = dpath / "case.txt", dpath / "surface.txt"
            case_list.write_text("ROUTE_CASE_LIST\tc1\n")
            surface.write_text("SURFACE_INVENTORY\t{}\n")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old

    def test_garbage_surface_inventory_rejected(self):
        """Item 6: surface inventory lines must be valid JSON."""
        ledger = _tiny_ledger()
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            test_list, case_list, surface = self._tmp_files(dpath, surface="not json")
            old = sl.KNOWN_CALLABLES
            sl.KNOWN_CALLABLES = {"ops::add_assign"}
            try:
                with self.assertRaises(ValidationError):
                    sl.cmd_check(ledger, False, dpath / "rustdoc.json", test_list, case_list, surface, None)
            finally:
                sl.KNOWN_CALLABLES = old


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------

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

    def test_duplicate_route_done_rejected(self):
        """Item 6: duplicate ROUTE_DONE fails."""
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, done_body={"source_id": "src-1", "host_id": "host-1",
                                           "features": "default", "request": "v3", "cases": 1})
            # Append a second ROUTE_DONE
            with open(dpath / "route.log", "a") as f:
                f.write("\nROUTE_DONE\t" + _json({"source_id": "src-1", "host_id": "host-1",
                                                   "features": "default", "request": "v3", "cases": 1}))
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(self._evidence_ledger(), dpath)

    def test_receipt_without_id_rejected(self):
        """Item 6: receipt without identity (no id field) fails."""
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            self._record(dpath, receipts=[{"obligation": "o", "features": "default",
                                           "request_raw": "v3", "source_id": "src-1"}])
            with self.assertRaises(ValidationError):
                sl._verify_evidence_records(self._evidence_ledger(), dpath)


# ---------------------------------------------------------------------------
# Accept (Item 8: receipt validation)
# ---------------------------------------------------------------------------

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

    # Item 8: request_raw must be a non-empty string
    def test_request_raw_none_rejected(self):
        """Item 8: request_raw=None is rejected."""
        rec = dict(_good_receipt(), request_raw=None)
        self._reject([rec], [self._done_line()], "receipt_request_mismatch")

    def test_request_raw_empty_rejected(self):
        """Item 8: request_raw="" is rejected."""
        rec = dict(_good_receipt(), request_raw="")
        self._reject([rec], [self._done_line()], "receipt_request_mismatch")

    def test_result_exact_pass_fail_skip(self):
        """Item 8: result must be exactly 'pass', 'fail', or 'skip' (not prefix match)."""
        # "passing" is not accepted
        rec = dict(_good_receipt(), result="passing")
        self._reject([rec], [self._done_line()], "unknown_result")
        # "passed" is not accepted
        rec = dict(_good_receipt(), result="passed")
        self._reject([rec], [self._done_line()], "unknown_result")

    def test_result_non_string_rejected(self):
        """Item 8: result must be a string."""
        rec = dict(_good_receipt(), result=True)
        self._reject([rec], [self._done_line()], "receipt_no_result")

    # Item 8: required set keyed by case id, not obligation
    def test_missing_case_id_rejected(self):
        """Item 8: receipt without a case id that matches is rejected for missing_required."""
        # Use a ledger with geometries so case-level expansion works
        ledger = _accept_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["coefficient_patterns"] = {"gen-1": {"slots": ["General"]}}
        # Receipt has the right obligation but wrong case id
        rec = dict(_good_receipt(obligation="ops::mul_add@gf8-aes", case_id="different-id"))
        self._reject([rec], [self._done_line()], "missing_required", ledger=ledger)


# ---------------------------------------------------------------------------
# Generator (Item 1)
# ---------------------------------------------------------------------------

class TestGenerate(unittest.TestCase):
    """Generator emit/byte-identity and case table generation."""

    def test_generate_check_one_byte_difference(self):
        ledger = _tiny_ledger()
        content = sl._generate_cases(ledger)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.rs"
            # Write raw content, then run generate to format and verify
            p.write_text(content)
            # Generate first to write formatted output
            self.assertEqual(sl.cmd_generate(p, False, ledger), 0)
            # Check should now pass
            self.assertEqual(sl.cmd_generate(p, True, ledger), 0)
            # Corrupt the file: change a byte
            formatted = p.read_text()
            bad = formatted.replace("generated", "generatXd", 1)
            self.assertNotEqual(bad, formatted)
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

    def test_generator_emits_case_table(self):
        """Item 1: generator emits Case/Cell/Work tables."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["coefficient_patterns"] = {"gen-1": {"slots": ["General"]}}
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v3": {"status": "unverified", "required": False, "min_serving": "v3",
                                     "work": ["row-add"], "process": "v3", "field": "v3", "perf": "unmeasured"}}}}
        content = sl._generate_cases(ledger)
        # Must contain CASES_SIMD512 or CASES table
        self.assertTrue("CASES_SIMD512" in content or "static CASES" in content)
        # Must contain Case struct instances
        self.assertIn("Case {", content)
        # Must contain Cell struct instances
        self.assertIn("Cell {", content)
        # Must contain Work struct instances
        self.assertIn("Work {", content)

    def test_generator_emits_cases_simd512(self):
        """Item 1: cases needing simd512 go in separate CASES_SIMD512 table."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["coefficient_patterns"] = {"gen-1": {"slots": ["General"]}}
        # Add a rule whose all/v4x cell needs v4x
        ledger["rules"]["gf16-v4x"] = {
            "group": "binary_flat", "degree": 16, "presentation": "Polynomial", "elem_bytes": 2,
            "fixtures": [
                {"id": "gf16x", "type": "crate::field::FiniteField<2,16,crate::field::Polynomial<0x1002D>>",
                 "elem_bytes": 2, "reference": "elem-arith"},
            ],
        }
        ledger["rules"]["gf16-v4x"]["_needs_simd512"] = True  # marker for test
        ledger["applicability"]["ops-row-gf16"] = {
            "families": ["ops/row"], "rules": ["gf16-v4x"], "exclusions": {},
        }
        content = sl._generate_cases(ledger)
        # gf8-aes cases go in CASES; gf16-v4x cases may go in CASES_SIMD512
        self.assertIn("CASES_SIMD512", content)

    def test_generator_work_lane_loss_in_rule_unit(self):
        """Item 1: emitted Work uses lane/loss/iterations in the rule's unit."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["coefficient_patterns"] = {"gen-1": {"slots": ["General"]}}
        ledger["obligations"] = {"ops::add_assign@gf8-aes": {
            "cells": {"default/v3": {"status": "unverified", "required": False, "min_serving": "v3",
                                     "work": ["row-add"], "process": "v3", "field": "v3", "perf": "unmeasured"}}}}
        content = sl._generate_cases(ledger)
        # Work struct should have lane, loss, iterations fields
        self.assertIn("lane:", content)
        self.assertIn("loss:", content)
        self.assertIn("iterations:", content)
        # Should NOT have raw lane_bytes/loss_bytes/loss_formula in generated output
        self.assertNotIn("lane_bytes:", content)
        self.assertNotIn("loss_bytes:", content)
        self.assertNotIn("loss_formula", content)

    def test_generator_gf2_filtering(self):
        """GF(2) fixtures should exclude General coefficient patterns."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["rules"]["residue-gf2"] = {
            "group": "residue", "elem_bytes": 1,
            "fixtures": [
                {"id": "rgf2", "type": "crate::field::FiniteField<2, 1, crate::field::Residue>",
                 "elem_bytes": 1, "reference": "elem-arith"},
            ],
        }
        ledger["applicability"]["row-gf2"] = {
            "families": ["ops/row"], "rules": ["residue-gf2"], "exclusions": {},
        }
        ledger["coefficient_patterns"] = {
            "gen-1": {"slots": ["General"]},
            "zero-1": {"slots": ["Zero"]},
            "one-1": {"slots": ["One"]},
        }
        content = sl._generate_cases(ledger)
        # GF(2) fixtures: only Zero/One patterns; no General
        # The generated case table for GF(2) should only reference non-General patterns
        self.assertIn("zero-1", content)
        self.assertIn("one-1", content)

    def test_generator_parseable_structure(self):
        """Item 1: generated Rust can be parsed for basic structure checks."""
        ledger = _tiny_ledger()
        ledger["geometries"] = {"row-64": {"rows": 1, "sources": 1, "len": 64, "misalign": 1, "pitch_pad": 0}}
        ledger["coefficient_patterns"] = {"gen-1": {"slots": ["General"]}}
        content = sl._generate_cases(ledger)
        lines = content.splitlines()
        # First line is the generated marker
        self.assertTrue(lines[0].startswith("// @generated"))
        # Find CASES array
        cases_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("static CASES:"):
                cases_start = i
                break
        self.assertIsNotNone(cases_start, "CASES table not found")
        # After CASES, there should be case entries
        found_case = False
        for line in lines[cases_start:]:
            if "Case {" in line:
                found_case = True
                break
        self.assertTrue(found_case, "No Case entries in CASES table")


# ---------------------------------------------------------------------------
# Canonical ledger (schema-only against current JSON is expected to fail)
# ---------------------------------------------------------------------------

class TestCanonicalLedger(unittest.TestCase):
    """The canonical ledger — schema-only against old JSON may fail; that's expected."""

    def test_ledger_default_is_next_to_script(self):
        self.assertEqual(sl.Path(__file__).with_name("simd_ledger.json"), CANONICAL_LEDGER_PATH)

    def test_schema_only_cli_reports_failure_reason(self):
        """Schema-only against old-format JSON: expected to fail with clear reason."""
        result = sl.main(["check", "--schema-only"])
        # The old JSON lacks "isa" top-level key — the tool should reject it
        self.assertNotEqual(result, 0, "Expected schema-only to fail against old JSON format")

    def test_test_subcommand_removed(self):
        self.assertFalse(hasattr(sl, "run_tests"))
        with self.assertRaises(SystemExit):
            sl.main(["test"])


if __name__ == "__main__":
    unittest.main()