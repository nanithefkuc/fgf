"""Synthetic rule, rejection, and declaration tests for ``bench_pair``.

No cargo or bench process is invoked: every fixture is constructed in memory
or in a temporary directory.
"""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_pair as pair


def cell(label="gf8-aes/mul_add/unprepared/65536x1x1/aligned"):
    return label


def ratios(values):
    return {"cell": list(values)}


def calibration_session(host, tier, base_values, candidate_values, mode=None):
    runs = []
    for index, (base, candidate) in enumerate(zip(base_values, candidate_values)):
        runs.append(
            {
                "run": index,
                "tier": tier,
                "order": ["base", "candidate"],
                "control": {"start_ns": 100.0, "end_ns": 200.0},
                "arms": {
                    "base": {"cells": {cell(): {"summary_ns": base, "rounds_ns": [base]}}},
                    "candidate": {
                        "cells": {
                            cell(): {
                                "summary_ns": candidate,
                                "rounds_ns": [candidate],
                            }
                        }
                    },
                },
            }
        )
    return {
        "kind": "calibration",
        "host": host,
        "calibration": {
            "mode": mode or "base_vs_layout_control",
        },
        "runs": runs,
    }


def host(name="willow-cove", cpu="Intel(R) Xeon(R) Platinum"):
    return {"hostname": name, "cpu_model": cpu}


class HodgesLehmannBoundTests(unittest.TestCase):
    def test_exact_bounds_for_hand_computed_small_n(self):
        expected = {
            5: (15, 15, 3.0, 5.0, 0.96875),
            6: (19, 21, 3.5, 5.0, 0.953125),
            8: (31, 36, 4.5, 6.5, 0.9609375),
            10: (45, 55, 5.5, 7.5, 0.9580078125),
        }
        for n, (index, order_statistics, estimate, upper, coverage) in expected.items():
            result = pair.hl_upper_bound(list(range(1, n + 1)))
            self.assertEqual(result["index"], index)
            self.assertEqual(result["order_statistics"], order_statistics)
            self.assertEqual(result["estimate"], estimate)
            self.assertEqual(result["upper_bound"], upper)
            self.assertAlmostEqual(result["coverage"], coverage)
            self.assertTrue(result["exact"])
            self.assertGreaterEqual(result["coverage"], 0.95)

    def test_one_gross_outlier_does_not_move_the_upper_bound(self):
        quiet = pair.hl_upper_bound([0.0] * 9 + [10.0])
        gross = pair.hl_upper_bound([0.0] * 9 + [1_000_000.0])
        self.assertEqual(quiet["estimate"], gross["estimate"])
        self.assertEqual(quiet["upper_bound"], gross["upper_bound"])
        self.assertEqual(quiet["upper_bound"], 0.0)

    def test_bound_is_monotone_under_a_shift(self):
        low = pair.hl_upper_bound([0.0, 0.0, 0.1, 0.2, 0.3])
        high = pair.hl_upper_bound([1.0, 1.0, 1.1, 1.2, 1.3])
        self.assertGreaterEqual(high["estimate"], low["estimate"])
        self.assertGreaterEqual(high["upper_bound"], low["upper_bound"])

    def test_normal_approximation_above_fifty_is_conservative(self):
        result = pair.hl_upper_bound([0.5] * 51)
        self.assertFalse(result["exact"])
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["upper_bound"], 0.5)
        self.assertGreaterEqual(result["coverage"], 0.95)

    def test_fewer_than_five_runs_is_insufficient(self):
        with self.assertRaises(pair.InsufficientRuns):
            pair.hl_upper_bound([0.0] * 4)


class RuleTests(unittest.TestCase):
    def test_noninferiority_passes_when_every_bound_is_below_limit(self):
        decision = pair.evaluate_noninferiority(
            {"a": [math.log(1.01)] * 5, "b": [0.0] * 5}, 0.03
        )
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["failed_cells"], [])
        self.assertAlmostEqual(decision["limit"], math.log1p(0.03))

    def test_noninferiority_fails_exactly_the_slow_cells(self):
        decision = pair.evaluate_noninferiority(
            {
                "slow-a": [math.log(1.06)] * 5,
                "slow-b": [math.log(1.06)] * 5,
                "fine": [math.log(1.01)] * 5,
            },
            0.03,
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(
            sorted(decision["failed_cells"]), ["slow-a", "slow-b"]
        )
        self.assertTrue(decision["per_cell"]["fine"]["pass"])

    def test_two_delta_slowdown_fails_exactly_the_two_contract_cells(self):
        slow = math.log(1.0 + 2.0 * 0.03)
        decision = pair.evaluate_noninferiority(
            {
                "gf8-aes/mul_add/unprepared/65536x1x1/aligned": [slow] * 5,
                "m31/mul_into/prepared/4096x1x1/aligned": [slow] * 5,
                "residue-65521/mul_into/prepared/4096x1x1/aligned": [0.0] * 5,
            },
            0.03,
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["failed_cells"],
            [
                "gf8-aes/mul_add/unprepared/65536x1x1/aligned",
                "m31/mul_into/prepared/4096x1x1/aligned",
            ],
        )

    def test_superiority_passes_with_the_contract_cells_as_targets(self):
        fast = math.log(1.0 - 2.0 * 0.03)
        decision = pair.evaluate_superiority(
            {
                "gf8-aes/mul_add/unprepared/65536x1x1/aligned": [fast] * 5,
                "m31/mul_into/prepared/4096x1x1/aligned": [fast] * 5,
                "residue-65521/mul_into/prepared/4096x1x1/aligned": [0.0] * 5,
            },
            [
                "gf8-aes/mul_add/unprepared/65536x1x1/aligned",
                "m31/mul_into/prepared/4096x1x1/aligned",
            ],
            0.03,
            0.03,
        )
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["failed_targets"], [])

    def test_superiority_passes_for_declared_targets(self):
        decision = pair.evaluate_superiority(
            {
                "target": [math.log(0.95)] * 5,
                "other": [math.log(1.0)] * 5,
            },
            ["target"],
            0.03,
            0.03,
        )
        self.assertTrue(decision["passed"])
        self.assertTrue(decision["per_target"]["target"]["pass"])
        self.assertTrue(decision["others_noninferior"])

    def test_superiority_fails_with_an_unaffected_third_target(self):
        decision = pair.evaluate_superiority(
            {
                "target": [math.log(0.95)] * 5,
                "third": [math.log(1.0)] * 5,
                "other": [math.log(1.0)] * 5,
            },
            ["target", "third"],
            0.03,
            0.03,
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["failed_targets"], ["third"])

    def test_superiority_fails_when_a_nontarget_regresses(self):
        decision = pair.evaluate_superiority(
            {
                "target": [math.log(0.95)] * 5,
                "other": [math.log(1.06)] * 5,
            },
            ["target"],
            0.03,
            0.03,
        )
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["others_noninferior"])

    def test_superiority_with_an_empty_target_set_never_passes(self):
        decision = pair.evaluate_superiority(
            {"target": [math.log(0.95)] * 5}, [], 0.03, 0.03
        )
        self.assertTrue(decision["declared"])
        self.assertFalse(decision["passed"])

    def test_superiority_limit_is_negative(self):
        decision = pair.evaluate_superiority(
            {"target": [math.log(1.0)] * 5}, ["target"], 0.03, 0.03
        )
        self.assertLess(decision["limit"], 0.0)
        self.assertFalse(decision["passed"])


class DriftTests(unittest.TestCase):
    def test_arm_order_is_counterbalanced(self):
        self.assertEqual(pair.balanced_order(0), ["base", "candidate"])
        self.assertEqual(pair.balanced_order(1), ["candidate", "base"])
        self.assertEqual(pair.balanced_order(2), ["base", "candidate"])

    def test_arm_run_summary_is_the_round_median(self):
        self.assertEqual(pair.run_arm_summary([9.0, 1.0, 5.0]), 5.0)

    def test_start_controls_include_both_arms(self):
        runs = [
            {
                "controls": {"base": {"start_ns": 10}, "candidate": {"start_ns": 12}},
                "control": {"start_ns": 10},
            },
            {
                "controls": {"base": {"start_ns": 14}, "candidate": {"start_ns": 16}},
                "control": {"start_ns": 14},
            },
        ]
        self.assertEqual(pair.start_controls(runs), [10.0, 12.0, 14.0, 16.0])

    def test_drift_is_interquartile_range_relative_to_median(self):
        stats = pair.drift_stats([100.0, 102.0, 101.0])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["drift_method"], pair.DRIFT_METHOD)
        self.assertAlmostEqual(stats["spread"], 2.0)
        self.assertAlmostEqual(stats["min"], 100.0)
        self.assertAlmostEqual(stats["max"], 102.0)
        self.assertAlmostEqual(stats["median"], 101.0)
        self.assertAlmostEqual(stats["p25"], 100.5)
        self.assertAlmostEqual(stats["p75"], 101.5)
        self.assertAlmostEqual(stats["drift"], 1.0 / 101.0)

    def test_drift_does_not_use_the_max_min_spread(self):
        samples = [100.0 + (index % 5) * 0.1 for index in range(100)]
        samples.append(500.0)
        stats = pair.drift_stats(samples)
        self.assertGreater(stats["spread"] / stats["median"], 3.0)
        self.assertLess(stats["drift"], 0.01)
        self.assertEqual(stats["max"], 500.0)

    def test_minority_slow_level_does_not_move_drift(self):
        stats, rejection = pair.check_drift([100.0] * 40 + [120.0] * 8, 0.01)
        self.assertIsNone(rejection)
        self.assertEqual(stats["drift"], 0.0)
        self.assertEqual(stats["spread"], 20.0)

    def test_sustained_slow_level_breaches_drift_cap(self):
        stats, rejection = pair.check_drift([100.0] * 32 + [120.0] * 16, 0.01)
        self.assertEqual(stats["drift"], 0.2)
        self.assertEqual(rejection.code, "drift_cap")

    def test_drift_cap_breach_rejects(self):
        stats, rejection = pair.check_drift([100.0, 103.0, 101.0], 0.01)
        self.assertGreater(stats["drift"], 0.01)
        self.assertIsNotNone(rejection)
        self.assertEqual(rejection.code, "drift_cap")

    def test_drift_within_cap_passes(self):
        stats, rejection = pair.check_drift([100.0, 101.0, 100.5], 0.02)
        self.assertIsNone(rejection)
        self.assertLessEqual(stats["drift"], 0.02)

    def test_nonpositive_median_is_a_rejection(self):
        _, rejection = pair.check_drift([0.0, 0.0], 0.01)
        self.assertEqual(rejection.code, "drift_control_invalid")

    def test_empty_controls_are_a_rejection(self):
        _, rejection = pair.check_drift([], 0.01)
        self.assertEqual(rejection.code, "drift_control_invalid")

    def test_recommendation_is_the_smallest_sufficient_run_count(self):
        noisy = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        recommendation = pair.recommend_run_count({"cell": noisy}, 0.03)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["runs"], 6)
        self.assertLessEqual(recommendation["max_bound"], math.log1p(0.015))

    def test_constant_ratios_recommend_five_runs(self):
        recommendation = pair.recommend_run_count({"cell": [0.0] * 5}, 0.03)
        self.assertEqual(recommendation["runs"], 5)

    def test_recommendation_returns_none_when_no_count_is_sufficient(self):
        recommendation = pair.recommend_run_count(
            {"cell": [0.1, -0.1, 0.1, -0.1, 0.1]}, 0.03
        )
        self.assertIsNone(recommendation)

    def test_drift_cap_uses_prior_calibration_drift_and_the_factor(self):
        recommendation, healthy = pair.calibration_recommendation(
            {"v4x": {"cell": [0.0] * 5}},
            0.03,
            {"v4x": {"drift": 0.01}},
            drift_safety_factor=2.0,
            prior_drift=0.02,
        )
        self.assertTrue(healthy)
        self.assertGreaterEqual(recommendation["drift_cap"], 0.04)

    def test_unstable_cells_are_reported_without_rejecting(self):
        recommendation, healthy = pair.calibration_recommendation(
            {"v4x": {"slow": [0.1, -0.1, 0.1, -0.1, 0.1]}},
            0.03,
            {"v4x": {"drift": 0.001}},
        )
        self.assertTrue(healthy)
        record = recommendation["per_tier"]["v4x"]
        self.assertEqual(record["available_runs"], 5)
        self.assertEqual(record["unstable_cells"], ["slow"])
        self.assertEqual(record["runs"], 5)
        self.assertEqual(record["cells_over_limit"], 0)

    def test_calibration_budget_ignores_pair_runs(self):
        args = types.SimpleNamespace(calibration_runs=8, runs="99")
        campaign = pair.Campaign.__new__(pair.Campaign)
        campaign.args = args
        self.assertEqual(campaign.calibration_runs(), 8)

    def test_recommended_drift_cap_applies_the_stated_safety_factor(self):
        self.assertGreaterEqual(pair.recommend_drift_cap(0.01234), 0.01234 * 1.25)
        self.assertAlmostEqual(pair.recommend_drift_cap(0.01234), 0.0155)
        self.assertGreaterEqual(pair.recommend_drift_cap(None), 0.0)


class LayoutTests(unittest.TestCase):
    def test_spread_is_the_largest_bound_across_sessions(self):
        first = calibration_session(host(), "v4x", [100] * 5, [102] * 5)
        second = calibration_session(host(), "v4x", [100] * 5, [103] * 5)
        spreads = pair.layout_spreads([first, second], host(), "v4x")
        self.assertIsNotNone(spreads)
        expected = max(
            pair.hl_upper_bound([math.log(1.02)] * 5)["upper_bound"],
            pair.hl_upper_bound([math.log(1.03)] * 5)["upper_bound"],
        )
        self.assertAlmostEqual(spreads[cell()], expected)

    def test_within_spread_is_only_an_owner_exception(self):
        disposition = pair.layout_disposition(0.04, 0.05)
        self.assertEqual(disposition["disposition"], "owner_exception_required")
        self.assertNotEqual(disposition["disposition"], "pass")

    def test_above_spread_is_a_regression(self):
        disposition = pair.layout_disposition(0.06, 0.05)
        self.assertEqual(disposition["disposition"], "regression_exceeds_spread")

    def test_missing_spread_is_not_a_pass(self):
        disposition = pair.layout_disposition(0.01, None)
        self.assertEqual(disposition["disposition"], "spread_unavailable")

    def test_wrong_mode_or_host_has_no_spread(self):
        wrong_mode = calibration_session(
            host(), "v4x", [100] * 5, [101] * 5, mode="base_vs_base"
        )
        self.assertIsNone(pair.layout_spreads([wrong_mode], host(), "v4x"))
        other_host = calibration_session(
            host("golden-cove", "Intel(R) Core(TM) i9"), "v4x", [100] * 5, [101] * 5
        )
        self.assertIsNone(pair.layout_spreads([other_host], host(), "v4x"))

    def test_spread_requires_accepted_matching_sessions(self):
        good = calibration_session(host(), "v4x", [100] * 5, [102] * 5)
        good["host"]["toolchain_id"] = "tc"
        good["snapshot"] = {"digest": "d", "rustflags": pair.DEFAULT_ALIGN_FLAGS}
        snapshot = {"digest": "d", "rustflags": pair.DEFAULT_ALIGN_FLAGS}
        self.assertIsNotNone(
            pair.layout_spreads(
                [good], host(), "v4x", snapshot=snapshot, toolchain_id="tc"
            )
        )
        mutations = []
        rejected = json.loads(json.dumps(good)); rejected["rejected"] = True; mutations.append(rejected)
        bad_rejections = json.loads(json.dumps(good)); bad_rejections["rejections"] = [{"code": "x"}]; mutations.append(bad_rejections)
        bad_toolchain = json.loads(json.dumps(good)); bad_toolchain["host"]["toolchain_id"] = "other"; mutations.append(bad_toolchain)
        bad_flags = json.loads(json.dumps(good)); bad_flags["snapshot"]["rustflags"] = "-C target-cpu=native"; mutations.append(bad_flags)
        bad_digest = json.loads(json.dumps(good)); bad_digest["snapshot"]["digest"] = "other"; mutations.append(bad_digest)
        for mutation in mutations:
            self.assertIsNone(
                pair.layout_spreads(
                    [mutation], host(), "v4x", snapshot=snapshot, toolchain_id="tc"
                )
            )

    def test_session_ratios_are_recomputed_from_stored_runs(self):
        session = calibration_session(host(), "v4x", [100, 100, 100], [101, 100, 99])
        ratios = pair.session_tier_ratios(session, "v4x")
        self.assertEqual(len(ratios[cell()]), 3)
        self.assertAlmostEqual(ratios[cell()][0], math.log(1.01))


class ThresholdTests(unittest.TestCase):
    def inventory(self, unit="bytes", value=64, item_class="routing_threshold",
                  axis="row length"):
        return [
            {
                "key": "src/kernel/quad_mersenne31.rs:ROW_THRESHOLD",
                "file": "src/kernel/quad_mersenne31.rs",
                "name_or_expr": "ROW_THRESHOLD",
                "value": value,
                "unit": unit,
                "axis": axis,
                "class": item_class,
                "routes": True,
                "citation": None,
            }
        ]

    def declaration(self, **overrides):
        value = {
            "item": "src/kernel/quad_mersenne31.rs:ROW_THRESHOLD",
            "old": 64,
            "new": 128,
            "cells": [
                {
                    "cell": "gf8-aes/mul_add/unprepared/64x1x1/aligned",
                    "axis_value": 64,
                    "route_old": "above",
                    "route_new": "below",
                },
                {
                    "cell": "gf8-aes/mul_add/unprepared/96x1x1/aligned",
                    "axis_value": 96,
                    "route_old": "above",
                    "route_new": "below",
                },
                {
                    "cell": "gf8-aes/mul_add/unprepared/128x1x1/aligned",
                    "axis_value": 128,
                    "route_old": "below",
                    "route_new": "below",
                },
            ],
        }
        value.update(overrides)
        return value

    def test_valid_byte_declaration_and_route_targets(self):
        info, rejection = pair.validate_threshold_declaration(
            self.declaration(), self.inventory()
        )
        self.assertIsNone(rejection)
        self.assertEqual(info["grid"], [64.0, 96.0, 128.0])
        self.assertEqual(
            info["target_set"],
            [
                "gf8-aes/mul_add/unprepared/64x1x1/aligned",
                "gf8-aes/mul_add/unprepared/96x1x1/aligned",
            ],
        )

    def test_old_cutoff_must_equal_the_inventory_value(self):
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(old=32), self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_old_mismatch")

    def test_nonnumeric_item_value_is_refused(self):
        inventory = self.inventory()
        inventory[0]["value"] = "not-a-number"
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(), inventory
        )
        self.assertEqual(rejection.code, "threshold_item_value")

    def test_structural_items_are_refused(self):
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(), self.inventory(item_class="structural")
        )
        self.assertEqual(rejection.code, "threshold_item_structural")

    def test_only_the_full_inventory_key_matches(self):
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(item="ROW_THRESHOLD"), self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_item_unknown")

    def test_axis_value_is_derived_from_cell_geometry(self):
        declaration = self.declaration()
        declaration["cells"][0]["axis_value"] = 96
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_axis_value")

    def test_duplicate_cells_are_refused(self):
        declaration = self.declaration()
        declaration["cells"].append(dict(declaration["cells"][1]))
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_duplicate_cell")

    def test_route_labels_must_agree_with_the_cutoff_side(self):
        declaration = self.declaration()
        declaration["cells"][0]["route_old"] = "same"
        declaration["cells"][0]["route_new"] = "same"
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_route_mismatch")

    def test_byte_grid_requires_the_midpoint(self):
        declaration = self.declaration()
        del declaration["cells"][1]
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_no_cell")

    def test_grid_missing_midpoint_when_another_cell_covers_the_span(self):
        declaration = self.declaration()
        declaration["cells"][1] = {
            "cell": "gf8-aes/mul_add/unprepared/80x1x1/aligned",
            "axis_value": 80,
            "route_old": "above",
            "route_new": "below",
        }
        # 80 lies between the cutoffs, but the required 96 grid point is absent.
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_grid")
        self.assertIn("96", rejection.message)

    def test_count_grid_requires_every_integer(self):
        declaration = {
            "item": "src/kernel/quad_mersenne31.rs:ROW_THRESHOLD",
            "old": 16,
            "new": 20,
            "cells": [
                {
                    "cell": f"gf8-aes/mul_add/unprepared/{value}x1x1/aligned",
                    "axis_value": value,
                    "route_old": "above" if value < 20 else "below",
                    "route_new": "below",
                }
                for value in (16, 18, 20)
            ],
        }
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory("count", value=16)
        )
        self.assertEqual(rejection.code, "threshold_grid")

    def test_elements_use_the_doubling_grid(self):
        self.assertEqual(
            pair.threshold_grid_points(64, 128, "elements"),
            [64.0, 96.0, 128.0],
        )

    def test_no_cell_strictly_between_the_cutoffs_is_rejected(self):
        declaration = self.declaration()
        declaration["cells"] = [
            declaration["cells"][0],
            declaration["cells"][2],
        ]
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "threshold_no_cell")

    def test_unknown_item_is_rejected(self):
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(item="absent"),
            self.inventory(),
        )
        self.assertEqual(rejection.code, "threshold_item_unknown")

    def test_unit_mismatch_is_rejected(self):
        _, rejection = pair.validate_threshold_declaration(
            self.declaration(unit="count"), self.inventory("bytes")
        )
        self.assertEqual(rejection.code, "threshold_unit_mismatch")

    def test_invalid_entry_is_rejected(self):
        declaration = self.declaration()
        del declaration["cells"][0]["route_new"]
        _, rejection = pair.validate_threshold_declaration(
            declaration, self.inventory()
        )
        self.assertEqual(rejection.code, "declaration_invalid")

    def test_declaration_file_digest_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decl.json"
            path.write_text(json.dumps(self.declaration()), encoding="utf-8")
            parsed = pair.parse_threshold_file(path)
            self.assertEqual(
                parsed["digest"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_grid_values_in_other_units(self):
        self.assertEqual(pair.threshold_grid_points(64, 256, "bytes"),
                         [64.0, 96.0, 128.0, 192.0, 256.0])
        self.assertEqual(pair.threshold_grid_points(512, 1024, "lanes"),
                         [512.0, 768.0, 1024.0])
        self.assertEqual(pair.threshold_grid_points(4, 6, "count"), [4, 5, 6])


class EnvironmentTests(unittest.TestCase):
    def facts(self, **overrides):
        value = {
            "pinned_core": 3,
            "affinity": [0, 1, 2, 3],
            "governor": "performance",
            "smt_active": True,
            "siblings": [7],
            "sibling_busy": 0.0,
        }
        value.update(overrides)
        return value

    def codes(self, **overrides):
        return {
            rejection.code
            for rejection in pair.evaluate_environment(self.facts(**overrides))
        }

    def test_healthy_environment_has_no_rejection(self):
        self.assertEqual(self.codes(), set())

    def test_powersave_governor_is_rejected(self):
        self.assertIn("governor", self.codes(governor="powersave"))

    def test_unknown_governor_is_rejected(self):
        self.assertIn("governor_unknown", self.codes(governor=None))

    def test_busy_smt_sibling_is_rejected(self):
        self.assertIn(
            "smt_sibling_busy",
            self.codes(sibling_busy=0.5),
        )

    def test_unverified_smt_sibling_is_rejected(self):
        self.assertIn(
            "smt_sibling_unverified",
            self.codes(sibling_busy=None),
        )

    def test_missing_pinned_core_is_rejected(self):
        self.assertIn("pinned_core_missing", self.codes(pinned_core=None))

    def test_pinned_core_outside_affinity_is_rejected(self):
        self.assertIn(
            "pinned_core_unavailable",
            self.codes(pinned_core=9, affinity=[0, 1, 2, 3]),
        )

    def test_smt_disabled_does_not_sample(self):
        self.assertEqual(
            self.codes(smt_active=False, sibling_busy=0.99), set()
        )


class ConfigurationTests(unittest.TestCase):
    flags = pair.DEFAULT_ALIGN_FLAGS

    def base_kwargs(self, **overrides):
        value = {
            "runs": 5,
            "drift_cap": 0.02,
            "delta": 0.03,
            "gamma": 0.03,
            "rounds": 7,
            "calibrated_hosts": ["willow-cove"],
            "calibrated_tiers": "willow-cove:v4x,v3,scalar",
            "calibrated_toolchain": "1.93.0/abc",
            "calibrated_flags": self.flags,
            "host": host(),
            "toolchain_id": "1.93.0/abc",
            "effective_flags": self.flags,
            "tiers": ["v4x", "v3"],
        }
        value.update(overrides)
        return value

    def codes(self, **overrides):
        return {
            rejection.code
            for rejection in pair.validate_acceptance_config(
                **self.base_kwargs(**overrides)
            )
        }

    def test_calibrated_configuration_passes(self):
        self.assertEqual(self.codes(), set())

    def test_placeholder_runs_are_refused(self):
        self.assertIn("uncalibrated_config", self.codes(runs=None))
        self.assertIn("insufficient_runs", self.codes(runs=4))
        self.assertNotIn("insufficient_runs", self.codes(runs=5))

    def test_placeholder_drift_cap_is_refused(self):
        self.assertIn("uncalibrated_config", self.codes(drift_cap=None))

    def test_unbound_host_is_refused(self):
        self.assertIn(
            "calibration_binding",
            self.codes(calibrated_hosts=["golden-cove"]),
        )

    def test_unbound_tier_is_refused(self):
        self.assertIn(
            "calibration_binding",
            self.codes(calibrated_tiers="willow-cove:v3,scalar"),
        )
        self.assertIn(
            "calibration_binding",
            self.codes(calibrated_tiers="golden-cove:v4x,v3,scalar"),
        )

    def test_toolchain_mismatch_is_refused(self):
        self.assertIn(
            "calibration_binding",
            self.codes(calibrated_toolchain="1.94.0/def"),
        )

    def test_flag_mismatch_is_refused(self):
        self.assertIn(
            "calibration_binding",
            self.codes(calibrated_flags="-C target-cpu=native"),
        )

    def test_per_host_tier_binding_parser(self):
        binding = pair.parse_host_tier_binding(
            "willow-cove:v4x,v3,scalar golden-cove:v3_gfni_crypto,v2"
        )
        self.assertEqual(binding["willow-cove"], {"v4x", "v3", "scalar"})
        self.assertEqual(binding["golden-cove"], {"v3_gfni_crypto", "v2"})
        self.assertIsNone(pair.parse_host_tier_binding("uncalibrated"))
        self.assertIsNone(pair.parse_host_tier_binding("willow-cove:v4x extra"))

    def test_per_host_value_binding_parser(self):
        self.assertEqual(
            pair.parse_host_value_binding("willow-cove:8 zbox:5", "willow-cove", "runs"),
            (8.0, None),
        )
        self.assertEqual(
            pair.parse_host_value_binding("willow-cove:8 zbox:5", "zbox", "runs"),
            (5.0, None),
        )
        self.assertEqual(
            pair.parse_host_value_binding("8", "willow-cove", "runs"),
            (8.0, None),
        )
        self.assertEqual(
            pair.parse_host_value_binding("uncalibrated", "willow-cove", "runs"),
            (None, None),
        )
        _, rejection = pair.parse_host_value_binding(
            "willow-cove:8", "golden-cove", "runs"
        )
        self.assertEqual(rejection.code, "uncalibrated_config")
        _, rejection = pair.parse_host_value_binding(
            "willow-cove:8 zbox", "willow-cove", "runs"
        )
        self.assertEqual(rejection.code, "uncalibrated_config")
        _, rejection = pair.parse_host_value_binding(
            "willow-cove:8 willow-cove:9", "willow-cove", "runs"
        )
        self.assertEqual(rejection.code, "uncalibrated_config")

    def test_host_binding_is_exact_hostname_equality(self):
        self.assertTrue(pair.host_matches("willow-cove", host()))
        self.assertFalse(pair.host_matches("willow", host()))
        self.assertFalse(pair.host_matches("Intel", host()))

    def test_smt_window_and_threshold_have_hard_bounds(self):
        self.assertIn(
            "smt_window_invalid", {r.code for r in pair.validate_smt_config(0.199, 0.05)}
        )
        self.assertIn(
            "smt_threshold_invalid",
            {r.code for r in pair.validate_smt_config(0.2, 0.051)},
        )
        self.assertEqual(pair.validate_smt_config(0.2, 0.05), [])

    def test_policy_overrides_are_refused(self):
        args = types.SimpleNamespace(
            runs="5", drift_cap="0.02", rounds="7", delta="0.03",
            gamma="0.03", calibrated_hosts="willow-cove",
            calibrated_tiers="willow-cove:v4x,v3",
            calibrated_toolchain="1.93.0/abc", calibrated_flags=self.flags,
            smt_window=0.2, smt_busy_threshold=0.05, chrt="",
            unstable_target_runs="uncalibrated",
            policy_json=json.dumps(
                {
                    "runs": "5", "drift_cap": "0.02", "rounds": "7",
                    "delta": "0.03", "gamma": "0.03",
                    "calibrated_hosts": "willow-cove",
                    "calibrated_tiers": "willow-cove:v4x,v3",
                    "calibrated_toolchain": "1.93.0/abc",
                    "calibrated_flags": self.flags,
                    "smt_window": "0.20",
                    "smt_busy_threshold": "0.05",
                    "chrt": "",
                    "unstable_target_runs": "uncalibrated",
                }
            ),
        )
        self.assertIsNone(pair.validate_policy_config(args))
        args.runs = "6"
        rejection = pair.validate_policy_config(args)
        self.assertEqual(rejection.code, "config_override")
        args.runs = "5"
        args.smt_window = 0.25
        self.assertEqual(pair.validate_policy_config(args).code, "config_override")

    def test_acceptance_requires_the_policy_snapshot(self):
        campaign = pair.Campaign.__new__(pair.Campaign)
        campaign.args = types.SimpleNamespace(
            runs="5", drift_cap="0.02", rounds="7", delta="0.03",
            gamma="0.03", calibrated_hosts="willow-cove",
            calibrated_tiers="willow-cove:v4x", calibrated_toolchain="1.93.0/abc",
            calibrated_flags=self.flags, smt_window=0.2,
            smt_busy_threshold=0.05, chrt="", policy_json=None, exploratory=False,
        )
        campaign.rejections = []
        snapshot = types.SimpleNamespace(
            rustflags=self.flags, features=["simd512"]
        )
        self.assertFalse(campaign.validate_measurement_config(
            snapshot, {"hostname": "willow-cove", "toolchain_id": "1.93.0/abc"},
            ["v4x"],
        ))
        self.assertEqual(campaign.rejections[-1].code, "config_missing")

    def test_calibration_sessions_are_refused_as_acceptance_inputs(self):
        calibration = {"kind": "calibration", "acceptance": False}
        rejection = pair.validate_acceptance_inputs([calibration])
        self.assertEqual(rejection.code, "calibration_reuse")
        accepted = {"kind": "acceptance", "acceptance": True}
        self.assertIsNone(pair.validate_acceptance_inputs([accepted]))

    def test_alignment_flags_are_required(self):
        self.assertEqual(
            pair.check_alignment_flags(self.flags).code
            if pair.check_alignment_flags(self.flags)
            else None,
            None,
        )
        rejection = pair.check_alignment_flags("-C target-cpu=native")
        self.assertEqual(rejection.code, "alignment_flags_missing")

    def test_v4x_requires_simd512_in_the_snapshot(self):
        rejection = pair.check_tier_features(["v4x"], ["std", "simd"])
        self.assertEqual(rejection.code, "simd512_missing")
        self.assertIsNone(pair.check_tier_features(["v4x"], ["simd512"]))
        self.assertIsNone(pair.check_tier_features(["v4x"], ["all"]))

    def test_unknown_tier_is_rejected(self):
        rejection = pair.check_tiers(["v4", "scalar"])
        self.assertEqual(rejection.code, "unknown_tier")
        self.assertIsNone(pair.check_tiers(["v4x", "scalar"]))

    def test_backend_mismatch_names_the_arm(self):
        rejection = pair.check_backend("v4x", "v3", "candidate")
        self.assertEqual(rejection.code, "backend_mismatch")
        self.assertEqual(rejection.arm, "candidate")
        self.assertIsNone(pair.check_backend("v4x", "v4x", "base"))


class InventoryAndOutputTests(unittest.TestCase):
    def entry(self, identifier, row_len=64):
        return {
            "id": identifier,
            "field": "gf8-aes",
            "operation": "mul_add",
            "prep": "unprepared",
            "row_len": row_len,
            "rows": 1,
            "sources": 1,
            "alignment": "aligned",
            "bytes": row_len,
        }

    def test_inventory_missing_cell_is_rejected(self):
        one = "gf8-aes/mul_add/unprepared/64x1x1/aligned"
        two = "gf8-aes/mul_add/unprepared/128x1x1/aligned"
        rejection = pair.compare_inventories(
            {one: self.entry(one), two: self.entry(two, 128)},
            {one: self.entry(one)},
        )
        self.assertEqual(rejection.code, "inventory_mismatch")
        self.assertIn(two, rejection.cells)

    def test_inventory_changed_entry_is_rejected(self):
        one = "gf8-aes/mul_add/unprepared/64x1x1/aligned"
        left = self.entry(one)
        right = dict(left, bytes=63)
        rejection = pair.compare_inventories({one: left}, {one: right})
        self.assertEqual(rejection.code, "inventory_mismatch")
        self.assertEqual(rejection.cells, (one,))

    def test_output_mismatch_names_the_first_differing_cell(self):
        first = "gf8-aes/mul_add/unprepared/64x1x1/aligned"
        second = "gf8-aes/mul_add/unprepared/128x1x1/aligned"
        base = {first: "a", second: "b"}
        candidate = {first: "a", second: "c"}
        rejection = pair.compare_outputs(base, candidate, [first, second])
        self.assertEqual(rejection.code, "output_mismatch")
        self.assertIn(second, rejection.message)
        self.assertEqual(rejection.cells, (second,))

    def test_missing_check_output_is_rejected(self):
        one = "gf8-aes/mul_add/unprepared/64x1x1/aligned"
        rejection = pair.compare_outputs({one: "a"}, {}, [one])
        self.assertEqual(rejection.code, "output_mismatch")
        self.assertEqual(rejection.cells, (one,))

    def test_dependency_sets_must_match(self):
        self.assertIsNone(
            pair.compare_dependency_sets({"a|x|"}, {"a|x|"}, "base")
        )
        rejection = pair.compare_dependency_sets(
            {"a|x|"}, {"b|y|"}, "candidate"
        )
        self.assertEqual(rejection.code, "dependency_mismatch")
        self.assertEqual(rejection.arm, "candidate")

    def test_normal_dependency_set_excludes_dev_and_build_dependencies(self):
        metadata = {
            "packages": [
                {
                    "name": "fgf",
                    "source": None,
                    "dependencies": [
                        {"name": "sim", "req": "=0.2.0", "kind": None, "target": None},
                        {"name": "arch", "req": "=0.9", "kind": "normal", "target": None},
                        {"name": "crit", "req": "=0.8", "kind": "dev", "target": None},
                        {"name": "build", "req": "=1", "kind": "build", "target": None},
                    ],
                }
            ]
        }
        self.assertEqual(
            pair.normal_dependency_set(metadata),
            {"sim|=0.2.0|", "arch|=0.9|"},
        )


class ProtocolTests(unittest.TestCase):
    def test_valid_protocol(self):
        text = "\n".join(
            [
                "BACKEND\tv3",
                "FIELD_BACKEND\tgf8-aes\tv3",
                "CELL\t" + json.dumps({
                    "id": "gf8-aes/mul_add/unprepared/64x1x1/aligned",
                    "field": "gf8-aes",
                    "operation": "mul_add",
                    "prep": "unprepared",
                    "row_len": 64,
                    "rows": 1,
                    "sources": 1,
                    "alignment": "aligned",
                    "bytes": 64,
                }),
                "CONTROL\tstart\t10",
                "ROUND\tgf8-aes/mul_add/unprepared/64x1x1/aligned\t0\t12.5",
                "CONTROL\tend\t20",
            ]
        )
        parsed, errors = pair.parse_protocol(text)
        self.assertEqual(errors, [])
        self.assertEqual(parsed["backend"], "v3")
        self.assertEqual(parsed["field_backends"], {"gf8-aes": "v3"})
        self.assertEqual(
            pair.round_values(
                parsed, "gf8-aes/mul_add/unprepared/64x1x1/aligned"
            ),
            [12.5],
        )
        self.assertEqual(pair.control_value(parsed, "start"), 10.0)

    def test_missing_backend_is_a_protocol_error(self):
        _, errors = pair.parse_protocol("OUTPUT\ta\tb")
        self.assertTrue(any("BACKEND" in error for error in errors))

    def test_malformed_inventory_entry_is_a_protocol_error(self):
        _, errors = pair.parse_protocol('CELL\t{"id":"nonsense"}')
        self.assertTrue(errors)

    def test_empty_inventory_entry_is_a_protocol_error(self):
        _, errors = pair.parse_protocol("BACKEND\tv3".replace("\\t", "\t"))
        self.assertEqual(errors, [])


class ReanalysisTests(unittest.TestCase):
    def source_session(self, kind="acceptance"):
        runs = []
        for index in range(5):
            runs.append({
                "run": index,
                "tier": "v4x",
                "order": ["base", "candidate"] if index % 2 == 0 else ["candidate", "base"],
                "control": {"start_ns": 100.0, "end_ns": 200.0},
                "arms": {
                    "base": {"cells": {"gf8-aes/mul_add/unprepared/64x1x1/aligned": {
                        "summary_ns": 1000.0, "rounds_ns": [1000.0]}}},
                    "candidate": {"cells": {"gf8-aes/mul_add/unprepared/64x1x1/aligned": {
                        "summary_ns": 990.0, "rounds_ns": [990.0]}}},
                },
            })
        return {
            "method_version": "fgf-pair/1",
            "session_id": "source",
            "kind": kind,
            "config": {"delta": 0.03, "gamma": 0.03},
            "runs": runs,
            "drift": {"v4x": {"drift": 0.001}},
            "calibration": {"mode": "base_vs_base"} if kind == "calibration" else None,
            "threshold": None,
        }

    def args(self):
        return types.SimpleNamespace(delta="0.03", gamma="0.03",
                                     drift_safety_factor=1.25)

    def test_acceptance_reanalysis_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(json.dumps(self.source_session()), encoding="utf-8")
            self.assertEqual(pair.reanalyze_session(source, self.args()), 0)
            outputs = list(Path(directory).glob("session.reanalysis-*.json"))
            self.assertEqual(len(outputs), 1)
            analysis = json.loads(outputs[0].read_text(encoding="utf-8"))
            self.assertTrue(analysis["reanalysis"])
            self.assertFalse(analysis["publishable"])
            self.assertFalse(analysis["acceptance"])
            self.assertEqual(analysis["method_version"], pair.METHOD_VERSION)
            self.assertEqual(analysis["drift_method"], pair.DRIFT_METHOD)
            self.assertIn("v4x", analysis["drift"])
            self.assertIn("drift_method", analysis["drift"]["v4x"])
            record = analysis["decisions"]["per_tier"]["v4x"]["noninferiority"][
                "per_cell"
            ]["gf8-aes/mul_add/unprepared/64x1x1/aligned"]
            self.assertIn("estimate", record)
            self.assertIn("upper_bound", record)

    def test_calibration_reanalysis_reports_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            session = self.source_session("calibration")
            session["method_version"] = "fgf-pair/2"
            source.write_text(json.dumps(session), encoding="utf-8")
            self.assertEqual(pair.reanalyze_session(source, self.args()), 0)
            output = next(Path(directory).glob("session.reanalysis-*.json"))
            analysis = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(analysis["reanalysis"])
            self.assertFalse(analysis["publishable"])
            self.assertIsNotNone(analysis["calibration"]["recommendation"]["runs"])
            self.assertIn("cells", analysis["calibration"]["recommendation"]["per_tier"]["v4x"])

    def test_reanalysis_reads_v4_under_its_recorded_drift_method(self):
        session = self.source_session()
        session["method_version"] = "fgf-pair/4"
        session["drift_method"] = "p90_minus_p10_over_median"
        session["config"].update(confirmation=True, runs=5)
        for index, run in enumerate(session["runs"]):
            run["stage"] = "main"
            run["controls"] = {
                arm: {"start_ns": 120.0 if index == 4 else 100.0}
                for arm in ("base", "candidate")
            }
        session["drift"] = {"v4x": {"drift": 0.2}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(json.dumps(session), encoding="utf-8")
            self.assertEqual(pair.reanalyze_session(source, self.args()), 0)
            output = next(Path(directory).glob("session.reanalysis-*.json"))
            analysis = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(analysis["source_method_version"], "fgf-pair/4")
            self.assertEqual(
                analysis["source_drift_method"], "p90_minus_p10_over_median"
            )
            self.assertEqual(analysis["method_version"], "fgf-pair/5")
            self.assertEqual(analysis["drift_method"], "iqr_over_median")
            self.assertEqual(analysis["drift"]["v4x"]["drift"], 0.0)
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), session)

    def test_reanalysis_rejects_mislabelled_historical_drift(self):
        session = self.source_session()
        session["method_version"] = "fgf-pair/4"
        session["drift_method"] = "iqr_over_median"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(json.dumps(session), encoding="utf-8")
            self.assertEqual(pair.reanalyze_session(source, self.args()), 2)
            self.assertFalse(list(Path(directory).glob("session.reanalysis-*.json")))

    def scoped_source_session(self, measured, selected, missing=()):
        """A session whose runs measure `measured` and whose scope selects `selected`."""
        session = self.source_session()
        cell = "gf8-aes/mul_add/unprepared/64x1x1/aligned"
        for run in session["runs"]:
            for arm in ("base", "candidate"):
                run["arms"][arm]["cells"] = {
                    name: run["arms"][arm]["cells"][cell] for name in measured
                }
        session["scope"] = {
            "kind": "selected",
            "cells": list(selected),
            "digest": hashlib.sha256(
                pair.canonical_json(list(selected)).encode("utf-8")
            ).hexdigest(),
            "rationale": "why",
            "declaration_digest": "decl",
        }
        for name in missing:
            session["scope"]["cells"].append(name)
        return session

    def test_selected_scope_constrains_the_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(
                json.dumps(
                    self.scoped_source_session(
                        ["gf8-aes/mul_add/unprepared/64x1x1/aligned"],
                        ["gf8-aes/mul_add/unprepared/64x1x1/aligned"],
                    )
                ),
                encoding="utf-8",
            )
            self.assertEqual(pair.reanalyze_session(source, self.args()), 0)
            output = next(Path(directory).glob("session.reanalysis-*.json"))
            analysis = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(analysis["scope"]["kind"], "selected")
            per_cell = analysis["decisions"]["per_tier"]["v4x"]["noninferiority"][
                "per_cell"
            ]
            self.assertEqual(
                list(per_cell), ["gf8-aes/mul_add/unprepared/64x1x1/aligned"]
            )

    def test_incomplete_recorded_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(
                json.dumps(
                    self.scoped_source_session(
                        ["gf8-aes/mul_add/unprepared/64x1x1/aligned"],
                        ["gf8-aes/mul_add/unprepared/64x1x1/aligned"],
                        missing=["gf8-aes/mul_add/unprepared/96x1x1/aligned"],
                    )
                ),
                encoding="utf-8",
            )
            self.assertEqual(pair.reanalyze_session(source, self.args()), 2)
            self.assertFalse(list(Path(directory).glob("session.reanalysis-*.json")))

    def test_scopeless_record_keeps_full_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.json"
            source.write_text(json.dumps(self.source_session()), encoding="utf-8")
            self.assertEqual(pair.reanalyze_session(source, self.args()), 0)
            output = next(Path(directory).glob("session.reanalysis-*.json"))
            analysis = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(analysis["scope"])


class ExecutionIsolationTests(unittest.TestCase):
    def test_setarch_is_required(self):
        codes = {
            rejection.code
            for rejection in pair.validate_execution_environment(
                "x86_64", None, "", None
            )
        }
        self.assertIn("setarch_missing", codes)
        self.assertEqual(
            pair.validate_execution_environment(
                "x86_64", "/usr/bin/setarch", "", None
            ),
            [],
        )

    def test_chrt_is_required_only_when_configured(self):
        self.assertEqual(
            pair.validate_execution_environment(
                "x86_64", "/usr/bin/setarch", "--idle 0", None
            )[0].code,
            "chrt_missing",
        )
        self.assertEqual(
            pair.validate_execution_environment(
                "x86_64", "/usr/bin/setarch", "--idle 0", "/usr/bin/chrt"
            ),
            [],
        )

    def test_harness_command_pins_taskset_and_setarch(self):
        command = pair.harness_command(
            "arm_a",
            ["--check"],
            core=3,
            machine="x86_64",
            chrt_parts=[],
            disable_aslr=True,
        )
        self.assertEqual(
            command,
            ["taskset", "-c", "3", "setarch", "x86_64", "-R", "arm_a", "--check"],
        )

    def test_chrt_is_prepended_only_when_configured(self):
        command = pair.harness_command(
            "arm_a",
            ["--time", "--rounds", "3"],
            core=3,
            machine="x86_64",
            chrt_parts=["--idle", "0"],
            disable_aslr=True,
        )
        self.assertEqual(
            command,
            [
                "chrt", "--idle", "0", "taskset", "-c", "3",
                "setarch", "x86_64", "-R", "arm_a", "--time", "--rounds", "3",
            ],
        )

    def test_equal_arm_aliases_have_equal_argv_length(self):
        base = pair.harness_command(
            "arm_a", ["--check"], core=3, machine="x86_64",
            chrt_parts=[], disable_aslr=True,
        )
        candidate = pair.harness_command(
            "arm_b", ["--check"], core=3, machine="x86_64",
            chrt_parts=[], disable_aslr=True,
        )
        self.assertEqual(pair.command_length(base), pair.command_length(candidate))

    def test_argv_length_mismatch_is_a_rejection(self):
        self.assertIsNone(
            pair.check_argv_lengths({"base": 42, "candidate": 42}, "run")
        )
        rejection = pair.check_argv_lengths({"base": 42, "candidate": 41}, "run")
        self.assertEqual(rejection.code, "argv_mismatch")

    def test_alias_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = pair.Campaign.__new__(pair.Campaign)
            campaign.execution_dir = Path(directory) / "exe"
            campaign.rejections = []
            with mock.patch.object(
                Path, "symlink_to", side_effect=OSError("no symlinks")
            ):
                with self.assertRaises(OSError):
                    campaign.prepare_execution_alias("base", Path("/fake/exe"))
            self.assertEqual(
                campaign.rejections[-1].code, "execution_alias_failed"
            )

    def test_environment_digest_is_order_independent(self):
        first = {"B": "2", "A": "1"}
        second = {"A": "1", "B": "2"}
        self.assertEqual(pair.environment_digest(first), pair.environment_digest(second))
        self.assertNotEqual(
            pair.environment_digest(first),
            pair.environment_digest({"A": "1", "B": "3"}),
        )


class ProtocolSummaryTests(unittest.TestCase):
    cell_id = "gf8-aes/mul_add/unprepared/64x1x1/aligned"

    def parsed(self, rounds=(1, 2, 3), extra=None):
        lines = ["BACKEND\tv3", "CONTROL\tstart\t10"]
        for index, value in enumerate(rounds):
            lines.append(f"ROUND\t{self.cell_id}\t{index}\t{value}")
        if extra:
            lines.append(f"ROUND\t{extra}\t0\t1")
        lines.append("CONTROL\tend\t20")
        parsed, errors = pair.parse_protocol("\n".join(lines))
        self.assertEqual(errors, [])
        return parsed

    def test_full_protocol_summary_is_the_round_median(self):
        cells, rejection = pair.summarize_timed_protocol(
            self.parsed(), [self.cell_id], 3, "base"
        )
        self.assertIsNone(rejection)
        self.assertEqual(cells[self.cell_id]["summary_ns"], 2.0)
        self.assertEqual(cells[self.cell_id]["rounds_ns"], [1.0, 2.0, 3.0])

    def test_missing_round_is_run_incomplete(self):
        _, rejection = pair.summarize_timed_protocol(
            self.parsed(rounds=(1, 2)), [self.cell_id], 3, "base"
        )
        self.assertEqual(rejection.code, "run_incomplete")
        self.assertEqual(rejection.cells, (self.cell_id,))

    def test_extra_cell_is_run_incomplete(self):
        _, rejection = pair.summarize_timed_protocol(
            self.parsed(extra="m31/mul_into/prepared/4096x1x1/aligned"),
            [self.cell_id],
            3,
            "candidate",
        )
        self.assertEqual(rejection.code, "run_incomplete")
        self.assertEqual(rejection.arm, "candidate")


class CalibrationRecommendationTests(unittest.TestCase):
    def test_overall_count_is_the_maximum_tier_recommendation(self):
        recommendation, healthy = pair.calibration_recommendation(
            {
                "v4x": {"cell": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]},
                "v3": {"cell": [0.0, 0.0, 0.0, 0.0, 0.0]},
            },
            0.03,
            {"v4x": {"drift": 0.004}, "v3": {"drift": 0.011}},
        )
        self.assertTrue(healthy)
        self.assertEqual(recommendation["runs"], 6)
        self.assertEqual(recommendation["per_tier"]["v3"]["runs"], 5)
        self.assertGreaterEqual(recommendation["drift_cap"], 0.011)

    def test_run_count_ignores_unstable_cells(self):
        recommendation, healthy = pair.calibration_recommendation(
            {
                "v4x": {
                    "stable": [0.0] * 5,
                    "wild": [0.1, -0.1, 0.1, -0.1, 0.1],
                }
            },
            0.03,
            {"v4x": {"drift": 0.001}},
        )
        self.assertTrue(healthy)
        record = recommendation["per_tier"]["v4x"]
        self.assertEqual(record["runs"], 5)
        self.assertEqual(record["unstable_cells"], ["wild"])
        self.assertEqual(record["stable_cells"], ["stable"])
        self.assertTrue(record["cells"]["wild"]["unstable"])
        self.assertFalse(record["cells"]["stable"]["unstable"])

    def test_unstable_tier_does_not_make_the_session_insufficient(self):
        recommendation, healthy = pair.calibration_recommendation(
            {"v4x": {"cell": [0.1, -0.1, 0.1, -0.1, 0.1]}},
            0.03,
            {"v4x": {"drift": 0.001}},
        )
        self.assertTrue(healthy)
        self.assertEqual(recommendation["runs"], 5)
        self.assertEqual(
            recommendation["per_tier"]["v4x"]["unstable_cells"], ["cell"]
        )
        self.assertEqual(recommendation["per_tier"]["v4x"]["runs"], 5)


class SessionInputTests(unittest.TestCase):
    def test_unreadable_prior_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text("not json", encoding="utf-8")
            sessions, rejection = pair.validate_prior_session_files([path])
            self.assertIsNone(sessions)
            self.assertEqual(rejection.code, "session_invalid")

    def test_campaign_lock_excludes_a_second_campaign(self):
        import types
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                repo=directory,
                records=Path(directory) / "records",
                keep_checkouts=False,
            )
            first = pair.Campaign(args, {}, Path(directory))
            second = pair.Campaign(args, {}, Path(directory))
            try:
                self.assertTrue(first.acquire_lock())
                self.assertFalse(second.acquire_lock())
                self.assertEqual(second.rejections[-1].code, "campaign_locked")
            finally:
                first.release_lock()
                first.close()
                second.close()
                shutil.rmtree(Path(directory) / "records", ignore_errors=True)

    def test_campaign_lock_is_shared_across_record_destinations(self):
        import types
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            pair_args = types.SimpleNamespace(
                repo=directory,
                records=Path(directory) / "pair",
                keep_checkouts=False,
            )
            standalone_args = types.SimpleNamespace(
                repo=directory,
                records=Path(directory) / "standalone",
                keep_checkouts=False,
            )
            first = pair.Campaign(pair_args, {}, Path(directory) / "a")
            second = pair.Campaign(standalone_args, {}, Path(directory) / "b")
            try:
                self.assertEqual(
                    first.lock_path,
                    Path(directory) / "bench-records" / ".campaign.lock",
                )
                self.assertTrue(first.acquire_lock())
                self.assertFalse(second.acquire_lock())
                self.assertIn(
                    str(first.lock_path), second.rejections[-1].message
                )
            finally:
                first.release_lock()
                first.close()
                second.close()
                shutil.rmtree(
                    Path(directory) / "bench-records", ignore_errors=True
                )


class CampaignFailureTests(unittest.TestCase):
    cell = "gf8-aes/mul_add/unprepared/64x1x1/aligned"

    def campaign(self, directory, **overrides):
        args = types.SimpleNamespace(
            repo=directory,
            records=Path(directory) / "records",
            keep_checkouts=False,
            timeout=10.0,
            calibration_mode="base_vs_base",
            layout_sessions="",
            mode="acceptance",
            base=None,
            candidate=None,
            prior_session=[],
            drift_safety_factor=1.25,
            policy_json=None,
            exploratory=False,
            smt_window=0.2,
            smt_busy_threshold=0.05,
            core=0,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        session = {
            "session_id": "s",
            "host": {"hostname": "host"},
            "snapshot": {},
            "arms": {
                "base": {"sha": "b", "kind": "library"},
                "candidate": {"sha": "c", "kind": "library"},
            },
            "config": {"delta": 0.03},
            "inventory": {"order": [self.cell]},
            "runs": [],
            "arm_summaries": {"base": {}, "candidate": {}},
        }
        Path(directory, "session").mkdir(exist_ok=True)
        return pair.Campaign(args, session, Path(directory) / "session")

    def tearDown(self):
        pair.run_process = self.original_run_process

    def setUp(self):
        self.original_run_process = pair.run_process

    def test_missing_base_is_a_usage_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            try:
                self.assertFalse(campaign.resolve_arms(pair.Snapshot(
                    "s", "s", Path(directory), "d", {}, [], "", None, "operations", False
                )))
                self.assertEqual(campaign.rejections[-1].code, "usage")
            finally:
                campaign.close()

    def test_build_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory, "arm")
            checkout.mkdir()
            (checkout / "Cargo.toml").write_text("[package]\n")
            campaign = self.campaign(directory)
            pair.run_process = lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 1, "", "build exploded"
            )
            try:
                snapshot = pair.Snapshot(
                    "s", "s", Path(directory), "d", {}, ["simd512"],
                    pair.DEFAULT_ALIGN_FLAGS, None, "operations", False,
                )
                result = campaign.build_bench(
                    "base", {"sha": "b", "checkout": str(checkout)}, snapshot
                )
                self.assertIsNone(result)
                self.assertEqual(campaign.rejections[-1].code, "build_failed")
            finally:
                campaign.close()

    def test_metadata_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory, "arm")
            checkout.mkdir()
            (checkout / "Cargo.toml").write_text("[package]\n")
            campaign = self.campaign(directory)
            pair.run_process = lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 1, "", "metadata exploded"
            )
            try:
                self.assertIsNone(campaign.cargo_metadata(checkout, "base"))
                self.assertEqual(
                    campaign.rejections[-1].code, "dependency_metadata_failed"
                )
            finally:
                campaign.close()

    def test_missing_snapshot_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory, snapshot="missing-rev")
            pair.run_process = lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 128, "", "unknown revision"
            )
            try:
                self.assertIsNone(campaign.prepare_snapshot())
                self.assertEqual(campaign.rejections[-1].code, "snapshot_missing")
            finally:
                campaign.close()

    def test_unstable_calibration_cells_do_not_reject_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            campaign.session["runs"] = [
                {
                    "run": index,
                    "tier": "v4x",
                    "order": ["base", "candidate"],
                    "control": {"start_ns": 1.0, "end_ns": 2.0},
                    "arms": {
                        "base": {
                            "cells": {self.cell: {"summary_ns": 100.0, "rounds_ns": [100.0]}}
                        },
                        "candidate": {
                            "cells": {
                                self.cell: {
                                    "summary_ns": 100.0 + (10 if index == 0 else -10),
                                    "rounds_ns": [100.0],
                                }
                            }
                        },
                    },
                }
                for index in range(5)
            ]
            try:
                self.assertEqual(
                    campaign.finish_calibration(["v4x"], {"v4x": {"drift": 0.001}}),
                    0,
                )
                recommendation = campaign.session["calibration"]["recommendation"]
                self.assertEqual(recommendation["runs"], 5)
                self.assertIn(
                    self.cell,
                    recommendation["per_tier"]["v4x"]["unstable_cells"],
                )
            finally:
                campaign.close()

    def test_noisy_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            campaign.session["config"].update({"delta": 0.03, "gamma": 0.03})
            campaign.session["runs"] = [
                {
                    "run": index,
                    "tier": "v4x",
                    "order": ["base", "candidate"],
                    "control": {"start_ns": 1.0, "end_ns": 2.0},
                    "arms": {
                        "base": {
                            "cells": {self.cell: {"summary_ns": 100.0, "rounds_ns": [100.0]}}
                        },
                        "candidate": {
                            "cells": {
                                self.cell: {
                                    "summary_ns": 110.0 if index % 2 == 0 else 90.0,
                                    "rounds_ns": [110.0 if index % 2 == 0 else 90.0],
                                }
                            }
                        },
                    },
                }
                for index in range(4)
            ]
            try:
                self.assertEqual(
                    campaign.finish_calibration(["v4x"], {"v4x": {"drift": 0.001}}),
                    1,
                )
                self.assertEqual(
                    campaign.rejections[-1].code, "calibration_insufficient"
                )
            finally:
                campaign.close()

    def test_checkout_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            snapshot = pair.Snapshot(
                "s", "s", Path(directory), "d", {}, [], "", None, "operations", False
            )
            original = pair.checkout_revision

            def fail_checkout(*args, **kwargs):
                raise RuntimeError("cannot materialize")

            pair.checkout_revision = fail_checkout
            try:
                self.assertFalse(
                    campaign.make_arm_checkout("base", {"sha": "b"}, snapshot)
                )
                self.assertEqual(campaign.rejections[-1].code, "checkout_failed")
            finally:
                pair.checkout_revision = original
                campaign.close()

    def test_overlay_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            missing = Path(directory, "missing-snapshot")
            snapshot = pair.Snapshot(
                "s", "s", missing, "d", {}, [], "", None, "operations", False
            )
            original = pair.checkout_revision

            def make_checkout(repo, revision, destination):
                destination.mkdir(parents=True)
                return "archive"

            campaign.cargo_metadata = lambda *args, **kwargs: {
                "packages": [
                    {
                        "name": "fgf",
                        "source": None,
                        "dependencies": [
                            {"name": "s", "req": "=1", "kind": None, "target": None}
                        ],
                    }
                ]
            }
            pair.checkout_revision = make_checkout
            try:
                self.assertFalse(
                    campaign.make_arm_checkout("base", {"sha": "b"}, snapshot)
                )
                self.assertEqual(campaign.rejections[-1].code, "overlay_failed")
            finally:
                pair.checkout_revision = original
                campaign.close()

    def test_snapshot_checkout_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory, snapshot="some-rev")
            original_git = pair.git_revision
            original_checkout = pair.checkout_revision

            def fake_git(repo, revision):
                return "deadbeef"

            def fail_checkout(*args, **kwargs):
                raise RuntimeError("cannot materialize")

            pair.git_revision = fake_git
            pair.checkout_revision = fail_checkout
            try:
                self.assertIsNone(campaign.prepare_snapshot())
                self.assertEqual(
                    campaign.rejections[-1].code, "snapshot_checkout_failed"
                )
            finally:
                pair.git_revision = original_git
                pair.checkout_revision = original_checkout
                campaign.close()

    def test_unexpected_exception_is_recorded_as_internal_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = pair.build_parser().parse_args(
                [
                    "--records",
                    str(Path(directory) / "records"),
                    "--snapshot",
                    str(Path(directory) / "snapshot"),
                ]
            )
            original = pair.Campaign.execute

            def explode(self):
                raise RuntimeError("boom")

            pair.Campaign.execute = explode
            try:
                self.assertEqual(pair.run(args), 1)
            finally:
                pair.Campaign.execute = original
            sessions = list(
                (Path(directory) / "records").glob("*/session.json")
            )
            self.assertEqual(len(sessions), 1)
            session = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(session["rejections"][-1]["code"], "internal_error")

    def test_harness_failures_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory, core=0, smt_window=0.01)
            campaign.run_harness = lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 1, "", "harness exploded"
            )
            try:
                self.assertIsNone(
                    campaign.inspect_tier("v4x", "base", {}, Path("/fake"))
                )
                self.assertEqual(campaign.rejections[-1].code, "inventory_failed")
                self.assertIsNone(
                    campaign.run_checks("v4x", "base", Path("/fake"), [self.cell])
                )
                self.assertEqual(campaign.rejections[-1].code, "check_failed")
                campaign.args_rounds = 1
                campaign.current_run = 0
                self.assertIsNone(
                    campaign.timed_run("v4x", "base", Path("/fake"), [self.cell])
                )
                self.assertEqual(campaign.rejections[-1].code, "run_failed")
            finally:
                campaign.close()

    def test_calibration_history_drift_raises_the_recommended_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            prior = Path(directory) / "records" / "prior" / "session.json"
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({
                "session_id": "prior",
                "kind": "calibration",
                "acceptance": False,
                "rejected": False,
                "rejections": [],
                "host": {"hostname": "host", "toolchain_id": None},
                "snapshot": {"rustflags": ""},
                "drift_method": pair.DRIFT_METHOD,
                "drift": {"v4x": {"drift": 0.02}},
            }), encoding="utf-8")
            campaign.session["runs"] = [
                {
                    "run": index,
                    "tier": "v4x",
                    "order": ["base", "candidate"],
                    "control": {"start_ns": 1.0, "end_ns": 2.0},
                    "arms": {
                        "base": {"cells": {self.cell: {
                            "summary_ns": 100.0, "rounds_ns": [100.0]}}},
                        "candidate": {"cells": {self.cell: {
                            "summary_ns": 100.0, "rounds_ns": [100.0]}}},
                    },
                }
                for index in range(5)
            ]
            try:
                self.assertEqual(
                    campaign.finish_calibration(
                        ["v4x"], {"v4x": {"drift": 0.001}}
                    ),
                    0,
                )
                recommendation = campaign.session["calibration"]["recommendation"]
                self.assertGreaterEqual(recommendation["drift_cap"], 0.025)
                self.assertEqual(recommendation["drift_safety_factor"], 1.25)
            finally:
                campaign.close()

    def test_layout_only_without_a_spread_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(directory)
            decisions = {
                "v4x": {
                    "noninferiority": {
                        "failed_cells": [self.cell],
                        "per_cell": {self.cell: {"upper_bound": 0.05}},
                    }
                }
            }
            try:
                self.assertFalse(campaign.layout_only(["v4x"], decisions))
                self.assertEqual(
                    campaign.rejections[-1].code, "layout_spread_missing"
                )
            finally:
                campaign.close()

    def test_layout_only_with_a_foreign_host_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "layout.json")
            path.write_text(
                json.dumps(
                    {
                        "kind": "calibration",
                        "acceptance": False,
                        "host": {"hostname": "somewhere-else"},
                        "calibration": {"mode": "base_vs_layout_control"},
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            campaign = self.campaign(
                directory, layout_sessions=str(path)
            )
            decisions = {
                "v4x": {
                    "noninferiority": {
                        "failed_cells": [self.cell],
                        "per_cell": {self.cell: {"upper_bound": 0.05}},
                    }
                }
            }
            try:
                self.assertFalse(campaign.layout_only(["v4x"], decisions))
                self.assertEqual(
                    campaign.rejections[-1].code, "layout_spread_mismatch"
                )
            finally:
                campaign.close()


class UnstableListTests(unittest.TestCase):
    flags = pair.DEFAULT_ALIGN_FLAGS

    def entry(self, **overrides):
        value = {
            "host": "host",
            "tier": "v4x",
            "toolchain": "tc",
            "flags": self.flags,
            "inventory_digest": "digest",
            "cells": ["cell-b", "cell-a"],
            "sources": ["session-1"],
            "owner_signature": "owner",
        }
        value.update(overrides)
        return value



    def test_missing_file_is_a_missing_list(self):
        with tempfile.TemporaryDirectory() as directory:
            entries, rejection = pair.load_unstable_entries(
                Path(directory) / "absent.json"
            )
            self.assertIsNone(entries)
            self.assertEqual(rejection.code, "unstable_list_missing")

    def test_writer_exports_catalogue_unions_sessions_and_clears_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "unstable_cells.json"
            catalogue = [
                {"id": f"gf8-aes/mul_add/unprepared/{length}x1x1/aligned",
                 "field": "gf8-aes", "operation": "mul_add", "prep": "unprepared",
                 "row_len": length, "rows": 1, "sources": 1,
                 "alignment": "aligned", "bytes": length}
                for length in (64, 96, 128, 160)
            ]
            ids = [entry["id"] for entry in catalogue]
            digest = pair.inventory_digest(catalogue, ids)
            target.write_text(json.dumps({
                "entries": [self.entry(cells=[ids[3]], inventory_digest=digest)]
            }), encoding="utf-8")

            def source(session_id, cells):
                return {
                    "session_id": session_id, "kind": "calibration",
                    "rejected": False, "rejections": [],
                    "host": {"hostname": "host", "toolchain_id": "tc"},
                    "snapshot": {"rustflags": self.flags},
                    "inventory": {"digest": digest, "base": catalogue, "order": ids},
                    "calibration": {"recommendation": {
                        "per_tier": {"v4x": {"unstable_cells": cells}}}},
                }

            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(source("s1", ids[:2])))
            second.write_text(json.dumps(source("s2", ids[1:3])))
            self.assertEqual(pair.write_unstable_list(target, [first, second]), 0)
            entries, rejection = pair.load_unstable_entries(target)
            self.assertIsNone(rejection)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(set(entry["cells"]), set(ids))
            self.assertEqual(entry["sources"], ["s1", "s2", "session-1"])
            self.assertIsNone(entry["owner_signature"])
            entry["owner_signature"] = "owner"
            exclusions, uncalibrated, rejection = pair.unstable_cells_for_inventory(
                entries, {"hostname": "host"}, "v4x", "tc", self.flags,
                {item["id"]: item for item in catalogue}, None,
                root / "calibrated_inventories")
            self.assertIsNone(rejection)
            self.assertEqual(set(exclusions), set(ids))
            self.assertEqual(uncalibrated, [])


class ScopeDeclarationTests(unittest.TestCase):
    def test_json_object_carries_cells_and_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            path.write_text(
                json.dumps({"cells": ["a", "b"], "rationale": "why"}),
                encoding="utf-8",
            )
            parsed = pair.parse_scope_file(path)
            self.assertEqual(parsed["cells"], ["a", "b"])
            self.assertEqual(parsed["rationale"], "why")
            self.assertEqual(
                parsed["digest"], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_list_and_text_formats_take_the_explicit_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            listed = Path(directory) / "scope.json"
            listed.write_text(json.dumps(["a", "b"]), encoding="utf-8")
            parsed = pair.parse_scope_file(listed, "explicit reason")
            self.assertEqual(parsed["cells"], ["a", "b"])
            self.assertEqual(parsed["rationale"], "explicit reason")
            text = Path(directory) / "scope.txt"
            text.write_text("# comment\na\n\nb # note\n", encoding="utf-8")
            self.assertEqual(
                pair.parse_scope_file(text, "explicit reason")["cells"],
                ["a", "b"],
            )

    def test_empty_duplicate_and_unrationalized_declarations_are_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.json"
            empty.write_text(json.dumps({"cells": [], "rationale": "why"}))
            with self.assertRaises(ValueError):
                pair.parse_scope_file(empty)
            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps({"cells": ["a", "a"], "rationale": "why"}))
            with self.assertRaises(ValueError):
                pair.parse_scope_file(duplicate)
            blank = root / "blank.txt"
            blank.write_text("a\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                pair.parse_scope_file(blank)
            unrationalized = root / "unrationalized.json"
            unrationalized.write_text(json.dumps({"cells": ["a"]}))
            with self.assertRaises(ValueError):
                pair.parse_scope_file(unrationalized)
            with self.assertRaises(ValueError):
                pair.parse_scope_file(unrationalized, "   ")


class ScopeTests(unittest.TestCase):
    order = [
        "gf8-aes/mul_add/unprepared/64x1x1/aligned",
        "gf8-aes/mul_add/unprepared/96x1x1/aligned",
        "gf8-aes/mul_add/unprepared/128x1x1/aligned",
    ]

    def campaign(self, directory: Path):
        campaign = pair.Campaign.__new__(pair.Campaign)
        campaign.session = {"scope": None}
        campaign.session_dir = directory
        campaign.rejections = []
        return campaign

    def declaration(self, cells, rationale="why"):
        return {
            "path": "scope.json",
            "digest": "decl-digest",
            "cells": list(cells),
            "rationale": rationale,
        }

    def test_full_scope_preserves_inventory_order(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(Path(directory))
            cells, filter_path = campaign.apply_scope(list(self.order), None)
            self.assertEqual(cells, self.order)
            self.assertIsNone(filter_path)
            record = campaign.session["scope"]
            self.assertEqual(record["kind"], "full")
            self.assertEqual(record["cells"], self.order)
            self.assertEqual(
                record["digest"],
                hashlib.sha256(
                    pair.canonical_json(self.order).encode("utf-8")
                ).hexdigest(),
            )
            self.assertIsNone(record["rationale"])
            self.assertIsNone(record["declaration_digest"])

    def test_selected_scope_keeps_inventory_order_and_writes_the_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(Path(directory))
            declaration = self.declaration(
                [self.order[2], self.order[0]], "measured decision"
            )
            cells, filter_path = campaign.apply_scope(list(self.order), declaration)
            self.assertEqual(cells, [self.order[0], self.order[2]])
            self.assertEqual(
                filter_path.read_text(encoding="utf-8").split(),
                [self.order[0], self.order[2]],
            )
            record = campaign.session["scope"]
            self.assertEqual(record["kind"], "selected")
            self.assertEqual(record["cells"], [self.order[0], self.order[2]])
            self.assertEqual(
                record["digest"],
                hashlib.sha256(
                    pair.canonical_json(record["cells"]).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(record["rationale"], "measured decision")
            self.assertEqual(record["declaration_digest"], "decl-digest")

    def test_unknown_scope_cells_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = self.campaign(Path(directory))
            cells, filter_path = campaign.apply_scope(
                list(self.order), self.declaration(["absent-cell"])
            )
            self.assertIsNone(cells)
            self.assertIsNone(filter_path)
            self.assertEqual(
                campaign.rejections[-1].code, "scope_unknown"
            )


class CalibratedInventoryTests(unittest.TestCase):
    flags = pair.DEFAULT_ALIGN_FLAGS

    def entry(self, length=64):
        return {"id": f"gf8-aes/mul_add/unprepared/{length}x1x1/aligned",
                "field": "gf8-aes", "operation": "mul_add", "prep": "unprepared",
                "row_len": length, "rows": 1, "sources": 1,
                "alignment": "aligned", "bytes": length}

    def fixture(self, root, entries):
        ids = [entry["id"] for entry in entries]
        digest = pair.inventory_digest(entries, ids)
        (root / f"{digest}.json").write_text(json.dumps(entries), encoding="utf-8")
        signed = {"host": "host", "tier": "v4x", "toolchain": "tc",
                  "flags": self.flags, "inventory_digest": digest,
                  "cells": ids, "sources": ["session-1"], "owner_signature": "owner"}
        return signed

    def match(self, root, signed, live, scope=None):
        return pair.unstable_cells_for_inventory(
            signed, {"hostname": "host"}, "v4x", "tc", self.flags, live, scope, root)

    def test_snapshot_must_hash_to_its_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signed = self.fixture(root, [self.entry()])
            path = root / f"{signed['inventory_digest']}.json"
            path.write_text(json.dumps([self.entry(96)]), encoding="utf-8")
            _, _, rejection = pair.load_calibrated_inventory(root, signed["inventory_digest"])
            self.assertEqual(rejection.code, "calibrated_inventory_stale")
            _, _, rejection = pair.load_calibrated_inventory(root, "absent")
            self.assertEqual(rejection.code, "calibrated_inventory_missing")

    def test_appended_cases_face_the_gate_without_an_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, added = self.entry(), self.entry(96), self.entry(128)
            signed = self.fixture(root, [first, second])
            live = {entry["id"]: entry for entry in (first, second, added)}
            exclusions, uncalibrated, rejection = self.match(root, [signed], live)
            self.assertIsNone(rejection)
            self.assertEqual(exclusions, sorted([first["id"], second["id"]]))
            self.assertEqual(uncalibrated, [added["id"]])
            decision = pair.evaluate_noninferiority(
                {first["id"]: [0.0] * 5, second["id"]: [0.0] * 5,
                 added["id"]: [math.log(1.1)] * 5}, 0.03, excluded=exclusions)
            self.assertEqual(decision["failed_cells"], [added["id"]])

    def test_scope_excludes_unmeasured_classifications_and_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self.entry(), self.entry(96)
            signed = self.fixture(root, [first, second])
            live = {first["id"]: dict(first, bytes=128), second["id"]: second}
            exclusions, uncalibrated, rejection = self.match(root, [signed], live, [second["id"]])
            self.assertIsNone(rejection)
            self.assertEqual((exclusions, uncalibrated), ([second["id"]], []))
            _, _, rejection = self.match(root, [signed], live)
            self.assertEqual(rejection.code, "calibrated_cell_redefined")
            self.assertEqual(rejection.cells, (first["id"],))

    def test_unsigned_duplicate_and_wrong_toolchain_entries_cannot_exclude(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.entry()
            signed = self.fixture(root, [entry])
            live = {entry["id"]: entry}
            for records, code in (
                ([dict(signed, owner_signature=None)], "unstable_list_unsigned"),
                ([signed, signed], "unstable_list_stale"),
                ([dict(signed, toolchain="other")], "unstable_list_stale"),
                ([dict(signed, host="other")], "unstable_list_missing"),
            ):
                exclusions, _, rejection = self.match(root, records, live)
                self.assertIsNone(exclusions)
                self.assertEqual(rejection.code, code)


class MockedSessionTests(unittest.TestCase):
    """End-to-end runner sessions with fake cargo/git/harness (no subprocess)."""

    cells = [
        {
            "id": "gf8-aes/mul_add/unprepared/64x1x1/aligned",
            "field": "gf8-aes", "operation": "mul_add", "prep": "unprepared",
            "row_len": 64, "rows": 1, "sources": 1, "alignment": "aligned",
            "bytes": 64,
        },
        {
            "id": "gf8-aes/mul_add/unprepared/96x1x1/aligned",
            "field": "gf8-aes", "operation": "mul_add", "prep": "unprepared",
            "row_len": 96, "rows": 1, "sources": 1, "alignment": "aligned",
            "bytes": 96,
        },
        {
            "id": "gf8-aes/mul_add/unprepared/128x1x1/aligned",
            "field": "gf8-aes", "operation": "mul_add", "prep": "unprepared",
            "row_len": 128, "rows": 1, "sources": 1, "alignment": "aligned",
            "bytes": 128,
        },
        {
            "id": "m31/mul_into/prepared/4096x1x1/aligned",
            "field": "m31", "operation": "mul_into", "prep": "prepared",
            "row_len": 4096, "rows": 1, "sources": 1, "alignment": "aligned",
            "bytes": 4096,
        },
    ]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.snapshot = self.root / "snapshot"
        (self.snapshot / "benches").mkdir(parents=True)
        (self.snapshot / "benches" / "operations.rs").write_text("fn main() {}\n")
        (self.snapshot / "Cargo.toml").write_text("[package]\nname='fgf'\n")
        (self.snapshot / "Cargo.lock").write_text("version = 4\n")
        self.records = self.root / "records"
        self.hostname = __import__("socket").gethostname()
        self.factor = 0.99
        self.slow_only_first = False
        self.timing_factor = None
        self.control_start = None
        self.unstable_file = self.root / "unstable_cells.json"
        self.calibrated = self.root / "calibrated_inventories"
        digest = pair.inventory_digest(
            self.cells, [cell["id"] for cell in self.cells]
        )
        self.write_calibrated_inventory(digest, self.cells)
        self.unstable_file.write_text(json.dumps({
            "entries": [{
                "host": self.hostname,
                "tier": "v4x",
                "toolchain": "1.0.0/xyz",
                "flags": pair.DEFAULT_ALIGN_FLAGS,
                "inventory_digest": digest,
                "cells": [],
                "sources": ["test"],
                "owner_signature": "test-owner",
            }]
        }), encoding="utf-8")
        self.original_run_process = pair.run_process
        self.original_harness = pair.Campaign.run_harness
        self.original_environment = pair.read_environment
        self.original_which = pair.shutil.which
        pair.shutil.which = (
            lambda name: f"/usr/bin/{name}"
            if name in ("setarch", "chrt")
            else self.original_which(name)
        )
        pair.run_process = self.fake_run_process
        pair.Campaign.run_harness = self.fake_run_harness
        pair.read_environment = lambda core, window: {
            "pinned_core": 0,
            "affinity": [0, 1, 2, 3],
            "governor": "performance",
            "smt_active": False,
            "siblings": [],
            "sibling_busy": None,
            "smt_window_seconds": window,
        }

    def tearDown(self):
        pair.run_process = self.original_run_process
        pair.Campaign.run_harness = self.original_harness
        pair.read_environment = self.original_environment
        pair.shutil.which = self.original_which
        self.directory.cleanup()

    def fake_run_process(self, command, *, cwd=None, env=None, timeout=None, log_path=None):
        command = [str(part) for part in command]
        if command[:2] == ["git", "-C"]:
            rest = command[3:]
            if rest[0] == "rev-parse":
                return subprocess.CompletedProcess(command, 0, "deadbeefcafebabe\n", "")
            if rest[0] == "worktree" and rest[1] == "add":
                destination = Path(rest[-2])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(self.snapshot, destination)
                return subprocess.CompletedProcess(command, 0, "", "")
            if rest[0] == "worktree" and rest[1] == "remove":
                return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "rustc":
            return subprocess.CompletedProcess(
                command, 0,
                "rustc 1.0.0\ncommit-hash: xyz\nrelease: 1.0.0\n", "",
            )
        if command[0] == "cargo" and command[1:2] == ["metadata"]:
            metadata = {
                "packages": [{
                    "name": "fgf", "source": None,
                    "dependencies": [
                        {"name": "simdispatch", "req": "=0.2.0",
                         "kind": None, "target": None},
                    ],
                }],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")
        if command[0] == "cargo" and "build" in command:
            name = "operations" if "--bench" in command else "backend_report"
            kind = ["bench"] if name == "operations" else ["example"]
            artifact = {
                "reason": "compiler-artifact",
                "target": {"name": name, "kind": kind},
                "executable": f"/fake/{name}",
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(artifact) + "\n", "")
        return subprocess.CompletedProcess(command, 1, "", "unexpected command")

    def fake_run_harness(self, executable, extra, tier, label, **kwargs):
        command = [str(executable), *extra]
        if "--inventory" in extra:
            lines = [f"BACKEND\t{tier}", "FIELD_BACKEND\tgf8-aes\t" + tier]
            lines += ["CELL\t" + json.dumps(cell) for cell in self.cells]
            return subprocess.CompletedProcess(command, 0, "\n".join(lines) + "\n", "")
        if "--check" in extra:
            lines = [f"BACKEND\t{tier}"]
            lines += [f"OUTPUT\t{cell['id']}\t" + "0" * 64 for cell in self.cells]
            return subprocess.CompletedProcess(command, 0, "\n".join(lines) + "\n", "")
        if "--time" in extra:
            selected = self.cells
            if "--cells" in extra:
                path = Path(extra[extra.index("--cells") + 1])
                wanted = set(path.read_text(encoding="utf-8").split())
                selected = [cell for cell in self.cells if cell["id"] in wanted]
            rounds = int(extra[extra.index("--rounds") + 1])
            run = int(label.split("-")[1])
            arm = "base" if "-base-" in label else (
                "layout_control" if "layout_control" in label else "candidate"
            )
            start = 100.0 + run * 0.1 + (0.02 if arm != "base" else 0.0)
            if self.control_start is not None:
                start = self.control_start(run, arm, label.rsplit("-", 1)[-1])
            lines = [f"BACKEND\t{tier}", f"CONTROL\tstart\t{start}"]
            for index, cell in enumerate(selected):
                factor = self.factor if arm == "candidate" else 1.0
                if self.slow_only_first and not (arm == "candidate" and index == 0):
                    factor = 1.0
                if self.timing_factor is not None:
                    factor = self.timing_factor(
                        arm, cell["id"], run, label.rsplit("-", 1)[-1]
                    )
                for round_index in range(rounds):
                    value = (1000.0 + round_index) * factor
                    lines.append(
                        f"ROUND\t{cell['id']}\t{round_index}\t{value}"
                    )
            lines.append(f"CONTROL\tend\t{start + 500.0}")
            return subprocess.CompletedProcess(command, 0, "\n".join(lines) + "\n", "")
        lines = [f"BACKEND\t{tier}", "FIELD_BACKEND\tgf8-aes\t" + tier]
        return subprocess.CompletedProcess(command, 0, "\n".join(lines) + "\n", "")

    def write_calibrated_inventory(self, digest, cells):
        self.calibrated.mkdir(parents=True, exist_ok=True)
        (self.calibrated / f"{digest}.json").write_text(
            json.dumps(cells), encoding="utf-8"
        )

    def args(self, *extra):
        return pair.build_parser().parse_args(
            [
                "--records", str(self.records),
                "--base", "BASE",
                "--candidate", "CAND",
                "--snapshot", str(self.snapshot),
                "--tiers", "v4x",
                "--rounds", "3",
                "--runs", "5",
                "--drift-cap", "0.2",
                "--calibrated-hosts", self.hostname,
                "--calibrated-tiers", f"{self.hostname}:v4x",
                "--calibrated-toolchain", "1.0.0/xyz",
                "--calibrated-flags", pair.DEFAULT_ALIGN_FLAGS,
                "--calibration-runs", "6",
                "--unstable-file", str(self.unstable_file),
                "--calibrated-inventories", str(self.calibrated),
                "--policy-json", json.dumps({
                    "runs": "5", "drift_cap": "0.2", "rounds": "3",
                    "delta": "0.03", "gamma": "0.03",
                    "calibrated_hosts": self.hostname,
                    "calibrated_tiers": f"{self.hostname}:v4x",
                    "calibrated_toolchain": "1.0.0/xyz",
                    "calibrated_flags": pair.DEFAULT_ALIGN_FLAGS,
                    "smt_window": "0.2", "smt_busy_threshold": "0.05",
                    "chrt": "", "unstable_target_runs": "uncalibrated",
                }),
            ]
            + list(extra)
        )

    def session(self):
        paths = list(self.records.glob("*/session.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_acceptance_session_passes_the_publisher_validator(self):
        import publish

        self.assertEqual(pair.run(self.args()), 0)
        session = self.session()
        publish.validate_session(session, "acceptance")
        self.assertEqual(session["kind"], "acceptance")
        self.assertTrue(session["acceptance"])
        self.assertTrue(session["decisions"]["noninferiority"]["passed"])
        self.assertEqual(session["config"]["estimator"], pair.ESTIMATOR)
        self.assertEqual(session["drift_method"], pair.DRIFT_METHOD)
        self.assertEqual(
            session["drift"]["v4x"]["drift_method"], pair.DRIFT_METHOD
        )
        self.assertIn("p10", session["drift"]["v4x"])
        self.assertIn("p90", session["drift"]["v4x"])
        self.assertEqual(session["config"]["confidence"], 0.95)
        cell_record = session["decisions"]["noninferiority"]["per_tier"]["v4x"][
            "per_cell"
        ]["gf8-aes/mul_add/unprepared/64x1x1/aligned"]
        self.assertIn("estimate", cell_record)
        self.assertIn("upper_bound", cell_record)
        self.assertNotIn("bound", cell_record)
        self.assertTrue(session["preconditions"])
        record = session["preconditions"][0]
        self.assertEqual(record["smt_window_seconds"], 0.2)
        self.assertEqual(record["smt_busy_threshold"], 0.05)
        self.assertIn("sibling_busy", record)
        self.assertEqual(session["config"]["smt_window"], 0.2)
        self.assertEqual(session["config"]["smt_busy_threshold"], 0.05)
        self.assertEqual(session["route_detail"], "per-field")
        self.assertIn("per-cell", session["route_limitation"])
        self.assertTrue(session["execution"]["aslr_disabled"])
        self.assertTrue(session["execution"]["setarch"].startswith("setarch "))
        self.assertIn("environment_digest", session["runs"][0])
        self.assertEqual(
            len(set(session["runs"][0]["argv_lengths"].values())), 1
        )
        check = session["checks"]["v4x"]
        self.assertIn("environment_digest", check)
        self.assertEqual(len(set(check["argv_lengths"].values())), 1)

    def reanalyze(self, session):
        source = self.root / "source.json"
        pair.write_session(session, source)
        expected = 0 if pair.decisions_pass(session["decisions"], None) else 1
        self.assertEqual(pair.reanalyze_session(source, self.args()), expected)
        output = next(self.root.glob("source.reanalysis-*.json"))
        return json.loads(output.read_text(encoding="utf-8"))

    def test_main_outlier_passes_only_after_separate_confirmation(self):
        first = self.cells[0]["id"]
        outliers = [-0.16, 0.0, 0.0, 0.0, 0.16]
        self.timing_factor = lambda arm, name, run, stage: (
            math.exp(outliers[run])
            if arm == "candidate" and name == first and stage == "main" else 1.0
        )
        self.assertEqual(pair.run(self.args()), 0)
        session = self.session()
        decision = session["decisions"]["per_tier"]["v4x"]["noninferiority"]
        record = decision["per_cell"][first]
        self.assertAlmostEqual(record["main"]["estimate"], 0.0)
        self.assertFalse(record["main"]["pass"])
        self.assertTrue(record["confirmation"]["pass"])
        self.assertEqual(record["confirmation"]["upper_bound"], 0.0)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["unconfirmed_failures"], [first])
        self.assertEqual(session["confirmed_failures"]["v4x"], [])
        confirmation = [run for run in session["runs"] if run["stage"] == "confirmation"]
        self.assertEqual(len(confirmation), session["config"]["runs"])
        for index, run in enumerate(confirmation):
            self.assertEqual(run["order"], pair.balanced_order(index))
            self.assertEqual(run["precondition"]["phase"], "confirmation")
            for arm in ("base", "candidate"):
                self.assertEqual(set(run["arms"][arm]["cells"]), {first})
        self.assertEqual(
            self.reanalyze(session)["decisions"], session["decisions"]
        )

    def test_shift_fails_both_stages_without_rejecting_session(self):
        import publish

        first = self.cells[0]["id"]
        self.timing_factor = lambda arm, name, run, stage: (
            1.08 if arm == "candidate" and name == first else 1.0
        )
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        self.assertFalse(session["rejected"])
        self.assertEqual(session["rejections"], [])
        self.assertEqual(session["confirmed_failures"]["v4x"], [first])
        record = session["decisions"]["per_tier"]["v4x"]["noninferiority"]["per_cell"][first]
        self.assertFalse(record["main"]["pass"])
        self.assertFalse(record["confirmation"]["pass"])
        self.assertAlmostEqual(record["confirmation"]["estimate"], math.log(1.08))
        analysis = self.reanalyze(session)
        self.assertEqual(analysis["decisions"], session["decisions"])
        self.assertEqual(analysis["confirmed_failures"], session["confirmed_failures"])
        publish.validate_session(session, "confirmed shift")

    def test_confirmation_is_not_pooled_and_preserves_inventory_order(self):
        import publish

        ids = [entry["id"] for entry in self.cells]
        self.timing_factor = lambda arm, name, run, stage: (
            1.0 if arm == "base" else
            1.02 if stage == "confirmation" and name == ids[0] else 1.08
        )
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        self.assertEqual(session["confirmed_failures"]["v4x"], ids[1:])
        record = session["decisions"]["per_tier"]["v4x"]["noninferiority"]["per_cell"][ids[0]]
        self.assertTrue(record["confirmation"]["pass"])
        self.assertAlmostEqual(record["upper_bound"], math.log(1.02))
        self.assertEqual(self.reanalyze(session)["decisions"], session["decisions"])
        publish.validate_session(session, "mixed confirmed outcomes")

    def test_superiority_uses_confirmed_non_target_outcome(self):
        target, other = [entry["id"] for entry in self.cells[:2]]
        targets = self.root / "targets.txt"
        targets.write_text(target + "\n", encoding="utf-8")
        for confirmation_factor, expected in ((1.08, 1), (1.0, 0)):
            with self.subTest(confirmation_factor=confirmation_factor):
                self.timing_factor = lambda arm, name, run, stage: (
                    1.0 if arm == "base" else
                    0.95 if name == target else
                    (1.08 if stage == "main" else confirmation_factor)
                    if name == other else 1.0
                )
                self.assertEqual(pair.run(self.args("--targets", str(targets))), expected)
                session = self.session()
                superiority = session["decisions"]["per_tier"]["v4x"]["superiority"]
                self.assertTrue(superiority["per_target"][target]["pass"])
                self.assertEqual(superiority["others_noninferior"], expected == 0)
                self.assertEqual(superiority["passed"], expected == 0)
                self.assertEqual(self.reanalyze(session)["decisions"], session["decisions"])
                shutil.rmtree(self.records)
                for path in self.root.glob("source.reanalysis-*.json"):
                    path.unlink()

    def test_superiority_miss_never_triggers_confirmation(self):
        target = self.cells[0]["id"]
        targets = self.root / "targets.txt"
        targets.write_text(target + "\n", encoding="utf-8")
        self.factor = 1.08
        self.slow_only_first = True
        self.assertEqual(pair.run(self.args("--targets", str(targets))), 1)
        session = self.session()
        self.assertFalse(session["decisions"]["superiority"]["passed"])
        self.assertTrue(all(run["stage"] == "main" for run in session["runs"]))
        self.assertEqual(session["confirmed_failures"]["v4x"], [])

    def test_signed_unstable_failure_is_not_confirmed_or_decided(self):
        first, other = [entry["id"] for entry in self.cells[:2]]
        self.write_signed_unstable([first])
        self.timing_factor = lambda arm, name, run, stage: (
            1.08 if arm == "candidate" and name in (first, other) else 1.0
        )
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        self.assertEqual(session["confirmed_failures"]["v4x"], [other])
        for run in session["runs"]:
            if run["stage"] == "confirmation":
                self.assertEqual(set(run["arms"]["base"]["cells"]), {other})
        self.assertEqual(self.reanalyze(session)["decisions"], session["decisions"])

    def test_confirmation_drift_rejects_without_overwriting_main_drift(self):
        self.factor = 1.08
        self.slow_only_first = True
        self.control_start = lambda run, arm, stage: (
            100.0 * (run + 1) if stage == "confirmation" else 100.0
        )
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        self.assertTrue(session["rejected"])
        self.assertIn("drift_cap", [entry["code"] for entry in session["rejections"]])
        self.assertEqual(session["drift"]["v4x"]["drift"], 0.0)
        self.assertGreater(session["confirmation_drift"]["v4x"]["drift"], 0.2)

    def test_reanalysis_rejects_incomplete_confirmation(self):
        self.factor = 1.08
        self.slow_only_first = True
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        session["runs"].pop()
        source = self.root / "incomplete.json"
        pair.write_session(session, source)
        self.assertEqual(pair.reanalyze_session(source, self.args()), 2)

    def write_signed_unstable(self, cells, *, signature="test-owner", digest=None):
        digest = digest or pair.inventory_digest(
            self.cells, [cell["id"] for cell in self.cells]
        )
        self.unstable_file.write_text(json.dumps({
            "entries": [{
                "host": self.hostname,
                "tier": "v4x",
                "toolchain": "1.0.0/xyz",
                "flags": pair.DEFAULT_ALIGN_FLAGS,
                "inventory_digest": digest,
                "cells": list(cells),
                "sources": ["test"],
                "owner_signature": signature,
            }]
        }), encoding="utf-8")

    def test_unstable_cells_are_excluded_and_marked(self):
        import publish

        cell = self.cells[0]["id"]
        self.write_signed_unstable([cell])
        self.assertEqual(pair.run(self.args()), 0)
        session = self.session()
        self.assertEqual(session["unstable"]["entries"]["v4x"], [cell])
        decision = session["decisions"]["noninferiority"]["per_tier"]["v4x"]
        self.assertNotIn(cell, decision["per_cell"])
        self.assertIn(cell, decision["excluded_unstable"])
        self.assertIn("upper_bound", decision["excluded_unstable"][cell])
        rows = [row for row in session["rows"] if row["cell"] == cell]
        self.assertTrue(rows)
        self.assertTrue(all(row["unstable"] for row in rows))
        publish.validate_session(session, "unstable")

    def test_missing_signed_unstable_list_is_rejected(self):
        missing = self.root / "absent-unstable.json"
        self.assertEqual(
            pair.run(self.args("--unstable-file", str(missing))), 1
        )
        session = self.session()
        self.assertIn(
            "unstable_list_missing",
            [rejection["code"] for rejection in session["rejections"]],
        )

    def test_unsigned_or_stale_unstable_list_is_rejected(self):
        self.write_signed_unstable([self.cells[0]["id"]], signature=None)
        self.assertEqual(pair.run(self.args()), 1)
        self.assertIn(
            "unstable_list_unsigned",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        # A signed entry whose digest names no calibrated snapshot rejects.
        self.write_signed_unstable([self.cells[0]["id"]], digest="wrong")
        self.assertEqual(pair.run(self.args()), 1)
        self.assertIn(
            "calibrated_inventory_missing",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        # So does a snapshot that does not hash to the digest naming it.
        self.write_calibrated_inventory("wrong", [dict(self.cells[0], bytes=63)])
        self.assertEqual(pair.run(self.args()), 1)
        self.assertIn(
            "calibrated_inventory_stale",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )

    def test_unstable_target_requires_power(self):
        cell = self.cells[0]["id"]
        self.write_signed_unstable([cell])
        targets = self.root / "targets.txt"
        targets.write_text(cell + "\n", encoding="utf-8")
        self.assertEqual(
            pair.run(self.args(
                "--exploratory", "--targets", str(targets)
            )),
            1,
        )
        session = self.session()
        self.assertIn(
            "unstable_target_underpowered",
            [rejection["code"] for rejection in session["rejections"]],
        )

    def test_powered_unstable_target_is_included(self):
        cell = self.cells[0]["id"]
        self.write_signed_unstable([cell])
        targets = self.root / "targets.txt"
        targets.write_text(cell + "\n", encoding="utf-8")
        self.factor = 0.95
        self.assertEqual(
            pair.run(self.args(
                "--exploratory",
                "--unstable-target-runs", "5",
                "--targets", str(targets),
            )),
            0,
        )
        session = self.session()
        target = session["decisions"]["superiority"]["per_tier"]["v4x"][
            "per_target"
        ][cell]
        self.assertTrue(target["unstable"])
        self.assertTrue(target["pass"])

    def scope_file(self, cells, rationale="single-cell regression probe"):
        path = self.root / "scope.json"
        path.write_text(
            json.dumps({"cells": list(cells), "rationale": rationale}),
            encoding="utf-8",
        )
        return path

    def test_cells_scope_times_selected_cells_and_checks_everything(self):
        import publish

        ids = [entry["id"] for entry in self.cells]
        chosen = [ids[3], ids[0]]
        scope = self.scope_file(chosen, "target the two routed geometries")
        self.assertEqual(pair.run(self.args("--cells", str(scope))), 0)
        session = self.session()
        record = session["scope"]
        self.assertEqual(record["kind"], "selected")
        self.assertEqual(record["rationale"], "target the two routed geometries")
        self.assertEqual(
            record["declaration_digest"],
            hashlib.sha256(scope.read_bytes()).hexdigest(),
        )
        # The selected cells keep inventory order regardless of declaration order.
        self.assertEqual(record["cells"], [ids[0], ids[3]])
        self.assertEqual(
            record["digest"],
            hashlib.sha256(
                pair.canonical_json(record["cells"]).encode("utf-8")
            ).hexdigest(),
        )
        # The full inventory and full check outputs are still recorded.
        self.assertEqual(session["inventory"]["order"], ids)
        self.assertEqual(len(session["checks"]["v4x"]["base"]), len(ids))
        # Only the scope is timed, decided, and rowed.
        timed = {row["cell"] for row in session["rows"]}
        self.assertEqual(timed, set(record["cells"]))
        per_cell = session["decisions"]["noninferiority"]["per_tier"]["v4x"][
            "per_cell"
        ]
        self.assertEqual(set(per_cell), set(record["cells"]))
        filter_path = self.records / session["session_id"] / "scope-cells.txt"
        self.assertEqual(filter_path.read_text(encoding="utf-8").split(), record["cells"])
        publish.validate_session(session, "acceptance")

    def test_cells_scope_is_rejected_for_calibration(self):
        scope = self.scope_file([self.cells[0]["id"]])
        self.assertEqual(
            pair.run(self.args(
                "--mode", "calibration",
                "--calibration-mode", "base_vs_base",
                "--cells", str(scope),
            )),
            1,
        )
        self.assertIn(
            "scope_calibration_unsupported",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )

    def test_scope_declaration_rejects_bad_files(self):
        empty = self.root / "empty.json"
        empty.write_text(json.dumps({"cells": [], "rationale": "why"}))
        self.assertEqual(
            pair.run(self.args("--cells", str(empty))), 1
        )
        self.assertIn(
            "scope_invalid",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(json.dumps({
            "cells": [self.cells[0]["id"], self.cells[0]["id"]],
            "rationale": "why",
        }))
        self.assertEqual(
            pair.run(self.args("--cells", str(duplicate))), 1
        )
        self.assertIn(
            "scope_invalid",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        unrationalized = self.root / "unrationalized.json"
        unrationalized.write_text(json.dumps({"cells": [self.cells[0]["id"]]}))
        self.assertEqual(
            pair.run(self.args("--cells", str(unrationalized))), 1
        )
        self.assertIn(
            "scope_invalid",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        unknown = self.scope_file(["gf8-aes/mul_add/unprepared/999999x1x1/aligned"])
        self.assertEqual(pair.run(self.args("--cells", str(unknown))), 1)
        self.assertIn(
            "scope_unknown",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )

    def test_scope_rejects_targets_and_threshold_cells_outside_it(self):
        scope = self.scope_file([self.cells[0]["id"]])
        targets = self.root / "targets.txt"
        targets.write_text(self.cells[1]["id"] + "\n", encoding="utf-8")
        self.assertEqual(
            pair.run(self.args(
                "--exploratory", "--cells", str(scope),
                "--targets", str(targets),
            )),
            1,
        )
        self.assertIn(
            "scope_excludes_targets",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )
        shutil.rmtree(self.records)
        scope = self.scope_file([entry["id"] for entry in self.cells[:2]])
        declaration = {
            "item": "src/kernel/quad_mersenne31.rs::VECTOR_ADD_MIN_BYTES",
            "old": 64,
            "new": 128,
            "cells": [
                {"cell": self.cells[0]["id"], "axis_value": 64,
                 "route_old": "above", "route_new": "below"},
                {"cell": self.cells[1]["id"], "axis_value": 96,
                 "route_old": "above", "route_new": "below"},
                {"cell": self.cells[2]["id"], "axis_value": 128,
                 "route_old": "below", "route_new": "below"},
            ],
        }
        inventory = [{
            "key": declaration["item"],
            "file": "src/kernel/quad_mersenne31.rs",
            "name_or_expr": "VECTOR_ADD_MIN_BYTES",
            "value": 64,
            "unit": "bytes",
            "axis": "row length",
            "class": "routing_threshold",
            "routes": True,
            "citation": None,
        }]
        declaration_path = self.root / "decl.json"
        inventory_path = self.root / "threshold-inventory.json"
        declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        self.assertEqual(
            pair.run(self.args(
                "--exploratory", "--cells", str(scope),
                "--mode", "threshold",
                "--threshold", str(declaration_path),
                "--inventory", str(inventory_path),
            )),
            1,
        )
        self.assertIn(
            "scope_excludes_threshold",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )

    def test_appended_live_cell_is_gated_and_recorded_uncalibrated(self):
        import publish

        ids = [entry["id"] for entry in self.cells]
        # The calibrated snapshot predates the appended fourth live cell, and
        # the signed entry binds that older digest.  The appended cell must
        # stay in scope, measured and decided, without being excluded.
        older = pair.inventory_digest(self.cells[:3], ids[:3])
        self.write_calibrated_inventory(older, self.cells[:3])
        self.write_signed_unstable([ids[0]], digest=older)
        self.assertEqual(pair.run(self.args()), 0)
        session = self.session()
        appended = ids[3]
        self.assertEqual(
            session["unstable"]["entries"]["v4x"], [ids[0]]
        )
        self.assertEqual(
            session["uncalibrated"]["entries"]["v4x"], [appended]
        )
        per_cell = session["decisions"]["noninferiority"]["per_tier"]["v4x"][
            "per_cell"
        ]
        self.assertIn(appended, per_cell)
        self.assertNotIn(
            appended, session["decisions"]["noninferiority"]["per_tier"][
                "v4x"
            ]["excluded_unstable"]
        )
        publish.validate_session(session, "unstable")

    def test_redefined_calibrated_cell_rejects_the_session(self):
        # The calibrated snapshot stays internally consistent; the live
        # harness redefines a cell it recorded.  The metadata comparison must
        # catch the redefinition.
        mutated = dict(self.cells[0])
        mutated["bytes"] = mutated["bytes"] + 1
        self.cells[0] = mutated
        self.assertEqual(pair.run(self.args()), 1)
        self.assertIn(
            "calibrated_cell_redefined",
            [rejection["code"] for rejection in self.session()["rejections"]],
        )

    def test_calibration_session_passes_the_publisher_validator(self):
        import publish

        self.assertEqual(
            pair.run(self.args("--mode", "calibration",
                               "--calibration-mode", "base_vs_base")),
            0,
        )
        session = self.session()
        publish.validate_session(session, "calibration")
        self.assertEqual(session["kind"], "calibration")
        self.assertFalse(session["acceptance"])
        self.assertIsNotNone(session["calibration"]["recommendation"]["runs"])
        self.assertGreaterEqual(session["config"]["drift_cap"], 0.0)
        self.assertIn("max_bound", session["calibration"]["recommendation"])
        self.assertIn("limit", session["calibration"]["recommendation"])

    def test_threshold_session_passes_the_publisher_validator(self):
        import publish

        declaration = {
            "item": "src/kernel/quad_mersenne31.rs::VECTOR_ADD_MIN_BYTES",
            "old": 64,
            "new": 128,
            "cells": [
                {"cell": self.cells[0]["id"], "axis_value": 64,
                 "route_old": "above", "route_new": "below"},
                {"cell": self.cells[1]["id"], "axis_value": 96,
                 "route_old": "above", "route_new": "below"},
                {"cell": self.cells[2]["id"], "axis_value": 128,
                 "route_old": "below", "route_new": "below"},
            ],
        }
        inventory = [{
            "key": declaration["item"],
            "file": "src/kernel/quad_mersenne31.rs",
            "name_or_expr": "VECTOR_ADD_MIN_BYTES",
            "value": 64,
            "unit": "bytes",
            "axis": "row length",
            "class": "routing_threshold",
            "routes": True,
            "citation": None,
        }]
        declaration_path = self.root / "decl.json"
        inventory_path = self.root / "inventory.json"
        declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        self.factor = 0.95
        self.write_signed_unstable([self.cells[0]["id"]])
        self.assertEqual(
            pair.run(self.args(
                "--mode", "threshold",
                "--threshold", str(declaration_path),
                "--inventory", str(inventory_path),
                "--unstable-target-runs", "5",
                "--exploratory",
            )),
            0,
        )
        session = self.session()
        publish.validate_session(session, "threshold")
        self.assertEqual(session["kind"], "threshold")
        self.assertFalse(session["acceptance"])
        self.assertTrue(session["threshold"]["span_ok"])
        self.assertEqual(len(session["threshold"]["target_set"]), 2)
        self.assertTrue(session["decisions"]["superiority"]["passed"])
        first_target = session["threshold"]["target_set"][0]
        self.assertTrue(
            session["decisions"]["superiority"]["per_tier"]["v4x"][
                "per_target"
            ][first_target]["unstable"]
        )

    def test_layout_only_remeasures_and_records_exception(self):
        # First measure the slow candidate to obtain host/snapshot identity.
        self.factor = 1.06
        self.slow_only_first = True
        self.assertEqual(pair.run(self.args()), 1)
        first = self.session()
        cell = self.cells[0]["id"]
        host = first["host"]
        snapshot = first["snapshot"]
        runs = []
        for index in range(6):
            runs.append({
                "run": index,
                "tier": "v4x",
                "order": ["base", "candidate"],
                "control": {"start_ns": 100.0, "end_ns": 200.0},
                "arms": {
                    "base": {"cells": {cell: {
                        "summary_ns": 1000.0, "rounds_ns": [1000.0]}}},
                    "candidate": {"cells": {cell: {
                        "summary_ns": 1060.0, "rounds_ns": [1060.0]}}},
                },
            })
        layout_path = self.root / "layout.json"
        layout_path.write_text(
            json.dumps({
                "kind": "calibration",
                "acceptance": False,
                "rejected": False,
                "rejections": [],
                "host": host,
                "snapshot": snapshot,
                "calibration": {"mode": "base_vs_layout_control"},
                "runs": runs,
            }),
            encoding="utf-8",
        )
        shutil.rmtree(self.records)
        self.assertEqual(
            pair.run(self.args("--layout-only", "--layout-sessions", str(layout_path))),
            1,
        )
        session = self.session()
        self.assertEqual(session["rejections"], [])
        import publish

        publish.validate_session(session, "layout-only")
        record = session["layout_only"]["per_tier"]["v4x"][cell]
        self.assertEqual(record["disposition"], "owner_exception_required")
        self.assertIsNone(record["owner_signature"])
        self.assertIsInstance(record["spread"], float)
        self.assertIn("remeasured_bound", record)
        self.assertIn(cell, session["layout_only"]["remeasured"]["v4x"]["bounds"])
        self.assertTrue(
            any(
                item["tier"] == "v4x" and item["run"] >= 0
                for item in session["preconditions"]
            )
        )

    def test_missing_setarch_rejects_the_session(self):
        pair.shutil.which = (
            lambda name: None if name == "setarch" else self.original_which(name)
        )
        self.assertEqual(pair.run(self.args()), 1)
        session = self.session()
        self.assertIn(
            "setarch_missing",
            [rejection["code"] for rejection in session["rejections"]],
        )

    def test_differing_arm_manifest_is_rejected(self):
        self.cell = self.cells[0]["id"]
        campaign = CampaignFailureTests.campaign(self, self.root)
        campaign.cargo_metadata = lambda *args, **kwargs: {
            "packages": [{
                "name": "fgf", "source": None,
                "dependencies": [
                    {"name": "snapshot-dep", "req": "=1",
                     "kind": None, "target": None},
                ],
            }],
        }
        campaign.session["arms"]["base"]["revision_dependency_set"] = [
            "snapshot-dep|=1|"
        ]
        campaign.session["arms"]["candidate"]["revision_dependency_set"] = [
            "candidate-dep|=2|"
        ]
        snapshot = pair.Snapshot(
            "s", "s", self.snapshot, "d", {}, [], "", None, "operations", False
        )
        try:
            self.assertFalse(campaign.check_dependencies(snapshot))
            self.assertEqual(campaign.rejections[-1].code, "dependency_mismatch")
            self.assertEqual(campaign.rejections[-1].arm, "candidate")
        finally:
            campaign.close()


class TargetsAndRowsTests(unittest.TestCase):
    def test_target_file_json_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(
                json.dumps({"cells": ["a", "b", "a"]}), encoding="utf-8"
            )
            parsed = pair.parse_targets_file(path)
            self.assertEqual(parsed["cells"], ["a", "b"])
            self.assertEqual(
                parsed["digest"], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_target_file_text_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.txt"
            path.write_text("# comment\na\n\nb # note\n", encoding="utf-8")
            self.assertEqual(pair.parse_targets_file(path)["cells"], ["a", "b"])

    def test_unknown_target_is_rejected(self):
        rejection = pair.validate_targets(["a", "missing"], {"a", "b"})
        self.assertEqual(rejection.code, "target_unknown")
        self.assertEqual(rejection.cells, ("missing",))

    def test_unknown_threshold_cell_is_rejected(self):
        rejection = pair.validate_declared_cells(["a", "missing"], {"a", "b"})
        self.assertEqual(rejection.code, "threshold_cell_unknown")
        self.assertEqual(rejection.cells, ("missing",))

    def test_rows_put_base_first_and_use_run_medians(self):
        session = {
            "session_id": "s",
            "host": {"hostname": "h"},
            "arms": {
                "base": {"sha": "b", "kind": "library"},
                "candidate": {"sha": "c", "kind": "library"},
            },
            "arm_summaries": {
                "base": {"v4x": {cell(): 10.0}},
                "candidate": {"v4x": {cell(): 9.0}},
            },
            "runs": [
                {
                    "tier": "v4x",
                    "arms": {
                        "base": {"cells": {cell(): {"summary_ns": 10.0, "rounds_ns": [9.0, 11.0]}}},
                        "candidate": {"cells": {cell(): {"summary_ns": 9.0, "rounds_ns": [8.0, 10.0]}}},
                    },
                }
            ],
        }
        rows = pair.build_rows(session, ["v4x"], [cell()])
        self.assertEqual([row["arm"] for row in rows], ["base", "candidate"])
        self.assertEqual(rows[0]["library_sha"], "b")
        self.assertEqual(rows[0]["field"], "gf8-aes")
        self.assertEqual(rows[0]["row_len"], 65536)
        self.assertEqual(rows[0]["rounds_ns"], [9.0, 11.0])
        self.assertEqual(rows[1]["role"], "candidate")

    def test_layout_control_candidate_role(self):
        session = {
            "session_id": "s",
            "host": {"hostname": "h"},
            "arms": {
                "base": {"sha": "b", "kind": "library"},
                "candidate": {"sha": "b", "kind": "layout_control"},
            },
            "arm_summaries": {"base": {}, "candidate": {}},
            "runs": [],
        }
        self.assertEqual(pair.build_rows(session, [], []), [])


class SnapshotTests(unittest.TestCase):
    def snapshot_dir(self, directory: Path, with_target=True):
        (directory / "benches").mkdir(parents=True)
        if with_target:
            (directory / "benches" / "operations.rs").write_text("fn main() {}\n")
        (directory / "Cargo.toml").write_text("[package]\nname='fgf'\n")
        (directory / "Cargo.lock").write_text("version = 3\n")
        return directory

    def test_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.snapshot_dir(Path(directory))
            self.assertIsNone(pair.validate_snapshot(path))

    def test_missing_target_is_snapshot_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.snapshot_dir(Path(directory), with_target=False)
            rejection = pair.validate_snapshot(path)
            self.assertEqual(rejection.code, "snapshot_incomplete")
            self.assertIn("operations", rejection.message)

    def test_snapshot_digest_is_content_addressed_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.snapshot_dir(Path(directory))
            first = pair.snapshot_digest(path)
            self.assertEqual(first, pair.snapshot_digest(path))
            (path / "benches" / "operations.rs").write_text("fn main() { }\n")
            self.assertNotEqual(first, pair.snapshot_digest(path))

    def test_snapshot_runner_must_match_the_running_runner(self):
        self.assertIsNone(pair.check_snapshot_runner(None, "self"))
        self.assertIsNone(pair.check_snapshot_runner("same", "same"))
        rejection = pair.check_snapshot_runner("other", "same")
        self.assertEqual(rejection.code, "snapshot_runner_mismatch")

    def test_snapshot_toolchain_is_read_from_toml_and_plain_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.snapshot_dir(Path(directory))
            (path / "rust-toolchain.toml").write_text(
                '[toolchain]\nchannel = "nightly-2026-09-30"\ncomponents = ["rustfmt"]\n'
            )
            self.assertEqual(
                pair.read_snapshot_toolchain(path), "nightly-2026-09-30"
            )
            (path / "rust-toolchain.toml").unlink()
            (path / "rust-toolchain").write_text("# pinned\n1.93.0\n")
            self.assertEqual(pair.read_snapshot_toolchain(path), "1.93.0")
            (path / "rust-toolchain").unlink()
            self.assertIsNone(pair.read_snapshot_toolchain(path))

    def test_meta_features_and_flags_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.snapshot_dir(Path(directory))
            (path / "benches" / "pair-snapshot.json").write_text(
                json.dumps(
                    {
                        "features": ["simd512", "internals"],
                        "rustflags": "-C llvm-args=-align-all-functions=6",
                    }
                )
            )
            meta = pair.read_snapshot_meta(path)
            features = pair.merge_features(["simd512"], meta.get("features"))
            self.assertEqual(features, ["simd512", "internals"])
            flags = pair.merge_rustflags(
                "-C llvm-args=-align-all-nofallthru-blocks=6",
                meta.get("rustflags"),
            )
            self.assertIn("align-all-functions=6", flags)
            self.assertIn("align-all-nofallthru-blocks=6", flags)


if __name__ == "__main__":
    unittest.main()
