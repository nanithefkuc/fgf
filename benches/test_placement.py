"""Behavioral checks for placement-distribution comparisons."""

import collections
import copy
import json
import math
import statistics
import unittest

import bench_pair as pair
import placement


def _make_valid_run(run_index=0, arms=("base", "candidate"), cells=("cell",),
                    rounds=3, base_ns=100.0, cand_ns=100.0):
    strata = []
    for point in placement.plan(run_index):
        order = placement.arm_order(arms, run_index, point["index"])
        observed = {}
        controls = {}
        stratum_arms = {}
        for arm_idx, arm in enumerate(arms):
            ns = base_ns if arm == arms[0] else cand_ns
            observed[arm] = {
                "dst_shift": point["dst_shift"],
                "src_shift": point["src_shift"],
                "stack_offset": (128 + arm_idx * 16 + point["stack_pad"]) % 4096,
            }
            controls[arm] = {"start_ns": 100.0 + arm_idx * 0.02, "end_ns": 600.0 + arm_idx * 0.02}
            stratum_arms[arm] = {
                "cells": {
                    cell: {
                        "summary_ns": ns,
                        "rounds_ns": [ns for _ in range(rounds)],
                    }
                    for cell in cells
                }
            }
        strata.append({
            **point,
            "order": order,
            "environment_digest": f"digest-{run_index}-{point['index']}",
            "argv_lengths": {arm: 120 for arm in arms},
            "observed": observed,
            "controls": controls,
            "arms": stratum_arms,
        })
    run_order = placement.arm_order(arms, run_index, 0)
    run_controls = {
        arm: {
            key: statistics.median(p["controls"][arm][key] for p in strata)
            for key in ("start_ns", "end_ns")
        }
        for arm in arms
    }
    return {
        "run": run_index,
        "stage": "main",
        "tier": "v4x",
        "order": run_order,
        "control": run_controls[run_order[0]],
        "controls": run_controls,
        "placements": strata,
        "arms": {
            arm: placement.summarize([p["arms"][arm] for p in strata], cells, rounds)
            for arm in arms
        },
    }


class PlacementTests(unittest.TestCase):
    def test_each_axis_and_relative_phase_covers_the_page_with_equal_weight(self):
        expected_lines = collections.Counter({i: 4 for i in range(0, 4096, 64)})
        for run in (0, 1, 2, 23):
            points = placement.plan(run)
            for axis in ("dst_shift", "src_shift"):
                self.assertEqual(collections.Counter(p[axis] for p in points), expected_lines)
            for src_base in (512, 960, 1408):
                rel = collections.Counter(
                    ((src_base + p["src_shift"]) - (64 + p["dst_shift"])) % 4096
                    for p in points
                )
                self.assertEqual(rel, expected_lines)
            self.assertEqual(sorted(p["stack_pad"] for p in points), list(range(0, 4096, 16)))
        self.assertNotEqual(placement.plan(0), placement.plan(1))
        self.assertNotEqual(placement.plan(1), placement.plan(2))

    def test_arm_order_is_counterbalanced_across_all_placement_axes(self):
        arms = ["base", "candidate"]
        for run in range(6):
            points = placement.plan(run)
            for key_fn in (
                lambda p: p["dst_shift"],
                lambda p: p["src_shift"],
                lambda p: (p["src_shift"] - p["dst_shift"]) % 4096,
            ):
                counts = collections.Counter(
                    (key_fn(p), placement.arm_order(arms, run, p["index"])[0])
                    for p in points
                )
                self.assertEqual(len(counts), 128)
                self.assertEqual(set(counts.values()), {2})
        for pair_idx in range(4):
            pad_orders = collections.Counter()
            for run in (2 * pair_idx, 2 * pair_idx + 1):
                for p in placement.plan(run):
                    first = placement.arm_order(arms, run, p["index"])[0]
                    pad_orders[(p["stack_pad"], first)] += 1
            self.assertEqual(len(pad_orders), 512)
            self.assertEqual(set(pad_orders.values()), {1})

        # Placement-dependent first-process penalty must not bias identical arms across paired runs.
        ratios = []
        for run in range(6):
            arm_pieces = {"base": [], "candidate": []}
            for p in placement.plan(run):
                order = placement.arm_order(arms, run, p["index"])
                penalty = 8.0 if (p["stack_pad"] // 16) % 2 == 0 else 0.0
                for pos, arm in enumerate(order):
                    ns = 100.0 + (penalty if pos == 0 else 0.0)
                    arm_pieces[arm].append({"cells": {"cell": {"summary_ns": ns, "rounds_ns": [ns]}}})
            b = placement.summarize(arm_pieces["base"], ["cell"], 1)["cells"]["cell"]["summary_ns"]
            c = placement.summarize(arm_pieces["candidate"], ["cell"], 1)["cells"]["cell"]["summary_ns"]
            ratios.append(math.log(c / b))
        self.assertEqual(ratios, [0.0] * 6)

    def test_link_address_rotation_does_not_rank_identical_code(self):
        # Model a destination tail colliding with a static after a link move.
        summaries = []
        for static in (0x140, 0x460):
            pieces = []
            for p in placement.plan():
                dst = (64 + p["dst_shift"]) % 4096
                collides = (static - dst) % 4096 < 384
                ns = 110.0 if collides else 100.0
                pieces.append({"cells": {"cell": {"summary_ns": ns, "rounds_ns": [ns]}}})
            summaries.append(placement.summarize(pieces, ["cell"], 1)["cells"]["cell"]["summary_ns"])
        self.assertEqual(summaries, [100.0, 100.0])
        self.assertTrue(pair.evaluate_noninferiority({"cell": [math.log(summaries[1] / summaries[0])] * 5}, .03)["passed"])

    def test_persistent_code_regression_and_invalid_summaries_fail(self):
        pieces = [{"cells": {"cell": {"summary_ns": 105.0, "rounds_ns": [105.0]}}}
                  for _ in placement.plan()]
        summary = placement.summarize(pieces, ["cell"], 1)["cells"]["cell"]["summary_ns"]
        self.assertFalse(pair.evaluate_noninferiority({"cell": [math.log(summary / 100.0)] * 5}, .03)["passed"])
        with self.assertRaises(ValueError):
            placement.summarize(pieces[:-1], ["cell"], 1)

        # Contradictory summary vs retained rounds must be rejected.
        tampered = copy.deepcopy(pieces)
        tampered[0]["cells"]["cell"]["summary_ns"] = 80.0
        with self.assertRaises(ValueError):
            placement.summarize(tampered, ["cell"], 1)

        # Non-finite or non-positive round values must be rejected.
        for bad in (float("nan"), 0.0, -5.0, True):
            broken = copy.deepcopy(pieces)
            broken[0]["cells"]["cell"]["rounds_ns"] = [bad]
            with self.assertRaises(ValueError):
                placement.summarize(broken, ["cell"], 1)

    def test_old_or_mismatched_harness_cannot_supply_a_placement(self):
        point = placement.plan()[1]
        with self.assertRaises(ValueError):
            placement.validate_record("BACKEND\tv3\n", point)
        with self.assertRaises(ValueError):
            placement.validate_record("PLACEMENT\t[]\n", point)
        with self.assertRaises(ValueError):
            placement.validate_record("PLACEMENT\t\"bad\"\n", point)
        record = {
            "dst_shift": (point["dst_shift"] + 64) % 4096,
            "src_shift": point["src_shift"],
            "stack_offset": 123,
        }
        with self.assertRaises(ValueError):
            placement.validate_record("PLACEMENT\t" + json.dumps(record), point)
        record["dst_shift"] = point["dst_shift"]
        self.assertEqual(placement.validate_record("PLACEMENT\t" + json.dumps(record), point), record)

    def test_missing_strata_tampered_evidence_and_biased_stacks_are_rejected(self):
        record = _make_valid_run(run_index=0)
        placement.validate_run(record, 3, expected_arms=("base", "candidate"), expected_cells=("cell",))
        for mutate in (
            lambda r: r["placements"].pop(),
            lambda r: r["placements"][1]["observed"]["base"].update(stack_offset=128),
            lambda r: [p["observed"]["base"].update(stack_offset=p["observed"]["base"]["stack_offset"] + 4096) for p in r["placements"]],
            lambda r: r["placements"][5]["observed"]["candidate"].update(
                stack_offset=(r["placements"][5]["observed"]["candidate"]["stack_offset"] + 32) % 4096
            ),
            lambda r: r["arms"]["base"]["cells"]["cell"].update(summary_ns=99.0),
            lambda r: r["placements"][0]["arms"]["base"]["cells"]["cell"].update(summary_ns=80.0),
            lambda r: r["controls"]["base"].update(start_ns=50.0),
            lambda r: r["placements"][0].update(order=["candidate", "base"]),
            lambda r: r["placements"][0]["argv_lengths"].update(candidate=121),
            lambda r: r["placements"][0].update(environment_digest=""),
        ):
            broken = copy.deepcopy(record)
            mutate(broken)
            with self.assertRaises(ValueError):
                placement.validate_run(broken, 3, expected_arms=("base", "candidate"), expected_cells=("cell",))

    def test_start_controls_reads_per_stratum_controls_for_drift(self):
        run = _make_valid_run(run_index=0)
        # Corrupt half of the strata inside one run; per-stratum drift must see it.
        for p in run["placements"][:128]:
            p["controls"]["base"]["start_ns"] = 130.0
        samples = pair.start_controls([run])
        self.assertEqual(len(samples), placement.COUNT * 2)
        stats = pair.drift_stats(samples)
        self.assertGreater(stats["drift"], 0.05)


if __name__ == "__main__":
    unittest.main()
