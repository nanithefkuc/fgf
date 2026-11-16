"""Behavioral checks for placement-distribution comparisons."""

import collections
import json
import math
import unittest

import bench_pair as pair
import placement


class PlacementTests(unittest.TestCase):
    def test_each_axis_covers_the_page_with_equal_weight(self):
        for run in (0, 1, 23):
            points = placement.plan(run)
            for axis in ("dst_shift", "src_shift"):
                self.assertEqual(collections.Counter(p[axis] for p in points),
                                 collections.Counter({i: 4 for i in range(0, 4096, 64)}))
            self.assertEqual(sorted(p["stack_pad"] for p in points), list(range(0, 4096, 16)))
        self.assertNotEqual(placement.plan(0), placement.plan(1))

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

    def test_persistent_code_regression_still_fails(self):
        pieces = [{"cells": {"cell": {"summary_ns": 105.0, "rounds_ns": [105.0]}}}
                  for _ in placement.plan()]
        summary = placement.summarize(pieces, ["cell"], 1)["cells"]["cell"]["summary_ns"]
        self.assertFalse(pair.evaluate_noninferiority({"cell": [math.log(summary / 100.0)] * 5}, .03)["passed"])
        with self.assertRaises(ValueError):
            placement.summarize(pieces[:-1], ["cell"], 1)
        pieces[0]["cells"]["cell"]["summary_ns"] = float("nan")
        with self.assertRaises(ValueError):
            placement.summarize(pieces, ["cell"], 1)

    def test_old_or_mismatched_harness_cannot_supply_a_placement(self):
        point = placement.plan()[1]
        with self.assertRaises(ValueError):
            placement.validate_record("BACKEND\tv3\n", point)
        record = {"dst_shift": 0, "src_shift": point["src_shift"], "stack_offset": 123}
        with self.assertRaises(ValueError):
            placement.validate_record("PLACEMENT\t" + json.dumps(record), point)
        record["dst_shift"] = point["dst_shift"]
        self.assertEqual(placement.validate_record("PLACEMENT\t" + json.dumps(record), point), record)

    def test_missing_strata_and_biased_stack_samples_are_rejected(self):
        import copy
        strata = []
        for point in placement.plan():
            arm = {"cells": {"cell": {"summary_ns": 100.0, "rounds_ns": [100.0]}}}
            strata.append({
                **point,
                "observed": {"base": {
                    "dst_shift": point["dst_shift"], "src_shift": point["src_shift"],
                    "stack_offset": (123 + point["stack_pad"]) % 4096,
                }},
                "arms": {"base": arm},
            })
        record = {
            "run": 0, "placements": strata,
            "arms": {"base": placement.summarize([p["arms"]["base"] for p in strata], ["cell"], 1)},
        }
        placement.validate_run(record, 1)
        for mutate in (
            lambda r: r["placements"].pop(),
            lambda r: r["placements"][1]["observed"]["base"].update(stack_offset=123),
            lambda r: r["arms"]["base"]["cells"]["cell"].update(summary_ns=99.0),
        ):
            broken = copy.deepcopy(record)
            mutate(broken)
            with self.assertRaises(ValueError):
                placement.validate_run(broken, 1)


if __name__ == "__main__":
    unittest.main()
