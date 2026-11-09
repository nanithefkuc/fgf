"""Behavioral checks for operation-time route observations."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import route_probe


class RouteObservationTests(unittest.TestCase):
    def test_only_operation_time_kernel_counts_are_observed(self):
        startup = [
            {"path": "fgf::kernel::routing::choose", "file": "src/kernel/routing.rs", "count": 8},
            {"path": "fgf::kernel::prepare", "file": "src/kernel.rs", "count": 3},
        ]
        functions = [
            {"path": "fgf::kernel::routing::choose", "file": "src/kernel/routing.rs", "count": 9},
            {"path": "fgf::kernel::prepare", "file": "src/kernel.rs", "count": 3},
            {"path": "fgf::kernel::scalar::mul", "file": "src/kernel/scalar.rs", "count": 4},
            {"path": "fgf::ops::mul", "file": "src/ops/single.rs", "count": 1},
            {"path": "other::kernel::mul", "file": "src/kernel/scalar.rs", "count": 7},
        ]
        self.assertEqual(route_probe.observed_routes(functions, startup), [
            {"path": "fgf::kernel::routing::choose", "file": "src/kernel/routing.rs", "count": 1},
            {"path": "fgf::kernel::scalar::mul", "file": "src/kernel/scalar.rs", "count": 4},
        ])

    def test_source_file_identity_prevents_cross_file_subtraction(self):
        startup = [{"path": "fgf::kernel::mul", "file": "src/kernel/other.rs", "count": 9}]
        functions = [
            {"path": "fgf::kernel::mul", "file": "src/kernel.rs", "count": 2},
            {"path": "fgf::kernel::mul", "file": "src/ops/single.rs", "count": 4},
        ]
        self.assertEqual(route_probe.observed_routes(functions, startup), [
            {"path": "fgf::kernel::mul", "file": "src/kernel.rs", "count": 2},
        ])


if __name__ == "__main__":
    unittest.main()
