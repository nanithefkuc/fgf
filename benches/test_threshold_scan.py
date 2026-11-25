"""Tests for the threshold scanner."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_scan as scan

REAL_CRATE = Path(__file__).resolve().parents[1]


def crate_root() -> Path | None:
    configured = os.environ.get("FGF_ROOT")
    if configured:
        candidate = Path(configured)
        return candidate if (candidate / "src").is_dir() else None
    return REAL_CRATE if (REAL_CRATE / "src").is_dir() else None


def write_fixture(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


DEMO = """\
//! Fixture module.

pub(crate) const REAL: usize = 8;

#[cfg(test)]
mod tests {
    const TEST_ONLY: usize = 9;

    fn checked(x: usize) -> bool {
        x < 77
    }
}

impl<const P: u64> Widget<P> {
    pub(crate) const WIDTH: usize = 4;

    fn pick<const N: usize>(value: usize) -> bool {
        value < 99
    }
}

#[cfg(test)]
pub(crate) const TEST_CONST: usize = 12;
"""


POSITIONED = """\
pub(crate) const ROW_MIN: usize = 32;

fn chooses(len: usize) -> bool {
    len < ROW_MIN
}
"""


class FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cfg_test_and_const_generic_parameters_are_excluded(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", DEMO)
        write_fixture(self.root, "src/kernel/tests.rs", "pub const LEAK: usize = 1;\n")
        write_fixture(self.root, "src/kernel/tests/extra.rs", "pub const LEAK: usize = 2;\n")
        write_fixture(self.root, "src/ops/tests.rs", "pub const LEAK: usize = 3;\n")

        items = scan.scan(self.root)
        keys = {item["key"] for item in items}

        self.assertIn("src/kernel/demo.rs::REAL", keys)
        self.assertIn("src/kernel/demo.rs::impl Widget::WIDTH", keys)
        self.assertFalse(any("TEST_ONLY" in key for key in keys))
        self.assertFalse(any("TEST_CONST" in key for key in keys))
        self.assertFalse(any("LEAK" in key for key in keys))
        self.assertFalse(any(key.endswith("::P") or key.endswith("::N") for key in keys))

    def test_literals_inside_generic_functions_survive(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", DEMO)
        items = scan.scan(self.root)
        self.assertTrue(any("cmp:value<99" in item["key"] for item in items))
        self.assertFalse(any("cmp:x<77" in item["key"] for item in items))

    def test_keys_are_stable_under_line_shifts(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", DEMO)
        before = [item["key"] for item in scan.scan(self.root)]

        shifted = "\n\n// an inserted leading comment\n\n" + DEMO.replace(
            "pub(crate) const REAL", "\n\npub(crate) const REAL"
        )
        write_fixture(self.root, "src/kernel/demo.rs", shifted)
        after = [item["key"] for item in scan.scan(self.root)]

        self.assertEqual(before, after)

    def test_serialization_is_byte_stable(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", DEMO)
        first = scan.serialize_json(scan.scan(self.root))
        second = scan.serialize_json(scan.scan(self.root))
        self.assertEqual(first, second)

    def test_positions_do_not_change_inventory_bytes(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", POSITIONED)
        items, positions = scan.scan_with_positions(self.root)
        self.assertEqual(scan.serialize_json(items), scan.serialize_json(scan.scan(self.root)))
        non_structural = {item["key"] for item in items if item["class"] != scan.STRUCTURAL}
        self.assertEqual(sorted(non_structural), sorted(positions))

    def test_positions_name_the_comparison_site(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", POSITIONED)
        _, positions = scan.scan_with_positions(self.root)
        payload = positions["src/kernel/demo.rs::ROW_MIN"]
        self.assertEqual(payload["file"], "src/kernel/demo.rs")
        self.assertEqual(payload["line"], 1)
        compare = [use for use in payload["uses"] if use["kind"] == "compare"]
        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["operator"], "<")
        self.assertEqual(compare[0]["line"], 4)
        self.assertEqual(compare[0]["condition_line"], 4)

    def test_cutoff_comparisons_follow_qualified_uses_and_operand_orientation(self) -> None:
        write_fixture(self.root, "src/kernel/routing.rs", """
pub const XOR_INLINE_MAX: usize = 31;
pub const GF8_NIBBLE_MUL_ASSIGN_MIN: usize = 65536;
""")
        write_fixture(self.root, "src/kernel/bytes.rs", """
fn inline(n: usize) -> bool { n <= crate::kernel::routing::XOR_INLINE_MAX }
fn dispatched(n: usize) -> bool { n > crate::kernel::routing::XOR_INLINE_MAX }
fn nibble(n: usize) -> bool { crate::kernel::routing::GF8_NIBBLE_MUL_ASSIGN_MIN <= n }
""")
        items = {item["name_or_expr"]: item for item in scan.scan(self.root)}
        self.assertEqual(items["XOR_INLINE_MAX"]["comparison"], "<=")
        self.assertEqual(items["GF8_NIBBLE_MUL_ASSIGN_MIN"]["comparison"], "<")

    def test_conflicting_cutoff_predicates_are_not_guessed(self) -> None:
        write_fixture(self.root, "src/kernel/routing.rs", """
pub const XOR_INLINE_MAX: usize = 31;
fn first(n: usize) -> bool { n <= XOR_INLINE_MAX }
fn second(n: usize) -> bool { n < XOR_INLINE_MAX }
""")
        item = next(
            item for item in scan.scan(self.root)
            if item["name_or_expr"] == "XOR_INLINE_MAX"
        )
        self.assertNotIn("comparison", item)

    def test_compound_constant_uses_do_not_claim_a_bare_scalar_cutoff(self) -> None:
        write_fixture(self.root, "src/kernel/x86.rs", """
pub const PREFETCH_AHEAD: usize = 512;
pub const ROW_MIN: usize = 32;
fn plain(n: usize) -> bool { n >= PREFETCH_AHEAD }
fn offset(n: usize) -> bool { n >= crate::kernel::x86::PREFETCH_AHEAD + 128 }
fn parenthesized_compound(n: usize) -> bool { n >= (ROW_MIN) + 8 }
""")
        items = {item["name_or_expr"]: item for item in scan.scan(self.root)}
        self.assertNotIn("comparison", items["PREFETCH_AHEAD"])
        self.assertNotIn("comparison", items["ROW_MIN"])

    def test_literal_cutoffs_keep_their_source_comparison(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", """
fn first(n: usize) -> bool { n <= 31 }
fn second(n: usize) -> bool { 31 < n }
""")
        comparisons = [
            item["comparison"] for item in scan.scan(self.root)
            if "::cmp:" in item["key"]
        ]
        self.assertEqual(comparisons, ["<=", "<="])

    def test_equality_and_inequality_cutoffs_select_the_same_changed_points(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", """
fn selected(n: usize) -> bool { n == 4 }
fn other(n: usize) -> bool { 4 != n }
""")
        comparisons = [
            item["comparison"] for item in scan.scan(self.root)
            if "::cmp:" in item["key"]
        ]
        self.assertEqual(comparisons, ["==", "=="])

    def test_positions_shift_with_leading_lines(self) -> None:
        write_fixture(self.root, "src/kernel/demo.rs", POSITIONED)
        _, before = scan.scan_with_positions(self.root)
        write_fixture(self.root, "src/kernel/demo.rs", "\n\n\n" + POSITIONED)
        _, after = scan.scan_with_positions(self.root)
        self.assertEqual(sorted(before), sorted(after))
        self.assertEqual(after["src/kernel/demo.rs::ROW_MIN"]["line"], 4)
        compare_before = [u for u in before["src/kernel/demo.rs::ROW_MIN"]["uses"] if u["kind"] == "compare"]
        compare_after = [u for u in after["src/kernel/demo.rs::ROW_MIN"]["uses"] if u["kind"] == "compare"]
        self.assertEqual(len(compare_before), len(compare_after))
        self.assertEqual(compare_after[0]["line"], compare_before[0]["line"] + 3)


@unittest.skipUnless(crate_root() is not None, "the fgf crate is not available")
class CrateTests(unittest.TestCase):
    def test_representative_items_are_classified(self) -> None:
        root = crate_root()
        assert root is not None
        items = {item["key"]: item for item in scan.scan(root)}

        self.assertEqual(
            items["src/kernel/routing.rs::ALIGN_PEEL_MIN"]["class"],
            scan.ROUTING,
        )
        self.assertEqual(
            items["src/kernel/x86.rs::PREFETCH_MIN"]["class"],
            scan.TUNING,
        )
        modulus = items["src/kernel/x86/mersenne31.rs::P"]
        self.assertEqual(modulus["class"], scan.STRUCTURAL)
        self.assertEqual(modulus["axis"], "modulus")
        self.assertEqual(scan.citation_status(modulus), "structural-exempt")

    def test_every_override_matches_an_item(self) -> None:
        root = crate_root()
        assert root is not None
        keys = {item["key"] for item in scan.scan(root)}
        self.assertEqual(sorted(key for key in scan.OVERRIDES if key not in keys), [])


if __name__ == "__main__":
    unittest.main()
