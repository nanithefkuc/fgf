"""Behavioral tests for `api/surface.py` over fixture crates.

The fixtures compile with the pinned nightly named by `API_NIGHTLY`, so the
checker reads the same rustdoc JSON format the inventory recipes produce.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "surface.py"
NIGHTLY = os.environ.get("API_NIGHTLY", "nightly-2026-09-30")


def rustdoc_json(source: Path, out: Path) -> Path:
    subprocess.run(
        [
            "rustdoc", f"+{NIGHTLY}", "--edition", "2024", "--crate-type", "lib",
            "--crate-name", "fixture", "-Z", "unstable-options", "--output-format", "json",
            "--document-hidden-items", "-o", str(out), str(source),
        ],
        check=True,
    )
    return out / "fixture.json"


def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], input=stdin, capture_output=True, text=True
    )


class OnePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        cls.duplicates = rustdoc_json(HERE / "fixtures" / "duplicates.rs", out / "duplicates")
        cls.clean = rustdoc_json(HERE / "fixtures" / "clean.rs", out / "clean")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_reports_exactly_the_duplicate_paths(self) -> None:
        result = run("paths", str(self.duplicates), "--mode", "report")
        self.assertEqual(result.returncode, 0, result.stderr)
        violations = sorted(line for line in result.stdout.splitlines() if not line.startswith("surface:"))
        self.assertEqual(
            violations,
            sorted(
                [
                    "duplicate item (2 paths): fixture::Direct | fixture::Renamed",
                    "duplicate item (2 paths): fixture::alias::Deep | fixture::outer::inner::Deep",
                    "duplicate module (2 paths): fixture::alias | fixture::outer::inner",
                    "duplicate item (2 paths): fixture::Cell | fixture::ext::Cell",
                    "duplicate item (2 paths): fixture::Hidden | fixture::hidden_again::Hidden",
                    "duplicate item (2 paths): fixture::globbed::Globbed | fixture::regathered::Globbed",
                    "glob re-export: fixture::regathered::super::globbed::*",
                ]
            ),
        )

    def test_counts_owned_and_external_items(self) -> None:
        result = run("paths", str(self.duplicates), "--mode", "report")
        self.assertIn(
            "surface: 7 crate-owned items through 11 paths; 1 external items through 2 paths",
            result.stdout,
        )

    def test_enforce_fails_on_violations(self) -> None:
        self.assertEqual(run("paths", str(self.duplicates), "--mode", "enforce").returncode, 1)

    def test_enforce_passes_without_violations(self) -> None:
        result = run("paths", str(self.clean), "--mode", "enforce")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("surface: 0 one-path violation(s)", result.stdout)


class Projection(unittest.TestCase):
    def test_drops_lines_naming_internals_and_items_of_unstable_impls(self) -> None:
        stdin = "\n".join(
            [
                "pub fn fgf::ops::mul_add(&mut [u8])",
                "pub fn fgf::internals::kernel::xor(&mut [u8], &[u8])",
                "impl<C: core::marker::Copy> fgf::internals::kernel::Matrix<C> for [(&[C], &[u8])]",
                "pub fn [(&[C], &[u8])]::coefficient(&self, usize, usize) -> C",
                "pub fn [(&[C], &[u8])]::len(&self) -> usize",
                "impl core::clone::Clone for fgf::Coeff",
                "pub fn fgf::Coeff::clone(&self) -> fgf::Coeff",
                "",
            ]
        )
        result = run("project", stdin=stdin)
        self.assertEqual(
            result.stdout,
            "pub fn fgf::ops::mul_add(&mut [u8])\n"
            "impl core::clone::Clone for fgf::Coeff\n"
            "pub fn fgf::Coeff::clone(&self) -> fgf::Coeff\n",
        )


class TargetEquality(unittest.TestCase):
    def test_names_configuration_and_differing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "default-x86_64-unknown-linux-gnu.txt"
            b = Path(tmp) / "default-wasm32-unknown-unknown.txt"
            a.write_text("pub fn fgf::backend()\npub fn fgf::x86_only()\n")
            b.write_text("pub fn fgf::backend()\n")
            result = run("same", "default", str(a), str(b))
            self.assertEqual(result.returncode, 1)
            self.assertIn("configuration default", result.stderr)
            self.assertIn("default-wasm32-unknown-unknown.txt", result.stderr)

    def test_identical_targets_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for name in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"):
                path = Path(tmp) / f"all-{name}.txt"
                path.write_text("pub fn fgf::backend()\n")
                files.append(str(path))
            self.assertEqual(run("same", "all", *files).returncode, 0)


if __name__ == "__main__":
    unittest.main()
