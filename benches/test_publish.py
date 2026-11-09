#!/usr/bin/env python3
"""Tests for the benchmark record publisher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_pair
import bench_baseline
import publish
from test_bench_baseline import descriptive_session

CELL_A = "gf8-aes/mul_add/unprepared/65536x1x1/aligned"
CELL_B = "m31/mul_into/prepared/4096x1x1/aligned"
BASE_SHA = "a" * 40
CANDIDATE_SHA = "b" * 40
HOSTS = [
    {"host": "willow", "name": "Willow Cove", "available": True},
    {"host": "golden", "name": "Golden Cove", "available": False},
]


def _split_cell(cell: str) -> tuple[str, str, str, str, str, int, int, int]:
    field, operation, prep, geometry, alignment = cell.split("/")
    row_len, rows, sources = (int(part) for part in geometry.split("x"))
    return field, operation, prep, alignment, cell, row_len, rows, sources


def make_row(cell: str = CELL_A, *, host: str = "willow", tier: str = "v4x",
             arm: str = "candidate", summary_ns: float = 100.0,
             session_id: str = "20261107T000000Z-willow") -> dict:
    field, operation, prep, alignment, _, row_len, rows, sources = _split_cell(cell)
    sha = CANDIDATE_SHA if arm == "candidate" else BASE_SHA
    return {
        "host": host,
        "tier": tier,
        "field": field,
        "operation": operation,
        "prep": prep,
        "row_len": row_len,
        "rows": rows,
        "sources": sources,
        "alignment": alignment,
        "cell": cell,
        "arm": arm,
        "library_sha": sha,
        "session_id": session_id,
        "summary_ns": summary_ns,
        "rounds_ns": [summary_ns, summary_ns * 1.01, summary_ns * 0.99],
        "role": "baseline" if arm == "base" else (
            "layout_control" if arm == "layout_control" else "candidate"),
        "unstable": False,
    }


def make_session(session_id: str = "20261107T000000Z-willow", *,
                 kind: str = "acceptance", host: str = "willow",
                 tier: str = "v4x", rows: list[dict] | None = None,
                 calibration: dict | None = None,
                 threshold: dict | None = None,
                 layout_only: dict | None = None,
                 decisions: dict | None = None,
                 rejected: bool = False,
                 base_sha: str = BASE_SHA,
                 candidate_sha: str = CANDIDATE_SHA,
                 candidate_kind: str = "library") -> dict:
    if rows is None:
        rows = [
            make_row(arm="base", summary_ns=101.0, session_id=session_id),
            make_row(arm="candidate", summary_ns=100.0, session_id=session_id),
        ]
    for row in rows:
        row["session_id"] = session_id
        row["host"] = host
        row["tier"] = tier
        row["library_sha"] = (base_sha if row["arm"] == "base"
                              else candidate_sha)
    session = {
        "method_version": "fgf-pair/3",
        "drift_method": "p90_minus_p10_over_median",
        "session_id": session_id,
        "created_utc": "2026-11-07T00:00:00Z",
        "kind": kind,
        "acceptance": kind == "acceptance",
        "reused_calibration": False,
        "rejected": rejected,
        "host": {
            "hostname": host,
            "cpu_model": "Example CPU",
            "os": "Linux",
            "arch": "x86_64",
            "toolchain": "rustc 1.93.0",
            "governor": "performance",
            "smt_active": True,
            "thread_siblings": [1],
            "pinned_core": 0,
            "affinity": "0",
            "kernel": "6.0.0",
            "measured_utc": "2026-11-07T00:00:00Z",
            "taskset": "taskset -c 0",
            "campaign_lock": "/tmp/fgf-benchmark-campaign.lock",
            "toolchain_id": "1.93.0/abcdef0",
        },
        "snapshot": {
            "source": "git",
            "resolved": "deadbeef",
            "digest": "c" * 64,
            "features": ["simd512"],
            "rustflags": " ".join(publish.ALIGNMENT_FLAGS),
            "toolchain": "rustc 1.93.0",
            "bench": "operations",
            "no_default_features": False,
            "cargo_metadata_seed": None,
            "runner_digest": "d" * 64,
            "runner_self_digest": "d" * 64,
            "runner_matches_snapshot": True,
        },
        "arms": {
            "base": {"kind": "library", "repo": ".", "sha": base_sha,
                     "metadata_seed": None},
            "candidate": {"kind": candidate_kind, "repo": ".", "sha": candidate_sha,
                          "metadata_seed": None},
        },
        "config": {
            "runs": 5,
            "rounds": 3,
            "delta": 0.03,
            "gamma": 0.03,
            "drift_cap": 0.01,
            "features": ["simd512"],
            "rustflags": " ".join(publish.ALIGNMENT_FLAGS),
            "toolchain": "rustc 1.93.0",
            "smt_window": 0.2,
            "smt_busy_threshold": 0.05,
            "calibrated": kind != "calibration",
            "calibration_binding": {},
            "targets_declaration": None,
            "threshold_declaration": None,
        },
        "tiers": [{
            "requested": tier,
            "reported": tier,
            "field_backends": {"gf8-aes": "test", "m31": "test"},
            "candidate_reported": tier,
            "candidate_field_backends": {"gf8-aes": "test", "m31": "test"},
            "field_backends_equal": True,
            "ok": True,
        }],
        "runs": [{
            "run": 0,
            "order": ["base", "candidate"],
            "tier": tier,
            "control": {"start_ns": 1, "end_ns": 2},
            "controls": {
                "base": {"start_ns": 1, "end_ns": 2},
                "candidate": {"start_ns": 3, "end_ns": 4},
            },
            "arms": {"base": {"cells": {}}, "candidate": {"cells": {}}},
        }],
        "arm_summaries": {"base": {}, "candidate": {}},
        "inventory": {"base": [], "candidate": [], "order": [], "equal": True},
        "checks": {},
        "route_diagnostics": {},
        "drift": {},
        "decisions": decisions if decisions is not None else {
            "noninferiority": {
                "passed": True,
                "per_tier": {
                    tier: {"passed": True, "delta": 0.03, "limit": 0.0296,
                           "per_cell": {}, "failed_cells": []},
                },
            },
            "superiority": {
                "declared": False,
                "passed": None,
                "per_tier": {
                    tier: {"declared": False, "gamma": 0.03, "delta": 0.03,
                           "limit": -0.0305, "targets": [], "per_target": {},
                           "failed_targets": [], "others_noninferior": True,
                           "other_bounds": {}, "passed": None},
                },
            },
            "per_tier": {
                tier: {
                    "noninferiority": {
                        "passed": True, "delta": 0.03, "limit": 0.0296,
                        "per_cell": {}, "failed_cells": [],
                    },
                    "superiority": {
                        "declared": False, "gamma": 0.03, "delta": 0.03,
                        "limit": -0.0305, "targets": [], "per_target": {},
                        "failed_targets": [], "others_noninferior": True,
                        "other_bounds": {}, "passed": None,
                    },
                },
            },
        },
        "rejections": [],
        "rows": rows,
        "layout_only": layout_only,
        "calibration": calibration,
        "threshold": threshold,
    }
    return session


def make_runner_session(session_id: str = "20261107T050000Z-willow", *,
                        tier: str = "v4x", cell: str = CELL_A,
                        base_ns: float = 101.0,
                        candidate_ns: float = 100.0,
                        confirmation_ns: float | None = None,
                        targets: list[str] | None = None) -> dict:
    args = argparse.Namespace(
        mode="acceptance", runs="5", rounds="3", delta="0.03", gamma="0.03",
        drift_cap="0.01", features="simd512",
        rustflags=" ".join(publish.ALIGNMENT_FLAGS), toolchain="",
    )
    session = bench_pair.base_session(args, session_id)
    session["config"].update({
        "runs": 5, "rounds": 3, "delta": 0.03, "gamma": 0.03,
        "drift_cap": 0.01, "features": ["simd512"],
        "rustflags": " ".join(publish.ALIGNMENT_FLAGS),
        "toolchain": "rustc 1.93.0", "calibrated": True,
        "calibration_binding": {},
    })
    session["host"].update({
        "hostname": "willow", "cpu_model": "Example CPU", "os": "Linux",
        "arch": "x86_64", "kernel": "6.0.0", "toolchain": "rustc 1.93.0",
        "toolchain_id": "1.93.0/abcdef0",
        "measured_utc": "2026-11-07T00:00:00Z", "governor": "performance",
        "smt_active": True, "thread_siblings": [1], "pinned_core": 0,
        "affinity": "0", "taskset": "taskset -c 0",
        "campaign_lock": "/tmp/fgf-benchmark-campaign.lock",
    })
    session["snapshot"].update({
        "source": "git", "resolved": "deadbeef", "digest": "c" * 64,
        "features": ["simd512"], "rustflags": " ".join(publish.ALIGNMENT_FLAGS),
        "toolchain": "rustc 1.93.0", "bench": "operations",
        "no_default_features": False, "runner_digest": "d" * 64,
        "runner_self_digest": "d" * 64, "runner_matches_snapshot": True,
    })
    session["arms"] = {
        "base": {"kind": "library", "repo": ".", "sha": BASE_SHA,
                 "metadata_seed": None},
        "candidate": {"kind": "library", "repo": ".", "sha": CANDIDATE_SHA,
                      "metadata_seed": None},
    }
    session["tiers"] = [{
        "requested": tier, "reported": tier,
        "field_backends": {"gf8-aes": "test"},
        "candidate_reported": tier,
        "candidate_field_backends": {"gf8-aes": "test"},
        "field_backends_equal": True, "ok": True,
    }]
    session["runs"] = [{
        "run": index, "stage": "main", "tier": tier,
        "order": bench_pair.balanced_order(index),
        "control": {"start_ns": 1, "end_ns": 2},
        "controls": {"base": {"start_ns": 1, "end_ns": 2},
                     "candidate": {"start_ns": 3, "end_ns": 4}},
        "arms": {
            "base": {"cells": {cell: {"summary_ns": base_ns,
                                      "rounds_ns": [base_ns]}}},
            "candidate": {"cells": {cell: {"summary_ns": candidate_ns,
                                           "rounds_ns": [candidate_ns]}}},
        },
    } for index in range(5)]
    if confirmation_ns is not None:
        confirmation = copy.deepcopy(session["runs"])
        for run in confirmation:
            run["stage"] = "confirmation"
            run["arms"]["candidate"]["cells"][cell] = {
                "summary_ns": confirmation_ns, "rounds_ns": [confirmation_ns],
            }
        session["runs"].extend(confirmation)
    session["arm_summaries"] = {
        "base": {tier: {cell: base_ns}},
        "candidate": {tier: {cell: candidate_ns}},
    }
    decision = bench_pair.recorded_staged_decision(session, tier, targets)
    session["decisions"] = bench_pair.publishable_decisions({tier: decision})
    session["confirmed_failures"] = {tier: decision["confirmed_failures"]}
    session["rows"] = bench_pair.build_rows(session, [tier], [cell])
    return session


def write_manifest(root: Path, sessions: list[dict], *,
                   hosts: list[dict] | None = None,
                   competitors: list[dict] | None = None) -> Path:
    records = root / "records"
    entries = []
    for session in sessions:
        directory = records / "pair" / session["session_id"]
        directory.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(session, indent=1) + "\n").encode("utf-8")
        path = directory / "session.json"
        path.write_bytes(content)
        entries.append({
            "path": f"pair/{session['session_id']}",
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    manifest = {
        "schema": 1,
        "metadata": {
            "hosts": HOSTS if hosts is None else hosts,
            "tiers": ["v4x", "v3_gfni_crypto", "v3", "v2", "scalar"],
        },
        "sessions": entries,
        "competitors": competitors or [],
    }
    manifest_path = root / "benchmarks" / "sessions.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_publish(root: Path, manifest: Path, *extra: str) -> tuple[int, str, str]:
    output = root / "out"
    args = ["--manifest", str(manifest), "--records", str(root / "records"),
            "--output", str(output), *extra]
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = publish.main(args)
    return code, stdout.getvalue(), stderr.getvalue()


def read_page(root: Path, relative: str) -> str:
    return (root / "out" / relative).read_text(encoding="utf-8")


class PublisherTests(unittest.TestCase):
    def test_empty_manifest_and_host_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertLess(page.index("Willow Cove"), page.index("Golden Cove"))
            self.assertIn("| Golden Cove | no | - | - | - | - |", page)
            verify, _, verify_stderr = run_publish(root, manifest, "--verify")
            self.assertEqual(verify, 0, verify_stderr)

    def test_unstable_marker_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_row(arm="base", summary_ns=101.0)
            candidate = make_row(arm="candidate", summary_ns=100.0)
            base["unstable"] = True
            candidate["unstable"] = True
            session = make_session(rows=[base, candidate])
            publish.validate_session(session, "unstable fixture")
            manifest = write_manifest(root, [session])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertIn("| yes |", page)

    def test_anchor_stability_across_sessions(self) -> None:
        anchor = f"willow/v4x/{CELL_A}"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_session(rows=[make_row(summary_ns=100.0)])
            manifest = write_manifest(root, [first])
            self.assertEqual(run_publish(root, manifest)[0], 0)
            page = read_page(root, "BENCHMARKS.md")
            ids = re.findall(r'<a id="([^"]+)"></a>', page)
            self.assertEqual(ids, [anchor])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            second = make_session(rows=[make_row(summary_ns=50.0)])
            manifest = write_manifest(root, [second])
            self.assertEqual(run_publish(root, manifest)[0], 0)
            page = read_page(root, "BENCHMARKS.md")
            ids = re.findall(r'<a id="([^"]+)"></a>', page)
            self.assertEqual(ids, [anchor])
            self.assertIn("`" + CELL_A + "`", page)

    def test_append_and_latest_current_semantics(self) -> None:
        first = make_session("20261107T000000Z-willow", rows=[
            make_row(arm="base", summary_ns=110.0),
            make_row(arm="candidate", summary_ns=100.0),
        ])
        second = make_session("20261107T010000Z-willow", rows=[
            make_row(arm="base", summary_ns=60.0),
            make_row(arm="candidate", summary_ns=50.0),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [first, second])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            current = read_page(root, "BENCHMARKS.md")
            history = read_page(root, "benchmarks/history.md")
            self.assertEqual(current.count(f'<a id="willow/v4x/{CELL_A}"></a>'), 1)
            self.assertIn("50.0", current)
            self.assertNotIn("100.0", current)
            self.assertLess(history.index(first["session_id"]),
                            history.index(second["session_id"]))
            self.assertIn("100", history)
            self.assertIn("50.0", history)
            self.assertIn(f"`{CANDIDATE_SHA}`", history)
            self.assertNotIn(f"`{BASE_SHA}`", history)

    def test_shared_library_sha_is_labelled_baseline(self) -> None:
        baseline = make_session(
            "20261107T000000Z-willow",
            base_sha=BASE_SHA, candidate_sha=BASE_SHA,
            rows=[make_row(arm="base", summary_ns=100.0),
                  make_row(arm="candidate", summary_ns=100.0)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [baseline])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertIn("| baseline |", page)
            self.assertNotIn("| candidate |", page)

    def test_digit_rule(self) -> None:
        self.assertEqual(publish.table_digits([3.2, 32.0]), 3)
        self.assertEqual(publish.fmt_value(3.2, 3), "3.20")
        self.assertEqual(publish.fmt_value(32.0, 3), "32.0")
        self.assertEqual(publish.fmt_value(None, 3), "-")
        self.assertEqual(publish.fmt_ratio(1.0), "1.0000")
        rows = [
            make_row(summary_ns=3.2),
            make_row(CELL_B, summary_ns=32.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [make_session(rows=rows)])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertIn("| 3.20 |", page)
            self.assertIn("| 32.0 |", page)

    def test_verify_fails_on_edited_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [make_session()])
            self.assertEqual(run_publish(root, manifest)[0], 0)
            path = root / "out" / "BENCHMARKS.md"
            path.write_bytes(path.read_bytes() + b"edited\n")
            code, _, stderr = run_publish(root, manifest, "--verify")
            self.assertNotEqual(code, 0)
            self.assertIn("changed", stderr)

    def test_verify_fails_on_manifest_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [make_session()])
            self.assertEqual(run_publish(root, manifest)[0], 0)
            path = root / "records" / "pair" / "20261107T000000Z-willow" / "session.json"
            path.write_bytes(path.read_bytes() + b"edited\n")
            code, _, stderr = run_publish(root, manifest, "--verify")
            self.assertNotEqual(code, 0)
            self.assertIn("digest", stderr)

    def test_calibration_rows_in_separate_table(self) -> None:
        calibration = {
            "mode": "base_vs_base",
            "recommendation": {
                "runs": 5,
                "drift_cap": 0.01,
                "max_bound": 0.005,
                "max_bound_signed": 0.005,
                "limit": 0.015,
                "drift_safety_factor": 1.5,
                "per_tier": {
                    "v4x": {
                        "runs": 5,
                        "limit": 0.015,
                        "max_bound": 0.005,
                        "bounds": {},
                    },
                },
            },
            "binding": {"hosts": ["willow"], "tiers": ["v4x"]},
        }
        calibrated = make_session(
            "20261107T020000Z-willow", kind="calibration",
            rows=[make_row(CELL_B, summary_ns=20.0)],
            calibration=calibration,
            base_sha=BASE_SHA, candidate_sha=BASE_SHA)
        accepted = make_session(
            "20261107T030000Z-willow", rows=[make_row(CELL_A, summary_ns=10.0)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [accepted, calibrated])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            benchmark = read_page(root, "BENCHMARKS.md")
            page = read_page(root, "benchmarks/calibration.md")
            self.assertNotIn(CELL_B, benchmark)
            self.assertIn(CELL_B, page)
            self.assertIn("## Measurements", page)
            self.assertIn("## Recommendations", page)
            self.assertIn("base_vs_base", page)

    def test_layout_only_per_tier_is_tolerated(self) -> None:
        layout_only = {
            "asserted": True,
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": 0.01,
                        "spread": 0.02,
                        "disposition": "owner_exception_required",
                        "owner_signature": "signed",
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(
                root, [make_session(layout_only=layout_only)])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)

    def test_null_owner_exception_is_tolerated(self) -> None:
        layout_only = {
            "asserted": True,
            "control": {
                "revision": "deadbeef",
                "metadata_seed": "fgf-pair-layout-control",
                "runs": 5,
                "rounds": 3,
            },
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": 0.01,
                        "calibration_spread": 0.02,
                        "remeasured_bound": 0.015,
                        "spread": 0.02,
                        "disposition": "owner_exception_required",
                        "owner_signature": None,
                    },
                },
            },
            "remeasured": {
                "v4x": {"bounds": {}, "controls": {}, "drift": {}, "runs": 5},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(
                root, [make_session(layout_only=layout_only)])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)

    def test_unavailable_layout_spread_is_tolerated(self) -> None:
        layout_only = {
            "asserted": True,
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": 0.01,
                        "spread": None,
                        "disposition": "spread_unavailable",
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(
                root, [make_session(layout_only=layout_only)])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)

    def test_wrong_drift_method_is_rejected(self) -> None:
        session = make_session()
        session["drift_method"] = "max_minus_min_over_median"
        with self.assertRaises(publish.PublishError):
            publish.validate_session(session, "wrong drift")

    def test_drift_method_must_match_record_version(self) -> None:
        for version, method in (
            ("fgf-pair/3", "p90_minus_p10_over_median"),
            ("fgf-pair/4", "p90_minus_p10_over_median"),
            ("fgf-pair/5", "iqr_over_median"),
        ):
            with self.subTest(version=version):
                session = make_session() if version.endswith("/3") else make_runner_session()
                session["method_version"] = version
                session["drift_method"] = method
                publish.validate_session(session, version)
                session["drift_method"] = (
                    "p90_minus_p10_over_median"
                    if method == "iqr_over_median" else "iqr_over_median"
                )
                with self.assertRaises(publish.PublishError):
                    publish.validate_session(session, "mislabelled drift")

    def test_failed_decision_is_rejected(self) -> None:
        decisions = {
            "noninferiority": {
                "passed": False,
                "per_tier": {
                    "v4x": {"passed": False, "delta": 0.03, "limit": 0.0296,
                            "per_cell": {}, "failed_cells": [CELL_A]},
                },
            },
            "superiority": {
                "declared": False, "passed": None,
                "per_tier": {
                    "v4x": {"declared": False, "gamma": 0.03, "delta": 0.03,
                            "limit": -0.0305, "targets": [], "per_target": {},
                            "failed_targets": [], "others_noninferior": True,
                            "other_bounds": {}, "passed": None},
                },
            },
            "per_tier": {
                "v4x": {
                    "noninferiority": {
                        "passed": False, "delta": 0.03, "limit": 0.0296,
                        "per_cell": {}, "failed_cells": [CELL_A],
                    },
                    "superiority": {
                        "declared": False, "gamma": 0.03, "delta": 0.03,
                        "limit": -0.0305, "targets": [], "per_target": {},
                        "failed_targets": [], "others_noninferior": True,
                        "other_bounds": {}, "passed": None,
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(
                root, [make_session(decisions=decisions)])
            code, _, stderr = run_publish(root, manifest)
            self.assertNotEqual(code, 0)
            self.assertIn("noninferiority", stderr)

    def test_layout_exception_is_recorded_not_current(self) -> None:
        failed_tier = {
            "passed": False, "delta": 0.03, "limit": 0.0296,
            "per_cell": {}, "failed_cells": [CELL_A],
        }
        decisions = {
            "noninferiority": {"passed": False,
                               "per_tier": {"v4x": failed_tier}},
            "superiority": {
                "declared": False, "passed": None,
                "per_tier": {
                    "v4x": {"declared": False, "gamma": 0.03, "delta": 0.03,
                            "limit": -0.0305, "targets": [], "per_target": {},
                            "failed_targets": [], "others_noninferior": True,
                            "other_bounds": {}, "passed": None},
                },
            },
            "per_tier": {
                "v4x": {
                    "noninferiority": failed_tier,
                    "superiority": {
                        "declared": False, "gamma": 0.03, "delta": 0.03,
                        "limit": -0.0305, "targets": [], "per_target": {},
                        "failed_targets": [], "others_noninferior": True,
                        "other_bounds": {}, "passed": None,
                    },
                },
            },
        }
        layout_only = {
            "asserted": True,
            "control": {"revision": BASE_SHA,
                        "metadata_seed": "fgf-pair-layout-control",
                        "runs": 5, "rounds": 3},
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": 0.04,
                        "calibration_spread": 0.05,
                        "remeasured_bound": 0.035,
                        "spread": 0.05,
                        "disposition": "owner_exception_required",
                        "owner_signature": None,
                    },
                },
            },
            "remeasured": {
                "v4x": {"bounds": {}, "controls": {}, "drift": {}, "runs": 5},
            },
        }
        session = make_session(decisions=decisions, layout_only=layout_only)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [session])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            benchmark = read_page(root, "BENCHMARKS.md")
            history = read_page(root, "benchmarks/history.md")
            page = read_page(root, "benchmarks/exceptions.md")
            self.assertNotIn(CELL_A, benchmark)
            self.assertNotIn(CELL_A, history)
            self.assertIn(CELL_A, page)
            self.assertIn("owner_exception_required", page)
            self.assertIn("| - |", page)

    def test_rejected_session_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [make_session(rejected=True)])
            code, _, stderr = run_publish(root, manifest)
            self.assertNotEqual(code, 0)
            self.assertIn("rejected", stderr)

    def test_unknown_host_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(
                root, [make_session(host="unknown")])
            code, _, stderr = run_publish(root, manifest)
            self.assertNotEqual(code, 0)
            self.assertIn("not declared", stderr)

    def test_publishes_runner_written_session(self) -> None:
        session = make_runner_session()
        self.assertEqual(session["method_version"], "fgf-pair/5")
        self.assertEqual(session["drift_method"], "iqr_over_median")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [session])
            path = (root / "records" / "pair" / session["session_id"]
                    / "session.json")
            bench_pair.write_session(session, path)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["sessions"][0]["sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(document, indent=2) + "\n",
                                encoding="utf-8")
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertIn("`" + CELL_A + "`", page)
            self.assertIn("`" + CANDIDATE_SHA + "`", page)

    def test_staged_diagnostics_render_without_disguising_confirmed_failure(self):
        unconfirmed = make_runner_session(
            "unconfirmed", cell=CELL_A, base_ns=100.0,
            candidate_ns=108.0, confirmation_ns=100.0,
        )
        unconfirmed["method_version"] = "fgf-pair/4"
        unconfirmed["drift_method"] = "p90_minus_p10_over_median"
        confirmed = make_runner_session(
            "confirmed", cell=CELL_B, base_ns=100.0,
            candidate_ns=108.0, confirmation_ns=108.0,
        )
        self.assertTrue(unconfirmed["decisions"]["noninferiority"]["passed"])
        self.assertFalse(confirmed["decisions"]["noninferiority"]["passed"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [unconfirmed, confirmed])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            for relative in ("BENCHMARKS.md", "benchmarks/history.md"):
                page = read_page(root, relative)
                rows = [line for line in page.splitlines() if line.startswith("| ")]
                unconfirmed_row = next(line for line in rows if f"`{CELL_A}`" in line)
                confirmed_row = next(line for line in rows if f"`{CELL_B}`" in line)
                self.assertIn("| unconfirmed_main_failure |", unconfirmed_row)
                self.assertIn("| confirmed_regression |", confirmed_row)
                self.assertNotIn("| accepted |", confirmed_row)
            self.assertEqual(run_publish(root, manifest, "--verify")[0], 0)

    def test_staged_failure_cannot_be_published_without_confirmation_evidence(self):
        for version, method in (
            ("fgf-pair/4", "p90_minus_p10_over_median"),
            ("fgf-pair/5", "iqr_over_median"),
        ):
            with self.subTest(version=version):
                session = make_runner_session(
                    base_ns=100.0, candidate_ns=108.0, confirmation_ns=108.0,
                )
                session["method_version"], session["drift_method"] = version, method
                session["runs"].pop()
                with self.assertRaises(publish.PublishError):
                    publish.validate_session(session, "incomplete confirmation")
                session = make_runner_session(
                    base_ns=100.0, candidate_ns=108.0, confirmation_ns=100.0,
                )
                session["method_version"], session["drift_method"] = version, method
                session["confirmed_failures"]["v4x"] = [CELL_A]
                with self.assertRaises(publish.PublishError):
                    publish.validate_session(session, "mislabelled confirmation")

    def test_competitor_logs_render_separate_table(self) -> None:
        log = "\n".join([
            "TRIO\tshape\t1073741824\t1000000000\t2000000000\t2000000000"
            "\t1\t2\t1\t2",
            "TRIO\tshape\t1073741824\t1000000000\t2000000000\t2000000000"
            "\t1\t2\t1\t2",
            "PRIME\trate shape\trate\t1073741824\t1000000000\t2000000000\t1\t2",
            "PRIME\trate shape\trate\t1073741824\t1000000000\t2000000000\t1\t2",
            "PRIME\tscalar shape\tscalar\t0\t2.5\t3.5\t2.4\t3.6",
            "RESULT\tgf16 shape\t1\t1\t1073741824\tfixed\tfgf"
            "\t1000000000\t1.0\t0.9\t1.1\t5",
            "RESULT\tgf16 shape\t1\t1\t1073741824\tfixed\tGF-Complete"
            "\t2000000000\t0.5\t0.4\t0.6\t5",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "records" / "comp" / "gf.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(log, encoding="utf-8")
            entry = {
                "path": "comp/gf.log",
                "host": "willow",
                "tier": "v4x",
                "label": "gf",
                "sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
            }
            manifest = write_manifest(root, [], competitors=[entry])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "benchmarks/competitors.md")
            self.assertIn("## Willow Cove / v4x / gf", page)
            self.assertIn("fgf (GiB/s)", page)
            self.assertIn("GF-Complete (GiB/s)", page)
            self.assertIn("Plonky3 scalar", page)
            self.assertIn("1.000", page)

    def test_threshold_rows_in_separate_table(self) -> None:
        threshold = {
            "inventory_path": "inventory.json",
            "item_key": "cutoff",
            "old_value": 2048,
            "new_value": 8192,
            "unit": "bytes",
            "comparison": "<",
            "axis": "row_len",
            "declared_cells": [CELL_B],
            "target_set": [CELL_B],
            "span_ok": True,
            "path": "threshold.json",
            "digest": "e" * 64,
            "grid": [2048, 4096, 8192],
            "cells": [{
                "cell": CELL_B,
                "axis_value": 4096,
                "route_old": "v2",
                "route_new": "v4x",
                "route_differs": True,
            }],
            "old": 2048,
            "new": 8192,
        }
        session = make_session(
            "20261107T040000Z-willow", kind="threshold",
            rows=[make_row(CELL_B, summary_ns=40.0)],
            threshold=threshold)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [session])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            benchmark = read_page(root, "BENCHMARKS.md")
            page = read_page(root, "benchmarks/thresholds.md")
            self.assertNotIn(CELL_B, benchmark)
            self.assertIn(CELL_B, page)
            self.assertIn("## Declarations (bytes)", page)
            self.assertIn("Old cutoff (bytes)", page)
            self.assertIn("New cutoff (bytes)", page)

    def test_threshold_rows_render_confirmed_and_unconfirmed_diagnostics(self):
        sessions = []
        for confirmation_ns in (100.0, 108.0):
            session = make_runner_session(
                f"threshold-{confirmation_ns}", cell=CELL_B, base_ns=100.0,
                candidate_ns=108.0, confirmation_ns=confirmation_ns,
            )
            session["kind"], session["acceptance"] = "threshold", False
            session["threshold"] = {
                "item_key": "cutoff", "unit": "bytes", "old": 32768, "new": 131072,
                "comparison": "<",
                "cells": [
                    {"cell": CELL_A, "axis_value": 65536},
                    {"cell": CELL_B, "axis_value": 4096},
                ],
                "target_set": [CELL_A],
            }
            for run in session["runs"]:
                if run["stage"] == "main":
                    for arm, value in (("base", 100.0), ("candidate", 95.0)):
                        run["arms"][arm]["cells"][CELL_A] = {
                            "summary_ns": value, "rounds_ns": [value],
                        }
            for arm, value in (("base", 100.0), ("candidate", 95.0)):
                session["arm_summaries"][arm]["v4x"][CELL_A] = value
            decision = bench_pair.recorded_staged_decision(session, "v4x", [CELL_A])
            session["decisions"] = bench_pair.publishable_decisions({"v4x": decision})
            session["confirmed_failures"] = {"v4x": decision["confirmed_failures"]}
            session["rows"] = bench_pair.build_rows(session, ["v4x"], [CELL_A, CELL_B])
            sessions.append(session)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, sessions)
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "benchmarks/thresholds.md")
            self.assertIn("| Record |", page)
            self.assertIn("| confirmed_regression |", page)
            self.assertIn("| unconfirmed_main_failure |", page)
            self.assertFalse(sessions[1]["decisions"]["superiority"]["passed"])

    def test_unsuccessful_thresholds_publish_as_evidence_not_current_results(self):
        for candidate_ns in (102.0, 108.0):
            with self.subTest(candidate_ns=candidate_ns):
                session = make_runner_session(
                    f"unsuccessful-threshold-{candidate_ns}", cell=CELL_B,
                    base_ns=100.0, candidate_ns=candidate_ns,
                    targets=[CELL_B],
                )
                session["kind"], session["acceptance"] = "threshold", False
                session["threshold"] = {
                    "item_key": "cutoff", "unit": "bytes", "old": 2048, "new": 8192,
                    "comparison": "<",
                    "cells": [{"cell": CELL_B, "axis_value": 4096}],
                    "target_set": [CELL_B],
                }
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    manifest = write_manifest(root, [session])
                    code, _, stderr = run_publish(root, manifest)
                    self.assertEqual(code, 0, stderr)
                    page = read_page(root, "benchmarks/thresholds.md")
                    declaration = next(
                        line for line in page.splitlines()
                        if 'id="threshold-' in line
                    )
                    self.assertEqual(
                        [cell.strip() for cell in declaration.split("|")][-3:-1],
                        ["yes" if candidate_ns == 102.0 else "no", "no"],
                    )
                    self.assertNotIn(CELL_B, read_page(root, "BENCHMARKS.md"))
                    self.assertNotIn(CELL_B, read_page(root, "benchmarks/history.md"))
                    self.assertNotIn("| accepted |", page)
                    self.assertEqual(run_publish(root, manifest, "--verify")[0], 0)
                session["runs"].pop()
                with self.assertRaises(publish.PublishError):
                    publish.validate_session(session, "incomplete threshold evidence")
                session["rejected"] = True
                with self.assertRaises(publish.PublishError):
                    publish.validate_session(session, "rejected threshold measurement")

    def test_threshold_predicate_provenance_is_required_and_rechecks_targets(self):
        session = make_runner_session(
            "predicate-provenance", cell=CELL_B, base_ns=100.0,
            candidate_ns=95.0, targets=[CELL_B],
        )
        session["kind"], session["acceptance"] = "threshold", False
        session["threshold"] = {
            "item_key": "inclusive-cutoff", "unit": "bytes",
            "old": 4096, "new": 8192,
            "cells": [{"cell": CELL_B, "axis_value": 4096}],
            "target_set": [CELL_B],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [session])
            capture = root / "records" / "pair" / session["session_id"] / "session.json"
            captured_bytes = capture.read_bytes()
            self.assertNotEqual(run_publish(root, manifest)[0], 0)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["sessions"][0]["threshold_comparison"] = "<="
            manifest.write_text(json.dumps(document), encoding="utf-8")
            self.assertNotEqual(run_publish(root, manifest)[0], 0)
            document["sessions"][0]["threshold_comparison"] = "<"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(run_publish(root, manifest, "--verify")[0], 0)
            self.assertEqual(capture.read_bytes(), captured_bytes)


class IncrementalDescriptiveTests(unittest.TestCase):
    def test_latest_rows_preserve_unmeasured_cells_and_measured_revision(self):
        old = make_session(rows=[make_row(CELL_A), make_row(CELL_B)])
        fresh = descriptive_session()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [old, fresh])
            hosts, sessions, _, metadata = publish.load_manifest(manifest, root / "records")
            rows = publish._collect_rows(sessions, "acceptance") + publish._collect_rows(sessions, "descriptive")
            index = publish.HostIndex(hosts, rows)
            latest = {row.cell: row for row in publish._current_rows(rows, index, metadata["tiers"])}
            self.assertEqual(latest[CELL_A].session.data["session_id"], fresh["session_id"])
            self.assertEqual(latest[CELL_A].data["library_sha"], fresh["arms"]["base"]["sha"])
            self.assertEqual(latest[CELL_B].session.data["session_id"], old["session_id"])
            self.assertEqual(latest[CELL_B].data["library_sha"], CANDIDATE_SHA)
            code, _, errors = run_publish(root, manifest)
            self.assertEqual(code, 0, errors)
            code, _, errors = run_publish(root, manifest, "--verify")
            self.assertEqual(code, 0, errors)

    def test_surface_measurements_do_not_replace_operations_anchors(self):
        operations = descriptive_session()
        surfaces = descriptive_session("20261108T010000Z-willow", bench="surfaces")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [operations, surfaces])
            code, _, errors = run_publish(root, manifest)
            self.assertEqual(code, 0, errors)
            self.assertIn(operations["rows"][0]["cell"], read_page(root, "BENCHMARKS.md"))
            self.assertNotIn(surfaces["rows"][0]["cell"], read_page(root, "BENCHMARKS.md"))
            self.assertIn(surfaces["rows"][0]["cell"], read_page(root, "benchmarks/surfaces.md"))

    def test_configuration_changes_cannot_silently_replace_existing_anchors(self):
        for key, value in (("features", ["std"]), ("rustflags", " ".join(publish.ALIGNMENT_FLAGS) + " -C opt-level=2")):
            old = descriptive_session()
            fresh = descriptive_session("20261108T010000Z-willow")
            fresh["snapshot"][key] = value
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = write_manifest(root, [old, fresh])
                with self.assertRaises(publish.PublishError):
                    publish.load_manifest(manifest, root / "records")

    def test_scoped_pair_rows_must_cover_both_arms_and_each_selected_cell(self):
        session = make_session()
        session["inventory"]["order"] = [CELL_A]
        session["scope"] = {"kind": "selected", "cells": [CELL_A],
                            "digest": bench_baseline.digest([CELL_A]), "rationale": "Changed byte-field path"}
        publish.validate_session(session, "scoped")
        session["rows"].pop()
        with self.assertRaises(publish.PublishError):
            publish.validate_session(session, "scoped")


if __name__ == "__main__":
    unittest.main()
