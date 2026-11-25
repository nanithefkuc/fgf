#!/usr/bin/env python3
"""Tests for the benchmark record publisher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
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
    """Construct a historical staged, single-placement measurement."""
    args = argparse.Namespace(
        mode="acceptance", runs="5", rounds="3", delta="0.03", gamma="0.03",
        drift_cap="0.01", features="simd512",
        rustflags=" ".join(publish.ALIGNMENT_FLAGS), toolchain="",
    )
    session = bench_pair.base_session(args, session_id)
    session["method_version"] = "fgf-pair/5"
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
        "control": {"start_ns": 100.0, "end_ns": 100.1},
        "controls": {"base": {"start_ns": 100.0, "end_ns": 100.1},
                     "candidate": {"start_ns": 100.0, "end_ns": 100.1}},
        "arms": {
            "base": {"cells": {cell: {"summary_ns": base_ns,
                                      "rounds_ns": [base_ns, base_ns, base_ns]}}},
            "candidate": {"cells": {cell: {"summary_ns": candidate_ns,
                                           "rounds_ns": [candidate_ns, candidate_ns, candidate_ns]}}},
        },
    } for index in range(5)]
    if confirmation_ns is not None:
        confirmation = copy.deepcopy(session["runs"])
        for run in confirmation:
            run["stage"] = "confirmation"
            run["arms"]["candidate"]["cells"][cell] = {
                "summary_ns": confirmation_ns,
                "rounds_ns": [confirmation_ns, confirmation_ns, confirmation_ns],
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
    saved_map = os.environ.pop("FGF_HOST_MAP", None) if "--host-map" not in extra else None
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = publish.main(args)
    finally:
        if saved_map is not None:
            os.environ["FGF_HOST_MAP"] = saved_map
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

    def test_single_placement_record_cannot_claim_distribution_measurement(self):
        session = make_runner_session()
        session["method_version"] = bench_pair.METHOD_VERSION
        with self.assertRaises(publish.PublishError):
            publish.validate_session(session, "missing placement measurements")

    def test_publishes_and_validates_distribution_session(self):
        import copy
        import statistics
        session = make_runner_session("20261124T050000Z-willow")
        session["method_version"] = bench_pair.METHOD_VERSION
        session["config"]["placement_method"] = bench_pair.placement.METHOD
        rounds = int(session["config"]["rounds"])
        for run in session["runs"]:
            run_index = run["run"]
            arms = ("base", "candidate")
            strata = []
            for point in bench_pair.placement.plan(run_index):
                order = bench_pair.placement.arm_order(arms, run_index, point["index"])
                observed = {}
                controls = {}
                stratum_arms = {}
                for arm_idx, arm in enumerate(arms):
                    observed[arm] = {
                        "dst_shift": point["dst_shift"],
                        "src_shift": point["src_shift"],
                        "stack_offset": (128 + arm_idx * 16 + point["stack_pad"]) % 4096,
                    }
                    controls[arm] = dict(run["controls"][arm])
                    stratum_arms[arm] = {
                        "cells": {
                            cid: {
                                "summary_ns": entry["summary_ns"],
                                "rounds_ns": [entry["summary_ns"]] * rounds,
                            }
                            for cid, entry in run["arms"][arm]["cells"].items()
                        }
                    }
                strata.append({
                    **point,
                    "order": order,
                    "environment_digest": f"env-{run_index}-{point['index']}",
                    "argv_lengths": {arm: 100 for arm in arms},
                    "observed": observed,
                    "controls": controls,
                    "arms": stratum_arms,
                })
            run["order"] = bench_pair.placement.arm_order(arms, run_index, 0)
            run["controls"] = {
                arm: {
                    key: statistics.median(p["controls"][arm][key] for p in strata)
                    for key in ("start_ns", "end_ns")
                }
                for arm in arms
            }
            run["control"] = run["controls"][run["order"][0]]
            run["placements"] = strata
            run["arms"] = {
                arm: bench_pair.placement.summarize(
                    [p["arms"][arm] for p in strata], list(run["arms"][arm]["cells"]), rounds
                )
                for arm in arms
            }
        session["rows"] = bench_pair.build_rows(session, ["v4x"], [CELL_A])
        publish.validate_session(session, "valid /6 session")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_manifest(root, [session])
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            self.assertIn("`" + CELL_A + "`", read_page(root, "BENCHMARKS.md"))

        # Corrupting a retained stratum round or control in a /6 session must fail validation.
        bad_round = copy.deepcopy(session)
        bad_round["runs"][0]["placements"][0]["arms"]["base"]["cells"][CELL_A]["rounds_ns"] = [50.0, 50.0, 101.0]
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_round, "tampered /6 round")

        bad_ctrl = copy.deepcopy(session)
        bad_ctrl["runs"][0]["controls"]["base"]["start_ns"] = 999.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_ctrl, "tampered /6 control")

        # Altering a published row timing without changing retained run data must fail validation.
        bad_row_summary = copy.deepcopy(session)
        bad_row_summary["rows"][0]["summary_ns"] = 77.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_row_summary, "tampered /6 row summary")

        bad_row_rounds = copy.deepcopy(session)
        bad_row_rounds["rows"][0]["rounds_ns"][0] = 77.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_row_rounds, "tampered /6 row rounds")

        bad_arm_summary = copy.deepcopy(session)
        bad_arm_summary["arm_summaries"]["candidate"]["v4x"][CELL_A] = 77.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_arm_summary, "tampered /6 arm_summaries")

        missing_row = copy.deepcopy(session)
        missing_row["rows"].pop()
        with self.assertRaises(publish.PublishError):
            publish.validate_session(missing_row, "missing /6 row")

        # Consistent per-placement start controls whose recomputed drift exceeds drift_cap must fail.
        bad_drift = copy.deepcopy(session)
        for run in bad_drift["runs"]:
            for idx, stratum in enumerate(run["placements"]):
                start = 100.0 if idx < 128 else 110.0
                for arm in ("base", "candidate"):
                    stratum["controls"][arm] = {"start_ns": start, "end_ns": start + 0.1}
            run["controls"] = {
                arm: {
                    key: statistics.median(p["controls"][arm][key] for p in run["placements"])
                    for key in ("start_ns", "end_ns")
                }
                for arm in ("base", "candidate")
            }
            run["control"] = run["controls"][run["order"][0]]
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_drift, "over-cap /6 per-placement drift")

        # Confirmation-stage over-cap drift must also fail validation (both /5 and /6).
        bad_conf_drift = make_runner_session(
            "20261124T060000Z-willow", base_ns=100.0, candidate_ns=108.0, confirmation_ns=108.0
        )
        for run in bad_conf_drift["runs"]:
            if run["stage"] == "confirmation":
                run["controls"] = {
                    "base": {"start_ns": 100.0, "end_ns": 100.1},
                    "candidate": {"start_ns": 110.0, "end_ns": 110.1},
                }
                run["control"] = run["controls"][run["order"][0]]
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_conf_drift, "over-cap /5 confirmation drift")

        # Extra per-placement control keys must not dilute /6 drift.
        extra_ctrl = copy.deepcopy(bad_drift)
        for run in extra_ctrl["runs"]:
            for stratum in run["placements"]:
                for k in ("pad0", "pad1", "pad2"):
                    stratum["controls"][k] = {"start_ns": 105.0, "end_ns": 105.1}
        with self.assertRaises(publish.PublishError):
            publish.validate_session(extra_ctrl, "extra control keys hiding /6 drift")

        # /6 confirmation-stage over-cap drift must fail validation.
        conf6 = copy.deepcopy(session)
        conf_runs = copy.deepcopy(session["runs"])
        for run in conf_runs:
            run["stage"] = "confirmation"
            for idx, stratum in enumerate(run["placements"]):
                start = 100.0 if idx < 128 else 110.0
                for arm in ("base", "candidate"):
                    stratum["controls"][arm] = {"start_ns": start, "end_ns": start + 0.1}
                    stratum["arms"]["candidate"]["cells"][CELL_A] = {
                        "summary_ns": 108.0,
                        "rounds_ns": [108.0] * rounds,
                    }
            run["controls"] = {
                arm: {
                    key: statistics.median(p["controls"][arm][key] for p in run["placements"])
                    for key in ("start_ns", "end_ns")
                }
                for arm in ("base", "candidate")
            }
            run["control"] = run["controls"][run["order"][0]]
            run["arms"]["candidate"] = bench_pair.placement.summarize(
                [p["arms"]["candidate"] for p in run["placements"]], [CELL_A], rounds
            )
        for run in conf6["runs"]:
            for stratum in run["placements"]:
                stratum["arms"]["candidate"]["cells"][CELL_A] = {
                    "summary_ns": 108.0,
                    "rounds_ns": [108.0] * rounds,
                }
            run["arms"]["candidate"] = bench_pair.placement.summarize(
                [p["arms"]["candidate"] for p in run["placements"]], [CELL_A], rounds
            )
        conf6["runs"].extend(conf_runs)
        conf6["arm_summaries"]["candidate"]["v4x"][CELL_A] = 108.0
        conf6["rows"] = bench_pair.build_rows(conf6, ["v4x"], [CELL_A])
        dec6 = bench_pair.recorded_staged_decision(conf6, "v4x")
        conf6["decisions"] = bench_pair.publishable_decisions({"v4x": dec6})
        conf6["confirmed_failures"] = {"v4x": dec6["confirmed_failures"]}
        with self.assertRaises(publish.PublishError):
            publish.validate_session(conf6, "over-cap /6 confirmation drift")

        # Flipping row metadata (role or unstable) must fail validation.
        bad_role = copy.deepcopy(session)
        bad_role["rows"][0]["role"] = "layout_control"
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_role, "flipped row role")

        bad_unstable = copy.deepcopy(session)
        bad_unstable["rows"][0]["unstable"] = True
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_unstable, "flipped row unstable flag")

        # Stray runs on undeclared tiers must fail validation.
        stray_tier = copy.deepcopy(session)
        extra_run = copy.deepcopy(session["runs"][0])
        extra_run["tier"] = "v2"
        stray_tier["runs"].append(extra_run)
        with self.assertRaises(publish.PublishError):
            publish.validate_session(stray_tier, "stray run on undeclared tier")

        # Missing layout_only.remeasured on a failing /6 session cannot bypass rejection.
        no_rem_layout = copy.deepcopy(conf6)
        for run in no_rem_layout["runs"]:
            if run["stage"] == "confirmation":
                for stratum in run["placements"]:
                    for arm in ("base", "candidate"):
                        stratum["controls"][arm] = {"start_ns": 100.0, "end_ns": 100.1}
                run["controls"] = {
                    arm: {"start_ns": 100.0, "end_ns": 100.1}
                    for arm in ("base", "candidate")
                }
                run["control"] = run["controls"][run["order"][0]]
        no_rem_layout["layout_only"] = {
            "asserted": True,
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": 0.0,
                        "spread": 1.0,
                        "disposition": "owner_exception_required",
                    }
                }
            },
        }
        with self.assertRaises(publish.PublishError):
            publish.validate_session(no_rem_layout, "missing /6 layout_only.remeasured")

        # Tampering with a nested layout_only placement record must fail validation.
        bad_layout = copy.deepcopy(session)
        bad_layout_run = copy.deepcopy(bad_layout["runs"][0])
        bad_layout_run["arms"] = {
            "base": bad_layout_run["arms"]["base"],
            "layout_control": bad_layout_run["arms"]["candidate"],
        }
        bad_layout_run["order"] = ["base", "layout_control"]
        bad_layout_run["controls"] = {
            "base": bad_layout_run["controls"]["base"],
            "layout_control": bad_layout_run["controls"]["candidate"],
        }
        for stratum in bad_layout_run["placements"]:
            stratum["order"] = ["layout_control" if a == "candidate" else a for a in stratum["order"]]
            stratum["argv_lengths"] = {"base": 100, "layout_control": 100}
            stratum["observed"] = {"base": stratum["observed"]["base"], "layout_control": stratum["observed"]["candidate"]}
            stratum["controls"] = {"base": stratum["controls"]["base"], "layout_control": stratum["controls"]["candidate"]}
            stratum["arms"] = {"base": stratum["arms"]["base"], "layout_control": stratum["arms"]["candidate"]}

        # Isolate each /6 layout_only failure cause: omitted remeasured_bound,
        # omitted calibration_spread, fabricated spread, and short placement count.
        rem_runs = []
        for idx in range(5):
            r = copy.deepcopy(bad_layout_run)
            r["run"] = idx
            r["order"] = bench_pair.placement.arm_order(("base", "layout_control"), idx, 0)
            for point, stratum in zip(bench_pair.placement.plan(idx), r["placements"]):
                stratum.update(point)
                stratum["order"] = bench_pair.placement.arm_order(("base", "layout_control"), idx, point["index"])
                for arm_i, arm in enumerate(("base", "layout_control")):
                    stratum["observed"][arm] = {
                        "dst_shift": point["dst_shift"],
                        "src_shift": point["src_shift"],
                        "stack_offset": (128 + arm_i * 16 + point["stack_pad"]) % 4096,
                    }
            rem_runs.append(r)
        rem_base = [{CELL_A: r["arms"]["base"]["cells"][CELL_A]["summary_ns"]} for r in rem_runs]
        rem_ctrl = [{CELL_A: r["arms"]["layout_control"]["cells"][CELL_A]["summary_ns"]} for r in rem_runs]
        rem_bound = bench_pair.hl_upper_bound(
            bench_pair.paired_log_ratios(rem_base, rem_ctrl, [CELL_A])[CELL_A]
        )["upper_bound"]
        valid_layout = copy.deepcopy(no_rem_layout)
        valid_layout["layout_only"] = {
            "asserted": True,
            "per_tier": {
                "v4x": {
                    CELL_A: {
                        "failed_bound": dec6["noninferiority"]["per_cell"][CELL_A]["upper_bound"],
                        "remeasured_bound": rem_bound,
                        "calibration_spread": 0.5,
                        "spread": max(0.5, rem_bound),
                        "disposition": "owner_exception_required",
                    }
                }
            },
            "remeasured": {"v4x": {"placements": rem_runs}},
        }
        publish.validate_session(valid_layout, "valid /6 layout exception")

        no_rem_bound = copy.deepcopy(valid_layout)
        no_rem_bound["layout_only"]["per_tier"]["v4x"][CELL_A].pop("remeasured_bound")
        with self.assertRaises(publish.PublishError):
            publish.validate_session(no_rem_bound, "omitted remeasured_bound")

        no_cal_spread = copy.deepcopy(valid_layout)
        no_cal_spread["layout_only"]["per_tier"]["v4x"][CELL_A].pop("calibration_spread")
        with self.assertRaises(publish.PublishError):
            publish.validate_session(no_cal_spread, "omitted calibration_spread")

        fewer_rem = copy.deepcopy(valid_layout)
        fewer_rem["layout_only"]["remeasured"]["v4x"]["placements"] = rem_runs[:4]
        with self.assertRaises(publish.PublishError):
            publish.validate_session(fewer_rem, "fewer than config.runs layout placements")
        over_drift_layout = copy.deepcopy(bad_layout)
        over_drift_run = copy.deepcopy(bad_layout_run)
        for idx, stratum in enumerate(over_drift_run["placements"]):
            start = 100.0 if idx < 128 else 110.0
            for arm in ("base", "layout_control"):
                stratum["controls"][arm] = {"start_ns": start, "end_ns": start + 0.1}
        over_drift_run["controls"] = {
            arm: {
                key: statistics.median(p["controls"][arm][key] for p in over_drift_run["placements"])
                for key in ("start_ns", "end_ns")
            }
            for arm in ("base", "layout_control")
        }
        over_drift_run["control"] = over_drift_run["controls"][over_drift_run["order"][0]]
        over_drift_layout["layout_only"] = {
            "asserted": True,
            "per_tier": {"v4x": {}},
            "remeasured": {"v4x": {"placements": [over_drift_run]}},
        }
        with self.assertRaises(publish.PublishError):
            publish.validate_session(over_drift_layout, "over-cap /6 layout drift")

        bad_layout_run["placements"][0]["observed"]["layout_control"]["stack_offset"] += 4096
        bad_layout["layout_only"] = {
            "asserted": True,
            "per_tier": {"v4x": {}},
            "remeasured": {"v4x": {"placements": [bad_layout_run]}},
        }
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_layout, "tampered /6 layout placement")

    def test_host_map_projects_legacy_private_hostnames_to_public_labels(self):
        legacy = make_runner_session("20261108T143928Z-AlphaBox")
        legacy["host"]["hostname"] = "AlphaBox"
        for row in legacy["rows"]:
            row["host"] = "AlphaBox"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records"
            sess_dir = records / "pair" / "20261108T143928Z-willow"
            sess_dir.mkdir(parents=True)
            content = (json.dumps(legacy, indent=1) + "\n").encode("utf-8")
            (sess_dir / "session.json").write_bytes(content)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": 1,
                "metadata": {"hosts": HOSTS, "tiers": ["v4x"]},
                "sessions": [{
                    "path": "pair/20261108T143928Z-willow/session.json",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }],
            }), encoding="utf-8")
            # Without a host map, the private hostname must fail closed.
            code, _, _ = run_publish(root, manifest)
            self.assertNotEqual(code, 0)
            # With an untracked local host map, public pages emit only the architecture label.
            host_map = records / "hosts.json"
            host_map.write_text(json.dumps({"AlphaBox": "willow"}), encoding="utf-8")
            code, _, stderr = run_publish(root, manifest)
            self.assertEqual(code, 0, stderr)
            page = read_page(root, "BENCHMARKS.md")
            self.assertIn('<a id="willow/v4x/', page)
            self.assertIn("`20261108T143928Z-willow`", page)
            for rel_page in ("BENCHMARKS.md", "benchmarks/history.md", "benchmarks/calibration.md", "benchmarks/thresholds.md", "benchmarks/exceptions.md"):
                self.assertNotIn("AlphaBox", read_page(root, rel_page))
            code, _, stderr = run_publish(root, manifest, "--verify")
            self.assertEqual(code, 0, stderr)
            # Manifest aliases must be rejected.
            bad_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            bad_manifest["metadata"]["hosts"][0]["aliases"] = ["AlphaBox"]
            bad_path = root / "bad_manifest.json"
            bad_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            with self.assertRaises(publish.PublishError):
                publish.load_manifest(bad_path, records, host_map=host_map)
            # An unrelated alias inside a mapped session_id must be rejected.
            bad_sid = copy.deepcopy(legacy)
            bad_sid["session_id"] = "20261108T143928Z-other-alias"
            for row in bad_sid["rows"]:
                row["session_id"] = bad_sid["session_id"]
            bad_dir = records / "pair" / bad_sid["session_id"]
            bad_dir.mkdir(parents=True)
            bad_bytes = (json.dumps(bad_sid, indent=1) + "\n").encode("utf-8")
            (bad_dir / "session.json").write_bytes(bad_bytes)
            bad_sid_manifest = root / "bad_sid_manifest.json"
            bad_sid_manifest.write_text(json.dumps({
                "schema": 1,
                "metadata": {"hosts": HOSTS, "tiers": ["v4x"]},
                "sessions": [{
                    "path": f"pair/{bad_sid['session_id']}/session.json",
                    "sha256": hashlib.sha256(bad_bytes).hexdigest(),
                }],
            }), encoding="utf-8")
            with self.assertRaises(publish.PublishError):
                publish.load_manifest(bad_sid_manifest, records, host_map=host_map)
            # A mapped private hostname appearing in a host display name must fail even with no session from that host,
            # including when the raw hostname is a substring of its public label (e.g. box -> golden-box).
            extra_map = records / "hosts_extra.json"
            extra_map.write_text(
                json.dumps({"AlphaBox": "willow", "SecretBox": "golden", "box": "golden-box"}),
                encoding="utf-8",
            )
            leak_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            leak_manifest["metadata"]["hosts"][1]["name"] = "SecretBox Display"
            leak_path = root / "leak_manifest.json"
            leak_path.write_text(json.dumps(leak_manifest), encoding="utf-8")
            h, s, c, m = publish.load_manifest(leak_path, records, host_map=extra_map)
            with self.assertRaises(publish.PublishError):
                publish.generate(h, s, c, m)
            sub_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            sub_manifest["metadata"]["hosts"][1] = {
                "host": "golden-box", "name": "box Display", "available": False
            }
            sub_path = root / "sub_manifest.json"
            sub_path.write_text(json.dumps(sub_manifest), encoding="utf-8")
            h, s, c, m = publish.load_manifest(sub_path, records, host_map=extra_map)
            with self.assertRaises(publish.PublishError):
                publish.generate(h, s, c, m)
            # Positive case: a session mapped from substring raw hostname `box` -> `golden-box`
            # renders valid `golden-box` anchors and session IDs without false positives.
            sub_session = make_runner_session("20261109T233036Z-box")
            sub_session["host"]["hostname"] = "box"
            for row in sub_session["rows"]:
                row["host"] = "box"
            sub_dir = records / "pair" / "20261109T233036Z-golden-box"
            sub_dir.mkdir(parents=True)
            sub_bytes = (json.dumps(sub_session, indent=1) + "\n").encode("utf-8")
            (sub_dir / "session.json").write_bytes(sub_bytes)
            ok_sub_manifest = root / "ok_sub_manifest.json"
            ok_sub_manifest.write_text(json.dumps({
                "schema": 1,
                "metadata": {
                    "hosts": [
                        {"host": "golden-box", "name": "Golden Cove", "available": True},
                    ],
                    "tiers": ["v4x"],
                },
                "sessions": [{
                    "path": "pair/20261109T233036Z-golden-box/session.json",
                    "sha256": hashlib.sha256(sub_bytes).hexdigest(),
                }],
            }), encoding="utf-8")
            code, _, stderr = run_publish(
                root, ok_sub_manifest, "--host-map", str(extra_map)
            )
            self.assertEqual(code, 0, stderr)
            sub_page = read_page(root, "BENCHMARKS.md")
            self.assertIn('<a id="golden-box/v4x/', sub_page)
            self.assertIn("`20261109T233036Z-golden-box`", sub_page)

    def test_v4_v5_rounds_controls_and_calibration_recomputation(self):
        # /5 run summary_ns disagreeing with rounds_ns must fail even if rows are rebuilt.
        for mv, dm in (("fgf-pair/4", "p90_minus_p10_over_median"), ("fgf-pair/5", "iqr_over_median")):
            forged = make_runner_session()
            forged["method_version"] = mv
            forged["drift_method"] = dm
            publish.validate_session(forged, f"valid {mv}")
            bad_med = copy.deepcopy(forged)
            for run in bad_med["runs"]:
                run["arms"]["candidate"]["cells"][CELL_A]["summary_ns"] = 50.0
            bad_med["arm_summaries"]["candidate"]["v4x"][CELL_A] = 50.0
            bad_med["rows"] = bench_pair.build_rows(bad_med, ["v4x"], [CELL_A])
            dec = bench_pair.recorded_staged_decision(bad_med, "v4x")
            bad_med["decisions"] = bench_pair.publishable_decisions({"v4x": dec})
            with self.assertRaises(publish.PublishError):
                publish.validate_session(bad_med, f"{mv} forged run median")

            # Dropping controls to fall back to single control must fail.
            missing_ctrls = copy.deepcopy(forged)
            missing_ctrls["runs"][0].pop("controls")
            with self.assertRaises(publish.PublishError):
                publish.validate_session(missing_ctrls, f"{mv} missing arm controls")

            # Malformed control start_ns or missing end_ns must raise PublishError.
            malformed = copy.deepcopy(forged)
            malformed["runs"][0]["controls"]["base"]["start_ns"] = "bad"
            with self.assertRaises(publish.PublishError):
                publish.validate_session(malformed, f"{mv} malformed control")

            missing_end = copy.deepcopy(forged)
            missing_end["runs"][0]["controls"]["base"].pop("end_ns")
            with self.assertRaises(publish.PublishError):
                publish.validate_session(missing_end, f"{mv} missing end_ns")

            # Injecting placements into a legacy /4 or /5 run cannot bypass median or drift checks.
            injected = copy.deepcopy(bad_med)
            for run in injected["runs"]:
                run["controls"] = {
                    "base": {"start_ns": 100.0, "end_ns": 100.1},
                    "candidate": {"start_ns": 130.0, "end_ns": 130.1},
                }
                run["placements"] = [{
                    "controls": {
                        "base": {"start_ns": 100.0, "end_ns": 100.1},
                        "candidate": {"start_ns": 100.0, "end_ns": 100.1},
                    }
                }]
            with self.assertRaises(publish.PublishError):
                publish.validate_session(injected, f"{mv} injected placements")

            # Over-cap drift must fail for both /4 (p90-p10) and /5 (iqr).
            over_cap = copy.deepcopy(forged)
            for idx, run in enumerate(over_cap["runs"]):
                val = 100.0 if idx < 2 else 115.0
                run["controls"] = {
                    "base": {"start_ns": val, "end_ns": val + 0.1},
                    "candidate": {"start_ns": val, "end_ns": val + 0.1},
                }
                run["control"] = run["controls"][run["order"][0]]
            with self.assertRaises(publish.PublishError):
                publish.validate_session(over_cap, f"{mv} over-cap drift")

        # Calibration with dropped runs or fabricated recommendation must fail.
        cal = make_runner_session(base_ns=100.0, candidate_ns=100.0)
        cal["kind"] = "calibration"
        cal["acceptance"] = False
        cal["config"]["confirmation"] = False
        cal["config"]["calibrated"] = False
        ratios = {"v4x": bench_pair.session_tier_ratios(cal, "v4x", "main")}
        drift = {"v4x": {"drift": 0.0}}
        rec, _ = bench_pair.calibration_recommendation(ratios, 0.03, drift)
        cal["calibration"] = {"mode": "base_vs_base", "recommendation": rec}
        publish.validate_session(cal, "valid calibration")

        dropped_runs = copy.deepcopy(cal)
        dropped_runs["runs"].pop()
        with self.assertRaises(publish.PublishError):
            publish.validate_session(dropped_runs, "calibration missing run")

        bad_rec_runs = copy.deepcopy(cal)
        bad_rec_runs["calibration"]["recommendation"]["runs"] = 99
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_rec_runs, "calibration forged runs")

        bad_rec_unstable = copy.deepcopy(cal)
        bad_rec_unstable["calibration"]["recommendation"]["per_tier"]["v4x"]["unstable_cells"] = [CELL_A]
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_rec_unstable, "calibration forged unstable_cells")

        bad_rec_max = copy.deepcopy(cal)
        bad_rec_max["calibration"]["recommendation"]["per_tier"]["v4x"]["max_bound"] = 999.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(bad_rec_max, "calibration forged max_bound")

        missing_limit = copy.deepcopy(cal)
        missing_limit["calibration"]["recommendation"]["per_tier"]["v4x"].pop("limit")
        with self.assertRaises(publish.PublishError):
            publish.validate_session(missing_limit, "calibration missing limit")

        inflated_cap = copy.deepcopy(cal)
        inflated_cap["calibration"]["prior_drift"] = 0.0
        inflated_cap["calibration"]["recommendation"]["drift_cap"] = 999.0
        with self.assertRaises(publish.PublishError):
            publish.validate_session(inflated_cap, "calibration inflated drift_cap")

        extra_conf_cal = copy.deepcopy(cal)
        extra_conf_run = copy.deepcopy(cal["runs"][0])
        extra_conf_run["stage"] = "confirmation"
        extra_conf_cal["runs"].append(extra_conf_run)
        with self.assertRaises(publish.PublishError):
            publish.validate_session(extra_conf_cal, "calibration non-main run")

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
                            "summary_ns": value, "rounds_ns": [value, value, value],
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
            hosts, sessions, _, metadata = publish.load_manifest(
                manifest, root / "records", host_map={}
            )
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
                    publish.load_manifest(manifest, root / "records", host_map={})

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
