#!/usr/bin/env python3
"""Paired before/after measurement runner for the public operations bench.

    python3 benches/bench_pair.py --base REV --candidate REV --snapshot REV_OR_DIR

The runner overlays a measurement snapshot (the bench target, this runner,
``Cargo.toml``, ``Cargo.lock``, any ``rust-toolchain`` file, the feature set,
and the pinned alignment flags) onto checkouts of the two library revisions,
then measures every requested backend tier in one campaign.  Only the bench
target laid down by the snapshot is executed; this file is never part of a
measured arm.

Session layout
--------------
One invocation writes ``bench-records/pair/<UTC stamp>-<host>/session.json``
plus raw logs.  ``session.json`` records the method version, both library
SHAs, the snapshot identity and digest, host facts, the requested and reported
tiers, the configuration, every run's per-cell arm summaries, the rule
decisions, and every rejection.

Protocol consumed
-----------------
The bench target prints tab-separated records: ``BACKEND``,
``FIELD_BACKEND``, ``CELL`` (``--inventory``), ``OUTPUT`` (``--check``), and
``CONTROL``/``ROUND`` (``--time``).  See ``AGENTS.md`` for the campaign
contract.

Modes
-----
``--mode acceptance`` (default) measures a candidate against a base, applying
the noninferiority rule to every cell and, when ``--targets FILE`` names a
target set before measurement, the superiority rule to those targets.

``--mode calibration`` measures base against base, or base against a layout
control (the same revision rebuilt with a different Cargo metadata seed), and
emits the run-count and drift-cap recommendation.  Calibration sessions are
marked non-acceptance and are refused as acceptance inputs.

``--mode threshold`` validates a declared threshold-evidence declaration
against the inventory, then measures its declared target set.

Declarations
------------
A target file (``--targets``) is a JSON cell-id array, a JSON object with a
``cells`` array, or one cell id per line (``#`` starts a comment).

A threshold declaration (``--threshold``) is a JSON object::

    {"item": "src/kernel/quad_mersenne31.rs:ROW_THRESHOLD",
     "old": 512, "new": 1024,
     "cells": [{"cell": "<cell id>", "axis_value": 512,
                "route_old": "v2", "route_new": "v4x"}, ...]}

``item`` names an entry in the inventory passed as ``--inventory``; ``old``
and ``new`` are the cutoff values on that item's axis; every declared cell
carries its axis value and the route the old and new gates select.  The
target set is every declared cell whose two routes differ.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

METHOD_VERSION = "fgf-pair/5"
DRIFT_METHOD = "iqr_over_median"
METHOD_DRIFT = {
    "fgf-pair/3": "p90_minus_p10_over_median",
    "fgf-pair/4": "p90_minus_p10_over_median",
    METHOD_VERSION: DRIFT_METHOD,
}
SUPPORTED_TIERS = ("v4x", "v3_gfni_crypto", "v3", "v2", "scalar")
DEFAULT_TIERS = " ".join(SUPPORTED_TIERS)
REQUIRED_ALIGN_TOKENS = (
    "align-all-functions=6",
    "align-all-nofallthru-blocks=6",
)
DEFAULT_ALIGN_FLAGS = (
    "-C llvm-args=-align-all-functions=6 "
    "-C llvm-args=-align-all-nofallthru-blocks=6"
)
CELL_RE = re.compile(
    r"^(?P<field>[^/]+)/(?P<operation>[^/]+)/(?P<prep>[^/]+)/"
    r"(?P<row_len>\d+)x(?P<rows>\d+)x(?P<sources>\d+)/(?P<alignment>.+)$"
)
INVENTORY_KEYS = (
    "id",
    "field",
    "operation",
    "prep",
    "row_len",
    "rows",
    "sources",
    "alignment",
    "bytes",
)
PLACEHOLDER = "uncalibrated"


# --------------------------------------------------------------------------
# Rejections and small utilities
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Rejection:
    code: str
    message: str
    cells: tuple[str, ...] = ()
    arm: str | None = None

    def as_dict(self) -> dict:
        record = {"code": self.code, "message": self.message}
        if self.cells:
            record["cells"] = list(self.cells)
        if self.arm is not None:
            record["arm"] = self.arm
        return record


def reject(code: str, message: str, *, cells=(), arm=None) -> Rejection:
    return Rejection(code, message, tuple(cells), arm)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def host_component(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    return cleaned.strip("-.") or "host"


def split_words(text: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", text or "") if part]


def parse_number(text, what: str) -> float:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if text is None:
        raise ValueError(f"{what} is required")
    text = str(text).strip()
    if not text or text.lower() == PLACEHOLDER:
        raise ValueError(f"{what} is not calibrated")
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"{what} is not a number: {text!r}") from error


def normalize_flags(flags: str) -> str:
    return " ".join((flags or "").split())


def cell_parts(cell: str) -> dict | None:
    match = CELL_RE.match(cell)
    if not match:
        return None
    parts = match.groupdict()
    for key in ("row_len", "rows", "sources"):
        parts[key] = int(parts[key])
    return parts


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# Hodges-Lehmann location bound
# --------------------------------------------------------------------------


MIN_RUNS_FOR_BOUND = 5
ESTIMATOR = "hodges-lehmann/wilcoxon-signed-rank"
CONFIDENCE = 0.95


class InsufficientRuns(ValueError):
    """Fewer than five paired runs cannot support a 95% HL upper bound."""


def walsh_averages(values) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("Walsh averages need at least one value")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("Walsh averages need finite values")
    return sorted(
        (samples[left] + samples[right]) / 2.0
        for left in range(len(samples))
        for right in range(left, len(samples))
    )


def signed_rank_counts(n: int) -> list[int]:
    """Exact null counts of T+ for n observations, by subset-sum DP."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for mass in range(total, rank - 1, -1):
            counts[mass] += counts[mass - rank]
    return counts


def hl_upper_bound(values, confidence: float = CONFIDENCE) -> dict:
    """Hodges-Lehmann estimate and one-sided distribution-free upper bound.

    The bound is an order statistic of the Walsh averages.  The index comes
    from the exact null distribution of the Wilcoxon signed-rank statistic for
    n <= 50 (subset-sum DP) and from the continuity-corrected normal
    approximation above 50.  The smallest index whose one-sided coverage is at
    least ``confidence`` is selected, so the bound is conservative.
    """
    samples = [float(value) for value in values]
    n = len(samples)
    if n < MIN_RUNS_FOR_BOUND:
        raise InsufficientRuns(
            f"{n} paired runs cannot support a 95% HL bound; need at least "
            f"{MIN_RUNS_FOR_BOUND}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("HL bounds need finite values")
    walsh = walsh_averages(samples)
    order_statistics = len(walsh)
    alpha = 1.0 - confidence
    if n <= 50:
        counts = signed_rank_counts(n)
        total = 1 << n
        cumulative = 0
        critical = -1
        cumulative_at_critical = 0
        for mass, count in enumerate(counts):
            cumulative += count
            if cumulative / total <= alpha:
                critical = mass
                cumulative_at_critical = cumulative
            else:
                break
        if critical < 0:
            raise ValueError(
                f"no exact one-sided {confidence:.3f} Wilcoxon bound for n={n}"
            )
        coverage = 1.0 - cumulative_at_critical / total
        exact = True
    else:
        mean = n * (n + 1) / 4.0
        deviation = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        normal = statistics.NormalDist()
        z = normal.inv_cdf(confidence)
        critical = math.floor(mean - z * deviation - 0.5)
        critical = max(0, min(order_statistics - 1, critical))
        coverage = 1.0 - normal.cdf((critical + 0.5 - mean) / deviation)
        exact = False
    index = order_statistics - critical
    if not 1 <= index <= order_statistics:
        raise ValueError("HL order-statistic index is out of range")
    return {
        "method": ESTIMATOR,
        "confidence": confidence,
        "n": n,
        "estimate": statistics.median(walsh),
        "upper_bound": walsh[index - 1],
        "index": index,
        "order_statistics": order_statistics,
        "coverage": coverage,
        "exact": exact,
    }


def paired_log_ratios(base_runs: list[dict], candidate_runs: list[dict], cells) -> dict:
    """Per-cell log(candidate/base), one value per paired independent run."""
    if len(base_runs) != len(candidate_runs):
        raise ValueError("both arms must contribute the same number of runs")
    ratios = {}
    for cell in cells:
        values = []
        for base, candidate in zip(base_runs, candidate_runs):
            base_value = base.get(cell)
            candidate_value = candidate.get(cell)
            if base_value is None or candidate_value is None:
                raise KeyError(cell)
            if base_value <= 0 or candidate_value <= 0:
                raise ValueError(f"nonpositive timing for {cell}")
            values.append(math.log(candidate_value / base_value))
        ratios[cell] = values
    return ratios


def evaluate_noninferiority(
    ratios: dict, delta: float, excluded=()
) -> dict:
    limit = math.log1p(delta)
    excluded_cells = set(excluded)
    per_cell = {}
    excluded_results = {}
    failed = []
    for cell, values in ratios.items():
        result = hl_upper_bound(values)
        if cell in excluded_cells:
            excluded_results[cell] = {
                "estimate": result["estimate"],
                "upper_bound": result["upper_bound"],
                "index": result["index"],
                "coverage": result["coverage"],
                "limit": limit,
                "excluded": "unstable",
            }
            continue
        passes = result["upper_bound"] <= limit
        per_cell[cell] = {
            "estimate": result["estimate"],
            "upper_bound": result["upper_bound"],
            "index": result["index"],
            "coverage": result["coverage"],
            "limit": limit,
            "pass": passes,
        }
        if not passes:
            failed.append(cell)
    return {
        "passed": not failed,
        "delta": delta,
        "limit": limit,
        "estimator": ESTIMATOR,
        "confidence": CONFIDENCE,
        "per_cell": per_cell,
        "excluded_unstable": excluded_results,
        "failed_cells": failed,
    }


def evaluate_superiority(
    ratios: dict,
    targets,
    gamma: float,
    delta: float,
    declared: bool = True,
    excluded=(),
    unstable_targets=(),
) -> dict:
    target_list = list(dict.fromkeys(targets))
    excluded_cells = set(excluded)
    unstable_target_set = set(unstable_targets)
    limit = math.log1p(-gamma)
    noninferiority_limit = math.log1p(delta)
    per_target = {}
    failed = []
    for cell in target_list:
        values = ratios.get(cell)
        if values is None:
            raise KeyError(cell)
        result = hl_upper_bound(values)
        passes = result["upper_bound"] < limit
        per_target[cell] = {
            "estimate": result["estimate"],
            "upper_bound": result["upper_bound"],
            "index": result["index"],
            "coverage": result["coverage"],
            "limit": limit,
            "pass": passes,
            "unstable": cell in unstable_target_set,
        }
        if not passes:
            failed.append(cell)
    others = [
        cell for cell in ratios
        if cell not in target_list and cell not in excluded_cells
    ]
    other_results = {cell: hl_upper_bound(ratios[cell]) for cell in others}
    other_bounds = {
        cell: result["upper_bound"] for cell, result in other_results.items()
    }
    others_noninferior = all(
        bound <= noninferiority_limit for bound in other_bounds.values()
    )
    excluded_results = {}
    for cell in sorted(excluded_cells - set(target_list)):
        if cell not in ratios:
            continue
        result = hl_upper_bound(ratios[cell])
        excluded_results[cell] = {
            "estimate": result["estimate"],
            "upper_bound": result["upper_bound"],
            "index": result["index"],
            "coverage": result["coverage"],
            "limit": noninferiority_limit,
            "excluded": "unstable",
        }
    return {
        "declared": declared,
        "gamma": gamma,
        "delta": delta,
        "limit": limit,
        "targets": target_list,
        "per_target": per_target,
        "failed_targets": failed,
        "others_noninferior": others_noninferior,
        "other_bounds": other_bounds,
        "other_cells": other_results,
        "excluded_unstable": excluded_results,
        "passed": bool(target_list) and not failed and others_noninferior,
    }


def confirmation_cells(noninferiority: dict, targets=()) -> list[str]:
    """Main-stage failures eligible for fresh noninferiority confirmation."""
    target_set = set(targets or ())
    return [
        cell for cell in noninferiority["failed_cells"] if cell not in target_set
    ]


def evaluate_staged_decisions(
    ratios: dict, confirmation_ratios: dict, delta: float, gamma: float,
    targets=None, excluded=(), unstable_targets=(),
) -> dict:
    """Decide noninferiority from separate stages and superiority from main runs."""
    main = evaluate_noninferiority(ratios, delta, excluded)
    expected = confirmation_cells(main, targets)
    if set(confirmation_ratios) != set(expected):
        raise ValueError("confirmation cells do not match eligible main failures")
    confirmation = evaluate_noninferiority(confirmation_ratios, delta)
    per_cell = {}
    for cell, main_result in main["per_cell"].items():
        confirmation_result = confirmation["per_cell"].get(cell)
        effective = confirmation_result or main_result
        per_cell[cell] = {
            **effective,
            "main": main_result,
            "confirmation": confirmation_result,
        }
    failed = [cell for cell, result in per_cell.items() if not result["pass"]]
    confirmed = confirmation["failed_cells"]
    noninferiority = {
        **main,
        "per_cell": per_cell,
        "passed": not failed,
        "failed_cells": failed,
        "main_failed_cells": main["failed_cells"],
        "confirmed_failures": confirmed,
        "unconfirmed_failures": [
            cell for cell in expected if cell not in confirmed
        ],
    }
    if targets is None:
        superiority = {"declared": False, "passed": None}
    else:
        superiority = evaluate_superiority(
            ratios, targets, gamma, delta, excluded=excluded,
            unstable_targets=unstable_targets,
        )
        others = {
            cell: result for cell, result in per_cell.items()
            if cell not in set(targets)
        }
        superiority["other_cells"] = others
        superiority["other_bounds"] = {
            cell: result["upper_bound"] for cell, result in others.items()
        }
        superiority["others_noninferior"] = all(
            result["pass"] for result in others.values()
        )
        superiority["passed"] = (
            bool(targets) and not superiority["failed_targets"]
            and superiority["others_noninferior"]
        )
    return {
        "noninferiority": noninferiority,
        "superiority": superiority,
        "confirmed_failures": confirmed,
    }


# --------------------------------------------------------------------------
# Drift, calibration recommendation, layout spread
# --------------------------------------------------------------------------


def balanced_order(run_index: int) -> list[str]:
    """AB in odd-numbered runs, BA in even-numbered runs."""
    return ["base", "candidate"] if run_index % 2 == 0 else ["candidate", "base"]


def run_arm_summary(round_timings) -> float:
    """An arm's run summary: the median of its rounds."""
    values = [float(value) for value in round_timings]
    if not values:
        raise ValueError("an arm summary needs at least one round")
    return statistics.median(values)


def start_controls(runs) -> list[float]:
    """Every arm's start control across the independent runs of one tier."""
    samples = []
    for run in runs:
        controls = run.get("controls") or {}
        if controls:
            samples.extend(float(entry["start_ns"]) for entry in controls.values())
        elif run.get("control") is not None:
            samples.append(float(run["control"]["start_ns"]))
    return samples


def session_tier_drift(session: dict, stage: str = "main") -> dict:
    """Per-tier start-control drift under the current method."""
    runs = [
        run for run in session.get("runs") or []
        if run.get("stage", "main") == stage
    ]
    tiers = sorted({
        run.get("tier") for run in runs if isinstance(run.get("tier"), str)
    })
    return {
        tier: drift_stats(
            start_controls([run for run in runs if run.get("tier") == tier])
        )
        for tier in tiers
    }


def percentile(values, fraction: float) -> float | None:
    """Linear-interpolated percentile (0..1) of a numeric sample."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def drift_stats(samples) -> dict:
    """Drift = interquartile range / median of the start controls.

    The max/min spread is retained for diagnosis but does not drive the cap.
    """
    values = [float(value) for value in samples]
    if not values:
        return {
            "n": 0,
            "drift": None,
            "drift_method": DRIFT_METHOD,
            "median": None,
            "p10": None,
            "p90": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
            "spread": None,
            "samples": [],
        }
    ordered = sorted(values)
    median = statistics.median(ordered)
    p10 = percentile(ordered, 0.10)
    p90 = percentile(ordered, 0.90)
    p25 = percentile(ordered, 0.25)
    p75 = percentile(ordered, 0.75)
    spread = ordered[-1] - ordered[0]
    ratio = (p75 - p25) / median if median > 0 else None
    return {
        "n": len(ordered),
        "drift": ratio,
        "drift_method": DRIFT_METHOD,
        "median": median,
        "p10": p10,
        "p90": p90,
        "p25": p25,
        "p75": p75,
        "min": ordered[0],
        "max": ordered[-1],
        "spread": spread,
        "samples": ordered,
    }


def check_drift(samples, cap: float) -> tuple[dict, Rejection | None]:
    stats = drift_stats(samples)
    if stats["drift"] is None:
        return stats, reject(
            "drift_control_invalid",
            "start controls have no positive median; drift is undefined",
        )
    if stats["drift"] > cap:
        return stats, reject(
            "drift_cap",
            f"start-control drift {stats['drift']:.6f} exceeds the cap {cap:.6f}",
        )
    return stats, None


def recommend_run_count(
    ratios_by_cell: dict, delta: float, minimum: int = MIN_RUNS_FOR_BOUND
) -> dict | None:
    """Smallest run count with every calibration cell's HL bound at most log(1+d/2)."""
    if not ratios_by_cell:
        return None
    available = min(len(values) for values in ratios_by_cell.values())
    limit = math.log1p(delta / 2.0)
    for runs in range(minimum, available + 1):
        results = {
            cell: hl_upper_bound(values[:runs])
            for cell, values in ratios_by_cell.items()
        }
        bounds = {
            cell: result["upper_bound"] for cell, result in results.items()
        }
        if all(bound <= limit for bound in bounds.values()):
            return {
                "runs": runs,
                "limit": limit,
                "max_bound": max(bounds.values()),
                "bounds": bounds,
                "estimates": {
                    cell: result["estimate"]
                    for cell, result in results.items()
                },
                "cells": {
                    cell: {
                        "estimate": result["estimate"],
                        "upper_bound": result["upper_bound"],
                    }
                    for cell, result in results.items()
                },
            }
    return None


def recommend_drift_cap(
    observed: float | None, safety_factor: float = 1.25
) -> float:
    if observed is None or observed <= 0:
        return 0.0001
    return math.ceil(observed * safety_factor * 1e4) / 1e4


def calibration_recommendation(
    ratios_by_tier: dict,
    delta: float,
    drift_by_tier: dict,
    *,
    drift_safety_factor: float = 1.25,
    prior_drift: float = 0.0,
):
    """Overall run count and drift cap from one calibration session's ratios.

    Unstable cells are those whose full-budget HL upper bound exceeds
    log(1 + delta/2); they are reported but do not make the session fail.
    The run count is the smallest prefix for which every non-unstable cell's
    bound is within that limit.
    """
    per_tier = {}
    recommended_runs = 0
    observed_drift = float(prior_drift or 0.0)
    healthy = True
    stable_bounds = []
    limit = math.log1p(delta / 2.0)
    for tier, ratios in ratios_by_tier.items():
        observed_drift = max(
            observed_drift, (drift_by_tier.get(tier) or {}).get("drift") or 0.0
        )
        available = min((len(values) for values in ratios.values()), default=0)
        if available < MIN_RUNS_FOR_BOUND:
            per_tier[tier] = {
                "runs": None,
                "available_runs": available,
                "limit": limit,
                "unstable_cells": [],
                "stable_cells": sorted(ratios),
                "max_bound": None,
                "max_bound_signed": None,
                "cells_over_limit": 0,
                "worst_cells": [],
                "cells": {},
                "estimates": {},
                "bounds": {},
            }
            healthy = False
            continue
        full = {
            cell: hl_upper_bound(values[:available])
            for cell, values in ratios.items()
        }
        unstable = sorted(
            cell for cell, result in full.items()
            if result["upper_bound"] > limit
        )
        stable = sorted(set(ratios) - set(unstable))
        stable_runs = None
        chosen = {}
        if not stable:
            stable_runs = MIN_RUNS_FOR_BOUND
        else:
            for runs in range(MIN_RUNS_FOR_BOUND, available + 1):
                candidate = {
                    cell: hl_upper_bound(ratios[cell][:runs])
                    for cell in stable
                }
                if all(
                    result["upper_bound"] <= limit
                    for result in candidate.values()
                ):
                    stable_runs = runs
                    chosen = candidate
                    break
        if stable_runs is None:
            healthy = False
            stable_results = {cell: full[cell] for cell in stable}
            bounds = {
                cell: result["upper_bound"]
                for cell, result in stable_results.items()
            }
            worst = sorted(bounds.items(), key=lambda item: item[1], reverse=True)
            reached = max(bounds.values()) if bounds else None
            per_tier[tier] = {
                "runs": None,
                "available_runs": available,
                "limit": limit,
                "unstable_cells": unstable,
                "stable_cells": stable,
                "max_bound": max(0.0, reached) if reached is not None else 0.0,
                "max_bound_signed": reached if reached is not None else 0.0,
                "cells_over_limit": sum(1 for b in bounds.values() if b > limit),
                "worst_cells": [[cell, bound] for cell, bound in worst[:10]],
                "cells": {
                    cell: {
                        "estimate": result["estimate"],
                        "upper_bound": result["upper_bound"],
                        "unstable": cell in unstable,
                        "at_runs": available,
                    }
                    for cell, result in full.items()
                },
                "estimates": {
                    cell: result["estimate"] for cell, result in full.items()
                },
                "bounds": {
                    cell: result["upper_bound"] for cell, result in full.items()
                },
            }
            if reached is not None:
                stable_bounds.append(reached)
        else:
            chosen = chosen or {cell: full[cell] for cell in stable}
            bounds = {
                cell: result["upper_bound"]
                for cell, result in chosen.items()
            }
            worst = sorted(bounds.items(), key=lambda item: item[1], reverse=True)
            reached = max(bounds.values()) if bounds else 0.0
            per_tier[tier] = {
                "runs": stable_runs,
                "available_runs": available,
                "limit": limit,
                "unstable_cells": unstable,
                "stable_cells": stable,
                "max_bound": max(0.0, reached),
                "max_bound_signed": reached,
                "cells_over_limit": 0,
                "worst_cells": [[cell, bound] for cell, bound in worst[:10]],
                "cells": {
                    cell: {
                        "estimate": full[cell]["estimate"],
                        "upper_bound": (
                            bounds[cell] if cell in bounds else full[cell]["upper_bound"]
                        ),
                        "unstable": cell in unstable,
                        "at_runs": stable_runs,
                    }
                    for cell in sorted(full)
                },
                "estimates": {
                    cell: full[cell]["estimate"] for cell in sorted(full)
                },
                "bounds": {
                    cell: (
                        bounds[cell] if cell in bounds else full[cell]["upper_bound"]
                    )
                    for cell in sorted(full)
                },
            }
            if reached is not None:
                stable_bounds.append(reached)
            recommended_runs = max(recommended_runs, stable_runs)
    signed = max(stable_bounds) if stable_bounds else 0.0
    return {
        "runs": recommended_runs if healthy else None,
        "estimator": ESTIMATOR,
        "confidence": CONFIDENCE,
        "drift_cap": recommend_drift_cap(observed_drift, drift_safety_factor),
        "max_bound": max(0.0, signed),
        "max_bound_signed": signed,
        "limit": limit,
        "drift_safety_factor": drift_safety_factor,
        "per_tier": per_tier,
    }, healthy


def layout_spreads(
    sessions,
    host: dict,
    tier: str,
    *,
    snapshot: dict | None = None,
    toolchain_id: str | None = None,
) -> dict | None:
    """Largest layout-control upper bound across accepted, matching sessions."""
    spreads: dict[str, float] = {}
    matched = False
    snapshot = snapshot or {}
    for session in sessions:
        if session.get("kind") != "calibration":
            continue
        if session.get("rejected") is True or session.get("rejections"):
            continue
        calibration = session.get("calibration") or {}
        if calibration.get("mode") != "base_vs_layout_control":
            continue
        if not same_host(session.get("host") or {}, host):
            continue
        session_snapshot = session.get("snapshot") or {}
        if (session.get("host") or {}).get("toolchain_id") != toolchain_id:
            continue
        if normalize_flags(session_snapshot.get("rustflags") or "") != normalize_flags(
            snapshot.get("rustflags") or ""
        ):
            continue
        if session_snapshot.get("digest") != snapshot.get("digest"):
            continue
        ratios = session_tier_ratios(session, tier)
        if ratios is None:
            continue
        matched = True
        for cell, values in ratios.items():
            if len(values) < MIN_RUNS_FOR_BOUND:
                continue
            bound = hl_upper_bound(values)["upper_bound"]
            if cell not in spreads or bound > spreads[cell]:
                spreads[cell] = bound
    if not matched:
        return None
    return spreads


def layout_disposition(failed_bound: float, spread: float | None) -> dict:
    if spread is None:
        return {
            "failed_bound": failed_bound,
            "spread": None,
            "disposition": "spread_unavailable",
        }
    within = failed_bound < spread or math.isclose(
        failed_bound, spread, rel_tol=1e-9, abs_tol=1e-12
    )
    if within:
        return {
            "failed_bound": failed_bound,
            "spread": spread,
            "disposition": "owner_exception_required",
            "owner_signature": None,
        }
    return {
        "failed_bound": failed_bound,
        "spread": spread,
        "disposition": "regression_exceeds_spread",
        "owner_signature": None,
    }


def session_tier_ratios(
    session: dict, tier: str, stage: str = "main"
) -> dict | None:
    runs = [
        run
        for run in session.get("runs") or []
        if run.get("tier") == tier and run.get("stage", "main") == stage
    ]
    if len(runs) < 2:
        return None

    def arm_runs(arm: str):
        collected = []
        for run in runs:
            cells = ((run.get("arms") or {}).get(arm) or {}).get("cells") or {}
            collected.append(
                {
                    cell: entry.get("summary_ns")
                    for cell, entry in cells.items()
                    if entry.get("summary_ns") is not None
                }
            )
        return collected

    base_arm = arm_runs("base")
    candidate_arm = arm_runs("candidate")
    cells = sorted(set.intersection(*(set(run) for run in base_arm + candidate_arm)))
    try:
        return paired_log_ratios(base_arm, candidate_arm, cells)
    except (KeyError, ValueError):
        return None


def recorded_staged_decision(session: dict, tier: str, targets=None) -> dict:
    """Recompute staged decisions, rejecting incomplete or mismatched run scopes."""
    config = session["config"]
    excluded = ((session.get("unstable") or {}).get("entries") or {}).get(tier, [])
    main_ratios = session_tier_ratios(session, tier) or {}
    cells = (session.get("scope") or {}).get("cells") or list(main_ratios)
    if not cells:
        raise ValueError(f"tier {tier} has no main-stage cells")
    if set(main_ratios) != set(cells):
        raise ValueError(f"tier {tier} has mismatched main cells")
    main_ratios = {cell: main_ratios[cell] for cell in cells}
    main = evaluate_noninferiority(main_ratios, float(config["delta"]), excluded)
    expected_confirmation = confirmation_cells(main, targets)
    ratios_by_stage = {}
    for stage, expected in (("main", cells), ("confirmation", expected_confirmation)):
        runs = [
            run for run in session.get("runs") or []
            if run.get("tier") == tier and run.get("stage") == stage
        ]
        count = int(config["runs"]) if expected else 0
        if len(runs) != count or {run["run"] for run in runs} != set(range(count)):
            raise ValueError(f"tier {tier} has incomplete {stage} runs")
        for run in runs:
            for arm in ("base", "candidate"):
                observed = ((run.get("arms") or {}).get(arm) or {}).get("cells") or {}
                if set(observed) != set(expected):
                    raise ValueError(f"tier {tier} has mismatched {stage} cells")
        ratios = session_tier_ratios(session, tier, stage) or {}
        ratios_by_stage[stage] = {cell: ratios[cell] for cell in expected}
    return evaluate_staged_decisions(
        ratios_by_stage["main"], ratios_by_stage["confirmation"],
        float(config["delta"]), float(config["gamma"]),
        targets=targets, excluded=excluded,
        unstable_targets=set(excluded) & set(targets or ()),
    )


# --------------------------------------------------------------------------
# Threshold declarations and inventory grid
# --------------------------------------------------------------------------


def load_inventory_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = None
        for key in ("items", "entries", "inventory"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if items is None:
            raise ValueError("inventory object has no item list")
    else:
        raise ValueError("inventory root is neither a list nor an object")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("inventory entries must be objects")
    return items


def find_inventory_item(items: list[dict], key: str) -> dict | None:
    """Find an inventory item by its full key only."""
    for item in items:
        if str(item.get("key", "")) == key:
            return item
    return None


def numeric_literal(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("_", "")
    if not text:
        return None
    try:
        return float(int(text, 0))
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def geometry_axis(axis: str | None) -> str | None:
    """Map an inventory axis label to the cell-id geometry field it reads."""
    text = (axis or "").casefold()
    if "lane" in text:
        return None
    if (
        "row length" in text
        or "destination length" in text
        or "buffer length" in text
    ):
        return "row_len"
    if (
        "row count" in text
        or "remaining row count" in text
        or "output row count" in text
        or "row group width" in text
        or "outputs per delayed tile" in text
    ):
        return "rows"
    if (
        "source count" in text
        or "sources per register group" in text
        or "cohort fill" in text
    ):
        return "sources"
    return None


def threshold_grid_points(old: float, new: float, unit: str) -> list[float]:
    """Grid values and integral midpoints between the old and new cutoff."""
    low, high = sorted((float(old), float(new)))
    unit = (unit or "").lower()
    if unit in ("bytes", "lanes", "elements"):
        points = {low, high}
        value = low
        if value <= 0.0:
            value = high
        while value * 2.0 <= high:
            value *= 2.0
            points.add(value)
    elif unit in ("count", "rows", "terms"):
        points = set(range(int(math.floor(low)), int(math.ceil(high)) + 1))
    else:
        points = {low, high}
    ordered = sorted(points)
    midpoints = set()
    for left, right in zip(ordered, ordered[1:]):
        middle = (left + right) / 2.0
        if left < middle < right and float(middle).is_integer():
            midpoints.add(middle)
    return sorted(points | midpoints)


def _same_number(left, right) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=1e-9, abs_tol=1e-9
    )


def threshold_route_differs(axis_value: float, old: float, new: float) -> bool:
    """Whether a cutoff change moves this axis value to the other route side."""
    return (axis_value < old) != (axis_value < new)


def inventory_digest(entries, order: list[str]) -> str:
    """Content digest of the measured cell inventory, in inventory order."""
    if isinstance(entries, dict):
        by_cell = entries
    else:
        by_cell = {entry["id"]: entry for entry in entries}
    canonical = [by_cell[cell] for cell in order if cell in by_cell]
    return sha256_bytes(canonical_json(canonical).encode("utf-8"))


def load_unstable_entries(path: Path):
    """Return (entries, rejection); the file must exist and be well formed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, reject(
            "unstable_list_missing",
            f"the signed unstable-cell list is missing: {path}",
        )
    except (OSError, json.JSONDecodeError) as error:
        return None, reject(
            "unstable_list_stale",
            f"cannot read the unstable-cell list {path}: {error}",
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        return None, reject(
            "unstable_list_stale",
            f"the unstable-cell list {path} must contain an entries array",
        )
    entries = []
    for index, entry in enumerate(payload["entries"]):
        if not isinstance(entry, dict):
            return None, reject(
                "unstable_list_stale",
                f"{path} entry {index} is not an object",
            )
        required = (
            "host", "tier", "toolchain", "flags", "inventory_digest",
            "cells", "sources", "owner_signature",
        )
        if any(key not in entry for key in required):
            return None, reject(
                "unstable_list_stale",
                f"{path} entry {index} is missing required fields",
            )
        if not isinstance(entry["cells"], list) or not all(
            isinstance(cell, str) for cell in entry["cells"]
        ):
            return None, reject(
                "unstable_list_stale",
                f"{path} entry {index} has malformed cells",
            )
        if not isinstance(entry["sources"], list) or not all(
            isinstance(source, str) for source in entry["sources"]
        ):
            return None, reject(
                "unstable_list_stale",
                f"{path} entry {index} has malformed sources",
            )
        signature = entry.get("owner_signature")
        if signature is not None and not isinstance(signature, str):
            return None, reject(
                "unstable_list_stale",
                f"{path} entry {index} has a malformed owner signature",
            )
        entries.append(entry)
    return entries, None




def load_calibrated_inventory(root: Path, digest: str):
    """Return (entries, order, rejection) for one calibrated snapshot.

    The snapshot under ``root`` is a raw JSON array of CELL records named by
    its own canonical inventory digest; a snapshot whose content does not
    hash to ``digest`` cannot vouch for the entry that cites it.
    """
    path = root / f"{digest}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None, reject(
            "calibrated_inventory_missing",
            f"no calibrated inventory snapshot for digest {digest} at {path}",
        )
    except (OSError, json.JSONDecodeError) as error:
        return None, None, reject(
            "calibrated_inventory_stale",
            f"cannot read the calibrated inventory {path}: {error}",
        )
    if not isinstance(payload, list) or not payload:
        return None, None, reject(
            "calibrated_inventory_stale",
            f"the calibrated inventory {path} must be a nonempty CELL array",
        )
    entries = {}
    order = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or validate_inventory_entry(entry) is not None:
            return None, None, reject(
                "calibrated_inventory_stale",
                f"calibrated inventory {path} entry {index} is not a CELL record",
            )
        cell = str(entry["id"])
        if cell in entries:
            return None, None, reject(
                "calibrated_inventory_stale",
                f"calibrated inventory {path} repeats cell {cell}",
                cells=(cell,),
            )
        entries[cell] = entry
        order.append(cell)
    if inventory_digest(entries, order) != digest:
        return None, None, reject(
            "calibrated_inventory_stale",
            f"calibrated inventory {path} does not hash to {digest}",
        )
    return entries, order, None


def unstable_cells_for_inventory(
    entries: list[dict],
    host: dict,
    tier: str,
    toolchain_id: str | None,
    flags: str,
    live_inventory: dict,
    scope_ids,
    calibrated_root: Path,
):
    """Match one signed unstable entry without the live inventory digest.

    Exactly one signed entry must bind (host, tier, toolchain, flags); the
    live full-inventory digest plays no part in the match, so appending cells
    to the bench inventory does not stale signed entries.  The matched entry's
    calibrated inventory snapshot must hash to the entry digest, and every
    selected cell recorded there must still match the live CELL record.
    Selected cells absent from the calibrated catalogue stay in scope,
    uncalibrated, and are never excluded; exclusions are the calibrated
    unstable cells inside the scope only.  ``scope_ids`` restricts the
    compared and excluded cells; ``None`` selects the whole live inventory.

    Returns (exclusions, uncalibrated, rejection).
    """
    hostname = host.get("hostname")
    candidates = [
        entry for entry in entries
        if entry.get("host") == hostname and entry.get("tier") == tier
    ]
    if not candidates:
        return None, None, reject(
            "unstable_list_missing",
            f"no unstable-cell entry for host {hostname!r} tier {tier!r}",
        )
    exact = [
        entry for entry in candidates
        if entry.get("toolchain") == toolchain_id
        and normalize_flags(entry.get("flags") or "") == normalize_flags(flags)
    ]
    if not exact:
        return None, None, reject(
            "unstable_list_stale",
            f"unstable-cell entry for host {hostname!r} tier {tier!r} does "
            "not match the toolchain or flags",
        )
    if len(exact) != 1:
        return None, None, reject(
            "unstable_list_stale",
            f"unstable-cell entry for host {hostname!r} tier {tier!r} is duplicated",
        )
    entry = exact[0]
    if not entry.get("owner_signature"):
        return None, None, reject(
            "unstable_list_unsigned",
            f"unstable-cell entry for host {hostname!r} tier {tier!r} has no "
            "owner signature",
        )
    digest = entry.get("inventory_digest")
    catalogue, _order, problem = load_calibrated_inventory(calibrated_root, digest)
    if problem is not None:
        return None, None, problem
    selected = (
        set(live_inventory) if scope_ids is None
        else set(scope_ids) & set(live_inventory)
    )
    calibrated = selected & set(catalogue)
    redefined = [
        cell for cell in sorted(calibrated)
        if catalogue[cell] != live_inventory[cell]
    ]
    if redefined:
        return None, None, reject(
            "calibrated_cell_redefined",
            "scoped cells differ from the calibrated inventory "
            f"{digest}: " + ", ".join(redefined),
            cells=tuple(redefined),
        )
    listed = set(entry.get("cells") or [])
    return sorted(calibrated & listed), sorted(selected - set(catalogue)), None


def validate_threshold_declaration(declaration: dict, items: list[dict]):
    """Return (info, rejection).  Rejections are declaration-only, pre-measurement."""
    if not isinstance(declaration, dict):
        return None, reject(
            "declaration_invalid", "threshold declaration must be a JSON object"
        )
    item_key = declaration.get("item") or declaration.get("item_key")
    if not isinstance(item_key, str) or not item_key:
        return None, reject(
            "declaration_invalid", "threshold declaration needs a nonempty item key"
        )
    item = find_inventory_item(items, item_key)
    if item is None:
        return None, reject(
            "threshold_item_unknown",
            f"threshold declaration names an item absent from the inventory: {item_key}",
        )
    if item.get("class") == "structural":
        return None, reject(
            "threshold_item_structural",
            f"structural inventory item {item_key!r} cannot carry threshold evidence",
        )
    item_value = numeric_literal(item.get("value"))
    if item_value is None:
        return None, reject(
            "threshold_item_value",
            f"inventory item {item_key!r} has no numeric cutoff value",
        )
    item_axis = item.get("axis")
    if not isinstance(item_axis, str) or not item_axis:
        return None, reject(
            "declaration_invalid",
            f"inventory item {item_key!r} has no axis label",
        )
    item_unit = item.get("unit")
    declared_unit = declaration.get("unit")
    if declared_unit is not None and item_unit and declared_unit != item_unit:
        return None, reject(
            "threshold_unit_mismatch",
            f"declared unit {declared_unit!r} differs from the inventory unit {item_unit!r}",
        )
    unit = item_unit or declared_unit or "other"
    old = declaration.get("old", declaration.get("old_value"))
    new = declaration.get("new", declaration.get("new_value"))
    if old is None or new is None:
        return None, reject(
            "declaration_invalid",
            "threshold declaration needs both the old and the new cutoff",
        )
    try:
        old_value = float(old)
        new_value = float(new)
    except (TypeError, ValueError):
        return None, reject(
            "declaration_invalid", "threshold cutoffs must be numbers"
        )
    if old_value < 0 or new_value < 0:
        return None, reject(
            "declaration_invalid", "threshold cutoffs must be nonnegative"
        )
    if not _same_number(old_value, item_value):
        return None, reject(
            "threshold_old_mismatch",
            f"declared old cutoff {old_value:g} differs from the inventory "
            f"item value {item_value:g}",
        )
    cells = declaration.get("cells")
    if not isinstance(cells, list) or not cells:
        return None, reject(
            "declaration_invalid", "threshold declaration needs a nonempty cell list"
        )
    entries = []
    axis_values = []
    seen_cells = set()
    for index, entry in enumerate(cells):
        if not isinstance(entry, dict):
            return None, reject(
                "declaration_invalid",
                f"threshold cell entry {index} is not an object",
            )
        cell = entry.get("cell", entry.get("id"))
        axis_value = entry.get("axis_value", entry.get("value"))
        route_old = entry.get("route_old")
        route_new = entry.get("route_new")
        if (
            not isinstance(cell, str)
            or cell_parts(cell) is None
            or axis_value is None
            or not isinstance(route_old, str)
            or not isinstance(route_new, str)
        ):
            return None, reject(
                "declaration_invalid",
                f"threshold cell entry {index} needs cell, axis_value, route_old, route_new",
            )
        if cell in seen_cells:
            return None, reject(
                "threshold_duplicate_cell",
                f"threshold declaration repeats cell {cell}",
                cells=(cell,),
            )
        seen_cells.add(cell)
        try:
            axis = float(axis_value)
        except (TypeError, ValueError):
            return None, reject(
                "declaration_invalid",
                f"threshold cell entry {index} has a nonnumeric axis value",
            )
        geometry = geometry_axis(item_axis)
        derived = None
        if geometry is not None:
            derived = float(cell_parts(cell)[geometry])
            if not _same_number(axis, derived):
                return None, reject(
                    "threshold_axis_value",
                    f"cell {cell} declares axis value {axis:g} but its "
                    f"{geometry} geometry is {derived:g}",
                    cells=(cell,),
                )
            axis = derived
        else:
            parts = cell_parts(cell)
            if parts is not None:
                # A derived geometry value is unavailable for this axis; use
                # the declared value but reject negative or nonfinite ones.
                if axis < 0 or not math.isfinite(axis):
                    return None, reject(
                        "declaration_invalid",
                        f"threshold cell entry {index} has an invalid axis value",
                    )
        derived_differs = threshold_route_differs(axis, old_value, new_value)
        declared_differs = route_old != route_new
        if derived_differs != declared_differs:
            return None, reject(
                "threshold_route_mismatch",
                f"cell {cell} declares routes {route_old!r} -> {route_new!r} "
                f"but axis value {axis:g} relative to the cutoffs "
                f"{old_value:g} -> {new_value:g} says "
                f"{'different' if derived_differs else 'unchanged'} routes",
                cells=(cell,),
            )
        entries.append(
            {
                "cell": cell,
                "axis_value": axis,
                "route_old": route_old,
                "route_new": route_new,
                "route_differs": derived_differs,
            }
        )
        axis_values.append(axis)
    span_low, span_high = sorted((old_value, new_value))
    between = [
        axis for axis in axis_values if span_low < axis < span_high
    ]
    if not between:
        return None, reject(
            "threshold_no_cell",
            "threshold declaration has no declared cell strictly between the "
            f"old cutoff {old_value:g} and the new cutoff {new_value:g}",
        )
    required = threshold_grid_points(old_value, new_value, unit)
    missing = [
        value
        for value in required
        if not any(_same_number(value, axis) for axis in axis_values)
    ]
    if missing:
        return None, reject(
            "threshold_grid",
            "threshold declaration misses grid values ("
            f"{unit}): " + ", ".join(f"{value:g}" for value in missing),
        )
    targets = list(
        dict.fromkeys(
            entry["cell"] for entry in entries if entry["route_differs"]
        )
    )
    return {
        "item_key": item_key,
        "unit": unit,
        "axis": item_axis,
        "old": old_value,
        "new": new_value,
        "cells": entries,
        "grid": required,
        "target_set": targets,
    }, None


# --------------------------------------------------------------------------
# Snapshot overlay helpers
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Snapshot:
    source: str
    resolved: str
    directory: Path
    digest: str
    meta: dict
    features: list[str]
    rustflags: str
    toolchain: str | None
    bench: str
    no_default_features: bool

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "resolved": self.resolved,
            "digest": self.digest,
            "features": self.features,
            "rustflags": self.rustflags,
            "toolchain": self.toolchain,
            "cargo_metadata_seed": self.meta.get("cargo_metadata_seed"),
            "bench": self.bench,
            "no_default_features": self.no_default_features,
        }


def _snapshot_meta(directory: Path) -> dict:
    for relative in ("benches/pair-snapshot.json", "benches/bench-pair.json"):
        candidate = directory / relative
        if candidate.is_file():
            meta = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"{relative} must contain a JSON object")
            return meta
    return {}


def _snapshot_files(directory: Path) -> list[Path]:
    files = []
    benches = directory / "benches"
    if benches.is_dir():
        for path in sorted(benches.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    for name in ("Cargo.toml", "Cargo.lock"):
        path = directory / name
        if path.is_file():
            files.append(path)
    for path in sorted(directory.glob("rust-toolchain*")):
        if path.is_file():
            files.append(path)
    return files


def snapshot_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(_snapshot_files(directory), key=lambda item: str(item.relative_to(directory))):
        relative = str(path.relative_to(directory))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_snapshot_meta(directory: Path) -> dict:
    return _snapshot_meta(directory)


def read_snapshot_toolchain(directory: Path) -> str | None:
    """The channel pinned by the snapshot's rust-toolchain file, if any."""
    toml_path = directory / "rust-toolchain.toml"
    if toml_path.is_file():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        channel = (data.get("toolchain") or {}).get("channel")
        return str(channel) if channel else None
    plain = directory / "rust-toolchain"
    if plain.is_file():
        for line in plain.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def validate_snapshot(directory: Path, bench: str = "operations") -> Rejection | None:
    missing = []
    for relative in ("Cargo.toml", "Cargo.lock"):
        if not (directory / relative).is_file():
            missing.append(relative)
    if not (directory / "benches").is_dir():
        missing.append("benches/")
    if not (directory / "benches" / f"{bench}.rs").is_file():
        missing.append(f"benches/{bench}.rs")
    if missing:
        return reject(
            "snapshot_incomplete",
            "snapshot is missing required files: " + ", ".join(missing),
        )
    return None


def merge_features(cli_features, meta_features) -> list[str]:
    merged = []
    for source in (cli_features or [], meta_features or []):
        if isinstance(source, str):
            source = split_words(source)
        for feature in source:
            feature = str(feature).strip()
            if feature and feature not in merged:
                merged.append(feature)
    return merged


def merge_rustflags(cli_flags: str, meta_flags) -> str:
    parts = [normalize_flags(cli_flags or "")]
    if isinstance(meta_flags, list):
        meta_flags = " ".join(str(part) for part in meta_flags)
    parts.append(normalize_flags(meta_flags or ""))
    return " ".join(part for part in parts if part)


def check_alignment_flags(rustflags: str) -> Rejection | None:
    missing = [token for token in REQUIRED_ALIGN_TOKENS if token not in (rustflags or "")]
    if missing:
        return reject(
            "alignment_flags_missing",
            "RUSTFLAGS do not pin 64-byte function and loop alignment; missing: "
            + ", ".join(missing),
        )
    return None


def check_tier_features(tiers, features) -> Rejection | None:
    if "v4x" in tiers and "all" not in features and "simd512" not in features:
        return reject(
            "simd512_missing",
            "tier v4x was requested but the snapshot feature set does not include simd512",
        )
    return None


def check_tiers(tiers) -> Rejection | None:
    unknown = [tier for tier in tiers if tier not in SUPPORTED_TIERS]
    if unknown:
        return reject(
            "unknown_tier",
            "unknown backend tier(s): " + ", ".join(unknown),
            cells=tuple(unknown),
        )
    if not tiers:
        return reject("unknown_tier", "no backend tiers were requested")
    return None


def check_snapshot_runner(snapshot_digest: str | None, self_digest: str) -> Rejection | None:
    if snapshot_digest is not None and snapshot_digest != self_digest:
        return reject(
            "snapshot_runner_mismatch",
            "the snapshot supplies a different bench_pair.py; run the snapshot's "
            "runner so the measurement snapshot identifies the whole campaign",
        )
    return None


def overlay_snapshot(checkout: Path, snapshot: Snapshot) -> None:
    destination = checkout / "benches"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        snapshot.directory / "benches",
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("Cargo.toml", "Cargo.lock"):
        shutil.copy2(snapshot.directory / name, checkout / name)
    for stale in checkout.glob("rust-toolchain*"):
        if stale.is_file():
            stale.unlink()
    for toolchain in snapshot.directory.glob("rust-toolchain*"):
        if toolchain.is_file():
            shutil.copy2(toolchain, checkout / toolchain.name)


# --------------------------------------------------------------------------
# Inventory, outputs, dependencies
# --------------------------------------------------------------------------


def validate_inventory_entry(entry: dict) -> Rejection | None:
    missing = [key for key in INVENTORY_KEYS if key not in entry]
    if missing:
        return reject(
            "protocol_error",
            "inventory cell is missing keys: " + ", ".join(missing),
        )
    cell = entry.get("id")
    if not isinstance(cell, str) or cell_parts(cell) is None:
        return reject(
            "protocol_error", f"inventory cell id is malformed: {cell!r}"
        )
    return None


def compare_inventories(base: dict, candidate: dict) -> Rejection | None:
    base_ids = set(base)
    candidate_ids = set(candidate)
    only_base = sorted(base_ids - candidate_ids)
    only_candidate = sorted(candidate_ids - base_ids)
    changed = sorted(
        cell
        for cell in base_ids & candidate_ids
        if canonical_json(base[cell]) != canonical_json(candidate[cell])
    )
    if only_base or only_candidate or changed:
        details = []
        if only_base:
            details.append("missing from candidate: " + ", ".join(only_base))
        if only_candidate:
            details.append("missing from base: " + ", ".join(only_candidate))
        if changed:
            details.append("differing entries: " + ", ".join(changed))
        return reject(
            "inventory_mismatch",
            "the two arms expose different cell inventories; " + "; ".join(details),
            cells=tuple(only_base + only_candidate + changed),
        )
    return None


def compare_outputs(
    base: dict, candidate: dict, order: list[str]
) -> Rejection | None:
    base_ids = set(base)
    candidate_ids = set(candidate)
    only_base = sorted(base_ids - candidate_ids)
    only_candidate = sorted(candidate_ids - base_ids)
    if only_base or only_candidate:
        return reject(
            "output_mismatch",
            "check output sets differ between arms; "
            + ("missing from candidate: " + ", ".join(only_base) if only_base else "")
            + ("; " if only_base and only_candidate else "")
            + ("missing from base: " + ", ".join(only_candidate) if only_candidate else ""),
            cells=tuple(only_base + only_candidate),
        )
    differing = [cell for cell in order if base.get(cell) != candidate.get(cell)]
    differing += [
        cell for cell in sorted(base_ids - set(order)) if base.get(cell) != candidate.get(cell)
    ]
    if differing:
        first = differing[0]
        return reject(
            "output_mismatch",
            f"check outputs differ between arms; first differing cell: {first}",
            cells=tuple(differing),
        )
    return None


def normal_dependency_set(metadata: dict) -> set[str]:
    packages = metadata.get("packages") or []
    package = next(
        (
            entry
            for entry in packages
            if entry.get("name") == "fgf" and entry.get("source") is None
        ),
        None,
    )
    if package is None:
        package = next(
            (entry for entry in packages if entry.get("name") == "fgf"), None
        )
    if package is None:
        raise ValueError("cargo metadata has no fgf package")
    dependencies = set()
    for dependency in package.get("dependencies") or []:
        kind = dependency.get("kind")
        if kind not in (None, "normal"):
            continue
        dependencies.add(
            "|".join(
                (
                    str(dependency.get("name")),
                    str(dependency.get("req")),
                    str(dependency.get("target") or ""),
                )
            )
        )
    return dependencies


def compare_dependency_sets(expected: set[str], actual: set[str], arm: str) -> Rejection | None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        return reject(
            "dependency_mismatch",
            f"{arm} normal-dependency set differs from the snapshot's; "
            + "; ".join(details),
            arm=arm,
        )
    return None


# --------------------------------------------------------------------------
# Environment and configuration validation
# --------------------------------------------------------------------------


def same_host(left: dict, right: dict) -> bool:
    """Sessions match only by exact hostname equality."""
    left_key = (left.get("hostname") or "").casefold()
    right_key = (right.get("hostname") or "").casefold()
    return bool(left_key) and left_key == right_key


def host_matches(entry: str, host: dict) -> bool:
    """Exact hostname equality; never a substring or CPU-model match."""
    needle = entry.strip().casefold()
    if not needle or needle == PLACEHOLDER:
        return False
    return needle == (host.get("hostname") or "").casefold()


def parse_host_value_binding(text, hostname: str | None, what: str = "value"):
    """Resolve ``host:value host:value`` for one exact hostname.

    A plain numeric value is accepted for a single-host campaign.  The
    placeholder resolves to ``(None, None)`` so the caller can report an
    uncalibrated configuration.
    """
    if text is None:
        return None, None
    text = str(text).strip()
    if not text or text.casefold() == PLACEHOLDER:
        return None, None
    tokens = split_words(text)
    if not tokens:
        return None, None
    if not any(":" in token for token in tokens):
        if len(tokens) != 1:
            return None, reject(
                "uncalibrated_config",
                f"{what} must be one value or host:value bindings",
            )
        try:
            return parse_number(tokens[0], what), None
        except ValueError as error:
            return None, reject("uncalibrated_config", str(error))
    entries: dict[str, float] = {}
    for token in tokens:
        if ":" not in token:
            return None, reject(
                "uncalibrated_config",
                f"{what} mixes host:value bindings with a bare value",
            )
        host, raw = token.split(":", 1)
        host = host.strip()
        raw = raw.strip()
        if not host or not raw:
            return None, reject(
                "uncalibrated_config",
                f"{what} has a malformed host:value binding {token!r}",
            )
        key = host.casefold()
        if key in entries:
            return None, reject(
                "uncalibrated_config",
                f"{what} repeats host {host!r}",
            )
        try:
            entries[key] = parse_number(raw, what)
        except ValueError as error:
            return None, reject("uncalibrated_config", str(error))
    key = (hostname or "").casefold()
    if not key or key not in entries:
        return None, reject(
            "uncalibrated_config",
            f"no per-host {what} value for host {hostname!r}",
        )
    return entries[key], None


def parse_host_tier_binding(text: str) -> dict[str, set[str]] | None:
    """Parse ``host1:t1,t2 host2:t3,...`` into a per-host tier map."""
    entries = (text or "").split()
    if not entries or any(entry.casefold() == PLACEHOLDER for entry in entries):
        return None
    binding: dict[str, set[str]] = {}
    for entry in entries:
        if ":" not in entry:
            return None
        host, tiers = entry.split(":", 1)
        host = host.strip().casefold()
        values = {tier.strip() for tier in tiers.split(",") if tier.strip()}
        if not host or not values:
            return None
        binding[host] = values
    return binding


def evaluate_environment(facts: dict, smt_threshold: float = 0.05) -> list[Rejection]:
    rejections = []
    core = facts.get("pinned_core")
    if core is None:
        rejections.append(
            reject(
                "pinned_core_missing",
                "FEC_GOLDEN_CORE is unset; a paired campaign needs a pinned core",
            )
        )
    else:
        affinity = facts.get("affinity")
        if affinity is not None and core not in affinity:
            rejections.append(
                reject(
                    "pinned_core_unavailable",
                    f"pinned core {core} is not an online CPU {sorted(affinity)}",
                )
            )
        governor = facts.get("governor")
        if governor is None:
            rejections.append(
                reject(
                    "governor_unknown",
                    f"cannot read the scaling governor for pinned core {core}",
                )
            )
        elif governor != "performance":
            rejections.append(
                reject(
                    "governor",
                    f"pinned core {core} runs under governor {governor!r}, not 'performance'",
                )
            )
    smt_active = facts.get("smt_active")
    siblings = facts.get("siblings") or []
    busy = facts.get("sibling_busy")
    if smt_active and siblings:
        if busy is None:
            rejections.append(
                reject(
                    "smt_sibling_unverified",
                    f"SMT siblings {siblings} of the pinned core could not be sampled",
                )
            )
        elif busy > smt_threshold:
            rejections.append(
                reject(
                    "smt_sibling_busy",
                    f"SMT sibling busy fraction {busy:.4f} exceeds {smt_threshold:.4f}",
                )
            )
    return rejections


def validate_smt_config(window: float, threshold: float) -> list[Rejection]:
    rejections = []
    if not math.isfinite(window) or window < 0.2:
        rejections.append(
            reject(
                "smt_window_invalid",
                f"SMT sampling window {window!r}s is below the fixed 0.2s floor",
            )
        )
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 0.05:
        rejections.append(
            reject(
                "smt_threshold_invalid",
                f"SMT busy threshold {threshold!r} is outside (0, 0.05]",
            )
        )
    return rejections


def validate_execution_environment(
    machine: str | None,
    setarch_path: str | None,
    chrt_command: str,
    chrt_path: str | None,
) -> list[Rejection]:
    rejections = []
    if not machine:
        rejections.append(
            reject("setarch_missing", "cannot determine the machine name for setarch")
        )
    if not setarch_path:
        rejections.append(
            reject(
                "setarch_missing",
                "setarch is required to disable ASLR for every timed and check "
                "bench process",
            )
        )
    if chrt_command and not chrt_path:
        rejections.append(
            reject(
                "chrt_missing",
                "chrt was configured but is not available on PATH",
            )
        )
    return rejections


def harness_command(
    executable,
    extra,
    *,
    core: int | None,
    machine: str,
    chrt_parts,
    disable_aslr: bool,
) -> list[str]:
    command = []
    if chrt_parts:
        command.extend(["chrt", *chrt_parts])
    if core is not None:
        command.extend(["taskset", "-c", str(core)])
    if disable_aslr:
        command.extend(["setarch", machine, "-R"])
    command.append(str(executable))
    command.extend(str(part) for part in extra)
    return command


def command_length(command) -> int:
    """A deterministic argv-length metric (NUL terminators included)."""
    return sum(len(str(part)) + 1 for part in command)


def environment_digest(env: dict) -> str:
    payload = "\0".join(f"{key}={env[key]}" for key in sorted(env))
    return sha256_bytes(payload.encode("utf-8", errors="replace"))


def check_argv_lengths(lengths: dict, where: str) -> Rejection | None:
    if len(set(lengths.values())) != 1:
        return reject(
            "argv_mismatch",
            f"{where} arms have different argv lengths: "
            + ", ".join(f"{arm}={length}" for arm, length in lengths.items()),
        )
    return None


def validate_policy_config(args) -> Rejection | None:
    """Refuse acceptance overrides of the crate.just policy values."""
    text = getattr(args, "policy_json", None)
    if not text:
        return None
    try:
        policy = json.loads(text)
    except json.JSONDecodeError as error:
        return reject("config_override", f"--policy-json is not valid JSON: {error}")
    if not isinstance(policy, dict):
        return reject("config_override", "--policy-json must be a JSON object")
    fields = (
        ("runs", "runs"),
        ("drift_cap", "drift_cap"),
        ("rounds", "rounds"),
        ("delta", "delta"),
        ("gamma", "gamma"),
        ("calibrated_hosts", "calibrated_hosts"),
        ("calibrated_tiers", "calibrated_tiers"),
        ("calibrated_toolchain", "calibrated_toolchain"),
        ("calibrated_flags", "calibrated_flags"),
        ("smt_window", "smt_window"),
        ("smt_busy_threshold", "smt_busy_threshold"),
        ("chrt", "chrt"),
        ("unstable_target_runs", "unstable_target_runs"),
    )
    numeric = {
        "runs", "drift_cap", "rounds", "delta", "gamma",
        "smt_window", "smt_busy_threshold",
    }
    mismatches = []
    for key, attribute in fields:
        if key not in policy:
            mismatches.append(f"{key} missing from policy")
            continue
        actual = getattr(args, attribute, None)
        actual_text = str(actual).strip()
        policy_text = str(policy[key]).strip()
        equal = actual_text == policy_text
        if not equal and key in numeric:
            try:
                equal = float(actual_text) == float(policy_text)
            except ValueError:
                equal = False
        if not equal:
            mismatches.append(
                f"{key}: argument {actual_text!r} != policy {policy_text!r}"
            )
    if mismatches:
        return reject(
            "config_override",
            "acceptance refuses policy overrides; use --exploratory for a "
            "non-publishable run: " + "; ".join(mismatches),
        )
    return None


def check_backend(requested: str, reported: str | None, arm: str) -> Rejection | None:
    if reported != requested:
        return reject(
            "backend_mismatch",
            f"{arm} resolved backend {reported!r} but {requested!r} was requested; "
            "the tier is not executable on this host or feature set",
            arm=arm,
        )
    return None


def validate_acceptance_config(
    *,
    runs,
    drift_cap,
    delta: float,
    gamma: float,
    rounds,
    calibrated_hosts,
    calibrated_tiers,
    calibrated_toolchain,
    calibrated_flags,
    host: dict,
    toolchain_id: str | None,
    effective_flags: str,
    tiers,
) -> list[Rejection]:
    rejections = []
    if runs is None:
        rejections.append(
            reject(
                "uncalibrated_config",
                "acceptance needs a calibrated --runs value; the recipe placeholder "
                "is not usable until a calibration session sets it",
            )
        )
    elif runs < MIN_RUNS_FOR_BOUND:
        rejections.append(
            reject(
                "insufficient_runs",
                f"acceptance needs at least {MIN_RUNS_FOR_BOUND} paired runs "
                "for a 95% HL bound",
            )
        )
    if drift_cap is None:
        rejections.append(
            reject(
                "uncalibrated_config",
                "acceptance needs a calibrated --drift-cap value; the recipe placeholder "
                "is not usable until a calibration session sets it",
            )
        )
    elif drift_cap <= 0:
        rejections.append(
            reject("uncalibrated_config", "the drift cap must be positive")
        )
    if rounds is None or rounds < 1:
        rejections.append(reject("uncalibrated_config", "rounds must be positive"))
    if not (0.0 < delta < 1.0):
        rejections.append(reject("uncalibrated_config", "delta must lie in (0, 1)"))
    if not (0.0 < gamma < 1.0):
        rejections.append(reject("uncalibrated_config", "gamma must lie in (0, 1)"))
    hosts = calibrated_hosts or []
    if not hosts or all(entry.strip().casefold() == PLACEHOLDER for entry in hosts):
        rejections.append(
            reject(
                "calibration_binding",
                "the calibrated host binding is unset; set PAIR_CALIBRATED_HOSTS "
                "from a calibration session before acceptance",
            )
        )
    elif not any(host_matches(entry, host) for entry in hosts):
        rejections.append(
            reject(
                "calibration_binding",
                f"host {host.get('hostname')!r} / CPU {host.get('cpu_model')!r} is not "
                "in the calibrated host binding",
            )
        )
    tier_binding = parse_host_tier_binding(
        " ".join(calibrated_tiers or [])
        if isinstance(calibrated_tiers, list)
        else str(calibrated_tiers or "")
    )
    hostname = (host.get("hostname") or "").casefold()
    if tier_binding is None:
        rejections.append(
            reject(
                "calibration_binding",
                "the calibrated tier binding is unset or malformed; set "
                "PAIR_CALIBRATED_TIERS to 'host:tiers' entries from calibration",
            )
        )
    elif hostname not in tier_binding:
        rejections.append(
            reject(
                "calibration_binding",
                f"host {host.get('hostname')!r} has no calibrated tier binding",
            )
        )
    else:
        missing = [tier for tier in tiers if tier not in tier_binding[hostname]]
        if missing:
            rejections.append(
                reject(
                    "calibration_binding",
                    "tier(s) not calibrated on this host: " + ", ".join(missing),
                    cells=tuple(missing),
                )
            )
    if not calibrated_toolchain or calibrated_toolchain.strip().casefold() == PLACEHOLDER:
        rejections.append(
            reject(
                "calibration_binding",
                "the calibrated toolchain binding is unset; set "
                "PAIR_CALIBRATED_TOOLCHAIN from a calibration session before acceptance",
            )
        )
    elif toolchain_id is None or calibrated_toolchain.strip() != toolchain_id:
        rejections.append(
            reject(
                "calibration_binding",
                f"toolchain {toolchain_id!r} differs from the calibrated binding "
                f"{calibrated_toolchain!r}",
            )
        )
    if not calibrated_flags or normalize_flags(calibrated_flags).casefold() == PLACEHOLDER:
        rejections.append(
            reject(
                "calibration_binding",
                "the calibrated flag binding is unset; set PAIR_CALIBRATED_FLAGS "
                "from a calibration session before acceptance",
            )
        )
    elif normalize_flags(calibrated_flags) != normalize_flags(effective_flags):
        rejections.append(
            reject(
                "calibration_binding",
                "the effective RUSTFLAGS differ from the calibrated flag binding",
            )
        )
    return rejections


def validate_acceptance_inputs(sessions) -> Rejection | None:
    for session in sessions:
        if session.get("kind") == "calibration" or not session.get("acceptance"):
            return reject(
                "calibration_reuse",
                "a calibration session cannot be reused as acceptance evidence; "
                "measure a fresh base-against-candidate session",
            )
    return None


def validate_targets(targets, inventory_ids) -> Rejection | None:
    unknown = [target for target in targets if target not in inventory_ids]
    if unknown:
        return reject(
            "target_unknown",
            "declared target cells are absent from the measured inventory: "
            + ", ".join(unknown),
            cells=tuple(unknown),
        )
    return None


def validate_declared_cells(cells, inventory_ids) -> Rejection | None:
    unknown = [target for target in cells if target not in inventory_ids]
    if unknown:
        return reject(
            "threshold_cell_unknown",
            "declared threshold cells are absent from the measured inventory: "
            + ", ".join(unknown),
            cells=tuple(unknown),
        )
    return None


def parse_targets_file(path: Path) -> dict:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    text = payload.decode("utf-8")
    cells = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        cells = parsed
    elif isinstance(parsed, dict):
        for key in ("cells", "targets"):
            if isinstance(parsed.get(key), list):
                cells = parsed[key]
                break
    if cells is None:
        cells = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                cells.append(line)
    resolved = []
    for entry in cells:
        if isinstance(entry, str):
            cell = entry.strip()
        elif isinstance(entry, dict):
            cell = str(entry.get("cell", entry.get("id", ""))).strip()
        else:
            cell = ""
        if cell and cell not in resolved:
            resolved.append(cell)
    if not resolved:
        raise ValueError(f"target declaration {path} names no cells")
    return {"path": str(path), "digest": digest, "cells": resolved}


def parse_threshold_file(path: Path) -> dict:
    payload = path.read_bytes()
    declaration = json.loads(payload.decode("utf-8"))
    return {"path": str(path), "digest": sha256_bytes(payload), "declaration": declaration}


def parse_scope_file(path: Path, rationale_override: str | None = None) -> dict:
    """Parse a timing-scope declaration.

    Accepted formats mirror ``parse_targets_file``: a JSON object with a
    ``cells`` array and a nonempty ``rationale``, a JSON cell-id array, or one
    cell id per line with ``#`` comments.  A declaration without its own
    rationale falls back to ``rationale_override`` (``--scope-rationale``);
    neither present is invalid.  Empty and duplicate cell entries are rejected
    here; unknown cells are rejected against the measured inventory later.
    The recorded digest covers the file bytes, so any recorded scope names the
    exact declaration it measured.
    """
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    text = payload.decode("utf-8")
    cells = None
    rationale = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        raw = parsed.get("cells")
        if isinstance(raw, list):
            cells = raw
        declared = parsed.get("rationale")
        if declared is not None:
            if not isinstance(declared, str) or not declared.strip():
                raise ValueError(f"scope declaration {path} has an empty rationale")
            rationale = declared.strip()
    elif isinstance(parsed, list):
        cells = parsed
    if cells is None:
        cells = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                cells.append(line)
    resolved = []
    seen = set()
    duplicates = []
    for entry in cells:
        if isinstance(entry, str):
            cell = entry.strip()
        elif isinstance(entry, dict):
            cell = str(entry.get("cell", entry.get("id", ""))).strip()
        else:
            cell = ""
        if not cell:
            raise ValueError(f"scope declaration {path} has an empty cell entry")
        if cell in seen:
            duplicates.append(cell)
            continue
        seen.add(cell)
        resolved.append(cell)
    if duplicates:
        raise ValueError(
            f"scope declaration {path} repeats cells: " + ", ".join(duplicates)
        )
    if not resolved:
        raise ValueError(f"scope declaration {path} names no cells")
    if rationale is None:
        override = (rationale_override or "").strip()
        if override:
            rationale = override
    if rationale is None:
        raise ValueError(
            f"scope declaration {path} carries no rationale; state one in the "
            "declaration or pass --scope-rationale"
        )
    return {
        "path": str(path),
        "digest": digest,
        "cells": resolved,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# Protocol parsing
# --------------------------------------------------------------------------


def parse_protocol(text: str):
    parsed = {
        "backend": None,
        "field_backends": {},
        "inventory": {},
        "inventory_order": [],
        "outputs": {},
        "rounds": {},
        "controls": [],
    }
    errors = []
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag == "BACKEND":
            if len(parts) != 2:
                errors.append("malformed BACKEND record")
            else:
                parsed["backend"] = parts[1]
        elif tag == "FIELD_BACKEND":
            if len(parts) != 3:
                errors.append("malformed FIELD_BACKEND record")
            else:
                parsed["field_backends"][parts[1]] = parts[2]
        elif tag == "CELL":
            if len(parts) != 2:
                errors.append("malformed CELL record")
                continue
            try:
                entry = json.loads(parts[1])
            except json.JSONDecodeError:
                errors.append("CELL payload is not JSON")
                continue
            problem = validate_inventory_entry(entry)
            if problem is not None:
                errors.append(problem.message)
                continue
            cell = entry["id"]
            if cell in parsed["inventory"]:
                errors.append(f"duplicate CELL record for {cell}")
                continue
            parsed["inventory"][cell] = entry
            parsed["inventory_order"].append(cell)
        elif tag == "OUTPUT":
            if len(parts) != 3:
                errors.append("malformed OUTPUT record")
                continue
            parsed["outputs"][parts[1]] = parts[2]
        elif tag == "ROUND":
            if len(parts) != 4:
                errors.append("malformed ROUND record")
                continue
            try:
                index = int(parts[2])
                value = float(parts[3])
            except ValueError:
                errors.append("ROUND payload is not numeric")
                continue
            parsed["rounds"].setdefault(parts[1], {})[index] = value
        elif tag == "CONTROL":
            if len(parts) != 3:
                errors.append("malformed CONTROL record")
                continue
            try:
                value = float(parts[2])
            except ValueError:
                errors.append("CONTROL payload is not numeric")
                continue
            parsed["controls"].append({"kind": parts[1], "ns": value})
    if parsed["backend"] is None:
        errors.append("missing BACKEND record")
    return parsed, errors


def round_values(parsed: dict, cell: str) -> list[float]:
    rounds = parsed["rounds"].get(cell) or {}
    return [rounds[index] for index in sorted(rounds)]


def control_value(parsed: dict, kind: str) -> float | None:
    values = [entry["ns"] for entry in parsed["controls"] if entry["kind"] == kind]
    if len(values) != 1:
        return None
    return values[0]


def summarize_timed_protocol(parsed: dict, order: list[str], rounds: int, arm: str):
    """Validate one arm's timed output and summarize each cell by round median."""
    cells = {}
    for cell in order:
        values = round_values(parsed, cell)
        if len(values) != rounds:
            return None, reject(
                "run_incomplete",
                f"{arm} timing has {len(values)} rounds for {cell}, expected {rounds}",
                cells=(cell,),
                arm=arm,
            )
        cells[cell] = {
            "summary_ns": run_arm_summary(values),
            "rounds_ns": values,
        }
    unexpected = [cell for cell in parsed["rounds"] if cell not in order]
    if unexpected:
        return None, reject(
            "run_incomplete",
            f"{arm} timing reported cells outside the inventory: "
            + ", ".join(sorted(unexpected)),
            cells=tuple(sorted(unexpected)),
            arm=arm,
        )
    return cells, None


def parse_route_diagnostic(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        match = re.match(r"^backend\s+([^=]+)=(.*)$", line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


# --------------------------------------------------------------------------
# Process execution
# --------------------------------------------------------------------------


def run_process(
    command,
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: float | None = None,
    log_path: Path | None = None,
):
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        if log_path is not None:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("$ " + shlex.join(str(part) for part in command) + "\n")
                handle.write(f"timeout after {timeout}s\n")
        output = expired.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        process = subprocess.CompletedProcess(
            [str(part) for part in command],
            124,
            output,
            f"timeout after {timeout}s",
        )
    except OSError as error:
        process = subprocess.CompletedProcess(
            [str(part) for part in command],
            127,
            "",
            str(error),
        )
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as handle:
            if "timeout after" not in (process.stderr or ""):
                handle.write("$ " + shlex.join(str(part) for part in command) + "\n")
                handle.write(f"exit={process.returncode}\n")
            handle.write(process.stdout or "")
            if process.stderr:
                handle.write("\n[stderr]\n" + process.stderr)
    return process


def git_revision(repo: Path, revision: str) -> str:
    process = run_process(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"]
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"git cannot resolve {revision!r} in {repo}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def checkout_revision(repo: Path, revision: str, destination: Path) -> str:
    """Materialize a revision; return the checkout mechanism used."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    worktree = run_process(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "--detach",
            "--force",
            str(destination),
            revision,
        ]
    )
    if worktree.returncode == 0:
        return "worktree"
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", revision],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if archive.returncode != 0:
        raise RuntimeError(
            f"git cannot archive {revision!r}: {archive.stderr.decode(errors='replace').strip()}"
        )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if extract.returncode != 0:
        raise RuntimeError(
            "tar cannot extract the revision archive: "
            + extract.stderr.decode(errors="replace").strip()
        )
    return "archive"


def remove_checkout(repo: Path, destination: Path, mechanism: str) -> None:
    if mechanism == "worktree":
        run_process(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(destination)]
        )
    else:
        shutil.rmtree(destination, ignore_errors=True)


def rustc_identity(toolchain: str | None) -> str | None:
    command = ["rustc"]
    if toolchain:
        command.append(f"+{toolchain}")
    command.append("-vV")
    process = run_process(command)
    if process.returncode != 0:
        return None
    fields = {}
    for line in process.stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip()
    release = fields.get("release")
    commit = fields.get("commit-hash")
    if not release:
        return None
    return f"{release}/{commit}" if commit else release


def read_governor(core: int) -> str | None:
    direct = Path(
        f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor"
    )
    if direct.is_file():
        return direct.read_text(encoding="utf-8").strip()
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        related = policy / "related_cpus"
        affected = policy / "affected_cpus"
        governor = policy / "scaling_governor"
        if not governor.is_file():
            continue
        cpus = set()
        for source in (related, affected):
            if source.is_file():
                for token in source.read_text(encoding="utf-8").split():
                    try:
                        cpus.add(int(token))
                    except ValueError:
                        pass
        if core in cpus:
            return governor.read_text(encoding="utf-8").strip()
    return None


def smt_siblings(core: int) -> list[int]:
    path = Path(
        f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list"
    )
    if not path.is_file():
        return []
    siblings = set()
    for token in path.read_text(encoding="utf-8").strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            try:
                siblings.update(range(int(low), int(high) + 1))
            except ValueError:
                pass
        else:
            try:
                siblings.add(int(token))
            except ValueError:
                pass
    siblings.discard(core)
    return sorted(siblings)


def smt_active() -> bool:
    path = Path("/sys/devices/system/cpu/smt/active")
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip() == "1"
        except OSError:
            return False
    return False


def _cpu_times() -> dict[int, tuple[float, float]]:
    times = {}
    try:
        lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return times
    for line in lines:
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0][3:])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        times[index] = (total, idle)
    return times


def measure_smt_busy(siblings: list[int], window: float) -> float | None:
    if not siblings:
        return None
    before = _cpu_times()
    time.sleep(window)
    after = _cpu_times()
    busiest = 0.0
    for core in siblings:
        if core not in before or core not in after:
            continue
        total = after[core][0] - before[core][0]
        idle = after[core][1] - before[core][1]
        if total <= 0:
            continue
        busiest = max(busiest, max(0.0, min(1.0, (total - idle) / total)))
    return busiest


def read_cpu_list(path: str) -> list[int] | None:
    """Parse a sysfs CPU list such as `0-2,3,8-11`."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    cpus: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        low, _, high = token.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return sorted(cpus)


def read_environment(core: int | None, smt_window: float) -> dict:
    # `taskset` may pin to any online CPU, including `isolcpus` cores that the
    # runner's own default affinity excludes, so availability is judged
    # against the online set and the inherited affinity is only recorded.
    online = read_cpu_list("/sys/devices/system/cpu/online")
    inherited = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    siblings = smt_siblings(core) if core is not None else []
    busy = measure_smt_busy(siblings, smt_window) if siblings else None
    return {
        "pinned_core": core,
        "affinity": online if online is not None else inherited,
        "inherited_affinity": inherited,
        "isolated": read_cpu_list("/sys/devices/system/cpu/isolated"),
        "governor": read_governor(core) if core is not None else None,
        "smt_active": smt_active(),
        "siblings": siblings,
        "sibling_busy": busy,
        "smt_window_seconds": smt_window,
    }


def read_host_facts(toolchain: str | None) -> dict:
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = os.uname().machine
    uname = os.uname()
    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model or uname.machine,
        "arch": uname.machine,
        "os": f"{uname.sysname} {uname.release}",
        "kernel": uname.release,
        "toolchain": toolchain or "",
        "toolchain_id": rustc_identity(toolchain),
        "measured_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# --------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------


def _pop_flag(args, name: str):
    value = getattr(args, name)
    if isinstance(value, str) and value.strip().casefold() == PLACEHOLDER:
        return None
    return value


class Campaign:
    def __init__(self, args, session: dict, session_dir: Path):
        self.args = args
        self.session = session
        self.session_dir = session_dir
        self.rejections: list[Rejection] = []
        self.scratch = Path(tempfile.mkdtemp(prefix="fgf-pair-"))
        self.checkouts: list[tuple[Path, str]] = []
        self.repo = Path(args.repo).resolve()
        # One campaign lock per repository root, shared by every runner that
        # composes Campaign here regardless of its --records destination.
        self.lock_path = self.repo / "bench-records" / ".campaign.lock"
        self._lock = None
        self.current_run = 0
        self.current_stage = "main"
        self.args_rounds = 0
        self.machine = os.uname().machine
        self.chrt_parts = []
        self.execution_dir = session_dir / "exe"
        self.setarch_command = None

    # -- bookkeeping ------------------------------------------------------

    def reject(self, rejection: Rejection | None) -> bool:
        if rejection is None:
            return False
        if not any(
            existing.code == rejection.code
            and existing.message == rejection.message
            for existing in self.rejections
        ):
            self.rejections.append(rejection)
        return True

    def log(self, message: str) -> None:
        line = f"{utc_stamp()} {message}\n"
        with open(self.session_dir / "campaign.log", "a", encoding="utf-8") as handle:
            handle.write(line)

    def close(self) -> None:
        if not getattr(self.args, "keep_checkouts", False):
            for path, mechanism in self.checkouts:
                try:
                    remove_checkout(self.repo, path, mechanism)
                except Exception:
                    pass
            shutil.rmtree(self.scratch, ignore_errors=True)
        else:
            self.log(f"checkouts kept under {self.scratch}")

    # -- setup ------------------------------------------------------------

    def acquire_lock(self) -> bool:
        lock_path = self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            self.reject(
                reject(
                    "campaign_locked",
                    f"another paired campaign holds {lock_path}",
                )
            )
            return False
        self._lock = handle
        return True

    def release_lock(self) -> None:
        if self._lock is not None:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def prepare_snapshot(self, bench: str | None = None) -> Snapshot | None:
        origin = self.args.snapshot
        source_path = Path(origin)
        mechanism = None
        if source_path.exists():
            directory = source_path.resolve()
            resolved = str(directory)
        else:
            try:
                resolved = git_revision(self.repo, origin)
            except RuntimeError as error:
                self.reject(reject("snapshot_missing", str(error)))
                return None
            directory = self.scratch / "snapshot"
            try:
                mechanism = checkout_revision(self.repo, resolved, directory)
            except RuntimeError as error:
                self.reject(reject("snapshot_checkout_failed", str(error)))
                return None
            self.checkouts.append((directory, mechanism))
        bench_name = "operations"
        try:
            meta = read_snapshot_meta(directory)
            bench_name = str(meta.get("bench") or bench_name)
        except (OSError, ValueError) as error:
            self.reject(reject("snapshot_incomplete", str(error)))
            return None
        if bench is not None:
            bench_name = str(bench)
        problem = validate_snapshot(directory, bench_name)
        if problem is not None:
            self.reject(problem)
            return None
        features = merge_features(split_words(self.args.features), meta.get("features"))
        rustflags = merge_rustflags(self.args.rustflags, meta.get("rustflags"))
        toolchain = (
            meta.get("toolchain")
            or read_snapshot_toolchain(directory)
            or self.args.toolchain
            or None
        )
        snapshot = Snapshot(
            source=origin,
            resolved=resolved,
            directory=directory,
            digest=snapshot_digest(directory),
            meta=meta,
            features=features,
            rustflags=rustflags,
            toolchain=str(toolchain) if toolchain else None,
            bench=bench_name,
            no_default_features=bool(meta.get("no_default_features", False)),
        )
        return snapshot

    # -- scope ------------------------------------------------------------

    def apply_scope(self, order_cells: list[str], declaration: dict | None):
        """Finalize the session scope and materialize the timing filter file.

        The scope always keeps inventory order.  Without a declaration the
        scope is the whole inventory; otherwise the selected cells are
        validated against the inventory, recorded with their rationale and
        declaration digest, and written to ``scope-cells.txt`` in the session
        directory for the harness's ``--cells`` timing filter.  Returns
        (cells, filter_path); cells is None on a rejection.
        """
        cells = list(order_cells)
        if declaration is None:
            self.session["scope"] = {
                "kind": "full",
                "cells": cells,
                "digest": sha256_bytes(canonical_json(cells).encode("utf-8")),
                "rationale": None,
                "declaration_digest": None,
            }
            return cells, None
        known = set(order_cells)
        unknown = [
            cell for cell in declaration["cells"] if cell not in known
        ]
        if unknown:
            self.reject(
                reject(
                    "scope_unknown",
                    "scope declaration names cells absent from the measured "
                    "inventory: " + ", ".join(unknown),
                    cells=tuple(unknown),
                )
            )
            return None, None
        declared = set(declaration["cells"])
        cells = [cell for cell in order_cells if cell in declared]
        self.session["scope"] = {
            "kind": "selected",
            "cells": cells,
            "digest": sha256_bytes(canonical_json(cells).encode("utf-8")),
            "rationale": declaration["rationale"],
            "declaration_digest": declaration["digest"],
        }
        filter_path = self.session_dir / "scope-cells.txt"
        filter_path.write_text("\n".join(cells) + "\n", encoding="utf-8")
        return cells, filter_path

    # -- validation -------------------------------------------------------

    def validate_common(self, snapshot: Snapshot, tiers: list[str]) -> bool:
        ok = True
        for problem in (
            check_tiers(tiers),
            check_alignment_flags(snapshot.rustflags),
            check_tier_features(tiers, snapshot.features),
        ):
            if problem is not None:
                ok = not self.reject(problem)
        return ok

    def validate_measurement_config(self, snapshot: Snapshot, host: dict, tiers) -> bool:
        for problem in validate_smt_config(
            float(self.args.smt_window), float(self.args.smt_busy_threshold)
        ):
            self.reject(problem)
        if not getattr(self.args, "exploratory", False):
            if not self.args.policy_json:
                self.reject(
                    reject(
                        "config_missing",
                        "acceptance needs the --policy-json snapshot supplied by "
                        "the recipe so policy overrides can be detected",
                    )
                )
                return False
            problem = validate_policy_config(self.args)
            if problem is not None:
                self.reject(problem)
                return False
        runs = _pop_flag(self.args, "runs")
        drift_cap = _pop_flag(self.args, "drift_cap")
        rounds = _pop_flag(self.args, "rounds")
        run_count, problem = parse_host_value_binding(
            runs, host.get("hostname"), "runs"
        )
        if problem is not None:
            self.reject(problem)
            return False
        if run_count is not None:
            run_count = int(run_count)
        drift_value, problem = parse_host_value_binding(
            drift_cap, host.get("hostname"), "drift cap"
        )
        if problem is not None:
            self.reject(problem)
            return False
        rounds_value = None
        if rounds is not None:
            try:
                rounds_value = int(parse_number(rounds, "rounds"))
            except ValueError as error:
                self.reject(reject("uncalibrated_config", str(error)))
                return False
        try:
            delta = parse_number(self.args.delta, "delta")
            gamma = parse_number(self.args.gamma, "gamma")
        except ValueError as error:
            self.reject(reject("uncalibrated_config", str(error)))
            return False
        problems = validate_acceptance_config(
            runs=run_count,
            drift_cap=drift_value,
            delta=delta,
            gamma=gamma,
            rounds=rounds_value,
            calibrated_hosts=split_words(self.args.calibrated_hosts),
            calibrated_tiers=self.args.calibrated_tiers,
            calibrated_toolchain=self.args.calibrated_toolchain,
            calibrated_flags=self.args.calibrated_flags,
            host=host,
            toolchain_id=host.get("toolchain_id"),
            effective_flags=snapshot.rustflags,
            tiers=tiers,
        )
        for problem in problems:
            self.reject(problem)
        return not problems

    def calibration_runs(self) -> int:
        """Calibration collects its own budget; PAIR_RUNS never caps it."""
        return int(self.args.calibration_runs)

    # -- checkouts and builds --------------------------------------------

    def resolve_arms(self, snapshot: Snapshot) -> bool:
        if not self.args.base:
            self.reject(reject("usage", "--base is required for a measurement"))
            return False
        candidate = self.args.candidate or (
            self.args.base if self.args.mode == "calibration" else None
        )
        if candidate is None:
            self.reject(reject("usage", "--candidate is required for a measurement"))
            return False
        try:
            base_sha = git_revision(self.repo, self.args.base)
            candidate_sha = git_revision(self.repo, candidate)
        except RuntimeError as error:
            self.reject(reject("usage", str(error)))
            return False
        self.session["arms"]["base"] = {
            "kind": "library",
            "repo": str(self.repo),
            "sha": base_sha,
            "metadata_seed": None,
        }
        self.session["arms"]["candidate"] = {
            "kind": "library",
            "repo": str(self.repo),
            "sha": candidate_sha,
            "metadata_seed": None,
        }
        if self.args.mode == "calibration" and self.args.calibration_mode == "base_vs_layout_control":
            if base_sha != candidate_sha:
                self.reject(
                    reject(
                        "usage",
                        "base_vs_layout_control calibration rebuilds the same "
                        "revision in both arms",
                    )
                )
                return False
            seed = self.args.meta_seed or "fgf-pair-layout-control"
            self.session["arms"]["candidate"]["kind"] = "layout_control"
            self.session["arms"]["candidate"]["metadata_seed"] = seed
            self.session["arms"]["base"]["metadata_seed"] = None
        elif self.args.mode == "calibration":
            if base_sha != candidate_sha:
                self.reject(
                    reject(
                        "usage",
                        "base_vs_base calibration needs the same revision in both arms",
                    )
                )
                return False
        return True

    def make_arm_checkout(self, arm: str, arm_record: dict, snapshot: Snapshot) -> bool:
        destination = self.scratch / f"arm-{arm}"
        try:
            mechanism = checkout_revision(self.repo, arm_record["sha"], destination)
        except RuntimeError as error:
            self.reject(reject("checkout_failed", str(error), arm=arm))
            return False
        self.checkouts.append((destination, mechanism))
        # Read the arm revision's own manifest and lock before the snapshot
        # overlay replaces them.  The comparison is only meaningful against
        # the revision itself, never against the overlaid copy.
        metadata = self.cargo_metadata(destination, f"{arm}-revision")
        if metadata is None:
            return False
        try:
            arm_record["revision_dependency_set"] = sorted(
                normal_dependency_set(metadata)
            )
        except ValueError as error:
            self.reject(
                reject("dependency_metadata_failed", str(error), arm=arm)
            )
            return False
        try:
            overlay_snapshot(destination, snapshot)
        except OSError as error:
            self.reject(reject("overlay_failed", str(error), arm=arm))
            return False
        arm_record["checkout"] = str(destination)
        return True

    def build_bench(self, arm: str, arm_record: dict, snapshot: Snapshot) -> Path | None:
        target_dir = self.scratch / f"target-{arm}"
        env = os.environ.copy()
        env.pop("CARGO_ENCODED_RUSTFLAGS", None)
        env["CARGO_TARGET_DIR"] = str(target_dir)
        flags = snapshot.rustflags
        seed = arm_record.get("metadata_seed")
        if seed:
            flags = f"{flags} -C metadata={seed}".strip()
            env["__CARGO_DEFAULT_LIB_METADATA"] = str(seed)
        else:
            env.pop("__CARGO_DEFAULT_LIB_METADATA", None)
        env["RUSTFLAGS"] = flags
        command = ["cargo"]
        if snapshot.toolchain:
            command.append(f"+{snapshot.toolchain}")
        command += [
            "build",
            "--profile",
            "bench",
            "--locked",
            "--message-format=json-render-diagnostics",
            "--manifest-path",
            str(Path(arm_record["checkout"]) / "Cargo.toml"),
            "--bench",
            snapshot.bench,
        ]
        if snapshot.no_default_features:
            command.append("--no-default-features")
        if snapshot.features:
            if "all" in snapshot.features:
                command.append("--all-features")
            else:
                command.extend(["--features", ",".join(snapshot.features)])
        process = run_process(
            command,
            env=env,
            timeout=self.args.timeout,
            log_path=self.session_dir / f"build-{arm}.log",
        )
        if process.returncode != 0:
            self.reject(
                reject(
                    "build_failed",
                    f"cargo build failed for the {arm} arm; see build-{arm}.log",
                    arm=arm,
                )
            )
            return None
        executables = []
        for line in process.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("reason") != "compiler-artifact":
                continue
            target = record.get("target") or {}
            if target.get("name") != snapshot.bench:
                continue
            if "bench" not in (target.get("kind") or []):
                continue
            if record.get("executable"):
                executables.append(Path(record["executable"]))
        if len(executables) != 1:
            self.reject(
                reject(
                    "build_failed",
                    f"cargo build did not yield exactly one {snapshot.bench} bench "
                    f"executable for the {arm} arm (found {len(executables)})",
                    arm=arm,
                )
            )
            return None
        arm_record["executable"] = str(executables[0])
        arm_record["target_dir"] = str(target_dir)
        return executables[0]

    def build_route_diagnostic(self, arm: str, arm_record: dict, snapshot: Snapshot):
        target_dir = arm_record.get("target_dir")
        if target_dir is None:
            return None
        env = os.environ.copy()
        env.pop("CARGO_ENCODED_RUSTFLAGS", None)
        env["CARGO_TARGET_DIR"] = target_dir
        flags = snapshot.rustflags
        seed = arm_record.get("metadata_seed")
        if seed:
            flags = f"{flags} -C metadata={seed}".strip()
            env["__CARGO_DEFAULT_LIB_METADATA"] = str(seed)
        else:
            env.pop("__CARGO_DEFAULT_LIB_METADATA", None)
        env["RUSTFLAGS"] = flags
        command = ["cargo"]
        if snapshot.toolchain:
            command.append(f"+{snapshot.toolchain}")
        command += [
            "build",
            "--profile",
            "bench",
            "--locked",
            "--message-format=json-render-diagnostics",
            "--manifest-path",
            str(Path(arm_record["checkout"]) / "Cargo.toml"),
            "--example",
            "backend_report",
            "--all-features",
        ]
        process = run_process(
            command,
            env=env,
            timeout=self.args.timeout,
            log_path=self.session_dir / f"diagnostic-build-{arm}.log",
        )
        if process.returncode != 0:
            return {
                "status": "build_failed",
                "stderr": process.stderr[-2000:],
            }
        executables = []
        for line in process.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("reason") != "compiler-artifact":
                continue
            target = record.get("target") or {}
            if target.get("name") == "backend_report" and record.get("executable"):
                executables.append(Path(record["executable"]))
        if not executables:
            return {"status": "build_failed", "stderr": "no backend_report executable"}
        return {"status": "built", "executable": str(executables[-1])}

    # -- harness execution ------------------------------------------------

    def harness_env(self, tier: str, seed=None) -> dict:
        env = os.environ.copy()
        env["SIMD_BACKEND"] = tier
        env["FEC_GOLDEN_CORE"] = str(self.args.core)
        if seed:
            env["__CARGO_DEFAULT_LIB_METADATA"] = str(seed)
        return env

    def prepare_execution_alias(self, key: str, executable: Path) -> Path:
        names = {
            "base": "arm_a",
            "candidate": "arm_b",
            "layout_control": "arm_l",
        }
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        alias = self.execution_dir / names[key]
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        try:
            alias.symlink_to(Path(executable).resolve())
        except OSError as error:
            self.reject(reject("execution_alias_failed", str(error), arm=key))
            raise
        return alias

    def run_harness(
        self,
        executable: Path,
        extra,
        tier: str,
        label: str,
        *,
        env: dict | None = None,
        disable_aslr: bool = False,
    ):
        command = harness_command(
            executable,
            extra,
            core=self.args.core,
            machine=self.machine,
            chrt_parts=self.chrt_parts,
            disable_aslr=disable_aslr,
        )
        return run_process(
            command,
            env=env if env is not None else self.harness_env(tier),
            timeout=self.args.timeout,
            log_path=self.session_dir / f"{label}.log",
        )

    # -- tiers ------------------------------------------------------------

    def inspect_tier(
        self, tier: str, arm: str, arm_record: dict, executable: Path
    ) -> dict | None:
        process = self.run_harness(
            executable, ["--inventory"], tier, f"inventory-{arm}-{tier}"
        )
        if process.returncode != 0:
            self.reject(
                reject(
                    "inventory_failed",
                    f"{arm} inventory run failed at tier {tier}; "
                    f"see inventory-{arm}-{tier}.log",
                    arm=arm,
                )
            )
            return None
        parsed, errors = parse_protocol(process.stdout)
        if errors:
            self.reject(
                reject(
                    "protocol_error",
                    f"{arm} inventory at tier {tier}: " + "; ".join(errors),
                    arm=arm,
                )
            )
            return None
        problem = check_backend(tier, parsed["backend"], arm)
        if problem is not None:
            self.reject(problem)
            return None
        if not parsed["inventory"]:
            self.reject(
                reject(
                    "protocol_error",
                    f"{arm} inventory at tier {tier} is empty",
                    arm=arm,
                )
            )
            return None
        return parsed

    def run_checks(
        self,
        tier: str,
        arm: str,
        executable: Path,
        order: list[str],
        env: dict | None = None,
    ):
        process = self.run_harness(
            executable,
            ["--check"],
            tier,
            f"check-{arm}-{tier}",
            env=env,
            disable_aslr=True,
        )
        if process.returncode != 0:
            self.reject(
                reject(
                    "check_failed",
                    f"{arm} check run failed at tier {tier} (oracle mismatch or "
                    f"unknown cell); see check-{arm}-{tier}.log",
                    arm=arm,
                )
            )
            return None
        parsed, errors = parse_protocol(process.stdout)
        if errors:
            self.reject(
                reject(
                    "protocol_error",
                    f"{arm} check at tier {tier}: " + "; ".join(errors),
                    arm=arm,
                )
            )
            return None
        missing = [cell for cell in order if cell not in parsed["outputs"]]
        extra = [cell for cell in parsed["outputs"] if cell not in order]
        if missing or extra:
            self.reject(
                reject(
                    "output_mismatch",
                    f"{arm} check output set differs from the inventory at tier {tier}",
                    cells=tuple(missing + extra),
                    arm=arm,
                )
            )
            return None
        return parsed["outputs"]

    def run_route_diagnostic(self, tier: str, arm: str, arm_record: dict):
        build = arm_record.get("diagnostic")
        record = {"status": "unavailable"}
        if build is None:
            record["stderr"] = "diagnostic build not attempted"
        elif build.get("status") != "built":
            record.update(build)
        else:
            process = self.run_harness(
                Path(build["executable"]),
                [],
                tier,
                f"diagnostic-{arm}-{tier}",
            )
            record = {
                "detail": "per-field",
                "status": "ok" if process.returncode == 0 else "failed",
                "exit": process.returncode,
                "fields": parse_route_diagnostic(process.stdout),
                "stdout": process.stdout[-2000:],
                "stderr": process.stderr[-2000:],
            }
        self.session["route_diagnostics"].setdefault(tier, {})[arm] = record

    def check_preconditions(
        self, tier: str, run_index: int, phase: str = "measurement"
    ) -> bool:
        environment = read_environment(self.args.core, self.args.smt_window)
        problems = evaluate_environment(
            environment, self.args.smt_busy_threshold
        )
        self.session["host"].update(
            {
                "governor": environment["governor"],
                "smt_active": environment["smt_active"],
                "thread_siblings": environment["siblings"],
                "pinned_core": environment["pinned_core"],
                "affinity": environment["affinity"],
                "sibling_busy": environment["sibling_busy"],
                "campaign_lock": str(self.lock_path),
                "taskset": (
                    f"taskset -c {environment['pinned_core']}"
                    if environment["pinned_core"] is not None
                    else None
                ),
            }
        )
        self.session["preconditions"].append(
            {
                "phase": phase,
                "tier": tier,
                "run": run_index,
                "governor": environment["governor"],
                "smt_active": environment["smt_active"],
                "thread_siblings": environment["siblings"],
                "sibling_busy": environment["sibling_busy"],
                "smt_window_seconds": environment["smt_window_seconds"],
                "smt_busy_threshold": self.args.smt_busy_threshold,
                "aslr_disabled": True,
                "setarch": self.setarch_command,
                "chrt": self.chrt_parts or None,
                "pinned_core": environment["pinned_core"],
                "affinity": environment["affinity"],
                "ok": not problems,
            }
        )
        for problem in problems:
            self.reject(problem)
        return not problems

    def timed_run(
        self,
        tier: str,
        arm: str,
        executable: Path,
        order: list[str],
        cells_filter: Path | None = None,
        env: dict | None = None,
    ):
        extra = ["--time", "--rounds", str(self.args_rounds)]
        if cells_filter is not None:
            extra.extend(["--cells", str(cells_filter)])
        process = self.run_harness(
            executable,
            extra,
            tier,
            f"run-{self.current_run}-{arm}-{tier}-{self.current_stage}",
            env=env,
            disable_aslr=True,
        )
        if process.returncode != 0:
            self.reject(
                reject(
                    "run_failed",
                    f"{arm} timing run failed at tier {tier}; "
                    f"see run-{self.current_run}-{arm}-{tier}-{self.current_stage}.log",
                    arm=arm,
                )
            )
            return None
        parsed, errors = parse_protocol(process.stdout)
        if errors:
            self.reject(
                reject(
                    "protocol_error",
                    f"{arm} timing at tier {tier}: " + "; ".join(errors),
                    arm=arm,
                )
            )
            return None
        start = control_value(parsed, "start")
        end = control_value(parsed, "end")
        if start is None or end is None:
            self.reject(
                reject(
                    "protocol_error",
                    f"{arm} timing at tier {tier} did not bracket the run with "
                    "one start and one end control",
                    arm=arm,
                )
            )
            return None
        cells, problem = summarize_timed_protocol(
            parsed, order, self.args_rounds, arm
        )
        if problem is not None:
            self.reject(problem)
            return None
        return {
            "run": self.current_run,
            "tier": tier,
            "order": order,
            "control": {"start_ns": start, "end_ns": end},
            "argv_length": command_length(
                harness_command(
                    executable,
                    extra,
                    core=self.args.core,
                    machine=self.machine,
                    chrt_parts=self.chrt_parts,
                    disable_aslr=True,
                )
            ),
            "arms": {
                arm: {
                    "cells": cells,
                }
            },
        }

    # -- rules ------------------------------------------------------------

    def tier_runs(self, tier: str, stage: str = "main") -> list[dict]:
        return [
            run for run in self.session["runs"]
            if run["tier"] == tier and run.get("stage", "main") == stage
        ]

    def build_ratios(
        self, tier: str, order: list[str], stage: str = "main"
    ) -> dict | None:
        runs = self.tier_runs(tier, stage)
        if not runs:
            return None
        base = [
            run["arms"]["base"]["cells"] for run in runs
            if "base" in run["arms"]
        ]
        candidate = [
            run["arms"]["candidate"]["cells"] for run in runs
            if "candidate" in run["arms"]
        ]
        base_summaries = [
            {cell: entry["summary_ns"] for cell, entry in cells.items()}
            for cells in base
        ]
        candidate_summaries = [
            {cell: entry["summary_ns"] for cell, entry in cells.items()}
            for cells in candidate
        ]
        if len(base_summaries) != len(runs) or len(candidate_summaries) != len(runs):
            self.reject(
                reject(
                    "run_incomplete",
                    f"tier {tier} does not have both arms in every run",
                )
            )
            return None
        try:
            return paired_log_ratios(base_summaries, candidate_summaries, order)
        except (KeyError, ValueError) as error:
            self.reject(
                reject(
                    "run_incomplete",
                    f"tier {tier} cannot be paired across arms: {error}",
                )
            )
            return None

    def summarize(self, tiers) -> None:
        for tier in tiers:
            for arm in ("base", "candidate"):
                cells = {}
                runs = self.tier_runs(tier)
                if not runs:
                    continue
                cell_ids = list(runs[0]["arms"][arm]["cells"])
                for cell in cell_ids:
                    values = [
                        run["arms"][arm]["cells"][cell]["summary_ns"]
                        for run in runs
                        if arm in run["arms"] and cell in run["arms"][arm]["cells"]
                    ]
                    if values:
                        cells[cell] = statistics.median(values)
                self.session["arm_summaries"][arm][tier] = cells

    def measure_stage(self, tier, aliases, cells, cells_filter, stage) -> bool:
        """Run fresh paired processes with the campaign's isolation checks."""
        self.current_stage = stage
        for run_index in range(self.session["config"]["runs"]):
            self.current_run = run_index
            if not self.check_preconditions(tier, run_index, stage):
                return False
            precondition = self.session["preconditions"][-1]
            run_env = self.harness_env(tier)
            arm_order = balanced_order(run_index)
            pieces = {}
            for arm in arm_order:
                piece = self.timed_run(
                    tier, arm, aliases[arm], cells,
                    env=run_env, cells_filter=cells_filter,
                )
                if piece is None:
                    return False
                pieces[arm] = piece
            lengths = {arm: pieces[arm]["argv_length"] for arm in arm_order}
            problem = check_argv_lengths(
                lengths, f"tier {tier} {stage} run {run_index}"
            )
            if problem is not None:
                self.reject(problem)
                return False
            self.session["runs"].append({
                "run": run_index,
                "stage": stage,
                "tier": tier,
                "order": arm_order,
                "control": pieces[arm_order[0]]["control"],
                "controls": {arm: pieces[arm]["control"] for arm in arm_order},
                "precondition": precondition,
                "environment_digest": environment_digest(run_env),
                "argv_lengths": lengths,
                "arms": {arm: pieces[arm]["arms"][arm] for arm in arm_order},
            })
        return True

    # -- main flow --------------------------------------------------------

    def execute(self) -> int:
        args = self.args
        if not self.acquire_lock():
            return 1
        session = self.session
        mode = args.mode
        if mode == "calibration" and getattr(args, "cells", None):
            self.reject(
                reject(
                    "scope_calibration_unsupported",
                    "--cells times a selected scope; calibration always "
                    "measures the whole inventory",
                )
            )
            return 1

        host = read_host_facts(args.toolchain)
        session["host"].update(
            {
                "hostname": host["hostname"],
                "cpu_model": host["cpu_model"],
                "arch": host["arch"],
                "os": host["os"],
                "kernel": host["kernel"],
                "toolchain": host["toolchain"],
                "toolchain_id": host["toolchain_id"],
                "measured_utc": host["measured_utc"],
            }
        )
        self.session["session_id"] = self.session_dir.name

        try:
            snapshot = self.prepare_snapshot()
            if snapshot is None:
                return 1
            self.snapshot = snapshot
            session["snapshot"] = snapshot.as_dict()
            snapshot_runner = snapshot.directory / "benches" / "bench_pair.py"
            self_runner = Path(__file__).resolve()
            session["snapshot"]["runner_digest"] = (
                sha256_file(snapshot_runner)
                if snapshot_runner.is_file()
                else None
            )
            session["snapshot"]["runner_self_digest"] = sha256_file(self_runner)
            session["snapshot"]["runner_matches_snapshot"] = (
                session["snapshot"]["runner_digest"]
                == session["snapshot"]["runner_self_digest"]
            )
            problem = check_snapshot_runner(
                session["snapshot"]["runner_digest"],
                session["snapshot"]["runner_self_digest"],
            )
            if problem is not None:
                self.reject(problem)
                return 1
            session["config"]["features"] = snapshot.features
            session["config"]["rustflags"] = snapshot.rustflags
            session["config"]["toolchain"] = snapshot.toolchain
            session["host"]["toolchain"] = snapshot.toolchain or ""
            session["host"]["toolchain_id"] = (
                host["toolchain_id"]
                if (snapshot.toolchain or "") == (args.toolchain or "")
                else rustc_identity(snapshot.toolchain)
            )

            tiers = list(dict.fromkeys(split_words(args.tiers)))
            if not self.validate_common(snapshot, tiers):
                return 1
            for problem in validate_smt_config(
                float(args.smt_window), float(args.smt_busy_threshold)
            ):
                self.reject(problem)
            for problem in validate_execution_environment(
                self.machine,
                shutil.which("setarch"),
                args.chrt,
                shutil.which("chrt"),
            ):
                self.reject(problem)
            if self.rejections:
                return 1
            self.chrt_parts = shlex.split(args.chrt) if args.chrt else []
            self.setarch_command = f"setarch {self.machine} -R"
            session["execution"] = {
                "aslr_disabled": True,
                "setarch": self.setarch_command,
                "setarch_path": shutil.which("setarch"),
                "taskset": (
                    f"taskset -c {args.core}"
                    if args.core is not None
                    else None
                ),
                "chrt": self.chrt_parts or None,
                "environment_isolation": (
                    "one identical environment block and argv length for each "
                    "run's two arms"
                ),
            }
            session["host"]["setarch"] = self.setarch_command
            session["host"]["chrt"] = self.chrt_parts or None

            if mode != "calibration":
                if not self.validate_measurement_config(snapshot, session["host"], tiers):
                    return 1
                try:
                    rounds = int(parse_number(args.rounds, "rounds"))
                except ValueError:
                    return 1
                self.args_rounds = rounds
                session["config"]["rounds"] = rounds
                run_value, problem = parse_host_value_binding(
                    args.runs, session["host"].get("hostname"), "runs"
                )
                if problem is not None or run_value is None:
                    self.reject(
                        problem
                        or reject(
                            "uncalibrated_config",
                            "acceptance needs a calibrated --runs value",
                        )
                    )
                    return 1
                drift_value, problem = parse_host_value_binding(
                    args.drift_cap, session["host"].get("hostname"), "drift cap"
                )
                if problem is not None or drift_value is None:
                    self.reject(
                        problem
                        or reject(
                            "uncalibrated_config",
                            "acceptance needs a calibrated --drift-cap value",
                        )
                    )
                    return 1
                session["config"]["runs"] = int(run_value)
                session["config"]["drift_cap"] = drift_value
                try:
                    session["config"]["delta"] = parse_number(args.delta, "delta")
                    session["config"]["gamma"] = parse_number(args.gamma, "gamma")
                except ValueError:
                    return 1
                prior, session_problem = validate_prior_session_files(
                    args.prior_session
                )
                if session_problem is not None:
                    self.reject(session_problem)
                    return 1
                if prior:
                    problem = validate_acceptance_inputs(prior)
                    if problem is not None:
                        self.reject(problem)
                        return 1
            else:
                self.args_rounds = int(parse_number(args.rounds, "rounds"))
                session["config"]["rounds"] = self.args_rounds
                session["config"]["runs"] = self.calibration_runs()
                session["config"]["delta"] = float(parse_number(args.delta, "delta"))
                session["config"]["gamma"] = float(parse_number(args.gamma, "gamma"))
                session["config"]["drift_cap"] = None

            if session["config"]["runs"] < MIN_RUNS_FOR_BOUND:
                self.reject(
                    reject(
                        "insufficient_runs",
                        f"a session collects at least {MIN_RUNS_FOR_BOUND} "
                        "paired runs before any HL bound is defined",
                    )
                )
                return 1
            session["config"]["runs_binding"] = args.runs
            session["config"]["drift_cap_binding"] = args.drift_cap
            session["config"]["drift_method"] = DRIFT_METHOD
            session["drift_method"] = DRIFT_METHOD
            session["config"]["calibrated"] = (
                mode != "calibration" and not getattr(args, "exploratory", False)
            )
            session["config"]["smt_window"] = float(args.smt_window)
            session["config"]["smt_busy_threshold"] = float(
                args.smt_busy_threshold
            )
            session["config"]["chrt"] = args.chrt
            session["config"]["drift_safety_factor"] = float(
                args.drift_safety_factor
            )
            session["config"]["policy_json"] = args.policy_json
            session["config"]["calibration_binding"] = {
                "hosts": split_words(args.calibrated_hosts),
                "tiers": str(args.calibrated_tiers),
                "tiers_by_host": {
                    host: sorted(tiers)
                    for host, tiers in (
                        parse_host_tier_binding(str(args.calibrated_tiers)) or {}
                    ).items()
                },
                "toolchain": args.calibrated_toolchain,
                "flags": args.calibrated_flags,
            }

            targets = None
            if getattr(args, "targets", None):
                try:
                    targets = parse_targets_file(Path(args.targets))
                except (OSError, ValueError) as error:
                    self.reject(reject("declaration_invalid", str(error)))
                    return 1
                session["config"]["targets_declaration"] = {
                    "path": targets["path"],
                    "digest": targets["digest"],
                    "cells": targets["cells"],
                }
                session["config"]["targets"] = targets["cells"]

            scope_declaration = None
            if getattr(args, "cells", None):
                try:
                    scope_declaration = parse_scope_file(
                        Path(args.cells), getattr(args, "scope_rationale", None)
                    )
                except (OSError, ValueError) as error:
                    self.reject(reject("scope_invalid", str(error)))
                    return 1

            threshold_info = None
            if mode == "threshold":
                if not args.threshold or not args.inventory:
                    self.reject(
                        reject(
                            "declaration_invalid",
                            "threshold mode needs --threshold and --inventory",
                        )
                    )
                    return 1
                try:
                    parsed_decl = parse_threshold_file(Path(args.threshold))
                except (OSError, ValueError) as error:
                    self.reject(reject("declaration_invalid", str(error)))
                    return 1
                try:
                    items = load_inventory_items(Path(args.inventory))
                except (OSError, ValueError) as error:
                    self.reject(reject("declaration_invalid", str(error)))
                    return 1
                info, problem = validate_threshold_declaration(
                    parsed_decl["declaration"], items
                )
                if problem is not None:
                    self.reject(problem)
                    return 1
                threshold_info = info
                threshold_info["path"] = parsed_decl["path"]
                threshold_info["digest"] = parsed_decl["digest"]
                session["config"]["threshold_declaration"] = {
                    "path": parsed_decl["path"],
                    "digest": parsed_decl["digest"],
                    "item_key": info["item_key"],
                    "old_value": info["old"],
                    "new_value": info["new"],
                    "unit": info["unit"],
                    "axis": info["axis"],
                    "declared_cells": [
                        entry["cell"] for entry in info["cells"]
                    ],
                }
                targets = {
                    "path": parsed_decl["path"],
                    "digest": parsed_decl["digest"],
                    "cells": info["target_set"],
                }
                session["threshold"] = {
                    "inventory_path": str(Path(args.inventory)),
                    "item_key": info["item_key"],
                    "old_value": info["old"],
                    "new_value": info["new"],
                    "old": info["old"],
                    "new": info["new"],
                    "unit": info["unit"],
                    "axis": info["axis"],
                    "declared_cells": [
                        entry["cell"] for entry in info["cells"]
                    ],
                    "target_set": info["target_set"],
                    "span_ok": True,
                    "path": parsed_decl["path"],
                    "digest": parsed_decl["digest"],
                    "grid": info["grid"],
                    "cells": info["cells"],
                }

            session["host"]["campaign_lock"] = str(self.lock_path)
            if not self.check_preconditions("preflight", -1, "preflight"):
                return 1

            if not self.resolve_arms(snapshot):
                return 1
            session["reused_calibration"] = False

            for arm in ("base", "candidate"):
                if not self.make_arm_checkout(arm, session["arms"][arm], snapshot):
                    return 1
            if not self.check_dependencies(snapshot):
                return 1

            executables = {}
            aliases = {}
            for arm in ("base", "candidate"):
                executable = self.build_bench(arm, session["arms"][arm], snapshot)
                if executable is None:
                    return 1
                executables[arm] = executable
                try:
                    aliases[arm] = self.prepare_execution_alias(arm, executable)
                except OSError:
                    return 1
                session["arms"][arm]["execution_path"] = str(aliases[arm])
                session["arms"][arm]["diagnostic"] = self.build_route_diagnostic(
                    arm, session["arms"][arm], snapshot
                )

            inventory_by_arm = {}
            unstable_by_tier = {}
            uncalibrated_by_tier = {}
            unstable_entries = None
            if mode != "calibration":
                unstable_entries, unstable_problem = load_unstable_entries(
                    Path(args.unstable_file)
                )
                if unstable_problem is not None:
                    self.reject(unstable_problem)
                    return 1
            scope_ids = (
                set(scope_declaration["cells"]) if scope_declaration else None
            )
            for tier in tiers:
                parsed = {}
                for arm in ("base", "candidate"):
                    parsed[arm] = self.inspect_tier(
                        tier, arm, session["arms"][arm], aliases[arm]
                    )
                    if parsed[arm] is None:
                        return 1
                session["tiers"].append(
                    {
                        "requested": tier,
                        "reported": parsed["base"]["backend"],
                        "field_backends": parsed["base"]["field_backends"],
                        "candidate_reported": parsed["candidate"]["backend"],
                        "candidate_field_backends": parsed["candidate"][
                            "field_backends"
                        ],
                        "field_backends_equal": parsed["base"]["field_backends"]
                        == parsed["candidate"]["field_backends"],
                        "ok": True,
                    }
                )
                base_inventory = parsed["base"]["inventory"]
                candidate_inventory = parsed["candidate"]["inventory"]
                problem = compare_inventories(base_inventory, candidate_inventory)
                if problem is not None:
                    self.reject(problem)
                    return 1
                inventory_by_arm.setdefault(
                    "base", base_inventory
                )
                inventory_by_arm.setdefault(
                    "candidate", candidate_inventory
                )
                order = parsed["base"]["inventory_order"]
                digest = inventory_digest(base_inventory, order)
                session["inventory"] = {
                    "base": list(base_inventory.values()),
                    "candidate": list(candidate_inventory.values()),
                    "order": order,
                    "equal": True,
                    "digest": digest,
                }
                if mode != "calibration":
                    cells, uncalibrated, unstable_problem = (
                        unstable_cells_for_inventory(
                            unstable_entries or [],
                            session["host"],
                            tier,
                            session["host"].get("toolchain_id"),
                            session["snapshot"].get("rustflags") or "",
                            base_inventory,
                            scope_ids,
                            Path(args.calibrated_inventories),
                        )
                    )
                    if unstable_problem is not None:
                        self.reject(unstable_problem)
                        return 1
                    unstable_by_tier[tier] = set(cells)
                    uncalibrated_by_tier[tier] = sorted(uncalibrated)
                check_env = self.harness_env(tier)
                base_outputs = self.run_checks(
                    tier, "base", aliases["base"], order, env=check_env
                )
                candidate_outputs = self.run_checks(
                    tier, "candidate", aliases["candidate"], order, env=check_env
                )
                if base_outputs is None or candidate_outputs is None:
                    return 1
                problem = compare_outputs(base_outputs, candidate_outputs, order)
                check_argv = {
                    "base": command_length(harness_command(
                        aliases["base"], ["--check"],
                        core=args.core, machine=self.machine,
                        chrt_parts=self.chrt_parts, disable_aslr=True,
                    )),
                    "candidate": command_length(harness_command(
                        aliases["candidate"], ["--check"],
                        core=args.core, machine=self.machine,
                        chrt_parts=self.chrt_parts, disable_aslr=True,
                    )),
                }
                problem = check_argv_lengths(check_argv, f"check tier {tier}")
                if problem is not None:
                    self.reject(problem)
                    return 1
                session["checks"][tier] = {
                    "base": base_outputs,
                    "candidate": candidate_outputs,
                    "identical": problem is None,
                    "environment_digest": environment_digest(check_env),
                    "argv_lengths": check_argv,
                }
                if problem is not None:
                    self.reject(problem)
                    return 1
                for arm in ("base", "candidate"):
                    self.run_route_diagnostic(tier, arm, session["arms"][arm])

            session["route_detail"] = "per-field"
            session["route_limitation"] = (
                "the stable-API bench and backend_report expose process and "
                "field backends only; per-cell route selectors are not "
                "available for every cell geometry without production changes"
            )
            order_cells = session["inventory"]["order"]
            scope_cells, scope_file = self.apply_scope(order_cells, scope_declaration)
            if scope_cells is None:
                return 1
            scope_set = set(scope_cells)
            if targets is not None:
                problem = validate_targets(targets["cells"], set(order_cells))
                if problem is not None:
                    self.reject(problem)
                    return 1
                outside = [
                    cell for cell in targets["cells"] if cell not in scope_set
                ]
                if outside:
                    self.reject(
                        reject(
                            "scope_excludes_targets",
                            "declared target cells are outside the measured "
                            "scope: " + ", ".join(outside),
                            cells=tuple(outside),
                        )
                    )
                    return 1
            if threshold_info is not None:
                declared_cells = [
                    entry["cell"] for entry in threshold_info["cells"]
                ]
                problem = validate_declared_cells(declared_cells, set(order_cells))
                if problem is not None:
                    self.reject(problem)
                    return 1
                outside = [
                    cell for cell in declared_cells if cell not in scope_set
                ]
                if outside:
                    self.reject(
                        reject(
                            "scope_excludes_threshold",
                            "declared threshold cells are outside the measured "
                            "scope: " + ", ".join(outside),
                            cells=tuple(outside),
                        )
                    )
                    return 1
            declared_targets = targets["cells"] if targets is not None else None
            unstable_targets_by_tier = {}
            if declared_targets is not None:
                try:
                    required_runs = int(
                        parse_number(args.unstable_target_runs, "unstable target runs")
                    )
                except ValueError:
                    required_runs = None
                for tier, cells in unstable_by_tier.items():
                    overlap = set(declared_targets) & cells
                    if not overlap:
                        unstable_targets_by_tier[tier] = set()
                        continue
                    if (
                        required_runs is None
                        or required_runs < MIN_RUNS_FOR_BOUND
                        or session["config"]["runs"] < required_runs
                    ):
                        self.reject(
                            reject(
                                "unstable_target_underpowered",
                                f"tier {tier} has unstable superiority targets "
                                f"but needs a calibrated --unstable-target-runs "
                                f"of at least {required_runs or 'uncalibrated'}; "
                                f"the session runs {session['config']['runs']}",
                                cells=tuple(sorted(overlap)),
                            )
                        )
                        return 1
                    unstable_targets_by_tier[tier] = overlap
            session["unstable"] = {
                "path": str(args.unstable_file),
                "entries": {
                    tier: sorted(cells)
                    for tier, cells in sorted(unstable_by_tier.items())
                },
            }
            session["uncalibrated"] = {
                "entries": {
                    tier: sorted(cells)
                    for tier, cells in sorted(uncalibrated_by_tier.items())
                },
            }
            session["config"]["unstable_file"] = str(args.unstable_file)
            session["config"]["unstable_target_runs"] = args.unstable_target_runs

            drift_by_tier = {}
            decisions = {}
            session["drift"] = drift_by_tier
            for tier in tiers:
                if not self.measure_stage(
                    tier, aliases, scope_cells, scope_file, "main"
                ):
                    return 1
                stats, problem = check_drift(
                    start_controls(self.tier_runs(tier)),
                    session["config"]["drift_cap"]
                    if mode != "calibration" else float("inf"),
                )
                drift_by_tier[tier] = stats
                if problem is not None and mode != "calibration":
                    self.reject(problem)
                    return 1
                if mode == "calibration":
                    continue
                ratios = self.build_ratios(tier, scope_cells)
                if ratios is None:
                    return 1
                excluded = unstable_by_tier.get(tier, set())
                main = evaluate_noninferiority(
                    ratios, session["config"]["delta"], excluded=excluded
                )
                cells = confirmation_cells(main, declared_targets)
                confirmation_ratios = {}
                if cells:
                    filter_path = self.session_dir / f"confirmation-cells-{tier}.txt"
                    filter_path.write_text("\n".join(cells) + "\n", encoding="utf-8")
                    if not self.measure_stage(
                        tier, aliases, cells, filter_path, "confirmation"
                    ):
                        return 1
                    stats, problem = check_drift(
                        start_controls(self.tier_runs(tier, "confirmation")),
                        session["config"]["drift_cap"],
                    )
                    session["confirmation_drift"][tier] = stats
                    if self.reject(problem):
                        return 1
                    confirmation_ratios = self.build_ratios(
                        tier, cells, "confirmation"
                    )
                    if confirmation_ratios is None:
                        return 1
                decisions[tier] = evaluate_staged_decisions(
                    ratios, confirmation_ratios,
                    session["config"]["delta"], session["config"]["gamma"],
                    targets=declared_targets, excluded=excluded,
                    unstable_targets=unstable_targets_by_tier.get(tier, set()),
                )
                session["confirmed_failures"][tier] = decisions[tier]["confirmed_failures"]
                session["decisions"] = publishable_decisions(decisions)

            if mode == "calibration":
                self.summarize(tiers)
                return self.finish_calibration(tiers, drift_by_tier)
            session["decisions"] = publishable_decisions(decisions)
            session["arm_summaries"] = {
                arm: {tier: {} for tier in tiers}
                for arm in ("base", "candidate")
            }
            self.summarize(tiers)
            session["rows"] = build_rows(
                session, tiers, scope_cells, unstable_by_tier
            )

            if args.layout_only:
                if not self.layout_only(tiers, decisions):
                    return 1

            return 0 if decisions_pass(decisions, declared_targets) else 1
        finally:
            self.release_lock()

    def check_dependencies(self, snapshot: Snapshot) -> bool:
        try:
            snapshot_meta = self.cargo_metadata(snapshot.directory, "snapshot")
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            self.reject(reject("dependency_metadata_failed", str(error)))
            return False
        if snapshot_meta is None:
            return False
        try:
            expected = normal_dependency_set(snapshot_meta)
        except ValueError as error:
            self.reject(reject("dependency_metadata_failed", str(error)))
            return False
        for arm, arm_record in self.session["arms"].items():
            actual = set(arm_record.get("revision_dependency_set") or [])
            problem = compare_dependency_sets(expected, actual, arm)
            if problem is not None:
                self.reject(problem)
                return False
        return True

    def cargo_metadata(self, checkout: Path, label: str):
        process = run_process(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--locked",
                "--all-features",
                "--manifest-path",
                str(checkout / "Cargo.toml"),
            ],
            timeout=self.args.timeout,
            log_path=self.session_dir / f"metadata-{label}.log",
        )
        if process.returncode != 0:
            self.reject(
                reject(
                    "dependency_metadata_failed",
                    f"cargo metadata failed for {label}; see metadata-{label}.log",
                    arm=label if label in ("base", "candidate") else None,
                )
            )
            return None
        return json.loads(process.stdout)

    def finish_calibration(self, tiers, drift_by_tier) -> int:
        order_cells = self.session["inventory"]["order"]
        ratios_by_tier = {}
        healthy = True
        for tier in tiers:
            ratios = self.build_ratios(tier, order_cells)
            if ratios is None:
                healthy = False
                continue
            ratios_by_tier[tier] = ratios
        prior_drift = 0.0
        prior_sessions, _ = validate_prior_session_files(
            getattr(self.args, "prior_session", [])
        )
        history = list(prior_sessions or [])
        for path in Path(self.args.records).glob("*/session.json"):
            if path.parent == self.session_dir:
                continue
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict):
                history.append(candidate)
        drift_sources = []
        for prior in history:
            if prior.get("kind") != "calibration":
                continue
            if prior.get("rejected") is True or prior.get("rejections"):
                continue
            if not same_host(prior.get("host") or {}, self.session.get("host") or {}):
                continue
            if (prior.get("host") or {}).get("toolchain_id") != (
                self.session.get("host") or {}
            ).get("toolchain_id"):
                continue
            if normalize_flags((prior.get("snapshot") or {}).get("rustflags") or "") != normalize_flags(
                (self.session.get("snapshot") or {}).get("rustflags") or ""
            ):
                continue
            drift_sources.append(prior.get("session_id"))
            prior_stats = session_tier_drift(prior)
            if not prior_stats and prior.get("drift_method") == DRIFT_METHOD:
                prior_stats = prior.get("drift") or {}
            for stats in prior_stats.values():
                prior_drift = max(prior_drift, float(stats.get("drift") or 0.0))
        recommendation, sufficient = calibration_recommendation(
            ratios_by_tier,
            self.session["config"]["delta"],
            drift_by_tier,
            drift_safety_factor=float(self.args.drift_safety_factor),
            prior_drift=prior_drift,
        )
        healthy = healthy and sufficient
        if any(
            record.get("available_runs", MIN_RUNS_FOR_BOUND) < MIN_RUNS_FOR_BOUND
            for record in recommendation["per_tier"].values()
        ):
            self.reject(
                reject(
                    "insufficient_runs",
                    f"calibration needs at least {MIN_RUNS_FOR_BOUND} paired "
                    "runs for a 95% HL bound",
                )
            )
        recommendation["drift_sources"] = drift_sources
        unstable_by_tier = {
            tier: list(record.get("unstable_cells") or [])
            for tier, record in recommendation["per_tier"].items()
        }
        recommendation["unstable_cells"] = unstable_by_tier
        self.session["unstable"] = {
            "path": None,
            "source": "calibration",
            "entries": {
                tier: sorted(cells) for tier, cells in sorted(unstable_by_tier.items())
            },
        }
        self.session["config"]["drift_cap"] = recommendation["drift_cap"]
        self.session["decisions"] = publishable_decisions({})
        snapshot = self.session.get("snapshot") or {}
        host = self.session.get("host") or {}
        self.session["calibration"] = {
            "mode": self.args.calibration_mode,
            "recommendation": recommendation,
            "inventory_digest": (self.session.get("inventory") or {}).get("digest"),
            "binding": {
                "hosts": [host.get("hostname")] if host.get("hostname") else [],
                "cpu_model": host.get("cpu_model"),
                "tiers": list(tiers),
                "tiers_by_host": (
                    {host.get("hostname"): list(tiers)}
                    if host.get("hostname")
                    else {}
                ),
                "toolchain": snapshot.get("toolchain"),
                "toolchain_id": host.get("toolchain_id"),
                "flags": snapshot.get("rustflags"),
            },
        }
        self.session["arm_summaries"] = {
            arm: {tier: {} for tier in tiers}
            for arm in ("base", "candidate")
        }
        self.summarize(tiers)
        self.session["rows"] = build_rows(
            self.session, tiers, order_cells, unstable_by_tier
        )
        if not healthy:
            self.reject(
                reject(
                    "calibration_insufficient",
                    "no run count in the calibration session reaches every stable "
                    "cell's bound at log(1 + delta/2); measure more runs",
                )
            )
            return 1
        return 0

    def build_layout_control(self) -> tuple[Path | None, dict]:
        base = self.session["arms"]["base"]
        record = {
            "sha": base["sha"],
            "checkout": base["checkout"],
            "metadata_seed": "fgf-pair-layout-control",
        }
        executable = self.build_bench("layout_control", record, self.snapshot)
        if executable is None:
            return None, record
        try:
            alias = self.prepare_execution_alias("layout_control", executable)
        except OSError:
            return None, record
        record["execution_path"] = str(alias)
        return alias, record

    def remeasure_layout_cells(self, tier: str, cells: list[str], executable: Path):
        """Fresh base-vs-layout-control runs for the failing cells of one tier."""
        base_executable = Path(
            self.session["arms"]["base"].get("execution_path")
            or self.session["arms"]["base"]["executable"]
        )
        filter_path = self.session_dir / f"layout-cells-{tier}.txt"
        filter_path.write_text("\n".join(cells) + "\n", encoding="utf-8")
        self.current_stage = "layout"
        base_runs = []
        control_runs = []
        controls = []
        environment_digests = []
        argv_lengths = []
        for run_index in range(self.session["config"]["runs"]):
            self.current_run = run_index
            if not self.check_preconditions(
                tier, run_index, "layout_remeasure"
            ):
                return None
            run_env = self.harness_env(tier)
            environment_digests.append(environment_digest(run_env))
            order = (
                ["base", "layout_control"]
                if run_index % 2 == 0
                else ["layout_control", "base"]
            )
            pieces = {}
            for arm in order:
                source = base_executable if arm == "base" else executable
                piece = self.timed_run(
                    tier,
                    arm,
                    source,
                    cells,
                    cells_filter=filter_path,
                    env=run_env,
                )
                if piece is None:
                    return None
                pieces[arm] = piece
            controls.append(
                {arm: pieces[arm]["control"] for arm in pieces}
            )
            lengths = {arm: pieces[arm]["argv_length"] for arm in pieces}
            problem = check_argv_lengths(
                lengths,
                f"layout-control re-measurement at tier {tier} run {run_index}",
            )
            if problem is not None:
                self.reject(problem)
                return None
            argv_lengths.append(lengths)
            base_runs.append(
                {
                    cell: entry["summary_ns"]
                    for cell, entry in pieces["base"]["arms"]["base"]["cells"].items()
                }
            )
            control_runs.append(
                {
                    cell: entry["summary_ns"]
                    for cell, entry in pieces["layout_control"]["arms"][
                        "layout_control"
                    ]["cells"].items()
                }
            )
        try:
            ratios = paired_log_ratios(base_runs, control_runs, cells)
        except (KeyError, ValueError) as error:
            self.reject(
                reject(
                    "layout_spread_mismatch",
                    f"layout-control re-measurement cannot be paired: {error}",
                )
            )
            return None
        starts = []
        for control in controls:
            starts.extend(entry["start_ns"] for entry in control.values())
        return {
            "bounds": {
                cell: hl_upper_bound(values)["upper_bound"]
                for cell, values in ratios.items()
            },
            "estimates": {
                cell: hl_upper_bound(values)["estimate"]
                for cell, values in ratios.items()
            },
            "controls": controls,
            "drift": drift_stats(starts),
            "runs": len(controls),
            "environment_digest": environment_digests,
            "argv_lengths": argv_lengths,
            "summaries": {
                "base": base_runs,
                "layout_control": control_runs,
            },
        }

    def layout_only(self, tiers, decisions) -> bool:
        failed_by_tier = {
            tier: decisions[tier]["noninferiority"]["failed_cells"]
            for tier in tiers
        }
        if not any(failed_by_tier.values()):
            self.session["layout_only"] = {
                "asserted": True,
                "per_tier": {tier: {} for tier in tiers},
            }
            return True
        paths = split_words(self.args.layout_sessions)
        if not paths:
            self.reject(
                reject(
                    "layout_spread_missing",
                    "layout-only mode needs --layout-sessions naming accepted "
                    "calibration sessions with layout controls",
                )
            )
            return False
        sessions, session_problem = validate_prior_session_files(paths)
        if session_problem is not None:
            self.reject(
                reject(
                    "layout_spread_mismatch",
                    session_problem.message,
                )
            )
            return False
        snapshot = self.session.get("snapshot") or {}
        host = self.session.get("host") or {}
        layout_executable = None
        layout_record = None
        disposition = {"asserted": True, "per_tier": {}}
        for tier in tiers:
            failed = failed_by_tier[tier]
            if not failed:
                disposition["per_tier"][tier] = {}
                continue
            spreads = layout_spreads(
                sessions,
                host,
                tier,
                snapshot=snapshot,
                toolchain_id=host.get("toolchain_id"),
            )
            if spreads is None:
                self.reject(
                    reject(
                        "layout_spread_mismatch",
                        f"no accepted layout-control calibration session matches "
                        f"this host, toolchain, flags, and snapshot at tier {tier}",
                    )
                )
                return False
            missing = [cell for cell in failed if spreads.get(cell) is None]
            if missing:
                self.reject(
                    reject(
                        "layout_spread_mismatch",
                        "accepted calibration sessions do not cover the failed "
                        "cells: " + ", ".join(missing),
                        cells=tuple(missing),
                    )
                )
                return False
            if layout_executable is None:
                layout_executable, layout_record = self.build_layout_control()
                if layout_executable is None:
                    return False
                disposition["control"] = {
                    "revision": "base",
                    "metadata_seed": layout_record["metadata_seed"],
                    "runs": self.session["config"]["runs"],
                    "rounds": self.args_rounds,
                }
            remeasured = self.remeasure_layout_cells(
                tier, failed, layout_executable
            )
            if remeasured is None:
                return False
            disposition.setdefault("remeasured", {})[tier] = remeasured
            tier_record = {}
            for cell in failed:
                bound = decisions[tier]["noninferiority"]["per_cell"][cell]["upper_bound"]
                calibration_spread = spreads[cell]
                session_bound = remeasured["bounds"].get(cell)
                if session_bound is None:
                    self.reject(
                        reject(
                            "layout_spread_mismatch",
                            f"layout-control re-measurement did not cover {cell}",
                            cells=(cell,),
                        )
                    )
                    return False
                spread = max(calibration_spread, session_bound)
                record = layout_disposition(bound, spread)
                record["calibration_spread"] = calibration_spread
                record["remeasured_bound"] = session_bound
                tier_record[cell] = record
            disposition["per_tier"][tier] = tier_record
        self.session["layout_only"] = disposition
        return True



def publishable_decisions(per_tier: dict) -> dict:
    """Aggregate top-level decisions plus every tier's detail."""
    noninferiority = {
        tier: decision["noninferiority"]
        for tier, decision in per_tier.items()
    }
    superiority = {
        tier: decision.get("superiority") or {
            "declared": False,
            "passed": None,
        }
        for tier, decision in per_tier.items()
    }
    declared = any(record.get("declared") for record in superiority.values())
    passed = all(
        record.get("passed") or record.get("declared") is not True
        for record in superiority.values()
    )
    return {
        "estimator": ESTIMATOR,
        "confidence": CONFIDENCE,
        "noninferiority": {
            "passed": all(
                record.get("passed") is True for record in noninferiority.values()
            ),
            "per_tier": noninferiority,
        },
        "superiority": {
            "declared": declared,
            "passed": passed if declared else None,
            "per_tier": superiority,
        },
        "per_tier": per_tier,
    }


def decisions_pass(decisions: dict, targets) -> bool:
    per_tier = decisions.get("per_tier", decisions)
    for tier_decision in per_tier.values():
        if not tier_decision["noninferiority"]["passed"]:
            return False
        superiority = tier_decision.get("superiority") or {}
        if superiority.get("declared") and not superiority.get("passed"):
            return False
    return True


def load_prior_sessions(paths) -> list[dict | None]:
    loaded = []
    for raw in paths or []:
        path = Path(raw)
        try:
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            loaded.append(None)
    return loaded


def validate_prior_session_files(paths):
    sessions = load_prior_sessions(paths)
    if any(not isinstance(session, dict) for session in sessions):
        return None, reject(
            "session_invalid",
            "a supplied session path is not a readable session.json object",
        )
    return sessions, None


def build_rows(
    session: dict, tiers, order_cells, unstable_by_tier=None
) -> list[dict]:
    unstable_by_tier = unstable_by_tier or {}
    rows = []
    for arm in ("base", "candidate"):
        arm_record = (session.get("arms") or {}).get(arm) or {}
        role = "candidate" if arm == "candidate" else "baseline"
        if arm == "candidate" and arm_record.get("kind") == "layout_control":
            role = "layout_control"
        if arm == "base" and arm_record.get("kind") == "layout_control":
            role = "layout_control"
        for tier in tiers:
            summaries = ((session.get("arm_summaries") or {}).get(arm) or {}).get(
                tier, {}
            )
            for cell in order_cells:
                if cell not in summaries:
                    continue
                parts = cell_parts(cell) or {}
                rounds = []
                for run in session.get("runs") or []:
                    if run.get("tier") != tier or run.get("stage", "main") != "main":
                        continue
                    entry = (
                        ((run.get("arms") or {}).get(arm) or {}).get("cells") or {}
                    ).get(cell)
                    if entry:
                        rounds.extend(entry.get("rounds_ns") or [])
                unstable = cell in set(unstable_by_tier.get(tier, ()))
                rows.append(
                    {
                        "host": session["host"].get("hostname"),
                        "tier": tier,
                        "field": parts.get("field"),
                        "operation": parts.get("operation"),
                        "prep": parts.get("prep"),
                        "row_len": parts.get("row_len"),
                        "rows": parts.get("rows"),
                        "sources": parts.get("sources"),
                        "alignment": parts.get("alignment"),
                        "cell": cell,
                        "arm": arm,
                        "library_sha": arm_record.get("sha"),
                        "session_id": session.get("session_id"),
                        "summary_ns": summaries[cell],
                        "rounds_ns": rounds,
                        "role": role,
                        "unstable": unstable,
                    }
                )
    rows.sort(key=lambda row: (row["arm"] != "base", row["tier"], row["cell"]))
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_execution_arguments(parser, *, records_default="bench-records/pair"):
    """Add the execution-environment arguments shared by every runner.

    Snapshot selection, repository, toolchain, pinned core, isolation
    sampling, record location, and checkout retention are identical for the
    paired campaign and any standalone baseline runner built on ``Campaign``.
    """
    parser.add_argument("--snapshot", default="HEAD",
                        help="measurement snapshot revision or directory")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                        help="repository holding the library revisions")
    parser.add_argument("--features", default="simd512",
                        help="snapshot feature request, comma or space separated")
    parser.add_argument("--rustflags", default=DEFAULT_ALIGN_FLAGS,
                        help="pinned RUSTFLAGS for both arms")
    parser.add_argument("--toolchain", default="",
                        help="toolchain override when the snapshot has no rust-toolchain file")
    parser.add_argument("--core", type=int,
                        default=int(os.environ["FEC_GOLDEN_CORE"])
                        if os.environ.get("FEC_GOLDEN_CORE", "").isdigit() else None,
                        help="pinned core (defaults to FEC_GOLDEN_CORE)")
    parser.add_argument("--smt-window", type=float, default=0.2,
                        help="seconds to sample the SMT sibling (floor 0.2)")
    parser.add_argument("--smt-busy-threshold", type=float, default=0.05,
                        help="maximum busy fraction tolerated (at most 0.05)")
    parser.add_argument("--chrt", default="",
                        help="chrt arguments prepended only when configured")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="seconds allowed per subprocess")
    parser.add_argument("--records", default=str(Path(records_default)),
                        help="session record root")
    parser.add_argument("--keep-checkouts", action="store_true",
                        help="keep the temporary arm checkouts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a candidate library revision against a base "
        "revision for the public operations bench."
    )
    parser.add_argument("--mode", choices=("acceptance", "calibration", "threshold"),
                        default="acceptance")
    parser.add_argument("--calibration-mode",
                        choices=("base_vs_base", "base_vs_layout_control"),
                        default="base_vs_base")
    parser.add_argument("--base", help="base library revision")
    parser.add_argument("--candidate", help="candidate library revision")
    add_execution_arguments(parser)
    parser.add_argument("--tiers", default=DEFAULT_TIERS,
                        help="backend tiers to measure, strongest first")
    parser.add_argument("--runs", default=PLACEHOLDER,
                        help="independent runs per tier (calibrated for acceptance)")
    parser.add_argument("--rounds", default="7",
                        help="rounds per arm per run")
    parser.add_argument("--drift-cap", default=PLACEHOLDER,
                        help="maximum relative start-control spread")
    parser.add_argument("--delta", default="0.03",
                        help="noninferiority policy delta")
    parser.add_argument("--gamma", default="0.03",
                        help="superiority policy gamma")
    parser.add_argument("--calibrated-hosts", default=PLACEHOLDER,
                        help="host binding recorded by calibration")
    parser.add_argument("--calibrated-tiers", default=PLACEHOLDER,
                        help="per-host tier binding, 'host:t1,t2 host2:t3,...'")
    parser.add_argument("--calibrated-toolchain", default=PLACEHOLDER,
                        help="toolchain binding recorded by calibration")
    parser.add_argument("--calibrated-flags", default=PLACEHOLDER,
                        help="flag binding recorded by calibration")
    parser.add_argument("--calibration-runs", type=int, default=8,
                        help="maximum runs collected by a calibration session")
    parser.add_argument("--meta-seed", default=None,
                        help="Cargo metadata seed for a layout-control arm")
    parser.add_argument("--targets", default=None,
                        help="target-cell declaration for the superiority rule")
    parser.add_argument("--cells", default=None,
                        help="timing-scope declaration restricting the timed cells")
    parser.add_argument("--scope-rationale", default=None,
                        help="selection rationale for a list or text --cells declaration")
    parser.add_argument("--calibrated-inventories",
                        default=str(
                            Path(__file__).resolve().parent / "calibrated_inventories"
                        ),
                        help="calibrated inventory snapshots named by inventory digest")
    parser.add_argument("--threshold", default=None,
                        help="threshold-evidence declaration (threshold mode)")
    parser.add_argument("--inventory", default=None,
                        help="threshold inventory.json (threshold mode)")
    parser.add_argument("--prior-session", action="append", default=[],
                        help="session.json consulted as prior evidence")
    parser.add_argument("--layout-only", action="store_true",
                        help="record layout-control dispositions for failed cells")
    parser.add_argument("--layout-sessions", default="",
                        help="layout-control calibration sessions, comma separated")
    parser.add_argument("--drift-safety-factor", type=float, default=1.25,
                        help="calibration drift-cap safety factor")
    parser.add_argument("--policy-json", default=None,
                        help="crate.just policy snapshot for override rejection")
    parser.add_argument("--exploratory", action="store_true",
                        help="allow policy overrides for a non-publishable run")
    parser.add_argument("--reanalyze", default=None,
                        help="recompute a stored session under the current HL method")
    parser.add_argument("--write-unstable", default=None,
                        help="merge calibration unstable cells into a JSON file")
    parser.add_argument("--unstable-session", action="append", default=[],
                        help="calibration session used by --write-unstable")
    parser.add_argument("--unstable-file",
                        default=str(Path(__file__).resolve().parent / "unstable_cells.json"),
                        help="signed unstable-cell list loaded by acceptance")
    parser.add_argument("--unstable-target-runs", default=PLACEHOLDER,
                        help="minimum runs for a listed superiority target")
    return parser


def _unstable_key(host: str, tier: str, toolchain: str, flags: str, digest: str) -> tuple:
    return (
        host,
        tier,
        toolchain,
        normalize_flags(flags),
        digest,
    )


def write_unstable_list(path: Path, session_paths) -> int:
    """Merge unstable cells from calibration sessions into the committed list."""
    existing: list[dict] = []
    if path.exists():
        entries, problem = load_unstable_entries(path)
        if problem is not None:
            print(f"bench_pair: {problem.message}", file=sys.stderr)
            return 2
        existing = entries
    merged: dict[tuple, dict] = {}
    for entry in existing:
        key = _unstable_key(
            entry["host"], entry["tier"], entry["toolchain"],
            entry["flags"], entry["inventory_digest"],
        )
        merged[key] = {
            "host": entry["host"],
            "tier": entry["tier"],
            "toolchain": entry["toolchain"],
            "flags": normalize_flags(entry["flags"]),
            "inventory_digest": entry["inventory_digest"],
            "cells": set(entry["cells"]),
            "sources": set(entry["sources"]),
            "owner_signature": entry.get("owner_signature"),
        }
    if not session_paths:
        print("bench_pair: --write-unstable needs --unstable-session", file=sys.stderr)
        return 2
    used = 0
    catalogues = {}
    selected_catalogues = {}
    for raw in session_paths:
        session_path = Path(raw)
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"bench_pair: cannot read {session_path}: {error}", file=sys.stderr)
            return 2
        if (
            not isinstance(session, dict)
            or session.get("kind") != "calibration"
            or session.get("rejected") is True
            or session.get("rejections")
        ):
            print(
                f"bench_pair: {session_path} is not a non-rejected calibration session",
                file=sys.stderr,
            )
            return 2
        calibration = session.get("calibration") or {}
        recommendation = calibration.get("recommendation") or {}
        per_tier = recommendation.get("per_tier") or {}
        host = session.get("host") or {}
        snapshot = session.get("snapshot") or {}
        inventory = session.get("inventory") or {}
        base = inventory.get("base") or []
        order = inventory.get("order") or []
        entries = {entry["id"]: entry for entry in base if isinstance(entry, dict) and "id" in entry}
        if (not order or len(order) != len(entries) or len(entries) != len(base) or set(entries) != set(order)
                or any(validate_inventory_entry(entry) is not None for entry in base)):
            print(f"bench_pair: {session_path} lacks a complete calibrated inventory", file=sys.stderr)
            return 2
        digest = inventory_digest(entries, order)
        if inventory.get("digest", digest) != digest:
            print(f"bench_pair: {session_path} has a mismatched inventory digest", file=sys.stderr)
            return 2
        if session.get("scope") and session["scope"].get("kind") != "full":
            print(f"bench_pair: {session_path} is a selected-scope calibration", file=sys.stderr)
            return 2
        catalogues[digest] = [entries[cell] for cell in order]
        toolchain = host.get("toolchain_id")
        flags = snapshot.get("rustflags") or ""
        hostname = host.get("hostname")
        if not hostname:
            print(f"bench_pair: {session_path} has no hostname", file=sys.stderr)
            return 2
        for tier, record in per_tier.items():
            cells = record.get("unstable_cells") or []
            if not isinstance(cells, list) or not set(cells) <= set(entries):
                print(f"bench_pair: {session_path} lists unstable cells outside its inventory", file=sys.stderr)
                return 2
            key = _unstable_key(hostname, tier, toolchain, flags, digest)
            context = key[:-1]
            if context in selected_catalogues and selected_catalogues[context] != digest:
                print("bench_pair: calibration inputs disagree on the inventory for one host/tier", file=sys.stderr)
                return 2
            selected_catalogues[context] = digest
            for previous in list(merged):
                if previous[:-1] == context and previous != key:
                    del merged[previous]
            entry = merged.setdefault(
                key,
                {
                    "host": hostname,
                    "tier": tier,
                    "toolchain": toolchain,
                    "flags": normalize_flags(flags),
                    "inventory_digest": digest,
                    "cells": set(),
                    "sources": set(),
                    "owner_signature": None,
                },
            )
            entry["cells"].update(cells)
            source_name = (
                session.get("session_id")
                or session.get("source_session_id")
                or session.get("source_session")
                or session_path.name
            )
            entry["sources"].add(Path(source_name).name)
            entry["owner_signature"] = None
            used += 1
    output = {
        "entries": [
            {
                "host": entry["host"],
                "tier": entry["tier"],
                "toolchain": entry["toolchain"],
                "flags": entry["flags"],
                "inventory_digest": entry["inventory_digest"],
                "cells": sorted(entry["cells"]),
                "sources": sorted(entry["sources"]),
                "owner_signature": entry["owner_signature"],
            }
            for _, entry in sorted(
                merged.items(),
                key=lambda item: tuple(str(part) for part in item[0]),
            )
        ]
    }
    catalogue_root = path.parent / "calibrated_inventories"
    catalogue_root.mkdir(parents=True, exist_ok=True)
    for digest, entries in catalogues.items():
        catalogue_path = catalogue_root / f"{digest}.json"
        catalogue_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: {used} tier records, {len(output['entries'])} entries")
    return 0


def reanalyze_session(path: Path, args) -> int:
    """Recompute a stored session without measuring; never publishable."""
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"bench_pair: cannot read {path}: {error}", file=sys.stderr)
        return 2
    if not isinstance(source, dict):
        print(f"bench_pair: {path} is not a session object", file=sys.stderr)
        return 2
    source_method = source.get("method_version")
    if source_method in METHOD_DRIFT:
        expected_drift = METHOD_DRIFT[source_method]
        if source.get("drift_method") != expected_drift:
            print(
                f"bench_pair: {path} drift_method must be {expected_drift} "
                f"for {source_method}",
                file=sys.stderr,
            )
            return 2
    kind = source.get("kind")
    if kind not in ("acceptance", "calibration", "threshold"):
        print(f"bench_pair: {path} has unsupported kind {kind!r}", file=sys.stderr)
        return 2
    config = source.get("config") or {}
    try:
        delta = float(config.get("delta", args.delta))
        gamma = float(config.get("gamma", args.gamma))
        factor = float(
            config.get(
                "drift_safety_factor",
                getattr(args, "drift_safety_factor", 1.25),
            )
        )
    except (TypeError, ValueError):
        print(f"bench_pair: {path} has invalid policy numbers", file=sys.stderr)
        return 2
    runs_by_tier = {}
    ratios_by_tier = {}
    for run in source.get("runs") or []:
        tier = run.get("tier")
        if isinstance(tier, str):
            runs_by_tier.setdefault(tier, 0)
            if run.get("stage", "main") == "main":
                runs_by_tier[tier] += 1
    for tier in sorted(runs_by_tier):
        ratios = session_tier_ratios(source, tier)
        if ratios is not None:
            ratios_by_tier[tier] = ratios
    # A recorded selected scope constrains the recomputation to exactly its
    # ordered ids; a session whose runs do not cover the recorded scope is
    # rejected instead of silently recomputing a different cell set.  Records
    # without a scope keep their original whole-inventory meaning.
    scope = source.get("scope")
    selected = None
    if isinstance(scope, dict) and scope.get("kind") == "selected":
        selected = [
            cell for cell in scope.get("cells") or []
            if isinstance(cell, str) and cell
        ]
    if selected is not None:
        selected_set = set(selected)
        for tier in sorted(ratios_by_tier):
            ratios = ratios_by_tier[tier]
            missing = [cell for cell in selected if cell not in ratios]
            if missing:
                print(
                    f"bench_pair: {path} records a selected scope but tier "
                    f"{tier} lacks runs for {', '.join(missing)}",
                    file=sys.stderr,
                )
                return 2
            ratios_by_tier[tier] = {
                cell: values for cell, values in ratios.items()
                if cell in selected_set
            }
    drift_by_tier = session_tier_drift(source)
    source_inventory = source.get("inventory") or {}
    inventory_digest_value = source_inventory.get("digest")
    if inventory_digest_value is None and source_inventory:
        inventory_digest_value = inventory_digest(
            source_inventory.get("base") or {},
            source_inventory.get("order") or [],
        )
    source_snapshot = source.get("snapshot") or {}
    output = {
        "method_version": METHOD_VERSION,
        "estimator": ESTIMATOR,
        "confidence": CONFIDENCE,
        "drift_method": DRIFT_METHOD,
        "reanalysis": True,
        "publishable": False,
        "acceptance": False,
        "kind": kind,
        "source_session": str(path),
        "source_session_id": source.get("session_id"),
        "source_method_version": source.get("method_version"),
        "source_drift_method": source.get("drift_method"),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": source.get("host"),
        "snapshot": {
            "rustflags": source_snapshot.get("rustflags"),
            "toolchain": source_snapshot.get("toolchain"),
            "digest": source_snapshot.get("digest"),
        },
        "inventory": (
            {
                "digest": inventory_digest_value,
                "equal": source_inventory.get("equal"),
            }
            if inventory_digest_value is not None
            else None
        ),
        "scope": source.get("scope"),
        "config": {
            "delta": delta,
            "confirmation": config.get("confirmation", False),
            "gamma": gamma,
            "drift_safety_factor": factor,
            "drift_method": DRIFT_METHOD,
            "estimator": ESTIMATOR,
            "confidence": CONFIDENCE,
        },
        "tiers": {
            tier: {
                "runs": runs_by_tier[tier],
                "available_runs": min(
                    (len(values) for values in ratios_by_tier.get(tier, {}).values()),
                    default=0,
                ),
            }
            for tier in sorted(runs_by_tier)
        },
        "drift": drift_by_tier,
        "confirmation_drift": session_tier_drift(source, "confirmation"),
        "confirmed_failures": {},
        "decisions": {},
        "calibration": None,
        "threshold": None,
        "insufficient_runs": False,
    }
    if kind == "calibration":
        recommendation, healthy = calibration_recommendation(
            ratios_by_tier,
            delta,
            drift_by_tier,
            drift_safety_factor=factor,
        )
        output["calibration"] = {
            "mode": (source.get("calibration") or {}).get("mode"),
            "recommendation": recommendation,
            "healthy": healthy,
        }
        output["insufficient_runs"] = any(
            record.get("available_runs", MIN_RUNS_FOR_BOUND) < MIN_RUNS_FOR_BOUND
            for record in recommendation["per_tier"].values()
        )
    else:
        targets = None
        if kind == "threshold":
            threshold = source.get("threshold") or {}
            targets = threshold.get("target_set")
            output["threshold"] = {
                "item_key": threshold.get("item_key"),
                "target_set": list(targets or []),
            }
        else:
            declaration = config.get("targets_declaration") or {}
            targets = declaration.get("cells")
        per_tier = {}
        insufficient = False
        decision_tiers = runs_by_tier if config.get("confirmation") else ratios_by_tier
        for tier in decision_tiers:
            ratios = ratios_by_tier.get(tier, {})
            try:
                excluded = ((source.get("unstable") or {}).get("entries") or {}).get(tier, [])
                if config.get("confirmation"):
                    tier_decision = recorded_staged_decision(source, tier, targets)
                    output["confirmed_failures"][tier] = tier_decision["confirmed_failures"]
                else:
                    tier_decision = {
                        "noninferiority": evaluate_noninferiority(ratios, delta, excluded)
                    }
                    if targets is not None:
                        tier_decision["superiority"] = evaluate_superiority(
                            ratios, targets, gamma, delta, excluded=excluded,
                            unstable_targets=set(excluded) & set(targets),
                        )
                    else:
                        tier_decision["superiority"] = {
                            "declared": False,
                            "passed": None,
                        }
                per_tier[tier] = tier_decision
            except InsufficientRuns as error:
                insufficient = True
                per_tier[tier] = {
                    "insufficient_runs": str(error),
                    "runs": min((len(values) for values in ratios.values()), default=0),
                }
            except (ValueError, KeyError, TypeError) as error:
                print(f"bench_pair: {path}: {error}", file=sys.stderr)
                return 2
        output["decisions"] = (
            {"estimator": ESTIMATOR, "confidence": CONFIDENCE, "per_tier": per_tier}
            if insufficient else publishable_decisions(per_tier)
        )
        output["insufficient_runs"] = insufficient
    stamp = utc_stamp()
    target = path.with_name(f"{path.stem}.reanalysis-{stamp}.json")
    counter = 2
    while target.exists():
        target = path.with_name(f"{path.stem}.reanalysis-{stamp}-{counter}.json")
        counter += 1
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(target)
    if config.get("confirmation") and kind != "calibration":
        return 0 if not output["insufficient_runs"] and decisions_pass(
            output["decisions"], targets
        ) else 1
    return 0


def base_session(args, session_id: str) -> dict:
    return {
        "method_version": METHOD_VERSION,
        "session_id": session_id,
        "drift_method": DRIFT_METHOD,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "kind": "calibration" if args.mode == "calibration" else (
            "threshold" if args.mode == "threshold" else "acceptance"
        ),
        "acceptance": args.mode == "acceptance"
        and not bool(getattr(args, "exploratory", False)),
        "exploratory": bool(getattr(args, "exploratory", False)),
        "reused_calibration": False,
        "rejected": False,
        "host": {},
        "snapshot": {},
        "arms": {},
        "config": {
            "runs": args.runs,
            "confirmation": args.mode in ("acceptance", "threshold"),
            "rounds": args.rounds,
            "delta": args.delta,
            "gamma": args.gamma,
            "drift_cap": args.drift_cap,
            "features": split_words(args.features),
            "rustflags": args.rustflags,
            "toolchain": args.toolchain or None,
            "calibrated": False,
            "targets_declaration": None,
            "threshold_declaration": None,
            "smt_window": 0.2,
            "smt_busy_threshold": 0.05,
            "drift_safety_factor": 1.25,
            "chrt": "",
            "estimator": ESTIMATOR,
            "confidence": CONFIDENCE,
            "unstable_file": getattr(
                args,
                "unstable_file",
                str(Path(__file__).resolve().parent / "unstable_cells.json"),
            ),
            "unstable_target_runs": getattr(
                args, "unstable_target_runs", PLACEHOLDER
            ),
        },
        "execution": {},
        "preconditions": [],
        "route_detail": None,
        "route_limitation": None,
        "tiers": [],
        "inventory": {},
        "scope": None,
        "uncalibrated": None,
        "checks": {},
        "route_diagnostics": {},
        "runs": [],
        "arm_summaries": {"base": {}, "candidate": {}},
        "drift": {},
        "confirmation_drift": {},
        "confirmed_failures": {},
        "decisions": {},
        "rejections": [],
        "rows": [],
        "layout_only": None,
        "calibration": None,
        "threshold": None,
    }


def write_session(session: dict, path: Path) -> None:
    path.write_text(
        json.dumps(session, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def run(args) -> int:
    records = Path(args.records)
    host = socket.gethostname()
    stamp = utc_stamp()
    base_name = f"{stamp}-{host_component(host)}"
    session_dir = records / base_name
    counter = 2
    while session_dir.exists():
        session_dir = records / f"{base_name}-{counter}"
        counter += 1
    session_dir.mkdir(parents=True, exist_ok=True)
    session = base_session(args, session_dir.name)
    campaign = Campaign(args, session, session_dir)
    exit_code = 1
    try:
        try:
            exit_code = campaign.execute()
        except Exception as error:  # keep the failed session on disk
            campaign.reject(
                reject(
                    "internal_error",
                    f"{type(error).__name__}: {error}",
                )
            )
            exit_code = 1
        finally:
            session["rejections"] = [
                rejection.as_dict() for rejection in campaign.rejections
            ]
            session["rejected"] = bool(campaign.rejections)
            if session.get("rejections"):
                exit_code = 1
            write_session(session, session_dir / "session.json")
            campaign.log(
                f"finish method={METHOD_VERSION} session={session['session_id']} "
                f"exit={exit_code} rejections={len(campaign.rejections)}"
            )
            campaign.close()
    finally:
        campaign.release_lock()
    return exit_code


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_unstable:
        return write_unstable_list(
            Path(args.write_unstable), args.unstable_session
        )
    if args.reanalyze:
        return reanalyze_session(Path(args.reanalyze), args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
