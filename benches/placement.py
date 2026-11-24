"""Balanced page-placement strata for paired operation measurements."""

import json
import math
import statistics

METHOD = "page-stratified/1"
COUNT = 256
_SUMMARY_TOL = 1e-12
_STORED_TOL = 1e-9


def plan(run=0):
    """Cover every cache-line buffer offset, relative phase, and stack offset.

    Each run visits every 64-byte destination offset, source offset, and
    relative `(src_shift - dst_shift) mod 4096` phase four times, and every
    16-byte ABI-aligned stack padding once. Across every consecutive pair of
    runs `(2k, 2k+1)`, every requested stack padding runs in both arm orders;
    within every run, every destination, source, and relative phase runs twice
    in each arm order.
    """
    shift = (37 * (run // 2)) % COUNT
    points = []
    for i in range(COUNT):
        r = i // 4
        # Cycle q as 0, 2, 1, 3 so consecutive strata alternate arm order while
        # balancing dst, src, and relative (src - dst) phases 2/2 within a run.
        q = (i % 2) * 2 + ((i // 2) % 2)
        # Rotate i right by 1 bit so stack-pad parity is balanced 64/64 within
        # every run; keeping `shift` constant across each run pair (2k, 2k+1)
        # while arm_order flips by `run` runs every pad in both orders.
        pad_idx = (((i >> 1) | ((i & 1) << 7)) + shift) % COUNT
        points.append({
            "index": i,
            "dst_shift": r * 64,
            "src_shift": ((3 * r + q + run) % 64) * 64,
            "stack_pad": pad_idx * 16,
        })
    return points


def arm_order(arms, run, index):
    """Balanced arm order for stratum `index` of run `run`."""
    ordered = list(arms)
    if (run + index) % 2:
        ordered.reverse()
    return ordered


def environment(base, point):
    env = dict(base)
    env["FGF_PLACEMENT_DST"] = f'{point["dst_shift"]:04d}'
    env["FGF_PLACEMENT_SRC"] = f'{point["src_shift"]:04d}'
    env["FGF_PLACEMENT_STACK"] = "x" * point["stack_pad"]
    return env


def _require_pos_number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {what}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid {what}")
    return number


def _validate_observed_mapping(observed, point):
    if not isinstance(observed, dict):
        raise ValueError("PLACEMENT record must be a JSON object")
    for key in ("dst_shift", "src_shift"):
        if type(observed.get(key)) is not int or observed[key] != point[key]:
            raise ValueError(f"placement mismatch: {key}")
    stack = observed.get("stack_offset")
    if type(stack) is not int or not 0 <= stack < 4096:
        raise ValueError("invalid stack page offset")
    return observed


def validate_record(text, point):
    records = [json.loads(line.split("\t", 1)[1]) for line in text.splitlines()
               if line.startswith("PLACEMENT\t")]
    if len(records) != 1:
        raise ValueError("expected exactly one PLACEMENT record")
    return _validate_observed_mapping(records[0], point)


def summarize(pieces, cells, rounds):
    """Median of equally weighted placement medians; never pseudoreplicate runs."""
    if len(pieces) != COUNT:
        raise ValueError("incomplete placement sweep")
    results = {}
    for cell in cells:
        entries = [piece["cells"][cell] for piece in pieces]
        values = []
        round_matrix = []
        for entry in entries:
            raw_rounds = entry["rounds_ns"]
            if not isinstance(raw_rounds, list) or len(raw_rounds) != rounds:
                raise ValueError("incomplete placement rounds")
            checked_rounds = [_require_pos_number(v, "placement round timing") for v in raw_rounds]
            summary_ns = _require_pos_number(entry["summary_ns"], "placement timing")
            expected_summary = statistics.median(checked_rounds)
            if not math.isclose(summary_ns, expected_summary, rel_tol=_SUMMARY_TOL, abs_tol=_SUMMARY_TOL):
                raise ValueError("placement summary disagrees with retained rounds")
            values.append(summary_ns)
            round_matrix.append(checked_rounds)
        results[cell] = {
            "summary_ns": statistics.median(values),
            "rounds_ns": [statistics.median(row[r] for row in round_matrix)
                          for r in range(rounds)],
        }
    return {"cells": results}


def _summaries_close(actual, expected):
    if not isinstance(actual, dict) or set(actual) != {"cells"}:
        return False
    actual_cells = actual["cells"]
    expected_cells = expected["cells"]
    if not isinstance(actual_cells, dict) or set(actual_cells) != set(expected_cells):
        return False
    for cell, exp in expected_cells.items():
        act = actual_cells[cell]
        if not isinstance(act, dict):
            return False
        try:
            act_sum = _require_pos_number(act["summary_ns"], "run summary")
        except (ValueError, KeyError):
            return False
        if not math.isclose(act_sum, exp["summary_ns"], rel_tol=_STORED_TOL, abs_tol=_STORED_TOL):
            return False
        act_rounds = act.get("rounds_ns")
        exp_rounds = exp["rounds_ns"]
        if not isinstance(act_rounds, list) or len(act_rounds) != len(exp_rounds):
            return False
        for a_val, e_val in zip(act_rounds, exp_rounds):
            try:
                num = _require_pos_number(a_val, "run round summary")
            except ValueError:
                return False
            if not math.isclose(num, e_val, rel_tol=_STORED_TOL, abs_tol=_STORED_TOL):
                return False
    return True


def validate_run(record, rounds, expected_arms=None, expected_cells=None):
    """Reject missing strata, biased orders, skewed stacks, and altered summaries."""
    run_index = record["run"]
    if type(run_index) is not int or run_index < 0:
        raise ValueError("invalid run index")
    strata = record["placements"]
    if not isinstance(strata, list) or len(strata) != COUNT:
        raise ValueError("incomplete placement sweep")
    expected_plan = plan(run_index)
    for actual, point in zip(strata, expected_plan):
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in point.items()):
            raise ValueError("placement strata differ from the declared design")

    arms_map = record["arms"]
    if not isinstance(arms_map, dict) or not arms_map:
        raise ValueError("missing arm summaries")
    if expected_arms is not None:
        base_arms = list(expected_arms)
        if set(arms_map) != set(base_arms):
            raise ValueError("run arms differ from expected arms")
    else:
        base_arms = list(arms_map)
    if "order" in record:
        expected_run_order = arm_order(base_arms, run_index, 0)
        if record["order"] != expected_run_order and (
            expected_arms is not None or set(record["order"]) != set(base_arms)
        ):
            raise ValueError("run arm order differs from the declared design")
        if expected_arms is None:
            # Recover canonical arm ordering when caller did not fix expected_arms.
            base_arms = arm_order(record["order"], run_index, 0)

    cell_sets = {arm: set(summary["cells"]) for arm, summary in arms_map.items()}
    first_cells = next(iter(cell_sets.values()))
    if not first_cells or any(cells != first_cells for cells in cell_sets.values()):
        raise ValueError("arm summaries do not cover the same non-empty cell set")
    if expected_cells is not None and first_cells != set(expected_cells):
        raise ValueError("arm summaries differ from expected cells")

    for point, stratum in zip(expected_plan, strata):
        if "order" in stratum:
            if stratum["order"] != arm_order(base_arms, run_index, point["index"]):
                raise ValueError("stratum arm order differs from the declared design")
        if "environment_digest" in stratum:
            digest = stratum["environment_digest"]
            if not isinstance(digest, str) or not digest:
                raise ValueError("stratum environment digest is missing")
        if "argv_lengths" in stratum:
            lengths = stratum["argv_lengths"]
            if (not isinstance(lengths, dict) or set(lengths) != set(base_arms)
                    or any(type(v) is not int or v <= 0 for v in lengths.values())
                    or len(set(lengths.values())) != 1):
                raise ValueError("stratum argv lengths differ across arms")

    stacks_by_arm = {}
    for arm, summary in arms_map.items():
        stack = []
        for point, p in zip(expected_plan, strata):
            observed = _validate_observed_mapping(p["observed"][arm], point)
            stack.append(observed["stack_offset"])
        if len(set(stack)) != COUNT or len({offset % 16 for offset in stack}) != 1:
            raise ValueError("observed stack offsets do not cover the page uniformly")
        stacks_by_arm[arm] = stack
        recomputed = summarize([p["arms"][arm] for p in strata], list(summary["cells"]), rounds)
        if not _summaries_close(summary, recomputed):
            raise ValueError("run summary differs from retained placement measurements")

    if len(base_arms) >= 2:
        ref = stacks_by_arm[base_arms[0]]
        for other in base_arms[1:]:
            diffs = {(a - b) % 4096 for a, b in zip(ref, stacks_by_arm[other])}
            if len(diffs) != 1:
                raise ValueError("inter-arm stack offset delta varies across strata")

    if "controls" in record:
        controls = record["controls"]
        if not isinstance(controls, dict) or set(controls) != set(base_arms):
            raise ValueError("run controls differ from expected arms")
        for arm in base_arms:
            for key in ("start_ns", "end_ns"):
                stratum_vals = [
                    _require_pos_number(p["controls"][arm][key], f"stratum control {key}")
                    for p in strata
                ]
                expected_ctrl = statistics.median(stratum_vals)
                actual_ctrl = _require_pos_number(controls[arm][key], f"run control {key}")
                if not math.isclose(actual_ctrl, expected_ctrl, rel_tol=_STORED_TOL, abs_tol=_STORED_TOL):
                    raise ValueError("run controls differ from retained placement controls")
        if "control" in record and "order" in record:
            first_arm = record["order"][0]
            for key in ("start_ns", "end_ns"):
                actual_top = _require_pos_number(record["control"][key], f"top control {key}")
                expected_top = _require_pos_number(controls[first_arm][key], f"run control {key}")
                if not math.isclose(actual_top, expected_top, rel_tol=_STORED_TOL, abs_tol=_STORED_TOL):
                    raise ValueError("top-level run control differs from leading arm control")
