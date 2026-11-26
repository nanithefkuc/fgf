"""Balanced page-placement strata for paired operation measurements."""

import json
import math
import statistics

METHOD = "page-sparse/1"
COUNT = 8
RUN_STRIDE = 8
PROBE_STRIDE = 4

STRATIFIED_METHOD = "page-stratified/1"
STRATIFIED_COUNT = 256

METHODS = {
    METHOD: COUNT,
    STRATIFIED_METHOD: STRATIFIED_COUNT,
}
_SUMMARY_TOL = 1e-12
_STORED_TOL = 1e-9


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"unsupported placement method: {method!r}")
    return METHODS[method]


def plan(run=0, method=METHOD):
    """Return the deterministic placement strata for paired run ``run``.

    ``page-sparse/1`` (``fgf-pair/7``, ``COUNT = 8``) covers all eight 512-byte
    page segments for destination, source, and stack padding within each run,
    balances all four mod-64 stack classes and all four run relative phases 1/1
    across arm orders within each run, covers all 64 destination and source
    cache lines across every 8-run block, covers all 64 destination, source,
    and relative phases twice across every 16-run cycle (with 1/1 destination
    order balance), and covers all 256 16-byte stack pads across 32 runs.

    ``page-stratified/1`` (frozen ``fgf-pair/6``, ``COUNT = 256``) is retained
    for historical session validation.
    """
    _check_method(method)
    if method == METHOD:
        m = (run // 2) % 4
        o = run % 2
        b = (run // 8) % 2
        k = (13 * run) % 32
        points = []
        for i in range(COUNT):
            dst_u = (8 * i + m + 4 * o) % 64
            rel_u = (16 * i + 2 * m + b + 8 * o) % 64
            src_u = (dst_u + rel_u) % 64
            stack_pad = 16 * (((32 * i + 4 * k) % 256) + (i % 4))
            points.append({
                "index": i,
                "dst_shift": dst_u * 64,
                "src_shift": src_u * 64,
                "stack_pad": stack_pad,
            })
        return points
    shift = (37 * (run // 2)) % STRATIFIED_COUNT
    points = []
    for i in range(STRATIFIED_COUNT):
        r = i // 4
        q = (i % 2) * 2 + ((i // 2) % 2)
        pad_idx = (((i >> 1) | ((i & 1) << 7)) + shift) % STRATIFIED_COUNT
        points.append({
            "index": i,
            "dst_shift": r * 64,
            "src_shift": ((3 * r + q + run) % 64) * 64,
            "stack_pad": pad_idx * 16,
        })
    return points


def arm_order(arms, run, index, method=METHOD):
    """Balanced arm order for stratum ``index`` of run ``run``."""
    _check_method(method)
    ordered = list(arms)
    if method == METHOD:
        if (run + index + (index // 4) + (run // 8)) % 2:
            ordered.reverse()
    elif (run + index) % 2:
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


def summarize(pieces, cells, rounds, method=METHOD):
    """Median of equally weighted placement medians; never pseudoreplicate runs."""
    expected_count = _check_method(method)
    if len(pieces) != expected_count:
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


def validate_run(record, rounds, expected_arms=None, expected_cells=None, method=METHOD):
    """Reject missing strata, biased orders, skewed stacks, and altered summaries."""
    expected_count = _check_method(method)
    run_index = record["run"]
    if type(run_index) is not int or run_index < 0:
        raise ValueError("invalid run index")
    strata = record["placements"]
    if not isinstance(strata, list) or len(strata) != expected_count:
        raise ValueError("incomplete placement sweep")
    expected_plan = plan(run_index, method=method)
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
        expected_run_order = arm_order(base_arms, run_index, 0, method=method)
        if record["order"] != expected_run_order and (
            expected_arms is not None or set(record["order"]) != set(base_arms)
        ):
            raise ValueError("run arm order differs from the declared design")
        if expected_arms is None:
            # Recover canonical arm ordering when caller did not fix expected_arms.
            base_arms = arm_order(record["order"], run_index, 0, method=method)

    cell_sets = {arm: set(summary["cells"]) for arm, summary in arms_map.items()}
    first_cells = next(iter(cell_sets.values()))
    if not first_cells or any(cells != first_cells for cells in cell_sets.values()):
        raise ValueError("arm summaries do not cover the same non-empty cell set")
    if expected_cells is not None and first_cells != set(expected_cells):
        raise ValueError("arm summaries differ from expected cells")

    for point, stratum in zip(expected_plan, strata):
        if method == METHOD:
            missing = [key for key in ("order", "environment_digest", "argv_lengths")
                       if key not in stratum]
            if missing:
                raise ValueError("stratum is missing " + ", ".join(missing))
        if "order" in stratum:
            if stratum["order"] != arm_order(base_arms, run_index, point["index"], method=method):
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
        if "controls" in stratum or "controls" in record:
            stratum_controls = stratum.get("controls")
            if not isinstance(stratum_controls, dict) or set(stratum_controls) != set(base_arms):
                raise ValueError("stratum controls differ from expected arms")
            for arm in base_arms:
                ctrl_entry = stratum_controls[arm]
                if not isinstance(ctrl_entry, dict):
                    raise ValueError(f"stratum control for {arm} must be an object")
                for key in ("start_ns", "end_ns"):
                    _require_pos_number(ctrl_entry.get(key), f"stratum control {arm}.{key}")

    stacks_by_arm = {}
    for arm, summary in arms_map.items():
        stack = []
        for point, p in zip(expected_plan, strata):
            observed = _validate_observed_mapping(p["observed"][arm], point)
            stack.append(observed["stack_offset"])
        if len(set(stack)) != expected_count or len({offset % 16 for offset in stack}) != 1:
            raise ValueError("observed stack offsets do not cover the page uniformly")
        if method == METHOD and len({(offset - stack[0]) % 64 for offset in stack}) != 4:
            raise ValueError("observed stack offsets do not cover all four mod-64 classes")
        stacks_by_arm[arm] = stack
        recomputed = summarize(
            [p["arms"][arm] for p in strata], list(summary["cells"]), rounds, method=method
        )
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
