"""Balanced page-placement strata for paired operation measurements."""

import math
import statistics

METHOD = "page-stratified/1"
COUNT = 256


def plan(run=0):
    """Cover every cache-line buffer offset and every ABI-aligned stack offset.

    This is a stratified design, not a Cartesian interaction sweep. The source
    permutation changes between independent runs to avoid pinning an interaction
    to one destination/stack pairing.
    """
    return [
        {"index": i, "dst_shift": (i % 64) * 64,
         "src_shift": ((17 * i + i // 64 + run) % 64) * 64,
         "stack_pad": ((i + 37 * run) % COUNT) * 16}
        for i in range(COUNT)
    ]


def environment(base, point):
    env = dict(base)
    env["FGF_PLACEMENT_DST"] = f'{point["dst_shift"]:04d}'
    env["FGF_PLACEMENT_SRC"] = f'{point["src_shift"]:04d}'
    env["FGF_PLACEMENT_STACK"] = "x" * point["stack_pad"]
    return env


def validate_record(text, point):
    import json
    records = [json.loads(line.split("\t", 1)[1]) for line in text.splitlines()
               if line.startswith("PLACEMENT\t")]
    if len(records) != 1:
        raise ValueError("expected exactly one PLACEMENT record")
    record = records[0]
    for key in ("dst_shift", "src_shift"):
        if type(record.get(key)) is not int or record[key] != point[key]:
            raise ValueError(f"placement mismatch: {key}")
    stack = record.get("stack_offset")
    if type(stack) is not int or not 0 <= stack < 4096:
        raise ValueError("invalid stack page offset")
    return record


def summarize(pieces, cells, rounds):
    """Median of equally weighted placement medians; never pseudoreplicate runs."""
    if len(pieces) != COUNT:
        raise ValueError("incomplete placement sweep")
    results = {}
    for cell in cells:
        entries = [piece["cells"][cell] for piece in pieces]
        values = [entry["summary_ns"] for entry in entries]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid placement timing")
        if any(len(entry["rounds_ns"]) != rounds for entry in entries):
            raise ValueError("incomplete placement rounds")
        results[cell] = {
            "summary_ns": statistics.median(values),
            "rounds_ns": [statistics.median(entry["rounds_ns"][r] for entry in entries)
                          for r in range(rounds)],
        }
    return {"cells": results}


def validate_run(record, rounds):
    """Reject missing strata, skewed stack coverage, and altered summaries."""
    strata = record["placements"]
    expected = plan(record["run"])
    if len(strata) != COUNT:
        raise ValueError("incomplete placement sweep")
    for actual, point in zip(strata, expected):
        if any(actual.get(key) != value for key, value in point.items()):
            raise ValueError("placement strata differ from the declared design")
    for arm, summary in record["arms"].items():
        stack = [p["observed"][arm]["stack_offset"] for p in strata]
        if len(set(stack)) != COUNT or len({offset % 16 for offset in stack}) != 1:
            raise ValueError("observed stack offsets do not cover the page uniformly")
        for p in strata:
            observed = p["observed"][arm]
            if any(observed[key] != p[key] for key in ("dst_shift", "src_shift")):
                raise ValueError("observed buffer shifts differ from the declared design")
        recomputed = summarize([p["arms"][arm] for p in strata], summary["cells"], rounds)
        if recomputed != summary:
            raise ValueError("run summary differs from retained placement measurements")
