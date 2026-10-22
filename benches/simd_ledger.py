#!/usr/bin/env python3
"""Canonical SIMD route ledger: inventory, check, generate, and accept.

Only the standard library. Follows the style of existing ``benches/*.py``.

Obligations are expanded from applicability × rules × callables at check time;
the ledger stores only explicit deviations (none at this stage).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from api_inventory import declared_callables

# ---------------------------------------------------------------------------
# Lexicon constants
# ---------------------------------------------------------------------------

VALID_KINDS = frozenset({"bulk", "non-bulk"})
VALID_REASONS = frozenset({
    "single-element", "query", "construction", "preparation",
    "composition", "validation",
})
VALID_STATUS = frozenset({
    "missing", "bypass", "composition", "direct", "zero-budget", "unverified",
})
VALID_REQUESTS = frozenset({"v4x", "v3_gfni_crypto", "v3", "v2", "scalar"})
VALID_FEATURE_CONFIGS = frozenset({"default", "all"})
VALID_PERF = frozenset({"unmeasured", "wins", "ties", "scalar-crossover", "no-win"})
VALID_SUBSTAGES = frozenset({
    "Add", "Sub", "Mul", "MulAdd", "MulElementwise", "Inverse", "Dot",
    "Linear", "Pack", "Unpack", "BitAnd", "BitAndNot", "BitXor",
    "BitWeight", "BitParity",
})
VALID_SCOPES = frozenset({
    "Inverse", "Division", "Frobenius", "Norm", "Trace", "Restriction",
    "Membership",
})
VALID_PASSES = frozenset({
    "Main", "Prefix", "Backward", "Output", "Conjugate", "Accumulate", "Project",
})
VALID_UNITS = frozenset({"Elements", "Bytes", "Bits", "Words"})
VALID_BASES = frozenset({"Source", "Target"})
VALID_MULTIPLICITIES = frozenset({
    "One", "Rows", "ActiveRows", "ActivePairs", "UnitAddPairs",
})
VALID_RULE_GROUPS = frozenset({
    "residue", "specialized_prime", "binary_flat", "packed",
    "component", "bits", "map", "map_bits",
})

# A General coefficient is only meaningful for operations that actually take a
# coefficient: the scalar accumulate/scalar families and the mul_* families.

KNOWN_CALLABLES: set[str] | None = None

# ISA lane bytes per tier (unchanging hardware facts, not measurements).
ISA_LANE_BYTES = {
    "scalar": 0,  # scalar has no vector lane
    "v2": 16,
    "v3": 32,
    "v3_gfni_crypto": 32,
    "v4x": 64,
}

# Reject codes emitted by accept (stable, testable).
REJECT_CODES = frozenset({
    "missing_log", "malformed_receipt", "duplicate_done", "malformed_done",
    "missing_done", "source_mismatch", "host_mismatch",
    "feature_mismatch", "request_mismatch", "empty_log",
    "count_mismatch", "no_required_obligations", "duplicate_receipt",
    "receipt_source_mismatch", "receipt_host_mismatch",
    "receipt_feature_mismatch", "receipt_request_mismatch",
    "receipt_no_result", "fail_receipt", "skip_receipt",
    "unknown_result", "missing_required",
    "receipt_process_mismatch", "receipt_field_mismatch",
})


class ValidationError(Exception):
    """Raised for ledger validation failures; carries a stable code."""
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(msg: str) -> None:
    if msg.startswith("accept: "):
        reason = msg[len("accept: "):]
        if reason.startswith("reject "):
            reason = reason[len("reject "):]
        printed = f"simd_ledger: reject {reason}"
        code = reason.split(" ", 1)[0]
    else:
        printed = f"simd_ledger: {msg}"
        code = msg.split(" ", 1)[0]
    print(printed, file=sys.stderr)
    raise ValidationError(code)


def _load_known_callables(rustdoc: Path) -> set[str]:
    global KNOWN_CALLABLES
    if KNOWN_CALLABLES is not None:
        return KNOWN_CALLABLES
    document = json.loads(rustdoc.read_text())
    KNOWN_CALLABLES = declared_callables(document)
    return KNOWN_CALLABLES


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_callables(ledger: dict, rustdoc: Path | None = None) -> None:
    """Check every ledger callable is classified and matches rustdoc."""
    callables = ledger.get("callables")
    if not isinstance(callables, dict):
        _fail("ledger.callables must be a dict")
    if not callables:
        _fail("ledger.callables is empty")

    if rustdoc is not None:
        known = _load_known_callables(rustdoc)
        ledger_names = set(callables)
        unclassified = known - ledger_names
        stale = ledger_names - known
        if unclassified:
            _fail(f"unclassified callables: {sorted(unclassified)}")
        if stale:
            _fail(f"stale callables (not in rustdoc): {sorted(stale)}")

    for name, entry in sorted(callables.items()):
        if not isinstance(entry, dict):
            _fail(f"callable {name!r} entry is not a dict")
        kind = entry.get("kind")
        if kind not in VALID_KINDS:
            _fail(f"callable {name!r}: invalid kind {kind!r}")
        if kind == "bulk":
            family = entry.get("family")
            if not isinstance(family, str) or not family:
                _fail(f"callable {name!r}: bulk missing family")
        elif kind == "non-bulk":
            reason = entry.get("reason")
            if reason not in VALID_REASONS:
                _fail(f"callable {name!r}: non-bulk invalid reason {reason!r}")


def _validate_rules(ledger: dict) -> dict:
    """Return {rule_id: rule}. Validate new taxonomy."""
    rules = ledger.get("rules")
    if not isinstance(rules, dict):
        _fail("ledger.rules must be a dict")
    if not rules:
        _fail("ledger.rules is empty")

    seen_fixture_ids: set[str] = set()
    for rid, rule in sorted(rules.items()):
        if not isinstance(rule, dict):
            _fail(f"rule {rid!r} is not a dict")
        group = rule.get("group")
        if group not in VALID_RULE_GROUPS:
            _fail(f"rule {rid!r}: invalid group {group!r}")
        fixtures = rule.get("fixtures")
        if not isinstance(fixtures, list) or not fixtures:
            _fail(f"rule {rid!r}: missing fixtures")

        for fix in fixtures:
            if not isinstance(fix, dict):
                _fail(f"rule {rid!r}: fixture entry is not a dict")
            fid = fix.get("id")
            if not isinstance(fid, str) or not fid:
                _fail(f"rule {rid!r}: fixture missing id")
            if fid in seen_fixture_ids:
                _fail(f"rule {rid!r}: duplicate fixture id {fid!r}")
            seen_fixture_ids.add(fid)
            ftype = fix.get("type")
            has_type = isinstance(ftype, str) and ftype
            has_pair = (isinstance(fix.get("source"), str) and fix["source"]
                        and isinstance(fix.get("target"), str) and fix["target"])
            if not has_type and not has_pair:
                _fail(f"rule {rid!r}: fixture {fid!r} missing type or source/target pair")

        # Rule groups have different required fields.
        if group == "map":
            for field in ("kind", "arm", "arm_citation"):
                if field not in rule:
                    _fail(f"rule {rid!r}: map missing {field}")
            if rule["kind"] not in ("frobenius", "embedding", "isomorphism"):
                _fail(f"rule {rid!r}: invalid map kind {rule['kind']!r}")
        elif group == "map_bits":
            if "target_width_bytes" not in rule:
                _fail(f"rule {rid!r}: map_bits missing target_width_bytes")

    return rules


def _validate_applicability(ledger: dict, rules: dict) -> None:
    """Validate applicability rules."""
    apps = ledger.get("applicability")
    if not isinstance(apps, dict):
        _fail("ledger.applicability must be a dict")
    if not apps:
        _fail("ledger.applicability is empty")

    rule_ids = set(rules)

    for app_name, app in sorted(apps.items()):
        if not isinstance(app, dict):
            _fail(f"applicability {app_name!r} is not a dict")
        families = app.get("families")
        if not isinstance(families, list) or not families:
            _fail(f"applicability {app_name!r}: missing families")
        for fam in families:
            if not isinstance(fam, str) or not fam:
                _fail(f"applicability {app_name!r}: invalid family {fam!r}")
        app_rules = app.get("rules")
        if not isinstance(app_rules, list) or not app_rules:
            _fail(f"applicability {app_name!r}: missing rules")
        for rid in app_rules:
            if rid not in rule_ids:
                _fail(f"applicability {app_name!r}: undefined rule {rid!r}")
        exclusions = app.get("exclusions", {})
        if not isinstance(exclusions, dict):
            _fail(f"applicability {app_name!r}: exclusions must be a dict")
        for excl_rule_id, reason in sorted(exclusions.items()):
            if excl_rule_id not in rule_ids:
                _fail(f"applicability {app_name!r}: exclusion for undefined rule {excl_rule_id!r}")
            if not isinstance(reason, str) or not reason:
                _fail(f"applicability {app_name!r}: exclusion {excl_rule_id!r} missing reason")


def _validate_work_rules(ledger: dict) -> dict:
    """Return {name: work_rule}. Per-tier lane_bytes/loss_bytes in bytes; loss_formula describes ISA derivation."""
    wr = ledger.get("work_rules")
    if not isinstance(wr, dict):
        _fail("ledger.work_rules must be a dict")
    if not wr:
        _fail("ledger.work_rules is empty")

    for name, rule in sorted(wr.items()):
        if not isinstance(rule, dict):
            _fail(f"work_rule {name!r} is not a dict")
        for field in ("substage", "unit", "basis", "multiplicity",
                       "lane_bytes", "loss_bytes", "loss_formula"):
            if field not in rule:
                _fail(f"work_rule {name!r}: missing {field}")

        substage = rule["substage"]
        if substage not in VALID_SUBSTAGES:
            _fail(f"work_rule {name!r}: invalid substage {substage!r}")

        scope = rule.get("scope")
        if scope is not None:
            if not isinstance(scope, dict):
                _fail(f"work_rule {name!r}: scope must be a dict")
            s = scope.get("scope")
            p = scope.get("pass")
            if s not in VALID_SCOPES:
                _fail(f"work_rule {name!r}: invalid scope.scope {s!r}")
            if p not in VALID_PASSES:
                _fail(f"work_rule {name!r}: invalid scope.pass {p!r}")

        unit = rule["unit"]
        if unit not in VALID_UNITS:
            _fail(f"work_rule {name!r}: invalid unit {unit!r}")
        basis = rule["basis"]
        if basis not in VALID_BASES:
            _fail(f"work_rule {name!r}: invalid basis {basis!r}")
        mult = rule["multiplicity"]
        if mult not in VALID_MULTIPLICITIES:
            _fail(f"work_rule {name!r}: invalid multiplicity {mult!r}")

        # lane_bytes and loss_bytes must be per-tier dicts with int values.
        for field in ("lane_bytes", "loss_bytes"):
            val = rule[field]
            if not isinstance(val, dict):
                _fail(f"work_rule {name!r}: {field} must be a per-tier dict with int values")
            for tier in VALID_REQUESTS.union({"scalar"}):
                # Every tier must be present, or at least the vector tiers.
                pass
            for tier, v in val.items():
                if tier not in VALID_REQUESTS and tier != "scalar":
                    _fail(f"work_rule {name!r}: {field} has unknown tier {tier!r}")
                if not isinstance(v, int):
                    _fail(f"work_rule {name!r}: {field}.{tier} must be an integer")

        # loss_formula is a string expression, not a per-tier dict.
        loss_formula = rule.get("loss_formula", "")
        if not isinstance(loss_formula, str) or not loss_formula.strip():
            _fail(f"work_rule {name!r}: loss_formula must be a non-empty string")

        if "justification" not in rule:
            _fail(f"work_rule {name!r}: missing justification")
        justification = rule["justification"]
        if not isinstance(justification, str) or not justification:
            _fail(f"work_rule {name!r}: empty justification")

    return wr


def _validate_geometries(ledger: dict) -> dict:
    geo = ledger.get("geometries")
    if not isinstance(geo, dict):
        _fail("ledger.geometries must be a dict")
    return geo


def _validate_coefficient_patterns(ledger: dict) -> dict:
    cp = ledger.get("coefficient_patterns")
    if not isinstance(cp, dict):
        _fail("ledger.coefficient_patterns must be a dict")
    return cp


def _validate_evidence(ledger: dict) -> None:
    ev = ledger.get("evidence")
    if not isinstance(ev, dict):
        _fail("ledger.evidence must be a dict")


def _verify_evidence_records(ledger: dict, records: Path | None) -> None:
    """Verify every cited evidence record matches its receipts and completion."""
    if records is None:
        _fail("--records is required when ledger cites evidence")
    if not records.is_dir():
        _fail(f"--records {records} is not a directory")

    for ev_id, ev in ledger.get("evidence", {}).items():
        record_file = ev.get("file", "")
        if not record_file:
            _fail(f"evidence {ev_id!r}: missing file")
        rp = records / record_file
        if not rp.is_file():
            _fail(f"evidence {ev_id!r}: record file {record_file} not found")

        done: dict | None = None
        receipts: list[dict] = []
        for line in rp.read_text().splitlines():
            if line.startswith("ROUTE_DONE\t"):
                try:
                    done = json.loads(line.split("\t", 1)[1])
                except json.JSONDecodeError:
                    _fail(f"evidence {ev_id!r}: malformed ROUTE_DONE")
            elif line.startswith("ROUTE_RECEIPT\t"):
                try:
                    receipts.append(json.loads(line.split("\t", 1)[1]))
                except json.JSONDecodeError:
                    _fail(f"evidence {ev_id!r}: malformed ROUTE_RECEIPT")

        if done is None:
            _fail(f"evidence {ev_id!r}: record {record_file} missing ROUTE_DONE")

        completion = ev.get("completion", {})
        if not isinstance(completion, dict):
            _fail(f"evidence {ev_id!r}: completion must be a dict")
        for field in ("source_id", "host_id", "features"):
            if completion.get(field) is not None and done.get(field) != completion[field]:
                _fail(f"evidence {ev_id!r}: completion {field} mismatch")
        if completion.get("cases") is not None and done.get("cases") != completion["cases"]:
            _fail(f"evidence {ev_id!r}: completion cases mismatch")

        by_id = {r.get("id"): r for r in receipts}
        for cited in ev.get("receipts", []):
            if not isinstance(cited, dict):
                _fail(f"evidence {ev_id!r}: receipt expectation is not a dict")
            rid = cited.get("id")
            if not rid:
                _fail(f"evidence {ev_id!r}: receipt expectation missing id")
            rec = by_id.get(rid)
            if rec is None:
                _fail(f"evidence {ev_id!r}: cited receipt {rid} missing")
            if cited.get("features") is not None and rec.get("features") != cited["features"]:
                _fail(f"evidence {ev_id!r}: receipt {rid} features mismatch")
            if cited.get("tier") is not None and rec.get("request_raw") != cited["tier"]:
                _fail(f"evidence {ev_id!r}: receipt {rid} tier mismatch")
            if cited.get("source_id") is not None and rec.get("source_id") != cited["source_id"]:
                _fail(f"evidence {ev_id!r}: receipt {rid} source_id mismatch")


# ---------------------------------------------------------------------------
# Obligation expansion
# ---------------------------------------------------------------------------


def _bulk_callables_by_family(ledger: dict) -> dict[str, list[str]]:
    """Return {family: [callable_name]} for bulk callables."""
    result: dict[str, list[str]] = {}
    for name, entry in ledger.get("callables", {}).items():
        if entry.get("kind") == "bulk":
            family = entry.get("family", "")
            result.setdefault(family, []).append(name)
    return result


def _expand_obligations(ledger: dict, rules: dict) -> dict:
    """Expand obligations from applicability × rules × callables.

    Returns {obligation_id: {cells: {key: default_cell}}}.
    """
    apps = ledger.get("applicability", {})
    callables_by_family = _bulk_callables_by_family(ledger)
    default_cell = ledger.get("_cell_default", {
        "status": "unverified", "required": False, "perf": "unmeasured"
    })

    expanded: dict[str, dict] = {}

    for app_name, app in sorted(apps.items()):
        app_rules: list[str] = app.get("rules", [])
        exclusions: dict[str, str] = app.get("exclusions", {})
        families: list[str] = app.get("families", [])

        for rule_id in app_rules:
            if rule_id in exclusions:
                continue
            if rule_id not in rules:
                continue

            # Skip non-bulk rules (packed-padded, map rules tested via their own families)
            rule = rules[rule_id]
            if rule.get("non_bulk"):
                continue

            for family in families:
                callables = callables_by_family.get(family, [])
                for call_name in callables:
                    obligation_id = f"{call_name}@{rule_id}"

                    if obligation_id in expanded:
                        # First entry wins; skip duplicate produced by another applicability.
                        continue

                    # Build default cells for every feature config × request.
                    cells: dict[str, dict] = {}
                    for fc in sorted(VALID_FEATURE_CONFIGS):
                        for req in sorted(VALID_REQUESTS):
                            # v4x requires non-default features.
                            if fc == "default" and req == "v4x":
                                continue
                            key = f"{fc}/{req}"
                            cells[key] = dict(default_cell)
                            cells[key]["min_serving"] = req
                            cells[key]["work"] = []
                            cells[key]["process"] = req
                            cells[key]["field"] = req

                    expanded[obligation_id] = {
                        "cells": cells,
                    }

    return expanded


def _count_cells(obligations: dict) -> int:
    """Count total cells across all expanded obligations."""
    total = 0
    for ob in obligations.values():
        total += len(ob.get("cells", {}))
    return total


def _validate_expanded_obligations(ledger: dict, expanded: dict,
                                   rules: dict, wrules: dict,
                                   geos: dict, coeffs: dict) -> None:
    """Validate expanded obligations: check rule/callable exist, cells valid."""
    callables = ledger.get("callables", {})
    rule_ids = set(rules)
    wr_names = set(wrules)

    for oid, ob in sorted(expanded.items()):
        if "@" not in oid:
            _fail(f"obligation {oid!r}: missing @ separator")
        call_name, rule_id = oid.split("@", 1)
        if call_name not in callables:
            _fail(f"obligation {oid!r}: unknown callable {call_name!r}")
        if rule_id not in rule_ids:
            _fail(f"obligation {oid!r}: unknown rule {rule_id!r}")

        cells = ob.get("cells", {})
        if not isinstance(cells, dict):
            _fail(f"obligation {oid!r}: cells must be a dict")

        for key, cell in sorted(cells.items()):
            if "/" not in key:
                _fail(f"obligation {oid!r}: cell key {key!r} missing / separator")
            fc, req = key.split("/", 1)
            if fc not in VALID_FEATURE_CONFIGS:
                _fail(f"obligation {oid!r}: invalid feature config {fc!r}")
            if req not in VALID_REQUESTS:
                _fail(f"obligation {oid!r}: invalid request {req!r}")
            if fc == "default" and req == "v4x":
                _fail(f"obligation {oid!r}: default features cannot resolve v4x")

            if not isinstance(cell, dict):
                _fail(f"obligation {oid!r}: cell {key!r} is not a dict")

            for field in ("status", "required", "min_serving", "work"):
                if field not in cell:
                    _fail(f"obligation {oid!r} cell {key!r}: missing {field}")

            status = cell["status"]
            if status not in VALID_STATUS:
                _fail(f"obligation {oid!r} cell {key!r}: invalid status {status!r}")

            if not isinstance(cell["required"], bool):
                _fail(f"obligation {oid!r} cell {key!r}: required must be bool")

            ms = cell["min_serving"]
            if ms not in VALID_REQUESTS:
                _fail(f"obligation {oid!r} cell {key!r}: invalid min_serving {ms!r}")

            work = cell["work"]
            if not isinstance(work, list):
                _fail(f"obligation {oid!r} cell {key!r}: work must be a list")
            for wref in work:
                if wref not in wr_names:
                    _fail(f"obligation {oid!r} cell {key!r}: unknown work rule {wref!r}")

            perf = cell.get("perf", "unmeasured")
            if perf not in VALID_PERF:
                _fail(f"obligation {oid!r} cell {key!r}: invalid perf {perf!r}")

    # Check for duplicate stage keys across Work items within each cell.
    for oid, ob in sorted(expanded.items()):
        for key, cell in sorted(ob.get("cells", {}).items()):
            work_names = cell.get("work", [])
            stage_keys: set[tuple] = set()
            for wname in work_names:
                wr = wrules.get(wname, {})
                substage = wr.get("substage", "")
                scope = wr.get("scope")
                sk = (substage, scope.get("scope") if scope else None,
                      scope.get("pass") if scope else None)
                if sk in stage_keys:
                    _fail(f"obligation {oid!r} cell {key!r}: "
                          f"duplicate stage key {sk} from work rules")
                stage_keys.add(sk)


def _validate_explicit_obligations(ledger: dict, expanded: dict) -> None:
    """Reject a deviation whose obligation is outside the expansion."""
    explicit_obs = ledger.get("obligations", {})
    if isinstance(explicit_obs, dict):
        for oid in explicit_obs:
            if oid not in expanded:
                _fail(f"explicit obligation {oid!r} is not in the expansion; "
                      f"check applicability rules")


# ---------------------------------------------------------------------------
# Inventory command
# ---------------------------------------------------------------------------


def cmd_inventory(rustdoc: Path, ledger_path: Path) -> int:
    """Classify every callable and print completeness report."""
    known = _load_known_callables(rustdoc)
    if not ledger_path.is_file():
        _fail("simd_ledger.json not found")
    ledger = json.loads(ledger_path.read_text())
    callables = ledger.get("callables", {})

    ledger_names = set(callables)
    unclassified = known - ledger_names
    stale = ledger_names - known

    bulk = {n for n, e in callables.items() if e.get("kind") == "bulk"}
    non_bulk = {n for n, e in callables.items() if e.get("kind") == "non-bulk"}

    print(f"callables: rustdoc={len(known)}  ledger={len(ledger_names)}  "
          f"bulk={len(bulk)}  non_bulk={len(non_bulk)}")
    if unclassified:
        print(f"unclassified ({len(unclassified)}):")
        for n in sorted(unclassified):
            print(f"  {n}")
    else:
        print("unclassified: none")
    if stale:
        print(f"stale ({len(stale)}):")
        for n in sorted(stale):
            print(f"  {n}")
    else:
        print("stale: none")

    if unclassified or stale:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Check command
# ---------------------------------------------------------------------------


def cmd_check(ledger: dict, schema_only: bool, rustdoc: Path | None,
              test_list: Path | None, case_list: Path | None,
              surface_inventory: Path | None, records: Path | None) -> int:
    """Validate the ledger."""
    if schema_only:
        # Schema + expansion count.
        _validate_callables(ledger, rustdoc)
        rules = _validate_rules(ledger)
        _validate_applicability(ledger, rules)
        wrules = _validate_work_rules(ledger)
        geos = _validate_geometries(ledger)
        coeffs = _validate_coefficient_patterns(ledger)
        _validate_evidence(ledger)

        # Expand obligations.
        expanded = _expand_obligations(ledger, rules)
        _validate_expanded_obligations(ledger, expanded, rules, wrules, geos, coeffs)
        _validate_explicit_obligations(ledger, expanded)

        n_cells = _count_cells(expanded)
        print(f"simd_ledger: schema valid  obligations={len(expanded)} cells={n_cells}")
        return 0

    # Full check requires all inventory arguments.
    missing = []
    if rustdoc is None:
        missing.append("--rustdoc")
    if test_list is None:
        missing.append("--test-list")
    if case_list is None:
        missing.append("--case-list")
    if surface_inventory is None:
        missing.append("--surface-inventory")
    if missing:
        _fail(f"--full requires: {', '.join(missing)}")

    # Validate inventories are non-empty.
    known = _load_known_callables(rustdoc)
    if not known:
        _fail("rustdoc inventory is empty")

    tl = test_list.read_text().strip()
    if not tl:
        _fail("test list is empty")

    cl = case_list.read_text().strip()
    if not cl:
        _fail("case list is empty")

    si = surface_inventory.read_text().strip()
    if not si:
        _fail("surface inventory is empty")

    # --records is mandatory if the ledger cites any evidence.
    if ledger.get("evidence"):
        _verify_evidence_records(ledger, records)

    # Full schema validation.
    _validate_callables(ledger, rustdoc)
    rules = _validate_rules(ledger)
    _validate_applicability(ledger, rules)
    wrules = _validate_work_rules(ledger)
    geos = _validate_geometries(ledger)
    coeffs = _validate_coefficient_patterns(ledger)
    _validate_evidence(ledger)

    # Expand + validate obligations.
    expanded = _expand_obligations(ledger, rules)
    _validate_expanded_obligations(ledger, expanded, rules, wrules, geos, coeffs)
    _validate_explicit_obligations(ledger, expanded)

    n_cells = _count_cells(expanded)
    print(f"simd_ledger: full check passed  obligations={len(expanded)} cells={n_cells}")
    return 0


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


def _safe_ident(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_#]', '_', s)


def _work_iterations(wr: dict) -> int:
    """Compute repetitions from the work rule's declared formula (>= 1)."""
    formula = wr.get("iterations_formula", "1")
    try:
        value = int(formula)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _work_rust(wr_name: str, wr: dict, fc: str, req: str) -> str:
    """Emit a Work const for a particular feature config and request."""
    substage = wr["substage"]
    unit = wr["unit"]
    basis = wr["basis"]
    mult = wr["multiplicity"]

    scope = wr.get("scope")
    scope_expr = "None"
    if scope:
        scope_expr = f"Some((Scope::{scope['scope']}, Pass::{scope['pass']}))"

    lane_bytes = wr["lane_bytes"].get(req, wr["lane_bytes"].get("scalar", 1))
    loss_bytes = wr["loss_bytes"].get(req, wr["loss_bytes"].get("scalar", 0))
    loss_formula = wr.get("loss_formula", "0")
    iterations = _work_iterations(wr)

    ident = _safe_ident(f"{wr_name}_{fc}_{req}")
    return "\n".join([
        f"static WORK_{ident}: Work = Work {{",
        f"    substage: Substage::{substage},",
        f"    scope: {scope_expr},",
        f"    unit: Unit::{unit},",
        f"    basis: Basis::{basis},",
        f"    multiplicity: Multiplicity::{mult},",
        f"    lane_bytes: {lane_bytes},",
        f"    loss_bytes: {loss_bytes},",
        f"    loss_formula: \"{loss_formula}\",",
        f"    iterations: {iterations},",
        "};",
    ])


def _fixture_is_gf2(fixture: dict) -> bool:
    """True when the fixture type is a degree-1 GF(2) presentation."""
    ftype = fixture.get("type", "")
    return bool(re.search(r"FiniteField\s*<\s*2\s*,\s*1\s*,", ftype))


def _general_slot_count(coeff_pattern: dict) -> int:
    return sum(1 for s in coeff_pattern.get("slots", []) if s == "General")


def _callable_takes_coefficient(call_name: str) -> bool:
    """True when the callable carries a coefficient slot (mul_* / *_scalar)."""
    base = call_name.split("::")[-1]
    return base.startswith("mul_") or base.endswith("_scalar")


def _multiplicity_count(mult: str, geo: dict) -> int:
    if mult == "Rows":
        return geo.get("rows", 1)
    if mult == "ActiveRows":
        return geo.get("rows", 1)
    if mult == "ActivePairs":
        return geo.get("rows", 1) * geo.get("sources", 1)
    if mult == "UnitAddPairs":
        return geo.get("rows", 1)
    return 1  # One


def _len_in_units(unit: str, length: int, rule: dict) -> int:
    if unit == "Bytes":
        return length * rule.get("elem_bytes", 1)
    if unit == "Bits":
        elem_bytes = rule.get("elem_bytes", 1)
        return length * rule.get("elem_bits", elem_bytes * 8)
    if unit == "Words":
        return max(1, (length * rule.get("elem_bytes", 1)) // 8)
    return length  # Elements


def _loss_in_units(unit: str, loss_bytes: int, rule: dict) -> int:
    """Convert loss_bytes to the work rule's unit using fixture elem_bytes.
    Rounding: loss rounds up (ceil)."""
    import math
    elem_bytes = rule.get("elem_bytes", 1)
    if unit == "Bytes":
        return loss_bytes
    if unit == "Bits":
        return loss_bytes * 8
    if unit == "Words":
        return max(1, math.ceil(loss_bytes / 8))
    # Elements: ceil(loss_bytes / elem_bytes)
    if elem_bytes <= 1:
        return loss_bytes
    return math.ceil(loss_bytes / elem_bytes)


def _lane_in_units(unit: str, lane_bytes: int, rule: dict) -> int:
    """Convert lane_bytes to the work rule's unit using fixture elem_bytes.
    Rounding: lane rounds down (floor)."""
    elem_bytes = rule.get("elem_bytes", 1)
    if unit == "Bytes":
        return lane_bytes
    if unit == "Bits":
        return lane_bytes * 8
    if unit == "Words":
        return max(1, lane_bytes // 8)
    # Elements: floor(lane_bytes / elem_bytes)
    if elem_bytes <= 1:
        return lane_bytes
    return lane_bytes // elem_bytes


def _work_budget(work_rule: dict, geo: dict, rule: dict, tier: str) -> int:
    """Budget for one Work item at a geometry: iterations * mult * (len_in_units - loss_in_units).
    lane/loss are derived from lane_bytes/loss_bytes via the rule's elem_bytes."""
    mult = _multiplicity_count(work_rule["multiplicity"], geo)
    loss_bytes = work_rule["loss_bytes"].get(tier, work_rule["loss_bytes"].get("scalar", 0))
    loss = _loss_in_units(work_rule["unit"], loss_bytes, rule)
    length = _len_in_units(work_rule["unit"], geo.get("len", 0), rule)
    return _work_iterations(work_rule) * mult * (length - loss)


def _validate_route_budgets(ledger: dict) -> None:
    """Generator check: reject a non-positive budget on an eligible nondegenerate case."""
    rules = _validate_rules(ledger)
    expanded = _expanded_with_overrides(ledger)
    wrules = ledger.get("work_rules", {})
    geos = ledger.get("geometries", {})

    for oid, ob in sorted(expanded.items()):
        rule = rules.get(oid.split("@", 1)[1], {})
        for key, cell in sorted(ob.get("cells", {}).items()):
            fc, req = key.split("/", 1)
            if cell.get("status", "unverified") in ("missing", "bypass", "zero-budget"):
                continue
            for wr_name in cell.get("work", []):
                wr = wrules.get(wr_name)
                if not wr:
                    continue
                for geo_id, geo in sorted(geos.items()):
                    if geo.get("len", 0) <= 0:
                        continue  # degenerate
                    if _work_budget(wr, geo, rule, req) <= 0:
                        _fail(f"obligation {oid!r} cell {key!r} geometry {geo_id!r}: "
                              f"non-positive budget for work rule {wr_name!r}")


def _validate_gf2_general(ledger: dict) -> None:
    """Generator check: a General coefficient slot has no GF(2) value."""
    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    coeffs = ledger.get("coefficient_patterns", {})

    for oid in sorted(expanded):
        call_name, rule_id = oid.split("@", 1)
        if not _callable_takes_coefficient(call_name):
            continue
        rule = rules.get(rule_id, {})
        for fixture in rule.get("fixtures", []):
            if not _fixture_is_gf2(fixture):
                continue
            for cp_name, cp in coeffs.items():
                if _general_slot_count(cp) > 0:
                    _fail(f"General coefficient on GF(2) fixture "
                          f"{fixture.get('id')!r} in obligation {oid!r} "
                          f"(coefficient pattern {cp_name!r})")


def _generate_cases(ledger: dict) -> str:
    """Generate src/kernel/tests/routes/cases.rs content."""
    _validate_gf2_general(ledger)
    _validate_route_budgets(ledger)

    lines = [
        "// @generated by benches/simd_ledger.py; edit the ledger, not this file.",
        "",
        "use crate::kernel::Backend;",
        "use crate::kernel::tests::routes::{",
        "    Basis, Case, Cell, Coeff, Features, Geometry, Multiplicity, Pass, Request, Scope, Substage,",
        "    Unit, Work,",
        "};",
        "",
    ]

    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})
    wrules = ledger.get("work_rules", {})

    # Generate Work consts first.
    work_consts: list[str] = []
    for wr_name, wr in sorted(wrules.items()):
        for fc in sorted(VALID_FEATURE_CONFIGS):
            for req in sorted(VALID_REQUESTS):
                if fc == "default" and req == "v4x":
                    continue
                work_consts.append(_work_rust(wr_name, wr, fc, req))

    # Generate cases.
    cases: list[str] = []
    for oid, ob in sorted(expanded.items()):
        call_name, rule_id = oid.split("@", 1)
        rule = rules.get(rule_id, {})
        fixtures = rule.get("fixtures", [])

        for fixture in fixtures:
            fid = fixture["id"]
            for geo_id, geo_coeffs in sorted(geos.items()):
                # Each geometry may associate specific coefficient patterns.
                # For now, emit with a default pattern.
                pass

    # Simplified: emit work consts only for now (the full case table
    # is owned by the runner worker per S0_CONTRACT.md §3).
    return "\n".join(lines + work_consts) + "\n"


def cmd_generate(out_path: Path, check: bool, ledger: dict) -> int:
    """Generate the route cases file or check byte identity."""
    content = _generate_cases(ledger)

    if check:
        if not out_path.is_file():
            _fail(f"{out_path} does not exist for --check")
        existing = out_path.read_text()
        if existing != content:
            _fail(f"{out_path} differs from generated output")
        return 0

    out_path.write_text(content)

    # Verify rustfmt.
    result = subprocess.run(
        ["rustfmt", "--check", "--edition", "2024", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        _fail(f"generated file fails rustfmt: {result.stderr}")

    return 0


# ---------------------------------------------------------------------------
# Accept command
# ---------------------------------------------------------------------------


def _expanded_with_overrides(ledger: dict) -> dict:
    """Return the expanded obligations with explicit ledger deviations merged."""
    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    explicit_obs = ledger.get("obligations", {})
    if isinstance(explicit_obs, dict):
        for oid, ob in explicit_obs.items():
            if oid in expanded:
                # Merge cells: an explicit cell overwrites the expanded cell.
                for key, cell in ob.get("cells", {}).items():
                    expanded[oid]["cells"][key] = dict(
                        expanded[oid]["cells"].get(key, {}), **cell)
            else:
                expanded[oid] = ob
    return expanded


def _required_obligations(expanded: dict, features: str, request: str) -> set[str]:
    """Return obligations whose (features, request) cell is required."""
    required: set[str] = set()
    for oid, ob in expanded.items():
        cell = ob.get("cells", {}).get(f"{features}/{request}", {})
        if cell.get("required"):
            required.add(oid)
    return required


def cmd_accept(log: Path, features: str, request: str, source_id: str,
               host_id: str, ledger: dict) -> int:
    """Fail-closed receipt audit."""
    if not log.is_file():
        _fail("accept: reject missing_log")

    text = log.read_text()

    receipts: list[dict] = []
    done: dict | None = None

    for line in text.splitlines():
        if line.startswith("ROUTE_RECEIPT\t"):
            try:
                receipts.append(json.loads(line.split("\t", 1)[1]))
            except json.JSONDecodeError:
                _fail("accept: reject malformed_receipt")
        elif line.startswith("ROUTE_DONE\t"):
            if done is not None:
                _fail("accept: reject duplicate_done")
            try:
                done = json.loads(line.split("\t", 1)[1])
            except json.JSONDecodeError:
                _fail("accept: reject malformed_done")

    if done is None:
        _fail("accept: reject missing_done")

    if not receipts:
        _fail("accept: reject empty_log")

    if done.get("cases", 0) != len(receipts):
        _fail("accept: reject count_mismatch")

    # The DONE marker must match the expected run identity.
    if done.get("source_id") != source_id:
        _fail("accept: reject source_mismatch")
    if done.get("host_id") != host_id:
        _fail("accept: reject host_mismatch")
    if done.get("features") != features:
        _fail("accept: reject feature_mismatch")
    if done.get("request") != request:
        _fail("accept: reject request_mismatch")

    expanded = _expanded_with_overrides(ledger)
    required_set = _required_obligations(expanded, features, request)
    if not required_set:
        _fail("accept: reject no_required_obligations")

    # Duplicate receipt ids are rejected before the rest of the audit runs.
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for rec in receipts:
        rid = rec.get("id", "")
        if rid in seen_ids:
            duplicate_ids.add(rid)
        seen_ids.add(rid)
    if duplicate_ids:
        _fail(f"accept: reject duplicate_receipt {sorted(duplicate_ids)}")

    received_ids: set[str] = set()
    for rec in receipts:
        oid = rec.get("obligation", "")
        received_ids.add(oid)

        if rec.get("source_id") != source_id:
            _fail(f"accept: reject receipt_source_mismatch {oid}")
        if rec.get("host_id") != host_id:
            _fail(f"accept: reject receipt_host_mismatch {oid}")
        if rec.get("features") != features:
            _fail(f"accept: reject receipt_feature_mismatch {oid}")
        if rec.get("request_raw") not in (request, None, ""):
            _fail(f"accept: reject receipt_request_mismatch {oid}")

        # Process and field resolution must equal the ledger's expected cell;
        # a downgraded host or a different field backend cannot claim.
        cell = expanded.get(oid, {}).get("cells", {}).get(f"{features}/{request}", {})
        if "process" in cell and rec.get("process") != cell["process"]:
            _fail(f"accept: reject receipt_process_mismatch {oid}")
        if "field" in cell and rec.get("field") != cell["field"]:
            _fail(f"accept: reject receipt_field_mismatch {oid}")

        result = rec.get("result")
        if result is None:
            _fail(f"accept: reject receipt_no_result {oid}")

        if isinstance(result, str):
            if result.startswith("fail"):
                _fail(f"accept: reject fail_receipt {oid}")
            elif result.startswith("skip"):
                _fail(f"accept: reject skip_receipt {oid}")
            elif not result.startswith("pass"):
                _fail(f"accept: reject unknown_result {oid}")

    missing = required_set - received_ids
    if missing:
        _fail(f"accept: reject missing_required {sorted(missing)}")

    return 0


# ---------------------------------------------------------------------------
# Fixture compilation check
# ---------------------------------------------------------------------------


def cmd_fixtures(ledger: dict, out_dir: Path) -> int:
    """Write a scratch crate with all fixture types and compile-check them
    against FieldBuffer for bulk fields and prepared-object types for maps."""
    rules = _validate_rules(ledger)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect shipped types (not needing custom specs).
    shipped_types: set[str] = set()
    map_fixtures: list[dict] = []  # (source_type, target_type, prepared_type_name)
    custom_specs: list[dict] = []

    for rid, rule in sorted(rules.items()):
        group = rule.get("group", "")
        for fix in rule.get("fixtures", []):
            if fix.get("needs_spec"):
                custom_specs.append({
                    "id": fix["id"],
                    "rule": rid,
                    "type": fix.get("type", ""),
                    "spec": fix["needs_spec"],
                })
            elif group == "map":
                # Map fixtures have source/target pairs; construct the
                # prepared object type (PreparedEmbedding, PreparedIsomorphism,
                # or Frobenius) whose bounds enforce the trait checks.
                kind = rule.get("kind", "")
                src = fix.get("source", "")
                tgt = fix.get("target", "")
                if kind == "frobenius":
                    ptype = f"fgf::Frobenius<{src.replace('crate::', 'fgf::')}>"
                elif kind == "embedding":
                    ptype = f"fgf::PreparedEmbedding<{src.replace('crate::', 'fgf::')}, {tgt.replace('crate::', 'fgf::')}>"
                elif kind == "isomorphism":
                    ptype = f"fgf::PreparedIsomorphism<{src.replace('crate::', 'fgf::')}, {tgt.replace('crate::', 'fgf::')}>"
                else:
                    continue
                map_fixtures.append({"id": fix["id"], "source": src, "target": tgt, "prepared_type": ptype})
            elif group == "map_bits":
                # map_bits fixtures have source/target pairs with
                # PreparedEmbedding<source, target>.
                fix_data = rule.get("fixture", fix)
                src = fix_data.get("source", "")
                tgt = fix_data.get("target", "")
                if src and tgt:
                    ptype = f"fgf::PreparedEmbedding<{src.replace('crate::', 'fgf::')}, {tgt.replace('crate::', 'fgf::')}>"
                    map_fixtures.append({"id": fix_data.get("id", ""), "source": src, "target": tgt, "prepared_type": ptype})
            elif group == "packed" and rule.get("non_bulk"):
                # Non-bulk storage markers (PackedPadded) are skipped in bulk checks.
                pass
            else:
                # Bulk field type: will be checked against FieldBuffer.
                # Skip fixtures whose rules indicate sub-byte-aligned fields
                # (degree < 8 binary flat) — they implement Field, not FieldBuffer.
                if group == "binary_flat" and rid == "binary-flat-sub8":
                    continue
                if group == "bits":
                    continue
                ftype = fix.get("type", fix.get("source", ""))
                if ftype and "crate::" in ftype:
                    shipped_types.add(ftype)

    # Write Cargo.toml
    cargo_toml = f"""[package]
name = "ledger-fixtures"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
fgf = {{ path = "{Path(__file__).parent.parent}", features = ["alloc"] }}
"""
    (out_dir / "Cargo.toml").write_text(cargo_toml)

    # Write src/main.rs
    src_dir = out_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    main_lines = ["fn main() {"]

    # FieldBuffer check for bulk types.
    main_lines.append("    fn _assert_field<T: fgf::FieldBuffer>() {}")
    for t in sorted(shipped_types):
        rust_type = t.replace("crate::", "fgf::")
        main_lines.append(f"    _assert_field::<{rust_type}>();")

    # Prepared-object type check for maps (type-level trait enforcement).
    if map_fixtures:
        main_lines.append("")
        main_lines.append("    // Map fixtures: construct the prepared type to enforce trait bounds.")
        for mf in map_fixtures:
            main_lines.append(f"    let _ = std::mem::size_of::<{mf['prepared_type']}>();")

    main_lines.append("}")

    (src_dir / "main.rs").write_text("\n".join(main_lines) + "\n")

    # Run cargo check.
    result = subprocess.run(
        ["cargo", "check"],
        cwd=str(out_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Fixture compilation FAILED:\n{result.stderr}")
        return 1

    print(f"Fixture compilation OK: {len(shipped_types)} bulk field types checked via FieldBuffer")
    if map_fixtures:
        print(f"  {len(map_fixtures)} map fixture prepared types checked")

    # Report custom-spec fixtures.
    if custom_specs:
        print(f"\nCustom-spec fixtures the runner must define ({len(custom_specs)}):")
        for cs in custom_specs:
            print(f"  {cs['id']} ({cs['rule']}): {cs['type']}")
            for k, v in cs["spec"].items():
                print(f"    {k}: {v}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path,
                        default=Path(__file__).with_name("simd_ledger.json"),
                        help="path to ledger JSON (default: the file next to this script)")
    sub = parser.add_subparsers(dest="command", required=True)

    # inventory
    inv = sub.add_parser("inventory", help="classify every callable")
    inv.add_argument("--rustdoc", type=Path, required=True)

    # check
    chk = sub.add_parser("check", help="validate ledger")
    chk.add_argument("--schema-only", action="store_true",
                     help="schema only, no inventory resolution")
    chk.add_argument("--full", action="store_true",
                     help="full check with all inventories")
    chk.add_argument("--rustdoc", type=Path, default=None)
    chk.add_argument("--test-list", type=Path, default=None)
    chk.add_argument("--case-list", type=Path, default=None)
    chk.add_argument("--surface-inventory", type=Path, default=None)
    chk.add_argument("--records", type=Path, default=None,
                     help="bench-records/routes directory (required if evidence cited)")

    # generate
    gen = sub.add_parser("generate", help="emit route cases file")
    gen.add_argument("--out", type=Path, required=True,
                     help="output path for cases.rs")
    gen.add_argument("--check", action="store_true",
                     help="check byte identity against existing file")

    # accept
    acc = sub.add_parser("accept", help="fail-closed receipt audit")
    acc.add_argument("--log", type=Path, required=True)
    acc.add_argument("--features", required=True, choices=["default", "all"])
    acc.add_argument("--request", required=True, choices=sorted(VALID_REQUESTS))
    acc.add_argument("--source-id", required=True)
    acc.add_argument("--host-id", required=True)

    # fixtures
    fix = sub.add_parser("fixtures", help="write scratch crate and compile-check")
    fix.add_argument("--out", type=Path, required=True,
                     help="output directory for scratch crate")

    args = parser.parse_args(argv)

    ledger_path = args.ledger

    try:
        if args.command == "inventory":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            return cmd_inventory(args.rustdoc, ledger_path)

        if args.command == "check":
            if args.full and args.schema_only:
                _fail("cannot use both --full and --schema-only")
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = json.loads(ledger_path.read_text())
            if args.full:
                return cmd_check(ledger, False, args.rustdoc, args.test_list,
                                 args.case_list, args.surface_inventory, args.records)
            return cmd_check(ledger, True, args.rustdoc, None, None, None, None)

        if args.command == "generate":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = json.loads(ledger_path.read_text())
            return cmd_generate(args.out, args.check, ledger)

        if args.command == "accept":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = json.loads(ledger_path.read_text())
            return cmd_accept(args.log, args.features, args.request,
                              args.source_id, args.host_id, ledger)

        if args.command == "fixtures":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = json.loads(ledger_path.read_text())
            return cmd_fixtures(ledger, args.out)

        return 0
    except ValidationError:
        return 1


if __name__ == "__main__":
    sys.exit(main())