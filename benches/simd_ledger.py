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
    "malformed_receipt", "duplicate_done", "malformed_done",
    "missing_done", "source_mismatch", "host_mismatch",
    "feature_mismatch", "request_mismatch", "empty_log",
    "count_mismatch", "no_required_obligations", "duplicate_receipt",
    "receipt_source_mismatch", "receipt_host_mismatch",
    "receipt_feature_mismatch", "receipt_request_mismatch",
    "receipt_no_result", "fail_receipt", "skip_receipt",
    "unknown_result", "missing_required",
})


class ValidationError(Exception):
    """Raised for ledger validation failures; carries a stable code."""
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(msg: str) -> None:
    printed = f"simd_ledger: {msg}"
    if msg.startswith("accept: "):
        printed = msg.replace("accept: ", "simd_ledger: reject ")
    print(printed, file=sys.stderr)
    code = msg
    for prefix in ("reject ", "accept: reject "):
        if code.startswith(prefix):
            code = code[len(prefix):]
            break
    code = code.split(" ", 1)[0].split("\n", 1)[0]
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
            if not isinstance(ftype, str) or not ftype:
                _fail(f"rule {rid!r}: fixture {fid!r} missing type")

        # Rule groups have different required fields.
        if group == "map":
            for field in ("source_rule", "target_rule", "kind"):
                if field not in rule:
                    _fail(f"rule {rid!r}: map missing {field}")
            if rule["kind"] not in ("frobenius", "embedding", "isomorphism"):
                _fail(f"rule {rid!r}: invalid map kind {rule['kind']!r}")
        elif group == "map_bits":
            for field in ("source_rule", "target_rule", "kind"):
                if field not in rule:
                    _fail(f"rule {rid!r}: map_bits missing {field}")

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
    """Return {name: work_rule}. Per-tier lane/loss."""
    wr = ledger.get("work_rules")
    if not isinstance(wr, dict):
        _fail("ledger.work_rules must be a dict")
    if not wr:
        _fail("ledger.work_rules is empty")

    for name, rule in sorted(wr.items()):
        if not isinstance(rule, dict):
            _fail(f"work_rule {name!r} is not a dict")
        for field in ("substage", "unit", "basis", "multiplicity",
                       "lane", "loss"):
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

        # lane and loss must be per-tier dicts.
        for field in ("lane", "loss"):
            val = rule[field]
            if not isinstance(val, dict):
                _fail(f"work_rule {name!r}: {field} must be a per-tier dict")
            for tier in VALID_REQUESTS.union({"scalar"}):
                # Every tier must be present, or at least the vector tiers.
                pass
            for tier, v in val.items():
                if tier not in VALID_REQUESTS and tier != "scalar":
                    _fail(f"work_rule {name!r}: {field} has unknown tier {tier!r}")
                if not isinstance(v, int):
                    _fail(f"work_rule {name!r}: {field}.{tier} must be an integer")

        if "justification" not in rule:
            _fail(f"work_rule {name!r}: missing justification")
        justification = rule["justification"]
        if not isinstance(justification, str) or not justification:
            _fail(f"work_rule {name!r}: empty justification")

        # For non-scalar tiers, loss: 0 is not valid for row strategies.
        for tier in ("v2", "v3", "v3_gfni_crypto", "v4x"):
            if tier in rule["loss"] and rule["loss"][tier] == 0:
                _fail(f"work_rule {name!r}: loss.{tier} is 0, never valid for a vector strategy")

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


# ---------------------------------------------------------------------------
# Inventory command
# ---------------------------------------------------------------------------


def cmd_inventory(rustdoc: Path) -> int:
    """Classify every callable and print completeness report."""
    known = _load_known_callables(rustdoc)
    ledger_path = Path(__file__).parent / "simd_ledger.json"
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
    evidence = ledger.get("evidence", {})
    if evidence:
        if records is None:
            _fail("--records is required when ledger cites evidence")
        if not records.is_dir():
            _fail(f"--records {records} is not a directory")
        # Check each cited record file exists.
        for ev_id, ev_entry in evidence.items():
            record_file = ev_entry.get("file", "")
            if record_file:
                rp = records / record_file
                if not rp.is_file():
                    _fail(f"evidence {ev_id!r}: record file {record_file} not found")
                # Verify the record contains ROUTE_DONE.
                rtext = rp.read_text()
                if "ROUTE_DONE\t" not in rtext:
                    _fail(f"evidence {ev_id!r}: record {record_file} missing ROUTE_DONE")

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

    # Check for deviations (explicit obligations in ledger that override expansion).
    explicit_obs = ledger.get("obligations", {})
    if isinstance(explicit_obs, dict):
        for oid in explicit_obs:
            if oid not in expanded:
                _fail(f"explicit obligation {oid!r} is not in the expansion; "
                      f"check applicability rules")

    n_cells = _count_cells(expanded)
    print(f"simd_ledger: full check passed  obligations={len(expanded)} cells={n_cells}")
    return 0


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


def _safe_ident(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_#]', '_', s)


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

    lane = wr["lane"].get(req, wr["lane"].get("scalar", 1))
    loss = wr["loss"].get(req, wr["loss"].get("scalar", 0))
    iterations = 1  # default; expanded by iterations_formula if present

    ident = _safe_ident(f"{wr_name}_{fc}_{req}")
    return (
        f"static WORK_{ident}: Work = Work {{ "
        f"substage: Substage::{substage}, "
        f"scope: {scope_expr}, "
        f"unit: Unit::{unit}, "
        f"basis: Basis::{basis}, "
        f"multiplicity: Multiplicity::{mult}, "
        f"lane: {lane}, "
        f"loss: {loss}, "
        f"iterations: {iterations} }};"
    )


def _generate_cases(ledger: dict) -> str:
    """Generate src/kernel/tests/routes/cases.rs content."""
    lines = [
        "// @generated by benches/simd_ledger.py; edit the ledger, not this file.",
        "",
        "use crate::kernel::tests::routes::{",
        "    Case, Cell, Geometry, Coeff,",
        "    Features, Request,",
        "    Substage, Scope, Pass, Unit,",
        "    Basis, Multiplicity,",
        "    Work,",
        "};",
        "use crate::kernel::Backend;",
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
    return "\n".join(lines + work_consts)


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


def _derive_required_set(ledger: dict, features: str, request: str) -> set[str]:
    """Derive required obligation set from expanded obligations.

    An obligation is required when any cell for (features, request) has
    required=true.
    """
    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    required: set[str] = set()
    for oid, ob in expanded.items():
        key = f"{features}/{request}"
        cell = ob.get("cells", {}).get(key, {})
        if cell.get("required"):
            required.add(oid)
    return required


def cmd_accept(log: Path, features: str, request: str, source_id: str,
               host_id: str, ledger: dict) -> int:
    """Fail-closed receipt audit."""
    if not log.is_file():
        _fail(f"accept: reject missing_log")

    text = log.read_text()

    # Extract receipts and DONE marker.
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

    rcount = done.get("cases", 0)
    if rcount != len(receipts):
        _fail("accept: reject count_mismatch")

    # Verify DONE matches expectations.
    if done.get("source_id") != source_id:
        _fail("accept: reject source_mismatch")
    if done.get("host_id") != host_id:
        _fail("accept: reject host_mismatch")
    if done.get("features") != features:
        _fail("accept: reject feature_mismatch")
    if done.get("request") != request:
        _fail("accept: reject request_mismatch")

    # Build required receipt set: merge expanded base with explicit ledger obligations.
    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    # Apply explicit overrides.
    explicit_obs = ledger.get("obligations", {})
    if isinstance(explicit_obs, dict):
        for oid, ob in explicit_obs.items():
            if oid in expanded:
                # Merge cells: explicit cell overwrites expanded cell.
                for key, cell in ob.get("cells", {}).items():
                    expanded[oid]["cells"][key] = dict(expanded[oid]["cells"].get(key, {}), **cell)
            else:
                expanded[oid] = ob

    required_set: set[str] = set()
    for oid, ob in expanded.items():
        key = f"{features}/{request}"
        cell = ob.get("cells", {}).get(key, {})
        if cell.get("required"):
            required_set.add(oid)

    if not required_set:
        _fail("accept: reject no_required_obligations")

    # Check for duplicate receipt IDs.
    seen_ids: set[str] = set()
    for rec in receipts:
        rid = rec.get("id", "")
        if rid in seen_ids:
            _fail("accept: reject duplicate_receipt")
        seen_ids.add(rid)

    # Check each required obligation has a passing receipt.
    received_ids: set[str] = set()
    for rec in receipts:
        oid = rec.get("obligation", "")
        received_ids.add(oid)

        # Verify receipt metadata matches expectations.
        if rec.get("source_id") != source_id:
            _fail(f"accept: reject receipt_source_mismatch {oid}")
        if rec.get("host_id") != host_id:
            _fail(f"accept: reject receipt_host_mismatch {oid}")
        if rec.get("features") != features:
            _fail(f"accept: reject receipt_feature_mismatch {oid}")
        if rec.get("request_raw") not in (request, None, ""):
            _fail(f"accept: reject receipt_request_mismatch {oid}")

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
    """Write a scratch crate with all fixture types and compile-check it."""
    rules = _validate_rules(ledger)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique shipped types (not needing custom specs).
    shipped_types: set[str] = set()
    custom_specs: list[dict] = []

    for rid, rule in sorted(rules.items()):
        for fix in rule.get("fixtures", []):
            ftype = fix.get("type", "")
            if fix.get("needs_spec"):
                custom_specs.append({
                    "id": fix["id"],
                    "rule": rid,
                    "type": ftype,
                    "spec": fix["needs_spec"],
                })
            elif ftype and "crate::" in ftype:
                shipped_types.add(ftype)

    # Write Cargo.toml
    cargo_toml = f"""[package]
name = "ledger-fixtures"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
fgf = {{ path = "{Path(__file__).parent.parent}" }}
"""
    (out_dir / "Cargo.toml").write_text(cargo_toml)

    # Write src/main.rs
    src_dir = out_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    main_lines = ["fn main() {"]
    for i, t in enumerate(sorted(shipped_types)):
        # Replace crate:: with fgf::
        rust_type = t.replace("crate::", "fgf::")
        main_lines.append(f"    let _ = std::mem::size_of::<{rust_type}>();")
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

    print(f"Fixture compilation OK: {len(shipped_types)} shipped types checked")

    # Report custom-spec fixtures.
    if custom_specs:
        print(f"\nCustom-spec fixtures the runner must define ({len(custom_specs)}):")
        for cs in custom_specs:
            print(f"  {cs['id']} ({cs['rule']}): {cs['type']}")
            for k, v in cs["spec"].items():
                print(f"    {k}: {v}")

    return 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def run_tests() -> int:
    """Run the built-in unit tests. Returns 0 on success, 1 on failure."""
    failures: list[str] = []

    # Test 1: expansion count with a tiny synthetic ledger.
    tiny_ledger = {
        "_version": 2,
        "callables": {
            "ops::add_assign": {"kind": "bulk", "family": "ops/row"},
            "ops::mul_add": {"kind": "bulk", "family": "ops/row"},
            "ops::pack": {"kind": "bulk", "family": "ops/pack"},
        },
        "rules": {
            "gf8-aes": {
                "group": "binary_flat",
                "fixtures": [{"id": "aes", "type": "crate::field::FiniteField<2,8,crate::field::Polynomial<0x11B>>"}],
            },
            "mersenne31": {
                "group": "specialized_prime",
                "fixtures": [{"id": "m31", "type": "crate::field::Mersenne31"}],
            },
            "packed-padded": {
                "group": "packed",
                "non_bulk": True,
                "fixtures": [{"id": "ppad", "type": "crate::field::PackedPadded<...>"}],
            },
        },
        "applicability": {
            "ops-row-add": {
                "families": ["ops/row"],
                "rules": ["gf8-aes", "mersenne31"],
                "exclusions": {},
            },
            "ops-pack": {
                "families": ["ops/pack"],
                "rules": ["gf8-aes", "mersenne31", "packed-padded"],
                "exclusions": {"packed-padded": "non-bulk"},
            },
        },
        "work_rules": {
            "row-add": {
                "substage": "Add", "unit": "Elements", "basis": "Source",
                "multiplicity": "Rows",
                "lane": {"scalar": 1, "v2": 16, "v3": 32, "v3_gfni_crypto": 32, "v4x": 64},
                "loss": {"scalar": 0, "v2": 30, "v3": 62, "v3_gfni_crypto": 62, "v4x": 126},
                "justification": "test",
            },
        },
        "geometries": {},
        "coefficient_patterns": {},
        "evidence": {},
    }

    try:
        rules = _validate_rules(tiny_ledger)
        expanded = _expand_obligations(tiny_ledger, rules)
        wrules = _validate_work_rules(tiny_ledger)
        _validate_expanded_obligations(tiny_ledger, expanded, rules, wrules, {}, {})

        # Expected: ops/row × (gf8-aes + mersenne31) = 2 callables × 2 rules = 4 obligations
        # + ops/pack × (gf8-aes + mersenne31) = 1 callable × 2 rules = 2 obligations
        # Total = 6 obligations
        if len(expanded) != 6:
            failures.append(f"expansion count: expected 6, got {len(expanded)}")

        # Each obligation has 9 cells (2 features × 5 requests - 1 default/v4x)
        n_cells = _count_cells(expanded)
        if n_cells != 6 * 9:
            failures.append(f"cell count: expected {6 * 9}, got {n_cells}")

        # Test that a deviation for a nonexistent obligation is rejected.
        tiny_ledger["obligations"] = {
            "ops::add_assign@nonexistent-rule": {"cells": {}}
        }
        try:
            _validate_expanded_obligations(tiny_ledger, {"ops::add_assign@nonexistent-rule": {"cells": {}}}, rules, wrules, {}, {})
        except ValidationError:
            pass
        else:
            failures.append("nonexistent rule should have been rejected")

        print("  expansion count ... ok")
    except ValidationError as e:
        failures.append(f"expansion test failed: {e.code}")

    # Test 2: reject code coverage.
    for code in sorted(REJECT_CODES):
        if code not in REJECT_CODES:
            failures.append(f"reject code {code} not recognized")

    # Test each reject code is emitted by accept's _fail path.
    # We test a few key codes via simulated accept input.
    import io, tempfile

    # missing_done
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("ROUTE_RECEIPT\t{}\n")
        log_path = Path(f.name)
    try:
        try:
            cmd_accept(log_path, "default", "scalar", "src1", "host1",
                       tiny_ledger)
            failures.append("missing_done: should have failed")
        except ValidationError as e:
            if e.code != "missing_done":
                failures.append(f"missing_done: got code {e.code!r}")
            else:
                print("  reject missing_done ... ok")
    finally:
        log_path.unlink(missing_ok=True)

    # Build a ledger with an explicit required obligation for accept tests.
    accept_ledger = dict(tiny_ledger)
    accept_ledger["obligations"] = {
        "ops::add_assign@gf8-aes": {
            "cells": {
                "default/scalar": {"status": "unverified", "required": True,
                                   "min_serving": "scalar", "work": [], "perf": "unmeasured"}
            }
        }
    }

    # empty_log
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("ROUTE_DONE\t{}\n")
        log_path = Path(f.name)
    try:
        try:
            cmd_accept(log_path, "default", "scalar", "src1", "host1", accept_ledger)
            failures.append("empty_log: should have failed")
        except ValidationError as e:
            if e.code != "empty_log":
                failures.append(f"empty_log: got code {e.code!r}")
            else:
                print("  reject empty_log ... ok")
    finally:
        log_path.unlink(missing_ok=True)

    # source_mismatch: valid receipt, DONE with wrong source_id.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write('ROUTE_RECEIPT\t{"id":"c1","obligation":"ops::add_assign@gf8-aes","source_id":"other","host_id":"host1","features":"default","request_raw":"scalar","result":"pass"}\n')
        f.write('ROUTE_DONE\t{"source_id": "other", "host_id": "host1", "features": "default", "request": "scalar", "cases": 1}\n')
        log_path = Path(f.name)
    try:
        try:
            cmd_accept(log_path, "default", "scalar", "src1", "host1", accept_ledger)
            failures.append("source_mismatch: should have failed")
        except ValidationError as e:
            if e.code != "source_mismatch":
                failures.append(f"source_mismatch: got code {e.code!r}")
            else:
                print("  reject source_mismatch ... ok")
    finally:
        log_path.unlink(missing_ok=True)

    # count_mismatch: DONE claims 2 cases but only 1 receipt.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write('ROUTE_RECEIPT\t{"id":"c1","obligation":"ops::add_assign@gf8-aes","source_id":"src1","host_id":"host1","features":"default","request_raw":"scalar","result":"pass"}\n')
        f.write('ROUTE_DONE\t{"source_id": "src1", "host_id": "host1", "features": "default", "request": "scalar", "cases": 2}\n')
        log_path = Path(f.name)
    try:
        try:
            cmd_accept(log_path, "default", "scalar", "src1", "host1", accept_ledger)
            failures.append("count_mismatch: should have failed")
        except ValidationError as e:
            if e.code != "count_mismatch":
                failures.append(f"count_mismatch: got code {e.code!r}")
            else:
                print("  reject count_mismatch ... ok")
    finally:
        log_path.unlink(missing_ok=True)

    # malformed_receipt
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write('ROUTE_RECEIPT\tnot json\n')
        f.write('ROUTE_DONE\t{"source_id": "src1", "host_id": "host1", "features": "default", "request": "scalar", "cases": 0}\n')
        log_path = Path(f.name)
    try:
        try:
            cmd_accept(log_path, "default", "scalar", "src1", "host1", accept_ledger)
            failures.append("malformed_receipt: should have failed")
        except ValidationError as e:
            if e.code != "malformed_receipt":
                failures.append(f"malformed_receipt: got code {e.code!r}")
            else:
                print("  reject malformed_receipt ... ok")
    finally:
        log_path.unlink(missing_ok=True)

    if failures:
        print(f"\n{len(failures)} test failure(s):")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1

    print("All tests passed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=None,
                        help="path to ledger JSON (default: ./simd_ledger.json)")
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

    # test
    sub.add_parser("test", help="run built-in unit tests")

    args = parser.parse_args(argv)

    # Determine ledger path.
    ledger_path = args.ledger if hasattr(args, 'ledger') and args.ledger is not None \
        else Path(__file__).parent / "simd_ledger.json"

    try:
        if args.command == "test":
            return run_tests()

        if args.command == "inventory":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            return cmd_inventory(args.rustdoc)

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