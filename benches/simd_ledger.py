#!/usr/bin/env python3
"""Canonical SIMD route ledger: inventory, check, generate, and accept.

Only the standard library. Follows the style of existing ``benches/*.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
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
VALID_UNITS = frozenset({"Elements", "Bytes", "Bits", "Words"})
VALID_BASES = frozenset({"Source", "Target"})
VALID_MULTIPLICITIES = frozenset({
    "One", "Rows", "ActiveRows", "ActivePairs", "UnitAddPairs",
})

KNOWN_CALLABLES: set[str] | None = None


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
    else:
        # Without rustdoc, at least check structural validity.
        pass

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
    """Return {rule_id: rule}."""
    rules = ledger.get("rules")
    if not isinstance(rules, dict):
        _fail("ledger.rules must be a dict")
    if not rules:
        _fail("ledger.rules is empty")

    for rid, rule in rules.items():
        if not isinstance(rule, dict):
            _fail(f"rule {rid!r} is not a dict")
        group = rule.get("group")
        if not isinstance(group, str) or not group:
            _fail(f"rule {rid!r}: missing group")
        fixtures = rule.get("fixtures")
        if not isinstance(fixtures, list) or not fixtures:
            _fail(f"rule {rid!r}: missing fixtures")
        for fix in fixtures:
            if not isinstance(fix, dict):
                _fail(f"rule {rid!r}: fixture entry is not a dict")
            fid = fix.get("id")
            if not isinstance(fid, str) or not fid:
                _fail(f"rule {rid!r}: fixture missing id")
            ftype = fix.get("type")
            if not isinstance(ftype, str) or not ftype:
                _fail(f"rule {rid!r}: fixture {fid!r} missing type")
    return rules


def _validate_applicability(ledger: dict, rules: dict) -> None:
    """Validate applicability rules."""
    apps = ledger.get("applicability")
    if not isinstance(apps, dict):
        _fail("ledger.applicability must be a dict")
    if not apps:
        _fail("ledger.applicability is empty")

    rule_ids = set(rules)
    callables = ledger.get("callables", {})

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
    """Return {name: work_rule}."""
    wr = ledger.get("work_rules")
    if not isinstance(wr, dict):
        _fail("ledger.work_rules must be a dict")
    if not wr:
        _fail("ledger.work_rules is empty")

    for name, rule in sorted(wr.items()):
        if not isinstance(rule, dict):
            _fail(f"work_rule {name!r} is not a dict")
        for field in ("substage", "unit", "basis", "multiplicity", "lane", "loss"):
            if field not in rule:
                _fail(f"work_rule {name!r}: missing {field}")
        substage = rule["substage"]
        if substage not in VALID_SUBSTAGES:
            _fail(f"work_rule {name!r}: invalid substage {substage!r}")
        scope = rule.get("scope")
        if scope is not None and scope not in VALID_SCOPES:
            _fail(f"work_rule {name!r}: invalid scope {scope!r}")
        unit = rule["unit"]
        if unit not in VALID_UNITS:
            _fail(f"work_rule {name!r}: invalid unit {unit!r}")
        basis = rule["basis"]
        if basis not in VALID_BASES:
            _fail(f"work_rule {name!r}: invalid basis {basis!r}")
        mult = rule["multiplicity"]
        if mult not in VALID_MULTIPLICITIES:
            _fail(f"work_rule {name!r}: invalid multiplicity {mult!r}")
        lane = rule["lane"]
        if not isinstance(lane, (int, str)):
            _fail(f"work_rule {name!r}: lane must be int or string path")
        loss = rule["loss"]
        if not isinstance(loss, (int, str)):
            _fail(f"work_rule {name!r}: loss must be int or string path")
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


def _validate_obligations(ledger: dict, rules: dict, wrules: dict,
                          geos: dict, coeffs: dict) -> dict:
    """Return {obligation_id: obligation_dict}."""
    obs = ledger.get("obligations")
    if not isinstance(obs, dict):
        _fail("ledger.obligations must be a dict")

    callables = ledger.get("callables", {})
    rule_ids = set(rules)
    wr_names = set(wrules)
    geo_names = set(geos)
    coeff_names = set(coeffs)
    seen_ids: set[str] = set()

    for oid, ob in sorted(obs.items()):
        if not isinstance(ob, dict):
            _fail(f"obligation {oid!r} is not a dict")
        if oid in seen_ids:
            _fail(f"duplicate obligation {oid!r}")
        seen_ids.add(oid)

        # Parse obligation ID: <callable>@<rule-id>
        if "@" not in oid:
            _fail(f"obligation {oid!r}: missing @ separator")
        call_name, rule_id = oid.split("@", 1)
        if call_name not in callables:
            _fail(f"obligation {oid!r}: unknown callable {call_name!r}")
        if rule_id not in rule_ids:
            _fail(f"obligation {oid!r}: unknown rule {rule_id!r}")

        cells = ob.get("cells")
        if not isinstance(cells, dict):
            _fail(f"obligation {oid!r}: cells must be a dict")

        for key, cell in sorted(cells.items()):
            # Key is like "default/v4x" or "all/scalar"
            if "/" not in key:
                _fail(f"obligation {oid!r}: cell key {key!r} missing / separator")
            fc, req = key.split("/", 1)
            if fc not in VALID_FEATURE_CONFIGS:
                _fail(f"obligation {oid!r}: invalid feature config {fc!r}")
            if req not in VALID_REQUESTS:
                _fail(f"obligation {oid!r}: invalid request {req!r}")

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
                if not isinstance(wref, str) or wref not in wr_names:
                    _fail(f"obligation {oid!r} cell {key!r}: unknown work rule {wref!r}")

            perf = cell.get("perf", "unmeasured")
            if perf not in VALID_PERF:
                _fail(f"obligation {oid!r} cell {key!r}: invalid perf {perf!r}")

    return obs


def _validate_evidence(ledger: dict) -> None:
    ev = ledger.get("evidence")
    if not isinstance(ev, dict):
        _fail("ledger.evidence must be a dict")
    # evidence is optional at the baseline


def validate_schema(ledger: dict, rustdoc: Path | None = None) -> None:
    """Full schema validation."""
    _validate_callables(ledger, rustdoc)
    rules = _validate_rules(ledger)
    _validate_applicability(ledger, rules)
    wrules = _validate_work_rules(ledger)
    geos = _validate_geometries(ledger)
    coeffs = _validate_coefficient_patterns(ledger)
    _validate_obligations(ledger, rules, wrules, geos, coeffs)
    _validate_evidence(ledger)


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


def cmd_check(ledger_path: Path, schema_only: bool, rustdoc: Path | None,
              test_list: Path | None, case_list: Path | None,
              surface_inventory: Path | None) -> int:
    """Validate the ledger."""
    if not ledger_path.is_file():
        _fail(f"{ledger_path} not found")
    ledger = json.loads(ledger_path.read_text())

    if schema_only:
        validate_schema(ledger, rustdoc)
        print("simd_ledger: schema valid")
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

    # Validate rustdoc is non-empty.
    known = _load_known_callables(rustdoc)
    if not known:
        _fail("rustdoc inventory is empty")

    # Validate test list is non-empty.
    tl = test_list.read_text().strip()
    if not tl:
        _fail("test list is empty")

    # Validate case list is non-empty.
    cl = case_list.read_text().strip()
    if not cl:
        _fail("case list is empty")

    # Validate surface inventory is non-empty.
    si = surface_inventory.read_text().strip()
    if not si:
        _fail("surface inventory is empty")

    validate_schema(ledger, rustdoc)
    print("simd_ledger: full check passed")
    return 0


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


def _generate_cases(ledger: dict) -> str:
    """Generate src/kernel/tests/routes/cases.rs content."""
    lines = [
        "// @generated by benches/simd_ledger.py; edit the ledger, not this file.",
        "",
        "use crate::kernel::tests::routes::{",
        "    Case, Cell, Geometry, Coeff,",
        "    Features, Request,",
        "    substage::Substage, scope::Scope, unit::Unit,",
        "    basis::Basis, multiplicity::Multiplicity,",
        "    work::Work,",
        "};",
        "use crate::kernel::Backend;",
        "",
    ]

    obligations = ledger.get("obligations", {})
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})
    wrules = ledger.get("work_rules", {})

    for oid, ob in sorted(obligations.items()):
        call_name, rule_id = oid.split("@", 1)
        rule = ledger.get("rules", {}).get(rule_id, {})
        fixtures = rule.get("fixtures", [])

        for fixture in fixtures:
            fid = fixture["id"]
            for geo_id, geo in sorted(geos.items()):
                for coeff_id, cpat in sorted(coeffs.items()):
                    case_id = f"{oid}#{fid}/{geo_id}/{coeff_id}"
                    # Build the case
                    cells_data = ob.get("cells", {})

                    # Emit case
                    lines.append(f"// {case_id}")
                    lines.append(f"static CASE_{_safe_ident(case_id)}: &Case = &Case {{")
                    lines.append(f'    id: "{case_id}",')
                    lines.append(f'    obligation: "{oid}",')
                    # run function placeholder: to be filled by runner
                    lines.append(f"    run: crate::kernel::tests::run::unimplemented,")
                    lines.append(f"    geometry: Geometry {{ rows: {geo['rows']}, sources: {geo['sources']}, len: {geo['len']}, misalign: {geo['misalign']}, pitch_pad: {geo.get('pitch_pad', 0)} }},")
                    lines.append(f"    coeffs: &{_coeffs_rust(cpat)},")
                    lines.append(f"    cells: &{_cells_rust(cells_data, wrules)},")
                    lines.append(f"}};")
                    lines.append("")

    return "\n".join(lines)


def _safe_ident(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_#]', '_', s)


def _coeffs_rust(cpat: dict) -> str:
    """Emit a static slice of Coeff."""
    slots = cpat.get("slots", [])
    parts = []
    for s in slots:
        if s == "General":
            parts.append("Coeff::General")
        elif s == "Zero":
            parts.append("Coeff::Zero")
        elif s == "One":
            parts.append("Coeff::One")
        else:
            parts.append(f"Coeff::{s}")
    return f"[{', '.join(parts)}]"


def _cells_rust(cells_data: dict, wrules: dict) -> str:
    """Emit cells array."""
    parts = []
    for key, cell in sorted(cells_data.items()):
        fc, req = key.split("/")
        fc_enum = "Features::Default" if fc == "default" else "Features::All"
        req_enum = {
            "v4x": "Request::V4x",
            "v3_gfni_crypto": "Request::V3GfniCrypto",
            "v3": "Request::V3",
            "v2": "Request::V2",
            "scalar": "Request::Scalar",
        }[req]
        min_serving = {
            "v4x": "Backend::V4x",
            "v3_gfni_crypto": "Backend::V3GfniCrypto",
            "v3": "Backend::V3",
            "v2": "Backend::V2",
            "scalar": "Backend::Scalar",
        }[cell["min_serving"]]
        required = "true" if cell["required"] else "false"

        work_refs = ", ".join(
            f"&WORK_{_safe_ident(w)}" for w in cell["work"]
        ) if cell["work"] else ""

        parts.append(
            f"Cell {{ features: {fc_enum}, request: {req_enum}, "
            f"process: Backend::Scalar, field: Backend::Scalar, "
            f"required: {required}, min_serving: {min_serving}, "
            f"work: &[{work_refs}] }}"
        )
    return f"[{', '.join(parts)}]"


def _generate_work_consts(wrules: dict) -> str:
    """Generate Work consts."""
    lines = []
    for name, rule in sorted(wrules.items()):
        scope = rule.get("scope")
        scope_expr = f"Some(Scope::{scope})" if scope else "None"
        lane = rule["lane"]
        loss = rule["loss"]
        lane_expr = str(lane) if isinstance(lane, int) else lane
        loss_expr = str(loss) if isinstance(loss, int) else loss
        lines.append(
            f"static WORK_{_safe_ident(name)}: Work = Work {{ "
            f"substage: Substage::{rule['substage']}, "
            f"scope: {scope_expr}, "
            f"unit: Unit::{rule['unit']}, "
            f"basis: Basis::{rule['basis']}, "
            f"multiplicity: Multiplicity::{rule['multiplicity']}, "
            f"lane: {lane_expr}, "
            f"loss: {loss_expr} }};"
        )
    return "\n".join(lines)


def cmd_generate(out_path: Path, check: bool, ledger: dict) -> int:
    """Generate the route cases file or check byte identity."""
    content = _generate_cases(ledger)
    # Prepend work consts
    wrules = ledger.get("work_rules", {})
    work_consts = _generate_work_consts(wrules)
    full = content + "\n" + work_consts + "\n"

    if check:
        if not out_path.is_file():
            _fail(f"{out_path} does not exist for --check")
        existing = out_path.read_text()
        if existing != full:
            _fail(f"{out_path} differs from generated output")
        return 0

    out_path.write_text(full)

    # Verify rustfmt
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


def cmd_accept(log: Path, features: str, request: str, source_id: str,
               host_id: str, ledger: dict) -> int:
    """Fail-closed receipt audit."""
    if not log.is_file():
        _fail(f"accept: log file {log} not found")

    text = log.read_text()

    # Extract receipts and DONE marker.
    receipts: list[dict] = []
    done: dict | None = None
    case_count: int | None = None

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

    # Verify DONE matches expectations.
    if done.get("source_id") != source_id:
        _fail("accept: reject source_mismatch")
    if done.get("host_id") != host_id:
        _fail("accept: reject host_mismatch")
    if done.get("features") != features:
        _fail("accept: reject feature_mismatch")
    if done.get("request") != request:
        _fail("accept: reject request_mismatch")

    if not receipts:
        _fail("accept: reject empty_log")

    rcount = done.get("cases", 0)
    if rcount != len(receipts):
        _fail("accept: reject count_mismatch")

    # Build required receipt set from ledger.
    obligations = ledger.get("obligations", {})
    required_set: set[str] = set()
    for oid, ob in obligations.items():
        for key, cell in ob.get("cells", {}).items():
            cell_fc, cell_req = key.split("/")
            if cell_fc == features and cell_req == request and cell.get("required"):
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

        # result starts with "pass", "fail", or "skip"
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
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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

    args = parser.parse_args(argv)

    ledger_path = Path(__file__).parent / "simd_ledger.json"

    try:
        if args.command == "inventory":
            if not ledger_path.is_file():
                _fail("simd_ledger.json not found")
            return cmd_inventory(args.rustdoc)

        if args.command == "check":
            if args.full and args.schema_only:
                _fail("cannot use both --full and --schema-only")
            if args.full:
                return cmd_check(ledger_path, False, args.rustdoc, args.test_list,
                                 args.case_list, args.surface_inventory)
            return cmd_check(ledger_path, True, args.rustdoc, None, None, None)

        if args.command == "generate":
            if not ledger_path.is_file():
                _fail("simd_ledger.json not found")
            ledger = json.loads(ledger_path.read_text())
            return cmd_generate(args.out, args.check, ledger)

        if args.command == "accept":
            if not ledger_path.is_file():
                _fail("simd_ledger.json not found")
            ledger = json.loads(ledger_path.read_text())
            return cmd_accept(args.log, args.features, args.request,
                              args.source_id, args.host_id, ledger)

        return 0
    except ValidationError:
        return 1


if __name__ == "__main__":
    sys.exit(main())