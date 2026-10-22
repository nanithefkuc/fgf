#!/usr/bin/env python3
"""Canonical SIMD route ledger: inventory, check, generate, and accept.

Only the standard library. Follows the style of existing ``benches/*.py``.

Schema v2: ISA facts at top level; work rules use expressions for iterations
and loss_bytes; lane comes from isa, not per-rule dicts. Every (features,
request) pair yields a cell including default/v4x.
"""

from __future__ import annotations

import argparse
import json
import math
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
VALID_REQUIRED_STATUSES = frozenset({"direct", "composition", "zero-budget"})
EXPR_VAR_NAMES = frozenset({
    "lane", "degree", "relative_degree", "power", "words", "elem_bytes",
})

KNOWN_CALLABLES: set[str] | None = None

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
# ISA facts (schema v2)
# ---------------------------------------------------------------------------

def _validate_isa(ledger: dict) -> dict:
    """Validate the isa facts block and return it."""
    isa = ledger.get("isa")
    if not isinstance(isa, dict):
        _fail("ledger.isa must be a dict")
    lane_bytes = isa.get("lane_bytes")
    if not isinstance(lane_bytes, dict):
        _fail("ledger.isa.lane_bytes must be a dict")
    for tier in ("v2", "v3", "v3_gfni_crypto", "v4x"):
        if tier not in lane_bytes or not isinstance(lane_bytes[tier], int) or lane_bytes[tier] <= 0:
            _fail(f"ledger.isa.lane_bytes.{tier} must be a positive integer")
    process = isa.get("process")
    if not isinstance(process, dict):
        _fail("ledger.isa.process must be a dict")
    for fc in VALID_FEATURE_CONFIGS:
        fc_map = process.get(fc)
        if not isinstance(fc_map, dict):
            _fail(f"ledger.isa.process.{fc} must be a dict")
        for req in VALID_REQUESTS:
            resolved = fc_map.get(req)
            if resolved not in VALID_REQUESTS:
                _fail(f"ledger.isa.process.{fc}.{req} must be a valid request tier, got {resolved!r}")
    return isa


def _lane_for_tier(isa: dict, tier: str) -> int:
    """Return lane bytes for a tier from ISA facts."""
    return isa["lane_bytes"][tier]


def _process_for(isa: dict, features: str, request: str) -> str:
    """Return the expected process resolution for (features, request)."""
    return isa["process"][features][request]


# ---------------------------------------------------------------------------
# Safe expression evaluator (no eval)
# ---------------------------------------------------------------------------

def _evaluate_expr(expr: str, variables: dict[str, int]) -> int:
    """Evaluate a simple arithmetic expression over integer variables.

    Allowed: +, -, *, //, (), integer literals, named variables.
    Disallowed: **, function calls, unary minus on expressions.
    Returns int; raises ValueError on parse/eval failure or negative result.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")

    tokens = _tokenize_expr(expr)
    result, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"unexpected token after expression: {tokens[pos:]!r}")
    value = _eval_ast(result, variables)
    if value < 0:
        raise ValueError(f"expression evaluated to negative: {value}")
    return value


def _tokenize_expr(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/()":
            # Check for // (integer division)
            if c == "/" and i + 1 < len(expr) and expr[i + 1] == "/":
                tokens.append("//")
                i += 2
            else:
                tokens.append(c)
                i += 1
        elif c.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            raise ValueError(f"unexpected character: {c!r}")
    return tokens


def _parse_expr(tokens: list[str], pos: int) -> tuple[dict, int]:
    """Parse additive expression: term ('+'|'-' term)*."""
    left, pos = _parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        right, pos = _parse_term(tokens, pos + 1)
        left = {"op": op, "left": left, "right": right}
    return left, pos


def _parse_term(tokens: list[str], pos: int) -> tuple[dict, int]:
    """Parse multiplicative term: factor ('*'|'//' factor)*."""
    left, pos = _parse_factor(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("*", "//"):
        op = tokens[pos]
        right, pos = _parse_factor(tokens, pos + 1)
        left = {"op": op, "left": left, "right": right}
    return left, pos


def _parse_factor(tokens: list[str], pos: int) -> tuple[dict, int]:
    """Parse factor: int | name | '(' expr ')' | '-' factor."""
    if pos >= len(tokens):
        raise ValueError("unexpected end of expression")
    t = tokens[pos]
    if t.isdigit():
        return {"op": "lit", "value": int(t)}, pos + 1
    if t == "(":
        node, pos = _parse_expr(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("expected ')'")
        return node, pos + 1
    if t == "-":
        # Unary minus only on literals
        node, pos = _parse_factor(tokens, pos + 1)
        if node["op"] != "lit":
            raise ValueError("unary minus only on literal")
        return {"op": "lit", "value": -node["value"]}, pos
    if re.match(r'^[a-zA-Z_]\w*$', t):
        return {"op": "var", "name": t}, pos + 1
    raise ValueError(f"unexpected token: {t!r}")


def _eval_ast(node: dict, variables: dict[str, int]) -> int:
    op = node["op"]
    if op == "lit":
        return node["value"]
    if op == "var":
        name = node["name"]
        if name not in variables:
            raise ValueError(f"unknown variable: {name!r}")
        return variables[name]
    if op in ("+", "-", "*", "//"):
        left = _eval_ast(node["left"], variables)
        right = _eval_ast(node["right"], variables)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "//":
            if right == 0:
                raise ValueError("division by zero")
            return left // right
    raise ValueError(f"unknown op: {op!r}")


def _validate_expr(expr: str, allowed_vars: frozenset[str]) -> None:
    """Validate an expression: parse it and check all variable references."""
    tokens = _tokenize_expr(expr)
    ast, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"trailing tokens: {tokens[pos:]!r}")
    _check_vars(ast, allowed_vars)


def _check_vars(node: dict, allowed: frozenset[str]) -> None:
    if node["op"] == "var":
        if node["name"] not in allowed:
            raise ValueError(f"variable {node['name']!r} not in {sorted(allowed)}")
    elif node["op"] in ("+", "-", "*", "//"):
        _check_vars(node["left"], allowed)
        _check_vars(node["right"], allowed)


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
            # elem_bytes and reference required for bulk fixtures
            # skip when the rule declares a 'reason' (non-bulk marker)
            if has_type and group not in ("map", "map_bits") and "reason" not in rule:
                if "elem_bytes" not in fix:
                    _fail(f"rule {rid!r}: fixture {fid!r} missing elem_bytes")
                if "reference" not in fix:
                    _fail(f"rule {rid!r}: fixture {fid!r} missing reference")

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
    """Return {name: work_rule}. Schema v2: iterations expression, loss_bytes expression."""
    wr = ledger.get("work_rules")
    if not isinstance(wr, dict):
        _fail("ledger.work_rules must be a dict")
    if not wr:
        _fail("ledger.work_rules is empty")

    for name, rule in sorted(wr.items()):
        if not isinstance(rule, dict):
            _fail(f"work_rule {name!r} is not a dict")
        for field in ("substage", "unit", "basis", "multiplicity",
                       "iterations", "loss_bytes"):
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

        # iterations must be a valid expression using allowed metadata vars
        iterations = rule["iterations"]
        if not isinstance(iterations, str) or not iterations:
            _fail(f"work_rule {name!r}: iterations must be a non-empty expression string")
        iter_vars = EXPR_VAR_NAMES - {"lane"}  # iterations doesn't use lane
        try:
            _validate_expr(iterations, iter_vars)
        except ValueError as e:
            _fail(f"work_rule {name!r}: invalid iterations expression: {e}")

        # loss_bytes must be a valid expression using 'lane' and metadata vars
        loss_bytes = rule["loss_bytes"]
        if not isinstance(loss_bytes, str) or not loss_bytes:
            _fail(f"work_rule {name!r}: loss_bytes must be a non-empty expression string")
        try:
            _validate_expr(loss_bytes, EXPR_VAR_NAMES)
        except ValueError as e:
            _fail(f"work_rule {name!r}: invalid loss_bytes expression: {e}")

        # extra_loss is optional and is a Rust const path
        extra_loss = rule.get("extra_loss")
        if extra_loss is not None and (not isinstance(extra_loss, str) or not extra_loss):
            _fail(f"work_rule {name!r}: extra_loss must be a non-empty string")

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


# ---------------------------------------------------------------------------
# Evidence record verification
# ---------------------------------------------------------------------------

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
                if done is not None:
                    _fail(f"evidence {ev_id!r}: duplicate ROUTE_DONE in {record_file}")
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

        # Reject receipts without identity
        by_id: dict[str, dict] = {}
        for r in receipts:
            rid = r.get("id")
            if not rid:
                _fail(f"evidence {ev_id!r}: receipt without id field in {record_file}")
            by_id[rid] = r

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
    Every (features, request) pair is included, including default/v4x.
    """
    apps = ledger.get("applicability", {})
    callables_by_family = _bulk_callables_by_family(ledger)
    isa = ledger.get("isa", {})
    default_cell = ledger.get("_cell_default", {
        "status": "unverified", "required": False, "perf": "unmeasured"
    })

    expanded: dict[str, dict] = {}
    # Track (callable, rule) pairs to catch overlapping applicability
    seen_pairs: set[tuple[str, str]] = set()

    for app_name, app in sorted(apps.items()):
        app_rules: list[str] = app.get("rules", [])
        exclusions: dict[str, str] = app.get("exclusions", {})
        families: list[str] = app.get("families", [])

        for rule_id in app_rules:
            if rule_id in exclusions:
                continue
            if rule_id not in rules:
                continue

            rule = rules[rule_id]
            if rule.get("non_bulk"):
                continue

            for family in families:
                callables = callables_by_family.get(family, [])
                for call_name in callables:
                    obligation_id = f"{call_name}@{rule_id}"
                    pair = (call_name, rule_id)
                    if pair in seen_pairs:
                        _fail(f"overlapping applicability: {call_name!r} with rule {rule_id!r} "
                              f"appears from multiple applicability entries "
                              f"(last: {app_name!r})")
                    seen_pairs.add(pair)

                    if obligation_id in expanded:
                        continue

                    # Build default cells for every feature config × request.
                    cells: dict[str, dict] = {}
                    for fc in sorted(VALID_FEATURE_CONFIGS):
                        for req in sorted(VALID_REQUESTS):
                            key = f"{fc}/{req}"
                            cells[key] = dict(default_cell)
                            cells[key]["min_serving"] = req
                            cells[key]["work"] = []
                            # Process resolution from ISA facts
                            if isa:
                                cells[key]["process"] = _process_for(isa, fc, req)
                            else:
                                cells[key]["process"] = req
                            cells[key]["field"] = cells[key]["process"]

                    expanded[obligation_id] = {"cells": cells}

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

            if cell["required"] and status not in VALID_REQUIRED_STATUSES:
                _fail(f"obligation {oid!r} cell {key!r}: required=true but status={status!r} "
                      f"(required statuses: {sorted(VALID_REQUIRED_STATUSES)})")

            ms = cell["min_serving"]
            if ms not in VALID_REQUESTS:
                _fail(f"obligation {oid!r} cell {key!r}: invalid min_serving {ms!r}")

            # Validate process and field backends
            process = cell.get("process")
            if process is not None and process not in VALID_REQUESTS:
                _fail(f"obligation {oid!r} cell {key!r}: invalid process {process!r}")
            field = cell.get("field")
            if field is not None and field not in VALID_REQUESTS:
                _fail(f"obligation {oid!r} cell {key!r}: invalid field {field!r}")

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


def _validate_bulk_family_coverage(ledger: dict, expanded: dict) -> None:
    """Every bulk callable family must receive at least one obligation."""
    callables_by_family = _bulk_callables_by_family(ledger)
    for family, callable_names in sorted(callables_by_family.items()):
        covered = False
        for oid in expanded:
            call_name = oid.split("@", 1)[0]
            if call_name in callable_names:
                covered = True
                break
        if not covered:
            _fail(f"bulk family {family!r} has no obligations; check applicability rules")


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

def _parse_test_list(text: str) -> set[str]:
    """Parse cargo test --lib -- --list output.

    Format: each non-empty line is one of:
      <path>: test
      <path>: bench
    We extract the test path (before ': test').
    """
    result: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ": test" in line or ": bench" in line:
            path = line.rsplit(":", 1)[0].strip()
            result.add(path)
    return result


def _parse_case_list(text: str) -> set[str]:
    """Parse ROUTE_CASE_LIST lines.

    Format: ROUTE_CASE_LIST\t<case-id>
    """
    result: set[str] = set()
    for line in text.splitlines():
        if line.startswith("ROUTE_CASE_LIST\t"):
            result.add(line.split("\t", 1)[1])
    return result


def _parse_surface_inventory(text: str) -> set[str]:
    """Parse SURFACE_INVENTORY JSON lines.

    Format: SURFACE_INVENTORY\t{json}
    Each record has a "case" field with the benchmark case id.
    """
    result: set[str] = set()
    for line in text.splitlines():
        if line.startswith("SURFACE_INVENTORY\t"):
            try:
                record = json.loads(line.split("\t", 1)[1])
            except json.JSONDecodeError:
                _fail(f"malformed SURFACE_INVENTORY line: {line[:80]}")
            case_id = record.get("case")
            if case_id:
                result.add(case_id)
    return result


def cmd_check(ledger: dict, schema_only: bool, rustdoc: Path | None,
              test_list: Path | None, case_list: Path | None,
              surface_inventory: Path | None, records: Path | None) -> int:
    """Validate the ledger."""
    if schema_only:
        _validate_isa(ledger)
        _validate_callables(ledger, rustdoc)
        rules = _validate_rules(ledger)
        _validate_applicability(ledger, rules)
        wrules = _validate_work_rules(ledger)
        geos = _validate_geometries(ledger)
        coeffs = _validate_coefficient_patterns(ledger)
        _validate_evidence(ledger)

        expanded = _expand_obligations(ledger, rules)
        _validate_expanded_obligations(ledger, expanded, rules, wrules, geos, coeffs)
        _validate_explicit_obligations(ledger, expanded)
        _validate_bulk_family_coverage(ledger, expanded)

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

    # Validate inventories are non-empty and parse them.
    known = _load_known_callables(rustdoc)
    if not known:
        _fail("rustdoc inventory is empty")

    tl_text = test_list.read_text().strip()
    if not tl_text:
        _fail("test list is empty")
    test_names = _parse_test_list(tl_text)
    if not test_names:
        _fail("test list contains no valid test entries")

    cl_text = case_list.read_text().strip()
    if not cl_text:
        _fail("case list is empty")
    case_ids = _parse_case_list(cl_text)
    if not case_ids:
        _fail("case list contains no ROUTE_CASE_LIST lines")

    si_text = surface_inventory.read_text().strip()
    if not si_text:
        _fail("surface inventory is empty")
    surface_ids = _parse_surface_inventory(si_text)
    if not surface_ids:
        _fail("surface inventory contains no SURFACE_INVENTORY lines")

    # --records is mandatory if the ledger cites any evidence.
    if ledger.get("evidence"):
        _verify_evidence_records(ledger, records)

    # Full schema validation.
    _validate_isa(ledger)
    _validate_callables(ledger, rustdoc)
    rules = _validate_rules(ledger)
    _validate_applicability(ledger, rules)
    wrules = _validate_work_rules(ledger)
    geos = _validate_geometries(ledger)
    coeffs = _validate_coefficient_patterns(ledger)
    _validate_evidence(ledger)

    expanded = _expand_obligations(ledger, rules)
    _validate_expanded_obligations(ledger, expanded, rules, wrules, geos, coeffs)
    _validate_explicit_obligations(ledger, expanded)
    _validate_bulk_family_coverage(ledger, expanded)

    n_cells = _count_cells(expanded)
    print(f"simd_ledger: full check passed  obligations={len(expanded)} cells={n_cells}")
    return 0


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------

def _backend_rust_name(backend: str) -> str:
    """Map a backend request string to its Rust enum variant name."""
    mapping = {
        "scalar": "Scalar", "v2": "V2", "v3": "V3",
        "v3_gfni_crypto": "V3GfniCrypto", "v4x": "V4x",
    }
    return mapping.get(backend, backend)


def _request_rust_name(request: str) -> str:
    """Map a request string to its Rust enum variant name."""
    mapping = {
        "scalar": "Scalar", "v2": "V2", "v3": "V3",
        "v3_gfni_crypto": "V3GfniCrypto", "v4x": "V4x",
    }
    return mapping.get(request, request)


def _safe_ident(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_#]', '_', s)


def _work_iterations(wr: dict, fixture_vars: dict | None = None) -> int:
    """Compute repetitions from the work rule's iterations expression."""
    iterations = wr.get("iterations", "1")
    try:
        value = int(iterations)
        return max(1, value)
    except (TypeError, ValueError):
        pass
    if fixture_vars is None:
        # Without metadata, evaluate with empty context for literal "1" etc.
        return max(1, _evaluate_expr(iterations, {}))
    return max(1, _evaluate_expr(iterations, fixture_vars))


def _fixture_vars(fixture: dict, rule: dict) -> dict[str, int]:
    """Extract metadata variables from fixture and rule for expression evaluation."""
    result: dict[str, int] = {}
    # elem_bytes from fixture or rule
    elem_bytes = fixture.get("elem_bytes", rule.get("elem_bytes", 1))
    result["elem_bytes"] = elem_bytes
    # degree from rule
    degree = rule.get("degree", 1)
    result["degree"] = degree
    result["relative_degree"] = rule.get("relative_degree", fixture.get("relative_degree", 1))
    result["power"] = fixture.get("power", rule.get("power", 1))
    result["words"] = fixture.get("words", max(1, math.ceil(elem_bytes / 8)))
    return result


def _fixture_is_gf2(fixture: dict) -> bool:
    """True when the fixture type is a degree-1 GF(2) presentation."""
    ftype = fixture.get("type", "")
    return bool(re.search(r"FiniteField\s*<\s*2\s*,\s*1\s*,", ftype))


def _general_slot_count(coeff_pattern: dict) -> int:
    return sum(1 for s in coeff_pattern.get("slots", []) if s == "General")


def _count_active_pairs(coeff_pattern: dict) -> int:
    """Count General slots (active coefficient pairs)."""
    return _general_slot_count(coeff_pattern)


def _count_unit_add_pairs(coeff_pattern: dict) -> int:
    """Count One slots (unit accumulate pairs)."""
    return sum(1 for s in coeff_pattern.get("slots", []) if s == "One")


def _callable_takes_coefficient(call_name: str) -> bool:
    """True when the callable carries a coefficient slot (mul_* / *_scalar)."""
    base = call_name.split("::")[-1]
    return base.startswith("mul_") or base.endswith("_scalar")


def _multiplicity_count(mult: str, geo: dict, coeff_pattern: dict) -> int:
    if mult == "Rows":
        return geo.get("rows", 1)
    if mult == "ActiveRows":
        return geo.get("rows", 1)
    if mult == "ActivePairs":
        return _count_active_pairs(coeff_pattern)
    if mult == "UnitAddPairs":
        return _count_unit_add_pairs(coeff_pattern)
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
    """Convert loss_bytes to the work rule's unit. Loss rounds up (ceil)."""
    elem_bytes = rule.get("elem_bytes", 1)
    if unit == "Bytes":
        return loss_bytes
    if unit == "Bits":
        return loss_bytes * 8
    if unit == "Words":
        return max(1, math.ceil(loss_bytes / 8))
    if elem_bytes <= 1:
        return loss_bytes
    return math.ceil(loss_bytes / elem_bytes)


def _lane_in_units(unit: str, lane_bytes: int, rule: dict) -> int:
    """Convert lane_bytes to the work rule's unit. Lane rounds down (floor)."""
    elem_bytes = rule.get("elem_bytes", 1)
    if unit == "Bytes":
        return lane_bytes
    if unit == "Bits":
        return lane_bytes * 8
    if unit == "Words":
        return max(1, lane_bytes // 8)
    if elem_bytes <= 1:
        return lane_bytes
    return lane_bytes // elem_bytes


def _compute_loss_bytes(wr: dict, lane: int, fixture_vars: dict) -> int:
    """Evaluate the loss_bytes expression with lane and fixture vars."""
    expr = wr["loss_bytes"]
    variables = dict(fixture_vars)
    variables["lane"] = lane
    return _evaluate_expr(expr, variables)


def _work_budget(work_rule: dict, geo: dict, source_rule: dict,
                 target_rule: dict, tier: str, isa: dict,
                 coeff_pattern: dict | None = None) -> int:
    """Budget for one Work item: iterations * mult * (len_in_units - loss_in_units)."""
    mult_rule = work_rule["multiplicity"]
    geo_count = _multiplicity_count(mult_rule, geo, coeff_pattern or {"slots": []})

    # Select the rule for length conversion based on basis
    basis = work_rule["basis"]
    unit = work_rule["unit"]
    geo_len = geo.get("len", 0)

    if basis == "Target":
        # Convert element length to target units
        src_eb = source_rule.get("elem_bytes", 1)
        tgt_eb = target_rule.get("elem_bytes", 1)
        if unit == "Elements":
            target_len = (geo_len * src_eb) // tgt_eb
        elif unit == "Bytes":
            target_len = geo_len * src_eb
        elif unit == "Bits":
            target_len = max(1, (geo_len * src_eb * 8) // target_rule.get("elem_bits", tgt_eb * 8))
        elif unit == "Words":
            target_len = max(1, (geo_len * src_eb) // 8)
        else:
            target_len = (geo_len * src_eb) // tgt_eb
        length = target_len
        length_rule = target_rule
    else:
        length = _len_in_units(unit, geo_len, source_rule)
        length_rule = source_rule

    lane_bytes = isa.get("lane_bytes", {}).get(tier, 1) if isa else 1
    fixture_vars = _fixture_vars({}, length_rule)
    loss_bytes = _compute_loss_bytes(work_rule, lane_bytes, fixture_vars)
    loss = _loss_in_units(unit, loss_bytes, length_rule)

    iter_count = _work_iterations(work_rule, fixture_vars)

    return iter_count * geo_count * (length - loss)


def _validate_route_budgets(ledger: dict) -> None:
    """Generator check: reject a non-positive budget on an eligible nondegenerate case."""
    isa = ledger.get("isa", {})
    if not isa:
        return  # Can't validate without ISA facts
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
                    budget = _work_budget(wr, geo, rule, rule, req, isa)
                    if budget <= 0:
                        _fail(f"obligation {oid!r} cell {key!r} geometry {geo_id!r}: "
                              f"non-positive budget for work rule {wr_name!r} (budget={budget})")


def _validate_gf2_general(ledger: dict) -> None:
    """Generator check: every GF(2) obligation must have at least one valid pattern without General slots."""
    rules = _validate_rules(ledger)
    expanded = _expand_obligations(ledger, rules)
    coeffs = ledger.get("coefficient_patterns", {})

    if not coeffs:
        return  # No patterns to validate against

    for oid in sorted(expanded):
        call_name, rule_id = oid.split("@", 1)
        if not _callable_takes_coefficient(call_name):
            continue
        rule = rules.get(rule_id, {})
        all_gf2 = True
        has_any_fixture = False
        for fixture in rule.get("fixtures", []):
            has_any_fixture = True
            if not _fixture_is_gf2(fixture):
                all_gf2 = False
                break
        if not (all_gf2 and has_any_fixture):
            continue
        # This obligation is entirely over GF(2) fixtures — need at least one valid pattern
        valid_cps = _valid_coeff_patterns_for_rule(coeffs, rule)
        if not valid_cps:
            _fail(f"GF(2) obligation {oid!r} has no valid coefficient patterns "
                  f"(all patterns have General slots which GF(2) cannot use)")


def _valid_coeff_patterns_for_rule(coeffs: dict, rule: dict) -> list[tuple[str, dict]]:
    """Return valid coefficient patterns for a rule, filtering General from GF(2)."""
    result: list[tuple[str, dict]] = []
    for cp_name, cp in sorted(coeffs.items()):
        if not isinstance(cp, dict):
            continue
        if _general_slot_count(cp) > 0:
            # General only valid on non-GF(2) fixtures
            gf2 = all(_fixture_is_gf2(f) for f in rule.get("fixtures", []))
            if gf2:
                continue
        result.append((cp_name, cp))
    return result


def _generate_cases(ledger: dict) -> str:
    """Generate src/kernel/tests/routes/cases.rs content.

    Produces complete Case/Cell/Work tables per contract §3.
    Cases using v4x in any cell go to CASES_SIMD512; rest to CASES.
    """
    isa = ledger.get("isa", {})
    _validate_gf2_general(ledger)
    if isa:
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
    expanded = _expanded_with_overrides(ledger)
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})
    wrules = ledger.get("work_rules", {})

    # Determine which obligations need simd512
    simd512_oids: set[str] = set()
    for oid, ob in sorted(expanded.items()):
        for key, cell in sorted(ob.get("cells", {}).items()):
            fc, req = key.split("/", 1)
            process = cell.get("process", req)
            field = cell.get("field", process)
            if req == "v4x" or process == "v4x" or field == "v4x":
                simd512_oids.add(oid)
                break
            if cell.get("min_serving") == "v4x":
                simd512_oids.add(oid)
                break

    # Collect all needed work combinations
    work_combos: dict[tuple[str, str, str], dict] = {}  # (wr_name, fc, req) -> work dict
    for oid, ob in sorted(expanded.items()):
        for key, cell in sorted(ob.get("cells", {}).items()):
            fc, req = key.split("/", 1)
            for wr_name in cell.get("work", []):
                wr = wrules.get(wr_name)
                if not wr:
                    continue
                combo_key = (wr_name, fc, req)
                if combo_key not in work_combos:
                    work_combos[combo_key] = {
                        "wr_name": wr_name,
                        "wr": wr,
                        "fc": fc,
                        "req": req,
                    }

    # Generate unique Work static items
    work_statics: dict[tuple[str, str, str], str] = {}
    for combo_key, combo in sorted(work_combos.items()):
        wr_name = combo["wr_name"]
        wr = combo["wr"]
        fc = combo["fc"]
        req = combo["req"]
        substage = wr["substage"]
        unit = wr["unit"]
        basis = wr["basis"]
        mult = wr["multiplicity"]

        scope = wr.get("scope")
        scope_expr = "None"
        if scope:
            scope_expr = f"Some((Scope::{scope['scope']}, Pass::{scope['pass']}))"

        ident = _safe_ident(f"{wr_name}_{fc}_{req}")
        work_statics[combo_key] = ident

        # Compute lane and loss in the rule's unit for the appropriate fixture
        lane_bytes = isa.get("lane_bytes", {}).get(req, 1) if isa else 1
        # Use a representative fixture to compute lane/loss in correct unit
        rep_rule = {"elem_bytes": 1}
        lane = _lane_in_units(unit, lane_bytes, rep_rule)
        loss_bytes = _compute_loss_bytes(wr, lane_bytes, {"elem_bytes": 1})
        loss = _loss_in_units(unit, loss_bytes, rep_rule)
        iterations = _work_iterations(wr)

        lines.append(f"static WORK_{ident}: Work = Work {{")
        lines.append(f"    substage: Substage::{substage},")
        lines.append(f"    scope: {scope_expr},")
        lines.append(f"    unit: Unit::{unit},")
        lines.append(f"    basis: Basis::{basis},")
        lines.append(f"    multiplicity: Multiplicity::{mult},")
        lines.append(f"    lane: {lane},")
        lines.append(f"    loss: {loss},")
        lines.append(f"    iterations: {iterations},")
        lines.append(f"}};")
        lines.append("")

    # Generate case tables
    cases_lines: list[str] = []
    cases_simd512_lines: list[str] = []

    for oid, ob in sorted(expanded.items()):
        call_name, rule_id = oid.split("@", 1)
        rule = rules.get(rule_id, {})
        fixtures = rule.get("fixtures", [])

        for fixture in fixtures:
            fid = fixture["id"]
            # Valid coefficient patterns for this rule
            valid_cps = _valid_coeff_patterns_for_rule(coeffs, rule)

            for geo_id, geo in sorted(geos.items()):
                if geo.get("len", 0) <= 0:
                    continue

                for cp_name, cp in valid_cps:
                    case_id = f"{oid}#{fid}/{geo_id}/{cp_name}"
                    # Build cells array
                    cell_entries: list[str] = []
                    for key, cell in sorted(ob.get("cells", {}).items()):
                        fc, req = key.split("/", 1)
                        features = "Features::Default" if fc == "default" else "Features::All"
                        request = f"Request::{_request_rust_name(req)}"
                        process_backend = f"Backend::{_backend_rust_name(cell.get('process', req))}"
                        field_backend = f"Backend::{_backend_rust_name(cell.get('field', cell.get('process', req)))}"
                        required = "true" if cell.get("required") else "false"
                        min_serving = f"Backend::{_backend_rust_name(cell['min_serving'])}"

                        work_refs: list[str] = []
                        for wr_name in cell.get("work", []):
                            wr_key = (wr_name, fc, req)
                            if wr_key in work_statics:
                                work_refs.append(f"&WORK_{work_statics[wr_key]}")

                        if work_refs:
                            work_array = "&[" + ", ".join(work_refs) + "]"
                        else:
                            work_array = "&[]"

                        cell_entry = (
                            f"        Cell {{ features: {features}, request: {request}, "
                            f"process: {process_backend}, field: {field_backend}, "
                            f"required: {required}, min_serving: {min_serving}, "
                            f"work: {work_array} }},"
                        )
                        cell_entries.append(cell_entry)

                    # Build coefficients array
                    coeff_entries: list[str] = []
                    for slot in cp.get("slots", []):
                        coeff_entries.append(f"Coeff::{slot}")

                    # Build geometry
                    rows = geo.get("rows", 1)
                    sources = geo.get("sources", 1)
                    length = geo.get("len", 0)
                    misalign = geo.get("misalign", 0)
                    pitch_pad = geo.get("pitch_pad", 0)
                    geo_str = (f"Geometry {{ rows: {rows}, sources: {sources}, "
                               f"len: {length}, misalign: {misalign}, "
                               f"pitch_pad: {pitch_pad} }}")

                    # Run function path
                    ftype = fixture.get("type", "").replace("crate::", "crate::")
                    run_fn = f"super::run_{_safe_ident(call_name)}::<{ftype}>"

                    case_body = [
                        f"    Case {{",
                        f"        id: \"{case_id}\",",
                        f"        obligation: \"{oid}\",",
                        f"        run: {run_fn},",
                        f"        geometry: {geo_str},",
                        f"        coeffs: &[{', '.join(coeff_entries)}],",
                        f"        cells: &[",
                    ] + cell_entries + [
                        f"        ],",
                        f"    }},",
                    ]

                    if oid in simd512_oids:
                        cases_simd512_lines.extend(case_body)
                    else:
                        cases_lines.extend(case_body)

    # Emit CASES table
    if cases_lines:
        lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\"))]")
        lines.append("static CASES: &[Case] = &[")
        lines.extend(cases_lines)
        lines.append("];")
        lines.append("")
    else:
        lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\"))]")
        lines.append("static CASES: &[Case] = &[];")
        lines.append("")

    # Emit CASES_SIMD512 table
    if cases_simd512_lines:
        lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\", feature = \"simd512\"))]")
        lines.append("static CASES_SIMD512: &[Case] = &[")
        lines.extend(cases_simd512_lines)
        lines.append("];")
        lines.append("")
    else:
        lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\", feature = \"simd512\"))]")
        lines.append("static CASES_SIMD512: &[Case] = &[];")
        lines.append("")

    return "\n".join(lines) + "\n"


def cmd_generate(out_path: Path, check: bool, ledger: dict) -> int:
    """Generate the route cases file or check byte identity."""
    content = _generate_cases(ledger)

    # Format with rustfmt in-place for both modes.
    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False) as tf:
        tf.write(content)
        tmp_path = Path(tf.name)

    result = subprocess.run(
        ["rustfmt", "--edition", "2024", str(tmp_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        _fail(f"generated file fails rustfmt: {result.stderr}")

    formatted = tmp_path.read_text()
    tmp_path.unlink()

    if check:
        if not out_path.is_file():
            _fail(f"{out_path} does not exist for --check")
        existing = out_path.read_text()
        if existing != formatted:
            _fail(f"{out_path} differs from generated output")
        return 0

    out_path.write_text(formatted)
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
                for key, cell in ob.get("cells", {}).items():
                    expanded[oid]["cells"][key] = dict(
                        expanded[oid]["cells"].get(key, {}), **cell)
            else:
                expanded[oid] = ob
    return expanded


def _expand_cases(ledger: dict, expanded: dict) -> set[str]:
    """Return the set of all case IDs from the generated case table."""
    rules = _validate_rules(ledger)
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})

    case_ids: set[str] = set()
    for oid, ob in sorted(expanded.items()):
        call_name, rule_id = oid.split("@", 1)
        rule = rules.get(rule_id, {})
        for fixture in rule.get("fixtures", []):
            fid = fixture["id"]
            valid_cps = _valid_coeff_patterns_for_rule(coeffs, rule)
            for geo_id, geo in sorted(geos.items()):
                if geo.get("len", 0) <= 0:
                    continue
                for cp_name, _cp in valid_cps:
                    case_ids.add(f"{oid}#{fid}/{geo_id}/{cp_name}")
    return case_ids


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
    all_case_ids = _expand_cases(ledger, expanded)

    # Build required set: case ids whose obligation is required in the cell.
    # When the ledger has no geometries/coefficient patterns, fall back to
    # obligation-level required set (backward compatibility with test fixtures).
    required_cases: set[str] = set()
    required_oids: set[str] = set()
    for oid, ob in expanded.items():
        cell = ob.get("cells", {}).get(f"{features}/{request}", {})
        if not cell.get("required"):
            continue
        required_oids.add(oid)
        if all_case_ids:
            for cid in all_case_ids:
                if cid.startswith(oid + "#"):
                    required_cases.add(cid)

    if not required_oids and not required_cases:
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
    received_oids: set[str] = set()
    for rec in receipts:
        cid = rec.get("id", "")
        oid = rec.get("obligation", "")
        if "#" in cid and not oid:
            oid = cid.split("#", 1)[0]
        received_ids.add(cid)
        received_oids.add(oid)

        if rec.get("source_id") != source_id:
            _fail(f"accept: reject receipt_source_mismatch {cid}")

        if rec.get("host_id") != host_id:
            _fail(f"accept: reject receipt_host_mismatch {cid}")

        if rec.get("features") != features:
            _fail(f"accept: reject receipt_feature_mismatch {cid}")

        # request_raw must be a non-empty string
        request_raw = rec.get("request_raw")
        if not isinstance(request_raw, str) or not request_raw:
            _fail(f"accept: reject receipt_request_mismatch {cid}")

        if request_raw != request:
            _fail(f"accept: reject receipt_request_mismatch {cid}")

        # Process and field resolution must equal the ledger's expected cell.
        cell = expanded.get(oid, {}).get("cells", {}).get(f"{features}/{request}", {})
        if "process" in cell and rec.get("process") != cell["process"]:
            _fail(f"accept: reject receipt_process_mismatch {cid}")
        if "field" in cell and rec.get("field") != cell["field"]:
            _fail(f"accept: reject receipt_field_mismatch {cid}")

        result = rec.get("result")
        if result is None or not isinstance(result, str):
            _fail(f"accept: reject receipt_no_result {cid}")

        # result must be exactly 'pass', 'fail', or 'skip'
        if result == "pass":
            pass
        elif result.startswith("fail"):
            _fail(f"accept: reject fail_receipt {cid}")
        elif result.startswith("skip"):
            _fail(f"accept: reject skip_receipt {cid}")
        else:
            _fail(f"accept: reject unknown_result {cid}")

    if all_case_ids:
        missing = required_cases - received_ids
    else:
        missing = required_oids - received_oids
    if missing:
        _fail(f"accept: reject missing_required {sorted(missing)}")

    return 0


# ---------------------------------------------------------------------------
# Fixture compilation check
# ---------------------------------------------------------------------------

def cmd_fixtures(ledger: dict, out_dir: Path) -> int:
    """Write a scratch crate with all fixture types and compile-check them."""
    rules = _validate_rules(ledger)
    out_dir.mkdir(parents=True, exist_ok=True)

    shipped_types: set[str] = set()
    map_fixtures: list[dict] = []
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
                src = fix.get("source", "")
                tgt = fix.get("target", "")
                if src and tgt:
                    ptype = f"fgf::PreparedEmbedding<{src.replace('crate::', 'fgf::')}, {tgt.replace('crate::', 'fgf::')}>"
                    map_fixtures.append({"id": fix.get("id", ""), "source": src, "target": tgt, "prepared_type": ptype})
            elif group == "packed" and rule.get("non_bulk"):
                pass
            else:
                if group == "binary_flat" and rid == "binary-flat-sub8":
                    continue
                if group == "bits":
                    continue
                ftype = fix.get("type", fix.get("source", ""))
                if ftype and "crate::" in ftype:
                    shipped_types.add(ftype)

    cargo_toml = f"""[package]
name = "ledger-fixtures"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
fgf = {{ path = "{Path(__file__).parent.parent}", features = ["alloc"] }}
"""
    (out_dir / "Cargo.toml").write_text(cargo_toml)

    src_dir = out_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    main_lines = ["fn main() {"]
    main_lines.append("    fn _assert_field<T: fgf::FieldBuffer>() {}")
    for t in sorted(shipped_types):
        rust_type = t.replace("crate::", "fgf::")
        main_lines.append(f"    _assert_field::<{rust_type}>();")

    if map_fixtures:
        main_lines.append("")
        main_lines.append("    // Map fixtures: construct the prepared type to enforce trait bounds.")
        for mf in map_fixtures:
            main_lines.append(f"    let _ = std::mem::size_of::<{mf['prepared_type']}>();")

    main_lines.append("}")
    (src_dir / "main.rs").write_text("\n".join(main_lines) + "\n")

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

    inv = sub.add_parser("inventory", help="classify every callable")
    inv.add_argument("--rustdoc", type=Path, required=True)

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

    gen = sub.add_parser("generate", help="emit route cases file")
    gen.add_argument("--out", type=Path, required=True,
                     help="output path for cases.rs")
    gen.add_argument("--check", action="store_true",
                     help="check byte identity against existing file")

    acc = sub.add_parser("accept", help="fail-closed receipt audit")
    acc.add_argument("--log", type=Path, required=True)
    acc.add_argument("--features", required=True, choices=["default", "all"])
    acc.add_argument("--request", required=True, choices=sorted(VALID_REQUESTS))
    acc.add_argument("--source-id", required=True)
    acc.add_argument("--host-id", required=True)

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