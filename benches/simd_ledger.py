#!/usr/bin/env python3
"""Canonical SIMD route ledger: inventory, check, generate, and accept.

Only the standard library. Follows the style of existing ``benches/*.py``.
One pipeline: ``load`` parses and raw-validates the ledger; ``obligations``
expands applicability with overrides merged and validated; ``cases`` expands
fixtures, geometries, and valid coefficient patterns with budgets resolved
per fixture width; ``check``, ``generate``, and ``accept`` all derive from
the same ``obligations()`` and ``cases()``. No other function re-derives
obligations, cases, or budgets.
"""

from __future__ import annotations

import argparse
import json
import math
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
ROUTE_TYPE_NAMES = frozenset({
    "Backend", "Basis", "Case", "Cell", "Coeff", "Features", "Geometry",
    "MapSpec", "Multiplicity", "Outcome", "Pass", "Request", "Scope",
    "Substage", "Unit", "Work",
})
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
    "receipt_required_mismatch", "receipt_min_serving_mismatch",
    "receipt_cell_missing", "receipt_obligation_mismatch",
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


def _validate_map_construction(rid: str, rule: dict, fix: dict) -> None:
    """Validate map power and caller-image byte encodings."""
    if rule.get("group") not in ("map", "map_bits"):
        return
    power = fix.get("power", rule.get("power"))
    if power is not None and (
            not isinstance(power, int) or isinstance(power, bool) or power < 0):
        _fail(f"rule {rid!r}: fixture {fix.get('id')!r} power must be a non-negative integer")

    target_width = fix.get("target_elem_bytes", rule.get("target_elem_bytes"))
    declared_width = rule.get("target_width_bytes")
    if (rule.get("group") == "map_bits" and declared_width is not None
            and declared_width != target_width):
        _fail(f"rule {rid!r}: fixture {fix.get('id')!r} target_elem_bytes "
              f"{target_width} differs from target_width_bytes {declared_width}")

    construction = rule.get("construction")
    images = fix.get("images", rule.get("images"))
    if images is None:
        if construction == "images":
            _fail(f"rule {rid!r}: fixture {fix.get('id')!r} declares image "
                  f"construction without image encodings")
        return
    if not isinstance(images, list):
        _fail(f"rule {rid!r}: fixture {fix.get('id')!r} images must be a list")
    if construction == "structural" and images:
        _fail(f"rule {rid!r}: fixture {fix.get('id')!r} is structural but "
              f"declares caller images")
    for image in images:
        if not isinstance(image, list) or not image:
            _fail(f"rule {rid!r}: fixture {fix.get('id')!r} image {image!r} "
                  f"is not a non-empty byte list")
        for byte in image:
            if (not isinstance(byte, int) or isinstance(byte, bool)
                    or byte < 0 or byte > 0xFF):
                _fail(f"rule {rid!r}: fixture {fix.get('id')!r} image byte "
                      f"{byte!r} is not in 0..=255")
        if (isinstance(target_width, int) and not isinstance(target_width, bool)
                and target_width > 0 and len(image) != target_width):
            _fail(f"rule {rid!r}: fixture {fix.get('id')!r} image has "
                  f"{len(image)} bytes, target_elem_bytes is {target_width}")


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
        for width_key in ("elem_bytes", "target_elem_bytes", "target_width_bytes"):
            if width_key in rule:
                value = rule[width_key]
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    _fail(f"rule {rid!r}: {width_key} must be a positive integer")
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
            if "needs_spec" in fix:
                _fail(f"rule {rid!r}: fixture {fid!r} uses the removed "
                      f"needs_spec key; declare spec only")
            for degree_key in ("degree", "relative_degree"):
                value = fix.get(degree_key)
                if (not isinstance(value, int) or isinstance(value, bool)
                        or value <= 0):
                    _fail(f"rule {rid!r}: fixture {fid!r} {degree_key} "
                          f"must be a positive integer")
            # Widths feed budget conversion directly, so a declared width is
            # validated on both sides of a width-changing map. Map fixtures
            # carry no width default.
            if group in ("map", "map_bits"):
                for width_key in ("elem_bytes", "target_elem_bytes"):
                    value = fix.get(width_key)
                    if (not isinstance(value, int) or isinstance(value, bool)
                            or value <= 0):
                        _fail(f"rule {rid!r}: fixture {fid!r} {width_key} "
                              f"must be a positive integer")
            for width_key in ("elem_bytes", "target_elem_bytes"):
                if width_key not in fix:
                    continue
                value = fix[width_key]
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    _fail(f"rule {rid!r}: fixture {fid!r} {width_key} must be a positive integer")
            _validate_map_construction(rid, rule, fix)

            # elem_bytes and reference required for bulk fixtures
            # skip when the rule declares a 'reason' (non-bulk marker)
            if has_type and group not in ("map", "map_bits") and "reason" not in rule:
                eb = fix.get("elem_bytes")
                if not isinstance(eb, int) or isinstance(eb, bool) or eb <= 0:
                    _fail(f"rule {rid!r}: fixture {fid!r} elem_bytes must be positive")
                ref = fix.get("reference")
                if ref not in VALID_REFERENCES:
                    _fail(f"rule {rid!r}: fixture {fid!r} has unknown reference {ref!r}")

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
    """Return {name: work_rule}. Iterations and loss_bytes are expression strings."""
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

        if "extra_loss" in rule:
            _fail(f"work_rule {name!r}: extra_loss is not part of the schema; "
                  f"route geometries must exceed the live crossover floor")

        if "justification" not in rule:
            _fail(f"work_rule {name!r}: missing justification")
        justification = rule["justification"]
        if not isinstance(justification, str) or not justification:
            _fail(f"work_rule {name!r}: empty justification")

    return wr


def _validate_geometries(ledger: dict) -> dict:
    """Raw shape plus strict field checks: positive integers, GF(2^16) even lengths."""
    geo = ledger.get("geometries")
    if not isinstance(geo, dict):
        _fail("ledger.geometries must be a dict")
    for gid, g in sorted(geo.items()):
        if not isinstance(g, dict):
            _fail(f"geometry {gid!r} is not a dict")
        for field in ("rows", "sources", "len", "misalign", "pitch_pad"):
            value = g.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                _fail(f"geometry {gid!r}: {field} must be an integer")
            if field in ("rows", "sources", "len") and value <= 0:
                _fail(f"geometry {gid!r}: {field} must be positive")
            if field in ("misalign", "pitch_pad") and value < 0:
                _fail(f"geometry {gid!r}: {field} must be non-negative")
    return geo


def _validate_coefficient_patterns(ledger: dict) -> dict:
    """Raw shape plus closed coefficient-variant vocabulary."""
    cp = ledger.get("coefficient_patterns")
    if not isinstance(cp, dict):
        _fail("ledger.coefficient_patterns must be a dict")
    for name, pattern in sorted(cp.items()):
        if not isinstance(pattern, dict):
            _fail(f"coefficient pattern {name!r} is not a dict")
        slots = pattern.get("slots")
        if not isinstance(slots, list) or not slots:
            _fail(f"coefficient pattern {name!r}: slots must be a non-empty list")
        for slot in slots:
            if slot not in VALID_COEFF_VARIANTS:
                _fail(f"coefficient pattern {name!r}: unknown variant {slot!r}")
    return cp


def _validate_evidence(ledger: dict) -> None:
    ev = ledger.get("evidence")
    if not isinstance(ev, dict):
        _fail("ledger.evidence must be a dict")


# ---------------------------------------------------------------------------
# Evidence record verification
# ---------------------------------------------------------------------------

EVIDENCE_IDENTITY_FIELDS = ("source_id", "host_id", "features", "request")


def _evidence_identity(record: dict, kind: str, ev_id: str) -> dict[str, str]:
    """Require a complete run identity on a citation, receipt, or marker."""
    identity: dict[str, str] = {}
    for field in EVIDENCE_IDENTITY_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            _fail(f"evidence {ev_id!r}: {kind} missing {field}")
        identity[field] = value
    if identity["features"] not in VALID_FEATURE_CONFIGS:
        _fail(f"evidence {ev_id!r}: {kind} has invalid features {identity['features']!r}")
    if identity["request"] not in VALID_REQUESTS:
        _fail(f"evidence {ev_id!r}: {kind} has invalid request {identity['request']!r}")
    return identity


def _receipt_identity(record: dict, ev_id: str) -> dict[str, str]:
    """Receipts name the requested tier as ``request_raw``."""
    normalised = dict(record)
    normalised["request"] = record.get("request_raw")
    return _evidence_identity(normalised, "receipt", ev_id)


def _read_evidence_record(ev_id: str, ev: dict, records: Path):
    """Parse one evidence record into its completion marker and receipts."""
    record_file = ev.get("file")
    if not isinstance(record_file, str) or not record_file:
        _fail(f"evidence {ev_id!r}: missing file")
    path = records / record_file
    if not path.is_file():
        _fail(f"evidence {ev_id!r}: record file {record_file} not found")

    done: dict | None = None
    receipts: list[dict] = []
    for line in path.read_text().splitlines():
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
    return done, receipts



# ---------------------------------------------------------------------------
# One pipeline: load / obligations / cases
#
# ``load`` parses and raw-validates the ledger. ``obligations`` expands
# applicability with overrides merged and validated. ``cases`` expands
# fixtures, geometries, and valid coefficient patterns with budgets resolved
# per fixture width. ``check``, ``generate``, and ``accept`` all derive from
# the same ``obligations()`` and ``cases()``.
# ---------------------------------------------------------------------------
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
        callable_filter = app.get("callable_filter")
        if callable_filter is not None and not isinstance(callable_filter, list):
            _fail(f"applicability {app_name!r}: callable_filter must be a list")

        for rule_id in app_rules:
            if rule_id in exclusions:
                continue
            if rule_id not in rules:
                continue

            rule = rules[rule_id]
            if rule.get("non_bulk"):
                continue
            rule_callables = rule.get("callables")
            if rule_callables is not None and not isinstance(rule_callables, list):
                _fail(f"rule {rule_id!r}: callables must be a list")

            for family in families:
                callables = callables_by_family.get(family, [])
                for call_name in callables:
                    if callable_filter is not None and call_name not in callable_filter:
                        continue
                    if rule_callables is not None and call_name not in rule_callables:
                        continue
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

# ---------------------------------------------------------------------------
# Pipeline dataclasses: Obligation / Cell / WorkSpec / Case
# ---------------------------------------------------------------------------

VALID_COEFF_VARIANTS = frozenset({"General", "Zero", "One"})
VALID_REFERENCES = frozenset({"elem-arith", "map-compact", "bits-scalar"})


_TYPE_CHARACTERISTIC = re.compile(
    r"FiniteField\s*<\s*(0[xX][0-9A-Fa-f_]+|\d[\d_]*)")


def _type_characteristic(text: str | None) -> int | None:
    """Characteristic named by a ``FiniteField`` type expression."""
    match = _TYPE_CHARACTERISTIC.search(text or "")
    if not match:
        return None
    return int(match.group(1).replace("_", ""), 0)


def _fixture_metadata(fixture: dict, rule: dict) -> dict[str, int]:
    """Resolve one fixture's width and arithmetic metadata once.

    Degree and relative degree come from the fixture's explicit metadata;
    type text only supplies the characteristic when no fixture or rule field
    declares it.
    """
    fixture = fixture or {}
    rule = rule or {}
    ftype = fixture.get("type") or fixture.get("source")

    characteristic = fixture.get("characteristic")
    if not isinstance(characteristic, int) or isinstance(characteristic, bool):
        rule_char = rule.get("characteristic")
        characteristic = (rule_char if isinstance(rule_char, int)
                          and not isinstance(rule_char, bool) else None)
    if characteristic is None:
        characteristic = _type_characteristic(
            ftype if isinstance(ftype, str) else None)

    degree = fixture.get("degree")
    relative_degree = fixture.get("relative_degree")
    for key, value in (("degree", degree),
                       ("relative_degree", relative_degree)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(f"fixture {fixture.get('id')!r}: {key} must be a "
                  f"positive integer")

    group = rule.get("group")
    if group in ("map", "map_bits"):
        elem_bytes = fixture.get("elem_bytes")
        target_elem_bytes = fixture.get("target_elem_bytes")
        if elem_bytes is None or target_elem_bytes is None:
            _fail(f"fixture {fixture.get('id')!r}: map fixtures declare "
                  f"elem_bytes and target_elem_bytes")
    else:
        elem_bytes = fixture.get("elem_bytes", rule.get("elem_bytes", 1))
        target_elem_bytes = fixture.get(
            "target_elem_bytes",
            rule.get("target_elem_bytes",
                     fixture.get("elem_bytes", rule.get("elem_bytes", 1))))

    meta: dict[str, int] = {
        "elem_bytes": elem_bytes,
        "target_elem_bytes": target_elem_bytes,
        "relative_degree": relative_degree,
        "power": fixture.get("power", rule.get("power", 1)),
        "words": fixture.get("words", max(1, math.ceil(elem_bytes / 8))),
    }
    if characteristic is not None:
        meta["characteristic"] = characteristic
    if degree is not None:
        meta["degree"] = degree
    return meta


def _is_gather_shape(call_name: str, call_family: str) -> bool:
    """Whether a callable folds several sources into one destination."""
    return (call_family in ("ops/gather", "bits/gather")
            or "gather" in call_name.split("::")[-1])


def _geometry_family(call_family: str, call_name: str = "") -> str:
    if call_family == "bits" and _is_gather_shape(call_name, call_family):
        return "bits-gather"
    mapping = {
        "ops/row": "single-row",
        "ops/scatter": "scatter",
        "ops/gather": "gather",
        "ops/matrix": "matrix",
        "ops/matrix_at": "matrix",
        "ops/reduction": "single-row",
        "ops/elementwise": "single-row",
        "ops/pack": "single-row",
        "ops/rows": "multi-row",
        "ops/scalar": "single-row",
        "ops/component": "single-row",
        "bits": "bits",
        "bits/gather": "bits-gather",
        "map": "single-row",
        "map_bits": "single-row",
    }
    return mapping.get(call_family, "single-row")


def _pattern_families_for(call_family: str) -> list[str]:
    mapping = {
        "ops/row": ["single-general", "single-zero", "single-one"],
        "ops/scatter": ["mixed", "multi-all-general"],
        "ops/gather": ["mixed", "multi-all-general"],
        "ops/matrix": ["mixed"],  # 3x4 mixed only; matrix_at shares it
        "ops/matrix_at": ["mixed"],
        "ops/reduction": ["single-general", "single-zero", "single-one"],
        "ops/elementwise": ["single-general", "single-zero", "single-one"],
        "ops/pack": ["single-general", "single-zero", "single-one"],
        "ops/rows": [],
        "ops/scalar": ["single-general", "single-zero", "single-one"],
        "ops/component": ["single-general", "single-zero", "single-one"],
        "bits": [],
        "bits/gather": [],
        "map": [],
        "map_bits": [],
    }
    return mapping.get(call_family, [])



def _slot_count_for(call_name: str, call_family: str, geo: dict) -> int:
    """Coefficient slots for one case: gather folds sources, scatter fans rows."""
    rows = geo.get("rows", 1)
    sources = geo.get("sources", 1)
    if call_family == "ops/gather":
        return sources
    if call_family == "ops/scatter":
        return rows
    return rows * sources

def _select_geometry_for_family(geos: dict, call_family: str,
                                call_name: str = "") -> tuple[str, dict] | None:
    """One vector-eligible geometry per callable family and shape."""
    want = _geometry_family(call_family, call_name)
    for geo_id, geo in sorted(geos.items()):
        if geo.get("family", "") == want:
            return (geo_id, geo)
    for geo_id, geo in sorted(geos.items()):
        if not geo.get("family", ""):
            return (geo_id, geo)
    return None


def _select_patterns_for(cps_by_family: dict, cps_all: list,
                         call_name: str, call_family: str,
                         meta: dict) -> list[tuple[str, dict]]:
    """Valid coefficient patterns for one fixture and callable shape."""
    takes = _callable_takes_coefficient(call_name, call_family)
    if not takes:
        return [("", {"slots": []})]
    selected: list[tuple[str, dict]] = []
    for cpfam in _pattern_families_for(call_family):
        selected.extend(cps_by_family.get(cpfam, []))
    if not selected:
        selected = list(cps_all)
    is_gf2 = _fixture_is_gf2_value(meta)
    out: list[tuple[str, dict]] = []
    for cp_name, cp in selected:
        slots = cp.get("slots", []) if isinstance(cp, dict) else []
        if is_gf2 and "General" in slots:
            continue
        out.append((cp_name, cp))
    return out


def load(path: Path) -> dict:
    """Parse the ledger JSON and raw-validate its top-level shape."""
    ledger = json.loads(path.read_text())
    if not isinstance(ledger, dict):
        _fail("ledger root must be a dict")
    _validate_isa(ledger)
    _validate_callables(ledger, None)
    rules = _validate_rules(ledger)
    _validate_applicability(ledger, rules)
    _validate_work_rules(ledger)
    _validate_geometries(ledger)
    _validate_coefficient_patterns(ledger)
    _validate_evidence(ledger)
    return ledger


def obligations(ledger: dict) -> dict:
    """Expand applicability into obligations with overrides merged and validated."""
    rules = _validate_rules(ledger)
    wrules = _validate_work_rules(ledger)
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})
    base = _expand_obligations(ledger, rules)
    expanded = {oid: {"cells": dict(ob["cells"])} for oid, ob in base.items()}
    explicit_obs = ledger.get("obligations", {})
    if explicit_obs and not isinstance(explicit_obs, dict):
        _fail("ledger.obligations must be a dict")
    for oid, ob in (explicit_obs or {}).items():
        # Applicability decides membership; an override only annotates a
        # (callable, rule) pair the expansion already produced.
        if oid not in base:
            _fail(f"explicit obligation {oid!r} is not in the expansion; "
                  f"check applicability rules")
        if not isinstance(ob, dict) or not isinstance(ob.get("cells", {}), dict):
            _fail(f"explicit obligation {oid!r}: cells must be a dict")
        for key, cell in (ob.get("cells", {}) or {}).items():
            if not isinstance(cell, dict):
                _fail(f"explicit obligation {oid!r} cell {key!r} is not a dict")
            merged = dict(expanded[oid]["cells"].get(key, {}), **cell)
            expanded[oid]["cells"][key] = merged
    _validate_expanded_obligations(ledger, expanded, rules, wrules, geos, coeffs)
    _validate_bulk_family_coverage(ledger, expanded)
    return expanded


def cases(ledger: dict, expanded: dict | None = None) -> list[dict]:
    """Expand fixtures, geometries, and valid patterns into case records."""
    rules = _validate_rules(ledger)
    wrules = ledger.get("work_rules", {})
    isa = ledger.get("isa", {})
    if expanded is None:
        expanded = obligations(ledger)
    geos = ledger.get("geometries", {})
    coeffs = ledger.get("coefficient_patterns", {})
    callables = ledger.get("callables", {})
    callable_families = {n: e.get("family", "") for n, e in callables.items() if e.get("family")}
    cps_by_family: dict[str, list[tuple]] = {}
    cps_all: list[tuple] = []
    for cp_name, cp in (coeffs or {}).items():
        cpfam = cp.get("family", "") if isinstance(cp, dict) else ""
        cps_by_family.setdefault(cpfam, []).append((cp_name, cp))
        if not cpfam:
            cps_all.append((cp_name, cp))
    out: list[dict] = []
    for oid, ob in sorted(expanded.items()):
        if "@" not in oid:
            _fail(f"obligation {oid!r}: missing @ separator")
        call_name, rule_id = oid.split("@", 1)
        rule = rules.get(rule_id, {})
        if not isinstance(rule, dict):
            _fail(f"obligation {oid!r}: unknown rule {rule_id!r}")
        call_family = callable_families.get(call_name, "")
        takes = _callable_takes_coefficient(call_name, call_family)
        geo_sel = _select_geometry_for_family(geos, call_family, call_name)
        if geo_sel is None:
            if takes:
                _fail(f"obligation {oid!r}: no geometry for family {call_family!r}")
            continue
        geo_id, geo = geo_sel
        for fixture in rule.get("fixtures", []):
            fid = fixture.get("id", "")
            meta = _fixture_metadata(fixture, rule)
            patterns = _select_patterns_for(cps_by_family, cps_all,
                                            call_name, call_family, meta)
            patterns = [p for p in patterns if isinstance(p[1], dict)]
            if takes:
                patterns = [p for p in patterns if p[1].get("slots")]
                if not patterns:
                    _fail(f"no_valid_pattern obligation {oid!r} fixture {fid!r}")
                need = _slot_count_for(call_name, call_family, geo)
                sized = [p for p in patterns if len(p[1].get("slots", [])) == need]
                if not sized:
                    _fail(f"no_sized_pattern obligation {oid!r} fixture {fid!r} "
                          f"geometry {geo_id!r} need {need}")
                patterns = sized
            cells = ob.get("cells", {})
            for cp_name, cp in patterns:
                case_id = f"{oid}#{fid}/{geo_id}/{cp_name or '_'}"
                out.append({
                    "id": case_id,
                    "obligation": oid,
                    "callable": call_name,
                    "rule_id": rule_id,
                    "fixture": fixture,
                    "geometry_id": geo_id,
                    "geometry": geo,
                    "pattern_id": cp_name,
                    "pattern": cp,
                    "cells": cells,
                    "cell_work": _resolve_case_work(
                        case_id, oid, cells, wrules, isa, meta, geo, cp,
                        gather=_is_gather_shape(call_name, call_family)),
                })
    return out


def _resolve_case_work(case_id: str, oid: str, cells: dict, wrules: dict,
                       isa: dict, meta: dict, geo: dict, pattern: dict,
                       gather: bool = False) -> dict:
    """Resolve every referenced work rule for one case and validate its budget."""
    resolved: dict[str, list[dict]] = {}
    for key, cell in sorted(cells.items()):
        req = key.split("/", 1)[1]
        items: list[dict] = []
        for wr_name in cell.get("work", []):
            wr = wrules.get(wr_name)
            if not isinstance(wr, dict):
                continue
            try:
                item = _resolve_work(wr, geo, meta, req, isa, pattern, gather)
            except ValueError as exc:
                _fail(f"obligation {oid!r} cell {key!r} case {case_id!r}: "
                      f"work rule {wr_name!r}: {exc}")
            if item["iterations"] <= 0:
                _fail(f"obligation {oid!r} cell {key!r} case {case_id!r}: "
                      f"work rule {wr_name!r} has non-positive iterations "
                      f"({item['iterations']})")
            status = cell.get("status", "unverified")
            # A zero-multiplicity composition requires no vector work, and a
            # cell that claims no route need not cover one.
            if item["multiplicity_count"] > 0 and status not in (
                    "missing", "bypass", "zero-budget") and item["budget"] <= 0:
                _fail(f"obligation {oid!r} cell {key!r} case {case_id!r}: "
                      f"non-positive budget for work rule {wr_name!r} "
                      f"(budget={item['budget']})")
            items.append({"rule": wr_name, **item})
        resolved[key] = items
    return resolved




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
    """Validate the ledger. Both modes derive from obligations() and cases()."""
    if schema_only:
        _validate_callables(ledger, rustdoc)
        expanded = obligations(ledger)
        _validate_geometries(ledger)
        _validate_coefficient_patterns(ledger)
        _validate_rule_references(ledger)
        cases(ledger, expanded)

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

    # Full schema validation derives from the same pipeline as schema-only.
    _validate_callables(ledger, rustdoc)
    expanded = obligations(ledger)
    _validate_geometries(ledger)
    _validate_coefficient_patterns(ledger)
    _validate_rule_references(ledger)
    case_list_derived = cases(ledger, expanded)

    _check_full_references(ledger, expanded, case_list_derived,
                           test_names, case_ids, surface_ids, records)

    n_cells = _count_cells(expanded)
    print(f"simd_ledger: full check passed  obligations={len(expanded)} cells={n_cells}")
    return 0


def _ledger_references(ledger: dict) -> dict:
    """Ledger-level citation sets: cases, tests, and benchmark ids."""
    refs = ledger.get("references", {})
    if not isinstance(refs, dict):
        _fail("ledger.references must be a dict")
    for key in ("cases", "tests", "bench"):
        value = refs.get(key, [])
        if not isinstance(value, list):
            _fail(f"ledger.references.{key} must be a list")
        for entry in value:
            if not isinstance(entry, str) or not entry:
                _fail(f"ledger.references.{key} entries must be non-empty strings")
    return refs


def _validate_rule_references(ledger: dict) -> None:
    """Every rule-declared callable and reference identity resolves."""
    callables = ledger.get("callables", {})
    rules = ledger.get("rules", {})
    for rid, rule in sorted(rules.items()):
        for name in rule.get("callables", []) or []:
            if name not in callables:
                _fail(f"rule {rid!r}: callable {name!r} is not a classified callable")
        for fix in rule.get("fixtures", []):
            ref = fix.get("reference")
            if ref is not None and ref not in VALID_REFERENCES:
                _fail(f"rule {rid!r}: fixture {fix.get('id')!r} has unknown reference {ref!r}")


def _route_claim_mismatch(rec: dict, case: dict, features: str,
                          request: str) -> str | None:
    """Compare one receipt's route claims with the case's cell."""
    cell = case.get("cells", {}).get(f"{features}/{request}")
    if not isinstance(cell, dict):
        return "receipt_cell_missing"
    if rec.get("process") != cell.get("process"):
        return "receipt_process_mismatch"
    if rec.get("field") != cell.get("field"):
        return "receipt_field_mismatch"
    if rec.get("required") != cell.get("required"):
        return "receipt_required_mismatch"
    if "min_serving" in rec and rec["min_serving"] != cell.get("min_serving"):
        return "receipt_min_serving_mismatch"
    return None


def _check_full_references(ledger: dict, expanded: dict, case_list_derived: list[dict],
                           test_names: set[str], runner_case_ids: set[str],
                           surface_ids: set[str], records: Path | None) -> None:
    """Resolve every citation against runner, test, and benchmark inventories."""
    refs = _ledger_references(ledger)
    for cid in refs.get("cases", []):
        # A generated case is only resolved when the runner also printed it.
        if cid not in runner_case_ids:
            _fail(f"reference case {cid!r} is not in the runner case list")
    for tid in refs.get("tests", []):
        if tid not in test_names:
            _fail(f"reference test {tid!r} is not in the test list")
    for bid in refs.get("bench", []):
        if bid not in surface_ids:
            _fail(f"reference bench {bid!r} is not in the surface inventory")
    _check_evidence_citations(ledger, expanded, case_list_derived, records,
                              runner_case_ids)


def _check_evidence_citations(ledger: dict, expanded: dict, case_list_derived: list[dict],
                             records: Path | None,
                             runner_case_ids: set[str] | None = None) -> None:
    """Evidence entries cite complete, consistent record identities.

    Every receipt in a record, cited or not, must carry the same complete
    identity as the completion marker; the citation repeats that identity.
    """
    evidence = ledger.get("evidence", {})
    if not evidence:
        return
    if records is None:
        _fail("--records is required when ledger cites evidence")
    if not records.is_dir():
        _fail(f"--records {records} is not a directory")
    by_case = {c["id"]: c for c in case_list_derived}
    for ev_id, ev in sorted(evidence.items()):
        if not isinstance(ev, dict):
            _fail(f"evidence {ev_id!r} is not a dict")
        citations = ev.get("receipts")
        if not isinstance(citations, list) or not citations:
            _fail(f"evidence {ev_id!r}: receipts must be a non-empty list")
        completion = ev.get("completion")
        if not isinstance(completion, dict):
            _fail(f"evidence {ev_id!r}: completion must be a dict")
        completion_identity = _evidence_identity(completion, "completion", ev_id)
        completion_cases = completion.get("cases")
        if (not isinstance(completion_cases, int) or isinstance(completion_cases, bool)
                or completion_cases < 0):
            _fail(f"evidence {ev_id!r}: completion missing cases")

        done, receipts = _read_evidence_record(ev_id, ev, records)
        done_identity = _evidence_identity(done, "ROUTE_DONE", ev_id)
        done_cases = done.get("cases")
        if (not isinstance(done_cases, int) or isinstance(done_cases, bool)
                or done_cases < 0):
            _fail(f"evidence {ev_id!r}: ROUTE_DONE missing cases")
        if done_cases != len(receipts):
            _fail(f"evidence {ev_id!r}: ROUTE_DONE count {done_cases} "
                  f"does not match {len(receipts)} receipts")
        if completion_cases != done_cases:
            _fail(f"evidence {ev_id!r}: completion cases mismatch")
        for field in EVIDENCE_IDENTITY_FIELDS:
            if completion_identity[field] != done_identity[field]:
                _fail(f"evidence {ev_id!r}: completion {field} mismatch")

        # Record-wide identity and uniqueness.
        seen: set[str] = set()
        by_rid: dict[str, dict] = {}
        for rec in receipts:
            if not isinstance(rec, dict):
                _fail(f"evidence {ev_id!r}: receipt is not a dict")
            rid = rec.get("id")
            if not isinstance(rid, str) or not rid:
                _fail(f"evidence {ev_id!r}: receipt without id field")
            if rid in seen:
                _fail(f"evidence {ev_id!r}: duplicate receipt {rid!r}")
            seen.add(rid)
            receipt_identity = _receipt_identity(rec, ev_id)
            result = rec.get("result")
            if not isinstance(result, str) or not result:
                _fail(f"evidence {ev_id!r}: receipt {rid} missing result")
            for field in EVIDENCE_IDENTITY_FIELDS:
                if receipt_identity[field] != done_identity[field]:
                    _fail(f"evidence {ev_id!r}: receipt {rid} {field} "
                          f"differs from ROUTE_DONE")
            by_rid[rid] = rec

        for cited in citations:
            if not isinstance(cited, dict):
                _fail(f"evidence {ev_id!r}: receipt expectation is not a dict")
            rid = cited.get("id")
            if not isinstance(rid, str) or not rid:
                _fail(f"evidence {ev_id!r}: receipt expectation missing id")
            citation_identity = _evidence_identity(cited, "citation", ev_id)
            if rid not in by_case:
                _fail(f"evidence {ev_id!r}: cited receipt {rid!r} is not a generated case")
            if runner_case_ids is not None and rid not in runner_case_ids:
                _fail(f"evidence {ev_id!r}: cited receipt {rid!r} is not in "
                      f"the runner case list")
            rec = by_rid.get(rid)
            if rec is None:
                _fail(f"evidence {ev_id!r}: cited receipt {rid} missing")
            if rec.get("result") != "pass":
                _fail(f"evidence {ev_id!r}: cited receipt {rid} result is not pass")
            case = by_case[rid]
            if rec.get("obligation") != case["obligation"]:
                _fail(f"evidence {ev_id!r}: cited receipt {rid} obligation mismatch")
            receipt_identity = _receipt_identity(rec, ev_id)
            for field in EVIDENCE_IDENTITY_FIELDS:
                if receipt_identity[field] != citation_identity[field]:
                    _fail(f"evidence {ev_id!r}: receipt {rid} {field} "
                          f"differs from its citation")
            code = _route_claim_mismatch(rec, case, receipt_identity["features"],
                                         receipt_identity["request"])
            if code is not None:
                _fail(f"evidence {ev_id!r}: cited receipt {rid} {code}")


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


def _work_iterations(wr: dict, fixture_vars: dict | None = None) -> int:
    """Repetitions from the work rule's iterations expression.

    Zero or negative results are rejected, never clamped: a rule that asks
    for no repetitions has no budget to cover any work.
    """
    iterations = wr.get("iterations", "1")
    try:
        return int(iterations)
    except (TypeError, ValueError):
        pass
    if fixture_vars is None:
        return _evaluate_expr(iterations, {})
    return _evaluate_expr(iterations, fixture_vars)


def _fixture_is_gf2_value(meta: dict[str, int]) -> bool:
    """True only for a fully known degree-1 GF(2) fixture shape."""
    return meta.get("characteristic") == 2 and meta.get("degree") == 1


def _general_slot_count(coeff_pattern: dict) -> int:
    return sum(1 for s in coeff_pattern.get("slots", []) if s == "General")


def _count_active_pairs(coeff_pattern: dict) -> int:
    """Count General slots (active coefficient pairs)."""
    return _general_slot_count(coeff_pattern)


def _count_unit_add_pairs(coeff_pattern: dict) -> int:
    """Count One slots (unit accumulate pairs)."""
    return sum(1 for s in coeff_pattern.get("slots", []) if s == "One")


def _callable_takes_coefficient(call_name: str, family: str = "") -> bool:
    """True when the callable carries a coefficient slot.

    Coefficient forms are ``mul_*`` row/fan shapes, ``mul_*_with``
    prepared-coefficient forms, ``*_scalar`` row forms, the elementwise
    multiply/divide pair, ``pack``/``unpack`` (element slots), and the single
    ``dot_product`` reduction. In-place add/sub, row copies, bit-range and
    weight queries, and map runners carry no coefficient slots.
    """
    base = call_name.split("::")[-1]
    if call_name.startswith("bits::") or call_name.startswith("map"):
        return False
    if base in ("pack", "pack_to_vec", "unpack", "unpack_to_vec",
                "mul_elementwise", "mul_elementwise_assign",
                "div_elementwise", "div_elementwise_assign"):
        return True
    if base in ("dot_product", "add_assign_rows", "sub_assign_rows",
                "add_assign_scalar", "sub_assign_scalar", "add_gather_offsets"):
        return False
    return base.startswith("mul_") or base.endswith("_with") or base.endswith("_scalar")


def _active_rows_count(geo: dict, coeff_pattern: dict) -> int:
    """Count rows with >= 1 nontrivial (General or One) coefficient slot."""
    rows = geo.get("rows", 1)
    sources = geo.get("sources", 1)
    slots = coeff_pattern.get("slots", [])
    if len(slots) != rows * sources:
        return 0
    count = 0
    for r in range(rows):
        row_slots = slots[r * sources:(r + 1) * sources]
        if any(s in ("General", "One") for s in row_slots):
            count += 1
    return count


def _multiplicity_count(mult: str, geo: dict, coeff_pattern: dict,
                        gather: bool = False) -> int:
    if mult == "Rows":
        return geo.get("rows", 1)
    if mult == "ActiveRows":
        slots = coeff_pattern.get("slots", [])
        if gather:
            # A gather folds selected sources: coefficients name sources,
            # and a coefficient-free callable selects every source.
            if not slots:
                return geo.get("sources", 1)
            return sum(1 for slot in slots if slot in ("General", "One"))
        if not slots:
            return geo.get("rows", 1)
        return _active_rows_count(geo, coeff_pattern)
    if mult == "ActivePairs":
        return _count_active_pairs(coeff_pattern)
    if mult == "UnitAddPairs":
        return _count_unit_add_pairs(coeff_pattern)
    return 1  # One


def _loss_in_units(unit: str, loss_bytes: int, elem_bytes: int) -> int:
    """Convert loss_bytes to the work rule's unit. Loss rounds up (ceil)."""
    if loss_bytes <= 0:
        return 0
    if unit == "Bytes":
        return loss_bytes
    if unit == "Bits":
        return loss_bytes * 8
    if unit == "Words":
        return math.ceil(loss_bytes / 8)
    if elem_bytes <= 1:
        return loss_bytes
    return math.ceil(loss_bytes / elem_bytes)


def _lane_in_units(unit: str, lane_bytes: int, elem_bytes: int) -> int:
    """Convert lane_bytes to the work rule's unit. Lane rounds down (floor)."""
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


def _length_in_units(unit: str, elements: int, elem_bytes: int) -> int:
    """Convert a geometry's length to the work rule's unit.

    A ``Bits`` geometry is already measured in bits; the element width only
    converts lane and loss for the byte-based work rules. Other units measure
    the geometry's element count.
    """
    if unit == "Elements":
        return elements
    if unit == "Bits":
        return elements
    if unit == "Bytes":
        return elements * elem_bytes
    if unit == "Words":
        return max(1, elements * elem_bytes // 8)
    raise ValueError(f"unknown work unit: {unit!r}")


def _resolve_work(work_rule: dict, geo: dict, meta: dict,
                  tier: str, isa: dict,
                  coeff_pattern: dict | None = None,
                  gather: bool = False) -> dict:
    """Resolve one Work item for a case: lane, loss, iterations, and budget.

    ``basis: Target`` converts the geometry length and the loss with the
    target element width; ``basis: Source`` uses the source width.
    """
    basis = work_rule["basis"]
    unit = work_rule["unit"]
    width_eb = meta["target_elem_bytes"] if basis == "Target" else meta["elem_bytes"]
    lane_bytes = (isa.get("lane_bytes", {}) or {}).get(tier) if isa else None
    if not isinstance(lane_bytes, int) or lane_bytes <= 0:
        raise ValueError(f"no lane width for tier {tier!r}")
    loss_bytes = _compute_loss_bytes(work_rule, lane_bytes, meta)
    iterations = _work_iterations(work_rule, meta)
    lane = _lane_in_units(unit, lane_bytes, width_eb)
    loss = _loss_in_units(unit, loss_bytes, width_eb)
    length = _length_in_units(unit, geo.get("len", 0), width_eb)
    count = _multiplicity_count(work_rule["multiplicity"], geo,
                                coeff_pattern or {"slots": []}, gather)
    scope = work_rule.get("scope")
    return {
        "substage": work_rule["substage"],
        "scope": (scope.get("scope"), scope.get("pass")) if scope else None,
        "unit": unit,
        "basis": basis,
        "multiplicity": work_rule["multiplicity"],
        "lane": lane,
        "loss": loss,
        "iterations": iterations,
        "multiplicity_count": count,
        "budget": iterations * count * (length - loss),
    }


def _work_budget(work_rule: dict, geo: dict, source_width: dict,
                 target_width: dict, tier: str, isa: dict,
                 coeff_pattern: dict | None = None,
                 fixture: dict | None = None, rule: dict | None = None) -> int:
    """Budget for one Work item: iterations * mult * (len_in_units - loss)."""
    meta = _fixture_metadata(fixture or {}, rule or {})
    meta["elem_bytes"] = source_width["elem_bytes"]
    meta["target_elem_bytes"] = target_width["elem_bytes"]
    return _resolve_work(work_rule, geo, meta, tier, isa,
                         coeff_pattern or {"slots": []})["budget"]


def _adapter_for(call_name: str, fixture: dict, rule: dict,
                 family: str = "") -> tuple[str, str, str]:
    """Adapter module, runner function, and type arguments for one case.

    Adapters are named ``<module>_<callable>`` and grouped in modules named
    for the operation family they run: rows, fans, elementwise, scalars,
    reductions, packing, bits, maps, map_bits. Map cases carry ``S, T``
    (or ``F``) type arguments from the fixture's source/target types.
    """
    module, _, base = call_name.partition("::")
    family_adapters = {
        "ops/row": "rows",
        "ops/rows": "rows",
        "ops/scatter": "fans",
        "ops/gather": "fans",
        "ops/matrix": "fans",
        "ops/matrix_at": "fans",
        "ops/elementwise": "elementwise",
        "ops/scalar": "scalars",
        "ops/reduction": "reductions",
        "ops/pack": "packing",
        "ops/component": "rows",
    }
    if module == "ops" and family in family_adapters:
        adapter = family_adapters[family]
    elif module == "ops":
        if base.startswith("mul_add_scatter") or base.startswith("mul_into_scatter"):
            adapter = "fans"
        elif base.startswith("mul_add_gather") or base.startswith("mul_into_gather"):
            adapter = "fans"
        elif base.startswith("mul_add_matrix") or base.startswith("mul_into_matrix"):
            adapter = "fans"
        elif base.startswith("mul_add") or base.startswith("mul_into") or base.startswith("mul_assign"):
            adapter = "rows"
        elif base.startswith("add_assign") or base.startswith("sub_assign"):
            adapter = "rows"
        elif "elementwise" in base or "inv_" in base or "div_" in base:
            adapter = "elementwise"
        elif base.endswith("_scalar"):
            adapter = "scalars"
        elif base in ("pack", "pack_to_vec", "unpack", "unpack_to_vec"):
            adapter = "packing"
        elif base == "dot_product":
            adapter = "reductions"
        else:
            adapter = "rows"
    elif module == "bits":
        adapter = "bits"
    elif module == "map":
        adapter = "maps"
    elif module == "map_bits":
        adapter = "map_bits"
    else:
        adapter = module
    # Runner functions are named ``<module>_<callable>``.
    fn = call_name.replace("::", "_")
    if module in ("map", "map_bits"):
        src_t = fixture.get("source", fixture.get("type", ""))
        tgt_t = fixture.get("target", fixture.get("type", ""))
        if base.startswith("frobenius"):
            type_args = f"{src_t}"
        else:
            type_args = f"{src_t}, {tgt_t}"
    else:
        type_args = fixture.get("type", fixture.get("source", ""))
    return (adapter, fn, type_args)


def _work_const(name: str, key: tuple) -> list[str]:
    """Render one ``const Work`` value from a resolved work tuple."""
    substage, scope, unit, basis, mult, lane, loss, iterations = key
    if scope is None:
        scope_expr = "None"
    else:
        scope_expr = f"Some((Scope::{scope[0]}, Pass::{scope[1]}))"
    return [
        "#[rustfmt::skip]",
        f"const {name}: Work = Work {{",
        f"    substage: Substage::{substage},",
        f"    scope: {scope_expr},",
        f"    unit: Unit::{unit},",
        f"    basis: Basis::{basis},",
        f"    multiplicity: Multiplicity::{mult},",
        f"    lane: {lane},",
        f"    loss: {loss},",
        f"    iterations: {iterations},",
        "};",
        "",
    ]


def _cells_const(name: str, cells_key: tuple) -> list[str]:
    """Render one ``const &[Cell]`` set from resolved cell tuples."""
    lines = ["#[rustfmt::skip]", f"const {name}: &[Cell] = &["]
    for fc, req, process, field, required, min_serving, refs in cells_key:
        features = "Features::Default" if fc == "default" else "Features::All"
        request = f"Request::{_request_rust_name(req)}"
        process_backend = f"Backend::{_backend_rust_name(process)}"
        field_backend = f"Backend::{_backend_rust_name(field)}"
        min_serving_backend = f"Backend::{_backend_rust_name(min_serving)}"
        required_str = "true" if required else "false"
        work_array = "&[" + ", ".join(refs) + "]" if refs else "&[]"
        lines.append(
            f"    Cell {{ features: {features}, request: {request}, "
            f"process: {process_backend}, field: {field_backend}, "
            f"required: {required_str}, min_serving: {min_serving_backend}, "
            f"work: {work_array} }},")
    lines.extend(["];", ""])
    return lines


def _map_spec(rule: dict, fixture: dict) -> tuple[int, list[list[int]]]:
    """Resolve a map case's construction data from the fixture, then the rule."""
    power = fixture.get("power", rule.get("power", 0))
    images = fixture.get("images", rule.get("images", []))
    return (int(power), [[int(byte) for byte in image] for image in images])


def _generate_cases(ledger: dict) -> str:
    """Emit the generated route case table from the pipeline.

    Renders the cases() expansion: one case per fixture, geometry, and
    valid coefficient pattern. Deduplicates cell sets, geometries, and
    coefficient arrays. Cases only go to CASES_SIMD512 when their consts
    reference simd512-gated items; numeric v4x lane/loss do not qualify.
    """
    rules = _validate_rules(ledger)
    expanded = obligations(ledger)
    derived = cases(ledger, expanded)
    callable_families = {name: entry.get("family", "")
                         for name, entry in ledger.get("callables", {}).items()
                         if entry.get("family")}

    work_defs: dict[tuple, str] = {}
    work_lines: list[str] = []
    map_defs: dict[tuple, str] = {}
    map_lines: list[str] = []
    run_defs: dict[str, str] = {}
    run_lines: list[str] = []
    cells_map: dict[tuple, str] = {}
    cells_lines: list[str] = []
    geos_map: dict[tuple, str] = {}
    geo_lines: list[str] = []
    coeffs_map: dict[tuple, str] = {}
    coeff_lines: list[str] = []
    case_records: list[dict] = []

    # Resolve case records; work values come from the pipeline's budgets.
    for case in derived:
        oid = case["obligation"]
        call_name = case["callable"]
        rule = rules.get(case["rule_id"], {})
        fixture = case["fixture"]
        geo = case["geometry"]
        cp = case["pattern"]

        cell_tuples = []
        for key, cell in sorted(case["cells"].items()):
            fc, req = key.split("/", 1)
            process = cell.get("process", req)
            field = cell.get("field", process)
            work_refs: list[str] = []
            for item in case["cell_work"].get(key, []):
                work_key = (item["substage"], item["scope"], item["unit"],
                            item["basis"], item["multiplicity"], item["lane"],
                            item["loss"], item["iterations"])
                if work_key not in work_defs:
                    name = f"WORK_{len(work_defs)}"
                    work_defs[work_key] = name
                    work_lines.extend(_work_const(name, work_key))
                work_refs.append(work_defs[work_key])
            cell_tuples.append((fc, req, process, field,
                                cell.get("required", False),
                                cell["min_serving"], tuple(work_refs)))

        cells_key = tuple(cell_tuples)
        if cells_key not in cells_map:
            name = f"CELLS_{len(cells_map)}"
            cells_map[cells_key] = name
            cells_lines.extend(_cells_const(name, cells_key))

        geo_key = (geo.get("rows", 1), geo.get("sources", 1),
                   geo.get("len", 0), geo.get("misalign", 0),
                   geo.get("pitch_pad", 0))
        if geo_key not in geos_map:
            name = f"GEO_{len(geos_map)}"
            geos_map[geo_key] = name
            rows, sources, length, misalign, pitch_pad = geo_key
            geo_lines.extend([
                "#[rustfmt::skip]",
                f"const {name}: Geometry = Geometry {{ rows: {rows}, "
                f"sources: {sources}, len: {length}, misalign: {misalign}, "
                f"pitch_pad: {pitch_pad} }};",
                "",
            ])

        coeff_key = tuple(cp.get("slots", []))
        if coeff_key not in coeffs_map:
            name = f"COEFFS_{len(coeffs_map)}"
            coeffs_map[coeff_key] = name
            entries = ", ".join(f"Coeff::{slot}" for slot in coeff_key)
            coeff_lines.extend([
                "#[rustfmt::skip]",
                f"const {name}: &[Coeff] = &[{entries}];",
                "",
            ])

        map_expr = "None"
        if rule.get("group") in ("map", "map_bits"):
            power, images = _map_spec(rule, fixture)
            map_key = (power, tuple(tuple(image) for image in images))
            if map_key not in map_defs:
                name = f"MAP_{len(map_defs)}"
                map_defs[map_key] = name
                images_expr = ", ".join(
                    "&[" + ", ".join(str(byte) for byte in image) + "]"
                    for image in images)
                map_lines.extend([
                    "#[rustfmt::skip]",
                    f"const {name}: MapSpec = MapSpec {{ power: {power}, "
                    f"images: &[{images_expr}] }};",
                    "",
                ])
            map_expr = f"Some({map_defs[map_key]})"

        family = callable_families.get(call_name, "")
        adapter, fn, type_args = _adapter_for(call_name, fixture, rule, family)
        run_fn = f"super::{adapter}::{fn}::<{type_args}>"
        if run_fn not in run_defs:
            name = f"RUN_{len(run_defs)}"
            run_defs[run_fn] = name
            run_lines.extend([
                "#[rustfmt::skip]",
                f"const {name}: fn(&Case, &Cell) -> Outcome = {run_fn};",
                "",
            ])

        case_records.append({
            "id": case["id"],
            "obligation": oid,
            "run_ref": run_defs[run_fn],
            "cells_ref": cells_map[cells_key],
            "geo_ref": geos_map[geo_key],
            "coeffs_ref": coeffs_map[coeff_key],
            "map_expr": map_expr,
        })

    # Render cases — one compact line each (as few as rustfmt allows).
    case_lines: list[str] = []
    for rec in case_records:
        case_lines.append(
            f"    Case {{ id: \"{rec['id']}\", obligation: \"{rec['obligation']}\", "
            f"run: {rec['run_ref']}, geometry: {rec['geo_ref']}, "
            f"coeffs: {rec['coeffs_ref']}, map: {rec['map_expr']}, "
            f"cells: {rec['cells_ref']} }},")

    values = "\n".join(work_lines + map_lines + run_lines + geo_lines
                       + coeff_lines + cells_lines + case_lines)
    used = {name for name in ROUTE_TYPE_NAMES
            if re.search(rf"\b{name}\b", values)}
    used.add("Case")

    lines: list[str] = [
        "// @generated by benches/simd_ledger.py; edit the ledger, not this file.",
        "",
    ]
    if "Backend" in used:
        used.discard("Backend")
        lines.append("use crate::kernel::Backend;")
    if used:
        lines.append("use crate::kernel::tests::routes::{")
        lines.append("    " + ", ".join(sorted(used)) + ",")
        lines.append("};")
    lines.append("")
    lines.extend(work_lines)
    lines.extend(map_lines)
    lines.extend(run_lines)
    lines.extend(geo_lines)
    lines.extend(coeff_lines)
    lines.extend(cells_lines)
    lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\"))]")
    lines.append("#[rustfmt::skip]")
    lines.append("pub(super) static CASES: &[Case] = &[")
    lines.extend(case_lines)
    lines.extend(["];", ""])
    # Cases whose const paths or kernels need a wider target live here.
    lines.append("#[cfg(all(test, target_arch = \"x86_64\", feature = \"simd\", feature = \"simd512\"))]")
    lines.append("#[rustfmt::skip]")
    lines.append("pub(super) static CASES_SIMD512: &[Case] = &[];")
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

    expanded = obligations(ledger)
    derived = cases(ledger, expanded)

    # Required set: cases() filtered to the requested cell's required flag.
    # Generation and acceptance share the same universe.
    by_id = {c["id"]: c for c in derived}
    required_cases: set[str] = set()
    for case in derived:
        cell = case["cells"].get(f"{features}/{request}", {})
        if cell.get("required"):
            required_cases.add(case["id"])

    if not required_cases:
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
        cid = rec.get("id", "")
        received_ids.add(cid)
        # A receipt counts only if its case id is known and its obligation
        # equals that case's obligation.
        case = by_id.get(cid)
        if case is None:
            _fail(f"accept: reject missing_required {sorted(required_cases - received_ids)}")
        if rec.get("obligation") != case["obligation"]:
            _fail(f"accept: reject receipt_obligation_mismatch {cid}")
        oid = case["obligation"]

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

        # Process, field, required, and min-serving expectations come from
        # the case's cell, never from the receipt.
        code = _route_claim_mismatch(rec, case, features, request)
        if code is not None:
            _fail(f"accept: reject {code} {cid}")

        result = rec.get("result")
        if not isinstance(result, str):
            _fail(f"accept: reject receipt_no_result {cid}")

        # result must be exactly 'pass', 'fail', or 'skip'
        if result == "pass":
            pass
        elif result == "fail":
            _fail(f"accept: reject fail_receipt {cid}")
        elif result == "skip":
            _fail(f"accept: reject skip_receipt {cid}")
        else:
            _fail(f"accept: reject unknown_result {cid}")

    missing = required_cases - received_ids
    if missing:
        _fail(f"accept: reject missing_required {sorted(missing)}")

    return 0


# ---------------------------------------------------------------------------
# Fixture compilation check
# ---------------------------------------------------------------------------

FIXTURE_MARKER = re.compile(
    r"crate::kernel::tests::routes::fixtures::(\w+)")


def _scratch_type(text: str) -> str:
    """A ledger type expression rooted in the scratch crate."""
    text = FIXTURE_MARKER.sub(r"crate::\1", text)
    return text.replace("crate::field::", "fgf::field::")


def _spec_marker(text: str) -> str | None:
    """The first route-fixture marker named by a type expression."""
    found = FIXTURE_MARKER.findall(text or "")
    return found[0] if found else None


def _spec_relation(spec: dict) -> str:
    """A const relation constructor appropriate to the spec's base field."""
    base = _scratch_type(spec["base"])
    values = ", ".join(str(value) for value in spec["relation"])
    characteristic = _type_characteristic(spec["base"])
    if characteristic == 2:
        return (f"fgf::Relation::from_binary_words::<{base}>"
                f"(&[{values}])")
    return f"fgf::Relation::from_coordinates(&[{values}])"


def cmd_fixtures(ledger: dict, out_dir: Path) -> int:
    """Write a scratch crate with all fixture types and compile-check them."""
    rules = _validate_rules(ledger)
    out_dir.mkdir(parents=True, exist_ok=True)

    type_widths: dict[str, set[int]] = {}
    map_fixtures: list[dict] = []
    specs: dict[str, dict] = {}
    used_types: list[str] = []
    valid_types: set[str] = set()

    def want(text: str, width: int | None, buffer: bool = True) -> str:
        scratch = _scratch_type(text)
        used_types.append(text)
        valid_types.add(scratch)
        if buffer and isinstance(width, int) and width > 0:
            type_widths.setdefault(scratch, set()).add(width)
        return scratch

    for rid, rule in sorted(rules.items()):
        group = rule.get("group", "")
        for fix in rule.get("fixtures", []):
            if "spec" in fix:
                own = fix.get("target") if group in ("map", "map_bits") else fix.get("type")
                marker = (_spec_marker(own or "")
                          or _spec_marker(fix.get("type") or fix.get("source") or ""))
                if marker is None:
                    _fail(f"rule {rid!r}: fixture {fix.get('id')!r} declares a "
                          f"spec but no route-fixture marker")
                structural = {key: fix["spec"].get(key)
                              for key in ("base", "degree", "relation", "storage")}
                structural["name"] = marker
                if marker in specs and specs[marker] != structural:
                    _fail(f"fixture marker {marker} has conflicting spec data")
                specs[marker] = structural

            if group in ("map", "map_bits"):
                src = fix["source"]
                tgt = fix["target"]
                # A degree-one binary source has no FieldBuffer; its width is
                # still a declared fixture fact and its field validity holds.
                src_buffer = not re.search(r"FiniteField<\s*2\s*,\s*1\s*,",
                                           src)
                src_t = want(src, fix["elem_bytes"], buffer=src_buffer)
                tgt_t = want(tgt, fix["target_elem_bytes"])
                kind = rule.get("kind", "")
                if group == "map_bits" or kind == "embedding":
                    ptype = f"fgf::PreparedEmbedding<{src_t}, {tgt_t}>"
                elif kind == "isomorphism":
                    ptype = f"fgf::PreparedIsomorphism<{src_t}, {tgt_t}>"
                elif kind == "frobenius":
                    ptype = f"fgf::Frobenius<{src_t}>"
                else:
                    ptype = None
                if ptype is not None:
                    map_fixtures.append({"id": fix.get("id", ""),
                                         "prepared_type": ptype})
            elif group == "bits":
                continue
            elif group == "packed" and rule.get("non_bulk"):
                continue
            elif group == "binary_flat" and rid == "binary-flat-sub8":
                continue
            else:
                ftype = fix.get("type") or fix.get("source")
                if isinstance(ftype, str) and ftype:
                    want(ftype, fix.get("elem_bytes"))

    referenced: set[str] = set()
    for text in used_types:
        referenced |= set(FIXTURE_MARKER.findall(text))
    for spec in specs.values():
        referenced |= set(FIXTURE_MARKER.findall(spec["base"]))
    missing = referenced - set(specs)
    if missing:
        _fail(f"fixtures reference undefined custom specs: {sorted(missing)}")

    spec_lines: list[str] = []
    for marker in sorted(specs):
        spec = specs[marker]
        storage = spec["storage"]
        if "<" not in storage:
            storage = f"fgf::{storage}"
        else:
            head, _, tail = storage.partition("<")
            storage = f"fgf::{head}<{tail}"
        spec_lines.extend([
            f"struct {marker};",
            f"impl fgf::ExtensionSpec<{spec['degree']}> for {marker} {{",
            f"    type Base = {_scratch_type(spec['base'])};",
            f"    type Storage = {storage};",
            f"    const NAME: &'static str = {json.dumps(spec.get('name', ''))};",
            f"    const RELATION: fgf::Relation = {_spec_relation(spec)};",
            "}",
            "",
        ])

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

    item_lines = ["fn _assert_field_buffer<T: fgf::FieldBuffer>() {}"]
    for rust_type in sorted(valid_types):
        item_lines.append(
            f"const _: () = <{rust_type} as fgf::Field>::VALID;")
    item_lines.extend(spec_lines)

    main_lines = ["fn main() {"]
    for rust_type, widths in sorted(type_widths.items()):
        main_lines.append(f"    _assert_field_buffer::<{rust_type}>();")
        for width in sorted(widths):
            main_lines.append(
                f"    assert_eq!(<{rust_type} as fgf::FieldBuffer>::BYTES, {width});")
    if map_fixtures:
        main_lines.append("")
        main_lines.append("    // Map fixtures: type-check the prepared forms.")
        for mf in map_fixtures:
            main_lines.append(
                f"    let _ = core::mem::size_of::<{mf['prepared_type']}>();")
    main_lines.append("}")

    (src_dir / "main.rs").write_text(
        "\n".join(item_lines + [""] + main_lines) + "\n")

    result = subprocess.run(
        ["cargo", "check"],
        cwd=str(out_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Fixture compilation FAILED:\n{result.stderr}")
        return 1

    print(f"Fixture compilation OK: {len(type_widths)} field types, "
          f"{len(specs)} custom specs")
    if map_fixtures:
        print(f"  {len(map_fixtures)} map fixture prepared types checked")

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
            ledger = load(ledger_path)
            if args.full:
                return cmd_check(ledger, False, args.rustdoc, args.test_list,
                                 args.case_list, args.surface_inventory, args.records)
            return cmd_check(ledger, True, args.rustdoc, None, None, None, None)

        if args.command == "generate":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = load(ledger_path)
            return cmd_generate(args.out, args.check, ledger)

        if args.command == "accept":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = load(ledger_path)
            return cmd_accept(args.log, args.features, args.request,
                              args.source_id, args.host_id, ledger)

        if args.command == "fixtures":
            if not ledger_path.is_file():
                _fail(f"{ledger_path} not found")
            ledger = load(ledger_path)
            return cmd_fixtures(ledger, args.out)

        return 0
    except ValidationError:
        return 1


if __name__ == "__main__":
    sys.exit(main())