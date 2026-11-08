#!/usr/bin/env python3
"""Capture descriptive public-API measurements without paired acceptance claims."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import statistics
import sys

import bench_pair as pair
import check_surfaces

METHOD_VERSION = "fgf-descriptive/1"


def digest(value) -> str:
    return pair.sha256_bytes(pair.canonical_json(value).encode("utf-8"))


def positive(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def parse_surfaces(text: str) -> dict:
    """Read the surface protocol, rejecting ambiguous identities and timings."""
    result = {"backend": None, "field_backends": {}, "inventory": {},
              "order": [], "timings": {}, "checked": [], "complete": None}
    for line in text.splitlines():
        if line.startswith("fgf surfaces benchmark — backend: "):
            if result["backend"] is not None:
                raise ValueError("duplicate surfaces backend banner")
            result["backend"] = line.rsplit(" ", 1)[-1]
        elif line.startswith("FIELD_BACKEND\t"):
            parts = line.split("\t")
            if len(parts) != 3 or parts[1] in result["field_backends"]:
                raise ValueError("ambiguous FIELD_BACKEND record")
            result["field_backends"][parts[1]] = parts[2]
        elif line.startswith(("SURFACE_INVENTORY\t", "SURFACE\t")):
            tag, raw = line.split("\t", 1)
            item = json.loads(raw)
            metadata = {key: item[key] for key in check_surfaces.SURFACE_INVENTORY_KEYS}
            check_surfaces.surface_object(metadata, section="surfaces", where=tag)
            key = item["case"]
            target = result["inventory"] if tag == "SURFACE_INVENTORY" else result["timings"]
            if key in target:
                raise ValueError(f"duplicate {tag} identity: {key}")
            if tag == "SURFACE_INVENTORY":
                result["order"].append(key)
                target[key] = metadata
            else:
                if not positive(item.get("ns")) or not isinstance(item.get("backend"), str):
                    raise ValueError(f"invalid surface timing: {key}")
                if item["backend"] not in check_surfaces.KNOWN_BACKENDS:
                    raise ValueError(f"unknown surface backend: {key}")
                if (not isinstance(item.get("samples"), int)
                        or isinstance(item["samples"], bool) or item["samples"] < 1):
                    raise ValueError(f"invalid surface sample count: {key}")
                target[key] = item
        elif line.startswith("SURFACE_CHECKED\t"):
            key = line.split("\t", 1)[1]
            if key in result["checked"]:
                raise ValueError(f"duplicate check identity: {key}")
            result["checked"].append(key)
        elif line.startswith("SURFACES_CHECK_COMPLETE\tcases="):
            if result["complete"] is not None:
                raise ValueError("duplicate full-check completion")
            result["complete"] = int(line.rsplit("=", 1)[-1])
    if result["backend"] is None or not result["inventory"]:
        raise ValueError("surfaces protocol lacks backend or inventory")
    if result["field_backends"].get("process") != result["backend"]:
        raise ValueError("surface process backend disagrees with banner")
    for key, timing in result["timings"].items():
        if key not in result["inventory"] or any(
            timing[name] != result["inventory"][key][name]
            for name in check_surfaces.SURFACE_INVENTORY_KEYS
        ):
            raise ValueError(f"surface timing metadata differs from inventory: {key}")
    return result


class Operations:
    """Adapt the existing bulk protocol without changing its measurement engine."""

    name = "operations"
    filter_flag = "--cells"

    @staticmethod
    def inventory(text):
        parsed, errors = pair.parse_protocol(text)
        if errors or not parsed["inventory"] or not parsed["backend"]:
            raise ValueError("invalid operations inventory: " + "; ".join(errors))
        return parsed, parsed["inventory"], parsed["inventory_order"]

    @staticmethod
    def check(text, inventory, order):
        parsed, errors = pair.parse_protocol(text)
        if errors or set(parsed["outputs"]) != set(order):
            raise ValueError("operations full check does not cover its inventory")
        return parsed

    @staticmethod
    def timing(text, inventory, order, rounds):
        parsed, errors = pair.parse_protocol(text)
        cells, problem = pair.summarize_timed_protocol(parsed, order, rounds, "base")
        start, end = pair.control_value(parsed, "start"), pair.control_value(parsed, "end")
        if errors or problem is not None or not positive(start) or not positive(end):
            raise ValueError("invalid or incomplete operations timing")
        return parsed, cells, {"start_ns": start, "end_ns": end}

    @staticmethod
    def arguments(rounds):
        return ["--time", "--rounds", str(rounds)]


class Surfaces:
    """Adapt the breadth protocol; controls never become measurement rows."""

    name = "surfaces"
    filter_flag = "--cases"

    @staticmethod
    def inventory(text):
        parsed = parse_surfaces(text)
        return parsed, parsed["inventory"], parsed["order"]

    @staticmethod
    def check(text, inventory, order):
        parsed = parse_surfaces(text)
        if (parsed["inventory"] != inventory or parsed["order"] != order
                or parsed["checked"] != order or parsed["complete"] != len(order)
                or parsed["timings"]):
            raise ValueError("surfaces full check does not cover its inventory")
        return parsed

    @staticmethod
    def timing(text, inventory, order, rounds):
        parsed = parse_surfaces(text)
        controls = [key for key, item in inventory.items() if item["control"]]
        if parsed["inventory"] != inventory or set(parsed["timings"]) != set(order + controls):
            raise ValueError("surfaces timing does not exactly cover selected cases and controls")
        phases = {}
        for key in controls:
            record = parsed["timings"][key]
            phase = record["phase"]
            if phase in phases:
                raise ValueError("duplicate surface control phase")
            phases[phase] = record["ns"]
        if set(phases) != {"entry", "exit"}:
            raise ValueError("surfaces timing lacks entry/exit controls")
        cells = {key: {"summary_ns": parsed["timings"][key]["ns"],
                       "rounds_ns": [parsed["timings"][key]["ns"]],
                       "backend": parsed["timings"][key]["backend"]} for key in order}
        return parsed, cells, {"start_ns": phases["entry"], "end_ns": phases["exit"]}

    @staticmethod
    def arguments(rounds):
        if rounds != 1:
            raise ValueError("surfaces requires exactly one round per process")
        return []


def scope_for(inventory: dict, order: list[str], bench: str, path: str | None) -> dict:
    """Resolve a descriptive selection without turning queries into headline data."""
    available = [key for key in order if not inventory[key].get("control", False)]
    declaration_digest = None
    rationale = None
    if path:
        payload = Path(path).read_bytes()
        declaration_digest = pair.sha256_bytes(payload)
        text = payload.decode("utf-8")
        try:
            declaration = json.loads(text)
        except json.JSONDecodeError:
            declaration = [line.split("#", 1)[0].strip() for line in text.splitlines()]
            declaration = [line for line in declaration if line]
        if isinstance(declaration, dict):
            rationale = declaration.get("rationale")
            wanted = declaration.get("cells")
        else:
            wanted = declaration
        if (not isinstance(wanted, list) or not wanted
                or any(not isinstance(key, str) or not key for key in wanted)
                or len(set(wanted)) != len(wanted)):
            raise ValueError("baseline selection must contain distinct nonempty case ids")
        unknown = set(wanted) - set(available)
        if unknown:
            raise ValueError("unknown or control-only baseline cases: " + ", ".join(sorted(unknown)))
        cells = [key for key in available if key in set(wanted)]
        kind, rule = "selected", None
    elif bench == "surfaces":
        cells = [key for key in available if inventory[key]["scope"] != "query"]
        kind, rule = "default", "scope != query"
    else:
        cells, kind, rule = available, "full", None
    if not cells:
        raise ValueError("baseline scope is empty")
    return {"kind": kind, "rule": rule, "cells": cells, "digest": digest(cells),
            "rationale": rationale, "declaration_digest": declaration_digest}


def build_rows(session: dict) -> list[dict]:
    rows = []
    inventory = session["inventory"]["entries"]
    for tier in session["tiers"]:
        name = tier["requested"]
        runs = [run for run in session["runs"] if run["tier"] == name]
        for key in session["scope"]["cells"]:
            metadata = inventory[key]
            values = [run["cells"][key]["summary_ns"] for run in runs]
            rows.append({
                "host": session["host"]["hostname"], "tier": name, "cell": key,
                "arm": "base", "role": "descriptive", "unstable": False,
                "library_sha": session["arms"]["base"]["sha"],
                "session_id": session["session_id"], "metadata": metadata,
                "field": metadata["field"], "operation": metadata["operation"],
                "prep": metadata.get("prep", metadata.get("scope", "")),
                "row_len": metadata.get("row_len", 0),
                "rows": metadata.get("rows", metadata.get("destinations", 0)),
                "sources": metadata["sources"], "alignment": metadata.get("alignment", "-"),
                "summary_ns": statistics.median(values), "run_summaries_ns": values,
                "min_ns": min(values), "max_ns": max(values),
                "rounds_ns": [value for run in runs for value in run["cells"][key]["rounds_ns"]],
            })
    return rows


def validate_session(session: dict) -> None:
    """Validate standalone evidence; a descriptive record cannot assert acceptance."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(session.get("method_version") == METHOD_VERSION, "unsupported descriptive method")
    require(session.get("kind") == "descriptive", "not a descriptive record")
    require(session.get("acceptance") is False, "descriptive record asserts acceptance")
    require(not session.get("rejected") and not session.get("rejections"), "rejected descriptive record")
    require("decisions" not in session, "descriptive record contains acceptance decisions")
    require(set(session["arms"]) == {"base"}, "descriptive record must contain exactly one base arm")
    require(session["arms"]["base"]["kind"] == "library", "descriptive arm is not a library revision")
    require(isinstance(session["arms"]["base"]["sha"], str) and bool(session["arms"]["base"]["sha"]),
            "missing descriptive revision")
    require(session["snapshot"]["bench"] in ("operations", "surfaces"), "unknown descriptive harness")
    require(session["snapshot"].get("runner_matches_snapshot") is True, "baseline runner differs from snapshot")
    require(session["snapshot"].get("no_default_features") is False, "baseline requires default features")
    require(isinstance(session["snapshot"]["features"], list), "invalid descriptive features")
    require("all" not in session["snapshot"]["features"] and "internals" not in session["snapshot"]["features"],
            "descriptive measurements require the stable feature set")
    require(pair.check_alignment_flags(session["snapshot"]["rustflags"]) is None, "baseline alignment flags missing")
    for key in ("session_id", "created_utc"):
        require(isinstance(session[key], str) and bool(session[key]), "missing descriptive identity")
    for key in ("hostname", "cpu_model", "os", "arch", "toolchain_id", "measured_utc", "campaign_lock"):
        require(isinstance(session["host"][key], str) and bool(session["host"][key]), "missing descriptive host provenance")
    count, rounds = session["config"]["runs"], session["config"]["rounds"]
    require(type(count) is int and count >= 3, "publication requires at least three standalone processes")
    require(type(rounds) is int and rounds >= 1, "invalid descriptive rounds")
    require(session["snapshot"]["bench"] != "surfaces" or rounds == 1, "surfaces supports one round")
    order, inventory = session["inventory"]["order"], session["inventory"]["entries"]
    require(isinstance(order, list) and bool(order) and len(set(order)) == len(order)
            and set(order) == set(inventory), "invalid descriptive inventory")
    require(digest([inventory[key] for key in order]) == session["inventory"]["digest"], "inventory digest mismatch")
    for key in order:
        entry = inventory[key]
        if session["snapshot"]["bench"] == "surfaces":
            check_surfaces.surface_object(entry, section="surfaces", where=key)
            require(entry["case"] == key, "surface identity differs from inventory key")
        else:
            require(pair.validate_inventory_entry(entry) is None and entry["id"] == key,
                    "invalid operations inventory entry")
    scope = session["scope"]
    cells = scope["cells"]
    require(isinstance(cells, list) and bool(cells) and len(set(cells)) == len(cells), "invalid descriptive scope")
    require(set(cells) <= set(order) and [key for key in order if key in set(cells)] == cells, "scope is outside inventory order")
    require(digest(cells) == scope["digest"], "scope digest mismatch")
    require(scope["kind"] in ("default", "full", "selected"), "unknown scope kind")
    available = [key for key in order if not inventory[key].get("control", False)]
    require(all(not inventory[key].get("control", False) for key in cells), "controls are not measurement rows")
    if scope["kind"] == "default":
        require(session["snapshot"]["bench"] == "surfaces" and scope["rule"] == "scope != query"
                and cells == [key for key in available if inventory[key]["scope"] != "query"], "default scope rule mismatch")
    elif scope["kind"] == "full":
        require(cells == available, "full scope omits cases")
    else:
        require(isinstance(scope.get("declaration_digest"), str) and bool(scope["declaration_digest"]), "selected scope lacks declaration")
    tiers = session["tiers"]
    names = [tier["requested"] for tier in tiers]
    require(bool(names) and len(set(names)) == len(names), "invalid descriptive tiers")
    for tier in tiers:
        require(tier["reported"] == tier["requested"] and tier["ok"] is True, "requested tier did not execute")
        require(bool(tier["field_backends"]), "missing field backend evidence")
        check = session["checks"][tier["requested"]]
        require(check["full"] is True and check["cells"] == order
                and check["inventory_digest"] == session["inventory"]["digest"], "descriptive check is incomplete")
    expected_runs = {(name, index) for name in names for index in range(count)}
    seen_runs = set()
    for run in session["runs"]:
        identity = (run["tier"], run["run"])
        require(identity in expected_runs and identity not in seen_runs, "unexpected/duplicate standalone run")
        seen_runs.add(identity)
        require(run["backend"] == run["tier"], "run backend mismatch")
        expected_fields = next(tier["field_backends"] for tier in tiers if tier["requested"] == run["tier"])
        require(run["field_backends"] == expected_fields, "run field backends differ from inspection")
        require(set(run["cells"]) == set(cells), "run does not cover scope")
        require(all(positive(run["control"][key]) for key in ("start_ns", "end_ns")), "invalid drift control")
        for item in run["cells"].values():
            require(len(item["rounds_ns"]) == rounds and all(positive(value) for value in item["rounds_ns"]), "invalid run samples")
            require(item["summary_ns"] == statistics.median(item["rounds_ns"]), "run median differs from samples")
    require(seen_runs == expected_runs, "missing standalone processes")
    require(session["rows"] == build_rows(session), "descriptive rows differ from scope, samples, or revision")
    require(session["rows"], "descriptive rows are empty")


class Baseline:
    """Compose the paired runner's build and isolation infrastructure."""

    def __init__(self, args, session, directory):
        self.args, self.session, self.directory = args, session, directory
        self.campaign = pair.Campaign(args, session, directory)
        self.adapter = Surfaces if args.bench == "surfaces" else Operations

    def invoke(self, executable, arguments, tier, label, env=None):
        process = self.campaign.run_harness(executable, arguments, tier, label,
                                            env=env, disable_aslr=True)
        if process.returncode:
            raise ValueError(f"harness failed; see {label}.log")
        return process.stdout

    def execute(self):
        args, session, campaign = self.args, self.session, self.campaign
        if not campaign.acquire_lock():
            return 1
        if args.runs < 3 or args.rounds < 1:
            raise ValueError("baseline requires at least three processes and a positive round count")
        self.adapter.arguments(args.rounds)
        snapshot = campaign.prepare_snapshot(bench=args.bench)
        if snapshot is None:
            return 1
        if "all" in snapshot.features or "internals" in snapshot.features or snapshot.no_default_features:
            raise ValueError("baseline requires default features without internals")
        session["snapshot"] = snapshot.as_dict()
        runners = ("bench_baseline.py", "bench_pair.py")
        matches = all((snapshot.directory / "benches" / name).is_file()
                      and pair.sha256_file(snapshot.directory / "benches" / name)
                      == pair.sha256_file(Path(__file__).with_name(name)) for name in runners)
        session["snapshot"]["runner_matches_snapshot"] = matches
        session["snapshot"]["runner_digests"] = {name: pair.sha256_file(Path(__file__).with_name(name)) for name in runners}
        if not matches:
            raise ValueError("baseline infrastructure differs from measurement snapshot")
        session["host"].update(pair.read_host_facts(snapshot.toolchain))
        for problem in pair.validate_smt_config(args.smt_window, args.smt_busy_threshold) + pair.validate_execution_environment(
                campaign.machine, shutil.which("setarch"), args.chrt, shutil.which("chrt")):
            campaign.reject(problem)
        if campaign.rejections:
            return 1
        campaign.chrt_parts = shlex.split(args.chrt)
        campaign.setarch_command = f"setarch {campaign.machine} -R"
        session["execution"] = {"aslr_disabled": True, "pinned_core": args.core,
                                "setarch": campaign.setarch_command, "chrt": campaign.chrt_parts}
        if not campaign.check_preconditions("preflight", -1, "preflight"):
            return 1
        arm = {"kind": "library", "repo": str(campaign.repo),
               "sha": pair.git_revision(campaign.repo, args.revision), "metadata_seed": None}
        session["arms"]["base"] = arm
        if not campaign.make_arm_checkout("base", arm, snapshot) or not campaign.check_dependencies(snapshot):
            return 1
        executable = campaign.build_bench("base", arm, snapshot)
        if executable is None:
            return 1
        executable = campaign.prepare_execution_alias("base", executable)
        auto_env = os.environ.copy()
        auto_env.pop("SIMD_BACKEND", None)
        auto_env["FEC_GOLDEN_CORE"] = str(args.core)
        text = self.invoke(executable, ["--inventory"], "auto", "inventory-auto", auto_env)
        discovered, inventory, order = self.adapter.inventory(text)
        tiers = list(dict.fromkeys(pair.split_words(args.tiers))) if args.tiers else [discovered["backend"]]
        if not campaign.validate_common(snapshot, tiers):
            return 1
        session["inventory"] = {"entries": inventory, "order": order,
                                "digest": digest([inventory[key] for key in order])}
        session["scope"] = scope_for(inventory, order, args.bench, args.cells)
        selected = session["scope"]["cells"]
        selection = self.directory / "cells.txt"
        selection.write_text("\n".join(selected) + "\n", encoding="utf-8")
        for tier in tiers:
            inspected, entries, ids = self.adapter.inventory(self.invoke(executable, ["--inventory"], tier, f"inventory-{tier}"))
            if inspected["backend"] != tier or entries != inventory or ids != order:
                raise ValueError("requested backend did not execute or inventory changed")
            fields = inspected["field_backends"]
            session["tiers"].append({"requested": tier, "reported": tier, "field_backends": fields, "ok": True})
            checked_text = self.invoke(executable, ["--check"], tier, f"check-{tier}")
            checked = self.adapter.check(checked_text, inventory, order)
            if checked["backend"] != tier or checked["field_backends"] != fields:
                raise ValueError("check backend differs from inspection")
            session["checks"][tier] = {"full": True, "cells": order,
                                        "inventory_digest": session["inventory"]["digest"],
                                        "log_digest": pair.sha256_bytes(checked_text.encode("utf-8"))}
            for index in range(args.runs):
                if not campaign.check_preconditions(tier, index):
                    return 1
                arguments = self.adapter.arguments(args.rounds) + [self.adapter.filter_flag, str(selection)]
                text = self.invoke(executable, arguments, tier, f"run-{index}-{tier}")
                parsed, cells, control = self.adapter.timing(text, inventory, selected, args.rounds)
                if parsed["backend"] != tier or parsed["field_backends"] != fields:
                    raise ValueError("timing backend differs from inspection")
                session["runs"].append({"run": index, "tier": tier, "backend": parsed["backend"],
                                         "field_backends": fields, "control": control, "cells": cells})
            session["drift"][tier] = pair.drift_stats([
                run["control"]["start_ns"] for run in session["runs"] if run["tier"] == tier])
        session["rows"] = build_rows(session)
        validate_session(session)
        return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    pair.add_execution_arguments(parser, records_default="bench-records/baseline")
    parser.add_argument("--revision", default="HEAD", help="library revision to measure")
    parser.add_argument("--bench", choices=("surfaces", "operations"), default="surfaces")
    parser.add_argument("--tiers", default=None, help="explicit tier list; default is the host's strongest resolved tier")
    parser.add_argument("--runs", type=int, default=5, help="standalone processes per tier")
    parser.add_argument("--rounds", type=int, default=1, help="operations rounds per process; surfaces requires one")
    parser.add_argument("--cells", help="optional exact case selection; JSON cells array or one id per line")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.records)
    directory = root / f"{pair.utc_stamp()}-{pair.host_component(socket.gethostname())}"
    suffix = 2
    while directory.exists():
        directory = root / f"{pair.utc_stamp()}-{pair.host_component(socket.gethostname())}-{suffix}"
        suffix += 1
    directory.mkdir(parents=True)
    session = {"method_version": METHOD_VERSION, "kind": "descriptive", "acceptance": False,
               "session_id": directory.name, "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "host": {}, "snapshot": {}, "arms": {}, "config": {"runs": args.runs, "rounds": args.rounds},
               "execution": {}, "preconditions": [], "tiers": [], "checks": {}, "runs": [],
               "drift": {}, "rows": [], "rejections": [], "rejected": False}
    baseline = Baseline(args, session, directory)
    code = 1
    try:
        code = baseline.execute()
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, check_surfaces.CheckError) as error:
        baseline.campaign.reject(pair.reject("baseline_failed", str(error)))
    finally:
        session["rejections"] = [item.as_dict() for item in baseline.campaign.rejections]
        session["rejected"] = bool(session["rejections"])
        pair.write_session(session, directory / "session.json")
        baseline.campaign.close()
        baseline.campaign.release_lock()
    print(directory / "session.json")
    for problem in session["rejections"]:
        print(f"baseline: {problem['message']}", file=sys.stderr)
    return 1 if session["rejected"] else code


if __name__ == "__main__":
    sys.exit(main())
