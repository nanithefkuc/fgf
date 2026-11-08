#!/usr/bin/env python3
"""Kernel-entry registry over rustdoc JSON: derivation, identity, and audit.

Every public `fgf::kernel::x86` function must take a mapped Archmage
capability token as its first parameter, and every token-first function
under `kernel::x86` must stay reachable through `fgf::internals`, so the
differential suites can name and exercise it. This tool derives the entry
registry from two all-features x86_64 rustdoc documents (one with
`--document-hidden-items`, one with `--document-private-items`), binds the
registry to the tested source tree, and audits the
`kernel_differential executed tier=TIER entry=PATH` lines the tier gate
emits after a successful differential.

Subcommands:

- `generate --public-doc P --private-doc Q --format F --channel C
  --target T [--features S] --out R`: derives `R`. Static checks, all
  fail-closed: the rustdoc format matches `F`; every public `kernel::x86`
  function has a mapped token first, or is one of the audited tokenless
  functions by name; every mapped token family (`v2`, `v3`,
  `v3_gfni_crypto`, `v4x`; `X64V1Token`/`X64V2Token` share `v2`,
  `X64V4Token`/`X64V4xToken` share `v4x`) has at least one entry; and the
  private pass rejects every token-first function under `kernel::x86`
  whose definition path is not reachable through the `fgf::internals`
  subtree of the public document, unless it is on the private-helper
  allowlist. The private pass includes module-local and restricted functions;
  paths omitted from rustdoc's public path table are reconstructed from
  the private declaration tree. Entries are canonical definition paths, one per
  function item including generic functions, not facade export paths.
- `identity [--target T] [--features S]`: prints the identity JSON the
  generator records: the actual tree hash of `src/`, `tests/`, the
  manifests, and this script, plus the feature set, target, pinned
  channel, and rustdoc JSON format.
- `check-identity REGISTRY [--target T] [--features S]`: recomputes the
  identity from the current tree and fails, naming each differing field,
  when the artifact is stale.
- `audit REGISTRY --floor TIER --actual TIER --log FILE`: verifies the
  tier gate's post-success marker lines against the registry. The floor is
  the required tier (`isa-accept EXPECTED`), the actual tier is the
  resolved process backend parsed from `backend_report`, so an unsupported
  request never widens the requirement. A marker naming an unknown path, a
  wrong tier, or a tier above the actual request ceiling fails; every entry
  at or below `min(floor, actual)` must have emitted its registered tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

GENERATOR_VERSION = "1"
KERNEL_ROOT = "fgf::kernel::x86"
CRATE = "fgf"
INTERNALS_PATH = f"{CRATE}::internals"

# First-parameter token type name to the tier whose kernels declare it.
# `V1` shares the `v2` family: dispatch reaches the SSE2 byte entries only
# from the V2 proof. `V4` and `V4x` share `v4x`.
TOKEN_TIERS = {
    "X64V1Token": "v2",
    "X64V2Token": "v2",
    "X64V3Token": "v3",
    "X64V3GfniCryptoToken": "v3_gfni_crypto",
    "X64V4Token": "v4x",
    "X64V4xToken": "v4x",
}

# Families the registry must never lose. `scalar` owns no token kernels.
REQUIRED_FAMILIES = ("v2", "v3", "v3_gfni_crypto", "v4x")

# Capability order, strongest last, mirroring the simdispatch x86 ladder
# the tier gate and `FGF_TIERS` resolve over. `Backend`'s own `Ord` sorts
# weaker tiers greater, which is why the audit keeps its own rank.
TIER_RANK = {
    "scalar": 0,
    "v2": 1,
    "v3": 2,
    "v3_gfni_crypto": 3,
    "v4x": 4,
}

# Public `kernel::x86` functions audited as non-entries: no first-parameter
# token, no marker obligation. Keyed by name; generation requires each name
# to match exactly one tokenless function and fails when the name no longer
# matches one, so disposing of an audited function removes its entry from
# this list in the same change.
TOKENLESS_ALLOWLIST = frozenset(
    {"multiply_vectors_avx2", "multiply_vectors_ssse3", "gf32_tiles", "gf64_tiles"}
)

# Definition paths of token-first helpers deliberately not reachable through
# internals. Each exception names the entry bodies that exercise it.
PRIVATE_ALLOWLIST: frozenset[str] = frozenset({
    # mul_add_scatter_avx2 checks the row windows consumed by this group.
    "fgf::kernel::x86::gf16::avx2::scatter_rows",
    # mul_add_matrix_avx512 drives disjoint reference-based row groups.
    "fgf::kernel::x86::gf16::avx512::matrix_group",
    # The corresponding public GFNI gather/matrix/scatter entries drive groups.
    "fgf::kernel::x86::gf16::gfni::gather::gather_group",
    "fgf::kernel::x86::gf16::gfni::matrix::matrix_group",
    "fgf::kernel::x86::gf16::gfni::scatter::scatter_group",
    # mul_add_gather_gfni drives both schedules and every tail through this chain.
    "fgf::kernel::x86::gf8::gfni::gather::mul_add_gather_impl",
    "fgf::kernel::x86::gf8::gfni::gather::gather_tiles",
    "fgf::kernel::x86::gf8::gfni::gather::gather_native_tail",
    "fgf::kernel::x86::gf8::gfni::gather::gather_remainder",
})

PRIVATE_PASS_VISIBILITIES = frozenset({"public", "crate", "restricted", "default"})

# Tree-hash inputs: everything the tested build consumes.
INPUT_FILES = ("Cargo.toml", "Cargo.lock", "build.rs")
INPUT_DIRS = ("src", "tests")
INPUT_SELF = "api/kernel_registry.py"
HASH_SKIP_DIRS = frozenset({"__pycache__", "target"})

MARKER_RE = re.compile(r"^kernel_differential executed tier=(\S+) entry=(\S+)$")

DEFAULT_FEATURES = "all-features"


class RegistryError(Exception):
    """A fail-closed registry condition with its evidence."""


def load(path: Path, expected_format: str) -> dict:
    doc = json.loads(path.read_text())
    version = doc.get("format_version")
    if str(version) != expected_format:
        raise RegistryError(
            f"{path}: rustdoc JSON format {version}, expected {expected_format}; "
            "regenerate with the pinned API_NIGHTLY"
        )
    return doc


def kind(item: dict) -> str:
    return next(iter(item["inner"]))


def canonical_path(doc: dict, item_id: str) -> str | None:
    entry = doc.get("paths", {}).get(item_id)
    if entry is None:
        return None
    return "::".join(entry["path"])


def definition_paths(doc: dict) -> dict[str, str]:
    """Include private declarations omitted from rustdoc's public path table."""
    paths = {str(key): "::".join(value["path"]) for key, value in doc["paths"].items()}
    seen: set[str] = set()

    def walk(module_id: str) -> None:
        if module_id in seen:
            return
        seen.add(module_id)
        module = doc["index"][module_id]
        prefix = paths[module_id]
        for child_id in module["inner"]["module"]["items"]:
            child_id = str(child_id)
            child = doc["index"].get(child_id)
            if child is None or kind(child) == "use":
                continue
            if child_id not in paths and child.get("name"):
                paths[child_id] = f"{prefix}::{child['name']}"
            if kind(child) == "module" and child_id in paths:
                walk(child_id)

    for item_id, item in doc["index"].items():
        if kind(item) == "module" and item_id in paths and paths[item_id].startswith(CRATE):
            walk(item_id)
    return paths

def functions_under_kernel_x86(doc: dict, visibilities: frozenset[str]) -> list[tuple[str, dict]]:
    """(definition path, index entry) for every function item under
    `kernel::x86` with one of `visibilities`, keyed by canonical path."""
    out = []
    definitions = definition_paths(doc)
    for item_id, entry in doc["index"].items():
        if kind(entry) != "function":
            continue
        visibility = entry.get("visibility")
        visibility = "restricted" if isinstance(visibility, dict) else visibility
        if visibility not in visibilities:
            continue
        path = definitions.get(item_id)
        if path is not None and path.startswith(KERNEL_ROOT + "::"):
            out.append((path, entry))
    return sorted(out)


def first_parameter(entry: dict, doc: dict) -> tuple[str, object]:
    """(classification, value) of the first parameter's type.

    `token` and `resolved` carry the resolved path; `generic` carries the
    parameter name; `missing` covers a zero-parameter function; any other
    rustdoc type shape (primitive, reference, slice, and kin) returns its
    shape name as an unmapped classification.
    """
    inputs = entry["inner"]["function"]["sig"]["inputs"]
    if not inputs:
        return ("missing", None)
    param_type = inputs[0][1]
    shape = next(iter(param_type))
    if shape == "resolved_path":
        resolved = param_type["resolved_path"]
        definition = doc.get("paths", {}).get(str(resolved["id"]))
        if definition is None:
            return ("resolved", f"unresolved::{resolved['path']}")
        path = definition["path"]
        if path[0] == "archmage" and path[-1] in TOKEN_TIERS:
            return ("resolved", f"archmage::{path[-1]}")
        return ("resolved", "::".join(path))
    if shape == "generic":
        return ("generic", param_type["generic"])
    return (shape, None)


def token_of(resolved: str) -> str | None:
    name = resolved.rsplit("::", 1)[-1]
    return name if resolved == f"archmage::{name}" and name in TOKEN_TIERS else None


def internals_reachable_paths(doc: dict) -> set[str]:
    """Canonical paths of every item reachable through the
    `fgf::internals` subtree: module children and re-export targets that
    a consumer outside the crate can name. A re-exported module's public
    contents are reachable through it; nothing else is."""
    internals_id = None
    for item_id, entry in doc["index"].items():
        if kind(entry) == "module" and canonical_path(doc, item_id) == INTERNALS_PATH:
            internals_id = item_id
            break
    if internals_id is None:
        raise RegistryError(
            f"no module with canonical path {INTERNALS_PATH}; "
            "the registry derives from the all-features x86_64 document"
        )
    seen: set[str] = set()
    reachable: set[str] = set()

    def walk(module_id: str) -> None:
        if module_id in seen:
            return
        seen.add(module_id)
        module = doc["index"].get(module_id)
        if module is None or kind(module) != "module":
            return
        for child in module["inner"]["module"]["items"]:
            child_id = str(child)
            entry = doc["index"].get(child_id)
            if entry is None or entry.get("visibility") != "public":
                continue
            path = canonical_path(doc, child_id)
            if path is not None:
                reachable.add(path)
            inner = kind(entry)
            if inner == "use":
                target_id = entry["inner"]["use"].get("id")
                if target_id is not None:
                    target_id = str(target_id)
                    # A re-export makes the target nameable under the
                    # re-export path, so its definition path is reachable.
                    target_path = canonical_path(doc, target_id)
                    if target_path is not None:
                        reachable.add(target_path)
                    walk(target_id)
            elif inner == "module":
                walk(child_id)

    walk(internals_id)
    return reachable


def tokenless_allowlist_matches(
    candidates: dict[str, list[str]], allowlist: frozenset[str] = TOKENLESS_ALLOWLIST
) -> tuple[list[str], list[str]]:
    """(canonical paths of the audited tokenless functions, errors): one
    path per allowlisted name, an error for a name that matches none (the
    function gained a token or disappeared) or matches more than one."""
    matched = []
    errors = []
    for name in sorted(allowlist):
        paths = candidates.get(name)
        if not paths:
            errors.append(
                f"audited tokenless function {name} matches no tokenless "
                "public kernel::x86 function; drop it from TOKENLESS_ALLOWLIST "
                "or restore the audited form"
            )
        elif len(paths) > 1:
            errors.append(
                f"audited tokenless function {name} is ambiguous across "
                f"{', '.join(sorted(paths))}"
            )
        else:
            matched.append(paths[0])
    return matched, errors


def generate(
    public_doc: dict,
    private_doc: dict,
    channel: str,
    fmt: str,
    target: str,
    features: str,
    tokenless_allowlist: frozenset[str] = TOKENLESS_ALLOWLIST,
    private_allowlist: frozenset[str] = PRIVATE_ALLOWLIST,
    root: Path | None = None,
) -> dict:
    """Derive the registry, running every static check. Returns the
    registry document; raises `RegistryError` listing all violations."""
    public_fns = functions_under_kernel_x86(public_doc, frozenset({"public"}))
    private_fns = functions_under_kernel_x86(private_doc, PRIVATE_PASS_VISIBILITIES)

    errors: list[str] = []
    entries: list[dict] = []
    tokenless_candidates: dict[str, list[str]] = {}

    for path, entry in public_fns:
        classification, value = first_parameter(entry, public_doc)
        if classification == "resolved":
            token = token_of(value)
            if token is not None:
                entries.append({"entry": path, "tier": TOKEN_TIERS[token], "token": token})
                continue
            if entry.get("name") in tokenless_allowlist:
                tokenless_candidates.setdefault(entry["name"], []).append(path)
                continue
            errors.append(
                f"unmapped first parameter `{value}` on public kernel::x86 function {path}; "
                "take a mapped capability token first or audit the function in "
                "TOKENLESS_ALLOWLIST"
            )
        elif classification == "generic":
            errors.append(
                f"generic first parameter `{value}` on public kernel::x86 function {path}; "
                "kernels take a concrete capability token first"
            )
        elif classification == "missing":
            if entry.get("name") in tokenless_allowlist:
                tokenless_candidates.setdefault(entry["name"], []).append(path)
            else:
                errors.append(
                    f"parameterless public kernel::x86 function {path}; "
                    "kernels take a capability token first"
                )
        else:
            errors.append(
                f"unmapped {classification} first parameter on public kernel::x86 "
                f"function {path}; take a mapped capability token first"
            )

    families = {entry["tier"] for entry in entries}
    for family in REQUIRED_FAMILIES:
        if family not in families:
            errors.append(f"no registry entry for token family {family}")
    present_tokens = {entry["token"] for entry in entries}
    for token in TOKEN_TIERS:
        if token not in present_tokens:
            errors.append(f"no registry entry for token type {token}")

    matched, allowlist_errors = tokenless_allowlist_matches(tokenless_candidates, tokenless_allowlist)
    errors.extend(allowlist_errors)
    audited = matched

    reachable = internals_reachable_paths(public_doc)
    for path, entry in private_fns:
        classification, value = first_parameter(entry, private_doc)
        if classification != "resolved" or token_of(value) is None:
            continue
        if path in reachable or path in private_allowlist:
            continue
        errors.append(
            f"token-first kernel::x86 function {path} is not reachable through "
            f"{INTERNALS_PATH}; re-export it through the internals facade or "
            "audit it in PRIVATE_ALLOWLIST"
        )

    public_paths = {path for path, _ in public_fns}
    private_public_paths = {
        path
        for path, entry in functions_under_kernel_x86(private_doc, frozenset({"public"}))
    }
    for label, only in (
        ("the private document", private_public_paths - public_paths),
        ("the public document", public_paths - private_public_paths),
    ):
        for path in sorted(only):
            errors.append(
                f"public kernel::x86 function {path} exists only in {label}; "
                "the two derivation documents must agree"
            )

    if errors:
        raise RegistryError(
            f"{len(errors)} registry violation(s):\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return {
        "identity": compute_identity(channel, fmt, target, features, root=root),
        "entries": sorted(entries, key=lambda entry: entry["entry"]),
        "audited_tokenless": sorted(audited),
        "private_allowlist": sorted(private_allowlist),
    }


def tree_hash(root: Path) -> str:
    """SHA-256 over the actual tested source tree: every file under the
    input directories, the manifests, and this script, framed by path and
    length so the stream is unambiguous."""
    root = Path(root)
    files: list[Path] = []
    for name in INPUT_FILES:
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    for directory in INPUT_DIRS:
        base = root / directory
        if not base.is_dir():
            raise RegistryError(f"tree hash input directory {base} is missing")
        for path in base.rglob("*"):
            if path.is_file() and not HASH_SKIP_DIRS.intersection(path.relative_to(root).parts):
                files.append(path)
    self_path = root / INPUT_SELF
    if not self_path.is_file():
        raise RegistryError(f"tree hash input {self_path} is missing")
    files.append(self_path)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        content = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


def script_hash(root: Path) -> str:
    return hashlib.sha256((Path(root) / INPUT_SELF).read_bytes()).hexdigest()


def compute_identity(
    channel: str, fmt: str, target: str, features: str, root: Path | None = None
) -> dict:
    return {
        "generator": f"kernel_registry/{GENERATOR_VERSION}",
        "generator_sha256": script_hash(root if root is not None else Path.cwd()),
        "source_tree_sha256": tree_hash(root if root is not None else Path.cwd()),
        "features": features,
        "target": target,
        "channel": channel,
        "rustdoc_json_format": str(fmt),
    }


def load_registry(path: Path) -> dict:
    try:
        registry = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise RegistryError(f"{path}: not valid JSON ({err})") from err
    for key in ("identity", "entries", "audited_tokenless", "private_allowlist"):
        if key not in registry:
            raise RegistryError(f"{path}: registry is missing `{key}`")
    return registry


def entry_tiers(registry: dict) -> dict[str, str]:
    tiers = {}
    for entry in registry["entries"]:
        path = entry.get("entry")
        tier = entry.get("tier")
        if path is None or tier is None or tier not in TIER_RANK:
            raise RegistryError(f"registry entry {entry!r} has no valid tier")
        if path in tiers:
            raise RegistryError(f"registry binds {path} twice")
        tiers[path] = tier
    return tiers


def audit(registry: dict, floor: str, actual: str, log_text: str) -> list[str]:
    """Check the executed markers against the registry. Returns the error
    lines; an empty list means the run satisfies the registry."""
    if floor not in TIER_RANK:
        raise RegistryError(f"unknown required floor tier {floor}")
    if actual not in TIER_RANK:
        raise RegistryError(f"unknown resolved actual tier {actual}")
    tiers = entry_tiers(registry)
    non_entries = set(registry["audited_tokenless"])

    ceiling = min(TIER_RANK[floor], TIER_RANK[actual])
    errors: list[str] = []
    executed: set[str] = set()
    for line in log_text.splitlines():
        marker = MARKER_RE.match(line.strip())
        if marker is None:
            continue
        tier, path = marker.group(1), marker.group(2)
        if tier not in TIER_RANK:
            errors.append(f"marker names unknown tier {tier}: {line.strip()}")
            continue
        if path in non_entries:
            errors.append(f"marker names audited non-entry {path}")
            continue
        expected = tiers.get(path)
        if expected is None:
            errors.append(f"marker names unknown entry {path}")
            continue
        if tier != expected:
            errors.append(f"marker reports tier {tier} for {path}, registry says {expected}")
            continue
        if TIER_RANK[tier] > TIER_RANK[actual]:
            errors.append(
                f"marker for {path} reports tier {tier} above the request "
                f"ceiling {actual} (floor {floor})"
            )
            continue
        executed.add(path)
    for path, tier in sorted(tiers.items()):
        if TIER_RANK[tier] <= ceiling and path not in executed:
            errors.append(f"missing executed marker for required entry {path} (tier {tier})")
    return errors


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"kernel-registry: set {name}; the recipes export it")
    return value


def cmd_generate(args: argparse.Namespace) -> int:
    public_doc = load(Path(args.public_doc), args.format)
    private_doc = load(Path(args.private_doc), args.format)
    registry = generate(
        public_doc,
        private_doc,
        channel=args.channel,
        fmt=args.format,
        target=args.target,
        features=args.features,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n")
    counts = {}
    for entry in registry["entries"]:
        counts[entry["tier"]] = counts.get(entry["tier"], 0) + 1
    print(
        f"kernel-registry: {len(registry['entries'])} entries "
        f"({', '.join(f'{tier}={counts[tier]}' for tier in sorted(counts))}), "
        f"{len(registry['audited_tokenless'])} audited tokenless, "
        f"{len(registry['private_allowlist'])} private allowlist entries"
    )
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    identity = compute_identity(
        require_env("API_NIGHTLY"), require_env("API_FORMAT"), args.target, args.features
    )
    print(json.dumps(identity, indent=2))
    return 0


def cmd_check_identity(args: argparse.Namespace) -> int:
    registry = load_registry(Path(args.registry))
    expected = compute_identity(
        require_env("API_NIGHTLY"), require_env("API_FORMAT"), args.target, args.features
    )
    recorded = registry["identity"]
    if recorded == expected:
        print("kernel-registry: registry artifact identity matches the tested tree")
        return 0
    print("kernel-registry: registry artifact is stale; regenerate it", file=sys.stderr)
    for key in sorted(set(expected) | set(recorded)):
        if recorded.get(key) != expected[key]:
            print(f"  {key}: registry={recorded.get(key)!r} tree={expected[key]!r}", file=sys.stderr)
    return 1


def cmd_audit(args: argparse.Namespace) -> int:
    registry = load_registry(Path(args.registry))
    errors = audit(registry, args.floor, args.actual, Path(args.log).read_text())
    if errors:
        print(f"kernel-registry: {len(errors)} audit violation(s)", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1
    tiers = entry_tiers(registry)
    ceiling = min(TIER_RANK[args.floor], TIER_RANK[args.actual])
    required = sum(1 for tier in tiers.values() if TIER_RANK[tier] <= ceiling)
    print(
        f"kernel-registry: audit ok (floor {args.floor}, resolved {args.actual}, "
        f"{required} required entries, every marker matched)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--public-doc", required=True)
    p.add_argument("--private-doc", required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--channel", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--out", required=True)
    p.set_defaults(run=cmd_generate)
    p = sub.add_parser("identity")
    p.add_argument("--target", required=True)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.set_defaults(run=cmd_identity)
    p = sub.add_parser("check-identity")
    p.add_argument("registry")
    p.add_argument("--target", required=True)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.set_defaults(run=cmd_check_identity)
    p = sub.add_parser("audit")
    p.add_argument("registry")
    p.add_argument("--floor", required=True)
    p.add_argument("--actual", required=True)
    p.add_argument("--log", required=True)
    p.set_defaults(run=cmd_audit)
    args = parser.parse_args()
    try:
        return args.run(args)
    except RegistryError as err:
        print(f"kernel-registry: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
