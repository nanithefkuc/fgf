# Contributing

Thanks for your interest. Contributions go through Pull Requests; what changed
and why belong in the PR and in `CHANGELOG.md`.

## Running the tests

```sh
just test        # host's selected backend
just test-tiers  # every supported backend tier
just features    # no-default, default, all-features
just examples-portable  # educational examples against the library without std or alloc
```

Backend dispatch resolves once per process, so a single run only covers the
host's best backend. `just test-tiers` walks the weaker ones through the
downgrade-only `SIMD_BACKEND` override.

## Benchmarks

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
```

Pin the run to one core and record the CPU, operating system, Rust version,
selected backend, and geometry with any quoted number. `BENCHMARKS.md` indexes
the current measurements in `benchmarks/`. `just benchmark-check` validates
portable fixtures and their compiler-derived public API inventory;
`just benchmark-check-native` validates native GF16 fixtures on capable hosts.

The competitor harnesses in `external/` need their toolchains: Intel ISA-L
discoverable through pkg-config, and the Go toolchain in PATH.

## SIMD kernels and routing

Requests and Pull Requests for new specialized SIMD kernels, or for new
backend routing, are reviewed case by case and are not accepted by default.
Each kernel and route adds maintenance burden, and a route can be slower
than the path it replaces on some hosts or geometries. A proposal states
the field, operations, and backends it targets, and carries differential
correctness against the scalar path and interleaved before/after
measurements against an unchanged control.

PRs adding public-API SIMD kernels, changing their routing or tuning, or
changing the route ledger, trace, or acceptance tooling reference the
tracking issue with `Refs #68`. Each applicable PR identifies:

- Obligation IDs and before/after route and performance state.
- Independent correctness references (differential, public-route, and
  production-smoke case IDs).
- Direct and public-route native execution evidence under default and all
  features.
- Pinned interleaved before/after measurements with unchanged controls.
- Exercised dispatch gates (tier requests and feature configurations).
- Applicable docs, CHANGELOG, residue-ledger, and unsafe-code updates.

The canonical SIMD route ledger is `benches/simd_ledger.json`, validated by
`benches/simd_ledger.py`. `just simd-ledger` checks callable classification
and the schema, runs the Python unit tests, and compiles the field and map
fixtures. The ledger is distinct from the unsafe residue ledger in `AGENTS.md`.

## Before opening a PR

```sh
just validate
```

This runs formatting, clippy at both feature ends, the dependency allowlist,
rustdoc, the feature matrix, the backend-tier tests, Miri, and coverage.

Release work runs the crate-specific acceptance, including the PR gate:
`just release-accept`. Its pieces also stand alone — `just isa-accept EXPECTED`
requires the requested process backend and positive kernel execution evidence;
`just test-gather-validation-32` runs the gather regression as release i686
code with overflow checks disabled. Native release acceptance uses dedicated
Willow Cove (`v4x`) and Golden Cove (`v3_gfni_crypto`) hosts. GitHub-hosted
runners cover only tiers they can prove.

New public items need doc comments. MSRV is 1.93 and is checked in CI; do not
reach for newer standard-library APIs without raising it deliberately.

## Commit messages

Subject lines are at most ~10 words and carry the change at a glance
(`fgf: short verb phrase`). Implementation detail belongs in the Pull Request
and `CHANGELOG.md`, not the message, and messages never reference planning
artifacts.
