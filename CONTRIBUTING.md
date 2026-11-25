# Contributing

Contributions start with an Issue and resolve through a linked Pull Request.
`CHANGELOG.md` is a concise release index, not a decision or implementation log.

## Issues, resolutions, and release notes

Every substantive change has an Issue describing the problem or specification,
constraints, alternatives, decision rationale, blockers, and observable
acceptance criteria. Keep the Issue current when the scope or decision changes.
Unresolved choices remain explicit rather than appearing as completed work.

The PR records the implemented resolution, deviations from the Issue, tradeoffs,
and verification actually exercised, including failures and limits. Partial work
uses `Refs #N`; a PR closes an Issue only when all its acceptance criteria are
resolved. Declined or superseded work closes with a reason and any replacement
link, not a claim of successful implementation.

Release entries use short, categorized user-facing summaries with Issue and PR
links as available. Breaking entries keep `**Breaking:**` and a migration
pointer. Detailed rationale and implementation belong in the Issue/PR; usage
and migration contracts belong in rustdoc or public guides. Retrospective
release-record Issues identify their source and do not invent missing PRs or
verification.

Public Issues and PRs stand alone: distill local development notes into a
reviewable specification or decision record without local document IDs, paths,
transcripts, or inaccessible evidence.

## Running the tests

```sh
just test        # host's selected backend
just test-tiers  # every supported backend tier
just features    # no-default, default, all-features
just examples    # every educational example in each feature configuration
```

Backend dispatch resolves once per process, so a single run only covers the
host's best backend. `just test-tiers` walks the weaker ones through the
downgrade-only `SIMD_BACKEND` override.

## Benchmarks

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
FEC_GOLDEN_CORE=<cpu> just bench-comp gf
```

Pin the run to one core and record the CPU, operating system, Rust version,
selected backend, and geometry with any quoted number; raw captures stay in
git-ignored `bench-records/`. `just benchmark-check` validates
portable fixtures and their compiler-derived public API inventory;
`just benchmark-check-native` validates native GF16 fixtures on capable hosts.

The unpublished competitor packages in `benches/compare/` keep native build
scripts and dependencies separate from the library. `just comparisons-check`
checks all three packages. Their toolchains include Intel ISA-L discoverable
through pkg-config, Go, and C/C++ compilers.
Public measurement records use processor-architecture labels and sanitized
environment facts, never SSH aliases, hostnames, or local account paths.

`just bench-pair` compares equally weighted page-placement distributions.
Each independent paired run measures a fixed 256-stratum design: destination
offsets, source offsets, and relative `(src - dst) mod 4096` phases each cover
every 64-byte cache-line offset equally (4× per run), and unread environment
padding covers every 16-byte ABI-aligned initial stack offset once per run.
This is a stratified design, not a Cartesian sweep of all interactions; source
buffers retain their role-relative offsets (`Dst = 64`, `Src = 512`,
`Src2 = 960`, `Region = 1408`).

Each stratum runs both arms under disabled ASLR with identical environments
and equal-length executable paths. Arm order alternates across strata and runs
so every destination, source, and relative phase is balanced 2/2 within each
run and every stack padding is balanced 1/1 across every consecutive pair of
runs. Per-placement round medians, controls, observed stack offsets, and raw
logs remain available; start-control drift gates per-stratum controls and
preconditions re-check periodically within each sweep. The arm's run summary is
the median across placements; only independent run summaries enter the paired
bound. A slow placement is not an individual noninferiority failure. Persistent
code regressions still face the unchanged margin and confidence rule. Old
single-placement sessions cannot establish a placement-distribution result.

`just bench-placement-check` exercises the placement and decision rules;
`just bench-placement-build` builds the operation harness used for native
placement verification. The harness asserts actual fixture page offsets
before timing and reports the observed stack page offset.

## SIMD kernels and routing

Requests and Pull Requests for new specialized SIMD kernels, or for new
backend routing, are reviewed case by case and are not accepted by default.
Each kernel and route adds maintenance burden, and a route can be slower
than the path it replaces on some hosts or geometries. A proposal states
the field, operations, and backends it targets, and carries differential
correctness against the scalar path and interleaved before/after
measurements against an unchanged control.

## Before opening a PR

```sh
just validate
```

This runs formatting, clippy at both feature ends, the dependency allowlist,
rustdoc, the feature matrix, the backend-tier tests, Miri, and coverage.

A pull request that changes the public API also runs `just api-update` and
commits the regenerated `api/` diff with the source change; the maintainer's
review of that diff authorizes the change. `just api` checks that the
committed inventory matches the source and is part of CI and
`just release-accept`. It needs the nightly, targets, and tool versions
pinned in `crate.just`, which holds their values.

`just api-semver` checks the stable feature sets — `internals` is never
enabled — against the published baseline, and accepts an optional baseline
version for development checks. The release gate that `just release-accept`
calls decides by the version's major number: releases below the current
major and the first release of the major with its prereleases are skipped,
every later release of the major including prereleases runs the check, and a
higher major fails until `API_MAJOR` in `crate.just` is raised.
`API_PATHS_MODE` in `crate.just` is `report` until every stable item has one
path, then `enforce`. `just api` never writes `api/`; only `just api-update`
rewrites it, inside an authorized change.

Release work runs the crate-specific acceptance, including the PR gate:
`just release-accept`. Its pieces also stand alone — `just isa-accept EXPECTED`
requires the requested process backend and positive kernel execution evidence;
`just test-gather-validation-32` runs the gather regression as release i686
code with overflow checks disabled. Native release acceptance uses dedicated
Willow Cove (`v4x`) and Golden Cove (`v3_gfni_crypto`) hosts. GitHub-hosted
runners cover only tiers they can prove.

New public items need doc comments. MSRV is 1.93 and is checked in CI; it
rises only in a minor release, never in a patch release, so do not reach for
newer standard-library APIs without raising it deliberately.

## Commit messages

Subject lines are at most ~10 words and carry the change at a glance
(`fgf: short verb phrase`). What changed and why belong in the Issue; the
linked PR records the resolution and verification. Neither the commit message
nor the changelog duplicates those details or references planning artifacts.
