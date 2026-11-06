# Contributing

Thanks for your interest. Contributions go through Pull Requests; what changed
and why belong in the PR and in `CHANGELOG.md`.

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
(`fgf: short verb phrase`). Implementation detail belongs in the Pull Request
and `CHANGELOG.md`, not the message, and messages never reference planning
artifacts.
