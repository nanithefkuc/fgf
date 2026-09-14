# Contributing

Thanks for your interest. Contributions go through Pull Requests; what changed
and why belong in the PR and in `CHANGELOG.md`.

## Running the tests

```sh
just test        # host's selected backend
just test-tiers  # every supported backend tier
just features    # no-default, default, all-features
```

Backend dispatch resolves once per process, so a single run only covers the
host's best backend. `just test-tiers` walks the weaker ones through the
downgrade-only `SIMD_BACKEND` override.

## Benchmarks

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
```

Pin the run to one core and record the CPU, operating system, Rust version,
selected backend, and geometry with any quoted number. `BENCHMARKS.md` holds
the current measurements.

## Before opening a PR

```sh
just validate
```

This runs formatting, clippy at both feature ends, the dependency allowlist,
rustdoc, the feature matrix, the backend-tier tests, Miri, and coverage.

New public items need doc comments. MSRV is 1.93 and is checked in CI; do not
reach for newer standard-library APIs without raising it deliberately.

## Commit messages

Subject lines are at most ~10 words and carry the change at a glance
(`fgf: short verb phrase`). Implementation detail belongs in the Pull Request
and `CHANGELOG.md`, not the message, and messages never reference planning
artifacts.
