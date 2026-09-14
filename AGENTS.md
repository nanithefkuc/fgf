# Repository Guidelines

This file is the operational manual for changing `fgf`. Public behavior belongs
in rustdoc and `README.md`; measurements belong in `BENCHMARKS.md`; user-visible
changes belong in `CHANGELOG.md`.

## Required workflow

Work from the crate root and use `just` for routine commands.

```sh
just test [ARGS]       # host's selected backend
just test-tiers        # every supported backend tier
just features          # no-default, default, all-features
just features-alloc    # alloc without std
just lint              # rustfmt and clippy at both feature ends
just doc               # rustdoc with warnings denied
just unsafe-check      # scalar-path Miri cases
just cover             # merged per-tier coverage, 95% minimum
just validate          # complete pull-request gate
```

Run `just validate` before submitting a change. Do not replace a recipe with a
bare Cargo command; fix the recipe when its supported behavior is insufficient.
The MSRV is Rust 1.93.

`justfile` is a shared, byte-identical command surface. Do not edit it here.
Crate-specific values and recipes belong in `crate.just`.

## Change discipline

- Preserve stable, fixed-width, little-endian field encodings.
- Preserve total scalar arithmetic: `inv(0) == 0` and `x / 0 == 0`.
- Keep inherent scalar operations `const` where the surrounding family is
  `const`.
- Validate public buffer geometry before dispatch. Use checked arithmetic for
  row spans and name the failing operand in panic messages.
- Keep hot steady-state operations allocation-free. Prepare coefficients and
  lookup data outside repeated calls.
- Gate owned prepared collections on `alloc`, not `std`.
- Do not serialize prepared coefficients; serialize scalar elements and rebuild
  preparation for the current process backend.
- Extend existing macros and family patterns instead of creating parallel
  implementations.
- Do not add a second spelling for an existing operation. Follow the public
  naming and argument-order grammar already used by `ops`.

Backend detection and ordering belong to `simdispatch`. `SIMD_BACKEND` is a
process-startup, downgrade-only request. Do not add local CPU detection, another
override, or per-call backend resolution. Use `backend_for::<F>()` when behavior
depends on a field's implemented backend.

## SIMD and unsafe code

Safe scalar behavior is the differential oracle. Add or change the scalar path
before an optimized path when no independent oracle exists.

Architecture kernels stay under `src/kernel/{x86,aarch64,wasm32}/`. A kernel
must take the exact Archmage capability token required by its instructions.
Dispatch resolves and caches that token once. Helpers use `#[archmage::rite]`
only within a token-proven entry.

Do not introduce module-level `#![allow(unsafe_code)]`. Every remaining unsafe
item needs a narrow `#[allow(unsafe_code)]`, and every unsafe operation needs an
adjacent SINCE–THUS proof with one labelled pair per obligation:

```text
// SAFETY:
// MEMORY VALIDITY
// SINCE: <local fact or named invariant>.
// THUS: <the exact requirement established by that fact>.
```

Prefer safe Archmage reference intrinsics. Unsafe is limited to cases their API
cannot express: offset-addressed disjoint rows, uninitialized scratch,
aligned or non-temporal stores, and provider callbacks with caller-owned
invariants. Update this list when a new residue class is unavoidable.

The x86 AVX-512 implementation is deferred and exposed only under `internals`.
Do not add it to production dispatch without executable differential coverage
and pinned measurements on AVX-512 hardware.

## Tests

Tests must defend observable contracts, not implementation wiring.

- `tests/algebra.rs`: field laws, representations, generators, independent
  arithmetic oracles.
- `tests/ops.rs`: checked public operations, geometry failures, prepared forms,
  empty inputs, and zero/one coefficients.
- `tests/bits.rs`: bit-packed GF(2) behavior and frozen layout conventions.
- `src/kernel/tests.rs`: direct scalar-versus-architecture differentials across
  lane, tail, row, and source-count boundaries.
- `tests/zero_alloc.rs`: allocation-free steady-state contracts.
- `tests/internals.rs`: explicitly unstable direct-kernel surfaces.

Reuse the deterministic test helpers and fixed-seed noise generator. Do not add
nondeterministic randomness or duplicate oracle conventions. GF(2^16) byte
lengths must be even. Unsupported direct-kernel tests report a skip and return;
they are not marked ignored.

A behavior fix should retain a regression test when a plausible future bug
would fail it. Do not add tests that assert source text, field copies, forwarding,
or exact panic prose.

`TIERS` is `v3_gfni_crypto v3 v2 scalar`. A forced tier on incapable hardware
may resolve to another supported tier; inspect reported backends before treating
a green run as ISA coverage.

Coverage excludes the x86 field-kernel subtrees and deferred AVX-512 code that
CI cannot execute. Keep `COV_IGNORE` narrow and document every exclusion here.

## Benchmarks

Use only the benchmark recipes, pinned to an identified core:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
```

`kernels` reports the public operation shapes. `compare` includes the in-process
`reed-solomon-erasure` comparison. `affine` and `dot_product` are internal
investigation harnesses, not headline public benchmarks.

Benchmark setup, allocation, coefficient construction, and input generation
must stay outside the timed region. Record the CPU, OS, Rust version, selected
backend, geometry, command, and aggregation rule in `BENCHMARKS.md`. Never
reuse an old number or claim a performance change from an unpinned run.

A performance change requires an unchanged callable control, differential
correctness, and interleaved before/after measurements in one session. Do not
change dispatch or a crossover from reasoning alone.

## Documentation and review

- Rustdoc owns item contracts, invariants, panics, safety, layout, and ownership.
- `README.md` owns user-facing scope, installation, features, and examples.
- `BENCHMARKS.md` owns reproducible current measurements.
- `CHANGELOG.md` owns user-visible changes and migration notes.
- Public files and source comments must not reference planning artifacts.
- Use third person and present tense. Avoid history, temporal status, marketing,
  and benchmark numbers outside `BENCHMARKS.md`.
- Every public item and module needs a one-line summary; `just doc` treats
  warnings as errors.

Review every exported-symbol change across all callers. Use a clean cutover:
migrate every caller and remove obsolete aliases, wrappers, comments, and
re-exports.

Commit subjects use `fgf: short verb phrase` and stay near ten words. Put what
changed and why in the pull request and `CHANGELOG.md`, not in a commit-message
essay.
