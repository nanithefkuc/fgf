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

## Residue ledger

Unsafe that survives the safe conversion ladder (K5), one entry per item.
Each carries a per-item `#[allow(unsafe_code)]` and a SINCE–THUS proof at the
site. The `aarch64` kernel subtree retains only the offset-addressed-row
class:

- `kernel/aarch64/gf8.rs`: `mul_add_scatter_impl`, `scatter_quad`,
  `mul_add_matrix_impl`, `matrix_quad`, `matrix_single`.
- `kernel/aarch64/gf16.rs`: `xor_row`, `scatter_group`,
  `mul_add_scatter_impl`, `matrix_group`, `mul_add_matrix_impl`.

All ten record row pointers as offsets into one uniquely borrowed flat row
buffer, a shape the borrow checker cannot express. Memory validity and
disjointness arguments are stated per obligation in each item's proof.

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

Coverage excludes the x86 field-kernel subtrees, the deferred AVX-512 code,
and the GFNI dispatch arms in `kernel/gf8.rs` and `kernel/tower.rs` — none of
which a GitHub-hosted runner can execute, since those hosts have no GFNI.
Keep `COV_IGNORE` narrow and document every exclusion here.

## Benchmarks

Use only the benchmark recipes, pinned to an identified core:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
FEC_GOLDEN_CORE=<cpu> just bench-isal
FEC_GOLDEN_CORE=<cpu> just bench-klauspost
FEC_GOLDEN_CORE=<cpu> just bench-prime-ntt
```

`kernels` reports the public operation shapes. `compare` includes the in-process
`reed-solomon-erasure` comparison. `affine` and `dot_product` are internal
investigation harnesses, not headline public benchmarks. `prime_ntt`
interleaves the QuadMersenne31 scalar control against the AVX2 kernels per
row length; its campaign set the dispatch thresholds in
`src/kernel/quad_mersenne31.rs`.

`bench-isal` builds and runs `bench-isal/`, a separate unpublished package
that links the system Intel ISA-L and interleaves it against `fgf` over
GF(2^8)/`0x11D`. It lives outside `src/` and `benches/` because linking the
C library needs a build script and `fgf` itself must never carry one; the
package allowlist keeps it out of `cargo package`. Every arm validates
against ISA-L's own output before timing, and the first table row is a
control that must read 1.00x.

`bench-klauspost` builds and runs `bench-klauspost/`, a separate unpublished
package that links klauspost/reedsolomon as a Go C archive and interleaves it
against `fgf` over GF(2^8)/`0x11D`. It lives outside `src/` and `benches/`
for the same reason as `bench-isal`: the Go link needs a build script and
`fgf` itself must never carry one. Requires the Go toolchain in PATH; the
build fails loudly when it is absent rather than dropping the arm. Every arm
supplies the coding matrix through `WithCustomMatrix` and validates against
the library's own output before timing, and the first table row is a control
that must read 1.00x.

`bench-prime-ntt` builds and runs `external-bench/prime-ntt/`, a separate
unpublished package that drives the real `butterfly-fft` NTT plans and its
`internals` schedule controls against the `fgf` working tree, so a kernel
change is measured through the consumer that depends on it without touching
any published pin. The package needs its `[patch.crates-io]` table for the
same reason a downstream consumer cannot mix sources: `butterfly-fft`
arrives from crates.io and declares its own registry dependency on `fgf`
and `simdispatch`, while the harness names the local `fgf` by path. Without
the patch, cargo would build two `fgf` copies — the working tree for the
harness's direct calls and the registry release inside `butterfly-fft` —
with incompatible types and no shared backend selection. The patch
redirects the registry's `fgf` and `simdispatch` onto the same local paths,
so the whole graph resolves one copy of each. `Goldilocks` and
`QuadMersenne31` are validated at every geometry — forced fused against
forced packed against production forward, inverse roundtrip, and a direct
DFT up to size 1024 — before any arm is timed; a mismatch panics instead of
producing numbers, and the harness prints the process and per-field
backends. Criterion filters apply (`just bench-prime-ntt goldilocks/n1024`),
and `--test` runs each arm once without timing. The package sits behind a
`.gitignore` path exception that keeps `prime-ntt/` tracked while its build
products stay ignored.

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
