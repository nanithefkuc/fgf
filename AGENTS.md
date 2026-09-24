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
just unsafe-check-gfni # owned-unsafe GFNI Miri cases
just cover             # merged per-tier coverage, 95% minimum
just validate          # complete pull-request gate
```

`just validate` does not run the GFNI Miri recipe; the dedicated `miri-gfni`
CI job runs `just unsafe-check-gfni`.

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

Unsafe that survives the safe conversion ladder is limited to the cases below.
The CPU token proves instruction availability; it does not prove bounds,
initialization, alignment, or aliasing. Each listed item has a per-item
`#[allow(unsafe_code)]` and a SINCE–THUS proof at its site.

| Where | Why unsafe remains |
|---|---|
| `kernel/aarch64/gf8.rs`: `mul_add_scatter_impl`, `scatter_quad`, `mul_add_matrix_impl`, `matrix_quad`, `matrix_single` | The kernels update multiple rows selected by runtime offsets into one uniquely borrowed flat buffer. The checked entry proves each span and the rows' disjointness; safe Rust cannot express those offset-addressed mutable windows together. |
| `kernel/aarch64/gf16.rs`: `xor_row`, `scatter_group`, `mul_add_scatter_impl`, `matrix_group`, `mul_add_matrix_impl` | The scatter and matrix bodies operate on grouped, disjoint row windows addressed within one live allocation. The entry establishes row bounds and disjointness, but the borrow checker cannot split rows selected by runtime offsets. |
| `kernel/x86.rs`: `store256`, `store128` | The store helpers expose raw-pointer vector stores. Their callers prove each writable window; streaming stores additionally require 32-byte or 16-byte alignment and a fence before another thread observes the writes. |
| `kernel/x86/avx512.rs`: `xor`, `xor_impl`, `gf8_mul_add`, `gf8_mul_add_impl`, `gf8_mul_assign`, `gf8_mul_assign_impl`, `gf8_mul_into`, `gf8_mul_into_impl`, `gf8_mul_elementwise`, `gf8_mul_elementwise_impl`, `gf16_mul_add`, `gf16_mul_add_impl`, `gf16_mul_assign`, `gf16_mul_assign_impl`, `gf16_mul_into`, `gf16_mul_into_impl`, `gf16_mul_elementwise`, `gf16_mul_elementwise_impl`, `swap_mask` | AVX-512 intrinsics require feature-enabled call boundaries and raw vector loads/stores over complete lanes. The wrappers' slice geometry bounds those windows; `swap_mask` also loads a statically aligned vector. This file is deferred and remains outside production dispatch. |
| `kernel/x86/gf16/gfni.rs`: `mul_into_gfni_lane` | The GFNI overwrite lane supports streaming stores through a raw-pointer helper. The caller peels the destination to the required boundary, keeps each lane in-bounds, and fences before observation. |
| `kernel/x86/gf16/matrix.rs`: `mul_add_matrix_gfni_with`, `matrix_group_gfni`, `mul_add_matrix_avx2_with`, `matrix_group_avx2`, `mul_add_matrix_ssse3_with`, `matrix_rows_ssse3` | Each matrix group writes multiple row windows selected by offsets into one destination allocation. Checked geometry establishes complete rows, source bounds, and disjointness; safe mutable slices cannot represent the grouped offset windows. |
| `kernel/x86/gf16/nibble.rs`: `mul_into_avx2_lane` | The nibble-shuffle overwrite lane uses the same aligned streaming-store primitive: checked slices bound every lane, the peel proves alignment, and the caller fences the stores. |
| `kernel/x86/gf16/scatter.rs`: `mul_add_scatter_gfni`, `scatter_group_gfni`, `mul_add_scatter_avx2`, `scatter_rows_avx2`, `mul_add_scatter_ssse3`, `scatter_rows_ssse3` | Scatter updates rows addressed by offsets within one borrowed flat buffer. The entry validates the row span and the body relies on pairwise-disjoint row windows that safe Rust cannot represent as simultaneous slices. |
| `kernel/x86/gf8/gfni.rs`: `mul_into_gfni_impl`, `mul_into_affine_impl` | These overwrite lanes write vector tiles through raw pointers; the tile split bounds every store, while the non-temporal branch additionally relies on the alignment peel and a final fence. |
| `kernel/x86/gf8/matrix.rs`: `mul_add_matrix_impl`, `mul_add_matrix_at_impl` | Matrix rows are addressed by checked offsets into one destination region. The entry proves each row is in-bounds and pairwise disjoint; the borrow checker cannot encode those runtime-selected windows. |
| `kernel/x86/gf8/nibble.rs`: `mul_into_avx2_impl`, `mul_into_ssse3_impl` | The overwrite loops use raw vector stores, including aligned-only streaming variants. Slice tiles prove memory validity; the aligned peel and fence prove the streaming-store obligations. |
| `kernel/x86/gf8/nibble_rows.rs`: `mul_add_scatter_avx2`, `mul_add_scatter_ssse3`, `mul_add_matrix_avx2_with`, `matrix_tiles_avx2`, `matrix_vector_avx2`, `mul_add_matrix_ssse3_with` | Scatter and matrix kernels batch writes to row windows selected by offsets in one allocation. The checked entry proves the spans and row geometry; the unsafe body preserves disjointness across vector stores and tails. |
| `kernel/x86/gf8/rows.rs`: `rows_body`, `rows_resolved_chunked`, `matrix_tail` | `rows_body` and `matrix_tail` operate on offset-addressed row pointers. `rows_resolved_chunked` additionally stages only the occupied prefix of `MaybeUninit` coefficient/source arrays; it reinterprets exactly the prefix written before reading it. |
| `kernel/x86/gf8/scatter.rs`: `mul_add_scatter_impl`, `scatter_rows4`, `scatter_rows2`, `scatter_span` | Scatter groups update disjoint rows by offsets into one destination allocation. The checked caller establishes each row's bounds and the grouped body maintains non-aliasing across stores. |
| `kernel/x86/gf8/experiments/grouped.rs`: `mul_into_matrix_chunk_8d`, `mul_into_matrix_external_grouped_8d` | Experimental grouped matrix kernels pass offset-addressed rows to pointer-based inner loops. Their checked entry proves row bounds and disjointness, plus the external coefficient/source ordering required by the walk. |
| `kernel/x86/gf8/experiments/resolve.rs`: `resolve_probe_8d` | The probe stages resolved coefficients and source references in `MaybeUninit` arrays to avoid initializing unused slots. It writes and reinterprets only the occupied prefix. |
| `kernel/x86/gf8/experiments/shuffle.rs`: `mul_into_matrix6_shuffle_impl`, `packed_nibble_product`, `mul_into_matrix6_shuffle_packed_impl`, `mul_into_matrix2_shuffle_body` | Matrix bodies use offset-addressed disjoint row windows and cache source/table pointers to avoid repeated bounds checks. The checked layer bounds every row, source, and packed table record; the pointer walk relies on those caller-established invariants. |
| `kernel/x86/gf8/experiments/tile.rs`: `mul_into_matrix2_checked`, `matrix2_body` | The two-row body keeps two checked, disjoint offset windows as raw pointers. Its streaming-store specialization additionally relies on the entry's aligned-buffer and row-length checks, then fences before observation. |

The `wasm32` kernel subtree retains no unsafe code: reference-based
`v128_load`/`v128_store` over 16-byte chunk arrays and `split_at_mut` row
groups express its memory access safely.

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
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
FEC_GOLDEN_CORE=<cpu> just bench-gdl-comp
FEC_GOLDEN_CORE=<cpu> just bench-m31-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf2-comp
```

The `bench-<field>` recipes run that family's self benchmarks in one pass;
`bench-<field>-comp` runs five complete rounds interleaving the self suite
with every wired competitor harness, shuffling the unit order within each
round and printing the realized order. `gf` is the binary tower fields,
`gdl` Goldilocks, `m31` Mersenne31 and QuadMersenne31, `gf2` bit-packed
GF(2). The granular self entry points stay available:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels [--gf|--gdl|--m31|--gf2]
FEC_GOLDEN_CORE=<cpu> just bench compare
FEC_GOLDEN_CORE=<cpu> just bench prime_ntt
```

`kernels` reports the public operation shapes and the self-comparison
panels; the family flags select panels, and no flag runs every panel.
`compare` reports `fgf` self-numbers over the competitor-harness fixture
family plus the six-row shuffle comparison. `affine` and `dot_product` are
internal investigation harnesses, not headline public benchmarks.
`prime_ntt` interleaves the QuadMersenne31 scalar control against the AVX2
kernels per row length; its campaign set the dispatch thresholds in
`src/kernel/quad_mersenne31.rs`.

Every external competitor harness lives in `external/`, one separate
unpublished package per harness with its own `build.rs`, outside `src/` and
`benches/` because linking the competitor needs a build script and `fgf`
itself must never carry one; the package allowlist keeps `external/` out of
`cargo package`. `external/bench-isal/` links the system Intel ISA-L and
interleaves it against `fgf` over GF(2^8)/`0x11D`;
`external/bench-klauspost/` links klauspost/reedsolomon as a Go C archive
and interleaves it against `fgf` over the same field, supplying the coding
matrix through `WithCustomMatrix`. Both run through `bench-gf-comp` inside
its shuffled rounds, validate every arm against the competitor's own output
before timing, and open with a control row that must read 1.00x. ISA-L needs
`libisal` discoverable through pkg-config and the klauspost harness needs
the Go toolchain in PATH; each build fails loudly when its toolchain is
absent rather than dropping the arm.

`external/prime-bench/` is the prime-field competitor harness: it compares
`fgf` against Plonky3 (`p3-goldilocks`, `p3-mersenne-31`, and the packed
quadratic extension of `p3-mersenne-31`) over Mersenne31, Goldilocks, and
QuadMersenne31 on each library's native layout. Every arm is validated
against the other library's output before timing, and the table opens with
a control row that must read 1.00x. Layout conversion is excluded from
every timed region, so the comparison is between kernels rather than
encodings, and throughput counts one operand pair per element. It runs
through `bench-gdl-comp` and `bench-m31-comp` inside their shuffled rounds;
a single rerun is `RUSTFLAGS="-C target-cpu=native" cargo run --release`
from its directory with a `gdl` or `m31` family argument. The package
builds with `-C target-cpu=native` because Plonky3 selects its packed
kernels at compile time, so its arms measure the AVX2 tier only; the record
states the flag alongside the numbers.

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
