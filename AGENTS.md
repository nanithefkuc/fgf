# Repository Guidelines

This file is the operational manual for changing `fgf`. Public behavior belongs
in rustdoc and `README.md`; measurements belong in `BENCHMARKS.md` and
`benchmarks/`; user-visible changes belong in `CHANGELOG.md`.

## Required workflow

Work from the crate root and use `just` for routine commands.

```sh
just test [ARGS]       # host's selected backend
just test-tiers        # every supported backend tier
just features          # no-default, default, all-features
just features-alloc    # alloc without std
just cross-check       # AArch64 and Wasm library compilation
just lint              # rustfmt and clippy at both feature ends
just doc               # rustdoc with warnings denied
just unsafe-check-gfni # owned-unsafe GFNI Miri cases
just cover             # merged per-tier coverage, 95% minimum
just cover-lcov        # export merged coverage profiles to coverage.lcov
just validate          # complete pull-request gate
just example NAME [ARGS] # one educational example, with Cargo options
just examples [ARGS]     # all educational examples, excluding timing probes
```

`just validate` does not run the GFNI Miri recipe; the dedicated `miri-gfni`
CI job runs `just unsafe-check-gfni`.

Run `just validate` before submitting a change. Do not replace a recipe with a
bare Cargo command; fix the recipe when its supported behavior is insufficient.
The MSRV is Rust 1.93.

`justfile` is a shared, byte-identical command surface. Do not edit it here.
Crate-specific values and recipes belong in `crate.just`.

The example recipes default to the normal library features. Run
`just examples --no-default-features --features alloc` for the portable
prepared-operation surface; scalar arithmetic, extension arithmetic,
embedding, and binary-syndrome examples also run without `alloc`.

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

The x86 subtree splits on field family, then instruction set: every family
directory holds one file per ISA its kernels compile for (`ssse3`/`sse2`/
`sse42`, `avx2`, `gfni`, `avx512`), and a file splits again into a directory
by fan shape only when it outgrows one file (`gf8/gfni/`, `gf8/avx512/`,
`gf16/gfni/`). Entry names carry the same ISA suffix as their file, in the
order operation, ISA, `_with` (`mul_add_matrix_at_gfni`,
`mul_add_matrix_avx512_with`). Byte-field entries are generic over the
polynomial or its typed prepared coefficient, without field-specific suffixes.
A helper private to one file carries no ISA or width marker; a helper shared
across sibling files carries its defining file's ISA
suffix (`bmul_gfni`, `store_avx2`, `fold_avx2`), and a half-width variant
inside a wider tier is `_half`. The
AVX-512 kernels dispatch on `V4x` under `simd512`; `Gf16` scatter and gather
retain the narrower GFNI path. Direct differentials execute on AVX-512
hardware, and the per-shape decisions are recorded in `benchmarks/gf8.md` and
`benchmarks/gf16.md`. The byte-field and `Gf16` `mul_add` alignment peel floors
remain measured thresholds with a benchmark record and misaligned-row coverage
in kernel tests.

## Residue ledger

Unsafe that survives the safe conversion ladder is limited to the cases below.
The CPU token proves instruction availability; it does not prove bounds,
initialization, alignment, or aliasing. Each listed item has a per-item
`#[allow(unsafe_code)]` and a SINCE–THUS proof at its site.

| Where | Why unsafe remains |
|---|---|
| `kernel/aarch64/gf8.rs`: `mul_add_scatter_impl`, `scatter_quad`, `mul_add_matrix_impl`, `matrix_quad`, `matrix_single` | The kernels update multiple rows selected by runtime offsets into one uniquely borrowed flat buffer. The checked entry proves each span and the rows' disjointness; safe Rust cannot express those offset-addressed mutable windows together. |
| `kernel/aarch64/gf16.rs`: `xor_row`, `scatter_group`, `mul_add_scatter_impl`, `matrix_group`, `mul_add_matrix_impl` | The scatter and matrix bodies operate on grouped, disjoint row windows addressed within one live allocation. The entry establishes row bounds and disjointness, but the borrow checker cannot split rows selected by runtime offsets. |
| `kernel/x86.rs`: `store_avx2`, `store_sse2` | The store helpers expose raw-pointer vector stores. Their callers prove each writable window; streaming stores additionally require 32-byte or 16-byte alignment and a fence before another thread observes the writes. |
| `kernel/x86/gf16/ssse3.rs`: `mul_add_scatter_ssse3`, `scatter_rows`, `mul_add_matrix_ssse3_with`, `matrix_rows` | Scatter and matrix groups write row windows selected by offsets into one destination allocation. The entry validates the row span, complete rows, and source bounds; the bodies rely on pairwise-disjoint windows that safe Rust cannot represent as simultaneous slices. |
| `kernel/x86/gf16/avx2.rs`: `mul_into_impl`, `mul_add_scatter_avx2`, `scatter_rows` | The overwrite lane uses the aligned streaming-store primitive: checked slices bound every lane, the peel proves alignment, and the caller fences the stores. The scatter group updates offset-addressed disjoint rows within one borrowed flat buffer. |
| `kernel/x86/gf16/gfni/single.rs`: `mul_into_impl` | The GFNI overwrite lane supports streaming stores through a raw-pointer helper. The caller peels the destination to the required boundary, keeps each lane in-bounds, and fences before observation. |
| `kernel/x86/gf16/gfni/scatter.rs`: `mul_add_scatter_gfni`, `scatter_group` | Scatter updates rows addressed by offsets within one borrowed flat buffer. The entry validates the row span and the body relies on pairwise-disjoint row windows that safe Rust cannot represent as simultaneous slices. |
| `kernel/x86/gf16/gfni/matrix.rs`: `mul_add_matrix_gfni_with`, `matrix_group` | Each matrix group writes multiple row windows selected by offsets into one destination allocation. Checked geometry establishes complete rows, source bounds, and disjointness; safe mutable slices cannot represent the grouped offset windows. |
| `kernel/x86/gf8/ssse3.rs`: `mul_into_ssse3_impl`, `mul_add_scatter_ssse3`, `mul_add_matrix_ssse3_with` | The overwrite loop uses raw vector stores, including aligned-only streaming variants; slice tiles prove memory validity and the aligned peel and fence prove the streaming-store obligations. Scatter and matrix kernels batch writes to row windows selected by offsets in one allocation; the checked entry proves the spans and the body preserves disjointness. |
| `kernel/x86/gf8/avx2.rs`: `mul_into_impl`, `mul_add_scatter_avx2`, `mul_add_matrix_avx2_with`, `matrix_tiles`, `matrix_vector` | As `kernel/x86/gf8/ssse3.rs`, at 32-byte lanes. |
| `kernel/x86/gf8/gfni/single.rs`: `mul_into_gfni_impl` | The overwrite body writes vector tiles through raw pointers; the tile split bounds every store, while the non-temporal branch additionally relies on the alignment peel and a final fence. |
| `kernel/x86/gf8/gfni/scatter.rs`: `mul_add_scatter_impl`, `scatter_rows4`, `scatter_rows2`, `scatter_span` | Scatter groups update disjoint rows by offsets into one destination allocation. The checked caller establishes each row's bounds and the grouped body maintains non-aliasing across stores. |
| `kernel/x86/gf8/gfni/matrix.rs`: `matrix_block`, `matrix_at_block` | Matrix rows are addressed by checked offsets into one destination region, advanced to the start of a column block that lies inside each row. The entry proves each row is in-bounds and pairwise disjoint; the borrow checker cannot encode those runtime-selected windows. |
| `kernel/x86/gf8/gfni/rows.rs`: `rows_body`, `rows_resolved`, `matrix_tail` | `rows_body` and `matrix_tail` operate on offset-addressed row pointers. `rows_resolved` additionally stages only the occupied prefix of `MaybeUninit` coefficient/source arrays; it reinterprets exactly the prefix written before reading it. |
| `kernel/x86/gf8/avx512/scatter.rs`: `mul_add_scatter_impl`, `scatter_rows4`, `scatter_rows2`, `scatter_span` | 64-byte byte-field scatter groups update disjoint rows by offsets into one destination allocation. The checked entry proves each row's bounds and disjointness; the grouped bodies keep those windows across 64-byte reference loads/stores. Dispatched on `V4x` under `simd512`. |
| `kernel/x86/gf8/avx512/matrix.rs`: `matrix_block`, `matrix_rows1`, `rows_tile4`, `rows_lane4`, `rows_tile2`, `rows_lane2`, `rows_tile1`, `rows_lane1`, `rows_resolved`, `rows_body`, `matrix_tail`, `matrix_at_block` | 64-byte byte-field matrix groups address rows by checked offsets into one region (contiguous or scattered), advanced to the start of a column block that lies inside each row. The entries prove each row in-bounds and pairwise disjoint; the tile bodies and tails preserve disjointness across 64-byte reference loads/stores. Direct tile and lane bodies use checked indexing on each captured provider source. `rows_resolved` additionally stages only the occupied prefix of `MaybeUninit` map-word/source arrays; it reinterprets exactly the prefix written before reading it. Dispatched on `V4x` under `simd512`. |
| `external/gf16-bench/src/native.rs`: foreign declarations, `Native::new`, `complete_mul`, `leopard_mul`, `complete_region`, `complete_assign`, `leopard_region`, `Drop` | Benchmark-only C ABI calls require a live uniquely owned GF-Complete context, initialized Leopard tables, a retained CPU capability token, and checked buffer geometry. Native contexts cannot cross threads. The declaration proof matches the bridge ABI; each wrapper proves its memory, aliasing and feature obligations. |

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
- `tests/tower_forward.rs`: frozen raw-word fixtures pinning the unified
  tower packing against the shipped tower identities.
- `tests/embedding.rs`: reference-ladder forward fixtures, inverse and
  commuting-triangle contracts, trace/norm, and failed restriction.
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

`TIERS` is `v4x v3_gfni_crypto v3 v2 scalar` (`v4x` resolves only with `simd512`). A forced tier on incapable hardware
may resolve to another supported tier; inspect reported backends before treating
a green run as ISA coverage.

Coverage excludes the x86 binary-field kernel subtrees and the GFNI dispatch
arms in `kernel/gf8.rs` and `kernel/tower.rs`, which require GFNI unavailable
on the GitHub-hosted coverage runner. It also excludes only
`kernel/x86/quad_mersenne31/avx512.rs`: its V4x instructions cannot execute on
that runner. The AVX2 kernel and mixed-ISA prime dispatch modules remain
included. Direct AVX-512 differentials run on capable hosts.
Keep `COV_IGNORE` narrow and document every exclusion here.

## Benchmarks

Use only the benchmark recipes, pinned to an identified core:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
FEC_GOLDEN_CORE=<cpu> just bench-gdl-comp
FEC_GOLDEN_CORE=<cpu> just bench-m31-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf2-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf16-comp
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
just bench-build NAME # build a portable artifact without running it
FEC_GOLDEN_CORE=<cpu> just bench prime_ntt
```

`kernels` reports the public operation shapes and the self-comparison panels;
the family flags select panels, and no flag runs every panel. The
`--network-diagnostic` flag isolates matched base offsets and network row
pitches for the byte-field payload operations.
`compare` reports `fgf` self-numbers over the competitor-harness fixture
family. Its `--peel-diagnostic` flag isolates matched source and destination
offsets around the AVX-512 peel floors.
`prime_ntt` interleaves the QuadMersenne31 scalar control against the AVX2
kernels per row length; its campaign set the dispatch thresholds in
`src/kernel/quad_mersenne31.rs`.

`prime_ntt` is a dispatch-threshold investigation, not part of the public
snapshot campaign. Its raw evidence stays in git-ignored `bench-records/`.

Every external competitor harness lives in `external/`, one separate
unpublished package per harness with its own `build.rs`, outside `src/` and
`benches/` because linking the competitor needs a build script and `fgf`
itself must never carry one; the package allowlist keeps `external/` out of
`cargo package`. `external/bench-trio/` links the system Intel ISA-L and
klauspost/reedsolomon as a Go C archive and interleaves all three arms —
`fgf`, ISA-L, klauspost — over one GF(2^8)/`0x11D` fixture set, rotating
through all six arm orders; the klauspost coding matrix arrives through
`WithCustomMatrix`. It runs through `bench-gf-comp` inside its shuffled
rounds, validates every arm against the competitor's own output before
timing, and opens with a control row that must read 1.00x. ISA-L needs
`libisal` discoverable through pkg-config and the klauspost arm needs the
Go toolchain in PATH; the build fails loudly when its toolchain is absent
rather than dropping the arm.

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
kernels at compile time. The banner reports packing widths; the record
states the flag and widths alongside the resolved `fgf` backends.

`external/gf16-bench/` compares native GF16 field operations against pinned
GF-Complete, reed-solomon-erasure, reed-solomon-simd and Leopard-RS 1.x
(pre-Leopard2). Leopard2 exposes no public field-region primitive, only codec
operations, so the harness pins the 1.x field kernels.
Its representation maps are bijective and preserve every basis product;
independent scalar arithmetic validates every native scalar and timed region.
Single-row cases cover fixed and cycling coefficients. Multi-row competitors
compose native region operations; unsupported primitives remain absent.
The duplicate fgf arm in every case measures bias. Each sampling cycle covers
every arm permutation, and each arm warms immediately before its timed batch.
Sampling stops only after a complete permutation cycle.
`bench-gf16-comp` runs the targeted five-round campaign; `bench-gf-comp`
includes the same harness in its shuffled binary-field rounds. `gf16-check`
validates all geometries without timing, `gf16-smoke` exercises a short panel,
and `gf16-lint` checks the external package. The native build requires Git,
C/C++ compilers and an x86-64 AVX2/GFNI/crypto host; missing inputs fail rather
than remove an arm. `gf16-fmt` formats only that package.

Benchmark setup, allocation, coefficient construction, and input generation
must stay outside the timed region. Record the CPU, OS, Rust version, selected
backend, geometry, command, and aggregation rule in the benchmark record. Never
reuse an old number or claim a performance change from an unpinned run.

The benchmark record is a current snapshot, not a measurement ledger. A
refresh runs all four family campaigns on both hosts' isolated CPUs in one
session and rewrites every published value, with the median of five per-run
medians for every published case. Competitor arms interleave inside each
harness process. Dates, before/after comparisons, threshold variants,
direct-kernel timings, and historical source fingerprints stay out of the
public snapshot. Raw outputs and historical measurement evidence stay in
`bench-records/`.

The record is split by field family. `BENCHMARKS.md` is the index: one row
per family page naming its fields, campaign recipe, and competitors, then the
host table, the method shared by every page, the number format, and the
reproduction commands. Each family page lives at `benchmarks/<family>.md`,
named for the field it measures, and holds a `Setup` table with only that
page's campaign, resolved backend, competitor versions and pins, build,
protocol, and throughput numerator; then `Self-timings`; then `Competitors`;
then a plain caveat list. A family with no competitor states that no matched
competitor measurement is available. A new field adds a page and one index
row; it never extends the shared method with page-specific provenance. A
table covers one family, so a self-timing table never compares fields from
different pages, and a competitor table includes only the arms that
implement every row. Arms differing in supported primitives get separate
tables, never `-` columns.

Every result cell is `Tiger Lake / Golden Cove`, in that order. Within one
table every value has the same digit count, counting leading zeros, so
paired cells line up down each column. The count is the fewest digits that
give the table's smallest value three significant figures, capped by the
resolution the harness prints for that table's values. Trailing zeros are
printed (`32.0`, `0.70`), integer parts are never rounded or written in
scientific notation, and ratios keep five digits. Different tables may use
different counts. Cells are generated from raw per-run values with
round-half-up at that count, never padded from previously published cells.

A performance change requires an unchanged callable control, differential
correctness, and interleaved before/after measurements in one session. Do not
change dispatch or a crossover from reasoning alone.

## Documentation and review

- Rustdoc owns item contracts, invariants, panics, safety, layout, and ownership.
- `README.md` owns user-facing scope, installation, features, and examples.
- `BENCHMARKS.md` and `benchmarks/` own reproducible current measurements.
- `CHANGELOG.md` owns user-visible changes and migration notes.
- Public files and source comments must not reference planning artifacts.
- Use third person and present tense. Avoid history, temporal status, marketing,
  and benchmark numbers outside `benchmarks/`.
- Every public item and module needs a one-line summary; `just doc` treats
  warnings as errors.

Review every exported-symbol change across all callers. Use a clean cutover:
migrate every caller and remove obsolete aliases, wrappers, comments, and
re-exports.

Commit subjects use `fgf: short verb phrase` and stay near ten words. Put what
changed and why in the pull request and `CHANGELOG.md`, not in a commit-message
essay.
