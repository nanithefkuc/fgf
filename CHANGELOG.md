# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `bench-klauspost/`, an in-process competitor comparison against
  `klauspost/reedsolomon` v1.14.2 over GF(2^8)/`0x11D`, run with
  `FEC_GOLDEN_CORE=<cpu> just bench-klauspost`. The package links the Go
  library as a C archive, interleaves it with the matched `fgf` operations
  over one fixture set, and validates every arm byte-for-byte before any
  timing. It replaces `catid/leopard` as the named codec-level competitor;
  the measured sweep is in `BENCHMARKS.md`, consolidated with the other
  competitor medians in its cross-library matrix.

### Changed

- The x86 fused overwrite kernels prefetch their destination. `mul_into`
  writes lines it never reads, so each ordinary store waited on the
  read-for-ownership fetch of the line it was about to replace whole.
  Naming those lines ahead of the cursor lifts single-source GF(2^8) and
  GF(2^16) overwrite throughput by about a third from 24 KiB up, and brings
  GF(2^8) to parity with Intel ISA-L's streaming `gf_vect_mul` without
  evicting the destination. Results are unchanged; buffers below 24 KiB and
  the non-temporal bodies above `NT_STORE_MIN` are untouched. The sweep
  that set the hint, the distance and the threshold is in `BENCHMARKS.md`.

## [1.0.0] - 2026-09-14

First stable release. The public API — fields, `ops`, `bits`, backend
reporting, and every stable byte encoding — is now under semantic versioning.
The `internals` feature stays explicitly unstable.

**Breaking: the MSRV is Rust 1.93.** The x86 kernels call safe intrinsics
inside their capability-token entries, which earlier compilers still require
an `unsafe` block for; 1.92 and below cannot build the crate.

### Added

- Inherent `neg` and `core::ops::Neg` on the binary-field elements — both
  flat GF(2^8) fields, the three tower levels, and the four Fan-Paar levels.
  In characteristic two negation is the identity, so `-x == x`; generic code
  over `F::Elem` can now reach for `-x` in every field the crate exposes.

- `field::gf8`: an umbrella module holding both flat GF(2^8) fields, since
  they are one construction under two reduction polynomials. `gf8b` and
  `gf8d` are now submodules of it, re-exported from `field` so every
  existing path — `fgf::gf8b::Elem`, `fgf::field::gf8d::Gf8D` — resolves
  unchanged.
- `field::tower` and `kernel::tower`: umbrella modules for the
  Rijndael-rooted quadratic tower. `gf16`, `gf32`, and `gf64` are now
  submodules of them, re-exported from `field` and `kernel` so every
  existing path resolves unchanged. The three levels share one macro-emitted
  scalar implementation, and the two GFNI-only kernel levels share one
  macro-emitted dispatch.

- The `alloc` feature: the prepared coefficient collections (`CoeffVec`,
  `CoeffMatrix`) and `pack_to_vec` own dynamically sized storage and now
  need only a heap, not an operating system; `std` implies `alloc`, so
  default-feature consumers are unaffected.
- `ops::CoeffMatrix::source` composes with `mul_add_scatter_with`: the
  borrowed `CoeffVecRef` view of one source's coefficients is exactly what
  the vector-shaped prepared operations consume, so a matrix column drives
  the scatter kernel with no copying or restaging.

- `Field::CHARACTERISTIC`: the field's characteristic as a first-class
  constant — `2` for every binary tower, the base prime `2^31 − 1` for
  `Mersenne31` and the quadratic extension `QuadMersenne31` (whose `ORDER`
  is `p²`), and the Goldilocks prime for `Goldilocks`. A required associated
  constant, so external `Field` implementers must add it; it cannot be
  derived from `ORDER`, which is the defect it exists to prevent.
- `ops::add_assign_rows`: pairwise row addition over two equal flat
  row buffers, `dst_row[j] += src_row[j]`. Semantically identical to
  `add_assign` (and currently implemented through it on every backend);
  the checked row geometry documents caller intent so backends may
  interleave independent row streams where a platform measures a win.
  The defaulted internal row-addition method carries the semantic
  contract; experimental four-stream interleaved XOR kernels for x86
  AVX2/SSE2 and an AArch64 NEON sketch are exposed behind `internals`,
  differentially tested, and deliberately not wired — they measured at
  parity or behind the flat XOR from L1 to DRAM on the reference host,
  and the additive-FFT derivative gap they targeted turned out to be a
  first-touch page-fault effect of out-of-place output buffers, not
  memory-level parallelism. See `BENCHMARKS.md`, "Row-interleaved XOR".
- `bits::XorRange` and `bits::xor_range_with`: the prepared form of
  `bits::xor_range`. The bit range's byte window and end masks are derived
  once and applied to many buffer pairs, the same prepare/apply split `ops`
  uses for coefficients; buffers of different lengths are accepted as long
  as each covers the planned window. `xor_range` itself now routes through
  the shared byte-window core, which replaces the word-assembly path for
  sub-word ranges (one or two masked byte operations) and the masked end
  words of longer ranges. On the eight-bit-range short-row shape the split
  measures 3.6x the one-shot form per call; see `BENCHMARKS.md`.
- `internals` entries for the crossed resolution x chunk panel:
  `matrix_overwrite1_fullinit_8d` (the production one-row path with the
  original full-array scratch policy), `matrix_overwrite1_chunk_8d` and
  `matrix_overwrite_chunk_8d` (resolved path at chunk 32/64/96, one row and
  grouped), `matrix_overwrite1_external_chunk_8d` and
  `matrix_overwrite_external_grouped_8d` (the production body over externally
  resolved maps, chunked or single-pass). Benchmark evidence only; production
  dispatch is unchanged. See `BENCHMARKS.md`, "Crossed resolution x chunk
  panel".
- `internals` entries for the plan-hoisting experiment:
  `matrix_affine_prepared_with` and `matrix_overwrite_affine_prepared_with`
  run the production row-group bodies over `Prepared8D` coefficients (the
  affine word read from the prepared form). Measured and rejected for the
  plan entries — parity-to-worse against the term-list baseline on both
  hosts, closing none of the gap to the resolution-free body — so production
  still
  resolves from scalar values; see `BENCHMARKS.md`, "Plan-hoisted
  resolution measured and rejected".
- CI now builds the docs with `-D warnings` in every feature configuration
  the test jobs exercise (default, `--no-default-features`,
  `--no-default-features --features std`, `--all-features`), compiles the
  `simd` feature for `wasm32-unknown-unknown` with `+simd128` enabled
  explicitly, and the MSRV job runs
  `cargo check --locked --all-features --all-targets` on Rust 1.89.

### Changed

- `README.md` leads with the crate's capabilities, installation, and operation
  tables; `BENCHMARKS.md` is a current two-host measurement record for the
  public operation shapes, replacing the accumulated experiment log.

- The Miri gate targets the scalar operation suite instead of interpreting
  every exhaustive algebra test and rustdoc example. The focused run keeps
  the checked geometry and portable-kernel coverage while removing work that
  the native validation stages already perform.

- **The `kernel::x86` subtree is restructured.** `x86/mod.rs` becomes
  `x86.rs` beside its directory, the field-independent XOR kernels move to
  `x86/bytes.rs`, and the three oversized kernel files are split along a
  named axis — multiply strategy first (`gfni`, `nibble`), then fan shape
  (`scatter`, `gather`, `matrix`) — leaving every implementation module at
  or under roughly 500 lines. GF(2^8)'s internals-only benchmark variants
  are gathered under `x86/gf8/experiments/`. No kernel body changed.

- **Kernel names converge on the `ops` grammar.** The domain operation now
  leads every x86 kernel name, and the shape qualifier follows it:
  `matrix_overwrite*` is `mul_into_matrix*`, `matrix_scattered*` is
  `mul_add_matrix_at*`, and `scatter_*`/`gather_*`/`matrix_*`/`elementwise_*`
  gain their `mul_add_`/`mul_elementwise_` prefixes. The AVX-512 family
  follows suit (`gf8_scatter` is `gf8_mul_add_scatter`). Renamed throughout
  the direct `kernel::x86` modules, so an `internals` consumer updates its
  paths; the benchmark identifiers in `BENCHMARKS.md` move with them.

- **The SSE Mersenne31 and Goldilocks kernels are named for the feature they
  require.** Every `*_sse41` entry point is `*_sse42`: the bodies enable
  `sse4.2` and depend on `pcmpgtq`, so the old name named the wrong ISA.

- **The x86 unsafe surface is countable.** The module-level
  `#![allow(unsafe_code)]` at the root of the subtree, which silently
  covered every child module, is replaced by a per-item
  `#[allow(unsafe_code)]` on each unsafe block and each `unsafe fn`.

- `kernel::x86::bytes::xor_avx2` and `xor_sse2` are direct kernel entries
  rather than wrappers. Both are safe `#[arcane]` functions that take a
  capability token and assert their own geometry. `xor_sse2` consequently
  takes the `X64V1Token` its instructions need instead of an `X64V2Token` it
  immediately narrowed.

- `gf2::Elem` stores a canonical byte as a type invariant: `to_raw` is now
  documented to return exactly `0` or `1` rather than "the byte as stored".
  Equality, hashing, and ordering are derived rather than masking on every
  comparison, and `canonical` is the identity. No reachable value changes;
  no constructor ever produced a wider pattern.

- The x86 byte-XOR kernels now use Archmage capability tokens and safe
  reference-based SIMD memory operations. Backend resolution caches the
  matching token with the selected tier, and differential coverage now spans
  unaligned inputs plus every AVX2/SSE2 lane and unroll boundary.

- All dispatched x86 kernel families now expose direct Archmage
  capability-token entries with release-mode geometry validation. Backend
  resolution caches the exact selected token once; steady-state calls pass
  that proof directly. The deferred AVX-512 compatibility entries are nested
  in `kernel::x86::avx512`, beside the implementation they validate.

- **Breaking: `ops::Plan` is split into `CoeffVec` and `CoeffMatrix`.** The
  one type served two incompatible interpretations — a flat coefficient
  vector and a source-major matrix — and reported dimensions
  `(1, coeffs.len())` for vector plans, so a gather plan's `source_count()`
  was really its output count. `CoeffVec::new(&[F::Elem])` is the flat
  vector, with no axes and no `source`/`get_at` accessors;
  `CoeffMatrix::from_source_major(sources, outputs, &[F::Elem])` is the
  source-major matrix. The prepared `mul_*_with` operations take
  `CoeffVecRef` by value at the vector shapes and `&CoeffMatrix` at the
  matrix shapes.
- **Breaking: the overwrite ops follow the ecosystem naming grammar.**
  `dot_product` → `mul_into_gather`, `dot_product_with` →
  `mul_into_gather_with`, `dot_product_matrix` → `mul_into_matrix`, and
  `dot_product_matrix_with` → `mul_into_matrix_with`: `mul_into_*` is the
  overwrite counterpart of `mul_add_*`. `mul_add_matrix_scattered` →
  `mul_add_matrix_at`, where `_at` marks the destination rows as addressed
  by explicit offsets. The sealed kernel-trait methods rename the same way.
- **Breaking: destination-first argument order for the row ops.**
  `ops::add_gather` is `ops::add_gather_offsets` (the name it already
  shared with the kernel-trait method) and now takes
  `(dst, region, offsets)` — `_offsets` marks the *sources* as
  offset-addressed into one backing region — and `ops::add_assign_rows`
  now takes `(dst, row_len, src)`, geometry second.
- **Breaking: `bits` surface renames and one reorder.** `bits::xor` is
  `bits::xor_assign`, `bits::parity_dot` is `bits::dot_product`, and
  `bits::RangeXor` is `bits::XorRange`; `bits::xor_gather` now takes
  `(dst, bits, selector, srcs)` — geometry, then the GF(2) coefficient
  vector, then sources.
- **Breaking: `Field::read`/`Field::write` are now `decode`/`encode`**, and
  the tower coordinate accessor `components()` is `to_components()`,
  pairing with `from_components()`.
- **Breaking: prime-field integer helpers share one verb.** The free
  `goldilocks::canonical` is `goldilocks::reduce` and
  `goldilocks::reduce128` is `goldilocks::reduce_wide`; `reduce*` maps a
  machine integer into the residue range while the inherent
  `Elem::canonical` normalizes an element (those inherent methods are
  unchanged, as is `mersenne31::reduce`).
- **Breaking: `QuadMersenne31::Elem::norm` returns the base-field
  element** `mersenne31::Elem` instead of a raw `u32`; the computation is
  unchanged.
- **Breaking: `FieldKernels::active_backend()` is
  `FieldKernels::backend()`.** The free functions `kernel::backend()` and
  `kernel::backend_for::<F>()` keep their names, and `FGF_TIERS` is
  re-exported at the crate root beside `Backend`.
- The flat GF(2^8) reference internals (`Elem::mul_xtime`,
  `Elem::inv_xtime`, `REDUCTION_LOW`) are `pub(crate)` rather than public,
  and the `$field::field_poly()` accessor is removed — the module const
  `REDUCTION_POLY` remains the one home for that fact.

- The GFNI GF(2^8) matrix kernels resolve each destination row group's
  coefficients into one stack array before their tile loops run, instead
  of loading a coefficient inside the loop. The per-tile coefficient load,
  its bounds check, the term pointer walk and the dependent affine-map
  lookup all leave the multiply loop; grouping, tile widths, and every
  public contract are unchanged, and the steady state stays allocation-free.
  Both GF(2^8) fields, overwrite and accumulate, plan forms and scattered
  rows inherit it. Measured on two GFNI microarchitectures — Core Ultra 7
  258V (Lunar Lake) at 1.07-1.34x and i7-12700K (Golden Cove) at 1.05-1.29x
  the previous throughput across 4-64 KiB rows, 6-16 sources and one to six
  output rows, with no shape regressing on either host; see `BENCHMARKS.md`,
  "Pre-resolved coefficients in the matrix row groups" and "Second GFNI
  host".
  Coefficient counts above 32 terms per call fold in several passes over
  the destination, so a caller that folds hundreds of sources into one call
  sees the destination read once per 32-term pass.
- The GFNI GF(2^8) resolve-chunk scratch is written only for the occupied
  chunk terms (`MaybeUninit` staging) instead of declaring the full arrays
  initialized. Measured kernel-neutral on both GFNI hosts — the full-init
  cost the old resolve probe showed was the probe's own artifact — but the
  probe now mirrors production, and `RESOLVE_CHUNK` stays 32: chunk 64/96
  fail the both-host noninferiority gate at 65 sources. No observable
  behaviour change; see `BENCHMARKS.md`, "Crossed resolution x chunk panel".
- **Breaking: scalar element tuple fields are now private.** Construct and
  read elements through the named `const` conversions — `Elem::from_raw` /
  `Elem::to_raw`, with `from_components` / `to_components` on the towers. Raw
  byte encodings and scalar arithmetic are unchanged; the migration is
  mechanical (`Elem(x)` → `Elem::from_raw(x)`, `e.0` → `e.to_raw()`).
- **Breaking: `Gf2` and the prime-family elements (`Mersenne31`,
  `Goldilocks`, `QuadMersenne31`) now compare, hash, and order by canonical
  field value**, so equivalent raw lanes (a prime lane holding `p` and one
  holding `0`) denote the same element. Binary-tower equality is unchanged:
  those fields still compare representations, which is value equality
  there. `to_raw` still returns the raw lane as stored — it is not a
  canonicalizing accessor; compare `to_raw()` values directly when raw
  representation equality is what you need.
- **Clarified packed prime-field inputs:** canonical input lanes produce
  canonical output lanes. Packed calls do not normalize arbitrary input
  bytes; raw-lane totality remains a scalar-arithmetic contract.
- **Breaking: `FieldKernels` remains a usable generic public bound, but its
  raw unchecked dispatch and preparation entry points are no longer exposed
  to normal consumers**, and `PreparedCoefficient` no longer lets consumers
  extract the implementation's prepared form. The checked operations retain
  their one-shot and borrowed prepared-coefficient usage flows; matrix-plan
  signature changes are listed below.
- `bits` padding is caller-maintained. Range operations leave padding
  untouched; whole-buffer operations preserve zero padding for valid
  inputs, but do not promise to sanitize nonzero padding. This corrects
  the earlier blanket zero-padding guarantee.
- `internals` is documented as explicitly unstable — nothing behind it is a
  compatibility promise — while remaining subject to Rust's safety and
  soundness contracts.
- Documentation corrections: the `no_std` story now states that the crate's
  own portable configuration exists but the `simdispatch`/`archmage`
  dependency graph still enables `std`, so bare-metal support awaits a
  separate dependency review; the fields table gained the missing
  `QuadMersenne31` row; value equality/hash/ordering, canonical prime
  lanes, caller-maintained bit padding, fused `+=` arithmetic (XOR only in
  characteristic two), and a crate-wide no-constant-time guarantee are
  stated explicitly.
- `cargo package` now uses an explicit include allowlist (source, the
  intentional tests/benches/examples, and the public doc/license set);
  local-only working files, markdownlint configuration, CI workflow
  definitions, and `external-bench` are excluded.
- `mul_add_matrix_with` and `mul_into_matrix_with` no longer take the
  row-count argument; it derives from the matrix's `output_count()`.
- Direct architecture entrypoints under `internals` now live in
  `kernel::{architecture}::proven` and require genuine Archmage capability
  tokens. Safe entrypoints validate geometry; generic raw-provider calls
  additionally require an unsafe caller contract. The normal dispatch path
  retains its existing once-per-process capability selection.
- `internals` enables optional `archmage =0.9.29`, matching the version
  already used by `simdispatch`. Default builds retain only `simdispatch`
  as a direct runtime dependency.
- **Breaking:** `bits::weight` returns `usize` rather than `u32`.

### Fixed

- Valid zero-byte scatter and matrix shapes are no-ops on every backend,
  while malformed coefficient/source geometry still panics.
- Prepared-matrix source lookup checks logical bounds before arithmetic,
  including zero-output matrices and `usize::MAX` indices.
- Bit population counts no longer overflow at `2^32` set bits; the public
  operation sums bounded segments without changing the inner kernel.
- Direct overwrite entrypoints replace addressed rows rather than accumulate
  into their previous contents, including zeroing them for an empty source
  set; surplus destination bytes remain untouched.
- GF16 scatter accepts surplus destination rows in debug builds, matching
  the checked buffer contract and release behavior.

### Removed

- The `internals` benchmark bodies for four candidates that two GFNI
  microarchitectures rejected: dense prepared affine-map records
  (`matrix_overwrite_affine_prepared`, `matrix_overwrite{1,2}_8d_prepared`,
  `gather_affine_prepared`, `gather_affine_prepared_tile`), ISA-L's
  three-row grouping (`matrix_overwrite{3,6_33}_8d_prepared`), the 32-byte
  replicated affine-map store (`Map32`, `prepare_map32_8d`,
  `dot_overwrite_replicated_8d`), and the pre-resolved reference body that
  the production path superseded (`dot_overwrite_compact_8d`). Nothing behind
  `internals` is a compatibility promise; the measurements that killed them
  are recorded in `BENCHMARKS.md`. The bodies the recorded follow-ups need —
  `matrix_overwrite1_tile_8d`, `matrix_overwrite1_external_8d` and
  `resolve_probe_8d` — stay.

## [0.7.1] - 2026-09-09

### Changed

- The `simdispatch` dependency now uses the exact crates.io `=0.1.0`
  release for the stack's published backend-selection source.

## [0.7.0] - 2026-08-24

### Added

- Native GF(2). The `Gf2` marker with `gf2::Elem` scalar arithmetic (add is
  XOR, multiply is AND, negation/squaring the identity, `const` throughout,
  total over raw bytes), and the `bits` module: a bit-packed vector surface
  over `&[u8]` buffers holding one element per bit LSB-first — `xor`,
  `xor_range`, `and_into`/`and_assign`/`andnot_assign`,
  `clear_range`/`set_range`, `weight`, `dot_product`, and `xor_gather`, with
  an explicit bit count where the byte length cannot recover it and padding
  bits kept zero on every output. The bit order is a frozen wire convention.
  `bits::xor_assign` reuses the dispatched byte-XOR kernel; the other kernels are
  portable `u64` word loops (intrinsic acceleration is measured-only and
  the bandwidth-bound shapes are expected to stay portable).
- CI now measures line coverage with `cargo-llvm-cov` and fails below 95%,
  merging one test run per forced `SIMD_BACKEND` tier so every dispatchable
  backend counts toward the total.
- Test coverage for surfaces the suites previously drove only through
  inherent methods: the `field::Elem` trait and its defaulted bodies, the
  `core::ops` operator overloads, `Sum`/`Product` folds, `Display`, the
  prime-field `Neg`, prepared-plan `_with` operations for every kernel
  family, `Coeff`/`Plan` accessors, the geometry-validation panic arms of
  every `ops` entry point, and runtime re-evaluation of the `const`
  table builders against the committed banks.

### Changed

- `ops::add_gather` (and `FieldKernels::add_gather_offsets`): the
  unit-coefficient gather — fold byte-offset rows of one backing region
  into one row. Fields whose addition is XOR dispatch to a new blocked
  kernel that holds the destination in AVX2 registers across the whole
  source list; prime fields fold per source through their canonical
  addition. Offsets are `u32` byte offsets, so callers fold an index table
  without staging a slice per source. Measured 1.4–2.4x over the staged
  all-ones `mul_add_gather` call it replaces — see `BENCHMARKS.md`,
  "Blocked XOR gather".

- `Gf8D`'s N-to-1 gather takes the source-fused body on short rows, the
  same selection rule `Gf8B`'s `GF2P8MULB` gather has used since the
  short-row measurement. It was wired to the unfused specialization when
  the blocked-affine seam landed, so rows below the 128-byte main tile fell
  through to one single-source AXPY per source. Results are unchanged;
  `Gf8B` is untouched. Measured on a 64-byte, sixty-four-source consumer
  shape at 1.5–3.5% end to end — see `BENCHMARKS.md`, "`Gf8D` inherits the
  source-fused short-row rule".

- The bit-packed GF(2) folds run split accumulators: `bits::weight` uses
  eight independent `popcount` lanes and `bits::dot_product` four independent
  AND-XOR lanes, folded once at the end, so the loops saturate execution-port
  throughput instead of riding one register's dependency chain. Measured
  1.2–1.5x (`weight`) and 1.2–2.0x (`dot_product`) across L1/DRAM buffer
  sizes on the reference host. Results are unchanged.

- `QuadMersenne31` kernels canonicalize each limb once per load instead of
  re-reducing operands in every helper: `mul_add`, `mul_assign`, `mul_into`,
  and `mul_elementwise` run about 1.2–1.4x faster on 64 KiB buffers.
  Raw-lane totality is unchanged. `Elem::square` uses a dedicated
  `(a²−b²) + 2ab·i` form (3 base multiplies instead of 4).

### Added

- `QuadMersenne31`, the quadratic extension GF((2^31 − 1)²) over `Mersenne31` with `i² = −1` (the QM31 construction). 8-byte interleaved `re,im` little-endian encoding, `const` scalar arithmetic (schoolbook multiply, conjugate/norm inverse), generator `(1,12)` of order `p²−1`, total over raw limbs, and the complete checked ops surface composed from the `Mersenne31` lanes (portable `scalar` today, no new `unsafe`).

## [0.6.0] - 2026-08-23

This release adds `QuadMersenne31`, the degree-2 extension of `Mersenne31` (`i²=−1`) used by circle STARKs. The binary-tower and prime fields are byte-identical to 0.5.0.

### Added

- `QuadMersenne31` (GF((2^31−1)²)). See Unreleased entry above.

## [0.5.0] - 2026-08-23

This release widens the crate from binary tower fields to also cover prime
fields: `Mersenne31` (GF(2^31 − 1)) and `Goldilocks` (GF(2^64 − 2^32 + 1)),
lane-packed integer arithmetic with x86 integer-SIMD kernels. The binary-tower
fields are byte-identical to 0.4.0.

### Added

- `Mersenne31`, the prime field GF(2^31 − 1). Full `const` scalar arithmetic
  over `u32` lanes with a Mersenne fold (`2^31 ≡ 1`), the complete checked ops
  surface, and x86 AVX2 (`v3`) / SSE4.2 (`v2`) integer-SIMD kernels for
  add/sub/multiply/elementwise using `vpmuludq` even/odd lane products. It is
  total over raw lanes and variable-time (not for secret data).
- `Goldilocks`, the prime field GF(2^64 − 2^32 + 1). Full `const` scalar
  arithmetic over `u64` lanes with the split-fold reduction (`2^64 ≡ 2^32 − 1`),
  the complete checked ops surface, and x86 AVX2/SSE4.2 kernels building the
  128-bit product from four `vpmuludq` half-products. Total over raw lanes and
  variable-time.
- `Elem::neg`, the additive inverse, defaulting to the identity in
  characteristic two and overridden by the prime fields.

### Changed

- `FieldKernels` now requires `add_assign`/`sub_assign` so `ops::add_assign`
  and `ops::sub_assign` route through the field: XOR for the binary towers,
  a modular fold for the prime fields. The binary-field bytes are unchanged.
- `Field::BITS` documents the element storage width (`8 * BYTES`); for the
  binary towers this still equals the extension degree over GF(2), while the
  prime fields set the lane width (32/64), not `log2(ORDER)`. No binary value
  changed.
- On non-x86 targets the prime fields use the portable path and report
  `scalar`, consistent with the crate's rule that a cross-compiled kernel is
  never promoted without validation on executing hardware.
- Faster prime-field x86 kernels: `Mersenne31` add/sub/multiply rebuilt on
  the Mersenne fold with a doubled-odd-limb multiply and loop-invariant
  coefficient canonicalization, and `Goldilocks` AVX2 multiplies rebuilt in
  the bias-shifted domain so borrow/carry detection is a single signed
  compare. On the reference AVX2 host the 64 KiB overwrite region runs about
  1.9x faster for `Mersenne31` and 1.3x for `Goldilocks`, accumulate about
  1.3x and 1.2x. The binary-tower paths are untouched.

### Fixed

- `Mersenne31` and `Goldilocks` now override `FieldKernels::mul_into`.
  The overwrite region op previously fell back to the copy-then-scale
  default, two passes over every buffer; it is now a single fused pass on
  every x86 backend, and the kernel differential tests cover it.

## [0.4.0] - 2026-08-22

This release adds `Gf8D`, a second GF(2^8) representation under the
Reed–Solomon polynomial `0x11D` for byte-identical interop, and the overwrite
operation shapes — `mul_into_gather`, `mul_into_matrix`, and
`mul_add_matrix_at` — for erasure encoding and in-place reconstruction.
The two GF(2^8) fields are now named by polynomial (`Gf8B`/`Gf8D`), a breaking
rename of the former `Gf8`.

### Added

- `mul_add_matrix_at` reconstructs disjoint destination rows directly
  in their final positions. GF(2^8) uses the register-blocked x86 GFNI matrix
  kernel; other backends use the portable path.
- `Gf8D`, a second GF(2^8) field under the polynomial `0x11D` (generator `2`),
  for byte-identical Reed–Solomon interop with ISA-L and
  `klauspost/reedsolomon`. It carries the full `const` scalar arithmetic and
  the complete checked ops surface: on x86 GFNI hosts it multiplies through
  `VGF2P8AFFINEQB` affine maps const-derived from its own scalar oracle
  (`GF2P8MULB` is the AES field and is never used), including the
  register-blocked scatter/gather/matrix shapes; elsewhere it runs the
  field-agnostic split-nibble shuffle kernels. Elementwise multiplication is
  vectorized on every x86 backend through a branchless shift/reduce that
  threads the `0x11D` reduction byte (the AES `GF2P8MULB` cannot serve it).
- `mul_into_gather` and `mul_into_gather_with` overwrite one destination with an
  N-to-1 field dot product, distinct from accumulating `mul_add_gather`.
  Empty/all-zero inputs clear the destination, one source uses `mul_into`, and
  prepared plans remain allocation-free.
- `mul_into_matrix` and `mul_into_matrix_with` overwrite multiple rows
  with field dot products for erasure encoding. On x86 GFNI, `Gf8B` and
  `Gf8D` seed blocked accumulators from zero in registers, eliminating the
  separate zero-fill and destination read; other backends use the equivalent
  zero-then-accumulate composition.

### Changed

- **Breaking: `Gf8` renamed to `Gf8B`, module `gf8` to `gf8b`.** The two
  GF(2^8) representations are now named consistently by polynomial
  (`Gf8B`/`0x11B`, `Gf8D`/`0x11D`). Update `use fgf::{Gf8, gf8}` to
  `use fgf::{Gf8B, gf8b}`; the field's bytes, tables, and kernels are unchanged.
- The flat GF(2^8) markers expose their reduction polynomial through
  `field_poly()` and the `REDUCTION_POLY` module constant, for compile-time
  representation introspection.
- Native GFNI N-to-1 gathers now keep 32/64/96-byte multi-source rows fused
  across sources instead of composing single-source AXPY calls. Measured
  16-byte and body-plus-tail variants remain on the prior path.
- `Gf8B` remains on native `GF2P8MULB` for every production gather shape.
  An internals-only prepared-affine gather and benchmark record the rejected
  fixed-map policy without changing public dispatch.
- The production GFNI gather retains its static 128-byte tile. Internals-only
  32/64/96-byte and split-accumulator bodies support page-controlled benchmark
  reruns; measured wins reversed across neighboring layouts.

## [0.3.0] - 2026-08-08

This release renames the crate from `fff` to `fgf` (Faster Galois Fields) and
moves backend detection onto the extracted Level 0 `simdispatch` crate. The
field arithmetic, kernels, and operation surface are unchanged.

### Changed

- **Breaking: crate renamed `fff` → `fgf`.** The repository moves to
  [github.com/nanithefkuc/fgf](https://github.com/nanithefkuc/fgf). Imports
  change from `use fff::…` to `use fgf::…` and the dependency key to `fgf`;
  the features (`std`, `simd`, `internals`) and the API are otherwise
  unchanged. The internals-only tier list `FFF_TIERS` is now `FGF_TIERS`.
- **Backend migration onto `simdispatch`.** `Backend` is now a
  re-export of `simdispatch::Backend`; detection, ordering, the `BACKEND`
  static, and the downgrade-only override are owned by `simdispatch` (Level 0,
  via `archmage` `summon()`). This crate's `detect()`, `resolve_backend()`,
  `BACKEND`, `Backend::ALL`, `is_for_current_arch`, local
  `Display`/`FromStr`/`ParseBackendError`, and the `FFF_BACKEND` override are
  deleted. Selection resolves `simdispatch::Selection` over `FGF_TIERS` once
  per process and is cached. The field-kernel policy methods `has_native_mul`
  and `has_blocked_rows` stay in this crate as the new `KernelBackend`
  extension trait (they are policy, not hardware facts).
- Tiers renamed to the `archmage` ladder: `Gfni` → `V3GfniCrypto` (`v3_gfni_crypto`),
  `Avx2` → `V3` (`v3`), `Ssse3` → `V2` (`v2`), `Pmull` → `NeonAes` (`neon_aes`),
  `Simd128` → `Wasm128` (`wasm128`). The 64-byte AVX-512 tier (`Avx512`) is
  deferred (V4x): its kernels remain under the `internals` feature,
  cross-compile-only, and are not in the ladder until validated on executing
  hardware; AVX-512 hosts resolve to the 32-byte GFNI tier.
- The `SIMD_BACKEND` environment variable replaces `FFF_BACKEND`, applied by
  `simdispatch`. Results are unchanged on hosts without AVX-512 (same kernels
  selected); a `simdispatch` git dependency (pinned to `v0.0.0`) is added —
  re-pin consuming crates in the same sitting.
- Tests: the cfg-based `is_for_current_arch` assertions are deleted; the
  per-backend differential-test gates now resolve `simdispatch` selection
  (`host_supports`) instead of `std::is_*_feature_detected!`. The AVX-512
  kernel differential test is gated behind `internals` (no tier to resolve).

## [0.2.0] - 2026-07-31

This release adds vector kernels for the fields that had none, widens the
shapes the existing kernels cover, and tunes x86 store and alignment
behaviour. Every performance claim, the hardware it was measured on, and the
experiments that were rejected are in [BENCHMARKS.md](BENCHMARKS.md); numbers
are deliberately not repeated here.

### Added

- Vector kernels for GF(2^32) and GF(2^64) on x86 GFNI, replacing the scalar
  path for `mul_add`, `mul_assign`, and `mul_into`.
- Vector kernels for the canonical Fan–Paar fields on x86: `FanPaar16` on AVX2
  and SSSE3, `FanPaar32` and `FanPaar64` on AVX2. `FanPaar8` keeps the portable
  path, which is already table-driven.
- `mul_elementwise` vector kernels for GF(2^8) and GF(2^16) on AVX2 and SSSE3.
  `has_vector_elementwise` now reports `true` for those fields on those
  backends.
- `Backend::Pmull`, the AArch64 NEON plus PMULL combination. It replaces a
  per-call feature probe inside `mul_elementwise`, so the capability is
  detected once per process and `FFF_BACKEND=neon` can turn it off. Accepted by
  `FFF_BACKEND`, `Backend::ALL`, `Display`, and `FromStr` like every other
  backend.
- Multi-row GF(2^8) kernels (`mul_add_scatter`, `mul_add_gather`,
  `mul_add_matrix`) and dedicated GF(2^16) scatter, gather, and matrix kernels
  for WebAssembly `simd128`. These shapes previously fell back to per-row or
  scalar loops.
- Fused GF(2^16) `mul_into` for NEON and WebAssembly `simd128`, which used
  copy-then-scale before.
- `internals` feature, exposing the kernel modules and coefficient preparation
  types for downstream crates that build directly on the kernels. It is not a
  stable semver surface.
- Differential kernel coverage for `mul_into` on every backend that implements
  it; the shape had no direct test before.
- Benchmark sections for the shapes the new tuning policies are set from:
  large destinations, destination alignment, small multi-row shapes,
  preparation crossover, and blocked kernels against repeated AXPY.
  `BENCHMARKS.md` documents the AArch64 and WebAssembly runner recipes.

### Changed

- **Breaking:** `Backend::ALL` is `[Backend; 8]`, up from `[Backend; 7]`.
- **Breaking:** AArch64 hosts with the PMULL extension now report
  `Backend::Pmull` from `backend()` rather than `Backend::Neon`.
- **Breaking:** `has_vector_elementwise::<Gf8>()` and `::<Gf16>()` return
  `true` on AVX2 and SSSE3 hosts, where they previously returned `false`.
- GF(2^16) scalar kernels, including the tail of every vector kernel, multiply
  through the shared nibble tables instead of a per-element Karatsuba multiply.
- x86 `mul_into` uses non-temporal stores once the destination is large enough
  that an ordinary store would evict more than it saves. `mul_add` and
  `mul_assign` read their destination and are unchanged.
- The GF(2^16) multi-row scatter kernels on GFNI and AVX2 peel a misaligned
  destination before the main pass, generalizing the existing GF(2^8) peel.
- The NEON and WebAssembly `simd128` GF(2^16) kernels process two lanes per
  iteration.
- GF(2^16) `mul_elementwise` no longer uses PMULL; the NEON path serves PMULL
  hosts. GF(2^8) `mul_elementwise` keeps it.
- GF(2^16) GFNI `mul_add_gather` and GF(2^8) AVX2 `mul_add_matrix` use their
  register-blocked kernels instead of dispatching to repeated AXPY.
- GF(2^16) 256-bit kernels descend through a 128-bit vector before the scalar
  tail, so short remainders are no longer fully scalar.

### Fixed

- WebAssembly `simd128` test build, which referenced GF(2^8) multi-row kernels
  that did not exist. The comparison dev-dependency is now scoped to non-wasm
  targets so `cargo test --target wasm32-*` builds.
- `--no-default-features --all-targets` builds. The bench targets use
  `ops::Plan` and the comparison dev-dependency, both of which need `std`, so
  they now declare `required-features = ["std"]`; a Fan–Paar test import that
  is only reachable on x86 SIMD builds is cfg-gated to match.

### Removed

- crates.io publishing metadata and docs.rs configuration; the crate is
  distributed through git only.

## [0.1.1] - 2026-07-29

### Added

- Prepared `Plan` consumers for scatter, gather, and matrix operations.
- Stable `pack`, `unpack`, and `pack_to_vec` element/buffer conversions.
- `Backend::ALL`, `Display`, `FromStr`, per-field `backend_for`, and
  `has_vector_elementwise` capability reporting.
- Uniform field element `Display`, assignment, iterator, byte-conversion,
  component, and raw-representation APIs.
- Fan–Paar level modules (`fan_paar::fp8` through `fp64`).
- Release metadata, CI, contributing guidance, benchmarks guide, and MIT
  license.

### Changed

- The public `FieldKernels` trait is sealed and re-exported from the crate root.
- The `Elem` trait remains at `fff::field::Elem` instead of colliding with
  concrete `Elem` types at the crate root.
- Row geometry and panic messages are consistent across vector operations.
- Internal scalar kernels, SIMD table layouts, and raw XOR dispatch are no
  longer public semver surface.
- x86 GF(2^8) kernels retain SIMD processing through 16-byte tails.
- Prepared GF(2^8) and GF(2^16) plans retain register-blocked x86 kernels
  where they outperform repeated prepared AXPY.

## [0.1.0] - 2026-07-29

Initial public release.

[Unreleased]: https://github.com/nanithefkuc/fgf/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/nanithefkuc/fgf/compare/v0.7.1...v1.0.0
[0.7.1]: https://github.com/nanithefkuc/fgf/compare/v0.7.0...v0.7.1
[0.6.0]: https://github.com/nanithefkuc/fgf/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/nanithefkuc/fgf/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/nanithefkuc/fgf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/nanithefkuc/fgf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nanithefkuc/fgf/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nanithefkuc/fgf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nanithefkuc/fgf/releases/tag/v0.1.0
