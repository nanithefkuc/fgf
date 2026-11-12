# fgf repository rules

Work from the crate root. `CONTRIBUTING.md` owns contributor workflow;
`crate.just` owns recipe definitions and tool versions; `just --list` lists them.
Do not duplicate their inventories here.

## Workflow

- Use `just` for routine commands; fix an insufficient recipe rather than
  bypassing it with Cargo. Run `just doctor` before assuming tools exist.
- Never edit the vendored `justfile`; crate-specific changes use `crate.just`.
- Run the relevant narrow checks while developing; `just validate` is required
  before a PR. `just release-accept` adds release-specific checks, not publication
  or native ISA acceptance. `just isa-accept TIER` requires actual execution.
- MSRV is Rust 1.93. Raise it only in a minor release, with a linked changelog entry.
- Dependencies resolve to their published pins, never local umbrella patches.
  `just msrv` refreshes/checks the lock; `just msrv-locked` checks it unchanged.
- Every substantive change has an Issue: problem/specification, constraints,
  alternatives, decisions and rationale, blockers, observable acceptance criteria.
  Linked PRs record resolutions, deviations, tradeoffs and exercised verification.
  Partial work uses `Refs #N`; close only fully resolved Issues. Retired work
  closes with its disposition, never a fabricated completion claim.
- Lucid is local working memory. Distill relevant content into self-contained
  public Issues before review; never publish local memory IDs, paths or transcripts.
- CHANGELOG is a short categorized release index with Issue/PR links, not a
  technical narrative. Breaking entries use `**Breaking:**` and a migration
  pointer. Rustdoc owns contracts; the wiki owns usage/migration guides; README
  stays a short overview. Commit subjects use `fgf: short verb phrase`.

## Stable API and placement

The 4.x module set is closed. Public modules are re-export-only facades over
private implementations; stable exports are explicit, never globbed.

| Module | Placement |
| --- | --- |
| `fgf` | Field-independent type model and process/backend queries |
| `fgf::prime` | Degree-one presentations and named prime fields |
| `fgf::binary` | Shipped characteristic-two presentations and tower specs |
| `fgf::extension` | The user-defined extension kit and named odd extensions |
| `fgf::maps` | Maps between fields |
| `fgf::ops` | Byte-lane operations and their prepared operands |
| `fgf::ops::bits` | Operations with bit-packed operands |
| `fgf::internals` | Unstable kernels, tables, descriptors, tiers and raw helpers |

Apply placement in order: field-independent model/query → root; shipped
presentation → binary for characteristic two, otherwise prime for degree one,
otherwise extension; names an `ExtensionSpec` implementer needs → extension;
map → maps; bulk operation → bits if any operand is bit-packed, otherwise ops.
Prepared state, views and capability traits follow their operation; backend
representations stay private or unstable. Otherwise stop and ask the maintainer.

- **A1–A4:** One canonical stable path per item, including hidden re-exports;
  aliases are distinct items, not alternate export paths. No new stable module.
- **A5:** `api/` is the source-level contract; rustdoc and fixtures own semantics.
  `just api` is read-only and checks all stable feature sets across x86_64,
  AArch64 and Wasm. `just api-update` requires an authorized API change and its
  reviewed diff. Never regenerate merely to turn CI green.
- **A6:** Existing signatures, effective bounds, supported impls and documented
  behavior remain compatible throughout 4.x; approved additions require a minor
  release. A new sealed-trait member is still an API addition.
- **A7–A8:** Stable backend facts are limited to `backend`, `backend_for` and
  the `simdispatch::Backend` re-export (the only external re-export). A field
  strategy query does not identify every operation's serving tier.
- **A9:** All public traits except `ExtensionSpec` are sealed. Callable plumbing
  belongs on private supertraits; negative consumers cover generic bounds and
  concrete types. `F: FieldBuffer` alone suffices for every applicable byte
  operation and coefficient type; never add a private consumer bound.
- **A10:** Inherent const counterparts preserve trait method names, meaning,
  validation and ordering; no second spelling of the same operation.
- `internals = []` changes reachability, never compilation or dependencies.
  Its sole feature gate declares `src/internals.rs`; no compatibility aliases.

### Frozen 4.x guarantees

Changing these requires a major release, not maintainer discretion:

- `FiniteField<const P: u64, const N: u32, R = Residue>` and `Elem<F>`;
  generic defaults, alias expansions, raw/associated type identities, public
  constants, supported presentations/storage, and inherent const-callability.
- Total degree at most 64; order below `2^128`; `u128` order, index and exponent;
  descriptor/coordinate capacities; no byte form for `PackedPadded`.
- `inv(0) == 0`, `x / 0 == 0`, `0^0 == 1`; canonical value equality/hash/order,
  with ordering by presentation. Stable Display contracts remain; Debug is unspecified.
- Fixed-width little-endian lanes and lane sizes; LSB-first bits with
  caller-maintained zero padding; field/basis/tower conventions, generators,
  canonical reference-ladder maps, image/generator/component/index ordering,
  and residue conversion semantics.
- Accepted inputs, alias decoding/canonicalization, byte-preserving zero/unit
  shortcuts, empty shapes, untouched destination regions, validation-before-write,
  allocation guarantees, argument order, suffix and geometry contracts.
- Error classification and identity, conditions, public traits and matchable
  structure; `RangeError`/`ImageError` remain opaque or non-exhaustive.
- Existing feature defaults/implications, item availability, no_std and
  alloc-without-std support, and portable targets. No stricter requirements.
- Existing explicit impls and their applicability/outputs, positive Send/Sync/
  Unpin/unwind-safety guarantees, iterators, operators, Default and Error.
  `Field::Raw` and `ExtensionStorage::Raw` remain Send + Sync.
- `ExtensionSpec` gains no required member. Existing seals remain sealed.
  An incompatible exposed Backend dependency update requires a major release.

Private layout, unstable APIs, routing and thresholds may change only while
these guarantees hold. Panic wording is unspecified unless its contract says otherwise.

### Error classification

E1 is mandatory const validation before any value/operation can bypass it;
merely naming an invalid type need not evaluate it. E2 is a documented
construction panic. E3 is a precondition panic naming operation and operand,
before mutation. E4 is a data-dependent Result/Option. Never reclassify by cost.

| Condition | Class |
| --- | --- |
| Prime characteristic in range; absolute degree `1..=64`; correct order below `2^128` | E1 |
| Irreducible binary polynomial; valid normal/Cantor basis | E1 |
| Valid extension base/characteristic; relative degree ≥2; correct total degree; relation count, coordinates and irreducibility; storage and descriptor capacity | E1 |
| Shipped generator has full multiplicative order | E1 |
| Embedding has distinct degrees with source dividing target; isomorphism has equal degrees; field validation precedes degree validation | E1 |
| A canonical map exists for the pair passed to `Embedding::new`/`Isomorphism::new` | E2 |
| Buffer lengths, row geometry, offsets, overlap and counts | E3 |
| Residue/index conversion range | E4: `RangeError` |
| Supplied image count or transported relation | E4: `ImageError` |
| Partial restriction, prime-subfield readout and collection lookup | E4: `Option` |

## Implementation and verification

- Validate checked buffer spans before dispatch; never unchecked row-span arithmetic.
  Hoist preparation outside steady state; prove allocation guarantees with the
  counting allocator. Owned preparations gate on `alloc`, not `std`, and are not serialized.
- Reuse family macros and the operation naming/argument grammar. Keep scalar
  arithmetic const where promised; add the scalar oracle before an optimized path.
- Dependencies point downward: facades → checked ops → engine → field
  arithmetic; maps depend on the field layer, not the engine. The field layer
  declares private dispatch/proof/prepared contracts; the engine implements
  them. Neither names the other's upper-layer wrappers.
- Backend policy belongs to `simdispatch::Selection`, resolved once over the
  implemented tiers. `SIMD_BACKEND` is startup-only and downgrade-only. No local
  detection, alternate override or per-call policy resolution.
- Kernels are x86_64-only under `src/kernel/x86/`; other targets run portable
  scalar bodies. Entries are safe Archmage functions with exact capability tokens;
  helpers use `#[rite]` inside those boundaries. Cache the token with selection.
- Paths split by field family, ISA, then fan shape when necessary. Entry names
  use operation, ISA, `_with`; private helpers omit redundant ISA markers and
  shared helpers carry their defining ISA suffix. Half-width helpers use `_half`.
  Residue single-row kernels split by `u8`/`u32`/`u64` lane type.
- Tests defend observable results/errors, not wording, copied fields or wiring.
  Reuse deterministic noise/oracles. Direct-kernel differentials straddle lane,
  unroll, tail, row and source-count boundaries. Unsupported direct tests report
  a skip, never `ignored`; green forced-tier tests are not execution evidence.
- `TIERS` is `v4x v3_gfni_crypto v3 v2 scalar`; V4x needs `simd512`.
  GF(2^16) byte lengths are even. Portable target/API checks remain required.

### Unsafe residue ledger

Prefer safe reference intrinsics. Each unsafe item needs its own allow and
adjacent obligation-labelled SINCE–THUS proof. Tokens prove features, not memory,
aliasing, initialization or alignment. No module-level unsafe allowance.

Classes: **R** offset-addressed disjoint rows; **U** initialized prefixes of
uninitialized scratch; **S** raw/aligned/non-temporal stores (alignment and fence
required); **F** foreign calls with checked geometry, live unique context and
retained CPU proof (benchmark contexts do not cross threads).

Paths below are relative to `src/`, except the comparison package.

| Where and items | Class |
| --- | --- |
| `kernel/x86.rs`: `store_avx2`, `store_sse2` | S |
| `kernel/x86/gf16/ssse3.rs`: `mul_add_scatter_ssse3`, `scatter_rows`, `mul_add_matrix_ssse3_with`, `matrix_rows` | R |
| `kernel/x86/gf16/avx2.rs`: `mul_into_impl`; `mul_add_scatter_avx2`, `scatter_rows` | S; R |
| `kernel/x86/gf16/gfni/single.rs`: `mul_into_impl` | S |
| `kernel/x86/gf16/gfni/scatter.rs`: `mul_add_scatter_gfni`, `scatter_group` | R |
| `kernel/x86/gf16/gfni/matrix.rs`: `mul_add_matrix_gfni_with`, `matrix_group` | R |
| `kernel/x86/gf8/ssse3.rs`: `mul_into_ssse3_impl`; `mul_add_scatter_ssse3`, `mul_add_matrix_ssse3_with` | S; R |
| `kernel/x86/gf8/avx2.rs`: `mul_into_impl`; `mul_add_scatter_avx2`, `mul_add_matrix_avx2_with`, `matrix_tiles`, `matrix_vector` | S; R |
| `kernel/x86/gf8/gfni/single.rs`: `mul_into_gfni_impl` | S |
| `kernel/x86/gf8/gfni/scatter.rs`: `mul_add_scatter_impl`, `scatter_rows4`, `scatter_rows2`, `scatter_span` | R |
| `kernel/x86/gf8/gfni/matrix.rs`: `matrix_block`, `matrix_at_block` | R |
| `kernel/x86/gf8/gfni/rows.rs`: `rows_body`, `rows_resolved`, `matrix_tail` | R; U in `rows_resolved` |
| `kernel/x86/gf8/avx512/scatter.rs`: `mul_add_scatter_impl`, `scatter_rows4`, `scatter_rows2`, `scatter_span` | R |
| `kernel/x86/gf8/avx512/matrix.rs`: `matrix_block`, `matrix_rows1`, `rows_tile4`, `rows_lane4`, `rows_tile2`, `rows_lane2`, `rows_tile1`, `rows_lane1`, `rows_resolved`, `rows_body`, `matrix_tail`, `matrix_at_block` | R; U in `rows_resolved` |
| `benches/compare/gf16/src/native.rs`: foreign declarations, `Native::new`, `complete_mul`, `leopard_mul`, `complete_region`, `complete_assign`, `leopard_region`, `Drop` | F |

### Coverage exclusions

`crate.just` owns the exact `COV_IGNORE` expression. The excluded binary-field
subtrees (`x86/gf8`, `gf16`, `gf32`, `gf64`) and complete `kernel/gf8.rs` and
`kernel/tower.rs` dispatch files also contain portable/narrower branches: the
exclusion is broad, not a claim that every line needs GFNI. Native differentials
provide separate execution evidence. AVX-512 QuadMersenne31 and residue bodies
cannot run on the hosted coverage runner and are excluded; AVX2 residue and
mixed-ISA prime dispatch remain included. The V4x weight arm in `kernel/gf1.rs`
is unreachable there but is not excluded. The remaining scope has a 95% gate.
Do not widen exclusions silently; document their justification here.

## Benchmarks

Use the pinned-core benchmark recipes; setup/allocation stays outside timing.
Keep unchanged controls, scalar differentials and same-session interleaved
before/after measurements. No routing threshold changes from reasoning alone.
Thresholds retain their `v3.0.2` public-record citations until new evidence replaces them.

Comparison packages live under `benches/compare/{gf8,gf16,prime}`, remain
unpublished, and stay outside the library's package allowlist. Native build
scripts/dependencies never move into the library. `just comparisons-check`
checks the packages; `just benchmark-check-native` executes GF16 fixture oracles.
Missing native tools/competitors fail loudly, never remove an arm.

Raw captures belong in ignored `bench-records/`, not public docs. Record source,
toolchain, CPU/OS, actual backend, geometry, commands, controls and aggregation.
Public records use processor-architecture labels and sanitized environment
facts, never SSH aliases, machine hostnames or local account paths. Do not
reuse stale captures after source/harness changes or invent unavailable results.
