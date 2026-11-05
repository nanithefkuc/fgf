# Repository Guidelines

This file is the operational manual for changing `fgf`. Public behavior belongs
in rustdoc and `README.md`; measurements belong in git-ignored
`bench-records/` captures; user-visible changes belong in `CHANGELOG.md`.

## Required workflow

Work from the crate root and use `just` for routine commands.

```sh
just test [ARGS]           # host's selected backend
just test-tiers            # every supported backend tier
just features              # no-default, default, all-features
just features-alloc        # alloc without std
just features-internals    # internals exposure closure across SIMD feature sets
just cross-check           # AArch64 and Wasm library compilation
just compile-fail          # negative consumer snippets must fail to compile
just msrv-accept           # compile-fail plus the consumer matrix on the MSRV
just msrv-locked           # locked published dependencies, outside umbrella patches
just package-check         # isolated archive build and portable example execution
just external-check        # type-check every package under external/
just lint                  # rustfmt and clippy at both feature ends
just doc                   # rustdoc with warnings denied
just unsafe-check-gfni     # owned-unsafe GFNI Miri cases
just cover                 # merged per-tier coverage, 95% minimum
just validate              # shared correctness and coverage gate
just release-accept        # crate-specific release checks, including native fixtures
just test-gather-validation-32 # i686 release regression, overflow checks disabled
just isa-accept TIER       # fail-closed native execution evidence and tier sweep
just benchmark-check       # portable fixtures and compiler-derived API coverage
just benchmark-check-native # native GF16 fixtures on capable hosts
just api                   # stable API inventory matches api/, one-path report
just api-update            # rewrite api/; only inside an authorized API change
just api-semver [BASELINE] # semver check of the stable feature sets
just example NAME [ARGS]   # one educational example, with Cargo options
just examples              # every educational example in each feature configuration
```

`just validate` is the shared gate, not every crate-specific CI or release
check. `just release-accept` includes the separate GFNI Miri, feature, MSRV,
cross-target, packaged-archive, example, benchmark-fixture, and 32-bit checks.
Publication dry-run and native tier acceptance remain explicit final steps.
`test-gather-validation-32` needs a 32-bit linker (`gcc-multilib` on Linux).
Native GF16 fixture execution requires the documented AVX2/GFNI/crypto host.
`benchmark-check` uses nightly rustdoc JSON to derive the stable callable set
independently of the benchmark's advertised inventory.

`just example backend_report --all-features` prints the requested, resolved,
and representative field backends. Degree-64 relations directly over GF(2)
compile and execute through `tests/compile_pass/binary_degree64.rs`, which the
consumer matrix includes.

`just api` regenerates the stable inventory for four feature configurations
— portable, alloc-only, default, and every feature except `internals` — on
`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, and
`wasm32-unknown-unknown`, requires the three targets to agree within each
configuration, and diffs the result against `api/inventory/` and
`api/metadata.txt`. No inventory line is filtered. An `--all-features`
rustdoc document is also built per target: `api/surface.py
stable-paths-equal` fails when its stable export path set differs from the
no-internals document's. That check covers export paths only; methods,
impls, or signatures added under `cfg(feature = "internals")` are excluded
by rule F4 and review, not by this check.
Each cell's rustdoc JSON is copied after its run and deleted before the next, so
a skipped rustdoc pass fails instead of reusing a stale document. Each
inventory closes with a `# external re-exports` section naming, per stable
re-export path, the re-exported crate and canonical path, and
`api/metadata.txt` records one `dependency <name> <version>` line per
re-exported crate, so a dependency bump changes the inventory. It runs the
one-path check from `api/surface.py` on the same rustdoc JSON, in the mode
`API_PATHS_MODE` names in `crate.just` (`report` until every stable item has
one path, then `enforce`); in report mode the full duplicate list of the
all/x86_64 cell is printed to `target/api/generated/paths.txt` and to
stdout. It needs the pinned `API_NIGHTLY` with the aarch64 and wasm32
targets, and exactly the `cargo-public-api` and `cargo-semver-checks`
versions pinned in `crate.just` (`cargo binstall
cargo-public-api@<version> cargo-semver-checks@<version>`); it checks those
versions before generating anything. `just api` never writes `api/`; CI
proves the committed inventory matches the source, and the maintainer's
review authorizes an inventory change. `just api-update` is run only inside
an explicitly authorized API change, and the regenerated diff is part of
that change.

`just api-semver` runs `cargo-semver-checks` for the portable, alloc-only,
default, and maximal stable feature sets with `internals` never enabled,
against the highest published non-yanked, non-prerelease `API_MAJOR`.x
release, sorted numerically whatever order crates.io returns; `BASELINE`
overrides only that selection for development checks. `release-accept` calls
`api-semver-release`, which decides by the version's major number: a major
below `API_MAJOR` is skipped, `API_MAJOR.0.0` and its prereleases are
skipped because the first release of a major has no baseline of its own,
every other `API_MAJOR` version including prereleases runs the check, and a
major above `API_MAJOR` fails with a message telling the maintainer to raise
`API_MAJOR`.

Run `just validate` before submitting a change. Do not replace a recipe with a
bare Cargo command; fix the recipe when its supported behavior is insufficient.
The MSRV is Rust 1.93. It rises only in a minor release, with a `CHANGELOG.md`
entry, never in a patch release.

CI's locked MSRV check uses published dependencies. `just msrv-locked`
checks the committed lock; regenerate a publication lock with `just msrv`,
not by changing pins.

`justfile` is vendored unchanged from a shared template. Do not edit it here.
Crate-specific values and recipes belong in `crate.just`.

`just examples` runs every example under the default features and under
`alloc` without `std`, then the allocation-free subset (scalar arithmetic,
extension arithmetic, embedding, binary-syndrome, and packed examples) with
every library feature disabled.

## Stable public surface

Every stable item has exactly one public path, and the module set is closed
for the whole major version. The table is the 4.x contract. Where the source
tree disagrees with it, the source is wrong: move it toward the table in a
change of its own, and never extend the disagreement.

| Module | Holds | Items |
|---|---|---|
| `fgf` | Field-independent type model and process query | `FiniteField`, `FiniteFieldRepr`, `Elem`, `Field`, `FieldElem`, `FieldBuffer`, `HasGenerator`, `RangeError`, `Backend` (re-export of `simdispatch::Backend`), `backend`, `backend_for` |
| `fgf::prime` | Shipped degree-one fields | `Residue`, `Mersenne31`, `MersenneFold`, `Goldilocks`, `GoldilocksFold`, `PrimeField`, `PrimeIdentity`, `PrimeCharacteristic`, `is_prime` |
| `fgf::binary` | Shipped characteristic-two presentations | `Polynomial`, `AES`, `RS`, `Normal`, `Cantor`, `BinaryRepr`, `BinaryField`, `BinaryDegree`, `Rijndael16`, `Rijndael32`, `Rijndael64`, `FanPaar2`, `FanPaar4`, `FanPaar8`, `FanPaar16`, `FanPaar32`, `FanPaar64` |
| `fgf::extension` | The kit for user-defined fields and the shipped odd-characteristic extensions | `Extension`, `ExtensionSpec`, `ExtensionField`, `ExtensionStorage`, `Relation`, `Components`, `Packed`, `PackedPadded`, `PackedWord`, `QuadMersenne31`, `ComplexMersenne31` |
| `fgf::maps` | Maps between fields | `Embedding`, `Isomorphism`, `Postcompose`, `ImageError` |
| `fgf::ops` | Bulk operations over byte lanes and their prepared operands | the row functions; `Coeff`, `CoeffVec`, `CoeffMatrix`, `CoeffRef`, `CoeffVecRef`, `PreparedCoefficient` |
| `fgf::ops::bits` | Bulk operations over bit-packed GF(2) | `bytes_for`, `xor_assign`, `xor_range`, `XorRange`, `xor_range_with`, `xor_gather`, `and_into`, `and_assign`, `andnot_assign`, `clear_range`, `set_range`, `weight`, `dot_product` |
| `fgf::internals` | Feature-gated and visibility-only, with no compatibility promise | kernels, tiers, thresholds, tables, descriptors, raw-lane helpers |

`FieldBuffer` is the sole stable bound for byte-lane operations: for arbitrary
downstream generic code, `F: FieldBuffer` suffices for every applicable `ops`
function and coefficient type, with no additional private bound. That
includes generic `R: BinaryRepr<8>`, custom `ExtensionSpec` component storage,
and every supported packed storage. `Field::Raw` is `Send + Sync`.

Internally, the public modules are facades over private implementation
layers, and production dependencies point strictly downward:

| Layer | Contents | Depends on |
|---|---|---|
| Facades | `lib.rs`, `prime`, `binary`, `extension`, `maps`, `ops`, `ops::bits`, and `internals` behind its feature | the layers below, by re-export only |
| `ops`, `ops::bits` implementations | validation and composition | field implementation, engine |
| `maps` implementation | scalar embeddings and isomorphisms | field implementation |
| Engine | detection proof, the one routing table holding every threshold, kernels per primitive and ISA, prepared forms | field implementation |
| Field implementation | type model, presentations, scalar arithmetic, descriptors, const validation | nothing in the crate |

The field implementation declares the private dispatch contract, its proof
types, and its associated prepared-type contract; the engine implements it.
`FieldBuffer` implies that contract, and the contract never extends
`FieldBuffer` back. Field-implementation declarations name no engine-owned
types, and engine implementations name no upper-layer coefficient wrappers.
Documentation links and tests may reference public facades. Sibling modules
inside one layer may depend on one another where their parent's module doc
permits it. A shared helper such as `bytes_for` has one lower-layer owner and
is re-exported, never duplicated. Backend policy stays with
`simdispatch::Selection`.

### Public API rules

- **A1. One path per item.** Each stable exported item has exactly one
  canonical module-export path. Count explicit module membership and
  re-export paths, including documentation-hidden public exports. Exclude
  `internals`, equivalent qualification through type aliases, and inherited
  trait or inherent-item lookup spellings. Named type aliases are distinct
  declared items, not duplicate exports. Stable re-exports are explicit,
  never globbed.
- **A2. Facades.** Public module files contain documentation, attributes,
  module declarations, conditional allocator linkage where required, and
  explicit re-exports, and no concrete operation bodies. Implementation
  modules are private. `src/internals.rs` keeps the visibility-only facade
  pattern.
- **A3. Placement.** Apply the first rule that matches:
  1. Field-independent type model or process query: the crate root.
  2. Shipped presentation or named field: characteristic two goes to
     `binary`; degree one goes to `prime`; anything else goes to `extension`.
  3. A type or trait a user-written `ExtensionSpec` must name: `extension`.
  4. A map between fields: `maps`.
  5. A bulk operation: `ops::bits` when any operand is bit-packed, otherwise
     `ops`.
  6. Stable prepared operation state, borrowed views, and their public
     capability traits follow their operation's module. Their backend
     representations stay private or unstable.
  7. Kernels, tiers, thresholds, tables, descriptors, and raw-lane helpers:
     `internals`, never the stable surface.

  If nothing matches, stop and ask the maintainer.
- **A4. Closed module set.** The modules in the table above are the stable
  modules for the whole major version.
- **A5. API inventory.** The committed API inventory records the source-level
  surface; rustdoc contracts and behavioral fixtures record semantics.
  Inventory generation records its toolchain and tools and covers the
  portable, alloc-without-std, default, and all-feature configurations. The
  stable surface is identical on every supported target, and the check
  compares the x86_64, aarch64, and wasm32 inventories. CI rejects unapproved
  changes. The one-path check traverses every stable export edge without
  collapsing alternate paths. Regenerate the inventory only as part of an
  explicitly authorized API change.
- **A6. Compatibility.** Within 4.x, existing stable items keep their
  signatures, effective public bounds, supported trait implementations, and
  documented behavior. Approved additions land in minor releases; no approval
  waives a breaking change or the closed module set. Private implementation
  requirements may change without altering those guarantees. A new item on a
  sealed trait is reviewed as an API addition, not presumed compatible because
  the trait is sealed. Publication compares against the latest published 4.x
  release.
- **A7. Backend facts.** The stable API exposes no backend-policy facts beyond
  `backend()`, `backend_for()`, and the `simdispatch::Backend` re-export. That
  re-export also exposes the upstream type's inherent items and trait
  implementations. It does not promise that every upstream backend is
  implemented here or that the reported field strategy identifies every
  operation's serving tier.
- **A8. External re-exports.** Only types appearing in public signatures are
  re-exported: `Backend` alone.
- **A9. Sealing.** Every public trait is sealed except `ExtensionSpec`, the one
  trait users implement. Items only the crate calls live on private
  supertraits. Moving plumbing behind a private supertrait removes its
  supported public access, not merely its documentation path; negative
  consumer cases cover generic public bounds as well as concrete types.
  Stable signatures never require `internals`.
- **A10. Inherent `const` counterparts.** A concrete `Elem` family may provide
  inherent `const fn` counterparts of `FieldElem` methods and
  `ExtensionField::{from_components, to_components}`. Each counterpart keeps
  the operation's name, meaning, validation, and component ordering. A trait
  function taking an element may use the equivalent inherent `self` receiver.
  These counterparts are not aliases; alternative names for the same trait
  operation are prohibited.

### Error rules

- **E1.** Enumerated, inexpensive type-constant validity checks are mandatory
  const assertions, evaluated before a value or operation can bypass
  validation. Merely naming an invalid type is not promised to trigger
  evaluation.
- **E2.** Enumerated type-dependent checks designated for runtime construction
  panic on failure and document that condition under `# Panics`. Checks are
  never reclassified by estimating their cost. Existing promised const
  validation stays const.
- **E3.** Violations of a consuming operation's runtime preconditions panic,
  naming the operation and the offending operand. This does not override APIs
  defined as total queries or data validators.
- **E4.** Data-dependent outcomes return `Result` or `Option`. Each error type
  represents one documented validation condition; unrelated conditions are
  never collected into a catch-all enum. `ImageError` covers rejection of a
  supplied generator-image sequence, including a wrong count or a failed
  transported relation.

The classification below is part of the API: changing a row is an API change
under A6.

| Check | Class |
|---|---|
| Characteristic is prime and in range | E1 |
| Absolute degree satisfies `1 ≤ N ≤ 64`; field order equals the characteristic raised to that degree and is below `2^128` | E1 |
| Binary presentation validity: irreducible polynomial, valid normal or Cantor basis | E1 |
| Extension base validity; base and row characteristic agreement; relative degree `D ≥ 2`; absolute degree equals base degree times `D`; defining relation has the required coefficient count and in-range coordinates and is irreducible; storage admissibility; descriptor capacity | E1 |
| Shipped `GENERATOR` has full multiplicative order | E1 |
| `Embedding`: source degree divides target degree and differs from it | E1 |
| `Isomorphism`: source and target degrees are equal | E1 |
| A canonical map exists for the pair (`Embedding::new`, `Isomorphism::new`) | E2 |
| Row geometry, lengths, offsets, overlap, coefficient counts | E3 |
| Residue or index range | E4, `RangeError` |
| Supplied generator images: count and transported relations | E4, `ImageError` |
| Partial restriction, prime-subfield readout, collection lookups | E4, `Option` |

`Embedding::new` and `Isomorphism::new` return `Self` and document their
construction panics.

### Frozen for 4.x

These guarantees hold for every 4.x release; changing one needs 5.0.

1. Every existing stable module path, trait, item, required trait member, and
   inherent const-callability guarantee stays available. Minor releases may
   add approved items within the closed module set. `ExtensionSpec` gains no
   required members, and provided additions undergo compatibility review.
   Existing trait seals stay sealed.
2. The type model: `FiniteField<const P: u64, const N: u32, R = Residue>` and
   `Elem<F>` as the one element type. Public generic defaults, named alias
   expansions, associated-type identities and equalities (including
   `Field::Raw` and the raw-word types exposed by scalar and storage APIs),
   public constant values other than explicitly variable backend facts, and
   the supported presentation and storage combinations.
3. Capacity: total degree at most 64, field order below `2^128`, `u128`
   `ORDER`, indices, and `pow` exponents; descriptor-node and coordinate
   capacities; `PackedPadded` has no byte form. Accepted geometry or
   presentations never become rejected in a minor release.
4. Arithmetic: `inv(0) == 0`, `x / 0 == 0`, `0^0 == 1`; `Eq`, `Hash`, and `Ord`
   by canonical storage, with `Ord` as presentation order.
5. Encodings and conventions: fixed-width little-endian lanes and each field's
   `BYTES`; the bit-packed LSB-first layout with caller-maintained zero
   padding; presentation data (AES `0x11B`, RS `0x11D`, the tower, normal, and
   Cantor bases); `GENERATOR` values; canonical embedding images from the
   reference ladder; explicit map-image and generator ordering and map
   algebra; `from_index` coordinate order; `from_residue` semantics.
6. The accepted input domain and documented output behavior: alias decoding
   and canonicalization, byte-preserving zero and unit shortcuts, empty and
   zero-work shapes, source and coefficient ordering, untouched destination
   regions, bit-padding behavior, validation-before-write guarantees, and
   allocation guarantees.
7. Every stable function signature: argument order, effective bounds, suffix
   semantics, and geometry contracts.
8. The E1–E4 classification of every API and check.
9. Error-type identity, documented conditions, implemented public traits, and
   externally matchable or constructible structure. `RangeError` and
   `ImageError` are opaque or `#[non_exhaustive]`; that never authorizes
   changing an existing error's meaning.
10. Feature defaults and implication edges, stable item availability under
    every supported feature combination, `no_std` and alloc-without-std
    support, and supported-target portability. New optional features may be
    added; existing items never acquire stricter requirements.
11. A dependency update that changes the exposed `Backend` type incompatibly,
    including a semver-breaking pre-1.0 `simdispatch` release, requires an
    fgf major release. The backend-query semantics and the downgrade-only
    request contract are preserved; exact routing results, implemented-tier
    sets, and thresholds are not.
12. Existing explicit trait implementations, their generic applicability,
    associated outputs, and positive auto-trait guarantees of public types and
    opaque return values: operators, `Default`, iterator contracts, `Error`,
    and the applicable `Send`, `Sync`, `Unpin`, and unwind-safety guarantees.
    Stable `Display` contracts are preserved; `Debug` output is unspecified.

Private implementation structure, `internals`, routing, thresholds,
performance, and unspecified prepared-value layout may change only while
every stable source, semantic, feature, const-callability, allocation, and
auto-trait guarantee above holds. Panic text and `Debug` output are
unspecified unless an individual contract says otherwise. "Documented
behavior" means the rustdoc contract text.

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

Architecture kernels stay under `src/kernel/x86/`; other targets compile
and run the portable scalar bodies. A kernel must take the exact Archmage
capability token required by its instructions. Dispatch resolves and caches
that token once. Helpers use `#[archmage::rite]` only within a token-proven
entry.

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
Residue single-row kernels split by arithmetic lane type into `u8.rs`,
`u32.rs`, and `u64.rs`; reduction parameters and source-provider windows
shared by AVX2 and AVX-512 remain in `kernel/x86/residue.rs`.
A helper private to one file carries no ISA or width marker; a helper shared
across sibling files carries its defining file's ISA
suffix (`bmul_gfni`, `store_avx2`, `fold_avx2`), and a half-width variant
inside a wider tier is `_half`. The
AVX-512 kernels dispatch on `V4x` under `simd512`; `FiniteField<2, 16, Extension<Rijndael16, 2>>` scatter and gather
retain the narrower GFNI path. Direct differentials execute on AVX-512
hardware, and the per-shape measurements behind the routing floors are the
3.0.2 record's gf8, gf16, and residue pages
(<https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf8.md>,
<https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf16.md>, and
<https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>).
The byte-field and `FiniteField<2, 16, Extension<Rijndael16, 2>>` `mul_add`
alignment peel floors remain measured thresholds, justified by that record,
with misaligned-row coverage in kernel tests.

## Residue ledger

Unsafe that survives the safe conversion ladder is limited to the cases below.
The CPU token proves instruction availability; it does not prove bounds,
initialization, alignment, or aliasing. Each listed item has a per-item
`#[allow(unsafe_code)]` and a SINCE–THUS proof at its site.

| Where | Why unsafe remains |
|---|---|
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
- `tests/binary_bases.rs`: ordered-basis and custom extension-spec arithmetic against
  base-field schoolbook oracles.
- `tests/prime_fields.rs`: primality, prime-field bulk operations on raw
  lanes, and independent modular oracles.
- `tests/extension.rs`: generic extension rows; `rows` covers each base
  shape, packed byte forms and the checked packed surface; `reference`
  checks arithmetic against an independent in-test schoolbook and
  brute-force reference, exhaustively on small fields and sampled at full
  width.
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

Coverage excludes the complete x86 binary-field kernel subtrees
(`gf8`, `gf16`, `gf32`, `gf64`) and the complete `kernel/gf8.rs` and
`kernel/tower.rs` dispatch modules. These file-level exclusions also omit
SSSE3/AVX2 kernels and portable dispatch branches; they are not a claim that
every excluded line requires GFNI. The 95% gate applies to the remaining scope.
`kernel/x86/quad_mersenne31/avx512.rs` and
`kernel/x86/residue/avx512.rs` with its child subtree are also excluded because
their AVX-512 instructions cannot execute on the hosted coverage runner.
The V4x weight arm in `kernel/gf1.rs` cannot execute on the coverage runner
either (documented, not excluded). The AVX2 residue kernels and the mixed-ISA
prime dispatch modules remain included. Direct differentials, with positive
execution markers and resolved-backend checks, provide separate acceptance
evidence on capable hosts.
Keep `COV_IGNORE` narrow and document every exclusion here.

## Benchmarks

Use only the benchmark recipes, pinned to an identified core:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-comp gf          # binary tower fields
FEC_GOLDEN_CORE=<cpu> just bench-comp gdl         # Goldilocks
FEC_GOLDEN_CORE=<cpu> just bench-comp m31         # Mersenne31 and QuadMersenne31
FEC_GOLDEN_CORE=<cpu> just bench-comp gf16 [ARGS] # native GF16 harness alone
```

`bench-comp` runs five complete rounds interleaving the family's self suite
with every wired competitor harness, shuffling the unit order within each
round and printing the realized order. Bit-packed GF(2) has no competitor
harness. The granular self entry points stay available:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels [--gf|--gdl|--m31|--gf2|--prime]
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
kernels per row length; it measures the dispatch thresholds in
`src/kernel/quad_mersenne31.rs`.

`surfaces` covers the stable scalar, representation, packing, bit-buffer,
prepared-coefficient, row-operation, capability-query, and map callables.
Its inventory is checked against compiler-derived rustdoc JSON, not a second
hand-maintained list of the same cases. `--check` exercises every fixture
without timing; `--smoke` shortens sampling without removing cases.
Metadata/query latencies include a timer-floor control. Representative
presentations distinguish arithmetic, storage, and special routing classes;
an infinite set of prime or relation parameters is not a finite case inventory.


Every external competitor harness lives in `external/`, one separate
unpublished package per harness with its own `build.rs`, outside `src/` and
`benches/` because linking the competitor needs a build script and `fgf`
itself must never carry one; the package allowlist keeps `external/` out of
`cargo package`. `external/bench-trio/` links the system Intel ISA-L and
klauspost/reedsolomon as a Go C archive and interleaves all three arms —
`fgf`, ISA-L, klauspost — over one GF(2^8)/`0x11D` fixture set, rotating
through all six arm orders; the klauspost coding matrix arrives through
`WithCustomMatrix`. It runs through `bench-comp gf` inside its shuffled
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
through `bench-comp gdl` and `bench-comp m31` inside their shuffled rounds;
a single rerun is `RUSTFLAGS="-C target-cpu=native" cargo run --release`
from its directory with a `gdl` or `m31` family argument. The package
builds with `-C target-cpu=native` because Plonky3 selects its packed
kernels at compile time. The banner reports packing widths; the captured
log states the flag and widths alongside the resolved `fgf` backends.

`external/gf16-bench/` compares native GF16 field operations against pinned
GF-Complete, reed-solomon-erasure, reed-solomon-simd and Leopard-RS 1.x
(pre-Leopard2). Leopard2 exposes no public field-region primitive, only codec
operations, so the harness pins the 1.x field kernels.
Its representation maps are bijective and preserve every basis product;
independent scalar arithmetic validates every native scalar and timed region.
Single-row cases cover fixed and cycling coefficients. Multi-row competitors
compose native region operations; unsupported primitives remain absent.
The duplicate fgf arm in every case measures bias. Arm orders follow a
Williams balanced Latin square: within one cycle every arm takes every
position and directly follows every other arm equally often. Each arm warms
immediately before its timed batch, and sampling stops only at a cycle
boundary.
`bench-comp gf16` runs the targeted five-round campaign; `bench-comp gf`
includes the same harness in its shuffled binary-field rounds.
`benchmark-check-native` validates every geometry without timing. The native
build requires Git, C/C++ compilers and an x86-64 AVX2/GFNI/crypto host;
missing inputs fail rather than remove an arm.

Benchmark setup, allocation, coefficient construction, and input generation
must stay outside the timed region. Record the CPU, OS, Rust version, selected
backend, geometry, command, and aggregation rule in the captured output. Never
reuse an old number or claim a performance change from an unpinned run.

Measured thresholds cite the 3.0.2 record by its `v3.0.2` tag URL; new
measurements are kept in git-ignored `bench-records/` until a public record
exists.

A performance change requires an unchanged callable control, differential
correctness, and interleaved before/after measurements in one session. Do not
change dispatch or a crossover from reasoning alone.

## Documentation and review

- Rustdoc owns item contracts, invariants, panics, safety, layout, and ownership.
- `README.md` is the short public overview: scope, a minimal example,
  features, platforms, safety, building, and license. Guides and background
  belong in the wiki, not the README.
- Raw benchmark captures live in git-ignored `bench-records/`.
- `CHANGELOG.md` owns user-visible changes and migration notes.
- Public files and source comments must not reference planning artifacts.
- Use third person and present tense. Avoid history, temporal status, marketing,
  and benchmark numbers outside a measurement capture.
- Every public item and module needs a one-line summary; `just doc` treats
  warnings as errors.

Review every exported-symbol change across all callers. Use a clean cutover:
migrate every caller and remove obsolete aliases, wrappers, comments, and
re-exports.

Commit subjects use `fgf: short verb phrase` and stay near ten words. Put what
changed and why in the pull request and `CHANGELOG.md`, not in a commit-message
essay.
