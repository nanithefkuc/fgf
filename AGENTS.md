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
just doc-stable            # default-feature rustdoc, the docs.rs configuration
just unsafe-check-gfni     # owned-unsafe GFNI Miri cases
just cover                 # merged per-tier coverage, 95% minimum
just cover-native          # full native coverage and excluded-ISA group figures
just validate              # shared correctness and coverage gate
just release-accept        # crate-specific release checks, including native fixtures
just test-gather-validation-32 # i686 release regression, overflow checks disabled
just isa-accept TIER [REGISTRY] # fail-closed native execution evidence and tier sweep
just kernel-registry-check # compiler-derived x86 entries and behavioral checker tests
just benchmark-check       # portable fixtures and compiler-derived API coverage
just benchmark-check-native # native GF16 fixtures on capable hosts
just bench-baseline *ARGS  # standalone descriptive breadth or bulk measurement
just bench-pair *ARGS      # paired acceptance, calibration, or threshold campaign
just bench-coverage-map TIER # instrumented coverage map of the operations bench
just bench-publish *ARGS   # render or verify the committed measurement record
just api                   # stable API inventory matches api/, one-path gate
just api-update            # rewrite api/; only inside an authorized API change
just api-semver [BASELINE] # semver check of the stable feature sets
just example NAME [ARGS]   # one educational example, with Cargo options
just examples              # every educational example in each feature configuration
```

`just validate` is the shared gate, not every crate-specific CI or release
check. `just release-accept` includes the separate GFNI Miri, feature, MSRV,
cross-target, stable-documentation, packaged-archive, example,
benchmark-fixture, and 32-bit checks.
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
one-path check from `api/surface.py` on the same rustdoc JSON, in the
`enforce` mode `API_PATHS_MODE` names in `crate.just`. The recipe writes the
all/x86_64 cell's full report to `target/api/generated/paths.txt` and prints
its last line to stdout. It
needs the pinned `API_NIGHTLY` with the aarch64 and wasm32
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

| Check | Class | Evidence |
|---|---|---|
| Characteristic is prime and in range | E1 | `tests/compile_fail/constructor_p0.rs`, `constructor_mr_composite.rs`, `identity_p4.rs` |
| Absolute degree satisfies `1 ≤ N ≤ 64`; field order equals the characteristic raised to that degree and is below `2^128` | E1 | `tests/compile_fail/extension_degree_zero.rs`, `extension_degree_above_cap.rs`, `binary_degree_above_cap.rs`, `extension_cardinality_overflow.rs` |
| Binary presentation validity: irreducible polynomial, valid normal or Cantor basis | E1 | `tests/compile_fail/binary_reducible_polynomial.rs`, `binary_invalid_normal_basis.rs`, `binary_invalid_cantor_basis.rs` |
| Extension base validity; base and row characteristic agreement; relative degree `D ≥ 2`; absolute degree equals base degree times `D`; defining relation has the required coefficient count and in-range coordinates and is irreducible; storage admissibility; descriptor capacity | E1 | `tests/compile_fail/extension_reducible_base.rs`, `extension_relative_degree_one.rs`, `extension_marker_degree_mismatch.rs`, `extension_relation_capacity.rs` |
| Shipped `GENERATOR` has full multiplicative order | E1 | `tests/consumers.rs::shipped_instance_const_surface_matches_the_trait_surface` (const generator use), `tests/algebra.rs::gf8_generator_has_full_order` |
| `Embedding`: source degree divides target degree and differs from it | E1 | `tests/compile_fail/embedding_equal_new.rs`, `embedding_equal_images.rs`, `embedding_reverse_new.rs`, `embedding_reverse_images.rs`, `embedding_unsupported_nondividing.rs`, `map_field_before_degree.rs` (field validity precedes the degree assertions) |
| `Isomorphism`: source and target degrees are equal | E1 | `tests/compile_fail/isomorphism_unequal_new.rs`, `isomorphism_unequal_images.rs` |
| A canonical map exists for the pair (`Embedding::new`, `Isomorphism::new`) | E2 | `tests/relations.rs::embedding_new_panics_on_an_unrelated_odd_pair`, `isomorphism_new_panics_on_an_unrelated_odd_pair`; `tests/embedding.rs::uncovered_presentations_reject_canonical_construction` |
| Row geometry, lengths, offsets, overlap, coefficient counts | E3 | `tests/ops.rs::mul_into_scatter_rejects_wrong_row_count`, `matrix_scattered_rejects_overlapping_rows`, `tests/bits.rs` geometry cases, `tests/embedding.rs::generator_rejects_a_level_at_the_count` |
| Residue or index range | E4, `RangeError` | `tests/residue.rs::residue_conversions_reject_out_of_range`, `range_error_is_a_std_error`; `tests/residue_index.rs::residue_rows_round_trip_indices_and_residues`, `mersenne31_rejects_the_shifted_alias`; `tests/named_rows.rs::residue_conversions_reject_out_of_range` |
| Supplied generator images: count and transported relations | E4, `ImageError` | `tests/relations.rs::image_sequences_reject_or_rebuild_across_both_constructors`, `gf9_conjugate_images_reach_gf81` |
| Partial restriction, prime-subfield readout, collection lookups | E4, `Option` | `tests/embedding.rs::restrict_rejects_elements_outside_the_subfield`; `tests/residue_index.rs::extension_rows_round_trip_every_index`, `normal_seven_two_marks_the_index_residue_split` (prime-subfield readout split) |

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
- `tests/embedding.rs`: frozen reference-ladder fixtures, mutual inverses,
  commuting triangles, trace/norm, canonical-map panics on uncovered
  pairs, and failed restriction.
- `tests/relations.rs`: characteristic-independent construction — the
  enumeration reference over image tuples, the image-error table across
  both constructors, canonical-map panics on unrelated odd pairs, and
  conjugate references.
- `tests/binary_bases.rs`: ordered-basis and custom extension-spec arithmetic against
  base-field schoolbook oracles.
- `tests/prime_fields.rs`: primality, prime-field bulk operations on raw
  lanes, and independent modular oracles.
- `tests/extension.rs`: generic extension rows; `rows` covers each base
  shape, packed byte forms and the checked packed surface; `reference`
  checks arithmetic against an independent in-test schoolbook and
  brute-force reference, exhaustively on small fields and sampled at full
  width.
- `src/kernel/tests.rs`: shared differential drivers; the children split direct
  comparisons by field family and ISA across lane, tail, row, and source-count
  boundaries, including the quadratic Mersenne kernels.
- `src/kernel/tests/fixtures.rs`: shared raw geometry, coefficient words, and
  frozen deterministic noise streams used by unit and integration tests.
- `src/kernel/tests/tier_gate.rs`: request-bounded tokens and post-comparison
  execution receipts shared by direct suites and allocation checks.
- `tests/zero_alloc.rs`: allocation-free steady-state contracts.
- `tests/internals.rs`: token-gated direct architecture kernels and direct
gather-offset validation.

Reuse the deterministic test helpers and fixed-seed noise generator. Do not add
nondeterministic randomness or duplicate oracle conventions. GF(2^16) byte
lengths must be even. Unsupported direct-kernel tests report a skip and return;
they are not marked ignored.

Direct arithmetic comparisons pass a concrete function item to `run_entry`
and call the supplied entry inside its body. The body completes its scalar
comparison before the gate emits a compiler-definition-path receipt.
`SIMD_BACKEND` limits execution to the resolved request and weaker tiers;
`FGF_REQUIRE_TIER` independently requires the host and feature set to support
its floor, including when the request is scalar. Geometry-panic and allocation
checks use the token gate without earning receipts.

`just kernel-registry-check` derives x86 entries from the pinned nightly's
all-feature hidden and private rustdoc documents. The registry audits exact
Archmage token types, public reachability, and explicit tokenless and private
helper exceptions, and binds the artifact to the source tree, features, and
generator. `just isa-accept TIER` generates it on the native host; supplying
`REGISTRY` requires an existing identity-matching artifact instead. CI downloads
that artifact from the registry job. Each initial and downgrade run audits its
receipts against its actual resolved ceiling and required floor.

The registry proves direct differential coverage per declaration, not per
generic specialization or public route. Specialization fixtures remain
independent obligations. Receipt identities use `std::any::type_name`, whose
format is not guaranteed; a toolchain format change fails the identity check
rather than weakening the registry match.

A behavior fix should retain a regression test when a plausible future bug
would fail it. Do not add tests that assert source text, field copies, forwarding,
or exact panic prose.

`TIERS` is `v4x v3_gfni_crypto v3 v2 scalar` (`v4x` resolves only with `simd512`). A forced tier on incapable hardware
may resolve to another supported tier; inspect reported backends before treating
a green run as ISA coverage.

Hosted coverage excludes only the explicit file lists in `crate.just`; the
hosted runner cannot guarantee GFNI or AVX-512.

| Group | Files relative to `src/kernel/` | Why hosted runners cannot cover it |
|---|---|---|
| GFNI | `x86/gf8/gfni.rs`, `x86/gf8/gfni/{elementwise,gather,matrix,rows,scatter,single}.rs`; `x86/gf16/gfni.rs`, `x86/gf16/gfni/{elementwise,gather,matrix,scatter,single}.rs`; `x86/gf32.rs`, `x86/gf64.rs` | GFNI with the selected crypto tier |
| AVX-512 | `x86/bytes/avx512.rs`; `x86/gf8/avx512.rs`, `x86/gf8/avx512/{elementwise,gather,matrix,scatter,single}.rs`; `x86/gf16/avx512.rs`; `x86/{goldilocks,mersenne31,quad_mersenne31}/avx512.rs`; `x86/residue/avx512.rs`, `x86/residue/avx512/{matrix,single}.rs`, `x86/residue/avx512/single/{u8,u32,u64}.rs` | AVX-512; the byte-field entries also require GFNI |
| Dispatch | `gf8.rs`, `tower.rs` | Generic dispatch arms are selected per field type and per tier, so no single monomorphized instance executes every arm; `llvm-cov` line totals score each function by its best instance, and hosted runners additionally miss the GFNI and `V4x` arms |

Portable bodies, narrower x86 kernels, and every other dispatch module remain
in the hosted 95% gate. `just cover-native` requires `v4x`, runs on a host that
executes every listed ISA, reports each group separately, and applies the 95%
gate to the hosted scope. Its group figures are informational:
`#[archmage::arcane]` kernel bodies carry almost no coverage mapping, so the
GFNI and AVX-512 figures measure wrapper lines rather than kernel bodies, and
the dispatch figure scores each generic function by its best monomorphized
instance. Kernel-body execution evidence comes from the registry receipts,
not from coverage. Adding an exclusion requires a per-file capability
justification here.

## Benchmarks

`just bench-pair` is this crate's B1 instantiation for the public operations
surface. A campaign checks out the base library revision and the candidate
library revision, overlays one measurement snapshot (a revision or directory)
on both — the snapshot's `benches/` tree (including the `operations` bench and
this runner), `Cargo.toml`, `Cargo.lock`, and any `rust-toolchain` file, with
`PAIR_FEATURES` and `PAIR_ALIGN` supplying the feature set and pinned flags —
and executes only the snapshot's bench target. The snapshot digest, runner
digest, method version `fgf-pair/3`, `drift_method`, both library SHAs, host
facts, execution isolation, requested and reported tiers, effective
configuration, per-run environment digests and argv lengths, unstable-cell
entries, and every rejection are written to
`bench-records/pair/<UTC stamp>-<host>/session.json` beside the raw logs. The
runner rejects a snapshot whose `bench_pair.py` differs from the one running.

```sh
FEC_GOLDEN_CORE=<cpu> just bench-pair --base REV --candidate REV [--targets FILE]
FEC_GOLDEN_CORE=<cpu> just bench-pair --mode calibration --base REV
FEC_GOLDEN_CORE=<cpu> just bench-pair --mode calibration \
    --calibration-mode base_vs_layout_control --base REV
FEC_GOLDEN_CORE=<cpu> just bench-pair --mode threshold \
    --base REV --candidate REV --threshold FILE --inventory FILE
FEC_GOLDEN_CORE=<cpu> just bench-pair --reanalyze SESSION_JSON
FEC_GOLDEN_CORE=<cpu> just bench-pair \
    --write-unstable benches/unstable_cells.json --unstable-session SESSION_JSON
```

### Descriptive baselines

`FEC_GOLDEN_CORE=<cpu> just bench-baseline --revision REV` measures one
library revision without a comparison arm or an acceptance claim. The default
`surfaces` catalogue covers scalar arithmetic, binary bases, storage forms,
generic extensions, maps, preparation, and bulk operations. Its recorded
default scope selects every non-query case; controls bracket execution but
never become measurement rows. `--bench operations` selects the page-aligned
bulk catalogue instead. The harnesses retain separate measurement identities.

The default uses `BASELINE_RUNS` standalone processes with
`BASELINE_ROUNDS` rounds on the host's strongest resolved tier. `--tiers`
selects an explicit tier list, and `--cells FILE` selects exact cases.
`surfaces` permits only a single round per process. Publication requires
at least three standalone processes. Run summaries produce a median and a
minimum–maximum range, not a confidence interval. Drift is recorded rather
than treated as paired acceptance evidence. Calibration and owner-signed
unstable-cell entries are not prerequisites for descriptive measurement.

Both runners share build, snapshot, isolation, and campaign-lock policy.
A descriptive session records one clean library revision, its harness
snapshot, effective configuration, exact scope, actual process and field
backends, full correctness preflight, and raw measurements under
`bench-records/baseline/`. It cannot contain a candidate arm or acceptance
decisions. A baseline runner must match its measurement snapshot; a
working-tree snapshot is selected explicitly with `--snapshot .`.

### Paired measurement scope

`bench-pair --cells FILE` restricts timing, never the full inventory or
cross-arm correctness preflight. The declaration is an object with a
`cells` array and a nonempty `rationale`; text or array selections supply
the rationale through `--scope-rationale`. Selected cells retain inventory
order. Empty, duplicate, unknown, or unexplained selections fail, as do
superiority targets or threshold-declared cells outside the selected scope.
Omitting the selection measures the full catalogue.

The scope covers every measured specialization affected by the change,
including shared kernels, prepared forms, relevant routing boundaries, and
controls. `bench-coverage-map` supplies execution evidence where available;
review records the reasoning where mappings cannot establish it. Changes to
shared dispatch, validation, or allocation paths require a full scope.
Paired bounds apply only to the declared scope. Calibrated run counts,
rounds, margins, and same-session pairing remain unchanged. Calibration
rejects selected scopes because its policy and unstable list cover the full
inventory.

Every requested tier is measured by executing each built arm under
`SIMD_BACKEND=<tier>`, in the requested order, which defaults to strongest
first. A reported process backend that differs from the requested tier rejects
the session; the per-field backends of both arms are recorded, and their
equality is part of the tier record. `v4x` requires the snapshot feature set to
include `simd512` or `all`. The ladder is the x86 ladder: `scalar` executes the
portable bodies and stands in for portable targets.

Before any timing, both arms must expose the same cell inventory, their
`--check` output digests must agree cell for cell, their normal dependency
sets must match the snapshot's, and the effective flags must carry the
required alignment tokens.

The runners take one exclusive `bench-records/.campaign.lock` in the crate
root for the whole session, regardless of capture destination, then re-probe
the host before the preflight and before every
measured run: `FEC_GOLDEN_CORE` must name an online CPU, that CPU must run
under the `performance` scaling governor, and, where SMT is active, its
siblings must stay below `PAIR_SMT_BUSY_THRESHOLD` over `PAIR_SMT_WINDOW`.
`setarch` must be available; every check and timing process runs under
`taskset` on the pinned core and under `setarch -R`, which disables ASLR for
the child. Both arms of a run receive one identical environment block, and
their argv lengths must match; a mismatch rejects the session. A failed
precondition rejects the session and is recorded.

Independent runs alternate arm order: base then candidate in one run,
candidate then base in the next. An arm's run summary is the median of its
rounds, and its session summary is the median of its run summaries. Every run
is bracketed by the harness's start and end controls. In acceptance and
threshold modes, the drift is `(p90 - p10) / median` of the start controls
across every arm's start control in a tier, recorded as `drift_method`, and
must not exceed the calibrated cap.

The estimator is the per-run log ratio `log(candidate / base)` for every cell.
Each cell's location estimate is the Hodges–Lehmann estimate of its ratios,
and its upper bound is the one-sided 95% distribution-free Wilcoxon
signed-rank bound, an order statistic of the Walsh averages. The exact null
distribution supplies the order statistic up to the enumeration limit; larger
run counts use a continuity-corrected normal approximation. A cell needs at
least the estimator's minimum paired-run count before a bound is defined.
Noninferiority holds when the upper bound is at most `log(1 + PAIR_DELTA)` for
every cell. Superiority holds when, for every cell of a target set declared
before measurement, the upper bound is below `log(1 - PAIR_GAMMA)`, and every
cell outside the target set stays noninferior. The target declaration is
digested into the session.

Calibration is the same recipe in `--mode calibration`: base against base, or
base against a layout control rebuilt from the same revision with a different
Cargo metadata seed. A calibration session reports, per tier, every cell whose
Hodges–Lehmann upper bound at the full calibration budget exceeds
`log(1 + PAIR_DELTA / 2)` as unstable, and recommends a stable-cell run count:
the smallest run count, at or above the estimator's minimum, that puts every
stable cell's bound at or below that limit. Unstable cells do not make the
session fail; a tier below the estimator's minimum run count is rejected as
insufficient. The recommendation also emits a drift cap from the largest
observed drift times `PAIR_DRIFT_SAFETY_FACTOR`; `PAIR_CALIBRATION_RUNS`
bounds how many runs it collects. The recommendation binds the exact
hostnames, measured tiers, toolchain identity, and effective flags, which the
recipe carries as `PAIR_CALIBRATED_HOSTS`, `PAIR_CALIBRATED_TIERS`,
`PAIR_CALIBRATED_TOOLCHAIN`, and `PAIR_CALIBRATED_FLAGS`. `PAIR_RUNS` and
`PAIR_DRIFT_CAP` are per-host `host:value` bindings for the exact hostname; a
single numeric value is accepted for a single-host campaign, and a host
without an entry is refused. Acceptance and threshold sessions refuse an
uncalibrated placeholder or a run count below the estimator's minimum, and
refuse a binding that does not match the live host and tree. A calibration
session is marked non-acceptance and is never reused as acceptance evidence.

The signed unstable-cell list is `benches/unstable_cells.json`, the default
`--unstable-file`. Each entry binds an exact host, tier, toolchain,
effective flags, and measured inventory digest to its cell list, calibration
sources, and owner signature. Acceptance and threshold sessions load the file,
require exactly one matching signed entry for every measured host and tier,
and reject a missing, stale, duplicate, or unsigned entry. A listed cell is
measured and published with the Unstable marker, is excluded from
noninferiority, and is excluded from the superiority rule's other-cell check.
A listed cell declared as a superiority target is evaluated as a target only
when the session runs at least a calibrated `PAIR_UNSTABLE_TARGET_RUNS`;
otherwise the session is rejected as underpowered. Calibration sessions report
their unstable cells per tier, and `--write-unstable FILE --unstable-session
SESSION_JSON` merges one or more calibration sessions into the file; every
updated entry loses its signature until the owner signs it again.

Each signed entry's inventory is retained as ordered cell metadata in
`benches/calibrated_inventories/<digest>.json`; its content must reproduce
that digest. Selected cells present in that catalogue must retain identical
metadata. New cells absent from it are recorded as uncalibrated and face the
paired gate without an unstable-cell exclusion. Adding a target does not
invalidate unrelated calibrated cells or require repeating calibration.
`--write-unstable` exports the matching catalogue beside the list, under
`calibrated_inventories/`. Recalibrating a host/tier/configuration replaces
its older inventory binding; conflicting inventories in one import fail.

`--layout-only` handles cells that fail noninferiority. It reads the largest
layout-control Hodges–Lehmann upper bound from matching non-rejected
calibration sessions with a defined bound and re-measures the failing cells
against a fresh layout-control arm in the same session; the larger bound
becomes the cell's spread. The record is
`spread_unavailable`, `regression_exceeds_spread`, or
`owner_exception_required`, and an owner signs the exception in the published
layout table. The procedure never records a pass for a failed cell.

`--mode threshold` measures threshold evidence. Its declaration names one
inventory item, the old and new cutoff values, and the cells with their axis
value and the routes the old and new gates select. The grid is in the cutoff's
unit: the endpoints, the doublings of the low cutoff that stay inside the span
for byte, lane, and element cutoffs (every integer for count, row, and term
cutoffs), and the integral midpoints between adjacent grid values. A
declaration with no cell strictly between the old and new cutoff is rejected,
as is a declared route that disagrees with the cutoff comparison. The measured
target set is every declared cell whose route changes.

`--reanalyze SESSION_JSON` recomputes a stored session's rule outcomes from
its per-run summaries under the current estimator and drift method, without
measuring. It writes a sidecar `<source stem>.reanalysis-<UTC stamp>.json`
beside the source, records the source method version, and marks the result
non-publishable and non-acceptance.

Policy values live in `crate.just` (`PAIR_DELTA`, `PAIR_GAMMA`, `PAIR_RUNS`,
`PAIR_DRIFT_CAP`, `PAIR_ROUNDS`, `PAIR_CALIBRATION_RUNS`,
`PAIR_DRIFT_SAFETY_FACTOR`, `PAIR_SMT_WINDOW`, `PAIR_SMT_BUSY_THRESHOLD`,
`PAIR_CHRT`, `PAIR_UNSTABLE_TARGET_RUNS`). `PAIR_RUNS` and `PAIR_DRIFT_CAP`
are per-host `host:value` bindings; `PAIR_CHRT` optionally prepends `chrt`
arguments to every harness invocation; `PAIR_UNSTABLE_TARGET_RUNS` is the
minimum session run count when a listed unstable cell is declared as a
superiority target. `--policy-json` carries the recipe's snapshot of the
acceptance policy values, and acceptance rejects an argument that differs;
`--exploratory` allows the override for a run that is not publishable.

### The operations bench

The `operations` bench is the stable-API measurement `bench-pair` drives. A
cell is one public `fgf::ops` or `fgf::ops::bits` call shape over deterministic
fixtures; the target names no `fgf::internals`, and every cell is checked
against an independent element-wise oracle before it is timed. A cell id is
`{field}/{operation}/{prep}/{row_len}x{rows}x{sources}/{alignment}`, where
`alignment` is `aligned` or `offsetN`. Registration order is fixed, so the
inventory is byte-identical across runs, hosts, and arms. Fixture storage is
page-aligned: destination, first source, second source, and the gather region
take fixed, distinct intra-page offsets; buffers at or above the
streaming-store size use huge-page alignment, and library-owned prepared
coefficient tables start at the page origin. `offsetN` shifts the destination
only.

The harness prints `BACKEND`, one `FIELD_BACKEND` line per registered field
label, and then `CELL` records for `--inventory`, `OUTPUT` digests for
`--check`, or a `CONTROL`-bracketed stream of `ROUND` records for
`--time --rounds N`. `--cells FILE` restricts a run to named cells; `--family`
selects one family panel, and omitting it runs every family. Setup, allocation,
and coefficient construction stay outside the timed region; a round is the
median of an adaptive inner batch.

### Coverage map

`just bench-coverage-map TIER` rebuilds the `operations` bench on the pinned
nightly with branch coverage, runs `--check` once per cell under
`SIMD_BACKEND=<TIER>`, and merges the per-cell profiles into one map. The map
relates every kernel-registry entry and threshold-inventory branch to the
cells that execute it, and it exits non-zero for an unexecuted required entry
or a one-sided required branch unless `benches/coverage_exceptions.json`
records the item with a reason. Exception classes are `unreachable`,
`unmapped_arcane`, and `ambiguous`; a stale exception is itself a failure. The
recipe needs `FGF_THRESHOLD_INVENTORY`, reads `scan.py` beside the inventory,
and regenerates the kernel registry when its identity does not match the tree.
The pinned compiler emits little or no mapping for `#[archmage::arcane]` kernel
bodies, so entry evidence is the wrapper's count or the nearest mapped caller
region, and an `unmapped_arcane` exception names the cells that straddle both
sides of the unmapped site.

### Publication

`just bench-publish` renders the committed record; `--verify` regenerates every
page in memory and fails on a manifest digest or byte mismatch without writing.
`benchmarks/sessions.json` is the committed manifest: it fixes the host order
and availability, lists every published `session.json` with its SHA-256 digest,
and lists the captured competitor logs. Operations rows retain their
`{host}/{tier}/{cell id}` anchors; surface rows use a separate harness-qualified
namespace and table. History appends in publication order. Latest measurements
retain the latest row per anchor with its measured revision and session, not
an assertion that every row describes the newest release. Descriptive rows
identify their standalone method, scope, repetition budget, and observed
range. Accepted sessions publish the candidate arm; same-revision paired
sessions retain their `baseline` label, and layout-control arms retain
`layout_control`. Calibration, threshold, layout exceptions, and competitor
captures have dedicated tables. Feature sets and effective flags must agree
within each published harness record.

The B6 digit rule applies to every generated table: the table uses the fewest
digits that give its smallest value three significant figures, capped at the
harness's recorded resolution; trailing zeros are printed, integer parts are
never rounded, and values never use scientific notation. Ratios keep five
digits. Rounding is half-up from the unrounded recorded values. Unavailable
numbers are `-`, units live in the table header, and caveats below a table
state host context, aggregation, and warm-up.

A publication refreshes only the anchors it measures. Other rows retain
their original provenance; they are neither relabelled nor compared across
sessions. Speedup, regression, crossover, and competitor claims require
matched same-session measurements of the relevant cases and implementations,
not a refresh of unrelated panels. Broad audits are explicit campaigns for
milestones or shared changes, not a prerequisite for every new number.
Raw captures remain in git-ignored `bench-records/`; only public-API timings
and the competitor matrix publish.

### Competitor campaigns

Competitor campaigns run `FEC_GOLDEN_CORE=<cpu> just bench-comp FAMILY`: `gf`
for the binary tower fields, `gdl` for Goldilocks, `m31` for Mersenne31 and
QuadMersenne31, and `gf16` for the native GF16 harness alone. A campaign runs
five complete rounds and shuffles the unit order within each round, printing
the realized order. The `self` unit runs the matching `operations` family
panel, plus `compare` for `gf`; the other units run the family's external
harness. Bit-packed GF(2) has no competitor harness. `gf16` runs under
`chrt --idle 0` in addition to the core pin.

Every external competitor harness is a separate unpublished package under
`external/`, outside `src/` and `benches/`; the package allowlist keeps the
directory out of `cargo package`. `external/bench-trio/` and
`external/gf16-bench/` carry a `build.rs` because they link native competitor
code; `external/prime-bench/` has no build script and builds with
`-C target-cpu=native`, which is how Plonky3 selects its packed kernels.

`external/bench-trio/` links the system Intel ISA-L and klauspost/reedsolomon
as a Go C archive and interleaves `fgf`, ISA-L, and klauspost over one
GF(2^8)/`0x11D` fixture set, rotating through all six arm orders. The
klauspost coding matrix arrives through `WithCustomMatrix`. Every arm is
validated against the competitor's own output before timing, and the control
row reports the harness's noise floor. ISA-L needs `libisal` through
pkg-config and the klauspost arm needs the Go toolchain in `PATH`; the build
fails rather than dropping an arm.

`external/prime-bench/` compares `fgf` against Plonky3 (`p3-goldilocks`,
`p3-mersenne-31`, and the packed quadratic extension of `p3-mersenne-31`) over
Mersenne31, Goldilocks, and QuadMersenne31 on each library's native layout.
Every arm is validated against the other library's output before timing, and
the control row reports the harness's noise floor. Layout conversion is
excluded from every timed region, and throughput counts one operand pair per
element. It runs through `bench-comp gdl` and `bench-comp m31`; a single rerun
is `RUSTFLAGS="-C target-cpu=native" cargo run --release -- [gdl|m31]` from
its directory. The banner reports packing widths; the captured log states the
flag and widths beside the resolved `fgf` backends.

`external/gf16-bench/` compares native GF16 field operations against pinned
GF-Complete, reed-solomon-erasure, reed-solomon-simd, and Leopard-RS 1.x
(pre-Leopard2); Leopard2 exposes no public field-region primitive, only codec
operations, so the harness pins the 1.x field kernels. Its representation maps
are bijective and preserve every basis product, and independent scalar
arithmetic validates every native scalar and timed region. Single-row cases
cover fixed and cycling coefficients; multi-row competitors compose native
region operations, and unsupported primitives remain absent. The duplicate
`fgf` arm measures bias. Arm orders follow a Williams balanced Latin square:
within one cycle every arm takes every position and directly follows every
other arm equally often. Each arm warms immediately before its timed batch,
and sampling stops only at a cycle boundary. `bench-comp gf16` runs the
targeted campaign; `bench-comp gf` includes the same harness in its shuffled
binary-field rounds. `benchmark-check-native` validates every geometry without
timing. The native build requires Git, C/C++ compilers, and an x86-64
AVX2/GFNI/crypto host; missing inputs fail rather than remove an arm.

### Other bench targets

`just bench-save NAME` and `just bench NAME` add Criterion's baseline flags
and apply only to the Criterion harness `prime_ntt`; every other bench target
is a `harness = false` binary with its own argument parser and ignores those
flags. `prime_ntt` is also declared `harness = false` and drives Criterion
through `criterion_main`. It interleaves the public QuadMersenne31 scalar
control, a duplicate control, and the direct AVX2 kernels per row length,
validates every output byte for byte before timing, and reports the process
and field backends. `just bench-build NAME` builds a named bench target
without running it for interleaved remote comparisons.

`just bench kernels` runs the internals-only direct prime-kernel panels;
`--blocked-diagnostic` replaces them with the blocked-versus-AXPY probe.
`just bench compare` reports `fgf` self-numbers over the competitor-harness
fixture family, and `--peel-diagnostic` isolates matched source and
destination offsets around the AVX-512 peel floors. `just bench field_maps`
measures the compact-map surface. `surfaces` covers the stable scalar,
representation, packing, bit-buffer, prepared-coefficient, row-operation,
capability-query, and map callables; `benchmark-check` checks its inventory
against compiler-derived rustdoc JSON, `--check` exercises every fixture
without timing, and `--smoke` shortens sampling without removing cases.
`--cases FILE` filters execution while retaining full inventory output and
automatic entry/exit controls. Selected checks use a distinct completion
marker; only an unfiltered check earns full-check completion.
The standalone baseline measures `surfaces` by default; paired campaigns
measure `operations`.

Benchmark setup, allocation, coefficient construction, and input generation
stay outside the timed region. Captured output records the CPU, OS, Rust
version, selected backend, geometry, command, and aggregation rule. A
performance change requires an unchanged callable control, differential
correctness, and a paired campaign in one session over its declared scope;
dispatch and crossover changes are never made from reasoning alone.

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
