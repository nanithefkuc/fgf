# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Mixed embedding composition over a sealed `Postcompose` right-hand
  operand: `Embedding::then` accepts a proper inclusion or an
  isomorphism out of the middle row, and
  `Isomorphism::then_embedding` composes an isomorphism into a proper
  inclusion.
- Prepared Frobenius maps with byte-row forms: `Frobenius` compiles one
  characteristic-power automorphism for repeated scalar and row use, and
  `ops::frobenius_into_with` / `ops::frobenius_assign_with` apply it
  across stable byte lanes.
- Prepared embedding and isomorphism maps with the complete byte-row
  surface: `PreparedEmbedding` compiles one canonical inclusion for
  repeated `embed`, `contains`, `restrict`, `frobenius`, `trace`, and
  `norm` use, `PreparedIsomorphism` compiles one presentation change for
  repeated `apply` / `unapply` use, and the `ops` row functions apply
  them across stable byte lanes with a validity bitmap for partial
  restriction.
- Bit-packed GF(2) map rows in `ops::map_bits`: `embed_into_with`,
  `trace_into_with`, `norm_into_with`, `restrict_into_with`, and
  `contains_into_with` move one prepared GF(2)-sourced map across packed
  bits and target lanes. Packed sides follow the `bits` LSB-first layout
  with caller-maintained zero padding; packed outputs are exactly
  `bits::bytes_for(bits)` bytes with cleared tail bits, lane sides hold
  whole target elements exactly, and empty rows are valid.
- Wiki reference for operation suffixes, destination behavior, prepared state,
- Prepared and overwrite forms of the offset-addressed matrix operations:
  `ops::mul_add_matrix_at_with`, `ops::mul_into_matrix_at`, and
  `ops::mul_into_matrix_at_with` complete the `mul_add_matrix_at` family with
  the same geometry checks and argument order as the contiguous forms. GF(2^8)
  dispatches the register-blocked GFNI and AVX-512 scattered kernels with
  overwrite seeded from zero in registers; every other field routes the
  dispatched single-row kernels over the named rows.

### Changed

- **Breaking:** the total extension degree cap is 64 at every
  characteristic. Declaring or validating a field whose total prime
  degree exceeds 64 is a build-time configuration error and panics:
  odd-characteristic extensions above degree 64 are no longer accepted,
  and index conversions share the same 64-coordinate capacity. Migrate
  wider odd towers (for example GF(3^66)) to degrees at or below 64.
- **Breaking:** `Embedding` names proper inclusions only: equal-degree
  construction reports `EmbeddingError::EqualDegree`, and
  `Isomorphism` is the only equal-degree map with `apply`/`unapply`.
  `Isomorphism::from_embedding` and `Isomorphism::to_embedding` are
  removed. Migrate equal-degree `Embedding::new`/`from_images` to
  `Isomorphism::new`/`from_images`, `embed` to `apply`, and `restrict`
  to `unapply` at equal degree.
- The wiki embedding guide covers the proper-inclusion and isomorphism
  split, mixed composition through `then` and `then_embedding`, and the
  `EqualDegree` and `IncompatibleDegree` rejection pair.
- The dependency lock records published sources for locked MSRV checks;
  umbrella-local patch resolution is excluded from the publication lock.

- Public documentation leads with `FiniteField`, its representations, and
  `Elem<F>`; named prime families appear as specializations. Wiki guides
  follow field selection, scalar arithmetic, packing, row operations,
  preparation, and deployment, with task-oriented navigation and a custom
  prime-base extension example.
- Benchmark publication refreshes all families and competitors together.
  Generic residue results use expandable, length-column tables with
  separate AVX2/scalar comparisons; explicitly unavailable hosts show `-`
  instead of historical measurements.

- Wiki guides use 3.0.1 as the base v3 release and lead from field construction
  and arithmetic through presentations, embeddings, and custom fields.
  Migration guidance goes directly from 2.x to 3.0.1.
- Wiki navigation groups the reading path in the home page and sidebar instead
  of per-page `Next:` footers.
- Four-byte generic residue rows vectorize elementwise multiplication on
  AVX2-capable x86 hosts.
- One-byte generic residue rows at moduli up to 127 vectorize prepared
  fixed-coefficient scaling through nibble-table lookups, four-byte rows
  through Shoup multiplication, and eight-byte rows at odd moduli through
  doubleword Montgomery multiplication; those rows report the vector
  backend, and vector-routed operations produce byte-identical output to
  the portable path on every input, applying the documented zero/unit
  shortcuts before vector dispatch.
- Four-byte generic residue rows delay reduction across shared sources in
  prepared overwrite/accumulate four-output matrices with enough shared
  sources and modulus headroom, as recorded in
  [the residue benchmark record](benchmarks/v3/residue.md), with delayed
  accumulation falling back to the repeated single-row path when an output
  row's coefficients are all zero.
- Operation implementations move from `ops.rs` into the `ops/` coefficient,
  single-row, scatter, gather, matrix, and packing modules, retaining public
  `ops::*` paths. Byte dispatch moves into `kernel/bytes.rs`.
- Generic residue single-row AVX2 kernels move from `residue/avx2/single.rs`
  into `residue/avx2/single/{u8,u32,u64}.rs`.
- **Breaking:** Unstable residue kernel entries use `u8`, `u32`, and `u64`
  lane names instead of `byte`, `word`, and `doubleword`; `WordParams`
  becomes `U32Params`. Migrate direct `internals` callers to the explicit
  lane-width spellings.
- **Breaking:** Unstable residue routing queries report an actual
  `Backend` through `single_row_backend` and `delayed_matrix_backend`.
  Replace boolean selector calls with comparisons to `Backend::Scalar`;
  use `avx2_floor_op_len` for the AVX2-specific row floor.
- The `alloc` feature documentation names the full prepared surface:
  the prepared maps, the `ops::map` and `ops::map_bits` modules, the
  prepared coefficient collections, the prepared scatter/gather/matrix
  `_with` forms, and `pack_to_vec`.

### Removed

- The `QuadraticExtension` public trait from the quadratic-conveniences
  proposal: no new public API ships. Degree-2 `Embedding` maps route
  internally through the Vieta closed forms with no public spelling
  change; `QuadMersenne31` keeps its inherent `const` conjugate/norm and
  drops the proposed inherent `trace` (use the embedding trace map).

- NEON and `simd128` acceleration: the `aarch64` and `wasm32` kernel
  subtrees and their dispatch arms are gone, and `internals` no longer
  exposes `kernel::{aarch64, wasm32}` or the `Neon`/`NeonAes`/`Wasm128`
  tokens. AArch64 and Wasm32 targets compile and run the portable scalar
  bodies; `backend_for::<F>()` reports `Backend::Scalar` for every field
  there (`backend()` still reports `simdispatch`'s process selection).
- Wiki migration guide from the withdrawn v3 release candidate.

### Fixed

- Mersenne31 x86 multiplication accepts lanes already inside its redundant
  residue range directly and normalizes wider raw lanes before multiplication,
  preserving arbitrary raw inputs and canonical outputs.
- Pinned extension scalar arithmetic exposes its instance bodies for
  cross-crate inlining, removing runtime instance dispatch from specialized
  QuadMersenne31 operator multiplication. Investigation and merge gates:
  [Issue #47](https://github.com/nanithefkuc/fgf/issues/47).
- QuadMersenne31 scalar products defer reduction across the real and imaginary
  product sums instead of reducing each base-field product separately.
- Mersenne31 multiply-accumulate accepts redundant destination zero, and
  elementwise products share input-range checks across vector tiles while
  retaining ISA-width remainders.
- QuadMersenne31 vector products retain redundant zero internally; scalar
  arithmetic consumes canonical element storage without repeating entry
  reductions. Constructors and packed-buffer decodes retain normalization.
- Mersenne31 AVX-512 products use explicit two-source word permutations,
  and multiply-accumulate screens each register group for the high bit,
  canonicalizing the group before any store when a lane carries it and
  retaining ISA-width tails. Measurements:
  [PR measurement record](https://github.com/nanithefkuc/fgf/pull/48#issuecomment-6163124505).
- Mersenne31 and QuadMersenne31 AVX-512 multiplication screens operand
  lanes for the high bit instead of canonicalizing every load: lanes
  already inside the redundant range multiply as loaded. A lane outside it
  triggers canonicalized recomputation of its register group for the
  in-place entries and of the whole call for the out-of-place entries,
  before any final store. Arbitrary raw inputs and canonical outputs are
  preserved.
- Goldilocks scalar multiplication feeds the operand product directly to
  the wide reduction, which is total over `u128` and needs no canonical
  operands.

## [3.0.1] - 2026-10-10

### Added

- The canonical generic field family `FiniteField<const P: u64, const N: u32,
  R: FiniteFieldRepr<P, N> = Residue>` with the `Residue` presentation at
  degree one over every `u64` prime. Each row exposes the full scalar and
  `const` element surface (`from_raw`, checked `from_residue`, `to_residue`,
  arithmetic, `pow`, `canonical`),
  total arithmetic under the crate-wide conventions (`inv(0) == 0`,
  `x / 0 == 0`, `0^0 == 1`, with the explicit inverse-zero branch at
  `P = 2`), a little-endian wire lane sized to the characteristic —
  characteristic two uses an eight-byte lane, and an odd characteristic
  uses one, two, four, or eight bytes as its residues fit within `2^8`,
  `2^16`, `2^32`, and `u64` — with `FieldBuffer::decode`/`encode` as the
  byte API of these rows, the checked packed
  operation surface with prepared coefficients, and `Scalar` routing on
  every host. Broadcast add/sub take the modular path for every
  characteristic, so packed lanes stay canonical at `P = 2`. Values live in
  `field::finite_field`, re-exported at the root.
- Sealed `PrimeField` marker for degree-one fields, the validated `const`
  integer reductions `field::prime::reduce` and `reduce_wide`, `RangeError`
  for out-of-range conversions, and the `PrimeIdentity` characteristic
  facts `BITS`, `TWO_ADICITY`, and `RESIDUE_MOD_8`.
- `just compile-fail` compiles the negative consumer snippets under
  `tests/compile_fail/` on the crate MSRV, each required to fail with its
  declared crate panic fragment, or simply to be rejected for trait-bound
  and privacy cases, and `just msrv-accept` runs
  those snippets plus the `tests/consumers.rs` compile-and-run matrix on
  the MSRV toolchain.
- Wiki source guides cover the generic type model, field presentations,
  embeddings, custom towers and scalar fields, packed workflows, preparation,
  prime and bit-packed arithmetic, deployment, and migration from 2.x.
  `wiki/Home.md` and the sidebar organize the reader paths.
- Sealed presentations `MersenneFold` (label `mersenne-fold`) and
  `GoldilocksFold` (label `goldilocks-fold`), re-exported at the root. Both
  rows implement `PrimeField` and gain `const` `from_residue(u64) ->
  Result<Self, RangeError>` and `to_residue(self) -> Option<u64>`. The
  default `Residue` row at either modulus remains a distinct type with its
  own lane width; values move between the two by residue conversion.
- The binary degree-one rows `FiniteField<2, 1, R>` implement `PrimeField`
  and gain `const` `from_residue(u64) -> Result<Self, RangeError>` and
  `to_residue(self) -> Option<u64>`; they stay distinct from
  `FiniteField<2, 1, Residue>`, which stores a `u64` residue.
- `just external-check` type-checks every package under `external/`
  against the working copy.
- The generic extension presentation `FiniteField<P, N, Extension<S, D>>`
  with data-only `ExtensionSpec` providers, the sealed storage family
  (`Components`, `Packed`, `PackedPadded`), the opaque const-constructible
  `Relation`, generic arithmetic under the crate-wide total conventions,
  component bytes and the checked packed surface with `Scalar` routing,
  `ExtensionField` component conversions, and `BinaryField` on
  characteristic-two rows over binary bases. The surface lives in
  `field::extension`, re-exported at the root. `EmbeddingError` gains the
  `NoCanonicalMap` variant for dividing degrees with no canonical map.
- `ComplexMersenne31`, the `ExtensionSpec<2>` of `QuadMersenne31`:
  `Mersenne31` extended by `t^2 + 1` with `Components` storage,
  re-exported at the root.
- An extension row whose base presentation, relation, and storage equal a
  shipped instance's runs that instance's specialized scalar arithmetic and
  reaches its kernel arms; every other extension row runs the generic body
  and reports `Scalar`.
- `FieldElem::from_residue`/`to_residue` and `from_index`/`to_index` on
  every row: the prime-subfield value and the presentation index share one
  generic body over the crate-private row hooks, and the inherent `const`
  residue constructors check through the same bound.
- The characteristic-independent relation surface in `field::embedding`,
  re-exported at the root: `Embedding<S, T>` over every pair of rows of
  equal characteristic whose degrees divide, with `from_images`,
  `to_images`, `generator_count`, `generator`, `then`, and
  `relative_degree`; `Isomorphism<S, T>` with `apply`, a total `unapply`,
  `inverse`, and composition; and the `NoCanonicalMap` and `InvalidImages`
  variants of `EmbeddingError`. Maps come from structural inclusions, the
  frozen binary ladder, or validated caller images; every pre-cutover
  binary result is unchanged.

### Changed

- **Breaking:** generic residue rows size their byte lane to the
  characteristic instead of always storing eight bytes: characteristic
  two keeps an eight-byte lane, and an odd characteristic uses one, two,
  four, or eight bytes as its residues fit within `2^8`, `2^16`, `2^32`,
  and `u64` — for example `FiniteField<7, 1>` moves one byte per element
  and `FiniteField<1_000_000_007, 1>` moves four. `FieldBuffer::decode`
  and `encode` are the byte API of these rows. Migration: the inherent
  fixed-array `from_bytes`/`to_bytes` are absent for these rows (every
  characteristic, `P = 2` included), and no `const` byte form exists for
  them — size buffers from `FieldBuffer::BYTES`, decode and encode
  through `FieldBuffer`, and in `const` code build from `from_raw` with
  `u64::from_le_bytes` and read back through `to_raw`. Extensions over a
  residue base shrink with the base lane; the named prime rows, binary
  rows, towers, and `QuadMersenne31` keep their lane widths and fixed-array
  methods.
- **Breaking:** `Mersenne31` and `Goldilocks` are type aliases of the
  canonical family rows `FiniteField<0x7FFF_FFFF, 1, MersenneFold>` and
  `FiniteField<0xFFFF_FFFF_0000_0001, 1, GoldilocksFold>` instead of unit
  marker structs. Moduli, raw widths, bytes, generators, arithmetic, and
  kernel routing are unchanged. The aliases are no longer value
  expressions: replace `Mersenne31` or `Goldilocks` used as a value with
  `Mersenne31::default()` (or drop the value where only the type was
  needed). Marker `Debug` now prints the family spelling, e.g.
  `FiniteField<2147483647, 1, mersenne-fold>`. Kernel receivers move to the
  rows with the aliases, so `FieldKernels` calls through the alias names
  compile unchanged.
- **Breaking:** the `Binary<N, R>` marker is removed without an alias;
  binary fields are the canonical family rows `FiniteField<2, N, R>`.
  Replace `Binary<8, Polynomial<AES>>` with `FiniteField<2, 8,
  Polynomial<AES>>` and `Binary::<…>` with `FiniteField::<2, …>`, and
  import `FiniteField` instead of `Binary`. Raw widths, bytes, descriptors,
  coordinates, embeddings, bit-packed buffers, and kernel routing are
  unchanged.
- **Breaking:** `BinaryRepr<const N: u32>` and `BinaryDegree<const N: u32>`
  take `u32` degree parameters instead of `u8`; generic code declaring
  `const N: u8` against them changes the parameter type.
- **Breaking:** `BinaryRepr` now extends `FiniteFieldRepr<2, N>`, which
  owns `NAME` and `VALID`. Replace `<R as BinaryRepr<N>>::NAME` and
  `::VALID` with `<R as FiniteFieldRepr<2, N>>::NAME` and `::VALID`;
  `DESCRIPTION` and `LOG_EXP` stay on `BinaryRepr`.
- **Breaking:** binary marker `Debug` prints the family spelling with the
  presentation label, e.g. `FiniteField<2, 8, GF(2^8)/0x11B>`, instead of
  the label alone. Element `Debug` and every wire format are unchanged.
- **Breaking:** binary rows validate their defining data at every entry
  point: `ZERO_RAW`, `ONE_RAW`, and every raw `Field` method now fail const
  evaluation for an invalid polynomial or basis, as `from_raw` already did.
  Valid presentations are unaffected.
- **Breaking:** `Field`, `HasGenerator`, and `FieldBuffer` are sealed.
  Implementing any of them outside the crate no longer compiles. Migrate a
  custom prime field to the family's degree-one row `FiniteField<P, 1>`
  (the default `Residue` presentation) at any checked `u64` prime: it
  supplies `Field`, `PrimeField`, `FieldBuffer`, the checked packed
  operation surface, and the `const` element API. Migrate a custom
  extension to an `ExtensionSpec` over a crate row.
- **Breaking:** the shipped towers are extension rows: `FiniteField<2, N,
  Tower<S>>` is spelled `FiniteField<2, N, Extension<S, 2>>` for `S` in
  `Rijndael16`, `Rijndael32`, `Rijndael64`, and `FanPaar2` through
  `FanPaar64`, each now an `ExtensionSpec<2>` with `Packed<W>` storage
  (`PackedPadded<u8>` for `FanPaar2` and `FanPaar4`). Raw words, bytes,
  coordinate words, generators, embeddings, and kernel routing are
  unchanged, and each instance keeps its `const` element surface and
  `mul_alpha`. Replace `Tower<S>` with `Extension<S, 2>` and import
  `Extension`.
- **Breaking:** the wide shipped instances' component conversions take
  and return arrays: `Elem::<F>::from_components(a, b)` becomes
  `from_components([a, b])` and `to_components()` returns `[a, b]` instead
  of `(a, b)`, matching `ExtensionField<2>`.
- **Breaking:** towers leave `BinaryRepr`: `BinaryRepr<N>` covers the flat
  presentations only, and no flat presentation exists at degrees 16, 32,
  and 64. Replace generic bounds `R: BinaryRepr<N>` that reached towers with
  field-level bounds (`F: BinaryDegree<N>` or `F: BinaryField`) and
  `<Tower<S> as BinaryRepr<N>>::DESCRIPTION` with `F::DESCRIPTION`.
- **Breaking:** `QuadMersenne31` is a type alias of
  `FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>` instead of
  a unit marker struct, and its raw word is `[u32; 2]` instead of
  `(u32, u32)`, with the same limbs in the same order. Replace
  `Elem::<QuadMersenne31>::from_raw(re, im)` with `from_raw([re, im])`,
  `to_raw().0` and `.1` with `to_raw()[0]` and `[1]`, `from_components(re,
  im)` with `from_components([re, im])`, and tuple destructuring of
  `to_components()` with array destructuring. Element `Display` prints
  `[re, im]` in the base field's form instead of `(re+imi)`; bytes,
  arithmetic, the generator, `I`, `conjugate`, `norm`, and kernel routing
  are unchanged. Replace `QuadMersenne31` used as a value with
  `QuadMersenne31::default()`.
- `BinaryDescription::inv` takes canonical coordinate words. On a
  polynomial-root description, a word with bits at or above `degree()`
  inverts its low `degree()` bits instead of returning an unspecified
  value; on other roots such a word still gives an unspecified result.
  Results on canonical words are unchanged.
- **Breaking:** `FieldElem` owns the residue and index conversions
  `from_residue`/`to_residue` and `from_index`/`to_index`, shared by every
  row. `BinaryField::to_coordinates`/`from_coordinates` are removed: use
  `to_index`/`from_index`, identical words below degree 64 widened to
  `u128`. `CoordinateError` is replaced by `fgf::RangeError` (also
  `fgf::field::RangeError`). Residues and indices are distinct: the index
  of one is the presentation's coordinate encoding of one.
- Binary embedding evaluation no longer uses precomputed linear columns:
  maps evaluate by Horner substitution and restrict through trace-dual
  bases, with unchanged results on every pair.
- **Breaking:** the `field::binary` embedding paths are removed without
  an alias: `fgf::field::binary::embedding`, `Embedding`, and
  `EmbeddingError` under it. Import from the crate root or
  `field::embedding` instead; the struct bounds widen from `BinaryField`
  to `Field`, and `Embedding` keeps `Clone`, `Copy`, and `Debug`.

### Removed

- **Breaking:** `Tower`, `TowerSpec`, and `TowerGeneratorSpec` are removed
  without an alias. A custom tower becomes an `ExtensionSpec<2>`: `impl
  TowerSpec for S` with `LINEAR_COEFFICIENT = A` and
  `CONSTANT_COEFFICIENT = B` becomes `impl ExtensionSpec<2> for S` with
  the same `Base` and `NAME`, `type Storage = Packed<W>` at the field's
  width (`PackedPadded<u8>` below one byte), and `const RELATION: Relation
  = Relation::from_binary_words::<Base>(&[B, A])` — constant coefficient
  first, the reverse of the old field order. The row is
  `FiniteField<2, N, Extension<S, 2>>`.
- **Breaking:** custom extension rows carry no generator:
  `TowerGeneratorSpec` has no replacement, and `HasGenerator` belongs to the
  shipped instances only. Request a shipped instance for a presentation that
  needs a checked generator.
- **Breaking:** custom extension rows have no `const` element surface:
  `from_raw`, `from_bytes`, `to_bytes`, the arithmetic methods, and the
  component conversions that custom towers exposed as `const fn` are gone.
  Use the `FieldElem` methods and operators, `FieldBuffer::decode` and
  `encode`, and `ExtensionField::from_components` and `to_components` at
  run time; `Elem::ZERO`, `Elem::ONE`, and `to_raw` stay `const`.

## [3.0.0-rc.1] - 2026-10-07

### Added

- Version-independent benchmark labels and separate absolute `benchmarks/v2/`
  and `benchmarks/v3/` records, with matched speed factors under
  `benchmarks/comparison/`. Released v2.1.0 replaces the generic-API baseline.
- Reproducible benchmark-only v2 preparation, paired host campaigns, and a
  fail-closed publisher with a raw-record cell evidence index.
- Custom RS-rooted `gf16d` one-shot, prepared, and scalar measurements, with
  no fabricated released-v2 counterpart.
- Explicit binary embeddings: `Embedding<S, T>` prepares the canonical
  inclusion of one binary field into another whose degree is a multiple of
  it, for every supported presentation — polynomial, ordered-basis, and
  tower, custom `TowerSpec` instances included. One frozen reference per
  degree (`Polynomial<3>`, `Polynomial<0x7>`, `Polynomial<0x13>`, the AES
  field, and the Rijndael towers `Binary<16, Tower<Rijndael16>>`,
  `Binary<32, Tower<Rijndael32>>`, and `Binary<64, Tower<Rijndael64>>`) fixes the
  consecutive inclusions: the identity into degree two, smallest-raw-root
  inclusions below the AES field, and low-component inclusions above it.
  `new` returns `EmbeddingError::IncompatibleDegree` when the source degree
  does not divide the target degree. `embed`, `contains`, `restrict`,
  `frobenius`, `trace`, and `norm` apply the prepared column arrays without
  allocating; equal presentation descriptions embed identically, and
  `Embedding<Binary<1, Polynomial<3>>, T>` supplies the absolute trace and norm.
- Recursive quadratic towers through degree 64. One open `TowerSpec`
  supplies `Base`, `LINEAR_COEFFICIENT`, `CONSTANT_COEFFICIENT`, and `NAME`
  for every quadratic tower at degrees 2, 4, 8, 16, 32, and 64 over a
  supported base of half the degree; `TowerGeneratorSpec` optionally pins a
  multiplicative generator checked to full order against the complete prime
  factors of `2^N - 1` at each degree. Shipped specs root at the AES field
  (`Rijndael16`, `Rijndael32`, `Rijndael64`) or nest from GF(2) through the
  Wiedemann relation (`FanPaar2` through `FanPaar64`, the degree-two and
  degree-four levels joining the ladder). Custom specs validate through the
  field description: the base first, excess coefficient bits, `A != 0`, and
  absolute `trace(B/A^2) == 1` for the relation `t^2 + A*t + B = 0`.
- Custom tower bulk operations: one dispatch implementation per degree
  serves every representation, routing pinned Rijndael and Fan-Paar
  descriptions to their vector kernels and every other tower to the typed
  scalar fallback with `backend_for` reporting `Scalar`; degree-eight
  towers join the single degree-eight dispatch. Towers at degrees 8, 16,
  32, and 64 implement `FieldBuffer`.
- Ordered-basis byte presentations: `Normal<P, E>` wraps a polynomial
  element into its conjugate basis (bit `i` selects `E^(2^i)`; one is all
  bits set) and `Cantor<P, SEED>` builds the chain `c_(N-1) = SEED`,
  `c_i = c_(i+1)^2 + c_(i+1)` (bit `i` selects `c_i`; validity asserts
  `c_0 == ONE`), at degrees 2, 4, and 8. Basis rank and inverse columns are
  checked by bounded enumeration; generators are the transported polynomial
  generators, verified to full order. Polynomial presentations extend to
  degrees two (`Polynomial<0x7>`) and four (`Polynomial<0x13>`,
  `Polynomial<0x19>`, `Polynomial<0x1F>`); degrees 1, 2, and 4 are
  scalar-only. Basis bulk operations run the typed scalar fallback;
  polynomial arithmetic never touches basis coordinates.
- Open scalar field definitions. `Field` is implementable downstream;
  `PrimeCharacteristic<P>` supplies a checked prime identity at any `u64`
  prime through deterministic Miller-Rabin, and the central validation
  block — reached from every generic construction — checks the
  characteristic identity, a positive degree, `ORDER ==
  CHARACTERISTIC^DEGREE` with overflow checking, and the binary degree cap
  of 64. Binary fields convert through `BinaryField::to_coordinates` and
  `from_coordinates`, which reject excess bits above the degree through
  `CoordinateError`.
- `just cross-check` compiles the AArch64 and Wasm library paths, including
  the explicit Wasm SIMD configuration, and `just bench-build NAME` builds a
  portable benchmark artifact without running it.
- `just bench-gf16-comp` runs an interleaved native GF(2^16) field
  comparison against GF-Complete, `reed-solomon-erasure`,
  `reed-solomon-simd`, and Leopard-RS 1.x; the harness validates basis maps
  and region outputs against independent arithmetic, includes
  duplicate-operation controls, and reports unavailable primitives
  separately. `bench-gf-comp` includes the same harness in its shuffled
  rounds.
- The benchmark record is split by field family. `BENCHMARKS.md` indexes
  the pages under `benchmarks/` and holds the hosts, shared method, number
  format, and reproduction commands; each page carries its own setup,
  self-timings, and competitor tables. Every value in a table has the same
  digit count, so paired Tiger Lake / Golden Cove cells line up down each
  column.

### Changed

- **Breaking:** fields have no public type aliases; `Binary<N, R>` is the
  only spelling. `Gf1`, `Gf16`, `Gf32`, `Gf64`, and `FanPaar2` through
  `FanPaar64` (the field aliases) are not provided: replace each with its
  canonical type — `Binary<1, Polynomial<3>>`,
  `Binary<16, Tower<Rijndael16>>`, `Binary<32, Tower<Rijndael32>>`,
  `Binary<64, Tower<Rijndael64>>`, and `Binary<N, Tower<FanPaarN>>` — or
  declare a local alias such as
  `type Gf2P16 = Binary<16, Tower<Rijndael16>>;`. The Fan-Paar spec markers
  are named like `Rijndael16`: `FanPaar2Spec` through `FanPaar64Spec` become
  `FanPaar2` through `FanPaar64`, so Fan-Paar GF(2^16) is
  `Binary<16, Tower<FanPaar16>>`.
- **Breaking:** `Binary<N, R>` replaces `Gf<N, R>` and explicitly names
  characteristic-two fields; `BinaryField` remains the capability trait.
  `Polynomial<P>` replaces `Poly<P>`, and `RS` replaces `REED_SOLOMON`
  for the unchanged `0x11D` polynomial convention. The `Gf8<R>` transition
  alias is removed: replace it with `Binary<8, R>`, including associated
  item paths such as `Binary::<8, Polynomial<AES>>::ORDER`. For example,
  `Elem<Gf8<Poly<REED_SOLOMON>>>` becomes
  `Elem<Binary<8, Polynomial<RS>>>`. No compatibility aliases remain.
- **Breaking:** `TowerSpec::LINEAR_COEFFICIENT` and
  `TowerSpec::CONSTANT_COEFFICIENT` replace `TowerSpec::A` and
  `TowerSpec::B`. Rename the associated constants in custom specs and their
  qualified uses; values remain raw coordinates in `Base` for the relation
  `t^2 + A*t + B = 0`. `TowerSpec` and `TowerGeneratorSpec` remain separate:
  arithmetic requires no selected generator. Field arithmetic, wire bytes,
  selected generators, and backend routing are unchanged.
- The canonical $GF(2)$ benchmark label is `gf1`, denoting extension degree.
  Its pages move from `gf2.md` to `gf1.md` in the v2, v3, and comparison
  directories; the released-v2 API mapping remains `Gf2`.
- **Breaking:** one element wrapper, `Elem<F>`, replaces every concrete
  element path. `mersenne31::Elem`, `goldilocks::Elem`,
  `quad_mersenne31::Elem`, the binary `Elem<N, R>` spelling, the
  `gf16`/`gf32`/`gf64` and Fan-Paar element structs, and `gf2::Elem` are
  removed, as are the module `GENERATOR` constants (`Elem::<F>::GENERATOR`
  replaces them all). `Gf2` becomes `Binary<1, Polynomial<3>>`, and
  `bits::dot_product` returns `Elem<Binary<1, Polynomial<3>>>`. Migrate
  `gf2::Elem` to `Elem<Binary<1, Polynomial<3>>>`, `mersenne31::Elem` to
  `Elem<Mersenne31>`, `gf16::Elem` to `Elem<Binary<16, Tower<Rijndael16>>>`,
  `fan_paar::fp16::Elem` to `Elem<Binary<16, Tower<FanPaar16>>>`, and the
  binary `Elem<8, Poly<POLY>>` to `Elem<Binary<8, Polynomial<POLY>>>`;
  spell elements of the wider towers and the prime families the same way.
  Inherent `const` arithmetic sits on the `Elem<F>` types; `ZERO`, `ONE`,
  `GENERATOR`, and `to_raw` are the generic items and are not redeclared
  per family. `Debug`
  for every element is the generic `NAME(raw)` format; family `Display`
  formats are unchanged. The scalar-arithmetic trait `field::Elem` is
  renamed `FieldElem` (re-exported as `fgf::FieldElem`): replace
  `use fgf::field::Elem;` trait imports with `use fgf::field::FieldElem;`.
  Wire bytes, generators, and relations are unchanged.
- **Breaking:** `Field` is split into three contracts. `Field` keeps scalar
  algebra and metadata (`Raw`, `Characteristic`, `NAME`, `DEGREE`, `ORDER`,
  `ZERO_RAW`, `ONE_RAW`, `VALID`, raw arithmetic); `FieldBuffer` carries the
  byte encoding (`BYTES`, `STORAGE_BITS`, `decode`, `encode`), implemented
  by binary degrees 8/16/32/64 and the prime families; `HasGenerator`
  carries the selected generator (`GENERATOR_RAW`). `Field::Elem`,
  `Field::BITS`, `Field::CHARACTERISTIC`, `Field::GENERATOR`, and
  `Field::elem_count` are removed. Generic code reads the characteristic as
  `<F::Characteristic as PrimeIdentity>::CHARACTERISTIC`, the storage width
  as `FieldBuffer::STORAGE_BITS`, and the generator as
  `Elem::<F>::GENERATOR` under a `HasGenerator` bound. `FieldKernels` and
  every `ops` operation — `pack`, `unpack`, and `pack_to_vec` included —
  bound on `FieldBuffer` and take `Elem<F>`. `FieldElem` is sealed and
  implemented only by the blanket `impl<F: Field> FieldElem for Elem<F>`;
  downstream custom fields implement the new `Field` contract and receive
  the trait. The trait exponent is `u128` (`pow(u128)`), the inherent
  `const pow` on every family takes `u128`, and `QuadMersenne31::pow_u128`
  is removed in its favour.
- **Breaking:** prime `from_raw` canonicalizes and `to_raw` returns
  canonical storage. In 2.x, `from_raw` and `decode` retained the
  unreduced lane and `to_raw` exposed the stored bits; every safely
  constructed element now holds canonical coordinates, so comparison,
  hashing, and packing see the reduced lane. Packed prime operations accept
  every lane bit pattern a decode accepts and emit canonical lanes on
  output — accumulate, in-place, and tails included — and zero-coefficient
  accumulation and unit-coefficient in-place scaling are defined no-ops
  that preserve their inputs byte for byte. Migrate by deleting manual
  canonicalization after construction; published canonical byte fixtures
  and mathematical values are unchanged. The inherent `canonical()` methods
  are removed; `pack`/`unpack` round-trip canonical lanes.
- **Breaking:** binary representations are descriptor-backed. `Repr<N>` and
  `ByteRepr` are removed in favour of `BinaryRepr<N>`, which carries the
  presentation `NAME`, an opaque `BinaryDescription`, an optional
  `ByteLogExp` table slot at degree eight, and the use-time `VALID` check;
  `BinaryField` adds the canonical coordinate conversions and
  `BinaryDegree<N>` pins the exact degree a generic tower row may build on.
  Migrate `Repr<8>`/`ByteRepr` bounds to `BinaryRepr<8>` and read tables
  through `LOG_EXP`; the kernel table banks
  move with the representations. The representations live in the binary
  subtree: `field::binary` (`Binary`, the coordinate surface),
  `field::binary::poly`, `field::binary::normal`, `field::binary::cantor`,
  and `field::binary::description` replace the removed `field::repr`,
  `field::poly`, and `field::gf1` modules.
- **Breaking:** `Poly<const P: u32>` becomes `Polynomial<const P: u128>`,
  and the `AES` and renamed `RS` constants widen to `u128` with the same
  values. `Polynomial` implements `BinaryRepr` at degrees 1, 2, 4, and 8;
  only `Polynomial<3>` is the degree-one form, giving GF(2) as
  `Binary<1, Polynomial<3>>`. Generic flat byte fields support any
  irreducible degree-eight polynomial — Data Matrix `0x12D` and the CCSDS
  basis `0x187` included — with x86 GFNI elementwise multiplication deriving
  an isomorphism onto the AES field for each polynomial. Migrating from the
  2.x names: `Gf8B`/`Gf8D` become `Binary<8, Polynomial<AES>>`/
  `Binary<8, Polynomial<RS>>`, `gf8b::Elem`/`gf8d::Elem` become
  `Elem<Binary<8, Polynomial<AES>>>`/`Elem<Binary<8, Polynomial<RS>>>`,
  and other conventions spell `Polynomial<0x12D>` or `Polynomial<0x187>`
  directly; import the constant (`Polynomial<AES>`) rather than qualifying
  it (`Polynomial<fgf::AES>`) in const-generic position. Values, wire
  encodings, and generators are unchanged.
- **Breaking:** the Rijndael and Fan-Paar hierarchies collapse into the
  recursive tower. `RijndaelTower` is renamed `Rijndael16` and joined by
  `Rijndael32` and `Rijndael64`: 2.x `Gf16`, `Gf32`, and `Gf64` become
  `Binary<16, Tower<Rijndael16>>`, `Binary<32, Tower<Rijndael32>>`, and
  `Binary<64, Tower<Rijndael64>>`. The `gf16`, `gf32`, and `gf64` modules,
  `quad_tower!`, the concrete Fan-Paar element and marker types, the
  `field::fan_paar` and `field::wiedemann` modules, and their
  `DELTA`/`ALPHA`/`GENERATOR` module constants are removed. Migrate the
  relation constants to `LINEAR_COEFFICIENT`/`CONSTANT_COEFFICIENT` in the
  spec; wire bytes, generators, and relations are unchanged. The
  independent Fan-Paar recurrence stays available as
  `field::binary::tower::{fp_multiply, fp_square, fp_invert,
  fp_mul_alpha}`, re-exported under `internals` as
  `internals::field::fan_paar`. `TowerSpec` is open: `LINEAR_COEFFICIENT`
  and `CONSTANT_COEFFICIENT` are `u64` base-coordinate words, the base
  bound becomes `BinaryField` at half the degree, and the generator moves
  to the optional `TowerGeneratorSpec`. The unstable
  `internals::kernel::fan_paar` dispatch
  module merges into `internals::kernel::tower::{gf16, gf32, gf64}`, whose
  `Prepared` enums carry raw words and tiles; `x86::fan_paar` keeps its
  token-bearing entries unchanged. Path moves: `field::tower` →
  `field::binary::tower`; `field::wiedemann` → the recurrence functions in
  `field::binary::tower`; the `kernel::fan_paar` dispatch merges into
  `kernel::tower`.
- Bulk routing follows the field description. Ordered-basis byte fields,
  degree-eight towers, custom tower specs, and towers over basis bases
  route every bulk operation and tail through the typed scalar fallback,
  with `backend_for` reporting `Scalar`; the GFNI degree-sixteen route that
  served non-AES polynomial bases through the AVX2 tables kernels is
  removed in favour of that fallback. `backend_for::<F>()` states the tier
  actually serving `F`'s bulk operations and
  `has_vector_elementwise::<F>()` reports elementwise eligibility
  separately. XOR-shaped operations — row addition, broadcast scalars, and
  `add`/`sub_assign` — stay field-independent and serve basis coordinates
  on the vector units.
- **Breaking (unstable surface only):** the direct kernels under
  `fgf::internals::kernel` are laid out by field family, then instruction
  set, with one entry per operation and ISA, representation-erased. The
  GFNI and AVX-512 fixed-coefficient entries take `Prepared` —
  unparameterized, built with `Prepared::new` — instead of a typed element
  or `Prepared<POLY>`; the GFNI byte bodies select their multiply through a
  `const NATIVE: bool` parameter (`true` for the AES encoding, `false` for
  the affine map). `scale_table_8d` and `affine_8b`/`affine_8d` become
  `scale_table` and `affine_map`. Elementwise ISA entries take the
  isomorphism onto the AES field and the reduction byte as explicit
  arguments (`isomorphism_to_aes::<POLY>()` and the polynomial's low byte)
  instead of a polynomial const parameter; the GF(2^16) elementwise entries
  take the relation's constant byte and the base reduction byte. The
  multi-row x86, AArch64, and Wasm entries take prepared coefficients — a
  `Matrix<Prepared>` provider such as `FlatMatrix<Prepared>` or a slice of
  `(&[Prepared], &[u8])` pairs — instead of element slices; the
  element-taking matrix wrappers (`mul_add_matrix_gfni`,
  `mul_into_matrix_avx512`, and kin) are removed in favour of the `_with`
  provider entries. The GF(2^16) kernel surface carries raw-erased
  payloads: `TowerCoeff` the raw coefficient word and the two broadcast
  words, `TowerTables` the raw word and four nibble tables, both built with
  `new` (representation inferred from the element);
  `internals::kernel::gf16::Prepared` carries those raw forms, with
  `Plain(u16)` for the fallback. The scatter, gather, and matrix entries
  take coefficients through provider forms — an element slice (resolved on
  access), a materialized table slice, or a matrix provider such as
  `FlatMatrix` or `gf16::FlatResolved`. Supported public API paths are
  unchanged.

## [2.1.0] - 2026-10-02

### Added

- Self-contained, runnable examples cover scalar arithmetic across every field,
  parity-shard encoding and erasure recovery, streaming GF(2^16) parity updates,
  bit-packed Hamming syndromes, extension-field embeddings and trace/norm, and
  canonical prime-field evaluation-vector arithmetic.
- `just example NAME [ARGS]` and `just examples [ARGS]` run one tutorial or the
  complete educational set, including portable builds with `alloc`.
- `QuadMersenne31` dispatches `V4x` under `simd512`: `mul_into`, `mul_add`,
  and `mul_elementwise` run AVX-512 kernels, and `backend_for::<QuadMersenne31>()`
  reports `v4x` on those hosts. The remaining operations run the AVX2 kernels.

### Changed

- `BENCHMARKS.md` presents a complete Tiger Lake / Golden Cove snapshot of
  public API and interleaved competitor timings. Measurement history and
  dispatch-threshold experiments are excluded from the public tables.
- Prime-field vector multiplication is faster. `Mersenne31` AVX2 and AVX-512
  multiplies skip re-folding canonical operands, and the AVX-512 product uses a
  masked recombination; `QuadMersenne31` AVX2 complex products defer reduction
  until four partial products are summed. `Goldilocks` AVX2 addition and
  subtraction use fewer compare steps. Outputs remain canonical, and the
  raw-destination guarantees of `add_assign_scalar` and `sub_assign_scalar`
  are unchanged. See `BENCHMARKS.md`.
- `bench-m31-comp` runs only the five-round public snapshot campaign;
  the `prime_ntt` threshold investigation remains a separate benchmark.
- The AVX-512 GF(2^8) matrix kernels (`Gf8D` and `Gf8B`, contiguous,
  scattered-row, provider, and prepared forms) use direct coefficient reads
  for short rows and resolve each row group's coefficients into map words
  once per chunk of terms for longer rows, folding terms in pairs. Results
  are unchanged.
- The GFNI and AVX-512 byte-field matrix kernels, contiguous and
  scattered-row, process long rows in column blocks when the destination
  spans more than one row group, so each source block is reused from cache
  across every row group instead of being re-read from memory per group.
  Results are unchanged.
- The GFNI byte-field matrix kernels prefetch each source ahead of the tile
  loop on long rows whose terms fit in one resolve chunk, and run the column
  blocks above only for larger term counts. Results are unchanged.
- The GFNI byte-field matrix kernels (`Gf8D` and `Gf8B`, contiguous,
  scattered-row, provider, and prepared forms) split the destination rows
  into the fewest width-balanced row groups of up to six rows, or four with
  one or two terms, instead of fixed groups of four, two, and one, so the
  sources are streamed once per group and fewer groups run. Overwrite seeds
  multi-row groups from the first product rather than from zero. Results are
  unchanged. `BENCHMARKS.md` is refreshed on both hosts for this change.
- The GFNI byte-field matrix kernels run a three-row group with three
  32-byte lanes per row, folding terms two at a time, when its terms number
  from six through one resolve chunk and its rows are cache-resident; other
  three-row groups keep two lanes. This covers three-destination calls and
  the trailing group of seven-destination calls, contiguous and
  scattered-row. Results are unchanged.

### Fixed

- The Plonky3 competitor harness aligns packed quadratic-extension fixtures
  for native AVX-512 builds and reports the actual packing widths.
- CI coverage uses `just cover`, including the portable feature run, and
  exports LCOV through `just cover-lcov` with the same exclusion set. The
  AVX-512-only `QuadMersenne31` kernel is excluded on incapable coverage
  runners; the AVX2 kernel and mixed-ISA dispatch remain included.
- NEON and Wasm row-geometry checks compile only on their consumer targets,
  rather than appearing as unreachable x86 coverage obligations.
- The GFNI and AVX-512 GF(2^8) matrix kernels over a generic `Matrix`
  provider check the length of each source slice they stage, so a provider
  that returns a shorter slice than the one the entry validated panics
  instead of reading out of bounds.

## [2.0.0] - 2026-09-30

### Added

- 64-byte AVX-512 kernels for both byte fields (`Gf8B`, `Gf8D`) behind the
  `simd512` feature: single-buffer `mul_add`/`mul_assign`/`mul_into`,
  blocked scatter/gather, blocked matrix (`mul_add`, `mul_into`, prepared
  `CoeffMatrix` forms, and `_at` scattered rows), and elementwise products
  (`mul_elementwise`, `mul_elementwise_assign`). `Gf8D` elementwise products
  conjugate `GF2P8MULB` by the field isomorphism onto `0x11B`; results are
  unchanged. With `simd512` on an AVX-512+GFNI host the new `V4x` tier
  resolves, both byte fields dispatch every operation there, and
  `has_vector_elementwise` reports `true` for both. Without `simd512` the
  tier never resolves and dispatch is unchanged.
- `Gf16` dispatches AVX-512 GFNI single-row multiplication, elementwise
  products, and register-blocked matrix accumulation (including prepared
  matrices) on `V4x`. Scatter and gather retain their existing GFNI kernels;
  non-`V4x` backends are unchanged. The new kernels use safe Archmage
  reference-based loads and stores.
- On `V4x`, misaligned scatter, gather, and matrix rows peel their heads to
  a 64-byte boundary above per-shape length floors set by measurement.
  `Gf16` `mul_add`
  destinations peel the same way above their own measured floor, and rows
  shorter than one 64-byte lane dispatch straight to the 32-byte GFNI
  kernel.
- On `V4x`, byte XOR — `add_assign` and `sub_assign` for every binary field,
  `bits::xor_assign`, and the other bit-packed XOR paths — runs 64-byte
  AVX-512 lanes from a measured length floor, with a 64-byte destination
  alignment peel on longer buffers. Shorter buffers keep the AVX2 kernel but
  pass one extra length test.
- On `V4x`, `bits::weight` counts whole words with `VPOPCNTQ`.
- `Mersenne31` and `Goldilocks` gain 64-byte AVX-512 kernels for every
  operation that had an AVX2 kernel, dispatched on `V4x`. Results are
  unchanged; `backend_for` reports `V4x` for both fields on that tier.

### Changed

- The prime-field competitor harness builds `fgf` with `simd512`, so a
  Tiger Lake run measures the `V4x` tier it resolves; the family recipes
  pass the feature. Tiger Lake prime columns recorded before this change
  measured `fgf` at `v3_gfni_crypto`.
- The `Gf16` AVX-512 matrix kernels broadcast each coefficient pair once
  per term block instead of once per destination lane.

- The binary-field competitor harnesses are one three-arm binary,
  `external/bench-trio/`, replacing `bench-isal` and `bench-klauspost`: it
  interleaves `fgf`, Intel ISA-L, and klauspost/reedsolomon over one fixture
  set, rotating through all six arm orders, and `just bench-gf-comp` runs it
  beside the self suite. `BENCHMARKS.md` carries the single-matrix record.
- **Breaking (unstable surface only):** the direct kernels under
  `fgf::internals::kernel::{x86,aarch64,wasm32}` are laid out by field
  family, then instruction set. Every entry is named operation, instruction
  set, field marker, then `_with`, and its instruction-set suffix matches
  its file. Supported public API paths are unchanged. Migration:
  - `x86::prime::*_gld_{sse42,avx2,avx512}` → `x86::goldilocks::*_{sse42,avx2,avx512}`,
    `*_m31_*` → `x86::mersenne31::*`, and `*_qm31_avx2` →
    `x86::quad_mersenne31::*_avx2` (for example `mul_add_gld_avx2` →
    `goldilocks::mul_add_avx2`).
  - `x86::bytes512::{xor512, count_ones512}` → `x86::bytes::{xor_avx512,
    count_ones_avx512}`.
  - `x86::fan_paar::{mul_add,mul_assign,mul_into}_{avx2,ssse3}` (GF(2^16)) →
    `{mul_add,mul_assign,mul_into}_{avx2,ssse3}_fp16`, and
    `*_{fp32,fp64}_avx2` → `*_avx2_{fp32,fp64}`.
  - `x86::gf16::{mul_add512, mul_assign512, mul_into512, mul_elementwise512,
    mul_elementwise_assign512, mul_add_matrix512, mul_add_matrix512_with}` →
    `mul_add_avx512`, `mul_assign_avx512`, `mul_into_avx512`,
    `mul_elementwise_avx512`, `mul_elementwise_assign_avx512`,
    `mul_add_matrix_avx512`, and `mul_add_matrix_avx512_with`.
  - `x86::gf8` 64-byte entries replace `affine512`/`gfni512` with `avx512`:
    `mul_{add,assign,into}_affine512`, `mul_add_{scatter,gather}_affine512`,
    `mul_{add,into}_matrix_affine512{,_with}`, `mul_add_matrix_at_affine512`,
    and `mul_elementwise{,_assign}_gfni512` become
    `mul_{add,assign,into}_avx512`, `mul_add_{scatter,gather}_avx512`,
    `mul_{add,into}_matrix_avx512{,_with}`, `mul_add_matrix_at_avx512`, and
    `mul_elementwise{,_assign}_avx512`.
  - `x86::gf8` `Gf8D` entries mark the field with `_8d` instead of naming the
    multiply strategy: `affine` becomes `gfni_8d` (`mul_add_affine` →
    `mul_add_gfni_8d`, `mul_add_matrix_at_affine` →
    `mul_add_matrix_at_gfni_8d`, `mul_add_matrix_affine_with` →
    `mul_add_matrix_gfni_8d_with`, and likewise for the `mul_assign`,
    `mul_into`, scatter, gather, and matrix entries);
    `mul_elementwise{,_assign}_iso_8d` → `mul_elementwise{,_assign}_gfni_8d`;
    `mul_elementwise{,_assign}_iso512_8d` →
    `mul_elementwise{,_assign}_avx512_8d`; and
    `mul_{add,into}_matrix_affine512_prepared_with` →
    `mul_{add,into}_matrix_avx512_8d_with`.
  - `x86::gf8::multiply_vectors_sse` → `x86::gf8::multiply_vectors_ssse3`.
  - `aarch64::{gf8,gf16}` and `wasm32::{gf8,gf16}` name the operation in
    full: `{scatter,gather,matrix,elementwise}_{neon,simd128}` →
    `mul_add_scatter_*`, `mul_add_gather_*`, `mul_add_matrix_*`, and
    `mul_elementwise_*`; `aarch64::gf8::elementwise_pmull` →
    `mul_elementwise_neon_aes`, named for the tier its token proves.
  - Every other entry keeps its path.

**Breaking:** `simdispatch` moves `=0.1.0` to `=0.2.0`. `fgf::Backend` is the
upstream type re-exported, so crates pinning both must move together; pin
`simdispatch = "=0.2.0"` alongside this `fgf` (no API names change; the
ladder gains `V4x`/`V4`).

### Removed

- The deferred `internals` entry `x86::avx512::proven::xor` and its
  unexported body. The dispatched `x86::bytes::xor_avx512` replaces it and
  carries no unsafe code.
- **Breaking (unstable surface only):** the rest of the deferred
  `internals` module `x86::avx512` (`gf8_*` and `gf16_*` single-buffer,
  scatter, gather, matrix, and elementwise entries, and `proven::*`). The
  dispatched `x86::gf8` and `x86::gf16` AVX-512 entries cover every shape
  it held.
- **Breaking (unstable surface only):** the `internals` measurement variants
  whose decisions are recorded: `x86::gf8::{mul_add_gather_affine_8b,
  prepare_affine_8b, Affine8BFactor, mul_add_gather_gfni_axpy_tail,
  mul_add_gather_gfni_tile, mul_add_gather_gfni_split,
  mul_into_matrix_chunk_8d, mul_into_matrix_external_grouped_8d,
  mul_into_matrix1_external_8d, mul_into_matrix1_fullinit_8d,
  mul_into_matrix1_chunk_8d, mul_into_matrix1_external_chunk_8d,
  mul_into_matrix1_tile_8d, mul_into_matrix2_8b, mul_into_matrix2_8d,
  mul_into_matrix2_shuffle_packed, mul_into_matrix6_shuffle_8b,
  mul_into_matrix6_shuffle_8d, mul_into_matrix6_shuffle_packed_8b,
  mul_into_matrix6_shuffle_packed_8d, resolve_probe_8d}`, the AVX2
  prepared-affine matrix `x86::gf8::{mul_add,mul_into}_matrix_affine_prepared_with`,
  the AVX2 blocked `Gf16` gather and matrix
  `x86::gf16::{mul_add_gather_avx2, mul_add_matrix_avx2,
  mul_add_matrix_avx2_with}` (dispatch serves both shapes with repeated
  AXPY), and the row-interleaved XOR `x86::bytes::{xor_rows_avx2,
  xor_rows_sse2}`. The `affine` and `dot_product` benchmarks and the
  `perf_p2` example that drove them are removed with them.
- **Breaking (unstable surface only):** `x86::gf8::{mul_add_matrix_at_gfni_with,
  mul_add_matrix_at_affine_with}`. No operation hands a prepared provider to
  scattered rows; a prepared scattered-row operation over `CoeffMatrix` was
  measured against the raw-term `mul_add_matrix_at` and not adopted.
  The raw-coefficient `_at` entries remain.

### Fixed

- `ops::mul_add_matrix_at` returned wrong results for `Mersenne31`,
  `Goldilocks`, and `QuadMersenne31` whenever a coefficient was one: the
  per-row fallback took the binary fields' XOR shortcut instead of modular
  addition. Every field without a blocked scattered kernel (`Gf16`, `Gf32`,
  `Gf64`, the Fan–Paar fields, the prime fields, and both byte fields on
  shuffle, `AArch64`, and Wasm backends) now runs one dispatched `mul_add`
  per row, which fixes the prime-field results and replaces the element-wise
  scalar loop with the field's vector kernel.
- The 32-byte GFNI and AVX2 kernels peel half-lane destinations: byte XOR,
  both byte fields' `mul_add`/`mul_assign`/`mul_into`, and the `Gf16`
  single-row kernels pay one 16-byte head when the destination sits 16
  bytes into a 32-byte lane, so their stores stop splitting cache lines
  above the shared measured floor. The AVX-512 `Gf16` `mul_into` gains the
  destination peel its `mul_add` already had.
- Prepared overwrite gathers route through the register-blocked one-row
  overwrite matrix where the change covers them: `Gf8D`'s prepared form
  fell back to the per-source AXPY loop on `V3GfniCrypto`, and `Gf8B`'s
  overwrite gather filled the destination and accumulated on `V4x` instead
  of overwriting in one pass.
- Half-lane-skewed GFNI in-place scaling and short blocked scatter avoid
  repeated split-line destination accesses. The AVX-512 scatter path avoids
  peeling when a quarter-line row pitch cannot align its row group.
- The AVX-512 peel-floor benchmark now compares the same offsets with and
  without peeling. The matched comparison supports the existing gather and
  matrix floors.

## [1.2.2] - 2026-09-30

### Changed

- `ops::add_assign_scalar` and `ops::sub_assign_scalar` dispatch the
  field-independent vector broadcast-XOR kernels for every binary field;
  `Gf32`, `Gf64`, and the Fan–Paar fields previously ran the portable
  byte-at-a-time loop. Results are unchanged.
- **Breaking (unstable surface only):** `internals`
  `x86::bytes::xor_broadcast_sse42` → `x86::bytes::xor_broadcast_sse2`,
  which takes an `X64V1Token`: the kernel uses only SSE2.

### Fixed

- `ops::mul_add_matrix_at` returned wrong results for `Mersenne31`,
  `Goldilocks`, and `QuadMersenne31` whenever a coefficient was one: the
  per-row fallback took the binary fields' XOR shortcut instead of modular
  addition. Every field without a blocked scattered kernel (`Gf16`, `Gf32`,
  `Gf64`, the Fan–Paar fields, the prime fields, and both byte fields on
  shuffle, `AArch64`, and Wasm backends) now runs one dispatched `mul_add`
  per row, which fixes the prime-field results and replaces the element-wise
  scalar loop with the field's vector kernel.

## [1.2.1] - 2026-09-24

### Fixed

- `ops::add_assign_scalar` and `ops::sub_assign_scalar` for the binary fields
  (`Gf8B`, `Gf8D`, `Gf16`) fell to the byte-at-a-time portable broadcast loop
  on every backend, while the prime fields dispatched to vector kernels.
  They now dispatch to field-independent vector broadcast-XOR kernels —
  AVX2/SSE4.2 on x86, NEON on AArch64, `simd128` on wasm32 — at the same
  tiers the fields' other byte kernels run; scalar fallbacks are unchanged.
  Measured throughput appears in `BENCHMARKS.md`, "Packed operations".

## [1.2.0] - 2026-09-24

### Added

- `ops::mul_elementwise_assign`: `dst[i] *= src[i]`, the in-place two-slice
  form of `mul_elementwise`. Existing three-slice callers that wrote the
  product over their first operand had to stage a second buffer; this
  spelling updates the accumulator in place.
- `ops::add_assign_scalar` and `ops::sub_assign_scalar`: add or subtract one
  field element broadcast across every lane. Consumers composing
  lane-parallel Horner steps no longer fill a broadcast buffer and re-read it
  through `add_assign`.
- Field-family benchmark recipes `bench-gf`, `bench-gdl`, `bench-m31`, and
  `bench-gf2`, each with a `-comp` variant adding every competitor harness
  wired for the family, and `--gf`, `--gdl`, `--m31`, and `--gf2` panel
  selection flags on the `kernels` bench target. A recorded campaign for one
  field family no longer re-runs every panel.
- Prime-field competitor benchmarking against Plonky3:
  `external/prime-bench/` compares `fgf` against `p3-goldilocks`,
  `p3-mersenne-31`, and its packed quadratic extension over the recorded
  Mersenne31, Goldilocks, and QuadMersenne31 shapes on each library's
  native layout, and `bench-gdl-comp` / `bench-m31-comp` interleave it with
  the self suite. `BENCHMARKS.md` carries the prime-field competitor
  matrix.

### Changed

- The direct `aarch64` and wasm32 `simd128` kernel entries under the
  `internals` feature are now safe
  [`archmage`](https://docs.rs/archmage) capability-token functions: each
  takes the exact token its instructions require (`NeonToken`,
  `NeonAesToken`, or `Wasm128Token`) as its first argument and validates its
  own geometry, replacing the `proven` token-proven compatibility facades,
  which are gone. The wasm32 subtree holds no unsafe code, and the
  module-level `#![allow(unsafe_code)]` over it is removed. `internals`
  carries no compatibility promise; entries may change or disappear in any
  release.
- The competitor harnesses now live in `external/`, one unpublished package
  per harness with its own build script (`bench-isal`, `bench-klauspost`,
  `prime-bench`). The `bench-isal` and `bench-klauspost` recipes
  are gone: `just bench-gf-comp` builds and runs both wired harnesses at the
  record's repetition.
- The `bench-<field>-comp` recipes run five complete rounds interleaving the
  family's self suite with every wired competitor harness, with the unit
  order shuffled within each round and printed into the run log, instead of
  one self pass followed by sequential harness runs.
- Benchmark timing carries `f64` nanosecond samples end to end instead of
  `Duration` values divided by the repetition count, which quantized every
  per-iteration estimate to whole nanoseconds; small-shape estimates and
  their displayed times keep sub-nanosecond precision.
- Campaigns now keep full-statistics records — run-level observations and
  spread statistics for every category — in the git-ignored
  `bench-records/` directory alongside the published `BENCHMARKS.md`
  medians.
- The Miri suite is split by what it can execute. The scalar run keeps one
  representative per memory-access shape — contract-sensitive coefficients,
  boundary lengths, and the empty, single, and group-plus-one row counts —
  with the full sweeps continuing in ordinary tests; the Goldilocks leg runs
  in ordinary tests only. A new `unsafe-check-gfni` recipe interprets the
  dispatched GFNI kernels under Miri with the tier promoted at compile time,
  covering the offset-addressed-row residue the scalar run cannot reach.
  Non-temporal stores stay proof-review-only. The residue ledger in
  `AGENTS.md` now enumerates the `x86` unsafe items alongside the `aarch64`
  ones.

### Removed

- The Plonky2 arm of the prime-field comparison: `plonky2_field` carries no
  packed field kernels worth timing, so `external/prime-bench/` compares
  against Plonky3 alone.
- The `reed-solomon-erasure` comparison from the benchmark suite: its
  development dependency, its arms in `benches/compare.rs`, and its column in
  `BENCHMARKS.md`. The ISA-L and `klauspost/reedsolomon` harnesses cover the
  same shapes with stronger aggregation; `compare` now reports `fgf`
  self-numbers over the competitor fixture family only.
- The `external-bench` microbenchmark lab: the gf-complete, FLINT, NTL, and
  Leopard comparators, their build artifacts, and the `bench-prime-ntt`
  recipe with its `external-bench/prime-ntt/` NTT consumer package. The
  retired comparators do not reach the tier of the surviving harnesses, and
  the NTT consumer measured `butterfly-fft` rather than `fgf`.
- The `isal-fgf` ISA-L attribution harness. The wired `bench-isal` harness
  covers the comparison role with its paired protocol and validation gate,
  and the internal attribution questions its panels priced are concluded.

## [1.1.1] - 2026-09-18

### Changed

- **Breaking (unstable surface only):** the `internals` feature is now a
  re-export-only facade at `fgf::internals`; implementation modules live in
  their private homes and are grouped under
  `fgf::internals::kernel::{gf8, gf16, gf32, gf64, tower, fan_paar, prime,
  scalar, tables}` (plus architecture modules under `simd` on their targets),
  with `fgf::internals::kernel::{xor, Matrix, FlatMatrix}`, the capability
  tokens, and `fgf::internals::field::wiedemann`. The old unstable
  `fgf::kernel::{…}` and `fgf::field::wiedemann` paths no longer resolve
  without the feature, and the feature no longer activates `archmage` (it is
  visibility-only: `internals = []`). Migration: `fgf::kernel::x86::gf8::…`
  becomes `fgf::internals::kernel::x86::gf8::…` (same shape for every other
  kernel module), and `fgf::field::wiedemann` becomes
  `fgf::internals::field::wiedemann`. Bench, example, and test targets that
  reach the facade now declare it in `required-features`; `tests/zero_alloc.rs`
  and `tests/goldilocks_wrap.rs` keep their public tests ungated, with the
  direct-kernel cases moved to `zero_alloc_internals` and
  `goldilocks_wrap_internals` targets. Supported public API paths are
  unchanged.

## [1.1.0] - 2026-09-16

### Fixed

- The AVX2 Goldilocks kernels computed wrong values whenever a product's low
  64 bits fell below its high 32: the split-fold reduction corrected a
  wrapped subtraction by adding where it must subtract, so every such lane
  shipped a canonical-looking result off by exactly `2^32 − 1`. Random
  operands hit the branch with probability ≈ 2⁻³³ per lane, but operands
  concentrated on powers of two — twiddle tables in transform consumers —
  hit it at ≈ 2⁻⁷, which is how the defect reached NTT workloads while
  every random-value differential passed. The SSE4.2 kernels were correct.
  A wrap-boundary regression against an independent `u128 % p` oracle now
  pins both widths; the operand family is `tests/goldilocks_wrap.rs`.

### Added

- AVX2 kernels for `QuadMersenne31` on `V3`/`V3GfniCrypto` hosts, four
  complex elements per vector over the Mersenne31 lane discipline, with a
  complex-by-coefficient multiply costing two limb multiplies. Production
  dispatch takes over at one full vector for the multiplies and at two
  vectors for add and sub.
- `vector_elementwise_min_bytes::<F>()`: the row length at which `F`'s
  elementwise multiplication dispatches to a vector kernel on this host
  (zero when no threshold applies), so schedule-selecting consumers route
  sub-vector rows to a scalar schedule instead of paying vector entry
  overhead over the scalar body.

- `bench-klauspost/`, an in-process competitor comparison against
  `klauspost/reedsolomon` v1.14.2 over GF(2^8)/`0x11D`, run with
  `FEC_GOLDEN_CORE=<cpu> just bench-klauspost`. The package links the Go
  library as a C archive, interleaves it with the matched `fgf` operations
  over one fixture set, and validates every arm byte-for-byte before any
  timing. It replaces `catid/leopard` as the named codec-level competitor.

### Changed

- The x86 fused overwrite kernels prefetch their destination. `mul_into`
  writes lines it never reads, so each ordinary store waited on the
  read-for-ownership fetch of the line it was about to replace whole.
  Naming those lines ahead of the cursor lifts single-source GF(2^8) and
  GF(2^16) overwrite throughput by about a third from 24 KiB up, and brings
  GF(2^8) to parity with Intel ISA-L's streaming `gf_vect_mul` without
  evicting the destination. Results are unchanged; buffers below 24 KiB and
  the non-temporal bodies above `NT_STORE_MIN` are untouched.

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
  memory-level parallelism.
- `bits::XorRange` and `bits::xor_range_with`: the prepared form of
  `bits::xor_range`. The bit range's byte window and end masks are derived
  once and applied to many buffer pairs, the same prepare/apply split `ops`
  uses for coefficients; buffers of different lengths are accepted as long
  as each covers the planned window. `xor_range` itself now routes through
  the shared byte-window core, which replaces the word-assembly path for
  sub-word ranges (one or two masked byte operations) and the masked end
  words of longer ranges.
- `internals` entries for the crossed resolution x chunk panel:
  `matrix_overwrite1_fullinit_8d` (the production one-row path with the
  original full-array scratch policy), `matrix_overwrite1_chunk_8d` and
  `matrix_overwrite_chunk_8d` (resolved path at chunk 32/64/96, one row and
  grouped), `matrix_overwrite1_external_chunk_8d` and
  `matrix_overwrite_external_grouped_8d` (the production body over externally
  resolved maps, chunked or single-pass). Benchmark evidence only; production
  dispatch is unchanged.
- `internals` entries for the plan-hoisting experiment:
  `matrix_affine_prepared_with` and `matrix_overwrite_affine_prepared_with`
  run the production row-group bodies over `Prepared8D` coefficients (the
  affine word read from the prepared form). Measured and rejected for the
  plan entries — parity-to-worse against the term-list baseline on both
  hosts, closing none of the gap to the resolution-free body — so production
  still resolves from scalar values.
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
  paths.

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
  output rows, with no shape regressing on either host.
  Coefficient counts above 32 terms per call fold in several passes over
  the destination, so a caller that folds hundreds of sources into one call
  sees the destination read once per 32-term pass.
- The GFNI GF(2^8) resolve-chunk scratch is written only for the occupied
  chunk terms (`MaybeUninit` staging) instead of declaring the full arrays
  initialized. Measured kernel-neutral on both GFNI hosts — the full-init
  cost the old resolve probe showed was the probe's own artifact — but the
  probe now mirrors production, and `RESOLVE_CHUNK` stays 32: chunk 64/96
  fail the both-host noninferiority gate at 65 sources. No observable
  behaviour change.
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
  `internals` is a compatibility promise. The remaining measurement bodies —
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
  without staging a slice per source.

- `Gf8D`'s N-to-1 gather takes the source-fused body on short rows, the
  same selection rule `Gf8B`'s `GF2P8MULB` gather has used since the
  short-row measurement. It was wired to the unfused specialization when
  the blocked-affine seam landed, so rows below the 128-byte main tile fell
  through to one single-source AXPY per source. Results are unchanged;
  `Gf8B` is untouched.

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
behaviour.

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

[Unreleased]: https://github.com/nanithefkuc/fgf/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/nanithefkuc/fgf/compare/v2.1.0...v3.0.0
[1.0.0]: https://github.com/nanithefkuc/fgf/compare/v0.7.1...v1.0.0
[0.7.1]: https://github.com/nanithefkuc/fgf/compare/v0.7.0...v0.7.1
[0.6.0]: https://github.com/nanithefkuc/fgf/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/nanithefkuc/fgf/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/nanithefkuc/fgf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/nanithefkuc/fgf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nanithefkuc/fgf/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nanithefkuc/fgf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nanithefkuc/fgf/releases/tag/v0.1.0

[1.1.0]: https://github.com/nanithefkuc/fgf/compare/v1.0.1...v1.1.0
[1.1.1]: https://github.com/nanithefkuc/fgf/compare/v1.1.0...v1.1.1
[1.2.0]: https://github.com/nanithefkuc/fgf/compare/v1.1.1...v1.2.0
[1.2.1]: https://github.com/nanithefkuc/fgf/compare/v1.2.0...v1.2.1
[1.2.2]: https://github.com/nanithefkuc/fgf/compare/v1.2.1...v1.2.2
[2.0.0]: https://github.com/nanithefkuc/fgf/compare/v1.2.2...v2.0.0
[2.1.0]: https://github.com/nanithefkuc/fgf/compare/v2.0.0...v2.1.0
