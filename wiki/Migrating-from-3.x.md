# Migrating from 3.x to 4.0

`fgf` 4.0 is a breaking release. Public modules are facades with one path
per item, trait plumbing is sealed, `FieldBuffer` is the sole byte-operation
bound, and map constructors follow the crate's error classification. The
[changelog](https://github.com/nanithefkuc/fgf/blob/main/CHANGELOG.md) under
`Unreleased` records each change with its migration note; this guide
collects them in one place.

## Paths

Every stable item has exactly one public path. The removed modules are
`fgf::field` and every submodule under it, `fgf::kernel`, `fgf::bits`,
`fgf::embedding`, `fgf::cantor`, `fgf::normal`, `fgf::poly`, `fgf::tower`,
`fgf::description`, `fgf::goldilocks`, `fgf::mersenne31`, and
`fgf::quad_mersenne31`. `fgf::extension` remains as a facade. Destinations
are selected by item, not by replacing a whole module prefix.

| 3.x items and paths | 4.0 path |
| --- | --- |
| `FiniteField`, `FiniteFieldRepr`, `Elem`, `Field`, `FieldElem`, `FieldBuffer`, `HasGenerator`, `RangeError` under `fgf::field` or the root | `fgf::{…}` |
| `Backend`, `backend`, `backend_for` under `fgf::kernel` | `fgf::{…}` |
| `Mersenne31`, `MersenneFold`, `Goldilocks`, `GoldilocksFold`, `Residue`, `PrimeField`, `PrimeIdentity`, `PrimeCharacteristic`, `is_prime` | `fgf::prime::{…}` |
| `Polynomial`, `AES`, `RS`, `Normal`, `Cantor`, `BinaryRepr`, `BinaryField`, `BinaryDegree` | `fgf::binary::{…}` |
| `Rijndael16`, `Rijndael32`, `Rijndael64`, `FanPaar2`, `FanPaar4`, `FanPaar8`, `FanPaar16`, `FanPaar32`, `FanPaar64` | `fgf::binary::{…}` |
| `Extension`, `ExtensionSpec`, `ExtensionField`, `ExtensionStorage`, `Relation`, `Components`, `Packed`, `PackedPadded`, `PackedWord` | `fgf::extension::{…}` |
| `QuadMersenne31`, `ComplexMersenne31` | `fgf::extension::{…}` |
| `Embedding`, `Isomorphism`, `Postcompose` | `fgf::maps::{…}` |
| `fgf::bits::{…}` | `fgf::ops::bits::{…}` |

Apply the table to every former export of its listed items, including
aliases under `fgf::field` and its submodules. Existing items already at
their listed destination retain that path.

## Deleted items

- `MODULUS` under `goldilocks`, `mersenne31`, and `quad_mersenne31` (and
  their `fgf::field` aliases). Use
  `<<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC`.
- `FieldKernels`. Use `F: FieldBuffer` as the sole bound for byte-lane
  operations and prepared coefficient types. Arbitrary downstream generic
  code needs no further bound, including generic `R: BinaryRepr<8>`, a
  custom `ExtensionSpec` over `Components` storage, and every supported
  `Packed<W>` storage.
- The prepared field maps: `Frobenius`, `PreparedEmbedding`,
  `PreparedIsomorphism`, `Elem::frobenius_with`, the nine `ops::map` row
  functions with their `ops` re-exports, and the five `ops::map_bits`
  row functions over packed GF(2) bits. Call the per-element `Embedding`
  and `Isomorphism` methods (`embed`, `contains`, `restrict`, `frobenius`,
  `trace`, `norm`, `apply`, `unapply`) in a loop, or a compact map method
  inside a decode/encode lane loop. `Embedding::frobenius` replaces the
  relative Frobenius `x^|S|`; an absolute power `Frobenius::new(k)` or
  `frobenius_with` becomes `x.pow(p^(k mod N))`, or `k` applications of
  the prime-subfield embedding's `frobenius`. The restriction rows'
  validity bitmap becomes the `Option` that `restrict` returns per element.
  The GF(2) bit/lane interop kernels behind `ops::map_bits` go with them;
  the bit-buffer operations in `bits` are unchanged.
- `EmbeddingError`, replaced by `fgf::maps::ImageError` (see Errors).
- `CoeffMatrix::values()`, replaced by `to_source_major()` (see
  Signatures).

Kernel internals move behind the `internals` feature: `KernelBackend`,
`FGF_TIERS`, `has_vector_elementwise`, `vector_elementwise_min_bytes`, the
`ParseBackendError` re-export, `BinaryDescription`, `ByteLogExp`,
`tower::{fp_mul_alpha, fp_multiply, fp_square, fp_invert}`,
`field::prime::{reduce, reduce_wide}`, `mersenne31::reduce`, and
`goldilocks::{reduce, reduce_wide}`. They carry no compatibility
guarantees. `has_vector_elementwise` and `vector_elementwise_min_bytes`
have no stable replacement; the backend query is
`fgf::backend_for::<F>()`.

## Sealed trait items

Raw arithmetic and validation plumbing moves behind crate-private
supertraits. `Field` no longer carries `canonical_raw`, `add_raw`,
`sub_raw`, `neg_raw`, `mul_raw`, `inv_raw`, `ZERO_RAW`, `ONE_RAW`, or
`VALID`; `HasGenerator` no longer carries `GENERATOR_RAW`; `BinaryRepr`
no longer carries `DESCRIPTION` or `LOG_EXP`; `BinaryField` no longer
carries `DESCRIPTION`; `FiniteFieldRepr` no longer carries `VALID`;
`PrimeIdentity` no longer carries `VALID`; and `Postcompose` no longer
carries its `postcompose` callback. Use `Elem::from_raw` and
`Elem::to_raw` for raw-word conversion, `FieldElem` operations for scalar
arithmetic, and `Elem::ZERO`, `Elem::ONE`, and `Elem::GENERATOR` for
identities and the generator. Map application uses `Embedding::embed` or
`Isomorphism::apply`; composition uses `Embedding::then` or
`Isomorphism::then`. `VALID` has no replacement: element constants and
constructors evaluate validation. `DESCRIPTION` and `LOG_EXP` have no
stable replacement. `PrimeIdentity` no longer exposes `C::Row`; generic
callers take an explicit row parameter such as
`F: PrimeField<Characteristic = C>`, or name the intended concrete row.

## Signatures

- `ops::bits::xor_range(dst, src, bits, from, to)` becomes
  `xor_range(dst, bits, from, to, src)`.
- `ops::bits::xor_range_with(dst, src, range)` becomes
  `xor_range_with(dst, range, src)`.
- `ops::add_gather_offsets(dst, region, offsets)` becomes
  `add_gather_offsets(dst, offsets, region)`.
- `ops::mul_elementwise` is `ops::mul_elementwise_into` and
  `ops::div_elementwise` is `ops::div_elementwise_into`. Arguments and
  behavior are unchanged; panic messages name the new operation.
- `CoeffMatrix::values()` is replaced by `to_source_major()`, which
  returns the source-major coefficients as a `Vec` and pairs with
  `from_source_major`. `matrix.values().collect::<Vec<_>>()` becomes
  `matrix.to_source_major()`; `get_at` and `source` read single
  coefficients without allocating.
- `backend_for<F>()` requires `F: FieldBuffer` and reports a
  field-strategy summary, not the serving tier of every individual
  operation.

## Bounds

Every `ops` function bounded `F: FieldKernels` in 3.0.2 is bounded
`F: FieldBuffer` in 4.0. `Field::Raw` is `Send + Sync` as well as
`Copy + Eq + Ord + Hash + Debug + 'static`. `Field` keeps `Raw`,
`Characteristic`, `NAME`, `DEGREE`, and `ORDER`; `FiniteFieldRepr` keeps
`NAME`.

Byte operations, packing, prepared-coefficient constructors, and
`backend_for` evaluate the row's central const validation before any
byte work, so an invalid row — a reducible polynomial, a non-bijective
basis, a composite characteristic, a reducible relation — is a compile
error even when no element is constructed and every buffer is empty.

## Errors

- `Embedding::new` and `Isomorphism::new` return `Self`. A pair outside
  the degree contract — equal degrees for `Embedding`, a source degree
  that does not divide the target degree, unequal degrees for
  `Isomorphism` — is a compile-time assertion. Drop the `Result`
  handling on `new`, move impossible pairs out of the codebase, and
  obtain noncanonical maps from `from_images` with validated images.
  Both constructors of both maps now evaluate the source and target
  rows' field validation, so a map over an invalid row no longer
  compiles. A pair inside the contracts that no structural or ladder
  rule covers panics under `# Panics`, naming the constructor and the
  source and target presentation types.
- `fgf::maps::ImageError` replaces `EmbeddingError` as a
  `#[non_exhaustive]` unit covering exactly the rejection of a supplied
  generator-image sequence (wrong count, or a failed transported
  relation), which `Embedding::from_images` and
  `Isomorphism::from_images` return. `from_images` no longer requires or
  searches for a canonical map. `RangeError` is `#[non_exhaustive]`.
  Neither error type can be constructed outside the crate: obtain the
  value from a rejected conversion or image sequence, match it with
  `Err(RangeError { .. })` or `Err(ImageError { .. })`, and replace
  variant matches on `EmbeddingError` with the single `ImageError` arm.
  Degree mismatches no longer reach runtime code.
- Runtime precondition panics of the row operations, field byte codecs,
  `Relation` constructors, and `Embedding::generator` name the operation
  and the offending operand. The checks, their order, and the accepted
  inputs are unchanged; tests matching panic text may need the new
  operation-prefixed fragments.

## Behavior

`ops::mul_add_matrix` and `ops::mul_add_matrix_at` return after
validation when every coefficient is zero, leaving the destination bytes
untouched, noncanonical lanes included. `ops::mul_into_scatter` copies
the source bytes verbatim into rows with a unit coefficient, as
`ops::mul_into` does; Mersenne31, Goldilocks, QuadMersenne31, and
residue rows previously canonicalized noncanonical source lanes.
