# Migrating from 2.x to 3.0.1

Upgrading to fgf 3.0.1 replaces field-specific element wrappers with `Elem<F>`. Scalar arithmetic, byte encoding, and generator selection have separate trait bounds, and prime elements store canonical values rather than retaining noncanonical input words.

This guide goes directly from 2.x to 3.0.1. The [changelog](https://github.com/nanithefkuc/fgf/blob/v3.0.1/CHANGELOG.md) records the release changes. Historical 2.x examples are fenced as text; every Rust example uses the 3.0.1 API.

## Replace field and element spellings together

| Released 2.x field | 3.0.1 field |
| --- | --- |
| `Gf8B` | `FiniteField<2, 8, Polynomial<AES>>` |
| `Gf8D` | `FiniteField<2, 8, Polynomial<RS>>` |
| `Gf16` | `FiniteField<2, 16, Extension<Rijndael16, 2>>` |
| `Gf32` | `FiniteField<2, 32, Extension<Rijndael32, 2>>` |
| `Gf64` | `FiniteField<2, 64, Extension<Rijndael64, 2>>` |
| `FanPaarN` field marker | `FiniteField<2, N, Extension<FanPaarN, 2>>`, where `FanPaarN` is a specification marker |
| `Gf2` | `FiniteField<2, 1, Polynomial<3>>` |
| `Mersenne31`, `Goldilocks` | Same names, now type aliases of `FiniteField<0x7FFF_FFFF, 1, MersenneFold>` and `FiniteField<0xFFFF_FFFF_0000_0001, 1, GoldilocksFold>`; usable as types, not as values |
| `QuadMersenne31` | Same name, a type alias of `FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>`; usable as a type, not as a value |

Every element type becomes `Elem<the_field>`. For example, `gf16::Elem` becomes `Elem<FiniteField<2, 16, Extension<Rijndael16, 2>>>`, and `mersenne31::Elem` becomes `Elem<Mersenne31>`. The crate supplies no compatibility field aliases. Applications can define their own aliases to keep signatures readable.

`QuadMersenne31` limb pairs are arrays: the two-argument `quad_mersenne31::Elem::from_raw` becomes `Elem::<QuadMersenne31>::from_raw`, which takes one `[re, im]` array; `from_components` takes `[re, im]`, and `to_raw` and `to_components` return `[re, im]` instead of a tuple.

A historical 2.x scalar computation:

```text
use fgf::gf16;
const A: gf16::Elem = gf16::Elem::from_raw(0x1234);
const B: gf16::Elem = gf16::Elem::from_raw(0x0108);
const PRODUCT: gf16::Elem = A.mul(B);
```

The replacement in 3.0.1:

```rust
use fgf::{Elem, Extension, FiniteField, Rijndael16};

type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
const A: Elem<F> = Elem::<F>::from_raw(0x1234);
const B: Elem<F> = Elem::<F>::from_raw(0x0108);
const PRODUCT: Elem<F> = A.mul(B);
assert_eq!(PRODUCT.div(B), A);
```

Import polynomial constants before using them in the const-generic position, such as `Polynomial<AES>`. The full polynomial type is `Polynomial<P>` with `P: u128`. `RS` keeps the `0x11D` convention.

## Update generic trait bounds

| Removed associated item or trait name | Replacement |
| --- | --- |
| `F::Elem` | `Elem<F>` |
| `Field::BITS` | `FieldBuffer::STORAGE_BITS` for lane width; `Field::DEGREE` for extension degree |
| `Field::CHARACTERISTIC` | `<F::Characteristic as PrimeIdentity>::CHARACTERISTIC` |
| `Field::GENERATOR` | `Elem::<F>::GENERATOR` with `F: HasGenerator` |
| `Field::BYTES`, `encode`, `decode` | `FieldBuffer` capability |
| `field::Elem` trait | `FieldElem`; import the trait separately from the `Elem<F>` wrapper |
| `Field::elem_count(bytes)` | Validate whole lanes, then divide by `F::BYTES` under `FieldBuffer` |

A scalar-only generic algorithm uses `F: Field`. Encoding uses `F: FieldBuffer`; packed arithmetic uses `F: FieldKernels`, which implies `FieldBuffer`. Generator selection is optional rather than a universal field property.

Historical 2.x coefficient signature:

```text
fn scale<F: fgf::FieldKernels>(dst: &mut [u8], c: F::Elem, src: &[u8]) {
    fgf::ops::mul_into::<F>(dst, c, src);
}
```

The replacement keeps the operation but changes the value type:

```rust
use fgf::{AES, FiniteField, Elem, FieldKernels, Polynomial, ops};

fn scale<F: FieldKernels>(dst: &mut [u8], c: Elem<F>, src: &[u8]) {
    ops::mul_into::<F>(dst, c, src);
}

type F = FiniteField<2, 8, Polynomial<AES>>;
let mut destination = [0xa5u8; 3];
scale::<F>(&mut destination, Elem::<F>::from_raw(2), &[1, 2, 3]);
assert_eq!(destination, [2, 4, 6]);
```

`pow` takes `u128` on both the scalar trait and inherent shipped-family methods. Existing `u64` variables need an explicit conversion such as `u128::from(exponent)`. The separate `QuadMersenne31::pow_u128` method is replaced by `pow`.

## Retain prepared-operation geometry

`Coeff<F>`, `CoeffVec<F>`, and `CoeffMatrix<F>` remain typed by the field; their constructors take `Elem<F>` coefficients. Matrix construction remains source-major, and application still borrows ordinary byte buffers. A field rename does not transpose coefficient axes or change row lengths.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let coefficient = ops::Coeff::<F>::new(Elem::<F>::from_raw(3));
let mut destination = [0xa5u8; 3];
ops::mul_into_with::<F>(&mut destination, &coefficient, &[1, 2, 3]);
assert_eq!(destination, [3, 6, 5]);
```

Prepared values are not a serialized compatibility surface. Rebuild them from scalar values in the new process. [Prepared coefficients](Prepared-coefficients) explains owned collections and borrowed views.

## Remove explicit post-construction canonicalization

In 2.x, prime construction and decoding could retain raw lanes, and `canonical()` reduced the element explicitly. In 3.0.1, construction and decoding store canonical coordinates, so `canonical()` is removed and `to_raw()` returns the reduced storage.

```rust
use fgf::{Elem, FieldBuffer, Mersenne31, mersenne31};

let input = mersenne31::MODULUS + 7;
let value = Elem::<Mersenne31>::from_raw(input);
assert_eq!(value.to_raw(), 7);
assert_eq!(Mersenne31::decode(&input.to_le_bytes()), value);
assert_eq!(value.to_bytes(), 7u32.to_le_bytes());
```

Applications that relied on retaining unreduced bits must retain the original bytes separately. Published canonical encodings and mathematical values remain unchanged, but decoding a noncanonical alias and encoding it again does not preserve its original bytes.

Packed prime operations accept raw lane patterns supported by decoding. Manual input canonicalization is not a precondition for arithmetic. Zero-coefficient accumulation, unit in-place scaling, and unit overwrite copying can preserve raw bytes; those operations must not be treated as normalization requests. [Prime fields](Prime-fields) gives an explicit decode–encode boundary and executable shortcut examples.

## Replace constants and tower paths

The `gf16`, `gf32`, `gf64`, and concrete Fan–Paar element modules are removed. The binary presentations live under `field::binary`; root re-exports provide the usual imports.

Module generator constants become `Elem::<F>::GENERATOR` under `HasGenerator`. A tower relation is the specification's `RELATION`, its coefficients listed constant term first; the tower indeterminate is not necessarily its multiplicative generator.

A custom tower implements `ExtensionSpec<2>` over a binary base, with `Packed` or `PackedPadded` storage and a relation built by `Relation::from_binary_words`. Custom rows declare no generator and carry no `const` element arithmetic. [Quadratic towers](Quadratic-towers) explains the relation and validity checks.

## Keep GF(2) on its bit-packed surface

`bits::dot_product` returns `Elem<FiniteField<2, 1, Polynomial<3>>>`. Scalar GF(2) is a `Field`, but its vectors remain bit-packed and do not implement the `FieldBuffer` contract.

```rust
use fgf::{FiniteField, Elem, Polynomial, bits};

type F = FiniteField<2, 1, Polynomial<3>>;
let parity: Elem<F> = bits::dot_product(&[0b101], &[0b001], 3);
assert_eq!(parity, Elem::<F>::ONE);
```

Logical bit length, LSB-first ordering, and caller-owned padding remain part of the buffer convention.

## Check the application boundary

Migration checks compare canonical wire fixtures, generator-dependent schedules, rectangular coefficient geometry, and prime inputs near the modulus. A passing type check alone does not establish that a raw-coordinate reinterpretation is correct.

From a crate checkout, `just examples` exercises complete workflows and `just doc` builds the API reference. [The type model](Type-model-and-generic-algorithms) is the entry point for rewriting generic library code; [deployment](Features-and-deployment) covers feature-specific verification.
