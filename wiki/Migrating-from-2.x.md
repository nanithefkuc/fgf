# Migrating from 2.x to 3.0.1

Migration replaces field-specific element wrappers with `Elem<F>`, separates
scalar, encoding, and generator capabilities, and canonicalizes named prime
elements at construction. Historical snippets below are text; Rust snippets
use 3.0.1. The [changelog](https://github.com/nanithefkuc/fgf/blob/v3.0.1/CHANGELOG.md)
records the release changes.

The `3.0.0` once published to crates.io went out by accident and was yanked;
it was never a supported release and is not API-compatible with 3.0.1 — the
`Binary<N, R>`, `Tower<S>`, `TowerSpec`, and `TowerGeneratorSpec` spellings it
exported are gone, and `Field` is sealed. Code locked to that artifact follows
the same guide: the 2.x steps below apply, and the 3.0.1 changelog entries
cover the remaining differences.

For 3.0.2, apply the 2.x-to-3.0.1 migration below, then use `Isomorphism`
for every equal-degree map: replace `Embedding::new`/`from_images` with
`Isomorphism::new`/`from_images`, `embed` with `apply`, and `restrict` with
`unapply`. `Isomorphism::from_embedding` and `to_embedding` are removed.
Total extension degrees must not exceed 64. AArch64 and Wasm32 use portable
scalar kernels. The [3.0.2 changelog](https://github.com/nanithefkuc/fgf/blob/v3.0.2/CHANGELOG.md)
lists the complete changes.

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

## Size generic-residue rows from the lane width

Generic residue rows `FiniteField<P, 1>` use a compact lane sized to the
characteristic: one, two, four, or eight bytes as residues fit within
`2^8`, `2^16`, `2^32`, and `u64` (characteristic two keeps an eight-byte
lane). Size buffers from `FieldBuffer::BYTES` and move lanes through
`FieldBuffer::decode`/`encode`; the inherent fixed-array `from_bytes` and
`to_bytes` are absent on these rows at every characteristic, and no
`const` byte form exists for them. Named rows keep their lane widths and
fixed-array methods. Extensions over a residue base shrink with the base
lane: a degree-`D` extension over a `B`-byte base moves `D * B` bytes per
element, with `Components`, `Packed`, and `PackedPadded` storage as before.

## Retain prepared-operation geometry

Prepared calls retain their row lengths, coefficient axes, and source counts.
Rebuild backend-bound preparation from scalar values after loading; it is not
a serialized compatibility surface. [Prepared coefficients](Prepared-coefficients)
shows reusable matrices and borrowed views.

## Remove explicit post-construction canonicalization

Named prime elements reduce input words during construction and decoding.
Remove their explicit `canonical()` calls; `to_raw()` returns reduced storage.

```rust
use fgf::{Elem, FieldBuffer, Mersenne31, mersenne31};

let input = mersenne31::MODULUS + 7;
let value = Elem::<Mersenne31>::from_raw(input);
assert_eq!(value.to_raw(), 7);
assert_eq!(Mersenne31::decode(&input.to_le_bytes()), value);
assert_eq!(value.to_bytes(), 7u32.to_le_bytes());
```

Retain original bytes separately if the application needs unreduced spellings.
Canonical wire values remain unchanged, but decode–encode can change aliases.
Packed operations accept decoder-supported raw lanes; zero/unit shortcuts can
preserve those bytes. Use an explicit [normalization boundary](Prime-fields)
rather than an identity multiplication.

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
