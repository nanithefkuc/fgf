# Coordinates and packed bytes

`Elem<F>` holds a canonical scalar value in a chosen presentation. `FieldBuffer` encodes that value as a fixed-width little-endian lane, and `ops` packs consecutive lanes without alignment or padding requirements.

This guide targets fgf 3.0.1. The [field reference](https://docs.rs/fgf/3.0.1/fgf/field/index.html) owns encoding contracts; the [ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) owns packed conversion and arithmetic contracts.

## Raw coordinates are not mathematical integers

For a binary presentation, raw bits select basis coordinates. `Elem::<F>::from_raw(2)` need not mean the sum of two copies of one: that sum is zero in characteristic two. In a polynomial basis, raw `2` is the indeterminate. In a normal basis, it selects the second conjugate basis vector.

For a prime field, the raw integer is reduced modulo the characteristic. `from_raw` canonicalizes storage; `to_raw` returns the stored canonical word. It cannot recover a noncanonical spelling used to construct the element. Equality and hashing see canonical field values, while ordering is coordinate order rather than an order compatible with field multiplication.

## Residues name values, indices name coordinates

`FieldElem::from_residue` maps an ordinary residue below the characteristic to its multiple of one, whatever the basis. `FieldElem::from_index` decodes the flattened prime-subfield coordinates instead. The two agree only where one is coordinate word one: in `Normal<7, 2>` the one is the sum of both basis vectors, so it reads residue one at index three.

```rust
use fgf::{Elem, FieldElem, FiniteField, Normal};

type Row = FiniteField<2, 2, Normal<7, 2>>;
let one = Elem::<Row>::ONE;
assert_eq!(one.to_residue(), Some(1));
assert_eq!(one.to_index(), 3);
assert_eq!(Elem::<Row>::from_residue(1).unwrap(), one);
assert_eq!(Elem::<Row>::from_index(3).unwrap(), one);
```

## Canonicalize or reject excess coordinates

Small binary element constructors mask away bits above the degree. The `FieldElem::from_index` conversion instead rejects an out-of-range word. The checked conversion is useful at a format boundary where excess bits indicate malformed input rather than an accepted alias.

```rust
use fgf::{Elem, FieldElem, FiniteField, Polynomial};

type Small = FiniteField<2, 4, Polynomial<0x13>>;
assert_eq!(Elem::<Small>::from_raw(0x12).to_raw(), 2);
assert!(Elem::<Small>::from_index(0x12).is_err());
let x = Elem::<Small>::from_index(2).unwrap();
assert_eq!(x.to_index(), 2);
```

Canonicalization operates within a presentation. It does not convert a polynomial element into normal coordinates or an AES value into RS coordinates. [Embeddings](Embeddings-and-presentation-changes) handle those conversions.

## Encode a scalar, then pack a vector

`FieldBuffer::encode` writes one scalar to a slice of exactly `F::BYTES` bytes. Concrete shipped families also provide fixed-array methods such as `to_bytes`. `ops::pack` and `ops::unpack` fill caller-owned buffers; the allocating `pack_to_vec` variant requires `alloc`.

```rust
use fgf::{Elem, Extension, FieldBuffer, FiniteField, Rijndael16, ops};

type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let values = [Elem::<F>::from_raw(0x1234), Elem::<F>::from_raw(0xabcd)];
let mut scalar = [0u8; 2];
F::encode(&mut scalar, values[0]);
assert_eq!(scalar, [0x34, 0x12]);
assert_eq!(F::decode(&scalar), values[0]);
let mut bytes = [0u8; 4];
ops::pack::<F>(&mut bytes, &values);
assert_eq!(bytes, [0x34, 0x12, 0xcd, 0xab]);
let mut decoded = [Elem::<F>::ZERO; 2];
ops::unpack::<F>(&mut decoded, &bytes);
assert_eq!(decoded, values);
```

Packing needs `FieldBuffer`, not `FieldKernels`: byte conversion does not require arithmetic kernels. A row with a byte format does not thereby gain the sealed `ops` arithmetic capability; the residue row carries the packed surface with `Scalar` routing.

## Apply arithmetic to an unaligned payload

Buffers hold complete lanes, but their starting address need not have native integer alignment. A framed payload can be borrowed as bytes without a transmute or a typed integer view.

```rust
use fgf::{Elem, Extension, FiniteField, Rijndael16, ops};

type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let mut frame = [0x7eu8; 6];
let input = [Elem::<F>::from_raw(0x1234), Elem::<F>::from_raw(0xabcd)];
ops::pack::<F>(&mut frame[1..5], &input);
let scale = Elem::<F>::from_raw(0x0108);
ops::mul_assign::<F>(&mut frame[1..5], scale);
let mut output = [Elem::<F>::ZERO; 2];
ops::unpack::<F>(&mut output, &frame[1..5]);
assert_eq!(output, [input[0] * scale, input[1] * scale]);
assert_eq!(frame[0], 0x7e);
assert_eq!(frame[5], 0x7e);
```

The slice begins after one metadata byte, and only the payload changes. Its length is still a multiple of `F::BYTES`. An odd byte count would contain a partial degree-sixteen element and would panic in checked packed arithmetic.

## Byte count, lane width, and logical length

A byte-lane buffer contains `bytes.len() / F::BYTES` elements only if the division is exact. Multiplication and pairwise operations require whole lanes and, where paired, equal byte lengths. Row operations additionally validate their declared geometry.

GF(2) vectors are different: eight logical elements share one byte, and the final byte may be partial. Degrees two and four have scalar storage but no `FieldBuffer`; the existence of a raw byte word is not permission to use packed `ops`. [Bit-packed GF(2)](Bit-packed-GF2) explains logical length and padding.

## Serialization needs a presentation agreement

A stable encoding fixes byte order and coordinate order. It does not identify the field in the byte stream. Peers must agree on the polynomial, ordered basis, or recursive tower relation, along with the payload's element count. A generic `Elem<F>` supplies no universal self-describing wire header.

Prime decoding accepts aliases and stores their canonical value. Decode–encode therefore need not reproduce the original bytes of an arbitrary raw payload. Applications requiring byte-for-byte preservation retain the original bytes separately. [Prime fields](Prime-fields) explains arithmetic passthrough cases as well.

Prepared coefficients are backend-bound state, not a wire format. Serialize their scalar values and rebuild preparation in the receiving process.

Next: [Linear operations and row geometry](Linear-operations-and-row-geometry).
