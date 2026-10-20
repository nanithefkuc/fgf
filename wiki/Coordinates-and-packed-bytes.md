# Coordinates and packed bytes

Pack values into fixed-width little-endian lanes before calling byte-buffer
operations. Lanes are adjacent, without padding, and their starting address
need not be aligned. `FieldBuffer` owns the encoding.

## Raw coordinates are not mathematical integers

Binary raw bits select basis coordinates, not ordinary integers. Polynomial
raw `2` means the indeterminate; normal raw `2` selects the second basis vector.
Prime raw integers reduce modulo the characteristic. `to_raw` returns canonical
coordinates, not the original input spelling.

## Residues name values, indices name coordinates

`from_residue` names a multiple of one; `from_index` names flattened prime-subfield
coordinates. In a normal basis these differ:

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

Use checked `from_index` at format boundaries. Unlike a masking `from_raw`
constructor, it rejects excess coordinates:

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

`encode` writes exactly `F::BYTES` bytes. `ops::pack` and `unpack` fill caller-owned
storage; `pack_to_vec` requires `alloc`. Generic residue rows use `encode` and
`decode`, not the fixed-array `to_bytes` methods of shipped element families.

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

Packing requires `FieldBuffer`; arithmetic on packed bytes requires `FieldKernels`.

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

The frame's metadata stays unchanged. Payload length must still be a multiple
of `F::BYTES`; a partial lane panics. Paired operations also require matching
lengths, and row operations check their declared geometry.

GF(2) vectors instead pack several logical elements into each byte. Their final
byte can be partial; use [bit-packed operations](Bit-packed-GF2), not byte-lane
`ops`. Binary degrees two and four have no `FieldBuffer`.

## Serialization needs a presentation agreement

Peers must agree on the polynomial, ordered basis or extension relation, byte
order, and element count. Field bytes do not contain a self-describing header.
Prime decode–encode can change noncanonical input bytes; retain the original
payload separately if exact byte preservation matters.

Continue with [row operations](Linear-operations-and-row-geometry). The
[packing reference](https://docs.rs/fgf/3.0.2/fgf/ops/index.html) owns exact sizes
and errors. The [packed no-allocation example](https://github.com/nanithefkuc/fgf/blob/v3.0.2/examples/packed_no_alloc.rs)
runs this surface against the library without `std` or `alloc`:
`just example packed_no_alloc --no-default-features`.

