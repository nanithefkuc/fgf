# Prime fields and their specializations

Use `FiniteField<P, 1>` for arithmetic modulo a `u64` prime. The default
`Residue` representation supplies scalar arithmetic, packed operations, and
prepared coefficients without implementing a field trait.

## Start with the modulus

```rust
use fgf::{Elem, FieldBuffer, FiniteField, ops};

type F = FiniteField<17, 1>;
let x = Elem::<F>::from_raw(24);
assert_eq!(x.to_raw(), 7);
assert_eq!(Elem::<F>::from_residue(17), Err(fgf::RangeError));

let source = [24u8];
let mut destination = [0xa5u8];
ops::mul_into::<F>(&mut destination, Elem::<F>::from_raw(3), &source);
assert_eq!(F::decode(&destination), x * Elem::<F>::from_raw(3));
assert_eq!(destination, [4]);
```

`from_raw` and decoding reduce integer aliases; `from_residue` rejects values
outside `0..P`. Constructors check that `P` is prime. Generic residue arithmetic
uses modular addition and subtraction, never bytewise XOR.

| Default residue characteristic | Little-endian lane |
| --- | --- |
| Odd prime below `2^8` | `u8` |
| Odd prime below `2^16` | `u16` |
| Odd prime below `2^32` | `u32` |
| Larger prime, or `P = 2` | `u64` |

Use `FieldBuffer::encode` and `decode` for scalar bytes, or `ops::pack` and
`unpack` for vectors. Generic residue rows do not have fixed-array `to_bytes`
methods. [Packing](Coordinates-and-packed-bytes) shows the shared workflow.

## Select a specialized representation when needed

Named prime families are choices within the same `FiniteField` model:

| Alias | Representation | Characteristic | Encoding |
| --- | --- | --- | --- |
| `Mersenne31` | `MersenneFold` | `2^31 - 1` | Little-endian `u32` |
| `Goldilocks` | `GoldilocksFold` | `2^64 - 2^32 + 1` | Little-endian `u64` |
| `QuadMersenne31` | `Extension<ComplexMersenne31, 2>` | `2^31 - 1` | Interleaved `[re: u32, im: u32]` |

A default residue row at the same modulus is a distinct type. Move values by
checked residue conversion, not by assuming the raw-word types match.
`QuadMersenne31` is an extension, not arithmetic modulo its order.

## Canonical values do not imply canonical passthrough bytes

Zero-coefficient accumulation leaves the destination unchanged. Unit in-place
scaling preserves it; unit overwrite scaling copies the source. These shortcuts
can retain noncanonical byte spellings. Normalize explicitly at a format boundary:

```rust
use fgf::{Elem, Mersenne31, mersenne31, ops};

let original = (mersenne31::MODULUS + 7).to_le_bytes();
let mut copied = [0u8; 4];
ops::mul_into::<Mersenne31>(&mut copied, Elem::<Mersenne31>::ONE, &original);
assert_eq!(copied, original);
let mut value = [Elem::<Mersenne31>::ZERO];
ops::unpack::<Mersenne31>(&mut value, &copied);
ops::pack::<Mersenne31>(&mut copied, &value);
assert_eq!(copied, 7u32.to_le_bytes());
```

Decode–encode loses the original alias. Applications needing byte-for-byte
preservation retain the incoming bytes separately. Accepting a lane for arithmetic
does not validate a proof, codeword, or evaluation vector.

## Combine pointwise and fixed-coefficient operations

This computes `out[i] = a[i]*b[i] + challenge*a[i] - a[i]`:

```rust
use fgf::{Elem, Mersenne31, ops};

let a = [1u32, 2, 3].map(Elem::<Mersenne31>::from_raw);
let b = [4u32, 5, 6].map(Elem::<Mersenne31>::from_raw);
let mut packed_a = [0u8; 12];
let mut packed_b = [0u8; 12];
ops::pack::<Mersenne31>(&mut packed_a, &a);
ops::pack::<Mersenne31>(&mut packed_b, &b);
let challenge = ops::Coeff::<Mersenne31>::new(Elem::<Mersenne31>::from_raw(13));
let mut output = [0u8; 12];
ops::mul_elementwise::<Mersenne31>(&mut output, &packed_a, &packed_b);
ops::mul_add_with::<Mersenne31>(&mut output, &challenge, &packed_a);
ops::sub_assign::<Mersenne31>(&mut output, &packed_a);
let mut values = [Elem::<Mersenne31>::ZERO; 3];
ops::unpack::<Mersenne31>(&mut values, &output);
assert_eq!(values.map(Elem::to_raw), [16, 34, 54]);
```

Transforms and interpolation belong to other libraries. `fgf` supplies their
field arithmetic, not the transform or evaluation schedule.

## Work with quadratic components

`QuadMersenne31` represents `re + im*i`, with `i² = -1`. Conjugation negates
the imaginary component; the norm returns `re² + im²` in `Mersenne31`:

```rust
use fgf::{Elem, Mersenne31, QuadMersenne31};

let z = Elem::<QuadMersenne31>::from_components([
    Elem::<Mersenne31>::from_raw(7), Elem::<Mersenne31>::from_raw(11),
]);
assert_eq!(z.norm().to_raw(), 170);
assert_eq!(z * z.conjugate(), Elem::<QuadMersenne31>::from_components([
    z.norm(), Elem::<Mersenne31>::ZERO,
]));
assert_eq!(z.to_bytes(), [7, 0, 0, 0, 11, 0, 0, 0]);
```

`conjugate` and `norm` equal the relative Frobenius and norm of
`Embedding<Mersenne31, QuadMersenne31>` (whose `trace` has no inherent
spelling here);
[embeddings](Embeddings-and-presentation-changes) define the relative maps.

For a custom relation or storage choice, continue with
[custom fields and extensions](Downstream-scalar-fields). Exact contracts live in
the [field reference](https://docs.rs/fgf/3.0.2/fgf/field/index.html).
