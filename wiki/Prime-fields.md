# Prime fields and quadratic prime extensions

Prime-field arithmetic adds and multiplies residues modulo a prime. `Mersenne31` and `Goldilocks` provide named fields with compact lanes; `QuadMersenne31` adds a second component to form a quadratic extension of `Mersenne31`.

The examples use fgf 3.0.1. The API references for [Mersenne31](https://docs.rs/fgf/3.0.1/fgf/field/mersenne31/index.html), [Goldilocks](https://docs.rs/fgf/3.0.1/fgf/field/goldilocks/index.html), and [QuadMersenne31](https://docs.rs/fgf/3.0.1/fgf/field/quad_mersenne31/index.html) give their encoding and arithmetic contracts.

## Modulus, order, and lane width

| Field | Characteristic | Degree | Order | Encoding |
| --- | --- | --- | --- | --- |
| `Mersenne31` | `2^31 - 1` | 1 | `2^31 - 1` | One little-endian `u32` |
| `Goldilocks` | `2^64 - 2^32 + 1` | 1 | `2^64 - 2^32 + 1` | One little-endian `u64` |
| `QuadMersenne31` | `2^31 - 1` | 2 | `(2^31 - 1)^2` | Interleaved `[re: u32, im: u32]`, little-endian |

Prime addition and subtraction are modular arithmetic, not bytewise XOR. `QuadMersenne31` has two base-field components and is an extension field, not arithmetic modulo its order.

## Construction loses noncanonical spellings

```rust
use fgf::{Elem, FieldBuffer, Mersenne31, mersenne31};

let p = mersenne31::MODULUS;
let x = Elem::<Mersenne31>::from_raw(p + 7);
assert_eq!(x.to_raw(), 7);
assert_eq!(Elem::<Mersenne31>::from_raw(p), Elem::<Mersenne31>::ZERO);
assert_eq!(Elem::<Mersenne31>::from_raw(u32::MAX).to_raw(), 1);
let raw_bytes = (p + 7).to_le_bytes();
let decoded = Mersenne31::decode(&raw_bytes);
let mut canonical_bytes = [0u8; 4];
Mersenne31::encode(&mut canonical_bytes, decoded);
assert_eq!(canonical_bytes, 7u32.to_le_bytes());
assert_ne!(canonical_bytes, raw_bytes);
```

Construction stores only the canonical value. `to_raw` cannot recover the original noncanonical word, so an application that needs the incoming payload bytes must retain them separately.

## Raw packed lanes are accepted

A packed prime row need not have been produced by `pack`. Arithmetic accepts the lane patterns accepted by field decoding. A nontrivial scale can consume raw lanes directly and emit reduced results.

```rust
use fgf::{Elem, Mersenne31, mersenne31, ops};

let source = (mersenne31::MODULUS + 7).to_le_bytes();
let mut destination = [0xa5u8; 4];
ops::mul_into::<Mersenne31>(&mut destination, Elem::<Mersenne31>::from_raw(3), &source);
assert_eq!(destination, 21u32.to_le_bytes());
```

Accepting a lane for field arithmetic does not validate it as a proof, codeword, or evaluation vector. The application still checks its own format and mathematical constraints.

## Identity operations are not normalization requests

Zero-coefficient accumulation contributes nothing and leaves the destination byte-for-byte unchanged. Unit-coefficient in-place scaling also preserves the input. Unit-coefficient overwrite scaling copies the source bytes. These shortcuts preserve the field value but can retain noncanonical raw spellings.

```rust
use fgf::{Elem, Mersenne31, mersenne31, ops};

let original = (mersenne31::MODULUS + 7).to_le_bytes();
let mut destination = original;
ops::mul_add::<Mersenne31>(&mut destination, Elem::<Mersenne31>::ZERO, &original);
assert_eq!(destination, original);
ops::mul_assign::<Mersenne31>(&mut destination, Elem::<Mersenne31>::ONE);
assert_eq!(destination, original);
let mut copied = [0u8; 4];
ops::mul_into::<Mersenne31>(&mut copied, Elem::<Mersenne31>::ONE, &original);
assert_eq!(copied, original);
let mut decoded = [Elem::<Mersenne31>::ZERO; 1];
ops::unpack::<Mersenne31>(&mut decoded, &copied);
ops::pack::<Mersenne31>(&mut copied, &decoded);
assert_eq!(copied, 7u32.to_le_bytes());
```

An explicit decode–encode or unpack–pack boundary produces canonical bytes even when an arithmetic passthrough does not. Identity coefficients must not be used as a substitute for that boundary.

## An evaluation-vector workflow

Pointwise products differ from multiplying every lane by one coefficient. The following calculation combines both: `out[i] = a[i]*b[i] + challenge*a[i] - a[i]`. Stack buffers avoid requiring `alloc`.

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

Transforms and interpolation are outside `fgf`; a transform library can compose these field operations rather than treating the vector example as a transform implementation.

## Quadratic components, conjugation, and norm

`QuadMersenne31` names the extension row `FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>` and represents `re + im*i` with `i² = -1`. Each limb canonicalizes independently. Conjugation negates the imaginary component, and the norm returns `re² + im²` in `Mersenne31`.

```rust
use fgf::{Elem, Mersenne31, QuadMersenne31, mersenne31};

let limbs = [mersenne31::MODULUS + 7, 11];
let z = Elem::<QuadMersenne31>::from_raw(limbs);
assert_eq!(z.to_raw(), [7, 11]);
let [re, im] = z.to_components();
assert_eq!(Elem::<QuadMersenne31>::I.square(), -Elem::<QuadMersenne31>::ONE);
assert_eq!(z.norm().to_raw(), 170);
assert_eq!(z * z.conjugate(), Elem::<QuadMersenne31>::from_components([z.norm(), Elem::<Mersenne31>::ZERO]));
assert_eq!(z.norm(), re.square() + im.square());
assert_eq!(z.to_bytes(), [7, 0, 0, 0, 11, 0, 0, 0]);
```

The inherent constructor and a generic `FieldElem::from_raw` call both take the field's `Raw` array `[re, im]`. The wrapper remains `Elem<QuadMersenne31>` in both cases.

All these operations are variable-time. From a checkout, `just example prime_vectors` runs the [complete multi-family workflow](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/prime_vectors.rs).

