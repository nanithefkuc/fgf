# Custom prime fields and extensions

A new field does not require a new element wrapper or an implementation of `Field`. Choose a prime modulus with `FiniteField<P, 1>`, or define an extension relation with `ExtensionSpec`. The resulting values still use `Elem<F>` and the same arithmetic operators as the named fields.

The examples use fgf 3.0.1. The [field reference](https://docs.rs/fgf/3.0.1/fgf/field/index.html) describes the supported types. `Field`, `FieldBuffer`, and `HasGenerator` are sealed; custom definitions supply parameters to crate-defined implementations rather than replacing those traits.

## Choose a modulus or an extension

`FiniteField<P, 1>` uses the default `Residue` presentation for a checked `u64` prime. Each element stores a canonical `u64` residue and encodes as an eight-byte little-endian lane. A custom extension instead uses `FiniteField<P, N, Extension<Spec, D>>`, with a data-only `ExtensionSpec` supplying its base, storage, and relation. [Quadratic towers](Quadratic-towers) gives a binary example.

`FiniteField<2, 1>` and `FiniteField<2, 1, Polynomial<3>>` have the same algebra but different storage APIs. The residue presentation has eight-byte lanes; the polynomial presentation has a scalar raw byte and uses `bits` for vectors, not `FieldBuffer`. Convert scalar values by residue rather than copying buffers between them. `PrimeCharacteristic<P>` checks a prime identity; the field row supplies the arithmetic.

## Use the residue row

The residue presentation provides operators, `FieldElem` methods, a `const` element API, and `FieldBuffer` encoding. It also supports checked packed arithmetic through `ops`, routed to the scalar backend.

```rust
use fgf::{Elem, FieldBuffer, FiniteField, ops};

type Seven = FiniteField<7, 1>;

let a = Elem::<Seven>::from_raw(12); // reduces on construction
let b = Elem::<Seven>::from_raw(3);
assert_eq!(a.to_raw(), 5);
assert_eq!((a * b).to_raw(), 1);
assert_eq!((a - b).to_raw(), 2);
assert_eq!((-a).to_raw(), 2);
assert_eq!(a * a.inv(), Elem::<Seven>::ONE);
assert_eq!(a / Elem::<Seven>::ZERO, Elem::<Seven>::ZERO);

// Construction outside `0..P` is rejected instead of narrowed.
assert!(Elem::<Seven>::from_residue(7).is_err());

// The `const` surface: a reciprocal table can be a `const` item.
const HALF: Elem<Seven> = Elem::<Seven>::from_raw(2).inv();
const _: () = assert!(HALF.to_raw() == 4);

// Eight-byte little-endian lanes: packing needs the buffer, not kernels.
let mut bytes = [0u8; 16];
ops::pack::<Seven>(&mut bytes, &[a, b]);
assert_eq!(&bytes[..8], &5u64.to_le_bytes());
assert_eq!(&bytes[8..], &3u64.to_le_bytes());
assert_eq!(Seven::decode(&12u64.to_le_bytes()), a);
```

Every raw lane is a legal input to construction and decoding, which reduce it into `0..P` on the way in. Arithmetic is total under the crate-wide conventions — `inv(0) == 0`, `x / 0 == 0`, `0^0 == 1` — variable-time and not intended for secret data.

## What central validation establishes

Construction checks that the characteristic is prime, the degree is positive, and the field order fits and equals the characteristic raised to that degree. Each presentation then validates its own defining data.

For a residue field, the defining parameter is the prime modulus. An invalid modulus is rejected when constructing an element. The named prime fields validate their moduli through the same field-level checks.

## Generators and kernels on named rows

The residue row implements `FieldBuffer` and the packed operations with `Scalar` routing; it does not implement `HasGenerator`, because a generic generator needs the factorization of `P − 1`, which fgf holds only for pinned moduli. Rows with a checked generator — the named prime rows, the binary rows, the shipped extension instances, and `QuadMersenne31` — prove full multiplicative order inside the provider constant.

```rust
use fgf::{Elem, Mersenne31};

// Order p - 1 = 2 * 3^2 * 7 * 11 * 31 * 151 * 331, checked by cofactors.
let g = Elem::<Mersenne31>::GENERATOR;
assert_eq!(g.pow(0x7FFF_FFFE), Elem::<Mersenne31>::ONE);
for q in [2u128, 3, 7, 11, 31, 151, 331] {
    assert_ne!(g.pow(0x7FFF_FFFE / q), Elem::<Mersenne31>::ONE);
}
```

## Verification before application use

The crate checks its field implementations, but an application can still misuse a correct field. Compare application results against an independent modular calculation, including boundary residues and zero handling, and check that every required inverse is nonzero.

