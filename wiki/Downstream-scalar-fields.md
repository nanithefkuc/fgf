# Downstream scalar fields

`Field`, `FieldBuffer`, and `HasGenerator` are sealed: fgf implements them for the canonical family rows and the named rows only, and the crate's relation algorithms read presentation facts that stay crate-owned. A downstream crate names a row; it does not implement the traits. A custom prime field is the family's degree-one row `FiniteField<P, 1>` over any checked `u64` prime, and a custom extension is an `ExtensionSpec` named inside an extension row.

The [field reference](https://docs.rs/fgf/latest/fgf/field/index.html) owns the row contracts. [The type model](Type-model-and-generic-algorithms) introduces the capability split.

## Choose the correct row

A custom extension, such as a binary quadratic tower over a byte field, implements the data-only `ExtensionSpec` trait and uses `FiniteField<P, N, Extension<Spec, D>>`, not a new scalar wrapper. It receives scalar arithmetic, component conversion, the byte-buffer surface when its storage has a byte form, and embeddings over a binary base; it carries no `const` element arithmetic and no generator. [Quadratic towers](Quadratic-towers) explains that route.

A prime field at any checked modulus is `FiniteField<P, 1>`, the default `Residue` presentation: one canonical `u64` residue per element, with eight-byte little-endian wire lanes. At characteristic two this row is distinct from the binary degree-one row `FiniteField<2, 1, Polynomial<3>>`, whose lane is one byte; values move between presentations by residue, never by buffer. `PrimeCharacteristic<P>` checks a prime identity; it is not an arithmetic engine for every `P`.

## Use the residue row

The residue row supplies the whole scalar surface `Elem<F>` wraps: operators, the `FieldElem` trait methods, `FieldBuffer` bytes, the checked packed operations with `Scalar` routing, and the `const` element API.

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

Generic construction checks a prime characteristic identity, positive degree, and `ORDER == CHARACTERISTIC^DEGREE` with checked exponentiation, before the row's own validity proof runs. These checks reject inconsistent metadata at first value use, in `const` context or at runtime.

The residue row's defining data is the characteristic itself, so the primality proof is the whole validation: an invalid `P` fails construction of any element. The named fold rows (`Mersenne31`, `Goldilocks`) carry the same central validation over their own moduli.

## Generators and kernels on named rows

The residue row implements `FieldBuffer` and the packed operations with `Scalar` routing; it does not implement `HasGenerator`, because a generic generator needs the factorization of `P − 1`, which fgf holds only for pinned moduli. Rows with a checked generator — the named prime rows, the binary rows, the shipped towers, and `QuadMersenne31` — prove full multiplicative order inside the provider constant.

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

A row chosen from the vocabulary arrives tested: known answers, field laws, byte codecs, and kernel routing are the crate's own contract. Application checks still need observable results against an independent modular reference — boundary residues, zero inversion, nonzero inverse products — because an algorithm misusing a correct field still produces incorrect results.

Next: [Features and deployment](Features-and-deployment), or [The type model and generic algorithms](Type-model-and-generic-algorithms).
