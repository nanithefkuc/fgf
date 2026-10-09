# The type model and generic algorithms

A field type fixes arithmetic and representation; `Elem<F>` holds one value in that field. A generic algorithm chooses capabilities through trait bounds rather than naming a particular field family.

This guide targets fgf 3.0.1. The [field module reference](https://docs.rs/fgf/3.0.1/fgf/field/index.html) owns the detailed contracts.

## A field is not an element

`FiniteField<2, 8, Polynomial<AES>>` names a presentation of GF(2^8). It is a zero-sized marker, not a byte containing a field value. `Elem<FiniteField<2, 8, Polynomial<AES>>>` contains that value. A local alias shortens both spellings without creating another library API.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial};

type F = FiniteField<2, 8, Polynomial<AES>>;
const A: Elem<F> = Elem::<F>::from_raw(0x53);
const B: Elem<F> = Elem::<F>::from_raw(0xca);
const PRODUCT: Elem<F> = A.mul(B);
assert_eq!(PRODUCT, Elem::<F>::ONE);
assert_eq!(PRODUCT.to_raw(), 1);
```

The representation is part of the type. An AES value and an RS value cannot be mixed in an arithmetic expression merely because both occupy one byte. [Embeddings](Embeddings-and-presentation-changes) perform an explicit value-preserving conversion.

## Choose the narrowest capability

| Bound | What an algorithm can use |
| --- | --- |
| `F: Field` | `Elem<F>`, scalar arithmetic, raw-word type, characteristic identity, degree, and order |
| `F: FieldBuffer` | The scalar contract plus fixed-width byte encoding |
| `F: HasGenerator` | A selected multiplicative generator as `Elem::<F>::GENERATOR` |
| `F: FieldKernels` | Byte encoding plus checked packed arithmetic through `ops` |
| `F: BinaryField` | Supported binary presentations and checked coordinate conversions |
| `F: BinaryDegree<N>` | A supported binary field of exactly degree `N` |

`FieldElem` is the scalar-arithmetic trait implemented for `Elem<F>`. It, `Field`, `FieldBuffer`, and `HasGenerator` are sealed: rows are crate-defined, and generic code works over the shipped vocabulary. A custom extension supplies only a data-only `ExtensionSpec` to a crate-defined extension row. Importing `FieldElem` exposes trait methods when a generic function cannot name a concrete family's inherent methods.

`FieldKernels` is also sealed. A row does not gain packed operations by declaring a byte format alone; the residue row carries the packed surface with `Scalar` routing. The [downstream field guide](Downstream-scalar-fields) separates these boundaries.

## One scalar algorithm, several fields

Horner evaluation needs scalar addition and multiplication, not a byte encoding or generator. Coefficients below are ordered from constant term upward; the reversed fold evaluates `c0 + c1*x + ...`.

```rust
use fgf::{AES, FiniteField, Elem, Field, Mersenne31, Polynomial};

fn evaluate<F: Field>(coefficients: &[Elem<F>], x: Elem<F>) -> Elem<F> {
    coefficients.iter().rev().fold(Elem::<F>::ZERO, |sum, &c| sum * x + c)
}

type ByteField = FiniteField<2, 8, Polynomial<AES>>;
let binary_coefficients = [
    Elem::<ByteField>::from_raw(1),
    Elem::<ByteField>::from_raw(2),
    Elem::<ByteField>::from_raw(3),
];
let prime_coefficients = [
    Elem::<Mersenne31>::from_raw(1),
    Elem::<Mersenne31>::from_raw(2),
    Elem::<Mersenne31>::from_raw(3),
];
assert_eq!(evaluate(&binary_coefficients, Elem::<ByteField>::from_raw(2)).to_raw(), 9);
assert_eq!(evaluate(&prime_coefficients, Elem::<Mersenne31>::from_raw(2)).to_raw(), 17);
assert_eq!(evaluate::<ByteField>(&[], Elem::<ByteField>::ONE), Elem::<ByteField>::ZERO);
```

The identical function computes different arithmetic because the coefficients and evaluation point belong to different fields. Binary addition is XOR; prime addition is modular integer addition.

## A packed algorithm needs kernels

A row operation takes byte slices and a typed coefficient. The coefficient pins the field while `ops` validates byte geometry and handles backend routing.

```rust
use fgf::{AES, FiniteField, Elem, FieldKernels, Polynomial, ops};

fn combine<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8], scale: Elem<F>) {
    ops::mul_into::<F>(dst, Elem::<F>::ONE, a);
    ops::mul_add::<F>(dst, scale, b);
}

type F = FiniteField<2, 8, Polynomial<AES>>;
let mut dst = [0xa5u8; 3];
combine::<F>(&mut dst, &[1, 2, 3], &[4, 5, 6], Elem::<F>::from_raw(2));
assert_eq!(dst, [9, 8, 15]);
```

Each operation validates its own operands. This wrapper is not transactional: if the second call has invalid geometry, the first call has already written `dst`. Application-level validation is needed before a multi-call workflow that must leave its output unchanged on failure.

## Metadata is not storage geometry

`Field::DEGREE` is the extension degree over the characteristic, while `Field::ORDER` is the number of elements. The characteristic is read through `PrimeIdentity`. `FieldBuffer::STORAGE_BITS` describes a byte lane, not the logarithm of the order.

```rust
use fgf::{Field, FieldBuffer, Mersenne31, PrimeIdentity, QuadMersenne31};

let p = <Mersenne31 as Field>::Characteristic::CHARACTERISTIC;
assert_eq!(p, 2_147_483_647);
assert_eq!(Mersenne31::DEGREE, 1);
assert_eq!(Mersenne31::STORAGE_BITS, 32);
assert_eq!(QuadMersenne31::DEGREE, 2);
assert_eq!(QuadMersenne31::ORDER, u128::from(p) * u128::from(p));
```

GF(2), spelled `FiniteField<2, 1, Polynomial<3>>`, has scalar arithmetic but no `FieldBuffer` implementation. Its vectors use [bit-packed operations](Bit-packed-GF2). Degrees two and four are likewise scalar-only rather than byte-lane bulk fields.

## Const arithmetic and zero division

Concrete shipped element families provide inherent `const` methods such as `mul` and `inv`. Custom extension rows provide only the `const` identities and `to_raw`; their arithmetic goes through `FieldElem` and the operators. Generic trait calls and overloaded operators are ordinary runtime calls; an `F: Field` bound does not make the call const-evaluable.

Inversion and division are total: `inv(0) == 0` and `x / 0 == 0`. An algorithm requiring a multiplicative inverse must reject zero explicitly. Zero division does not signal an error on its own.

Next: [Choosing a field and basis](Choosing-a-field-and-basis).
