# Choosing a field and basis

A field choice fixes both the algebra and the coordinates used to represent values. Interoperability requires agreement on that presentation, not only the number of elements.

This guide targets fgf 3.0.1. The [binary field reference](https://docs.rs/fgf/3.0.1/fgf/field/binary/index.html) lists the representation contracts.

## Start with the application's convention

An existing protocol or algebraic construction determines the characteristic, degree, and often the basis. A new application can choose a presentation, but must record that choice wherever another process reconstructs the same values. Backend availability is a separate deployment question; it does not justify silently changing wire coordinates.

Common choices are `FiniteField<2, 8, Polynomial<RS>>` for the `0x11D` convention and `FiniteField<2, 8, Polynomial<AES>>` for the `0x11B` convention. A Reed–Solomon code is not enough to determine a polynomial or a generator schedule. Matching the field does not by itself match a codec's coefficients or wire protocol.

## Supported presentations

| Presentation | Binary degrees | Scalar arithmetic | Byte-lane `ops` |
| --- | --- | --- | --- |
| `Polynomial<P>` | 1, 2, 4, 8 | Yes | Degree 8 |
| `Normal<P, E>` | 2, 4, 8 | Yes | Degree 8, scalar fallback |
| `Cantor<P, SEED>` | 2, 4, 8 | Yes | Degree 8, scalar fallback |
| `Extension<S, 2>` over a binary base | 2, 4, 8, 16, 32, 64 | Yes | Degrees 8, 16, 32, 64 with `Packed` storage |

The presentation parameters must pass their validity checks. This table does not mean that every integer polynomial, normal element, seed, or tower relation is valid. Polynomial degree one is specifically `Polynomial<3>`. `bits` supplies the vector surface for GF(2); degrees two and four have no packed arithmetic surface.

The shipped nonbinary families are `Mersenne31`, `Goldilocks`, and `QuadMersenne31`, all with scalar and byte-buffer arithmetic. `QuadMersenne31` is a quadratic extension of a prime field, not itself a prime field. Their coordinate rules are explained in [Prime fields](Prime-fields).

## Polynomial coordinates

In `FiniteField<2, N, Polynomial<P>>`, bit `i` selects the coefficient of `x^i`. Addition XORs those coefficients; multiplication reduces the polynomial product modulo `P`. The polynomial includes its leading term and must be irreducible of degree `N`.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, RS};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
let aes_product = Elem::<Aes>::from_raw(0x80) * Elem::<Aes>::from_raw(2);
let rs_product = Elem::<Rs>::from_raw(0x80) * Elem::<Rs>::from_raw(2);
assert_eq!(aes_product.to_raw(), 0x1b);
assert_eq!(rs_product.to_raw(), 0x1d);
```

Here `0x80` denotes `x^7` and `2` denotes `x`. Their product is reduced using different defining polynomials. Copying the raw byte into the other type changes the value's interpretation; it is not a field conversion.

## Normal coordinates

`Normal<P, E>` starts from a polynomial-field element `E` and uses its conjugates `E, E^2, E^4, ...` as an ordered basis. Bit `i` selects `E^(2^i)`. The conjugates must form a full-rank basis. One is represented by all the live coordinate bits being set, not by the integer `1`.

```rust
use fgf::{AES, FiniteField, Elem, Embedding, Normal, Polynomial};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type NormalField = FiniteField<2, 8, Normal<AES, 0x20>>;
let change = Embedding::<PolynomialField, NormalField>::new().unwrap();
assert_eq!(Elem::<NormalField>::ONE.to_raw(), 0xff);
assert_eq!(change.embed(Elem::<PolynomialField>::ONE), Elem::<NormalField>::ONE);
let x = Elem::<PolynomialField>::from_raw(0x53);
assert_eq!(change.restrict(change.embed(x)), Some(x));
```

`from_raw(1)` means the first basis vector. Application code uses `Elem::<F>::ONE` when it needs the multiplicative identity, independent of the basis.

## Cantor coordinates

`Cantor<P, SEED>` builds an ordered chain with `c_(N-1) = SEED` and `c_i = c_(i+1)^2 + c_(i+1)`. Bit `i` selects `c_i`. A valid chain reaches `c_0 = ONE` and spans the field.

```rust
use fgf::{AES, FiniteField, Cantor, Elem, Embedding, Polynomial};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type CantorField = FiniteField<2, 8, Cantor<AES, 0x20>>;
let change = Embedding::<PolynomialField, CantorField>::new().unwrap();
let x = Elem::<PolynomialField>::from_raw(0x53);
let y = Elem::<PolynomialField>::from_raw(0xca);
assert_eq!(change.embed(x * y), change.embed(x) * change.embed(y));
assert_eq!(change.embed(Elem::<PolynomialField>::ONE), Elem::<CantorField>::ONE);
```

An ordered basis does not provide an FFT or a polynomial evaluation algorithm. `fgf` owns the field representation and arithmetic; transform algorithms belong to their own libraries.

## Extension components

A quadratic extension holds `a + b*t` over a base field. Low coordinate bits hold `a`, high coordinate bits hold `b`, and the specification's `Relation` fixes the relation for `t`. The base presentation is part of the construction.

`FiniteField<2, 16, Extension<Rijndael16, 2>>` roots at the AES field. `FiniteField<2, 16, Extension<FanPaar16, 2>>` uses the recursive Fan–Paar presentation. Both have the same order but different coordinates. [Quadratic towers](Quadratic-towers) explains their components and custom relations; [Embeddings](Embeddings-and-presentation-changes) converts values between them.

## Generators and acceleration are separate capabilities

A selected multiplicative generator is not the same thing as the polynomial indeterminate or an extension indeterminate. Algorithms that enumerate nonzero elements use `HasGenerator`, which custom extension rows do not implement; extension arithmetic itself requires no selected generator.

The [deployment guide](Features-and-deployment) explains backend queries. Shipped presentations can have architecture kernels, while ordered-basis presentations and custom extension rows use a typed scalar fallback unless their base, relation, and storage match a shipped instance. Representation selection is not a promise that every operation executes SIMD.

Next: [Coordinates and packed bytes](Coordinates-and-packed-bytes), or [Embeddings and presentation changes](Embeddings-and-presentation-changes).
