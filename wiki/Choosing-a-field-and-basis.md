# Field presentations and bases

Choose the coordinates required by the application or protocol. Equal field
orders do not make raw words interchangeable; the presentation is part of
the value's meaning.

## Start with the application's convention

For an existing protocol, match its characteristic, degree, and basis.
For a new format, record these conventions. Backend selection must not change
wire coordinates. A Reed–Solomon code alone does not specify its field
polynomial, coefficients, or generator schedule.

## Supported presentations

| Presentation | Binary degrees | Scalar arithmetic | Byte-lane `ops` |
| --- | --- | --- | --- |
| `Polynomial<P>` | 1, 2, 4, 8 | Yes | Degree 8 |
| `Normal<P, E>` | 2, 4, 8 | Yes | Degree 8, scalar fallback |
| `Cantor<P, SEED>` | 2, 4, 8 | Yes | Degree 8, scalar fallback |
| `Extension<S, 2>` over a binary base | 2, 4, 8, 16, 32, 64 | Yes | Degrees 8, 16, 32, 64 with `Packed` storage |

Defining parameters are checked; a listed shape does not make every polynomial,
normal element, seed, or relation valid. GF(2) uses `Polynomial<3>` for scalars
and `bits` for vectors. Binary degrees two and four are scalar-only.

For a prime characteristic, start with `FiniteField<P, 1>`. Mersenne31 and
Goldilocks select specialized representations of prime fields;
QuadMersenne31 is a quadratic extension. See [prime fields](Prime-fields).

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

The same raw product reduces differently under the two polynomials.
Copying coordinates is not a value-preserving conversion.

## Normal coordinates

`Normal<P, E>` starts from a polynomial-field element `E` and uses its conjugates `E, E^2, E^4, ...` as an ordered basis. Bit `i` selects `E^(2^i)`. The conjugates must form a full-rank basis. One is represented by all the live coordinate bits being set, not by the integer `1`.

```rust
use fgf::{AES, FiniteField, Elem, Isomorphism, Normal, Polynomial};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type NormalField = FiniteField<2, 8, Normal<AES, 0x20>>;
let change = Isomorphism::<PolynomialField, NormalField>::new().unwrap();
assert_eq!(Elem::<NormalField>::ONE.to_raw(), 0xff);
assert_eq!(change.apply(Elem::<PolynomialField>::ONE), Elem::<NormalField>::ONE);
let x = Elem::<PolynomialField>::from_raw(0x53);
assert_eq!(change.unapply(change.apply(x)), x);
```

`from_raw(1)` means the first basis vector. Application code uses `Elem::<F>::ONE` when it needs the multiplicative identity, independent of the basis.

## Cantor coordinates

`Cantor<P, SEED>` builds an ordered chain with `c_(N-1) = SEED` and `c_i = c_(i+1)^2 + c_(i+1)`. Bit `i` selects `c_i`. A valid chain reaches `c_0 = ONE` and spans the field.

```rust
use fgf::{AES, FiniteField, Cantor, Elem, Isomorphism, Polynomial};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type CantorField = FiniteField<2, 8, Cantor<AES, 0x20>>;
let change = Isomorphism::<PolynomialField, CantorField>::new().unwrap();
let x = Elem::<PolynomialField>::from_raw(0x53);
let y = Elem::<PolynomialField>::from_raw(0xca);
assert_eq!(change.apply(x * y), change.apply(x) * change.apply(y));
assert_eq!(change.apply(Elem::<PolynomialField>::ONE), Elem::<CantorField>::ONE);
```

Ordered bases supply coordinates and arithmetic, not an FFT algorithm.

## Extension components

A quadratic extension holds `a + b*t` over a base field. Low coordinate bits hold `a`, high coordinate bits hold `b`, and the specification's `Relation` fixes the relation for `t`. The base presentation is part of the construction.

`FiniteField<2, 16, Extension<Rijndael16, 2>>` roots at the AES field. `FiniteField<2, 16, Extension<FanPaar16, 2>>` uses the recursive Fan–Paar presentation. Both have the same order but different coordinates. [Quadratic towers](Quadratic-towers) explains their components and custom relations; [Embeddings](Embeddings-and-presentation-changes) converts values between them.

## Generators and acceleration are separate capabilities

Use `HasGenerator` only when an algorithm needs a selected multiplicative
generator; arithmetic itself does not need it. The generator need not be the
polynomial indeterminate, and custom rows need not supply one.

Continue with [packing](Coordinates-and-packed-bytes), [quadratic extensions](Quadratic-towers),
or [presentation changes](Embeddings-and-presentation-changes).
The [binary reference](https://docs.rs/fgf/3.0.2/fgf/field/binary/index.html) owns
parameter validity; [deployment](Features-and-deployment) explains acceleration.

