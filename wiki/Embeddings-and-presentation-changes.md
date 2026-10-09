# Embeddings and presentation changes

`Embedding<S, T>` prepares a value-preserving map between supported binary presentations when the source degree divides the target degree. Equal degrees give a change of presentation; a larger target gives a subfield inclusion.

This guide targets fgf 3.0.0. It builds on [Choosing a field and basis](Choosing-a-field-and-basis). The [embedding reference](https://docs.rs/fgf/3.0.0/fgf/field/binary/embedding/index.html) owns the canonical-map convention and method contracts.

## A smaller field is not a prefix of arbitrary coordinates

GF(2^4) occurs as a subfield of GF(2^8), but its polynomial indeterminate need not be the AES indeterminate. The canonical map sends the degree-four indeterminate to a root of its defining polynomial in the target.

```rust
use fgf::{AES, FiniteField, Elem, Embedding, Polynomial};

type Small = FiniteField<2, 4, Polynomial<0x13>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;
let inclusion = Embedding::<Small, Aes>::new().unwrap();
let x = Elem::<Small>::from_raw(2);
let image = inclusion.embed(x);
assert_eq!(image.to_raw(), 0x5c);
assert_eq!(inclusion.restrict(image), Some(x));
assert!(!inclusion.contains(Elem::<Aes>::from_raw(2)));
assert_eq!(inclusion.restrict(Elem::<Aes>::from_raw(2)), None);
```

The literal target byte `2` lies outside this subfield image. An unchecked raw copy would not preserve the source arithmetic. `restrict` answers membership and returns the source value only for elements in the image.

## Equal degree changes the presentation

A change from AES polynomial coordinates to RS polynomial coordinates preserves addition, multiplication, zero, and one. It is not a relabelling of raw bytes.

```rust
use fgf::{AES, FiniteField, Elem, Embedding, Polynomial, RS};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
let change = Embedding::<Aes, Rs>::new().unwrap();
let a = Elem::<Aes>::from_raw(0x53);
let b = Elem::<Aes>::from_raw(0xca);
assert_eq!(change.embed(a * b), change.embed(a) * change.embed(b));
assert_eq!(change.embed(Elem::<Aes>::ONE), Elem::<Rs>::ONE);
assert_eq!(change.restrict(change.embed(a)), Some(a));
```

The same surface handles normal, Cantor, and tower presentations. Equal-degree maps are bijective, so every target element has a source preimage. A larger-degree inclusion is injective but not surjective.

## One convention makes composed inclusions agree

An abstract pair of finite fields can admit several embeddings. `fgf` selects one convention through a frozen reference ladder: polynomial presentations at degrees one, two, and four; the AES field at degree eight; and Rijndael towers above it. Presentation changes factor through that ladder.

The consequence for applications is composability: embedding through an intermediate supported degree agrees with embedding directly. The reference's root selection and coordinate ordering are interoperability facts, not knobs for a caller to tune.

```rust
use fgf::{AES, Elem, Embedding, Extension, FiniteField, Polynomial, Rijndael16, Rijndael64};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Mid = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Wide = FiniteField<2, 64, Extension<Rijndael64, 2>>;
let first = Embedding::<Aes, Mid>::new().unwrap();
let second = Embedding::<Mid, Wide>::new().unwrap();
let direct = Embedding::<Aes, Wide>::new().unwrap();
let x = Elem::<Aes>::from_raw(0x53);
assert_eq!(second.embed(first.embed(x)), direct.embed(x));
```

## Relative Frobenius, trace, and norm

For source order `q` and relative extension degree `r`, relative Frobenius sends `z` to `z^q`. Relative trace sums `z, z^q, ..., z^(q^(r-1))`; relative norm multiplies them. Trace and norm return source-field elements, not merely target elements known to lie in the image.

In the Rijndael degree-sixteen relation `t² + t + 0x20 = 0`, the conjugate of `a + b*t` is `(a + b) + b*t`. Its trace is `b`; its norm is `a² + a*b + 0x20*b²`.

```rust
use fgf::{AES, Elem, Embedding, Extension, FiniteField, Polynomial, Rijndael16};

type Base = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let map = Embedding::<Base, Gf16>::new().unwrap();
let a = Elem::<Base>::from_raw(0x53);
let b = Elem::<Base>::from_raw(0xca);
let z = Elem::<Gf16>::from_components([a, b]);
assert_eq!(map.frobenius(z), Elem::<Gf16>::from_components([a + b, b]));
assert_eq!(map.frobenius(map.frobenius(z)), z);
assert_eq!(map.trace(z), b);
assert_eq!(map.norm(z), a.square() + a * b + Elem::<Base>::from_raw(0x20) * b.square());
assert_eq!(map.restrict(z), None);
```

The last assertion is not a failed conversion: `z` has a nonzero extension component, so it is not a base-field value. `Embedding<FiniteField<2, 1, Polynomial<3>>, T>` provides the absolute trace and norm to GF(2).

## Reuse and rejected requests

Construction and application use fixed-size prepared arrays and allocate nothing. A map can be retained across values and shared immutably. It prepares scalar coordinate maps, not an `ops` coefficient or a bulk conversion buffer.

A source degree larger than the target cannot divide it. Construction returns `EmbeddingError::IncompatibleDegree`; restriction in the opposite direction is performed through a valid smaller-to-larger map.

```rust
use fgf::{AES, Embedding, EmbeddingError, Extension, FiniteField, Polynomial, Rijndael16};

type Small = FiniteField<2, 8, Polynomial<AES>>;
type Large = FiniteField<2, 16, Extension<Rijndael16, 2>>;
assert_eq!(
    Embedding::<Large, Small>::new().unwrap_err(),
    EmbeddingError::IncompatibleDegree { source: 16, target: 8 },
);
```

The [complete embeddings example](https://github.com/nanithefkuc/fgf/blob/v3.0.0/examples/embeddings.rs) also covers cross-tower conversions and absolute trace. From a checkout, `just example embeddings` runs it.

Next: [Quadratic towers](Quadratic-towers), or [Coordinates and packed bytes](Coordinates-and-packed-bytes).
