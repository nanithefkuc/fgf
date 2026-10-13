# Embeddings and presentation changes

Use `Embedding<S, T>` to preserve values when changing field types.
Equal degrees give a presentation change; a larger target can contain the
source as a subfield. `Isomorphism<S, T>` supplies a total inverse in the
equal-degree case. Raw-coordinate copying is not a conversion.

## A smaller field is not a prefix of arbitrary coordinates

Embed the smaller field through a root of its defining polynomial in the
target, not by treating its coordinates as a byte prefix:

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

`restrict` returns a source value only inside the image; target raw `2` is
outside this particular subfield.

## Equal degree changes the presentation

AES-to-RS conversion preserves arithmetic, not raw bytes:

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

The same surface handles normal, Cantor, and extension presentations. Equal-degree maps are bijective, so every target element has a source preimage. A larger-degree inclusion is injective but not surjective.

## One convention makes composed inclusions agree

Canonical maps agree through intermediate supported fields. An extension's
base embeds as constant components; other binary maps follow a fixed
reference convention. Root and coordinate choices are interoperability rules,
not tuning options:

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

`z` has a nonzero extension component, so restriction returns `None`.
Embedding from polynomial GF(2) gives absolute trace and norm.

## Reuse and rejected requests

Prepare a map once and reuse it. Construction and application use fixed-size
arrays without allocation. `from_images` checks caller-supplied generator
images when a canonical map is unavailable.

Source degree must divide target degree. Restrict through a valid
smaller-to-larger map rather than constructing the reverse inclusion:

```rust
use fgf::{AES, Embedding, EmbeddingError, Extension, FiniteField, Polynomial, Rijndael16};

type Small = FiniteField<2, 8, Polynomial<AES>>;
type Large = FiniteField<2, 16, Extension<Rijndael16, 2>>;
assert_eq!(
    Embedding::<Large, Small>::new().unwrap_err(),
    EmbeddingError::IncompatibleDegree { source: 16, target: 8 },
);
```

Run the [complete example](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/embeddings.rs)
with `just example embeddings`. The
[embedding reference](https://docs.rs/fgf/3.0.1/fgf/field/embedding/index.html)
owns map construction and error contracts.

