# Embeddings and presentation changes

Use `Embedding<S, T>` for a proper inclusion of a smaller field in a
larger target, and `Isomorphism<S, T>` for a presentation change at
equal degree with a total inverse. The inclusion pairs `embed` with the
partial `restrict`; the isomorphism pairs `apply` with the total
`unapply`. The two maps carry different degree contracts and convert
into no common form. Raw-coordinate copying is not a conversion.

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
use fgf::{AES, FiniteField, Elem, Isomorphism, Polynomial, RS};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
let change = Isomorphism::<Aes, Rs>::new().unwrap();
let a = Elem::<Aes>::from_raw(0x53);
let b = Elem::<Aes>::from_raw(0xca);
assert_eq!(change.apply(a * b), change.apply(a) * change.apply(b));
assert_eq!(change.apply(Elem::<Aes>::ONE), Elem::<Rs>::ONE);
assert_eq!(change.unapply(change.apply(a)), a);
```

The same surface handles normal, Cantor, and extension presentations. Equal-degree maps are bijective, so every target element has a source preimage; `inverse` returns the reversed isomorphism from the same prepared images. A larger-degree inclusion is injective but not surjective, and no inclusion or isomorphism converts into the other.

## One convention makes composed inclusions agree

Canonical maps agree through intermediate supported fields. An extension's
base embeds as constant components; other binary maps follow a fixed
reference convention. Root and coordinate choices are interoperability rules,
not tuning options. `then` composes by image transport, so the composed
inclusion equals the direct one:

```rust
use fgf::{AES, Elem, Embedding, Extension, FiniteField, Polynomial, Rijndael16, Rijndael64};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Mid = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Wide = FiniteField<2, 64, Extension<Rijndael64, 2>>;
let first = Embedding::<Aes, Mid>::new().unwrap();
let second = Embedding::<Mid, Wide>::new().unwrap();
let direct = Embedding::<Aes, Wide>::new().unwrap();
let composed = first.then(&second);
let x = Elem::<Aes>::from_raw(0x53);
assert_eq!(composed.embed(x), direct.embed(x));
```

## Mixed composition keeps one verb

`Embedding::then` accepts a proper inclusion or an isomorphism out of the
middle row, through the sealed `Postcompose` trait, so a presentation change
at the far end still yields an `Embedding`. `Isomorphism::then_embedding`
composes the other order, an isomorphism into a proper inclusion;
`Isomorphism::then` composes two isomorphisms:

```rust
use fgf::{AES, Elem, Embedding, Extension, FiniteField, Isomorphism, Polynomial, RS, Rijndael16};

type Nibble = FiniteField<2, 4, Polynomial<0x13>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
type Tower = FiniteField<2, 16, Extension<Rijndael16, 2>>;

let lift = Embedding::<Nibble, Aes>::new().unwrap();
let swap = Isomorphism::<Aes, Rs>::new().unwrap();
let raise = Embedding::<Rs, Tower>::new().unwrap();

let into_rs = lift.then(&swap);
let nibble = Elem::<Nibble>::from_raw(2);
assert_eq!(into_rs.embed(nibble), swap.apply(lift.embed(nibble)));

let from_aes = swap.then_embedding(&raise);
let byte = Elem::<Aes>::from_raw(0x53);
assert_eq!(from_aes.embed(byte), raise.embed(swap.apply(byte)));
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

An `Embedding` needs a source degree that strictly divides the target
degree, and an `Isomorphism` needs equal degrees; each constructor rejects
the pair the other owns, reporting `EqualDegree` and `IncompatibleDegree`
respectively. Restrict through a valid smaller-to-larger map rather than
constructing the reverse inclusion:

```rust
use fgf::{AES, Embedding, EmbeddingError, Extension, FiniteField, Isomorphism, Polynomial, RS, Rijndael16};

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
type Large = FiniteField<2, 16, Extension<Rijndael16, 2>>;
assert_eq!(
    Embedding::<Aes, Rs>::new().unwrap_err(),
    EmbeddingError::EqualDegree { degree: 8 },
);
assert_eq!(
    Isomorphism::<Aes, Large>::new().unwrap_err(),
    EmbeddingError::IncompatibleDegree { source: 8, target: 16 },
);
assert_eq!(
    Embedding::<Large, Aes>::new().unwrap_err(),
    EmbeddingError::IncompatibleDegree { source: 16, target: 8 },
);
```

Run the [complete example](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/embeddings.rs)
with `just example embeddings`. The
[embedding reference](https://docs.rs/fgf/3.0.1/fgf/field/embedding/index.html)
owns map construction and error contracts.

