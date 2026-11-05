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
use fgf::{FiniteField, Elem};
use fgf::binary::{AES, Polynomial};
use fgf::maps::Embedding;

type Small = FiniteField<2, 4, Polynomial<0x13>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;
let inclusion = Embedding::<Small, Aes>::new();
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
use fgf::{FiniteField, Elem};
use fgf::binary::{AES, Polynomial, RS};
use fgf::maps::Isomorphism;

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
let change = Isomorphism::<Aes, Rs>::new();
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
use fgf::{Elem, FiniteField};
use fgf::binary::{AES, Polynomial, Rijndael16, Rijndael64};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Mid = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Wide = FiniteField<2, 64, Extension<Rijndael64, 2>>;
let first = Embedding::<Aes, Mid>::new();
let second = Embedding::<Mid, Wide>::new();
let direct = Embedding::<Aes, Wide>::new();
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
use fgf::{Elem, FiniteField};
use fgf::binary::{AES, Polynomial, RS, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::{Embedding, Isomorphism};

type Nibble = FiniteField<2, 4, Polynomial<0x13>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Rs = FiniteField<2, 8, Polynomial<RS>>;
type Tower = FiniteField<2, 16, Extension<Rijndael16, 2>>;

let lift = Embedding::<Nibble, Aes>::new();
let swap = Isomorphism::<Aes, Rs>::new();
let raise = Embedding::<Rs, Tower>::new();

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
use fgf::{Elem, FiniteField};
use fgf::binary::{AES, Polynomial, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type Base = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let map = Embedding::<Base, Gf16>::new();
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
images when a canonical map is unavailable, and no canonical map is required:
any image sequence that satisfies the transported relations fixes a map.

Degree contracts reject at code generation, not at runtime: an `Embedding`
needs a source degree that strictly divides the target degree, and an
`Isomorphism` needs equal degrees, so an impossible pair fails the build
(`cargo build` and `cargo test` catch it; `cargo check` alone may skip it).
On a pair inside the contracts that no structural or ladder rule covers,
canonical construction panics instead of guessing; restrict through a valid
smaller-to-larger map rather than constructing the reverse inclusion:

```rust
use fgf::{Elem, FiniteField};
use fgf::binary::{AES, Polynomial};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type Aes = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<fgf::binary::Rijndael16, 2>>;
let tower = Embedding::<Aes, Gf16>::new();
let byte = tower.restrict(Elem::<Gf16>::from_components([
    Elem::<Aes>::from_raw(0x53),
    Elem::<Aes>::ZERO,
])).unwrap();

// A wrong-count image sequence rejects with `ImageError` as a value.
assert!(Embedding::<Aes, Gf16>::from_images(&[]).is_err());
```

Run the [complete example](https://github.com/nanithefkuc/fgf/blob/v3.0.2/examples/embeddings.rs)
with `just example embeddings`, and the
[basis-maps example](https://github.com/nanithefkuc/fgf/blob/v3.0.2/examples/basis_maps.rs)
with `just example basis_maps` for the ordered-basis presentations. The
[embedding reference](https://docs.rs/fgf/3.0.2/fgf/field/embedding/index.html)
owns map construction and error contracts.
