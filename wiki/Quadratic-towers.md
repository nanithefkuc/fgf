# Quadratic towers

Represent an extension value as `a + b*t` over a base field, with a relation
for reducing `t²`. `FiniteField<2, N, Extension<Spec, 2>>` selects a binary
quadratic extension; repeating it produces a tower.

## Components are values in the base presentation

In a degree-sixteen packed tower, the low byte holds `a` and the high byte
holds `b`. Components belong to the declared base presentation:

```rust
use fgf::{AES, Elem, Extension, FiniteField, Polynomial, Rijndael16};

type Base = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let a = Elem::<Base>::from_raw(0x53);
let b = Elem::<Base>::from_raw(0xca);
let z = Elem::<Gf16>::from_components([a, b]);
assert_eq!(z.to_components(), [a, b]);
assert_eq!(z.to_raw(), 0xca53);
assert_eq!(z.to_bytes(), [0x53, 0xca]);
assert_eq!(Elem::<Gf16>::ONE.to_components(), [Elem::<Base>::ONE, Elem::<Base>::ZERO]);
```

The same low/high component convention recurses at wider degrees. Raw `1` is not a universal identity if the base uses an ordered basis; the tower embeds the base's `ONE` in its low component.

## Deriving arithmetic from the relation

Characteristic two gives `t² = A*t + B`. Expanding a product and replacing `t²` yields:

```text
(a + b*t)(c + d*t) = (a*c + B*b*d) + (a*d + b*c + A*b*d)*t
(a + b*t)^2       = (a^2 + B*b^2) + (A*b^2)*t
conjugate(a+b*t)  = (a + A*b) + b*t
norm(a+b*t)       = a^2 + A*a*b + B*b^2
```

Every coefficient operation in these equations belongs to the base field. In particular, `A` and `B` are coordinates in the declared base, not ordinary integers added to a Rust word.

## Shipped specifications

`Rijndael16` starts at the AES byte field; `Rijndael32` and `Rijndael64`
continue its tower. `FanPaar2` through `FanPaar64` nest from GF(2).
These names are specifications, not standalone field markers.

Shipped rows provide `const` arithmetic and selected generators, with component
conversions on the 16-, 32-, and 64-bit instances. Equal-degree Rijndael and
Fan–Paar values need an [embedding](Embeddings-and-presentation-changes).

## Define a custom relation

Supply the base, storage, name, and constant-first relation.
`[B, A]` specifies `t² + A*t + B` over a binary base.

| Storage | Coordinates | Byte form |
| --- | --- | --- |
| `Packed<W>` | Integer word matching the binary degree | Little-endian lanes at supported buffer degrees |
| `PackedPadded<W>` | Narrower coordinates in a padded integer | Scalar only |
| `Components` | Individual base elements | Depends on the base |

This example uses the RS byte field and packed storage:

```rust
use fgf::{
    Elem, Embedding, Extension, ExtensionField, ExtensionSpec, FiniteField, Packed, Polynomial,
    RS, Relation,
};

type Base = FiniteField<2, 8, Polynomial<RS>>;

struct RsQuadratic;
impl ExtensionSpec<2> for RsQuadratic {
    type Base = Base;
    type Storage = Packed<u16>;
    const NAME: &'static str = "RS quadratic tower";
    // `t^2 + 2*t + 0x80`: the constant coefficient first, then the linear one.
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0x80, 2]);
}

type Gf16 = FiniteField<2, 16, Extension<RsQuadratic, 2>>;
let pair = <Gf16 as ExtensionField<2>>::from_components;
let a = Elem::<Base>::from_raw(0x53);
let b = Elem::<Base>::from_raw(0xca);
let linear = Elem::<Base>::from_raw(2);
let constant = Elem::<Base>::from_raw(0x80);
let z = pair([a, b]);
let conjugate = pair([a + linear * b, b]);
let expected_norm = a.square() + linear * a * b + constant * b.square();
assert_eq!(z * conjugate, pair([expected_norm, Elem::<Base>::ZERO]));
let base_map = Embedding::<Base, Gf16>::new().unwrap();
assert_eq!(base_map.trace(z), linear * b);
assert_eq!(base_map.norm(z), expected_norm);
```

Custom rows use trait arithmetic and component conversion, with `const`
identities and `to_raw`. Using a value or identity forces validation of the
base, storage, degree, coordinates, and irreducible relation; an alias alone
does not. Binary packed towers stop at degree 64.

## A generator is optional

`HasGenerator` exposes a selected generator on shipped rows. Custom rows
declare none; arithmetic, component conversion, and embeddings need none.
The indeterminate `t` is not automatically primitive.

## Bulk operations and application boundaries

Packed binary rows at degrees 8, 16, 32, and 64 support `ops`.
Routing follows base, relation, and storage: matching a shipped instance uses
its kernels; other custom rows use typed scalar fallback.

For arbitrary relative degree and a prime base, continue with
[custom extensions](Downstream-scalar-fields). The
[tower reference](https://docs.rs/fgf/3.0.1/fgf/field/binary/tower/index.html) and
[extension reference](https://docs.rs/fgf/3.0.1/fgf/field/extension/index.html)
own the specification and storage contracts.
