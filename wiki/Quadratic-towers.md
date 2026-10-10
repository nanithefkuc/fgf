# Quadratic towers

A quadratic extension builds a larger field from pairs of base-field elements. Write a value as `a + b*t`, where a relation specifies how to reduce `t²`. Repeating the construction produces a tower; the shipped binary towers reach degree 64.

The examples use fgf 3.0.1. A tower type is `FiniteField<2, N, Extension<Spec, 2>>`. The [tower reference](https://docs.rs/fgf/3.0.1/fgf/field/binary/tower/index.html) lists the shipped specifications, and the [extension reference](https://docs.rs/fgf/3.0.1/fgf/field/extension/index.html) describes custom relations and storage.

## Components are values in the base presentation

For a degree-sixteen tower over a byte field, the low byte contains `a` and the high byte contains `b`. The stable encoding is little-endian, so the constant component comes first. Components are typed base-field elements, not interchangeable bytes from another basis.

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

`Rijndael16` roots at `FiniteField<2, 8, Polynomial<AES>>`; `Rijndael32` and `Rijndael64` continue that tower. `FanPaar2` through `FanPaar64` nest from GF(2) through the Fan–Paar relation. Each marker is a specification, used inside `FiniteField<2, N, Extension<Spec, 2>>`, rather than a standalone field type.

Use a local alias such as `type F = FiniteField<2, 16, Extension<Rijndael16, 2>>` for repeated code. Each shipped specification fixes its relation and selected generator. The shipped rows provide `const` constructors and arithmetic, with component conversions on the 16-, 32-, and 64-bit instances. Rijndael and Fan–Paar coordinates are different even at equal degree; [embeddings](Embeddings-and-presentation-changes) convert values between them.

## Define a custom relation

A custom specification declares a base row, a storage, a descriptive name, and the relation. Over a binary base, `Relation::from_binary_words` takes one raw base word per coefficient, constant term first, so `[B, A]` names `t² + A*t + B`. `Packed<W>` storage keeps the coordinate word in an integer `W` whose width equals the extension degree and gives the row a little-endian byte form; `PackedPadded<W>` keeps a narrower coordinate word in a padded integer and has no byte form. The example below uses the RS byte field as its base.

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

A custom row carries the `const` identities `Elem::<F>::ZERO` and `Elem::<F>::ONE` and a `const` `to_raw`, but no `const` element arithmetic and no inherent constructors. Construction and arithmetic go through the `FieldElem` trait, the operators, `ExtensionField::from_components` and `to_components`, and `FieldBuffer` when the storage has a byte form.

The relation is not valid merely because its coefficients fit. Validation checks the base, that the extension degree is twice the base degree, that every coefficient word fits the base, that the storage admits the degree, and that the quadratic is irreducible over the base.

Declaring a type does not by itself validate the relation. Using an identity or a value-producing operation forces validation of its defining data; invalid constants are rejected at compile time. Binary tower rows stop at degree 64.

## A generator is optional

Each shipped specification carries a selected generator, read as `Elem::<F>::GENERATOR` under a `HasGenerator` bound. A custom specification declares no generator: arithmetic, component conversion, and embeddings need none. `HasGenerator` is sealed, so a custom row offers no `GENERATOR`.

The tower indeterminate `t` is not automatically a multiplicative generator. Algorithms that need one add a `HasGenerator` bound and use a shipped row. Arithmetic-only applications need not search for or declare a generator.

## Bulk operations and application boundaries

Rows with `Packed` storage have a byte form at degrees 8, 16, 32, and 64 and implement the packed `ops` surface. Routing follows the presentation, not the spelling of a specification marker: a custom specification whose base, relation, and storage equal a shipped specification's runs that row's arithmetic and kernels, while every other custom row uses the typed scalar fallback. `backend_for::<F>()` reports the field's backend. Fan–Paar degrees two and four keep a padded byte word and have no byte-lane surface.

`ExtensionSpec` is not limited to binary quadratic towers: its base is any row fgf implements, and its relative degree is any `D` that `Extension<Spec, D>` names. `Components` storage keeps one base element per component and suits a prime base. [Downstream scalar fields](Downstream-scalar-fields) covers the sealed scalar rows and the residue row for custom prime fields.

The [extension-fields example](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/extension_fields.rs) compares binary towers with `QuadMersenne31`.
