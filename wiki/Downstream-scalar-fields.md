# Custom prime fields and extensions

Custom fields use the same `Elem<F>` as shipped fields. Choose parameters for
`FiniteField`; do not implement the sealed arithmetic traits.

## Decide what needs to vary

| Need | Choice |
| --- | --- |
| A new prime modulus | `FiniteField<P, 1>`; see [prime fields](Prime-fields) |
| A binary polynomial or ordered basis | A supported [presentation](Choosing-a-field-and-basis) |
| An extension relation or storage representation | `ExtensionSpec<D>` with `Extension<Spec, D>` |

An extension specification supplies a base field, storage, name, and monic
relation. The field's absolute degree is the base degree multiplied by the
relative degree `D`.

## Define an extension over a prime base

This constructs GF(49) as GF(7)[t]/(t² - 3). The relation omits the monic
leading coefficient and lists the remaining coordinates constant-first:

```rust
use fgf::{Components, Elem, Extension, ExtensionField, ExtensionSpec,
          FieldElem, FiniteField, Relation};

type Base = FiniteField<7, 1>;
struct Quadratic;
impl ExtensionSpec<2> for Quadratic {
    type Base = Base;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type F = FiniteField<7, 2, Extension<Quadratic, 2>>;
let pair = <F as ExtensionField<2>>::from_components;
let t = pair([Elem::<Base>::ZERO, Elem::<Base>::ONE]);
assert_eq!(t * t, pair([Elem::<Base>::from_raw(3), Elem::<Base>::ZERO]));
let x = pair([Elem::<Base>::from_raw(2), Elem::<Base>::from_raw(3)]);
assert_eq!(x * x.inv(), Elem::<F>::ONE);
```

`Components` stores base elements individually. For compact binary words,
[quadratic towers](Quadratic-towers) uses `Packed<W>`; padded binary storage
uses `PackedPadded<W>`. Byte encoding and packed arithmetic depend on the
chosen storage and base, not merely the existence of a scalar raw word.

## Validate before application use

Value-producing operations and identities force validation. The characteristic
must be prime, the degree and order consistent, the storage suitable, and the
extension relation irreducible with valid base coordinates. A type alias alone
does not validate its parameters.

Custom extensions have `const` identities and `to_raw`; their arithmetic and
component conversion use traits and operators, not shipped-family inherent
`const` methods. Import `FieldElem` for trait methods on elements and
`ExtensionField` for component conversion.

## Request only supplied capabilities

Generic residue rows and custom extensions do not supply `HasGenerator`.
Arithmetic needs no selected generator. Algorithms requiring one must choose a
row with that capability rather than assuming the extension indeterminate is
primitive.

Packed routing follows the representation: custom specifications matching a
shipped base, relation, and storage can use its kernels; other custom rows use
typed scalar fallback. [Deployment](Features-and-deployment) explains backend
queries and feature choices.

The [extension reference](https://docs.rs/fgf/3.0.2/fgf/field/extension/index.html)
owns validity and storage contracts. Application checks should also cover
expected relations, nonzero inverses, and agreed wire encodings. The
[custom-extension example](https://github.com/nanithefkuc/fgf/blob/v3.0.2/examples/custom_extensions.rs)
runs this surface as a program: `just example custom_extensions`.
