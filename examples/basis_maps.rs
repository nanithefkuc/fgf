//! Basis presentations and the isomorphisms between them.
//!
//! Run with `just example basis_maps`, optionally with
//! `--no-default-features`. Normal and Cantor bases re-coordinate the same
//! polynomial fields: the multiplicative one is a field value, so a normal
//! basis stores it in the all-ones word rather than the coordinate word
//! `1`, and residues read the same under every basis while coordinate
//! indices do not. [`Isomorphism`] maps between equal-degree
//! presentations with a total inverse, from the canonical ladder or from
//! caller-supplied generator images. In odd characteristic nothing is
//! canonical: construction without images refuses, supplied images are
//! validated against the transported relations, and restriction through a
//! proper subfield rejects elements outside its image.

#![forbid(unsafe_code)]

use fgf::poly::AES;
use fgf::{
    Cantor, Components, Elem, Embedding, EmbeddingError, Extension, ExtensionField, ExtensionSpec,
    Field, FieldElem, FiniteField, Isomorphism, Normal, Polynomial, Relation,
};

// Callers name their own fields; the canonical `FiniteField<2, N, R>` type is the API.
type Aes = FiniteField<2, 8, Polynomial<AES>>;
type NormalAes = FiniteField<2, 8, Normal<AES, 0x20>>;
type CantorAes = FiniteField<2, 8, Cantor<AES, 0x20>>;
type Gf7 = FiniteField<7, 1>;

/// GF(49) as `t^2 - 3` over GF(7).
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = Gf7;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

/// GF(49) as `t^2 + 1` over GF(7): an unrelated presentation of the same
/// field, since `-1` is a non-residue modulo seven.
struct Gf49AltSpec;
impl ExtensionSpec<2> for Gf49AltSpec {
    type Base = Gf7;
    type Storage = Components;
    const NAME: &'static str = "GF(49), t^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type Gf49Alt = FiniteField<7, 2, Extension<Gf49AltSpec, 2>>;

fn main() {
    bases();
    binary_maps();
    odd_maps();
}

/// What a basis changes: the coordinate word of one.
fn bases() {
    // A normal basis holds the conjugate `E^(2^i)` in bit `i`, and the
    // conjugates of the basis element sum to the field one: the all-ones
    // word, not word `1`.
    let one = Elem::<NormalAes>::ONE;
    assert_eq!(one.to_raw(), 0xFF);
    assert_eq!(one.to_residue(), Some(1));
    assert_eq!(one.to_index(), 0xFF);
    assert_eq!(Elem::<NormalAes>::from_residue(1).unwrap(), one);
    assert_ne!(Elem::<NormalAes>::from_index(1).unwrap(), one);
    assert_eq!(Elem::<NormalAes>::from_index(1).unwrap().to_raw(), 1);

    // A Cantor chain runs down to the polynomial one, so this basis keeps
    // one in word `1`. Which word holds one is a basis fact; the residue
    // reading is not.
    let cantor_one = Elem::<CantorAes>::ONE;
    assert_eq!(cantor_one.to_raw(), 1);
    assert_eq!(cantor_one.to_residue(), Some(1));
    assert_eq!(Elem::<CantorAes>::from_residue(1).unwrap(), cantor_one);

    println!(
        "normal one has word {:#x}, cantor one has word {:#x}, both read residue 1",
        one.to_raw(),
        cantor_one.to_raw()
    );
}

/// The canonical isomorphism, its inverse, composition, and supplied
/// images in characteristic two.
fn binary_maps() {
    // The frozen ladder covers the ordered bases: one call prepares the
    // canonical map.
    let iso = Isomorphism::<NormalAes, Aes>::new().unwrap();
    let (g, h) = (
        Elem::<NormalAes>::GENERATOR,
        Elem::<NormalAes>::from_raw(0x96),
    );

    // `apply` is a field homomorphism and `unapply` its total inverse.
    assert_eq!(iso.apply(g * h), iso.apply(g) * iso.apply(h));
    assert_eq!(iso.unapply(iso.apply(h)), h);
    assert_eq!(iso.apply(Elem::<NormalAes>::ONE), Elem::<Aes>::ONE);

    // `inverse` swaps the directions; `then` composes by image transport.
    let back = iso.inverse();
    assert_eq!(back.apply(iso.apply(g)), g);
    let to_cantor = Isomorphism::<Aes, CantorAes>::new().unwrap();
    let composed = iso.then(&to_cantor);
    assert_eq!(composed.apply(g), to_cantor.apply(iso.apply(g)));
    assert_eq!(composed.inverse().apply(composed.apply(h)), h);

    // Stored images rebuild a map: `from_images` accepts `to_images`'s
    // order and reproduces the canonical map.
    let rebuilt = Isomorphism::<NormalAes, Aes>::from_images(iso.to_images()).unwrap();
    assert_eq!(rebuilt.apply(g), iso.apply(g));

    // A conjugate of the reference root satisfies the same polynomial, so
    // it fixes a second isomorphism between the same fields.
    let conjugate =
        Isomorphism::<NormalAes, Aes>::from_images(&[Elem::<Aes>::from_raw(4)]).unwrap();
    assert_eq!(
        conjugate.apply(g * h),
        conjugate.apply(g) * conjugate.apply(h)
    );
    assert_eq!(conjugate.unapply(conjugate.apply(g)), g);

    println!(
        "normal -> AES -> cantor composed; images rebuild the map; the conjugate root gives a second map"
    );
}

/// Odd characteristic: no canonical map, validated images, rejected
/// restriction.
fn odd_maps() {
    // Unrelated odd-characteristic presentations name no canonical map, and
    // an equal-degree `Embedding` is rejected outright: only an
    // `Isomorphism` maps between fields of the same degree.
    assert_eq!(
        Isomorphism::<Gf49Alt, Gf49>::new().unwrap_err(),
        EmbeddingError::NoCanonicalMap
    );
    assert_eq!(
        Embedding::<Gf49Alt, Gf49>::new().unwrap_err(),
        EmbeddingError::EqualDegree { degree: 2 }
    );

    // In `t^2 = 3`, the element `3t` squares to `2 * 3 = 6 = -1`, so it is
    // a root of the alternative relation `u^2 + 1 = 0` and fixes an
    // isomorphism.
    let root =
        <Gf49 as ExtensionField<2>>::from_components([Elem::<Gf7>::ZERO, Elem::<Gf7>::from_raw(3)]);
    let mapped = Isomorphism::<Gf49Alt, Gf49>::from_images(&[root]).unwrap();
    let (p, q) = (
        <Gf49Alt as ExtensionField<2>>::from_components([
            Elem::<Gf7>::from_raw(2),
            Elem::<Gf7>::from_raw(5),
        ]),
        <Gf49Alt as ExtensionField<2>>::from_components([
            Elem::<Gf7>::from_raw(6),
            Elem::<Gf7>::from_raw(1),
        ]),
    );
    assert_eq!(mapped.apply(p * q), mapped.apply(p) * mapped.apply(q));
    assert_eq!(mapped.unapply(mapped.apply(p)), p);

    // Images that fail the transported relation, or the wrong count, are
    // rejected: the indeterminate squares to 3, not to -1.
    let t = <Gf49 as ExtensionField<2>>::from_components([Elem::<Gf7>::ZERO, Elem::<Gf7>::ONE]);
    assert_eq!(
        Isomorphism::<Gf49Alt, Gf49>::from_images(&[t]).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    assert_eq!(
        Isomorphism::<Gf49Alt, Gf49>::from_images(&[]).unwrap_err(),
        EmbeddingError::InvalidImages
    );

    // Restriction through the prime subfield is total on its image alone:
    // the indeterminate lies outside GF(7).
    let subfield = Embedding::<Gf7, Gf49>::new().unwrap();
    assert_eq!(subfield.restrict(t), None);
    assert_eq!(
        subfield.restrict(subfield.embed(Elem::<Gf7>::from_raw(5))),
        Some(Elem::<Gf7>::from_raw(5))
    );

    println!(
        "{} <- {}: caller image {} validates, the indeterminate and the empty list reject",
        Gf49::NAME,
        Gf49Alt::NAME,
        root
    );
}
