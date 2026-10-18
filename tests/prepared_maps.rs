//! Prepared embedding and isomorphism acceptance: the compiled scalar
//! surface against the compact maps.
//!
//! Every valid image tuple of the `tests/relations.rs` inventory runs
//! through the prepared types: each method agrees with its compact
//! counterpart, `None`-agreement holds on every non-member, and the
//! generator-level counts match. The structural lift, the conjugate-image
//! pair, the normal-basis injection trap, the characteristic-divisible relative-degree case,
//! the Frobenius equality with `Frobenius<F>`, and the QM31 anchors
//! follow, with the GF(3^64) tower covering the at-cap odd row.
//!
//! Row-function equality (decode-method-encode at every lane count,
//! bitmap discipline, geometry panics, zero allocation) lives in
//! `tests/prepared_map_rows.rs`; this file pins the scalar surface those
//! rows decode into.

#![cfg(feature = "alloc")]

extern crate alloc;
use fgf::poly::{AES, RS};
use fgf::{
    Cantor, Components, Elem, Embedding, Extension, ExtensionSpec, Field, FieldElem, FiniteField,
    Frobenius, Isomorphism, Normal, Polynomial, PreparedEmbedding, PreparedIsomorphism,
    QuadMersenne31, Relation, Residue,
};

// ---------------------------------------------------------------------------
// Shared inventory rows (mirroring `tests/relations.rs`)
// ---------------------------------------------------------------------------

type R3 = FiniteField<3, 1, Residue>;
type R5 = FiniteField<5, 1, Residue>;
type R7 = FiniteField<7, 1, Residue>;

struct G9Spec;
impl ExtensionSpec<2> for G9Spec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as x^2 + 2x + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 2]);
}
type G9 = FiniteField<3, 2, Extension<G9Spec, 2>>;

struct G81QuarticSpec;
impl ExtensionSpec<4> for G81QuarticSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as x^4 + 2x^3 + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
}
type G81Quartic = FiniteField<3, 4, Extension<G81QuarticSpec, 4>>;

struct G9AltSpec;
impl ExtensionSpec<2> for G9AltSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as a^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type G9Alt = FiniteField<3, 2, Extension<G9AltSpec, 2>>;

struct G81ExtSpec;
impl ExtensionSpec<2> for G81ExtSpec {
    type Base = G9Alt;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as t^2 - (1 + a)";
    // `-(1 + a)`: constant `2 + 2a`, linear zero.
    const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
}
type G81Ext = FiniteField<3, 4, Extension<G81ExtSpec, 2>>;

struct G25Spec;
impl ExtensionSpec<2> for G25Spec {
    type Base = R5;
    type Storage = Components;
    const NAME: &'static str = "GF(25) as x^2 + 4x + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 4]);
}
type G25 = FiniteField<5, 2, Extension<G25Spec, 2>>;

struct G625Spec;
impl ExtensionSpec<4> for G625Spec {
    type Base = R5;
    type Storage = Components;
    const NAME: &'static str = "GF(625) as x^4 + 4x^2 + 4x + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 4, 4, 0]);
}
type G625 = FiniteField<5, 4, Extension<G625Spec, 4>>;

struct G49OddSpec;
impl ExtensionSpec<2> for G49OddSpec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "GF(49) as x^2 - 3";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type G49Odd = FiniteField<7, 2, Extension<G49OddSpec, 2>>;

struct G2401ExtSpec;
impl ExtensionSpec<2> for G2401ExtSpec {
    type Base = G49Odd;
    type Storage = Components;
    const NAME: &'static str = "GF(7^4) as t^2 - (1 + a)";
    // `1 + a` is a non-square in GF(49): its norm is five, a non-square
    // modulo seven. The constant is `-(1 + a)`, linear zero.
    const RELATION: Relation = Relation::from_coordinates(&[6, 6, 0, 0]);
}
type G2401Ext = FiniteField<7, 4, Extension<G2401ExtSpec, 2>>;

struct G343Spec;
impl ExtensionSpec<3> for G343Spec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "GF(343) as x^3 - 2";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}
type G343 = FiniteField<7, 3, Extension<G343Spec, 3>>;

type Poly2 = FiniteField<2, 2, Polynomial<0x7>>;
type Poly4 = FiniteField<2, 4, Polynomial<0x13>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;

struct G16ExtSpec;
impl ExtensionSpec<2> for G16ExtSpec {
    type Base = Poly2;
    type Storage = Components;
    const NAME: &'static str = "GF(16) tower over GF(4)";
    // `t^2 + t + 2`: two has trace one over GF(2), so the relation is
    // irreducible (constant one would split: `t = 2` is a root).
    const RELATION: Relation = Relation::from_binary_words::<Poly2>(&[2, 1]);
}
type G16Ext = FiniteField<2, 4, Extension<G16ExtSpec, 2>>;

struct G4Spec;
impl ExtensionSpec<2> for G4Spec {
    type Base = FiniteField<2, 1, Polynomial<3>>;
    type Storage = Components;
    const NAME: &'static str = "GF(4)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
}
type G4 = FiniteField<2, 2, Extension<G4Spec, 2>>;

type Gf16 = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;
type Gf64 = FiniteField<2, 64, Extension<fgf::Rijndael64, 2>>;
type Fp64 = FiniteField<2, 64, Extension<fgf::FanPaar64, 2>>;
type Normal8 = FiniteField<2, 8, Normal<0x11B, 0x20>>;
type Cantor8 = FiniteField<2, 8, Cantor<0x11B, 0x20>>;
type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Rs8 = FiniteField<2, 8, Polynomial<RS>>;

// The at-cap odd tower: GF(3^4) by the `x^4 + 2x^2 + 2` quartic, then
// quadratics `t^2 - n` with Euler-searched non-square constants (each
// constant's sparsity noted below). Relation lengths are D blocks of
// base-degree coordinates: 4, 8, 16, 32, 64.
struct G3e4Spec;
impl ExtensionSpec<4> for G3e4Spec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^4) as x^4 + 2x^2 + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
}
type G3e4 = FiniteField<3, 4, Extension<G3e4Spec, 4>>;

struct G3e8Spec;
impl ExtensionSpec<2> for G3e8Spec {
    type Base = G3e4;
    type Storage = Components;
    const NAME: &'static str = "GF(3^8) as t^2 - x";
    const RELATION: Relation = Relation::from_coordinates(&[0, 1, 0, 0, 0, 0, 0, 0]);
}
type G3e8 = FiniteField<3, 8, Extension<G3e8Spec, 2>>;

struct G3e16Spec;
impl ExtensionSpec<2> for G3e16Spec {
    type Base = G3e8;
    type Storage = Components;
    const NAME: &'static str = "GF(3^16) as t^2 - s^4";
    const RELATION: Relation =
        Relation::from_coordinates(&[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}
type G3e16 = FiniteField<3, 16, Extension<G3e16Spec, 2>>;

struct G3e32Spec;
impl ExtensionSpec<2> for G3e32Spec {
    type Base = G3e16;
    type Storage = Components;
    const NAME: &'static str = "GF(3^32) as t^2 - s^8";
    const RELATION: Relation = Relation::from_coordinates(&[
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ]);
}
type G3e32 = FiniteField<3, 32, Extension<G3e32Spec, 2>>;

struct G3e64Spec;
impl ExtensionSpec<2> for G3e64Spec {
    type Base = G3e32;
    type Storage = Components;
    const NAME: &'static str = "GF(3^64) as t^2 - s^16";
    const RELATION: Relation = Relation::from_coordinates(&[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]);
}
type G3e64 = FiniteField<3, 64, Extension<G3e64Spec, 2>>;

// ---------------------------------------------------------------------------
// Enumeration helpers (mirroring `tests/relations.rs`)
// ---------------------------------------------------------------------------

/// Every element of a small field, in index order.
fn elements<F: Field>(order: u128) -> Vec<Elem<F>> {
    (0..order)
        .map(|i| Elem::<F>::from_index(i).expect("an index below the order"))
        .collect()
}

/// Every valid image tuple of the pair: the full cartesian product over
/// the target filtered by `from_images`.
fn valid_tuples<S: Field, T: Field<Characteristic = S::Characteristic>>(
    order_t: u128,
) -> Vec<Vec<Elem<T>>> {
    let count = Embedding::<S, T>::generator_count();
    let target = elements::<T>(order_t);
    let mut tuples = vec![Vec::new()];
    for _ in 0..count {
        let mut next = Vec::new();
        for prefix in &tuples {
            for &y in &target {
                let mut tuple = prefix.clone();
                tuple.push(y);
                next.push(tuple);
            }
        }
        tuples = next;
    }
    tuples
        .into_iter()
        .filter(|tuple| Embedding::<S, T>::from_images(tuple).is_ok())
        .collect()
}

/// Every valid image tuple of an equal-degree pair, filtered by the
/// isomorphism constructor.
fn valid_iso_tuples<S: Field, T: Field<Characteristic = S::Characteristic>>(
    order_t: u128,
) -> Vec<Vec<Elem<T>>> {
    let count = Embedding::<S, T>::generator_count();
    let target = elements::<T>(order_t);
    let mut tuples = vec![Vec::new()];
    for _ in 0..count {
        let mut next = Vec::new();
        for prefix in &tuples {
            for &y in &target {
                let mut tuple = prefix.clone();
                tuple.push(y);
                next.push(tuple);
            }
        }
        tuples = next;
    }
    tuples
        .into_iter()
        .filter(|tuple| Isomorphism::<S, T>::from_images(tuple).is_ok())
        .collect()
}

/// Exhaustive agreement of the prepared embedding with its compact map on
/// every valid image tuple: every method matches, `None`-agreement holds
/// on every non-member, and the generator-level counts match.
fn agree_prepared_on_tuples<S: Field, T: Field<Characteristic = S::Characteristic>>(
    order_s: u128,
    order_t: u128,
) {
    let tuples = valid_tuples::<S, T>(order_t);
    assert!(!tuples.is_empty(), "a pair holds a valid image tuple");
    // The prepared constructor covers every generator level exactly once.
    assert_eq!(
        tuples[0].len(),
        Embedding::<S, T>::generator_count(),
        "tuple width matches the generator count"
    );
    let source = elements::<S>(order_s);
    let target = elements::<T>(order_t);
    for tuple in &tuples {
        let compact = Embedding::<S, T>::from_images(tuple).unwrap();
        let prepared = PreparedEmbedding::<S, T>::new(&compact);
        assert_eq!(prepared.embedding().to_images(), compact.to_images());
        assert_eq!(prepared.relative_degree(), T::DEGREE / S::DEGREE);
        for &s in &source {
            assert_eq!(prepared.embed(s), compact.embed(s), "embed");
            let y = compact.embed(s);
            assert!(prepared.contains(y), "a member image is contained");
            assert_eq!(prepared.restrict(y), Some(s), "round trip");
        }
        for &y in &target {
            assert_eq!(prepared.contains(y), compact.contains(y), "contains");
            assert_eq!(prepared.restrict(y), compact.restrict(y), "restriction");
            assert_eq!(prepared.frobenius(y), compact.frobenius(y), "frobenius");
            assert_eq!(prepared.trace(y), compact.trace(y), "trace");
            assert_eq!(prepared.norm(y), compact.norm(y), "norm");
        }
    }
}

/// Exhaustive agreement of the prepared isomorphism with its compact map
/// on every valid image tuple, in both directions.
fn agree_prepared_iso_on_tuples<S: Field, T: Field<Characteristic = S::Characteristic>>(
    order_s: u128,
    order_t: u128,
) {
    let tuples = valid_iso_tuples::<S, T>(order_t);
    assert!(!tuples.is_empty(), "a pair holds a valid image tuple");
    let source = elements::<S>(order_s);
    let target = elements::<T>(order_t);
    for tuple in &tuples {
        let compact = Isomorphism::<S, T>::from_images(tuple).unwrap();
        let prepared = PreparedIsomorphism::<S, T>::new(&compact);
        assert_eq!(prepared.isomorphism().to_images(), compact.to_images());
        for &s in &source {
            assert_eq!(prepared.apply(s), compact.apply(s), "apply");
            assert_eq!(prepared.unapply(prepared.apply(s)), s, "round trip");
        }
        for &y in &target {
            assert_eq!(prepared.unapply(y), compact.unapply(y), "unapply");
            assert_eq!(prepared.apply(prepared.unapply(y)), y, "reverse round trip");
        }
    }
}

// ---------------------------------------------------------------------------
// Full-tuple agreement on the inventory pairs
// ---------------------------------------------------------------------------

#[test]
fn prepared_agrees_over_gf3_towers() {
    agree_prepared_on_tuples::<R3, G9>(3, 9);
    agree_prepared_on_tuples::<R3, G81Quartic>(3, 81);
    agree_prepared_on_tuples::<R3, G9Alt>(3, 9);
    agree_prepared_on_tuples::<R3, G81Ext>(3, 81);
    agree_prepared_on_tuples::<G9, G81Ext>(9, 81);
    agree_prepared_on_tuples::<G9Alt, G81Ext>(9, 81);
    agree_prepared_iso_on_tuples::<G81Quartic, G81Quartic>(81, 81);
    agree_prepared_iso_on_tuples::<G81Ext, G81Ext>(81, 81);
    agree_prepared_iso_on_tuples::<G81Quartic, G81Ext>(81, 81);
    agree_prepared_iso_on_tuples::<G81Ext, G81Quartic>(81, 81);
}

#[test]
fn prepared_agrees_over_gf5_towers() {
    agree_prepared_on_tuples::<R5, G25>(5, 25);
    agree_prepared_on_tuples::<R5, G625>(5, 625);
    agree_prepared_on_tuples::<G25, G625>(25, 625);
    agree_prepared_iso_on_tuples::<G625, G625>(625, 625);
}

#[test]
fn prepared_agrees_over_gf7_towers() {
    agree_prepared_on_tuples::<R7, G49Odd>(7, 49);
    agree_prepared_on_tuples::<R7, G343>(7, 343);
    agree_prepared_on_tuples::<R7, G2401Ext>(7, 2401);
    agree_prepared_on_tuples::<G49Odd, G2401Ext>(49, 2401);
    agree_prepared_iso_on_tuples::<G343, G343>(343, 343);
}

#[test]
fn prepared_agrees_on_small_binary_pairs() {
    agree_prepared_on_tuples::<Poly2, Poly4>(4, 16);
    agree_prepared_on_tuples::<Poly2, Aes>(4, 256);
    agree_prepared_on_tuples::<Poly4, Aes>(16, 256);
    agree_prepared_on_tuples::<G4, G16Ext>(4, 16);
    agree_prepared_iso_on_tuples::<Poly4, G16Ext>(16, 16);
    agree_prepared_iso_on_tuples::<G16Ext, G16Ext>(16, 16);
}

// ---------------------------------------------------------------------------
// Canonical pairs: the prepared map matches the canonical compact map
// ---------------------------------------------------------------------------

#[test]
fn prepared_canonical_pairs_match() {
    // Tower inclusions and the prime subfield.
    let tower = Embedding::<Aes, Gf16>::new().unwrap();
    let prepared = PreparedEmbedding::<Aes, Gf16>::new(&tower);
    let x = Elem::<Aes>::from_raw(0x53);
    assert_eq!(prepared.embed(x), tower.embed(x));
    assert_eq!(prepared.restrict(prepared.embed(x)), Some(x));
    // Isomorphism pairs route through the isomorphism type.
    let iso = Isomorphism::<Fp64, Gf64>::new().unwrap();
    let prepared_iso = PreparedIsomorphism::<Fp64, Gf64>::new(&iso);
    let y = Elem::<Fp64>::from_raw(0x1234_5678);
    assert_eq!(prepared_iso.apply(y), iso.apply(y));
    assert_eq!(prepared_iso.unapply(prepared_iso.apply(y)), y);
    // The quadratic Mersenne extension over its prime row.
    let qm = Embedding::<fgf::Mersenne31, QuadMersenne31>::new().unwrap();
    let prepared_qm = PreparedEmbedding::<fgf::Mersenne31, QuadMersenne31>::new(&qm);
    let m = Elem::<fgf::Mersenne31>::from_residue(7).unwrap();
    assert_eq!(prepared_qm.restrict(prepared_qm.embed(m)), Some(m));
    // Basis-leaf rows embed through the ladder like the compact map.
    let normal = Embedding::<Normal8, Gf16>::new().unwrap();
    let prepared_normal = PreparedEmbedding::<Normal8, Gf16>::new(&normal);
    let element = Elem::<Normal8>::from_index(0x53).unwrap();
    assert_eq!(prepared_normal.embed(element), normal.embed(element));
    assert_eq!(
        prepared_normal.restrict(prepared_normal.embed(element)),
        Some(element)
    );
    let cantor = Embedding::<Cantor8, Gf16>::new().unwrap();
    let prepared_cantor = PreparedEmbedding::<Cantor8, Gf16>::new(&cantor);
    let element = Elem::<Cantor8>::from_index(0x53).unwrap();
    assert_eq!(prepared_cantor.embed(element), cantor.embed(element));
    assert_eq!(
        prepared_cantor.restrict(prepared_cantor.embed(element)),
        Some(element)
    );
}

// ---------------------------------------------------------------------------
// Relative degree divisible by the characteristic: the weighted-trace
// projection stays exact there, where dividing the absolute trace by
// the degree would fail.
// ---------------------------------------------------------------------------

#[test]
fn prepared_restricts_when_the_characteristic_divides_the_degree() {
    // GF(9) over GF(3): `r = 2`, `p = 3` does not divide `r`; the
    // constructor still succeeds and every member round-trips, but the
    // proptest below pins the exact lift where the naive trace-division
    // applies directly.
    let check = Embedding::<R3, G9>::new().unwrap();
    let prepared = PreparedEmbedding::<R3, G9>::new(&check);
    for i in 0..9u128 {
        let y = Elem::<G9>::from_index(i).unwrap();
        assert_eq!(prepared.restrict(y), check.restrict(y));
    }
    // GF(81) over GF(9): `r = 2`, `p = 3` — and GF(81) over GF(3) with
    // `r = 4`; both divide nothing, yet round-trip exactly. The true
    // case is GF(8) over GF(2): `r = 3` is odd, so take GF(4)
    // over GF(2) with `r = 2`: `2 | 2`.
    let two_four = Embedding::<Poly2, Poly4>::new().unwrap();
    let prepared_two_four = PreparedEmbedding::<Poly2, Poly4>::new(&two_four);
    assert_eq!(2 % 2, 0, "the characteristic divides the relative degree");
    for i in 0..16u128 {
        let y = Elem::<Poly4>::from_index(i).unwrap();
        assert_eq!(prepared_two_four.restrict(y), two_four.restrict(y));
        assert_eq!(prepared_two_four.trace(y), two_four.trace(y));
        assert_eq!(prepared_two_four.norm(y), two_four.norm(y));
    }
    // A cubic tower over GF(2): `r = 3`, odd — and GF(64) over GF(8):
    // `r = 3` again. The even-`r` binary ladder case above is the divisible
    // witness; this loop pins the odd-`r` agreement beside it.
    let two_eight = Embedding::<Poly2, Aes>::new().unwrap();
    let prepared_two_eight = PreparedEmbedding::<Poly2, Aes>::new(&two_eight);
    for i in 0..256u128 {
        let y = Elem::<Aes>::from_index(i).unwrap();
        assert_eq!(prepared_two_eight.restrict(y), two_eight.restrict(y));
    }
}

// ---------------------------------------------------------------------------
// Two conjugate-image embeddings of one pair with different preimages
// ---------------------------------------------------------------------------

#[test]
fn prepared_distinguishes_conjugate_image_embeddings() {
    // GF(4) into the AES field admits three valid image tuples (the three
    // nonzero roots of the transported relation besides one); any two
    // distinct tuples fix different embeddings with different preimages.
    let tuples = valid_tuples::<Poly2, Aes>(256);
    assert!(tuples.len() > 1, "the pair admits conjugate images");
    let first = Embedding::<Poly2, Aes>::from_images(&tuples[0]).unwrap();
    let second = Embedding::<Poly2, Aes>::from_images(&tuples[1]).unwrap();
    assert_ne!(tuples[0], tuples[1], "distinct image tuples");
    let prepared_first = PreparedEmbedding::<Poly2, Aes>::new(&first);
    let prepared_second = PreparedEmbedding::<Poly2, Aes>::new(&second);
    // The images of a fixed nonzero source element differ.
    let probe = Elem::<Poly2>::from_index(2).unwrap();
    assert_ne!(
        prepared_first.embed(probe),
        prepared_second.embed(probe),
        "conjugate images embed distinctly"
    );
    // Each prepared map inverts its own image. The two images lie in
    // one shared subfield image set, so cross restriction succeeds with
    // the conjugated preimage rather than rejecting.
    let image_first = prepared_first.embed(probe);
    let image_second = prepared_second.embed(probe);
    assert_eq!(prepared_first.restrict(image_first), Some(probe));
    assert_eq!(prepared_second.restrict(image_second), Some(probe));
    let cross = prepared_first.restrict(image_second);
    assert_eq!(
        cross,
        first.restrict(image_second),
        "cross restriction agrees"
    );
    assert_ne!(
        cross,
        Some(probe),
        "the shared image holds a different preimage"
    );
    assert!(cross.is_some(), "a conjugate image is never rejected");
    // Unequal generator-level counts are rejected before construction:
    // the tuple width always equals the generator count.
    for tuple in &tuples {
        assert_eq!(tuple.len(), Embedding::<Poly2, Aes>::generator_count());
    }
}

// ---------------------------------------------------------------------------
// The injection-detection trap: a nonidentity normal-basis automorphism
// composed with an inclusion is columns, not an injection
// ---------------------------------------------------------------------------

#[test]
fn prepared_normal_basis_composite_takes_the_column_arm() {
    // A nonidentity automorphism of the normal-basis byte row, composed
    // with the inclusion into GF(2^16): the forward columns are a permuted
    // set of unit columns, which must NOT detect as an injection — only
    // the exact identity prefix counts.
    let iso_tuples = valid_iso_tuples::<Normal8, Normal8>(256);
    let nonidentity = iso_tuples
        .iter()
        .map(|tuple| Isomorphism::<Normal8, Normal8>::from_images(tuple).unwrap())
        .find(|iso| {
            iso.apply(Elem::<Normal8>::from_index(2).unwrap())
                != Elem::<Normal8>::from_index(2).unwrap()
        })
        .expect("the normal-basis row admits a nonidentity automorphism");
    // The inclusion targets an extension built directly over the
    // normal-basis row, so the composite's forward map is genuinely the
    // normal automorphism followed by the structural lift — no
    // AES-tower conversion stands between the unit columns and the
    // permutation the automorphism induces.
    let inclusion = Embedding::<Normal8, Normal16>::new().unwrap();
    let composite = nonidentity.then_embedding(&inclusion);
    let prepared = PreparedEmbedding::<Normal8, Normal16>::new(&composite);
    // The composite still equals the compact map everywhere, even though
    // it routes around the injection arm.
    for word in 0..256u128 {
        let s = Elem::<Normal8>::from_index(word).unwrap();
        assert_eq!(prepared.embed(s), composite.embed(s), "embed at {word}");
    }
    for word in 0..65536u128 {
        let y = Elem::<Normal16>::from_index(word).unwrap();
        assert_eq!(
            prepared.restrict(y),
            composite.restrict(y),
            "restrict at {word}"
        );
        assert_eq!(
            prepared.contains(y),
            composite.contains(y),
            "contains at {word}"
        );
        assert_eq!(prepared.trace(y), composite.trace(y), "trace at {word}");
        assert_eq!(prepared.norm(y), composite.norm(y), "norm at {word}");
    }
}

// An extension row built directly over the normal-basis byte row:
// `t^2 + 0xFF*t + 1` (the normal-basis one is `0x00FF`, not raw one).
// Proven in `tests/binary_bases.rs` (`BasesNormal16`).
struct Normal16Spec;
impl ExtensionSpec<2> for Normal16Spec {
    type Base = Normal8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) over the normal-basis row";
    const RELATION: Relation = Relation::from_binary_words::<Normal8>(&[0x1, 0xFF]);
}
type Normal16 = FiniteField<2, 16, Extension<Normal16Spec, 2>>;

// A quadratic extension of the normal-basis GF(4): the base word one is
// raw 3, so raw 1 is not field one anywhere on this tower. `t^2 + u*t + b`
// with `u` the base one (raw 3) and `b` the base element of raw 2 (trace
// one over GF(2)) is irreducible by the Artin-Schreier criterion, the
// same shape as the `G16Ext` tower above with the linear coefficient
// spelled in the base's own words.
type Normal4 = FiniteField<2, 2, Normal<7, 2>>;
struct OverNormal4Spec;
impl ExtensionSpec<2> for OverNormal4Spec {
    type Base = Normal4;
    type Storage = Components;
    const NAME: &'static str = "GF(16) over normal-basis GF(4)";
    const RELATION: Relation = Relation::from_binary_words::<Normal4>(&[2, 3]);
}
type OverNormal16 = FiniteField<2, 4, Extension<OverNormal4Spec, 2>>;
#[test]
fn prepared_agrees_over_the_normal_basis_tower() {
    // Raw 1 is not field one on this tower: the regression needs the
    // prepared map to agree with the compact map here, where a raw-word
    // comparison against 1 misfires.
    assert_eq!(
        Elem::<Normal4>::ONE.to_raw(),
        3,
        "raw 1 is not field one on the normal-basis row"
    );
    // The reported case from the conjugation fix: word `[0, 3]` is the
    // base one in the upper coordinate.
    let reported = Elem::<OverNormal16>::from_raw([0, 3]);
    let compact = Embedding::<Normal4, OverNormal16>::new().unwrap();
    let prepared = PreparedEmbedding::<Normal4, OverNormal16>::new(&compact);
    for i in 0..16u128 {
        let y = Elem::<OverNormal16>::from_index(i).unwrap();
        assert_eq!(
            prepared.frobenius(y),
            compact.frobenius(y),
            "frobenius at {i}"
        );
        assert_eq!(prepared.contains(y), compact.contains(y), "contains at {i}");
        assert_eq!(prepared.restrict(y), compact.restrict(y), "restrict at {i}");
        assert_eq!(prepared.trace(y), compact.trace(y), "trace at {i}");
        assert_eq!(prepared.norm(y), compact.norm(y), "norm at {i}");
    }
    assert_eq!(
        prepared.frobenius(reported),
        reported.pow(4),
        "the reported case matches scalar exponentiation"
    );
}

// ---------------------------------------------------------------------------
// A structural polynomial→basis→extension lift, with exhaustive preimages
// and outsider rejection for both directions
// ---------------------------------------------------------------------------

#[test]
fn prepared_structural_lift_round_trips_exhaustively() {
    // A genuine polynomial -> normal -> extension lift: the AES byte row
    // changes presentation into the normal-basis row, then lifts
    // structurally into the extension directly over it. The prepared map
    // reproduces the composite, with exhaustive preimages and outsider
    // rejection on the small pair beside it.
    let change = Isomorphism::<Aes8, Normal8>::new().unwrap();
    let lift = Embedding::<Normal8, Normal16>::new().unwrap();
    let reference = change.then_embedding(&lift);
    let prepared = PreparedEmbedding::<Aes8, Normal16>::new(&reference);
    assert_eq!(prepared.relative_degree(), 2);
    for word in 0..256u128 {
        let s = Elem::<Aes8>::from_index(word).unwrap();
        assert_eq!(prepared.embed(s), reference.embed(s), "embed at {word}");
        assert_eq!(prepared.restrict(prepared.embed(s)), Some(s));
    }
    // Exhaustive outsider rejection over the small pair: every non-image
    // target restricts to `None` under both maps.
    let small = Embedding::<Poly2, Poly4>::new().unwrap();
    let prepared_small = PreparedEmbedding::<Poly2, Poly4>::new(&small);
    for word in 0..16u128 {
        let y = Elem::<Poly4>::from_index(word).unwrap();
        assert_eq!(
            prepared_small.restrict(y),
            small.restrict(y),
            "outsider at {word}"
        );
        assert_eq!(
            prepared_small.contains(y),
            small.contains(y),
            "contains at {word}"
        );
    }
}

// ---------------------------------------------------------------------------
// Frobenius equality with `Frobenius<F>` at the same power
// ---------------------------------------------------------------------------

#[test]
fn prepared_frobenius_matches_the_frobenius_type() {
    // Over GF(2^16): the relative Frobenius of the AES inclusion is
    // `x^256`, the 8th absolute power.
    let inclusion = Embedding::<Aes, Gf16>::new().unwrap();
    let prepared = PreparedEmbedding::<Aes, Gf16>::new(&inclusion);
    let absolute = Frobenius::<Gf16>::new(8);
    assert_eq!(absolute.power(), 8);
    for word in 0..65536u128 {
        let y = Elem::<Gf16>::from_index(word).unwrap();
        assert_eq!(
            prepared.frobenius(y),
            y.frobenius_with(&absolute),
            "frobenius at {word}"
        );
    }
    // Over the odd tower: GF(9) over GF(3) has relative Frobenius `x^3`,
    // the 1st absolute power.
    let odd = Embedding::<R3, G9>::new().unwrap();
    let prepared_odd = PreparedEmbedding::<R3, G9>::new(&odd);
    let absolute_odd = Frobenius::<G9>::new(1);
    for i in 0..9u128 {
        let y = Elem::<G9>::from_index(i).unwrap();
        assert_eq!(prepared_odd.frobenius(y), y.frobenius_with(&absolute_odd));
    }
    // The identity power agrees trivially.
    let identity = Frobenius::<Gf16>::new(0);
    let y = Elem::<Gf16>::from_raw(0xBEEF);
    assert_eq!(y.frobenius_with(&identity), y);
}

// ---------------------------------------------------------------------------
// QM31 anchors: `frobenius == conjugate`, `norm == Elem::norm`
// ---------------------------------------------------------------------------

#[test]
fn prepared_qm31_matches_conjugation_and_norm() {
    type M31 = fgf::Mersenne31;
    type Qm31 = QuadMersenne31;
    let compact = Embedding::<M31, Qm31>::new().expect("m31 embeds in qm31");
    let prepared = PreparedEmbedding::<M31, Qm31>::new(&compact);
    let elems = [
        Elem::<Qm31>::from_raw([7, 11]),
        Elem::<Qm31>::from_raw([0x7FFF_FFFE, 0x7FFF_FFFF]),
        Elem::<Qm31>::ZERO,
        Elem::<Qm31>::ONE,
    ];
    for (i, &elem) in elems.iter().enumerate() {
        assert_eq!(
            prepared.frobenius(elem),
            elem.conjugate(),
            "frobenius anchor {i}"
        );
        assert_eq!(prepared.norm(elem), elem.norm(), "norm anchor {i}");
        let sum = elem + elem.conjugate();
        assert_eq!(
            prepared.trace(elem),
            compact
                .restrict(sum)
                .expect("the conjugate sum lies in the image"),
            "trace anchor {i}"
        );
    }
    // Exhaustive agreement across the whole (small) target is infeasible
    // at QM31 order; the anchors above plus the row-gate sample cover it.
}

// ---------------------------------------------------------------------------
// The at-cap odd row: GF(3^64) fills the coordinate capacity exactly
// ---------------------------------------------------------------------------

#[test]
fn prepared_at_cap_odd_row_agrees() {
    const { assert!(G3e64::DEGREE == 64) };
    let compact = Embedding::<R3, G3e64>::new().expect("r3 embeds in g3e64");
    let prepared = PreparedEmbedding::<R3, G3e64>::new(&compact);
    assert_eq!(prepared.relative_degree(), 64);
    assert_eq!(prepared.embedding().to_images(), compact.to_images());
    // Every prime element round-trips; member arithmetic matches.
    for residue in 0..3u64 {
        let s = Elem::<R3>::from_residue(residue).unwrap();
        assert_eq!(prepared.embed(s), compact.embed(s));
        assert_eq!(prepared.restrict(prepared.embed(s)), Some(s));
        assert_eq!(
            prepared.trace(prepared.embed(s)),
            compact.trace(prepared.embed(s))
        );
        assert_eq!(
            prepared.norm(prepared.embed(s)),
            compact.norm(prepared.embed(s))
        );
    }
    // The mid-tower generator image sits outside the prime image and is
    // rejected by both maps; member combinations agree on every method.
    let mid = Embedding::<G3e4, G3e64>::new().expect("g3e4 embeds in g3e64");
    let g4: Elem<G3e4> = Embedding::<G3e4, G3e8>::generator(0);
    let tower_gen = mid.embed(g4);
    assert!(!compact.contains(tower_gen));
    assert!(!prepared.contains(tower_gen));
    assert_eq!(prepared.restrict(tower_gen), None);
    let one = prepared.embed(Elem::<R3>::ONE);
    for w in [one, tower_gen, one + tower_gen, tower_gen * tower_gen] {
        assert_eq!(prepared.contains(w), compact.contains(w));
        assert_eq!(prepared.restrict(w), compact.restrict(w));
        assert_eq!(prepared.trace(w), compact.trace(w));
        assert_eq!(prepared.norm(w), compact.norm(w));
        assert_eq!(prepared.frobenius(w), compact.frobenius(w));
    }
    // A wider deterministic sweep through tower arithmetic.
    let mut state = tower_gen;
    for _ in 0..32 {
        state = state * tower_gen + one;
        assert_eq!(prepared.contains(state), compact.contains(state));
        assert_eq!(prepared.restrict(state), compact.restrict(state));
        assert_eq!(prepared.trace(state), compact.trace(state));
        assert_eq!(prepared.norm(state), compact.norm(state));
    }
}

// ---------------------------------------------------------------------------
// Isomorphism coverage: mutual pairs, basis changes, and the tower iso
// ---------------------------------------------------------------------------

#[test]
fn prepared_isomorphisms_cover_mutual_pairs() {
    // AES/RS byte presentations are mutual isomorphisms.
    let forward = Isomorphism::<Aes8, Rs8>::new().unwrap();
    let prepared = PreparedIsomorphism::<Aes8, Rs8>::new(&forward);
    for word in 0..256u128 {
        let s = Elem::<Aes8>::from_index(word).unwrap();
        assert_eq!(prepared.apply(s), forward.apply(s));
        assert_eq!(prepared.unapply(prepared.apply(s)), s);
    }
    for word in 0..256u128 {
        let y = Elem::<Rs8>::from_index(word).unwrap();
        assert_eq!(prepared.unapply(y), forward.unapply(y));
    }
    // The normal-basis byte change routes through columns, not identity.
    let basis = Isomorphism::<Normal8, Aes8>::new().unwrap();
    let prepared_basis = PreparedIsomorphism::<Normal8, Aes8>::new(&basis);
    for word in 0..256u128 {
        let s = Elem::<Normal8>::from_index(word).unwrap();
        assert_eq!(prepared_basis.apply(s), basis.apply(s));
        assert_eq!(prepared_basis.unapply(prepared_basis.apply(s)), s);
    }
}

// ---------------------------------------------------------------------------
// Structural traits: `Clone` + `Debug`, not `Copy`; `Send + Sync` with
// `u64`-raw endpoints
// ---------------------------------------------------------------------------

#[test]
fn prepared_types_are_clone_debug_and_thread_safe() {
    let compact = Embedding::<Aes, Gf16>::new().unwrap();
    let prepared = PreparedEmbedding::<Aes, Gf16>::new(&compact);
    let cloned = prepared.clone();
    assert_eq!(
        cloned.embed(Elem::<Aes>::from_raw(0x53)),
        prepared.embed(Elem::<Aes>::from_raw(0x53))
    );
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PreparedEmbedding<Aes, Gf16>>();
    assert_send_sync::<PreparedIsomorphism<Fp64, Gf64>>();
    assert_send_sync::<PreparedEmbedding<fgf::Mersenne31, QuadMersenne31>>();
}

// ---------------------------------------------------------------------------
// Coverage: clone/debug of isomorphisms, prefix-mask edges, non-identity
// prefix maps, and the Frobenius closed-form edges
// ---------------------------------------------------------------------------

#[test]
fn prepared_isomorphism_clone_and_debug_agree() {
    // Clone-and-compare over a column-arm isomorphism: every method of
    // the clone agrees with the original, and Debug names the type.
    let compact = Isomorphism::<Normal8, Aes8>::new().unwrap();
    let prepared = PreparedIsomorphism::<Normal8, Aes8>::new(&compact);
    let cloned = prepared.clone();
    assert_eq!(
        cloned.isomorphism().to_images(),
        prepared.isomorphism().to_images()
    );
    for word in 0..256u128 {
        let s = Elem::<Normal8>::from_index(word).unwrap();
        assert_eq!(cloned.apply(s), prepared.apply(s), "apply at {word}");
    }
    for word in 0..256u128 {
        let y = Elem::<Aes8>::from_index(word).unwrap();
        assert_eq!(cloned.unapply(y), prepared.unapply(y), "unapply at {word}");
    }
    // The identity isomorphism clones through the injection arm too.
    let identity = Isomorphism::<Aes8, Aes8>::new().unwrap();
    let prepared_identity = PreparedIsomorphism::<Aes8, Aes8>::new(&identity);
    let cloned_identity = prepared_identity.clone();
    let probe = Elem::<Aes8>::from_raw(0x53);
    assert_eq!(cloned_identity.apply(probe), probe);
    assert_eq!(cloned_identity.unapply(probe), probe);
}

#[test]
fn prepared_injection_prefix_covers_degree_64() {
    // GF(2^64) over its GF(2^32) base through the tower inclusion, and
    // the degree-64 Frobenius identity for the full-word mask arm.
    type Gf32 = FiniteField<2, 32, Extension<fgf::Rijndael32, 2>>;
    let tower = Embedding::<Gf32, Gf64>::new().unwrap();
    let prepared = PreparedEmbedding::<Gf32, Gf64>::new(&tower);
    let probe = Elem::<Gf32>::from_raw(0xDEAD_BEEF);
    assert_eq!(prepared.restrict(prepared.embed(probe)), Some(probe));
    let absolute = Frobenius::<Gf64>::new(64);
    assert_eq!(absolute.power(), 0);
    let wide = Elem::<Gf64>::from_raw(0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(wide.frobenius_with(&absolute), wide);
    // A non-injection map trips the per-column mismatch: the normal-basis
    // composite from the trap test agrees with its compact map here too.
    let tuples = valid_iso_tuples::<Normal8, Normal8>(256);
    let nonidentity = tuples
        .iter()
        .map(|tuple| Isomorphism::<Normal8, Normal8>::from_images(tuple).unwrap())
        .find(|iso| {
            iso.apply(Elem::<Normal8>::from_index(2).unwrap())
                != Elem::<Normal8>::from_index(2).unwrap()
        })
        .expect("a nonidentity normal-basis automorphism");
    let inclusion = Embedding::<Normal8, Gf16>::new().unwrap();
    let composite = nonidentity.then_embedding(&inclusion);
    let prepared_composite = PreparedEmbedding::<Normal8, Gf16>::new(&composite);
    let probe8 = Elem::<Normal8>::from_index(0x53).unwrap();
    assert_eq!(prepared_composite.embed(probe8), composite.embed(probe8));
    assert_eq!(
        prepared_composite.restrict(prepared_composite.embed(probe8)),
        Some(probe8)
    );
}

#[test]
fn prepared_odd_non_prefix_map_agrees() {
    // Conjugate image tuples of a level-bearing odd pair: every tuple's
    // prepared map agrees with its compact map on the whole target,
    // tripping the residue per-entry mismatch arm on non-identity tuples.
    let tuples = valid_tuples::<G9, G81Ext>(81);
    assert!(tuples.len() > 1, "conjugate images exist");
    for tuple in &tuples {
        let compact = Embedding::<G9, G81Ext>::from_images(tuple).unwrap();
        let prepared = PreparedEmbedding::<G9, G81Ext>::new(&compact);
        for y in elements::<G81Ext>(81) {
            assert_eq!(prepared.restrict(y), compact.restrict(y));
            assert_eq!(prepared.trace(y), compact.trace(y));
            assert_eq!(prepared.norm(y), compact.norm(y));
        }
    }
}

#[test]
fn prepared_frobenius_closed_form_edges_agree() {
    // The even-multiple-of-base identity arm: power 16 over GF(2^16) is
    // the identity, matching scalar exponentiation.
    let identity_arm = Frobenius::<Gf16>::new(16);
    assert_eq!(identity_arm.power(), 0);
    let probe = Elem::<Gf16>::from_raw(0xBEEF);
    assert_eq!(probe.frobenius_with(&identity_arm), probe);
    // The odd-multiple conjugation arm over the quadratic Mersenne row.
    let conjugate_arm = Frobenius::<QuadMersenne31>::new(1);
    let qm_probe = Elem::<QuadMersenne31>::from_raw([0x1234_5678, 0x1111_1111]);
    assert_eq!(
        qm_probe.frobenius_with(&conjugate_arm),
        qm_probe.conjugate()
    );
    // The even-multiple identity arm over the same row.
    let qm_identity = Frobenius::<QuadMersenne31>::new(2);
    assert_eq!(qm_identity.power(), 0);
    assert_eq!(qm_probe.frobenius_with(&qm_identity), qm_probe);
}

// ---------------------------------------------------------------------------
// Odd injection detection must check every output coordinate
// ---------------------------------------------------------------------------

/// `GF(3)[t]/(t^4 + 2t^3 + t^2 + 1)`: irreducible, with `t + t^2` a root
/// of `a^2 + 1` whose prime-coordinate columns carry a nonzero upper
/// coordinate. An identity-prefix check that stops at the source degree
/// accepts those columns and then embeds by truncation.
struct QuarticTSpec;
impl ExtensionSpec<4> for QuarticTSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^4) as t^4 + 2t^3 + t^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0, 1, 2]);
}
type QuarticT = FiniteField<3, 4, Extension<QuarticTSpec, 4>>;

#[test]
fn odd_injection_detection_rejects_upper_coordinates() {
    // `a |-> t + t^2`: satisfies the transported relation `x^2 + 1 = 0`
    // because `(t + t^2)^2 = t^2 + 2t^3 + t^4 = -1` in this presentation.
    // Index digits are prime coordinates, least significant first.
    let generator = Elem::<QuarticT>::from_index(3 + 9).unwrap(); // t + t^2
    let map = Embedding::<G9Alt, QuarticT>::from_images(&[generator]).unwrap();
    let prepared = PreparedEmbedding::<G9Alt, QuarticT>::new(&map);
    let a = Elem::<G9Alt>::from_index(3).unwrap(); // the generator a
    let lifted = prepared.embed(a);
    assert_eq!(
        lifted, generator,
        "the compiled map reproduces the image tuple"
    );
    assert_eq!(
        lifted,
        map.embed(a),
        "the compiled map is the validated map"
    );
    assert_eq!(prepared.restrict(lifted), Some(a));
    // The map is no identity prefix: the image carries upper coordinates,
    // so detection must route it through the residue columns.
    let unit = Elem::<G9Alt>::ONE;
    let image_one = prepared.embed(unit);
    assert_eq!(prepared.restrict(image_one), Some(unit));
}
