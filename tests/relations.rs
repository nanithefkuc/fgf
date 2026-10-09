//! Characteristic-independent embedding and isomorphism acceptance.
//!
//! The enumeration reference checks every valid image tuple — enumerated
//! as full cartesian products filtered by `from_images` — with exhaustive
//! round trips, brute-force restriction, and homomorphism samples.
//! Construction, error precedence, conjugates, and the boundary cases
//! follow.

use fgf::{
    AES, Components, Elem, Embedding, EmbeddingError, Extension, ExtensionSpec, Field, FieldElem,
    FiniteField, Isomorphism, Mersenne31, Polynomial, QuadMersenne31, Relation, Residue,
};

// ---------------------------------------------------------------------------
// Inventory rows for the enumeration reference
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

// ---------------------------------------------------------------------------
// Enumeration helpers
// ---------------------------------------------------------------------------

/// Every element of a small field, in index order.
fn elements<F: Field>(order: u128) -> Vec<Elem<F>> {
    (0..order)
        .map(|i| Elem::<F>::from_index(i).expect("an index below the order"))
        .collect()
}

/// Every valid image tuple of the pair: the full cartesian product over
/// the target filtered by `from_images`, which validates each transported
/// relation.
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

/// Exhaustive agreement on every valid image tuple: every source element
/// round-trips, every non-image target restricts to `None` by brute-force
/// scan, and samples preserve addition and multiplication.
fn agree_on_tuples<S: Field, T: Field<Characteristic = S::Characteristic>>(
    order_s: u128,
    order_t: u128,
) {
    let tuples = valid_tuples::<S, T>(order_t);
    assert!(!tuples.is_empty(), "a pair holds a valid image tuple");
    let source = elements::<S>(order_s);
    let target = elements::<T>(order_t);
    for tuple in &tuples {
        let e = Embedding::<S, T>::from_images(tuple).unwrap();
        for &s in &source {
            let y = e.embed(s);
            assert_eq!(e.restrict(y), Some(s), "round trip");
            assert!(e.contains(y));
        }
        for &y in &target {
            let expected = source.iter().find(|&&s| e.embed(s) == y).copied();
            assert_eq!(e.restrict(y), expected, "brute-force restriction");
        }
        for (i, &a) in source.iter().enumerate() {
            let b = source[(i * 7 + 3) % source.len()];
            assert_eq!(e.embed(a + b), e.embed(a) + e.embed(b));
            assert_eq!(e.embed(a * b), e.embed(a) * e.embed(b));
        }
    }
}

// ---------------------------------------------------------------------------
// Full-tuple agreement on the inventory pairs
// ---------------------------------------------------------------------------

#[test]
fn enumeration_agrees_over_gf3_towers() {
    agree_on_tuples::<R3, G9>(3, 9);
    agree_on_tuples::<R3, G81Quartic>(3, 81);
    agree_on_tuples::<R3, G9Alt>(3, 9);
    agree_on_tuples::<R3, G81Ext>(3, 81);
    agree_on_tuples::<G9, G81Ext>(9, 81);
    agree_on_tuples::<G9Alt, G81Ext>(9, 81);
    agree_on_tuples::<G81Quartic, G81Quartic>(81, 81);
    agree_on_tuples::<G81Ext, G81Ext>(81, 81);
    // The tower and the flat presentation are mutual isomorphisms.
    agree_on_tuples::<G81Quartic, G81Ext>(81, 81);
    agree_on_tuples::<G81Ext, G81Quartic>(81, 81);
}

#[test]
fn enumeration_agrees_over_gf5_towers() {
    agree_on_tuples::<R5, G25>(5, 25);
    agree_on_tuples::<R5, G625>(5, 625);
    agree_on_tuples::<G25, G625>(25, 625);
    agree_on_tuples::<G625, G625>(625, 625);
}

#[test]
fn enumeration_agrees_over_gf7_towers() {
    agree_on_tuples::<R7, G49Odd>(7, 49);
    agree_on_tuples::<R7, G343>(7, 343);
    agree_on_tuples::<R7, G2401Ext>(7, 2401);
    agree_on_tuples::<G49Odd, G2401Ext>(49, 2401);
    agree_on_tuples::<G343, G343>(343, 343);
}

#[test]
fn enumeration_agrees_on_small_binary_pairs() {
    agree_on_tuples::<Poly2, Poly4>(4, 16);
    agree_on_tuples::<Poly2, Aes>(4, 256);
    agree_on_tuples::<Poly4, Aes>(16, 256);
    agree_on_tuples::<G4, G16Ext>(4, 16);
    agree_on_tuples::<Poly4, G16Ext>(16, 16);
    agree_on_tuples::<G16Ext, G16Ext>(16, 16);
}

// ---------------------------------------------------------------------------
// Construction and error precedence
// ---------------------------------------------------------------------------

#[test]
fn shipped_instances_embed_over_every_base_and_the_prime_subfield() {
    // Rijndael16 over the AES byte field and over GF(2).
    type R16 = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;
    let to_base = Embedding::<Aes, R16>::new().unwrap();
    let to_prime = Embedding::<FiniteField<2, 1, Polynomial<3>>, R16>::new().unwrap();
    let x = Elem::<Aes>::from_raw(0x53);
    let lifted = to_base.embed(x);
    assert_eq!(to_base.restrict(lifted), Some(x));
    assert_eq!(to_prime.embed(Elem::ONE), Elem::ONE);
    // FanPaar64 over FanPaar32 is a tower inclusion.
    type Fp32 = FiniteField<2, 32, Extension<fgf::FanPaar32, 2>>;
    type Fp64 = FiniteField<2, 64, Extension<fgf::FanPaar64, 2>>;
    let tower = Embedding::<Fp32, Fp64>::new().unwrap();
    let y = Elem::<Fp32>::from_raw(0x1234_5678);
    assert_eq!(tower.restrict(tower.embed(y)), Some(y));
    // QuadMersenne31 over Mersenne31.
    let qm = Embedding::<Mersenne31, QuadMersenne31>::new().unwrap();
    assert_eq!(
        Embedding::<Mersenne31, QuadMersenne31>::generator_count(),
        0,
        "prime sources hold no levels"
    );
    let m = Elem::<Mersenne31>::from_residue(7).unwrap();
    assert_eq!(qm.restrict(qm.embed(m)), Some(m));
}

#[test]
fn unrelated_odd_presentations_have_no_canonical_map() {
    // GF(9) in two unrelated presentations of the same order: equal
    // degrees, so no degree error, and no structural or ladder map.
    assert_eq!(
        Embedding::<G9, G9Alt>::new().unwrap_err(),
        EmbeddingError::NoCanonicalMap
    );
    assert_eq!(
        Embedding::<G9Alt, G9>::new().unwrap_err(),
        EmbeddingError::NoCanonicalMap
    );
}

#[test]
fn from_images_rejects_counts_and_non_roots() {
    let e = Embedding::<G9Alt, G81Ext>::new().unwrap();
    // Wrong count: the source holds one level.
    assert_eq!(
        Embedding::<G9Alt, G81Ext>::from_images(&[]).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    let doubled = [e.to_images()[0], e.to_images()[0]];
    assert_eq!(
        Embedding::<G9Alt, G81Ext>::from_images(&doubled).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    // A perturbed image fails the transported relation: the tower pair
    // holds two levels, so its top image perturbs.
    let t = Embedding::<G81Ext, G81Ext>::new().unwrap();
    let worse = [t.to_images()[0], t.to_images()[1] + Elem::ONE];
    assert_eq!(
        Embedding::<G81Ext, G81Ext>::from_images(&worse).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    // Round trip through images reproduces the map.
    let rebuilt = Embedding::<G9Alt, G81Ext>::from_images(e.to_images()).unwrap();
    for s in elements::<G9Alt>(9) {
        assert_eq!(rebuilt.embed(s), e.embed(s));
    }
}

#[test]
fn incompatible_degree_precedes_every_other_error() {
    // Every constructor reports the degree first, even with malformed
    // images and no canonical embedding on unequal but dividing degrees.
    let bad = [Elem::<G81Ext>::ZERO; 3];
    assert_eq!(
        Embedding::<G9Alt, G81Ext>::from_images(&bad).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    assert_eq!(
        Isomorphism::<G9Alt, G81Ext>::new().unwrap_err(),
        EmbeddingError::IncompatibleDegree {
            source: 2,
            target: 4
        }
    );
    assert_eq!(
        Isomorphism::<G9Alt, G81Ext>::from_images(&bad).unwrap_err(),
        EmbeddingError::IncompatibleDegree {
            source: 2,
            target: 4
        }
    );
    let embedding = Embedding::<G9Alt, G81Ext>::new().unwrap();
    assert_eq!(
        Isomorphism::<G9Alt, G81Ext>::from_embedding(embedding).unwrap_err(),
        EmbeddingError::IncompatibleDegree {
            source: 2,
            target: 4
        }
    );
    // Equal degrees with a short image list report invalid images.
    assert_eq!(
        Isomorphism::<G9Alt, G9Alt>::from_images(&[Elem::<G9Alt>::ZERO]).unwrap_err(),
        EmbeddingError::InvalidImages
    );
}

#[test]
fn isomorphisms_are_total_and_invertible_on_small_fields() {
    let iso = Isomorphism::<G9Alt, G9Alt>::new().unwrap();
    for s in elements::<G9Alt>(9) {
        assert_eq!(iso.unapply(iso.apply(s)), s);
    }
    for t in elements::<G9Alt>(9) {
        assert_eq!(iso.apply(iso.unapply(t)), t);
    }
    assert_eq!(iso.inverse().inverse().to_images(), iso.to_images());
    // The denoted embedding agrees on every element.
    let embedding = iso.to_embedding();
    for s in elements::<G9Alt>(9) {
        assert_eq!(iso.apply(s), embedding.embed(s));
    }
    // Composition equals sequential application.
    let forth = Isomorphism::<G9Alt, G9Alt>::new().unwrap();
    let composed = iso.then(&forth);
    for s in elements::<G9Alt>(9) {
        assert_eq!(composed.apply(s), forth.apply(iso.apply(s)));
    }
    // Unequal level counts across the flat and tower GF(81) rows: the
    // inverse image set rebuilds, and composition agrees sequentially.
    // No canonical map exists between these unrelated odd presentations,
    // so the map comes from a validated image tuple.
    let tuple: Vec<Elem<G81Ext>> = valid_tuples::<G81Quartic, G81Ext>(81)[0].clone();
    let flat_tower = Isomorphism::<G81Quartic, G81Ext>::from_images(&tuple).unwrap();
    let rebuilt =
        Isomorphism::<G81Ext, G81Quartic>::from_images(flat_tower.inverse().to_images()).unwrap();
    for t in elements::<G81Ext>(81) {
        assert_eq!(rebuilt.apply(t), flat_tower.inverse().apply(t));
    }
    let back = Isomorphism::<G81Ext, G81Quartic>::from_images(rebuilt.to_images()).unwrap();
    let cycle = flat_tower.then(&back);
    for s in elements::<G81Quartic>(81) {
        assert_eq!(cycle.apply(s), back.apply(flat_tower.apply(s)));
        assert_eq!(cycle.unapply(s), flat_tower.unapply(back.unapply(s)));
    }
    // Embedding composition agrees too.
    let e1 = Embedding::<R3, G9Alt>::new().unwrap();
    let e2 = Embedding::<G9Alt, G81Ext>::new().unwrap();
    let composed = e1.then(&e2);
    for s in elements::<R3>(3) {
        assert_eq!(composed.embed(s), e2.embed(e1.embed(s)));
    }
}

// ---------------------------------------------------------------------------
// Conjugate references
// ---------------------------------------------------------------------------

#[test]
fn trace_and_norm_match_independent_conjugates() {
    let e = Embedding::<G9Alt, G81Ext>::new().unwrap();
    for t in elements::<G81Ext>(81) {
        // Independent conjugates from raw powering, mapped back by a
        // brute-force preimage scan rather than `restrict`.
        let mut sum = t;
        let mut product = t;
        let mut power = t;
        for _ in 1..2 {
            power = power.pow(9);
            sum += power;
            product *= power;
        }
        let expected_sum = elements::<G9Alt>(9)
            .into_iter()
            .find(|&s| e.embed(s) == sum)
            .expect("a trace lies in the embedding image");
        let expected_product = elements::<G9Alt>(9)
            .into_iter()
            .find(|&s| e.embed(s) == product)
            .expect("a norm lies in the embedding image");
        assert_eq!(e.trace(t), expected_sum);
        assert_eq!(e.norm(t), expected_product);
    }
    // QuadMersenne31 norm over Mersenne31 equals the inherent method.
    let qm = Embedding::<Mersenne31, QuadMersenne31>::new().unwrap();
    for (re, im) in [(0u32, 0u32), (1, 0), (0, 1), (3, 4), (7, 11)] {
        let z = Elem::<QuadMersenne31>::from_raw([re, im]);
        assert_eq!(qm.norm(z).to_raw(), z.norm().to_raw());
    }
}

// ---------------------------------------------------------------------------
// Boundary cases
// ---------------------------------------------------------------------------

#[test]
fn gf9_conjugate_images_reach_gf81() {
    // Both roots of `a^2 + 1` over GF(3) embed GF(9) into the tower GF(81);
    // each choice validates, and the two maps differ.
    let mut maps = Vec::new();
    for a_image in elements::<G81Ext>(81) {
        if let Ok(e) = Embedding::<G9Alt, G81Ext>::from_images(&[a_image]) {
            maps.push(e);
        }
    }
    assert_eq!(maps.len(), 2, "two conjugate images of GF(9)");
    let (first, second) = (maps[0], maps[1]);
    let a = Elem::<G9Alt>::from_raw([0, 1]);
    assert_ne!(first.embed(a), second.embed(a));
    for s in elements::<G9Alt>(9) {
        assert_eq!(first.restrict(first.embed(s)), Some(s));
        assert_eq!(second.restrict(second.embed(s)), Some(s));
    }
}

#[test]
fn tower_transport_rejects_the_unchanged_top_image() {
    // The tower `t^2 - (1 + a)` with the lower image `a ↦ -a`: the
    // transported relation at the unchanged top image evaluates to `2a`,
    // nonzero, so the pair is rejected. The unchanged pair validates.
    let e = Embedding::<G81Ext, G81Ext>::new().unwrap();
    let (lower, top) = (e.to_images()[0], e.to_images()[1]);
    let neg_lower = lower.neg();
    assert_ne!(neg_lower, lower);
    assert_eq!(
        Embedding::<G81Ext, G81Ext>::from_images(&[neg_lower, top]).unwrap_err(),
        EmbeddingError::InvalidImages
    );
    assert!(Embedding::<G81Ext, G81Ext>::from_images(&[lower, top]).is_ok());
}

#[test]
fn non_identity_equal_degree_maps_return_preimages() {
    // A non-identity automorphism of GF(9): trace and norm return the
    // preimage in the source rather than the target value.
    let conj =
        Isomorphism::<G9Alt, G9Alt>::from_images(&[Elem::<G9Alt>::from_raw([0, 2])]).unwrap();
    let a = Elem::<G9Alt>::from_raw([1, 1]);
    let image = conj.apply(a);
    assert_ne!(image, a);
    let embedding = conj.to_embedding();
    assert_eq!(embedding.trace(image), a);
    assert_eq!(embedding.norm(image), a);
}

#[test]
fn prime_rows_embed_with_no_images() {
    // Residue at 2 against the binary degree-one row, both directions, and
    // their inclusions into extensions.
    type Res2 = FiniteField<2, 1, Residue>;
    type Bin1 = FiniteField<2, 1, Polynomial<3>>;
    assert_eq!(Embedding::<Res2, Bin1>::generator_count(), 0);
    assert_eq!(Embedding::<Bin1, Res2>::generator_count(), 0);
    let there = Embedding::<Res2, Bin1>::new().unwrap();
    let back = Embedding::<Bin1, Res2>::new().unwrap();
    assert!(there.to_images().is_empty() && back.to_images().is_empty());
    for r in 0..2u64 {
        let x = Elem::<Res2>::from_residue(r).unwrap();
        let y = Elem::<Bin1>::from_residue(r).unwrap();
        assert_eq!(there.embed(x), y);
        assert_eq!(there.restrict(y), Some(x));
        assert_eq!(back.embed(y), x);
        assert_eq!(back.restrict(x), Some(y));
    }
    // Into extensions: the prime subfield of a tower.
    let into_tower = Embedding::<Res2, G16Ext>::new().unwrap();
    assert_eq!(
        into_tower.to_images().len(),
        Embedding::<Res2, G16Ext>::generator_count()
    );
    let one = into_tower.embed(Elem::ONE);
    assert_eq!(into_tower.restrict(one), Some(Elem::ONE));
    // An extension over Residue at 2, which lacks `BinaryField`.
    struct OverRes2Spec;
    impl ExtensionSpec<2> for OverRes2Spec {
        type Base = Res2;
        type Storage = Components;
        const NAME: &'static str = "GF(4) over Residue at 2";
        const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
    }
    type OverRes2 = FiniteField<2, 2, Extension<OverRes2Spec, 2>>;
    let e = Embedding::<Res2, OverRes2>::new().unwrap();
    for r in 0..2u64 {
        let x = Elem::<Res2>::from_residue(r).unwrap();
        assert_eq!(e.restrict(e.embed(x)), Some(x));
    }
}

#[test]
fn widest_prime_readout_embeds_cleanly() {
    const P: u64 = 0xFFFF_FFFF_FFFF_FFC5;
    type Wide = FiniteField<P, 1, Residue>;
    struct WideSquareSpec;
    impl ExtensionSpec<2> for WideSquareSpec {
        type Base = Wide;
        type Storage = Components;
        const NAME: &'static str = "widest prime square";
        const RELATION: Relation = Relation::from_coordinates(&[P - 2, 0]);
    }
    type WideSquare = FiniteField<P, 2, Extension<WideSquareSpec, 2>>;
    let e = Embedding::<Wide, WideSquare>::new().unwrap();
    let max = Elem::<Wide>::from_residue(u64::MAX % P).unwrap();
    assert_eq!(e.restrict(e.embed(max)), Some(max));
}

#[test]
fn composition_crosses_layouts_and_counts() {
    // Through a third field with a different `Raw` layout and a different
    // generator count: GF(2) into GF(4) into GF(16).
    type Bin1 = FiniteField<2, 1, Polynomial<3>>;
    let e1 = Embedding::<Bin1, G4>::new().unwrap();
    let e2 = Embedding::<G4, G16Ext>::new().unwrap();
    let composed = e1.then(&e2);
    for s in elements::<Bin1>(2) {
        assert_eq!(composed.embed(s), e2.embed(e1.embed(s)));
    }
    assert_eq!(
        Embedding::<Bin1, G16Ext>::generator_count(),
        0,
        "prime sources hold no levels"
    );
}
