//! Residue and index conversions across every row family.
//!
//! `from_residue`/`to_residue` name the prime-subfield value independently
//! of presentation; `from_index`/`to_index` name the presentation index,
//! the flattened prime-subfield coordinates as base-characteristic digits.
//! Exhaustive rows round-trip every element; wide rows check boundaries
//! and deterministic samples. Residues agree with repeated addition at
//! small primes and double-and-add at full width.

use fgf::{
    AES, Cantor, ComplexMersenne31, Components, Elem, Extension, ExtensionField, ExtensionSpec,
    Field, FieldElem, FiniteField, Goldilocks, Mersenne31, Normal, Polynomial, QuadMersenne31,
    RangeError, Relation, Residue,
};

/// Every index round-trips, and the order and `u128::MAX` are rejected.
fn index_round_trip<F: Field>() {
    for index in 0..F::ORDER {
        let element = Elem::<F>::from_index(index).unwrap();
        assert_eq!(element.to_index(), index, "index {index}");
    }
    assert_eq!(Elem::<F>::from_index(F::ORDER), Err(RangeError));
    assert_eq!(Elem::<F>::from_index(u128::MAX), Err(RangeError));
}

/// Every residue round-trips through the prime subfield.
fn residue_round_trip<F: Field>(characteristic: u64) {
    for residue in 0..u128::from(characteristic) {
        let residue = u64::try_from(residue).unwrap();
        let element = Elem::<F>::from_residue(residue).unwrap();
        assert_eq!(element.to_residue(), Some(residue), "residue {residue}");
    }
    assert_eq!(
        Elem::<F>::from_residue(characteristic),
        Err(RangeError),
        "characteristic rejected"
    );
}

/// `from_residue(n)` equals `n · ONE` by repeated addition.
fn residue_matches_repeated_addition<F: Field>(characteristic: u64) {
    let mut sum = Elem::<F>::ZERO;
    for residue in 0..characteristic {
        assert_eq!(Elem::<F>::from_residue(residue).unwrap(), sum);
        sum += Elem::<F>::ONE;
    }
}

/// `from_residue` rejects narrowed aliases: values whose low bits look
/// valid but sit at or above the characteristic.
fn residue_rejects_aliases<F: Field>(values: &[u64]) {
    for &value in values {
        assert_eq!(
            Elem::<F>::from_residue(value),
            Err(RangeError),
            "alias {value:#x}"
        );
    }
}

#[test]
fn residue_rows_round_trip_indices_and_residues() {
    index_round_trip::<FiniteField<2, 1, Residue>>();
    index_round_trip::<FiniteField<3, 1, Residue>>();
    index_round_trip::<FiniteField<5, 1, Residue>>();
    index_round_trip::<FiniteField<7, 1, Residue>>();
    residue_round_trip::<FiniteField<2, 1, Residue>>(2);
    residue_round_trip::<FiniteField<3, 1, Residue>>(3);
    residue_round_trip::<FiniteField<5, 1, Residue>>(5);
    residue_round_trip::<FiniteField<7, 1, Residue>>(7);
    residue_matches_repeated_addition::<FiniteField<2, 1, Residue>>(2);
    residue_matches_repeated_addition::<FiniteField<3, 1, Residue>>(3);
    residue_matches_repeated_addition::<FiniteField<5, 1, Residue>>(5);
    residue_matches_repeated_addition::<FiniteField<7, 1, Residue>>(7);
    // On every degree-one row the index is the ordinary residue.
    for residue in 0..7u64 {
        assert_eq!(
            Elem::<FiniteField<7, 1, Residue>>::from_index(residue as u128).unwrap(),
            Elem::<FiniteField<7, 1, Residue>>::from_residue(residue).unwrap()
        );
    }
    for residue in 0..2u64 {
        assert_eq!(
            Elem::<FiniteField<2, 1, Polynomial<3>>>::from_index(residue as u128).unwrap(),
            Elem::<FiniteField<2, 1, Polynomial<3>>>::from_residue(residue).unwrap()
        );
    }
}

#[test]
fn binary_flat_rows_round_trip_every_index() {
    index_round_trip::<FiniteField<2, 1, Polynomial<3>>>();
    index_round_trip::<FiniteField<2, 2, Polynomial<0x7>>>();
    index_round_trip::<FiniteField<2, 2, Normal<7, 2>>>();
    index_round_trip::<FiniteField<2, 2, Cantor<7, 2>>>();
    index_round_trip::<FiniteField<2, 4, Polynomial<0x13>>>();
    index_round_trip::<FiniteField<2, 4, Normal<0x13, 8>>>();
    index_round_trip::<FiniteField<2, 4, Cantor<0x13, 8>>>();
    index_round_trip::<FiniteField<2, 8, Polynomial<AES>>>();
    // The binary degree-one index is the coordinate word, and the residue
    // covers exactly the prime subfield.
    residue_round_trip::<FiniteField<2, 1, Polynomial<3>>>(2);
    residue_round_trip::<FiniteField<2, 8, Polynomial<AES>>>(2);
    residue_rejects_aliases::<FiniteField<2, 8, Polynomial<AES>>>(&[256, 512, u64::MAX]);
}

#[test]
fn extension_rows_round_trip_every_index() {
    struct Gf4Spec;
    impl ExtensionSpec<2> for Gf4Spec {
        type Base = FiniteField<2, 1, Polynomial<3>>;
        type Storage = Components;
        const NAME: &'static str = "GF(4)";
        const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
    }
    struct Gf9Spec;
    impl ExtensionSpec<2> for Gf9Spec {
        type Base = FiniteField<3, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(9)";
        const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
    }
    struct Gf25Spec;
    impl ExtensionSpec<2> for Gf25Spec {
        type Base = FiniteField<5, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(25) as x^2 + 4x + 2";
        const RELATION: Relation = Relation::from_coordinates(&[2, 4]);
    }
    struct Gf49Spec;
    impl ExtensionSpec<2> for Gf49Spec {
        type Base = FiniteField<7, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(49) as x^2 - 3";
        const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
    }
    struct Gf81QuarticSpec;
    impl ExtensionSpec<4> for Gf81QuarticSpec {
        type Base = FiniteField<3, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(81) as x^4 + 2x^2 + 2";
        const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
    }
    index_round_trip::<FiniteField<2, 2, Extension<Gf4Spec, 2>>>();
    index_round_trip::<FiniteField<3, 2, Extension<Gf9Spec, 2>>>();
    index_round_trip::<FiniteField<5, 2, Extension<Gf25Spec, 2>>>();
    index_round_trip::<FiniteField<7, 2, Extension<Gf49Spec, 2>>>();
    index_round_trip::<FiniteField<3, 4, Extension<Gf81QuarticSpec, 4>>>();
    residue_round_trip::<FiniteField<2, 2, Extension<Gf4Spec, 2>>>(2);
    residue_round_trip::<FiniteField<3, 2, Extension<Gf9Spec, 2>>>(3);
    residue_round_trip::<FiniteField<5, 2, Extension<Gf25Spec, 2>>>(5);
    residue_round_trip::<FiniteField<7, 2, Extension<Gf49Spec, 2>>>(7);
    residue_matches_repeated_addition::<FiniteField<3, 2, Extension<Gf9Spec, 2>>>(3);
    residue_matches_repeated_addition::<FiniteField<5, 2, Extension<Gf25Spec, 2>>>(5);
    // Off the prime subfield the readout is `None`, counted across the
    // row: every non-constant index reads back `None`.
    let off = Elem::<FiniteField<3, 2, Extension<Gf9Spec, 2>>>::from_index(4).unwrap();
    assert_eq!(off.to_residue(), None);
    let mut none_count = 0;
    for index in 0..9u128 {
        if Elem::<FiniteField<3, 2, Extension<Gf9Spec, 2>>>::from_index(index)
            .unwrap()
            .to_residue()
            .is_none()
        {
            none_count += 1;
        }
    }
    assert_eq!(none_count, 9 - 3);
}

#[test]
fn normal_seven_two_marks_the_index_residue_split() {
    type Row = FiniteField<2, 2, Normal<7, 2>>;
    // The normal basis one is the sum of the basis vectors, index 3, while
    // its prime-subfield value is residue 1.
    let one = Elem::<Row>::ONE;
    assert_eq!(one.to_index(), 3);
    assert_eq!(one.to_residue(), Some(1));
    assert_eq!(Elem::<Row>::from_index(3).unwrap(), one);
    assert_eq!(Elem::<Row>::from_residue(1).unwrap(), one);
}

#[test]
fn quad_mersenne31_index_is_real_plus_imaginary_times_p() {
    const P: u64 = 0x7FFF_FFFF;
    // Limbs stay below the modulus, so the index is `re + im * p`, and
    // decoding the index recovers the limbs.
    for (re, im) in [(0u64, 0u64), (1, 0), (0, 1), (1, 1), (P - 1, P - 1)] {
        let element = Elem::<QuadMersenne31>::from_raw([re as u32, im as u32]);
        let index = re as u128 + im as u128 * P as u128;
        assert_eq!(element.to_index(), index);
        assert_eq!(Elem::<QuadMersenne31>::from_index(index).unwrap(), element);
    }
    assert_eq!(
        Elem::<QuadMersenne31>::from_index(P as u128 * P as u128),
        Err(RangeError)
    );
    // The prime subfield is the real axis.
    assert_eq!(
        Elem::<QuadMersenne31>::from_residue(5)
            .unwrap()
            .to_residue(),
        Some(5)
    );
    assert_eq!(
        Elem::<QuadMersenne31>::from_index(P as u128)
            .unwrap()
            .to_residue(),
        None
    );
}

#[test]
fn complex_mersenne31_shares_the_quad_mersenne31_index() {
    type Spelled = FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>;
    let element = Elem::<Spelled>::from_raw([3, 5]);
    assert_eq!(
        element.to_index(),
        Elem::<QuadMersenne31>::from_raw([3, 5]).to_index()
    );
}

#[test]
fn widest_prime_rows_agree_with_double_and_add() {
    const P: u64 = 0xFFFF_FFFF_FFFF_FFC5;
    type Row = FiniteField<P, 1, Residue>;
    // `from_residue(n)` equals the double-and-add multiple of one over
    // sampled residues, accumulating `ONE` into a binary decomposition.
    for n in [0u64, 1, 2, 3, 17, 255, 65535, P - 1, P - 2, u64::MAX % P] {
        let mut acc = Elem::<Row>::ZERO;
        let mut bit = 1u64;
        let mut addend = Elem::<Row>::ONE;
        while bit != 0 {
            if n & bit != 0 {
                acc += addend;
            }
            addend += addend;
            bit = bit.wrapping_mul(2);
            if bit == 0 {
                break;
            }
        }
        assert_eq!(Elem::<Row>::from_residue(n).unwrap(), acc, "residue {n}");
    }
    assert_eq!(Elem::<Row>::from_residue(P), Err(RangeError));
    assert_eq!(Elem::<Row>::from_residue(u64::MAX), Err(RangeError));
    assert_eq!(Elem::<Row>::from_index(P as u128), Err(RangeError));
    assert_eq!(Elem::<Row>::from_index(u128::MAX), Err(RangeError));
    // At degree one the index is the ordinary residue.
    assert_eq!(Elem::<Row>::from_residue(17).unwrap().to_index(), 17);
}
#[test]
fn mersenne31_rejects_the_shifted_alias() {
    residue_rejects_aliases::<Mersenne31>(&[1 << 31, 1 << 32, u64::MAX]);
    assert_eq!(
        Elem::<Mersenne31>::from_residue(0x7FFF_FFFF),
        Err(RangeError)
    );
    assert_eq!(Elem::<Mersenne31>::from_index(0x7FFF_FFFF), Err(RangeError));
}

#[test]
fn residue_at_two_matches_binary_degree_one() {
    type Res2 = FiniteField<2, 1, Residue>;
    type Bin1 = FiniteField<2, 1, Polynomial<3>>;
    // Equal residues name equal values, and equal words name equal
    // indices, across the two degree-one presentations.
    for residue in 0..2u64 {
        assert_eq!(
            Elem::<Res2>::from_residue(residue).unwrap().to_residue(),
            Elem::<Bin1>::from_residue(residue).unwrap().to_residue(),
        );
    }
    for index in 0..2u128 {
        assert_eq!(
            Elem::<Res2>::from_index(index).unwrap().to_index(),
            Elem::<Bin1>::from_index(index).unwrap().to_index(),
        );
    }
}

struct GoldilocksSquareSpec;
impl ExtensionSpec<2> for GoldilocksSquareSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(p^2) over Goldilocks";
    // `t^2 - 7`: seven is a quadratic nonresidue, per the reference suite.
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}

#[test]
fn goldilocks_quadratic_indices_reach_the_order_edge() {
    type Row = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GoldilocksSquareSpec, 2>>;
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    // The last index names `(p - 1, p - 1)`: both components at their
    // maximum, ruling out a transposed digit order.
    let top = Elem::<Row>::from_index(Row::ORDER - 1).unwrap();
    assert_eq!(top.to_index(), Row::ORDER - 1);
    assert_eq!(
        <Row as ExtensionField<2>>::to_components(top),
        [
            Elem::<Goldilocks>::from_residue(P - 1).unwrap(),
            Elem::<Goldilocks>::from_residue(P - 1).unwrap(),
        ]
    );
    for index in [Row::ORDER - 1, Row::ORDER - 2] {
        assert_eq!(Elem::<Row>::from_index(index).unwrap().to_index(), index);
    }
    assert_eq!(Elem::<Row>::from_index(Row::ORDER), Err(RangeError));
}
