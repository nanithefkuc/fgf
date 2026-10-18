//! The prepared Frobenius map of one field: identity, Vieta conjugation
//! over a quadratic relation root, or compiled prime-coordinate columns.
//!
//! `Frobenius<F>` compiles the arm of `x |-> x^(p^k)` once, so repeated
//! application never re-evaluates exponentiation. The column arms evaluate
//! the scalar power on the prime-coordinate basis vectors at construction;
//! application only folds the stored columns. The scalar spelling of the
//! map is `Elem::pow`; the compact [`Embedding::frobenius`](super::Embedding::frobenius)
//! stays pow-based as the reference behaviour these prepared arms must
//! match. Map evaluation is variable-time and unsuitable for secret data.
//!
//! Requires `alloc`: the column arms own their entries, so this module is
//! alloc-only.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::super::extension::descriptor::COORDINATES;
use super::super::private::RowAccess;
use super::super::{Elem, Field};
use super::super::{FieldElem, PrimeIdentity, ResidueRow, Validate};
use super::quadratic;

/// The closed-form conjugation coefficient of a quadratic relation root,
/// shared with the compact embedding arms.
use super::quadratic::ConjugationCoefficient;

/// A compiled Frobenius arm.
///
/// Identity and Vieta conjugation run through the crate-private descriptor
/// interpreter; the column arms own their entries.
#[derive(Clone, Debug)]
enum CompiledMap<F: Field> {
    /// The identity arm, at a multiple of the field degree.
    Identity,
    /// Closed-form Vieta conjugation over a quadratic relation root.
    Conjugation {
        base_degree: u8,
        coefficient: ConjugationCoefficient,
    },
    /// Characteristic-two coordinate columns in source-major order.
    BinaryColumns {
        inputs: u8,
        outputs: u8,
        entries: Box<[u64]>,
    },
    /// Odd-characteristic columns as prime-row elements, source-major.
    ResidueColumns {
        inputs: u8,
        outputs: u8,
        entries: Box<[Elem<ResidueRow<F>>]>,
    },
}

/// Compiled from structure and checked against the scalar path:
/// construction evaluates the scalar power on the prime-coordinate basis
/// vectors, and every arm agrees with `Elem::pow` element by element.
///
/// Holds the compiled arm of `x |-> x^(p^k)` with `k` normalized to the
/// field degree, so application never re-evaluates exponentiation.
/// Requires `alloc`: the column arms own their entries.
pub struct Frobenius<F: Field> {
    power: u32,
    map: CompiledMap<F>,
}

// A derive would bound on `F: Clone`; the manual impl needs only the
// field contract the type already carries. Cloning copies the column
// entries.
impl<F: Field> Clone for Frobenius<F> {
    fn clone(&self) -> Self {
        Self {
            power: self.power,
            map: self.map.clone(),
        }
    }
}
impl<F: Field> Frobenius<F> {
    /// Prepare the Frobenius map `x |-> x^(p^k)`.
    ///
    /// The power normalizes modulo the field degree, so zero, multiples
    /// of the degree, and `u32::MAX` are all accepted.
    #[must_use]
    pub fn new(power: u32) -> Self {
        let () = Validate::<F>::OK;
        let degree = F::DEGREE;
        debug_assert!(degree > 0, "field degree must be positive");
        let normalized = power % degree;
        let map = compile::<F>(normalized);
        Self {
            power: normalized,
            map,
        }
    }

    /// The normalized power: the requested `k` modulo the field degree.
    #[must_use]
    pub fn power(&self) -> u32 {
        self.power
    }
}

/// The closed-form arm of the `k`th Frobenius power of `F`, when one
/// applies: identity at a vanishing power, Vieta conjugation at a
/// quadratic relation root whose base is fixed pointwise with the
/// indeterminate moved. The column compiler covers every other root shape.
fn closed_form<F: Field>(power: u32) -> Option<CompiledMap<F>> {
    if power == 0 {
        return Some(CompiledMap::Identity);
    }
    let (base_degree, coefficient) = conjugation::<F>()?;
    let base = u32::from(base_degree);
    if !power.is_multiple_of(base) {
        return None;
    }
    // The power is a multiple of the base degree, so the base is fixed
    // pointwise; an odd multiple moves the indeterminate once, an even
    // multiple returns it.
    if (power / base).is_multiple_of(2) {
        return Some(CompiledMap::Identity);
    }
    Some(CompiledMap::Conjugation {
        base_degree,
        coefficient,
    })
}

/// Compile the arm of the `k`th Frobenius power of `F`.
///
/// The closed-form arms come first; compiled prime-coordinate columns
/// cover every other root shape.
fn compile<F: Field>(power: u32) -> CompiledMap<F> {
    closed_form::<F>(power).unwrap_or_else(|| compile_columns::<F>(power))
}

/// The Vieta data of a quadratic relation root, when the field has one.
///
/// Returns the absolute base degree with the linear-coefficient case:
/// vanishing, unit, or general. Any other root shape — prime leaves,
/// flat polynomial leaves, basis leaves, wider relations — reports none
/// and compiles to columns.
fn conjugation<F: Field>() -> Option<(u8, ConjugationCoefficient)> {
    quadratic::conjugation_shape::<F>()
}

/// The scalar exponent of the `k`th Frobenius power: `p^k`.
///
/// The normalized power stays below the degree, so the product stays
/// below the field order, which fits `u128` by validation.
fn pow_exp(characteristic: u64, power: u32) -> u128 {
    let mut exponent = 1u128;
    let mut k = 0;
    while k < power {
        exponent *= u128::from(characteristic);
        k += 1;
    }
    exponent
}

/// The `j`th prime-coordinate basis vector of `F`: coordinate `j` set to
/// the prime one, every other coordinate zero.
///
/// Shared with the prepared embeddings and isomorphisms, which compile
/// their columns over the same basis.
pub(super) fn basis_vector<F: Field>(j: usize) -> Elem<F> {
    let degree = F::DEGREE as usize;
    let mut coordinates = [0u64; COORDINATES];
    coordinates[j] = 1;
    Elem::<F>::from_raw(F::from_prime_coordinates(&coordinates[..degree]))
}

/// Apply one compiled arm to an element.
fn apply_map<F: Field>(map: &CompiledMap<F>, value: Elem<F>) -> Elem<F> {
    match map {
        CompiledMap::Identity => value,
        CompiledMap::Conjugation {
            base_degree,
            coefficient,
        } => quadratic::apply_conjugation::<F>(value, *base_degree, *coefficient),
        CompiledMap::BinaryColumns { .. } => apply_binary::<F>(map, value),
        CompiledMap::ResidueColumns { .. } => apply_residue::<F>(map, value),
    }
}

/// Compile the general arm: the prime-coordinate columns of the map.
///
/// Each column evaluates the existing scalar power on one
/// prime-coordinate basis vector, so exponentiation runs once per column
/// at construction and never at application.
fn compile_columns<F: Field>(power: u32) -> CompiledMap<F> {
    let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
    let exponent = pow_exp(characteristic, power);
    let degree = F::DEGREE as usize;
    // Every degree the crate builds fits the column tag width.
    #[allow(clippy::cast_possible_truncation)]
    let width = degree as u8;
    if characteristic == 2 {
        let mut entries = Vec::with_capacity(degree);
        for j in 0..degree {
            let basis = basis_vector::<F>(j);
            entries.push(F::coordinate_word(basis.pow(exponent).raw));
        }
        CompiledMap::BinaryColumns {
            inputs: width,
            outputs: width,
            entries: entries.into_boxed_slice(),
        }
    } else {
        let mut entries = Vec::with_capacity(degree * degree);
        for j in 0..degree {
            let basis = basis_vector::<F>(j);
            let image = basis.pow(exponent);
            let mut coordinates = [0u64; COORDINATES];
            F::prime_coordinates_into(&mut coordinates[..degree], image.raw);
            for coordinate in &coordinates[..degree] {
                entries.push(Elem::<ResidueRow<F>>::from_raw(
                    ResidueRow::<F>::from_prime(*coordinate),
                ));
            }
        }
        CompiledMap::ResidueColumns {
            inputs: width,
            outputs: width,
            entries: entries.into_boxed_slice(),
        }
    }
}

/// The shared binary column fold: the image is the XOR of the entries
/// selected by the set bits of the input coordinate word.
///
/// Used by every binary column application — Frobenius, forward and
/// inverse maps, and the trace — so the hot loop compiles once.
#[inline]
pub(super) fn fold_binary_columns(entries: &[u64], mut word: u64) -> u64 {
    let mut image = 0u64;
    let mut j = 0;
    while word != 0 {
        if word & 1 == 1 {
            image ^= entries[j];
        }
        word >>= 1;
        j += 1;
    }
    image
}

/// The shared residue column accumulation: each image coordinate
/// accumulates the input coordinates against the stored prime-row
/// entries, through the prime row's own arithmetic.
///
/// Entries hold one row of `width` per input coordinate. Used by every
/// residue column application — Frobenius, forward and inverse maps, and
/// the trace — so the hot loop compiles once.
#[inline]
pub(super) fn accumulate_residue_columns<T: Field>(
    entries: &[Elem<ResidueRow<T>>],
    coordinates: &[u64],
    width: usize,
    image: &mut [Elem<ResidueRow<T>>; COORDINATES],
) {
    for (j, coordinate) in coordinates.iter().enumerate() {
        if *coordinate == 0 {
            continue;
        }
        let factor = Elem::<ResidueRow<T>>::from_residue(*coordinate)
            .expect("input coordinate below the characteristic");
        for (o, slot) in image[..width].iter_mut().enumerate() {
            *slot += entries[j * width + o] * factor;
        }
    }
}

/// Apply binary columns: the image is the XOR of the columns selected by
/// the set bits of the input coordinate word.
fn apply_binary<F: Field>(map: &CompiledMap<F>, value: Elem<F>) -> Elem<F> {
    let CompiledMap::BinaryColumns {
        inputs,
        outputs,
        entries,
    } = map
    else {
        debug_assert!(false, "binary application on a non-binary arm");
        return value;
    };
    debug_assert!(usize::from(*inputs) == entries.len(), "column count");
    debug_assert!(usize::from(*outputs) == F::DEGREE as usize, "row count");
    let image = fold_binary_columns(entries, F::coordinate_word(value.raw));
    Elem {
        raw: F::from_coordinate_word(image),
    }
}

/// Apply residue columns: each image coordinate accumulates the input
/// coordinates against the stored prime-row entries, through the prime
/// row's own arithmetic.
fn apply_residue<F: Field>(map: &CompiledMap<F>, value: Elem<F>) -> Elem<F> {
    let CompiledMap::ResidueColumns {
        inputs,
        outputs,
        entries,
    } = map
    else {
        debug_assert!(false, "residue application on a non-residue arm");
        return value;
    };
    let degree = F::DEGREE as usize;
    debug_assert!(usize::from(*inputs) == degree, "column count");
    debug_assert!(usize::from(*outputs) == degree, "row count");
    let mut coordinates = [0u64; COORDINATES];
    F::prime_coordinates_into(&mut coordinates[..degree], value.raw);
    let mut image = [Elem::<ResidueRow<F>>::ZERO; COORDINATES];
    accumulate_residue_columns::<F>(entries, &coordinates[..degree], degree, &mut image);
    let mut residues = [0u64; COORDINATES];
    for (o, slot) in image[..degree].iter().enumerate() {
        residues[o] = slot
            .to_residue()
            .expect("degree-one readout covers the row");
    }
    Elem {
        raw: F::from_prime_coordinates(&residues[..degree]),
    }
}

impl<F: Field> Elem<F> {
    /// Apply a prepared Frobenius map: the image of `self` under
    /// `x |-> x^(p^k)` for the power the map was prepared with.
    ///
    /// Construction normalizes the power and compiles the arm, so steady
    /// state application allocates nothing.
    #[must_use]
    pub fn frobenius_with(self, prepared: &Frobenius<F>) -> Self {
        apply_map::<F>(&prepared.map, self)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    //! Differentials of every compiled arm against scalar exponentiation.
    //!
    //! Each arm's application equals `pow(p^k)` element by element:
    //! exhaustive at small degrees across every storage class, sampled
    //! at the wide towers and odd rows, with composition, identity, and
    //! normalization checks beside the anchors.

    use super::super::super::binary::tower::{
        FanPaar2Field, FanPaar4Field, FanPaar8Field, FanPaar16Field, FanPaar32Field,
        FanPaar64Field, Rijndael16Field, Rijndael32Field, Rijndael64Field,
    };
    use super::super::super::extension::Extension;
    use super::super::super::finite_field::FiniteField;
    use super::super::super::quad_mersenne31::QuadMersenne31;
    use super::super::super::{AES, Cantor, Normal, Polynomial, RS};
    use super::super::super::{Components, FieldElem, Relation};
    use super::super::super::{ExtensionSpec, Field, Mersenne31, Residue};
    use super::{Frobenius, pow_exp};

    /// GF(4) over the residue-two base: the word hooks recurse through
    /// the residue leaf.
    struct OverRes2Spec;
    impl ExtensionSpec<2> for OverRes2Spec {
        type Base = FiniteField<2, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(4) over Residue at 2";
        const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
    }

    /// GF(9) as `x^2 + 2x + 2`: a general linear coefficient.
    struct G9Spec;
    impl ExtensionSpec<2> for G9Spec {
        type Base = FiniteField<3, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(9) as x^2 + 2x + 2";
        const RELATION: Relation = Relation::from_coordinates(&[2, 2]);
    }

    /// GF(49) as `x^2 - 3`: a vanishing linear coefficient.
    struct G49OddSpec;
    impl ExtensionSpec<2> for G49OddSpec {
        type Base = FiniteField<7, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(49) as x^2 - 3";
        const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
    }

    /// GF(9) as `a^2 + 1`: the base of the towered odd conjugation.
    struct G9AltSpec;
    impl ExtensionSpec<2> for G9AltSpec {
        type Base = FiniteField<3, 1, Residue>;
        type Storage = Components;
        const NAME: &'static str = "GF(9) as a^2 + 1";
        const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
    }
    /// GF(9) as `a^2 + 1`, named for the towered extension over it.
    type G9Alt = FiniteField<3, 2, Extension<G9AltSpec, 2>>;

    /// GF(81) as `t^2 - (1 + a)`: towered odd conjugation over a
    /// quadratic base.
    struct G81ExtSpec;
    impl ExtensionSpec<2> for G81ExtSpec {
        type Base = G9Alt;
        type Storage = Components;
        const NAME: &'static str = "GF(81) as t^2 - (1 + a)";
        const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
    }

    /// The scalar oracle: `x^(p^k)` by the existing exponentiation.
    fn oracle<F: Field>(
        value: super::super::super::Elem<F>,
        power: u32,
    ) -> super::super::super::Elem<F> {
        let characteristic =
            <<F as Field>::Characteristic as super::super::super::PrimeIdentity>::CHARACTERISTIC;
        value.pow(pow_exp(characteristic, power))
    }

    /// Every element of `F` agrees with the oracle at every power,
    /// alongside the composition and normalization laws.
    fn check_exhaustive<F: Field>(powers: &[u32]) {
        let degree = F::DEGREE;
        for index in 0..F::ORDER {
            let value = super::super::super::Elem::<F>::from_index(index)
                .expect("index below the field order");
            for &power in powers {
                let prepared = Frobenius::<F>::new(power);
                assert_eq!(
                    value.frobenius_with(&prepared),
                    oracle(value, power % degree)
                );
            }
        }
        // Composition adds powers; the degree power is the identity.
        for index in 0..F::ORDER {
            let value = super::super::super::Elem::<F>::from_index(index)
                .expect("index below the field order");
            for &a in powers {
                for &b in powers {
                    let first = Frobenius::<F>::new(a);
                    let second = Frobenius::<F>::new(b);
                    let composed = Frobenius::<F>::new(
                        u32::try_from((u64::from(a) + u64::from(b)) % u64::from(degree))
                            .expect("reduced power fits in u32"),
                    );
                    assert_eq!(
                        value.frobenius_with(&first).frobenius_with(&second),
                        value.frobenius_with(&composed)
                    );
                }
            }
        }
        let identity = Frobenius::<F>::new(degree);
        let zero = Frobenius::<F>::new(0);
        let wrapped = Frobenius::<F>::new(u32::MAX);
        assert_eq!(wrapped.power(), u32::MAX % degree);
        for index in 0..F::ORDER {
            let value = super::super::super::Elem::<F>::from_index(index)
                .expect("index below the field order");
            assert_eq!(value.frobenius_with(&identity), value);
            assert_eq!(value.frobenius_with(&zero), value);
        }
    }

    /// A deterministic sample of `F` agrees with the oracle at `powers`.
    fn check_sampled<F: Field>(count: u64, powers: &[u32]) {
        let mut state = 0x9E37_79B9_7F4A_7C15u128;
        for _ in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let value = super::super::super::Elem::<F>::from_index(state % F::ORDER)
                .expect("index below the field order");
            for &power in powers {
                let prepared = Frobenius::<F>::new(power);
                assert_eq!(
                    value.frobenius_with(&prepared),
                    oracle(value, power % F::DEGREE)
                );
            }
        }
    }

    #[test]
    fn flat_and_basis_rows_match_scalar_powers() {
        check_exhaustive::<FiniteField<2, 1, Polynomial<3>>>(&[0, 1, 2, u32::MAX]);
        check_exhaustive::<FiniteField<2, 2, Polynomial<0x7>>>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<FiniteField<2, 2, Normal<7, 2>>>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<FiniteField<2, 2, Cantor<7, 2>>>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<FiniteField<2, 4, Polynomial<0x13>>>(&[0, 1, 3, 4, 5, u32::MAX]);
        check_exhaustive::<FiniteField<2, 4, Normal<0x13, 8>>>(&[0, 1, 3, 4, 5, u32::MAX]);
        check_exhaustive::<FiniteField<2, 8, Polynomial<AES>>>(&[0, 1, 2, 7, 8, 9, u32::MAX]);
        check_sampled::<FiniteField<2, 8, Polynomial<RS>>>(256, &[0, 1, 7, 8, 9, u32::MAX]);
        check_sampled::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(256, &[0, 1, 7, 8, 9, u32::MAX]);
    }

    #[test]
    fn tower_rows_match_scalar_powers() {
        check_exhaustive::<FanPaar2Field>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<FanPaar4Field>(&[0, 1, 2, 3, 4, 5, u32::MAX]);
        check_sampled::<FanPaar8Field>(512, &[0, 1, 7, 8, 9, u32::MAX]);
        check_sampled::<FanPaar16Field>(256, &[0, 1, 8, 15, 16, 17, u32::MAX]);
        check_sampled::<FanPaar32Field>(128, &[0, 1, 16, 31, 32, 33, u32::MAX]);
        check_sampled::<FanPaar64Field>(64, &[0, 1, 32, 63, 64, 65, u32::MAX]);
        check_sampled::<Rijndael16Field>(256, &[0, 1, 8, 15, 16, 17, u32::MAX]);
        check_sampled::<Rijndael32Field>(128, &[0, 1, 16, 31, 32, 33, u32::MAX]);
        check_sampled::<Rijndael64Field>(64, &[0, 1, 32, 63, 64, 65, u32::MAX]);
    }

    #[test]
    fn packed_and_component_rows_match_scalar_powers() {
        check_sampled::<FiniteField<2, 32, Extension<super::super::super::Rijndael32, 2>>>(
            64,
            &[0, 1, 31, 32, 33, u32::MAX],
        );
        check_sampled::<FiniteField<2, 16, Extension<super::super::super::FanPaar16, 2>>>(
            128,
            &[0, 1, 15, 16, 17, u32::MAX],
        );
        check_exhaustive::<FiniteField<2, 2, Extension<OverRes2Spec, 2>>>(&[0, 1, 2, 3, u32::MAX]);
    }

    /// A quadratic extension of the normal-basis GF(4): the base's word
    /// one is raw 3, so a raw-word comparison against 1 misfires here.
    struct OverNormal4Spec;
    impl ExtensionSpec<2> for OverNormal4Spec {
        type Base = FiniteField<2, 2, Normal<7, 2>>;
        type Storage = Components;
        const NAME: &'static str = "GF(16) over normal-basis GF(4)";
        const RELATION: Relation =
            Relation::from_binary_words::<FiniteField<2, 2, Normal<7, 2>>>(&[1, 1]);
    }

    #[test]
    fn normal_basis_quadratic_matches_scalar_powers() {
        type E = FiniteField<2, 4, Extension<OverNormal4Spec, 2>>;
        type N = FiniteField<2, 2, Normal<7, 2>>;
        // The base word one is raw 3, not raw 1: the conjugation arm must
        // compare against the base identity, not the raw word.
        assert_eq!(
            super::super::super::Elem::<N>::ONE.to_raw(),
            3,
            "this regression needs raw 1 below the base one"
        );
        // The reported case: the order-two prepared map against `x^4`.
        let reported = super::super::super::Elem::<E>::from_raw([0, 3]);
        assert_eq!(
            reported.frobenius_with(&Frobenius::<E>::new(2)),
            reported.pow(4)
        );
        check_exhaustive::<E>(&[0, 1, 2, 3, 4, u32::MAX]);
    }

    #[test]
    fn odd_rows_match_scalar_powers() {
        check_exhaustive::<FiniteField<3, 1, Residue>>(&[0, 1, 2, u32::MAX]);
        check_exhaustive::<FiniteField<7, 1, Residue>>(&[0, 1, 2, u32::MAX]);
        check_sampled::<Mersenne31>(512, &[0, 1, 2, u32::MAX]);
        check_exhaustive::<FiniteField<3, 2, Extension<G9Spec, 2>>>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<FiniteField<7, 2, Extension<G49OddSpec, 2>>>(&[0, 1, 2, 3, u32::MAX]);
        check_exhaustive::<G9Alt>(&[0, 1, 2, 3, u32::MAX]);
        check_sampled::<FiniteField<3, 4, Extension<G81ExtSpec, 2>>>(
            512,
            &[0, 1, 3, 4, 5, u32::MAX],
        );
        check_sampled::<QuadMersenne31>(512, &[0, 1, 2, 3, u32::MAX]);
    }

    #[test]
    fn qm31_preparation_is_conjugation() {
        let prepared = Frobenius::<QuadMersenne31>::new(1);
        let values = [
            super::super::super::Elem::<QuadMersenne31>::from_raw([7, 11]),
            super::super::super::Elem::<QuadMersenne31>::from_raw([0x7FFF_FFFE, 0x7FFF_FFFF]),
            super::super::super::Elem::<QuadMersenne31>::ZERO,
            super::super::super::Elem::<QuadMersenne31>::ONE,
        ];
        for value in values {
            assert_eq!(value.frobenius_with(&prepared), value.conjugate());
        }
    }

    #[test]
    fn rijndael_step_is_tower_conjugation() {
        // The order-eight power fixes the GF(2^8) base pointwise and
        // shears the upper coordinate: the tower conjugation.
        let prepared = Frobenius::<Rijndael16Field>::new(8);
        for index in 0..Rijndael16Field::ORDER {
            let value = super::super::super::Elem::<Rijndael16Field>::from_index(index)
                .expect("index below the field order");
            assert_eq!(value.frobenius_with(&prepared), oracle(value, 8));
        }
        let word_case = super::super::super::Elem::<Rijndael16Field>::from_raw(0x0100);
        let sheared = word_case.frobenius_with(&prepared).to_raw();
        assert_eq!(sheared >> 8, 1, "upper coordinate negates to itself here");
        assert_eq!(oracle(word_case, 8).to_raw(), sheared);
    }
}
