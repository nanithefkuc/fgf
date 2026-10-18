//! The closed-form quadratic maps of a relative-degree-2 target row: the
//! shape of the row's own relation over its immediate base, and the Vieta
//! conjugate, trace, and norm of one element through it.
//!
//! The shape detector reads the target descriptor only: relative degree
//! exactly two at the root, with the base sub-descriptor structurally the
//! source descriptor, and the coefficient class computed with
//! [`word_one`](super::super::extension::arithmetic::word_one), never a
//! raw-word comparison against one. The applicators run the same
//! interpreter arithmetic the prepared [`Frobenius`](super::frobenius::Frobenius)
//! conjugation arm uses, over the target row's coordinate hooks, so the
//! compact [`Embedding`](super::Embedding) methods and the prepared arms
//! share one implementation with no per-call allocation.
//!
//! Ungated: no `alloc` is needed anywhere on this path.

use super::super::extension::arithmetic;
use super::super::extension::descriptor::{COORDINATES, Node};
use super::super::{Elem, Field, FieldElem, PrimeIdentity};

/// The closed-form conjugation coefficient of a quadratic relation root.
///
/// The Vieta step sees only whether the linear coefficient vanishes, is
/// one in characteristic two, or needs its stored value. The general case
/// carries no payload: application re-reads the coefficient from the
/// field descriptor, which already stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConjugationCoefficient {
    /// Odd characteristic with a vanishing linear coefficient.
    Zero,
    /// Characteristic two with unit linear coefficient.
    One,
    /// A general stored linear coefficient, read from the descriptor.
    General,
}

/// The closed-form shape of the target row's own quadratic relation over
/// its immediate base: the absolute base degree with the
/// linear-coefficient case.
///
/// `None` where the pair does not route: a non-extension target, a wider
/// relative degree, a towered degree-2 pair whose root relation sits above
/// another extension level, or a source that is not the relation's base.
/// A `const` context cannot compare descriptors for structural equality,
/// so callers confirm `S::DESCRIPTOR` against the base sub-descriptor at
/// the call site; the shape alone never authorizes the arm.
pub(super) const fn conjugation_shape<T: Field>() -> Option<(u8, ConjugationCoefficient)> {
    let descriptor = T::DESCRIPTOR;
    let root = descriptor.root_index();
    let node = descriptor.node(root);
    let (base, base_degree, base_words, coefficients, characteristic) = match node {
        Node::Extension {
            base,
            relative_degree: 2,
            base_degree,
            base_words,
            coefficients,
            characteristic,
            ..
        } => (
            *base,
            *base_degree,
            *base_words,
            *coefficients,
            *characteristic,
        ),
        _ => return None,
    };
    // The base degree fits the tag width on every row the crate builds.
    #[allow(clippy::cast_possible_truncation)]
    let base_degree = base_degree as u8;
    if characteristic == 2 {
        if coefficients[1] == arithmetic::word_one(descriptor, base) {
            return Some((base_degree, ConjugationCoefficient::One));
        }
        return Some((base_degree, ConjugationCoefficient::General));
    }
    let mut linear = arithmetic::zero_element();
    let mut k = 0;
    while k < base_words {
        linear[k] = coefficients[base_words + k];
        k += 1;
    }
    if arithmetic::element_is_zero(descriptor, base, &linear) {
        return Some((base_degree, ConjugationCoefficient::Zero));
    }
    Some((base_degree, ConjugationCoefficient::General))
}

/// Whether `S` is the immediate base of `T`'s own quadratic relation: the
/// relative degree is two, the target root carries the relation, and the
/// base sub-descriptor is structurally the source descriptor.
pub(super) const fn routes<S: Field, T: Field>() -> bool {
    if T::DEGREE != 2 * S::DEGREE {
        return false;
    }
    let target = T::DESCRIPTOR;
    let root = target.root_index();
    let (base_degree,) = match target.node(root) {
        Node::Extension {
            relative_degree: 2,
            base_degree,
            ..
        } => (*base_degree,),
        _ => return false,
    };
    if base_degree != S::DEGREE {
        return false;
    }
    target.base_descriptor().same_structure(S::DESCRIPTOR)
}

/// Whether the pair is the tiny GF(4)-shaped row: absolute degrees
/// one into two at characteristic two. The relative Frobenius there is
/// a single square, so the Vieta coordinate round-trip costs more than
/// the power on the `frobenius`, `contains`, and `restrict` arms, while
/// `trace` and `norm` still route. A `const` degree/characteristic
/// check, resolved per monomorphization with the shape.
pub(super) const fn is_tiny_gf4<S: Field, T: Field>() -> bool {
    S::DEGREE == 1
        && T::DEGREE == 2
        && <S::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2
        && <T::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2
}

/// The stored linear coefficient of a quadratic relation root, as a base
/// element in interpreter form.
fn linear_coefficient<T: Field>() -> arithmetic::Element {
    let descriptor = T::DESCRIPTOR;
    let root = descriptor.root_index();
    let Node::Extension {
        base_words,
        coefficients,
        ..
    } = descriptor.node(root)
    else {
        debug_assert!(false, "stored coefficient on a non-relation root");
        return arithmetic::zero_element();
    };
    let characteristic = arithmetic::characteristic_of(descriptor, root);
    let mut coefficient = arithmetic::zero_element();
    if characteristic == 2 {
        coefficient[0] = coefficients[1];
        return coefficient;
    }
    let mut k = 0;
    while k < *base_words {
        coefficient[k] = coefficients[*base_words + k];
        k += 1;
    }
    coefficient
}

fn mask_of(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// The closed-form Vieta conjugate `x^|base|` of one element: `(a0 -
/// c1*a1) - a1*t` for `x = a0 + a1*t` under `t^2 + c1*t + c0`.
pub(super) fn apply_conjugation<T: Field>(
    value: Elem<T>,
    base_degree: u8,
    coefficient: ConjugationCoefficient,
) -> Elem<T> {
    let descriptor = T::DESCRIPTOR;
    let root = descriptor.root_index();
    let Node::Extension { base, .. } = descriptor.node(root) else {
        debug_assert!(false, "conjugation on a non-relation root");
        return value;
    };
    let base = *base;
    let characteristic = arithmetic::characteristic_of(descriptor, root);
    if characteristic == 2 {
        let word = T::coordinate_word(value.raw);
        let half = u32::from(base_degree);
        let mask = mask_of(half);
        let (lo, hi) = (word & mask, (word >> half) & mask);
        let c1 = linear_coefficient::<T>();
        let sheared = match coefficient {
            ConjugationCoefficient::One => lo ^ hi,
            ConjugationCoefficient::General => {
                lo ^ arithmetic::word_mul(descriptor, base, c1[0], hi)
            }
            ConjugationCoefficient::Zero => {
                debug_assert!(false, "vanishing coefficient is inseparable here");
                lo
            }
        };
        return Elem {
            raw: T::from_coordinate_word(sheared | (hi << half)),
        };
    }
    let degree = T::DEGREE as usize;
    let half = usize::from(base_degree);
    debug_assert!(degree == 2 * half, "conjugation spans the whole root");
    let mut coordinates = [0u64; COORDINATES];
    T::prime_coordinates_into(&mut coordinates[..degree], value.raw);
    let mut lower = arithmetic::zero_element();
    let mut upper = arithmetic::zero_element();
    let mut k = 0;
    while k < half {
        lower[k] = coordinates[k];
        upper[k] = coordinates[half + k];
        k += 1;
    }
    let image_upper = arithmetic::element_neg(descriptor, base, &upper);
    let image_lower = match coefficient {
        ConjugationCoefficient::Zero => lower,
        ConjugationCoefficient::General => {
            let product =
                arithmetic::element_mul(descriptor, base, &linear_coefficient::<T>(), &upper);
            arithmetic::element_sub(descriptor, base, &lower, &product)
        }
        ConjugationCoefficient::One => {
            debug_assert!(false, "unit coefficient is inseparable here");
            lower
        }
    };
    let mut image = [0u64; COORDINATES];
    let mut k = 0;
    while k < half {
        image[k] = image_lower[k];
        image[half + k] = image_upper[k];
        k += 1;
    }
    Elem {
        raw: T::from_prime_coordinates(&image[..degree]),
    }
}

/// The closed-form trace `x + x^|base| = 2*a - c1*b` of one element,
/// through the target row's own shape. In characteristic two the doubled
/// term vanishes and the trace is `c1*b`.
///
/// Callers pass the shape from [`conjugation_shape`]; the arm is
/// authorized only where [`routes`] also holds.
pub(super) fn apply_trace<S: Field, T: Field>(
    value: Elem<T>,
    base_degree: u8,
    coefficient: ConjugationCoefficient,
) -> Elem<S> {
    let _ = coefficient;
    let _ = base_degree;
    let conjugate = apply_conjugation::<T>(value, base_degree, coefficient);
    let sum = Elem::<T> {
        raw: add_raw::<T>(value.raw, conjugate.raw),
    };
    restrict_quad::<S, T>(sum)
}

/// The closed-form norm `x * x^|base| = a^2 - c1*a*b + c0*b^2` of one
/// element, through the target row's own shape.
///
/// Callers pass the shape from [`conjugation_shape`]; the arm is
/// authorized only where [`routes`] also holds.
pub(super) fn apply_norm<S: Field, T: Field>(
    value: Elem<T>,
    base_degree: u8,
    coefficient: ConjugationCoefficient,
) -> Elem<S> {
    let conjugate = apply_conjugation::<T>(value, base_degree, coefficient);
    let product = Elem::<T> {
        raw: mul_raw::<T>(value.raw, conjugate.raw),
    };
    restrict_quad::<S, T>(product)
}

fn add_raw<T: Field>(left: T::Raw, right: T::Raw) -> T::Raw {
    T::add_raw(left, right)
}

fn mul_raw<T: Field>(left: T::Raw, right: T::Raw) -> T::Raw {
    T::mul_raw(left, right)
}

/// Read the base component of a target element known to lie over the
/// source: the conjugate sum and product do by the Vieta identities, so
/// the upper coordinates vanish and the lower names the trace and norm.
/// Uses the source row's coordinate hooks, never the embedding images.
pub(super) fn restrict_quad<S: Field, T: Field>(value: Elem<T>) -> Elem<S> {
    let degree = T::DEGREE as usize;
    let span = S::DEGREE as usize;
    debug_assert!(degree == 2 * span, "quadratic restriction spans the row");
    let mut coordinates = [0u64; COORDINATES];
    T::prime_coordinates_into(&mut coordinates[..degree], value.raw);
    Elem::<S>::from_raw(S::from_prime_coordinates(&coordinates[..span]))
}
