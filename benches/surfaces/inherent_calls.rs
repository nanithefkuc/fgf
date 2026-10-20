//! Inherent `const` scalar methods of the concrete field families: one
//! separate timed case per method per family that declares the method.
//!
//! The inherent surface resolves only on concrete types, so every case is
//! spelled through a macro instantiation on a family alias rather than a
//! generic `F: Field`, which would bind the sealed [`FieldElem`] trait
//! instead. Fixture operands come from the deterministic LCG, verification
//! compares each inherent result against the trait arithmetic or an
//! independent identity before timing, and every timed body is one method
//! call over black-boxed operands. The residue row declares no inherent byte
//! conversion and the specialized primes declare no inherent canonical or
//! getter methods, so those families register the subsets their `const`
//! surface actually provides.

use std::hint::black_box;

use fgf::field::FieldElem;
use fgf::{Elem, Field};

use crate::fields::{
    Gf1e9p7, Gf2p1Poly, Gf2p2Poly, Gf2p4Poly, Gf2p8Aes, Gf2p16Rijndael, Gf2p32Rijndael,
    Gf2p64Rijndael, Gld, M31, Qm31,
};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// A nonzero operand: substitution keeps division and inversion total in the
/// verification identities without branching in the timed body.
fn nonzero<F: Field>(value: Elem<F>) -> Elem<F> {
    if <Elem<F> as FieldElem>::is_zero(value) {
        <Elem<F> as FieldElem>::ONE
    } else {
        value
    }
}

const API_FROM_RAW: &[&str] = &["fgf::Elem::from_raw (inherent const)"];
const API_FROM_BYTES: &[&str] = &["fgf::Elem::from_bytes (inherent const)"];
const API_TO_BYTES: &[&str] = &["fgf::Elem::to_bytes (inherent const)"];
const API_ADD: &[&str] = &["fgf::Elem::add (inherent const)"];
const API_SUB: &[&str] = &["fgf::Elem::sub (inherent const)"];
const API_NEG: &[&str] = &["fgf::Elem::neg (inherent const)"];
const API_MUL: &[&str] = &["fgf::Elem::mul (inherent const)"];
const API_SQUARE: &[&str] = &["fgf::Elem::square (inherent const)"];
const API_INV: &[&str] = &["fgf::Elem::inv (inherent const)"];
const API_DIV: &[&str] = &["fgf::Elem::div (inherent const)"];
const API_POW: &[&str] = &["fgf::Elem::pow (inherent const)"];
const API_CANONICAL: &[&str] = &["fgf::Elem::canonical (inherent const)"];
const API_IS_ZERO: &[&str] = &["fgf::Elem::is_zero (inherent const)"];
const API_IS_ONE: &[&str] = &["fgf::Elem::is_one (inherent const)"];

fn inherent_case(
    label: &str,
    method: &'static str,
    api: &'static [&'static str],
    scope: Scope,
    sources: usize,
    run: fn(&Case, &CaseCtx),
) -> Case {
    Case {
        case: format!("scalar/inherent/{method}/{label}"),
        api,
        field: label.to_string(),
        operation: method.to_string(),
        scope,
        bytes: 0,
        lanes: 1,
        sources,
        destinations: 1,
        control: false,
        phase: None,
        run,
    }
}

macro_rules! raw_case {
    ($label:literal, $F:ty) => {
        registry::add(inherent_case(
            $label,
            "from_raw",
            API_FROM_RAW,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                let word = a.to_raw();
                {
                    assert_eq!(
                        Elem::<F>::from_raw(word),
                        a,
                        "{}: raw-word oracle",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let rebuilt = Elem::<F>::from_raw(black_box(word));
                    fixture::observe(rebuilt);
                });
            },
        ));
    };
}

macro_rules! byte_cases {
    ($label:literal, $F:ty) => {
        registry::add(inherent_case(
            $label,
            "from_bytes",
            API_FROM_BYTES,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                let bytes = a.to_bytes();
                {
                    assert_eq!(
                        Elem::<F>::from_bytes(bytes),
                        a,
                        "{}: byte-word oracle",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let decoded = Elem::<F>::from_bytes(black_box(bytes));
                    fixture::observe(decoded);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "to_bytes",
            API_TO_BYTES,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert_eq!(
                        Elem::<F>::from_bytes(a.to_bytes()),
                        a,
                        "{}: byte-form oracle",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let bytes = Elem::<F>::to_bytes(black_box(a));
                    fixture::observe(bytes);
                });
            },
        ));
    };
}

macro_rules! arithmetic_cases {
    ($label:literal, $F:ty) => {
        registry::add(inherent_case(
            $label,
            "add",
            API_ADD,
            Scope::Steady,
            2,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let elems = fixture::elements::<F>(4, &mut seed);
                let a = elems[0];
                let b = elems[1];
                {
                    assert_eq!(
                        a.add(b),
                        <Elem<F> as FieldElem>::add(a, b),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.add(b).sub(b), a, "{}: addition identity", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let sum = Elem::<F>::add(black_box(a), black_box(b));
                    fixture::observe(sum);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "sub",
            API_SUB,
            Scope::Steady,
            2,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let elems = fixture::elements::<F>(4, &mut seed);
                let a = elems[0];
                let b = elems[1];
                {
                    assert_eq!(
                        a.sub(b),
                        <Elem<F> as FieldElem>::sub(a, b),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.sub(b).add(b), a, "{}: subtraction identity", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let difference = Elem::<F>::sub(black_box(a), black_box(b));
                    fixture::observe(difference);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "neg",
            API_NEG,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert_eq!(
                        a.neg(),
                        <Elem<F> as FieldElem>::neg(a),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(
                        a.neg().add(a),
                        Elem::<F>::ZERO,
                        "{}: negation identity",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let negated = Elem::<F>::neg(black_box(a));
                    fixture::observe(negated);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "mul",
            API_MUL,
            Scope::Steady,
            2,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let elems = fixture::elements::<F>(4, &mut seed);
                let a = elems[0];
                let b = nonzero(elems[1]);
                {
                    assert_eq!(
                        a.mul(b),
                        <Elem<F> as FieldElem>::mul(a, b),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.mul(b).mul(b.inv()), a, "{}: inverse identity", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let product = Elem::<F>::mul(black_box(a), black_box(b));
                    fixture::observe(product);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "square",
            API_SQUARE,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert_eq!(
                        a.square(),
                        <Elem<F> as FieldElem>::square(a),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.square(), a.mul(a), "{}: product identity", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let squared = Elem::<F>::square(black_box(a));
                    fixture::observe(squared);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "inv",
            API_INV,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = nonzero(fixture::elements::<F>(4, &mut seed).remove(0));
                {
                    assert_eq!(
                        a.inv(),
                        <Elem<F> as FieldElem>::inv(a),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(
                        a.inv().mul(a),
                        Elem::<F>::ONE,
                        "{}: inverse identity",
                        case.case
                    );
                    assert_eq!(
                        Elem::<F>::ZERO.inv(),
                        Elem::<F>::ZERO,
                        "{}: total inversion convention",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let inverted = Elem::<F>::inv(black_box(a));
                    fixture::observe(inverted);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "div",
            API_DIV,
            Scope::Steady,
            2,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let elems = fixture::elements::<F>(4, &mut seed);
                let a = elems[0];
                let b = nonzero(elems[1]);
                {
                    assert_eq!(
                        a.div(b),
                        <Elem<F> as FieldElem>::div(a, b),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.div(b), a.mul(b.inv()), "{}: quotient identity", case.case);
                    assert_eq!(
                        a.div(Elem::<F>::ZERO),
                        Elem::<F>::ZERO,
                        "{}: total division convention",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let quotient = Elem::<F>::div(black_box(a), black_box(b));
                    fixture::observe(quotient);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "pow",
            API_POW,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert_eq!(a.pow(3), a.square().mul(a), "{}: cube identity", case.case);
                    assert_eq!(
                        a.pow(7),
                        <Elem<F> as FieldElem>::pow(a, 7),
                        "{}: trait oracle",
                        case.case
                    );
                    assert_eq!(a.pow(0), Elem::<F>::ONE, "{}: zero exponent", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let power = Elem::<F>::pow(black_box(a), black_box(7u128));
                    fixture::observe(power);
                });
            },
        ));
    };
}

macro_rules! classification_cases {
    ($label:literal, $F:ty) => {
        registry::add(inherent_case(
            $label,
            "canonical",
            API_CANONICAL,
            Scope::Steady,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert_eq!(
                        a.canonical(),
                        <Elem<F> as FieldElem>::from_raw(a.to_raw()),
                        "{}: canonical oracle",
                        case.case
                    );
                    assert_eq!(a.canonical(), a, "{}: canonical identity", case.case);
                }
                registry::time(case, ctx, "scalar", || {
                    let normalized = Elem::<F>::canonical(black_box(a));
                    fixture::observe(normalized);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "is_zero",
            API_IS_ZERO,
            Scope::Query,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert!(
                        Elem::<F>::ZERO.is_zero() && !Elem::<F>::ZERO.is_one(),
                        "{}: zero-element queries",
                        case.case
                    );
                    assert!(!Elem::<F>::ONE.is_zero(), "{}: unit is not zero", case.case);
                    assert_eq!(
                        a.is_zero(),
                        a == <Elem<F> as FieldElem>::ZERO,
                        "{}: zero oracle",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let flag = Elem::<F>::is_zero(black_box(a));
                    fixture::observe(flag);
                });
            },
        ));
        registry::add(inherent_case(
            $label,
            "is_one",
            API_IS_ONE,
            Scope::Query,
            1,
            |case, ctx| {
                type F = $F;
                let mut seed = seed_for(&case.case);
                let a = fixture::elements::<F>(4, &mut seed).remove(0);
                {
                    assert!(
                        Elem::<F>::ONE.is_one() && !Elem::<F>::ONE.is_zero(),
                        "{}: unit-element queries",
                        case.case
                    );
                    assert!(!Elem::<F>::ZERO.is_one(), "{}: zero is not one", case.case);
                    assert_eq!(
                        a.is_one(),
                        a == <Elem<F> as FieldElem>::ONE,
                        "{}: unit oracle",
                        case.case
                    );
                }
                registry::time(case, ctx, "scalar", || {
                    let flag = Elem::<F>::is_one(black_box(a));
                    fixture::observe(flag);
                });
            },
        ));
    };
}

/// Register one case per inherent `const` method per concrete family, in the
/// family order of the former combined inherent panel.
pub fn register() {
    macro_rules! inherent {
        ($label:literal, $F:ty) => {
            raw_case!($label, $F);
            byte_cases!($label, $F);
            arithmetic_cases!($label, $F);
            classification_cases!($label, $F);
        };
    }
    inherent!("gf2p1-poly", Gf2p1Poly);
    inherent!("gf2p2-poly", Gf2p2Poly);
    inherent!("gf2p4-poly", Gf2p4Poly);
    inherent!("gf2p8-aes", Gf2p8Aes);
    inherent!("gf2p16-rijndael", Gf2p16Rijndael);
    inherent!("gf2p32-rijndael", Gf2p32Rijndael);
    inherent!("gf2p64-rijndael", Gf2p64Rijndael);
    // The residue row declares no inherent byte conversion.
    raw_case!("gf1e9p7", Gf1e9p7);
    arithmetic_cases!("gf1e9p7", Gf1e9p7);
    classification_cases!("gf1e9p7", Gf1e9p7);
    // The specialized primes declare no inherent canonical or getter methods.
    raw_case!("m31", M31);
    byte_cases!("m31", M31);
    arithmetic_cases!("m31", M31);
    raw_case!("goldilocks", Gld);
    byte_cases!("goldilocks", Gld);
    arithmetic_cases!("goldilocks", Gld);
    inherent!("qm31", Qm31);
}
