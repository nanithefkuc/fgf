//! Scalar arithmetic surfaces: the sealed [`FieldElem`] trait methods, the
//! operator spellings, per-family inherent `const` methods, and the trivial
//! element queries.
//!
//! Every registered field runs the generic trait panel, so one spelling of
//! each case covers all forty-five presentations; the eleven family-shape
//! cases exercise the inherent `const` method surface that only concrete
//! families provide. Fixture values come from presentation indices through
//! the LCG, verification uses independent identities (re-adding a subtracted
//! operand, multiplying by an inverse, `pow` against explicit products), and
//! steady bodies are one call per invocation with black-boxed operands.

use std::hint::black_box;

use fgf::Elem;
use fgf::field::{Field, FieldElem};

use crate::fields::{for_each_field, for_each_field_small};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};

/// Deterministic per-case seed from the stable case string.
pub fn seed_for(case: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in case.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// A nonzero operand: substitution keeps division and inversion total in the
/// verification identities without branching in the timed body.
fn nonzero<F: Field>(value: Elem<F>) -> Elem<F> {
    if value.is_zero() {
        <Elem<F> as FieldElem>::ONE
    } else {
        value
    }
}

macro_rules! scalar_op_cases {
    ($label:literal, $F:ty) => {
        $crate::scalar::register_op::<$F>($label);
    };
}

fn case_id(label: &str, op: &str) -> String {
    format!("scalar/{op}/{label}")
}

const API_ADD: &[&str] = &["fgf::field::FieldElem::add", "fgf::field::Field::add_raw"];
const API_SUB: &[&str] = &["fgf::field::FieldElem::sub", "fgf::field::Field::sub_raw"];
const API_NEG: &[&str] = &["fgf::field::FieldElem::neg", "fgf::field::Field::neg_raw"];
const API_MUL: &[&str] = &["fgf::field::FieldElem::mul", "fgf::field::Field::mul_raw"];
const API_SQUARE: &[&str] = &["fgf::field::FieldElem::square"];
const API_INV: &[&str] = &["fgf::field::FieldElem::inv", "fgf::field::Field::inv_raw"];
const API_DIV: &[&str] = &["fgf::field::FieldElem::div"];
const API_POW: &[&str] = &["fgf::field::FieldElem::pow"];
const API_OPERATORS: &[&str] = &[
    "core::ops::Add for fgf::Elem",
    "core::ops::Sub for fgf::Elem",
    "core::ops::Mul for fgf::Elem",
    "core::ops::Div for fgf::Elem",
    "core::ops::Neg for fgf::Elem",
    "core::ops::AddAssign for fgf::Elem",
    "core::ops::SubAssign for fgf::Elem",
    "core::ops::MulAssign for fgf::Elem",
    "core::ops::DivAssign for fgf::Elem",
    "core::iter::Sum for fgf::Elem",
    "core::iter::Product for fgf::Elem",
];
const API_QUERIES: &[&str] = &[
    "fgf::field::FieldElem::from_raw",
    "fgf::field::FieldElem::is_zero",
    "fgf::field::FieldElem::is_one",
    "fgf::Elem::to_raw",
];

fn base_case(op: &'static str, api: &'static [&'static str], label: &str) -> Case {
    Case {
        case: case_id(label, op),
        api,
        field: label.to_string(),
        operation: op.to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: 1,
        sources: if op == "operators" { 8 } else { 2 },
        destinations: 1,
        control: false,
        phase: None,
        run: |_, _| unreachable!("registered cases carry their own run fn"),
    }
}

/// One field's ten generic scalar cases, spelled once per operation so each
/// api list names exactly the callable under measurement.
pub fn register_op<F: Field>(label: &'static str) {
    macro_rules! timed_case {
        ($op:literal, $api:expr, $sources:expr, $run:expr) => {
            registry::add(Case {
                run: $run,
                sources: $sources,
                ..base_case($op, $api, label)
            });
        };
    }

    timed_case!("add", API_ADD, 2, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        let b = fixture::elements::<F>(4, &mut seed).remove(1);
        {
            assert_eq!(a.add(b).sub(b), a, "{}: addition oracle", case.case);
            assert_eq!(
                a.add(a.neg()),
                <Elem<F> as FieldElem>::ZERO,
                "{}: negation oracle",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let sum = <Elem<F> as FieldElem>::add(black_box(a), black_box(b));
            fixture::observe(sum.to_raw());
        });
    });

    timed_case!("sub", API_SUB, 2, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        let b = fixture::elements::<F>(4, &mut seed).remove(1);
        {
            assert_eq!(a.sub(b).add(b), a, "{}: subtraction oracle", case.case);
        }
        registry::time(case, ctx, "scalar", || {
            let diff = <Elem<F> as FieldElem>::sub(black_box(a), black_box(b));
            fixture::observe(diff.to_raw());
        });
    });

    timed_case!("neg", API_NEG, 1, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        {
            assert_eq!(
                a.neg().add(a),
                <Elem<F> as FieldElem>::ZERO,
                "{}: negation oracle",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let negated = <Elem<F> as FieldElem>::neg(black_box(a));
            fixture::observe(negated.to_raw());
        });
    });

    timed_case!("mul", API_MUL, 2, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        let b = fixture::elements::<F>(4, &mut seed).remove(1);
        {
            let divisor = nonzero::<F>(b);
            assert_eq!(
                a.mul(divisor).mul(divisor.inv()),
                a,
                "{}: multiplication oracle",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let product = <Elem<F> as FieldElem>::mul(black_box(a), black_box(b));
            fixture::observe(product.to_raw());
        });
    });

    timed_case!("square", API_SQUARE, 1, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        {
            assert_eq!(a.square(), a.mul(a), "{}: square oracle", case.case);
        }
        registry::time(case, ctx, "scalar", || {
            let squared = <Elem<F> as FieldElem>::square(black_box(a));
            fixture::observe(squared.to_raw());
        });
    });

    timed_case!("inv", API_INV, 1, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = nonzero::<F>(fixture::elements::<F>(4, &mut seed).remove(0));
        {
            assert_eq!(
                a.inv().mul(a),
                <Elem<F> as FieldElem>::ONE,
                "{}: inversion oracle",
                case.case
            );
            assert_eq!(
                <Elem<F> as FieldElem>::ZERO.inv(),
                <Elem<F> as FieldElem>::ZERO,
                "{}: total inversion convention",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let inverted = <Elem<F> as FieldElem>::inv(black_box(a));
            fixture::observe(inverted.to_raw());
        });
    });

    timed_case!("div", API_DIV, 2, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        let b = fixture::elements::<F>(4, &mut seed).remove(1);
        {
            assert_eq!(a.div(b), a.mul(b.inv()), "{}: division oracle", case.case);
            assert_eq!(
                a.div(<Elem<F> as FieldElem>::ZERO),
                <Elem<F> as FieldElem>::ZERO,
                "{}: total division convention",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let quotient = <Elem<F> as FieldElem>::div(black_box(a), black_box(b));
            fixture::observe(quotient.to_raw());
        });
    });

    timed_case!("pow", API_POW, 1, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        {
            let squared = a.square();
            let expected = squared.square().mul(squared).mul(a);
            assert_eq!(a.pow(7), expected, "{}: pow oracle", case.case);
            assert_eq!(
                a.pow(0),
                <Elem<F> as FieldElem>::ONE,
                "{}: pow zero exponent",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let power = <Elem<F> as FieldElem>::pow(black_box(a), black_box(7u128));
            fixture::observe(power.to_raw());
        });
    });

    timed_case!(
        "operators",
        API_OPERATORS,
        8,
        |case: &Case, ctx: &CaseCtx| {
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(8, &mut seed);
            let a = elems[0];
            let b = nonzero::<F>(elems[1]);
            {
                let mut acc = a + b;
                acc -= b;
                acc *= b;
                acc /= b;
                acc = -acc;
                acc += b;
                assert_eq!(acc, b - a, "{}: operator chain oracle", case.case);
                let sum: Elem<F> = elems.iter().copied().sum();
                let mut manual = <Elem<F> as FieldElem>::ZERO;
                for value in &elems {
                    manual = <Elem<F> as FieldElem>::add(manual, *value);
                }
                assert_eq!(sum, manual, "{}: sum oracle", case.case);
                let mut manual_product = <Elem<F> as FieldElem>::ONE;
                for value in &elems {
                    manual_product = <Elem<F> as FieldElem>::mul(manual_product, *value);
                }
                let product: Elem<F> = elems.iter().copied().product();
                assert_eq!(product, manual_product, "{}: product oracle", case.case);
            }
            registry::time(case, ctx, "scalar", || {
                let b = black_box(b);
                let mut acc = black_box(a) + b;
                acc -= b;
                acc *= b;
                acc /= b;
                acc = -acc;
                acc += b;
                let sum: Elem<F> = black_box(&elems).iter().copied().sum();
                let product: Elem<F> = black_box(&elems).iter().copied().product();
                fixture::observe(acc.to_raw());
                fixture::observe(sum.to_raw());
                fixture::observe(product.to_raw());
            });
        }
    );

    timed_case!("queries", API_QUERIES, 1, |case: &Case, ctx: &CaseCtx| {
        let mut seed = seed_for(&case.case);
        let a = fixture::elements::<F>(4, &mut seed).remove(0);
        {
            assert_eq!(
                <Elem<F> as FieldElem>::from_raw(a.to_raw()),
                a,
                "{}: raw round trip",
                case.case
            );
            assert!(
                <Elem<F> as FieldElem>::ONE.is_one() && !<Elem<F> as FieldElem>::ONE.is_zero(),
                "{}: identity queries",
                case.case
            );
        }
        registry::time(case, ctx, "scalar", || {
            let rebuilt = <Elem<F> as FieldElem>::from_raw(black_box(a.to_raw()));
            fixture::observe((rebuilt.is_zero(), rebuilt.is_one(), rebuilt.to_raw()));
        });
    });
}

/// Register the generic scalar panel over every registered field.
pub fn register() {
    macro_rules! walk {
        ($label:literal, $F:ty) => {
            scalar_op_cases!($label, $F);
        };
    }
    for_each_field!(walk);
    macro_rules! walk_small {
        ($label:literal, $F:ty) => {
            scalar_op_cases!($label, $F);
        };
    }
    for_each_field_small!(walk_small);
}
