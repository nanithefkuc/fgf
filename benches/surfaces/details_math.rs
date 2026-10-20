//! Detail mathematics surfaces: the binary description predicates and
//! exponentiation, the degree-eight log/exp table build, the concrete-field
//! component, conjugation, norm, and residue constructors, the canonical
//! raw-word fold, and the Fan-Paar tower recurrence functions.
//!
//! Each case times exactly the callable its `api` list names over
//! prebuilt fixtures; the checks before the timed region use independent
//! identities (round trips through the paired conversion, the frozen
//! component layouts, conjugate-norm products, base-field schoolbook
//! arithmetic, and the description interpreter against the instance
//! arithmetic). Every callable here is small-field scalar work, so every
//! record reports the `scalar` backend.

use std::hint::black_box;

use fgf::field::binary::tower::{fp_invert, fp_mul_alpha, fp_multiply, fp_square};
use fgf::{BinaryField, ByteLogExp, Elem, Field};

use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

const API_IS_POLYNOMIAL_ROOT: &[&str] =
    &["fgf::field::binary::BinaryDescription::is_polynomial_root"];
const API_IS_BASIS_ROOT: &[&str] = &["fgf::field::binary::BinaryDescription::is_basis_root"];
const API_DESCRIPTION_POW: &[&str] = &["fgf::field::binary::BinaryDescription::pow"];
const API_LOG_EXP_BUILD: &[&str] = &["fgf::field::binary::ByteLogExp::build"];
const API_FROM_COMPONENTS: &[&str] = &["fgf::Elem::from_components (inherent const)"];
const API_TO_COMPONENTS: &[&str] = &["fgf::Elem::to_components (inherent const)"];
const API_CONJUGATE: &[&str] = &["fgf::Elem::conjugate (inherent const)"];
const API_NORM: &[&str] = &["fgf::Elem::norm (inherent const)"];
const API_MUL_ALPHA: &[&str] = &["fgf::Elem::mul_alpha (inherent const)"];
const API_FROM_RESIDUE: &[&str] = &["fgf::Elem::from_residue (inherent const)"];
const API_TO_RESIDUE: &[&str] = &["fgf::Elem::to_residue (inherent const)"];
const API_CANONICAL_RAW: &[&str] = &["fgf::field::Field::canonical_raw"];
const API_FP_MULTIPLY: &[&str] = &["fgf::field::binary::tower::fp_multiply"];
const API_FP_SQUARE: &[&str] = &["fgf::field::binary::tower::fp_square"];
const API_FP_MUL_ALPHA: &[&str] = &["fgf::field::binary::tower::fp_mul_alpha"];
const API_FP_INVERT: &[&str] = &["fgf::field::binary::tower::fp_invert"];

/// Repeated hot-loop work over prebuilt scalar fixtures.
fn steady_case(
    api: &'static [&'static str],
    label: &str,
    slug: &'static str,
    operation: &'static str,
    sources: usize,
    run: fn(&Case, &CaseCtx),
) -> Case {
    Case {
        case: format!("details-math/{slug}/{label}"),
        api,
        field: label.to_string(),
        operation: operation.to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: 1,
        sources,
        destinations: 1,
        control: false,
        phase: None,
        run,
    }
}

/// One public preparation call whose inputs are built outside the timer.
fn setup_case(
    api: &'static [&'static str],
    label: &str,
    slug: &'static str,
    operation: &'static str,
    run: fn(&Case, &CaseCtx),
) -> Case {
    Case {
        case: format!("details-math/{slug}/{label}"),
        api,
        field: label.to_string(),
        operation: operation.to_string(),
        scope: Scope::Setup,
        bytes: 0,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run,
    }
}

/// One trivial accessor read, timed at query scope with zero geometry.
fn query_case(
    api: &'static [&'static str],
    label: &str,
    slug: &'static str,
    operation: &'static str,
    run: fn(&Case, &CaseCtx),
) -> Case {
    Case {
        case: format!("details-math/{slug}/{label}"),
        api,
        field: label.to_string(),
        operation: operation.to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 0,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run,
    }
}

/// Register the details-math panel over the concrete shipped fields whose
/// presentations expose each callable.
pub fn register() {
    use crate::fields::{
        Gf1e9p7, Gf2p8Aes, Gf2p8FanPaar, Gf2p8Normal, Gf2p16FanPaar, Gf2p16Rijndael, Gf2p32FanPaar,
        Gf2p64FanPaar, Gld, M31, Qm31,
    };

    // --- binary description root predicates ---------------------------------

    registry::add(query_case(
        API_IS_POLYNOMIAL_ROOT,
        "gf2p8-aes",
        "description-is-polynomial-root",
        "description polynomial-root query",
        |case, ctx| {
            type F = Gf2p8Aes;
            let description = <F as BinaryField>::DESCRIPTION;
            assert!(
                description.is_polynomial_root(),
                "{}: AES row roots at a polynomial",
                case.case
            );
            assert!(
                !description.is_basis_root(),
                "{}: AES row has no basis root",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(description).is_polynomial_root());
            });
        },
    ));

    registry::add(query_case(
        API_IS_BASIS_ROOT,
        "gf2p8-normal",
        "description-is-basis-root",
        "description basis-root query",
        |case, ctx| {
            type F = Gf2p8Normal;
            let description = <F as BinaryField>::DESCRIPTION;
            assert!(
                description.is_basis_root(),
                "{}: normal row roots at an ordered basis",
                case.case
            );
            assert!(
                !description.is_polynomial_root(),
                "{}: normal row has no polynomial root",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(description).is_basis_root());
            });
        },
    ));

    // --- binary description exponentiation ----------------------------------

    registry::add(steady_case(
        API_DESCRIPTION_POW,
        "gf2p8-aes",
        "description-pow",
        "description pow",
        2,
        |case, ctx| {
            type F = Gf2p8Aes;
            let description = <F as BinaryField>::DESCRIPTION;
            let mut seed = seed_for(&case.case);
            let elem = fixture::elements::<F>(4, &mut seed).remove(0);
            let elem = if elem.is_zero() { Elem::<F>::ONE } else { elem };
            let a = u64::from(elem.to_raw());
            let one = description.one();
            assert_eq!(
                description.pow(a, 0),
                one,
                "{}: zeroth power is one",
                case.case
            );
            assert_eq!(
                description.pow(a, 3),
                description.mul(description.mul(a, a), a),
                "{}: cube against explicit products",
                case.case
            );
            assert_eq!(
                description.pow(a, 7),
                u64::from(elem.pow(7).to_raw()),
                "{}: element exponentiation oracle",
                case.case
            );
            let generator = u64::from(Elem::<F>::GENERATOR.to_raw());
            assert_eq!(
                description.pow(generator, 255),
                one,
                "{}: full-order generator closes the group",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(description.pow(black_box(a), 7));
            });
        },
    ));

    // --- degree-eight log/exp table build ------------------------------------

    registry::add(setup_case(
        API_LOG_EXP_BUILD,
        "gf2p8-aes",
        "log-exp-build",
        "log-exp build",
        |case, ctx| {
            type F = Gf2p8Aes;
            let description = <F as BinaryField>::DESCRIPTION;
            let generator = u64::from(Elem::<F>::GENERATOR.to_raw());
            assert_ne!(generator, 0, "{}: generator is nonzero", case.case);
            assert!(
                description.is_generator(generator),
                "{}: the element generator generates the description",
                case.case
            );
            assert_eq!(
                description.pow(generator, 255),
                description.one(),
                "{}: generator has full order",
                case.case
            );
            // The built table exposes no public readers, so the validation
            // proves the inputs against the description's own group law and
            // exercises the build path; the timed body measures that call.
            let table = ByteLogExp::build(description, generator);
            fixture::observe(table);
            registry::time(case, ctx, "scalar", || {
                let table = ByteLogExp::build(black_box(description), black_box(generator));
                fixture::observe(table);
            });
        },
    ));

    // --- component conversions over the shipped quadratic instances ----------

    registry::add(steady_case(
        API_FROM_COMPONENTS,
        "gf2p16-rijndael",
        "from-components",
        "from components",
        2,
        |case, ctx| {
            type F = Gf2p16Rijndael;
            type Base = Gf2p8Aes;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<Base>(2, &mut seed);
            let components = [elems[0], elems[1]];
            let value = Elem::<F>::from_components(components);
            assert_eq!(
                value.to_components(),
                components,
                "{}: component round trip",
                case.case
            );
            assert_eq!(
                u16::from(components[0].to_raw()) | (u16::from(components[1].to_raw()) << 8),
                value.to_raw(),
                "{}: low/high packing",
                case.case
            );
            assert!(
                Elem::<F>::from_components([Elem::<Base>::ONE, Elem::<Base>::ZERO]).is_one(),
                "{}: identity components",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                let value = Elem::<F>::from_components(black_box(components));
                fixture::observe(value.to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_FROM_COMPONENTS,
        "qm31",
        "from-components",
        "from components",
        2,
        |case, ctx| {
            type F = Qm31;
            type Base = M31;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<Base>(2, &mut seed);
            let components = [elems[0], elems[1]];
            let value = Elem::<F>::from_components(components);
            assert_eq!(
                value.to_components(),
                components,
                "{}: component round trip",
                case.case
            );
            assert_eq!(
                Elem::<F>::from_raw([components[0].to_raw(), components[1].to_raw()]),
                value,
                "{}: [re, im] limb layout",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                let value = Elem::<F>::from_components(black_box(components));
                fixture::observe(value.to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_TO_COMPONENTS,
        "gf2p16-rijndael",
        "to-components",
        "to components",
        1,
        |case, ctx| {
            type F = Gf2p16Rijndael;
            let mut seed = seed_for(&case.case);
            let value = fixture::elements::<F>(4, &mut seed).remove(0);
            let components = value.to_components();
            assert_eq!(
                Elem::<F>::from_components(components),
                value,
                "{}: round trip",
                case.case
            );
            assert_eq!(
                u16::from(components[0].to_raw()) | (u16::from(components[1].to_raw()) << 8),
                value.to_raw(),
                "{}: low/high packing",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).to_components());
            });
        },
    ));

    registry::add(steady_case(
        API_TO_COMPONENTS,
        "qm31",
        "to-components",
        "to components",
        1,
        |case, ctx| {
            type F = Qm31;
            let mut seed = seed_for(&case.case);
            let value = fixture::elements::<F>(4, &mut seed).remove(0);
            let components = value.to_components();
            assert_eq!(
                Elem::<F>::from_components(components),
                value,
                "{}: round trip",
                case.case
            );
            assert_eq!(
                Elem::<F>::from_raw([components[0].to_raw(), components[1].to_raw()]),
                value,
                "{}: [re, im] limb layout",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).to_components());
            });
        },
    ));

    // --- conjugation and norm over the quadratic Mersenne extension ----------

    registry::add(steady_case(
        API_CONJUGATE,
        "qm31",
        "conjugate",
        "conjugate",
        1,
        |case, ctx| {
            type F = Qm31;
            type Base = M31;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let value = elems[0];
            let [re, im] = value.to_components();
            let conjugated = value.conjugate();
            assert_eq!(conjugated.conjugate(), value, "{}: involution", case.case);
            assert_eq!(
                conjugated.to_components(),
                [re, im.neg()],
                "{}: imaginary part negates",
                case.case
            );
            assert_eq!(
                value.mul(conjugated),
                Elem::<F>::from_components([value.norm(), Elem::<Base>::ZERO]),
                "{}: conjugate product holds the norm",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).conjugate().to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_NORM,
        "qm31",
        "norm",
        "norm",
        2,
        |case, ctx| {
            type F = Qm31;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let value = elems[0];
            let other = elems[1];
            let [re, im] = value.to_components();
            assert_eq!(
                value.norm(),
                re.mul(re).add(im.mul(im)),
                "{}: norm is a² + b² over i² = −1",
                case.case
            );
            assert_eq!(
                value.mul(other).norm(),
                value.norm().mul(other.norm()),
                "{}: norm is multiplicative",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).norm().to_raw());
            });
        },
    ));

    // --- multiply by the tower indeterminate over the Fan-Paar rows ----------

    registry::add(steady_case(
        API_MUL_ALPHA,
        "gf2p8-fanpaar",
        "mul-alpha",
        "mul alpha",
        1,
        |case, ctx| {
            type F = Gf2p8FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            // The level indeterminate `X` is components `[0, 1]`: raw `0x10`.
            let alpha = Elem::<F>::from_raw(0x10);
            for value in &elems {
                assert_eq!(
                    value.mul_alpha(),
                    value.mul(alpha),
                    "{}: X·x over the Fan-Paar relation",
                    case.case
                );
            }
            assert!(
                Elem::<F>::ZERO.mul_alpha().is_zero(),
                "{}: zero absorbs",
                case.case
            );
            let value = elems[0];
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).mul_alpha().to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_MUL_ALPHA,
        "gf2p16-fanpaar",
        "mul-alpha",
        "mul alpha",
        1,
        |case, ctx| {
            type F = Gf2p16FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            // The level indeterminate `X` is components `[0, 1]`: raw `0x0100`.
            let alpha = Elem::<F>::from_raw(0x0100);
            for value in &elems {
                assert_eq!(
                    value.mul_alpha(),
                    value.mul(alpha),
                    "{}: X·x over the Fan-Paar relation",
                    case.case
                );
            }
            assert!(
                Elem::<F>::ZERO.mul_alpha().is_zero(),
                "{}: zero absorbs",
                case.case
            );
            let value = elems[0];
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).mul_alpha().to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_MUL_ALPHA,
        "gf2p32-fanpaar",
        "mul-alpha",
        "mul alpha",
        1,
        |case, ctx| {
            type F = Gf2p32FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            // The level indeterminate `X` is components `[0, 1]`:
            // raw `0x0001_0000`.
            let alpha = Elem::<F>::from_raw(0x0001_0000);
            for value in &elems {
                assert_eq!(
                    value.mul_alpha(),
                    value.mul(alpha),
                    "{}: X·x over the Fan-Paar relation",
                    case.case
                );
            }
            assert!(
                Elem::<F>::ZERO.mul_alpha().is_zero(),
                "{}: zero absorbs",
                case.case
            );
            let value = elems[0];
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).mul_alpha().to_raw());
            });
        },
    ));

    // --- inherent residue constructors over the prime rows -------------------

    registry::add(steady_case(
        API_FROM_RESIDUE,
        "m31",
        "from-residue",
        "from residue",
        1,
        |case, ctx| {
            type F = M31;
            const MODULUS: u64 = 0x7FFF_FFFF;
            let residue = u64::from(0x1234_5678u32);
            assert_eq!(
                Elem::<F>::from_residue(residue),
                Ok(Elem::<F>::from_raw(0x1234_5678)),
                "{}: residue below the modulus",
                case.case
            );
            assert!(
                Elem::<F>::from_residue(MODULUS).is_err(),
                "{}: the modulus is rejected",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                let value = Elem::<F>::from_residue(black_box(residue)).expect("valid residue");
                fixture::observe(value.to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_FROM_RESIDUE,
        "gf1e9p7",
        "from-residue",
        "from residue",
        1,
        |case, ctx| {
            type F = Gf1e9p7;
            const MODULUS: u64 = 1_000_000_007;
            let residue = 987_654_321u64;
            assert_eq!(
                Elem::<F>::from_residue(residue),
                Ok(Elem::<F>::from_raw(residue)),
                "{}: residue below the modulus",
                case.case
            );
            assert!(
                Elem::<F>::from_residue(MODULUS).is_err(),
                "{}: the modulus is rejected",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                let value = Elem::<F>::from_residue(black_box(residue)).expect("valid residue");
                fixture::observe(value.to_raw());
            });
        },
    ));

    registry::add(steady_case(
        API_TO_RESIDUE,
        "m31",
        "to-residue",
        "to residue",
        1,
        |case, ctx| {
            type F = M31;
            let value = Elem::<F>::from_raw(0x1234_5678);
            assert_eq!(
                value.to_residue(),
                Some(u64::from(0x1234_5678u32)),
                "{}: canonical limb reads back",
                case.case
            );
            assert_eq!(
                Elem::<F>::ZERO.to_residue(),
                Some(0),
                "{}: zero reads back",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).to_residue());
            });
        },
    ));

    registry::add(steady_case(
        API_TO_RESIDUE,
        "gf1e9p7",
        "to-residue",
        "to residue",
        1,
        |case, ctx| {
            type F = Gf1e9p7;
            let value = Elem::<F>::from_raw(987_654_321);
            assert_eq!(
                value.to_residue(),
                Some(987_654_321),
                "{}: canonical limb reads back",
                case.case
            );
            assert_eq!(
                Elem::<F>::ZERO.to_residue(),
                Some(0),
                "{}: zero reads back",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(black_box(value).to_residue());
            });
        },
    ));

    // --- canonical raw-word fold over three presentations --------------------

    registry::add(steady_case(
        API_CANONICAL_RAW,
        "gf2p8-aes",
        "canonical-raw",
        "canonical raw",
        1,
        |case, ctx| {
            type F = Gf2p8Aes;
            let raw = 0x53u8;
            assert_eq!(
                <F as Field>::canonical_raw(raw),
                Elem::<F>::from_raw(raw).to_raw(),
                "{}: matches the element word",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(<F as Field>::canonical_raw(black_box(raw)));
            });
        },
    ));

    registry::add(steady_case(
        API_CANONICAL_RAW,
        "m31",
        "canonical-raw",
        "canonical raw",
        1,
        |case, ctx| {
            type F = M31;
            const MODULUS: u64 = 0x7FFF_FFFF;
            let raw = 0x1234_5678u32;
            assert_eq!(
                <F as Field>::canonical_raw(raw),
                raw,
                "{}: canonical words hold",
                case.case
            );
            assert_eq!(
                u64::from(<F as Field>::canonical_raw(u32::MAX)),
                u64::from(u32::MAX) % MODULUS,
                "{}: reduction matches modular arithmetic",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(<F as Field>::canonical_raw(black_box(raw)));
            });
        },
    ));

    registry::add(steady_case(
        API_CANONICAL_RAW,
        "goldilocks",
        "canonical-raw",
        "canonical raw",
        1,
        |case, ctx| {
            type F = Gld;
            const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;
            let raw = 0x1234_5678_9ABC_DEF0u64;
            assert_eq!(
                <F as Field>::canonical_raw(raw),
                raw,
                "{}: canonical words hold",
                case.case
            );
            assert_eq!(
                <F as Field>::canonical_raw(MODULUS + 5),
                5,
                "{}: reduction matches modular arithmetic",
                case.case
            );
            registry::time(case, ctx, "scalar", || {
                fixture::observe(<F as Field>::canonical_raw(black_box(raw)));
            });
        },
    ));

    // --- Fan-Paar tower recurrence functions at the widest word --------------

    registry::add(steady_case(
        API_FP_MULTIPLY,
        "gf2p64-fanpaar",
        "fp-multiply",
        "fp multiply",
        2,
        |case, ctx| {
            type F = Gf2p64FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let x = elems[0].to_raw();
            let y = elems[1].to_raw();
            assert_eq!(
                fp_multiply(x, y, 64),
                Elem::<F>::from_raw(x).mul(Elem::<F>::from_raw(y)).to_raw(),
                "{}: recursion against the instance product",
                case.case
            );
            assert_eq!(fp_multiply(x, 1, 64), x, "{}: identity", case.case);
            registry::time(case, ctx, "scalar", || {
                fixture::observe(fp_multiply(black_box(x), black_box(y), 64));
            });
        },
    ));

    registry::add(steady_case(
        API_FP_SQUARE,
        "gf2p64-fanpaar",
        "fp-square",
        "fp square",
        1,
        |case, ctx| {
            type F = Gf2p64FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let x = elems[0].to_raw();
            assert_eq!(
                fp_square(x, 64),
                Elem::<F>::from_raw(x).mul(Elem::<F>::from_raw(x)).to_raw(),
                "{}: recursion against the instance square",
                case.case
            );
            assert_eq!(fp_square(0, 64), 0, "{}: zero squares to zero", case.case);
            registry::time(case, ctx, "scalar", || {
                fixture::observe(fp_square(black_box(x), 64));
            });
        },
    ));

    registry::add(steady_case(
        API_FP_MUL_ALPHA,
        "gf2p64-fanpaar",
        "fp-mul-alpha",
        "fp mul alpha",
        1,
        |case, ctx| {
            type F = Gf2p64FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let x = elems[0].to_raw();
            // The level indeterminate `X` is components `[0, 1]`: raw `1 << 32`.
            let alpha = 1u64 << 32;
            assert_eq!(
                fp_mul_alpha(x, 64),
                Elem::<F>::from_raw(x)
                    .mul(Elem::<F>::from_raw(alpha))
                    .to_raw(),
                "{}: X·x over the Fan-Paar relation",
                case.case
            );
            assert_eq!(fp_mul_alpha(0, 64), 0, "{}: zero absorbs", case.case);
            registry::time(case, ctx, "scalar", || {
                fixture::observe(fp_mul_alpha(black_box(x), 64));
            });
        },
    ));

    registry::add(steady_case(
        API_FP_INVERT,
        "gf2p64-fanpaar",
        "fp-invert",
        "fp invert",
        1,
        |case, ctx| {
            type F = Gf2p64FanPaar;
            let mut seed = seed_for(&case.case);
            let elems = fixture::elements::<F>(4, &mut seed);
            let value = elems[0].to_raw();
            let value = if value == 0 { 1u64 } else { value };
            assert_eq!(
                fp_invert(value, 64),
                Elem::<F>::from_raw(value).inv().to_raw(),
                "{}: recursion against the instance inverse",
                case.case
            );
            assert_eq!(
                fp_multiply(fp_invert(value, 64), value, 64),
                1,
                "{}: product with the inverse is one",
                case.case
            );
            assert_eq!(fp_invert(0, 64), 0, "{}: zero maps to zero", case.case);
            registry::time(case, ctx, "scalar", || {
                fixture::observe(fp_invert(black_box(value), 64));
            });
        },
    ));
}
