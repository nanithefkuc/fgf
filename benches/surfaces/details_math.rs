//! Detail mathematics surfaces: the concrete-field component, conjugation,
//! norm, and residue constructors, and the canonical raw-word fold.
//!
//! Each case times exactly the callable its `api` list names over
//! prebuilt fixtures; the checks before the timed region use independent
//! identities (round trips through the paired conversion, the frozen
//! component layouts, conjugate-norm products, and base-field schoolbook
//! arithmetic). Every callable here is small-field scalar work, so every
//! record reports the `scalar` backend.

use std::hint::black_box;

use fgf::Elem;

use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

const API_FROM_COMPONENTS: &[&str] = &["fgf::Elem::from_components (inherent const)"];
const API_TO_COMPONENTS: &[&str] = &["fgf::Elem::to_components (inherent const)"];
const API_CONJUGATE: &[&str] = &["fgf::Elem::conjugate (inherent const)"];
const API_NORM: &[&str] = &["fgf::Elem::norm (inherent const)"];
const API_MUL_ALPHA: &[&str] = &["fgf::Elem::mul_alpha (inherent const)"];
const API_FROM_RESIDUE: &[&str] = &["fgf::Elem::from_residue (inherent const)"];
const API_TO_RESIDUE: &[&str] = &["fgf::Elem::to_residue (inherent const)"];
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

/// Register the details-math panel over the concrete shipped fields whose
/// presentations expose each callable.
pub fn register() {
    use crate::fields::{
        Gf1e9p7, Gf2p8Aes, Gf2p8FanPaar, Gf2p16FanPaar, Gf2p16Rijndael, Gf2p32FanPaar, M31, Qm31,
    };

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
}
