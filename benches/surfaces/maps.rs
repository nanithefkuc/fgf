//! The embedding and isomorphism surface: canonical construction, image
//! round trips, application, restriction with in-image and out-of-image
//! results, membership, relative Frobenius, trace, and norm, composition,
//! inversion, generator queries, and the failure modes.
//!
//! Every map runs on scalar arithmetic, so records report the `scalar`
//! backend. Construction-heavy cases (build, from_images, then) are
//! end-to-end records with the calibration cap bounding their samples;
//! per-value methods are steady records over fixed fixture elements; the
//! pure queries are query records. Oracles are the algebra itself:
//! homomorphism on a sum, `restrict ∘ embed` round trips, membership versus
//! restriction agreement, linearity of trace, and `norm(embed(x)) == x^r`.
//! The failure cases name `IncompatibleDegree`, `EqualDegree`, and
//! `InvalidImages`; no registered pair falls outside the structural and
//! ladder map sources.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::{Elem, Embedding, EmbeddingError, Isomorphism};

use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// The alternate GF(7²) presentation uses `t² + 1`; `-1` is nonsquare in GF(7).
struct Alternate49;

impl fgf::ExtensionSpec<2> for Alternate49 {
    type Base = fgf::FiniteField<7, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(7^2)/alternate";
    const RELATION: fgf::Relation = fgf::Relation::from_coordinates(&[1, 0]);
}

const API_BUILD: &[&str] = &["fgf::Embedding::new"];
const API_FROM_IMAGES: &[&str] = &["fgf::Embedding::from_images", "fgf::Embedding::to_images"];
const API_APPLY: &[&str] = &["fgf::Embedding::embed"];
const API_ISO_APPLY: &[&str] = &[
    "fgf::Isomorphism::new",
    "fgf::Isomorphism::apply",
    "fgf::Isomorphism::unapply",
];
const API_ISO_INVERSE: &[&str] = &["fgf::Isomorphism::inverse"];
const API_RESTRICT: &[&str] = &["fgf::Embedding::restrict", "fgf::Embedding::contains"];
const API_CONTAINS: &[&str] = &["fgf::Embedding::contains"];
const API_FROBENIUS: &[&str] = &["fgf::Embedding::frobenius"];
const API_TRACE: &[&str] = &["fgf::Embedding::trace"];
const API_NORM: &[&str] = &["fgf::Embedding::norm"];
const API_THEN: &[&str] = &["fgf::Embedding::then"];
const API_THEN_EMBEDDING: &[&str] = &[
    "fgf::Isomorphism::then_embedding",
    "fgf::Isomorphism::new",
    "fgf::Embedding::new",
];
const API_GENERATORS: &[&str] = &[
    "fgf::Embedding::generator_count",
    "fgf::Embedding::generator",
];
const API_RELATIVE_DEGREE: &[&str] = &["fgf::Embedding::relative_degree"];
const API_FAIL_DEGREE: &[&str] = &[
    "fgf::Embedding::new",
    "fgf::EmbeddingError::IncompatibleDegree",
];
const API_FAIL_EQUAL_DEGREE: &[&str] = &["fgf::Embedding::new", "fgf::EmbeddingError::EqualDegree"];
const API_FAIL_IMAGES: &[&str] = &[
    "fgf::Embedding::from_images",
    "fgf::EmbeddingError::InvalidImages",
];

/// The degree-one rows that only appear as map sources.
pub type Gf3Row = fgf::FiniteField<3, 1>;
pub type Gf5Row = fgf::FiniteField<5, 1>;
pub type Gf7Row = fgf::FiniteField<7, 1>;

/// The scalar backend name every map record reports.
fn map_backend() -> String {
    crate::fields::scalar_backend_name()
}

#[allow(clippy::too_many_arguments)] // Fields mirror the measurement record.
fn pair_case(
    op: &'static str,
    scope: Scope,
    s_label: &'static str,
    t_label: &'static str,
    api: &'static [&'static str],
    sources: usize,
    destinations: usize,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case: format!("maps/{op}/{s_label}->{t_label}"),
        api,
        field: format!("{s_label}->{t_label}"),
        operation: op.to_string(),
        scope,
        bytes: 0,
        lanes: 1,
        sources,
        destinations,
        control: false,
        phase: None,
        run,
    });
}

/// A deterministic element of `S` from the case seed.
fn sample<S: Field>(case: &str) -> Elem<S> {
    let mut seed = seed_for(case);
    let index = u128::from(fixture::lcg(&mut seed)) % S::ORDER.max(1);
    <Elem<S> as FieldElem>::from_index(index).expect("index below ORDER")
}

fn run_embed_apply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let x = sample::<S>(&case.case);
    {
        let y = fixture::elements::<S>(2, &mut seed_for(&case.case));
        let sum = map.embed(x.add(y[0]));
        assert_eq!(
            sum,
            map.embed(x).add(map.embed(y[0])),
            "{}: additivity",
            case.case
        );
        assert_eq!(
            map.embed(<Elem<S> as FieldElem>::ONE),
            <Elem<T> as FieldElem>::ONE,
            "{}: one maps to one",
            case.case
        );
        let _ = y;
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.embed(black_box(x)).to_raw());
    });
}

fn run_build<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    {
        let map = Embedding::<S, T>::new().expect("canonical map exists");
        let x = sample::<S>(&case.case);
        let y = map.embed(x);
        assert_eq!(map.restrict(y), Some(x), "{}: build round trip", case.case);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let map = Embedding::<S, T>::new().expect("canonical map exists");
        fixture::observe(map.relative_degree());
    });
}

fn run_from_images<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    {
        let map = Embedding::<S, T>::new().expect("canonical map exists");
        let rebuilt =
            Embedding::<S, T>::from_images(map.to_images()).expect("canonical images are valid");
        let x = sample::<S>(&case.case);
        assert_eq!(
            rebuilt.embed(x),
            map.embed(x),
            "{}: image round trip",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let map = Embedding::<S, T>::new().expect("canonical map exists");
        let rebuilt =
            Embedding::<S, T>::from_images(map.to_images()).expect("canonical images are valid");
        fixture::observe(rebuilt.relative_degree());
    });
}

fn run_restrict<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let x = sample::<S>(&case.case);
    let y_hit = map.embed(x);
    // Membership and restriction must agree over a sample spread.
    for probe in 0..8u64 {
        let candidate =
            fixture::elements::<T>(1, &mut seed_for(&format!("{}/{probe}", case.case))).remove(0);
        assert_eq!(
            map.contains(candidate),
            map.restrict(candidate).is_some(),
            "{}: contains agrees with restrict",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.restrict(black_box(y_hit)).map(|v| v.to_raw()));
    });
}

fn run_contains<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let y_hit = map.embed(sample::<S>(&case.case));
    assert!(
        map.contains(y_hit),
        "{}: image element is contained",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.contains(black_box(y_hit)));
    });
}

fn run_frobenius<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let x = sample::<S>(&case.case);
    let y = map.embed(x);
    assert_eq!(
        map.frobenius(y),
        y.pow(S::ORDER),
        "{}: relative Frobenius oracle",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.frobenius(black_box(y)).to_raw());
    });
}

fn run_trace<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let x = sample::<S>(&case.case);
    let relative = map.relative_degree();
    // A subfield element repeats under every relative conjugate, so its
    // trace is the relative degree times itself, spelled as repeated
    // addition to stay inside the source field at every characteristic.
    let mut multiple = <Elem<S> as FieldElem>::ZERO;
    for _ in 0..relative {
        multiple = multiple.add(<Elem<S> as FieldElem>::ONE);
    }
    assert_eq!(
        map.trace(map.embed(x)),
        x.mul(multiple),
        "{}: trace of a subfield element",
        case.case
    );
    let y = map.embed(x);
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.trace(black_box(y)).to_raw());
    });
}

fn run_norm<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    let x = sample::<S>(&case.case);
    let relative = map.relative_degree();
    assert_eq!(
        map.norm(map.embed(x)),
        x.pow(u128::from(relative)),
        "{}: norm of a subfield element",
        case.case
    );
    let y = map.embed(x);
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.norm(black_box(y)).to_raw());
    });
}

fn run_relative_degree<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let map = Embedding::<S, T>::new().expect("canonical map exists");
    assert_eq!(
        map.relative_degree(),
        T::DEGREE / S::DEGREE,
        "{}",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(map.relative_degree());
    });
}

fn run_iso_apply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let iso = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let x = sample::<S>(&case.case);
    {
        assert_eq!(
            iso.unapply(iso.apply(x)),
            x,
            "{}: iso round trip",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let y = iso.apply(black_box(x));
        fixture::observe((y.to_raw(), iso.unapply(y).to_raw()));
    });
}

fn run_iso_inverse<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let iso = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let x = sample::<S>(&case.case);
    {
        let back = iso.inverse();
        assert_eq!(back.apply(iso.apply(x)), x, "{}: round trip", case.case);
        assert_eq!(
            iso.inverse().inverse().apply(x),
            iso.apply(x),
            "{}: double inversion returns the same map",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let back = iso.inverse();
        let y = iso.apply(black_box(x));
        fixture::observe(back.apply(y).to_raw());
    });
}

/// Composition of two inclusions through an intermediate presentation.
fn run_then<A, B, C>(case: &Case, ctx: &CaseCtx)
where
    A: Field,
    B: Field<Characteristic = A::Characteristic>,
    C: Field<Characteristic = A::Characteristic>,
{
    let first = Embedding::<A, B>::new().expect("canonical map exists");
    let second = Embedding::<B, C>::new().expect("canonical map exists");
    let composed = first.then(&second);
    let x = sample::<A>(&case.case);
    assert_eq!(
        composed.embed(x),
        second.embed(first.embed(x)),
        "{}: composition oracle",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let composed = first.then(&second);
        fixture::observe(composed.embed(black_box(x)).to_raw());
    });
}

/// Composition of an isomorphism with a following proper inclusion.
fn run_then_embedding<A, B, C>(case: &Case, ctx: &CaseCtx)
where
    A: Field,
    B: Field<Characteristic = A::Characteristic>,
    C: Field<Characteristic = A::Characteristic>,
{
    let first = Isomorphism::<A, B>::new().expect("canonical isomorphism exists");
    let second = Embedding::<B, C>::new().expect("canonical map exists");
    let composed = first.then_embedding(&second);
    let x = sample::<A>(&case.case);
    assert_eq!(
        composed.embed(x),
        second.embed(first.apply(x)),
        "{}: composition oracle",
        case.case
    );
    assert_eq!(
        composed.restrict(composed.embed(x)),
        Some(x),
        "{}: composition round trip",
        case.case
    );
    assert_eq!(
        composed.relative_degree(),
        C::DEGREE / A::DEGREE,
        "{}: relative degree spans the chain",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let composed = first.then_embedding(&second);
        fixture::observe(composed.embed(black_box(x)).to_raw());
    });
}

/// Generator-level queries on one source field.
fn run_generators<S: Field>(case: &Case, ctx: &CaseCtx) {
    let count = Embedding::<S, S>::generator_count();
    let mut nonzero = 0usize;
    for level in 0..count {
        let g = Embedding::<S, S>::generator(level);
        assert!(!g.is_zero(), "{}: generator {level} is nonzero", case.case);
        nonzero += 1;
    }
    assert_eq!(nonzero, count, "{}", case.case);
    let identity = Isomorphism::<S, S>::new().expect("identity map exists");
    let probe = sample::<S>(&format!("{}/identity", case.case));
    assert_eq!(
        identity.apply(probe),
        probe,
        "{}: identity forward",
        case.case
    );
    assert_eq!(
        identity.unapply(probe),
        probe,
        "{}: identity reverse",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let mut observed = 0usize;
        for level in 0..count {
            if !Embedding::<S, S>::generator(level).is_zero() {
                observed += 1;
            }
        }
        fixture::observe((count, observed));
    });
}

fn run_fail_degree<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    match Embedding::<S, T>::new() {
        Err(EmbeddingError::IncompatibleDegree { source, target }) => {
            assert_eq!(source, S::DEGREE);
            assert_eq!(target, T::DEGREE);
        }
        other => panic!(
            "{}: expected IncompatibleDegree, got {:?}",
            case.case,
            other.map(|_| ())
        ),
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let failure = Embedding::<S, T>::new().err();
        fixture::observe(failure.is_some());
    });
}

fn run_fail_equal_degree<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    match Embedding::<S, T>::new() {
        Err(EmbeddingError::EqualDegree { degree }) => {
            assert_eq!(degree, S::DEGREE);
            assert_eq!(degree, T::DEGREE);
        }
        other => panic!(
            "{}: expected EqualDegree, got {:?}",
            case.case,
            other.map(|_| ())
        ),
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let failure = Embedding::<S, T>::new().err();
        fixture::observe(failure.is_some());
    });
}

fn run_fail_images<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    match Embedding::<S, T>::from_images(&[]) {
        Err(EmbeddingError::InvalidImages) => {}
        other => panic!(
            "{}: expected InvalidImages, got {:?}",
            case.case,
            other.map(|_| ())
        ),
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let failure = Embedding::<S, T>::from_images(&[]).err();
        fixture::observe(failure.is_some());
    });
}

/// Every registered map pair: structural inclusions across all shipped and
/// custom towers, the odd-characteristic extensions, the equal-degree
/// presentation changes, and the failure pairs.
pub fn register() {
    macro_rules! inclusion {
        ($s:literal, $S:ty, $t:literal, $T:ty) => {
            pair_case(
                "build",
                Scope::EndToEnd,
                $s,
                $t,
                API_BUILD,
                1,
                1,
                run_build::<$S, $T>,
            );
            pair_case(
                "from-images",
                Scope::EndToEnd,
                $s,
                $t,
                API_FROM_IMAGES,
                1,
                1,
                run_from_images::<$S, $T>,
            );
            pair_case(
                "apply",
                Scope::Steady,
                $s,
                $t,
                API_APPLY,
                1,
                1,
                run_embed_apply::<$S, $T>,
            );
            pair_case(
                "restrict",
                Scope::Steady,
                $s,
                $t,
                API_RESTRICT,
                1,
                1,
                run_restrict::<$S, $T>,
            );
            pair_case(
                "contains",
                Scope::Steady,
                $s,
                $t,
                API_CONTAINS,
                1,
                1,
                run_contains::<$S, $T>,
            );
            pair_case(
                "frobenius",
                Scope::Steady,
                $s,
                $t,
                API_FROBENIUS,
                1,
                1,
                run_frobenius::<$S, $T>,
            );
            pair_case(
                "trace",
                Scope::Steady,
                $s,
                $t,
                API_TRACE,
                1,
                1,
                run_trace::<$S, $T>,
            );
            pair_case(
                "norm",
                Scope::Steady,
                $s,
                $t,
                API_NORM,
                1,
                1,
                run_norm::<$S, $T>,
            );
            pair_case(
                "relative-degree",
                Scope::Query,
                $s,
                $t,
                API_RELATIVE_DEGREE,
                0,
                0,
                run_relative_degree::<$S, $T>,
            );
        };
    }
    macro_rules! iso_pair {
        ($s:literal, $S:ty, $t:literal, $T:ty) => {
            pair_case(
                "apply-unapply",
                Scope::Steady,
                $s,
                $t,
                API_ISO_APPLY,
                1,
                1,
                run_iso_apply::<$S, $T>,
            );
            pair_case(
                "iso-inverse",
                Scope::Query,
                $s,
                $t,
                API_ISO_INVERSE,
                1,
                1,
                run_iso_inverse::<$S, $T>,
            );
        };
    }

    inclusion!(
        "gf2p1-poly",
        crate::fields::Gf2p1Poly,
        "gf2p2-poly",
        crate::fields::Gf2p2Poly
    );
    inclusion!(
        "gf2p2-poly",
        crate::fields::Gf2p2Poly,
        "gf2p4-poly",
        crate::fields::Gf2p4Poly
    );
    inclusion!(
        "gf2p4-poly",
        crate::fields::Gf2p4Poly,
        "gf2p8-aes",
        crate::fields::Gf2p8Aes
    );
    inclusion!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p16-rijndael",
        crate::fields::Gf2p16Rijndael
    );
    inclusion!(
        "gf2p16-rijndael",
        crate::fields::Gf2p16Rijndael,
        "gf2p32-rijndael",
        crate::fields::Gf2p32Rijndael
    );
    inclusion!(
        "gf2p32-rijndael",
        crate::fields::Gf2p32Rijndael,
        "gf2p64-rijndael",
        crate::fields::Gf2p64Rijndael
    );
    inclusion!(
        "gf2p2-fanpaar",
        crate::fields::Gf2p2FanPaar,
        "gf2p4-fanpaar",
        crate::fields::Gf2p4FanPaar
    );
    inclusion!(
        "gf2p4-fanpaar",
        crate::fields::Gf2p4FanPaar,
        "gf2p8-fanpaar",
        crate::fields::Gf2p8FanPaar
    );
    inclusion!(
        "gf2p8-fanpaar",
        crate::fields::Gf2p8FanPaar,
        "gf2p16-fanpaar",
        crate::fields::Gf2p16FanPaar
    );
    inclusion!(
        "gf2p16-fanpaar",
        crate::fields::Gf2p16FanPaar,
        "gf2p32-fanpaar",
        crate::fields::Gf2p32FanPaar
    );
    inclusion!(
        "gf2p32-fanpaar",
        crate::fields::Gf2p32FanPaar,
        "gf2p64-fanpaar",
        crate::fields::Gf2p64FanPaar
    );
    inclusion!(
        "gf2p8-rs",
        crate::fields::Gf2p8Rs,
        "gf2p16-custom-rs",
        crate::fields::Gf2p16CustomRs
    );
    inclusion!(
        "gf2p8-normal",
        crate::fields::Gf2p8Normal,
        "gf2p16-custom-normal",
        crate::fields::Gf2p16CustomNormal
    );
    inclusion!(
        "gf2p8-cantor",
        crate::fields::Gf2p8Cantor,
        "gf2p16-custom-cantor",
        crate::fields::Gf2p16CustomCantor
    );
    inclusion!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p16-custom-rijndael",
        crate::fields::Gf2p16CustomRijndael
    );
    inclusion!(
        "gf2p1-poly",
        crate::fields::Gf2p1Poly,
        "gf2p5-padded",
        crate::fields::Gf2p5Padded
    );
    inclusion!("gf7", Gf7Row, "gf49", crate::fields::Gf49);
    inclusion!("gf5", Gf5Row, "gf125", crate::fields::Gf125);
    inclusion!("gf3", Gf3Row, "gf9", crate::fields::Gf9);
    inclusion!("gf9", crate::fields::Gf9, "gf81", crate::fields::Gf81);
    inclusion!(
        "goldilocks",
        crate::fields::Gld,
        "goldilocks-squared",
        crate::fields::GoldilocksSquared
    );
    inclusion!(
        "gfmaxp",
        crate::fields::GfMaxPrime,
        "maxprime-squared",
        crate::fields::MaxPrimeSquared
    );
    inclusion!("m31", crate::fields::M31, "qm31", crate::fields::Qm31);

    iso_pair!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p8-rs",
        crate::fields::Gf2p8Rs
    );
    iso_pair!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p8-normal",
        crate::fields::Gf2p8Normal
    );
    iso_pair!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p8-cantor",
        crate::fields::Gf2p8Cantor
    );
    iso_pair!(
        "gf2p2-poly",
        crate::fields::Gf2p2Poly,
        "gf2p2-normal",
        crate::fields::Gf2p2Normal
    );
    iso_pair!(
        "gf2p2-poly",
        crate::fields::Gf2p2Poly,
        "gf2p2-cantor",
        crate::fields::Gf2p2Cantor
    );
    iso_pair!(
        "gf2p4-poly",
        crate::fields::Gf2p4Poly,
        "gf2p4-normal",
        crate::fields::Gf2p4Normal
    );
    iso_pair!(
        "gf2p4-poly",
        crate::fields::Gf2p4Poly,
        "gf2p4-cantor",
        crate::fields::Gf2p4Cantor
    );
    iso_pair!(
        "gf2p16-rijndael",
        crate::fields::Gf2p16Rijndael,
        "gf2p16-fanpaar",
        crate::fields::Gf2p16FanPaar
    );
    iso_pair!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p8-aes",
        crate::fields::Gf2p8Aes
    );

    macro_rules! then_chain {
        ($a:literal, $A:ty, $b:literal, $B:ty, $c:literal, $C:ty) => {
            registry::add(Case {
                case: format!("maps/then/{}->{}->{}", $a, $b, $c),
                api: API_THEN,
                field: format!("{}->{}->{}", $a, $b, $c),
                operation: "then".to_string(),
                scope: Scope::EndToEnd,
                bytes: 0,
                lanes: 1,
                sources: 1,
                destinations: 1,
                control: false,
                phase: None,
                run: run_then::<$A, $B, $C>,
            });
        };
    }
    then_chain!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p16-rijndael",
        crate::fields::Gf2p16Rijndael,
        "gf2p32-rijndael",
        crate::fields::Gf2p32Rijndael
    );
    then_chain!(
        "gf2p4-fanpaar",
        crate::fields::Gf2p4FanPaar,
        "gf2p8-fanpaar",
        crate::fields::Gf2p8FanPaar,
        "gf2p16-fanpaar",
        crate::fields::Gf2p16FanPaar
    );
    then_chain!(
        "gf3",
        Gf3Row,
        "gf9",
        crate::fields::Gf9,
        "gf81",
        crate::fields::Gf81
    );
    then_chain!(
        "gf2p2-poly",
        crate::fields::Gf2p2Poly,
        "gf2p4-poly",
        crate::fields::Gf2p4Poly,
        "gf2p8-aes",
        crate::fields::Gf2p8Aes
    );

    macro_rules! then_iso_chain {
        ($a:literal, $A:ty, $b:literal, $B:ty, $c:literal, $C:ty) => {
            registry::add(Case {
                case: format!("maps/then-embedding/{}->{}->{}", $a, $b, $c),
                api: API_THEN_EMBEDDING,
                field: format!("{}->{}->{}", $a, $b, $c),
                operation: "then-embedding".to_string(),
                scope: Scope::EndToEnd,
                bytes: 0,
                lanes: 1,
                sources: 1,
                destinations: 1,
                control: false,
                phase: None,
                run: run_then_embedding::<$A, $B, $C>,
            });
        };
    }
    then_iso_chain!(
        "gf2p8-aes",
        crate::fields::Gf2p8Aes,
        "gf2p8-rs",
        crate::fields::Gf2p8Rs,
        "gf2p16-custom-rs",
        crate::fields::Gf2p16CustomRs
    );

    macro_rules! generators {
        ($label:literal, $S:ty) => {
            registry::add(Case {
                case: format!("maps/generators/{}", $label),
                api: API_GENERATORS,
                field: $label.to_string(),
                operation: "generator queries".to_string(),
                scope: Scope::Query,
                bytes: 0,
                lanes: 1,
                sources: 0,
                destinations: 0,
                control: false,
                phase: None,
                run: run_generators::<$S>,
            });
        };
    }
    generators!("gf2p8-aes", crate::fields::Gf2p8Aes);
    generators!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
    generators!("gf2p64-rijndael", crate::fields::Gf2p64Rijndael);
    generators!("gf2p64-fanpaar", crate::fields::Gf2p64FanPaar);
    generators!("m31", crate::fields::M31);
    generators!("goldilocks", crate::fields::Gld);
    generators!("qm31", crate::fields::Qm31);
    generators!("gf49", crate::fields::Gf49);
    generators!("gf81", crate::fields::Gf81);
    generators!("gf2p5-padded", crate::fields::Gf2p5Padded);

    pair_case(
        "fail-degree",
        Scope::Query,
        "gf2p16-rijndael",
        "gf2p8-aes",
        API_FAIL_DEGREE,
        0,
        0,
        run_fail_degree::<crate::fields::Gf2p16Rijndael, crate::fields::Gf2p8Aes>,
    );
    pair_case(
        "fail-equal-degree",
        Scope::Query,
        "gf49",
        "gf49-alternate",
        API_FAIL_EQUAL_DEGREE,
        0,
        0,
        run_fail_equal_degree::<
            crate::fields::Gf49,
            fgf::FiniteField<7, 2, fgf::Extension<Alternate49, 2>>,
        >,
    );
    pair_case(
        "fail-images",
        Scope::Query,
        "gf2p8-aes",
        "gf2p16-rijndael",
        API_FAIL_IMAGES,
        0,
        0,
        run_fail_images::<crate::fields::Gf2p8Aes, crate::fields::Gf2p16Rijndael>,
    );
}
