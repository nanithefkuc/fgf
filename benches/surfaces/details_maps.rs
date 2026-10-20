//! Detail cases for stable public callables whose timed bodies no other panel
//! carries: the prepared overwrite gather and matrix operations across the
//! representative routing classes, the coefficient value and prepared-view
//! accessors, the direct vector-elementwise capability pair, and the
//! isomorphism construction/composition/image reads and the defining-relation
//! constructors and read.
//!
//! Overwrite cases run fixed 4 KiB rows with four sources and three matrix
//! outputs; coefficient preparation and the independent per-lane scalar oracle
//! run before timing, and every timed call overwrites a correctly seeded
//! destination, so the overwrite contract itself is what the oracle pins. Map
//! and relation work runs on scalar arithmetic and reports the `scalar`
//! backend; constructions are setup records and pure reads are query records.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::ops::{CoeffMatrix, CoeffVec};
use fgf::{Elem, FieldBuffer, FieldKernels, Isomorphism, Relation, ops};

use crate::fields::{self, Gf2p8Aes, Gf2p8Rs, Gf2p16Rijndael, Qm31};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// Fixed row size for the overwrite cases, rounded down to whole elements.
const ROW_TARGET: usize = 4096;
/// Sources feeding one overwrite call.
const SOURCES: usize = 4;
/// Output rows of one prepared matrix call.
const OUTPUTS: usize = 3;
/// Prepared coefficients behind one view accessor case.
const VIEW_LEN: usize = 8;

fn row_bytes<F: FieldBuffer>() -> usize {
    ROW_TARGET - ROW_TARGET % F::BYTES
}

/// Deterministic nonzero coefficients, one per requested slot.
fn nonzero_coeffs<F: Field>(label: &str, count: usize) -> Vec<Elem<F>> {
    use FieldElem as _;
    let mut seed = seed_for(label);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let candidate = fixture::element::<F>(&mut seed);
        if !candidate.is_zero() && (F::ORDER == 2 || !candidate.is_one()) {
            out.push(candidate);
        }
    }
    out
}

/// A deterministic element of `S` from the case seed.
fn sample<S: Field>(case: &str) -> Elem<S> {
    let mut seed = seed_for(case);
    let index = u128::from(fixture::lcg(&mut seed)) % S::ORDER.max(1);
    <Elem<S> as FieldElem>::from_index(index).expect("index below ORDER")
}

/// The scalar backend name every map and relation record reports.
fn map_backend() -> String {
    fields::scalar_backend_name()
}

/// Independent overwrite oracle: `out[j] = sum_t coeff_sets[t][j] * srcs[t]`,
/// computed per lane with `Elem` arithmetic and packed for byte comparison.
fn overwrite_oracle<F: FieldBuffer>(
    outputs: usize,
    lanes: usize,
    coeff_sets: &[Vec<Elem<F>>],
    srcs: &[&[u8]],
) -> Vec<u8> {
    let src_elems: Vec<Vec<Elem<F>>> = srcs.iter().map(|s| fixture::lanes::<F>(s)).collect();
    let mut expected = vec![0u8; outputs * lanes * F::BYTES];
    for row in 0..outputs {
        let mut lane_elems = vec![<Elem<F> as FieldElem>::ZERO; lanes];
        for (set, src) in coeff_sets.iter().zip(&src_elems) {
            let c = set[row];
            for (acc, &value) in lane_elems.iter_mut().zip(src) {
                *acc += c * value;
            }
        }
        let window = &mut expected[row * lanes * F::BYTES..(row + 1) * lanes * F::BYTES];
        fixture::write_lanes::<F>(window, &lane_elems);
    }
    expected
}

const API_GATHER_INTO_PREPARED: &[&str] = &[
    "fgf::ops::mul_into_gather_with",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_MATRIX_INTO_PREPARED: &[&str] = &[
    "fgf::ops::mul_into_matrix_with",
    "fgf::ops::CoeffMatrix::from_source_major",
];
const API_COEFF_VALUE: &[&str] = &["fgf::ops::Coeff::value", "fgf::ops::Coeff::new"];
const API_COEFFVECREF_GET: &[&str] = &[
    "fgf::ops::CoeffVecRef::get",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_COEFFVECREF_LEN: &[&str] = &[
    "fgf::ops::CoeffVecRef::len",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_COEFFVECREF_IS_EMPTY: &[&str] = &[
    "fgf::ops::CoeffVecRef::is_empty",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_HAS_VECTOR_ELEMENTWISE: &[&str] = &["fgf::kernel::FieldKernels::has_vector_elementwise"];
const API_VECTOR_ELEMENTWISE_MIN_BYTES: &[&str] = &["fgf::kernel::vector_elementwise_min_bytes"];
const API_ISO_FROM_IMAGES: &[&str] = &[
    "fgf::Isomorphism::from_images",
    "fgf::Isomorphism::to_images",
    "fgf::Isomorphism::new",
];
const API_ISO_THEN: &[&str] = &[
    "fgf::Isomorphism::then",
    "fgf::Isomorphism::inverse",
    "fgf::Isomorphism::new",
];
const API_ISO_TO_IMAGES: &[&str] = &["fgf::Isomorphism::to_images", "fgf::Isomorphism::new"];
const API_RELATION_FROM_COORDINATES: &[&str] = &["fgf::Relation::from_coordinates"];
const API_RELATION_FROM_BINARY_WORDS: &[&str] = &["fgf::Relation::from_binary_words"];
const API_RELATION_TO_COORDINATES: &[&str] = &["fgf::Relation::to_coordinates"];

fn register_gather_into_prepared<F: FieldBuffer + FieldKernels>(label: &'static str) {
    let row = row_bytes::<F>();
    registry::add(Case {
        case: format!("details-maps/gather-into-prepared/{label}"),
        api: API_GATHER_INTO_PREPARED,
        field: label.to_string(),
        operation: "gather-into-prepared".to_string(),
        scope: Scope::Steady,
        bytes: row * SOURCES,
        lanes: row * SOURCES / F::BYTES,
        sources: SOURCES,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let row = row_bytes::<F>();
            let lanes = row / F::BYTES;
            let mut seed = seed_for(&case.case);
            let srcs: Vec<Vec<u8>> = (0..SOURCES)
                .map(|_| fixture::packed::<F>(lanes, fixture::lcg(&mut seed)))
                .collect();
            let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
            let cs = nonzero_coeffs::<F>(&case.case, SOURCES);
            let sets: Vec<Vec<Elem<F>>> = cs.iter().map(|c| vec![*c]).collect();
            let prepared = CoeffVec::<F>::new(&cs);
            let view = prepared.as_ref();
            let mut dst = fixture::packed::<F>(lanes, fixture::lcg(&mut seed));
            {
                assert_eq!(
                    view.len(),
                    SOURCES,
                    "{}: one coefficient per source",
                    case.case
                );
                let expected = overwrite_oracle::<F>(1, lanes, &sets, &refs);
                let mut fresh = fixture::packed::<F>(lanes, fixture::lcg(&mut seed));
                ops::mul_into_gather_with::<F>(&mut fresh, view, &refs);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_into_gather_with::<F>(black_box(&mut dst), view, black_box(&refs));
                fixture::observe(dst.len());
            });
        },
    });
}

fn register_matrix_into_prepared<F: FieldBuffer + FieldKernels>(label: &'static str) {
    let row = row_bytes::<F>();
    registry::add(Case {
        case: format!("details-maps/matrix-into-prepared/{label}"),
        api: API_MATRIX_INTO_PREPARED,
        field: label.to_string(),
        operation: "matrix-into-prepared".to_string(),
        scope: Scope::Steady,
        bytes: row * SOURCES * OUTPUTS,
        lanes: row * SOURCES * OUTPUTS / F::BYTES,
        sources: SOURCES,
        destinations: OUTPUTS,
        control: false,
        phase: None,
        run: |case, ctx| {
            let row = row_bytes::<F>();
            let lanes = row / F::BYTES;
            let mut seed = seed_for(&case.case);
            let srcs: Vec<Vec<u8>> = (0..SOURCES)
                .map(|_| fixture::packed::<F>(lanes, fixture::lcg(&mut seed)))
                .collect();
            let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
            let coeff_sets: Vec<Vec<Elem<F>>> = (0..SOURCES)
                .map(|t| nonzero_coeffs::<F>(&format!("{}/source{t}", case.case), OUTPUTS))
                .collect();
            let flat: Vec<Elem<F>> = coeff_sets.iter().flatten().copied().collect();
            let matrix = CoeffMatrix::<F>::from_source_major(SOURCES, OUTPUTS, &flat);
            let mut rows = fixture::packed::<F>(lanes * OUTPUTS, fixture::lcg(&mut seed));
            {
                assert_eq!(
                    matrix.source_count(),
                    SOURCES,
                    "{}: matrix sources",
                    case.case
                );
                assert_eq!(
                    matrix.output_count(),
                    OUTPUTS,
                    "{}: matrix outputs",
                    case.case
                );
                let expected = overwrite_oracle::<F>(OUTPUTS, lanes, &coeff_sets, &refs);
                let mut fresh = fixture::packed::<F>(lanes * OUTPUTS, fixture::lcg(&mut seed));
                ops::mul_into_matrix_with::<F>(&mut fresh, row, &matrix, &refs);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_into_matrix_with::<F>(
                    black_box(&mut rows),
                    row,
                    &matrix,
                    black_box(&refs),
                );
                fixture::observe(rows.len());
            });
        },
    });
}

fn register_coeff_value<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/coeff-value/{label}"),
        api: API_COEFF_VALUE,
        field: label.to_string(),
        operation: "coeff-value".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 1,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let c = nonzero_coeffs::<F>(&case.case, 1).remove(0);
            let prepared = ops::Coeff::<F>::new(c);
            assert_eq!(
                ops::Coeff::<F>::value(&prepared),
                c,
                "{}: value round trip",
                case.case
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                let value = ops::Coeff::<F>::value(black_box(&prepared));
                fixture::observe(value.to_raw());
            });
        },
    });
}

fn register_coeffvecref_get<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/coeffvecref-get/{label}"),
        api: API_COEFFVECREF_GET,
        field: label.to_string(),
        operation: "coeffvecref-get".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: VIEW_LEN,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let cs = nonzero_coeffs::<F>(&case.case, VIEW_LEN);
            let prepared = CoeffVec::<F>::new(&cs);
            let view = prepared.as_ref();
            assert_eq!(view.get(0).map(|c| c.value()), Some(cs[0]), "{}", case.case);
            assert!(view.get(VIEW_LEN).is_none(), "{}", case.case);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                let found = black_box(view).get(black_box(VIEW_LEN - 3));
                fixture::observe(found);
            });
        },
    });
}

fn register_coeffvecref_len<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/coeffvecref-len/{label}"),
        api: API_COEFFVECREF_LEN,
        field: label.to_string(),
        operation: "coeffvecref-len".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: VIEW_LEN,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let cs = nonzero_coeffs::<F>(&case.case, VIEW_LEN);
            let prepared = CoeffVec::<F>::new(&cs);
            let view = prepared.as_ref();
            assert_eq!(view.len(), VIEW_LEN, "{}", case.case);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                fixture::observe(black_box(view).len());
            });
        },
    });
}

fn register_coeffvecref_is_empty<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/coeffvecref-is-empty/{label}"),
        api: API_COEFFVECREF_IS_EMPTY,
        field: label.to_string(),
        operation: "coeffvecref-is-empty".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: VIEW_LEN,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let cs = nonzero_coeffs::<F>(&case.case, VIEW_LEN);
            let prepared = CoeffVec::<F>::new(&cs);
            let view = prepared.as_ref();
            let empty = CoeffVec::<F>::new(&[]);
            assert!(!view.is_empty(), "{}", case.case);
            assert!(empty.as_ref().is_empty(), "{}", case.case);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                fixture::observe(black_box(view).is_empty());
            });
        },
    });
}

fn register_has_vector_elementwise<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/has-vector-elementwise/{label}"),
        api: API_HAS_VECTOR_ELEMENTWISE,
        field: label.to_string(),
        operation: "has-vector-elementwise".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 1,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let has = <F as FieldKernels>::has_vector_elementwise();
            assert_eq!(
                has,
                fgf::has_vector_elementwise::<F>(),
                "{}: free wrapper agrees with the trait method",
                case.case
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                fixture::observe(<F as FieldKernels>::has_vector_elementwise());
            });
        },
    });
}

fn register_vector_elementwise_min_bytes<F: FieldKernels>(label: &'static str) {
    registry::add(Case {
        case: format!("details-maps/vector-elementwise-min-bytes/{label}"),
        api: API_VECTOR_ELEMENTWISE_MIN_BYTES,
        field: label.to_string(),
        operation: "vector-elementwise-min-bytes".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 1,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: |case, ctx| {
            let min_bytes = fgf::kernel::vector_elementwise_min_bytes::<F>();
            assert_eq!(
                min_bytes,
                <F as FieldKernels>::vector_elementwise_min_bytes(),
                "{}: free wrapper agrees with the trait method",
                case.case
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                fixture::observe(fgf::kernel::vector_elementwise_min_bytes::<F>());
            });
        },
    });
}

#[allow(clippy::too_many_arguments)] // Parameters mirror the measurement record.
fn iso_case(
    op: &'static str,
    scope: Scope,
    chain: &str,
    api: &'static [&'static str],
    lanes: usize,
    sources: usize,
    destinations: usize,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case: format!("details-maps/{op}/{chain}"),
        api,
        field: chain.to_string(),
        operation: op.to_string(),
        scope,
        bytes: 0,
        lanes,
        sources,
        destinations,
        control: false,
        phase: None,
        run,
    });
}

fn run_iso_from_images<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let canonical = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let images: Vec<Elem<T>> = canonical.to_images().to_vec();
    let x = sample::<S>(&case.case);
    {
        let rebuilt =
            Isomorphism::<S, T>::from_images(&images).expect("canonical images are valid");
        let y = canonical.apply(x);
        assert_eq!(rebuilt.apply(x), y, "{}: forward agreement", case.case);
        assert_eq!(rebuilt.unapply(y), x, "{}: unapply round trip", case.case);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let iso = Isomorphism::<S, T>::from_images(black_box(&images));
        fixture::observe(iso.map_or(0, |m| m.to_images().len()));
    });
}

fn run_iso_then<A, B>(case: &Case, ctx: &CaseCtx)
where
    A: Field,
    B: Field<Characteristic = A::Characteristic>,
{
    let first = Isomorphism::<A, B>::new().expect("canonical isomorphism exists");
    let second = first.inverse();
    let x = sample::<A>(&case.case);
    {
        let composed = first.then(&second);
        assert_eq!(
            second.apply(first.apply(x)),
            x,
            "{}: inverse composition round trip",
            case.case
        );
        assert_eq!(
            composed.apply(x),
            x,
            "{}: then with the inverse is the identity",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let composed = first.then(&second);
        fixture::observe(composed.to_images().len());
    });
}

fn run_iso_to_images<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let iso = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    {
        let rebuilt =
            Isomorphism::<S, T>::from_images(iso.to_images()).expect("canonical images are valid");
        let x = sample::<S>(&case.case);
        assert_eq!(
            rebuilt.apply(x),
            iso.apply(x),
            "{}: images reproduce the map",
            case.case
        );
        assert_eq!(iso.to_images(), rebuilt.to_images(), "{}", case.case);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let held = black_box(iso);
        let borrowed = held.to_images();
        fixture::observe(borrowed.len());
    });
}

fn run_relation_from_coordinates(case: &Case, ctx: &CaseCtx) {
    // The `Gf49Spec` relation: `t² + t + 3` over GF(7).
    const COORDINATES: &[u64] = &[3, 1];
    {
        let relation = Relation::from_coordinates(COORDINATES);
        assert_eq!(
            relation.to_coordinates(),
            COORDINATES,
            "{}: coordinate readback",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let relation = Relation::from_coordinates(black_box(COORDINATES));
        fixture::observe(relation.to_coordinates().len());
    });
}

fn run_relation_from_binary_words(case: &Case, ctx: &CaseCtx) {
    // The custom RS tower relation: `t² + t + 0x21` over GF(2^8)/RS, one
    // word per coefficient, expanded least significant bit first.
    const WORDS: &[u64] = &[0x21, 0x01];
    {
        let relation = Relation::from_binary_words::<Gf2p8Rs>(WORDS);
        let mut expected = Vec::<u64>::new();
        for &word in WORDS {
            for bit in 0..<Gf2p8Rs as Field>::DEGREE as usize {
                expected.push((word >> bit) & 1);
            }
        }
        assert_eq!(
            relation.to_coordinates(),
            expected.as_slice(),
            "{}: word expansion",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let relation = Relation::from_binary_words::<Gf2p8Rs>(black_box(WORDS));
        fixture::observe(relation.to_coordinates().len());
    });
}

fn run_relation_to_coordinates(case: &Case, ctx: &CaseCtx) {
    let relation = Relation::from_coordinates(&[3, 1]);
    assert_eq!(
        relation.to_coordinates(),
        &[3, 1],
        "{}: readback",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let held = black_box(relation);
        let borrowed = held.to_coordinates();
        fixture::observe(borrowed.len());
    });
}

/// The representative routing classes for the prepared overwrite operations.
macro_rules! overwrite_fields {
    ($callback:ident) => {
        // Byte fields: AES and RS polynomials.
        $callback!("gf2p8-aes", crate::fields::Gf2p8Aes);
        $callback!("gf2p8-rs", crate::fields::Gf2p8Rs);
        // Packed towers: Rijndael16 and Fan-Paar16.
        $callback!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
        $callback!("gf2p16-fanpaar", crate::fields::Gf2p16FanPaar);
        // Named prime folds.
        $callback!("m31", crate::fields::M31);
        $callback!("goldilocks", crate::fields::Gld);
        // Generic residue rows at one lane per u8/u16/u32/u64, the u64 row
        // being the largest prime.
        $callback!("gf101", crate::fields::Gf101);
        $callback!("gf65521", crate::fields::Gf65521);
        $callback!("gfu32max", crate::fields::GfU32Max);
        $callback!("gfmaxp", crate::fields::GfMaxPrime);
        // A user-side extension spec over the RS polynomial.
        $callback!("gf2p16-custom-rs", crate::fields::Gf2p16CustomRs);
    };
}

/// Register the details panel: one independently timed case per stable
/// callable no other panel times.
pub fn register() {
    macro_rules! overwrite {
        ($label:literal, $F:ty) => {
            register_gather_into_prepared::<$F>($label);
            register_matrix_into_prepared::<$F>($label);
        };
    }
    overwrite_fields!(overwrite);

    register_coeff_value::<Gf2p8Aes>("gf2p8-aes");
    register_coeffvecref_get::<Gf2p16Rijndael>("gf2p16-rijndael");
    register_coeffvecref_len::<Gf2p16Rijndael>("gf2p16-rijndael");
    register_coeffvecref_is_empty::<Gf2p16Rijndael>("gf2p16-rijndael");
    register_has_vector_elementwise::<Qm31>("qm31");
    register_vector_elementwise_min_bytes::<Qm31>("qm31");

    iso_case(
        "iso-from-images",
        Scope::Setup,
        "gf2p8-aes->gf2p8-rs",
        API_ISO_FROM_IMAGES,
        8,
        1,
        1,
        run_iso_from_images::<Gf2p8Aes, Gf2p8Rs>,
    );
    iso_case(
        "iso-then",
        Scope::Setup,
        "gf2p8-aes->gf2p8-rs->gf2p8-aes",
        API_ISO_THEN,
        1,
        1,
        1,
        run_iso_then::<Gf2p8Aes, Gf2p8Rs>,
    );
    iso_case(
        "iso-to-images",
        Scope::Query,
        "gf2p8-aes->gf2p8-rs",
        API_ISO_TO_IMAGES,
        8,
        0,
        0,
        run_iso_to_images::<Gf2p8Aes, Gf2p8Rs>,
    );

    registry::add(Case {
        case: "details-maps/relation-from-coordinates/gf49".to_string(),
        api: API_RELATION_FROM_COORDINATES,
        field: "gf49".to_string(),
        operation: "relation-from-coordinates".to_string(),
        scope: Scope::Setup,
        bytes: 0,
        lanes: 2,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: run_relation_from_coordinates,
    });
    registry::add(Case {
        case: "details-maps/relation-from-binary-words/gf2p16-custom-rs".to_string(),
        api: API_RELATION_FROM_BINARY_WORDS,
        field: "gf2p16-custom-rs".to_string(),
        operation: "relation-from-binary-words".to_string(),
        scope: Scope::Setup,
        bytes: 0,
        lanes: 16,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: run_relation_from_binary_words,
    });
    registry::add(Case {
        case: "details-maps/relation-to-coordinates/gf49".to_string(),
        api: API_RELATION_TO_COORDINATES,
        field: "gf49".to_string(),
        operation: "relation-to-coordinates".to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 2,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run: run_relation_to_coordinates,
    });
}
