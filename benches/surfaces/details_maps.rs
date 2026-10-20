//! Detail cases for stable public callables whose timed bodies no other panel
//! carries: the prepared overwrite gather and matrix operations across the
//! representative routing classes, the coefficient value and prepared-view
//! accessors, the direct vector-elementwise capability pair, the
//! isomorphism construction/composition/image reads, the defining-relation
//! constructors and read, and the prepared map surface: compact and prepared
//! scalar maps, the prepared map row functions, and the bit-packed GF(2)
//! map rows.
//!
//! Overwrite cases run fixed 4 KiB rows with four sources and three matrix
//! outputs; coefficient preparation and the independent per-lane scalar oracle
//! run before timing, and every timed call overwrites a correctly seeded
//! destination, so the overwrite contract itself is what the oracle pins. Map
//! and relation work runs on scalar arithmetic and reports the `scalar`
//! backend; constructions are setup records and pure reads are query records.
//! Prepared-map oracles are the compact maps themselves (a prepared map must
//! agree with the compact map element by element), the defining identities
//! (`restrict ∘ embed` round trips, relative Frobenius against `pow`, trace
//! linearity), and the bitmap conventions of `fgf::bits`.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::ops::{CoeffMatrix, CoeffVec};
use fgf::{
    Elem, Embedding, FieldBuffer, FieldKernels, Frobenius, Isomorphism, PreparedEmbedding,
    PreparedIsomorphism, Relation, ops,
};

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

const API_PREPARED_EMBED_BUILD: &[&str] = &["fgf::PreparedEmbedding::new", "fgf::Embedding::new"];
const API_PREPARED_EMBED_ACCESSORS: &[&str] = &[
    "fgf::PreparedEmbedding::embedding",
    "fgf::PreparedEmbedding::relative_degree",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_EMBED: &[&str] = &[
    "fgf::PreparedEmbedding::embed",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_RESTRICT: &[&str] = &[
    "fgf::PreparedEmbedding::restrict",
    "fgf::PreparedEmbedding::embed",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_CONTAINS: &[&str] = &[
    "fgf::PreparedEmbedding::contains",
    "fgf::PreparedEmbedding::embed",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_FROBENIUS: &[&str] = &[
    "fgf::PreparedEmbedding::frobenius",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_TRACE: &[&str] = &[
    "fgf::PreparedEmbedding::trace",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_EMBED_NORM: &[&str] = &[
    "fgf::PreparedEmbedding::norm",
    "fgf::PreparedEmbedding::new",
];
const API_PREPARED_ISO_BUILD: &[&str] = &["fgf::PreparedIsomorphism::new", "fgf::Isomorphism::new"];
const API_PREPARED_ISO_ACCESSORS: &[&str] = &[
    "fgf::PreparedIsomorphism::isomorphism",
    "fgf::PreparedIsomorphism::new",
];
const API_PREPARED_ISO_APPLY: &[&str] = &[
    "fgf::PreparedIsomorphism::apply",
    "fgf::PreparedIsomorphism::new",
];
const API_PREPARED_ISO_UNAPPLY: &[&str] = &[
    "fgf::PreparedIsomorphism::unapply",
    "fgf::PreparedIsomorphism::new",
];
const API_MAP_ROW_EMBED: &[&str] = &[
    "fgf::ops::map::embed_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_ROW_TRACE: &[&str] = &[
    "fgf::ops::map::trace_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_ROW_NORM: &[&str] = &[
    "fgf::ops::map::norm_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_ROW_RESTRICT: &[&str] = &[
    "fgf::ops::map::restrict_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_ROW_CONTAINS: &[&str] = &[
    "fgf::ops::map::contains_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_ROW_APPLY: &[&str] = &[
    "fgf::ops::map::apply_into_with",
    "fgf::PreparedIsomorphism::new",
    "fgf::Isomorphism::new",
];
const API_MAP_ROW_UNAPPLY: &[&str] = &[
    "fgf::ops::map::unapply_into_with",
    "fgf::PreparedIsomorphism::new",
    "fgf::Isomorphism::new",
];
const API_MAP_ROW_FROBENIUS: &[&str] =
    &["fgf::ops::map::frobenius_into_with", "fgf::Frobenius::new"];
const API_MAP_ROW_FROBENIUS_ASSIGN: &[&str] = &[
    "fgf::ops::map::frobenius_assign_with",
    "fgf::Frobenius::new",
];
const API_MAP_BITS_EMBED: &[&str] = &[
    "fgf::ops::map_bits::embed_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_BITS_TRACE: &[&str] = &[
    "fgf::ops::map_bits::trace_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_BITS_NORM: &[&str] = &[
    "fgf::ops::map_bits::norm_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_BITS_RESTRICT: &[&str] = &[
    "fgf::ops::map_bits::restrict_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];
const API_MAP_BITS_CONTAINS: &[&str] = &[
    "fgf::ops::map_bits::contains_into_with",
    "fgf::PreparedEmbedding::new",
    "fgf::Embedding::new",
];

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

/// Lanes in one prepared map row case.
const MAP_ROW_LANES: usize = 64;
/// Bit count of one bit-packed map row case.
const MAP_BITS: usize = 64;

/// Prepared map row fixture: the compact map, its prepared form, the packed
/// source row, and the target lanes.
type MapRowFixture<S, T> = (Embedding<S, T>, PreparedEmbedding<S, T>, Vec<u8>, Vec<u8>);

/// One `maps/`-prefixed record over a composite `S->T` presentation field.
#[allow(clippy::too_many_arguments)] // Parameters mirror the measurement record.
fn map_case(
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
        case: format!("maps/{op}/{chain}"),
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

fn run_prepared_embed_build<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    {
        let prepared = PreparedEmbedding::<S, T>::new(&compact);
        assert_eq!(
            prepared.relative_degree(),
            compact.relative_degree(),
            "{}: prepared relative degree",
            case.case
        );
        let x = sample::<S>(&case.case);
        assert_eq!(
            prepared.embed(x),
            compact.embed(x),
            "{}: prepared forward agrees with the compact map",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let prepared = PreparedEmbedding::<S, T>::new(black_box(&compact));
        fixture::observe(prepared.relative_degree());
    });
}

fn run_prepared_embed_accessors<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    {
        assert_eq!(
            prepared.embedding().to_images(),
            compact.to_images(),
            "{}: readback reproduces the compact map",
            case.case
        );
        assert_eq!(
            prepared.relative_degree(),
            T::DEGREE / S::DEGREE,
            "{}: relative degree spans the pair",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let held = black_box(&prepared);
        fixture::observe((held.embedding().relative_degree(), held.relative_degree()));
    });
}

fn run_prepared_embed_embed<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let mut seed = seed_for(&case.case);
    let xs = fixture::elements::<S>(4, &mut seed);
    {
        for x in &xs {
            assert_eq!(
                prepared.embed(*x),
                compact.embed(*x),
                "{}: prepared forward agrees with the compact map",
                case.case
            );
        }
        let (a, b) = (xs[0], xs[1]);
        assert_eq!(
            prepared.embed(a.add(b)),
            prepared.embed(a).add(prepared.embed(b)),
            "{}: additive homomorphism",
            case.case
        );
        assert_eq!(
            prepared.embed(a.mul(b)),
            prepared.embed(a).mul(prepared.embed(b)),
            "{}: multiplicative homomorphism",
            case.case
        );
    }
    let x = xs[0];
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.embed(black_box(x)).to_raw());
    });
}

fn run_prepared_embed_restrict<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let x = sample::<S>(&case.case);
    {
        assert_eq!(
            prepared.restrict(prepared.embed(x)),
            Some(x),
            "{}: image restricts to its preimage",
            case.case
        );
        for probe in 0..8u64 {
            let candidate =
                fixture::elements::<T>(1, &mut seed_for(&format!("{}/{probe}", case.case)))
                    .remove(0);
            assert_eq!(
                prepared.contains(candidate),
                prepared.restrict(candidate).is_some(),
                "{}: contains agrees with restrict",
                case.case
            );
            assert_eq!(
                prepared.restrict(candidate),
                compact.restrict(candidate),
                "{}: restriction agrees with the compact map",
                case.case
            );
        }
    }
    let y = prepared.embed(x);
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.restrict(black_box(y)).map(|v| v.to_raw()));
    });
}

fn run_prepared_embed_contains<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let y_hit = prepared.embed(sample::<S>(&case.case));
    {
        assert!(
            prepared.contains(y_hit),
            "{}: image element is contained",
            case.case
        );
        for probe in 0..8u64 {
            let candidate =
                fixture::elements::<T>(1, &mut seed_for(&format!("{}/{probe}", case.case)))
                    .remove(0);
            assert_eq!(
                prepared.contains(candidate),
                compact.contains(candidate),
                "{}: membership agrees with the compact map",
                case.case
            );
        }
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.contains(black_box(y_hit)));
    });
}

fn run_prepared_embed_frobenius<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let y = prepared.embed(sample::<S>(&case.case));
    assert_eq!(
        prepared.frobenius(y),
        y.pow(S::ORDER),
        "{}: relative Frobenius oracle",
        case.case
    );
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.frobenius(black_box(y)).to_raw());
    });
}

fn run_prepared_embed_trace<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let x = sample::<S>(&case.case);
    let relative = prepared.relative_degree();
    let mut multiple = <Elem<S> as FieldElem>::ZERO;
    for _ in 0..relative {
        multiple = multiple.add(<Elem<S> as FieldElem>::ONE);
    }
    let mut seed = seed_for(&case.case);
    let elems = fixture::elements::<T>(2, &mut seed);
    {
        assert_eq!(
            prepared.trace(prepared.embed(x)),
            x.mul(multiple),
            "{}: trace of a subfield element",
            case.case
        );
        let (a, b) = (elems[0], elems[1]);
        assert_eq!(
            prepared.trace(a.add(b)),
            prepared.trace(a).add(prepared.trace(b)),
            "{}: trace is additive",
            case.case
        );
    }
    let y = elems[0];
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.trace(black_box(y)).to_raw());
    });
}

fn run_prepared_embed_norm<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let x = sample::<S>(&case.case);
    assert_eq!(
        prepared.norm(prepared.embed(x)),
        x.pow(u128::from(prepared.relative_degree())),
        "{}: norm of a subfield element",
        case.case
    );
    let mut seed = seed_for(&case.case);
    let y = fixture::elements::<T>(1, &mut seed).remove(0);
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.norm(black_box(y)).to_raw());
    });
}

fn run_prepared_iso_build<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    {
        let prepared = PreparedIsomorphism::<S, T>::new(&compact);
        let x = sample::<S>(&case.case);
        assert_eq!(
            prepared.apply(x),
            compact.apply(x),
            "{}: prepared forward agrees with the compact map",
            case.case
        );
        assert_eq!(
            prepared.unapply(compact.apply(x)),
            x,
            "{}: prepared reverse round trip",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        let prepared = PreparedIsomorphism::<S, T>::new(black_box(&compact));
        fixture::observe(prepared.isomorphism().to_images().len());
    });
}

fn run_prepared_iso_accessors<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let prepared = PreparedIsomorphism::<S, T>::new(&compact);
    {
        assert_eq!(
            prepared.isomorphism().to_images(),
            compact.to_images(),
            "{}: readback reproduces the compact map",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(&prepared).isomorphism().to_images().len());
    });
}

fn run_prepared_iso_apply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let prepared = PreparedIsomorphism::<S, T>::new(&compact);
    let mut seed = seed_for(&case.case);
    let xs = fixture::elements::<S>(4, &mut seed);
    {
        for x in &xs {
            assert_eq!(
                prepared.apply(*x),
                compact.apply(*x),
                "{}: prepared forward agrees with the compact map",
                case.case
            );
        }
        let (a, b) = (xs[0], xs[1]);
        assert_eq!(
            prepared.apply(a.mul(b)),
            prepared.apply(a).mul(prepared.apply(b)),
            "{}: multiplicative homomorphism",
            case.case
        );
    }
    let x = xs[0];
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.apply(black_box(x)).to_raw());
    });
}

fn run_prepared_iso_unapply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let prepared = PreparedIsomorphism::<S, T>::new(&compact);
    let x = sample::<S>(&case.case);
    let y = compact.apply(x);
    {
        assert_eq!(
            prepared.unapply(y),
            compact.unapply(y),
            "{}: prepared reverse agrees with the compact map",
            case.case
        );
        assert_eq!(
            prepared.unapply(prepared.apply(x)),
            x,
            "{}: round trip",
            case.case
        );
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(prepared.unapply(black_box(y)).to_raw());
    });
}

/// The shared prepared embedding row fixture: the compact map, its prepared
/// form, a source row, and a mixed hit/outsider target row.
fn embed_row_fixture<S, T>(case: &Case) -> MapRowFixture<S, T>
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let src = fixture::packed::<S>(MAP_ROW_LANES, seed_for(&case.case));
    let mut seed = seed_for(&format!("{}/targets", case.case));
    let mut targets = Vec::with_capacity(MAP_ROW_LANES * T::BYTES);
    for lane in 0..MAP_ROW_LANES {
        let value = if lane % 2 == 0 {
            // Image lane: the embedding of a deterministic source element.
            let mut state = fixture::lcg(&mut seed);
            compact.embed(fixture::element::<S>(&mut state))
        } else {
            fixture::element::<T>(&mut seed)
        };
        let mut encoded = vec![0u8; T::BYTES];
        T::encode(&mut encoded, value);
        targets.extend_from_slice(&encoded);
    }
    (compact, prepared, src, targets)
}

fn run_row_embed<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, src, _) = embed_row_fixture::<S, T>(case);
    let mut dst = vec![0u8; MAP_ROW_LANES * T::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (out, inp) in expected
            .chunks_exact_mut(T::BYTES)
            .zip(src.chunks_exact(S::BYTES))
        {
            T::encode(out, compact.embed(S::decode(inp)));
        }
        ops::map::embed_into_with::<S, T>(&mut dst, &prepared, &src);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::embed_into_with::<S, T>(black_box(&mut dst), &prepared, black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_row_trace<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = embed_row_fixture::<S, T>(case);
    let mut dst = vec![0u8; MAP_ROW_LANES * S::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (out, inp) in expected
            .chunks_exact_mut(S::BYTES)
            .zip(targets.chunks_exact(T::BYTES))
        {
            S::encode(out, compact.trace(T::decode(inp)));
        }
        ops::map::trace_into_with::<S, T>(&mut dst, &prepared, &targets);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::trace_into_with::<S, T>(black_box(&mut dst), &prepared, black_box(&targets));
        fixture::observe(dst.len());
    });
}

fn run_row_norm<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = embed_row_fixture::<S, T>(case);
    let mut dst = vec![0u8; MAP_ROW_LANES * S::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (out, inp) in expected
            .chunks_exact_mut(S::BYTES)
            .zip(targets.chunks_exact(T::BYTES))
        {
            S::encode(out, compact.norm(T::decode(inp)));
        }
        ops::map::norm_into_with::<S, T>(&mut dst, &prepared, &targets);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::norm_into_with::<S, T>(black_box(&mut dst), &prepared, black_box(&targets));
        fixture::observe(dst.len());
    });
}

fn run_row_restrict<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = embed_row_fixture::<S, T>(case);
    let valid_len = fgf::bits::bytes_for(MAP_ROW_LANES);
    let mut dst = vec![0u8; MAP_ROW_LANES * S::BYTES];
    let mut valid = vec![0u8; valid_len];
    {
        let mut expected = vec![0u8; dst.len()];
        let mut expected_valid = vec![0u8; valid_len];
        for (index, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            match compact.restrict(T::decode(inp)) {
                Some(preimage) => {
                    S::encode(
                        &mut expected[index * S::BYTES..(index + 1) * S::BYTES],
                        preimage,
                    );
                    expected_valid[index / 8] |= 1 << (index % 8);
                }
                None => {
                    S::encode(
                        &mut expected[index * S::BYTES..(index + 1) * S::BYTES],
                        Elem::<S>::ZERO,
                    );
                }
            }
        }
        ops::map::restrict_into_with::<S, T>(&mut dst, &mut valid, &prepared, &targets);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
        fixture::assert_bytes_equal(&case.case, &valid, &expected_valid);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::restrict_into_with::<S, T>(
            black_box(&mut dst),
            black_box(&mut valid),
            &prepared,
            black_box(&targets),
        );
        fixture::observe(dst.len());
    });
}

fn run_row_contains<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = embed_row_fixture::<S, T>(case);
    let valid_len = fgf::bits::bytes_for(MAP_ROW_LANES);
    let mut valid = vec![0u8; valid_len];
    {
        let mut expected_valid = vec![0u8; valid_len];
        for (index, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            if compact.contains(T::decode(inp)) {
                expected_valid[index / 8] |= 1 << (index % 8);
            }
        }
        ops::map::contains_into_with::<S, T>(&mut valid, &prepared, &targets);
        fixture::assert_bytes_equal(&case.case, &valid, &expected_valid);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::contains_into_with::<S, T>(black_box(&mut valid), &prepared, black_box(&targets));
        fixture::observe(valid.len());
    });
}

fn run_row_apply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let prepared = PreparedIsomorphism::<S, T>::new(&compact);
    let src = fixture::packed::<S>(MAP_ROW_LANES, seed_for(&case.case));
    let mut dst = vec![0u8; MAP_ROW_LANES * T::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (out, inp) in expected
            .chunks_exact_mut(T::BYTES)
            .zip(src.chunks_exact(S::BYTES))
        {
            T::encode(out, compact.apply(S::decode(inp)));
        }
        ops::map::apply_into_with::<S, T>(&mut dst, &prepared, &src);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::apply_into_with::<S, T>(black_box(&mut dst), &prepared, black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_row_unapply<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
    let prepared = PreparedIsomorphism::<S, T>::new(&compact);
    let targets = fixture::packed::<T>(MAP_ROW_LANES, seed_for(&case.case));
    let mut dst = vec![0u8; MAP_ROW_LANES * S::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (out, inp) in expected
            .chunks_exact_mut(S::BYTES)
            .zip(targets.chunks_exact(T::BYTES))
        {
            S::encode(out, compact.unapply(T::decode(inp)));
        }
        ops::map::unapply_into_with::<S, T>(&mut dst, &prepared, &targets);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::unapply_into_with::<S, T>(black_box(&mut dst), &prepared, black_box(&targets));
        fixture::observe(dst.len());
    });
}

/// One single-field Frobenius row: `dst[i]` becomes `x^(p^k)` of `src[i]`.
fn frobenius_row_fixture<F: FieldBuffer>(case: &Case, power: u32) -> (Frobenius<F>, Vec<u8>) {
    let map = Frobenius::<F>::new(power);
    let src = fixture::packed::<F>(MAP_ROW_LANES, seed_for(&case.case));
    (map, src)
}

fn run_row_frobenius(case: &Case, ctx: &CaseCtx) {
    type F = Gf2p16Rijndael;
    let (map, src) = frobenius_row_fixture::<F>(case, 1);
    let mut dst = vec![0u8; src.len()];
    {
        // Characteristic two: the first Frobenius power is the square.
        let mut expected = vec![0u8; src.len()];
        for (out, inp) in expected
            .chunks_exact_mut(F::BYTES)
            .zip(src.chunks_exact(F::BYTES))
        {
            F::encode(out, F::decode(inp).square());
        }
        ops::map::frobenius_into_with::<F>(&mut dst, &map, &src);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::frobenius_into_with::<F>(black_box(&mut dst), &map, black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_row_frobenius_assign(case: &Case, ctx: &CaseCtx) {
    type F = crate::fields::M31;
    const P: u128 = 0x7FFF_FFFF;
    let (map, mut row) = frobenius_row_fixture::<F>(case, 1);
    {
        let mut expected = vec![0u8; row.len()];
        for (out, inp) in expected
            .chunks_exact_mut(F::BYTES)
            .zip(row.chunks_exact(F::BYTES))
        {
            F::encode(out, F::decode(inp).pow(P));
        }
        ops::map::frobenius_assign_with::<F>(&mut row, &map);
        fixture::assert_bytes_equal(&case.case, &row, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map::frobenius_assign_with::<F>(black_box(&mut row), &map);
        fixture::observe(row.len());
    });
}

/// One bit-packed GF(2) map row: the packed source bits, the target lanes,
/// and the compact map that provides the oracle.
fn bits_row_fixture<S, T>(case: &Case) -> MapRowFixture<S, T>
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("canonical map exists");
    let prepared = PreparedEmbedding::<S, T>::new(&compact);
    let packed_len = fgf::bits::bytes_for(MAP_BITS);
    let src = fixture::noise(packed_len, seed_for(&case.case));
    let mut seed = seed_for(&format!("{}/targets", case.case));
    let mut targets = Vec::with_capacity(MAP_BITS * T::BYTES);
    for lane in 0..MAP_BITS {
        let bit = (src[lane / 8] >> (lane % 8)) & 1;
        let value = if lane % 2 == 0 {
            compact.embed(if bit == 1 {
                <Elem<S> as FieldElem>::ONE
            } else {
                <Elem<S> as FieldElem>::ZERO
            })
        } else {
            fixture::element::<T>(&mut seed)
        };
        let mut encoded = vec![0u8; T::BYTES];
        T::encode(&mut encoded, value);
        targets.extend_from_slice(&encoded);
    }
    (compact, prepared, src, targets)
}

fn run_bits_embed<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, src, _) = bits_row_fixture::<S, T>(case);
    let mut dst = vec![0u8; MAP_BITS * T::BYTES];
    {
        let mut expected = vec![0u8; dst.len()];
        for (lane, out) in expected.chunks_exact_mut(T::BYTES).enumerate() {
            let bit = (src[lane / 8] >> (lane % 8)) & 1;
            let value = compact.embed(if bit == 1 {
                <Elem<S> as FieldElem>::ONE
            } else {
                <Elem<S> as FieldElem>::ZERO
            });
            T::encode(out, value);
        }
        ops::map_bits::embed_into_with::<S, T>(&mut dst, &prepared, &src, MAP_BITS);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map_bits::embed_into_with::<S, T>(
            black_box(&mut dst),
            &prepared,
            black_box(&src),
            MAP_BITS,
        );
        fixture::observe(dst.len());
    });
}

fn run_bits_trace<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = bits_row_fixture::<S, T>(case);
    let packed_len = fgf::bits::bytes_for(MAP_BITS);
    let mut dst = vec![0u8; packed_len];
    {
        let mut expected = vec![0u8; packed_len];
        for (lane, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            if compact.trace(T::decode(inp)) == <Elem<S> as FieldElem>::ONE {
                expected[lane / 8] |= 1 << (lane % 8);
            }
        }
        ops::map_bits::trace_into_with::<S, T>(&mut dst, &prepared, &targets, MAP_BITS);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map_bits::trace_into_with::<S, T>(
            black_box(&mut dst),
            &prepared,
            black_box(&targets),
            MAP_BITS,
        );
        fixture::observe(dst.len());
    });
}

fn run_bits_norm<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = bits_row_fixture::<S, T>(case);
    let packed_len = fgf::bits::bytes_for(MAP_BITS);
    let mut dst = vec![0u8; packed_len];
    {
        let mut expected = vec![0u8; packed_len];
        for (lane, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            if compact.norm(T::decode(inp)) == <Elem<S> as FieldElem>::ONE {
                expected[lane / 8] |= 1 << (lane % 8);
            }
        }
        ops::map_bits::norm_into_with::<S, T>(&mut dst, &prepared, &targets, MAP_BITS);
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map_bits::norm_into_with::<S, T>(
            black_box(&mut dst),
            &prepared,
            black_box(&targets),
            MAP_BITS,
        );
        fixture::observe(dst.len());
    });
}

fn run_bits_restrict<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = bits_row_fixture::<S, T>(case);
    let packed_len = fgf::bits::bytes_for(MAP_BITS);
    let mut dst = vec![0u8; packed_len];
    let mut valid = vec![0u8; packed_len];
    {
        let mut expected = vec![0u8; packed_len];
        let mut expected_valid = vec![0u8; packed_len];
        for (lane, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            if let Some(preimage) = compact.restrict(T::decode(inp)) {
                if preimage == <Elem<S> as FieldElem>::ONE {
                    expected[lane / 8] |= 1 << (lane % 8);
                }
                expected_valid[lane / 8] |= 1 << (lane % 8);
            }
        }
        ops::map_bits::restrict_into_with::<S, T>(
            &mut dst, &mut valid, &prepared, &targets, MAP_BITS,
        );
        fixture::assert_bytes_equal(&case.case, &dst, &expected);
        fixture::assert_bytes_equal(&case.case, &valid, &expected_valid);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map_bits::restrict_into_with::<S, T>(
            black_box(&mut dst),
            black_box(&mut valid),
            &prepared,
            black_box(&targets),
            MAP_BITS,
        );
        fixture::observe(dst.len());
    });
}

fn run_bits_contains<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: fgf::PrimeField<Characteristic = fgf::PrimeCharacteristic<2>>,
    T: fgf::BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared, _, targets) = bits_row_fixture::<S, T>(case);
    let packed_len = fgf::bits::bytes_for(MAP_BITS);
    let mut valid = vec![0u8; packed_len];
    {
        let mut expected_valid = vec![0u8; packed_len];
        for (lane, inp) in targets.chunks_exact(T::BYTES).enumerate() {
            if compact.contains(T::decode(inp)) {
                expected_valid[lane / 8] |= 1 << (lane % 8);
            }
        }
        ops::map_bits::contains_into_with::<S, T>(&mut valid, &prepared, &targets, MAP_BITS);
        fixture::assert_bytes_equal(&case.case, &valid, &expected_valid);
    }
    let backend = map_backend();
    registry::time(case, ctx, &backend, || {
        ops::map_bits::contains_into_with::<S, T>(
            black_box(&mut valid),
            &prepared,
            black_box(&targets),
            MAP_BITS,
        );
        fixture::observe(valid.len());
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

    map_case(
        "prepared-embed-build",
        Scope::Setup,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_BUILD,
        1,
        1,
        1,
        run_prepared_embed_build::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-accessors",
        Scope::Query,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_ACCESSORS,
        1,
        0,
        0,
        run_prepared_embed_accessors::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-embed",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_EMBED,
        1,
        1,
        1,
        run_prepared_embed_embed::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-restrict",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_RESTRICT,
        1,
        1,
        1,
        run_prepared_embed_restrict::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-contains",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_CONTAINS,
        1,
        1,
        1,
        run_prepared_embed_contains::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-frobenius",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_FROBENIUS,
        1,
        1,
        1,
        run_prepared_embed_frobenius::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-trace",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_TRACE,
        1,
        1,
        1,
        run_prepared_embed_trace::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-embed-norm",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_PREPARED_EMBED_NORM,
        1,
        1,
        1,
        run_prepared_embed_norm::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "prepared-iso-build",
        Scope::Setup,
        "gf2p8-aes->gf2p8-rs",
        API_PREPARED_ISO_BUILD,
        1,
        1,
        1,
        run_prepared_iso_build::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "prepared-iso-accessors",
        Scope::Query,
        "gf2p8-aes->gf2p8-rs",
        API_PREPARED_ISO_ACCESSORS,
        1,
        0,
        0,
        run_prepared_iso_accessors::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "prepared-iso-apply",
        Scope::Steady,
        "gf2p8-aes->gf2p8-rs",
        API_PREPARED_ISO_APPLY,
        1,
        1,
        1,
        run_prepared_iso_apply::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "prepared-iso-unapply",
        Scope::Steady,
        "gf2p8-aes->gf2p8-rs",
        API_PREPARED_ISO_UNAPPLY,
        1,
        1,
        1,
        run_prepared_iso_unapply::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "row-embed",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_MAP_ROW_EMBED,
        MAP_ROW_LANES,
        1,
        1,
        run_row_embed::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "row-trace",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_MAP_ROW_TRACE,
        MAP_ROW_LANES,
        1,
        1,
        run_row_trace::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "row-norm",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_MAP_ROW_NORM,
        MAP_ROW_LANES,
        1,
        1,
        run_row_norm::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "row-restrict",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_MAP_ROW_RESTRICT,
        MAP_ROW_LANES,
        1,
        1,
        run_row_restrict::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "row-contains",
        Scope::Steady,
        "gf2p8-aes->gf2p16-rijndael",
        API_MAP_ROW_CONTAINS,
        MAP_ROW_LANES,
        1,
        1,
        run_row_contains::<Gf2p8Aes, Gf2p16Rijndael>,
    );
    map_case(
        "row-apply",
        Scope::Steady,
        "gf2p8-aes->gf2p8-rs",
        API_MAP_ROW_APPLY,
        MAP_ROW_LANES,
        1,
        1,
        run_row_apply::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "row-unapply",
        Scope::Steady,
        "gf2p8-aes->gf2p8-rs",
        API_MAP_ROW_UNAPPLY,
        MAP_ROW_LANES,
        1,
        1,
        run_row_unapply::<Gf2p8Aes, Gf2p8Rs>,
    );
    map_case(
        "row-frobenius",
        Scope::Steady,
        "gf2p16-rijndael",
        API_MAP_ROW_FROBENIUS,
        MAP_ROW_LANES,
        1,
        1,
        run_row_frobenius,
    );
    map_case(
        "row-frobenius-assign",
        Scope::Steady,
        "m31",
        API_MAP_ROW_FROBENIUS_ASSIGN,
        MAP_ROW_LANES,
        1,
        1,
        run_row_frobenius_assign,
    );
    map_case(
        "bits-embed",
        Scope::Steady,
        "gf2p1-poly->gf2p16-rijndael",
        API_MAP_BITS_EMBED,
        MAP_BITS,
        1,
        1,
        run_bits_embed::<crate::fields::Gf2p1Poly, Gf2p16Rijndael>,
    );
    map_case(
        "bits-trace",
        Scope::Steady,
        "gf2p1-poly->gf2p16-rijndael",
        API_MAP_BITS_TRACE,
        MAP_BITS,
        1,
        1,
        run_bits_trace::<crate::fields::Gf2p1Poly, Gf2p16Rijndael>,
    );
    map_case(
        "bits-norm",
        Scope::Steady,
        "gf2p1-poly->gf2p16-rijndael",
        API_MAP_BITS_NORM,
        MAP_BITS,
        1,
        1,
        run_bits_norm::<crate::fields::Gf2p1Poly, Gf2p16Rijndael>,
    );
    map_case(
        "bits-restrict",
        Scope::Steady,
        "gf2p1-poly->gf2p16-rijndael",
        API_MAP_BITS_RESTRICT,
        MAP_BITS,
        1,
        1,
        run_bits_restrict::<crate::fields::Gf2p1Poly, Gf2p16Rijndael>,
    );
    map_case(
        "bits-contains",
        Scope::Steady,
        "gf2p1-poly->gf2p16-rijndael",
        API_MAP_BITS_CONTAINS,
        MAP_BITS,
        1,
        1,
        run_bits_contains::<crate::fields::Gf2p1Poly, Gf2p16Rijndael>,
    );
}
