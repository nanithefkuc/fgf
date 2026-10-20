//! Separate timed cases for stable public callables whose timed bodies no
//! other panel carries: the direct isomorphism constructors and the
//! generator accessor, the scalar description identity and structure
//! comparator, the prepared-collection accessors, and the one-shot
//! tuple-terms matrix overwrite over every bulk-kernel presentation.
//!
//! Map, description, and collection work runs scalar arithmetic and reports
//! the `scalar` backend; each accessor case times one call over a container
//! built outside the timer. The one-shot matrix case runs 4 KiB rows with
//! four sources and three outputs; coefficient preparation and the
//! independent per-lane scalar oracle stay outside the timer, and every
//! timed call overwrites a nonzero-seeded destination.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::ops::{CoeffMatrix, CoeffVec};
use fgf::{Elem, Embedding, FieldBuffer, FieldKernels, Isomorphism, ops};

use crate::fields::{self, Gf2p8Aes, Gf2p8Rs, Gf2p16Rijndael, for_each_bulk_field};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// Fixed row size of the one-shot matrix overwrite, rounded down to whole
/// elements.
const ROW_TARGET: usize = 4096;
/// Sources feeding one bulk matrix call.
const SOURCES: usize = 4;
/// Output rows of one bulk matrix call.
const OUTPUTS: usize = 3;
/// Prepared coefficients behind one collection accessor case.
const VIEW_LEN: usize = 8;

const API_ISO_NEW: &[&str] = &["fgf::Isomorphism::new"];
const API_ISO_FROM_EMBEDDING: &[&str] = &["fgf::Isomorphism::from_embedding"];
const API_GENERATOR_COUNT: &[&str] = &["fgf::Embedding::generator_count"];
const API_DESCRIPTION_ONE: &[&str] = &[
    "fgf::BinaryField::DESCRIPTION",
    "fgf::BinaryDescription::one",
];
const API_DESCRIPTION_SAME_STRUCTURE: &[&str] = &[
    "fgf::BinaryField::DESCRIPTION",
    "fgf::BinaryDescription::same_structure",
];
const API_COEFFVEC_IS_EMPTY: &[&str] = &["fgf::ops::CoeffVec::is_empty"];
const API_COEFFVEC_GET: &[&str] = &["fgf::ops::CoeffVec::get"];
const API_COEFFVEC_VALUES: &[&str] = &["fgf::ops::CoeffVec::values"];
const API_COEFFVECREF_VALUES: &[&str] = &["fgf::ops::CoeffVecRef::values"];
const API_COEFFMATRIX_OUTPUT_COUNT: &[&str] = &["fgf::ops::CoeffMatrix::output_count"];
const API_COEFFMATRIX_IS_EMPTY: &[&str] = &["fgf::ops::CoeffMatrix::is_empty"];
const API_COEFFMATRIX_VALUES: &[&str] = &["fgf::ops::CoeffMatrix::values"];
const API_MATRIX_INTO_ONE_SHOT: &[&str] = &["fgf::ops::mul_into_matrix"];

fn row_bytes<F: FieldBuffer>() -> usize {
    ROW_TARGET - ROW_TARGET % F::BYTES
}

/// The scalar backend name every map, description, and collection record
/// reports.
fn scalar_backend() -> String {
    fields::scalar_backend_name()
}

/// A deterministic element of `S` from the case seed.
fn sample<S: Field>(case: &str) -> Elem<S> {
    let mut seed = seed_for(case);
    let index = u128::from(fixture::lcg(&mut seed)) % S::ORDER.max(1);
    <Elem<S> as FieldElem>::from_index(index).expect("index below ORDER")
}

/// Deterministic coefficients rejecting zero and, above GF(2), one.
fn nontrivial_coeffs<F: Field>(label: &str, count: usize) -> Vec<Elem<F>> {
    use FieldElem as _;
    let mut seed = seed_for(label);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let candidate = fixture::elements::<F>(1, &mut seed).remove(0);
        if candidate.is_zero() {
            continue;
        }
        if F::ORDER > 2 && candidate == <Elem<F> as FieldElem>::ONE {
            continue;
        }
        out.push(candidate);
    }
    out
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

/// One registered map case on the canonical AES-to-RS pair.
fn add_map_case(
    op: &'static str,
    scope: Scope,
    api: &'static [&'static str],
    lanes: usize,
    sources: usize,
    destinations: usize,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case: format!("maps/direct-{op}/gf2p8-aes->gf2p8-rs"),
        api,
        field: "gf2p8-aes->gf2p8-rs".to_string(),
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

/// One registered query-scope case with zero logical bytes.
fn add_query_case(
    case: String,
    field: &str,
    operation: &'static str,
    api: &'static [&'static str],
    lanes: usize,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case,
        api,
        field: field.to_string(),
        operation: operation.to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run,
    });
}

fn run_iso_new<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    {
        let iso = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
        let embedding = Embedding::<S, T>::new().expect("canonical embedding exists");
        let x = sample::<S>(&case.case);
        let y = sample::<S>(&format!("{}/probe", case.case));
        assert_eq!(
            iso.apply(x),
            embedding.embed(x),
            "{}: forward matches the canonical embedding",
            case.case
        );
        assert_eq!(
            iso.unapply(iso.apply(x)),
            x,
            "{}: reverse round trip",
            case.case
        );
        assert_eq!(
            iso.apply(x.add(y)),
            iso.apply(x).add(iso.apply(y)),
            "{}: additive homomorphism",
            case.case
        );
    }
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        let iso = Isomorphism::<S, T>::new().expect("canonical isomorphism exists");
        fixture::observe(iso.to_images().len());
    });
}

fn run_iso_from_embedding<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let embedding = Embedding::<S, T>::new().expect("canonical embedding exists");
    {
        let iso = Isomorphism::<S, T>::from_embedding(embedding)
            .expect("the canonical embedding yields an isomorphism");
        let x = sample::<S>(&case.case);
        assert_eq!(
            iso.apply(x),
            embedding.embed(x),
            "{}: map agrees with its embedding",
            case.case
        );
        assert_eq!(iso.unapply(iso.apply(x)), x, "{}: round trip", case.case);
    }
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        let iso = Isomorphism::<S, T>::from_embedding(black_box(embedding));
        fixture::observe(iso.map_or(0, |m| m.to_images().len()));
    });
}

fn run_generator_count<S, T>(case: &Case, ctx: &CaseCtx)
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let embedding = Embedding::<S, T>::new().expect("canonical embedding exists");
    assert_eq!(
        Embedding::<S, T>::generator_count(),
        embedding.to_images().len(),
        "{}: generator count matches the canonical image count",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(Embedding::<S, T>::generator_count());
    });
}

fn run_description_one(case: &Case, ctx: &CaseCtx) {
    let description = <Gf2p8Aes as fgf::BinaryField>::DESCRIPTION;
    assert_eq!(
        description.one(),
        u64::from(<Elem<Gf2p8Aes> as FieldElem>::ONE.to_raw()),
        "{}: description one matches the field's raw one encoding",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(description).one());
    });
}

fn run_description_same_structure(case: &Case, ctx: &CaseCtx) {
    let description = <Gf2p8Aes as fgf::BinaryField>::DESCRIPTION;
    let different = <Gf2p8Rs as fgf::BinaryField>::DESCRIPTION;
    assert!(
        description.same_structure(description),
        "{}: a description agrees with itself",
        case.case
    );
    assert!(
        !description.same_structure(different),
        "{}: different defining polynomials are different structures",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(description).same_structure(black_box(different)));
    });
}

fn run_coeffvec_is_empty(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, VIEW_LEN);
    let prepared = CoeffVec::<Gf2p16Rijndael>::new(&cs);
    assert!(
        !prepared.is_empty(),
        "{}: prepared vector is nonempty",
        case.case
    );
    assert!(
        CoeffVec::<Gf2p16Rijndael>::new(&[]).is_empty(),
        "{}: empty construction is empty",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(&prepared).is_empty());
    });
}

fn run_coeffvec_get(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, VIEW_LEN);
    let prepared = CoeffVec::<Gf2p16Rijndael>::new(&cs);
    for (index, expected) in cs.iter().enumerate() {
        assert_eq!(
            prepared.get(index).map(|c| c.value()),
            Some(*expected),
            "{}: entry {index}",
            case.case
        );
    }
    assert!(
        prepared.get(VIEW_LEN).is_none(),
        "{}: out-of-range read",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(&prepared).get(black_box(VIEW_LEN / 2)));
    });
}

fn run_coeffvec_values(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, VIEW_LEN);
    let prepared = CoeffVec::<Gf2p16Rijndael>::new(&cs);
    {
        let collected: Vec<Elem<Gf2p16Rijndael>> = prepared.values().collect();
        assert_eq!(
            collected, cs,
            "{}: values reproduce the coefficients in order",
            case.case
        );
    }
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        let _ = fixture::observe(black_box(&prepared).values());
    });
}

fn run_coeffvecref_values(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, VIEW_LEN);
    let prepared = CoeffVec::<Gf2p16Rijndael>::new(&cs);
    let view = prepared.as_ref();
    {
        let collected: Vec<Elem<Gf2p16Rijndael>> = view.values().collect();
        assert_eq!(
            collected, cs,
            "{}: view values reproduce the coefficients in order",
            case.case
        );
    }
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        let _ = fixture::observe(black_box(view).values());
    });
}

fn run_coeffmatrix_output_count(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, SOURCES * OUTPUTS);
    let matrix = CoeffMatrix::<Gf2p16Rijndael>::from_source_major(SOURCES, OUTPUTS, &cs);
    assert_eq!(matrix.output_count(), OUTPUTS, "{}: output rows", case.case);
    assert_eq!(
        matrix.source_count(),
        SOURCES,
        "{}: source terms",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(&matrix).output_count());
    });
}

fn run_coeffmatrix_is_empty(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, SOURCES * OUTPUTS);
    let matrix = CoeffMatrix::<Gf2p16Rijndael>::from_source_major(SOURCES, OUTPUTS, &cs);
    assert!(
        !matrix.is_empty(),
        "{}: prepared matrix is nonempty",
        case.case
    );
    assert!(
        CoeffMatrix::<Gf2p16Rijndael>::from_source_major(0, 0, &[]).is_empty(),
        "{}: empty construction is empty",
        case.case
    );
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        fixture::observe(black_box(&matrix).is_empty());
    });
}

fn run_coeffmatrix_values(case: &Case, ctx: &CaseCtx) {
    let cs = nontrivial_coeffs::<Gf2p16Rijndael>(&case.case, SOURCES * OUTPUTS);
    let matrix = CoeffMatrix::<Gf2p16Rijndael>::from_source_major(SOURCES, OUTPUTS, &cs);
    {
        let collected: Vec<Elem<Gf2p16Rijndael>> = matrix.values().collect();
        assert_eq!(
            collected, cs,
            "{}: values reproduce the source-major order",
            case.case
        );
    }
    let backend = scalar_backend();
    registry::time(case, ctx, &backend, || {
        let _ = fixture::observe(black_box(&matrix).values());
    });
}

fn register_matrix_into_one_shot<F>(label: &'static str)
where
    F: FieldBuffer + FieldKernels,
{
    let row = row_bytes::<F>();
    let traffic = row * SOURCES * OUTPUTS;
    registry::add(Case {
        case: format!("bulk/matrix-into-one-shot/{label}"),
        api: API_MATRIX_INTO_ONE_SHOT,
        field: label.to_string(),
        operation: "matrix-into-one-shot".to_string(),
        scope: Scope::Steady,
        bytes: traffic,
        lanes: traffic / F::BYTES,
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
                .map(|t| nontrivial_coeffs::<F>(&format!("{}/source{t}", case.case), OUTPUTS))
                .collect();
            let terms: Vec<(&[Elem<F>], &[u8])> = coeff_sets
                .iter()
                .zip(&refs)
                .map(|(set, src)| (set.as_slice(), *src))
                .collect();
            let mut dst = fixture::packed::<F>(lanes * OUTPUTS, fixture::lcg(&mut seed));
            {
                let expected = overwrite_oracle::<F>(OUTPUTS, lanes, &coeff_sets, &refs);
                let mut fresh = fixture::packed::<F>(lanes * OUTPUTS, fixture::lcg(&mut seed));
                assert!(
                    fresh.iter().zip(&expected).any(|(a, b)| a != b),
                    "{}: seeded destination must differ from the oracle result",
                    case.case
                );
                ops::mul_into_matrix::<F>(&mut fresh, row, OUTPUTS, &terms);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_into_matrix::<F>(black_box(&mut dst), row, OUTPUTS, black_box(&terms));
                fixture::observe(dst.len());
            });
        },
    });
}

/// Register the missing-call panel: one separate timed case per stable
/// callable whose timed body no other panel carries.
pub fn register() {
    add_map_case(
        "new",
        Scope::Setup,
        API_ISO_NEW,
        1,
        1,
        1,
        run_iso_new::<Gf2p8Aes, Gf2p8Rs>,
    );
    add_map_case(
        "from_embedding",
        Scope::Setup,
        API_ISO_FROM_EMBEDDING,
        1,
        1,
        1,
        run_iso_from_embedding::<Gf2p8Aes, Gf2p8Rs>,
    );
    add_map_case(
        "generator_count",
        Scope::Query,
        API_GENERATOR_COUNT,
        1,
        0,
        0,
        run_generator_count::<Gf2p8Aes, Gf2p8Rs>,
    );

    add_query_case(
        "scalar/description-one/gf2p8-aes".to_string(),
        "gf2p8-aes",
        "description one",
        API_DESCRIPTION_ONE,
        0,
        run_description_one,
    );
    add_query_case(
        "scalar/description-same_structure/gf2p8-aes".to_string(),
        "gf2p8-aes",
        "description structure comparison",
        API_DESCRIPTION_SAME_STRUCTURE,
        0,
        run_description_same_structure,
    );

    add_query_case(
        "scalar/collection-coeffvec-is_empty/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient vector emptiness",
        API_COEFFVEC_IS_EMPTY,
        VIEW_LEN,
        run_coeffvec_is_empty,
    );
    add_query_case(
        "scalar/collection-coeffvec-get/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient vector indexed read",
        API_COEFFVEC_GET,
        VIEW_LEN,
        run_coeffvec_get,
    );
    add_query_case(
        "scalar/collection-coeffvec-values/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient vector value iterator",
        API_COEFFVEC_VALUES,
        VIEW_LEN,
        run_coeffvec_values,
    );
    add_query_case(
        "scalar/collection-coeffvecref-values/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "borrowed coefficient vector value iterator",
        API_COEFFVECREF_VALUES,
        VIEW_LEN,
        run_coeffvecref_values,
    );
    add_query_case(
        "scalar/collection-coeffmatrix-output_count/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient matrix output count",
        API_COEFFMATRIX_OUTPUT_COUNT,
        SOURCES * OUTPUTS,
        run_coeffmatrix_output_count,
    );
    add_query_case(
        "scalar/collection-coeffmatrix-is_empty/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient matrix emptiness",
        API_COEFFMATRIX_IS_EMPTY,
        SOURCES * OUTPUTS,
        run_coeffmatrix_is_empty,
    );
    add_query_case(
        "scalar/collection-coeffmatrix-values/gf2p16-rijndael".to_string(),
        "gf2p16-rijndael",
        "coefficient matrix value iterator",
        API_COEFFMATRIX_VALUES,
        SOURCES * OUTPUTS,
        run_coeffmatrix_values,
    );

    macro_rules! bulk {
        ($label:literal, $F:ty) => {
            register_matrix_into_one_shot::<$F>($label);
        };
    }
    for_each_bulk_field!(bulk);
}
