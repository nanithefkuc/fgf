//! Multi-row operation surfaces: scatter, gather, and matrix shapes in
//! one-shot and prepared forms, the offset row and offset source routes,
//! coefficient-container preparation and accessors, and the zero-coefficient
//! and empty-input special routes.
//!
//! Geometries are the production erasure-code shapes — eight rows of about
//! four KiB, eight sources into the matrix shapes, strided row starts for
//! the offset route — with oracles recomputed per lane from independent
//! `Elem` arithmetic on small fixtures. Container preparation is timed as
//! setup-scope records; accessor reads are query-scope records with the
//! prepared containers black-boxed so no dynamic dispatch enters trivial
//! timings.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::ops::{CoeffMatrix, CoeffVec};
use fgf::{Elem, FieldBuffer, ops};

use crate::fields::{self, for_each_bulk_field};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// Rows per multi-row shape.
const NROWS: usize = 8;
/// Sources feeding the matrix shapes.
const NSOURCES: usize = 8;
/// Target bytes of one row, rounded to whole elements.
const ROW_TARGET: usize = 4096;
/// Lanes in one oracle row.
const ORACLE_ROW_LANES: usize = 4;

fn row_bytes<F: FieldBuffer + fgf::FieldKernels>() -> usize {
    ROW_TARGET - ROW_TARGET % F::BYTES
}

fn oracle_row_bytes<F: FieldBuffer + fgf::FieldKernels>() -> usize {
    ORACLE_ROW_LANES * F::BYTES
}

const API_SCATTER: &[&str] = &["fgf::ops::mul_add_scatter"];
const API_SCATTER_WITH: &[&str] = &[
    "fgf::ops::mul_add_scatter_with",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_GATHER: &[&str] = &["fgf::ops::mul_add_gather"];
const API_GATHER_WITH: &[&str] = &[
    "fgf::ops::mul_add_gather_with",
    "fgf::ops::CoeffVec::new",
    "fgf::ops::CoeffVec::as_ref",
];
const API_MATRIX: &[&str] = &["fgf::ops::mul_add_matrix"];
const API_MATRIX_WITH: &[&str] = &[
    "fgf::ops::mul_add_matrix_with",
    "fgf::ops::CoeffMatrix::from_source_major",
];
const API_MATRIX_AT: &[&str] = &["fgf::ops::mul_add_matrix_at"];
const API_GATHER_OFFSETS: &[&str] = &["fgf::ops::add_gather_offsets"];
const API_COEFF_NEW: &[&str] = &["fgf::ops::Coeff::new"];
const API_COEFFVEC_NEW: &[&str] = &["fgf::ops::CoeffVec::new"];
const API_COEFFMATRIX_NEW: &[&str] = &["fgf::ops::CoeffMatrix::from_source_major"];
const API_COEFF_ACCESSORS: &[&str] = &["fgf::ops::PreparedCoefficient::value"];
const API_COEFFVEC_ACCESSORS: &[&str] = &[
    "fgf::ops::CoeffVec::len",
    "fgf::ops::CoeffVec::is_empty",
    "fgf::ops::CoeffVec::get",
    "fgf::ops::CoeffVec::coeffs",
    "fgf::ops::CoeffVec::values",
    "fgf::ops::CoeffVecRef::into_coeffs",
    "fgf::ops::CoeffVecRef::values",
    "fgf::ops::CoeffRef::value",
];
const API_COEFFMATRIX_ACCESSORS: &[&str] = &[
    "fgf::ops::CoeffMatrix::source_count",
    "fgf::ops::CoeffMatrix::output_count",
    "fgf::ops::CoeffMatrix::len",
    "fgf::ops::CoeffMatrix::is_empty",
    "fgf::ops::CoeffMatrix::get_at",
    "fgf::ops::CoeffMatrix::source",
    "fgf::ops::CoeffMatrix::values",
];
const API_ZERO_ROUTES: &[&str] = &[
    "fgf::ops::mul_add_scatter",
    "fgf::ops::mul_add_gather",
    "fgf::ops::mul_into_gather",
];
const API_EMPTY_ROUTES: &[&str] = &[
    "fgf::ops::mul_add_matrix",
    "fgf::ops::mul_into_matrix",
    "fgf::ops::mul_add_gather",
];

/// Deterministic coefficients excluding shortcut values where possible.
fn coeffs<F: Field>(label: &str, count: usize) -> Vec<Elem<F>> {
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

/// Schoolbook `rows[j] += coeffs[j] * src`.
fn scatter_oracle<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    let src_lanes = fixture::lanes::<F>(src);
    for (index, c) in coeffs.iter().enumerate() {
        let row = &mut rows[index * row_len..(index + 1) * row_len];
        let mut lanes = fixture::lanes::<F>(row);
        assert_eq!(lanes.len(), src_lanes.len(), "{case}: row lane count");
        for (lane, s) in lanes.iter_mut().zip(&src_lanes) {
            *lane += *c * *s;
        }
        fixture::write_lanes::<F>(row, &lanes);
    }
}

/// Schoolbook `dst += sum coeffs[i] * srcs[i]`.
fn gather_oracle<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    dst: &mut [u8],
    coeffs: &[Elem<F>],
    srcs: &[&[u8]],
) {
    for (c, s) in coeffs.iter().zip(srcs) {
        let mut lanes = fixture::lanes::<F>(dst);
        let src_lanes = fixture::lanes::<F>(s);
        assert_eq!(lanes.len(), src_lanes.len(), "{case}: gather lane count");
        for (lane, value) in lanes.iter_mut().zip(&src_lanes) {
            *lane += *c * *value;
        }
        fixture::write_lanes::<F>(dst, &lanes);
    }
}

#[allow(clippy::too_many_arguments)] // Fields mirror the measurement record.
fn spread_case<F: FieldBuffer + fgf::FieldKernels>(
    op: &'static str,
    api: &'static [&'static str],
    label: &'static str,
    bytes: usize,
    sources: usize,
    destinations: usize,
    scope: Scope,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case: format!("spread/{op}/{label}"),
        api,
        field: label.to_string(),
        operation: op.to_string(),
        scope,
        bytes,
        lanes: bytes / F::BYTES,
        sources,
        destinations,
        control: false,
        phase: None,
        run,
    });
}

fn register_scatter<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS;
    spread_case::<F>(
        "scatter",
        API_SCATTER,
        label,
        traffic,
        1,
        NROWS,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            // Oracle: same operation on a small geometry.
            {
                let oracle_row = oracle_row_bytes::<F>();
                let mut rows = vec![0u8; oracle_row * NROWS];
                let src = fixture::packed::<F>(ORACLE_ROW_LANES, 0xD1);
                let cs = coeffs::<F>(&case.case, NROWS);
                scatter_oracle::<F>(&case.case, &mut rows, oracle_row, &cs, &src);
                let mut fresh = vec![0u8; oracle_row * NROWS];
                ops::mul_add_scatter::<F>(&mut fresh, oracle_row, &cs, &src);
                fixture::assert_bytes_equal(&case.case, &fresh, &rows);
            }
            let mut seed = seed_for(&case.case);
            let src = fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed));
            let mut rows =
                fixture::packed::<F>(row_len / F::BYTES * NROWS, fixture::lcg(&mut seed));
            let cs = coeffs::<F>(&case.case, NROWS);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_scatter::<F>(black_box(&mut rows), row_len, &cs, black_box(&src));
                fixture::observe(rows.len());
            });
        },
    );
}

fn register_scatter_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS;
    spread_case::<F>(
        "scatter-prepared",
        API_SCATTER_WITH,
        label,
        traffic,
        1,
        NROWS,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            let mut seed = seed_for(&case.case);
            let src = fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed));
            let mut rows =
                fixture::packed::<F>(row_len / F::BYTES * NROWS, fixture::lcg(&mut seed));
            let cs = coeffs::<F>(&case.case, NROWS);
            let prepared = CoeffVec::<F>::new(&cs);
            // The prepared form is the same arithmetic as the one-shot; the
            // small-geometry oracle pins the one-shot in `register_scatter`.
            let mut oracle_rows = vec![0u8; oracle_row_bytes::<F>() * NROWS];
            let oracle_src = fixture::packed::<F>(ORACLE_ROW_LANES, 0xD2);
            scatter_oracle::<F>(
                &case.case,
                &mut oracle_rows,
                oracle_row_bytes::<F>(),
                &cs,
                &oracle_src,
            );
            let mut fresh = vec![0u8; oracle_row_bytes::<F>() * NROWS];
            ops::mul_add_scatter_with::<F>(
                &mut fresh,
                oracle_row_bytes::<F>(),
                prepared.as_ref(),
                &oracle_src,
            );
            fixture::assert_bytes_equal(&case.case, &fresh, &oracle_rows);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_scatter_with::<F>(
                    black_box(&mut rows),
                    row_len,
                    prepared.as_ref(),
                    black_box(&src),
                );
                fixture::observe(rows.len());
            });
        },
    );
}

fn register_gather<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS;
    spread_case::<F>(
        "gather",
        API_GATHER,
        label,
        traffic,
        NROWS,
        1,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            {
                let oracle_row = oracle_row_bytes::<F>();
                let srcs: Vec<Vec<u8>> = (0..NROWS)
                    .map(|t| fixture::packed::<F>(ORACLE_ROW_LANES, 0xE0 + t as u64))
                    .collect();
                let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
                let cs = coeffs::<F>(&case.case, NROWS);
                let mut expected = vec![0u8; oracle_row];
                gather_oracle::<F>(&case.case, &mut expected, &cs, &refs);
                let mut fresh = vec![0u8; oracle_row];
                ops::mul_add_gather::<F>(&mut fresh, &cs, &refs);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let mut seed = seed_for(&case.case);
            let srcs: Vec<Vec<u8>> = (0..NROWS)
                .map(|_| fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed)))
                .collect();
            let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
            let mut dst = fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed));
            let cs = coeffs::<F>(&case.case, NROWS);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_gather::<F>(black_box(&mut dst), &cs, black_box(&refs));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_gather_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS;
    spread_case::<F>(
        "gather-prepared",
        API_GATHER_WITH,
        label,
        traffic,
        NROWS,
        1,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            let mut seed = seed_for(&case.case);
            let srcs: Vec<Vec<u8>> = (0..NROWS)
                .map(|_| fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed)))
                .collect();
            let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
            let mut dst = fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed));
            let cs = coeffs::<F>(&case.case, NROWS);
            let prepared = CoeffVec::<F>::new(&cs);
            {
                let oracle_row = oracle_row_bytes::<F>();
                let oracle_srcs: Vec<Vec<u8>> = (0..NROWS)
                    .map(|t| fixture::packed::<F>(ORACLE_ROW_LANES, 0xE8 + t as u64))
                    .collect();
                let oracle_refs: Vec<&[u8]> = oracle_srcs.iter().map(Vec::as_slice).collect();
                let mut expected = vec![0u8; oracle_row];
                gather_oracle::<F>(&case.case, &mut expected, &cs, &oracle_refs);
                let mut fresh = vec![0u8; oracle_row];
                ops::mul_add_gather_with::<F>(&mut fresh, prepared.as_ref(), &oracle_refs);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_gather_with::<F>(
                    black_box(&mut dst),
                    prepared.as_ref(),
                    black_box(&refs),
                );
                fixture::observe(dst.len());
            });
        },
    );
}

/// The shared matrix fixture: coefficient vector per source, one packed row
/// per source, one packed destination region of `NROWS` rows.
#[allow(clippy::type_complexity)] // Source rows, coefficient columns, destination rows.
fn matrix_fixtures<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    row_len: usize,
    seed: u64,
) -> (Vec<Vec<u8>>, Vec<Vec<Elem<F>>>, Vec<u8>) {
    let mut state = seed | 1;
    let sources: Vec<Vec<u8>> = (0..NSOURCES)
        .map(|_| fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut state)))
        .collect();
    let coeff_sets: Vec<Vec<Elem<F>>> = (0..NSOURCES)
        .map(|t| coeffs::<F>(&format!("{case}/source{t}"), NROWS))
        .collect();
    let dst = fixture::packed::<F>(row_len / F::BYTES * NROWS, fixture::lcg(&mut state));
    (sources, coeff_sets, dst)
}

/// Schoolbook `rows[j] += sum_t coeff_sets[t][j] * sources[t]`.
fn matrix_oracle<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    rows: &mut [u8],
    row_len: usize,
    coeff_sets: &[Vec<Elem<F>>],
    sources: &[&[u8]],
) {
    for (set, src) in coeff_sets.iter().zip(sources) {
        for (index, c) in set.iter().enumerate() {
            let row = &mut rows[index * row_len..(index + 1) * row_len];
            let mut lanes = fixture::lanes::<F>(row);
            let src_lanes = fixture::lanes::<F>(src);
            for (lane, value) in lanes.iter_mut().zip(&src_lanes) {
                *lane += *c * *value;
            }
            fixture::write_lanes::<F>(row, &lanes);
        }
    }
    let _ = case;
}

fn register_matrix<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS * NSOURCES;
    spread_case::<F>(
        "matrix",
        API_MATRIX,
        label,
        traffic,
        NSOURCES,
        NROWS,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            {
                let oracle_row = oracle_row_bytes::<F>();
                let (oracle_sources, oracle_sets, init) =
                    matrix_fixtures::<F>(&case.case, oracle_row, 0xF0);
                let oracle_refs: Vec<&[u8]> = oracle_sources.iter().map(Vec::as_slice).collect();
                let mut expected = init.clone();
                matrix_oracle::<F>(
                    &case.case,
                    &mut expected,
                    oracle_row,
                    &oracle_sets,
                    &oracle_refs,
                );
                let oracle_terms: Vec<(&[Elem<F>], &[u8])> = oracle_sets
                    .iter()
                    .zip(&oracle_sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut fresh = init;
                ops::mul_add_matrix::<F>(&mut fresh, oracle_row, NROWS, &oracle_terms);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let (sources, coeff_sets, mut rows) =
                matrix_fixtures::<F>(&case.case, row_len, seed_for(&case.case));
            let terms: Vec<(&[Elem<F>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_matrix::<F>(black_box(&mut rows), row_len, NROWS, black_box(&terms));
                fixture::observe(rows.len());
            });
        },
    );
}

fn register_matrix_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS * NSOURCES;
    spread_case::<F>(
        "matrix-prepared",
        API_MATRIX_WITH,
        label,
        traffic,
        NSOURCES,
        NROWS,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            let (sources, coeff_sets, mut rows) =
                matrix_fixtures::<F>(&case.case, row_len, seed_for(&case.case));
            let flat: Vec<Elem<F>> = coeff_sets.iter().flatten().copied().collect();
            let matrix = CoeffMatrix::<F>::from_source_major(NSOURCES, NROWS, &flat);
            let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            {
                let oracle_row = oracle_row_bytes::<F>();
                let (oracle_sources, oracle_sets, init) =
                    matrix_fixtures::<F>(&case.case, oracle_row, 0xF8);
                let oracle_refs: Vec<&[u8]> = oracle_sources.iter().map(Vec::as_slice).collect();
                let mut expected = init.clone();
                matrix_oracle::<F>(
                    &case.case,
                    &mut expected,
                    oracle_row,
                    &oracle_sets,
                    &oracle_refs,
                );
                let oracle_flat: Vec<Elem<F>> = oracle_sets.iter().flatten().copied().collect();
                let oracle_matrix =
                    CoeffMatrix::<F>::from_source_major(NSOURCES, NROWS, &oracle_flat);
                let mut fresh = init;
                ops::mul_add_matrix_with::<F>(&mut fresh, oracle_row, &oracle_matrix, &oracle_refs);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_matrix_with::<F>(
                    black_box(&mut rows),
                    row_len,
                    &matrix,
                    black_box(&refs),
                );
                fixture::observe(rows.len());
            });
        },
    );
}

/// The offset-row route: destination rows at strided, non-contiguous starts.
fn register_matrix_at<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS * NSOURCES;
    spread_case::<F>(
        "matrix-at",
        API_MATRIX_AT,
        label,
        traffic,
        NSOURCES,
        NROWS,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            let stride = row_len + F::BYTES;
            let starts: Vec<usize> = (0..NROWS).map(|j| j * stride).collect();
            let dst_len = (NROWS - 1) * stride + row_len;
            {
                // Oracle at the small geometry, with the same inter-row gaps
                // so the offset route is exercised.
                let oracle_row = oracle_row_bytes::<F>();
                let oracle_stride = oracle_row + F::BYTES;
                let oracle_starts: Vec<usize> = (0..NROWS).map(|j| j * oracle_stride).collect();
                let oracle_len = (NROWS - 1) * oracle_stride + oracle_row;
                let (oracle_sources, oracle_sets, _) =
                    matrix_fixtures::<F>(&case.case, oracle_row, 0x114);
                let oracle_refs: Vec<&[u8]> = oracle_sources.iter().map(Vec::as_slice).collect();
                let mut expected = vec![0u8; oracle_len];
                for (j, &start) in oracle_starts.iter().enumerate() {
                    let row = &mut expected[start..start + oracle_row];
                    let mut lanes = fixture::lanes::<F>(row);
                    for (set, src) in oracle_sets.iter().zip(&oracle_refs) {
                        let src_lanes = fixture::lanes::<F>(src);
                        let c = set[j];
                        for (lane, value) in lanes.iter_mut().zip(&src_lanes) {
                            *lane += c * *value;
                        }
                    }
                    fixture::write_lanes::<F>(row, &lanes);
                }
                let mut fresh = vec![0u8; oracle_len];
                let oracle_terms: Vec<(&[Elem<F>], &[u8])> = oracle_sets
                    .iter()
                    .zip(&oracle_sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                ops::mul_add_matrix_at::<F>(&mut fresh, oracle_row, &oracle_starts, &oracle_terms);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let (sources, coeff_sets, mut dst) =
                matrix_fixtures::<F>(&case.case, row_len, seed_for(&case.case));
            let terms: Vec<(&[Elem<F>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            dst.resize(dst_len, 0);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_matrix_at::<F>(
                    black_box(&mut dst),
                    row_len,
                    &starts,
                    black_box(&terms),
                );
                fixture::observe(dst.len());
            });
        },
    );
}

/// The offset-source route: row-sized windows gathered out of one region.
fn register_gather_offsets<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let row_len = row_bytes::<F>();
    let traffic = row_len * NROWS;
    spread_case::<F>(
        "gather-offsets",
        API_GATHER_OFFSETS,
        label,
        traffic,
        NROWS,
        1,
        Scope::Steady,
        |case, ctx| {
            let row_len = row_bytes::<F>();
            let mut seed = seed_for(&case.case);
            let region = fixture::packed::<F>(
                (row_len + F::BYTES) * NROWS / F::BYTES,
                fixture::lcg(&mut seed),
            );
            let offsets: Vec<u32> = (0..NROWS)
                .map(|t| u32::try_from(t * (row_len + F::BYTES)).unwrap_or(u32::MAX))
                .collect();
            let mut dst = fixture::packed::<F>(row_len / F::BYTES, fixture::lcg(&mut seed));
            {
                // Oracle: unit-coefficient gather over the same windows.
                let oracle_row = oracle_row_bytes::<F>();
                let oracle_region = fixture::packed::<F>(ORACLE_ROW_LANES * NROWS + 1, 0x118);
                let oracle_offsets: Vec<u32> = (0..NROWS)
                    .map(|t| u32::try_from(t * oracle_row).unwrap_or(u32::MAX))
                    .collect();
                let mut expected = fixture::packed::<F>(ORACLE_ROW_LANES, 0x119);
                for &off in &oracle_offsets {
                    let off = off as usize;
                    let window = &oracle_region[off..off + oracle_row];
                    let mut lanes = fixture::lanes::<F>(&expected);
                    let window_lanes = fixture::lanes::<F>(window);
                    for (lane, value) in lanes.iter_mut().zip(&window_lanes) {
                        *lane += *value;
                    }
                    fixture::write_lanes::<F>(&mut expected, &lanes);
                }
                let mut fresh = fixture::packed::<F>(ORACLE_ROW_LANES, 0x119);
                ops::add_gather_offsets::<F>(&mut fresh, &oracle_region, &oracle_offsets);
                fixture::assert_bytes_equal(&case.case, &fresh, &expected);
            }
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::add_gather_offsets::<F>(
                    black_box(&mut dst),
                    black_box(&region),
                    black_box(&offsets),
                );
                fixture::observe(dst.len());
            });
        },
    );
}

/// Setup-scope records for the prepared containers, on representative
/// fields, plus the accessor/query records over the prepared values.
fn register_containers() {
    macro_rules! containers {
        ($label:literal, $F:ty) => {
            registry::add(Case {
                case: format!("setup/coeff-new/{}", $label),
                api: API_COEFF_NEW,
                field: $label.to_string(),
                operation: "coeff preparation".to_string(),
                scope: Scope::Setup,
                bytes: 0,
                lanes: 1,
                sources: 1,
                destinations: 1,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let c = coeffs::<$F>(&case.case, 1).remove(0);
                    let prepared = ops::Coeff::<$F>::new(c);
                    assert_eq!(ops::PreparedCoefficient::value(&prepared), c);
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        let p = ops::Coeff::<$F>::new(black_box(c));
                        fixture::observe(ops::PreparedCoefficient::value(&p).to_raw());
                    });
                },
            });
            registry::add(Case {
                case: format!("setup/coeffvec-new/{}", $label),
                api: API_COEFFVEC_NEW,
                field: $label.to_string(),
                operation: "coeff vector preparation".to_string(),
                scope: Scope::Setup,
                bytes: 0,
                lanes: NROWS,
                sources: NROWS,
                destinations: 1,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let cs = coeffs::<$F>(&case.case, NROWS);
                    let prepared = CoeffVec::<$F>::new(&cs);
                    assert_eq!(prepared.len(), NROWS);
                    assert!(!prepared.is_empty());
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        let p = CoeffVec::<$F>::new(black_box(&cs));
                        fixture::observe(p.len());
                    });
                },
            });
            registry::add(Case {
                case: format!("setup/coeffmatrix-new/{}", $label),
                api: API_COEFFMATRIX_NEW,
                field: $label.to_string(),
                operation: "coefficient matrix preparation".to_string(),
                scope: Scope::Setup,
                bytes: 0,
                lanes: NROWS * NSOURCES,
                sources: NSOURCES,
                destinations: NROWS,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let flat = coeffs::<$F>(&case.case, NROWS * NSOURCES);
                    let matrix = CoeffMatrix::<$F>::from_source_major(NSOURCES, NROWS, &flat);
                    assert_eq!(matrix.source_count(), NSOURCES);
                    assert_eq!(matrix.output_count(), NROWS);
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        let m =
                            CoeffMatrix::<$F>::from_source_major(NSOURCES, NROWS, black_box(&flat));
                        fixture::observe(m.len());
                    });
                },
            });
            registry::add(Case {
                case: format!("query/coeff-accessors/{}", $label),
                api: API_COEFF_ACCESSORS,
                field: $label.to_string(),
                operation: "coeff value read".to_string(),
                scope: Scope::Query,
                bytes: 0,
                lanes: 1,
                sources: 0,
                destinations: 0,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let c = coeffs::<$F>(&case.case, 1).remove(0);
                    let prepared = ops::Coeff::<$F>::new(c);
                    assert_eq!(prepared.value(), c);
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        fixture::observe(prepared.value().to_raw());
                    });
                },
            });
            registry::add(Case {
                case: format!("query/coeffvec-accessors/{}", $label),
                api: API_COEFFVEC_ACCESSORS,
                field: $label.to_string(),
                operation: "coeff vector reads".to_string(),
                scope: Scope::Query,
                bytes: 0,
                lanes: NROWS,
                sources: 0,
                destinations: 0,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let cs = coeffs::<$F>(&case.case, NROWS);
                    let prepared = CoeffVec::<$F>::new(&cs);
                    let view = prepared.as_ref();
                    assert_eq!(view.len(), NROWS);
                    assert!(!view.is_empty());
                    assert_eq!(view.get(0).map(|c| c.value()), Some(cs[0]));
                    assert!(view.get(NROWS).is_none());
                    assert_eq!(view.into_coeffs().count(), NROWS);
                    assert_eq!(prepared.coeffs().count(), NROWS);
                    assert_eq!(
                        prepared
                            .values()
                            .zip(cs.iter())
                            .filter(|(a, b)| a == *b)
                            .count(),
                        NROWS
                    );
                    assert_eq!(view.values().count(), NROWS);
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        let mut seen = 0usize;
                        for c in prepared.coeffs() {
                            seen += usize::from(c.value() == cs[seen.min(NROWS - 1)]);
                        }
                        let view = prepared.as_ref();
                        fixture::observe((view.len(), view.get(1).unwrap().value().to_raw(), seen));
                    });
                },
            });
            registry::add(Case {
                case: format!("query/coeffmatrix-accessors/{}", $label),
                api: API_COEFFMATRIX_ACCESSORS,
                field: $label.to_string(),
                operation: "coefficient matrix reads".to_string(),
                scope: Scope::Query,
                bytes: 0,
                lanes: NROWS * NSOURCES,
                sources: 0,
                destinations: 0,
                control: false,
                phase: None,
                run: |case, ctx| {
                    let flat = coeffs::<$F>(&case.case, NROWS * NSOURCES);
                    let matrix = CoeffMatrix::<$F>::from_source_major(NSOURCES, NROWS, &flat);
                    assert_eq!(
                        matrix.get_at(1, 2).map(|c| c.value()),
                        Some(flat[1 * NROWS + 2])
                    );
                    assert!(matrix.get_at(NSOURCES, 0).is_none());
                    assert_eq!(matrix.source(1).map(|v| v.len()), Some(NROWS));
                    assert!(matrix.source(NSOURCES).is_none());
                    assert_eq!(matrix.values().count(), NROWS * NSOURCES);
                    assert!(!matrix.is_empty());
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        let mut seen = 0usize;
                        for source in 0..NSOURCES {
                            if let Some(view) = matrix.source(source) {
                                for c in view.into_coeffs() {
                                    seen += usize::from(!c.value().is_zero());
                                }
                            }
                        }
                        fixture::observe((
                            matrix.source_count(),
                            matrix.get_at(0, 0).unwrap().value().to_raw(),
                            seen,
                        ));
                    });
                },
            });
        };
    }
    containers!("gf2p8-aes", crate::fields::Gf2p8Aes);
    containers!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
    containers!("gf2p16-fanpaar", crate::fields::Gf2p16FanPaar);
    containers!("m31", crate::fields::M31);
    containers!("gfmaxp", crate::fields::GfMaxPrime);
}

/// The multi-row special routes: all-zero coefficients skip work, a single
/// source degrades the overwrite gather to `mul_into`, and empty terms leave
/// the destination untouched (or zero-filled in the overwrite form).
fn register_special_routes() {
    macro_rules! specials {
        ($label:literal, $F:ty) => {
            let row_len = row_bytes::<$F>();
            let traffic = row_len * NROWS;

            spread_case::<$F>(
                "scatter-zero-coeffs",
                API_ZERO_ROUTES,
                $label,
                traffic,
                1,
                NROWS,
                Scope::Steady,
                |case, ctx| {
                    let row_len = row_bytes::<$F>();
                    let mut seed = seed_for(&case.case);
                    let src = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES,
                        fixture::lcg(&mut seed),
                    );
                    let mut rows = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES * NROWS,
                        fixture::lcg(&mut seed),
                    );
                    {
                        let zeros = vec![<Elem<$F> as FieldElem>::ZERO; NROWS];
                        let untouched = rows.clone();
                        ops::mul_add_scatter::<$F>(&mut rows, row_len, &zeros, &src);
                        assert_eq!(
                            rows, untouched,
                            "{}: zero scatter leaves rows untouched",
                            case.case
                        );
                    }
                    let zeros = vec![<Elem<$F> as FieldElem>::ZERO; NROWS];
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        ops::mul_add_scatter::<$F>(
                            black_box(&mut rows),
                            row_len,
                            &zeros,
                            black_box(&src),
                        );
                        fixture::observe(rows.len());
                    });
                },
            );

            spread_case::<$F>(
                "gather-single-source",
                API_ZERO_ROUTES,
                $label,
                traffic,
                1,
                1,
                Scope::Steady,
                |case, ctx| {
                    let row_len = row_bytes::<$F>();
                    let mut seed = seed_for(&case.case);
                    let src = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES,
                        fixture::lcg(&mut seed),
                    );
                    let cs = coeffs::<$F>(&case.case, 1);
                    {
                        let mut expected = vec![0u8; row_len];
                        ops::mul_into::<$F>(&mut expected, cs[0], &src);
                        let mut fresh = vec![0u8; row_len];
                        let src_ref: &[u8] = &src;
                        ops::mul_into_gather::<$F>(&mut fresh, &cs, &[src_ref]);
                        fixture::assert_bytes_equal(&case.case, &fresh, &expected);
                    }
                    let mut dst = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES,
                        fixture::lcg(&mut seed),
                    );
                    let backend = fields::backend_name::<$F>();
                    let src_ref: &[u8] = &src;
                    let srcs: &[&[u8]] = &[src_ref];
                    registry::time(case, ctx, &backend, || {
                        ops::mul_into_gather::<$F>(black_box(&mut dst), &cs, black_box(srcs));
                        fixture::observe(dst.len());
                    });
                },
            );

            spread_case::<$F>(
                "matrix-empty-terms",
                API_EMPTY_ROUTES,
                $label,
                0,
                0,
                NROWS,
                Scope::Steady,
                |case, ctx| {
                    let row_len = row_bytes::<$F>();
                    let mut seed = seed_for(&case.case);
                    let mut rows = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES * NROWS,
                        fixture::lcg(&mut seed),
                    );
                    {
                        let untouched = rows.clone();
                        let empty: [(&[Elem<$F>], &[u8]); 0] = [];
                        ops::mul_add_matrix::<$F>(&mut rows, row_len, NROWS, &empty);
                        assert_eq!(
                            rows, untouched,
                            "{}: empty accumulate leaves rows untouched",
                            case.case
                        );
                        ops::mul_into_matrix::<$F>(&mut rows, row_len, NROWS, &empty);
                        assert!(
                            rows.iter().all(|byte| *byte == 0),
                            "{}: empty overwrite fills zero rows",
                            case.case
                        );
                    }
                    let empty: [(&[Elem<$F>], &[u8]); 0] = [];
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        ops::mul_add_matrix::<$F>(
                            black_box(&mut rows),
                            row_len,
                            NROWS,
                            black_box(&empty),
                        );
                        fixture::observe(rows.len());
                    });
                },
            );

            spread_case::<$F>(
                "gather-empty-sources",
                API_EMPTY_ROUTES,
                $label,
                0,
                0,
                1,
                Scope::Steady,
                |case, ctx| {
                    let row_len = row_bytes::<$F>();
                    let mut seed = seed_for(&case.case);
                    let mut dst = fixture::packed::<$F>(
                        row_len / <$F as FieldBuffer>::BYTES,
                        fixture::lcg(&mut seed),
                    );
                    {
                        let untouched = dst.clone();
                        let cs: Vec<Elem<$F>> = Vec::new();
                        let refs: [&[u8]; 0] = [];
                        ops::mul_add_gather::<$F>(&mut dst, &cs, &refs);
                        assert_eq!(
                            dst, untouched,
                            "{}: empty gather leaves dst untouched",
                            case.case
                        );
                        ops::mul_into_gather::<$F>(&mut dst, &cs, &refs);
                        assert!(
                            dst.iter().all(|byte| *byte == 0),
                            "{}: empty overwrite gather fills zero",
                            case.case
                        );
                    }
                    let cs: Vec<Elem<$F>> = Vec::new();
                    let refs: [&[u8]; 0] = [];
                    let backend = fields::backend_name::<$F>();
                    registry::time(case, ctx, &backend, || {
                        ops::mul_add_gather::<$F>(black_box(&mut dst), &cs, black_box(&refs));
                        fixture::observe(dst.len());
                    });
                },
            );
        };
    }
    specials!("gf2p8-aes", crate::fields::Gf2p8Aes);
    specials!("m31", crate::fields::M31);
}

/// Register the multi-row panel over every bulk field.
pub fn register() {
    macro_rules! walk {
        ($label:literal, $F:ty) => {
            register_scatter::<$F>($label);
            register_scatter_prepared::<$F>($label);
            register_gather::<$F>($label);
            register_gather_prepared::<$F>($label);
            register_matrix::<$F>($label);
            register_matrix_prepared::<$F>($label);
            register_matrix_at::<$F>($label);
            register_gather_offsets::<$F>($label);
        };
    }
    for_each_bulk_field!(walk);
    register_containers();
    register_special_routes();
}
