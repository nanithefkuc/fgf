//! Single-row bulk operations over every bulk-capable presentation:
//! accumulate, overwrite, in-place, elementwise, and broadcast-scalar
//! operations in one-shot and prepared forms, at the resident geometry, with
//! short and nonresident sweeps and the zero/unit/empty special routes on
//! representative fields.
//!
//! Oracles are independent per-lane `Elem` arithmetic on small fixtures
//! outside every timed region; steady bodies operate on preallocated packed
//! buffers whose lanes stay canonical across repeated calls. Prepared-form
//! cases build their [`fgf::ops::Coeff`] outside the timed body; coefficient
//! preparation itself is a separate setup-scope case in `spread`.

use std::hint::black_box;

use fgf::field::{Field, FieldElem};
use fgf::{Elem, FieldBuffer, ops};

use crate::fields::{self, for_each_bulk_field};
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

/// Resident geometry: about 256 KiB, rounded down to a whole sixteen-element
/// row so `add_assign_rows` and every lane width divide it exactly.
const RESIDENT_TARGET: usize = 256 * 1024;
/// Short geometry: sixty-four bytes of packed lanes.
const SHORT_BYTES: usize = 64;
/// Streaming geometry: 128 MiB, larger than four times either measured LLC.
const NONRESIDENT_TARGET: usize = 128 * 1024 * 1024;
/// Lane count of the oracle fixtures.
const ORACLE_LANES: usize = 7;

fn resident_bytes<F: FieldBuffer + fgf::FieldKernels>() -> usize {
    let row = 16 * F::BYTES;
    RESIDENT_TARGET - RESIDENT_TARGET % row
}

fn nonresident_bytes<F: FieldBuffer + fgf::FieldKernels>() -> usize {
    let row = 16 * F::BYTES;
    NONRESIDENT_TARGET - NONRESIDENT_TARGET % row
}

fn short_bytes<F: FieldBuffer + fgf::FieldKernels>() -> usize {
    let row = 16 * F::BYTES;
    let bytes = SHORT_BYTES.max(row);
    bytes - bytes % row
}

const API_ADD_ASSIGN: &[&str] = &["fgf::ops::add_assign"];
const API_SUB_ASSIGN: &[&str] = &["fgf::ops::sub_assign"];
const API_ADD_ROWS: &[&str] = &["fgf::ops::add_assign_rows"];
const API_MUL_ADD: &[&str] = &["fgf::ops::mul_add"];
const API_MUL_ADD_WITH: &[&str] = &[
    "fgf::ops::mul_add_with",
    "fgf::ops::Coeff::new",
    "fgf::ops::PreparedCoefficient",
];
const API_MUL_INTO: &[&str] = &["fgf::ops::mul_into"];
const API_MUL_INTO_WITH: &[&str] = &["fgf::ops::mul_into_with", "fgf::ops::Coeff::new"];
const API_MUL_ASSIGN: &[&str] = &["fgf::ops::mul_assign"];
const API_MUL_ASSIGN_WITH: &[&str] = &["fgf::ops::mul_assign_with", "fgf::ops::Coeff::new"];
const API_ELEMENTWISE: &[&str] = &["fgf::ops::mul_elementwise"];
const API_ELEMENTWISE_ASSIGN: &[&str] = &["fgf::ops::mul_elementwise_assign"];
const API_ADD_SCALAR: &[&str] = &["fgf::ops::add_assign_scalar"];
const API_SUB_SCALAR: &[&str] = &["fgf::ops::sub_assign_scalar"];
const API_ZERO_UNIT_ROUTES: &[&str] = &[
    "fgf::ops::mul_add",
    "fgf::ops::mul_into",
    "fgf::ops::add_assign",
];
const API_EMPTY_ROUTES: &[&str] = &[
    "fgf::ops::add_assign",
    "fgf::ops::mul_add",
    "fgf::ops::mul_into",
    "fgf::ops::mul_assign",
];

/// Deterministic coefficient excluding shortcut values where possible.
fn coeff<F: Field>(label: &str) -> Elem<F> {
    if F::ORDER == 2 {
        return <Elem<F> as FieldElem>::ONE;
    }
    let mut seed = seed_for(label);
    loop {
        let candidate = fixture::element::<F>(&mut seed);
        if !candidate.is_zero() && !candidate.is_one() {
            return candidate;
        }
    }
}

/// Independent canonical destination and source buffers from fixed seeds.
fn packed_pair<F: FieldBuffer + fgf::FieldKernels>(lanes: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    (
        fixture::packed::<F>(lanes, seed),
        fixture::packed::<F>(lanes, seed.wrapping_add(1)),
    )
}

/// Run one bulk operation on a small fixture and compare against independent
/// per-lane arithmetic: `run` performs the operation, `combine` names the
/// lane function of the destination element and the source element.
fn verify_pair<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    run: impl FnOnce(&mut [u8], &[u8]),
    combine: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
) {
    let (mut dst, _) = packed_pair::<F>(ORACLE_LANES, 0xA1);
    let (src, _) = packed_pair::<F>(ORACLE_LANES, 0xB2);
    let d = fixture::lanes::<F>(&dst);
    let s = fixture::lanes::<F>(&src);
    run(&mut dst, &src);
    let expected_lanes: Vec<Elem<F>> = d.iter().zip(&s).map(|(a, b)| combine(*a, *b)).collect();
    let mut expected = vec![0u8; ORACLE_LANES * F::BYTES];
    fixture::write_lanes::<F>(&mut expected, &expected_lanes);
    fixture::assert_bytes_equal(case, &dst, &expected);
}

/// The unary variant: the operation touches only the destination.
fn verify_unary<F: FieldBuffer + fgf::FieldKernels>(
    case: &str,
    run: impl FnOnce(&mut [u8]),
    combine: impl Fn(Elem<F>) -> Elem<F>,
) {
    let (mut dst, _) = packed_pair::<F>(ORACLE_LANES, 0xC3);
    let d = fixture::lanes::<F>(&dst);
    run(&mut dst);
    let expected_lanes: Vec<Elem<F>> = d.iter().map(|a| combine(*a)).collect();
    let mut expected = vec![0u8; ORACLE_LANES * F::BYTES];
    fixture::write_lanes::<F>(&mut expected, &expected_lanes);
    fixture::assert_bytes_equal(case, &dst, &expected);
}

#[allow(clippy::too_many_arguments)] // Fields mirror the measurement record.
fn bulk_case<F: FieldBuffer + fgf::FieldKernels>(
    op: &'static str,
    api: &'static [&'static str],
    label: &'static str,
    size: Option<&'static str>,
    bytes: usize,
    lanes: usize,
    sources: usize,
    run: fn(&Case, &CaseCtx),
) {
    assert_eq!(lanes, bytes / F::BYTES, "bulk measurement geometry");
    let id = match size {
        Some(size) => format!("bulk/{op}/{size}/{label}"),
        None => format!("bulk/{op}/{label}"),
    };
    let unit = F::ORDER == 2;
    let no_op = op == "zero-coeff" || (unit && matches!(op, "mul-assign" | "mul-assign-prepared"));
    let operation = match (unit, op) {
        (_, "zero-coeff") => "mul-add (zero no-op)".to_string(),
        (true, "mul-assign" | "mul-assign-prepared") => format!("{op} (unit no-op)"),
        (true, "mul-into" | "mul-into-prepared") => format!("{op} (unit copy)"),
        (true, "mul-add" | "mul-add-prepared") => format!("{op} (unit add)"),
        _ => op.to_string(),
    };
    registry::add(Case {
        case: id,
        api,
        field: label.to_string(),
        operation,
        scope: Scope::Steady,
        bytes: if no_op { 0 } else { bytes },
        lanes,
        sources,
        destinations: 1,
        control: false,
        phase: None,
        run,
    });
}

fn register_add_assign<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "add-assign",
        API_ADD_ASSIGN,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            verify_pair::<F>(&case.case, |d, s| ops::add_assign::<F>(d, s), |a, b| a + b);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::add_assign::<F>(black_box(&mut dst), black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_sub_assign<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "sub-assign",
        API_SUB_ASSIGN,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            verify_pair::<F>(&case.case, |d, s| ops::sub_assign::<F>(d, s), |a, b| a - b);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::sub_assign::<F>(black_box(&mut dst), black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_add_rows<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "add-rows",
        API_ADD_ROWS,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let row_len = 16 * F::BYTES;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            let (mut got, input) = packed_pair::<F>(48, 0xadd);
            let expected: Vec<_> = fixture::lanes::<F>(&got)
                .into_iter()
                .zip(fixture::lanes::<F>(&input))
                .map(|(a, b)| a + b)
                .collect();
            ops::add_assign_rows::<F>(&mut got, row_len, &input);
            let mut encoded = vec![0; got.len()];
            fixture::write_lanes::<F>(&mut encoded, &expected);
            fixture::assert_bytes_equal(&case.case, &got, &encoded);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::add_assign_rows::<F>(black_box(&mut dst), row_len, black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_add<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-add",
        API_MUL_ADD,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            verify_pair::<F>(
                &case.case,
                |d, s| ops::mul_add::<F>(d, c, s),
                |a, b| a + c * b,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add::<F>(black_box(&mut dst), c, black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_add_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-add-prepared",
        API_MUL_ADD_WITH,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            let prepared = ops::Coeff::<F>::new(c);
            verify_pair::<F>(
                &case.case,
                |d, s| ops::mul_add_with::<F>(d, &prepared, s),
                |a, b| a + c * b,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_add_with::<F>(black_box(&mut dst), &prepared, black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_into<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-into",
        API_MUL_INTO,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            verify_pair::<F>(&case.case, |d, s| ops::mul_into::<F>(d, c, s), |_, b| c * b);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_into::<F>(black_box(&mut dst), c, black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_into_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-into-prepared",
        API_MUL_INTO_WITH,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            let prepared = ops::Coeff::<F>::new(c);
            verify_pair::<F>(
                &case.case,
                |d, s| ops::mul_into_with::<F>(d, &prepared, s),
                |_, b| c * b,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_into_with::<F>(black_box(&mut dst), &prepared, black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_assign<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-assign",
        API_MUL_ASSIGN,
        label,
        None,
        bytes,
        lanes,
        0,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, _) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            verify_unary::<F>(&case.case, |d| ops::mul_assign::<F>(d, c), |a| a * c);
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_assign::<F>(black_box(&mut dst), c);
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_mul_assign_prepared<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "mul-assign-prepared",
        API_MUL_ASSIGN_WITH,
        label,
        None,
        bytes,
        lanes,
        0,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, _) = packed_pair::<F>(lanes, seed_for(&case.case));
            let c = coeff::<F>(&case.case);
            let prepared = ops::Coeff::<F>::new(c);
            verify_unary::<F>(
                &case.case,
                |d| ops::mul_assign_with::<F>(d, &prepared),
                |a| a * c,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_assign_with::<F>(black_box(&mut dst), &prepared);
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_elementwise<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "elementwise",
        API_ELEMENTWISE,
        label,
        None,
        bytes,
        lanes,
        2,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, a) = packed_pair::<F>(lanes, seed_for(&case.case));
            let (b, _) = packed_pair::<F>(lanes, seed_for(&case.case) ^ 0x5EED);
            verify_pair::<F>(
                &case.case,
                |d, s| {
                    let left = d.to_vec();
                    ops::mul_elementwise::<F>(d, &left, s);
                },
                |x, y| x * y,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_elementwise::<F>(black_box(&mut dst), black_box(&a), black_box(&b));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_elementwise_assign<F: FieldBuffer + fgf::FieldKernels>(label: &'static str) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    bulk_case::<F>(
        "elementwise-assign",
        API_ELEMENTWISE_ASSIGN,
        label,
        None,
        bytes,
        lanes,
        1,
        |case, ctx| {
            let lanes = case.lanes;
            let (mut dst, src) = packed_pair::<F>(lanes, seed_for(&case.case));
            verify_pair::<F>(
                &case.case,
                |d, s| ops::mul_elementwise_assign::<F>(d, s),
                |a, b| a * b,
            );
            let backend = fields::backend_name::<F>();
            registry::time(case, ctx, &backend, || {
                ops::mul_elementwise_assign::<F>(black_box(&mut dst), black_box(&src));
                fixture::observe(dst.len());
            });
        },
    );
}

fn register_scalar_broadcast<F: FieldBuffer + fgf::FieldKernels>(
    label: &'static str,
    subtract: bool,
) {
    let bytes = resident_bytes::<F>();
    let lanes = bytes / F::BYTES;
    let (op, api): (&'static str, &'static [&'static str]) = if subtract {
        ("sub-scalar", API_SUB_SCALAR)
    } else {
        ("add-scalar", API_ADD_SCALAR)
    };
    bulk_case::<F>(op, api, label, None, bytes, lanes, 0, |case, ctx| {
        let lanes = case.lanes;
        let (mut dst, _) = packed_pair::<F>(lanes, seed_for(&case.case));
        let v = coeff::<F>(&case.case);
        if case.operation == "sub-scalar" {
            verify_unary::<F>(&case.case, |d| ops::sub_assign_scalar::<F>(d, v), |a| a - v);
        } else {
            verify_unary::<F>(&case.case, |d| ops::add_assign_scalar::<F>(d, v), |a| a + v);
        }
        let backend = fields::backend_name::<F>();
        registry::time(case, ctx, &backend, || {
            if case.operation == "sub-scalar" {
                ops::sub_assign_scalar::<F>(black_box(&mut dst), v);
            } else {
                ops::add_assign_scalar::<F>(black_box(&mut dst), v);
            }
            fixture::observe(dst.len());
        });
    });
}

/// Short and nonresident sweeps of the three traffic-dominant operations on
/// the routing-representative subset.
fn register_size_sweep() {
    macro_rules! sweep {
        ($label:literal, $F:ty) => {
            for (size, bytes) in [
                ("short", short_bytes::<$F>()),
                ("nonresident", nonresident_bytes::<$F>()),
            ] {
                let lanes = bytes / <$F as FieldBuffer>::BYTES;
                for op in ["add-assign", "mul-add", "mul-into"] {
                    match op {
                        "add-assign" => bulk_case::<$F>(
                            "add-assign",
                            API_ADD_ASSIGN,
                            $label,
                            Some(size),
                            bytes,
                            lanes,
                            1,
                            |case, ctx| {
                                let lanes = case.bytes / <$F as FieldBuffer>::BYTES;
                                let (mut dst, src) = packed_pair::<$F>(lanes, seed_for(&case.case));
                                verify_pair::<$F>(
                                    &case.case,
                                    |d, s| ops::add_assign::<$F>(d, s),
                                    |a, b| a + b,
                                );
                                let backend = fields::backend_name::<$F>();
                                registry::time(case, ctx, &backend, || {
                                    ops::add_assign::<$F>(black_box(&mut dst), black_box(&src));
                                    fixture::observe(dst.len());
                                });
                            },
                        ),
                        "mul-add" => bulk_case::<$F>(
                            "mul-add",
                            API_MUL_ADD,
                            $label,
                            Some(size),
                            bytes,
                            lanes,
                            1,
                            |case, ctx| {
                                let lanes = case.bytes / <$F as FieldBuffer>::BYTES;
                                let (mut dst, src) = packed_pair::<$F>(lanes, seed_for(&case.case));
                                let c = coeff::<$F>(&case.case);
                                verify_pair::<$F>(
                                    &case.case,
                                    |d, s| ops::mul_add::<$F>(d, c, s),
                                    |a, b| a + c * b,
                                );
                                let backend = fields::backend_name::<$F>();
                                registry::time(case, ctx, &backend, || {
                                    ops::mul_add::<$F>(black_box(&mut dst), c, black_box(&src));
                                    fixture::observe(dst.len());
                                });
                            },
                        ),
                        _ => bulk_case::<$F>(
                            "mul-into",
                            API_MUL_INTO,
                            $label,
                            Some(size),
                            bytes,
                            lanes,
                            1,
                            |case, ctx| {
                                let lanes = case.bytes / <$F as FieldBuffer>::BYTES;
                                let (mut dst, src) = packed_pair::<$F>(lanes, seed_for(&case.case));
                                let c = coeff::<$F>(&case.case);
                                verify_pair::<$F>(
                                    &case.case,
                                    |d, s| ops::mul_into::<$F>(d, c, s),
                                    |_, b| c * b,
                                );
                                let backend = fields::backend_name::<$F>();
                                registry::time(case, ctx, &backend, || {
                                    ops::mul_into::<$F>(black_box(&mut dst), c, black_box(&src));
                                    fixture::observe(dst.len());
                                });
                            },
                        ),
                    }
                }
            }
        };
    }
    sweep!("gf2p8-aes", crate::fields::Gf2p8Aes);
    sweep!("gf2p8-rs", crate::fields::Gf2p8Rs);
    sweep!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
    sweep!("m31", crate::fields::M31);
    sweep!("goldilocks", crate::fields::Gld);
    sweep!("gfmaxp", crate::fields::GfMaxPrime);
}

/// The zero-coefficient, unit-coefficient, and empty-buffer special routes:
/// the byte-preserving shortcuts the prime kernels take and the total no-op
/// paths every kernel shares.
fn register_special_routes() {
    macro_rules! specials {
        ($label:literal, $F:ty) => {
            let bytes = resident_bytes::<$F>();
            let lanes = bytes / <$F as FieldBuffer>::BYTES;

            bulk_case::<$F>(
                "zero-coeff",
                API_ZERO_UNIT_ROUTES,
                $label,
                None,
                bytes,
                lanes,
                1,
                |case, ctx| {
                    let lanes = case.lanes;
                    let (mut dst, src) = packed_pair::<$F>(lanes, seed_for(&case.case));
                    {
                        let zero = <Elem<$F> as FieldElem>::ZERO;
                        let untouched = dst.clone();
                        ops::mul_add::<$F>(&mut dst, zero, &src);
                        assert_eq!(
                            dst, untouched,
                            "{}: zero coefficient leaves dst untouched",
                            case.case
                        );
                        ops::mul_into::<$F>(&mut dst, zero, &src);
                        assert!(
                            dst.iter().all(|byte| *byte == 0),
                            "{}: zero overwrite fills zero lanes",
                            case.case
                        );
                    }
                    let backend = fields::backend_name::<$F>();
                    let c = <Elem<$F> as FieldElem>::ZERO;
                    registry::time(case, ctx, &backend, || {
                        ops::mul_add::<$F>(black_box(&mut dst), c, black_box(&src));
                        fixture::observe(dst.len());
                    });
                },
            );

            bulk_case::<$F>(
                "unit-coeff",
                API_ZERO_UNIT_ROUTES,
                $label,
                None,
                bytes,
                lanes,
                1,
                |case, ctx| {
                    let lanes = case.bytes / <$F as FieldBuffer>::BYTES;
                    let (mut dst, src) = packed_pair::<$F>(lanes, seed_for(&case.case));
                    {
                        let one = <Elem<$F> as FieldElem>::ONE;
                        let mut expected = dst.clone();
                        ops::add_assign::<$F>(&mut expected, &src);
                        ops::mul_add::<$F>(&mut dst, one, &src);
                        assert_eq!(dst, expected, "{}: unit accumulate equals add", case.case);
                        ops::mul_into::<$F>(&mut dst, one, &src);
                        assert_eq!(dst, src, "{}: unit overwrite copies source", case.case);
                    }
                    let backend = fields::backend_name::<$F>();
                    let c = <Elem<$F> as FieldElem>::ONE;
                    registry::time(case, ctx, &backend, || {
                        ops::mul_add::<$F>(black_box(&mut dst), c, black_box(&src));
                        fixture::observe(dst.len());
                    });
                },
            );

            bulk_case::<$F>(
                "empty-buffers",
                API_EMPTY_ROUTES,
                $label,
                None,
                0,
                0,
                0,
                |case, ctx| {
                    {
                        let mut dst: Vec<u8> = Vec::new();
                        let src: Vec<u8> = Vec::new();
                        ops::add_assign::<$F>(&mut dst, &src);
                        ops::mul_add::<$F>(&mut dst, <Elem<$F> as FieldElem>::ONE, &src);
                        ops::mul_into::<$F>(&mut dst, <Elem<$F> as FieldElem>::ONE, &src);
                        ops::mul_assign::<$F>(&mut dst, <Elem<$F> as FieldElem>::ONE);
                        assert!(dst.is_empty(), "{}: empty buffers stay empty", case.case);
                    }
                    let backend = fields::backend_name::<$F>();
                    let mut dst: Vec<u8> = Vec::new();
                    let src: Vec<u8> = Vec::new();
                    let c = <Elem<$F> as FieldElem>::ONE;
                    registry::time(case, ctx, &backend, || {
                        ops::add_assign::<$F>(black_box(&mut dst), black_box(&src));
                        ops::mul_add::<$F>(black_box(&mut dst), c, black_box(&src));
                        ops::mul_into::<$F>(black_box(&mut dst), c, black_box(&src));
                        ops::mul_assign::<$F>(black_box(&mut dst), c);
                        fixture::observe(dst.len());
                    });
                },
            );
        };
    }
    specials!("gf2p8-aes", crate::fields::Gf2p8Aes);
    specials!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
    specials!("m31", crate::fields::M31);
}

/// Register the single-row bulk panel.
pub fn register() {
    macro_rules! walk {
        ($label:literal, $F:ty) => {
            register_add_assign::<$F>($label);
            register_sub_assign::<$F>($label);
            register_add_rows::<$F>($label);
            register_mul_add::<$F>($label);
            register_mul_add_prepared::<$F>($label);
            register_mul_into::<$F>($label);
            register_mul_into_prepared::<$F>($label);
            register_mul_assign::<$F>($label);
            register_mul_assign_prepared::<$F>($label);
            register_elementwise::<$F>($label);
            register_elementwise_assign::<$F>($label);
            register_scalar_broadcast::<$F>($label, false);
            register_scalar_broadcast::<$F>($label, true);
        };
    }
    for_each_bulk_field!(walk);
    register_size_sweep();
    register_special_routes();
}
