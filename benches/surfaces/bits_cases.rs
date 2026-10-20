//! The bit-packed GF(2) surface: whole-buffer and range XOR, elementwise
//! AND, the prepared [`XorRange`], weight, dot product, and subset-XOR
//! gather, at short, resident, and nonresident geometries, plus the
//! empty-range and zero-length special routes.
//!
//! The field label of every case is `gf2-packed-bits`: elements are bits,
//! eight per byte, LSB-first, with caller-maintained zero padding past the
//! logical length. Oracles are byte-level manual bit loops outside every
//! timed region; steady bodies run on preallocated packed buffers whose
//! padding stays zero across calls. Every run body re-derives its geometry
//! from the case record — `lanes` is the bit count, `bytes` the packed byte
//! length — so the entry points stay plain function pointers.

use std::hint::black_box;

use fgf::bits::{self, XorRange};

use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

const SHORT_BITS: usize = 512;
const RESIDENT_BITS: usize = 2 * 1024 * 1024;
const NONRESIDENT_BITS: usize = 128 * 1024 * 1024 * 8;

const API_BYTES_FOR: &[&str] = &["fgf::bits::bytes_for"];
const API_XOR_ASSIGN: &[&str] = &["fgf::bits::xor_assign"];
const API_XOR_RANGE: &[&str] = &["fgf::bits::xor_range"];
const API_XOR_RANGE_NEW: &[&str] = &["fgf::bits::XorRange::new"];
const API_XOR_RANGE_WITH: &[&str] = &["fgf::bits::xor_range_with", "fgf::bits::XorRange::new"];
const API_AND_INTO: &[&str] = &["fgf::bits::and_into"];
const API_AND_ASSIGN: &[&str] = &["fgf::bits::and_assign"];
const API_ANDNOT: &[&str] = &["fgf::bits::andnot_assign"];
const API_CLEAR_RANGE: &[&str] = &["fgf::bits::clear_range"];
const API_SET_RANGE: &[&str] = &["fgf::bits::set_range"];
const API_WEIGHT: &[&str] = &["fgf::bits::weight"];
const API_DOT: &[&str] = &["fgf::bits::dot_product", "fgf::field::FieldElem::is_one"];
const API_XOR_GATHER: &[&str] = &["fgf::bits::xor_gather"];
const API_EMPTY_ROUTES: &[&str] = &[
    "fgf::bits::xor_range",
    "fgf::bits::xor_range_with",
    "fgf::bits::xor_gather",
    "fgf::bits::xor_assign",
];

const FIELD_LABEL: &str = "gf2-packed-bits";
const BACKEND: &str = "scalar";

/// A packed buffer with `bits` live elements: deterministic noise with the
/// padding bits of the final byte forced to zero.
fn packed_bits(len_bytes: usize, bits: usize, seed: u64) -> Vec<u8> {
    let mut buf = fixture::noise(len_bytes, seed);
    let live = bits % 8;
    if live != 0 {
        let last = buf.len() - 1;
        buf[last] &= (1u8 << live) - 1;
    }
    buf
}

/// Byte-level oracle for a masked range XOR.
fn xor_range_oracle(dst: &mut [u8], src: &[u8], from: usize, to: usize) {
    for bit in from..to {
        let byte = bit / 8;
        let mask = 1u8 << (bit % 8);
        dst[byte] ^= src[byte] & mask;
    }
}

#[allow(clippy::too_many_arguments)] // Fields mirror the measurement record.
fn bits_case(
    op: &'static str,
    size: &'static str,
    api: &'static [&'static str],
    bits: usize,
    bytes: usize,
    sources: usize,
    destinations: usize,
    scope: Scope,
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case: format!("bits/{op}/{size}"),
        api,
        field: FIELD_LABEL.to_string(),
        operation: op.to_string(),
        scope,
        bytes,
        lanes: bits,
        sources,
        destinations,
        control: false,
        phase: None,
        run,
    });
}

/// Register the whole bit-packed panel.
pub fn register() {
    for (size, bits) in [
        ("short", SHORT_BITS),
        ("resident", RESIDENT_BITS),
        ("nonresident", NONRESIDENT_BITS),
    ] {
        bits_case(
            "xor-assign",
            size,
            API_XOR_ASSIGN,
            bits,
            bits::bytes_for(bits),
            1,
            1,
            Scope::Steady,
            run_xor_assign,
        );
    }
    bits_case(
        "xor-range",
        "resident",
        API_XOR_RANGE,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        1,
        1,
        Scope::Steady,
        run_xor_range,
    );
    bits_case(
        "xor-range-prepared",
        "resident",
        API_XOR_RANGE_WITH,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        1,
        1,
        Scope::Steady,
        run_xor_range_prepared,
    );
    bits_case(
        "xor-range-new",
        "resident",
        API_XOR_RANGE_NEW,
        RESIDENT_BITS,
        0,
        0,
        0,
        Scope::Setup,
        run_xor_range_new,
    );
    bits_case(
        "and-into",
        "resident",
        API_AND_INTO,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        2,
        1,
        Scope::Steady,
        run_and_into,
    );
    bits_case(
        "and-assign",
        "resident",
        API_AND_ASSIGN,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        1,
        1,
        Scope::Steady,
        run_and_assign,
    );
    bits_case(
        "andnot-assign",
        "resident",
        API_ANDNOT,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        1,
        1,
        Scope::Steady,
        run_andnot_assign,
    );
    bits_case(
        "clear-range",
        "resident",
        API_CLEAR_RANGE,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        0,
        1,
        Scope::Steady,
        run_clear_range,
    );
    bits_case(
        "set-range",
        "resident",
        API_SET_RANGE,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        0,
        1,
        Scope::Steady,
        run_set_range,
    );
    bits_case(
        "weight",
        "resident",
        API_WEIGHT,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        1,
        0,
        Scope::Steady,
        run_weight,
    );
    bits_case(
        "dot-product",
        "resident",
        API_DOT,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        2,
        1,
        Scope::Steady,
        run_dot_product,
    );
    bits_case(
        "xor-gather",
        "resident",
        API_XOR_GATHER,
        RESIDENT_BITS,
        bits::bytes_for(RESIDENT_BITS),
        4,
        1,
        Scope::Steady,
        run_xor_gather,
    );
    bits_case(
        "empty-range",
        "small",
        API_EMPTY_ROUTES,
        64,
        0,
        1,
        1,
        Scope::Steady,
        run_empty_range,
    );
    bits_case(
        "empty-buffers",
        "small",
        API_EMPTY_ROUTES,
        0,
        0,
        1,
        1,
        Scope::Steady,
        run_empty_buffers,
    );
    bits_case(
        "bytes-for",
        "query",
        API_BYTES_FOR,
        0,
        0,
        0,
        0,
        Scope::Query,
        run_bytes_for,
    );
}

/// The live range the masked operations exercise, inside every geometry.
fn window(bits: usize) -> (usize, usize) {
    (11, bits - 17)
}

fn run_xor_assign(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let src = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for (d, s) in expected.iter_mut().zip(&src) {
            *d ^= s;
        }
        bits::xor_assign(&mut dst, &src);
        assert_eq!(dst, expected, "{}: xor oracle", case.case);
    }
    registry::time(case, ctx, fgf::backend().name(), || {
        bits::xor_assign(black_box(&mut dst), black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_xor_range(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let (from, to) = window(bits);
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let src = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        xor_range_oracle(&mut expected, &src, from, to);
        bits::xor_range(&mut dst, &src, bits, from, to);
        assert_eq!(dst, expected, "{}: range oracle", case.case);
    }
    registry::time(case, ctx, fgf::backend().name(), || {
        bits::xor_range(black_box(&mut dst), black_box(&src), bits, from, to);
        fixture::observe(dst.len());
    });
}

fn run_xor_range_prepared(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let (from, to) = window(bits);
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let src = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let range = XorRange::new(bits, from, to);
    {
        let mut expected = dst.clone();
        xor_range_oracle(&mut expected, &src, from, to);
        bits::xor_range_with(&mut dst, &src, &range);
        assert_eq!(dst, expected, "{}: prepared range oracle", case.case);
    }
    registry::time(case, ctx, fgf::backend().name(), || {
        bits::xor_range_with(black_box(&mut dst), black_box(&src), &range);
        fixture::observe(dst.len());
    });
}

fn run_xor_range_new(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let (from, to) = window(bits);
    let range = XorRange::new(bits, from, to);
    assert_eq!(range, XorRange::new(bits, from, to));
    registry::time(case, ctx, BACKEND, || {
        let range = XorRange::new(black_box(bits), black_box(from), black_box(to));
        fixture::observe(range);
    });
}

fn run_and_into(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let a = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let b = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for (d, (x, y)) in expected.iter_mut().zip(a.iter().zip(&b)) {
            *d = x & y;
        }
        bits::and_into(&mut dst, &a, &b);
        assert_eq!(dst, expected, "{}: and oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::and_into(black_box(&mut dst), black_box(&a), black_box(&b));
        fixture::observe(dst.len());
    });
}

fn run_and_assign(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let src = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for (d, s) in expected.iter_mut().zip(&src) {
            *d &= s;
        }
        bits::and_assign(&mut dst, &src);
        assert_eq!(dst, expected, "{}: and-assign oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::and_assign(black_box(&mut dst), black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_andnot_assign(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let mask = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for (d, m) in expected.iter_mut().zip(&mask) {
            *d &= !m;
        }
        bits::andnot_assign(&mut dst, &mask);
        assert_eq!(dst, expected, "{}: andnot oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::andnot_assign(black_box(&mut dst), black_box(&mask));
        fixture::observe(dst.len());
    });
}

fn run_clear_range(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let (from, to) = window(bits);
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        xor_range_oracle(&mut expected, &dst, from, to);
        bits::clear_range(&mut dst, bits, from, to);
        assert_eq!(dst, expected, "{}: clear oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::clear_range(black_box(&mut dst), bits, from, to);
        fixture::observe(dst.len());
    });
}

fn run_set_range(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let (from, to) = window(bits);
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for bit in from..to {
            expected[bit / 8] |= 1u8 << (bit % 8);
        }
        bits::set_range(&mut dst, bits, from, to);
        assert_eq!(dst, expected, "{}: set oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::set_range(black_box(&mut dst), bits, from, to);
        fixture::observe(dst.len());
    });
}

fn run_weight(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let seed = seed_for(&case.case);
    let buf = packed_bits(bytes, bits, seed);
    {
        // Padding bits are zero, so the whole-buffer popcount is the live
        // weight.
        let expected: usize = buf.iter().map(|byte| byte.count_ones() as usize).sum();
        assert_eq!(
            bits::weight(&buf, bits),
            expected,
            "{}: weight oracle",
            case.case
        );
    }
    let backend = if cfg!(feature = "simd512") && fgf::backend() == fgf::Backend::V4x {
        "v4x"
    } else {
        "scalar"
    };
    registry::time(case, ctx, backend, || {
        fixture::observe(bits::weight(black_box(&buf), bits));
    });
}

fn run_dot_product(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let a = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    let b = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let overlap: usize = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum();
        let expected = overlap % 2 == 1;
        let parity = bits::dot_product(&a, &b, bits);
        assert_eq!(parity.is_one(), expected, "{}: dot oracle", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        fixture::observe(bits::dot_product(black_box(&a), black_box(&b), bits).to_raw());
    });
}

fn run_xor_gather(case: &Case, ctx: &CaseCtx) {
    let bits = case.lanes;
    let bytes = case.bytes;
    let mut seed = seed_for(&case.case);
    let srcs: Vec<Vec<u8>> = (0..4u64)
        .map(|t| packed_bits(bytes, bits, fixture::lcg(&mut seed).wrapping_add(t)))
        .collect();
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let selector = 0b1011u64;
    let mut dst = packed_bits(bytes, bits, fixture::lcg(&mut seed));
    {
        let mut expected = dst.clone();
        for (bit, source) in srcs.iter().enumerate() {
            if (selector >> bit) & 1 == 1 {
                for (d, s) in expected.iter_mut().zip(source) {
                    *d ^= s;
                }
            }
        }
        bits::xor_gather(&mut dst, bits, selector, &refs);
        assert_eq!(dst, expected, "{}: gather oracle", case.case);
    }
    registry::time(case, ctx, fgf::backend().name(), || {
        bits::xor_gather(black_box(&mut dst), bits, selector, black_box(&refs));
        fixture::observe(dst.len());
    });
}

fn run_empty_range(case: &Case, ctx: &CaseCtx) {
    let mut seed = seed_for(&case.case);
    let mut dst = packed_bits(8, 64, fixture::lcg(&mut seed));
    let src = packed_bits(8, 64, fixture::lcg(&mut seed));
    {
        let untouched = dst.clone();
        bits::xor_range(&mut dst, &src, 64, 31, 31);
        assert_eq!(dst, untouched, "{}: empty range is a no-op", case.case);
        let range = XorRange::new(64, 0, 0);
        bits::xor_range_with(&mut dst, &src, &range);
        assert_eq!(
            dst, untouched,
            "{}: prepared empty range is a no-op",
            case.case
        );
    }
    registry::time(case, ctx, BACKEND, || {
        bits::xor_range(black_box(&mut dst), black_box(&src), 64, 31, 31);
        fixture::observe(dst.len());
    });
}

fn run_empty_buffers(case: &Case, ctx: &CaseCtx) {
    let mut dst: Vec<u8> = Vec::new();
    let src: Vec<u8> = Vec::new();
    {
        bits::xor_assign(&mut dst, &src);
        bits::xor_gather(&mut dst, 0, 0, &[]);
        assert!(dst.is_empty(), "{}: empty buffers stay empty", case.case);
    }
    registry::time(case, ctx, BACKEND, || {
        bits::xor_assign(black_box(&mut dst), black_box(&src));
        fixture::observe(dst.len());
    });
}

fn run_bytes_for(case: &Case, ctx: &CaseCtx) {
    assert_eq!(bits::bytes_for(0), 0);
    assert_eq!(bits::bytes_for(1), 1);
    assert_eq!(bits::bytes_for(9), 2);
    assert_eq!(bits::bytes_for(65), 9);
    registry::time(case, ctx, BACKEND, || {
        fixture::observe(bits::bytes_for(black_box(123_457)));
    });
}
