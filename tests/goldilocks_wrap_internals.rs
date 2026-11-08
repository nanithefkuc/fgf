//! Direct Goldilocks wrap differentials bounded by the host and request tiers.

#![cfg(all(feature = "std", feature = "simd", target_arch = "x86_64"))]
#![allow(clippy::cast_possible_truncation)]

#[path = "common/goldilocks_wrap.rs"]
mod common;
#[path = "../src/kernel/tests/tier_gate.rs"]
mod tier_gate;

use common::{addmulmod, boundary_family, check_lanes, lanes_bytes, mulmod};
use fgf::internals::kernel::{FGF_TIERS, x86};

enum CoefficientOperation {
    Into,
    Assign,
    Add,
}

fn check_coefficient(
    label: &str,
    operation: CoefficientOperation,
    kernel: impl Fn(&mut [u8], u64, &[u8]),
) {
    let family = boundary_family();
    let src = lanes_bytes(&family);
    let rotated: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 3) % family.len()])
        .collect();
    for &coeff in &family {
        let mut got = match operation {
            CoefficientOperation::Into => vec![0u8; src.len()],
            CoefficientOperation::Assign | CoefficientOperation::Add => lanes_bytes(&rotated),
        };
        kernel(&mut got, coeff, &src);
        let want: Vec<u64> = family
            .iter()
            .zip(&rotated)
            .map(|(&value, &prior)| match operation {
                CoefficientOperation::Into => mulmod(coeff, value),
                CoefficientOperation::Assign => mulmod(coeff, prior),
                CoefficientOperation::Add => addmulmod(prior, coeff, value),
            })
            .collect();
        check_lanes(label, coeff, &got, &want);
    }
}

fn check_elementwise(label: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    let family = boundary_family();
    let src = lanes_bytes(&family);
    let shifted: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 1) % family.len()])
        .collect();
    let mut got = vec![0u8; src.len()];
    kernel(&mut got, &src, &lanes_bytes(&shifted));
    let want: Vec<u64> = family
        .iter()
        .zip(&shifted)
        .map(|(&a, &b)| mulmod(a, b))
        .collect();
    check_lanes(label, 0, &got, &want);
}

#[test]
fn avx2_wrap_boundary_matches_u128_oracle() {
    tier_gate::run_entry(x86::goldilocks::mul_into_avx2, |token, entry| {
        check_coefficient(
            "gld avx2 mul",
            CoefficientOperation::Into,
            |dst, coeff, src| {
                entry(token, dst, coeff, src);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_assign_avx2, |token, entry| {
        check_coefficient(
            "gld avx2 mul",
            CoefficientOperation::Assign,
            |dst, coeff, _src| {
                entry(token, dst, coeff);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_add_avx2, |token, entry| {
        check_coefficient(
            "gld avx2 mul",
            CoefficientOperation::Add,
            |dst, coeff, src| {
                entry(token, dst, coeff, src);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_elementwise_avx2, |token, entry| {
        check_elementwise("gld avx2 mul", |dst, a, b| entry(token, dst, a, b));
    });
}

#[test]
fn sse42_wrap_boundary_matches_u128_oracle() {
    tier_gate::run_entry(x86::goldilocks::mul_into_sse42, |token, entry| {
        check_coefficient(
            "gld sse4.2 mul",
            CoefficientOperation::Into,
            |dst, coeff, src| {
                entry(token, dst, coeff, src);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_assign_sse42, |token, entry| {
        check_coefficient(
            "gld sse4.2 mul",
            CoefficientOperation::Assign,
            |dst, coeff, _src| {
                entry(token, dst, coeff);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_add_sse42, |token, entry| {
        check_coefficient(
            "gld sse4.2 mul",
            CoefficientOperation::Add,
            |dst, coeff, src| {
                entry(token, dst, coeff, src);
            },
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_elementwise_sse42, |token, entry| {
        check_elementwise("gld sse4.2 mul", |dst, a, b| entry(token, dst, a, b));
    });
}
