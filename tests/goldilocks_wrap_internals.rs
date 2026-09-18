#![cfg(all(
    feature = "std",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#![allow(clippy::cast_possible_truncation)]

#[path = "common/goldilocks_wrap.rs"]
mod common;

use common::{addmulmod, boundary_family, check_lanes, lanes_bytes, mulmod};
use fgf::internals::kernel::{SimdToken, X64V2Token, X64V3Token, x86};

/// Drive one (token, width) pair over the family: every overwrite and
/// in-place multiply shape, plus elementwise, each against the oracle.
fn drive<F: Fn(&mut [u8], u64, &[u8]), G: Fn(&mut [u8], u64), H: Fn(&mut [u8], u64, &[u8])>(
    label: &str,
    mul_into: F,
    mul_assign: G,
    mul_add: H,
    elementwise: impl Fn(&mut [u8], &[u8], &[u8]),
) {
    let family = boundary_family();
    let src = lanes_bytes(&family);
    let rotated: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 3) % family.len()])
        .collect();
    let rotated_bytes = lanes_bytes(&rotated);
    let shifted: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 1) % family.len()])
        .collect();

    for &coeff in &family {
        let mut got = vec![0u8; src.len()];
        mul_into(&mut got, coeff, &src);
        check_lanes(
            label,
            coeff,
            &got,
            &family.iter().map(|&v| mulmod(coeff, v)).collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        mul_assign(&mut got, coeff);
        check_lanes(
            label,
            coeff,
            &got,
            &rotated
                .iter()
                .map(|&v| mulmod(coeff, v))
                .collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        mul_add(&mut got, coeff, &src);
        check_lanes(
            label,
            coeff,
            &got,
            &family
                .iter()
                .zip(&rotated)
                .map(|(&v, &d)| addmulmod(d, coeff, v))
                .collect::<Vec<_>>(),
        );
    }

    let mut got = vec![0u8; src.len()];
    elementwise(&mut got, &src, &lanes_bytes(&shifted));
    check_lanes(
        label,
        0,
        &got,
        &family
            .iter()
            .zip(&shifted)
            .map(|(&a, &b)| mulmod(a, b))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn avx2_wrap_boundary_matches_u128_oracle() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    drive(
        "gld avx2 mul",
        |dst, coeff, src| x86::prime::mul_into_gld_avx2(token, dst, coeff, src),
        |dst, coeff| x86::prime::mul_assign_gld_avx2(token, dst, coeff),
        |dst, coeff, src| x86::prime::mul_add_gld_avx2(token, dst, coeff, src),
        |dst, a, b| x86::prime::mul_elementwise_gld_avx2(token, dst, a, b),
    );
}

#[test]
fn sse42_wrap_boundary_matches_u128_oracle() {
    let Some(token) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE4.2 on this host");
        return;
    };
    drive(
        "gld sse4.2 mul",
        |dst, coeff, src| x86::prime::mul_into_gld_sse42(token, dst, coeff, src),
        |dst, coeff| x86::prime::mul_assign_gld_sse42(token, dst, coeff),
        |dst, coeff, src| x86::prime::mul_add_gld_sse42(token, dst, coeff, src),
        |dst, a, b| x86::prime::mul_elementwise_gld_sse42(token, dst, a, b),
    );
}
