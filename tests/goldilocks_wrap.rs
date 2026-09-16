//! Goldilocks vector multiplication at the split-fold wrap boundary.
//!
//! The Goldilocks reduction folds a 128-bit product as
//! `lo + hi_lo * eps - hi_hi` with `eps = 2^32 - 1`. When the low 64
//! product bits fall below `hi_hi` (the top 32), the `lo - hi_hi`
//! subtraction wraps and the fold must give back the `2^64 == eps` the
//! wrap added. The operand family below forces that branch on real
//! products — powers of two reaching past `2^96`, the epsilon and
//! modulus extremes — so every lane crosses the boundary inside the
//! vector body and the portable tail. Each lane is checked against the
//! independent `u128 % p` oracle, never the crate's own reduction, over
//! the public dispatching operations and the direct AVX2 and SSE4.2
//! kernel entries (which skip with a notice where the host cannot
//! summon the token).

#![allow(clippy::cast_possible_truncation)]

use fgf::{Goldilocks, goldilocks, ops};

/// `a * b (mod p)` by direct 128-bit integer reduction.
fn mulmod(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(goldilocks::MODULUS)) as u64
}

/// `d + a * b (mod p)` by direct 128-bit integer reduction.
fn addmulmod(d: u64, a: u64, b: u64) -> u64 {
    ((u128::from(d) + u128::from(a) * u128::from(b)) % u128::from(goldilocks::MODULUS)) as u64
}

/// Canonical operands whose pairwise products cross the wrap branch:
/// every power of two (pairs past `2^96` wrap), the `2^32` epsilon
/// neighborhood, the `2^48` neighborhood, the modulus extremes, and
/// 128-divisible lanes at and above `2^39`.
fn boundary_family() -> Vec<u64> {
    let mut values = vec![0, 1, 2];
    values.extend((0..=63).map(|k| 1_u64 << k));
    values.extend([0xFFFF_FFFE, 0xFFFF_FFFF, 0x1_0000_0000]);
    values.extend([(1 << 48) - 1, 1 << 48, (1 << 48) + 1]);
    let p = goldilocks::MODULUS;
    values.extend([p - 1, p - 2, (p - 1) / 2]);
    values.extend([128 << 32, 128 * ((1 << 45) + 7)]);
    values
}

/// Pack canonical lanes into the stable little-endian representation.
fn lanes_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Assert every lane of `got` against `want`, naming the operation,
/// coefficient, lane index, and source lane on failure.
fn check_lanes(name: &str, coeff: u64, got: &[u8], want: &[u64]) {
    for (index, (chunk, &expected)) in got.as_chunks::<8>().0.iter().zip(want.iter()).enumerate() {
        let lane = u64::from_le_bytes(*chunk);
        assert_eq!(
            lane, expected,
            "{name} coeff {coeff:#x} lane {index}: got {lane:#x}, want {expected:#x}"
        );
    }
}

#[test]
fn wrap_boundary_public_ops_match_u128_oracle() {
    let family = boundary_family();
    let src = lanes_bytes(&family);
    let rotated: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 3) % family.len()])
        .collect();
    let rotated_bytes = lanes_bytes(&rotated);

    for &coeff in &family {
        let element = goldilocks::Elem::from_raw(coeff);
        let prepared = ops::Coeff::<Goldilocks>::new(element);

        let mut got = vec![0u8; src.len()];
        ops::mul_into::<Goldilocks>(&mut got, element, &src);
        check_lanes(
            "mul_into",
            coeff,
            &got,
            &family.iter().map(|&v| mulmod(coeff, v)).collect::<Vec<_>>(),
        );

        let mut got = vec![0u8; src.len()];
        ops::mul_into_with::<Goldilocks>(&mut got, &prepared, &src);
        check_lanes(
            "mul_into_with",
            coeff,
            &got,
            &family.iter().map(|&v| mulmod(coeff, v)).collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        ops::mul_assign::<Goldilocks>(&mut got, element);
        check_lanes(
            "mul_assign",
            coeff,
            &got,
            &rotated
                .iter()
                .map(|&v| mulmod(coeff, v))
                .collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        ops::mul_assign_with::<Goldilocks>(&mut got, &prepared);
        check_lanes(
            "mul_assign_with",
            coeff,
            &got,
            &rotated
                .iter()
                .map(|&v| mulmod(coeff, v))
                .collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        ops::mul_add::<Goldilocks>(&mut got, element, &src);
        check_lanes(
            "mul_add",
            coeff,
            &got,
            &family
                .iter()
                .zip(&rotated)
                .map(|(&v, &d)| addmulmod(d, coeff, v))
                .collect::<Vec<_>>(),
        );

        let mut got = rotated_bytes.clone();
        ops::mul_add_with::<Goldilocks>(&mut got, &prepared, &src);
        check_lanes(
            "mul_add_with",
            coeff,
            &got,
            &family
                .iter()
                .zip(&rotated)
                .map(|(&v, &d)| addmulmod(d, coeff, v))
                .collect::<Vec<_>>(),
        );
    }

    // Elementwise: both operands vary per lane, so the wrap branch is
    // crossed from the b-lane gather side as well.
    let shifted: Vec<u64> = (0..family.len())
        .map(|i| family[(i + 1) % family.len()])
        .collect();
    let mut got = vec![0u8; src.len()];
    ops::mul_elementwise::<Goldilocks>(&mut got, &src, &lanes_bytes(&shifted));
    check_lanes(
        "mul_elementwise",
        0,
        &got,
        &family
            .iter()
            .zip(&shifted)
            .map(|(&a, &b)| mulmod(a, b))
            .collect::<Vec<_>>(),
    );
}

#[cfg(all(
    feature = "internals",
    feature = "simd",
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod direct {
    use super::{addmulmod, boundary_family, check_lanes, lanes_bytes, mulmod};
    use fgf::kernel::{SimdToken, X64V2Token, X64V3Token, x86};

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
}
