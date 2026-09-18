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
//! the public dispatching operations. `goldilocks_wrap_internals` exercises
//! direct AVX2 and SSE4.2 entries against the same fixtures.

#![allow(clippy::cast_possible_truncation)]

use fgf::{Goldilocks, goldilocks, ops};

#[path = "common/goldilocks_wrap.rs"]
mod common;
use common::{addmulmod, boundary_family, check_lanes, lanes_bytes, mulmod};

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
