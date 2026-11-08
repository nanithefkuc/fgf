//! AVX-512 Goldilocks lane-edge differentials against the `u128 % p`
//! model.

use super::*;
use crate::field::goldilocks;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

/// Goldilocks lane edges noise buffers rarely hit — non-canonical
/// inputs, operands at `p − 1`, and products whose fold both borrows and
/// wraps — against the `u128 % p` model.
#[test]
fn gld_avx512_lane_edges_match_u128_model() {
    let modulus = goldilocks::MODULUS;
    let edges: [u64; 12] = [
        0,
        1,
        2,
        0xFFFF_FFFF,
        0x1_0000_0000,
        modulus - 2,
        modulus - 1,
        modulus,
        modulus + 1,
        u64::MAX - 1,
        u64::MAX,
        0xFFFF_FFFE_FFFF_FFFF,
    ];
    let (lhs, rhs): (Vec<u64>, Vec<u64>) = edges
        .iter()
        .flat_map(|&x| edges.iter().map(move |&y| (x, y)))
        .unzip();
    let bytes = |v: &[u64]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let (lhs_bytes, rhs_bytes) = (bytes(&lhs), bytes(&rhs));
    let word = |buf: &[u8], lane: usize| {
        u64::from_le_bytes(buf[8 * lane..8 * lane + 8].try_into().unwrap())
    };
    let wide = u128::from(modulus);

    tier_gate::run_entry(
        x86::goldilocks::mul_elementwise_avx512,
        |token: X64V4xToken, entry| {
            let mut product = vec![0u8; lhs_bytes.len()];
            entry(token.v4(), &mut product, &lhs_bytes, &rhs_bytes);
            for (lane, (&left, &right)) in lhs.iter().zip(&rhs).enumerate() {
                let (x, y) = (u128::from(left) % wide, u128::from(right) % wide);
                assert_eq!(
                    u128::from(word(&product, lane)),
                    x * y % wide,
                    "gld mul {left:#x} * {right:#x}"
                );
            }
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::add_assign_avx512,
        |token: X64V4xToken, entry| {
            let mut sum = lhs_bytes.clone();
            entry(token.v4(), &mut sum, &rhs_bytes);
            for (lane, (&left, &right)) in lhs.iter().zip(&rhs).enumerate() {
                let (x, y) = (u128::from(left) % wide, u128::from(right) % wide);
                assert_eq!(
                    u128::from(word(&sum, lane)),
                    (x + y) % wide,
                    "gld add {left:#x} {right:#x}"
                );
            }
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::sub_assign_avx512,
        |token: X64V4xToken, entry| {
            let mut diff = lhs_bytes.clone();
            entry(token.v4(), &mut diff, &rhs_bytes);
            for (lane, (&left, &right)) in lhs.iter().zip(&rhs).enumerate() {
                let (x, y) = (u128::from(left) % wide, u128::from(right) % wide);
                assert_eq!(
                    u128::from(word(&diff, lane)),
                    (x + wide - y) % wide,
                    "gld sub {left:#x} {right:#x}"
                );
            }
        },
    );
}
