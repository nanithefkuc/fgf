//! Prepared Frobenius row acceptance: decode-apply-encode equality,
//! in-place decoding, geometry panics, and the conjugation anchor.
//!
//! The row entry points decode each lane, apply the prepared automorphism,
//! and encode canonically, so every row equals the per-element scalar loop
//! of decode into `frobenius_with` into encode.

#![cfg(feature = "alloc")]
// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use fgf::field::FieldBuffer;
use fgf::poly::AES;
use fgf::{Elem, FiniteField, Frobenius, Polynomial, QuadMersenne31, ops};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf64 = FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;

/// Deterministic pseudo-random bytes. No dependency, reproducible failures.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Per-element oracle: decode, apply the prepared map, encode.
fn oracle_into<F: FieldBuffer>(dst: &mut [u8], map: &Frobenius<F>, src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(s).frobenius_with(map));
    }
}

/// Row lengths straddle lane and unroll boundaries, plus one large count.
const COUNTS: [usize; 12] = [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1024];

fn check_counts<F: FieldBuffer>(powers: &[u32]) {
    for &power in powers {
        let map = Frobenius::<F>::new(power);
        for &count in &COUNTS {
            let src = noise(count * F::BYTES, 0xF20 + u64::from(power));
            let mut expected = vec![0u8; count * F::BYTES];
            oracle_into::<F>(&mut expected, &map, &src);
            let mut dst = vec![0xA5u8; count * F::BYTES];
            ops::frobenius_into_with(&mut dst, &map, &src);
            assert_eq!(dst, expected, "into_with count {count} power {power}");
            let mut row = src.clone();
            ops::frobenius_assign_with(&mut row, &map);
            assert_eq!(row, expected, "assign_with count {count} power {power}");
        }
    }
}

#[test]
fn prepared_rows_match_the_scalar_loop() {
    check_counts::<Aes8>(&[0, 1, 2, 7, 8, 9, u32::MAX]);
    check_counts::<Gf16>(&[0, 1, 2, 15, 16, 17, u32::MAX]);
    check_counts::<Gf64>(&[0, 1, 2, 63, 64, 65, u32::MAX]);
    check_counts::<QuadMersenne31>(&[0, 1, 2, 3, u32::MAX]);
}

/// Noncanonical alias lanes decode before the map runs: row output still
/// equals the scalar loop over the same bytes.
#[test]
fn noncanonical_alias_lanes_match_the_scalar_loop() {
    // Mersenne31's maximum raw word is an alias; every lane here is one.
    type M31 = fgf::Mersenne31;
    let map = Frobenius::<M31>::new(1);
    for &count in &COUNTS {
        let src = vec![0xFFu8; count * M31::BYTES];
        let mut expected = vec![0u8; count * M31::BYTES];
        oracle_into::<M31>(&mut expected, &map, &src);
        let mut dst = vec![0u8; count * M31::BYTES];
        ops::frobenius_into_with(&mut dst, &map, &src);
        assert_eq!(dst, expected, "alias into_with count {count}");
        let mut row = src.clone();
        ops::frobenius_assign_with(&mut row, &map);
        assert_eq!(row, expected, "alias assign_with count {count}");
    }
}

/// Over the quadratic Mersenne extension the order-one prepared map is the
/// scalar conjugation.
#[test]
fn qm31_prepared_row_is_conjugation() {
    let map = Frobenius::<QuadMersenne31>::new(1);
    let elems = [
        Elem::<QuadMersenne31>::from_raw([7, 11]),
        Elem::<QuadMersenne31>::from_raw([0x7FFF_FFFE, 0x7FFF_FFFF]),
        Elem::<QuadMersenne31>::ZERO,
        Elem::<QuadMersenne31>::ONE,
    ];
    for (i, &elem) in elems.iter().enumerate() {
        assert_eq!(
            elem.frobenius_with(&map),
            elem.conjugate(),
            "conjugate anchor {i}"
        );
    }
    let mut src = Vec::new();
    for &elem in &elems {
        let mut lane = vec![0u8; QuadMersenne31::BYTES];
        QuadMersenne31::encode(&mut lane, elem);
        src.extend_from_slice(&lane);
    }
    let mut dst = vec![0u8; src.len()];
    ops::frobenius_into_with(&mut dst, &map, &src);
    for (lane, &elem) in dst.chunks_exact(QuadMersenne31::BYTES).zip(elems.iter()) {
        assert_eq!(QuadMersenne31::decode(lane), elem.conjugate());
    }
}

#[cfg(feature = "std")]
mod panics {
    use super::*;

    fn catch(body: impl FnOnce()) -> (String, bool) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        match result {
            Ok(()) => (String::new(), false),
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_default();
                (message, true)
            }
        }
    }

    #[test]
    fn geometry_failures_name_the_operand_and_leave_outputs_unchanged() {
        let map = Frobenius::<Gf16>::new(1);
        // Short destination: the message names the operation and `dst`, and
        // the destination keeps its bytes.
        let mut dst = vec![0xA5u8; 8];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::frobenius_into_with(&mut dst, &map, &[0u8; 16]);
        });
        assert!(panicked, "short dst must panic");
        assert!(
            message.contains("frobenius_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed into_with must not write dst");
        // Short source: names `src`, destination untouched.
        let mut dst = vec![0xA5u8; 16];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::frobenius_into_with(&mut dst, &map, &[0u8; 8]);
        });
        assert!(panicked, "short src must panic");
        assert!(
            message.contains("frobenius_into_with") && message.contains("src"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed into_with must not write dst");
        // Partial trailing element in the in-place row.
        let mut row = vec![0xA5u8; 7];
        let before = row.clone();
        let (message, panicked) = catch(|| ops::frobenius_assign_with(&mut row, &map));
        assert!(panicked, "partial lane must panic");
        assert!(
            message.contains("frobenius_assign_with"),
            "unexpected message: {message}"
        );
        assert_eq!(row, before, "failed assign_with must not write row");
    }
}
