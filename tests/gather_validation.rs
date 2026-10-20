//! Gather offset validation rejects every invalid offset before any
//! destination byte changes.
//!
//! The public operations validate the whole offset table before dispatch;
//! the direct AVX2 gather must do the same before choosing its lane or
//! per-source fallback shape. Every rejection case asserts the destination
//! bytes survived the panic untouched, so a fold that mutates before
//! validating fails here. No case pins panic prose.

use std::panic::{AssertUnwindSafe, catch_unwind};

use fgf::poly::AES;
use fgf::{FiniteField, Goldilocks, Polynomial, ops};

type Gf8 = FiniteField<2, 8, Polynomial<AES>>;

/// Deterministic LCG noise, the crate's shared convention.
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

/// Asserts `body` panics on `dst` and that `dst` still holds `sentinel`
/// bytes afterwards, proving the rejection preceded every write.
fn rejects_without_mutating(sentinel: &[u8], body: impl FnOnce(&mut [u8])) {
    let mut got = sentinel.to_vec();
    let outcome = catch_unwind(AssertUnwindSafe(|| body(&mut got)));
    assert!(outcome.is_err(), "invalid gather offsets must panic");
    assert_eq!(
        got, sentinel,
        "a rejected gather must not write any destination byte",
    );
}

#[test]
fn invalid_later_offset_never_mutates() {
    let region = noise(256, 0x51);
    let dst = noise(32, 0x52);
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Gf8>(got, &region, &[0, 64, 4_096]);
    });
}

/// A field whose addition is not XOR folds one `add_assign` per source, but
/// the whole offset table is still validated before the fold starts.
#[test]
fn non_xor_residue_field_rejects_before_mutation() {
    let region = noise(256, 0x53);
    let dst = noise(64, 0x54);
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Goldilocks>(got, &region, &[0, 8, 65_536]);
    });
}

/// Only on a 32-bit `usize` does `u32::MAX + dst.len()` overflow. With
/// overflow checks off (`--release`), an unchecked add wraps the range check
/// to a pass and the fold reaches the bad slice with earlier sources already
/// XORed in; the checked sum rejects the table before dispatch. Geometry:
/// a 0x3c-byte region and a 0x3c-byte destination filled with 0xa5, sources
/// at offsets 0 and `u32::MAX`.
#[test]
#[cfg(target_pointer_width = "32")]
fn overflowing_offset_rejects_before_mutation_on_32_bit() {
    if std::env::var_os("FGF_REQUIRE_WRAPPING").is_some() {
        let sum = catch_unwind(|| std::hint::black_box(usize::MAX) + std::hint::black_box(1usize));
        assert_eq!(
            sum.ok(),
            Some(0),
            "the 32-bit acceptance job must disable overflow checks",
        );
    }
    let region = noise(0x3c, 0x55);
    let dst = vec![0xa5u8; 0x3c];
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Gf8>(got, &region, &[0, u32::MAX]);
    });
    let region = noise(64, 0x56);
    let dst = vec![0xa5u8; 64];
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Goldilocks>(got, &region, &[0, u32::MAX]);
    });
}

/// Direct AVX2 entries demand a capability token the host must prove;
/// `#[should_panic]` cannot express the documented skip, so rejection is
/// asserted through `catch_unwind` after the token summons.
#[cfg(all(feature = "simd", feature = "internals", target_arch = "x86_64"))]
mod direct_avx2 {
    use fgf::internals::kernel::{SimdToken, X64V3Token, x86};

    use super::{noise, rejects_without_mutating};

    /// 33 bytes is one whole 32-byte lane plus a 1-byte tail: the shape that
    /// falls back to the per-source fold.
    #[test]
    fn tail_row_rejects_invalid_later_offset_before_mutation() {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping: no AVX2 on this host");
            return;
        };
        let region = noise(128, 0x61);
        let dst = noise(33, 0x62);
        rejects_without_mutating(&dst, |got| {
            x86::bytes::xor_gather_avx2(token, &region, got, &[0, 96]);
        });
    }

    /// 288 bytes is nine whole lanes: wider than eight vectors, the other
    /// shape that falls back to the per-source fold.
    #[test]
    fn wide_row_rejects_invalid_later_offset_before_mutation() {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping: no AVX2 on this host");
            return;
        };
        let region = noise(512, 0x63);
        let dst = noise(288, 0x64);
        rejects_without_mutating(&dst, |got| {
            x86::bytes::xor_gather_avx2(token, &region, got, &[0, 32, 480]);
        });
    }
}
