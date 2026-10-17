//! x86 tier dispatch for the GF(2) bit-packed interop seam.
//!
//! The GitHub-hosted coverage runner resolves `V3` without GFNI or AVX-512,
//! so the `V3GfniCrypto` and `V4x` arms below cannot execute there — the
//! same situation as the GFNI dispatch arms of `kernel/gf8.rs` and
//! `kernel/tower.rs`, and this dispatch file is excluded from coverage the
//! same way (`COV_IGNORE` in `crate.just`).
//!
//! Tier floors are measured, not reasoned: a row shorter than the tier's
//! vector unit pays setup without vector work and loses to the portable
//! body (i7-12700K pinned core, 2026-10-17: at 8 lanes of 2 bytes SSE4.2
//! beat the portable body 6.6 ns to 18.8 ns while AVX2 lost at 22.9 ns;
//! from one full unit up every tier wins). Each arm therefore requires one
//! full vector unit of row coverage: 16 bytes at `V2`, 32 at `V3` and
//! `V3GfniCrypto`, 64 at `V4x`.

use super::super::{X86Proof, x86, x86_proof};

/// Dispatch [`super::expand`] to the selected tier. Returns `true` when an
/// arm handled the row; `false` leaves the portable body to run.
pub(crate) fn expand(dst: &mut [u8], one_w: u64, src: &[u8], bits: usize, bytes: usize) -> bool {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 64 =>
        {
            x86::gf1::expand_avx512(token, dst, one_w, src, bits, bytes);
            true
        }
        X86Proof::V3GfniCrypto(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::expand_avx2(token.v3(), dst, one_w, src, bits, bytes);
            true
        }
        X86Proof::V3(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::expand_avx2(token, dst, one_w, src, bits, bytes);
            true
        }
        X86Proof::V2(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 16 =>
        {
            x86::gf1::expand_sse42(token, dst, one_w, src, bits, bytes);
            true
        }
        _ => false,
    }
}

/// Dispatch [`super::compress`] to the selected tier. Returns `true` when an
/// arm handled the row; `false` leaves the portable body to run.
pub(crate) fn compress(
    dst: &mut [u8],
    src: &[u8],
    bits: usize,
    bytes: usize,
    test: &super::LaneTest,
) -> bool {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 64 =>
        {
            x86::gf1::compress_avx512(token, dst, src, bits, bytes, test);
            true
        }
        X86Proof::V3GfniCrypto(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::compress_avx2(token.v3(), dst, src, bits, bytes, test);
            true
        }
        X86Proof::V3(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::compress_avx2(token, dst, src, bits, bytes, test);
            true
        }
        X86Proof::V2(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 16 =>
        {
            x86::gf1::compress_sse42(token, dst, src, bits, bytes, test);
            true
        }
        _ => false,
    }
}

/// Dispatch [`super::restrict`] to the selected tier. Returns `true` when an
/// arm handled the row; `false` leaves the portable body to run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn restrict(
    dst: &mut [u8],
    valid: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) -> bool {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 64 =>
        {
            x86::gf1::restrict_avx512(token, dst, valid, one_w, src, bits, bytes);
            true
        }
        X86Proof::V3GfniCrypto(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::restrict_avx2(token.v3(), dst, valid, one_w, src, bits, bytes);
            true
        }
        X86Proof::V3(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 32 =>
        {
            x86::gf1::restrict_avx2(token, dst, valid, one_w, src, bits, bytes);
            true
        }
        X86Proof::V2(token)
            if matches!(bytes, 1 | 2 | 4 | 8) && bits.saturating_mul(bytes) >= 16 =>
        {
            x86::gf1::restrict_sse42(token, dst, valid, one_w, src, bits, bytes);
            true
        }
        _ => false,
    }
}
