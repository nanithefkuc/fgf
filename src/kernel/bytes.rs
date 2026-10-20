//! Field-independent byte XOR operations and backend dispatch.

use super::*;

/// `dst ^= sum(srcs[i])` — a blocked XOR gather over byte offsets into one
/// backing region.
///
/// The field-independent shape of back-substitution: every coefficient is
/// one, so a many-source fold is pure XOR, and the sources are rows of one
/// buffer addressed by their start offsets. Addressing sources by offset
/// rather than slice lets a caller walk an index table without staging fat
/// pointers per source.
///
/// This is the XOR primitive under [`FieldKernels::add_gather_offsets`];
/// only fields whose addition is XOR dispatch to it.
///
/// # Panics
/// Panics if any offset plus `dst.len()` falls outside `region`. Every
/// offset is checked, with overflow-safe arithmetic, before any destination
/// byte is written.
pub(crate) fn xor_gather(region: &[u8], dst: &mut [u8], offsets: &[u32]) {
    let live = dst.len();
    for (index, &start) in offsets.iter().enumerate() {
        let start = start as usize;
        let end = start
            .checked_add(live)
            .expect("xor_gather: offset + length overflows");
        assert!(
            end <= region.len(),
            "xor_gather: offset {index} ({start}) + {live} exceeds region of {} bytes",
            region.len(),
        );
    }
    if offsets.is_empty() {
        return;
    }
    if live <= XOR_INLINE_MAX {
        for &start in offsets {
            scalar::xor(dst, &region[start as usize..start as usize + live]);
        }
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => {
            x86::xor_gather_avx2(token.v3(), region, dst, offsets);
        }
        X86Proof::V3GfniCrypto(token) => {
            x86::xor_gather_avx2(token.v3(), region, dst, offsets);
        }
        X86Proof::V3(token) => x86::xor_gather_avx2(token, region, dst, offsets),
        X86Proof::V2(token) => {
            for &start in offsets {
                x86::xor_sse2(
                    token.v1(),
                    dst,
                    &region[start as usize..start as usize + live],
                );
            }
        }
        X86Proof::Scalar => {
            for &start in offsets {
                scalar::xor(dst, &region[start as usize..start as usize + live]);
            }
        }
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for &start in offsets {
            scalar::xor(dst, &region[start as usize..start as usize + live]);
        }
    }
}

/// Buffers at most this long skip the dispatched SIMD XOR for the inline
/// portable one: below a single vector the out-of-line SIMD entry boundary
/// costs more than the body saves, and short GF(2) rows (eight
/// elements per byte) sit almost entirely under it.
const XOR_INLINE_MAX: usize = 31;

/// `dst ^= src` over raw bytes.
///
/// Field-independent: addition in every binary field is XOR, and XOR of a
/// packed element array is XOR of its bytes regardless of element width.
///
/// # Panics
/// Panics before mutation if the slices differ in length.
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fgf::xor: length mismatch");

    if dst.len() <= XOR_INLINE_MAX {
        scalar::xor(dst, src);
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match x86_proof() {
        // The length guard sits on the `V4x` arm alone, so every narrower
        // tier reaches its body without testing it.
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) if dst.len() >= x86::bytes::XOR_AVX512_MIN => {
            x86::bytes::xor_avx512(token.v4(), dst, src);
        }
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => x86::xor_avx2(token.v3(), dst, src),
        X86Proof::V3GfniCrypto(token) => x86::xor_avx2(token.v3(), dst, src),
        X86Proof::V3(token) => x86::xor_avx2(token, dst, src),
        X86Proof::V2(token) => x86::xor_sse2(token.v1(), dst, src),
        X86Proof::Scalar => scalar::xor(dst, src),
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    scalar::xor(dst, src);
}

/// `dst[i] ^= value_bytes[i % value_bytes.len()]` for every byte — one
/// field element broadcast across every lane, the binary fields'
/// `add_assign_scalar`.
///
/// Field-independent: addition in every binary field is XOR, and
/// broadcast addition over a packed element array is the cyclic XOR of
/// the element's little-endian encoding regardless of element width.
/// `value_bytes` holds one element's encoding, 1 to 16 bytes; every lane
/// boundary the vector kernels peel is a multiple of the element width,
/// so the pattern's phase matches the destination's element phase
/// throughout.
///
/// Tier mapping mirrors [`xor`]: the tier each architecture byte
/// kernel runs at is the tier this dispatches to.
///
/// # Panics
/// Panics before mutation if `value_bytes` is empty or longer than 16
/// bytes.
pub fn xor_broadcast(dst: &mut [u8], value_bytes: &[u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => {
            x86::bytes::xor_broadcast_avx2(token.v3(), dst, value_bytes);
        }
        X86Proof::V3GfniCrypto(token) => {
            x86::bytes::xor_broadcast_avx2(token.v3(), dst, value_bytes);
        }
        X86Proof::V3(token) => x86::bytes::xor_broadcast_avx2(token, dst, value_bytes),
        X86Proof::V2(token) => x86::bytes::xor_broadcast_sse2(token.v1(), dst, value_bytes),
        X86Proof::Scalar => scalar::xor_broadcast_bytes(dst, value_bytes),
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    scalar::xor_broadcast_bytes(dst, value_bytes);
}
