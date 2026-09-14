//! x86 / `x86_64` SIMD kernels.
//!
//! One submodule per field family, plus `bytes` for the field-independent
//! byte kernels. This module-named file keeps only what the families share:
//! the non-temporal store threshold and its alignment helpers, and the two
//! width-parameterized store primitives every `mul_into` kernel calls.
//!
//! Selected entries use [`archmage`] capability tokens and reference-based
//! intrinsics. Unsafe remains only where the borrow checker cannot express
//! offset-addressed rows, uninitialized scratch, or non-temporal stores;
//! each residue item carries a per-item `#[allow(unsafe_code)]` and local
//! SINCE–THUS proofs.
//!
//! Two multiply strategies:
//!
//! - **GFNI.** `GF2P8MULB` multiplies bytes in `GF(2)[x] / 0x11B` — exactly
//!   this crate's GF(2^8) — 32 lanes per instruction, no table, no shuffle
//!   port pressure. This is why the field uses the AES polynomial.
//! - **Nibble shuffle.** Without GFNI, `PSHUFB` performs a 16-entry lookup
//!   per lane, so `c * x` becomes two shuffles and an XOR against the
//!   precomputed [`ScaleTable`](crate::kernel::tables::ScaleTable).

#![allow(clippy::incompatible_msrv)]

// The 64-byte AVX-512 kernels are the deferred V4x tier (cross-compile-only
// today; not in the shared ladder until validated on executing hardware).
// They compile only for `internals` experiments, where they are reachable
// (and differentially tested on a host that has AVX-512).
#[cfg(feature = "internals")]
pub mod avx512;
pub mod bytes;
pub mod fan_paar;
pub mod gf16;
pub mod gf32;
pub mod gf64;
pub mod gf8;
pub mod prime;

pub(crate) use bytes::{xor_avx2, xor_gather_avx2, xor_sse2};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

///
/// `mul_into` writes a destination it never reads, so an ordinary store pays
/// a read-for-ownership fetch of every line it is about to overwrite
/// completely. `vmovntdq` skips that fetch and the allocation, at the price
/// of evicting the destination from cache.
///
/// The threshold is set by the workload this pessimizes — encode, then read
/// the destination back — because that is the case where the eviction costs
/// something. Below it the read-back loop loses; at 2 MiB it breaks even while
/// the write-only case is already several times ahead. `mul_add` and
/// `mul_assign` read their destination anyway and keep ordinary stores. The
/// measurements are under "Crossover and dispatch decisions" in
/// BENCHMARKS.md.
pub(super) const NT_STORE_MIN: usize = 2 << 20;

/// Head bytes to store normally so that a non-temporal body starts on a
/// 32-byte boundary, or `None` when this destination should stay temporal.
///
/// `None` covers a destination too small to repay the eviction, and the case
/// where the alignment peel would not be a whole number of `elem_bytes`
/// elements — a kernel may only split a buffer on an element boundary.
#[inline]
pub(super) fn nt_split(dst: &[u8], elem_bytes: usize) -> Option<usize> {
    if dst.len() < NT_STORE_MIN {
        return None;
    }
    let peel = dst.as_ptr().align_offset(32);
    (peel != usize::MAX && peel.is_multiple_of(elem_bytes)).then_some(peel)
}

/// Bytes of lead-in that put a row group's vector accesses on a 32-byte
/// boundary, or `0` when the row is too short to repay it.
///
/// A 32-byte `vmovdqu` at an odd multiple of 32 straddles two cache lines,
/// and a multi-row body issues one load and one store per row per vector: a
/// misaligned destination therefore doubles the line traffic of every access
/// but the shared source load. Peeling at most 31 bytes per row group buys
/// the aligned body for the rest of the pass; the measurements, and the
/// shapes where the same peel was tried and rejected, are in BENCHMARKS.md.
/// Rows of a group sit `row_len` apart, so aligning one aligns them all
/// whenever `row_len` is a multiple of 32 — the usual case. The source is
/// left where it falls: it is one access against many, and only the
/// destination is ours to choose.
///
/// The peel is rounded up to a whole number of `elem_bytes` elements, because
/// a kernel may only split a buffer on an element boundary. For an odd
/// destination pointer in a 2-byte field that is unreachable, and the peel is
/// skipped.
#[inline]
pub(super) fn peel_to_align(ptr: *const u8, len: usize, elem_bytes: usize) -> usize {
    // The peel runs the narrow single-row kernels, which descend to 128-bit
    // lanes and then scalar code. Keep the conservative crossover measured for
    // the former all-scalar GF(2^8) peel: it won from about 2 KiB up and lost
    // badly below 1 KiB. Sixteen turns of the 128-byte body remains the floor.
    const FLOOR: usize = 16 * 128;
    let head = ptr.align_offset(32);
    // `align_offset` reports `usize::MAX` only for an unreachable alignment,
    // which cannot happen for a byte pointer; `head >= len` also covers the
    // degenerate case of a peel longer than the row.
    if len < FLOOR || head >= len || !head.is_multiple_of(elem_bytes) {
        0
    } else {
        head
    }
}

/// Store 32 bytes, non-temporally when `NT`.
///
/// # Safety
/// `ptr` must be writable for 32 bytes, and when `NT` it must additionally be
/// 32-byte aligned. A `NT` caller must `_mm_sfence()` before any other thread
/// observes the stores.
#[allow(unsafe_code)]
#[archmage::rite(v3)]
pub(super) unsafe fn store256<const NT: bool>(ptr: *mut u8, value: __m256i) {
    // SAFETY:
    // CPU FEATURES
    // SINCE: the `#[archmage::rite(v3)]` context enables AVX2 for this body.
    // THUS: the store intrinsics below satisfy their CPU-feature
    //      precondition.
    //
    // MEMORY VALIDITY
    // SINCE: the `# Safety` contract guarantees `ptr` is writable for 32
    //        bytes.
    // THUS: the store stays within the writable window.
    //
    // ALIGNMENT
    // SINCE: the `# Safety` contract guarantees a 32-byte aligned `ptr` when
    //        `NT`.
    // THUS: the streaming store address is 32-byte aligned.
    unsafe {
        if NT {
            _mm256_stream_si256(ptr.cast(), value);
        } else {
            _mm256_storeu_si256(ptr.cast(), value);
        }
    }
}

/// Store 16 bytes, non-temporally when `NT`.
///
/// # Safety
/// As [`store256`], for 16 bytes and 16-byte alignment.
#[allow(unsafe_code)]
#[archmage::rite(v1)]
pub(super) unsafe fn store128<const NT: bool>(ptr: *mut u8, value: __m128i) {
    // SAFETY:
    // CPU FEATURES
    // SINCE: the `#[archmage::rite(v1)]` context enables SSE2 for this body.
    // THUS: the store intrinsics below satisfy their CPU-feature
    //      precondition.
    //
    // MEMORY VALIDITY
    // SINCE: the `# Safety` contract guarantees `ptr` is writable for 16
    //        bytes.
    // THUS: the store stays within the writable window.
    //
    // ALIGNMENT
    // SINCE: the `# Safety` contract guarantees a 16-byte aligned `ptr` when
    //        `NT`.
    // THUS: the streaming store address is 16-byte aligned.
    unsafe {
        if NT {
            _mm_stream_si128(ptr.cast(), value);
        } else {
            _mm_storeu_si128(ptr.cast(), value);
        }
    }
}
