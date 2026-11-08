//! Field-independent byte-lane kernel differentials.
//!
//! The direct SSE2 and AVX2 byte suites here span two instruction sets in
//! one differential; the per-set suites live in the ISA modules.

use super::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::tests::tier_gate;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::x86;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;
#[cfg(all(feature = "simd", feature = "simd512", target_arch = "x86_64"))]
mod avx512;

#[test]
#[should_panic(expected = "xor_gather: offset 1")]
fn xor_gather_rejects_a_source_past_the_region() {
    // Offset 1 leaves the 8-byte window one byte short of the region's end.
    let region = noise(16, 0x5c);
    let mut dst = noise(8, 0x5d);
    crate::kernel::xor_gather(&region, &mut dst, &[8, 9]);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn vector_xor_matches_scalar_xor() {
    tier_gate::run_entry(x86::bytes::xor_avx2, |token, entry| {
        for &len in XOR_LENGTHS {
            for &(dst_prefix, src_prefix) in &[(0usize, 0usize), (1, 3), (15, 17), (31, 7)] {
                let mut dst_storage = noise(dst_prefix + len, 0x2d);
                let src_storage = noise(src_prefix + len, 0x1c);
                let dst = &mut dst_storage[dst_prefix..];
                let src = &src_storage[src_prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                entry(token, dst, src);
                assert_eq!(
                    &*dst, want,
                    "avx2 xor: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                );
            }
        }
    });
    tier_gate::run_entry(x86::xor_sse2, |token, entry| {
        for &len in XOR_LENGTHS {
            for &(dst_prefix, src_prefix) in &[(0usize, 0usize), (1, 3), (15, 17), (31, 7)] {
                let mut dst_storage = noise(dst_prefix + len, 0x2d);
                let src_storage = noise(src_prefix + len, 0x1c);
                let dst = &mut dst_storage[dst_prefix..];
                let src = &src_storage[src_prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                entry(token, dst, src);
                assert_eq!(
                    &*dst, want,
                    "sse2 xor: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                );
            }
        }
    });
}

/// The broadcast XOR kernels directly — every value width the dispatch
/// can produce (1-byte GF(2^8), 2-byte GF(2^16), 4-byte GF(2^32),
/// 8-byte GF(2^64), and a 16-byte width), plus the zero value and a
/// width that divides neither lane, against the portable byte-cyclic
/// reference.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn xor_broadcast_matches_scalar_reference() {
    const VALUES: [&[u8]; 8] = [
        &[0x00],
        &[0x53],
        &[0xa7, 0x53],
        &[0x01, 0x00],
        &[0x1a, 0x2b, 0x3c, 0x4d],
        &[0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78],
        &[
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ],
        // A width that divides neither lane, exercising the portable
        // fallback in both kernels.
        &[0x11, 0x22, 0x33],
    ];
    tier_gate::run_entry(x86::bytes::xor_broadcast_avx2, |token, entry| {
        for &len in XOR_LENGTHS {
            for value in VALUES {
                let dst = &mut noise(len, 0x2e);
                let mut want = dst.clone();
                for (w, &b) in want.iter_mut().zip(value.iter().cycle()) {
                    *w ^= b;
                }
                entry(token, dst, value);
                assert_eq!(
                    &**dst,
                    want.as_slice(),
                    "avx2 xor_broadcast: len {len}, value {value:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::bytes::xor_broadcast_sse2, |token, entry| {
        for &len in XOR_LENGTHS {
            for value in VALUES {
                let dst = &mut noise(len, 0x2e);
                let mut want = dst.clone();
                for (w, &b) in want.iter_mut().zip(value.iter().cycle()) {
                    *w ^= b;
                }
                entry(token, dst, value);
                assert_eq!(
                    &**dst,
                    want.as_slice(),
                    "sse2 xor_broadcast: len {len}, value {value:?}"
                );
            }
        }
    });
}

/// The documented value-width contract of the direct broadcast entries:
/// an empty value and a 17-byte value both panic, and the panic message
/// names the operation.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn xor_broadcast_entries_reject_invalid_value_widths() {
    let mut dst = [0u8; 32];
    let Some(sse) = tier_gate::summon::<X64V1Token>() else {
        eprintln!("skipping: no SSE2 on this host");
        return;
    };
    for bad in [&[][..], &[0u8; 17][..]] {
        if let Some(token) = tier_gate::summon::<X64V3Token>() {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                x86::bytes::xor_broadcast_avx2(token, &mut dst, bad);
            }))
            .expect_err("xor_broadcast_avx2 must reject an invalid value width");
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.starts_with("xor_broadcast_avx2: value is"),
                "panic must name the operation: {msg}"
            );
        }
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x86::bytes::xor_broadcast_sse2(sse, &mut dst, bad);
        }))
        .expect_err("xor_broadcast_sse2 must reject an invalid value width");
        let msg = err.downcast_ref::<String>().expect("string panic payload");
        assert!(
            msg.starts_with("xor_broadcast_sse2: value is"),
            "panic must name the operation: {msg}"
        );
    }
}
