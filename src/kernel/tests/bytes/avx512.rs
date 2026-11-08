//! AVX-512 byte-kernel differentials.

use super::*;
use crate::kernel::tests::tier_gate;

/// The 64-byte byte kernels: XOR over every lane boundary and mismatched
/// source/destination offsets, and `VPOPCNTQ` counts against `count_ones`.
#[test]
fn bytes512_kernels_match_scalar() {
    let peel = x86::bytes::XOR_PEEL_MIN;
    let edges = [peel - 1, peel, peel + 1, peel + 63, 1024 + 255, 4096 + 63];
    tier_gate::run_entry(x86::bytes::xor_avx512, |token: X64V4xToken, entry| {
        for &len in XOR_LENGTHS.iter().chain(&edges) {
            for &(dst_prefix, src_prefix) in &[(0usize, 0usize), (1, 3), (16, 16), (31, 7), (63, 0)]
            {
                let mut dst_storage = noise(dst_prefix + len, 0x2e);
                let src_storage = noise(src_prefix + len, 0x1d);
                let dst = &mut dst_storage[dst_prefix..];
                let src = &src_storage[src_prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                entry(token.v4(), dst, src);
                assert_eq!(
                    &*dst, want,
                    "xor_avx512: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                );
            }
            // The dispatched entry routes on the short-buffer threshold.
            for prefix in [0usize, 16] {
                let mut dst_storage = noise(prefix + len, 0x4e);
                let src_storage = noise(prefix + len, 0x4d);
                let dst = &mut dst_storage[prefix..];
                let src = &src_storage[prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                crate::kernel::xor(dst, src);
                assert_eq!(&*dst, want, "dispatched xor: len {len}, prefix {prefix}");
            }
        }
    });
    tier_gate::run_entry(x86::bytes::count_ones_avx512, |token, entry| {
        for &len in XOR_LENGTHS.iter().chain(&edges) {
            for prefix in [0usize, 5, 17] {
                let storage = noise(prefix + len, 0x3f);
                let buf = &storage[prefix..];
                let want: u64 = buf.iter().map(|b| u64::from(b.count_ones())).sum();
                assert_eq!(
                    entry(token, buf),
                    want,
                    "count_ones_avx512: len {len}, prefix {prefix}",
                );
            }
        }
        assert_eq!(entry(token, &[0xff; 1000]), 8000);
    });
}
