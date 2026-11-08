//! AVX2 byte-kernel differentials.

use super::*;
use crate::kernel::tests::tier_gate;

/// Field-independent XOR across the half-lane peel floor: the 16-mod-32
/// residues that peel, their odd neighbours that decline, and every
/// destination offset against the byte-wise reference.
#[test]
fn bytes_avx2_peels_half_lane() {
    tier_gate::run_entry(x86::bytes::xor_avx2, |token, entry| {
        let floor = x86::HALF_LANE_PEEL_MIN;
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c1);
                let mut dst_storage = noise(len + 128, 0x2c2);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                entry(token, dst, src);
                assert_eq!(&*dst, want, "xor_avx2: len {len}, offset {offset}");
            }
        }
    });
}

/// The gathered byte XOR against offset-addressed sources: the fused
/// lane loop, the per-source tail fallback, and the wide fallback,
/// over empty, single-source, unaligned, and region-end source lists.
#[test]
fn xor_gather_avx2_matches_scalar() {
    tier_gate::run_entry(x86::xor_gather_avx2, |token, entry| {
        // 32 and 256 bytes take the fused loop, 100 the sub-vector tail
        // fallback, and 288 the wide fallback.
        for live in [32usize, 100, 256, 288] {
            // The last source of `[0, 32, 412]` ends exactly at the region end.
            let region = noise(live + 412, 0x2d1);
            for offsets in [
                vec![0u32],
                vec![0u32, 32, 412],
                vec![7u32, 131, 260, 411],
                vec![],
            ] {
                let mut got = noise(live, 0x2d2);
                let mut want = got.clone();
                entry(token, &region, &mut got, &offsets);
                for &start in &offsets {
                    let start = start as usize;
                    scalar::xor(&mut want, &region[start..start + live]);
                }
                assert_eq!(
                    got, want,
                    "xor_gather_avx2: live {live}, offsets {offsets:?}"
                );
            }
        }
    });
}
