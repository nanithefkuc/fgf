//! Field-independent byte kernels.

use crate::kernel::scalar;

/// `dst ^= src` over 16-byte NEON lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_neon(_token: archmage::NeonToken, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        vst1q_u8(d, veorq_u8(vld1q_u8(&*d), vld1q_u8(s)));
    }
    scalar::xor(dst_tail, src_tail);
}
