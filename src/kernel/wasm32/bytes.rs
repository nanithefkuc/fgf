//! Field-independent byte kernels.

use crate::kernel::scalar;

/// `dst ^= src` over 16-byte `simd128` lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_simd128(_token: archmage::Wasm128Token, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        v128_store(d, v128_xor(v128_load(&*d), v128_load(s)));
    }
    scalar::xor(dst_tail, src_tail);
}
