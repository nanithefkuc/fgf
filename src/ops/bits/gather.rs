//! Subset-XOR combination over a dense row set.

use super::geometry::check_holds;
use crate::kernel::gf1;

/// `dst ^= srcs[i]` for every set bit `i` of `selector`: the combination
/// step of subset-XOR over a dense row set.
///
/// An empty selector is a no-op. `dst` must not alias any source.
///
/// # Panics
/// Panics if any source differs in length from `dst`, if `selector` names a
/// source at or beyond `srcs.len()`, or if `dst` is shorter than
/// [`bytes_for`](super::bytes_for)`(bits)`.
pub fn xor_gather(dst: &mut [u8], bits: usize, selector: u64, srcs: &[&[u8]]) {
    check_holds("bits::xor_gather", "dst", dst.len(), bits);
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            dst.len(),
            "bits::xor_gather: source {index} is {} bytes but dst is {} bytes",
            src.len(),
            dst.len()
        );
    }
    if srcs.len() < 64 {
        let excess = selector >> srcs.len();
        let count = srcs.len();
        assert!(
            excess == 0,
            "bits::xor_gather: selector names source {} but only {count} sources exist",
            count + excess.trailing_zeros() as usize
        );
    }
    gf1::xor_gather(dst, srcs, selector);
}
