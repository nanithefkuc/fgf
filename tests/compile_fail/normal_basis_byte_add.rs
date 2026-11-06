// expect: basis map is not bijective
// expect-code: E0080
use fgf::{FiniteField, ops};
use fgf::binary::Normal;
pub fn add(dst: &mut [u8], src: &[u8]) {
    ops::add_assign::<FiniteField<2, 8, Normal<0x11B, 1>>>(dst, src);
}
