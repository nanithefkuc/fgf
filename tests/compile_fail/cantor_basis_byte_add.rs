// expect: basis map is not bijective
// expect-code: E0080
use fgf::{FiniteField, ops};
use fgf::binary::Cantor;
pub fn add(dst: &mut [u8], src: &[u8]) {
    ops::add_assign::<FiniteField<2, 8, Cantor<0x11B, 0>>>(dst, src);
}
