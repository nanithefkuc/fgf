//! Prepared Frobenius rows over stable byte lanes.
//!
//! [`Frobenius`] compiles one characteristic-power
//! automorphism for repeated application; these rows apply it lane by lane
//! over [`FieldBuffer`] encodings. The scalar spelling is
//! `Elem::frobenius_with`; the row below decodes, applies, and encodes each
//! lane, so row output always equals the per-element scalar loop.

#[cfg(feature = "alloc")]
use super::{check_pair, check_width};
#[cfg(feature = "alloc")]
use crate::Frobenius;
#[cfg(feature = "alloc")]
use crate::field::FieldBuffer;

/// Apply a prepared Frobenius map across one row: `dst[i]` becomes the image
/// of `src[i]`.
///
/// Each source lane is decoded, run through the prepared automorphism, and
/// encoded canonically. Output lanes are canonical unconditionally.
///
/// # Panics
///
/// Panics when `dst` and `src` differ in length or hold a partial trailing
/// element, naming the operation and the failing operand. Validation runs
/// before any destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn frobenius_into_with<F: FieldBuffer>(dst: &mut [u8], map: &Frobenius<F>, src: &[u8]) {
    check_pair::<F>("frobenius_into_with", "dst", dst.len(), "src", src.len());
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(s).frobenius_with(map));
    }
}

/// Apply a prepared Frobenius map in place: `row[i]` becomes its own image.
///
/// Each old lane is decoded before its bytes are overwritten, so every lane
/// reads its original value. Output lanes are canonical; empty rows are
/// valid. There is no cross-field in-place form: equal degree does not
/// promise equal encoded width.
///
/// # Panics
///
/// Panics when `row` holds a partial trailing element, naming the operation.
#[cfg(feature = "alloc")]
#[inline]
pub fn frobenius_assign_with<F: FieldBuffer>(row: &mut [u8], map: &Frobenius<F>) {
    check_width::<F>("frobenius_assign_with", row.len());
    for lane in row.chunks_exact_mut(F::BYTES) {
        let image = F::decode(lane).frobenius_with(map);
        F::encode(lane, image);
    }
}
