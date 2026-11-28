//! Packing between field elements and their packed byte representation.
//!
//! Element storage is canonical, so packed bytes are canonical lanes and
//! buffers produced here satisfy the packed-operation input contract
//! directly.

use crate::field::{Elem, FieldBuffer, FieldElem, PrimeIdentity, Validate};

use super::geometry::{check_pair, check_width};

/// Returns whether every lane decodes to zero, including in an empty row.
///
/// # Panics
/// Panics if `row` holds a partial lane.
#[must_use]
pub fn is_zero<F: FieldBuffer>(row: &[u8]) -> bool {
    check_width::<F>("is_zero", "row", row.len());
    if has_unique_binary_lanes::<F>() {
        return row.iter().all(|&byte| byte == 0);
    }
    row.chunks_exact(F::BYTES)
        .all(|lane| F::decode(lane).is_zero())
}

/// Returns whether corresponding lanes of two rows decode to equal values.
///
/// # Panics
/// Panics if the lengths differ or a row holds a partial lane.
#[must_use]
pub fn is_equal<F: FieldBuffer>(left: &[u8], right: &[u8]) -> bool {
    check_pair::<F>("is_equal", "left", left.len(), "right", right.len());
    if has_unique_binary_lanes::<F>() {
        return left == right;
    }
    left.chunks_exact(F::BYTES)
        .zip(right.chunks_exact(F::BYTES))
        .all(|(a, b)| a == b || F::decode(a) == F::decode(b))
}

/// Rewrites a row in place to the canonical encoding of its lane values.
///
/// Uses no scratch buffer. Leaves already canonical lanes unchanged.
///
/// # Panics
/// Panics if `dst` holds a partial lane, before writing any bytes.
pub fn canonicalize_assign<F: FieldBuffer>(dst: &mut [u8]) {
    check_width::<F>("canonicalize_assign", "dst", dst.len());
    if has_unique_binary_lanes::<F>() {
        return;
    }
    for lane in dst.chunks_exact_mut(F::BYTES) {
        let value = F::decode(lane);
        F::encode(lane, value);
    }
}

#[inline]
const fn has_unique_binary_lanes<F: FieldBuffer>() -> bool {
    F::Characteristic::CHARACTERISTIC == 2 && F::STORAGE_BITS == F::DEGREE
}

/// Packs field elements into their stable little-endian byte representation.
///
/// Element storage is canonical, so packed bytes are canonical lanes.
/// Buffers produced here satisfy the packed-operation input contract
/// directly.
///
/// # Panics
/// Panics unless `dst.len() == elems.len() * F::BYTES`.
pub fn pack<F: FieldBuffer>(dst: &mut [u8], elems: &[Elem<F>]) {
    let () = Validate::<F>::OK;
    let expected = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack: buffer geometry overflows");
    assert_eq!(
        dst.len(),
        expected,
        "pack: dst is {} bytes but {} {} elements need {expected}",
        dst.len(),
        elems.len(),
        F::NAME,
    );
    for (bytes, &elem) in dst.chunks_exact_mut(F::BYTES).zip(elems) {
        F::encode(bytes, elem);
    }
}

/// Decodes packed little-endian bytes into field elements.
///
/// Every lane is accepted and canonicalized as
/// [`FieldBuffer::decode`](crate::FieldBuffer::decode) specifies.
///
/// # Panics
/// Panics unless `src.len() == dst.len() * F::BYTES`.
pub fn unpack<F: FieldBuffer>(dst: &mut [Elem<F>], src: &[u8]) {
    let () = Validate::<F>::OK;
    let expected = dst
        .len()
        .checked_mul(F::BYTES)
        .expect("unpack: buffer geometry overflows");
    assert_eq!(
        src.len(),
        expected,
        "unpack: src is {} bytes but dst holds {} {} elements ({expected} bytes)",
        src.len(),
        dst.len(),
        F::NAME,
    );
    for (elem, bytes) in dst.iter_mut().zip(src.chunks_exact(F::BYTES)) {
        *elem = F::decode(bytes);
    }
}

/// Packs field elements into a newly allocated byte vector.
///
/// Like [`pack`], this emits canonical lanes.
///
/// # Panics
/// Panics if the required byte length overflows `usize`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn pack_to_vec<F: FieldBuffer>(elems: &[Elem<F>]) -> alloc::vec::Vec<u8> {
    let () = Validate::<F>::OK;
    let len = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack_to_vec: buffer geometry overflows");
    let mut bytes = alloc::vec![0; len];
    pack::<F>(&mut bytes, elems);
    bytes
}

/// Decodes packed little-endian bytes into a newly allocated element vector.
///
/// Every lane is accepted and canonicalized as
/// [`FieldBuffer::decode`](crate::FieldBuffer::decode) specifies.
/// Buffers produced by [`pack`] or [`pack_to_vec`] round-trip exactly.
///
/// # Panics
/// Panics unless `src.len()` is a whole number of field elements.
#[cfg(feature = "alloc")]
#[must_use]
pub fn unpack_to_vec<F: FieldBuffer>(src: &[u8]) -> alloc::vec::Vec<Elem<F>> {
    let () = Validate::<F>::OK;
    assert!(
        src.len().is_multiple_of(F::BYTES),
        "unpack_to_vec: src is {} bytes, not a whole number of {} elements",
        src.len(),
        F::NAME,
    );
    let len = src.len() / F::BYTES;
    let mut elems = alloc::vec![Elem::<F>::ZERO; len];
    unpack::<F>(&mut elems, src);
    elems
}
