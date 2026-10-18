//! Packing between field elements and their packed byte representation.
//!
//! Element storage is canonical, so packed bytes are canonical lanes and
//! buffers produced here satisfy the packed-operation input contract
//! directly.

use crate::field::{Elem, FieldBuffer};

/// Pack field elements into their stable little-endian byte representation.
///
/// Element storage is canonical, so packed bytes are canonical lanes.
/// Buffers produced here satisfy the packed-operation input contract
/// directly.
///
/// # Panics
/// Panics unless `dst.len() == elems.len() * F::BYTES`.
pub fn pack<F: FieldBuffer>(dst: &mut [u8], elems: &[Elem<F>]) {
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

/// Decode packed little-endian bytes into field elements.
///
/// Decoding canonicalizes each lane into element storage.
///
/// # Panics
/// Panics unless `src.len() == dst.len() * F::BYTES`.
pub fn unpack<F: FieldBuffer>(dst: &mut [Elem<F>], src: &[u8]) {
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

/// Pack field elements into a newly allocated byte vector.
///
/// Like [`pack`], this emits canonical lanes.
///
/// # Panics
/// Panics if the required byte length overflows `usize`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn pack_to_vec<F: FieldBuffer>(elems: &[Elem<F>]) -> alloc::vec::Vec<u8> {
    let len = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack_to_vec: buffer geometry overflows");
    let mut bytes = alloc::vec![0; len];
    pack::<F>(&mut bytes, elems);
    bytes
}

/// Decode packed little-endian bytes into a newly allocated element vector.
///
/// Like [`unpack`], this canonicalizes each lane into element storage.
/// Buffers produced by [`pack`] or [`pack_to_vec`] round-trip exactly.
///
/// # Panics
/// Panics unless `src.len()` is a whole number of field elements.
#[cfg(feature = "alloc")]
#[must_use]
pub fn unpack_to_vec<F: FieldBuffer>(src: &[u8]) -> alloc::vec::Vec<Elem<F>> {
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
