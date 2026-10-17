//! Prepared field-map rows over stable byte lanes.
//!
//! [`Frobenius`] compiles one characteristic-power
//! automorphism for repeated application; [`PreparedEmbedding`] compiles one
//! canonical proper inclusion and [`PreparedIsomorphism`] one presentation
//! change. These rows apply the compiled maps lane by lane over
//! [`FieldBuffer`] encodings. Each row decodes, applies the scalar prepared
//! method, and encodes canonically, so row output always equals the
//! per-element scalar loop.

#[cfg(feature = "alloc")]
use super::{check_pair, check_width};
#[cfg(feature = "alloc")]
use crate::Frobenius;
#[cfg(feature = "alloc")]
use crate::bits;
#[cfg(feature = "alloc")]
use crate::field::{Elem, FieldBuffer};
#[cfg(feature = "alloc")]
use crate::{PreparedEmbedding, PreparedIsomorphism};

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

/// Validate one row span before any destination write: `src` holds whole
/// `S` elements and `dst` holds exactly the matching `T` lanes, with the
/// span product checked. Panics name the operation and the failing operand.
#[cfg(feature = "alloc")]
#[inline]
fn check_row<S: FieldBuffer, T: FieldBuffer>(name: &str, dst_len: usize, src_len: usize) {
    assert!(
        src_len.is_multiple_of(S::BYTES),
        "{name}: src is {src_len} bytes, not a whole number of {} elements",
        S::NAME
    );
    let count = src_len / S::BYTES;
    let Some(expected) = count.checked_mul(T::BYTES) else {
        panic!("{name}: dst is {dst_len} bytes but count {count} overflows `usize`");
    };
    assert_eq!(
        dst_len,
        expected,
        "{name}: dst is {dst_len} bytes but src holds {count} {} elements needing {expected}",
        S::NAME
    );
}

/// Apply a prepared embedding across one row: `dst[i]` becomes the image of
/// `src[i]`.
///
/// Each source lane is decoded, run through [`PreparedEmbedding::embed`],
/// and encoded canonically. Output lanes are canonical unconditionally.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing source element or `dst` does
/// not hold exactly one target lane per source element, naming the
/// operation and the failing operand. Validation runs before any
/// destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn embed_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_row::<S, T>("embed_into_with", dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(T::BYTES)
        .zip(src.chunks_exact(S::BYTES))
    {
        T::encode(d, map.embed(S::decode(s)));
    }
}

/// Apply a prepared relative trace across one row: `dst[i]` becomes the
/// trace of `src[i]` in the source field.
///
/// Each target lane is decoded, run through [`PreparedEmbedding::trace`],
/// and encoded canonically. Output lanes are canonical unconditionally.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing target element or `dst` does
/// not hold exactly one source lane per target element, naming the
/// operation and the failing operand. Validation runs before any
/// destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn trace_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_row::<T, S>("trace_into_with", dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(S::BYTES)
        .zip(src.chunks_exact(T::BYTES))
    {
        S::encode(d, map.trace(T::decode(s)));
    }
}

/// Apply a prepared relative norm across one row: `dst[i]` becomes the norm
/// of `src[i]` in the source field.
///
/// Each target lane is decoded, run through [`PreparedEmbedding::norm`],
/// and encoded canonically. Output lanes are canonical unconditionally.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing target element or `dst` does
/// not hold exactly one source lane per target element, naming the
/// operation and the failing operand. Validation runs before any
/// destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn norm_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_row::<T, S>("norm_into_with", dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(S::BYTES)
        .zip(src.chunks_exact(T::BYTES))
    {
        S::encode(d, map.norm(T::decode(s)));
    }
}

/// Validate a restriction row before any destination write: `src` holds
/// whole target elements, `dst` holds exactly one source lane per element,
/// and `valid` holds exactly [`bits::bytes_for`]`(count)` bytes.
#[cfg(feature = "alloc")]
#[inline]
fn check_restrict_row<S: FieldBuffer, T: FieldBuffer>(
    name: &str,
    dst_len: usize,
    valid_len: usize,
    src_len: usize,
) {
    check_row::<T, S>(name, dst_len, src_len);
    let count = src_len / T::BYTES;
    let expected_valid = bits::bytes_for(count);
    assert_eq!(
        valid_len, expected_valid,
        "{name}: valid is {valid_len} bytes but {count} elements need {expected_valid}"
    );
}

/// Restrict a prepared embedding across one row: `dst[i]` becomes the
/// preimage of `src[i]` when it lies in the image, with `valid` marking
/// the successes.
///
/// Each target lane is decoded and run through
/// [`PreparedEmbedding::restrict`]. Valid lanes encode the canonical
/// preimage; failed lanes read as canonical source zero. `valid` is the
/// LSB-first bitmap ([`bits`] convention) with
/// [`bits::bytes_for`]`(count)` bytes and cleared tail bits. Destinations
/// are write-only and canonically encoded on success unconditionally; zero
/// is not a failure sentinel — a valid zero has its bit set.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing target element, `dst` does
/// not hold exactly one source lane per target element, or `valid` is not
/// exactly the bitmap length, naming the operation and the failing
/// operand. Validation runs before any destination write; empty rows are
/// valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn restrict_into_with<S, T>(
    dst: &mut [u8],
    valid: &mut [u8],
    map: &PreparedEmbedding<S, T>,
    src: &[u8],
) where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_restrict_row::<S, T>("restrict_into_with", dst.len(), valid.len(), src.len());
    valid.fill(0);
    for (i, (d, s)) in dst
        .chunks_exact_mut(S::BYTES)
        .zip(src.chunks_exact(T::BYTES))
        .enumerate()
    {
        match map.restrict(T::decode(s)) {
            Some(preimage) => {
                S::encode(d, preimage);
                valid[i / 8] |= 1 << (i % 8);
            }
            None => {
                S::encode(d, Elem::<S>::ZERO);
            }
        }
    }
}

/// Test a prepared embedding across one row: `valid` marks the lanes of
/// `src` lying in the image.
///
/// Each target lane is decoded and run through
/// [`PreparedEmbedding::contains`]. `valid` is the LSB-first bitmap
/// ([`bits`] convention) with [`bits::bytes_for`]`(count)` bytes and
/// cleared tail bits; only the bitmap is written.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing target element or `valid`
/// is not exactly the bitmap length, naming the operation and the failing
/// operand. Validation runs before any destination write; empty rows are
/// valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn contains_into_with<S, T>(valid: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    assert!(
        src.len().is_multiple_of(T::BYTES),
        "contains_into_with: src is {} bytes, not a whole number of {} elements",
        src.len(),
        T::NAME
    );
    let count = src.len() / T::BYTES;
    let expected_valid = bits::bytes_for(count);
    assert_eq!(
        valid.len(),
        expected_valid,
        "contains_into_with: valid is {} bytes but {count} elements need {expected_valid}",
        valid.len()
    );
    valid.fill(0);
    for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
        if map.contains(T::decode(s)) {
            valid[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Apply a prepared isomorphism across one row: `dst[i]` becomes the image
/// of `src[i]`.
///
/// Each source lane is decoded, run through
/// [`PreparedIsomorphism::apply`], and encoded canonically. Output lanes
/// are canonical unconditionally.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing source element or `dst` does
/// not hold exactly one target lane per source element, naming the
/// operation and the failing operand. Validation runs before any
/// destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn apply_into_with<S, T>(dst: &mut [u8], map: &PreparedIsomorphism<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_row::<S, T>("apply_into_with", dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(T::BYTES)
        .zip(src.chunks_exact(S::BYTES))
    {
        T::encode(d, map.apply(S::decode(s)));
    }
}

/// Unapply a prepared isomorphism across one row: `dst[i]` becomes the
/// preimage of `src[i]`.
///
/// Each target lane is decoded, run through
/// [`PreparedIsomorphism::unapply`], and encoded canonically. Output lanes
/// are canonical unconditionally.
///
/// # Panics
///
/// Panics when `src` holds a partial trailing target element or `dst` does
/// not hold exactly one source lane per target element, naming the
/// operation and the failing operand. Validation runs before any
/// destination write; empty rows are valid.
#[cfg(feature = "alloc")]
#[inline]
pub fn unapply_into_with<S, T>(dst: &mut [u8], map: &PreparedIsomorphism<S, T>, src: &[u8])
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    check_row::<T, S>("unapply_into_with", dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(S::BYTES)
        .zip(src.chunks_exact(T::BYTES))
    {
        S::encode(d, map.unapply(T::decode(s)));
    }
}
