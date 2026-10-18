//! Prepared field-map rows over the bit-packed GF(2) representation.
//!
//! The bit-packed counterparts of the lane rows in [`super::map`]: the GF(2)
//! endpoint rides as packed bits in [`crate::bits`] LSB-first layout while the
//! binary target rides as stable little-endian lanes. The rows read the
//! precomputed `(one_w, trace_word)` pair from the prepared map, so row output
//! equals the per-element scalar loop by construction.
//!
//! # Representation contract
//!
//! Bits-side buffers follow the [`crate::bits`] LSB-first layout with
//! caller-maintained zero padding: packed inputs may carry surplus bytes past
//! [`crate::bits::bytes_for`]`(bits)` (padding like any other), while packed
//! outputs are exactly [`crate::bits::bytes_for`]`(bits)` bytes with cleared
//! tail bits. Lane-side buffers hold whole `T` elements exactly.
//!
//! # Lane-word invariant
//!
//! Every `BinaryField + FieldBuffer` satisfies `8 * BYTES == DEGREE <= 64`:
//! by induction, `Packed` widths equal the degree and `Components` widths are
//! `D` times a base that is itself `FieldBuffer`, so each lane is one
//! coordinate word and the kernel seam reads it straight from the lane bytes.
//! See [`super::map`] for the lane rows.

use crate::PreparedEmbedding;
use crate::bits::{bytes_for, check_holds};
use crate::field::{BinaryField, FieldBuffer, PrimeCharacteristic, PrimeField};
use crate::kernel::gf1::{self, LaneTest};

/// Validate one lane-side row span: `len` holds exactly `bits` `T` lanes,
/// with the span product checked. Panics name the operation and operand.
#[inline]
fn check_lanes<T: FieldBuffer>(name: &str, operand: &str, len: usize, bits: usize) {
    let Some(expected) = bits.checked_mul(T::BYTES) else {
        panic!("{name}: {operand} is {len} bytes but count {bits} overflows `usize`");
    };
    assert_eq!(
        len, expected,
        "{name}: {operand} is {len} bytes but {bits} elements need {expected}"
    );
}

/// Assert one packed output is exactly the bitmap length.
#[inline]
fn check_packed_out(name: &str, operand: &str, len: usize, bits: usize) {
    let expected = bytes_for(bits);
    assert_eq!(
        len, expected,
        "{name}: {operand} is {len} bytes but {bits} bits need {expected}"
    );
}

/// The precomputed bit-map words of a GF(2)-sourced map.
#[inline]
fn bit_words<S, T>(map: &PreparedEmbedding<S, T>) -> (u64, u64)
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    map.bit_map_words()
        .expect("bit_map_words: GF(2)-sourced maps carry precomputed bit words")
}

/// Expand a packed GF(2) row into target lanes: `dst[i]` becomes the image of
/// bit `i` of `src`.
///
/// `src` needs at least [`bytes_for`]`(bits)` bytes (surplus padding
/// allowed); `dst` holds exactly `bits` `T` lanes. Empty rows are valid.
///
/// # Panics
///
/// Panics when `src` is short of the packed coverage or `dst` does not hold
/// exactly `bits` lanes, naming the operation and the failing operand.
/// Validation runs before any destination write.
#[inline]
pub fn embed_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8], bits: usize)
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_holds("embed_into_with", "src", src.len(), bits);
    check_lanes::<T>("embed_into_with", "dst", dst.len(), bits);
    let (one_w, _) = bit_words(map);
    gf1::expand(dst, one_w, src, bits, T::BYTES);
}

/// Compress target lanes through the relative trace: bit `i` of `dst` is the
/// trace of `src[i]`.
///
/// `src` holds exactly `bits` `T` lanes; `dst` is exactly
/// [`bytes_for`]`(bits)` bytes with cleared tail bits. Empty rows are valid.
///
/// # Panics
///
/// Panics when `src` does not hold exactly `bits` lanes or `dst` is not
/// exactly the bitmap length, naming the operation and the failing operand.
/// Validation runs before any destination write.
#[inline]
pub fn trace_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8], bits: usize)
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_lanes::<T>("trace_into_with", "src", src.len(), bits);
    check_packed_out("trace_into_with", "dst", dst.len(), bits);
    let (_, tau) = bit_words(map);
    gf1::compress(dst, src, bits, T::BYTES, &LaneTest::Parity(tau));
}

/// Compress target lanes through the relative norm: bit `i` of `dst` is
/// whether `src[i]` is nonzero.
///
/// `src` holds exactly `bits` `T` lanes; `dst` is exactly
/// [`bytes_for`]`(bits)` bytes with cleared tail bits. Empty rows are valid.
///
/// # Panics
///
/// Panics when `src` does not hold exactly `bits` lanes or `dst` is not
/// exactly the bitmap length, naming the operation and the failing operand.
/// Validation runs before any destination write.
#[inline]
pub fn norm_into_with<S, T>(dst: &mut [u8], map: &PreparedEmbedding<S, T>, src: &[u8], bits: usize)
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_lanes::<T>("norm_into_with", "src", src.len(), bits);
    check_packed_out("norm_into_with", "dst", dst.len(), bits);
    let _ = bit_words(map);
    gf1::compress(dst, src, bits, T::BYTES, &LaneTest::NonZero);
}

/// Restrict target lanes to the image: `dst` bit `i` is the preimage of
/// `src[i]` when it lies in the image, with `valid` marking the successes.
///
/// Outsiders and valid zeros both write a zero value bit; a valid zero keeps
/// its bit set. `src` holds exactly `bits` `T` lanes; `dst` and `valid` are
/// each exactly [`bytes_for`]`(bits)` bytes with cleared tails. Empty rows
/// are valid.
///
/// # Panics
///
/// Panics when `src` does not hold exactly `bits` lanes or either output is
/// not exactly the bitmap length, naming the operation and the failing
/// operand. Both outputs are validated before either is written.
#[inline]
pub fn restrict_into_with<S, T>(
    dst: &mut [u8],
    valid: &mut [u8],
    map: &PreparedEmbedding<S, T>,
    src: &[u8],
    bits: usize,
) where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_lanes::<T>("restrict_into_with", "src", src.len(), bits);
    check_packed_out("restrict_into_with", "dst", dst.len(), bits);
    check_packed_out("restrict_into_with", "valid", valid.len(), bits);
    let (one_w, _) = bit_words(map);
    gf1::restrict(dst, valid, one_w, src, bits, T::BYTES);
}

/// Test target lanes for image membership: `valid` marks the lanes of `src`
/// lying in the image.
///
/// `src` holds exactly `bits` `T` lanes; `valid` is exactly
/// [`bytes_for`]`(bits)` bytes with cleared tail bits. Empty rows are valid.
///
/// # Panics
///
/// Panics when `src` does not hold exactly `bits` lanes or `valid` is not
/// exactly the bitmap length, naming the operation and the failing operand.
/// Validation runs before any destination write.
#[inline]
pub fn contains_into_with<S, T>(
    valid: &mut [u8],
    map: &PreparedEmbedding<S, T>,
    src: &[u8],
    bits: usize,
) where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_lanes::<T>("contains_into_with", "src", src.len(), bits);
    check_packed_out("contains_into_with", "valid", valid.len(), bits);
    let (one_w, _) = bit_words(map);
    gf1::compress(valid, src, bits, T::BYTES, &LaneTest::ZeroOrOne(one_w));
}
