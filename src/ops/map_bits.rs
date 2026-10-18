//! Prepared field-map rows over the bit-packed GF(2) representation.
//!
//! The bit-packed counterparts of the lane rows in [`super::map`]: the GF(2)
//! endpoint rides as packed bits in [`crate::bits`] LSB-first layout while the
//! binary target rides as stable little-endian lanes. The embedding rows read
//! the precomputed `(one_w, trace_word)` pair from the prepared map, and the
//! isomorphism rows read the image of one through the scalar `apply`, so row
//! output equals the per-element scalar loop by construction.
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

use crate::bits::{bytes_for, check_holds};
use crate::field::{BinaryField, Field, FieldBuffer, PrimeCharacteristic, PrimeField};
use crate::kernel::gf1::{self, LaneTest};
use crate::{Elem, PreparedEmbedding, PreparedIsomorphism};

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

/// The `T` coordinate word of the image of one under a characteristic-two
/// isomorphism whose source endpoint rides as packed bits.
///
/// Bit `i` of the packed side selects the image of zero or one, so the row
/// needs only the image of one: the zero element maps to the zero lane and
/// the one element to `one_w`. `T` is `BinaryField`, so every canonical
/// lane is a coordinate word and the kernel seam reads it straight from
/// the lane bytes. Read through the scalar `apply`, so the row equals the
/// per-element scalar loop on the packed endpoint by construction.
#[inline]
fn iso_one_word<S, T>(map: &PreparedIsomorphism<S, T>) -> u64
where
    S: Field<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let mut lane = [0u8; 8];
    T::encode(&mut lane[..T::BYTES], map.apply(Elem::<S>::ONE));
    let mut word = 0u64;
    for (shift, &byte) in lane[..T::BYTES].iter().enumerate() {
        word |= u64::from(byte) << (8 * shift);
    }
    word
}

/// Apply a characteristic-two isomorphism across one row: `dst[i]` becomes
/// the image of bit `i` of `src`.
///
/// The bit-packed counterpart of [`super::map::apply_into_with`] for an
/// endpoint that rides as packed bits: bit `i` selects the image of zero
/// or one, written as the `T` lane beside it. A degree-one char-2
/// isomorphism between bit-packed GF(2) and a byte-lane GF(2)
/// presentation (e.g. polynomial GF(2) beside eight-byte residue GF(2))
/// has no [`FieldBuffer`] form on the packed side, so the packed endpoint
/// rides as bits here; wider char-2 endpoints ride the same bridge with
/// their zero/one images.
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
pub fn apply_into_with<S, T>(
    dst: &mut [u8],
    map: &PreparedIsomorphism<S, T>,
    src: &[u8],
    bits: usize,
) where
    S: Field<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_holds("apply_into_with", "src", src.len(), bits);
    check_lanes::<T>("apply_into_with", "dst", dst.len(), bits);
    let one_w = iso_one_word(map);
    gf1::expand(dst, one_w, src, bits, T::BYTES);
}
/// Unapply a characteristic-two isomorphism across one row: bit `i` of
/// `dst` is the preimage of `src[i]`.
///
/// The bit-packed counterpart of [`super::map::unapply_into_with`]: each
/// `T` lane is decoded and run through [`PreparedIsomorphism::unapply`],
/// and the preimage-equal-to-one bit is written packed LSB-first. On the
/// degree-one bridge every canonical lane is 0 or one, so non-member lanes
/// cannot occur; wider endpoints compress with the same predicate the
/// scalar loop uses. `src` holds exactly `bits` `T` lanes; `dst` is
/// exactly [`bytes_for`]`(bits)` bytes with cleared tail bits. Empty rows
/// are valid.
///
/// # Panics
///
/// Panics when `src` does not hold exactly `bits` lanes or `dst` is not
/// exactly the bitmap length, naming the operation and the failing operand.
/// Validation runs before any destination write.
#[inline]
pub fn unapply_into_with<S, T>(
    dst: &mut [u8],
    map: &PreparedIsomorphism<S, T>,
    src: &[u8],
    bits: usize,
) where
    S: Field<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    check_lanes::<T>("unapply_into_with", "src", src.len(), bits);
    check_packed_out("unapply_into_with", "dst", dst.len(), bits);
    dst.fill(0);
    for (i, lane) in src.chunks_exact(T::BYTES).enumerate().take(bits) {
        let bit = u8::from(map.unapply(T::decode(lane)) == Elem::<S>::ONE);
        dst[i / 8] |= bit << (i % 8);
    }
}
