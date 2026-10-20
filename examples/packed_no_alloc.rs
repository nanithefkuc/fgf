//! Packed operations without the heap, at the widest prime lane.
//!
//! Run with `just example packed_no_alloc`, optionally with
//! `--no-default-features`. The largest prime below 2^64 keeps one
//! canonical residue in an eight-byte lane: `from_raw` reduces any word
//! into the canonical range, `from_residue` checks the range instead, and
//! the byte form is the little-endian lane. Every packed operation runs
//! from stack buffers with one coefficient prepared once and reused across
//! calls, so the library's `alloc` feature stays off.
//!
//! The Goldilocks modulus also exists under the generic residue
//! presentation. The two rows carry different reduction strategies and
//! different names; values move between them by residue, and both agree
//! value for value.

#![forbid(unsafe_code)]

use fgf::{Elem, Field, FieldBuffer, FiniteField, Goldilocks, backend_for, ops};

/// The largest `u64` prime: eight-byte lanes, no fold identities.
type WidePrime = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1>;
/// The Goldilocks modulus `2^64 - 2^32 + 1` under the generic residue
/// presentation, the default for a prime row.
type GldResidue = FiniteField<0xFFFF_FFFF_0000_0001, 1>;

const LANES: usize = 6;
/// Lane width of the wide prime row, asserted below.
const LANE_BYTES: usize = 8;

fn main() {
    lanes();
    residue_and_fold();
}

/// Eight-byte lanes: checked and reducing construction, and packed
/// operations from stack buffers with one reusable prepared coefficient.
fn lanes() {
    // Reducing construction folds any word into the canonical range: the
    // modulus itself lands on zero, and `u64::MAX` lands on its excess 58.
    assert_eq!(
        Elem::<WidePrime>::from_raw(0xFFFF_FFFF_FFFF_FFC5).to_raw(),
        0
    );
    assert_eq!(Elem::<WidePrime>::from_raw(u64::MAX).to_raw(), 58);

    // Checked construction rejects out-of-range words instead of
    // narrowing them.
    assert!(Elem::<WidePrime>::from_residue(u64::MAX).is_err());
    let x = Elem::<WidePrime>::from_residue(12_345_678).unwrap();
    assert_eq!(x.to_residue(), Some(12_345_678));

    // The lane is the eight-byte little-endian residue.
    assert_eq!(<WidePrime as FieldBuffer>::BYTES, LANE_BYTES);
    let mut lane = [0u8; LANE_BYTES];
    <WidePrime as FieldBuffer>::encode(&mut lane, x);
    assert_eq!(lane, 12_345_678u64.to_le_bytes());
    assert_eq!(<WidePrime as FieldBuffer>::decode(&lane), x);

    // One prepared coefficient, two packed calls, zero allocation.
    let xs: [Elem<WidePrime>; LANES] = std::array::from_fn(|i| {
        Elem::<WidePrime>::from_raw((i as u64 + 1) * 0x0123_4567_89AB_CDEF)
    });
    let k = Elem::<WidePrime>::from_residue(0x0123_4567_89AB_CDEF).unwrap();
    let prepared = ops::Coeff::<WidePrime>::new(k);

    const WIDTH: usize = LANES * LANE_BYTES;
    let mut source = [0u8; WIDTH];
    ops::pack::<WidePrime>(&mut source, &xs);

    // `mul_into_with` overwrites; `mul_add_with` accumulates onto zero
    // with the same prepared coefficient.
    let mut scaled = [0u8; WIDTH];
    ops::mul_into_with::<WidePrime>(&mut scaled, &prepared, &source);
    let mut acc = [0u8; WIDTH];
    ops::mul_add_with::<WidePrime>(&mut acc, &prepared, &source);
    assert_eq!(scaled, acc);

    let mut lanes = [Elem::<WidePrime>::ZERO; LANES];
    ops::unpack::<WidePrime>(&mut lanes, &acc);
    for (value, &x) in lanes.iter().zip(&xs) {
        assert_eq!(*value, k * x, "prepared coefficient agrees per lane");
    }

    // Elementwise products agree with scalar arithmetic lane by lane.
    let mut product = [0u8; WIDTH];
    ops::mul_elementwise::<WidePrime>(&mut product, &source, &source);
    ops::unpack::<WidePrime>(&mut lanes, &product);
    for (value, &x) in lanes.iter().zip(&xs) {
        assert_eq!(*value, x.square(), "elementwise product agrees per lane");
    }

    // The total conventions hold at this width: `inv(0) == 0` and
    // `x / 0 == 0`.
    assert_eq!(Elem::<WidePrime>::ZERO.inv(), Elem::<WidePrime>::ZERO);
    assert_eq!(x.div(Elem::<WidePrime>::ZERO), Elem::<WidePrime>::ZERO);

    println!(
        "{} ({:?}): eight-byte lanes, prepared coefficient reused, no allocation",
        WidePrime::NAME,
        backend_for::<WidePrime>()
    );
}

/// The generic residue row and the named fold row at one modulus.
fn residue_and_fold() {
    // Both rows reduce the same word to the same canonical residue: the
    // conditional subtract, since `2^64 < 2p`.
    let residue = Elem::<GldResidue>::from_raw(u64::MAX);
    let folded = Elem::<Goldilocks>::from_raw(u64::MAX);
    assert_eq!(residue.to_raw(), folded.to_raw());
    assert_eq!(folded.to_raw(), (1u64 << 32) - 2);

    // Values move by residue, never by buffer.
    let moved = Elem::<Goldilocks>::from_residue(residue.to_raw()).unwrap();
    assert_eq!(moved, folded);

    // Arithmetic agrees value for value across the presentations.
    let (c, d) = (
        Elem::<Goldilocks>::from_raw(0xFFFF_FFFF),
        Elem::<Goldilocks>::from_raw(0x1_0000_0000),
    );
    let rc = Elem::<GldResidue>::from_residue(c.to_raw()).unwrap();
    let rd = Elem::<GldResidue>::from_residue(d.to_raw()).unwrap();
    assert_eq!((rc * rd).to_raw(), (c * d).to_raw());

    // One fold fact the plain row cannot name: `(2^32 - 1)^2` reduces
    // through `2^64 = 2^32 - 1` to `2^64 - 2^33 + 1`.
    let eps = Elem::<Goldilocks>::from_residue((1u64 << 32) - 1).unwrap();
    assert_eq!(eps.square().to_raw(), 0xFFFF_FFFE_0000_0001);

    // The named row keeps its own label.
    println!(
        "{} and {} agree by residue at one modulus",
        GldResidue::NAME,
        Goldilocks::NAME
    );
}
