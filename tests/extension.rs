//! Smoke tests of generic extension rows over every storage shape.
//!
//! Five rows cover the extension surface: GF(49) as `t^2 - 3` and GF(7^3) as
//! `t^3 - 2` over the residue row at seven (component storage, quadratic
//! and cubic relations, norm-form and Euclid inversion), GF(p^2) as
//! `t^2 + 4` over the Mersenne fold row, a packed GF(2^16) as
//! `t^2 + t + 0x21` over the AES byte field, and a padded packed GF(2^4)
//! as `t^2 + t + omega` over GF(4). Each row checks the inverse identity
//! and the defining relation — exhaustive products, inverses, and
//! component round trips of the odd rows, GF(7^3) included, live in
//! `extension_reference` — and the byte-backed rows check byte round
//! trips and report the scalar backend through one packed call. The
//! shipped `QuadMersenne31` name is checked to be the extension row itself.

use core::any::TypeId;

use fgf::field::mersenne31::Mersenne31;
use fgf::{
    AES, Backend, ComplexMersenne31, Components, Elem, Extension, ExtensionField, ExtensionSpec,
    FieldBuffer, FieldElem, FiniteField, Packed, PackedPadded, Polynomial, QuadMersenne31,
    Relation, backend_for, ops,
};

// ---------------------------------------------------------------------------
// Rows under test
// ---------------------------------------------------------------------------

struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    // `t^2 - 3`: `4 = -3` constant first.
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;
type Base7 = FiniteField<7, 1>;

struct Gf343Spec;
impl ExtensionSpec<3> for Gf343Spec {
    type Base = Base7;
    type Storage = Components;
    const NAME: &'static str = "GF(7^3)";
    // `t^3 - 2`: `5 = -2` constant first.
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}

type Gf343 = FiniteField<7, 3, Extension<Gf343Spec, 3>>;

struct MersenneSquareSpec;
impl ExtensionSpec<2> for MersenneSquareSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "GF(p^2) over GF(2^31 - 1)";
    // `t^2 + 4`: `-4` is a nonresidue modulo the Mersenne prime, whose
    // characteristic is three modulo four.
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

type MersenneSquare = FiniteField<0x7FFF_FFFF, 2, Extension<MersenneSquareSpec, 2>>;

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;

struct Packed16Spec;
impl ExtensionSpec<2> for Packed16Spec {
    type Base = Aes8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)";
    // `t^2 + t + 0x21`: constant word first, then the linear word.
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x21, 0x01]);
}

type Packed16 = FiniteField<2, 16, Extension<Packed16Spec, 2>>;

type Gf4 = FiniteField<2, 2, Polynomial<7>>;

struct Padded4Spec;
impl ExtensionSpec<2> for Padded4Spec {
    type Base = Gf4;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(2^4)";
    // `t^2 + t + omega` with omega the generator of GF(4).
    const RELATION: Relation = Relation::from_binary_words::<Gf4>(&[2, 1]);
}

type Padded4 = FiniteField<2, 4, Extension<Padded4Spec, 2>>;

// ---------------------------------------------------------------------------
// Independent references
// ---------------------------------------------------------------------------

/// `(a0 + a1 t)(b0 + b1 t)` over GF(7) with `t^2 = 3`.
fn gf49_mul_ref(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    ((a.0 * b.0 + 3 * a.1 * b.1) % 7, (a.0 * b.1 + a.1 * b.0) % 7)
}

/// GF(2^8) product under the AES polynomial.
fn gf8_mul_ref(a: u8, b: u8) -> u8 {
    let (mut a, mut b) = (u16::from(a), u16::from(b));
    let mut product = 0u16;
    while b != 0 {
        if b & 1 == 1 {
            product ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & 0x100 != 0 {
            a ^= 0x11B;
        }
    }
    product as u8
}

/// `(a0 + a1 t)(b0 + b1 t)` over the AES field with `t^2 = t + 0x21`.
fn packed16_mul_ref(a: (u8, u8), b: (u8, u8)) -> (u8, u8) {
    let low = gf8_mul_ref(a.0, b.0);
    let cross = gf8_mul_ref(a.0, b.1) ^ gf8_mul_ref(a.1, b.0);
    let high = gf8_mul_ref(a.1, b.1);
    // `t^2 = t + 0x21` folds the high component into both.
    (low ^ gf8_mul_ref(high, 0x21), cross ^ high)
}

/// GF(4) product under `x^2 + x + 1`.
fn gf4_mul_ref(a: u8, b: u8) -> u8 {
    let (mut a, mut b) = (u16::from(a), u16::from(b));
    let mut product = 0u16;
    while b != 0 {
        if b & 1 == 1 {
            product ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & 0x4 != 0 {
            a ^= 0x7;
        }
    }
    product as u8
}

/// `(a0 + a1 t)(b0 + b1 t)` over GF(4) with `t^2 = t + omega`, `omega = 2`.
fn padded4_mul_ref(a: (u8, u8), b: (u8, u8)) -> (u8, u8) {
    let low = gf4_mul_ref(a.0, b.0);
    let cross = gf4_mul_ref(a.0, b.1) ^ gf4_mul_ref(a.1, b.0);
    let high = gf4_mul_ref(a.1, b.1);
    // `t^2 = t + omega` folds the high component into both.
    (low ^ gf4_mul_ref(high, 2), cross ^ high)
}

/// A deterministic sample generator for larger fields.
struct Sample {
    state: u64,
}

impl Sample {
    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}

// ---------------------------------------------------------------------------
// GF(49): components over the residue row, exhaustively
// ---------------------------------------------------------------------------

#[test]
fn gf49_matches_reference_exhaustively() {
    for a0 in 0..7 {
        for a1 in 0..7 {
            let a = <Gf49 as ExtensionField<2>>::from_components([
                Elem::<Base7>::from_raw(a0),
                Elem::<Base7>::from_raw(a1),
            ]);
            for b0 in 0..7 {
                for b1 in 0..7 {
                    let b = <Gf49 as ExtensionField<2>>::from_components([
                        Elem::<Base7>::from_raw(b0),
                        Elem::<Base7>::from_raw(b1),
                    ]);
                    let expected = gf49_mul_ref((a0, a1), (b0, b1));
                    let components = <Gf49 as ExtensionField<2>>::to_components(a * b);
                    assert_eq!(components[0].to_raw(), expected.0);
                    assert_eq!(components[1].to_raw(), expected.1);
                }
            }
        }
    }
}

#[test]
fn gf49_inverses_and_relation() {
    const ONE: Elem<Gf49> = Elem::<Gf49>::ONE;
    for a0 in 0..7 {
        for a1 in 0..7 {
            let a = <Gf49 as ExtensionField<2>>::from_components([
                Elem::<Base7>::from_raw(a0),
                Elem::<Base7>::from_raw(a1),
            ]);
            // `a * a^-1 = 1` for every nonzero element; zero maps to zero.
            assert_eq!(a * a.inv(), if a.is_zero() { Elem::ZERO } else { ONE });
            // Components round-trip.
            let components = <Gf49 as ExtensionField<2>>::to_components(a);
            assert_eq!(<Gf49 as ExtensionField<2>>::from_components(components), a);
        }
    }
    let t = <Gf49 as ExtensionField<2>>::from_components([Elem::<Base7>::ZERO, Elem::<Base7>::ONE]);
    // `t^2 = 3`, so `t^2 + 4 = 0` over GF(7).
    let square = <Gf49 as ExtensionField<2>>::to_components(t * t);
    assert_eq!(square[0].to_raw(), 3);
    assert_eq!(square[1].to_raw(), 0);
    let four = <Gf49 as ExtensionField<2>>::from_components([
        Elem::<Base7>::from_raw(4),
        Elem::<Base7>::ZERO,
    ]);
    assert_eq!(t.square() + four, Elem::<Gf49>::ZERO);
}

#[test]
fn gf49_zero_and_division_conventions() {
    let a = <Gf49 as ExtensionField<2>>::from_components([
        Elem::<Base7>::from_raw(3),
        Elem::<Base7>::from_raw(5),
    ]);
    assert!(Elem::<Gf49>::ZERO.inv().is_zero());
    assert_eq!(a / Elem::<Gf49>::ZERO, Elem::<Gf49>::ZERO);
    assert_eq!(Elem::<Gf49>::ZERO.pow(0), Elem::<Gf49>::ONE);
    assert_eq!(a.pow(0), Elem::<Gf49>::ONE);
    // Distributivity and negation.
    assert_eq!(a * (a.square() + a), a * a.square() + a * a);
    assert_eq!(a + -a, Elem::<Gf49>::ZERO);
    // The full multiplicative order divides the group order.
    assert_eq!(a.pow(49 - 1), Elem::<Gf49>::ONE);
}

#[test]
fn gf49_bytes_and_backend() {
    assert_eq!(<Gf49 as FieldBuffer>::BYTES, 16);
    let a = <Gf49 as ExtensionField<2>>::from_components([
        Elem::<Base7>::from_raw(6),
        Elem::<Base7>::from_raw(2),
    ]);
    let mut bytes = [0u8; 16];
    <Gf49 as FieldBuffer>::encode(&mut bytes, a);
    assert_eq!(<Gf49 as FieldBuffer>::decode(&bytes), a);

    // One packed call through the checked surface, reporting the scalar
    // backend.
    assert_eq!(backend_for::<Gf49>(), Backend::Scalar);
    let coeff = a.inv();
    let mut row = [0u8; 32];
    ops::pack(&mut row, &[Elem::<Gf49>::ONE, a]);
    let before = row;
    ops::mul_add(&mut row, coeff, &before);
    let mut unpacked = [Elem::<Gf49>::ZERO; 2];
    ops::unpack(&mut unpacked, &row);
    // `mul_add` accumulates: `1 + coeff` and `a + coeff*a`.
    assert_eq!(unpacked[0], Elem::<Gf49>::ONE + coeff);
    assert_eq!(unpacked[1], a + coeff * a);
}

// ---------------------------------------------------------------------------
// GF(7^3): the Euclid inversion path at relative degree three
// ---------------------------------------------------------------------------

#[test]
fn gf343_relation_holds() {
    // Exhaustive products, inverses, and component round trips against
    // the independent reference live in `extension_reference`.
    let two = <Gf343 as ExtensionField<3>>::from_components([
        Elem::<Base7>::from_raw(2),
        Elem::<Base7>::ZERO,
        Elem::<Base7>::ZERO,
    ]);
    // The defining relation `t^3 = 2` holds at `t`.
    let t = <Gf343 as ExtensionField<3>>::from_components([
        Elem::<Base7>::ZERO,
        Elem::<Base7>::ONE,
        Elem::<Base7>::ZERO,
    ]);
    assert_eq!(t * t * t, two);
}

// ---------------------------------------------------------------------------
// GF(p^2) over the Mersenne fold row
// ---------------------------------------------------------------------------

#[test]
fn mersenne_square_inverses_relation_bytes() {
    const ONE: Elem<MersenneSquare> = Elem::<MersenneSquare>::ONE;
    assert_eq!(<MersenneSquare as FieldBuffer>::BYTES, 8);
    let mut sample = Sample { state: 0xC0FF_EE01 };
    for _ in 0..500 {
        let re = (sample.next() % 0x7FFF_FFFF) as u32;
        let im = (sample.next() % 0x7FFF_FFFF) as u32;
        let a = <MersenneSquare as ExtensionField<2>>::from_components([
            Elem::<Mersenne31>::from_raw(re),
            Elem::<Mersenne31>::from_raw(im),
        ]);
        assert_eq!(a * a.inv(), if a.is_zero() { Elem::ZERO } else { ONE });
        let mut bytes = [0u8; 8];
        <MersenneSquare as FieldBuffer>::encode(&mut bytes, a);
        assert_eq!(<MersenneSquare as FieldBuffer>::decode(&bytes), a);
    }
    // The defining relation `t^2 = -4`.
    let t = <MersenneSquare as ExtensionField<2>>::from_components([
        Elem::<Mersenne31>::ZERO,
        Elem::<Mersenne31>::ONE,
    ]);
    let four = <MersenneSquare as ExtensionField<2>>::from_components([
        Elem::<Mersenne31>::from_raw(4),
        Elem::<Mersenne31>::ZERO,
    ]);
    assert_eq!(t * t, -four);
}

// ---------------------------------------------------------------------------
// Packed GF(2^16) over the AES byte field
// ---------------------------------------------------------------------------

#[test]
fn packed16_matches_reference_exhaustively() {
    // Every element against the reference on rotated partners, and the
    // full field on the inverse identity.
    const ONE: Elem<Packed16> = Elem::<Packed16>::ONE;
    for word in 0u16..=u16::MAX {
        let a = Elem::<Packed16>::from_raw(word);
        assert_eq!(a * a.inv(), if a.is_zero() { Elem::ZERO } else { ONE });
        let partner = word.rotate_left(7);
        let b = Elem::<Packed16>::from_raw(partner);
        let expected = packed16_mul_ref(
            (word as u8, (word >> 8) as u8),
            (partner as u8, (partner >> 8) as u8),
        );
        assert_eq!(
            <Packed16 as ExtensionField<2>>::to_components(a * b),
            [
                Elem::<Aes8>::from_raw(expected.0),
                Elem::<Aes8>::from_raw(expected.1),
            ]
        );
    }
}

#[test]
fn packed16_relation_bytes_backend() {
    // `t^2 + t + 0x21 = 0` at `t`.
    let t =
        <Packed16 as ExtensionField<2>>::from_components([Elem::<Aes8>::ZERO, Elem::<Aes8>::ONE]);
    let constant = <Packed16 as ExtensionField<2>>::from_components([
        Elem::<Aes8>::from_raw(0x21),
        Elem::<Aes8>::ZERO,
    ]);
    assert_eq!(t.square() + t + constant, Elem::<Packed16>::ZERO);

    // Component round trip through the packed word.
    let components = <Packed16 as ExtensionField<2>>::to_components(t);
    assert_eq!(
        <Packed16 as ExtensionField<2>>::from_components(components),
        t
    );

    // The byte form is the little-endian word.
    assert_eq!(<Packed16 as FieldBuffer>::BYTES, 2);
    let mut bytes = [0u8; 2];
    <Packed16 as FieldBuffer>::encode(&mut bytes, t);
    assert_eq!(bytes, 0x0100u16.to_le_bytes());
    assert_eq!(<Packed16 as FieldBuffer>::decode(&bytes), t);

    // One packed call through the checked surface, reporting the scalar
    // backend.
    assert_eq!(backend_for::<Packed16>(), Backend::Scalar);
    let mut row = [0u8; 4];
    ops::pack(&mut row, &[Elem::<Packed16>::ONE, t]);
    let before = row;
    ops::mul_add(&mut row, t, &before);
    let mut unpacked = [Elem::<Packed16>::ZERO; 2];
    ops::unpack(&mut unpacked, &row);
    // `mul_add` accumulates: `1 + t` and `t + t^2`.
    assert_eq!(unpacked[0], Elem::<Packed16>::ONE + t);
    assert_eq!(unpacked[1], t + t * t);
}

// ---------------------------------------------------------------------------
// Padded packed GF(2^4) over GF(4)
// ---------------------------------------------------------------------------

#[test]
fn padded4_matches_reference_and_clears_padding() {
    for word in 0u8..=0x0F {
        let a = Elem::<Padded4>::from_raw(word);
        assert_eq!(a.to_raw(), word);
        // Construction clears padding bits.
        assert_eq!(Elem::<Padded4>::from_raw(word | 0xF0).to_raw(), word);
        for other in 0u8..=0x0F {
            let b = Elem::<Padded4>::from_raw(other);
            let expected = padded4_mul_ref((word & 3, word >> 2), (other & 3, other >> 2));
            assert_eq!(
                <Padded4 as ExtensionField<2>>::to_components(a * b),
                [
                    Elem::<Gf4>::from_raw(expected.0),
                    Elem::<Gf4>::from_raw(expected.1),
                ]
            );
        }
        assert_eq!(
            a * a.inv(),
            if a.is_zero() {
                Elem::ZERO
            } else {
                Elem::<Padded4>::ONE
            }
        );
    }
    // The defining relation `t^2 + t + omega = 0` at `t`.
    let t = <Padded4 as ExtensionField<2>>::from_components([Elem::<Gf4>::ZERO, Elem::<Gf4>::ONE]);
    let omega = <Padded4 as ExtensionField<2>>::from_components([
        Elem::<Gf4>::from_raw(2),
        Elem::<Gf4>::ZERO,
    ]);
    assert_eq!(t.square() + t + omega, Elem::<Padded4>::ZERO);
}

// ---------------------------------------------------------------------------
// The packed surface over the checked wrappers
// ---------------------------------------------------------------------------

/// Exercise the checked packed surface over one odd-characteristic and one
/// packed binary row, against element-level expectations.
#[test]
fn packed_surface_matches_element_arithmetic() {
    let rows: Vec<(Elem<Gf49>, Elem<Gf49>)> = [(6u64, 2u64), (3, 5), (1, 0)]
        .iter()
        .map(|&(re, im)| {
            (
                <Gf49 as ExtensionField<2>>::from_components([
                    Elem::<Base7>::from_raw(re),
                    Elem::<Base7>::from_raw(im),
                ]),
                <Gf49 as ExtensionField<2>>::from_components([
                    Elem::<Base7>::from_raw(im),
                    Elem::<Base7>::from_raw(re),
                ]),
            )
        })
        .collect();

    const LANE: usize = <Gf49 as FieldBuffer>::BYTES;
    let mut buffer = [0u8; 3 * LANE];
    let flat: Vec<Elem<Gf49>> = rows.iter().map(|pair| pair.0).collect();
    ops::pack(&mut buffer, &flat);

    // Elementwise addition and subtraction.
    let mut sum = buffer;
    ops::add_assign::<Gf49>(&mut sum, &buffer);
    let mut double = [Elem::<Gf49>::ZERO; 3];
    ops::unpack(&mut double, &sum);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, pair.0 + pair.0);
    }
    ops::sub_assign::<Gf49>(&mut sum, &buffer);
    ops::unpack(&mut double, &sum);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, pair.0);
    }

    // Broadcast-scalar addition and subtraction.
    let shift = rows[1].1;
    ops::add_assign_scalar::<Gf49>(&mut sum, shift);
    ops::unpack(&mut double, &sum);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, pair.0 + shift);
    }
    ops::sub_assign_scalar::<Gf49>(&mut sum, shift);
    ops::unpack(&mut double, &sum);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, pair.0);
    }

    // Overwrite scaling and accumulating scaling.
    let mut scaled = buffer;
    ops::mul_into::<Gf49>(&mut scaled, shift, &buffer);
    ops::unpack(&mut double, &scaled);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, shift * pair.0);
    }
    ops::mul_add::<Gf49>(&mut scaled, shift, &buffer);
    ops::unpack(&mut double, &scaled);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, shift * pair.0 + shift * pair.0);
    }
    ops::mul_assign::<Gf49>(&mut scaled, shift);
    ops::unpack(&mut double, &scaled);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, shift * (shift * pair.0 + shift * pair.0));
    }

    // Elementwise multiplication.
    let mut product = buffer;
    ops::mul_elementwise::<Gf49>(&mut product, &buffer, &buffer);
    ops::unpack(&mut double, &product);
    for (element, pair) in double.iter().zip(&rows) {
        assert_eq!(*element, pair.0 * pair.0);
    }

    // The scatter shape: two coefficients from one source row.
    let mut scattered = [0u8; 2 * LANE];
    ops::mul_add_scatter::<Gf49>(&mut scattered, LANE, &flat[1..3], &buffer[..LANE]);
    ops::unpack(&mut double[..2], &scattered);
    assert_eq!(double[0], flat[1] * flat[0]);
    assert_eq!(double[1], flat[2] * flat[0]);

    // The gather shape: two sources into one row.
    let mut gathered = [0u8; LANE];
    ops::mul_add_gather::<Gf49>(
        &mut gathered,
        &flat[1..3],
        &[&buffer[..LANE], &buffer[LANE..2 * LANE]],
    );
    ops::unpack(&mut double[..1], &gathered);
    assert_eq!(double[0], flat[1] * flat[0] + flat[2] * flat[1]);

    // The matrix shape: one term of three coefficients into three
    // two-element rows, overwriting the destination.
    let mut matrix = [0u8; 3 * 2 * LANE];
    ops::mul_into_matrix::<Gf49>(
        &mut matrix,
        2 * LANE,
        3,
        &[(&flat[0..3], &buffer[..2 * LANE])],
    );
    let mut columns = [Elem::<Gf49>::ZERO; 6];
    ops::unpack(&mut columns, &matrix);
    for row in 0..3 {
        assert_eq!(columns[2 * row], flat[row] * flat[0]);
        assert_eq!(columns[2 * row + 1], flat[row] * flat[1]);
    }

    // The offset-addressed gather matches in-place addition.
    let mut region = [0u8; 4 * LANE];
    region[..2 * LANE].copy_from_slice(&buffer[..2 * LANE]);
    let source = region;
    ops::add_gather_offsets::<Gf49>(&mut region[2 * LANE..], &source, &[0, 0]);
    ops::unpack(&mut double[..2], &region[2 * LANE..]);
    assert_eq!(double[0], flat[0] + flat[0]);
    assert_eq!(double[1], flat[1] + flat[1]);

    // The same surface on the packed binary row: XOR addition and word
    // lanes.
    let packed_rows = [
        Elem::<Packed16>::from_raw(0x1234),
        Elem::<Packed16>::from_raw(0x0F0F),
    ];
    let mut packed_buffer = [0u8; 4];
    ops::pack(&mut packed_buffer, &packed_rows);
    let packed_source = packed_buffer;
    ops::add_assign::<Packed16>(&mut packed_buffer, &packed_source);
    let mut unpacked = [Elem::<Packed16>::ZERO; 2];
    ops::unpack(&mut unpacked, &packed_buffer);
    assert_eq!(unpacked[0].to_raw(), 0);
    assert_eq!(unpacked[1].to_raw(), 0);
    ops::mul_elementwise::<Packed16>(&mut packed_buffer, &packed_source, &packed_source);
    ops::unpack(&mut unpacked, &packed_buffer);
    let first = Elem::<Packed16>::from_raw(0x1234);
    let second = Elem::<Packed16>::from_raw(0x0F0F);
    assert_eq!(unpacked[0], first * first);
    assert_eq!(unpacked[1], second * second);
}

/// `QuadMersenne31` names the quadratic extension row over Mersenne31; it
/// is the same type as the spelled-out row, not a distinct marker.
#[test]
fn quad_mersenne31_is_the_spelled_extension_row() {
    type Spelled = FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>;
    // Compiles only when the two spellings are one type.
    fn through_spelled(value: Elem<Spelled>) -> Elem<Spelled> {
        value
    }
    let x = Elem::<QuadMersenne31>::from_raw([3, 5]);
    let y: Elem<QuadMersenne31> = through_spelled(x);
    assert_eq!(y, x);
    assert_eq!(TypeId::of::<QuadMersenne31>(), TypeId::of::<Spelled>());
    assert_eq!(
        TypeId::of::<Elem<QuadMersenne31>>(),
        TypeId::of::<Elem<Spelled>>()
    );
}
