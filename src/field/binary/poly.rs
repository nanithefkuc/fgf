//! Polynomial-basis binary presentations at degrees 1, 2, 4, and 8.
//!
//! [`Polynomial`] is the construction `GF(2)[x] / p(x)` for an irreducible
//! degree-`N` `p`. The representation, its descriptor, and its generator
//! derive from the polynomial at compile time, so each polynomial is a
//! distinct representation with fully `const` scalar arithmetic.
//!
//! | Constant | Polynomial | Convention |
//! | --- | --- | --- |
//! | [`AES`] | `0x11B` | AES/Rijndael; the field `GF2P8MULB` multiplies natively |
//! | [`RS`] | `0x11D` | Intel ISA-L, `klauspost/reedsolomon`, QR codes |
//!
//! Any other degree-8 irreducible polynomial is spelled directly, such as
//! `Polynomial<0x12D>` for Data Matrix or `Polynomial<0x187>` for the CCSDS Reed–Solomon
//! field. Distinct polynomials are unrelated encodings: a byte has different
//! products under each, so a buffer carries one convention and never two.
//!
//! ```
//! use fgf::{AES, Elem, Binary, Polynomial, RS};
//!
//! // The same bytes multiply differently under the two conventions.
//! let (a, b) = (0x53, 0xca);
//! assert_eq!(
//!     Elem::<Binary<8, Polynomial<AES>>>::from_raw(a)
//!         .mul(Elem::<Binary<8, Polynomial<AES>>>::from_raw(b))
//!         .to_raw(),
//!     0x01
//! );
//! assert_eq!(
//!     Elem::<Binary<8, Polynomial<RS>>>::from_raw(a)
//!         .mul(Elem::<Binary<8, Polynomial<RS>>>::from_raw(b))
//!         .to_raw(),
//!     0x8f
//! );
//!
//! // Any irreducible polynomial is a field; its generator is derived.
//! assert_eq!(
//!     Elem::<Binary<8, Polynomial<0x12D>>>::GENERATOR.pow(255),
//!     Elem::<Binary<8, Polynomial<0x12D>>>::ONE
//! );
//! ```
//!
//! ```compile_fail,E0080
//! // x^8 + 1 = (x + 1)^8 is reducible, so this field does not exist.
//! let _ = fgf::Elem::<fgf::Binary<8, fgf::Polynomial<0x101>>>::from_raw(1);
//! ```
//!
//! ```compile_fail,E0080
//! // The degree-eight polynomial does not match a degree-four field.
//! let _ = fgf::Elem::<fgf::Binary<4, fgf::Polynomial<0x11B>>>::from_raw(1);
//! ```

use super::description::{BinaryDescription, ByteLogExp};
use super::{BinaryRepr, private};

/// The AES/Rijndael polynomial `x^8 + x^4 + x^3 + x + 1`.
///
/// It is the field the x86 `GF2P8MULB` instruction implements, so
/// `Binary<8, Polynomial<AES>>` multiplies with one instruction per vector
/// on GFNI hosts. Its derived generator is `0x03`.
pub const AES: u128 = 0x11B;

/// The `0x11D` Reed–Solomon convention, `x^8 + x^4 + x^3 + x^2 + 1`.
///
/// Intel ISA-L, `klauspost/reedsolomon`, and QR codes use this polynomial;
/// `Binary<8, Polynomial<RS>>` shards are byte-identical to those ecosystems.
/// Reed–Solomon codes also use other polynomials. Its derived generator is
/// `0x02`.
pub const RS: u128 = 0x11D;

/// The polynomial-basis representation of GF(2^N) under the reduction
/// polynomial `P`.
///
/// `P` is the full polynomial including the leading term, and must be
/// irreducible of the field's degree; the check runs when the
/// representation's constants are evaluated. The same type spells GF(2) as
/// [`Polynomial<3>`](Polynomial), the polynomial `x + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Polynomial<const P: u128>;

impl<const P: u128> private::Sealed for Polynomial<P> {}

impl<const P: u128> Polynomial<P> {
    /// The full reduction polynomial including the leading term.
    pub const POLY: u128 = P;

    /// The low byte `XORed` in when a shift overflows the field.
    ///
    /// Read by the kernel differential tests as their independent oracle;
    /// non-`simd` builds have no other user.
    #[allow(dead_code)]
    pub(crate) const REDUCTION_LOW: u8 = (P & 0xFF) as u8;

    /// The bytes of the degree-eight field name, `"GF(2^8)/0x1XY"`.
    const NAME_BYTES8: [u8; 13] = name_bytes8(P);
    /// The bytes of the degree-four field name, `"GF(2^4)/0x1X"`.
    const NAME_BYTES4: [u8; 12] = name_bytes4(P);
    /// The bytes of the degree-two field name, `"GF(2^2)/0xX"`.
    const NAME_BYTES2: [u8; 11] = name_bytes2(P);

    /// The smallest primitive element of the described field.
    ///
    /// An element generates the group exactly when no proper cofactor power
    /// returns it to one. The search starts at two, so the AES field finds
    /// `0x03` and the Reed–Solomon field `0x02`, matching the frozen
    /// conventions.
    pub(crate) const fn smallest_generator(description: &BinaryDescription) -> u64 {
        let limit = 1u64 << description.degree();
        let mut candidate = 2u64;
        loop {
            assert!(candidate < limit, "Polynomial P has no primitive element");
            if description.is_generator(candidate) {
                break candidate;
            }
            candidate += 1;
        }
    }
}

/// The bytes of the degree-eight field name: `"GF(2^8)/0x1XY"`.
const fn name_bytes8(poly: u128) -> [u8; 13] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut name = *b"GF(2^8)/0x000";
    name[10] = HEX[((poly >> 8) & 0xf) as usize];
    name[11] = HEX[((poly >> 4) & 0xf) as usize];
    name[12] = HEX[(poly & 0xf) as usize];
    name
}

/// The bytes of the degree-four field name: `"GF(2^4)/0x1X"`.
const fn name_bytes4(poly: u128) -> [u8; 12] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut name = *b"GF(2^4)/0x00";
    name[10] = HEX[((poly >> 4) & 0xf) as usize];
    name[11] = HEX[(poly & 0xf) as usize];
    name
}

/// The bytes of the degree-two field name: `"GF(2^2)/0xX"`.
const fn name_bytes2(poly: u128) -> [u8; 11] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut name = *b"GF(2^2)/0x0";
    name[10] = HEX[(poly & 0xf) as usize];
    name
}

macro_rules! poly_degree8 {
    () => {
        impl<const P: u128> BinaryRepr<8> for Polynomial<P> {
            const NAME: &'static str = match core::str::from_utf8(&Polynomial::<P>::NAME_BYTES8) {
                Ok(name) => name,
                Err(_) => panic!("Polynomial field names are ASCII"),
            };
            const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(8, P);
            const LOG_EXP: Option<&'static ByteLogExp> = Some(&ByteLogExp::build(
                &BinaryDescription::polynomial(8, P),
                Polynomial::<P>::smallest_generator(&BinaryDescription::polynomial(8, P)),
            ));
            const VALID: () = {
                assert!(
                    super::description::poly_degree(P) == 8,
                    "polynomial degree does not match field degree"
                );
                <Self as BinaryRepr<8>>::DESCRIPTION.validate();
                assert!(
                    <Self as BinaryRepr<8>>::DESCRIPTION.is_generator(
                        Polynomial::<P>::smallest_generator(<Self as BinaryRepr<8>>::DESCRIPTION)
                    ),
                    "byte representation generator does not have full order"
                );
            };
        }
    };
}

poly_degree8!();

impl<const P: u128> BinaryRepr<4> for Polynomial<P> {
    const NAME: &'static str = match core::str::from_utf8(&Polynomial::<P>::NAME_BYTES4) {
        Ok(name) => name,
        Err(_) => panic!("Polynomial field names are ASCII"),
    };
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(4, P);
    const LOG_EXP: Option<&'static ByteLogExp> = None;
    const VALID: () = {
        assert!(
            super::description::poly_degree(P) == 4,
            "polynomial degree does not match field degree"
        );
        <Self as BinaryRepr<4>>::DESCRIPTION.validate();
    };
}

impl<const P: u128> BinaryRepr<2> for Polynomial<P> {
    const NAME: &'static str = match core::str::from_utf8(&Polynomial::<P>::NAME_BYTES2) {
        Ok(name) => name,
        Err(_) => panic!("Polynomial field names are ASCII"),
    };
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(2, P);
    const LOG_EXP: Option<&'static ByteLogExp> = None;
    const VALID: () = {
        assert!(
            super::description::poly_degree(P) == 2,
            "polynomial degree does not match field degree"
        );
        <Self as BinaryRepr<2>>::DESCRIPTION.validate();
    };
}

impl<const P: u128> BinaryRepr<1> for Polynomial<P> {
    const NAME: &'static str = "GF(2)";
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(1, P);
    const LOG_EXP: Option<&'static ByteLogExp> = None;
    const VALID: () = {
        assert!(P == 3, "Polynomial<3> is the only degree-one form");
        <Self as BinaryRepr<1>>::DESCRIPTION.validate();
    };
}

use super::super::HasGenerator;
use super::Binary;

impl<const P: u128> HasGenerator for Binary<8, Polynomial<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Polynomial::<P>::smallest_generator(<Polynomial<P> as BinaryRepr<8>>::DESCRIPTION);
        assert!(
            <Polynomial<P> as BinaryRepr<8>>::DESCRIPTION.is_generator(g),
            "byte representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Binary<4, Polynomial<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Polynomial::<P>::smallest_generator(<Polynomial<P> as BinaryRepr<4>>::DESCRIPTION);
        assert!(
            <Polynomial<P> as BinaryRepr<4>>::DESCRIPTION.is_generator(g),
            "representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Binary<2, Polynomial<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Polynomial::<P>::smallest_generator(<Polynomial<P> as BinaryRepr<2>>::DESCRIPTION);
        assert!(
            <Polynomial<P> as BinaryRepr<2>>::DESCRIPTION.is_generator(g),
            "representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Binary<1, Polynomial<P>> {
    const GENERATOR_RAW: u8 = 1;
}

#[cfg(test)]
pub(super) mod tests {
    extern crate std;

    use core::hint::black_box;

    use super::super::description::poly_irreducible;
    use super::*;

    /// Independent shift-reduce product in `GF(2)[x] / poly`, degree `n`.
    pub(in crate::field::binary) fn oracle_mul(poly: u128, n: u32, x: u64, y: u64) -> u64 {
        let mut acc: u128 = 0;
        for k in 0..n {
            if (y >> k) & 1 == 1 {
                acc ^= u128::from(x) << k;
            }
        }
        for k in (n..2 * n).rev() {
            if (acc >> k) & 1 == 1 {
                acc ^= poly << (k - n);
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        let product = acc as u64;
        product
    }

    /// The smallest element of `GF(2)[x] / poly` from two upward whose
    /// powers first return to one after exactly `2^n - 1` steps.
    pub(in crate::field::binary) fn oracle_smallest_generator(poly: u128, n: u32) -> u64 {
        let order = (1u64 << n) - 1;
        (2..=order)
            .find(|&candidate| {
                let mut power = candidate;
                let mut steps = 1;
                while power != 1 && steps <= order {
                    power = oracle_mul(poly, n, power, candidate);
                    steps += 1;
                }
                steps == order
            })
            .expect("a field has a primitive element")
    }

    /// Apply coordinate columns: bit `i` of `word` selects `columns[i]`.
    pub(in crate::field::binary) fn apply_columns(columns: [u8; 8], word: u64) -> u64 {
        (0..8)
            .filter(|&i| (word >> i) & 1 == 1)
            .fold(0, |image, i| image ^ u64::from(columns[i]))
    }

    /// Every irreducible polynomial of degree `n`.
    fn irreducibles(n: u32) -> impl Iterator<Item = u128> {
        (1u128 << n..2u128 << n).filter(move |&p| poly_irreducible(p, n))
    }

    /// The generator search, evaluated at runtime over every irreducible
    /// polynomial of degrees 2, 4, and 8, finds the brute-force smallest
    /// element of full multiplicative order.
    #[test]
    fn smallest_generator_matches_brute_force_order() {
        for n in [2u8, 4, 8] {
            for poly in irreducibles(u32::from(n)) {
                let description = BinaryDescription::polynomial(black_box(n), black_box(poly));
                assert_eq!(
                    Polynomial::<AES>::smallest_generator(&description),
                    oracle_smallest_generator(poly, u32::from(n)),
                    "degree {n}, polynomial {poly:#x}"
                );
            }
        }
    }

    /// Field names carry the polynomial: the hexadecimal digits after
    /// `0x` parse back to every irreducible polynomial of the degree.
    #[test]
    fn field_names_round_trip_their_polynomial() {
        fn parsed(name: &[u8]) -> u128 {
            let name = core::str::from_utf8(name).expect("ASCII name");
            let (_, digits) = name.rsplit_once("0x").expect("hexadecimal suffix");
            u128::from_str_radix(digits, 16).expect("hexadecimal digits")
        }
        for poly in irreducibles(8) {
            assert_eq!(parsed(&name_bytes8(black_box(poly))), poly);
        }
        for poly in irreducibles(4) {
            assert_eq!(parsed(&name_bytes4(black_box(poly))), poly);
        }
        for poly in irreducibles(2) {
            assert_eq!(parsed(&name_bytes2(black_box(poly))), poly);
        }
    }

    /// Every degree-8 polynomial, against an independent irreducibility
    /// count: GF(2) has exactly 30 irreducible polynomials of degree 8.
    #[test]
    fn irreducibility_check_admits_exactly_the_thirty_fields() {
        let count = (0x100..=0x1FFu128)
            .filter(|&p| super::super::description::poly_irreducible(p, 8))
            .count();
        assert_eq!(count, 30);
        assert!(super::super::description::poly_irreducible(AES, 8));
        assert!(super::super::description::poly_irreducible(RS, 8));
        assert!(super::super::description::poly_irreducible(0x12D, 8));
        assert!(super::super::description::poly_irreducible(0x187, 8));
        assert!(!super::super::description::poly_irreducible(0x101, 8));
        assert!(!super::super::description::poly_irreducible(0x0FF, 8));
        assert!(!super::super::description::poly_irreducible(0x200, 8));
    }
}
