//! Polynomial-basis binary presentations at degrees 1, 2, 4, and 8.
//!
//! [`Poly`] is the construction `GF(2)[x] / p(x)` for an irreducible
//! degree-`N` `p`. The representation, its descriptor, and its generator
//! derive from the polynomial at compile time, so each polynomial is a
//! distinct representation with fully `const` scalar arithmetic.
//!
//! | Constant | Polynomial | Convention |
//! | --- | --- | --- |
//! | [`AES`] | `0x11B` | AES/Rijndael; the field `GF2P8MULB` multiplies natively |
//! | [`REED_SOLOMON`] | `0x11D` | Intel ISA-L, `klauspost/reedsolomon`, QR codes |
//!
//! Any other degree-8 irreducible polynomial is spelled directly, such as
//! `Poly<0x12D>` for Data Matrix or `Poly<0x187>` for the CCSDS Reed–Solomon
//! field. Distinct polynomials are unrelated encodings: a byte has different
//! products under each, so a buffer carries one convention and never two.
//!
//! ```
//! use fgf::{AES, Elem, Gf, Poly, REED_SOLOMON};
//!
//! // The same bytes multiply differently under the two conventions.
//! let (a, b) = (0x53, 0xca);
//! assert_eq!(
//!     Elem::<Gf<8, Poly<AES>>>::from_raw(a)
//!         .mul(Elem::<Gf<8, Poly<AES>>>::from_raw(b))
//!         .to_raw(),
//!     0x01
//! );
//! assert_eq!(
//!     Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(a)
//!         .mul(Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(b))
//!         .to_raw(),
//!     0x8f
//! );
//!
//! // Any irreducible polynomial is a field; its generator is derived.
//! assert_eq!(
//!     Elem::<Gf<8, Poly<0x12D>>>::GENERATOR.pow(255),
//!     Elem::<Gf<8, Poly<0x12D>>>::ONE
//! );
//! ```
//!
//! ```compile_fail,E0080
//! // x^8 + 1 = (x + 1)^8 is reducible, so this field does not exist.
//! let _ = fgf::Elem::<fgf::Gf<8, fgf::Poly<0x101>>>::from_raw(1);
//! ```
//!
//! ```compile_fail,E0080
//! // The degree-eight polynomial does not match a degree-four field.
//! let _ = fgf::Elem::<fgf::Gf<4, fgf::Poly<0x11B>>>::from_raw(1);
//! ```

use super::description::{BinaryDescription, ByteLogExp};
use super::{BinaryRepr, private};

/// The AES/Rijndael polynomial `x^8 + x^4 + x^3 + x + 1`.
///
/// It is the field the x86 `GF2P8MULB` instruction implements, so
/// `Gf8<Poly<AES>>` multiplies with one instruction per vector on GFNI
/// hosts. Its derived generator is `0x03`.
pub const AES: u128 = 0x11B;

/// The polynomial `x^8 + x^4 + x^3 + x^2 + 1`.
///
/// It is the field of Intel ISA-L, `klauspost/reedsolomon`, QR codes, and
/// the classical Reed–Solomon tables; `Gf8<Poly<REED_SOLOMON>>` shards are
/// byte-identical to those ecosystems. Its derived generator is `0x02`.
pub const REED_SOLOMON: u128 = 0x11D;

/// The polynomial-basis representation of GF(2^N) under the reduction
/// polynomial `P`.
///
/// `P` is the full polynomial including the leading term, and must be
/// irreducible of the field's degree; the check runs when the
/// representation's constants are evaluated. The same type spells GF(2) as
/// [`Poly<3>`](Poly), the polynomial `x + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Poly<const P: u128>;

impl<const P: u128> private::Sealed for Poly<P> {}

impl<const P: u128> Poly<P> {
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
            assert!(candidate < limit, "Poly P has no primitive element");
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
        impl<const P: u128> BinaryRepr<8> for Poly<P> {
            const NAME: &'static str = match core::str::from_utf8(&Poly::<P>::NAME_BYTES8) {
                Ok(name) => name,
                Err(_) => panic!("Poly field names are ASCII"),
            };
            const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(8, P);
            const LOG_EXP: Option<&'static ByteLogExp> = Some(&ByteLogExp::build(
                &BinaryDescription::polynomial(8, P),
                Poly::<P>::smallest_generator(&BinaryDescription::polynomial(8, P)),
            ));
            const VALID: () = {
                assert!(
                    super::description::poly_degree(P) == 8,
                    "polynomial degree does not match field degree"
                );
                <Self as BinaryRepr<8>>::DESCRIPTION.validate();
                assert!(
                    <Self as BinaryRepr<8>>::DESCRIPTION.is_generator(
                        Poly::<P>::smallest_generator(<Self as BinaryRepr<8>>::DESCRIPTION)
                    ),
                    "byte representation generator does not have full order"
                );
            };
        }
    };
}

poly_degree8!();

impl<const P: u128> BinaryRepr<4> for Poly<P> {
    const NAME: &'static str = match core::str::from_utf8(&Poly::<P>::NAME_BYTES4) {
        Ok(name) => name,
        Err(_) => panic!("Poly field names are ASCII"),
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

impl<const P: u128> BinaryRepr<2> for Poly<P> {
    const NAME: &'static str = match core::str::from_utf8(&Poly::<P>::NAME_BYTES2) {
        Ok(name) => name,
        Err(_) => panic!("Poly field names are ASCII"),
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

impl<const P: u128> BinaryRepr<1> for Poly<P> {
    const NAME: &'static str = "GF(2)";
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::polynomial(1, P);
    const LOG_EXP: Option<&'static ByteLogExp> = None;
    const VALID: () = {
        assert!(P == 3, "Poly<3> is the only degree-one form");
        <Self as BinaryRepr<1>>::DESCRIPTION.validate();
    };
}

use super::super::HasGenerator;
use super::Gf;

impl<const P: u128> HasGenerator for Gf<8, Poly<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Poly::<P>::smallest_generator(<Poly<P> as BinaryRepr<8>>::DESCRIPTION);
        assert!(
            <Poly<P> as BinaryRepr<8>>::DESCRIPTION.is_generator(g),
            "byte representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Gf<4, Poly<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Poly::<P>::smallest_generator(<Poly<P> as BinaryRepr<4>>::DESCRIPTION);
        assert!(
            <Poly<P> as BinaryRepr<4>>::DESCRIPTION.is_generator(g),
            "representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Gf<2, Poly<P>> {
    const GENERATOR_RAW: u8 = {
        let g = Poly::<P>::smallest_generator(<Poly<P> as BinaryRepr<2>>::DESCRIPTION);
        assert!(
            <Poly<P> as BinaryRepr<2>>::DESCRIPTION.is_generator(g),
            "representation generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128> HasGenerator for Gf<1, Poly<P>> {
    const GENERATOR_RAW: u8 = 1;
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    /// Every degree-8 polynomial, against an independent irreducibility
    /// count: GF(2) has exactly 30 irreducible polynomials of degree 8.
    #[test]
    fn irreducibility_check_admits_exactly_the_thirty_fields() {
        let count = (0x100..=0x1FFu128)
            .filter(|&p| super::super::description::poly_irreducible(p, 8))
            .count();
        assert_eq!(count, 30);
        assert!(super::super::description::poly_irreducible(AES, 8));
        assert!(super::super::description::poly_irreducible(REED_SOLOMON, 8));
        assert!(super::super::description::poly_irreducible(0x12D, 8));
        assert!(super::super::description::poly_irreducible(0x187, 8));
        assert!(!super::super::description::poly_irreducible(0x101, 8));
        assert!(!super::super::description::poly_irreducible(0x0FF, 8));
        assert!(!super::super::description::poly_irreducible(0x200, 8));
    }
}
