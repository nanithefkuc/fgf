//! The polynomial-basis binary fields: GF(2) and the flat GF(2^8) family.
//!
//! Every flat byte field here is the same construction `GF(2)[x] / p(x)` for
//! an irreducible degree-8 `p`. [`Poly`] is that construction with `p` as a
//! const parameter: the representation, its element
//! [`Elem<Gf<8, Poly<P>>>`](crate::field::Elem), and every table derive from the
//! polynomial at compile time, so each polynomial is a distinct
//! representation with no runtime polynomial and fully `const` scalar
//! arithmetic.
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
//! # Totality and canonicalization
//!
//! A polynomial that is not irreducible of degree eight is a compile-time
//! error wherever an element is constructed or the representation's tables
//! are used: the check runs when the const items of that `Poly<P>` are
//! evaluated, which happens at monomorphization rather than at type
//! checking.
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
//! ```compile_fail
//! // x^8 + 1 = (x + 1)^8 is reducible, so this field does not exist.
//! let _ = fgf::Elem::<fgf::Gf<8, fgf::Poly<0x101>>>::from_raw(1);
//! ```
//!
//! ```compile_fail
//! // Default construction also rejects a reducible polynomial.
//! let _ = fgf::Elem::<fgf::Gf<8, fgf::Poly<0x101>>>::default();
//! ```
//!
//! # Compile-time coding matrices
//!
//! Every scalar operation is `const`, so a Reed–Solomon generator matrix is
//! a `const` item rather than a lazily built table. This builds the 3-by-4
//! Vandermonde matrix `V[i][j] = x_j^i` at compile time:
//!
//! ```
//! use fgf::{AES, Elem, Gf, Poly};
//!
//! type Byte = Elem<Gf<8, Poly<AES>>>;
//!
//! const POINTS: [Byte; 4] = [
//!     Byte::from_raw(1),
//!     Byte::from_raw(2),
//!     Byte::from_raw(3),
//!     Byte::from_raw(4),
//! ];
//! const V: [[Byte; 4]; 3] = {
//!     let mut rows = [[Byte::ZERO; 4]; 3];
//!     let mut i = 0;
//!     while i < 3 {
//!         let mut j = 0;
//!         while j < 4 {
//!             rows[i][j] = POINTS[j].pow(i as u128);
//!             j += 1;
//!         }
//!         i += 1;
//!     }
//!     rows
//! };
//!
//! assert_eq!(V[0], [Byte::ONE; 4]);
//! assert_eq!(V[1], POINTS);
//! assert_eq!(V[2][3], Byte::from_raw(4).square());
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Byte::from_raw(0x57).div(Byte::ZERO).to_raw() == 0);
//! ```

use core::fmt;

use super::repr::{ByteRepr, Gf, Repr};
use super::{Elem, Validate};

/// The AES/Rijndael polynomial `x^8 + x^4 + x^3 + x + 1`.
///
/// It is the field the x86 `GF2P8MULB` instruction implements, so
/// `Gf8<Poly<AES>>` multiplies with one instruction per vector on GFNI
/// hosts. Its derived generator is `0x03`.
pub const AES: u32 = 0x11B;

/// The polynomial `x^8 + x^4 + x^3 + x^2 + 1`.
///
/// It is the field of Intel ISA-L, `klauspost/reedsolomon`, QR codes, and
/// the classical Reed–Solomon tables; `Gf8<Poly<REED_SOLOMON>>` shards are
/// byte-identical to those ecosystems. Its derived generator is `0x02`.
pub const REED_SOLOMON: u32 = 0x11D;

/// The polynomial-basis representation of GF(2^8) under the reduction
/// polynomial `P`.
///
/// `P` is the full polynomial including `x^8`, in `0x100..=0x1FF`, and must
/// be irreducible; the check runs when the representation's constants are
/// evaluated. The same type spells GF(2) as [`Poly<3>`](Poly), the
/// polynomial `x + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Poly<const P: u32>;

/// Whether `poly` is an irreducible polynomial of degree eight.
///
/// Trial division by every polynomial of degree one through four: a
/// reducible degree-8 polynomial has a factor of degree at most four.
const fn is_irreducible(poly: u32) -> bool {
    if poly < 0x100 || poly > 0x1FF {
        return false;
    }
    let mut divisor = 2u32;
    while divisor < 32 {
        if carryless_rem(poly, divisor) == 0 {
            return false;
        }
        divisor += 1;
    }
    true
}

/// `value mod divisor` over GF(2)[x].
const fn carryless_rem(mut value: u32, divisor: u32) -> u32 {
    let shift = divisor.ilog2();
    while value != 0 && value.ilog2() >= shift {
        value ^= divisor << (value.ilog2() - shift);
    }
    value
}

impl<const P: u32> Poly<P> {
    /// The bytes of the field name, `"GF(2^8)/0x1XY"`.
    const NAME_BYTES: [u8; 13] = name_bytes(Self::POLY);

    /// The reduction polynomial, including `x^8`.
    pub const POLY: u32 = {
        assert!(
            is_irreducible(P),
            "Poly must be an irreducible polynomial of degree 8"
        );
        P
    };

    /// The low byte of [`Poly::POLY`], `XORed` in when a shift overflows the
    /// field.
    pub(crate) const REDUCTION_LOW: u8 = Self::POLY.to_le_bytes()[0];

    /// The smallest primitive element: the generator of the multiplicative
    /// group the discrete-log tables are built from.
    ///
    /// An element generates the order-255 group exactly when no proper
    /// divisor `255 / q` of the order, for the prime factors `q` of
    /// `255 = 3 * 5 * 17`, already returns it to one.
    const GENERATOR_RAW_VALUE: u8 = {
        let mut candidate = 2u32;
        loop {
            assert!(candidate < 256, "Poly P has no primitive element");
            let g = candidate.to_le_bytes()[0];
            if Self::pow_xtime(g, 85) != 1
                && Self::pow_xtime(g, 51) != 1
                && Self::pow_xtime(g, 15) != 1
            {
                break g;
            }
            candidate += 1;
        }
    };

    /// `EXP[i] = GENERATOR^i` for `i in 0..255`.
    const fn build_exp() -> [u8; 255] {
        let mut table = [0u8; 255];
        let mut value = 1u8;
        let mut i = 0;
        while i < 255 {
            table[i] = value;
            value = Self::mul_xtime(value, Self::GENERATOR_RAW_VALUE);
            i += 1;
        }
        table
    }

    /// `LOG[EXP[i]] = i` for nonzero elements. `LOG[0]` is undefined and set
    /// to zero; callers short-circuit on zero before indexing.
    // `i < 255`, so the cast is exact; `const` rules out `try_into`.
    #[allow(clippy::cast_possible_truncation)]
    const fn build_log() -> [u8; 256] {
        let exp = Self::build_exp();
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 255 {
            table[exp[i] as usize] = i as u8;
            i += 1;
        }
        table
    }

    /// Reference multiplication: shift-and-XOR ("Russian peasant").
    ///
    /// `const`, allocation-free, and independent of the tables. This is the
    /// oracle the table and SIMD backends are validated against, and the
    /// multiply every table is built from.
    #[must_use]
    pub(crate) const fn mul_xtime(a: u8, b: u8) -> u8 {
        let mut a = a;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            let overflow = a & 0x80 != 0;
            a <<= 1;
            if overflow {
                a ^= Self::REDUCTION_LOW;
            }
            i += 1;
        }
        acc
    }

    /// Square-and-multiply over [`Poly::mul_xtime`], for the generator
    /// search and the table builders.
    const fn pow_xtime(a: u8, mut exponent: u32) -> u8 {
        let mut base = a;
        let mut result = 1u8;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = Self::mul_xtime(result, base);
            }
            base = Self::mul_xtime(base, base);
            exponent >>= 1;
        }
        result
    }
}

impl<const P: u32> Repr<8> for Poly<P> {
    type Raw = u8;

    const NAME: &'static str = match core::str::from_utf8(&Self::NAME_BYTES) {
        Ok(name) => name,
        Err(_) => panic!("Poly field names are ASCII"),
    };
    const ONE_RAW: u8 = 1;
    const GENERATOR_RAW: u8 = Self::GENERATOR_RAW_VALUE;
}

impl<const P: u32> ByteRepr for Poly<P> {
    const EXP: &'static [u8; 255] = &Self::build_exp();
    const LOG: &'static [u8; 256] = &Self::build_log();
    const VALID: () = {
        let _ = Self::POLY;
    };
}

impl Repr<1> for Poly<3> {
    type Raw = u8;

    const NAME: &'static str = "GF(2)";
    const ONE_RAW: u8 = 1;
    const GENERATOR_RAW: u8 = 1;
}

/// The bytes of the field name for `poly`: `"GF(2^8)/0x1XY"`.
const fn name_bytes(poly: u32) -> [u8; 13] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut name = *b"GF(2^8)/0x000";
    name[10] = HEX[((poly >> 8) & 0xf) as usize];
    name[11] = HEX[((poly >> 4) & 0xf) as usize];
    name[12] = HEX[(poly & 0xf) as usize];
    name
}

impl Elem<Gf<1, Poly<3>>> {
    /// Wrap a raw byte, keeping only the low bit.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        let () = Validate::<Gf<1, Poly<3>>>::OK;
        Elem { raw: value & 1 }
    }

    /// Decode from the stable one-byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        Self::from_raw(bytes[0])
    }

    /// Encode to the stable one-byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 1] {
        [self.raw]
    }

    /// The canonical representative of this element.
    ///
    /// The stored byte is canonical by construction, so this is the
    /// identity. It exists because every field in the crate answers the
    /// question.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        self
    }

    /// Field addition. XOR of the low bits.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Elem {
            raw: self.raw ^ rhs.raw,
        }
    }

    /// Field subtraction. Identical to [`Elem::add`](Self::add): characteristic two.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }

    /// Additive inverse. The identity: `x + x = 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        self
    }

    /// Field multiplication. AND of the low bits.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: self.raw & rhs.raw,
        }
    }

    /// Square. The identity: `x² = x` in GF(2).
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self
    }

    /// Multiplicative inverse. `inv(1) = 1` and `inv(0) = 0` by convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        self
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if rhs.raw == 0 { Self::ZERO } else { self }
    }

    /// Raise to an unsigned integer power. `pow(_, 0) == ONE`.
    ///
    /// Every element satisfies `x² = x`, so any positive power is `x`
    /// itself.
    #[inline]
    #[must_use]
    pub const fn pow(self, exponent: u128) -> Self {
        if exponent == 0 { Self::ONE } else { self }
    }

    /// Whether this element is the additive identity.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.raw == 0
    }

    /// Whether this element is the multiplicative identity.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.raw == 1
    }
}

impl fmt::Display for Elem<Gf<1, Poly<3>>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every degree-8 polynomial, against an independent irreducibility
    /// count: GF(2) has exactly 30 irreducible polynomials of degree 8.
    #[test]
    fn irreducibility_check_admits_exactly_the_thirty_fields() {
        let count = (0x100..=0x1FFu32).filter(|&p| is_irreducible(p)).count();
        assert_eq!(count, 30);
        assert!(is_irreducible(AES) && is_irreducible(REED_SOLOMON));
        assert!(is_irreducible(0x12D) && is_irreducible(0x187));
        assert!(!is_irreducible(0x101) && !is_irreducible(0x0FF) && !is_irreducible(0x200));
    }

    fn check_tables<const P: u32>() {
        for (i, &g) in Poly::<P>::EXP.iter().enumerate() {
            assert_eq!(Poly::<P>::LOG[g as usize] as usize, i, "log/exp at {i}");
        }
        for a in 0..=255u8 {
            let a = Elem::<Gf<8, Poly<P>>>::from_raw(a);
            // Fermat: a^254 is the inverse, through the table-free multiply.
            assert_eq!(
                a.inv().to_raw(),
                Poly::<P>::pow_xtime(a.to_raw(), 254),
                "{a:?} inverse"
            );
            for b in 0..=255u8 {
                let b = Elem::<Gf<8, Poly<P>>>::from_raw(b);
                assert_eq!(
                    a.mul(b).to_raw(),
                    Poly::<P>::mul_xtime(a.to_raw(), b.to_raw()),
                    "{a:?} * {b:?}"
                );
            }
        }
    }

    #[test]
    fn tables_match_the_reference_multiply() {
        check_tables::<AES>();
        check_tables::<REED_SOLOMON>();
        check_tables::<0x12D>();
        check_tables::<0x187>();
    }

    /// The frozen generators of the two named conventions.
    #[test]
    fn derived_generators_match_the_frozen_conventions() {
        assert_eq!(Elem::<Gf<8, Poly<AES>>>::GENERATOR.to_raw(), 0x03);
        assert_eq!(Elem::<Gf<8, Poly<REED_SOLOMON>>>::GENERATOR.to_raw(), 0x02);
    }
}
