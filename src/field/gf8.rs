//! The flat (non-tower) GF(2^8) fields, one per reduction polynomial.
//!
//! Every flat byte field is the same construction `GF(2)[x] / p(x)` for an
//! irreducible degree-8 `p`. [`Gf8`] is that construction with `p` as a
//! const parameter: the marker, its [`Elem`], and every table derive from the
//! polynomial at compile time, so each polynomial is a distinct type with no
//! runtime polynomial and fully `const` scalar arithmetic.
//!
//! | Constant | Polynomial | Convention |
//! | --- | --- | --- |
//! | [`AES`] | `0x11B` | AES/Rijndael; the field `GF2P8MULB` multiplies natively |
//! | [`REED_SOLOMON`] | `0x11D` | Intel ISA-L, `klauspost/reedsolomon`, QR codes |
//!
//! Any other degree-8 irreducible polynomial is spelled directly, such as
//! `Gf8<0x12D>` for Data Matrix or `Gf8<0x187>` for the CCSDS Reed–Solomon
//! field. Distinct polynomials are unrelated encodings: a byte has different
//! products under each, so a buffer carries one convention and never two.
//!
//! # Totality and canonicalization
//!
//! A polynomial that is not irreducible of degree eight is a compile-time
//! error wherever an element is constructed or the field's tables are used:
//! the check runs when the const items of that `Gf8<POLY>` are evaluated,
//! which happens at monomorphization rather than at type checking.
//!
//! ```
//! use fgf::gf8::{AES, Elem, REED_SOLOMON};
//!
//! // The same bytes multiply differently under the two conventions.
//! let (a, b) = (0x53, 0xca);
//! assert_eq!(Elem::<AES>::from_raw(a).mul(Elem::from_raw(b)).to_raw(), 0x01);
//! assert_eq!(Elem::<REED_SOLOMON>::from_raw(a).mul(Elem::from_raw(b)).to_raw(), 0x8f);
//!
//! // Any irreducible polynomial is a field; its generator is derived.
//! assert_eq!(Elem::<0x12D>::GENERATOR.pow(255), Elem::ONE);
//! ```
//!
//! ```compile_fail
//! // x^8 + 1 = (x + 1)^8 is reducible, so this field does not exist.
//! let _ = fgf::gf8::Elem::<0x101>::from_raw(1);
//! ```
//!
//! ```compile_fail
//! // Default construction also rejects a reducible polynomial.
//! let _ = fgf::gf8::Elem::<0x101>::default();
//! ```
//!
//! # Compile-time coding matrices
//!
//! Every scalar operation is `const`, so a Reed–Solomon generator matrix is
//! a `const` item rather than a lazily built table. This builds the 3-by-4
//! Vandermonde matrix `V[i][j] = x_j^i` at compile time:
//!
//! ```
//! use fgf::gf8::{AES, Elem};
//!
//! const POINTS: [Elem<AES>; 4] = [
//!     Elem::from_raw(1),
//!     Elem::from_raw(2),
//!     Elem::from_raw(3),
//!     Elem::from_raw(4),
//! ];
//! const V: [[Elem<AES>; 4]; 3] = {
//!     let mut rows = [[Elem::ZERO; 4]; 3];
//!     let mut i = 0;
//!     while i < 3 {
//!         let mut j = 0;
//!         while j < 4 {
//!             rows[i][j] = POINTS[j].pow(i as u64);
//!             j += 1;
//!         }
//!         i += 1;
//!     }
//!     rows
//! };
//!
//! assert_eq!(V[0], [Elem::ONE; 4]);
//! assert_eq!(V[1], POINTS);
//! assert_eq!(V[2][3], Elem::from_raw(4).square());
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Elem::<AES>::from_raw(0x57).div(Elem::ZERO).to_raw() == 0);
//! ```

/// The AES/Rijndael polynomial `x^8 + x^4 + x^3 + x + 1`.
///
/// It is the field the x86 `GF2P8MULB` instruction implements, so
/// `Gf8<AES>` multiplies with one instruction per vector on GFNI hosts.
/// Its derived generator is `0x03`.
pub const AES: u16 = 0x11B;

/// The polynomial `x^8 + x^4 + x^3 + x^2 + 1`.
///
/// It is the field of Intel ISA-L, `klauspost/reedsolomon`, QR codes, and
/// the classical Reed–Solomon tables; `Gf8<REED_SOLOMON>` shards are
/// byte-identical to those ecosystems. Its derived generator is `0x02`.
pub const REED_SOLOMON: u16 = 0x11D;

/// Marker type for GF(2^8) under the reduction polynomial `POLY`.
///
/// `POLY` is the full polynomial including `x^8`, in `0x100..=0x1FF`, and
/// must be irreducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf8<const POLY: u16>;

/// An element of GF(2^8) under `POLY`, stored as its polynomial coefficient
/// vector.
///
/// Every byte is a distinct field value, so the derived
/// [`PartialEq`]/[`Hash`]/[`Ord`] — raw-byte order — compare field values.
/// That order is a deterministic total order for map keys and sorting; no
/// order compatible with addition exists in characteristic two.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Elem<const POLY: u16>(pub(crate) u8);

/// Whether `poly` is an irreducible polynomial of degree eight.
///
/// Trial division by every polynomial of degree one through four: a
/// reducible degree-8 polynomial has a factor of degree at most four.
const fn is_irreducible(poly: u16) -> bool {
    if poly < 0x100 || poly > 0x1FF {
        return false;
    }
    let mut divisor = 2u16;
    while divisor < 32 {
        if carryless_rem(poly, divisor) == 0 {
            return false;
        }
        divisor += 1;
    }
    true
}

/// `value mod divisor` over GF(2)[x].
const fn carryless_rem(mut value: u16, divisor: u16) -> u16 {
    let shift = divisor.ilog2();
    while value != 0 && value.ilog2() >= shift {
        value ^= divisor << (value.ilog2() - shift);
    }
    value
}

impl<const POLY: u16> Elem<POLY> {
    /// The additive identity.
    pub const ZERO: Self = Self::from_raw(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self::from_raw(1);

    /// The reduction polynomial, including `x^8`.
    pub const POLY: u16 = {
        assert!(
            is_irreducible(POLY),
            "Gf8 POLY must be an irreducible polynomial of degree 8"
        );
        POLY
    };

    /// The low byte of [`Elem::POLY`], `XORed` in when a shift overflows the
    /// field.
    pub(crate) const REDUCTION_LOW: u8 = Self::POLY.to_le_bytes()[0];

    /// The smallest primitive element: the generator of the multiplicative
    /// group the discrete-log tables are built from.
    ///
    /// An element generates the order-255 group exactly when no proper
    /// divisor `255 / q` of the order, for the prime factors `q` of
    /// `255 = 3 * 5 * 17`, already returns it to one.
    pub const GENERATOR: Self = {
        let mut candidate = 2u16;
        loop {
            assert!(candidate < 256, "Gf8 POLY has no primitive element");
            let g = Self(candidate.to_le_bytes()[0]);
            if g.pow_xtime(85).0 != 1 && g.pow_xtime(51).0 != 1 && g.pow_xtime(15).0 != 1 {
                break g;
            }
            candidate += 1;
        }
    };

    /// `EXP[i] = GENERATOR^i` for `i in 0..255`.
    pub(crate) const EXP: [u8; 255] = {
        let mut table = [0u8; 255];
        let mut value = Self::ONE;
        let mut i = 0;
        while i < 255 {
            table[i] = value.0;
            value = value.mul_xtime(Self::GENERATOR);
            i += 1;
        }
        table
    };

    /// `LOG[EXP[i]] = i` for nonzero elements. `LOG[0]` is undefined and set
    /// to zero; callers short-circuit on zero before indexing.
    // `i < 255`, so the cast is exact; `const` rules out `try_into`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const LOG: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 255 {
            table[Self::EXP[i] as usize] = i as u8;
            i += 1;
        }
        table
    };

    /// Wrap a raw coefficient vector.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        let _ = Self::POLY;
        Self(value)
    }

    /// Unwrap to the raw coefficient vector.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u8 {
        self.0
    }

    /// Decode from the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        Self::from_raw(bytes[0])
    }

    /// Encode to the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 1] {
        [self.0]
    }

    /// Field addition. Identical to subtraction and to bitwise XOR.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    /// Field subtraction. Identical to [`Elem::add`].
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

    /// Reference multiplication: shift-and-XOR ("Russian peasant").
    ///
    /// `const`, allocation-free, and independent of the tables. This is the
    /// oracle the table and SIMD backends are validated against, and the
    /// multiply every table is built from.
    #[must_use]
    pub(crate) const fn mul_xtime(self, rhs: Self) -> Self {
        let mut a = self.0;
        let b = rhs.0;
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
        Self(acc)
    }

    /// Square-and-multiply over [`Elem::mul_xtime`], for the table builders.
    const fn pow_xtime(self, mut exponent: u32) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul_xtime(base);
            }
            base = base.mul_xtime(base);
            exponent >>= 1;
        }
        result
    }

    /// Field multiplication through the discrete-log tables.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        // LOG has no entry for zero, so the absorbing case must
        // short-circuit.
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        let la = Self::LOG[self.0 as usize] as usize;
        let lb = Self::LOG[rhs.0 as usize] as usize;
        Self(Self::EXP[(la + lb) % 255])
    }

    /// Square.
    ///
    /// A flat field has no cheaper form than a general multiply; the tower
    /// fields above do. Present for parity with them.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse. Maps zero to zero by crate convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        // The `% 255` matters only for `self == 1`, where `LOG` is 0 and
        // `255 - 0` would run off the end of the 255-entry table.
        let l = Self::LOG[self.0 as usize] as usize;
        Self(Self::EXP[(255 - l) % 255])
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division
    /// total leaves hot loops branch-free and keeps this callable from
    /// `const` context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power.
    #[must_use]
    pub const fn pow(self, mut exponent: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }
}

/// `"GF(2^8)/0x1XY"`, the field name for `poly`.
const fn field_name(poly: u16) -> [u8; 13] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut name = *b"GF(2^8)/0x000";
    name[10] = HEX[((poly >> 8) & 0xf) as usize];
    name[11] = HEX[((poly >> 4) & 0xf) as usize];
    name[12] = HEX[(poly & 0xf) as usize];
    name
}

impl<const POLY: u16> Gf8<POLY> {
    const NAME_BYTES: [u8; 13] = field_name(Elem::<POLY>::POLY);
}

impl<const POLY: u16> crate::field::Elem for Elem<POLY> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
    #[inline]
    fn square(self) -> Self {
        Elem::square(self)
    }
    #[inline]
    fn inv(self) -> Self {
        Elem::inv(self)
    }
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
    #[inline]
    fn pow(self, exponent: u64) -> Self {
        Elem::pow(self, exponent)
    }
}

impl<const POLY: u16> crate::field::Field for Gf8<POLY> {
    type Elem = Elem<POLY>;

    const NAME: &'static str = match core::str::from_utf8(&Self::NAME_BYTES) {
        Ok(name) => name,
        Err(_) => panic!("Gf8 field names are ASCII"),
    };
    const BITS: u32 = 8;
    const BYTES: usize = 1;
    const ORDER: u128 = 256;
    const CHARACTERISTIC: u64 = 2;
    const GENERATOR: Elem<POLY> = Elem::<POLY>::GENERATOR;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<POLY> {
        let bytes: [u8; 1] = bytes
            .try_into()
            .expect("GF(2^8) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<POLY>) {
        assert_eq!(bytes.len(), 1, "GF(2^8) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl<const POLY: u16> Default for Elem<POLY> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const POLY: u16> core::fmt::Debug for Elem<POLY> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Gf8<{POLY:#05x}>({:#04x})", self.0)
    }
}

impl<const POLY: u16> core::fmt::Display for Elem<POLY> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:02x}", self.0)
    }
}

impl<const POLY: u16> core::ops::Add for Elem<POLY> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
}

impl<const POLY: u16> core::ops::Sub for Elem<POLY> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
}

impl<const POLY: u16> core::ops::Neg for Elem<POLY> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Elem::neg(self)
    }
}

impl<const POLY: u16> core::ops::Mul for Elem<POLY> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
}

impl<const POLY: u16> core::ops::Div for Elem<POLY> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
}

impl<const POLY: u16> core::ops::AddAssign for Elem<POLY> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Elem::add(*self, rhs);
    }
}

impl<const POLY: u16> core::ops::SubAssign for Elem<POLY> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Elem::sub(*self, rhs);
    }
}

impl<const POLY: u16> core::ops::MulAssign for Elem<POLY> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Elem::mul(*self, rhs);
    }
}

impl<const POLY: u16> core::ops::DivAssign for Elem<POLY> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Elem::div(*self, rhs);
    }
}

impl<const POLY: u16> core::iter::Sum for Elem<POLY> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Elem::add)
    }
}

impl<'a, const POLY: u16> core::iter::Sum<&'a Elem<POLY>> for Elem<POLY> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem<POLY>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| Elem::add(acc, x))
    }
}

impl<const POLY: u16> core::iter::Product for Elem<POLY> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Elem::mul)
    }
}

impl<'a, const POLY: u16> core::iter::Product<&'a Elem<POLY>> for Elem<POLY> {
    #[inline]
    fn product<I: Iterator<Item = &'a Elem<POLY>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| Elem::mul(acc, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every degree-8 polynomial, against an independent irreducibility
    /// count: GF(2) has exactly 30 irreducible polynomials of degree 8.
    #[test]
    fn irreducibility_check_admits_exactly_the_thirty_fields() {
        let count = (0x100..=0x1FFu16).filter(|&p| is_irreducible(p)).count();
        assert_eq!(count, 30);
        assert!(is_irreducible(AES) && is_irreducible(REED_SOLOMON));
        assert!(is_irreducible(0x12D) && is_irreducible(0x187));
        assert!(!is_irreducible(0x101) && !is_irreducible(0x0FF) && !is_irreducible(0x200));
    }

    fn check_tables<const P: u16>() {
        for (i, &g) in Elem::<P>::EXP.iter().enumerate() {
            assert_eq!(Elem::<P>::LOG[g as usize] as usize, i, "log/exp at {i}");
        }
        for a in 0..=255u8 {
            let a = Elem::<P>(a);
            // Fermat: a^254 is the inverse, through the table-free multiply.
            assert_eq!(a.pow_xtime(254), a.inv(), "{a:?} inverse");
            for b in 0..=255u8 {
                let b = Elem::<P>(b);
                assert_eq!(a.mul(b), a.mul_xtime(b), "{a:?} * {b:?}");
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
        assert_eq!(Elem::<AES>::GENERATOR.0, 0x03);
        assert_eq!(Elem::<REED_SOLOMON>::GENERATOR.0, 0x02);
    }
}
