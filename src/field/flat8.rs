//! Shared implementation of the flat (non-tower) GF(2^8) fields.
//!
//! Both public byte fields — [`gf8b`](crate::field::gf8b) under the AES
//! polynomial `0x11B` and `gf8d` under `0x11D` — are the
//! same construction `GF(2)[x] / p(x)` for an irreducible degree-8 `p`. They
//! differ only in the reduction polynomial and the multiplicative generator;
//! every algorithm, table shape, and byte encoding is identical. This macro is
//! that single implementation, instantiated once per field so each stays a
//! distinct type with its own compile-time tables and no runtime polynomial.

/// Emit the scalar algebra of one flat GF(2^8) field into the calling module.
///
/// Parameters: the zero-sized marker identifier, the full reduction polynomial
/// (`0x1_00..=0x1FF`), its low byte (the polynomial minus `x^8`, `XORed` in on
/// overflow), a primitive generator of the multiplicative group, and the
/// human-readable field name.
macro_rules! flat_gf8 {
    ($field:ident, $reduction_poly:literal, $reduction_low:literal, $generator:literal, $name:literal) => {
        /// Irreducible reduction polynomial of this field.
        pub const REDUCTION_POLY: u16 = $reduction_poly;

        /// Low byte of [`REDUCTION_POLY`], `XORed` in when a shift overflows the
        /// field.
        pub const REDUCTION_LOW: u8 = $reduction_low;

        /// A generator of the multiplicative group under this polynomial.
        pub const GENERATOR: Elem = Elem($generator);

        #[doc = concat!("Marker type for ", $name, ".")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
        pub struct $field;

        /// An element of this GF(2^8) field, stored as its polynomial
        /// coefficient vector.
        ///
        /// The derived [`Ord`] is raw-representation order, useful for map keys
        /// and deterministic iteration; it carries no field-theoretic meaning.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
        pub struct Elem(pub u8);

        impl $field {
            /// This field's irreducible reduction polynomial, e.g. `0x11B` for
            /// the AES field. Lets generic code introspect the representation
            /// at compile time without carrying a runtime polynomial.
            #[inline]
            #[must_use]
            pub const fn field_poly() -> u16 {
                REDUCTION_POLY
            }
        }

        impl Elem {
            /// The additive identity.
            pub const ZERO: Self = Self(0);
            /// The multiplicative identity.
            pub const ONE: Self = Self(1);

            /// Wrap a raw coefficient vector.
            #[inline]
            #[must_use]
            pub const fn from_raw(value: u8) -> Self {
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
                Self(bytes[0])
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

            /// Reference multiplication: shift-and-XOR ("Russian peasant").
            ///
            /// `const`, allocation-free, and independent of the tables. This is
            /// the oracle the table and SIMD backends are validated against.
            #[must_use]
            pub const fn mul_xtime(self, rhs: Self) -> Self {
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
                        a ^= REDUCTION_LOW;
                    }
                    i += 1;
                }
                Self(acc)
            }

            /// Reference inverse via Fermat's little theorem: `a^254 == a^-1`.
            #[must_use]
            pub const fn inv_xtime(self) -> Self {
                if self.0 == 0 {
                    return Self::ZERO;
                }
                // 254 = 0b1111_1110. Square every step; multiply on every set
                // bit, scanning most-significant first.
                let mut result = Self::ONE;
                let mut i = 0;
                while i < 8 {
                    result = result.mul_xtime(result);
                    if (254u32 >> (7 - i)) & 1 == 1 {
                        result = result.mul_xtime(self);
                    }
                    i += 1;
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
                let la = LOG[self.0 as usize] as usize;
                let lb = LOG[rhs.0 as usize] as usize;
                Self(EXP[(la + lb) % 255])
            }

            /// Square.
            ///
            /// A flat field has no cheaper form than a general multiply; the
            /// tower fields above do. Present for parity with them.
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
                let l = LOG[self.0 as usize] as usize;
                Self(EXP[(255 - l) % 255])
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

        impl crate::field::Elem for Elem {
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

        impl crate::field::Field for $field {
            type Elem = Elem;

            const NAME: &'static str = $name;
            const BITS: u32 = 8;
            const BYTES: usize = 1;
            const ORDER: u128 = 256;
            const GENERATOR: Elem = GENERATOR;

            #[inline]
            fn read(bytes: &[u8]) -> Elem {
                let bytes: [u8; 1] = bytes
                    .try_into()
                    .expect(concat!($name, " element has the wrong byte width"));
                Elem::from_bytes(bytes)
            }

            #[inline]
            fn write(bytes: &mut [u8], value: Elem) {
                assert_eq!(
                    bytes.len(),
                    1,
                    concat!($name, " element has the wrong byte width")
                );
                bytes.copy_from_slice(&value.to_bytes());
            }
        }

        impl core::fmt::Debug for Elem {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($field), "({:#04x})"), self.0)
            }
        }

        impl core::fmt::Display for Elem {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{:02x}", self.0)
            }
        }

        impl core::ops::Add for Elem {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Elem::add(self, rhs)
            }
        }

        impl core::ops::Sub for Elem {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Elem::sub(self, rhs)
            }
        }

        impl core::ops::Mul for Elem {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Elem::mul(self, rhs)
            }
        }

        impl core::ops::Div for Elem {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                Elem::div(self, rhs)
            }
        }

        impl core::ops::AddAssign for Elem {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = Elem::add(*self, rhs);
            }
        }

        impl core::ops::SubAssign for Elem {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = Elem::sub(*self, rhs);
            }
        }

        impl core::ops::MulAssign for Elem {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = Elem::mul(*self, rhs);
            }
        }

        impl core::ops::DivAssign for Elem {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = Elem::div(*self, rhs);
            }
        }

        impl core::iter::Sum for Elem {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, Elem::add)
            }
        }

        impl<'a> core::iter::Sum<&'a Elem> for Elem {
            #[inline]
            fn sum<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |acc, &x| Elem::add(acc, x))
            }
        }

        impl core::iter::Product for Elem {
            #[inline]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ONE, Elem::mul)
            }
        }

        impl<'a> core::iter::Product<&'a Elem> for Elem {
            #[inline]
            fn product<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
                iter.fold(Self::ONE, |acc, &x| Elem::mul(acc, x))
            }
        }

        // -------------------------------------------------------------------
        // Compile-time discrete-log tables.
        // -------------------------------------------------------------------

        /// `EXP[i] = GENERATOR^i` for `i in 0..255`.
        pub(crate) const EXP: [u8; 255] = build_exp();

        /// `LOG[EXP[i]] = i` for nonzero elements. `LOG[0]` is undefined and set
        /// to zero; callers must short-circuit on zero before indexing.
        pub(crate) const LOG: [u8; 256] = build_log();

        const fn build_exp() -> [u8; 255] {
            let mut table = [0u8; 255];
            let mut value: u8 = 1;
            let mut i = 0;
            while i < 255 {
                table[i] = value;
                value = Elem(value).mul_xtime(GENERATOR).0;
                i += 1;
            }
            table
        }

        // `i < 255`, so the cast is exact; `const fn` rules out `try_into`.
        #[allow(clippy::cast_possible_truncation)]
        const fn build_log() -> [u8; 256] {
            let mut table = [0u8; 256];
            let mut i = 0;
            while i < 255 {
                table[EXP[i] as usize] = i as u8;
                i += 1;
            }
            table
        }
    };
}

pub(crate) use flat_gf8;
