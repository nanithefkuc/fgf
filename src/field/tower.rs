//! The Rijndael-rooted quadratic tower: GF(2^16), GF(2^32), GF(2^64).
//!
//! Each level is a degree-two extension of the one below, rooted at the AES
//! byte field [`crate::field::gf8b`]. An element of a level is `a + b*r` for
//! base-field components `a, b` and a root satisfying `r^2 + r + DELTA = 0`
//! with `DELTA` chosen to have absolute trace one, so the quadratic is
//! irreducible. Every level therefore shares one implementation — Karatsuba
//! multiplication with a `DELTA` fold, squaring with vanishing cross terms,
//! and inversion through the conjugate and norm — instantiated once per
//! level by the macro below.
//!
//! | Level | Module | Marker | Element | Root |
//! | --- | --- | --- | --- | --- |
//! | GF(2^16) | [`gf16`] | [`gf16::Gf16`] | [`gf16::Elem`] | `u` over GF(2^8) |
//! | GF(2^32) | [`gf32`] | [`gf32::Gf32`] | [`gf32::Elem`] | `v` over GF(2^16) |
//! | GF(2^64) | [`gf64`] | [`gf64::Gf64`] | [`gf64::Elem`] | `w` over GF(2^32) |
//!
//! The stable byte form of every level is the little-endian component
//! encoding `[a, b]`, so a buffer is a byte-interleaved pair of base-field
//! planes. This is the layout the SIMD kernels read; the canonical Fan-Paar
//! tower in [`crate::field::fan_paar`] is a different basis and does not
//! interoperate with it.

/// Emit one quadratic tower level into the calling module.
///
/// Parameters: the marker identifier, the level's raw integer type, the
/// base-field element type and its raw integer type, the bit offset of the
/// extension component, the raw `DELTA` and generator values, the symbol
/// naming the extension root, the field name, its storage width in bits and
/// bytes, its order, and the `Debug`/`Display` format strings.
macro_rules! quad_tower {
    (
        $marker:ident, $raw:ty, $base:ty, $base_raw:ty, $shift:literal,
        $delta:literal, $generator:literal, $root:literal, $name:literal,
        $bits:literal, $bytes:literal, $order:expr,
        $debug_fmt:literal, $display_fmt:literal $(,)?
    ) => {
        use core::fmt;

        type Base = $base;
        use crate::field::{Elem as ElemTrait, Field};

        #[doc = concat!("Constant term of the irreducible tower polynomial `", $root, "^2 + ", $root, " + DELTA`.")]
        pub const DELTA: Base = Base::from_raw($delta);

        /// A primitive element of the extension field: its multiplicative
        /// order is the whole group.
        pub const GENERATOR: Elem = Elem($generator);

        #[doc = concat!("Marker type for ", $name, ".")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
        pub struct $marker;

        #[doc = concat!("An element of ", $name, ", stored as `a + b*", $root, "` with `a` in the low half.")]
        ///
        /// Every bit pattern is a distinct field value, so the derived
        /// [`PartialEq`]/[`Hash`]/[`Ord`] — raw-bit order — compare field
        /// values. That order is a deterministic total order for map keys and
        /// sorting; no order compatible with addition exists in
        /// characteristic two.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
        pub struct Elem(pub(crate) $raw);

        impl Elem {
            /// The additive identity.
            pub const ZERO: Self = Self(0);
            /// The multiplicative identity.
            pub const ONE: Self = Self(1);

            #[doc = concat!("Construct `a + b*", $root, "` from its two base-field components.")]
            #[inline]
            #[must_use]
            pub const fn from_components(a: Base, b: Base) -> Self {
                Self((a.to_raw() as $raw) | ((b.to_raw() as $raw) << $shift))
            }

            #[doc = concat!("Return the `(a, b)` components of `a + b*", $root, "`.")]
            #[inline]
            #[must_use]
            // Splitting the raw word into halves: the truncation IS the
            // operation.
            #[allow(clippy::cast_possible_truncation)]
            pub const fn to_components(self) -> (Base, Base) {
                (
                    Base::from_raw(self.0 as $base_raw),
                    Base::from_raw((self.0 >> $shift) as $base_raw),
                )
            }

            /// Decode from the stable little-endian component representation.
            #[inline]
            #[must_use]
            pub const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                Self(<$raw>::from_le_bytes(bytes))
            }

            /// Encode to the stable little-endian component representation.
            #[inline]
            #[must_use]
            pub const fn to_bytes(self) -> [u8; $bytes] {
                self.0.to_le_bytes()
            }

            /// Wrap raw component bits.
            #[inline]
            #[must_use]
            pub const fn from_raw(value: $raw) -> Self {
                Self(value)
            }

            /// Unwrap to the raw component bits.
            #[inline]
            #[must_use]
            pub const fn to_raw(self) -> $raw {
                self.0
            }

            /// Field addition: component-wise XOR.
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

            #[doc = concat!("Karatsuba multiplication followed by `", $root, "^2 = ", $root, " + DELTA` reduction.")]
            ///
            /// Three base multiplies for the Karatsuba products plus one for
            /// the `DELTA` fold, versus four for the schoolbook form.
            #[inline]
            #[must_use]
            pub const fn mul(self, rhs: Self) -> Self {
                let (a, b) = self.to_components();
                let (c, d) = rhs.to_components();
                let ac = a.mul(c);
                let bd = b.mul(d);
                let constant = ac.add(DELTA.mul(bd));
                // (a+b)(c+d) + ac = ad + bc + bd, which is the coefficient of
                // the root once bd*root^2 is rewritten as bd*root + DELTA*bd.
                let extension = a.add(b).mul(c.add(d)).add(ac);
                Self::from_components(constant, extension)
            }

            /// Square using the tower relation. Cheaper than a general
            /// multiply: cross terms vanish in characteristic two.
            #[inline]
            #[must_use]
            pub const fn square(self) -> Self {
                let (a, b) = self.to_components();
                let a2 = a.square();
                let b2 = b.square();
                Self::from_components(a2.add(DELTA.mul(b2)), b2)
            }

            /// Multiplicative inverse through the quadratic conjugate and
            /// norm.
            ///
            #[doc = concat!("`conj(a + b*", $root, ") = (a + b) + b*", $root, "` and `norm = a^2 + a*b + DELTA*b^2`,")]
            /// so the inverse is `conj / norm` with a single base-field
            /// inversion.
            #[inline]
            #[must_use]
            pub const fn inv(self) -> Self {
                let (a, b) = self.to_components();
                if a.to_raw() == 0 && b.to_raw() == 0 {
                    return Self::ZERO;
                }
                let norm = a.square().add(a.mul(b)).add(DELTA.mul(b.square()));
                let norm_inv = norm.inv();
                Self::from_components(a.add(b).mul(norm_inv), b.mul(norm_inv))
            }

            /// Field division. Returns zero when either operand is zero.
            ///
            /// `x / 0 == 0` is a definition, not an oversight: keeping
            /// division total leaves hot loops branch-free and keeps this
            /// callable from `const` context. A `debug_assert!` here would
            /// turn any `const` division by zero into a compile error in
            /// debug builds and silence in release ones.
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

        impl ElemTrait for Elem {
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

        impl Field for $marker {
            type Elem = Elem;

            const NAME: &'static str = $name;
            const BITS: u32 = $bits;
            const BYTES: usize = $bytes;
            const ORDER: u128 = $order;
            const CHARACTERISTIC: u64 = 2;
            const GENERATOR: Elem = GENERATOR;

            #[inline]
            fn decode(bytes: &[u8]) -> Elem {
                let bytes: [u8; $bytes] = bytes
                    .try_into()
                    .expect(concat!($name, " element has the wrong byte width"));
                Elem::from_bytes(bytes)
            }

            #[inline]
            fn encode(bytes: &mut [u8], value: Elem) {
                assert_eq!(
                    bytes.len(),
                    $bytes,
                    concat!($name, " element has the wrong byte width")
                );
                bytes.copy_from_slice(&value.to_bytes());
            }
        }

        impl fmt::Debug for Elem {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let (a, b) = self.to_components();
                write!(f, $debug_fmt, a.to_raw(), b.to_raw())
            }
        }

        impl fmt::Display for Elem {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $display_fmt, self.0)
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

        impl core::ops::Neg for Elem {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Elem::neg(self)
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
    };
}

pub mod gf16 {
    //! GF(2^16) as a quadratic tower over [`crate::field::gf8b`].
    //!
    //! With `F = GF(2^8)` under the Rijndael polynomial, an element here is
    //! `a + b*u` with `a, b in F` and
    //!
    //! ```text
    //! u^2 + u + DELTA = 0,   DELTA = 0x20.
    //! ```
    //!
    //! The absolute trace of `0x20` in `F` is one, so the quadratic is
    //! irreducible.
    //!
    //! # Why a tower and not a primitive 16-bit polynomial
    //!
    //! A flat GF(2^16) needs either a 128 KiB log table (blows L1 and is not
    //! vectorizable) or a four-level nibble shuffle. The tower instead reduces
    //! every 16-bit multiply to base-field multiplies, which the *byte-wide*
    //! hardware — `GF2P8MULB` on x86, `PMULL`/nibble-shuffle on NEON — already
    //! does 16 or 32 lanes at a time. The whole 16-bit kernel is then two
    //! byte-wide multiplies plus a lane swap.
    //!
    //! # Representation
    //!
    //! The stable byte form is `[a, b]`: constant component first, extension
    //! component second. That is exactly the little-endian encoding of the
    //! wrapped `u16`, so a buffer of GF(2^16) elements is a byte-interleaved
    //! pair of base-field planes at stride two.
    //!
    //! ```
    //! use fgf::gf16::{self, Elem, DELTA};
    //! use fgf::gf8b;
    //!
    //! let x = Elem::from_components(gf8b::Elem::from_raw(0x12), gf8b::Elem::from_raw(0x34));
    //! assert_eq!(x.to_raw(), 0x3412);
    //! assert_eq!(x.to_bytes(), [0x12, 0x34]);
    //!
    //! // The defining relation: u^2 == u + DELTA.
    //! const U: Elem = Elem::from_components(gf8b::Elem::ZERO, gf8b::Elem::ONE);
    //! const DELTA_LIFTED: Elem = Elem::from_components(DELTA, gf8b::Elem::ZERO);
    //! const _: () = assert!(U.square().to_raw() == U.add(DELTA_LIFTED).to_raw());
    //!
    //! // The documented generator really does have order 65535.
    //! assert_eq!(gf16::GENERATOR.pow(65_535), Elem::ONE);
    //!
    //! // Division is total: `x / 0` is zero, in `const` context too.
    //! const _: () = assert!(Elem::from_raw(0x0108).div(Elem::ZERO).to_raw() == 0);
    //! ```

    quad_tower!(
        Gf16,
        u16,
        crate::field::gf8b::Elem,
        u8,
        8,
        0x20,
        0x0108,
        "u",
        "GF(2^16)",
        16,
        2,
        65_536,
        "Gf16({:#04x}+{:#04x}u)",
        "{:04x}",
    );
}

pub mod gf32 {
    //! GF(2^32) as a quadratic tower over [`crate::field::gf16`].
    //!
    //! An element is `a + b*v` with `a, b in GF(2^16)` and
    //!
    //! ```text
    //! v^2 + v + DELTA = 0,   DELTA = 0x2000.
    //! ```
    //!
    //! The absolute trace of `DELTA` in GF(2^16) is one, so the quadratic is
    //! irreducible. The stable byte form is the little-endian component encoding
    //! `[a, b]`.
    //!
    //! ```
    //! use fgf::gf32::{self, Elem};
    //! use fgf::gf16;
    //!
    //! const X: Elem = Elem::from_components(gf16::Elem::from_raw(0x1234), gf16::Elem::from_raw(0x5678));
    //! const _: () = assert!(X.to_raw() == 0x5678_1234);
    //!
    //! // `inv` is `const`, so a reciprocal table can be a `const` item.
    //! const RECIP: Elem = X.inv();
    //! const _: () = assert!(X.mul(RECIP).to_raw() == Elem::ONE.to_raw());
    //!
    //! // Division is total: `x / 0` is zero, in `const` context too.
    //! const _: () = assert!(X.div(Elem::ZERO).to_raw() == 0);
    //!
    //! assert_eq!(X.to_components(), (gf16::Elem::from_raw(0x1234), gf16::Elem::from_raw(0x5678)));
    //! assert_eq!(gf32::GENERATOR.pow(u64::from(u32::MAX)), Elem::ONE);
    //! ```

    quad_tower!(
        Gf32,
        u32,
        super::gf16::Elem,
        u16,
        16,
        0x2000,
        0x0001_0002,
        "v",
        "GF(2^32)",
        32,
        4,
        1u128 << 32,
        "Gf32({:#06x}+{:#06x}v)",
        "{:08x}",
    );
}

pub mod gf64 {
    //! GF(2^64) as a quadratic tower over [`crate::field::gf32`].
    //!
    //! An element is `a + b*w` with `a, b in GF(2^32)` and
    //!
    //! ```text
    //! w^2 + w + DELTA = 0,   DELTA = 0x20000000.
    //! ```
    //!
    //! The absolute trace of `DELTA` in GF(2^32) is one, so the quadratic is
    //! irreducible. The stable byte form is the little-endian component encoding
    //! `[a, b]`.
    //!
    //! ```
    //! use fgf::gf64::Elem;
    //! use fgf::gf32;
    //!
    //! let x = Elem::from_components(gf32::Elem::from_raw(0xdead_beef), gf32::Elem::from_raw(0x0123_4567));
    //! assert_eq!(x.to_bytes(), 0x0123_4567_dead_beefu64.to_le_bytes());
    //! assert_eq!(Elem::from_bytes(x.to_bytes()), x);
    //! assert_eq!(Elem::from_raw(x.to_raw()), x);
    //!
    //! // Division is total: `x / 0` is zero, in `const` context too.
    //! const _: () = assert!(Elem::from_raw(7).div(Elem::ZERO).to_raw() == 0);
    //!
    //! // Summing a row is XOR, the parity operation the vector kernels vectorize.
    //! let row = [Elem::from_raw(1), Elem::from_raw(2), Elem::from_raw(3)];
    //! assert_eq!(row.iter().sum::<Elem>(), Elem::from_raw(1 ^ 2 ^ 3));
    //! ```

    quad_tower!(
        Gf64,
        u64,
        super::gf32::Elem,
        u32,
        32,
        0x2000_0000,
        0x0000_0001_0000_0004,
        "w",
        "GF(2^64)",
        64,
        8,
        1u128 << 64,
        "Gf64({:#010x}+{:#010x}w)",
        "{:016x}",
    );
}
