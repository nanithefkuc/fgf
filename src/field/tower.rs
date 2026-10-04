//! The quadratic tower family: GF(2^16), GF(2^32), GF(2^64).
//!
//! GF(2^16) is a degree-two extension of a byte field under the relation
//! `t^2 + A*t + B = 0`: an element is `a + b*t` for base-field components
//! `a, b`, stored low component first. [`TowerSpec`] pins the base field and
//! the relation; [`Tower<S>`] is the representation, so every tower shares
//! one implementation — Karatsuba multiplication with the relation fold,
//! squaring with vanishing cross terms, and inversion through the conjugate
//! and norm — instantiated once per spec. [`RijndaelTower`], the tower over
//! [`Gf8<Poly<AES>>`](crate::field::Gf8) rooted at the AES field, is today's
//! GF(2^16) and is aliased by [`gf16`].
//!
//! The wider levels are fixed Rijndael-rooted towers over the level below:
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

use core::fmt;

use super::poly::{AES, Poly};
use super::repr::private;
use super::repr::{ByteRepr, Elem, Gf, Repr};
use super::{Field, FieldElem};

/// The defining data of one quadratic tower over a byte field.
///
/// The relation is `t^2 + A*t + B = 0` over the degree-eight base
/// [`TowerSpec::Base`], with `A` and `B` raw bytes in the base's encoding.
/// The spec is sealed: the crate's towers are fixed, and each carries its
/// [`Tower::SPEC`] and [`Tower::GENERATOR_VALID`] use-time checks.
#[allow(private_bounds)]
pub trait TowerSpec:
    Copy + core::fmt::Debug + core::hash::Hash + PartialEq + Eq + 'static + private::Sealed
{
    /// The degree-8 base representation.
    type Base: ByteRepr;
    /// The linear coefficient `A` of the relation, raw in `Base`'s encoding.
    const A: u8;
    /// The constant coefficient `B` of the relation, raw in `Base`'s
    /// encoding.
    const B: u8;
    /// The pinned generator of the order-`2^16 - 1` multiplicative group,
    /// raw `lo | hi << 8`.
    ///
    /// Pinned rather than derived: a const search over GF(2^16) exceeds the
    /// const evaluator's step budget.
    const GENERATOR: u16;
    /// Human-readable name of the tower field.
    const NAME: &'static str;
}

/// The quadratic tower representation `GF(2^16)` over the spec's base.
///
/// The raw form of an element is `lo | hi << 8` for the components
/// `lo + hi*t`, so the wire encoding is `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Tower<S: TowerSpec>(core::marker::PhantomData<S>);

impl<S: TowerSpec> private::Sealed for Tower<S> {}

impl<S: TowerSpec> Repr<16> for Tower<S> {
    type Raw = u16;

    const NAME: &'static str = S::NAME;
    const ONE_RAW: u16 = u16::from_le_bytes([<S::Base as Repr<8>>::ONE_RAW, 0]);
    const GENERATOR_RAW: u16 = S::GENERATOR;
}

/// Whether `S::A` is nonzero: the first conjunct of [`Tower::SPEC`].
pub(crate) const fn spec_a_nonzero<S: TowerSpec>() -> bool {
    S::A != 0
}

/// Whether `S::A` is the base's multiplicative identity: the relation's
/// linear coefficient is then a unit, so the arithmetic formulas can drop
/// its multiplies.
pub(crate) const fn spec_a_is_one<S: TowerSpec>() -> bool {
    S::A == <S::Base as Repr<8>>::ONE_RAW
}

/// Whether the absolute trace of `S::B / S::A^2` over the base is one: the
/// second conjunct of [`Tower::SPEC`], and the irreducibility criterion of
/// the relation.
pub(crate) const fn spec_trace_one<S: TowerSpec>() -> bool {
    let a = Elem::<8, S::Base>::from_raw(S::A);
    let b = Elem::<8, S::Base>::from_raw(S::B);
    // Absolute trace over the base: the sum of the conjugates.
    let mut acc = b.div(a.square());
    let mut power = acc;
    let mut i = 1;
    while i < 8 {
        power = power.square();
        acc = acc.add(power);
        i += 1;
    }
    acc.to_raw() == <S::Base as Repr<8>>::ONE_RAW
}

impl<S: TowerSpec> Tower<S> {
    /// The checked relation of this tower.
    ///
    /// `A` nonzero and the absolute trace of `B / A^2` over the base one,
    /// which makes `t^2 + A*t + B` irreducible and the tower a field.
    ///
    /// Referencing this constant evaluates the check, so an invalid spec
    /// fails to compile at the use site.
    pub const SPEC: () = {
        assert!(spec_a_nonzero::<S>(), "TowerSpec A must be nonzero");
        assert!(
            spec_trace_one::<S>(),
            "TowerSpec B must have absolute trace one over A squared"
        );
    };

    /// The checked generator of this tower.
    ///
    /// `S::GENERATOR` generates the order-`2^16 - 1` multiplicative group:
    /// no proper cofactor power `65535 / q`, for the prime factors
    /// `q` of `65535`, returns it to one.
    pub const GENERATOR_VALID: () = {
        let one = Elem::<16, Tower<S>>::ONE.0;
        let g = Elem::<16, Tower<S>>(S::GENERATOR);
        assert!(
            g.pow(65535 / 3).0 != one
                && g.pow(65535 / 5).0 != one
                && g.pow(65535 / 17).0 != one
                && g.pow(65535 / 257).0 != one,
            "TowerSpec GENERATOR must have order 65535"
        );
    };
}

/// The Rijndael-rooted tower: GF(2^16) over [`Gf8<Poly<AES>>`](crate::field::Gf8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct RijndaelTower;

impl private::Sealed for RijndaelTower {}

impl TowerSpec for RijndaelTower {
    type Base = Poly<AES>;
    const A: u8 = 1;
    const B: u8 = 0x20;
    const GENERATOR: u16 = 0x0108;
    const NAME: &'static str = "GF(2^16)";
}

impl<S: TowerSpec> Elem<16, Tower<S>> {
    /// The additive identity, and the absorbing element for multiplication.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(u16::from_le_bytes([<S::Base as Repr<8>>::ONE_RAW, 0]));
    /// The canonical generator of the multiplicative group.
    ///
    /// The pinned generator of the spec, validated by the spec's order
    /// check.
    pub const GENERATOR: Self = {
        // Reading the check forces its evaluation; the unit value is the
        // point.
        #[allow(clippy::let_unit_value)]
        #[allow(clippy::ignored_unit_patterns)]
        let _ = Tower::<S>::GENERATOR_VALID;
        Self(S::GENERATOR)
    };

    /// Construct `a + b*t` from its two base-field components.
    #[inline]
    #[must_use]
    pub const fn from_components(a: Elem<8, S::Base>, b: Elem<8, S::Base>) -> Self {
        Self(u16::from_le_bytes([a.to_raw(), b.to_raw()]))
    }

    /// Return the `(a, b)` components of `a + b*t`.
    #[inline]
    #[must_use]
    // Splitting the raw word into halves: the truncation IS the operation.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn to_components(self) -> (Elem<8, S::Base>, Elem<8, S::Base>) {
        (
            Elem::<8, S::Base>::from_raw(self.0 as u8),
            Elem::<8, S::Base>::from_raw((self.0 >> 8) as u8),
        )
    }

    /// Decode from the stable little-endian component representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian component representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Wrap raw component bits.
    ///
    /// Referencing this constructor evaluates the spec's validity checks, so
    /// a spec with an invalid relation or generator fails to compile at the
    /// use site.
    // Reading the checks forces their evaluation; the unit value is the point.
    #[allow(clippy::let_unit_value)]
    #[allow(clippy::ignored_unit_patterns)]
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u16) -> Self {
        let _ = Tower::<S>::SPEC;
        let _ = Tower::<S>::GENERATOR_VALID;
        Self(value)
    }

    /// Unwrap to the raw component bits.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u16 {
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

    /// Karatsuba multiplication followed by the `t^2 = A*t + B` reduction.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        let con = Elem::<8, S::Base>::from_raw(S::B);
        let (x, y) = self.to_components();
        let (c, d) = rhs.to_components();
        let xc = x.mul(c);
        let yd = y.mul(d);
        let constant = xc.add(con.mul(yd));
        // (x+y)(c+d) + xc expands to `xd + yc + yd`, which is the extension
        // component when the relation's linear coefficient is the base's
        // one: the `A*yd` term the relation contributes is then `yd`
        // itself. Any other `A` cancels that `yd` back out and multiplies
        // it in, leaving `xd + yc + A*yd`.
        let extension = if spec_a_is_one::<S>() {
            x.add(y).mul(c.add(d)).add(xc)
        } else {
            let lin = Elem::<8, S::Base>::from_raw(S::A);
            x.add(y).mul(c.add(d)).add(xc).add(yd).add(lin.mul(yd))
        };
        Self::from_components(constant, extension)
    }

    /// Square using the tower relation. Cheaper than a general multiply:
    /// cross terms vanish in characteristic two.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        let con = Elem::<8, S::Base>::from_raw(S::B);
        let (x, y) = self.to_components();
        let y2 = y.square();
        let extension = if spec_a_is_one::<S>() {
            y2
        } else {
            Elem::<8, S::Base>::from_raw(S::A).mul(y2)
        };
        Self::from_components(x.square().add(con.mul(y2)), extension)
    }

    /// The quadratic conjugate `conj(a + b*t) = (a + A*b) + b*t`.
    #[inline]
    #[must_use]
    pub(crate) const fn conjugate(self) -> Self {
        let (x, y) = self.to_components();
        let constant = if spec_a_is_one::<S>() {
            x.add(y)
        } else {
            x.add(Elem::<8, S::Base>::from_raw(S::A).mul(y))
        };
        Self::from_components(constant, y)
    }

    /// The norm to the base field: `a^2 + A*a*b + B*b^2`.
    #[inline]
    #[must_use]
    pub(crate) const fn norm(self) -> Elem<8, S::Base> {
        let con = Elem::<8, S::Base>::from_raw(S::B);
        let (x, y) = self.to_components();
        let middle = if spec_a_is_one::<S>() {
            x.mul(y)
        } else {
            Elem::<8, S::Base>::from_raw(S::A).mul(x.mul(y))
        };
        x.square().add(middle).add(con.mul(y.square()))
    }

    /// Multiplicative inverse through the quadratic conjugate and norm.
    ///
    /// The inverse is `conj / norm` with a single base-field inversion.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        let norm_inv = self.norm().inv();
        let conjugate = self.conjugate();
        let (x, y) = conjugate.to_components();
        Self::from_components(x.mul(norm_inv), y.mul(norm_inv))
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division
    /// total leaves hot loops branch-free and keeps this callable from
    /// `const` context.
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

impl<S: TowerSpec> FieldElem for Elem<16, Tower<S>> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::sub(self, rhs)
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::mul(self, rhs)
    }
    #[inline]
    fn square(self) -> Self {
        Self::square(self)
    }
    #[inline]
    fn inv(self) -> Self {
        Self::inv(self)
    }
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::div(self, rhs)
    }
    #[inline]
    fn pow(self, exponent: u64) -> Self {
        Self::pow(self, exponent)
    }
}

impl<S: TowerSpec> Field for Gf<16, Tower<S>> {
    type Elem = Elem<16, Tower<S>>;

    const NAME: &'static str = S::NAME;
    const BITS: u32 = 16;
    const BYTES: usize = 2;
    const ORDER: u128 = 65_536;
    const CHARACTERISTIC: u64 = 2;
    const GENERATOR: Elem<16, Tower<S>> = Elem::<16, Tower<S>>::GENERATOR;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<16, Tower<S>> {
        let bytes: [u8; 2] = bytes
            .try_into()
            .expect("GF(2^16) element has the wrong byte width");
        Self::Elem::from_bytes(bytes)
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<16, Tower<S>>) {
        assert_eq!(bytes.len(), 2, "GF(2^16) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl<S: TowerSpec> fmt::Display for Elem<16, Tower<S>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}", self.0)
    }
}

pub mod gf16 {
    //! GF(2^16) as the Rijndael-rooted quadratic tower over
    //! [`Gf8<Poly<AES>>`](crate::field::Gf8).
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
    //! use fgf::poly::AES;
    //! use fgf::{Elem as Byte, Poly};
    //!
    //! let base = |raw: u8| Byte::<8, Poly<AES>>::from_raw(raw);
    //! let x = Elem::from_components(base(0x12), base(0x34));
    //! assert_eq!(x.to_raw(), 0x3412);
    //! assert_eq!(x.to_bytes(), [0x12, 0x34]);
    //!
    //! // The defining relation: u^2 == u + DELTA.
    //! const U: Elem = Elem::from_components(
    //!     Byte::<8, Poly<AES>>::ZERO,
    //!     Byte::<8, Poly<AES>>::ONE,
    //! );
    //! const DELTA_LIFTED: Elem =
    //!     Elem::from_components(DELTA, Byte::<8, Poly<AES>>::ZERO);
    //! const _: () = assert!(U.square().to_raw() == U.add(DELTA_LIFTED).to_raw());
    //!
    //! // The documented generator really does have order 65535.
    //! assert_eq!(gf16::GENERATOR.pow(65_535), Elem::ONE);
    //!
    //! // Division is total: `x / 0` is zero, in `const` context too.
    //! const _: () = assert!(Elem::from_raw(0x0108).div(Elem::ZERO).to_raw() == 0);
    //! ```

    use super::RijndaelTower;
    use super::TowerSpec;
    use crate::field::poly::{AES, Poly};
    use crate::field::repr::Gf;
    use crate::field::tower::Tower;

    /// Marker for GF(2^16):
    /// [`Gf<16, Tower<RijndaelTower>>`](Gf).
    pub type Gf16 = Gf<16, Tower<RijndaelTower>>;

    /// An element of GF(2^16), stored as `a + b*u` with `a` in the low half.
    ///
    /// Every bit pattern is a distinct field value, so the
    /// [`PartialEq`]/[`Hash`]/[`Ord`] — raw-bit order — compare field
    /// values. That order is a deterministic total order for map keys and
    /// sorting; no order compatible with addition exists in characteristic
    /// two.
    pub type Elem = crate::field::repr::Elem<16, Tower<RijndaelTower>>;

    /// Constant term of the irreducible tower polynomial `u^2 + u + DELTA`.
    pub const DELTA: crate::field::repr::Elem<8, Poly<AES>> =
        <crate::field::repr::Elem<8, Poly<AES>>>::from_raw(RijndaelTower::B);

    /// A primitive element of the extension field: its multiplicative
    /// order is the whole group.
    pub const GENERATOR: Elem = Elem::GENERATOR;
}

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
        use crate::field::{Field, FieldElem as ElemTrait};

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

/// The Reed–Solomon-rooted tower: a second, crate-internal spec whose
/// relation has a non-unit linear coefficient and whose base is not the AES
/// field. Test-only — it exists to prove the degree-16 arithmetic and the
/// kernels generic over the spec.
///
/// The relation is `t^2 + 2*t + 0x80 = 0` over `Poly<REED_SOLOMON>` (the
/// absolute trace of `0x80 / 2^2` is one), and the root itself generates the
/// multiplicative group.
#[cfg(test)]
pub(crate) mod rs_tower {
    use super::super::poly::{Poly, REED_SOLOMON};
    use super::super::repr::{Elem, Gf};
    use super::{Tower, TowerSpec, private};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct RsTower;

    impl private::Sealed for RsTower {}

    impl TowerSpec for RsTower {
        type Base = Poly<REED_SOLOMON>;
        const A: u8 = 2;
        const B: u8 = 0x80;
        const GENERATOR: u16 = 0x0100;
        const NAME: &'static str = "GF(2^16)/0x11D";
    }

    pub(crate) type RsElem = Elem<16, Tower<RsTower>>;
    pub(crate) type RsField = Gf<16, Tower<RsTower>>;
}

/// An AES-base tower with a non-unit linear coefficient: the only spec
/// that drives the GFNI `Compact` route with a general relation. Found by
/// the same independent Python model of `GF(2^8)/0x11B` as the
/// Reed–Solomon-rooted tower: `A = 2` differs from one, `B = 0x01` is the
/// smallest byte with `Tr(B / A^2) = 1`, and `GENERATOR = 0x0104` is the
/// smallest raw word whose cofactor powers avoid one.
#[cfg(test)]
pub(crate) mod aes_tower {
    use super::super::poly::{AES, Poly};
    use super::super::repr::{Elem, Gf};
    use super::{Tower, TowerSpec, private};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct AesTower;

    impl private::Sealed for AesTower {}

    impl TowerSpec for AesTower {
        type Base = Poly<AES>;
        const A: u8 = 2;
        const B: u8 = 0x01;
        const GENERATOR: u16 = 0x0104;
        const NAME: &'static str = "GF(2^16)/0x11B/A2";
    }

    #[allow(dead_code)]
    pub(crate) type AesElem = Elem<16, Tower<AesTower>>;
    #[allow(dead_code)]
    pub(crate) type AesField = Gf<16, Tower<AesTower>>;
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::vec;
    use std::vec::Vec;

    use super::super::poly::{AES, Poly, REED_SOLOMON};
    use super::super::repr::Elem;
    use super::aes_tower::AesTower;
    use super::rs_tower::RsTower;
    use super::{Field, FieldElem, RijndaelTower, Tower, TowerSpec, private};
    use super::{spec_a_nonzero, spec_trace_one};

    type RijndaelElem = Elem<16, Tower<RijndaelTower>>;

    /// A spec whose linear coefficient is zero: rejected by the first
    /// conjunct of the spec check.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct ZeroA;

    impl private::Sealed for ZeroA {}

    impl TowerSpec for ZeroA {
        type Base = Poly<REED_SOLOMON>;
        const A: u8 = 0;
        const B: u8 = 0x80;
        const GENERATOR: u16 = 0x0100;
        const NAME: &'static str = "GF(2^16)/A0";
    }

    /// A spec whose constant term has absolute trace zero over `A^2`: the
    /// relation is reducible and rejected by the second conjunct.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct TraceZeroB;

    impl private::Sealed for TraceZeroB {}

    impl TowerSpec for TraceZeroB {
        type Base = Poly<REED_SOLOMON>;
        const A: u8 = 1;
        const B: u8 = 1;
        const GENERATOR: u16 = 0x0100;
        const NAME: &'static str = "GF(2^16)/T0";
    }

    #[test]
    fn spec_check_rejects_invalid_conjuncts() {
        assert!(spec_a_nonzero::<RijndaelTower>());
        assert!(spec_trace_one::<RijndaelTower>());
        assert!(!spec_a_nonzero::<ZeroA>());
        assert!(!spec_trace_one::<TraceZeroB>());
    }
    /// Independent schoolbook oracle over one tower's base: reduce
    /// `(a + b t)(c + d t)` by `t^2 = A t + B` using the base's own
    /// multiply, never the tower's formulas.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::many_single_char_names)]
    fn tower_schoolbook<S: TowerSpec>(x: u16, y: u16) -> u16 {
        let mul = |p: u8, q: u8| {
            Elem::<8, S::Base>::from_raw(p)
                .mul(Elem::<8, S::Base>::from_raw(q))
                .to_raw()
        };
        let (a, b) = (x as u8, (x >> 8) as u8);
        let (c, d) = (y as u8, (y >> 8) as u8);
        let constant = mul(a, c) ^ mul(S::B, mul(b, d));
        let extension = mul(a, d) ^ mul(b, c) ^ mul(S::A, mul(b, d));
        u16::from_le_bytes([constant, extension])
    }

    #[allow(clippy::cast_possible_truncation)]
    fn tower_sample(generator: u16) -> Vec<u16> {
        let mut values = vec![0u16, 1, 2, 0x00ff, 0x0100, 0xff00, 0xffff, generator];
        let mut state = 0x0123_4567_89ab_cdefu64;
        for _ in 0..128 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            values.push((state >> 32) as u16);
        }
        values
    }

    fn check_tower_matches_schoolbook<S: TowerSpec>() {
        for &x in &tower_sample(S::GENERATOR) {
            for &y in &tower_sample(S::GENERATOR) {
                let got = Elem::<16, Tower<S>>::from_raw(x)
                    .mul(Elem::<16, Tower<S>>::from_raw(y))
                    .to_raw();
                assert_eq!(got, tower_schoolbook::<S>(x, y), "{x:04x} * {y:04x}");
            }
        }
    }

    fn check_tower_field_laws<S: TowerSpec>() {
        let one = Elem::<16, Tower<S>>::ONE;
        for &x in &tower_sample(S::GENERATOR) {
            let a = Elem::<16, Tower<S>>::from_raw(x);
            assert_eq!(a.add(a), Elem::<16, Tower<S>>::ZERO, "x + x");
            assert_eq!(a.sub(a), Elem::<16, Tower<S>>::ZERO, "x - x");
            assert_eq!(a.mul(one), a, "x * 1");
            assert_eq!(a.square(), a.mul(a), "x^2");
            if !a.is_zero() {
                assert_eq!(a.mul(a.inv()), one, "x * x^-1");
                assert_eq!(a.mul(a.inv()).mul(a), a, "round trip");
            }
            assert_eq!(
                a.div(Elem::<16, Tower<S>>::ZERO),
                Elem::<16, Tower<S>>::ZERO,
                "x / 0"
            );
            assert_eq!(a.pow(0), one, "x^0");
            // The conjugate and norm satisfy `x * conj(x) = N(x)` lifted.
            let norm = a.norm();
            assert_eq!(
                a.mul(a.conjugate()),
                Elem::<16, Tower<S>>::from_components(norm, Elem::<8, S::Base>::ZERO),
                "x * conj(x) = N(x)"
            );
        }
    }

    fn check_tower_generator_order<S: TowerSpec>() {
        let g = Elem::<16, Tower<S>>::GENERATOR;
        assert_eq!(g.pow(65_535), Elem::<16, Tower<S>>::ONE);
        for q in [3u64, 5, 17, 257] {
            assert_ne!(g.pow(65_535 / q), Elem::<16, Tower<S>>::ONE, "cofactor {q}");
        }
    }

    #[test]
    fn rs_tower_matches_schoolbook() {
        check_tower_matches_schoolbook::<RsTower>();
    }

    #[test]
    fn rs_tower_field_laws() {
        check_tower_field_laws::<RsTower>();
    }

    #[test]
    fn rs_tower_generator_order() {
        check_tower_generator_order::<RsTower>();
    }

    #[test]
    fn aes_tower_matches_schoolbook() {
        check_tower_matches_schoolbook::<AesTower>();
    }

    #[test]
    fn aes_tower_field_laws() {
        check_tower_field_laws::<AesTower>();
    }

    #[test]
    fn aes_tower_generator_order() {
        check_tower_generator_order::<AesTower>();
    }

    #[test]
    fn rijndael_tower_wire_values_hold() {
        // Byte-identical wire facts of the Rijndael alias: raw layout,
        // relation, generator order, and total division.
        const _: () = assert!(
            RijndaelElem::from_raw(0x0108)
                .div(RijndaelElem::ZERO)
                .to_raw()
                == 0
        );
        let base = |raw: u8| Elem::<8, Poly<AES>>::from_raw(raw);
        let x = RijndaelElem::from_components(base(0x12), base(0x34));
        assert_eq!(x.to_raw(), 0x3412);
        assert_eq!(x.to_bytes(), [0x12, 0x34]);
        let u =
            RijndaelElem::from_components(Elem::<8, Poly<AES>>::ZERO, Elem::<8, Poly<AES>>::ONE);
        let delta = RijndaelElem::from_components(super::gf16::DELTA, Elem::<8, Poly<AES>>::ZERO);
        assert_eq!(u.square(), u.add(delta));
        assert_eq!(super::gf16::GENERATOR.pow(65_535), RijndaelElem::ONE);
    }

    #[test]
    fn tower_field_impl_facts() {
        use super::gf16::Gf16;
        assert_eq!(<Gf16 as Field>::NAME, "GF(2^16)");
        assert_eq!(<Gf16 as Field>::BITS, 16);
        assert_eq!(<Gf16 as Field>::BYTES, 2);
        assert_eq!(<Gf16 as Field>::ORDER, 65_536);
        assert_eq!(<Gf16 as Field>::CHARACTERISTIC, 2);
        let mut bytes = [0u8; 2];
        <Gf16 as Field>::encode(&mut bytes, RijndaelElem::from_raw(0x0108));
        assert_eq!(bytes, [0x08, 0x01]);
        assert_eq!(
            <Gf16 as Field>::decode(&bytes).to_raw(),
            0x0108,
            "decode/encode round trip"
        );
        assert_eq!(format!("{}", RijndaelElem::from_raw(0x0108)), "0108");
    }
}
