#![allow(clippy::cast_possible_truncation)]

//! Optimized Fan–Paar arithmetic over canonical Wiedemann tower fields.
//!
//! Starting from GF(2), each level doubles the extension degree. If `alpha`
//! is the previous level's tower generator, the next generator `X` satisfies
//!
//! ```text
//! X^2 + alpha*X + 1 = 0.
//! ```
//!
//! Elements use the canonical recursive tower basis: the low half is the
//! constant component and the high half is the `X` component. This is the
//! representation standardized by binary-field proof systems. Squaring and
//! multiplication by a tower generator are linear XOR-only recurrences;
//! inversion descends to one inversion in the direct subfield.
//!
//! # Layout
//!
//! One submodule per level, each shaped like [`crate::field::gf16`]: a marker
//! type, an `Elem`, the level's tower generator [`fp16::ALPHA`], and a
//! primitive [`fp16::GENERATOR`].
//!
//! | Level | Module | Marker | Element |
//! | --- | --- | --- | --- |
//! | GF(2^8) | [`fp8`] | [`FanPaar8`] | [`fp8::Elem`] |
//! | GF(2^16) | [`fp16`] | [`FanPaar16`] | [`fp16::Elem`] |
//! | GF(2^32) | [`fp32`] | [`FanPaar32`] | [`fp32::Elem`] |
//! | GF(2^64) | [`fp64`] | [`FanPaar64`] | [`fp64::Elem`] |
//!
//! The levels nest: a byte embedded in a wider level keeps its bit pattern
//! and its products.
//!
//! ```
//! use fgf::fan_paar::{FanPaar16, fp8, fp16};
//! use fgf::field::Field;
//!
//! let (a, b) = (fp8::Elem::from_raw(0x1b), fp8::Elem::from_raw(0xa8));
//! assert_eq!(a.mul(b), fp8::Elem::from_raw(0x09));
//! assert_eq!(
//!     fp16::Elem::from_components(a, fp8::Elem::ZERO)
//!         .mul(fp16::Elem::from_components(b, fp8::Elem::ZERO)),
//!     fp16::Elem::from_components(fp8::Elem::from_raw(0x09), fp8::Elem::ZERO),
//! );
//!
//! // The defining quadratic, checked at compile time: X^2 = alpha*X + 1,
//! // with `alpha` the subfield's tower generator lifted into this level.
//! const X: fp16::Elem = fp16::ALPHA;
//! const ALPHA: fp16::Elem = fp16::Elem::from_components(fp8::ALPHA, fp8::Elem::ZERO);
//! const _: () = assert!(X.square().to_raw() == X.mul(ALPHA).add(fp16::Elem::ONE).to_raw());
//!
//! // Multiplying by ALPHA is the XOR-only recurrence.
//! assert_eq!(fp16::Elem::from_raw(0x1234).mul_alpha(), fp16::Elem::from_raw(0x1234).mul(fp16::ALPHA));
//!
//! assert_eq!(FanPaar16::NAME, "Fan-Paar GF(2^16)");
//! ```

#[inline]
const fn low_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Multiply by the top-level tower generator of a `bits`-wide field.
const fn mul_alpha(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    a1 | ((a0 ^ mul_alpha(a1, half)) << half)
}

const fn build_exp8() -> [u8; 255] {
    let mut table = [0u8; 255];
    let mut value = 1u8;
    let mut i = 0;
    while i < table.len() {
        table[i] = value;
        value = super::wiedemann::multiply(value as u64, 0x2d, 8) as u8;
        i += 1;
    }
    table
}

const EXP8: [u8; 255] = build_exp8();

#[allow(clippy::cast_possible_truncation)]
const fn build_log8() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < EXP8.len() {
        table[EXP8[i] as usize] = i as u8;
        i += 1;
    }
    table
}

const LOG8: [u8; 256] = build_log8();

/// Recursive Fan–Paar multiplication with a log-table byte base.
const fn multiply(lhs: u64, rhs: u64, bits: u32) -> u64 {
    if bits == 8 {
        if lhs == 0 || rhs == 0 {
            return 0;
        }
        let log = LOG8[lhs as usize] as usize + LOG8[rhs as usize] as usize;
        return EXP8[log % 255] as u64;
    }
    if bits == 1 {
        return lhs & rhs & 1;
    }
    let half = bits / 2;
    let mask = low_mask(half);
    let a0 = lhs & mask;
    let a1 = lhs >> half;
    let b0 = rhs & mask;
    let b1 = rhs >> half;
    let z0 = multiply(a0, b0, half);
    let z2 = multiply(a1, b1, half);
    let z1 = multiply(a0 ^ a1, b0 ^ b1, half) ^ z0 ^ z2;
    (z0 ^ z2) | ((z1 ^ mul_alpha(z2, half)) << half)
}

/// Recursive squaring. This is linear: no general field multiplication.
const fn square(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    let z0 = square(a0, half);
    let z2 = square(a1, half);
    (z0 ^ z2) | (mul_alpha(z2, half) << half)
}

/// Recursive conjugate-and-norm inversion, with `inv(0) = 0`.
const fn invert(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    let a0z1 = a0 ^ mul_alpha(a1, half);
    let norm = multiply(a0, a0z1, half) ^ square(a1, half);
    let norm_inv = invert(norm, half);
    multiply(norm_inv, a0z1, half) | (multiply(norm_inv, a1, half) << half)
}

/// Emit one tower level as its own module, shaped like [`crate::field::gf16`].
///
/// The optional `base` group is the half-width level's element type and raw
/// integer; the lowest level has no in-crate subfield type, so it is omitted
/// and the component accessors are not generated for it.
macro_rules! define_fan_paar_level {
    (
        $module:ident,
        $field:ident,
        $raw:ty,
        $bits:literal,
        $bytes:literal,
        $generator:expr,
        $field_name:literal,
        $module_doc:literal,
        $elem_doc:literal,
        $field_doc:literal
        $(, base = $base:path, base_raw = $base_raw:ty)?
    ) => {
        #[doc = $module_doc]
        #[doc = ""]
        #[doc = "```"]
        #[doc = concat!("use fgf::fan_paar::", stringify!($module), "::{ALPHA, Elem, GENERATOR};")]
        #[doc = ""]
        #[doc = "assert_eq!(Elem::ONE.mul_alpha(), ALPHA);"]
        #[doc = "assert_eq!(Elem::from_raw(GENERATOR.to_raw()), GENERATOR);"]
        #[doc = "assert_eq!(GENERATOR.mul(GENERATOR.inv()), Elem::ONE);"]
        #[doc = ""]
        #[doc = "// Division is total, in `const` context too."]
        #[doc = "const _: () = assert!(Elem::ONE.div(Elem::ZERO).to_raw() == 0);"]
        #[doc = "```"]
        pub mod $module {
            use core::fmt;

            use crate::field::{Field, FieldBuffer, HasGenerator, PrimeCharacteristic};

            #[doc = $field_doc]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
            pub struct $field;

            #[doc = $elem_doc]
            #[doc = ""]
            #[doc = "Every bit pattern is a distinct field value, so the"]
            #[doc = "`Ord` — raw-bit order — compares field values. It is a deterministic"]
            #[doc = "total order for map keys and sorting; no order compatible with"]
            #[doc = "addition exists in characteristic two."]
            pub type Elem = crate::field::Elem<$field>;

            /// This level's tower generator `X`: the basis element of the high
            /// half, and the constant [`Elem::mul_alpha`] multiplies by.
            ///
            /// With `alpha` the tower generator of the half-width subfield,
            /// `X^2 + alpha*X + 1 = 0`. It is *not* a primitive element; see
            /// [`GENERATOR`] for that.
            pub const ALPHA: Elem = Elem { raw: 1 << ($bits / 2) };

            /// A generator of the multiplicative group of this level.
            pub const GENERATOR: Elem = Elem::GENERATOR;

            impl crate::field::Elem<$field> {
                /// Decode from the canonical little-endian tower representation.
                #[inline]
                #[must_use]
                pub const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                    Self {
                        raw: <$raw>::from_le_bytes(bytes),
                    }
                }

                /// Encode to the canonical little-endian tower representation.
                #[inline]
                #[must_use]
                pub const fn to_bytes(self) -> [u8; $bytes] {
                    self.raw.to_le_bytes()
                }

                /// Wrap raw canonical tower-basis bits.
                #[inline]
                #[must_use]
                pub const fn from_raw(value: $raw) -> Self {
                    let () = crate::field::Validate::<$field>::OK;
                    Self { raw: value }
                }

                /// Field addition: component-wise XOR.
                #[inline]
                #[must_use]
                pub const fn add(self, rhs: Self) -> Self {
                    Self {
                        raw: self.raw ^ rhs.raw,
                    }
                }

                /// Field subtraction. Identical to addition.
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

                /// Recursive Karatsuba multiplication in the Wiedemann tower.
                #[inline]
                #[must_use]
                // Widening a raw lane to the tower word is exact; `const`
                // rules out `u64::from`.
                #[allow(clippy::cast_lossless)]
                pub const fn mul(self, rhs: Self) -> Self {
                    Self {
                        raw: super::multiply(self.raw as u64, rhs.raw as u64, $bits) as $raw,
                    }
                }

                /// Square through the linear Wiedemann recurrence.
                #[inline]
                #[must_use]
                #[allow(clippy::cast_lossless)]
                pub const fn square(self) -> Self {
                    Self {
                        raw: super::square(self.raw as u64, $bits) as $raw,
                    }
                }

                /// Multiply by [`ALPHA`], this field's top-level tower generator.
                #[inline]
                #[must_use]
                #[allow(clippy::cast_lossless)]
                pub const fn mul_alpha(self) -> Self {
                    Self {
                        raw: super::mul_alpha(self.raw as u64, $bits) as $raw,
                    }
                }

                /// Multiplicative inverse. Maps zero to zero by convention.
                #[inline]
                #[must_use]
                #[allow(clippy::cast_lossless)]
                pub const fn inv(self) -> Self {
                    Self {
                        raw: super::invert(self.raw as u64, $bits) as $raw,
                    }
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
                    if self.raw == 0 || rhs.raw == 0 {
                        return Self::ZERO;
                    }
                    self.mul(rhs.inv())
                }

                /// Raise to an unsigned integer power.
                #[must_use]
                pub const fn pow(self, mut exponent: u128) -> Self {
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

            $(
                impl crate::field::Elem<$field> {
                    /// Construct `a + b*ALPHA` from its two subfield components.
                    #[inline]
                    #[must_use]
                    pub const fn from_components(a: $base, b: $base) -> Self {
                        Self {
                            raw: (a.to_raw() as $raw) | ((b.to_raw() as $raw) << ($bits / 2)),
                        }
                    }

                    /// Return the `(a, b)` components of `a + b*ALPHA`.
                    ///
                    /// The low half is the constant component, the high half the
                    /// [`ALPHA`] component.
                    #[inline]
                    #[must_use]
                    #[allow(clippy::cast_possible_truncation)]
                    pub const fn to_components(self) -> ($base, $base) {
                        (
                            <$base>::from_raw(self.raw as $base_raw),
                            <$base>::from_raw((self.raw >> ($bits / 2)) as $base_raw),
                        )
                    }
                }
            )?

            impl Field for $field {
                type Raw = $raw;
                type Characteristic = PrimeCharacteristic<2>;

                const NAME: &'static str = $field_name;
                const DEGREE: u32 = $bits;
                const ORDER: u128 = 1u128 << $bits;
                const ZERO_RAW: $raw = 0;
                const ONE_RAW: $raw = 1;
                const VALID: () = ();

                #[inline]
                fn canonical_raw(raw: $raw) -> $raw {
                    raw
                }

                #[inline]
                fn add_raw(left: $raw, right: $raw) -> $raw {
                    left ^ right
                }

                #[inline]
                fn sub_raw(left: $raw, right: $raw) -> $raw {
                    left ^ right
                }

                #[inline]
                fn neg_raw(value: $raw) -> $raw {
                    value
                }

                #[inline]
                #[allow(clippy::cast_lossless)]
                fn mul_raw(left: $raw, right: $raw) -> $raw {
                    super::multiply(left as u64, right as u64, $bits) as $raw
                }

                #[inline]
                #[allow(clippy::cast_lossless)]
                fn inv_raw(value: $raw) -> $raw {
                    super::invert(value as u64, $bits) as $raw
                }
            }

            impl HasGenerator for $field {
                const GENERATOR_RAW: $raw = $generator;
            }

            impl FieldBuffer for $field {
                const BYTES: usize = $bytes;
                const STORAGE_BITS: u32 = $bits;

                #[inline]
                fn decode(bytes: &[u8]) -> Elem {
                    let bytes: [u8; $bytes] = bytes
                        .try_into()
                        .expect("Fan-Paar element has the wrong byte width");
                    Elem::from_bytes(bytes)
                }

                #[inline]
                fn encode(bytes: &mut [u8], value: Elem) {
                    assert_eq!(
                        bytes.len(),
                        $bytes,
                        "Fan-Paar element has the wrong byte width"
                    );
                    bytes.copy_from_slice(&value.to_bytes());
                }
            }

            impl fmt::Display for Elem {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:0width$x}", self.raw, width = 2 * $bytes)
                }
            }
        }
    };
}

define_fan_paar_level!(
    fp8,
    FanPaar8,
    u8,
    8,
    1,
    0x2d,
    "Fan-Paar GF(2^8)",
    "Canonical Fan–Paar GF(2^8), the tower's byte level.",
    "An 8-bit element in the canonical Fan–Paar tower basis.",
    "Marker type for canonical Fan–Paar GF(2^8)."
);
define_fan_paar_level!(
    fp16,
    FanPaar16,
    u16,
    16,
    2,
    0xe2de,
    "Fan-Paar GF(2^16)",
    "Canonical Fan–Paar GF(2^16), a quadratic tower over [`crate::field::fan_paar::fp8`].",
    "A 16-bit element in the canonical Fan–Paar tower basis.",
    "Marker type for canonical Fan–Paar GF(2^16).",
    base = crate::field::fan_paar::fp8::Elem,
    base_raw = u8
);
define_fan_paar_level!(
    fp32,
    FanPaar32,
    u32,
    32,
    4,
    0x03e2_1cea,
    "Fan-Paar GF(2^32)",
    "Canonical Fan–Paar GF(2^32), a quadratic tower over [`crate::field::fan_paar::fp16`].",
    "A 32-bit element in the canonical Fan–Paar tower basis.",
    "Marker type for canonical Fan–Paar GF(2^32).",
    base = crate::field::fan_paar::fp16::Elem,
    base_raw = u16
);
define_fan_paar_level!(
    fp64,
    FanPaar64,
    u64,
    64,
    8,
    0x070f_870d_cd9c_1d88,
    "Fan-Paar GF(2^64)",
    "Canonical Fan–Paar GF(2^64), a quadratic tower over [`crate::field::fan_paar::fp32`].",
    "A 64-bit element in the canonical Fan–Paar tower basis.",
    "Marker type for canonical Fan–Paar GF(2^64).",
    base = crate::field::fan_paar::fp32::Elem,
    base_raw = u32
);

pub use fp8::FanPaar8;
pub use fp16::FanPaar16;
pub use fp32::FanPaar32;
pub use fp64::FanPaar64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::wiedemann;

    #[test]
    fn optimized_scalar_matches_wiedemann_oracle() {
        assert_eq!(build_exp8(), EXP8);
        assert_eq!(build_log8(), LOG8);
        for (i, &g) in EXP8.iter().enumerate() {
            assert_eq!(LOG8[g as usize] as usize, i, "log/exp inverse at {i}");
        }

        let mut state = 0x0123_4567_89ab_cdefu64;
        for _ in 0..64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let rhs = state >> 32;
            for bits in [8u32, 16, 32, 64] {
                let mask = low_mask(bits);
                let lhs = state & mask;
                let rhs = rhs & mask;
                assert_eq!(
                    multiply(lhs, rhs, bits),
                    wiedemann::multiply(lhs, rhs, bits),
                    "multiply at {bits} bits"
                );
                assert_eq!(
                    square(lhs, bits),
                    wiedemann::square(lhs, bits),
                    "square at {bits} bits"
                );
                assert_eq!(
                    mul_alpha(lhs, bits),
                    wiedemann::mul_generator(lhs, bits),
                    "generator multiply at {bits} bits"
                );
                assert_eq!(
                    invert(lhs, bits),
                    wiedemann::invert(lhs, bits),
                    "inverse at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn generator_orbit_generates_the_byte_field() {
        let mut seen = [false; 256];
        let mut value = 1u8;
        for _ in 0..255 {
            seen[value as usize] = true;
            value = multiply(u64::from(value), 0x2d, 8) as u8;
        }
        assert!(
            seen[1..].iter().all(|&hit| hit),
            "generator misses elements"
        );
        assert_eq!(value, 1, "generator order is not 255");
    }
}
