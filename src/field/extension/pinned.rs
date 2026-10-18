//! The shipped instances: their pinned descriptors, structural membership
//! in that set, and the specialized scalar bodies a member runs.
//!
//! A descriptor structurally equal to a pinned descriptor skips the
//! irreducibility test: the shipped row's validity holds through its own
//! known-answer fixtures, and every pinned descriptor's irreducibility is
//! proven by the full test in this module's tests. A row whose descriptor
//! is pinned also runs that instance's specialized scalar body, over the
//! instance word its storage converts to: the coordinate word of a packed
//! storage, and `re | im << 32` for the prime extension's limbs.

use super::descriptor::Descriptor;
use crate::field::binary::tower::{
    self, FanPaar2Field, FanPaar4Field, FanPaar8Field, FanPaar16Field, FanPaar32Field,
    FanPaar64Field, Rijndael16Field, Rijndael32Field, Rijndael64Field,
};
use crate::field::private::RowAccess;
use crate::field::quad_mersenne31::{self, QuadMersenne31};

/// One shipped extension instance, naming its specialized scalar body.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Instance {
    /// GF(2^16) over the AES field.
    Rijndael16,
    /// GF(2^32) over Rijndael GF(2^16).
    Rijndael32,
    /// GF(2^64) over Rijndael GF(2^32).
    Rijndael64,
    /// Fan-Paar GF(2^2) over GF(2).
    FanPaar2,
    /// Fan-Paar GF(2^4).
    FanPaar4,
    /// Fan-Paar GF(2^8).
    FanPaar8,
    /// Fan-Paar GF(2^16).
    FanPaar16,
    /// Fan-Paar GF(2^32).
    FanPaar32,
    /// Fan-Paar GF(2^64).
    FanPaar64,
    /// GF((2^31 - 1)^2) over the Mersenne fold row.
    QuadMersenne31,
}

/// The shipped instances, in [`PINNED`] order.
const INSTANCES: [Instance; 10] = [
    Instance::Rijndael16,
    Instance::Rijndael32,
    Instance::Rijndael64,
    Instance::FanPaar2,
    Instance::FanPaar4,
    Instance::FanPaar8,
    Instance::FanPaar16,
    Instance::FanPaar32,
    Instance::FanPaar64,
    Instance::QuadMersenne31,
];

/// The pinned shipped descriptors: every shipped instance row's descriptor,
/// in [`INSTANCES`] order.
#[allow(clippy::large_const_arrays)]
pub const PINNED: [Descriptor; 10] = [
    *<Rijndael16Field as RowAccess>::DESCRIPTOR,
    *<Rijndael32Field as RowAccess>::DESCRIPTOR,
    *<Rijndael64Field as RowAccess>::DESCRIPTOR,
    *<FanPaar2Field as RowAccess>::DESCRIPTOR,
    *<FanPaar4Field as RowAccess>::DESCRIPTOR,
    *<FanPaar8Field as RowAccess>::DESCRIPTOR,
    *<FanPaar16Field as RowAccess>::DESCRIPTOR,
    *<FanPaar32Field as RowAccess>::DESCRIPTOR,
    *<FanPaar64Field as RowAccess>::DESCRIPTOR,
    *<QuadMersenne31 as RowAccess>::DESCRIPTOR,
];

/// The shipped instance structurally equal to `candidate`, if any.
pub const fn instance(candidate: &Descriptor) -> Option<Instance> {
    let mut i = 0;
    while i < PINNED.len() {
        if candidate.same_structure(&PINNED[i]) {
            return Some(INSTANCES[i]);
        }
        i += 1;
    }
    None
}

/// Whether `candidate` is structurally equal to a pinned descriptor.
pub const fn is_pinned(candidate: &Descriptor) -> bool {
    instance(candidate).is_some()
}

/// The two limbs of a prime-extension instance word.
// Splitting the word into its two limbs: the truncation is the operation.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn limbs(word: u64) -> [u32; 2] {
    [word as u32, (word >> 32) as u32]
}

/// The instance word of two prime-extension limbs.
#[inline]
const fn limb_word(limbs: [u32; 2]) -> u64 {
    (limbs[0] as u64) | ((limbs[1] as u64) << 32)
}

// The specialized bodies over instance words. Every packed word holds the
// row's coordinate word, so the narrowing casts drop only zero bits.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
impl Instance {
    /// The canonical form of an instance word.
    #[inline]
    pub const fn canonical(self, word: u64) -> u64 {
        match self {
            Self::FanPaar2 => word & 0x3,
            Self::FanPaar4 => word & 0xF,
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_canonical(limbs(word))),
            _ => word,
        }
    }

    /// Field addition of two canonical instance words.
    #[inline]
    pub const fn add(self, left: u64, right: u64) -> u64 {
        match self {
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_add(limbs(left), limbs(right))),
            _ => left ^ right,
        }
    }

    /// Field subtraction of two canonical instance words.
    #[inline]
    pub const fn sub(self, left: u64, right: u64) -> u64 {
        match self {
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_sub(limbs(left), limbs(right))),
            _ => left ^ right,
        }
    }

    /// The additive inverse of a canonical instance word.
    #[inline]
    pub const fn neg(self, value: u64) -> u64 {
        match self {
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_neg(limbs(value))),
            _ => value,
        }
    }

    /// Field multiplication of two canonical instance words.
    #[inline]
    pub const fn mul(self, left: u64, right: u64) -> u64 {
        match self {
            Self::Rijndael16 => tower::rijndael16_mul(left as u16, right as u16) as u64,
            Self::Rijndael32 => tower::rijndael32_mul(left as u32, right as u32) as u64,
            Self::Rijndael64 => tower::rijndael64_mul(left, right),
            Self::FanPaar2 => tower::FANPAAR2_DESC.mul(left, right),
            Self::FanPaar4 => tower::FANPAAR4_DESC.mul(left, right),
            Self::FanPaar8 => tower::FANPAAR8_DESC.mul(left, right),
            Self::FanPaar16 => tower::fp_multiply(left, right, 16),
            Self::FanPaar32 => tower::fp_multiply(left, right, 32),
            Self::FanPaar64 => tower::fp_multiply(left, right, 64),
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_mul(limbs(left), limbs(right))),
        }
    }

    /// The multiplicative inverse of a canonical instance word, zero to
    /// zero.
    #[inline]
    pub const fn inv(self, value: u64) -> u64 {
        match self {
            Self::Rijndael16 => tower::rijndael16_inv(value as u16) as u64,
            Self::Rijndael32 => tower::rijndael32_inv(value as u32) as u64,
            Self::Rijndael64 => tower::rijndael64_inv(value),
            Self::FanPaar2 => tower::FANPAAR2_DESC.inv(value),
            Self::FanPaar4 => tower::FANPAAR4_DESC.inv(value),
            Self::FanPaar8 => tower::FANPAAR8_DESC.inv(value),
            Self::FanPaar16 => tower::fp_invert(value, 16),
            Self::FanPaar32 => tower::fp_invert(value, 32),
            Self::FanPaar64 => tower::fp_invert(value, 64),
            Self::QuadMersenne31 => limb_word(quad_mersenne31::raw_inv(limbs(value))),
        }
    }
}

/// The shipped instances' descriptors rebuilt from their defining table
/// data — base presentations, storages, and quadratic coefficient words —
/// with no constant published by the rows.
#[cfg(test)]
pub(crate) mod shipped_instances {
    use super::super::descriptor::{Descriptor, PrimePresentation, Storage};

    /// The AES GF(2^8) leaf.
    pub(crate) fn aes_leaf() -> Descriptor {
        Descriptor::polynomial(8, 0x11B)
    }

    /// The GF(2) leaf the Fan-Paar chain rises from.
    pub(crate) fn gf2_leaf() -> Descriptor {
        Descriptor::polynomial(1, 0b11)
    }

    /// The flattened relation coordinates of `t^2 + linear*t + constant`
    /// over a binary base of absolute degree `base`, constant term first.
    const fn quadratic_coordinates(base: usize, constant: u64, linear: u64) -> [u64; 64] {
        let mut out = [0u64; 64];
        let mut i = 0;
        while i < base {
            out[i] = (constant >> i) & 1;
            out[base + i] = (linear >> i) & 1;
            i += 1;
        }
        out
    }

    /// One binary tower level over `base` with the table's storage word and
    /// coefficient words.
    pub(crate) fn binary_level(
        base: &Descriptor,
        padded: bool,
        width: u8,
        constant: u64,
        linear: u64,
    ) -> Descriptor {
        let base_degree = base.degree() as usize;
        let degree = 2 * base.degree();
        let relation = &quadratic_coordinates(base_degree, constant, linear)[..degree as usize];
        let storage = if padded {
            Storage::PackedPadded { width }
        } else {
            Storage::Packed { width }
        };
        Descriptor::extension(base, 2, degree, 2, relation, storage)
    }

    /// `GF(2^16)` over the AES leaf: `t^2 + t + 0x20`.
    pub(crate) fn rijndael16() -> Descriptor {
        binary_level(&aes_leaf(), false, 16, 0x20, 0x01)
    }

    /// `GF(2^32)` over Rijndael16: `t^2 + t + 0x2000`.
    pub(crate) fn rijndael32() -> Descriptor {
        binary_level(&rijndael16(), false, 32, 0x2000, 0x0001)
    }

    /// `GF(2^64)` over Rijndael32: `t^2 + t + 0x2000_0000`.
    pub(crate) fn rijndael64() -> Descriptor {
        binary_level(&rijndael32(), false, 64, 0x2000_0000, 0x0000_0001)
    }

    /// Fan-Paar `GF(2^2)` over GF(2): `t^2 + t + 1`.
    pub(crate) fn fanpaar2() -> Descriptor {
        binary_level(&gf2_leaf(), true, 8, 1, 1)
    }

    /// Fan-Paar `GF(2^4)`: `t^2 + 0x2*t + 1`.
    pub(crate) fn fanpaar4() -> Descriptor {
        binary_level(&fanpaar2(), true, 8, 1, 0x2)
    }

    /// Fan-Paar `GF(2^8)`: `t^2 + 0x4*t + 1`.
    pub(crate) fn fanpaar8() -> Descriptor {
        binary_level(&fanpaar4(), false, 8, 1, 0x4)
    }

    /// Fan-Paar `GF(2^16)`: `t^2 + 0x10*t + 1`.
    pub(crate) fn fanpaar16() -> Descriptor {
        binary_level(&fanpaar8(), false, 16, 1, 0x10)
    }

    /// Fan-Paar `GF(2^32)`: `t^2 + 0x0100*t + 1`.
    pub(crate) fn fanpaar32() -> Descriptor {
        binary_level(&fanpaar16(), false, 32, 1, 0x0100)
    }

    /// Fan-Paar `GF(2^64)`: `t^2 + 0x0001_0000*t + 1`.
    pub(crate) fn fanpaar64() -> Descriptor {
        binary_level(&fanpaar32(), false, 64, 1, 0x0001_0000)
    }

    /// `QuadMersenne31` over the Mersenne fold leaf: `t^2 + 1`.
    pub(crate) fn quad_mersenne31() -> Descriptor {
        Descriptor::extension(
            &Descriptor::prime(0x7FFF_FFFF, PrimePresentation::MersenneFold),
            0x7FFF_FFFF,
            2,
            2,
            &[1, 0],
            Storage::Components,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::descriptor::{Descriptor, PrimePresentation, Storage};
    use super::PINNED;
    use super::shipped_instances as table;
    use crate::field::BinaryField;
    use crate::field::QuadMersenne31;
    use crate::field::binary::description::BinaryDescription;
    use crate::field::binary::tower::{
        FanPaar2Field, FanPaar4Field, FanPaar8Field, FanPaar16Field, FanPaar32Field,
        FanPaar64Field, Rijndael16Field, Rijndael32Field, Rijndael64Field,
    };
    use crate::field::private::RowAccess;

    #[test]
    fn pinned_descriptors_prove_irreducible_by_the_full_test() {
        for (index, descriptor) in PINNED.iter().enumerate() {
            assert!(
                descriptor.is_irreducible(),
                "pinned descriptor {index} has a reducible relation"
            );
        }
    }

    /// Every pinned entry and every shipped row's binary description equals
    /// the descriptor built from the instance's table row.
    #[test]
    fn pinned_entries_and_row_hooks_match_the_instance_table() {
        // One descriptor occupies the shared node array's width.
        #[allow(clippy::large_stack_arrays)]
        let expected: [Descriptor; 10] = [
            table::rijndael16(),
            table::rijndael32(),
            table::rijndael64(),
            table::fanpaar2(),
            table::fanpaar4(),
            table::fanpaar8(),
            table::fanpaar16(),
            table::fanpaar32(),
            table::fanpaar64(),
            table::quad_mersenne31(),
        ];
        let hooks: [&Descriptor; 10] = [
            <Rijndael16Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <Rijndael32Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <Rijndael64Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar2Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar4Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar8Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar16Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar32Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <FanPaar64Field as BinaryField>::DESCRIPTION.as_descriptor(),
            <QuadMersenne31 as RowAccess>::DESCRIPTOR,
        ];
        for index in 0..expected.len() {
            assert!(
                PINNED[index].same_structure(&expected[index]),
                "pinned entry {index} differs from its instance table row"
            );
            assert!(
                hooks[index].same_structure(&expected[index]),
                "row hook {index} differs from its instance table row"
            );
        }
    }

    #[test]
    fn pinned_membership_decides_by_structure() {
        // A custom quadratic tower over the AES field is not pinned.
        let custom_row =
            BinaryDescription::append_quadratic(&BinaryDescription::polynomial(8, 0x11B), 1, 0x21);
        let custom = custom_row.as_descriptor();
        assert!(!super::is_pinned(custom));
        // The QuadMersenne31 relation rebuilt from scratch stays pinned.
        let rebuilt = Descriptor::extension(
            &Descriptor::prime(0x7FFF_FFFF, PrimePresentation::MersenneFold),
            0x7FFF_FFFF,
            2,
            2,
            &[1, 0],
            Storage::Components,
        );
        assert!(super::is_pinned(&rebuilt));
        // The same modulus under the residue presentation is a different
        // row and is not pinned.
        let residue_row = Descriptor::extension(
            &Descriptor::prime(0x7FFF_FFFF, PrimePresentation::Residue),
            0x7FFF_FFFF,
            2,
            2,
            &[1, 0],
            Storage::Components,
        );
        assert!(!super::is_pinned(&residue_row));
    }
}
