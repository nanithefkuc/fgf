//! The pinned shipped descriptors and structural membership in that set.
//!
//! A descriptor structurally equal to a pinned descriptor skips the
//! irreducibility test: the shipped row's validity holds through its own
//! known-answer fixtures, and every pinned descriptor's irreducibility is
//! proven by the full test in this module's tests.

use super::descriptor::{Descriptor, PrimePresentation, Storage};
use crate::field::binary::BinaryRepr;
use crate::field::binary::tower::{self, FanPaar2, FanPaar4, FanPaar8, Tower};

/// The pinned shipped descriptors: every shipped tower row, taken through
/// its binary description, and the `QuadMersenne31` row over the Mersenne
/// fold leaf under `t^2 + 1`.
#[allow(clippy::large_const_arrays)]
pub const PINNED: [Descriptor; 10] = [
    *tower::RIJNDAEL16_DESC.as_descriptor(),
    *tower::RIJNDAEL32_DESC.as_descriptor(),
    *tower::RIJNDAEL64_DESC.as_descriptor(),
    *<Tower<FanPaar2> as BinaryRepr<2>>::DESCRIPTION.as_descriptor(),
    *<Tower<FanPaar4> as BinaryRepr<4>>::DESCRIPTION.as_descriptor(),
    *<Tower<FanPaar8> as BinaryRepr<8>>::DESCRIPTION.as_descriptor(),
    *tower::FANPAAR16_DESC.as_descriptor(),
    *tower::FANPAAR32_DESC.as_descriptor(),
    *tower::FANPAAR64_DESC.as_descriptor(),
    QUAD_MERSENNE31,
];

/// The pinned `QuadMersenne31` descriptor: the Mersenne fold leaf extended
/// by `t^2 + 1`.
// No production reader: the pinned set and the row's hook read it.
#[allow(dead_code)]
pub const QUAD_MERSENNE31: Descriptor = Descriptor::extension(
    &Descriptor::prime(0x7FFF_FFFF, PrimePresentation::MersenneFold),
    0x7FFF_FFFF,
    2,
    2,
    &[1, 0],
    Storage::Components,
);

/// Whether `candidate` is structurally equal to a pinned descriptor.
// No production reader: the validator entry and the tests call it.
#[allow(dead_code)]
pub const fn is_pinned(candidate: &Descriptor) -> bool {
    let mut i = 0;
    while i < PINNED.len() {
        if candidate.same_structure(&PINNED[i]) {
            return true;
        }
        i += 1;
    }
    false
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
    use crate::field::binary::description::BinaryDescription;
    use crate::field::binary::tower::{
        FanPaar16, FanPaar32, FanPaar64, Rijndael16, Rijndael32, Rijndael64,
    };
    use crate::field::private::RowAccess;
    use crate::field::{FanPaar2, FanPaar4, FanPaar8, FiniteField, QuadMersenne31, Tower};

    #[test]
    fn pinned_descriptors_prove_irreducible_by_the_full_test() {
        for (index, descriptor) in PINNED.iter().enumerate() {
            assert!(
                descriptor.is_irreducible(),
                "pinned descriptor {index} has a reducible relation"
            );
        }
    }

    /// Every pinned entry and every shipped row's hook descriptor equals
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
            <FiniteField<2, 16, Tower<Rijndael16>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 32, Tower<Rijndael32>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 64, Tower<Rijndael64>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 2, Tower<FanPaar2>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 4, Tower<FanPaar4>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 8, Tower<FanPaar8>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 16, Tower<FanPaar16>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 32, Tower<FanPaar32>> as RowAccess>::DESCRIPTOR,
            <FiniteField<2, 64, Tower<FanPaar64>> as RowAccess>::DESCRIPTOR,
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
