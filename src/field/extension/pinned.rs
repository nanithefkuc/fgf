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
#[allow(dead_code)]
// No production caller: the validator reaches this set through
// `validate_relation`, and tests exercise it directly. The array is `const`
// so structural comparison reads it in const evaluation.
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
    Descriptor::extension(
        &Descriptor::prime(0x7FFF_FFFF, PrimePresentation::MersenneFold),
        0x7FFF_FFFF,
        2,
        2,
        &[1, 0],
        Storage::Components,
    ),
];

/// Whether `candidate` is structurally equal to a pinned descriptor.
#[allow(dead_code)]
// No production caller: the validator reaches this test through
// `validate_relation`, and tests exercise it directly.
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

/// The index of the pinned `QuadMersenne31` descriptor in [`PINNED`].
// No production caller: tests read this entry while sampling elements.
#[allow(dead_code)]
pub const QUAD_MERSENNE31: usize = 9;

#[cfg(test)]
mod tests {
    use super::super::descriptor::{Descriptor, PrimePresentation, Storage};
    use super::PINNED;
    use crate::field::binary::description::BinaryDescription;

    #[test]
    fn pinned_descriptors_prove_irreducible_by_the_full_test() {
        for (index, descriptor) in PINNED.iter().enumerate() {
            assert!(
                descriptor.is_irreducible(),
                "pinned descriptor {index} has a reducible relation"
            );
        }
    }

    #[test]
    fn pinned_membership_matches_the_set() {
        for descriptor in &PINNED {
            assert!(super::is_pinned(descriptor));
        }
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
