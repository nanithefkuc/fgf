//! The defining-relation data an extension spec carries.
//!
//! A [`Relation`] holds the non-leading coefficients of the monic polynomial
//! `t^D + c_{D-1} t^{D-1} + ... + c_1 t + c_0` as flattened prime-subfield
//! coordinates, constant term first, each coefficient's coordinates in the
//! base's own order. The flattened form is identical for every storage of
//! the same base presentation.

use super::descriptor::COORDINATES;
use crate::field::BinaryField;
use crate::field::Field;

/// The defining relation of an extension row, as flattened prime-subfield
/// coordinates.
///
/// Opaque and `Copy`; built only through the const constructors. A
/// construction past the coordinate capacity, or from a binary word with
/// bits at or above the base's degree, panics with a message naming the
/// constructor and its operand, in const evaluation and at runtime alike.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Relation {
    coordinates: [u64; COORDINATES],
    len: u8,
}

impl Relation {
    /// A relation from flattened prime-subfield coordinates, constant term
    /// first.
    ///
    /// # Panics
    ///
    /// Panics when `coordinates` holds more than the coordinate capacity.
    #[must_use]
    // The capacity check bounds the length below `u8::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_coordinates(coordinates: &[u64]) -> Self {
        assert!(
            coordinates.len() <= COORDINATES,
            "Relation::from_coordinates: coordinates: relation exceeds the coordinate capacity"
        );
        let mut out = [0; COORDINATES];
        let mut i = 0;
        while i < coordinates.len() {
            out[i] = coordinates[i];
            i += 1;
        }
        Self {
            coordinates: out,
            len: coordinates.len() as u8,
        }
    }

    /// A relation over a binary base, taking one coordinate word per
    /// coefficient.
    ///
    /// Each word expands to the base's coordinate bits, least significant
    /// first, so `[constant, linear]` names `t^2 + linear*t + constant` in
    /// the base's own packing.
    ///
    /// Only `B::DEGREE` is read; the relation is coordinate data. The base
    /// row is validated where the relation is used: an [`Extension`](crate::extension::Extension) over an
    /// invalid base fails const validation like any other invalid row.
    ///
    /// # Panics
    ///
    /// Panics when a word holds bits at or above the base's degree, or the
    /// expansion exceeds the coordinate capacity.
    // The casts are the const form of the lossless conversions: `From` is
    // not const-callable on the crate MSRV, and every value is bounded by
    // the capacity checks above.
    #[must_use]
    #[allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
    pub const fn from_binary_words<B: BinaryField>(words: &[u64]) -> Self {
        let degree = <B as Field>::DEGREE;
        let Some(total) = (words.len() as u64).checked_mul(degree as u64) else {
            panic!("Relation::from_binary_words: words: relation exceeds the coordinate capacity");
        };
        assert!(
            total <= COORDINATES as u64,
            "Relation::from_binary_words: words: relation exceeds the coordinate capacity"
        );
        let mut out = [0; COORDINATES];
        let mut i = 0;
        while i < words.len() {
            let word = words[i];
            assert!(
                degree >= 64 || word >> degree == 0,
                "Relation::from_binary_words: words: relation word has bits at or above the base degree"
            );
            let mut j = 0;
            while j < degree as usize {
                out[i * degree as usize + j] = (word >> j) & 1;
                j += 1;
            }
            i += 1;
        }
        Self {
            coordinates: out,
            len: total as u8,
        }
    }

    /// The flattened prime-subfield coordinates, constant term first.
    #[must_use]
    pub const fn to_coordinates(&self) -> &[u64] {
        self.coordinates.split_at(self.len as usize).0
    }

    /// The relation as one coordinate word per coefficient over the binary
    /// base `B`, constant term first: the form
    /// [`from_binary_words`](Self::from_binary_words) accepts.
    ///
    /// Allocation-free; the words are assembled as the iterator advances.
    ///
    /// # Panics
    ///
    /// Panics when the coordinate count is not a multiple of `B::DEGREE`, or
    /// a coordinate is not a bit.
    #[must_use = "the word iterator is lazy; nothing is read unless it is consumed"]
    pub fn to_binary_words<B: BinaryField>(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        let degree = <B as Field>::DEGREE as usize;
        let coordinates = self.to_coordinates();
        assert!(
            coordinates.len().is_multiple_of(degree),
            "Relation::to_binary_words: relation of {} coordinates is not a whole number of degree-{degree} words",
            coordinates.len()
        );
        assert!(
            coordinates.iter().all(|&bit| bit <= 1),
            "Relation::to_binary_words: relation holds a coordinate that is not a bit"
        );
        coordinates.chunks_exact(degree).map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u64, |word, (i, &bit)| word | (bit << i))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::COORDINATES;
    use super::Relation;
    use crate::field::FiniteField;
    use crate::field::binary::Polynomial;

    type Gf4 = FiniteField<2, 2, Polynomial<7>>;

    #[test]
    fn const_construction_reads_back() {
        const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
        const BINARY: Relation = Relation::from_binary_words::<Gf4>(&[2, 1]);
        const _: () = assert!(BINARY.to_coordinates().len() == 4);
        const _: () = assert!(BINARY.to_coordinates()[0] == 0);

        // Runtime construction walks the same copy and expansion loops as
        // const evaluation.
        let runtime = Relation::from_coordinates(&[4, 0]);
        assert_eq!(runtime.to_coordinates(), &[4, 0]);
        let runtime_binary = Relation::from_binary_words::<Gf4>(&[2, 1]);
        assert_eq!(runtime_binary.to_coordinates(), &[0, 1, 1, 0]);

        assert_eq!(RELATION.to_coordinates(), &[4, 0]);

        // One word per coefficient over GF(4): constant `2`, linear `1`,
        // expanding each word least significant first.
        assert_eq!(BINARY.to_coordinates(), &[0, 1, 1, 0]);
    }

    #[test]
    #[should_panic(expected = "relation exceeds the coordinate capacity")]
    fn capacity_panics() {
        const OVER: [u64; COORDINATES + 1] = [1; COORDINATES + 1];
        let relation = Relation::from_coordinates(&OVER);
        assert_eq!(relation.to_coordinates().len(), COORDINATES + 1);
    }

    #[test]
    #[should_panic(expected = "relation word has bits at or above the base degree")]
    fn binary_word_above_degree_panics() {
        // GF(4) words hold at most two bits.
        const BAD: [u64; 1] = [4];
        let relation = Relation::from_binary_words::<Gf4>(&BAD);
        assert_eq!(relation.to_coordinates().len(), 2);
    }

    #[test]
    #[should_panic(expected = "relation exceeds the coordinate capacity")]
    fn binary_expansion_capacity_panics() {
        // Sixty words of two coordinates each exceed the capacity.
        const LONG: [u64; 60] = [1; 60];
        let relation = Relation::from_binary_words::<Gf4>(&LONG);
        assert_eq!(relation.to_coordinates().len(), 120);
    }
}
