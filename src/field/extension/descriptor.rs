//! The general field descriptor: node data, checks, and constructors.
//!
//! A [`Descriptor`] is a topologically ordered node array, exactly like the
//! binary description it shares characteristic-two arithmetic with. The node
//! kinds split on what defines the presentation: prime leaves name a modulus
//! and its presentation, polynomial and basis leaves hold the flat binary
//! leaf data, and relation nodes hold a defining relation over an earlier
//! node in flattened prime-subfield coordinates. The element interpreter
//! lives in [`super::arithmetic`] and the irreducibility tests in
//! [`super::irreducibility`].

/// Node capacity of one descriptor.
pub const NODES: usize = 8;

/// Presentation equality as a const comparison.
const fn presentation_eq(left: PrimePresentation, right: PrimePresentation) -> bool {
    matches!(
        (left, right),
        (PrimePresentation::Residue, PrimePresentation::Residue)
            | (
                PrimePresentation::MersenneFold,
                PrimePresentation::MersenneFold
            )
            | (
                PrimePresentation::GoldilocksFold,
                PrimePresentation::GoldilocksFold
            )
    )
}

/// Storage equality as a const comparison.
const fn storage_eq(left: Storage, right: Storage) -> bool {
    match (left, right) {
        (Storage::Components, Storage::Components) => true,
        (Storage::Packed { width: a }, Storage::Packed { width: b })
        | (Storage::PackedPadded { width: a }, Storage::PackedPadded { width: b }) => a == b,
        _ => false,
    }
}

/// Prime-coordinate capacity of one relation, and of one element.
///
/// The cardinality check bounds every relation first: a characteristic-two
/// root holds at most 64 coordinates, and an odd characteristic satisfies
/// `p^degree < 2^128`, which caps `degree` at 80 because `3^81` exceeds
/// `u128`. Coefficient packing therefore never writes past this capacity.
pub const COORDINATES: usize = 80;

/// The presentation of a prime-field leaf.
///
/// Prime rows of one modulus exist under several presentations with distinct
/// raw and byte layouts; structural equality distinguishes them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrimePresentation {
    /// The ordinary-residue row: one `u64` residue per element.
    Residue,
    /// The Mersenne fold row of `2^31 - 1`: one `u32` lane per element.
    MersenneFold,
    /// The Goldilocks fold row of `2^64 - 2^32 + 1`: one `u64` lane.
    GoldilocksFold,
}

/// The element storage a relation node selects.
///
/// `Packed` and `PackedPadded` name the characteristic-two coordinate word
/// of the row; `Components` stores one base element per component.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    /// One base element per component, in the base's coordinate order.
    Components,
    /// The row's coordinate word fills the storage word exactly.
    Packed {
        /// Storage word width in bits: 8, 16, 32, or 64.
        width: u8,
    },
    /// The row's coordinate word sits below the storage word width.
    PackedPadded {
        /// Storage word width in bits: 8, 16, 32, or 64.
        width: u8,
    },
}

/// One node of a field descriptor.
// The relation variant carries its full coefficient array inline, so the
// interpreter reads the coefficients by reference through the node; the
// variant's width over the leaf variants is the cost of that layout.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
pub enum Node {
    /// An unused capacity slot past the descriptor length.
    Unused,
    /// A prime-field leaf. At characteristic two with `Residue` the word
    /// arithmetic is `AND` and the identity word is `1`.
    Prime {
        /// The prime modulus.
        modulus: u64,
        /// The presentation naming the row.
        presentation: PrimePresentation,
    },
    /// A flat polynomial root of the stated degree.
    Polynomial {
        /// Total degree of this node.
        degree: u8,
        /// The full reduction polynomial including the leading term.
        full: u128,
    },
    /// An ordered basis over an earlier node.
    Basis {
        /// Index of the base node.
        base: u8,
        /// Basis vectors in base coordinates; bit `i` selects `to_base[i]`.
        to_base: [u8; 8],
        /// Inverse columns; bit `j` of a base word selects `from_base[j]`.
        from_base: [u8; 8],
    },
    /// A defining relation over an earlier node.
    ///
    /// The cached fields hold the characteristic, the base's absolute
    /// degree, the base's interpreter word count, and the relation's
    /// non-leading coefficients in interpreter form: one packed coordinate
    /// word per coefficient in characteristic two, and `base_words` prime
    /// coordinates per coefficient otherwise, constant term first. Unused
    /// coefficient words are zero, so structural equality compares whole
    /// arrays.
    Extension {
        /// Index of the base node.
        base: u8,
        /// Degree of the relation over its base, at least two.
        relative_degree: usize,
        /// Characteristic of the root, cached from the base.
        characteristic: u64,
        /// Absolute degree of the base over the prime field.
        base_degree: u32,
        /// Interpreter word count of one base element.
        base_words: usize,
        /// Non-leading relation coefficients, constant term first.
        coefficients: [u64; COORDINATES],
        /// The storage the relation's row selects.
        storage: Storage,
    },
}

/// A field presentation as const data: a topologically ordered node array
/// with a length and a root, interpreted by the crate-private arithmetic in
/// [`super::arithmetic`].
#[derive(Clone, Copy)]
pub struct Descriptor {
    nodes: [Node; NODES],
    len: u8,
    root: u8,
}

impl Descriptor {
    /// A prime-field leaf of `modulus` under `presentation`.
    #[must_use]
    pub const fn prime(modulus: u64, presentation: PrimePresentation) -> Self {
        let mut nodes = [Node::Unused; NODES];
        nodes[0] = Node::Prime {
            modulus,
            presentation,
        };
        Self {
            nodes,
            len: 1,
            root: 0,
        }
    }

    /// A lone polynomial leaf of the stated degree.
    #[must_use]
    pub const fn polynomial(degree: u8, full: u128) -> Self {
        let mut nodes = [Node::Unused; NODES];
        nodes[0] = Node::Polynomial { degree, full };
        Self {
            nodes,
            len: 1,
            root: 0,
        }
    }

    /// Append an ordered-basis wrapper over the root of `self`.
    ///
    /// # Panics
    ///
    /// Panics when the node capacity is exhausted.
    // `NODES` is 8, far below `u8::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn append_basis(&self, to_base: [u8; 8], from_base: [u8; 8]) -> Self {
        assert!(self.len < NODES as u8, "description capacity exhausted");
        let mut next = *self;
        next.nodes[self.len as usize] = Node::Basis {
            base: self.root,
            to_base,
            from_base,
        };
        next.root = next.len;
        next.len += 1;
        next
    }

    /// Append a characteristic-two quadratic relation over the root of
    /// `self`, taking both coefficients as packed base-coordinate words.
    ///
    /// The words are stored verbatim, so a later validation can still
    /// reject excess bits: the tests build malformed nodes through it. The
    /// storage follows the packed width rule: `PackedPadded` at one byte
    /// below it, `Packed` from one byte up.
    ///
    /// # Panics
    ///
    /// Panics when the node capacity is exhausted.
    // `NODES` is 8 and the doubled base degree stays below the storage word
    // widths, both far below `u8::MAX`.
    #[cfg(test)]
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn append_quadratic_words(&self, linear: u64, constant: u64) -> Self {
        assert!(self.len < NODES as u8, "description capacity exhausted");
        let half = super::arithmetic::degree_of(self, self.root);
        let width = 2 * half;
        let storage = if width < 8 {
            Storage::PackedPadded { width: 8 }
        } else {
            Storage::Packed { width: width as u8 }
        };
        let mut coefficients = [0u64; COORDINATES];
        coefficients[0] = constant;
        coefficients[1] = linear;
        let mut next = *self;
        next.nodes[self.len as usize] = Node::Extension {
            base: self.root,
            relative_degree: 2,
            characteristic: 2,
            base_degree: half,
            base_words: super::arithmetic::element_words(self, self.root),
            coefficients,
            storage,
        };
        next.root = next.len;
        next.len += 1;
        next
    }

    /// Append a defining relation over the root of `base`, checking the
    /// relation's geometry in this order: the base characteristic, the
    /// relative degree, the degree identity, the field cardinality, the
    /// coordinate count and range, the storage admissibility, and the node
    /// capacity. Irreducibility is not checked here; callers run
    /// [`Self::validate_relation`] last.
    ///
    /// `relation` holds the non-leading coefficients in flattened
    /// prime-subfield coordinates, constant term first, each coefficient's
    /// coordinates in the base's own order. The relation length equals the
    /// root's absolute degree.
    ///
    /// # Panics
    ///
    /// Panics with a check-specific message when any check fails.
    // `NODES` is 8 and every coordinate count and width stays far below
    // `u8::MAX` and `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn extension(
        base: &Self,
        characteristic: u64,
        degree: u32,
        relative_degree: usize,
        relation: &[u64],
        storage: Storage,
    ) -> Self {
        assert!(
            relation.len() <= COORDINATES,
            "relation exceeds the coordinate capacity"
        );
        assert!(
            super::arithmetic::characteristic_of(base, base.root) == characteristic,
            "extension base characteristic differs from the field characteristic"
        );
        assert!(
            relative_degree >= 2,
            "extension relative degree is below two"
        );
        let base_degree = super::arithmetic::degree_of(base, base.root);
        let Some(product) = (base_degree as u64).checked_mul(relative_degree as u64) else {
            panic!("extension degree is not the base degree times the relative degree");
        };
        assert!(
            product == degree as u64,
            "extension degree is not the base degree times the relative degree"
        );
        crate::field::finite_field::metadata_order(characteristic, degree);
        assert!(
            relation.len() == degree as usize,
            "relation coordinate count differs from the field degree"
        );
        let mut i = 0;
        while i < relation.len() {
            assert!(
                relation[i] < characteristic,
                "relation coordinate is not below the characteristic"
            );
            i += 1;
        }
        match storage {
            Storage::Components => {}
            Storage::Packed { width } => {
                assert!(
                    characteristic == 2,
                    "packed storage requires characteristic two"
                );
                assert!(
                    matches!(width, 8 | 16 | 32 | 64),
                    "packed storage word width is unsupported"
                );
                assert!(
                    width as u32 == degree,
                    "packed storage width differs from the field degree"
                );
            }
            Storage::PackedPadded { width } => {
                assert!(
                    characteristic == 2,
                    "packed storage requires characteristic two"
                );
                assert!(
                    matches!(width, 8 | 16 | 32 | 64),
                    "packed storage word width is unsupported"
                );
                assert!(
                    degree < width as u32,
                    "padded packed storage is not narrower than its word"
                );
            }
        }
        assert!(base.len < NODES as u8, "description capacity exhausted");
        // The cardinality check above bounds `degree` by `COORDINATES`, so
        // the packing below stays inside the coefficient array.
        let base_words = super::arithmetic::element_words(base, base.root);
        let mut coefficients = [0u64; COORDINATES];
        if characteristic == 2 {
            let mut j = 0;
            while j < relative_degree {
                let mut word = 0u64;
                let mut k = 0;
                while k < base_degree as usize {
                    word |= (relation[j * base_degree as usize + k] & 1) << k;
                    k += 1;
                }
                coefficients[j] = word;
                j += 1;
            }
        } else {
            let mut i = 0;
            while i < relation.len() {
                coefficients[i] = relation[i];
                i += 1;
            }
        }
        let mut next = *base;
        next.nodes[base.len as usize] = Node::Extension {
            base: base.root,
            relative_degree,
            characteristic,
            base_degree,
            base_words,
            coefficients,
            storage,
        };
        next.root = base.len;
        next.len = base.len + 1;
        next
    }

    /// The node at `index`.
    #[must_use]
    pub const fn node(&self, index: u8) -> &Node {
        &self.nodes[index as usize]
    }

    /// Index of the root node.
    #[must_use]
    pub const fn root_index(&self) -> u8 {
        self.root
    }

    /// Number of used node slots.
    #[must_use]
    pub const fn node_count(&self) -> u8 {
        self.len
    }

    /// Characteristic of the root.
    #[must_use]
    // No production caller: the tests read it.
    #[allow(dead_code)]
    pub const fn characteristic(&self) -> u64 {
        super::arithmetic::characteristic_of(self, self.root)
    }

    /// Absolute degree of the root over the prime field.
    #[must_use]
    pub const fn degree(&self) -> u32 {
        super::arithmetic::degree_of(self, self.root)
    }

    /// The sub-descriptor rooted at the root node's base.
    ///
    /// # Panics
    ///
    /// Panics when the root node holds no base.
    // `NODES` is 8, far below `u8::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn base_descriptor(&self) -> Self {
        let base = match self.node(self.root) {
            Node::Basis { base, .. } | Node::Extension { base, .. } => *base,
            _ => panic!("the root node has no base"),
        };
        let mut sub = *self;
        sub.root = base;
        sub.len = base + 1;
        let mut index = sub.len;
        while index < NODES as u8 {
            sub.nodes[index as usize] = Node::Unused;
            index += 1;
        }
        sub
    }

    /// Structural equality ignoring unused slots.
    ///
    /// Prime leaves compare by modulus and presentation, polynomial and
    /// basis leaves by all data, and relation nodes by base index, relative
    /// degree, the full coefficient array, and storage.
    // The node cursor stays below `NODES` (8), far below `u8::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn same_structure(&self, other: &Self) -> bool {
        if self.len != other.len || self.root != other.root {
            return false;
        }
        let mut i = 0;
        while i < self.len as usize {
            let same = match (self.node(i as u8), other.node(i as u8)) {
                (
                    Node::Prime {
                        modulus: m0,
                        presentation: p0,
                    },
                    Node::Prime {
                        modulus: m1,
                        presentation: p1,
                    },
                ) => *m0 == *m1 && presentation_eq(*p0, *p1),
                (
                    Node::Polynomial {
                        degree: d0,
                        full: f0,
                    },
                    Node::Polynomial {
                        degree: d1,
                        full: f1,
                    },
                ) => *d0 == *d1 && *f0 == *f1,
                (
                    Node::Basis {
                        base: b0,
                        to_base: t0,
                        from_base: i0,
                    },
                    Node::Basis {
                        base: b1,
                        to_base: t1,
                        from_base: i1,
                    },
                ) => {
                    let mut eq = *b0 == *b1;
                    let mut k = 0;
                    while k < 8 {
                        eq = eq && t0[k] == t1[k] && i0[k] == i1[k];
                        k += 1;
                    }
                    eq
                }
                (
                    Node::Extension {
                        base: b0,
                        relative_degree: d0,
                        coefficients: c0,
                        storage: s0,
                        ..
                    },
                    Node::Extension {
                        base: b1,
                        relative_degree: d1,
                        coefficients: c1,
                        storage: s1,
                        ..
                    },
                ) => {
                    let mut eq = *b0 == *b1 && *d0 == *d1 && storage_eq(*s0, *s1);
                    let mut k = 0;
                    while k < COORDINATES {
                        eq = eq && c0[k] == c1[k];
                        k += 1;
                    }
                    eq
                }
                _ => false,
            };
            if !same {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Whether the root node's relation is irreducible over its base.
    ///
    /// Runs the full test; a descriptor structurally equal to a pinned
    /// shipped descriptor skips it only through [`Self::validate_relation`].
    ///
    /// # Panics
    ///
    /// Panics when the root node holds no relation.
    #[must_use]
    // No production caller: the tests and the const proofs run it.
    #[allow(dead_code)]
    pub const fn is_irreducible(&self) -> bool {
        super::irreducibility::root_relation_irreducible(self)
    }

    /// Validation entry point of a relation node: a pinned descriptor is
    /// accepted as shipped, every other root must prove its relation
    /// irreducible.
    ///
    /// # Panics
    ///
    /// Panics when the root relation is reducible.
    pub const fn validate_relation(&self) {
        super::irreducibility::validate_relation(self);
    }

    /// A descriptor from raw node slots, for tests that validate malformed
    /// shapes no checked constructor builds.
    #[cfg(test)]
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub const fn from_parts(nodes: [Node; NODES], len: u8, root: u8) -> Self {
        Self { nodes, len, root }
    }

    /// A copy of `self` with the node at `index` replaced, for tests.
    #[cfg(test)]
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub const fn with_node(&self, index: u8, node: Node) -> Self {
        let mut next = *self;
        next.nodes[index as usize] = node;
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::binary::description::BinaryDescription;

    const RESIDUE_SEVEN: Descriptor = Descriptor::prime(7, PrimePresentation::Residue);
    const RESIDUE_TWO: Descriptor = Descriptor::prime(2, PrimePresentation::Residue);

    /// `extension` panics with each check's own message, in the checked
    /// order: one test per message.
    fn build(
        base: &Descriptor,
        characteristic: u64,
        degree: u32,
        relative_degree: usize,
        relation: &[u64],
        storage: Storage,
    ) -> Descriptor {
        Descriptor::extension(
            base,
            characteristic,
            degree,
            relative_degree,
            relation,
            storage,
        )
    }

    #[test]
    #[should_panic(expected = "extension base characteristic differs")]
    fn extension_rejects_a_characteristic_mismatch() {
        build(&RESIDUE_SEVEN, 5, 2, 2, &[1, 0], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "extension relative degree is below two")]
    fn extension_rejects_a_linear_relation() {
        build(&RESIDUE_SEVEN, 7, 7, 1, &[0], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "extension degree is not the base degree")]
    fn extension_rejects_a_wrong_degree() {
        build(&RESIDUE_SEVEN, 7, 8, 2, &[0; 8], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "binary field degree exceeds 64")]
    fn extension_rejects_a_binary_degree_above_sixty_four() {
        let base = <crate::field::binary::tower::FanPaar64Field as crate::field::private::RowAccess>::DESCRIPTOR;
        let wide = build(base, 2, 128, 2, &[], Storage::Packed { width: 64 });
        let _ = wide;
    }

    #[test]
    #[should_panic(expected = "field order overflows u128")]
    fn extension_rejects_a_cardinality_overflow() {
        build(&RESIDUE_SEVEN, 7, 81, 81, &[], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "relation coordinate count differs")]
    fn extension_rejects_a_wrong_coordinate_count() {
        build(&RESIDUE_SEVEN, 7, 2, 2, &[0], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "relation coordinate is not below")]
    fn extension_rejects_an_out_of_range_coordinate() {
        build(&RESIDUE_SEVEN, 7, 2, 2, &[7, 0], Storage::Components);
    }

    #[test]
    #[should_panic(expected = "packed storage requires characteristic two")]
    fn extension_rejects_packed_storage_in_odd_characteristic() {
        build(
            &RESIDUE_SEVEN,
            7,
            2,
            2,
            &[0, 0],
            Storage::Packed { width: 16 },
        );
    }

    #[test]
    #[should_panic(expected = "packed storage word width is unsupported")]
    fn extension_rejects_an_unsupported_word_width() {
        build(
            &RESIDUE_TWO,
            2,
            2,
            2,
            &[1, 1],
            Storage::PackedPadded { width: 24 },
        );
    }

    #[test]
    #[should_panic(expected = "packed storage width differs from the field degree")]
    fn extension_rejects_a_packed_width_above_the_degree() {
        build(&RESIDUE_TWO, 2, 2, 2, &[1, 1], Storage::Packed { width: 8 });
    }

    #[test]
    #[should_panic(expected = "padded packed storage is not narrower")]
    fn extension_rejects_a_padded_width_at_the_degree() {
        let aes_row = BinaryDescription::polynomial(8, 0x11B);
        let aes = aes_row.as_descriptor();
        build(aes, 2, 16, 2, &[0; 16], Storage::PackedPadded { width: 16 });
    }

    #[test]
    #[should_panic(expected = "description capacity exhausted")]
    fn extension_rejects_a_full_node_array() {
        // Basis wrappers do not raise the degree, so a chain of them fills
        // the node array without tripping the cardinality bound first.
        let identity = [1u8, 2, 4, 8, 0, 0, 0, 0];
        let mut base = BinaryDescription::polynomial(8, 0x11B);
        for _ in 0..7 {
            base = BinaryDescription::append_basis(&base, identity, identity);
        }
        build(
            base.as_descriptor(),
            2,
            16,
            2,
            &[0; 16],
            Storage::Packed { width: 16 },
        );
    }

    #[test]
    fn structural_equality_distinguishes_presentations_and_data() {
        let residue = RESIDUE_SEVEN;
        let fold = Descriptor::prime(7, PrimePresentation::MersenneFold);
        // One modulus under two presentations differs.
        assert!(residue.same_structure(&residue));
        assert!(!residue.same_structure(&fold));
        // Two storages with equal relations differ.
        let components = build(&RESIDUE_TWO, 2, 2, 2, &[1, 1], Storage::Components);
        let padded = build(
            &RESIDUE_TWO,
            2,
            2,
            2,
            &[1, 1],
            Storage::PackedPadded { width: 8 },
        );
        assert!(!components.same_structure(&padded));
        // Two coefficient arrays differ.
        let other = build(&RESIDUE_TWO, 2, 2, 2, &[1, 0], Storage::Components);
        assert!(!components.same_structure(&other));
        assert!(components.same_structure(&build(
            &RESIDUE_TWO,
            2,
            2,
            2,
            &[1, 1],
            Storage::Components
        )));
    }

    #[test]
    fn accessors_read_the_root() {
        let quadratic = build(&RESIDUE_SEVEN, 7, 2, 2, &[3, 0], Storage::Components);
        assert_eq!(quadratic.characteristic(), 7);
        assert_eq!(quadratic.degree(), 2);
        // The sub-descriptor restricts to the prime leaf under its own
        // presentation.
        let base = quadratic.base_descriptor();
        assert_eq!(base.degree(), 1);
        assert_eq!(base.characteristic(), 7);
        assert!(base.same_structure(&RESIDUE_SEVEN));
    }
}
