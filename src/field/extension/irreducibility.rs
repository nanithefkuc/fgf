//! Irreducibility of a relation node over its base, decided exactly in
//! const: the quadratic criteria as a fast path, and the generalized Ben-Or
//! test for every higher relative degree.
//!
//! The quadratic test reads nonzero `c_1` plus the absolute trace of
//! `c_0 / c_1^2` equal to one in characteristic two — against the base
//! node's identity word, which a basis leaf fixes — and a nonresidue
//! discriminant `c_1^2 - 4*c_0` by Euler's criterion otherwise. The general
//! test checks `gcd(f, t^(q^i) - t) = 1` for `1 <= i <= floor(D/2)`, `q` the
//! base order, exiting at the first nontrivial gcd.

use super::arithmetic::{
    WIDE, Wide, element_eq, element_is_zero, element_mul, element_one, element_pow, element_sub,
    order_of, relation_coefficient, wide_get, wide_put, word_inv, word_mul, word_one,
};
use super::descriptor::{Descriptor, Node};
use super::pinned;

/// Whether the relation node at `index` is irreducible over its base, by
/// the quadratic fast path at relative degree two and the generalized
/// Ben-Or test otherwise.
///
/// # Panics
///
/// Panics when the node at `index` holds no relation.
#[must_use]
pub const fn node_relation_irreducible(descriptor: &Descriptor, index: u8) -> bool {
    match descriptor.node(index) {
        Node::Extension {
            relative_degree, ..
        } => {
            if *relative_degree == 2 {
                quadratic_relation_holds(descriptor, index)
            } else {
                ben_or_irreducible(descriptor, index)
            }
        }
        _ => panic!("not a relation node"),
    }
}

/// Whether the root node's relation is irreducible over its base.
///
/// # Panics
///
/// Panics when the root node holds no relation.
#[must_use]
pub const fn root_relation_irreducible(descriptor: &Descriptor) -> bool {
    node_relation_irreducible(descriptor, descriptor.root_index())
}

/// Validation entry point of a relation node: a descriptor structurally
/// equal to a pinned shipped descriptor is accepted as shipped, every other
/// root must prove its relation irreducible.
///
/// # Panics
///
/// Panics when the root relation is reducible.
// No production caller: the tests reach the underlying test directly.
#[allow(dead_code)]
pub const fn validate_relation(descriptor: &Descriptor) {
    if pinned::is_pinned(descriptor) {
        return;
    }
    assert!(
        root_relation_irreducible(descriptor),
        "extension relation is reducible"
    );
}

/// Whether the degree-two relation of the node at `index` is irreducible
/// over its base.
pub const fn quadratic_relation_holds(descriptor: &Descriptor, index: u8) -> bool {
    let (base, characteristic, coefficients) = match descriptor.node(index) {
        Node::Extension {
            base,
            characteristic,
            coefficients,
            ..
        } => (*base, *characteristic, coefficients),
        _ => panic!("not a relation node"),
    };
    if characteristic == 2 {
        // Nonzero linear coefficient and absolute trace of c0 / c1^2 equal
        // to one; the trace compares against the base's identity word.
        let (c0, c1) = (coefficients[0], coefficients[1]);
        if c1 == 0 {
            return false;
        }
        let c1_inverse = word_inv(descriptor, base, c1);
        let mut value = word_mul(
            descriptor,
            base,
            c0,
            word_mul(descriptor, base, c1_inverse, c1_inverse),
        );
        let base_degree = super::arithmetic::degree_of(descriptor, base);
        let mut trace = 0;
        let mut k = 0;
        while k < base_degree {
            trace ^= value;
            value = word_mul(descriptor, base, value, value);
            k += 1;
        }
        trace == word_one(descriptor, base)
    } else {
        // Nonzero discriminant that is a quadratic nonresidue.
        let c0 = relation_coefficient(descriptor, index, 0);
        let c1 = relation_coefficient(descriptor, index, 1);
        let base_order = order_of(descriptor, base);
        let mut four = super::arithmetic::zero_element();
        four[0] = 4 % characteristic;
        let square = element_mul(descriptor, base, &c1, &c1);
        let scaled = element_mul(descriptor, base, &four, &c0);
        let discriminant = element_sub(descriptor, base, &square, &scaled);
        if element_is_zero(descriptor, base, &discriminant) {
            return false;
        }
        let power = element_pow(descriptor, base, &discriminant, (base_order - 1) / 2);
        !element_eq(descriptor, base, &power, &element_one(descriptor, base))
    }
}

/// The generalized Ben-Or test for a relation node of relative degree at
/// least three: `gcd(f, t^(q^i) - t)` must be a unit for every
/// `1 <= i <= floor(D/2)`, `q` the base order.
// No production caller: reached through the irreducibility entry points.
#[allow(dead_code)]
const fn ben_or_irreducible(descriptor: &Descriptor, index: u8) -> bool {
    let (base, relative_degree, base_words) = match descriptor.node(index) {
        Node::Extension {
            base,
            relative_degree,
            base_words,
            ..
        } => (*base, *relative_degree, *base_words),
        _ => panic!("not a relation node"),
    };
    let base_order = order_of(descriptor, base);
    let base_one = element_one(descriptor, base);
    let mut relation: Wide = [0; WIDE];
    let mut i = 0;
    while i < relative_degree {
        wide_put(
            &mut relation,
            i,
            base_words,
            &relation_coefficient(descriptor, index, i),
        );
        i += 1;
    }
    wide_put(&mut relation, relative_degree, base_words, &base_one);
    let mut power = super::arithmetic::zero_element();
    super::arithmetic::set_component(descriptor, index, &mut power, 1, &base_one);
    let mut round = 1;
    while round <= relative_degree / 2 {
        power = element_pow(descriptor, index, &power, base_order);
        let mut difference: Wide = [0; WIDE];
        i = 0;
        while i < relative_degree {
            wide_put(
                &mut difference,
                i,
                base_words,
                &super::arithmetic::component(descriptor, index, &power, i),
            );
            i += 1;
        }
        let shifted = element_sub(
            descriptor,
            base,
            &wide_get(&difference, 1, base_words),
            &base_one,
        );
        wide_put(&mut difference, 1, base_words, &shifted);
        let mut a = relation;
        let mut b = difference;
        while super::arithmetic::poly_degree(descriptor, base, &b, base_words, relative_degree) >= 0
        {
            let (_, remainder) = super::arithmetic::poly_divmod(
                descriptor,
                base,
                &a,
                &b,
                base_words,
                relative_degree,
            );
            a = b;
            b = remainder;
        }
        if super::arithmetic::poly_degree(descriptor, base, &a, base_words, relative_degree) > 0 {
            return false;
        }
        round += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::extension::descriptor::{
        COORDINATES, NODES, Node, PrimePresentation, Storage,
    };

    const M31: u64 = 0x7FFF_FFFF;
    const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
    const BABY_BEAR: u64 = 2_013_265_921;
    const KOALA_BEAR: u64 = 2_130_706_433;

    /// `modulus - value % modulus` for a relation coefficient.
    const fn neg_mod(value: u64, modulus: u64) -> u64 {
        (modulus - value % modulus) % modulus
    }

    // ---- Independent trial-division reference over GF(p), written here.

    /// Dense polynomial remainder of the `left_len` low coefficients of
    /// `left` modulo monic `right` of degree `right_len - 1`, shrinking
    /// `left_len` in place.
    fn poly_remainder(
        left: &mut [u64; 8],
        left_len: &mut usize,
        right: &[u64; 8],
        right_len: usize,
        p: u64,
    ) {
        while *left_len >= right_len {
            let lead = left[*left_len - 1];
            if lead != 0 {
                let shift = *left_len - right_len;
                let mut k = 0;
                while k < right_len {
                    left[shift + k] = (left[shift + k] + p - lead * right[k] % p) % p;
                    k += 1;
                }
            }
            *left_len -= 1;
        }
    }

    /// Whether the monic relation `coefficients[..degree]` (constant term
    /// first, so the polynomial is `t^D + c_{D-1} t^{D-1} + ... + c_0`) has
    /// a monic factor of degree `1..=D/2` over GF(p).
    fn brute_irreducible(coefficients: &[u64; 4], degree: usize, p: u64) -> bool {
        let mut full = [0u64; 8];
        full[..degree].copy_from_slice(&coefficients[..degree]);
        full[degree] = 1;
        let mut factor_degree = 1;
        while factor_degree <= degree / 2 {
            // Every monic candidate of this degree, by odometer.
            let mut candidate = [0u64; 8];
            candidate[factor_degree] = 1;
            let mut digits = [0u64; 4];
            loop {
                let mut remainder = full;
                let mut remainder_len = degree + 1;
                poly_remainder(
                    &mut remainder,
                    &mut remainder_len,
                    &candidate,
                    factor_degree + 1,
                    p,
                );
                if remainder[..remainder_len].iter().all(|&c| c == 0) {
                    return false;
                }
                let mut index = 0;
                while index < factor_degree {
                    digits[index] += 1;
                    candidate[index] = digits[index];
                    if digits[index] < p {
                        break;
                    }
                    digits[index] = 0;
                    candidate[index] = 0;
                    index += 1;
                }
                if index == factor_degree {
                    break;
                }
            }
            factor_degree += 1;
        }
        true
    }

    /// Every monic relation of relative degree two through four over every
    /// prime up to seven, on the residue leaf and, in characteristic two,
    /// on the degree-one binary leaf too: `is_irreducible` equals the
    /// trial-division search on every one of them.
    #[test]
    // The relative degrees stay at most four.
    #[allow(clippy::cast_possible_truncation)]
    fn brute_force_verdicts_agree_with_trial_division() {
        let mut relations = 0;
        let mut mismatches = 0;
        for p in [2u64, 3, 5, 7] {
            for degree in [2usize, 3, 4] {
                let mut relation = [0u64; 4];
                'relations: loop {
                    let bases = [
                        Descriptor::prime(p, PrimePresentation::Residue),
                        Descriptor::polynomial(1, 0b11),
                    ];
                    for base in &bases {
                        // The binary leaf only covers characteristic two.
                        if p != 2 && !matches!(base.node(0), Node::Prime { .. }) {
                            continue;
                        }
                        let descriptor = Descriptor::extension(
                            base,
                            p,
                            degree as u32,
                            degree,
                            &relation[..degree],
                            Storage::Components,
                        );
                        if descriptor.is_irreducible() != brute_irreducible(&relation, degree, p) {
                            mismatches += 1;
                        }
                    }
                    relations += 1;
                    let mut index = 0;
                    while index < degree {
                        relation[index] += 1;
                        if relation[index] < p {
                            continue 'relations;
                        }
                        relation[index] = 0;
                        index += 1;
                    }
                    break;
                }
            }
        }
        assert_eq!(mismatches, 0);
        assert_eq!(relations, 3713);
    }

    // ---- Published relations.

    /// A relation node appended past the checked constructor, for published
    /// relations whose cardinality exceeds the descriptor's `u128` domain:
    /// `extension` rejects those rows by contract, while the interpreter
    /// still decides their relations.
    #[allow(clippy::large_types_passed_by_value)]
    fn unchecked_extension(
        base: &Descriptor,
        characteristic: u64,
        relative_degree: usize,
        relation: &[u64],
    ) -> Descriptor {
        let base_degree = super::super::arithmetic::degree_of(base, base.root_index());
        let base_words = super::super::arithmetic::element_words(base, base.root_index());
        let mut coefficients = [0u64; COORDINATES];
        if characteristic == 2 {
            let mut j = 0;
            while j < relative_degree {
                let mut k = 0;
                while k < base_degree as usize {
                    coefficients[j] |= (relation[j * base_degree as usize + k] & 1) << k;
                    k += 1;
                }
                j += 1;
            }
        } else {
            let mut i = 0;
            while i < relation.len() {
                coefficients[i] = relation[i] % characteristic;
                i += 1;
            }
        }
        let mut nodes = [Node::Unused; NODES];
        let mut i = 0;
        while i < base.node_count() {
            nodes[i as usize] = *base.node(i);
            i += 1;
        }
        nodes[base.node_count() as usize] = Node::Extension {
            base: base.root_index(),
            relative_degree,
            characteristic,
            base_degree,
            base_words,
            coefficients,
            storage: Storage::Components,
        };
        Descriptor::from_parts(nodes, base.node_count() + 1, base.node_count())
    }

    #[test]
    fn published_relations_match_published_verdicts() {
        let residue = |p: u64| Descriptor::prime(p, PrimePresentation::Residue);
        let fold31 = || Descriptor::prime(M31, PrimePresentation::MersenneFold);
        let goldilocks = || Descriptor::prime(GOLDILOCKS, PrimePresentation::GoldilocksFold);
        let binary_leaf = || Descriptor::polynomial(1, 0b11);

        // A prime-base relation at a degree the cardinality bound admits.
        let quad = |base: &Descriptor, p: u64, d: u32, rel: &[u64]| {
            Descriptor::extension(base, p, d, rel.len(), rel, Storage::Components)
        };
        // CM31 relations over the pinned QuadMersenne31 descriptor: the
        // cubic exceeds the cardinality bound and runs past the checks.
        let qm31 = super::super::pinned::QUAD_MERSENNE31;
        assert!(
            Descriptor::extension(
                &qm31,
                M31,
                4,
                2,
                &[M31 - 2, M31 - 1, 0, 0],
                Storage::Components
            )
            .is_irreducible()
        );
        assert!(unchecked_extension(&qm31, M31, 3, &[0, M31 - 5, 0, 0, 0, 0]).is_irreducible());
        // BabyBear and KoalaBear relations; the quintics and octics exceed
        // the cardinality bound and run past the checks.
        assert!(
            quad(
                &residue(BABY_BEAR),
                BABY_BEAR,
                4,
                &[neg_mod(11, BABY_BEAR), 0, 0, 0]
            )
            .is_irreducible()
        );
        assert!(
            unchecked_extension(
                &residue(BABY_BEAR),
                BABY_BEAR,
                5,
                &[neg_mod(2, BABY_BEAR), 0, 0, 0, 0]
            )
            .is_irreducible()
        );
        assert!(
            unchecked_extension(
                &residue(BABY_BEAR),
                BABY_BEAR,
                8,
                &[neg_mod(11, BABY_BEAR), 0, 0, 0, 0, 0, 0, 0]
            )
            .is_irreducible()
        );
        assert!(
            quad(
                &residue(KOALA_BEAR),
                KOALA_BEAR,
                4,
                &[neg_mod(3, KOALA_BEAR), 0, 0, 0]
            )
            .is_irreducible()
        );
        assert!(
            unchecked_extension(
                &residue(KOALA_BEAR),
                KOALA_BEAR,
                8,
                &[neg_mod(3, KOALA_BEAR), 0, 0, 0, 0, 0, 0, 0]
            )
            .is_irreducible()
        );
        // Goldilocks: the quadratic fits, the cubic and quintic do not.
        assert!(quad(&goldilocks(), GOLDILOCKS, 2, &[neg_mod(7, GOLDILOCKS), 0]).is_irreducible());
        assert!(
            unchecked_extension(
                &goldilocks(),
                GOLDILOCKS,
                3,
                &[neg_mod(1, GOLDILOCKS), neg_mod(1, GOLDILOCKS), 0]
            )
            .is_irreducible()
        );
        assert!(
            unchecked_extension(
                &goldilocks(),
                GOLDILOCKS,
                5,
                &[neg_mod(3, GOLDILOCKS), 0, 0, 0, 0]
            )
            .is_irreducible()
        );
        // Mersenne31 quadratic and cubic, GF(7) reducible, GF(9), GF(25),
        // and the GF(2) quintic and degree-13 relations.
        assert!(quad(&fold31(), M31, 2, &[1, 0]).is_irreducible());
        assert!(quad(&fold31(), M31, 3, &[neg_mod(5, M31), 0, 0]).is_irreducible());
        assert!(!quad(&residue(7), 7, 2, &[5, 0]).is_irreducible());
        assert!(quad(&residue(3), 3, 2, &[1, 0]).is_irreducible());
        assert!(quad(&residue(5), 5, 2, &[3, 0]).is_irreducible());
        assert!(quad(&binary_leaf(), 2, 5, &[1, 0, 1, 0, 0]).is_irreducible());
        assert!(
            quad(
                &binary_leaf(),
                2,
                13,
                &[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            .is_irreducible()
        );
    }

    /// `t^2 + linear*t + constant` over a GF(2^8) leaf, as sixteen
    /// flattened coordinates.
    fn rs_words(linear: u64, constant: u64) -> [u64; 16] {
        let mut relation = [0u64; 16];
        #[allow(clippy::needless_range_loop)]
        for k in 0..8 {
            relation[k] = (constant >> k) & 1;
            relation[8 + k] = (linear >> k) & 1;
        }
        relation
    }

    /// The Rijndael16 descriptor `t^2 + t + 0x20` over the AES leaf.
    fn rijndael16(aes: &Descriptor) -> Descriptor {
        Descriptor::extension(
            aes,
            2,
            16,
            2,
            &rs_words(1, 0x20),
            Storage::Packed { width: 16 },
        )
    }

    /// The Rijndael32 descriptor `t^2 + t + 0x2000` over Rijndael16: the
    /// constant word carries bit 13 and the linear word is one.
    fn rijndael32(aes: &Descriptor) -> Descriptor {
        let mut relation = [0u64; 32];
        relation[13] = 1;
        relation[16] = 1;
        let base = rijndael16(aes);
        Descriptor::extension(&base, 2, 32, 2, &relation, Storage::Packed { width: 32 })
    }

    /// The Rijndael64 descriptor `t^2 + t + 0x2000_0000` over Rijndael32.
    fn rijndael64(aes: &Descriptor) -> Descriptor {
        let mut relation = [0u64; 64];
        relation[29] = 1;
        relation[32] = 1;
        let base = rijndael32(aes);
        Descriptor::extension(&base, 2, 64, 2, &relation, Storage::Packed { width: 64 })
    }

    /// The Fan-Paar chain from GF(2): one descriptor per level, by the
    /// Wiedemann relation `t^2 + alpha*t + 1` with `alpha` the previous
    /// level's indeterminate.
    // Each chain slot holds a whole descriptor and the level widths stay
    // far below `u8::MAX`.
    #[allow(clippy::large_stack_arrays, clippy::cast_possible_truncation)]
    fn fan_paar_chain() -> [Descriptor; 6] {
        let mut chain = [Descriptor::polynomial(1, 0b11); 6];
        let mut base = Descriptor::polynomial(1, 0b11);
        for (count, level) in [2u32, 4, 8, 16, 32, 64].into_iter().enumerate() {
            let mut relation = [0u64; 64];
            let used = match level {
                2 => {
                    relation[1] = 1;
                    2
                }
                4 => {
                    relation[3] = 1;
                    4
                }
                8 => {
                    relation[6] = 1;
                    8
                }
                wide => {
                    relation[wide as usize / 2 + wide as usize / 4] = 1;
                    wide as usize
                }
            };
            relation[0] = 1;
            let storage = if level < 8 {
                Storage::PackedPadded { width: 8 }
            } else {
                let width = level as u8;
                Storage::Packed { width }
            };
            base = Descriptor::extension(&base, 2, level, 2, &relation[..used], storage);
            chain[count] = base;
        }
        chain
    }

    /// The published tower verdicts: Rijndael levels, the two reducible
    /// quadratics, and every Fan-Paar level.
    #[test]
    // The relative degrees and level widths stay far below their types' max.
    #[allow(clippy::cast_possible_truncation)]
    fn published_towers_match_published_verdicts() {
        let aes = || Descriptor::polynomial(8, 0x11B);
        let rs = || Descriptor::polynomial(8, 0x11D);
        assert!(rijndael16(&aes()).is_irreducible());
        assert!(rijndael32(&aes()).is_irreducible());
        assert!(rijndael64(&aes()).is_irreducible());
        // t^2 + t + 1 over 0x11D is reducible: GF(4) is a subfield.
        assert!(
            !Descriptor::extension(
                &rs(),
                2,
                16,
                2,
                &rs_words(1, 1),
                Storage::Packed { width: 16 }
            )
            .is_irreducible()
        );
        // t^2 + 0x20 over AES is reducible: the linear coefficient is zero.
        assert!(
            !Descriptor::extension(
                &aes(),
                2,
                16,
                2,
                &rs_words(0, 0x20),
                Storage::Packed { width: 16 }
            )
            .is_irreducible()
        );
        for level in fan_paar_chain() {
            assert!(level.is_irreducible());
        }
    }

    // ---- Runtime proofs that the const relations below are irreducible.

    /// Independent GF(256) product under `0x11B`.
    // The accumulator stays one byte under the reduction.
    #[allow(clippy::cast_possible_truncation)]
    fn aes_mul(x: u8, y: u8) -> u8 {
        let (mut a, mut b, mut acc) = (u16::from(x), u16::from(y), 0u8);
        while b != 0 {
            if b & 1 == 1 {
                acc ^= a as u8;
            }
            b >>= 1;
            a <<= 1;
            if a & 0x100 != 0 {
                a ^= 0x11B;
            }
        }
        acc
    }

    /// `t^3 + t + 1` has no root in GF(256): a cubic over a field is
    /// irreducible exactly when it has no linear factor, checked here with
    /// the independent multiplication.
    #[test]
    fn aes_cubic_has_no_root_independent_of_the_interpreter() {
        for x in 0..=255u8 {
            let value = aes_mul(x, aes_mul(x, x)) ^ x ^ 1;
            assert_ne!(value, 0, "root {x:#x}");
        }
    }

    /// Independent GF(16) product under `0x13` for the basis-row checks.
    fn gf16_mul(x: u64, y: u64) -> u64 {
        let mut acc = 0u64;
        #[allow(clippy::needless_range_loop)]
        for k in 0..4 {
            if (y >> k) & 1 == 1 {
                acc ^= x << k;
            }
        }
        for k in (4..7).rev() {
            if (acc >> k) & 1 == 1 {
                acc ^= 1 << k;
                acc ^= 1 << (k - 3);
                acc ^= 1 << (k - 4);
            }
        }
        acc & 0xF
    }

    /// The relations pinned into the const proofs over the shipped basis
    /// rows have no root, so the quadratics are irreducible; the basis maps
    /// come from the descriptor's stored node data, the multiplication from
    /// this test.
    #[test]
    fn basis_row_relations_have_no_root_independent_of_the_interpreter() {
        for row in [
            <crate::field::binary::Normal<0x13, 8> as crate::field::binary::BinaryRepr<4>>::DESCRIPTION,
            <crate::field::binary::Cantor<0x13, 8> as crate::field::binary::BinaryRepr<4>>::DESCRIPTION,
        ] {
            let descriptor = row.as_descriptor();
            let to_base = match descriptor.node(descriptor.root_index()) {
                Node::Basis { to_base, .. } => *to_base,
                _ => panic!("expected a basis root"),
            };
            let transport = |word: u64| crate::field::binary::description::apply(to_base, word);
            // Linear word 0x2, constant word 0x5, in the row's coordinates.
            let c0 = transport(5);
            let c1 = transport(2);
            for word in 0..16u64 {
                let x = transport(word);
                let value = gf16_mul(c1, x) ^ gf16_mul(x, x) ^ c0;
                assert_ne!(value, 0, "root word {word:#x}");
            }
        }
    }

    // ---- Const proofs on the crate MSRV; every one runs the full test.

    const _: () = assert!(
        crate::field::extension::pinned::PINNED[8].is_irreducible(),
        "FanPaar64"
    );
    const _: () = assert!(
        relation_over_prime(PrimePresentation::Residue, 7, 2, 2, &[4, 0]),
        "GF(49)"
    );
    const _: () = assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            GOLDILOCKS,
            2,
            2,
            &[GOLDILOCKS - 7, 0]
        ),
        "goldilocks residue t^2 - 7"
    );
    const _: () = assert!(
        relation_over_prime(
            PrimePresentation::GoldilocksFold,
            GOLDILOCKS,
            2,
            2,
            &[GOLDILOCKS - 7, 0]
        ),
        "goldilocks fold t^2 - 7"
    );
    const _: () = assert!(
        relation_over_prime(PrimePresentation::Residue, M31, 3, 3, &[M31 - 5, 0, 0]),
        "mersenne31 residue t^3 - 5"
    );
    const _: () = assert!(
        relation_over_prime(PrimePresentation::MersenneFold, M31, 3, 3, &[M31 - 5, 0, 0]),
        "mersenne31 fold t^3 - 5"
    );
    const _: () = assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            BABY_BEAR,
            4,
            4,
            &[BABY_BEAR - 11, 0, 0, 0]
        ),
        "babybear"
    );
    const _: () = assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            KOALA_BEAR,
            4,
            4,
            &[KOALA_BEAR - 3, 0, 0, 0]
        ),
        "koalabear"
    );
    const _: () = assert!(
        quad_over_quad_mersenne31(),
        "t^2 - (2 + i) over QuadMersenne31"
    );
    const _: () = assert!(aes_cubic(), "cubic over AES");
    const _: () = assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            2,
            13,
            13,
            &[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ),
        "degree 13 over GF(2)"
    );
    const _: () = assert!(cubic_over_gf49(), "cubic over GF(49)");
    const _: () = assert!(gf64_over_gf8(), "GF(64)");
    const _: () = assert!(normal_row_extension(), "normal row");
    const _: () = assert!(cantor_row_extension(), "cantor row");
    const _: () = assert!(
        relation_over_prime(PrimePresentation::Residue, 2, 3, 3, &[1, 1, 0]),
        "cubic over GF(2)"
    );

    /// A prime-base relation built in const over the named presentation.
    const fn relation_over_prime(
        presentation: PrimePresentation,
        p: u64,
        degree: u32,
        relative_degree: usize,
        relation: &[u64],
    ) -> bool {
        Descriptor::extension(
            &Descriptor::prime(p, presentation),
            p,
            degree,
            relative_degree,
            relation,
            Storage::Components,
        )
        .is_irreducible()
    }

    /// `t^2 - (2 + i)` over the pinned `QuadMersenne31` descriptor.
    const fn quad_over_quad_mersenne31() -> bool {
        let base = crate::field::extension::pinned::QUAD_MERSENNE31;
        Descriptor::extension(
            &base,
            0x7FFF_FFFF,
            4,
            2,
            &[0x7FFF_FFFD, 0x7FFF_FFFE, 0, 0],
            Storage::Components,
        )
        .is_irreducible()
    }

    /// `t^3 + t + 1` over the AES GF(2^8) leaf.
    const fn aes_cubic() -> bool {
        let mut relation = [0u64; 24];
        relation[0] = 1;
        relation[8] = 1;
        let aes = Descriptor::polynomial(8, 0x11B);
        Descriptor::extension(&aes, 2, 24, 3, &relation, Storage::Components).is_irreducible()
    }

    /// A cubic over GF(49), itself `t^2 - 3` over the residue leaf at 7.
    const fn cubic_over_gf49() -> bool {
        let gf49 = Descriptor::extension(
            &Descriptor::prime(7, PrimePresentation::Residue),
            7,
            2,
            2,
            &[4, 0],
            Storage::Components,
        );
        Descriptor::extension(&gf49, 7, 6, 3, &[5, 0, 0, 0, 0, 0], Storage::Components)
            .is_irreducible()
    }

    /// GF(64) as `t^2 + alpha*t + 1` over GF(8) = `x^3 + x + 1` over GF(2).
    const fn gf64_over_gf8() -> bool {
        let gf8 = Descriptor::extension(
            &Descriptor::prime(2, PrimePresentation::Residue),
            2,
            3,
            3,
            &[1, 1, 0],
            Storage::Components,
        );
        let gf8_holds = gf8.is_irreducible();
        let gf64 = Descriptor::extension(&gf8, 2, 6, 2, &[1, 0, 0, 0, 1, 0], Storage::Components);
        gf8_holds & gf64.is_irreducible()
    }

    /// `t^2 + 0x2*t + 0x5` over the shipped normal GF(2^4) row.
    const fn normal_row_extension() -> bool {
        let row = <crate::field::binary::Normal<0x13, 8> as crate::field::binary::BinaryRepr<4>>::DESCRIPTION;
        let base = row.as_descriptor();
        let mut relation = [0u64; 8];
        // Constant word 0x5 (bits 0 and 2), linear word 0x2 (bit 1).
        relation[0] = 1;
        relation[2] = 1;
        relation[4 + 1] = 1;
        Descriptor::extension(base, 2, 8, 2, &relation, Storage::Packed { width: 8 })
            .is_irreducible()
    }

    /// `t^2 + 0x2*t + 0x5` over the shipped Cantor GF(2^4) row.
    const fn cantor_row_extension() -> bool {
        let row = <crate::field::binary::Cantor<0x13, 8> as crate::field::binary::BinaryRepr<4>>::DESCRIPTION;
        let base = row.as_descriptor();
        let mut relation = [0u64; 8];
        relation[0] = 1;
        relation[2] = 1;
        relation[4 + 1] = 1;
        Descriptor::extension(base, 2, 8, 2, &relation, Storage::Packed { width: 8 })
            .is_irreducible()
    }
}
