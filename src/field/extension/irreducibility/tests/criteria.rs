//! Quadratic, packed binary and published relation criteria.

use super::super::*;
use super::ben_or::ben_or_irreducible;
use crate::field::extension::descriptor::{COORDINATES, NODES, Node, PrimePresentation, Storage};

const M31: u64 = 0x7FFF_FFFF;
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
const BABY_BEAR: u64 = 2_013_265_921;
const KOALA_BEAR: u64 = 2_130_706_433;

#[test]
fn binary_rabin_matches_exhaustive_trial_division() {
    for degree in 3..=10 {
        for low in 0..1u128 << degree {
            let coefficients: [u64; 64] =
                core::array::from_fn(|i| u64::from(low & (1u128 << i) != 0));
            assert_eq!(
                binary_rabin_irreducible(&coefficients, degree),
                crate::field::binary::description::poly_irreducible(
                    (1u128 << degree) | low,
                    u32::try_from(degree).unwrap(),
                ),
                "degree={degree} low={low:#x}",
            );
        }
    }
}

#[test]
fn binary_rabin_boundary_relations_match_generic_ben_or() {
    for (degree, low) in [
        (64, 0x1bu64),
        (64, 0xa1c6_7ae0_5c18_5c83),
        (64, (1 << 32) | 1),
        (64, 2),
        (63, 3),
        (61, 0x5b),
    ] {
        let relation: [u64; 64] = core::array::from_fn(|i| (low >> i) & 1);
        for base in [
            Descriptor::prime(2, PrimePresentation::Residue),
            Descriptor::polynomial(1, 3),
        ] {
            let descriptor = Descriptor::extension(
                &base,
                2,
                u32::try_from(degree).unwrap(),
                degree,
                &relation[..degree],
                Storage::Components,
            );
            assert_eq!(
                descriptor.is_irreducible(),
                ben_or_irreducible(&descriptor, descriptor.root_index()),
                "degree={degree} low={low:#x}",
            );
        }
    }
}

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
pub(super) fn unchecked_extension(
    base: &Descriptor,
    characteristic: u64,
    relative_degree: usize,
    relation: &[u64],
) -> Descriptor {
    let base_degree = crate::field::extension::arithmetic::degree_of(base, base.root_index());
    let base_words = crate::field::extension::arithmetic::element_words(base, base.root_index());
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
    let qm31 = *<crate::field::quad_mersenne31::QuadMersenne31 as crate::field::private::RowAccess>::DESCRIPTOR;
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
pub(super) fn fan_paar_chain() -> [Descriptor; 6] {
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
            <crate::field::binary::Normal<0x13, 8> as crate::field::binary::binary_private::BinaryReprData<4>>::DESCRIPTION,
            <crate::field::binary::Cantor<0x13, 8> as crate::field::binary::binary_private::BinaryReprData<4>>::DESCRIPTION,
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
    let base = *<crate::field::quad_mersenne31::QuadMersenne31 as crate::field::private::RowAccess>::DESCRIPTOR;
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
    Descriptor::extension(&gf49, 7, 6, 3, &[5, 0, 0, 0, 0, 0], Storage::Components).is_irreducible()
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
    let row = <crate::field::binary::Normal<0x13, 8> as crate::field::binary::binary_private::BinaryReprData<4>>::DESCRIPTION;
    let base = row.as_descriptor();
    let mut relation = [0u64; 8];
    // Constant word 0x5 (bits 0 and 2), linear word 0x2 (bit 1).
    relation[0] = 1;
    relation[2] = 1;
    relation[4 + 1] = 1;
    Descriptor::extension(base, 2, 8, 2, &relation, Storage::Packed { width: 8 }).is_irreducible()
}

/// `t^2 + 0x2*t + 0x5` over the shipped Cantor GF(2^4) row.
const fn cantor_row_extension() -> bool {
    let row = <crate::field::binary::Cantor<0x13, 8> as crate::field::binary::binary_private::BinaryReprData<4>>::DESCRIPTION;
    let base = row.as_descriptor();
    let mut relation = [0u64; 8];
    relation[0] = 1;
    relation[2] = 1;
    relation[4 + 1] = 1;
    Descriptor::extension(base, 2, 8, 2, &relation, Storage::Packed { width: 8 }).is_irreducible()
}

/// The const proofs above evaluate at compile time; the same fixtures
/// re-run here exercise the identical checks at runtime.
#[test]
fn const_relation_fixtures_hold_at_runtime_too() {
    assert!(
        relation_over_prime(PrimePresentation::Residue, 7, 2, 2, &[4, 0]),
        "GF(49)"
    );
    assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            GOLDILOCKS,
            2,
            2,
            &[GOLDILOCKS - 7, 0]
        ),
        "goldilocks residue t^2 - 7"
    );
    assert!(
        relation_over_prime(
            PrimePresentation::GoldilocksFold,
            GOLDILOCKS,
            2,
            2,
            &[GOLDILOCKS - 7, 0]
        ),
        "goldilocks fold t^2 - 7"
    );
    assert!(
        relation_over_prime(PrimePresentation::Residue, M31, 3, 3, &[M31 - 5, 0, 0]),
        "mersenne31 residue t^3 - 5"
    );
    assert!(
        relation_over_prime(PrimePresentation::MersenneFold, M31, 3, 3, &[M31 - 5, 0, 0]),
        "mersenne31 fold t^3 - 5"
    );
    assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            BABY_BEAR,
            4,
            4,
            &[BABY_BEAR - 11, 0, 0, 0]
        ),
        "babybear"
    );
    assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            KOALA_BEAR,
            4,
            4,
            &[KOALA_BEAR - 3, 0, 0, 0]
        ),
        "koalabear"
    );
    assert!(
        quad_over_quad_mersenne31(),
        "t^2 - (2 + i) over QuadMersenne31"
    );
    assert!(aes_cubic(), "cubic over AES");
    assert!(
        relation_over_prime(
            PrimePresentation::Residue,
            2,
            13,
            13,
            &[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ),
        "degree 13 over GF(2)"
    );
    assert!(cubic_over_gf49(), "cubic over GF(49)");
    assert!(gf64_over_gf8(), "GF(64)");
    assert!(normal_row_extension(), "normal row");
    assert!(cantor_row_extension(), "cantor row");
    assert!(
        relation_over_prime(PrimePresentation::Residue, 2, 3, 3, &[1, 1, 0]),
        "cubic over GF(2)"
    );
}

#[test]
#[should_panic(expected = "not a relation node")]
fn node_relation_irreducible_rejects_a_prime_root() {
    let _ = node_relation_irreducible(&Descriptor::prime(2, PrimePresentation::Residue), 0);
}

#[test]
fn validate_relation_accepts_an_unpinned_irreducible_quadratic() {
    // GF(25): the discriminant of `t^2 + 2` is `-8 = 2`, a quadratic
    // nonresidue over GF(5), and the descriptor matches no pinned row.
    let descriptor = Descriptor::extension(
        &Descriptor::prime(5, PrimePresentation::Residue),
        5,
        2,
        2,
        &[2, 0],
        Storage::Components,
    );
    assert!(brute_irreducible(&[2, 0, 0, 0], 2, 5));
    assert!(quadratic_relation_holds(
        &descriptor,
        descriptor.root_index()
    ));
    validate_relation(&descriptor);
}

#[test]
#[should_panic(expected = "extension relation is reducible")]
fn validate_relation_rejects_a_reducible_quadratic() {
    // `t^2 + 1` splits over GF(5): its discriminant `-4 = 1` is a square.
    validate_relation(&Descriptor::extension(
        &Descriptor::prime(5, PrimePresentation::Residue),
        5,
        2,
        2,
        &[1, 0],
        Storage::Components,
    ));
}

#[test]
fn unchecked_binary_relations_decide_with_the_checked_constructor() {
    let leaf = Descriptor::polynomial(1, 0b11);
    // `t^3 + t + 1` is irreducible and `t^3 + t^2 + t + 1` factors as
    // `(t + 1)(t^2 + t + 1)` over GF(2).
    let irreducible = unchecked_extension(&leaf, 2, 3, &[1, 1, 0]);
    let reducible = unchecked_extension(&leaf, 2, 3, &[1, 1, 1]);
    let checked = Descriptor::extension(&leaf, 2, 3, 3, &[1, 1, 0], Storage::Components);
    assert!(irreducible.is_irreducible());
    assert!(!reducible.is_irreducible());
    assert_eq!(irreducible.is_irreducible(), checked.is_irreducible());
}
