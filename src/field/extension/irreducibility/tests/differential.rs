//! Runtime regressions: differential verdicts, factor products, basis
//! identity, independent resultants, and every policy branch.

use super::super::prime as prime_policy;
use super::super::*;
use super::ben_or::ben_or_irreducible;
use super::criteria::{fan_paar_chain, unchecked_extension};
use crate::field::extension::arithmetic::{
    Element, degree_of, element_add, element_mul, word_one, zero_element,
};
use crate::field::extension::descriptor::{
    COORDINATES, Descriptor, Node, PrimePresentation, Storage,
};

/// Compare the production predicate with the Ben-Or oracle.
fn assert_agrees(descriptor: &Descriptor) {
    let index = descriptor.root_index();
    assert_agrees_at(descriptor, index);
}

/// Compare one node index, root or not.
fn assert_agrees_at(descriptor: &Descriptor, index: u8) {
    assert_eq!(
        node_relation_irreducible(descriptor, index),
        ben_or_irreducible(descriptor, index),
        "descriptor node {index}",
    );
}

fn prime_base(modulus: u64) -> Descriptor {
    Descriptor::prime(modulus, PrimePresentation::Residue)
}

fn extension(
    base: &Descriptor,
    characteristic: u64,
    degree: u32,
    relative_degree: usize,
    relation: &[u64],
) -> Descriptor {
    Descriptor::extension(
        base,
        characteristic,
        degree,
        relative_degree,
        relation,
        Storage::Components,
    )
}

fn next_relation(relation: &mut [u64], radix: u64) -> bool {
    let mut index = 0;
    while index < relation.len() {
        relation[index] += 1;
        if relation[index] < radix {
            return true;
        }
        relation[index] = 0;
        index += 1;
    }
    false
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn digit(&mut self, radix: u64) -> u64 {
        self.next() % radix
    }
}

fn fill_relation(relation: &mut [u64], radix: u64, rng: &mut Rng) {
    let mut index = 0;
    while index < relation.len() {
        relation[index] = rng.digit(radix);
        index += 1;
    }
}

// ---- Exhaustive prime fields through the differential degree bounds.

#[test]
fn exhaustive_prime_fields_match_ben_or() {
    for (modulus, maximum) in [(2u64, 9usize), (3, 6), (5, 4), (7, 3)] {
        let base = prime_base(modulus);
        for degree in 2..=maximum {
            let mut relation = [0u64; COORDINATES];
            loop {
                let descriptor = extension(
                    &base,
                    modulus,
                    degree.try_into().unwrap(),
                    degree,
                    &relation[..degree],
                );
                assert_agrees(&descriptor);
                if !next_relation(&mut relation[..degree], modulus) {
                    break;
                }
            }
        }
    }
}

// ---- Exhaustive small towers.

fn exhaustive_tower(base: &Descriptor, characteristic: u64, maximum: usize) {
    let base_degree = degree_of(base, base.root_index()) as usize;
    for degree in 2..=maximum {
        let coordinates = degree * base_degree;
        let mut relation = [0u64; COORDINATES];
        loop {
            let descriptor = extension(
                base,
                characteristic,
                coordinates.try_into().unwrap(),
                degree,
                &relation[..coordinates],
            );
            assert_agrees(&descriptor);
            if !next_relation(&mut relation[..coordinates], characteristic) {
                break;
            }
        }
    }
}

#[test]
fn exhaustive_gf4_towers_match_ben_or() {
    let gf4 = extension(&prime_base(2), 2, 2, 2, &[1, 1]);
    assert!(gf4.is_irreducible());
    exhaustive_tower(&gf4, 2, 4);
}

#[test]
fn exhaustive_gf9_towers_match_ben_or() {
    let gf9 = extension(&prime_base(3), 3, 2, 2, &[1, 0]);
    assert!(gf9.is_irreducible());
    exhaustive_tower(&gf9, 3, 4);
}

// ---- Deterministic samples of larger towers and polynomial leaves.

fn sampled(base: &Descriptor, characteristic: u64, minimum: usize, maximum: usize, count: usize) {
    let base_degree = degree_of(base, base.root_index()) as usize;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for degree in minimum..=maximum {
        let coordinates = degree * base_degree;
        let mut sample = 0;
        while sample < count {
            let mut relation = [0u64; COORDINATES];
            fill_relation(&mut relation[..coordinates], characteristic, &mut rng);
            let descriptor = extension(
                base,
                characteristic,
                coordinates.try_into().unwrap(),
                degree,
                &relation[..coordinates],
            );
            assert_agrees(&descriptor);
            sample += 1;
        }
    }
}

#[test]
fn sampled_larger_towers_match_ben_or() {
    let gf4 = extension(&prime_base(2), 2, 2, 2, &[1, 1]);
    let gf9 = extension(&prime_base(3), 3, 2, 2, &[1, 0]);
    let gf16 = Descriptor::polynomial(4, 0x13);
    let polynomial_six = Descriptor::polynomial(6, 0x43);
    sampled(&gf4, 2, 5, 8, 12);
    sampled(&gf9, 3, 5, 8, 12);
    sampled(&gf16, 2, 3, 6, 8);
    sampled(&polynomial_six, 2, 4, 5, 4);
}

// ---- Two- and three-factor products, repeated factors, and the gate.

fn prime_full_coefficient(coefficients: &[u64], degree: usize, index: usize, modulus: u64) -> u64 {
    if index == degree {
        1
    } else {
        coefficients[index] % modulus
    }
}

fn prime_product(
    out: &mut [u64],
    left: &[u64],
    left_degree: usize,
    right: &[u64],
    right_degree: usize,
    modulus: u64,
) -> usize {
    let degree = left_degree + right_degree;
    let mut i = 0;
    while i <= left_degree {
        let a = prime_full_coefficient(left, left_degree, i, modulus);
        let mut j = 0;
        while j <= right_degree {
            let b = prime_full_coefficient(right, right_degree, j, modulus);
            let index = i + j;
            out[index] = u64::try_from(
                (u128::from(out[index]) + u128::from(a) * u128::from(b)) % u128::from(modulus),
            )
            .unwrap();
            j += 1;
        }
        i += 1;
    }
    degree
}

fn prime_monic_descriptor(modulus: u64, coefficients: &[u64], degree: usize) -> Descriptor {
    extension(
        &prime_base(modulus),
        modulus,
        degree.try_into().unwrap(),
        degree,
        &coefficients[..degree],
    )
}

#[test]
fn repeated_and_multiple_prime_factors_are_rejected() {
    // t^2 + 1 and t^3 + 2 are irreducible over GF(7); t^2 + 1 has a
    // nonsquare discriminant and t^3 + 2 a square discriminant.
    let quadratic = [1u64, 0];
    let cubic = [2u64, 0, 0];
    let mut repeated = [0u64; COORDINATES];
    let repeated_degree = prime_product(&mut repeated, &quadratic, 2, &quadratic, 2, 7);
    assert_eq!(repeated_degree, 4);
    let repeated_descriptor = prime_monic_descriptor(7, &repeated, repeated_degree);
    assert!(!repeated_descriptor.is_irreducible());
    assert_agrees(&repeated_descriptor);

    let mut two = [0u64; COORDINATES];
    let two_degree = prime_product(&mut two, &quadratic, 2, &cubic, 3, 7);
    assert_eq!(two_degree, 5);
    let two_descriptor = prime_monic_descriptor(7, &two, two_degree);
    assert!(!prime_policy::discriminant_gate(
        &two[..=two_degree],
        two_degree,
        7
    ));
    assert!(!two_descriptor.is_irreducible());
    assert_agrees(&two_descriptor);

    // Three nonsquare-discriminant quadratics pass the parity filter and
    // must continue to rank testing.
    let mut found = false;
    for b0 in 0..7u64 {
        for c0 in 0..7u64 {
            let f0 = [c0, b0];
            if prime_monic_descriptor(7, &f0, 2).is_irreducible() {
                for b1 in 0..7u64 {
                    for c1 in 0..7u64 {
                        let f1 = [c1, b1];
                        if prime_monic_descriptor(7, &f1, 2).is_irreducible() {
                            for b2 in 0..7u64 {
                                for c2 in 0..7u64 {
                                    let f2 = [c2, b2];
                                    if !prime_monic_descriptor(7, &f2, 2).is_irreducible() {
                                        continue;
                                    }
                                    let mut pair = [0u64; COORDINATES];
                                    let pair_degree = prime_product(&mut pair, &f0, 2, &f1, 2, 7);
                                    let mut three = [0u64; COORDINATES];
                                    let degree =
                                        prime_product(&mut three, &pair, pair_degree, &f2, 2, 7);
                                    if prime_policy::discriminant_gate(&three[..=degree], degree, 7)
                                    {
                                        let descriptor = prime_monic_descriptor(7, &three, degree);
                                        assert!(!descriptor.is_irreducible());
                                        assert_agrees(&descriptor);
                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(found, "no three-factor product passed the parity filter");
}

// ---- Odd extension products: square-free gate and rank without parity.

fn gf9() -> Descriptor {
    extension(&prime_base(3), 3, 2, 2, &[1, 0])
}

fn gf9_element(first: u64, second: u64) -> Element {
    let mut value = zero_element();
    value[0] = first;
    value[1] = second;
    value
}

fn odd_polynomial_product(
    out: &mut [Element; 8],
    base: &Descriptor,
    left: &[Element],
    right: &[Element],
) -> usize {
    let left_degree = left.len() - 1;
    let right_degree = right.len() - 1;
    let degree = left_degree + right_degree;
    let mut i = 0;
    while i <= left_degree {
        let mut j = 0;
        while j <= right_degree {
            let product = element_mul(base, base.root_index(), &left[i], &right[j]);
            out[i + j] = element_add(base, base.root_index(), &out[i + j], &product);
            j += 1;
        }
        i += 1;
    }
    degree
}

fn flatten_odd(base_degree: usize, coefficients: &[Element], degree: usize) -> [u64; COORDINATES] {
    let mut relation = [0u64; COORDINATES];
    let mut j = 0;
    while j < degree {
        let mut k = 0;
        while k < base_degree {
            relation[j * base_degree + k] = coefficients[j][k];
            k += 1;
        }
        j += 1;
    }
    relation
}

fn first_gf9_quadratics() -> [[Element; 3]; 3] {
    let base = gf9();
    let mut factors = [[zero_element(); 3]; 3];
    let mut count = 0;
    for b in 0..9u64 {
        for c in 0..9u64 {
            let mut coefficients = [zero_element(); 3];
            coefficients[0] = gf9_element(c % 3, c / 3);
            coefficients[1] = gf9_element(b % 3, b / 3);
            let relation = flatten_odd(2, &coefficients, 2);
            let descriptor = extension(&base, 3, 4, 2, &relation[..4]);
            if descriptor.is_irreducible() {
                factors[count] = coefficients;
                count += 1;
                if count == 3 {
                    return factors;
                }
            }
        }
    }
    panic!("fewer than three irreducible quadratics over GF(9)");
}

#[test]
fn repeated_and_multiple_gf9_factors_are_rejected() {
    let base = gf9();
    let factors = first_gf9_quadratics();
    let mut repeated = [zero_element(); 8];
    let repeated_degree =
        odd_polynomial_product(&mut repeated, &base, &factors[0][..3], &factors[0][..3]);
    let relation = flatten_odd(2, &repeated, repeated_degree);
    let descriptor = extension(
        &base,
        3,
        (repeated_degree * 2).try_into().unwrap(),
        repeated_degree,
        &relation[..repeated_degree * 2],
    );
    assert!(!descriptor.is_irreducible());
    assert_agrees(&descriptor);

    let mut two = [zero_element(); 8];
    let two_degree = odd_polynomial_product(&mut two, &base, &factors[0][..3], &factors[1][..3]);
    let relation = flatten_odd(2, &two, two_degree);
    let descriptor = extension(
        &base,
        3,
        (two_degree * 2).try_into().unwrap(),
        two_degree,
        &relation[..two_degree * 2],
    );
    assert!(!descriptor.is_irreducible());
    assert_agrees(&descriptor);

    let mut pair = [zero_element(); 8];
    let pair_degree = odd_polynomial_product(&mut pair, &base, &factors[0][..3], &factors[1][..3]);
    let mut three = [zero_element(); 8];
    let three_degree =
        odd_polynomial_product(&mut three, &base, &pair[..=pair_degree], &factors[2][..3]);
    let relation = flatten_odd(2, &three, three_degree);
    let descriptor = extension(
        &base,
        3,
        (three_degree * 2).try_into().unwrap(),
        three_degree,
        &relation[..three_degree * 2],
    );
    assert!(!descriptor.is_irreducible());
    assert_agrees(&descriptor);
}

// ---- A basis whose identity word is not one.

fn basis_relations(base: &Descriptor) -> (Descriptor, Descriptor, Descriptor) {
    let base_degree = degree_of(base, base.root_index()) as usize;
    let identity = word_one(base, base.root_index());
    let mut bits = [0u64; 24];
    let mut k = 0;
    while k < base_degree {
        bits[k] = (identity >> k) & 1;
        bits[2 * base_degree + k] = (identity >> k) & 1;
        k += 1;
    }
    let cubic = extension(
        base,
        2,
        (3 * base_degree).try_into().unwrap(),
        3,
        &bits[..3 * base_degree],
    );
    let mut quartic_bits = [0u64; 32];
    let mut k = 0;
    while k < base_degree {
        quartic_bits[k] = (identity >> k) & 1;
        quartic_bits[base_degree + k] = (identity >> k) & 1;
        k += 1;
    }
    let quartic = extension(
        base,
        2,
        (4 * base_degree).try_into().unwrap(),
        4,
        &quartic_bits[..4 * base_degree],
    );
    let mut rng = Rng(0xC0DE_BA5E_F00D_1234);
    let mut quintic_bits = [0u64; 32];
    fill_relation(&mut quintic_bits[..5 * base_degree], 2, &mut rng);
    let quintic = extension(
        base,
        2,
        (5 * base_degree).try_into().unwrap(),
        5,
        &quintic_bits[..5 * base_degree],
    );
    (cubic, quartic, quintic)
}

#[test]
fn basis_identity_semantics_hold_through_descent_and_rank() {
    let row = <crate::field::binary::Normal<0x13, 8> as crate::field::binary::binary_private::BinaryReprData<4>>::DESCRIPTION;
    let base = row.as_descriptor();
    let identity = word_one(base, base.root_index());
    assert_ne!(identity, 1, "the chosen basis must hide identity one");
    let (cubic, quartic, quintic) = basis_relations(base);
    assert!(cubic.is_irreducible());
    assert_agrees(&cubic);
    assert!(!quartic.is_irreducible());
    assert_agrees(&quartic);
    assert_agrees(&quintic);
}

// ---- Independent Sylvester determinant.

fn determinant_mod(mut matrix: [[u64; 32]; 32], size: usize, modulus: u64) -> u64 {
    let mut determinant = 1u64;
    let mut column = 0;
    while column < size {
        let mut pivot = column;
        while pivot < size && matrix[pivot][column] == 0 {
            pivot += 1;
        }
        if pivot == size {
            return 0;
        }
        if pivot != column {
            matrix.swap(pivot, column);
            determinant = (modulus - determinant) % modulus;
        }
        let lead = matrix[column][column];
        determinant =
            u64::try_from(u128::from(determinant) * u128::from(lead) % u128::from(modulus))
                .unwrap();
        let inverse = (1..modulus)
            .find(|&value| u128::from(lead) * u128::from(value) % u128::from(modulus) == 1)
            .unwrap();
        let mut row = column + 1;
        while row < size {
            if matrix[row][column] != 0 {
                let factor = u64::try_from(
                    u128::from(matrix[row][column]) * u128::from(inverse) % u128::from(modulus),
                )
                .unwrap();
                let mut k = column;
                while k < size {
                    let product =
                        u128::from(factor) * u128::from(matrix[column][k]) % u128::from(modulus);
                    matrix[row][k] = u64::try_from(
                        (u128::from(matrix[row][k]) + u128::from(modulus) - product)
                            % u128::from(modulus),
                    )
                    .unwrap();
                    k += 1;
                }
            }
            row += 1;
        }
        column += 1;
    }
    determinant
}

#[allow(clippy::cast_possible_truncation)]
fn sylvester_resultant(full: &[u64], degree: usize, modulus: u64) -> u64 {
    let mut derivative = [0u64; 32];
    let mut i = 0;
    while i < degree {
        derivative[i] = (i as u64 + 1) % modulus * full[i + 1] % modulus;
        i += 1;
    }
    let size = 2 * degree - 1;
    let mut matrix = [[0u64; 32]; 32];
    let mut row = 0;
    while row < degree - 1 {
        let mut k = 0;
        while k <= degree {
            matrix[row][row + k] = full[k];
            k += 1;
        }
        row += 1;
    }
    let mut offset = 0;
    while offset < degree {
        let mut k = 0;
        while k < degree {
            matrix[degree - 1 + offset][offset + k] = derivative[k];
            k += 1;
        }
        offset += 1;
    }
    determinant_mod(matrix, size, modulus)
}

#[test]
fn prime_resultants_match_independent_sylvester_values() {
    let cases: [(u64, usize, [u64; 8]); 6] = [
        (7, 2, [1, 0, 1, 0, 0, 0, 0, 0]),
        (7, 3, [2, 0, 0, 1, 0, 0, 0, 0]),
        (5, 4, [1, 1, 0, 0, 1, 0, 0, 0]),
        (11, 4, [3, 1, 4, 0, 1, 0, 0, 0]),
        (7, 5, [1, 2, 3, 4, 5, 1, 0, 0]),
        (13, 6, [2, 3, 5, 7, 11, 6, 1, 0]),
    ];
    for (modulus, degree, full) in cases {
        let expected = sylvester_resultant(&full[..=degree], degree, modulus);
        let actual = prime_policy::resultant(&full[..=degree], degree, modulus);
        assert_eq!(actual, expected, "modulus={modulus} degree={degree}");
    }
    assert_eq!(
        prime_policy::resultant(&[1, 0, 1][..=2], 2, 7),
        sylvester_resultant(&[1, 0, 1][..=2], 2, 7)
    );
}

// ---- Every policy branch.

#[test]
fn specialized_and_prime_descent_relations_match_oracle() {
    // Pinned acceptance.
    let pinned = crate::field::extension::pinned::PINNED;
    let mut i = 0;
    while i < pinned.len() {
        pinned[i].validate_relation();
        i += 1;
    }

    // Shipped quadratic criteria and packed Rabin.
    let gf4 = extension(&prime_base(2), 2, 2, 2, &[1, 1]);
    assert!(gf4.is_irreducible());
    assert_agrees(&gf4);
    let quadratic = extension(&prime_base(5), 5, 2, 2, &[2, 0]);
    assert!(quadratic.is_irreducible());
    assert_agrees(&quadratic);
    let cubic = extension(&prime_base(2), 2, 3, 3, &[1, 1, 0]);
    assert!(cubic.is_irreducible());
    assert_agrees(&cubic);

    // Prime-subfield descent, including a non-coprime immediate rejection.
    let gf49 = extension(&prime_base(7), 7, 2, 2, &[4, 0]);
    let cubic_over_gf49 = extension(&gf49, 7, 6, 3, &[5, 0, 0, 0, 0, 0]);
    assert!(cubic_over_gf49.is_irreducible());
    assert_agrees(&cubic_over_gf49);
    let mut noncoprime = [0u64; COORDINATES];
    noncoprime[0] = 1;
    let noncoprime_descriptor = extension(&gf4, 2, 8, 4, &noncoprime[..8]);
    assert!(!noncoprime_descriptor.is_irreducible());
    assert_agrees(&noncoprime_descriptor);
}

#[test]
fn explicit_ancestor_descent_relations_match_oracle() {
    // Explicit odd ancestor descent, GF(81) over GF(9).
    let gf9 = gf9();
    let factors = first_gf9_quadratics();
    let quadratic_relation = flatten_odd(2, &factors[0], 2);
    let gf81 = extension(&gf9, 3, 4, 2, &quadratic_relation[..4]);
    assert!(gf81.is_irreducible());
    let mut ancestor_relation = [0u64; COORDINATES];
    let mut j = 0;
    while j < 3 {
        let coefficient = if j == 0 {
            gf9_element(0, 1)
        } else {
            factors[j % 3][0]
        };
        ancestor_relation[j * 4] = coefficient[0];
        ancestor_relation[j * 4 + 1] = coefficient[1];
        j += 1;
    }
    let ancestor_descriptor = extension(&gf81, 3, 12, 3, &ancestor_relation[..12]);
    assert_agrees(&ancestor_descriptor);

    // Explicit characteristic-two ancestor descent, GF(256) over GF(16).
    let gf16 = extension(&prime_base(2), 2, 4, 4, &[1, 1, 0, 0]);
    assert!(gf16.is_irreducible());
    let mut binary_quadratic = [0u64; COORDINATES];
    let mut found_binary_quadratic = false;
    let mut low = 1u64;
    while low < 256 {
        let mut bits = [0u64; COORDINATES];
        let mut bit = 0;
        while bit < 8 {
            bits[bit] = (low >> bit) & 1;
            bit += 1;
        }
        let candidate = extension(&gf16, 2, 8, 2, &bits[..8]);
        if candidate.is_irreducible() {
            binary_quadratic = bits;
            found_binary_quadratic = true;
            break;
        }
        low += 1;
    }
    assert!(found_binary_quadratic);
    let gf256 = extension(&gf16, 2, 8, 2, &binary_quadratic[..8]);
    assert!(gf256.is_irreducible());
    let mut binary_ancestor_relation = [0u64; COORDINATES];
    let mut j = 0;
    while j < 3 {
        binary_ancestor_relation[j * 8] = (j as u64 + 1) & 1;
        binary_ancestor_relation[j * 8 + 1] = 1;
        j += 1;
    }
    let binary_ancestor = extension(&gf256, 2, 24, 3, &binary_ancestor_relation[..24]);
    assert_agrees(&binary_ancestor);
}

#[test]
fn rank_policy_relations_match_oracle() {
    let gf9 = gf9();

    // Relative degrees four and five use normalized rank.
    let quartic = extension(&prime_base(7), 7, 4, 4, &[2, 0, 0, 0]);
    assert_agrees(&quartic);
    let gf9_quintic = extension(&gf9, 3, 10, 5, &[1, 0, 2, 1, 1, 0, 2, 0, 0, 0]);
    assert_agrees(&gf9_quintic);

    // Odd prime degree six: gate and skyline rank, support shifts, lazy sums.
    assert!(prime_policy::wide_safe(3, 6));
    assert!(!prime_policy::wide_safe(u64::MAX - 58, 6));
    let mut prime_six_relation = [0u64; COORDINATES];
    let mut found_prime_six = false;
    loop {
        let candidate = extension(&prime_base(3), 3, 6, 6, &prime_six_relation[..6]);
        if ben_or_irreducible(&candidate, candidate.root_index()) {
            found_prime_six = true;
            break;
        }
        if !next_relation(&mut prime_six_relation[..6], 3) {
            break;
        }
    }
    assert!(found_prime_six);
    let prime_six = extension(&prime_base(3), 3, 6, 6, &prime_six_relation[..6]);
    let mut prime_six_monic = [0u64; COORDINATES];
    prime_six_monic[..6].copy_from_slice(&prime_six_relation[..6]);
    prime_six_monic[6] = 1;
    assert!(prime_policy::discriminant_gate(
        &prime_six_monic[..=6],
        6,
        3
    ));
    assert!(prime_six.is_irreducible());
    assert_agrees(&prime_six);
    // The parity filter rejects before rank when it fails.
    let mut two_factor = [0u64; COORDINATES];
    let two_factor_degree = prime_product(&mut two_factor, &[1, 0], 2, &[2, 0, 0], 3, 7);
    assert!(!prime_policy::discriminant_gate(
        &two_factor[..=two_factor_degree],
        two_factor_degree,
        7
    ));
    let two_factor_descriptor = prime_monic_descriptor(7, &two_factor, two_factor_degree);
    assert!(!two_factor_descriptor.is_irreducible());
    assert_agrees(&two_factor_descriptor);

    // Extension bases at degree six keep rank without a parity gate.
    let mut extension_six = [0u64; COORDINATES];
    extension_six[1] = 1;
    extension_six[2] = 1;
    let extension_six = extension(&gf9, 3, 12, 6, &extension_six[..12]);
    assert_agrees(&extension_six);

    // Lazy-free and bounded-lazy policy helpers remain exercised directly.
    assert!(prime_policy::wide_safe(3, 64));
    assert!(!prime_policy::wide_safe((1u64 << 63) - 25, 8));
}

// ---- Arbitrary node indices.

#[test]
fn arbitrary_node_indices_keep_the_existing_interface() {
    let chain = fan_paar_chain();
    let descriptor = chain[5];
    let mut index = 0;
    while index < descriptor.node_count() {
        if matches!(descriptor.node(index), Node::Extension { .. }) {
            assert_agrees_at(&descriptor, index);
        }
        index += 1;
    }
}

// ---- A checked constructor and an unchecked published row agree.

#[test]
fn unchecked_and_checked_binary_relations_agree_on_the_new_policy() {
    let leaf = Descriptor::polynomial(1, 0b11);
    let checked = extension(&leaf, 2, 3, 3, &[1, 1, 0]);
    let unchecked = unchecked_extension(&leaf, 2, 3, &[1, 1, 0]);
    assert!(checked.is_irreducible());
    assert_agrees(&checked);
    assert_eq!(checked.is_irreducible(), unchecked.is_irreducible());
    assert_agrees(&unchecked);
}

// ---- Larger odd base at a small relative degree.

#[test]
fn larger_odd_base_differentials_exercise_coefficient_inversion() {
    let relation = [2u64, 1, 0, 0, 2, 1, 1, 0, 0, 0, 2, 1, 2, 2, 1, 1];
    let base = extension(&prime_base(3), 3, 16, 16, &relation);
    assert!(base.is_irreducible());
    assert_agrees(&base);

    // x^3 - x - (2 + u), with u the base root: trace(2 + u) = 1, so the
    // Artin-Schreier cubic is irreducible and its constant is 1 + 2u.
    let mut outer = [0u64; COORDINATES];
    outer[0] = 1;
    outer[1] = 2;
    outer[16] = 2;
    let outer = extension(&base, 3, 48, 3, &outer[..48]);
    assert!(outer.is_irreducible());
    assert_agrees(&outer);

    let mut rng = Rng(0xA5A5_5A5A_1234_5678);
    let mut quartic = [0u64; COORDINATES];
    fill_relation(&mut quartic[..64], 3, &mut rng);
    let quartic = extension(&base, 3, 64, 4, &quartic[..64]);
    assert_agrees(&quartic);
}

#[test]
fn nonquadratic_binary_base_inversion_differentials() {
    let gf8 = extension(&prime_base(2), 2, 3, 3, &[1, 1, 0]);
    assert!(gf8.is_irreducible());
    let mut cubic = [0u64; COORDINATES];
    cubic[1] = 1;
    cubic[3] = 1;
    let cubic = extension(&gf8, 2, 9, 3, &cubic[..9]);
    assert_agrees(&cubic);

    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    let mut quartic = [0u64; COORDINATES];
    fill_relation(&mut quartic[..12], 2, &mut rng);
    let quartic = extension(&gf8, 2, 12, 4, &quartic[..12]);
    assert_agrees(&quartic);

    let gf64 = extension(&gf8, 2, 6, 2, &[1, 0, 0, 0, 1, 0]);
    assert!(gf64.is_irreducible());
    let mut recursive = [0u64; COORDINATES];
    recursive[3] = 1;
    recursive[6] = 1;
    let recursive = extension(&gf64, 2, 18, 3, &recursive[..18]);
    assert_agrees(&recursive);
}

#[test]
fn wide_lazy_disabled_rank_matches_ben_or() {
    // 2^64 - 59 is prime; at degree six the bounded lazy sum does not fit
    // u128, so the eager normalized skyline path runs.
    let modulus = u64::MAX - 58;
    assert!(!prime_policy::wide_safe(modulus, 6));
    let base = prime_base(modulus);
    let mut relation = [0u64; 6];
    let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
    let mut found = false;
    let mut attempt = 0;
    while attempt < 128 {
        fill_relation(&mut relation, 1024, &mut rng);
        let candidate = unchecked_extension(&base, modulus, 6, &relation);
        if ben_or_irreducible(&candidate, candidate.root_index()) {
            found = true;
            break;
        }
        attempt += 1;
    }
    assert!(found, "no irreducible sextic found over the large prime");
    let descriptor = unchecked_extension(&base, modulus, 6, &relation);
    assert!(descriptor.is_irreducible());
    assert_agrees(&descriptor);
}

#[test]
fn compact_binary_inverses_match_field_products_and_euclid() {
    let gf8 = extension(&prime_base(2), 2, 3, 3, &[1, 1, 0]);
    let gf64 = extension(&gf8, 2, 6, 2, &[1, 0, 0, 0, 1, 0]);
    for base in [gf8, gf64] {
        let index = base.root_index();
        let shape = coefficient::Shape::of_node(&base, index);
        for value in 0..1u64 << degree_of(&base, index) {
            let mut inverse = [0u64; 1];
            coefficient::inverse(&mut inverse, &base, shape, &[value]);
            assert_eq!(
                inverse[0],
                crate::field::extension::arithmetic::word_inv(&base, index, value)
            );
            let product =
                crate::field::extension::arithmetic::word_mul(&base, index, value, inverse[0]);
            assert_eq!(
                product,
                if value == 0 {
                    0
                } else {
                    word_one(&base, index)
                }
            );
        }
    }
}

#[test]
fn compact_odd_inverses_match_field_products_and_euclid() {
    let smaller = gf9();
    let relation = flatten_odd(2, &first_gf9_quadratics()[0], 2);
    let larger = extension(&smaller, 3, 4, 2, &relation[..4]);
    for base in [smaller, larger] {
        let node = base.root_index();
        let shape = coefficient::Shape::of_node(&base, node);
        let words = shape.words();
        let order = crate::field::extension::arithmetic::order_of(&base, node);
        for index in 0..u64::try_from(order).unwrap() {
            let mut value = zero_element();
            let mut digits = index;
            for coordinate in value.iter_mut().take(words) {
                *coordinate = digits % 3;
                digits /= 3;
            }
            let mut inverse = zero_element();
            coefficient::inverse(&mut inverse[..words], &base, shape, &value[..words]);
            let expected = crate::field::extension::arithmetic::element_inv(&base, node, &value);
            assert_eq!(&inverse[..words], &expected[..words]);
            let product = element_mul(&base, node, &value, &inverse);
            let expected_product = if index == 0 {
                zero_element()
            } else {
                crate::field::extension::arithmetic::element_one(&base, node)
            };
            assert_eq!(&product[..words], &expected_product[..words]);
        }
    }
}
