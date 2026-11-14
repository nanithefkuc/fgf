//! Irreducibility of a relation node over its base, decided exactly in
//! const.
//!
//! The shipped quadratic criterion and the packed GF(2) Rabin test keep
//! their positions. Every other relation follows generalized subfield
//! descent first, the exact single Ben-Or checkpoint at relative degree
//! three, normalized streaming rank at degrees four and five, and for odd
//! prime bases at degree six and above the accumulated-resultant parity
//! filter followed by normalized skyline rank. Extension bases keep
//! normalized rank without fraction-free pivots or a parity gate.
//!
//! The general evaluator uses a flat packed occupied prefix and a private
//! coefficient shape, never a capacity-sized [`Element`](super::arithmetic::Element)
//! copy. Pinned descriptor acceptance still bypasses the predicate in
//! [`validate_relation`].

mod checkpoint;
mod coefficient;
mod descent;
mod policy;
mod polynomial;
mod prime;
mod rank;

use super::arithmetic::{
    element_eq, element_is_zero, element_mul, element_one, element_pow, element_sub, order_of,
    relation_coefficient, word_inv, word_mul, word_one,
};
use super::descriptor::{Descriptor, Node};
use super::pinned;

/// Whether the relation node at `index` is irreducible over its base.
///
/// Uses the quadratic criterion at relative degree two, packed Rabin
/// validation over GF(2), and the dispatched general policy otherwise. The
/// index need not be the descriptor root: every path reads the node's own
/// base.
///
/// # Panics
///
/// Panics when the node at `index` holds no relation.
#[must_use]
pub const fn node_relation_irreducible(descriptor: &Descriptor, index: u8) -> bool {
    match descriptor.node(index) {
        Node::Extension {
            base,
            relative_degree,
            characteristic,
            base_degree,
            base_words,
            coefficients,
            ..
        } => {
            let degree = *relative_degree;
            if degree == 2 {
                return quadratic_relation_holds(descriptor, index);
            }
            if *characteristic == 2 && *base_degree == 1 {
                return binary_rabin_irreducible(coefficient::prefix(coefficients, degree), degree);
            }
            let shape = coefficient::Shape::of_node(descriptor, *base);
            policy::decide(
                descriptor,
                shape,
                degree,
                coefficient::prefix(coefficients, degree * *base_words),
            )
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

/// Rabin's criterion for a monic relation over GF(2), packed into bits.
///
/// The descriptor bounds the degree by 64. Residues have degree below 64,
/// so their squares fit in `u128`, including the degree-64 modulus's leading
/// bit. Each distinct prime divisor `p` of `degree` requires a unit GCD at
/// Frobenius step `degree / p`; the final step must return the indeterminate.
const fn binary_rabin_irreducible(coefficients: &[u64], degree: usize) -> bool {
    let mut modulus = 1u128 << degree;
    let mut i = 0;
    while i < degree {
        modulus |= (coefficients[i] as u128) << i;
        i += 1;
    }
    let mut checkpoints = 0u64;
    let mut remaining = degree;
    let mut prime = 2;
    while prime * prime <= remaining {
        if remaining.is_multiple_of(prime) {
            checkpoints |= 1u64 << (degree / prime);
            while remaining.is_multiple_of(prime) {
                remaining /= prime;
            }
        }
        prime += 1;
    }
    if remaining > 1 {
        checkpoints |= 1u64 << (degree / remaining);
    }
    let mut power = 2u128;
    let mut step = 1;
    while step <= degree {
        let mut square = 0u128;
        let mut bits = power;
        while bits != 0 {
            square |= 1u128 << (2 * bits.trailing_zeros());
            bits &= bits - 1;
        }
        power = binary_remainder(square, modulus);
        if step < degree && checkpoints & (1u64 << step) != 0 {
            let mut a = modulus;
            let mut b = power ^ 2;
            while b != 0 {
                let remainder = binary_remainder(a, b);
                a = b;
                b = remainder;
            }
            if a != 1 {
                return false;
            }
        }
        step += 1;
    }
    power == 2
}

/// Remainder of packed binary polynomials with a nonzero divisor.
const fn binary_remainder(mut dividend: u128, divisor: u128) -> u128 {
    let degree = divisor.ilog2();
    while dividend != 0 && dividend.ilog2() >= degree {
        dividend ^= divisor << (dividend.ilog2() - degree);
    }
    dividend
}

#[cfg(test)]
mod tests;
