//! The relation-evaluation dispatch order.
//!
//! The shipped quadratic criterion and packed GF(2) Rabin test are selected
//! before this dispatcher. Descent runs before the tower evaluator; degree
//! three takes the exact single checkpoint; degrees four and five take
//! normalized rank; odd prime bases at degree six and above take the gate and
//! skyline rank; extension bases keep normalized rank without fraction-free
//! pivots or parity.

use super::super::descriptor::Descriptor;
use super::checkpoint;
use super::coefficient::{Shape, prefix};
use super::descent;
use super::prime;
use super::rank;

/// Evaluate a relation by its relative degree and base shape.
pub(super) const fn decide(
    descriptor: &Descriptor,
    shape: Shape,
    degree: usize,
    relation: &[u64],
) -> bool {
    if shape.characteristic() == 2 && shape.degree() == 1 {
        return super::binary_rabin_irreducible(prefix(relation, degree), degree);
    }
    if let Some(verdict) = descent::evaluate(descriptor, shape, degree, relation) {
        return verdict;
    }
    if degree == 3 {
        return checkpoint::single(descriptor, shape, relation, degree);
    }
    if let Some(modulus) = shape.prime_modulus(descriptor) {
        return if degree <= 5 {
            prime::normalized(descriptor, modulus, degree, relation)
        } else {
            prime::decide(descriptor, modulus, degree, relation)
        };
    }
    rank::normalized(descriptor, shape, relation, degree)
}
