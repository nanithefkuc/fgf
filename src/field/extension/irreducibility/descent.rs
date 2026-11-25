//! Generalized subfield descent before the tower evaluator.
//!
//! Prime-subfield coefficients and coefficients of an explicitly represented
//! ancestor are detected with the base's own identity and component layout.
//! A non-coprime relation degree and base-extension index reject immediately;
//! an accepted descent recurses through the same evaluator over the smaller
//! represented base. The recursion never needs the node to be the descriptor
//! root, so arbitrary-index validation keeps working.

use super::super::descriptor::{COORDINATES, Descriptor, Node};
use super::coefficient::{Shape, Source, block, identity, is_zero, prefix, prefix_mut};
use super::policy;

/// Greatest common divisor of two indices.
const fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Mask the low `bits` bits without shifting a `u64` by 64.
const fn low_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Detect a prime-subfield or explicit-ancestor descent.
///
/// `None` means the coefficients do not lie in the prime subfield or an
/// explicitly represented ancestor.
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn evaluate(
    descriptor: &Descriptor,
    shape: Shape,
    degree: usize,
    relation: &[u64],
) -> Option<bool> {
    let words = shape.words();
    if shape.degree() <= 1 {
        return None;
    }
    let characteristic = shape.characteristic();
    let mut prime = [0u64; COORDINATES];
    let mut prime_members = true;
    if characteristic == 2 {
        let mut one = [0u64; COORDINATES];
        identity(prefix_mut(&mut one, words), descriptor, shape);
        let mut j = 0;
        while j < degree {
            let coefficient = block(relation, j, words);
            if is_zero(shape, coefficient) {
                prime[j] = 0;
            } else if super::coefficient::eq(shape, coefficient, prefix(&one, words)) {
                prime[j] = 1;
            } else {
                prime_members = false;
                break;
            }
            j += 1;
        }
    } else {
        let mut j = 0;
        while j < degree {
            let coefficient = block(relation, j, words);
            let mut k = 1;
            while k < words {
                if coefficient[k] != 0 {
                    prime_members = false;
                    break;
                }
                k += 1;
            }
            if !prime_members {
                break;
            }
            prime[j] = coefficient[0];
            j += 1;
        }
    }
    if prime_members {
        if gcd(degree, shape.degree()) != 1 {
            return Some(false);
        }
        return Some(policy::decide(
            descriptor,
            Shape::prime(characteristic),
            degree,
            prefix(&prime, degree),
        ));
    }
    let base = match shape.source() {
        Source::Node(index) => index,
        Source::Prime(_) => return None,
    };
    let ancestor = match descriptor.node(base) {
        Node::Extension { base, .. } => *base,
        _ => return None,
    };
    let ancestor_degree = super::super::arithmetic::degree_of(descriptor, ancestor) as usize;
    let ancestor_words = super::super::arithmetic::element_words(descriptor, ancestor);
    let mut lower = [0u64; COORDINATES];
    let mut j = 0;
    while j < degree {
        let coefficient = block(relation, j, words);
        if characteristic == 2 {
            // Component zero is the low word of the packed base element.
            let low = coefficient[0] & low_mask(ancestor_degree as u32);
            if coefficient[0] != low {
                return None;
            }
            lower[j] = low;
        } else {
            let mut k = ancestor_words;
            while k < words {
                if coefficient[k] != 0 {
                    return None;
                }
                k += 1;
            }
            let target = j * ancestor_words;
            let mut k = 0;
            while k < ancestor_words {
                lower[target + k] = coefficient[k];
                k += 1;
            }
        }
        j += 1;
    }
    if gcd(degree, shape.degree() / ancestor_degree) != 1 {
        return Some(false);
    }
    Some(policy::decide(
        descriptor,
        Shape::of_node(descriptor, ancestor),
        degree,
        prefix(&lower, degree * ancestor_words),
    ))
}
