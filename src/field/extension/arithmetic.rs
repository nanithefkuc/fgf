//! The descriptor's element interpreter, in both characteristics.
//!
//! Characteristic-two elements are one packed `u64` coordinate word and
//! travel through the word layer, which the binary description shares.
//! Odd-characteristic elements are fixed-capacity prime-coordinate arrays
//! with `u128` widening products and overflowing-add reduction, so no `u64`
//! overflow occurs for any prime below `2^64`. Relation nodes multiply by
//! schoolbook over their base with reduction by the cached coefficients;
//! quadratic characteristic-two nodes multiply by word-level Karatsuba.
//! Inversion never uses Fermat exponentiation in an extension node:
//! polynomial leaves invert by extended Euclid, basis leaves through their
//! base, quadratic characteristic-two nodes through the norm, and every
//! other relation node by extended Euclid over the base.
//!
//! Identities are read from the node: a basis leaf's identity word is
//! `apply(from_base, 1)`, never assumed to be `1`.

use super::descriptor::{COORDINATES, Descriptor, Node};

/// Interpreter word capacity of one polynomial over a base: one block per
/// coefficient below the relation capacity, plus the monic leading term.
pub const WIDE: usize = 2 * COORDINATES + 2;

/// One element of any node: a packed coordinate word in characteristic two
/// (word `0`), or `COORDINATES` prime coordinates otherwise.
pub type Element = [u64; COORDINATES];

/// One fixed-capacity polynomial over a base, as `Element` blocks.
pub type Wide = [u64; WIDE];

/// The zero element.
#[must_use]
pub const fn zero_element() -> Element {
    [0; COORDINATES]
}

/// Degree of the node at `index`: absolute, over the prime field.
///
/// # Panics
///
/// Panics when the node is unused.
// The relative degree never exceeds the descriptor's coordinate capacity,
// far below `u32::MAX`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub const fn degree_of(descriptor: &Descriptor, index: u8) -> u32 {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { .. } => 1,
        Node::Polynomial { degree, .. } => *degree as u32,
        Node::Basis { base, .. } => degree_of(descriptor, *base),
        Node::Extension {
            base_degree,
            relative_degree,
            ..
        } => *base_degree * (*relative_degree as u32),
    }
}

/// Characteristic of the node at `index`.
///
/// # Panics
///
/// Panics when the node is unused.
#[must_use]
pub const fn characteristic_of(descriptor: &Descriptor, index: u8) -> u64 {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus, .. } => *modulus,
        Node::Polynomial { .. } | Node::Basis { .. } => 2,
        Node::Extension { characteristic, .. } => *characteristic,
    }
}

/// Interpreter word count of one element of the node at `index`.
///
/// # Panics
///
/// Panics when the node is unused.
#[must_use]
pub const fn element_words(descriptor: &Descriptor, index: u8) -> usize {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { .. } | Node::Polynomial { .. } | Node::Basis { .. } => 1,
        Node::Extension {
            characteristic,
            base_degree,
            relative_degree,
            ..
        } => {
            if *characteristic == 2 {
                1
            } else {
                *base_degree as usize * *relative_degree
            }
        }
    }
}

/// Order of the node at `index`, with overflow checking.
///
/// # Panics
///
/// Panics when the node is unused or the order overflows `u128`.
#[must_use]
pub const fn order_of(descriptor: &Descriptor, index: u8) -> u128 {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus, .. } => *modulus as u128,
        Node::Polynomial { degree, .. } => 1u128 << *degree,
        Node::Basis { base, .. } => order_of(descriptor, *base),
        Node::Extension {
            base,
            relative_degree,
            ..
        } => {
            let base_order = order_of(descriptor, *base);
            let mut order = 1u128;
            let mut i = 0;
            while i < *relative_degree {
                order = match order.checked_mul(base_order) {
                    Some(value) => value,
                    None => panic!("field order overflows u128"),
                };
                i += 1;
            }
            order
        }
    }
}

// ---------------------------------------------------------------------------
// Characteristic two: one packed coordinate word per element.

/// The identity word of the characteristic-two node at `index`.
///
/// # Panics
///
/// Panics when the node is unused or not characteristic two.
#[must_use]
pub const fn word_one(descriptor: &Descriptor, index: u8) -> u64 {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus: 2, .. } | Node::Polynomial { .. } => 1,
        Node::Basis { from_base, .. } => crate::field::binary::description::apply(*from_base, 1),
        Node::Extension {
            characteristic: 2,
            base,
            ..
        } => word_one(descriptor, *base),
        Node::Prime { .. } | Node::Extension { .. } => panic!("word identity on an odd node"),
    }
}

/// Word multiplication in the characteristic-two node at `index`.
///
/// # Panics
///
/// Panics when the node is unused or not characteristic two.
#[must_use]
pub const fn word_mul(descriptor: &Descriptor, index: u8, x: u64, y: u64) -> u64 {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus: 2, .. } => x & y,
        Node::Polynomial { degree, full } => {
            let n = *degree as u32;
            polynomial_leaf_mul(*full, n, x, y)
        }
        Node::Basis {
            base,
            to_base,
            from_base,
        } => crate::field::binary::description::apply(
            *from_base,
            word_mul(
                descriptor,
                *base,
                crate::field::binary::description::apply(*to_base, x),
                crate::field::binary::description::apply(*to_base, y),
            ),
        ),
        Node::Extension {
            base,
            relative_degree,
            base_degree,
            characteristic: 2,
            coefficients,
            ..
        } => {
            let base_degree = *base_degree;
            let mask = mask_of(base_degree);
            if *relative_degree == 2 {
                // t^2 = A*t + B with A = coefficients[1], B = coefficients[0].
                let (x0, x1, y0, y1) = (x & mask, x >> base_degree, y & mask, y >> base_degree);
                let ac = word_mul(descriptor, *base, x0, y0);
                let bd = word_mul(descriptor, *base, x1, y1);
                let mid = word_mul(descriptor, *base, x0 ^ x1, y0 ^ y1) ^ ac ^ bd;
                let lo = ac ^ word_mul(descriptor, *base, coefficients[0], bd);
                let hi = mid ^ word_mul(descriptor, *base, coefficients[1], bd);
                (lo & mask) | ((hi & mask) << base_degree)
            } else {
                relation_word_product(
                    descriptor,
                    *base,
                    *relative_degree,
                    base_degree,
                    coefficients,
                    x,
                    y,
                )
            }
        }
        Node::Prime { .. } | Node::Extension { .. } => panic!("word multiplication on an odd node"),
    }
}

/// `low_bits` set bits, never shifting a `u64` by 64.
const fn mask_of(low_bits: u32) -> u64 {
    if low_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << low_bits) - 1
    }
}

/// Leaf multiplication with the pinned degree-eight tables.
#[inline]
const fn polynomial_leaf_mul(full: u128, n: u32, x: u64, y: u64) -> u64 {
    use crate::field::binary::description::{PINNED_AES_LEAF, PINNED_RS_LEAF};
    // Leaf coordinates hold one byte by construction.
    #[allow(clippy::cast_possible_truncation)]
    let pinned = if n == 8 && full == 0x11B {
        Some(PINNED_AES_LEAF.mul(x as u8, y as u8) as u64)
    } else if n == 8 && full == 0x11D {
        Some(PINNED_RS_LEAF.mul(x as u8, y as u8) as u64)
    } else {
        None
    };
    match pinned {
        Some(product) => product,
        None => crate::field::binary::description::poly_mul_serial(full, n, x, y),
    }
}

/// Schoolbook relation product with reduction by the cached coefficients,
/// for characteristic-two nodes of relative degree at least three.
// The single-character locals mirror the node fields they combine, and the
// component counters stay below the characteristic-two degree cap of 64.
#[allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]
const fn relation_word_product(
    descriptor: &Descriptor,
    base: u8,
    relative_degree: usize,
    base_degree: u32,
    coefficients: &[u64; COORDINATES],
    x: u64,
    y: u64,
) -> u64 {
    let mask = mask_of(base_degree);
    let mut words = [0u64; WIDE];
    let mut i = 0;
    while i < relative_degree {
        let xi = (x >> (i as u32 * base_degree)) & mask;
        if xi != 0 {
            let mut j = 0;
            while j < relative_degree {
                words[i + j] ^=
                    word_mul(descriptor, base, xi, (y >> (j as u32 * base_degree)) & mask);
                j += 1;
            }
        }
        i += 1;
    }
    let mut k = 2 * relative_degree - 2;
    while k >= relative_degree {
        let top = words[k];
        if top != 0 {
            let mut j = 0;
            while j < relative_degree {
                words[k - relative_degree + j] ^= word_mul(descriptor, base, top, coefficients[j]);
                j += 1;
            }
        }
        k -= 1;
    }
    let mut result = 0u64;
    i = 0;
    while i < relative_degree {
        result |= (words[i] & mask) << (i as u32 * base_degree);
        i += 1;
    }
    result
}

/// Word exponentiation from the node's identity.
#[must_use]
pub const fn word_pow(descriptor: &Descriptor, index: u8, mut x: u64, mut exponent: u128) -> u64 {
    let mut result = word_one(descriptor, index);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = word_mul(descriptor, index, result, x);
        }
        x = word_mul(descriptor, index, x, x);
        exponent >>= 1;
    }
    result
}

/// Word inverse in the node at `index`, mapping zero to zero.
///
/// # Panics
///
/// Panics when the node is unused or not characteristic two.
#[must_use]
pub const fn word_inv(descriptor: &Descriptor, index: u8, x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus: 2, .. } => x,
        Node::Polynomial { full, .. } => polynomial_leaf_inv(*full, x),
        Node::Basis {
            base,
            to_base,
            from_base,
        } => crate::field::binary::description::apply(
            *from_base,
            word_inv(
                descriptor,
                *base,
                crate::field::binary::description::apply(*to_base, x),
            ),
        ),
        Node::Extension {
            base,
            base_degree,
            relative_degree,
            characteristic: 2,
            coefficients,
            ..
        } => {
            let base_degree = *base_degree;
            let mask = mask_of(base_degree);
            if *relative_degree == 2 {
                // Conjugate (x0 + A*x1, x1) over the norm
                // x0^2 + A*x0*x1 + B*x1^2, with one base inverse.
                let (x0, x1) = (x & mask, x >> base_degree);
                let norm = word_mul(descriptor, *base, x0, x0)
                    ^ word_mul(
                        descriptor,
                        *base,
                        coefficients[1],
                        word_mul(descriptor, *base, x0, x1),
                    )
                    ^ word_mul(
                        descriptor,
                        *base,
                        coefficients[0],
                        word_mul(descriptor, *base, x1, x1),
                    );
                let norm_inverse = word_inv(descriptor, *base, norm);
                (word_mul(
                    descriptor,
                    *base,
                    x0 ^ word_mul(descriptor, *base, coefficients[1], x1),
                    norm_inverse,
                ) & mask)
                    | ((word_mul(descriptor, *base, x1, norm_inverse) & mask) << base_degree)
            } else {
                let mut value = zero_element();
                value[0] = x;
                euclid_inverse(descriptor, index, &value)[0]
            }
        }
        // An odd prime leaf and an odd-characteristic relation node both
        // sit outside the word layer.
        Node::Prime { .. } | Node::Extension { .. } => panic!("word inversion on an odd node"),
    }
}

/// Extended Euclid on the polynomial leaf: `g` with `x*g = 1 (mod full)`.
///
/// The inverse is of the low `degree(full)` bits of `x`, so the loop always
/// terminates, and a word whose low bits are zero inverts to zero.
// The swap loop reuses both operand degrees, so the leading-bit counts stay
// spelled as `127 - leading_zeros` rather than `ilog2`.
#[allow(clippy::manual_ilog2)]
const fn polynomial_leaf_inv(full: u128, x: u64) -> u64 {
    let degree = crate::field::binary::description::poly_degree(full);
    let mask = if degree >= 64 {
        u64::MAX
    } else {
        (1u64 << degree) - 1
    };
    let masked = x & mask;
    if masked == 0 {
        return 0;
    }
    // Operands stay below 2^128: `masked` below the leaf's degree and
    // `full` of degree at most 127.
    let mut u = masked as u128;
    let mut v = full;
    let mut g1 = 1u128;
    let mut g2 = 0u128;
    while u != 1 {
        let mut du = 127 - u.leading_zeros();
        let mut dv = 127 - v.leading_zeros();
        if du < dv {
            core::mem::swap(&mut u, &mut v);
            core::mem::swap(&mut g1, &mut g2);
            core::mem::swap(&mut du, &mut dv);
        }
        let shift = du - dv;
        u ^= v << shift;
        g1 ^= g2 << shift;
    }
    // The Bezout coefficient of a leaf coordinate holds its degree's bits.
    #[allow(clippy::cast_possible_truncation)]
    let inverse = g1 as u64;
    inverse
}

/// Absolute trace to GF(2) in the characteristic-two node at `index`.
#[must_use]
pub const fn word_trace(descriptor: &Descriptor, index: u8, x: u64) -> u64 {
    let n = degree_of(descriptor, index);
    let mut acc = 0;
    let mut power = x;
    let mut k = 0;
    while k < n {
        acc ^= power;
        power = word_mul(descriptor, index, power, power);
        k += 1;
    }
    acc
}

// ---------------------------------------------------------------------------
// Elements: both characteristics.

/// The identity element of the node at `index`.
///
/// # Panics
///
/// Panics when the node is unused.
#[must_use]
pub const fn element_one(descriptor: &Descriptor, index: u8) -> Element {
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { .. } => prime_element(1),
        Node::Polynomial { .. } | Node::Basis { .. } => {
            let mut element = zero_element();
            element[0] = word_one(descriptor, index);
            element
        }
        Node::Extension {
            base, base_words, ..
        } => {
            let base_one = element_one(descriptor, *base);
            let mut element = zero_element();
            let mut k = 0;
            while k < *base_words {
                element[k] = base_one[k];
                k += 1;
            }
            element
        }
    }
}

/// One prime-coordinate element holding the residue `value`.
const fn prime_element(value: u64) -> Element {
    let mut element = zero_element();
    element[0] = value;
    element
}

/// Whether the element is zero in the node at `index`.
#[must_use]
pub const fn element_is_zero(descriptor: &Descriptor, index: u8, element: &Element) -> bool {
    let words = element_words(descriptor, index);
    let mut k = 0;
    while k < words {
        if element[k] != 0 {
            return false;
        }
        k += 1;
    }
    true
}

/// Element equality in the node at `index`.
#[must_use]
pub const fn element_eq(
    descriptor: &Descriptor,
    index: u8,
    left: &Element,
    right: &Element,
) -> bool {
    let words = element_words(descriptor, index);
    let mut k = 0;
    while k < words {
        if left[k] != right[k] {
            return false;
        }
        k += 1;
    }
    true
}

/// Element addition in the node at `index`.
#[must_use]
pub const fn element_add(
    descriptor: &Descriptor,
    index: u8,
    left: &Element,
    right: &Element,
) -> Element {
    if characteristic_of(descriptor, index) == 2 {
        let mut sum = zero_element();
        sum[0] = left[0] ^ right[0];
        return sum;
    }
    let modulus = characteristic_of(descriptor, index);
    let words = element_words(descriptor, index);
    let mut sum = zero_element();
    let mut k = 0;
    while k < words {
        // Operands are canonical residues, so one overflowing add and one
        // conditional subtract reduce the sum for every prime below 2^64.
        let (value, carry) = left[k].overflowing_add(right[k]);
        sum[k] = if carry || value >= modulus {
            value.wrapping_sub(modulus)
        } else {
            value
        };
        k += 1;
    }
    sum
}

/// Additive inverse in the node at `index`.
#[must_use]
pub const fn element_neg(descriptor: &Descriptor, index: u8, element: &Element) -> Element {
    if characteristic_of(descriptor, index) == 2 {
        return *element;
    }
    let modulus = characteristic_of(descriptor, index);
    let words = element_words(descriptor, index);
    let mut negated = zero_element();
    let mut k = 0;
    while k < words {
        negated[k] = if element[k] == 0 {
            0
        } else {
            modulus - element[k]
        };
        k += 1;
    }
    negated
}

/// Element subtraction in the node at `index`.
#[must_use]
pub const fn element_sub(
    descriptor: &Descriptor,
    index: u8,
    left: &Element,
    right: &Element,
) -> Element {
    element_add(
        descriptor,
        index,
        left,
        &element_neg(descriptor, index, right),
    )
}

/// Element multiplication in the node at `index`.
///
/// # Panics
///
/// Panics when the node is unused.
#[must_use]
pub const fn element_mul(
    descriptor: &Descriptor,
    index: u8,
    left: &Element,
    right: &Element,
) -> Element {
    if characteristic_of(descriptor, index) == 2 {
        let mut product = zero_element();
        product[0] = word_mul(descriptor, index, left[0], right[0]);
        return product;
    }
    match descriptor.node(index) {
        Node::Prime { modulus, .. } => {
            let product = left[0] as u128 * (right[0] as u128) % (*modulus as u128);
            #[allow(clippy::cast_possible_truncation)]
            let residue = product as u64;
            prime_element(residue)
        }
        Node::Extension {
            base,
            relative_degree,
            base_words,
            ..
        } => {
            // Schoolbook over the components, then fold
            // t^D = -(c_{D-1} t^{D-1} + ... + c_0) from the top down.
            let (base, relative_degree, base_words) = (*base, *relative_degree, *base_words);
            let mut wide: Wide = [0; WIDE];
            let mut i = 0;
            while i < relative_degree {
                let ai = component(descriptor, index, left, i);
                if !element_is_zero(descriptor, base, &ai) {
                    let mut j = 0;
                    while j < relative_degree {
                        let product = element_mul(
                            descriptor,
                            base,
                            &ai,
                            &component(descriptor, index, right, j),
                        );
                        let sum = element_add(
                            descriptor,
                            base,
                            &wide_get(&wide, i + j, base_words),
                            &product,
                        );
                        wide_put(&mut wide, i + j, base_words, &sum);
                        j += 1;
                    }
                }
                i += 1;
            }
            let mut k = 2 * relative_degree - 2;
            while k >= relative_degree {
                let top = wide_get(&wide, k, base_words);
                if !element_is_zero(descriptor, base, &top) {
                    let mut j = 0;
                    while j < relative_degree {
                        let product = element_mul(
                            descriptor,
                            base,
                            &top,
                            &relation_coefficient(descriptor, index, j),
                        );
                        let difference = element_sub(
                            descriptor,
                            base,
                            &wide_get(&wide, k - relative_degree + j, base_words),
                            &product,
                        );
                        wide_put(&mut wide, k - relative_degree + j, base_words, &difference);
                        j += 1;
                    }
                }
                k -= 1;
            }
            let mut product = zero_element();
            let mut i = 0;
            while i < relative_degree * base_words {
                product[i] = wide[i];
                i += 1;
            }
            product
        }
        _ => panic!("unused node"),
    }
}

/// Element exponentiation from the node's identity.
#[must_use]
pub const fn element_pow(
    descriptor: &Descriptor,
    index: u8,
    element: &Element,
    mut exponent: u128,
) -> Element {
    if characteristic_of(descriptor, index) == 2 {
        let mut result = zero_element();
        result[0] = word_pow(descriptor, index, element[0], exponent);
        return result;
    }
    let mut result = element_one(descriptor, index);
    let mut base = *element;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = element_mul(descriptor, index, &result, &base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = element_mul(descriptor, index, &base, &base);
        }
    }
    result
}

/// Element inverse in the node at `index`, mapping zero to zero.
///
/// # Panics
///
/// Panics when the node is unused.
#[must_use]
pub const fn element_inv(descriptor: &Descriptor, index: u8, element: &Element) -> Element {
    if characteristic_of(descriptor, index) == 2 {
        let mut inverse = zero_element();
        inverse[0] = word_inv(descriptor, index, element[0]);
        return inverse;
    }
    match descriptor.node(index) {
        Node::Prime { modulus, .. } => prime_inverse(*modulus, element[0]),
        Node::Extension { .. } => euclid_inverse(descriptor, index, element),
        _ => panic!("unused node"),
    }
}

/// The inverse of a nonzero residue modulo the odd prime `modulus`, by
/// extended Euclid on integers with `u128` arithmetic.
#[allow(clippy::cast_possible_truncation)]
const fn prime_inverse(modulus: u64, value: u64) -> Element {
    if value == 0 {
        return zero_element();
    }
    // Quotients and reduced cofactors stay below the modulus, so their
    // product fits `u128` for every prime below 2^64.
    let prime = modulus as u128;
    let (mut r, mut new_r) = (prime, value as u128);
    let (mut t, mut new_t) = (0u128, 1u128);
    while new_r != 0 {
        let quotient = r / new_r;
        let remainder = r - quotient * new_r;
        let next_t = (t + prime - quotient * new_t % prime) % prime;
        r = new_r;
        new_r = remainder;
        t = new_t;
        new_t = next_t;
    }
    prime_element(t as u64)
}

// ---------------------------------------------------------------------------
// Relation-node coordinate packing, shared by multiplication, inversion,
// and the irreducibility tests.

#[allow(clippy::cast_possible_truncation)]
// The component index stays below the characteristic-two degree cap of 64.
/// Component `i` of a relation node's element, as a base element.
pub(super) const fn component(
    descriptor: &Descriptor,
    index: u8,
    element: &Element,
    i: usize,
) -> Element {
    let (characteristic, base_degree, base_words) = match descriptor.node(index) {
        Node::Extension {
            characteristic,
            base_degree,
            base_words,
            ..
        } => (*characteristic, *base_degree, *base_words),
        _ => panic!("not a relation node"),
    };
    let mut result = zero_element();
    if characteristic == 2 {
        let shift = i as u32 * base_degree;
        result[0] = (element[0] >> shift) & mask_of(base_degree);
    } else {
        let mut k = 0;
        while k < base_words {
            result[k] = element[i * base_words + k];
            k += 1;
        }
    }
    result
}

#[allow(clippy::cast_possible_truncation)]
// The component index stays below the characteristic-two degree cap of 64.
/// Write component `i` of a relation node's element.
pub(super) const fn set_component(
    descriptor: &Descriptor,
    index: u8,
    element: &mut Element,
    i: usize,
    value: &Element,
) {
    let (characteristic, base_degree, base_words) = match descriptor.node(index) {
        Node::Extension {
            characteristic,
            base_degree,
            base_words,
            ..
        } => (*characteristic, *base_degree, *base_words),
        _ => panic!("not a relation node"),
    };
    if characteristic == 2 {
        let shift = i as u32 * base_degree;
        element[0] = (element[0] & !(mask_of(base_degree) << shift)) | (value[0] << shift);
    } else {
        let mut k = 0;
        while k < base_words {
            element[i * base_words + k] = value[k];
            k += 1;
        }
    }
}

/// Relation coefficient `j` of the node at `index`, as a base element.
pub(super) const fn relation_coefficient(descriptor: &Descriptor, index: u8, j: usize) -> Element {
    let (characteristic, base_words, coefficients) = match descriptor.node(index) {
        Node::Extension {
            characteristic,
            base_words,
            coefficients,
            ..
        } => (*characteristic, *base_words, coefficients),
        _ => panic!("not a relation node"),
    };
    let mut result = zero_element();
    if characteristic == 2 {
        result[0] = coefficients[j];
    } else {
        let mut k = 0;
        while k < base_words {
            result[k] = coefficients[j * base_words + k];
            k += 1;
        }
    }
    result
}

/// Read block `i` of a wide polynomial over a base of `words` words.
pub(super) const fn wide_get(array: &Wide, i: usize, words: usize) -> Element {
    let mut result = zero_element();
    let mut k = 0;
    while k < words {
        result[k] = array[i * words + k];
        k += 1;
    }
    result
}

/// Write block `i` of a wide polynomial over a base of `words` words.
pub(super) const fn wide_put(array: &mut Wide, i: usize, words: usize, value: &Element) {
    let mut k = 0;
    while k < words {
        array[i * words + k] = value[k];
        k += 1;
    }
}

// ---------------------------------------------------------------------------
// Polynomials over a base, for Euclidean inversion and the irreducibility
// tests.

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
// Degree cursors run from a bound down to -1; the index casts match that.
/// Degree of the wide polynomial, or `-1` when it is zero, searching from
/// `from` down.
pub(super) const fn poly_degree(
    descriptor: &Descriptor,
    base: u8,
    array: &Wide,
    words: usize,
    from: usize,
) -> isize {
    let mut i = from as isize;
    while i >= 0 {
        if !element_is_zero(descriptor, base, &wide_get(array, i as usize, words)) {
            return i;
        }
        i -= 1;
    }
    -1
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
// Degree cursors run from a bound down to -1; the index casts match that.
/// Quotient and remainder of `left` by `right`, both over the node `base`.
pub(super) const fn poly_divmod(
    descriptor: &Descriptor,
    base: u8,
    left: &Wide,
    right: &Wide,
    words: usize,
    top: usize,
) -> (Wide, Wide) {
    let divisor_degree = poly_degree(descriptor, base, right, words, top);
    assert!(divisor_degree >= 0, "division by zero polynomial");
    let divisor_degree = divisor_degree as usize;
    let leading_inverse = element_inv(descriptor, base, &wide_get(right, divisor_degree, words));
    let mut remainder = *left;
    let mut quotient: Wide = [0; WIDE];
    let mut remainder_degree = poly_degree(descriptor, base, &remainder, words, top);
    while remainder_degree >= divisor_degree as isize {
        let degree = remainder_degree as usize;
        let factor = element_mul(
            descriptor,
            base,
            &wide_get(&remainder, degree, words),
            &leading_inverse,
        );
        let shift = degree - divisor_degree;
        wide_put(&mut quotient, shift, words, &factor);
        let mut j = 0;
        while j <= divisor_degree {
            let product = element_mul(descriptor, base, &factor, &wide_get(right, j, words));
            let difference = element_sub(
                descriptor,
                base,
                &wide_get(&remainder, shift + j, words),
                &product,
            );
            wide_put(&mut remainder, shift + j, words, &difference);
            j += 1;
        }
        remainder_degree = if degree == 0 {
            -1
        } else {
            poly_degree(descriptor, base, &remainder, words, degree - 1)
        };
    }
    (quotient, remainder)
}

#[allow(clippy::cast_sign_loss)]
// Degree cursors hold nonnegative polynomial degrees; the casts match.
/// Product of two wide polynomials over the node `base`.
const fn poly_mul(
    descriptor: &Descriptor,
    base: u8,
    left: &Wide,
    right: &Wide,
    words: usize,
    top: usize,
) -> Wide {
    let left_degree = poly_degree(descriptor, base, left, words, top);
    let right_degree = poly_degree(descriptor, base, right, words, top);
    let mut product: Wide = [0; WIDE];
    if left_degree < 0 || right_degree < 0 {
        return product;
    }
    let mut i = 0;
    while i <= left_degree as usize {
        let ai = wide_get(left, i, words);
        let mut j = 0;
        while j <= right_degree as usize {
            let term = element_mul(descriptor, base, &ai, &wide_get(right, j, words));
            let sum = element_add(descriptor, base, &wide_get(&product, i + j, words), &term);
            wide_put(&mut product, i + j, words, &sum);
            j += 1;
        }
        i += 1;
    }
    product
}

/// Difference of two wide polynomials over the node `base`, block by block
/// up to `upto`.
const fn poly_sub(
    descriptor: &Descriptor,
    base: u8,
    left: &Wide,
    right: &Wide,
    words: usize,
    upto: usize,
) -> Wide {
    let mut difference: Wide = [0; WIDE];
    let mut i = 0;
    while i <= upto {
        let value = element_sub(
            descriptor,
            base,
            &wide_get(left, i, words),
            &wide_get(right, i, words),
        );
        wide_put(&mut difference, i, words, &value);
        i += 1;
    }
    difference
}

/// Extended Euclid over the relation of the node at `index`: the inverse of
/// a nonzero element, scaled by the inverse of the unit gcd.
pub(super) const fn euclid_inverse(
    descriptor: &Descriptor,
    index: u8,
    element: &Element,
) -> Element {
    let (base, relative_degree, base_words) = match descriptor.node(index) {
        Node::Extension {
            base,
            relative_degree,
            base_words,
            ..
        } => (*base, *relative_degree, *base_words),
        _ => panic!("not a relation node"),
    };
    if element_is_zero(descriptor, index, element) {
        return zero_element();
    }
    // r0 walks the relation, r1 the element; s1 tracks the element's
    // Bezout coefficient. The loop leaves the gcd in r0 and the inverse's
    // cofactor in s0.
    let mut r0: Wide = [0; WIDE];
    let mut i = 0;
    while i < relative_degree {
        wide_put(
            &mut r0,
            i,
            base_words,
            &relation_coefficient(descriptor, index, i),
        );
        i += 1;
    }
    wide_put(
        &mut r0,
        relative_degree,
        base_words,
        &element_one(descriptor, base),
    );
    let mut r1: Wide = [0; WIDE];
    i = 0;
    while i < relative_degree {
        wide_put(
            &mut r1,
            i,
            base_words,
            &component(descriptor, index, element, i),
        );
        i += 1;
    }
    let mut s0: Wide = [0; WIDE];
    let mut s1: Wide = [0; WIDE];
    wide_put(&mut s1, 0, base_words, &element_one(descriptor, base));
    while poly_degree(descriptor, base, &r1, base_words, relative_degree) >= 0 {
        let (quotient, remainder) =
            poly_divmod(descriptor, base, &r0, &r1, base_words, relative_degree);
        r0 = r1;
        r1 = remainder;
        let next = poly_sub(
            descriptor,
            base,
            &s0,
            &poly_mul(
                descriptor,
                base,
                &quotient,
                &s1,
                base_words,
                relative_degree,
            ),
            base_words,
            relative_degree,
        );
        s0 = s1;
        s1 = next;
    }
    let gcd_inverse = element_inv(descriptor, base, &wide_get(&r0, 0, base_words));
    let mut inverse = zero_element();
    i = 0;
    while i < relative_degree {
        let value = element_mul(
            descriptor,
            base,
            &wide_get(&s0, i, base_words),
            &gcd_inverse,
        );
        set_component(descriptor, index, &mut inverse, i, &value);
        i += 1;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::extension::descriptor::{PrimePresentation, Storage};
    use crate::field::extension::pinned;
    use crate::field::{Elem, Goldilocks, Mersenne31, QuadMersenne31};

    /// Deterministic 64-bit words from the crate's shared LCG constants.
    fn words(seed: u64) -> impl Iterator<Item = u64> {
        let mut state = seed | 1;
        core::iter::repeat_with(move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            state >> 24
        })
    }

    #[test]
    #[should_panic(expected = "word inversion on an odd node")]
    fn word_inversion_rejects_an_odd_extension_node() {
        let descriptor = super::super::descriptor::Descriptor::extension(
            &super::super::descriptor::Descriptor::prime(7, PrimePresentation::Residue),
            7,
            2,
            2,
            &[4, 0],
            Storage::Components,
        );
        let _ = word_inv(&descriptor, descriptor.root_index(), 1);
    }

    /// Carry-less product of two GF(2) polynomials below degree six.
    fn clmul_small(x: u64, y: u64) -> u64 {
        let mut acc = 0;
        let mut k = 0;
        while k < 6 {
            if (y >> k) & 1 == 1 {
                acc ^= x << k;
            }
            k += 1;
        }
        acc
    }

    /// Independent GF(8) product under `x^3 + x + 1`.
    fn gf8_mul(x: u64, y: u64) -> u64 {
        let mut acc = clmul_small(x, y);
        let mut k = 5;
        while k >= 3 {
            if (acc >> k) & 1 == 1 {
                acc ^= 1 << k;
                acc ^= 1 << (k - 2);
                acc ^= 1 << (k - 3);
            }
            k -= 1;
        }
        acc & 0b111
    }

    /// Independent GF(27) product under `x^3 - x - 1` over GF(3).
    fn gf27_mul(x: &[u64; 3], y: &[u64; 3]) -> [u64; 3] {
        let p = 3u64;
        let mut w = [0u64; 5];
        for i in 0..3 {
            for j in 0..3 {
                w[i + j] = (w[i + j] + x[i] * y[j]) % p;
            }
        }
        // t^3 = t + 1.
        for k in (3..5).rev() {
            let top = w[k];
            if top != 0 {
                w[k] = 0;
                w[k - 3 + 1] = (w[k - 3 + 1] + top) % p;
                w[k - 3] = (w[k - 3] + top) % p;
            }
        }
        [w[0], w[1], w[2]]
    }

    /// Independent GF(64) product under `t^2 + alpha*t + 1` over the
    /// independent GF(8) multiplication.
    fn gf64_mul(x: u64, y: u64) -> u64 {
        let (x0, x1, y0, y1) = (x & 7, x >> 3, y & 7, y >> 3);
        let top = gf8_mul(x1, y1);
        let lo = gf8_mul(x0, y0) ^ gf8_mul(1, top);
        let hi = gf8_mul(x0, y1) ^ gf8_mul(x1, y0) ^ gf8_mul(0b010, top);
        lo | hi << 3
    }

    /// Independent BabyBear-quartic product under `t^4 - 11`.
    // The widened products fall back below the modulus before the cast.
    #[allow(clippy::cast_possible_truncation)]
    fn babybear_mul(x: &[u64; 4], y: &[u64; 4]) -> [u64; 4] {
        const P: u64 = 2_013_265_921;
        let mut w = [0u64; 7];
        for i in 0..4 {
            for j in 0..4 {
                w[i + j] = (u128::from(w[i + j]) + u128::from(x[i]) * u128::from(y[j])) as u64 % P;
            }
        }
        // t^4 = 11.
        for k in (4..7).rev() {
            let top = w[k];
            if top != 0 {
                w[k] = 0;
                w[k - 4] = (u128::from(w[k - 4]) + u128::from(top) * 11u128) as u64 % P;
            }
        }
        [w[0], w[1], w[2], w[3]]
    }

    /// GF(8) as `x^3 + x + 1` over the GF(2) residue leaf.
    const GF8: Descriptor = Descriptor::extension(
        &Descriptor::prime(2, PrimePresentation::Residue),
        2,
        3,
        3,
        &[1, 1, 0],
        Storage::Components,
    );

    /// GF(64) as `t^2 + alpha*t + 1` over that GF(8).
    const GF64: Descriptor =
        Descriptor::extension(&GF8, 2, 6, 2, &[1, 0, 0, 0, 1, 0], Storage::Components);

    #[test]
    fn inversion_over_non_quadratic_relations_matches_independent_searches() {
        const BABY_BEAR_QUARTIC: u64 = 2_013_265_921;
        let root = GF8.root_index();
        for a in 1..8u64 {
            let mut value = zero_element();
            value[0] = a;
            let inverse = element_inv(&GF8, root, &value)[0];
            // Brute-force search with the independent multiplication.
            let mut found = 0u64;
            for b in 1..8u64 {
                if gf8_mul(a, b) == 1 {
                    found = b;
                    break;
                }
            }
            assert_ne!(found, 0, "no inverse found for {a}");
            assert_eq!(inverse, found, "inverse of {a} in GF(8)");
        }
        // GF(27) as x^3 - x - 1 over GF(3): every nonzero element.
        let gf27 = Descriptor::extension(
            &Descriptor::prime(3, PrimePresentation::Residue),
            3,
            3,
            3,
            &[2, 2, 0],
            Storage::Components,
        );
        let gf27_root = gf27.root_index();
        for digits in 1..27u64 {
            let coords = [digits % 3, (digits / 3) % 3, digits / 9];
            if coords == [0, 0, 0] {
                continue;
            }
            let mut value = zero_element();
            value[..3].copy_from_slice(&coords);
            let inverse = element_inv(&gf27, gf27_root, &value);
            let product = gf27_mul(&coords, &[inverse[0], inverse[1], inverse[2]]);
            assert_eq!(product, [1, 0, 0], "inverse of {digits} in GF(27)");
        }
        // GF(64) as t^2 + alpha*t + 1 over GF(8): every nonzero element.
        let gf64_root = GF64.root_index();
        for a in 1..64u64 {
            let mut value = zero_element();
            value[0] = a;
            let inverse = element_inv(&GF64, gf64_root, &value)[0];
            assert_eq!(gf64_mul(a, inverse), 1, "inverse of {a:#x} in GF(64)");
        }
        // BabyBear quartic samples: deterministic elements with several
        // nonzero components.
        let quartic = Descriptor::extension(
            &Descriptor::prime(BABY_BEAR_QUARTIC, PrimePresentation::Residue),
            BABY_BEAR_QUARTIC,
            4,
            4,
            &[BABY_BEAR_QUARTIC - 11, 0, 0, 0],
            Storage::Components,
        );
        let quartic_root = quartic.root_index();
        for (index, w) in words(0xB4B4).enumerate().take(16) {
            let mut coords = [0u64; 4];
            let mut shifted = w;
            for coord in &mut coords {
                *coord = shifted % BABY_BEAR_QUARTIC;
                shifted = (shifted / BABY_BEAR_QUARTIC) ^ (shifted << 7);
            }
            if coords == [0, 0, 0, 0] {
                continue;
            }
            let mut value = zero_element();
            value[..4].copy_from_slice(&coords);
            let inverse = element_inv(&quartic, quartic_root, &value);
            let product = babybear_mul(&coords, &[inverse[0], inverse[1], inverse[2], inverse[3]]);
            assert_eq!(product, [1, 0, 0, 0], "inverse of sample {index}");
        }
    }

    #[test]
    fn prime_leaves_match_runtime_field_arithmetic() {
        // Widened products reduce below `P` before the narrowing casts.
        #[allow(clippy::cast_possible_truncation)]
        fn check<const P: u64>(presentation: PrimePresentation, boundary: &[u64]) {
            let descriptor = Descriptor::prime(P, presentation);
            let root = descriptor.root_index();
            let one = element_one(&descriptor, root);
            assert_eq!(one[0], 1);
            for &a in boundary {
                for &b in boundary {
                    let (x, y) = (a % P, b % P);
                    let mut xa = zero_element();
                    xa[0] = x;
                    let mut ya = zero_element();
                    ya[0] = y;
                    assert_eq!(
                        element_add(&descriptor, root, &xa, &ya)[0],
                        u64::try_from((u128::from(x) + u128::from(y)) % u128::from(P)).unwrap(),
                        "sum {x} + {y}"
                    );
                    assert_eq!(
                        element_mul(&descriptor, root, &xa, &ya)[0],
                        (u128::from(x) * u128::from(y) % u128::from(P)) as u64,
                        "product {x} * {y}"
                    );
                    if x != 0 {
                        let inverse = element_inv(&descriptor, root, &xa)[0];
                        assert_eq!(
                            (u128::from(x) * u128::from(inverse) % u128::from(P)) as u64,
                            1,
                            "inverse of {x}"
                        );
                    }
                }
            }
        }
        check::<7>(PrimePresentation::Residue, &[0, 1, 2, 6]);
        check::<2_013_265_921>(PrimePresentation::Residue, &[0, 1, 2, 2_013_265_920]);
        check::<0xFFFF_FFFF_FFFF_FFC5>(
            PrimePresentation::Residue,
            &[0, 1, 2, 0xFFFF_FFFF_FFFF_FFC4],
        );
        check::<0x7FFF_FFFF>(PrimePresentation::MersenneFold, &[0, 1, 2, 0x7FFF_FFFE]);
        check::<0xFFFF_FFFF_0000_0001>(
            PrimePresentation::GoldilocksFold,
            &[0, 1, 2, 0xFFFF_FFFF_0000_0000],
        );
        // The named rows agree with the interpreter on their own elements.
        let m31 = Descriptor::prime(0x7FFF_FFFF, PrimePresentation::MersenneFold);
        let m31_root = m31.root_index();
        for w in words(0x31).take(24) {
            let x = Elem::<Mersenne31>::from_raw((w % 0x7FFF_FFFF) as u32);
            let y = Elem::<Mersenne31>::from_raw(((w >> 20) % 0x7FFF_FFFF) as u32);
            let mut xa = zero_element();
            xa[0] = u64::from(x.to_raw());
            let mut ya = zero_element();
            ya[0] = u64::from(y.to_raw());
            assert_eq!(
                element_mul(&m31, m31_root, &xa, &ya)[0],
                u64::from((x * y).to_raw())
            );
            assert_eq!(
                element_inv(&m31, m31_root, &xa)[0],
                u64::from(x.inv().to_raw())
            );
        }
        let gld = Descriptor::prime(0xFFFF_FFFF_0000_0001, PrimePresentation::GoldilocksFold);
        let gld_root = gld.root_index();
        for w in words(0x64D).take(24) {
            let x = Elem::<Goldilocks>::from_raw(w);
            let y = Elem::<Goldilocks>::from_raw(w.rotate_left(17));
            let mut xa = zero_element();
            xa[0] = x.to_raw();
            let mut ya = zero_element();
            ya[0] = y.to_raw();
            assert_eq!(element_mul(&gld, gld_root, &xa, &ya)[0], (x * y).to_raw());
            assert_eq!(element_inv(&gld, gld_root, &xa)[0], x.inv().to_raw());
        }
    }

    #[test]
    // Limbs stay below the 31-bit modulus, so the narrowing casts are exact.
    #[allow(clippy::cast_possible_truncation)]
    fn quad_mersenne31_descriptor_matches_field_arithmetic() {
        let descriptor = &pinned::QUAD_MERSENNE31;
        let root = descriptor.root_index();
        let from_parts = |raw: (u32, u32)| {
            let mut element = zero_element();
            element[0] = u64::from(raw.0);
            element[1] = u64::from(raw.1);
            element
        };
        for w in words(0x931).take(32) {
            let x = Elem::<QuadMersenne31>::from_raw(
                (w % 0x7FFF_FFFF) as u32,
                ((w >> 24) % 0x7FFF_FFFF) as u32,
            );
            let y = Elem::<QuadMersenne31>::from_raw(
                ((w >> 11) % 0x7FFF_FFFF) as u32,
                ((w >> 7) % 0x7FFF_FFFF) as u32,
            );
            let product = element_mul(
                descriptor,
                root,
                &from_parts(x.to_raw()),
                &from_parts(y.to_raw()),
            );
            let expected = (x * y).to_raw();
            assert_eq!(
                [product[0] as u32, product[1] as u32],
                [expected.0, expected.1]
            );
            let inverse = element_inv(descriptor, root, &from_parts(x.to_raw()));
            let expected = x.inv().to_raw();
            assert_eq!(
                [inverse[0] as u32, inverse[1] as u32],
                [expected.0, expected.1]
            );
        }
    }
}
