//! Flat coefficient arithmetic for the general relation evaluator.
//!
//! One coefficient of a polynomial over a base node is an occupied slice of
//! `Shape::words()` words. A shape names the base's arithmetic and element
//! layout, so every operation here works on the occupied prefix only. Prime
//! coefficients reduce scalars; characteristic-two coefficients use the
//! descriptor's word layer, which keeps basis and identity semantics;
//! relation-node coefficients recurse over their components and invert by
//! the same extended Euclidean pass the runtime interpreter uses, keeping the
//! quadratic norm specialization. All buffer spans come from const
//! `split_at`/`split_at_mut` helpers; no range indexing is assumed in const.

use super::super::arithmetic::{
    characteristic_of, degree_of, element_words, order_of, word_inv, word_mul, word_one,
};
use super::super::descriptor::{COORDINATES, Descriptor, Node};

/// Capacity of one flat polynomial scratch, in `u64` words.
pub(super) const SCRATCH: usize = 2 * COORDINATES + 2;

/// Occupied prefix of `data`.
#[must_use]
pub(super) const fn prefix(data: &[u64], words: usize) -> &[u64] {
    let (head, _) = data.split_at(words);
    head
}

/// Mutable occupied prefix of `data`.
pub(super) const fn prefix_mut(data: &mut [u64], words: usize) -> &mut [u64] {
    let (head, _) = data.split_at_mut(words);
    head
}

/// Block `index` of `data`, `words` wide.
#[must_use]
pub(super) const fn block(data: &[u64], index: usize, words: usize) -> &[u64] {
    let (_, tail) = data.split_at(index * words);
    let (block, _) = tail.split_at(words);
    block
}

/// Mutable block `index` of `data`, `words` wide.
pub(super) const fn block_mut(data: &mut [u64], index: usize, words: usize) -> &mut [u64] {
    let (_, tail) = data.split_at_mut(index * words);
    let (block, _) = tail.split_at_mut(words);
    block
}

/// Copy `count` blocks from `from` into `to`.
pub(super) const fn copy_blocks(to: &mut [u64], from: &[u64], shape: Shape, count: usize) {
    let used = count * shape.words();
    prefix_mut(to, used).copy_from_slice(prefix(from, used));
}

/// Zero `count` blocks of `out`.
pub(super) const fn zero_blocks(out: &mut [u64], shape: Shape, count: usize) {
    let words = shape.words();
    let mut index = 0;
    while index < count {
        zero(block_mut(out, index, words), shape);
        index += 1;
    }
}

/// Where a coefficient shape's arithmetic comes from.
#[derive(Clone, Copy)]
pub(super) enum Source {
    /// A virtual prime field named by its modulus.
    Prime(u64),
    /// The descriptor node at this index.
    Node(u8),
}

/// The arithmetic and element layout of one base coefficient.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    source: Source,
    words: usize,
    degree: usize,
    characteristic: u64,
}

impl Shape {
    /// The shape of the descriptor node at `index`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub(super) const fn of_node(descriptor: &Descriptor, index: u8) -> Self {
        Self {
            source: Source::Node(index),
            words: element_words(descriptor, index),
            degree: degree_of(descriptor, index) as usize,
            characteristic: characteristic_of(descriptor, index),
        }
    }

    /// A virtual prime field, used by a prime-subfield descent.
    #[must_use]
    pub(super) const fn prime(modulus: u64) -> Self {
        Self {
            source: Source::Prime(modulus),
            words: 1,
            degree: 1,
            characteristic: modulus,
        }
    }

    /// Occupied words of one coefficient.
    #[must_use]
    pub(super) const fn words(self) -> usize {
        self.words
    }

    /// Absolute degree of the base over the prime field.
    #[must_use]
    pub(super) const fn degree(self) -> usize {
        self.degree
    }

    /// Characteristic of the coefficient field.
    #[must_use]
    pub(super) const fn characteristic(self) -> u64 {
        self.characteristic
    }

    /// The arithmetic source.
    #[must_use]
    pub(super) const fn source(self) -> Source {
        self.source
    }

    /// The prime modulus when this shape is a prime field.
    #[must_use]
    pub(super) const fn prime_modulus(self, descriptor: &Descriptor) -> Option<u64> {
        match self.source {
            Source::Prime(modulus) => Some(modulus),
            Source::Node(index) => match descriptor.node(index) {
                Node::Prime { modulus, .. } => Some(*modulus),
                _ => None,
            },
        }
    }

    /// Multiplicative order of the base.
    #[must_use]
    pub(super) const fn order(self, descriptor: &Descriptor) -> u128 {
        match self.source {
            Source::Prime(modulus) => modulus as u128,
            Source::Node(index) => order_of(descriptor, index),
        }
    }
}

/// Write the shape's multiplicative identity into `out`.
pub(super) const fn identity(out: &mut [u64], descriptor: &Descriptor, shape: Shape) {
    let words = shape.words();
    let mut k = 1;
    while k < words {
        out[k] = 0;
        k += 1;
    }
    out[0] = match shape.source() {
        Source::Prime(_) => 1,
        Source::Node(index) => {
            if shape.characteristic() == 2 {
                word_one(descriptor, index)
            } else {
                1
            }
        }
    };
}

/// Zero an occupied coefficient.
pub(super) const fn zero(out: &mut [u64], shape: Shape) {
    let words = shape.words();
    let mut k = 0;
    while k < words {
        out[k] = 0;
        k += 1;
    }
}

/// Copy one occupied coefficient from `from` into `to`.
pub(super) const fn copy(to: &mut [u64], from: &[u64], shape: Shape) {
    let mut k = 0;
    while k < shape.words {
        to[k] = from[k];
        k += 1;
    }
}

/// Whether an occupied coefficient is zero.
#[must_use]
pub(super) const fn is_zero(shape: Shape, value: &[u64]) -> bool {
    let words = shape.words();
    let mut k = 0;
    while k < words {
        if value[k] != 0 {
            return false;
        }
        k += 1;
    }
    true
}

/// Whether two occupied coefficients are equal.
#[must_use]
pub(super) const fn eq(shape: Shape, left: &[u64], right: &[u64]) -> bool {
    let words = shape.words();
    let mut k = 0;
    while k < words {
        if left[k] != right[k] {
            return false;
        }
        k += 1;
    }
    true
}

/// `left + right` modulo the odd characteristic.
#[must_use]
const fn add_mod(left: u64, right: u64, modulus: u64) -> u64 {
    let complement = modulus - right;
    if left >= complement {
        left - complement
    } else {
        left + right
    }
}

/// `left - right` modulo the odd characteristic.
#[must_use]
const fn sub_mod(left: u64, right: u64, modulus: u64) -> u64 {
    if left >= right {
        left - right
    } else {
        modulus - (right - left)
    }
}

/// `left * right` modulo the odd characteristic.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
const fn mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    (left as u128 * right as u128 % modulus as u128) as u64
}

/// Product of scalar polynomials modulo a monic prime-field relation.
pub(super) const fn prime_mulmod(
    out: &mut [u64],
    modulus: u64,
    relation: &[u64],
    degree: usize,
    left: &[u64],
    right: &[u64],
) {
    if degree == 2 {
        let low = mul_mod(left[0], right[0], modulus);
        let high = mul_mod(left[1], right[1], modulus);
        let cross = add_mod(
            mul_mod(left[0], right[1], modulus),
            mul_mod(left[1], right[0], modulus),
            modulus,
        );
        out[0] = sub_mod(low, mul_mod(relation[0], high, modulus), modulus);
        out[1] = sub_mod(cross, mul_mod(relation[1], high, modulus), modulus);
        return;
    }
    let mut left_degree = degree - 1;
    while left_degree > 0 && left[left_degree] == 0 {
        left_degree -= 1;
    }
    let mut right_degree = degree - 1;
    while right_degree > 0 && right[right_degree] == 0 {
        right_degree -= 1;
    }
    let mut wide = [0u64; SCRATCH];
    let mut i = 0;
    while i <= left_degree {
        if left[i] != 0 {
            let mut j = 0;
            while j <= right_degree {
                if right[j] != 0 {
                    wide[i + j] =
                        add_mod(wide[i + j], mul_mod(left[i], right[j], modulus), modulus);
                }
                j += 1;
            }
        }
        i += 1;
    }
    let mut k = left_degree + right_degree;
    while k >= degree {
        let top = wide[k];
        if top != 0 {
            let mut j = 0;
            while j < degree {
                if relation[j] != 0 {
                    let target = k - degree + j;
                    wide[target] =
                        sub_mod(wide[target], mul_mod(top, relation[j], modulus), modulus);
                }
                j += 1;
            }
        }
        k -= 1;
    }
    let mut k = 0;
    while k < degree {
        out[k] = wide[k];
        k += 1;
    }
}

/// Difference of two occupied coefficients into `out`.
pub(super) const fn sub(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
) {
    let words = shape.words();
    let characteristic = shape.characteristic();
    let mut k = 0;
    while k < words {
        out[k] = if characteristic == 2 {
            left[k] ^ right[k]
        } else {
            sub_mod(left[k], right[k], characteristic)
        };
        k += 1;
    }
    let _ = descriptor;
}

/// `out -= other` over occupied words.
pub(super) const fn sub_assign(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    other: &[u64],
) {
    let words = shape.words();
    let characteristic = shape.characteristic();
    let mut k = 0;
    while k < words {
        out[k] = if characteristic == 2 {
            out[k] ^ other[k]
        } else {
            sub_mod(out[k], other[k], characteristic)
        };
        k += 1;
    }
    let _ = descriptor;
}

/// Multiply an occupied coefficient by a prime-subfield scalar into `out`.
pub(super) const fn scalar_mul(out: &mut [u64], shape: Shape, value: &[u64], factor: u64) {
    let words = shape.words();
    if shape.characteristic() == 2 {
        if factor & 1 == 0 {
            zero(out, shape);
        } else {
            copy(out, value, shape);
        }
        return;
    }
    let factor = factor % shape.characteristic();
    let mut k = 0;
    while k < words {
        out[k] = mul_mod(value[k], factor, shape.characteristic());
        k += 1;
    }
}

/// Add or subtract `left * right` into `out`.
const fn mul_acc(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
    subtract: bool,
) {
    if shape.characteristic() == 2 {
        let product = match shape.source() {
            Source::Prime(_) => left[0] & right[0],
            Source::Node(index) => word_mul(descriptor, index, left[0], right[0]),
        };
        out[0] ^= product;
        return;
    }
    if let Some(modulus) = shape.prime_modulus(descriptor) {
        let product = mul_mod(left[0], right[0], modulus);
        out[0] = if subtract {
            sub_mod(out[0], product, modulus)
        } else {
            add_mod(out[0], product, modulus)
        };
        return;
    }
    extension_mul_acc(out, descriptor, shape, left, right, subtract);
}

/// Add or subtract the product of two odd-extension coefficients.
const fn extension_mul_acc(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
    subtract: bool,
) {
    let (base, relative_degree, base_words, coefficients) = match shape.source() {
        Source::Node(index) => match descriptor.node(index) {
            Node::Extension {
                base,
                relative_degree,
                base_words,
                coefficients,
                ..
            } => (*base, *relative_degree, *base_words, coefficients),
            _ => panic!("unsupported coefficient shape"),
        },
        Source::Prime(_) => panic!("unsupported coefficient shape"),
    };
    let base_shape = Shape::of_node(descriptor, base);
    if let Some(modulus) = base_shape.prime_modulus(descriptor) {
        let mut product = [0u64; COORDINATES];
        prime_mulmod(
            &mut product,
            modulus,
            coefficients,
            relative_degree,
            left,
            right,
        );
        let mut k = 0;
        while k < relative_degree {
            out[k] = if subtract {
                sub_mod(out[k], product[k], modulus)
            } else {
                add_mod(out[k], product[k], modulus)
            };
            k += 1;
        }
        return;
    }
    let blocks = 2 * relative_degree - 1;
    let mut wide = [0u64; SCRATCH];
    let mut i = 0;
    while i < relative_degree {
        let ai = block(left, i, base_words);
        if !is_zero(base_shape, ai) {
            let mut j = 0;
            while j < relative_degree {
                let bj = block(right, j, base_words);
                if !is_zero(base_shape, bj) {
                    mul_acc(
                        block_mut(&mut wide, i + j, base_words),
                        descriptor,
                        base_shape,
                        ai,
                        bj,
                        false,
                    );
                }
                j += 1;
            }
        }
        i += 1;
    }
    let mut k = blocks - 1;
    while k >= relative_degree {
        if !is_zero(base_shape, block(&wide, k, base_words)) {
            let mut top = [0u64; COORDINATES];
            copy(
                prefix_mut(&mut top, base_words),
                block(&wide, k, base_words),
                base_shape,
            );
            let mut j = 0;
            while j < relative_degree {
                let relation = block(coefficients, j, base_words);
                if !is_zero(base_shape, relation) {
                    mul_acc(
                        block_mut(&mut wide, k - relative_degree + j, base_words),
                        descriptor,
                        base_shape,
                        prefix(&top, base_words),
                        relation,
                        true,
                    );
                }
                j += 1;
            }
        }
        k -= 1;
    }
    let used = relative_degree * base_words;
    let characteristic = shape.characteristic();
    let mut t = 0;
    while t < used {
        out[t] = if subtract {
            sub_mod(out[t], wide[t], characteristic)
        } else {
            add_mod(out[t], wide[t], characteristic)
        };
        t += 1;
    }
}

/// `out += left * right`.
pub(super) const fn mul_add_assign(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
) {
    mul_acc(out, descriptor, shape, left, right, false);
}

/// `out -= left * right`.
pub(super) const fn sub_mul_assign(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
) {
    mul_acc(out, descriptor, shape, left, right, true);
}

/// Product of two occupied coefficients into `out`.
pub(super) const fn mul(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
) {
    zero(out, shape);
    mul_acc(out, descriptor, shape, left, right, false);
}

/// Degree of a packed polynomial over `shape`.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
const fn poly_degree(shape: Shape, polynomial: &[u64], from: usize) -> isize {
    let words = shape.words();
    let mut index = from as isize;
    while index >= 0 {
        if !is_zero(shape, block(polynomial, index as usize, words)) {
            return index;
        }
        index -= 1;
    }
    -1
}

/// Quotient and remainder of two packed polynomials.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
const fn poly_divmod(
    quotient: &mut [u64],
    remainder: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    left: &[u64],
    right: &[u64],
    top: usize,
) {
    let words = shape.words();
    let count = top + 1;
    copy_blocks(remainder, left, shape, count);
    zero_blocks(quotient, shape, count);
    let divisor_degree = poly_degree(shape, right, top);
    assert!(divisor_degree >= 0, "division by zero polynomial");
    let divisor_degree = divisor_degree as usize;
    let mut leading_inverse = [0u64; COORDINATES];
    inverse(
        prefix_mut(&mut leading_inverse, words),
        descriptor,
        shape,
        block(right, divisor_degree, words),
    );
    let mut remainder_degree = poly_degree(shape, remainder, top);
    while remainder_degree >= divisor_degree as isize {
        let degree = remainder_degree as usize;
        let mut factor = [0u64; COORDINATES];
        mul(
            prefix_mut(&mut factor, words),
            descriptor,
            shape,
            block(remainder, degree, words),
            prefix(&leading_inverse, words),
        );
        copy(
            block_mut(quotient, degree - divisor_degree, words),
            prefix(&factor, words),
            shape,
        );
        let mut j = 0;
        while j <= divisor_degree {
            sub_mul_assign(
                block_mut(remainder, degree - divisor_degree + j, words),
                descriptor,
                shape,
                prefix(&factor, words),
                block(right, j, words),
            );
            j += 1;
        }
        remainder_degree = if degree == 0 {
            -1
        } else {
            poly_degree(shape, remainder, degree - 1)
        };
    }
}

/// Inverse of a relation-node coefficient by its quadratic norm or polynomial Euclid.
#[allow(clippy::too_many_lines)]
const fn extension_inverse(out: &mut [u64], descriptor: &Descriptor, shape: Shape, value: &[u64]) {
    let (base, relative_degree, base_words, coefficients) = match shape.source() {
        Source::Node(index) => match descriptor.node(index) {
            Node::Extension {
                base,
                relative_degree,
                base_words,
                coefficients,
                ..
            } => (*base, *relative_degree, *base_words, coefficients),
            _ => panic!("unsupported coefficient shape"),
        },
        Source::Prime(_) => panic!("unsupported coefficient shape"),
    };
    let base_shape = Shape::of_node(descriptor, base);
    if relative_degree == 2
        && let Some(modulus) = base_shape.prime_modulus(descriptor)
    {
        let low = value[0];
        let high = value[1];
        let norm = add_mod(
            sub_mod(
                mul_mod(low, low, modulus),
                mul_mod(coefficients[1], mul_mod(low, high, modulus), modulus),
                modulus,
            ),
            mul_mod(coefficients[0], mul_mod(high, high, modulus), modulus),
            modulus,
        );
        let inverse = prime_inverse(modulus, norm);
        out[0] = mul_mod(
            sub_mod(low, mul_mod(coefficients[1], high, modulus), modulus),
            inverse,
            modulus,
        );
        out[1] = sub_mod(0, mul_mod(high, inverse, modulus), modulus);
        return;
    }
    let count = relative_degree + 1;
    let mut r0 = [0u64; SCRATCH];
    let mut r1 = [0u64; SCRATCH];
    let mut s0 = [0u64; SCRATCH];
    let mut s1 = [0u64; SCRATCH];
    let mut quotient = [0u64; SCRATCH];
    let mut remainder = [0u64; SCRATCH];
    let mut index = 0;
    while index < relative_degree {
        copy(
            block_mut(&mut r0, index, base_words),
            block(coefficients, index, base_words),
            base_shape,
        );
        copy(
            block_mut(&mut r1, index, base_words),
            block(value, index, base_words),
            base_shape,
        );
        index += 1;
    }
    identity(
        block_mut(&mut r0, relative_degree, base_words),
        descriptor,
        base_shape,
    );
    identity(block_mut(&mut s1, 0, base_words), descriptor, base_shape);
    while poly_degree(base_shape, &r1, relative_degree) >= 0 {
        poly_divmod(
            &mut quotient,
            &mut remainder,
            descriptor,
            base_shape,
            &r0,
            &r1,
            relative_degree,
        );
        copy_blocks(&mut r0, &r1, base_shape, count);
        copy_blocks(&mut r1, &remainder, base_shape, count);
        let quotient_len =
            (poly_degree(base_shape, &quotient, relative_degree) + 1).cast_unsigned();
        let cofactor_len = (poly_degree(base_shape, &s1, relative_degree) + 1).cast_unsigned();
        let mut product = [0u64; SCRATCH];
        let mut i = 0;
        while i < quotient_len {
            let mut j = 0;
            while j < cofactor_len {
                mul_add_assign(
                    block_mut(&mut product, i + j, base_words),
                    descriptor,
                    base_shape,
                    block(&quotient, i, base_words),
                    block(&s1, j, base_words),
                );
                j += 1;
            }
            i += 1;
        }
        let mut next = [0u64; SCRATCH];
        let mut block_index = 0;
        while block_index <= relative_degree {
            sub(
                block_mut(&mut next, block_index, base_words),
                descriptor,
                base_shape,
                block(&s0, block_index, base_words),
                block(&product, block_index, base_words),
            );
            block_index += 1;
        }
        copy_blocks(&mut s0, &s1, base_shape, count);
        copy_blocks(&mut s1, &next, base_shape, count);
    }
    let mut gcd_inverse = [0u64; COORDINATES];
    inverse(
        prefix_mut(&mut gcd_inverse, base_shape.words()),
        descriptor,
        base_shape,
        block(&r0, 0, base_words),
    );
    let mut index = 0;
    while index < relative_degree {
        mul(
            block_mut(out, index, base_words),
            descriptor,
            base_shape,
            block(&s0, index, base_words),
            prefix(&gcd_inverse, base_shape.words()),
        );
        index += 1;
    }
}

/// Mask the low `bits` bits without shifting a `u64` by 64.
const fn mask_low(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Inverse of one characteristic-two word, with compact relation-node
/// recursion: the quadratic norm stays specialized and higher relative
/// degrees use the occupied-slice Euclidean path.
#[allow(clippy::cast_possible_truncation)]
const fn binary_inverse(descriptor: &Descriptor, index: u8, value: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    match descriptor.node(index) {
        Node::Unused => panic!("unused node"),
        Node::Prime { modulus: 2, .. } => value,
        Node::Polynomial { .. } => word_inv(descriptor, index, value),
        Node::Basis {
            base,
            to_base,
            from_base,
        } => crate::field::binary::description::apply(
            *from_base,
            binary_inverse(
                descriptor,
                *base,
                crate::field::binary::description::apply(*to_base, value),
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
            if *relative_degree == 2 {
                let degree = *base_degree;
                let mask = mask_low(degree);
                let low = value & mask;
                let high = value >> degree;
                let norm = word_mul(descriptor, *base, low, low)
                    ^ word_mul(
                        descriptor,
                        *base,
                        coefficients[1],
                        word_mul(descriptor, *base, low, high),
                    )
                    ^ word_mul(
                        descriptor,
                        *base,
                        coefficients[0],
                        word_mul(descriptor, *base, high, high),
                    );
                let norm_inverse = binary_inverse(descriptor, *base, norm);
                (word_mul(
                    descriptor,
                    *base,
                    low ^ word_mul(descriptor, *base, coefficients[1], high),
                    norm_inverse,
                ) & mask)
                    | ((word_mul(descriptor, *base, high, norm_inverse) & mask) << degree)
            } else {
                let degree = *base_degree;
                let shape = Shape::of_node(descriptor, index);
                let mask = mask_low(degree);
                let mut input = [0u64; COORDINATES];
                let mut index_component = 0;
                while index_component < *relative_degree {
                    let shift = index_component as u32 * degree;
                    input[index_component] = (value >> shift) & mask;
                    index_component += 1;
                }
                let mut output = [0u64; COORDINATES];
                extension_inverse(&mut output, descriptor, shape, &input);
                let mut packed = 0u64;
                let mut index_component = 0;
                while index_component < *relative_degree {
                    let shift = index_component as u32 * degree;
                    packed |= (output[index_component] & mask) << shift;
                    index_component += 1;
                }
                packed
            }
        }
        // An odd prime leaf and an odd-characteristic relation node both
        // sit outside the word layer.
        Node::Prime { .. } | Node::Extension { .. } => panic!("word inversion on an odd node"),
    }
}

/// Inverse of an occupied coefficient into `out`; zero maps to zero.
pub(super) const fn inverse(out: &mut [u64], descriptor: &Descriptor, shape: Shape, value: &[u64]) {
    zero(out, shape);
    if is_zero(shape, value) {
        return;
    }
    if let Some(modulus) = shape.prime_modulus(descriptor) {
        out[0] = prime_inverse(modulus, value[0]);
        return;
    }
    match shape.source() {
        Source::Node(index) => {
            if shape.characteristic() == 2 {
                out[0] = binary_inverse(descriptor, index, value[0]);
            } else {
                extension_inverse(out, descriptor, shape, value);
            }
        }
        Source::Prime(_) => panic!("unsupported coefficient shape"),
    }
}

/// Inverse of a nonzero residue modulo an odd prime.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) const fn prime_inverse(modulus: u64, value: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    // Euclidean cofactors have magnitude at most the modulus. Their signed
    // updates fit i128 for every u64 modulus, without reducing each update.
    let prime = modulus as i128;
    let (mut remainder, mut next_remainder) = (prime, value as i128);
    let (mut cofactor, mut next_cofactor) = (0i128, 1i128);
    while next_remainder != 0 {
        let quotient = remainder / next_remainder;
        let step = remainder - quotient * next_remainder;
        let next = cofactor - quotient * next_cofactor;
        remainder = next_remainder;
        next_remainder = step;
        cofactor = next_cofactor;
        next_cofactor = next;
    }
    let reduced = cofactor % prime;
    if reduced < 0 {
        (reduced + prime) as u64
    } else {
        reduced as u64
    }
}
