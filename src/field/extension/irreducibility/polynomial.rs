//! Flat packed polynomials over one coefficient shape.
//!
//! Every operation addresses occupied blocks only and obtains buffer spans
//! from const `split_at`/`split_at_mut` helpers. Destination-first free
//! functions write their output; mutating operations carry an `_assign`
//! suffix. The relation passed to [`mulmod`], [`powmod`], and [`gcd_unit`]
//! includes its monic leading block.

use super::super::descriptor::{COORDINATES, Descriptor};
use super::coefficient::{
    SCRATCH, Shape, block, block_mut, copy, copy_blocks, identity, is_zero, mul, mul_add_assign,
    prefix, prefix_mut, prime_mulmod, scalar_mul, sub_assign, sub_mul_assign, zero, zero_blocks,
};

/// Degree of a packed polynomial, scanning from `from` down, or `-1` when
/// zero.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
#[must_use]
pub(super) const fn degree(
    descriptor: &Descriptor,
    shape: Shape,
    polynomial: &[u64],
    from: usize,
) -> isize {
    let _ = descriptor;
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

/// Multiply a packed polynomial by the indeterminate modulo the monic
/// relation `relation` of degree `degree`.
pub(super) const fn shift_once_assign(
    polynomial: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
) {
    let words = shape.words();
    let mut top = [0u64; COORDINATES];
    copy(
        prefix_mut(&mut top, words),
        block(polynomial, degree - 1, words),
        shape,
    );
    let mut temporary = [0u64; COORDINATES];
    let mut current = degree - 1;
    while current > 0 {
        copy(
            prefix_mut(&mut temporary, words),
            block(polynomial, current - 1, words),
            shape,
        );
        copy(
            block_mut(polynomial, current, words),
            prefix(&temporary, words),
            shape,
        );
        current -= 1;
    }
    zero(block_mut(polynomial, 0, words), shape);
    if !is_zero(shape, prefix(&top, words)) {
        let mut j = 0;
        while j < degree {
            sub_mul_assign(
                block_mut(polynomial, j, words),
                descriptor,
                shape,
                prefix(&top, words),
                block(relation, j, words),
            );
            j += 1;
        }
    }
}

/// Product of two packed polynomials reduced modulo the monic relation
/// `relation` of degree `degree`.
#[allow(clippy::cast_sign_loss)]
pub(super) const fn mulmod(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
    left: &[u64],
    right: &[u64],
) {
    if let Some(modulus) = shape.prime_modulus(descriptor) {
        prime_mulmod(out, modulus, relation, degree, left, right);
        return;
    }
    let words = shape.words();
    let left_degree = self::degree(descriptor, shape, left, degree - 1);
    let right_degree = self::degree(descriptor, shape, right, degree - 1);
    if left_degree < 0 || right_degree < 0 {
        zero_blocks(out, shape, degree);
        return;
    }
    let left_degree = left_degree as usize;
    let right_degree = right_degree as usize;
    let mut wide = [0u64; SCRATCH];
    let mut i = 0;
    while i <= left_degree {
        if !is_zero(shape, block(left, i, words)) {
            let mut j = 0;
            while j <= right_degree {
                if !is_zero(shape, block(right, j, words)) {
                    mul_add_assign(
                        block_mut(&mut wide, i + j, words),
                        descriptor,
                        shape,
                        block(left, i, words),
                        block(right, j, words),
                    );
                }
                j += 1;
            }
        }
        i += 1;
    }
    let mut k = 2 * degree - 2;
    while k >= degree {
        if !is_zero(shape, block(&wide, k, words)) {
            let mut top = [0u64; COORDINATES];
            copy(prefix_mut(&mut top, words), block(&wide, k, words), shape);
            let mut j = 0;
            while j < degree {
                sub_mul_assign(
                    block_mut(&mut wide, k - degree + j, words),
                    descriptor,
                    shape,
                    prefix(&top, words),
                    block(relation, j, words),
                );
                j += 1;
            }
        }
        k -= 1;
    }
    copy_blocks(out, &wide, shape, degree);
}

/// `base^exponent` modulo the monic relation `relation` of degree `degree`.
pub(super) const fn powmod(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
    base: &[u64],
    mut exponent: u128,
) {
    let words = shape.words();
    let mut result_buffer = [0u64; COORDINATES];
    let mut square_buffer = [0u64; COORDINATES];
    let mut product_buffer = [0u64; COORDINATES];
    let mut result = &mut result_buffer;
    let mut square = &mut square_buffer;
    let mut product = &mut product_buffer;
    identity(prefix_mut(result, words), descriptor, shape);
    copy_blocks(square, base, shape, degree);
    let prime = shape.prime_modulus(descriptor);
    while exponent > 0 {
        if exponent & 1 == 1 {
            match prime {
                Some(modulus) => prime_mulmod(product, modulus, relation, degree, result, square),
                None => mulmod(product, descriptor, shape, relation, degree, result, square),
            }
            (result, product) = (product, result);
        }
        exponent >>= 1;
        if exponent > 0 {
            match prime {
                Some(modulus) => prime_mulmod(product, modulus, relation, degree, square, square),
                None => mulmod(product, descriptor, shape, relation, degree, square, square),
            }
            (square, product) = (product, square);
        }
    }
    copy_blocks(out, result, shape, degree);
}

/// Formal derivative of the monic relation, without its leading term.
pub(super) const fn derivative(
    out: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
) {
    let words = shape.words();
    let characteristic = shape.characteristic();
    let mut i = 0;
    while i < degree {
        let factor = (i as u64 + 1) % characteristic;
        if factor == 0 {
            zero(block_mut(out, i, words), shape);
        } else if characteristic == 2 {
            copy(
                block_mut(out, i, words),
                block(relation, i + 1, words),
                shape,
            );
        } else {
            scalar_mul(
                block_mut(out, i, words),
                shape,
                block(relation, i + 1, words),
                factor,
            );
        }
        i += 1;
    }
    let _ = descriptor;
}

/// Whether `gcd(relation, other)` is a unit.
#[allow(clippy::cast_sign_loss)]
pub(super) const fn gcd_unit(
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
    other: &[u64],
) -> bool {
    let words = shape.words();
    let mut left_buffer = [0u64; SCRATCH];
    let mut right_buffer = [0u64; SCRATCH];
    let mut left = &mut left_buffer;
    let mut right = &mut right_buffer;
    copy_blocks(left, relation, shape, degree + 1);
    copy_blocks(right, other, shape, degree);
    let mut left_degree = self::degree(descriptor, shape, left, degree);
    let mut right_degree = self::degree(descriptor, shape, right, degree - 1);
    if right_degree < 0 {
        return false;
    }
    while right_degree > 0 {
        let divisor_degree = right_degree as usize;
        let mut inverse = [0u64; COORDINATES];
        super::coefficient::inverse(
            prefix_mut(&mut inverse, words),
            descriptor,
            shape,
            block(right, divisor_degree, words),
        );
        let mut current = left_degree;
        while current >= right_degree {
            let index = current as usize;
            if !is_zero(shape, block(left, index, words)) {
                let mut factor = [0u64; COORDINATES];
                mul(
                    prefix_mut(&mut factor, words),
                    descriptor,
                    shape,
                    block(left, index, words),
                    prefix(&inverse, words),
                );
                let mut j = 0;
                while j <= divisor_degree {
                    sub_mul_assign(
                        block_mut(left, index - divisor_degree + j, words),
                        descriptor,
                        shape,
                        prefix(&factor, words),
                        block(right, j, words),
                    );
                    j += 1;
                }
            }
            current -= 1;
            while current >= 0 {
                if !is_zero(shape, block(left, current as usize, words)) {
                    break;
                }
                current -= 1;
            }
        }
        if current < 0 {
            return false;
        }
        if current == 0 {
            return true;
        }
        (left, right) = (right, left);
        left_degree = right_degree;
        right_degree = current;
    }
    true
}

/// Subtract the identity from coefficient `position`.
pub(super) const fn clear_identity(
    polynomial: &mut [u64],
    descriptor: &Descriptor,
    shape: Shape,
    position: usize,
) {
    let words = shape.words();
    let mut one = [0u64; COORDINATES];
    identity(prefix_mut(&mut one, words), descriptor, shape);
    sub_assign(
        block_mut(polynomial, position, words),
        descriptor,
        shape,
        prefix(&one, words),
    );
}
