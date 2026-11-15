//! Normalized streaming Berlekamp rank over an arbitrary coefficient shape.
//!
//! The relation is gated square-free first, then the `q`-Frobenius columns
//! `t^(j*q) - t^j` are streamed against an online reduced-row-echelon basis.
//! The first dependent column proves a rank below `d - 1`. All storage is a
//! flat occupied prefix: `d * words` coefficient words per column and
//! `d * d * words` for the basis.

use super::super::descriptor::{COORDINATES, Descriptor};
use super::coefficient::{
    SCRATCH, Shape, block, block_mut, copy, copy_blocks, identity, inverse, is_zero, mul, prefix,
    prefix_mut, sub_mul_assign,
};
use super::polynomial::{clear_identity, derivative, gcd_unit, mulmod, powmod, shift_once_assign};

/// Capacity of the flat echelon basis.
const RANK: usize = COORDINATES * COORDINATES;

/// Whether the relation is irreducible by normalized streaming rank.
// Const validation keeps the bounded echelon workspace on the stack.
#[allow(clippy::large_stack_arrays)]
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(super) const fn normalized(
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
) -> bool {
    let words = shape.words();
    let used = degree * words;
    let mut monic = [0u64; SCRATCH];
    copy_blocks(&mut monic, relation, shape, degree);
    identity(block_mut(&mut monic, degree, words), descriptor, shape);
    let mut formal_derivative = [0u64; COORDINATES];
    derivative(
        prefix_mut(&mut formal_derivative, used),
        descriptor,
        shape,
        &monic,
        degree,
    );
    if !gcd_unit(
        descriptor,
        shape,
        &monic,
        degree,
        prefix(&formal_derivative, used),
    ) {
        return false;
    }
    let mut one = [0u64; COORDINATES];
    identity(prefix_mut(&mut one, words), descriptor, shape);
    let order = shape.order(descriptor);
    let mut frobenius = [0u64; COORDINATES];
    if order >= degree as u128 {
        let mut t = [0u64; COORDINATES];
        identity(block_mut(&mut t, 1, words), descriptor, shape);
        powmod(
            prefix_mut(&mut frobenius, used),
            descriptor,
            shape,
            &monic,
            degree,
            prefix(&t, used),
            order,
        );
    }
    let mut basis = [0u64; RANK];
    let mut high = [0usize; COORDINATES];
    let mut occupied = [false; COORDINATES];
    let mut column = [0u64; COORDINATES];
    identity(block_mut(&mut column, 0, words), descriptor, shape);
    let mut value = [0u64; COORDINATES];
    let mut product = [0u64; COORDINATES];
    let mut j = 1;
    while j < degree {
        if order < degree as u128 {
            let steps = order as usize;
            let mut step = 0;
            while step < steps {
                shift_once_assign(&mut column, descriptor, shape, &monic, degree);
                step += 1;
            }
        } else {
            mulmod(
                prefix_mut(&mut product, used),
                descriptor,
                shape,
                &monic,
                degree,
                prefix(&column, used),
                prefix(&frobenius, used),
            );
            copy_blocks(&mut column, &product, shape, degree);
        }
        copy_blocks(&mut value, &column, shape, degree);
        clear_identity(&mut value, descriptor, shape, j);
        let mut end = degree;
        while end > 0 && is_zero(shape, block(&value, end - 1, words)) {
            end -= 1;
        }
        let mut pivot = 0;
        let mut found = false;
        while pivot < end {
            let mut remainder = [0u64; COORDINATES];
            copy(
                prefix_mut(&mut remainder, words),
                block(&value, pivot, words),
                shape,
            );
            if !is_zero(shape, prefix(&remainder, words)) {
                if occupied[pivot] {
                    let row_high = high[pivot];
                    if row_high > end {
                        end = row_high;
                    }
                    let mut k = pivot + 1;
                    while k < row_high {
                        let basis_index = pivot * degree + k;
                        if !is_zero(shape, block(&basis, basis_index, words)) {
                            sub_mul_assign(
                                block_mut(&mut value, k, words),
                                descriptor,
                                shape,
                                prefix(&remainder, words),
                                block(&basis, basis_index, words),
                            );
                        }
                        k += 1;
                    }
                } else {
                    let mut inverse_value = [0u64; COORDINATES];
                    inverse(
                        prefix_mut(&mut inverse_value, words),
                        descriptor,
                        shape,
                        prefix(&remainder, words),
                    );
                    let pivot_index = pivot * degree + pivot;
                    copy(
                        block_mut(&mut basis, pivot_index, words),
                        prefix(&one, words),
                        shape,
                    );
                    let mut k = pivot + 1;
                    while k < end {
                        let basis_index = pivot * degree + k;
                        if is_zero(shape, block(&value, k, words)) {
                            let mut zero_word = 0;
                            let target = block_mut(&mut basis, basis_index, words);
                            while zero_word < words {
                                target[zero_word] = 0;
                                zero_word += 1;
                            }
                        } else {
                            mul(
                                block_mut(&mut basis, basis_index, words),
                                descriptor,
                                shape,
                                block(&value, k, words),
                                prefix(&inverse_value, words),
                            );
                        }
                        k += 1;
                    }
                    let mut row_high = end;
                    while row_high > pivot + 1
                        && is_zero(shape, block(&basis, pivot * degree + row_high - 1, words))
                    {
                        row_high -= 1;
                    }
                    high[pivot] = row_high;
                    occupied[pivot] = true;
                    found = true;
                    break;
                }
            }
            pivot += 1;
        }
        if !found {
            return false;
        }
        j += 1;
    }
    true
}
