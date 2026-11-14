//! Odd-prime relative degrees at least six.
//!
//! The square-free Euclidean pass accumulates the resultant, including swap
//! signs, leading-coefficient powers, and degree drops. The odd-parity
//! discriminant gate is a rejection filter only: passing it never proves
//! irreducibility. Only a passing relation reaches the normalized skyline
//! rank, which uses support shifts when the prime is below the relation
//! degree, zero-guarded reductions, row skylines, and bounded lazy `u128`
//! sums when the prime bound allows them.

use super::super::descriptor::{COORDINATES, Descriptor};
use super::coefficient::{Shape, prefix, prefix_mut};
use super::polynomial::{mulmod, powmod};

/// Capacity of the flat echelon basis.
const RANK: usize = COORDINATES * COORDINATES;

/// `left - right` modulo an odd prime.
const fn sub_mod(left: u64, right: u64, modulus: u64) -> u64 {
    if left >= right {
        left - right
    } else {
        left.wrapping_add(modulus).wrapping_sub(right)
    }
}

/// `left * right` modulo an odd prime.
#[allow(clippy::cast_possible_truncation)]
const fn mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    (left as u128 * right as u128 % modulus as u128) as u64
}

/// `base^exponent` modulo an odd prime.
const fn pow_mod(base: u64, mut exponent: u128, modulus: u64) -> u64 {
    let mut result = 1;
    let mut square = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, square, modulus);
        }
        exponent >>= 1;
        if exponent > 0 {
            square = mul_mod(square, square, modulus);
        }
    }
    result
}

/// Inverse of a nonzero residue modulo an odd prime.
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn inverse_mod(value: u64, modulus: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    let prime = modulus as u128;
    let (mut remainder, mut next_remainder) = (prime, value as u128);
    let (mut cofactor, mut next_cofactor) = (0u128, 1u128);
    while next_remainder != 0 {
        let quotient = remainder / next_remainder;
        let step = remainder - quotient * next_remainder;
        let next = (cofactor + prime - quotient * next_cofactor % prime) % prime;
        remainder = next_remainder;
        next_remainder = step;
        cofactor = next_cofactor;
        next_cofactor = next;
    }
    cofactor as u64
}

/// Formal derivative of a monic prime relation, without its leading term.
const fn derivative(out: &mut [u64], relation: &[u64], degree: usize, modulus: u64) {
    let mut i = 0;
    while i < degree {
        out[i] = mul_mod((i as u64 + 1) % modulus, relation[i + 1], modulus);
        i += 1;
    }
}

/// Degree of a scalar polynomial, scanning from `from` down.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
const fn degree(polynomial: &[u64], from: usize) -> isize {
    let mut index = from as isize;
    while index >= 0 {
        if polynomial[index as usize] != 0 {
            return index;
        }
        index -= 1;
    }
    -1
}

/// Copy the occupied prefix of a scalar polynomial into `to`.
const fn copy_polynomial(to: &mut [u64], from: &[u64], count: usize) {
    let mut k = 0;
    while k < count {
        to[k] = from[k];
        k += 1;
    }
}

/// Resultant of a monic polynomial and its derivative by the Euclidean
/// recurrence, with the square-free unit flag accumulated in the same pass.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
const fn resultant_parts(
    relation: &[u64],
    formal_derivative: &[u64],
    degree: usize,
    modulus: u64,
) -> (bool, u64) {
    let mut current = [0u64; COORDINATES + 2];
    let mut divisor = [0u64; COORDINATES + 2];
    let mut remainder = [0u64; COORDINATES + 2];
    copy_polynomial(&mut current, relation, degree + 1);
    copy_polynomial(&mut divisor, formal_derivative, degree);
    let mut current_degree = degree as isize;
    let mut divisor_degree = self::degree(&divisor, degree - 1);
    let mut result = 1;
    if divisor_degree < 0 {
        return (false, 0);
    }
    while divisor_degree > 0 {
        let lead = divisor[divisor_degree as usize];
        let lead_inverse = inverse_mod(lead, modulus);
        copy_polynomial(&mut remainder, &current, degree + 1);
        let mut top = current_degree;
        while top >= divisor_degree {
            let value = remainder[top as usize];
            if value != 0 {
                let factor = mul_mod(value, lead_inverse, modulus);
                let mut j = 0;
                while j <= divisor_degree as usize {
                    let target = top as usize - divisor_degree as usize + j;
                    let product = mul_mod(factor, divisor[j], modulus);
                    remainder[target] = sub_mod(remainder[target], product, modulus);
                    j += 1;
                }
            }
            top -= 1;
        }
        let next = self::degree(&remainder, divisor_degree as usize - 1);
        if next < 0 {
            return (false, 0);
        }
        let power = current_degree - next;
        if (current_degree * divisor_degree) & 1 != 0 {
            result = sub_mod(0, result, modulus);
        }
        result = mul_mod(result, pow_mod(lead, power as u128, modulus), modulus);
        copy_polynomial(&mut current, &divisor, degree + 1);
        copy_polynomial(&mut divisor, &remainder, degree + 1);
        current_degree = divisor_degree;
        divisor_degree = next;
    }
    let constant = divisor[0];
    if constant == 0 {
        return (false, 0);
    }
    result = mul_mod(
        result,
        pow_mod(constant, current_degree as u128, modulus),
        modulus,
    );
    (true, result)
}

/// `Res(f, f')` for a monic prime relation.
#[must_use]
pub(super) const fn resultant(relation: &[u64], degree: usize, modulus: u64) -> u64 {
    let mut formal_derivative = [0u64; COORDINATES];
    derivative(
        prefix_mut(&mut formal_derivative, degree),
        relation,
        degree,
        modulus,
    );
    resultant_parts(
        relation,
        prefix(&formal_derivative, degree),
        degree,
        modulus,
    )
    .1
}

/// The odd-parity discriminant rejection filter.
///
/// `true` means "not rejected"; it is never an irreducibility proof.
#[must_use]
pub(super) const fn discriminant_gate(relation: &[u64], degree: usize, modulus: u64) -> bool {
    let mut discriminant = resultant(relation, degree, modulus);
    if (degree * (degree - 1) / 2) & 1 != 0 {
        discriminant = sub_mod(0, discriminant, modulus);
    }
    if discriminant == 0 {
        return false;
    }
    let expected = if degree & 1 == 0 {
        sub_mod(0, 1, modulus)
    } else {
        1
    };
    pow_mod(discriminant, (modulus as u128 - 1) / 2, modulus) == expected
}

/// Whether the bounded lazy `u128` accumulation is licensed.
#[must_use]
pub(super) const fn wide_safe(modulus: u64, degree: usize) -> bool {
    let pm1 = (modulus - 1) as u128;
    match pm1.checked_mul(pm1) {
        Some(square) => match (degree as u128).checked_mul(square) {
            Some(value) => value.checked_add(pm1).is_some(),
            None => false,
        },
        None => false,
    }
}

/// Multiply a scalar column by the indeterminate and fold its support.
const fn shift_support_assign(
    column: &mut [u64],
    relation: &[u64],
    degree: usize,
    support: &[usize],
    support_len: usize,
    modulus: u64,
) {
    let top = column[degree - 1];
    let mut k = degree - 1;
    while k > 0 {
        column[k] = column[k - 1];
        k -= 1;
    }
    column[0] = 0;
    if top != 0 {
        let mut j = 0;
        while j < support_len {
            let position = support[j];
            let product = mul_mod(top, relation[position], modulus);
            column[position] = sub_mod(column[position], product, modulus);
            j += 1;
        }
    }
}

/// Normalized skyline rank for an odd prime relation.
// Const validation keeps the bounded echelon workspace on the stack.
#[allow(clippy::large_stack_arrays)]
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
const fn rank(descriptor: &Descriptor, relation: &[u64], degree: usize, modulus: u64) -> bool {
    let shape = Shape::prime(modulus);
    let mut support = [0usize; COORDINATES];
    let mut support_len = 0;
    let mut k = 0;
    while k < degree {
        if relation[k] != 0 {
            support[support_len] = k;
            support_len += 1;
        }
        k += 1;
    }
    let order = modulus as u128;
    let mut frobenius = [0u64; COORDINATES];
    if order >= degree as u128 {
        let mut t = [0u64; COORDINATES];
        t[1] = 1;
        powmod(
            prefix_mut(&mut frobenius, degree),
            descriptor,
            shape,
            relation,
            degree,
            prefix(&t, degree),
            order,
        );
    }
    let lazy = wide_safe(modulus, degree);
    let mut basis = [0u64; RANK];
    let mut high = [0usize; COORDINATES];
    let mut occupied = [false; COORDINATES];
    let mut column = [0u64; COORDINATES];
    column[0] = 1;
    let mut value = [0u64; COORDINATES];
    let mut product = [0u64; COORDINATES];
    let mut j = 1;
    while j < degree {
        if order < degree as u128 {
            let steps = order as usize;
            let mut step = 0;
            while step < steps {
                shift_support_assign(
                    &mut column,
                    relation,
                    degree,
                    &support,
                    support_len,
                    modulus,
                );
                step += 1;
            }
        } else {
            mulmod(
                prefix_mut(&mut product, degree),
                descriptor,
                shape,
                relation,
                degree,
                prefix(&column, degree),
                prefix(&frobenius, degree),
            );
            copy_polynomial(&mut column, &product, degree);
        }
        copy_polynomial(&mut value, &column, degree);
        value[j] = sub_mod(value[j], 1, modulus);
        let mut end = degree;
        while end > 0 && value[end - 1] == 0 {
            end -= 1;
        }
        let mut acc = [0u128; COORDINATES];
        if lazy {
            let mut l = 0;
            while l < end {
                acc[l] = value[l] as u128;
                l += 1;
            }
        }
        let mut pivot = 0;
        let mut found = false;
        while pivot < end {
            let remainder = if lazy {
                if acc[pivot] == 0 {
                    0
                } else {
                    (acc[pivot] % modulus as u128) as u64
                }
            } else {
                value[pivot]
            };
            if remainder != 0 {
                if occupied[pivot] {
                    let row_high = high[pivot];
                    if row_high > end {
                        end = row_high;
                    }
                    if lazy {
                        let factor = (modulus - remainder) as u128;
                        let mut l = pivot + 1;
                        while l < row_high {
                            let entry = basis[pivot * degree + l];
                            if entry != 0 {
                                acc[l] += factor * entry as u128;
                            }
                            l += 1;
                        }
                    } else {
                        let mut l = pivot + 1;
                        while l < row_high {
                            let entry = basis[pivot * degree + l];
                            if entry != 0 {
                                value[l] =
                                    sub_mod(value[l], mul_mod(remainder, entry, modulus), modulus);
                            }
                            l += 1;
                        }
                    }
                } else {
                    let inverse = inverse_mod(remainder, modulus);
                    basis[pivot * degree + pivot] = 1;
                    let mut l = pivot + 1;
                    while l < end {
                        let current = if lazy {
                            if acc[l] == 0 {
                                0
                            } else {
                                (acc[l] % modulus as u128) as u64
                            }
                        } else {
                            value[l]
                        };
                        basis[pivot * degree + l] = if current == 0 {
                            0
                        } else {
                            mul_mod(current, inverse, modulus)
                        };
                        l += 1;
                    }
                    let mut row_high = end;
                    while row_high > pivot + 1 && basis[pivot * degree + row_high - 1] == 0 {
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

/// Odd prime base at relative degree at least six: gate, then rank.
pub(super) const fn decide(
    descriptor: &Descriptor,
    modulus: u64,
    degree: usize,
    relation: &[u64],
) -> bool {
    let mut monic = [0u64; COORDINATES + 2];
    copy_polynomial(&mut monic, relation, degree);
    monic[degree] = 1;
    if !discriminant_gate(prefix(&monic, degree + 1), degree, modulus) {
        return false;
    }
    rank(descriptor, prefix(&monic, degree + 1), degree, modulus)
}
