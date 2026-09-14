//! Direct scalar arithmetic for the canonical Wiedemann binary tower.
//!
//! The tower starts at GF(2). At each extension, an element is `a + b*X`
//! over the direct subfield and the new generator satisfies
//!
//! ```text
//! X^2 = 1 + x*X,
//! ```
//!
//! where `x` is the direct subfield's top-level generator. Raw integers store
//! the constant component in the low half and the `X` component in the high
//! half.
//!
//! This module follows the defining schoolbook recurrence directly. It has no
//! tables, Karatsuba step, prepared coefficients, or vector operations, so it
//! is the independent oracle for the optimized Fan–Paar implementation of the
//! same representation.

/// Returns the low `bits` bits set.
const fn low_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Checks that `bits` names a level of the tower representable in `u64`.
const fn assert_tower_width(bits: u32) {
    assert!(bits.is_power_of_two() && bits <= 64);
}

/// Multiplies by the top-level generator of a tower level.
///
/// `value` uses the low `bits` bits of the canonical tower representation.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than the width of `u64`.
#[must_use]
pub const fn mul_generator(value: u64, bits: u32) -> u64 {
    assert_tower_width(bits);
    if bits == 1 {
        return value & 1;
    }

    let half = bits / 2;
    let mask = low_mask(half);
    let constant = value & mask;
    let coefficient = (value >> half) & mask;

    coefficient | ((constant ^ mul_generator(coefficient, half)) << half)
}

/// Multiplies two canonical Wiedemann tower elements.
///
/// Each operand uses the low `bits` bits of the recursive tower basis. The
/// result uses the same representation.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than the width of `u64`.
#[must_use]
pub const fn multiply(lhs: u64, rhs: u64, bits: u32) -> u64 {
    assert_tower_width(bits);
    if bits == 1 {
        return lhs & rhs & 1;
    }

    let half = bits / 2;
    let mask = low_mask(half);
    let lhs_constant = lhs & mask;
    let lhs_coefficient = (lhs >> half) & mask;
    let rhs_constant = rhs & mask;
    let rhs_coefficient = (rhs >> half) & mask;

    let constant_product = multiply(lhs_constant, rhs_constant, half);
    let outer_product = multiply(lhs_constant, rhs_coefficient, half);
    let inner_product = multiply(lhs_coefficient, rhs_constant, half);
    let coefficient_product = multiply(lhs_coefficient, rhs_coefficient, half);

    let constant = constant_product ^ coefficient_product;
    let coefficient = outer_product ^ inner_product ^ mul_generator(coefficient_product, half);
    constant | (coefficient << half)
}

/// Squares a canonical Wiedemann tower element by direct multiplication.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than the width of `u64`.
#[must_use]
pub const fn square(value: u64, bits: u32) -> u64 {
    multiply(value, value, bits)
}

/// Inverts a canonical Wiedemann tower element by Fermat exponentiation.
///
/// Zero maps to zero. The routine deliberately uses [`multiply`] and
/// [`square`] rather than the tower norm recurrence.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than the width of `u64`.
#[must_use]
pub const fn invert(value: u64, bits: u32) -> u64 {
    assert_tower_width(bits);
    if value & low_mask(bits) == 0 {
        return 0;
    }
    if bits == 1 {
        return 1;
    }

    let mut exponent = if bits == 64 {
        u64::MAX - 1
    } else {
        (1u64 << bits) - 2
    };
    let mut base = value & low_mask(bits);
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = multiply(result, base, bits);
        }
        base = square(base, bits);
        exponent >>= 1;
    }
    result
}

fn decode(bytes: &[u8]) -> u64 {
    let mut encoded = [0u8; 8];
    encoded[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(encoded)
}

fn encode(bytes: &mut [u8], value: u64) {
    bytes.copy_from_slice(&value.to_le_bytes()[..bytes.len()]);
}

fn element_bytes(bits: u32) -> usize {
    assert!(matches!(bits, 8 | 16 | 32 | 64));
    usize::try_from(bits / 8).expect("tower width fits usize")
}

/// Accumulates a fixed-coefficient product over packed Wiedemann elements.
///
/// `dst` and `src` contain equal-length sequences of canonical little-endian
/// elements at the tower level selected by `bits`.
///
/// # Panics
///
/// Panics unless `bits` is a whole-byte tower width, the slices have equal
/// lengths, and their lengths are a multiple of one element.
pub fn mul_add(dst: &mut [u8], coefficient: u64, src: &[u8], bits: u32) {
    let bytes = element_bytes(bits);
    assert_eq!(dst.len(), src.len());
    assert_eq!(dst.len() % bytes, 0);

    for (dst_element, src_element) in dst.chunks_exact_mut(bytes).zip(src.chunks_exact(bytes)) {
        let value = decode(dst_element) ^ multiply(decode(src_element), coefficient, bits);
        encode(dst_element, value);
    }
}

/// Scales packed Wiedemann elements in place.
///
/// `dst` contains canonical little-endian elements at the tower level selected
/// by `bits`.
///
/// # Panics
///
/// Panics unless `bits` is a whole-byte tower width and the slice length is a
/// multiple of one element.
pub fn mul_assign(dst: &mut [u8], coefficient: u64, bits: u32) {
    let bytes = element_bytes(bits);
    assert_eq!(dst.len() % bytes, 0);

    for element in dst.chunks_exact_mut(bytes) {
        let value = multiply(decode(element), coefficient, bits);
        encode(element, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_basis_vectors_match() {
        assert_eq!(multiply(0b10, 0b11, 2), 0b01);
        assert_eq!(square(0x10, 8), 0x41);
        assert_eq!(invert(0b0100, 4), 0b0110);
        assert_eq!(invert(0x10, 8), 0x14);
    }

    #[test]
    fn every_level_satisfies_its_defining_quadratic() {
        for bits in [2, 4, 8, 16, 32, 64] {
            let half = bits / 2;
            let generator = 1u64 << half;
            let subfield_generator = if half == 1 { 1 } else { 1u64 << (half / 2) };
            assert_eq!(square(generator, bits), 1 | (subfield_generator << half));
        }
    }
}
