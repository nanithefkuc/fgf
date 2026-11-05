//! Byte geometry: buffer sizing and the shared buffer checks.

/// Number of bytes needed to hold `bits` packed elements.
///
/// ```text
/// bits  0  1  8  9 64 65
/// bytes 0  1  1  2  8  9
/// ```
#[must_use]
pub const fn bytes_for(bits: usize) -> usize {
    bits.div_ceil(8)
}

/// Asserts two named operand lengths agree.
#[inline]
pub(super) fn check_pair(name: &str, left_name: &str, left: usize, right_name: &str, right: usize) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
}

/// Asserts a named operand holds at least the bytes `bits` needs.
#[inline]
pub(super) fn check_holds(name: &str, operand: &str, len: usize, bits: usize) {
    assert!(
        bytes_for(bits) <= len,
        "{name}: {bits} bits need at least {} bytes but {operand} holds {len}",
        bytes_for(bits)
    );
}
