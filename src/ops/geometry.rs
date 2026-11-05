//! Shared buffer-geometry validation for the row operations.

use crate::field::FieldBuffer;
use crate::field::Validate;

/// Asserts a buffer holds a whole number of field elements.
///
/// The row's central const validation runs first, before the byte width is
/// read, so every operation sharing this check rejects an invalid row at
/// compile time whatever geometry its buffers hold.
#[inline]
pub(super) fn check_width<F: FieldBuffer>(name: &str, len: usize) {
    let () = Validate::<F>::OK;
    assert!(
        len.is_multiple_of(F::BYTES),
        "{name}: buffer of {len} bytes is not a whole number of {} elements",
        F::NAME
    );
}

/// Asserts two named operand lengths agree, then checks the element width.
#[inline]
pub(super) fn check_pair<F: FieldBuffer>(
    name: &str,
    left_name: &str,
    left: usize,
    right_name: &str,
    right: usize,
) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
    check_width::<F>(name, left);
}
