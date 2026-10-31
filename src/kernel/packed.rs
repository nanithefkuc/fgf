//! Packed binary multiplication through coordinate-linear and product kernels.
//!
//! Existing specialized tower kernels retain their routes. These entries serve
//! the remaining packed rows without enlarging their prepared coefficients;
//! coefficient maps are bounded per-execution stack state, not prepared values.

use crate::field::Field;
use crate::kernel::Backend;

/// Compatible serving tier of the shared packed kernels.
pub(crate) fn serving_backend() -> Backend {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match super::backend() {
        Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => return Backend::V3,
        Backend::V2 => return Backend::V2,
        _ => {}
    }
    Backend::Scalar
}

/// Smallest row containing a shared vector product body.
pub(crate) fn elementwise_min_bytes() -> usize {
    if serving_backend() == Backend::Scalar {
        0
    } else {
        16
    }
}

/// Destination disposition of a fixed-coefficient row.
#[derive(Clone, Copy)]
pub(crate) enum Fixed<'a> {
    /// Accumulates the transformed source.
    Add(&'a [u8]),
    /// Overwrites with the transformed source.
    Into(&'a [u8]),
    /// Transforms the destination in place.
    Assign,
}

/// Runs a fixed product when a compatible vector body can execute.
///
/// Geometry is checked by the public operation. A false result leaves the
/// destination untouched and asks the caller to retain its scalar row body.
pub(crate) fn try_fixed<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    coefficient: u64,
    operation: Fixed<'_>,
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use super::x86::binary_linear;
        use crate::field::extension::arithmetic::word_one;

        let tier = serving_backend();
        if tier == Backend::Scalar || dst.len() < 16 {
            return false;
        }
        if coefficient == 0 {
            if !matches!(operation, Fixed::Add(_)) {
                dst.fill(0);
            }
            return true;
        }
        if coefficient == word_one(F::DESCRIPTOR, F::DESCRIPTOR.root_index()) {
            match operation {
                Fixed::Add(src) => super::xor(dst, src),
                Fixed::Into(src) => dst.copy_from_slice(src),
                Fixed::Assign => {}
            }
            return true;
        }
        #[cfg(test)]
        let _general = matches!(operation, Fixed::Add(_))
            .then(|| super::trace::term(super::trace::Term::General));
        let map = binary_linear::multiplication(F::DESCRIPTOR, coefficient);
        if tier == Backend::V3 && dst.len() >= 32 {
            let token = super::x86_v3_token();
            match operation {
                Fixed::Add(src) => binary_linear::mul_add_avx2::<BYTES>(token, dst, &map, src),
                Fixed::Into(src) => binary_linear::mul_into_avx2::<BYTES>(token, dst, &map, src),
                Fixed::Assign => binary_linear::mul_assign_avx2::<BYTES>(token, dst, &map),
            }
        } else {
            let token = super::x86_v2_token();
            match operation {
                Fixed::Add(src) => binary_linear::mul_add_sse41::<BYTES>(token, dst, &map, src),
                Fixed::Into(src) => binary_linear::mul_into_sse41::<BYTES>(token, dst, &map, src),
                Fixed::Assign => binary_linear::mul_assign_sse41::<BYTES>(token, dst, &map),
            }
        }
        true
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        let _ = (core::marker::PhantomData::<F>, dst, coefficient);
        match operation {
            Fixed::Add(src) | Fixed::Into(src) => {
                let _ = src;
            }
            Fixed::Assign => {}
        }
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    false
}

/// Runs a varying product when a compatible vector body can execute.
pub(crate) fn try_elementwise<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use super::x86::binary_product::{avx2, sse41};
        match serving_backend() {
            Backend::V3 if dst.len() >= 32 => {
                avx2::mul_elementwise_avx2::<BYTES>(
                    super::x86_v3_token(),
                    F::DESCRIPTOR,
                    dst,
                    a,
                    b,
                );
                return true;
            }
            Backend::V3 | Backend::V2 if dst.len() >= 16 => {
                sse41::mul_elementwise_sse41::<BYTES>(
                    super::x86_v2_token(),
                    F::DESCRIPTOR,
                    dst,
                    a,
                    b,
                );
                return true;
            }
            _ => {}
        }
    }
    let _ = (core::marker::PhantomData::<F>, dst, a, b);
    false
}

/// Runs an in-place varying product without staging a destination copy.
pub(crate) fn try_elementwise_assign<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    src: &[u8],
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use super::x86::binary_product::{avx2, sse41};
        match serving_backend() {
            Backend::V3 if dst.len() >= 32 => {
                avx2::mul_elementwise_assign_avx2::<BYTES>(
                    super::x86_v3_token(),
                    F::DESCRIPTOR,
                    dst,
                    src,
                );
                return true;
            }
            Backend::V3 | Backend::V2 if dst.len() >= 16 => {
                sse41::mul_elementwise_assign_sse41::<BYTES>(
                    super::x86_v2_token(),
                    F::DESCRIPTOR,
                    dst,
                    src,
                );
                return true;
            }
            _ => {}
        }
    }
    let _ = (core::marker::PhantomData::<F>, dst, src);
    false
}
