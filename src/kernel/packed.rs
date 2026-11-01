//! Packed binary multiplication through coordinate-linear and product kernels.
//!
//! Existing specialized tower kernels retain their routes. These entries serve
//! the remaining packed rows without enlarging their prepared coefficients;
//! coefficient maps are bounded per-execution stack state, not prepared values.
//!
//! Each entry serves a whole-vector prefix of the row and returns its length
//! in bytes. The caller finishes the remaining bytes with the field's own
//! scalar body. A prefix shorter than the entry's measured floor is not served
//! at all; the floors are the crossovers recorded in `BENCHMARKS.md`.

use crate::field::Field;
use crate::kernel::Backend;

/// Minimum row size accepted by the shared varying-product entries.
pub(crate) const PRODUCT_MIN_BYTES: usize = 64;

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

/// Vector width in bytes of a serving tier.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const fn vector_bytes(tier: Backend) -> usize {
    match tier {
        Backend::V2 => 16,
        _ => 32,
    }
}

/// Smallest fixed-coefficient prefix, in elements, that the linear kernels
/// serve for one element width and tier.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const fn fixed_floor_elements(bytes: usize, tier: Backend) -> usize {
    match (bytes, tier) {
        (4 | 8, Backend::V2) => 128,
        _ => 64,
    }
}

/// Whether the varying-product kernels serve this element width.
///
/// The shared AVX2 body serves two-byte rows; other widths use portable products.
fn product_serves(bytes: usize) -> bool {
    bytes == 2 && serving_backend() == Backend::V3
}

/// Whether a varying product of this element width has a vector body.
pub(crate) fn has_vector_elementwise(bytes: usize) -> bool {
    product_serves(bytes)
}

/// Smallest row containing a vector product body, or zero without one.
pub(crate) fn elementwise_min_bytes(bytes: usize) -> usize {
    if product_serves(bytes) {
        PRODUCT_MIN_BYTES
    } else {
        0
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

/// Runs a fixed product over the longest servable whole-vector prefix.
///
/// Geometry is checked by the public operation. Returns the number of bytes
/// written; the caller applies its scalar row body to the rest. Zero and unit
/// coefficients are served over the whole row on every vector tier.
pub(crate) fn try_fixed<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    coefficient: u64,
    operation: Fixed<'_>,
) -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use super::x86::binary_linear;
        use crate::field::extension::arithmetic::word_one;

        let tier = serving_backend();
        if tier == Backend::Scalar {
            return 0;
        }
        if coefficient == 0 {
            if !matches!(operation, Fixed::Add(_)) {
                dst.fill(0);
            }
            return dst.len();
        }
        if coefficient == word_one(F::DESCRIPTOR, F::DESCRIPTOR.root_index()) {
            match operation {
                Fixed::Add(src) => super::xor(dst, src),
                Fixed::Into(src) => dst.copy_from_slice(src),
                Fixed::Assign => {}
            }
            return dst.len();
        }
        let head = dst.len() - dst.len() % vector_bytes(tier);
        if head < fixed_floor_elements(BYTES, tier) * BYTES {
            return 0;
        }
        #[cfg(test)]
        let _general = matches!(operation, Fixed::Add(_))
            .then(|| super::trace::term(super::trace::Term::General));
        let map = binary_linear::multiplication(F::DESCRIPTOR, coefficient);
        let dst = &mut dst[..head];
        if tier == Backend::V3 {
            let token = super::x86_v3_token();
            match operation {
                Fixed::Add(src) => {
                    binary_linear::mul_add_avx2::<BYTES>(token, dst, &map, &src[..head]);
                }
                Fixed::Into(src) => {
                    binary_linear::mul_into_avx2::<BYTES>(token, dst, &map, &src[..head]);
                }
                Fixed::Assign => binary_linear::mul_assign_avx2::<BYTES>(token, dst, &map),
            }
        } else {
            let token = super::x86_v2_token();
            match operation {
                Fixed::Add(src) => {
                    binary_linear::mul_add_sse41::<BYTES>(token, dst, &map, &src[..head]);
                }
                Fixed::Into(src) => {
                    binary_linear::mul_into_sse41::<BYTES>(token, dst, &map, &src[..head]);
                }
                Fixed::Assign => binary_linear::mul_assign_sse41::<BYTES>(token, dst, &map),
            }
        }
        head
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
        0
    }
}

/// Runs a two-byte varying product over the longest whole-vector prefix.
///
/// Returns the number of bytes written; the caller applies its scalar
/// product to the rest.
pub(crate) fn try_elementwise<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let head = dst.len() - dst.len() % vector_bytes(Backend::V3);
        if head >= PRODUCT_MIN_BYTES && product_serves(BYTES) {
            super::x86::binary_product::avx2::mul_elementwise_avx2(
                super::x86_v3_token(),
                F::DESCRIPTOR,
                &mut dst[..head],
                &a[..head],
                &b[..head],
            );
            return head;
        }
    }
    let _ = (core::marker::PhantomData::<F>, dst, a, b);
    0
}

/// Runs a two-byte in-place varying product over the longest whole-vector
/// prefix, without staging a destination copy.
///
/// Returns the number of bytes written; the caller applies its scalar
/// product to the rest.
pub(crate) fn try_elementwise_assign<F: Field, const BYTES: usize>(
    dst: &mut [u8],
    src: &[u8],
) -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let head = dst.len() - dst.len() % vector_bytes(Backend::V3);
        if head >= PRODUCT_MIN_BYTES && product_serves(BYTES) {
            super::x86::binary_product::avx2::mul_elementwise_assign_avx2(
                super::x86_v3_token(),
                F::DESCRIPTOR,
                &mut dst[..head],
                &src[..head],
            );
            return head;
        }
    }
    let _ = (core::marker::PhantomData::<F>, dst, src);
    0
}
