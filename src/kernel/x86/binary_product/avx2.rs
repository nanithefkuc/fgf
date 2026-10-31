//! AVX2 descriptor-recursive binary varying products over 32-byte lanes.
//!
//! One module per lane width expands the shared
//! [`binary_product_lanes!`](crate::kernel::x86::binary_product)
//! algorithm over the AVX2 vector ops; the entries check the geometry and
//! hand whole `BYTES`-byte lanes to the matching module.

use crate::field::extension::Descriptor;
use crate::kernel::x86::binary_product::check_entry_geometry;

mod lanes16 {
    // The invocation names the intrinsics, so the namespace must be in
    // scope inside this module.
    use core::arch::x86_64::*;

    use crate::kernel::x86::binary_product::binary_product_lanes;

    binary_product_lanes!(
        v3,
        __m256i,
        32,
        2,
        zero: _mm256_setzero_si256,
        xor: _mm256_xor_si256,
        and: _mm256_and_si256,
        or: _mm256_or_si256,
        cmpeq: _mm256_cmpeq_epi16,
        splat: _mm256_set1_epi16,
        shl: _mm256_slli_epi16,
        shr: _mm256_srli_epi16,
        load: _mm256_loadu_si256,
        store: _mm256_storeu_si256
    );
}

mod lanes32 {
    // The invocation names the intrinsics, so the namespace must be in
    // scope inside this module.
    use core::arch::x86_64::*;

    use crate::kernel::x86::binary_product::binary_product_lanes;

    binary_product_lanes!(
        v3,
        __m256i,
        32,
        4,
        zero: _mm256_setzero_si256,
        xor: _mm256_xor_si256,
        and: _mm256_and_si256,
        or: _mm256_or_si256,
        cmpeq: _mm256_cmpeq_epi32,
        splat: _mm256_set1_epi32,
        shl: _mm256_slli_epi32,
        shr: _mm256_srli_epi32,
        load: _mm256_loadu_si256,
        store: _mm256_storeu_si256
    );
}

mod lanes64 {
    // The invocation names the intrinsics, so the namespace must be in
    // scope inside this module.
    use core::arch::x86_64::*;

    use crate::kernel::x86::binary_product::binary_product_lanes;

    binary_product_lanes!(
        v3,
        __m256i,
        32,
        8,
        zero: _mm256_setzero_si256,
        xor: _mm256_xor_si256,
        and: _mm256_and_si256,
        or: _mm256_or_si256,
        cmpeq: _mm256_cmpeq_epi64,
        splat: _mm256_set1_epi64x,
        shl: _mm256_slli_epi64,
        shr: _mm256_srli_epi64,
        load: _mm256_loadu_si256,
        store: _mm256_storeu_si256
    );
}

/// `dst[i] = a[i] * b[i]` under `descriptor`, 32-byte lanes (AVX2).
///
/// One full raw element occupies one lane of `BYTES` bytes, `BYTES` in
/// `{2, 4, 8}`. The recursion is the descriptor's own presentation, and the
/// scalar tail multiplies through the same descriptor.
///
/// # Panics
///
/// Panics unless all three buffers match in length, hold whole elements,
/// and the descriptor is characteristic two with its degree inside the
/// lane width.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2<const BYTES: usize>(
    _token: archmage::X64V3Token,
    descriptor: &Descriptor,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    check_entry_geometry::<BYTES>(
        "binary_product::mul_elementwise_avx2",
        descriptor,
        dst.len(),
        a.len(),
        b.len(),
    );
    match BYTES {
        2 => lanes16::mul_elementwise_impl::<BYTES>(descriptor, dst, a, b),
        4 => lanes32::mul_elementwise_impl::<BYTES>(descriptor, dst, a, b),
        8 => lanes64::mul_elementwise_impl::<BYTES>(descriptor, dst, a, b),
        _ => unreachable!("check_entry_geometry rejects other lane widths"),
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "binary_product.avx2.mul_elementwise",
        crate::kernel::trace::Substage::MulElementwise,
        crate::kernel::trace::Unit::Elements,
        dst.len() / 32 * (32 / BYTES),
        crate::kernel::Backend::V3,
    );
}

/// `dst[i] *= src[i]` under `descriptor`, 32-byte lanes (AVX2).
///
/// The destination is read and rewritten in place; no copy of the
/// destination is taken and nothing is allocated.
///
/// # Panics
///
/// Panics unless both buffers match in length and hold whole elements, and
/// the descriptor is characteristic two with its degree inside the lane
/// width.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2<const BYTES: usize>(
    _token: archmage::X64V3Token,
    descriptor: &Descriptor,
    dst: &mut [u8],
    src: &[u8],
) {
    check_entry_geometry::<BYTES>(
        "binary_product::mul_elementwise_assign_avx2",
        descriptor,
        dst.len(),
        src.len(),
        src.len(),
    );
    match BYTES {
        2 => lanes16::mul_elementwise_assign_impl::<BYTES>(descriptor, dst, src),
        4 => lanes32::mul_elementwise_assign_impl::<BYTES>(descriptor, dst, src),
        8 => lanes64::mul_elementwise_assign_impl::<BYTES>(descriptor, dst, src),
        _ => unreachable!("check_entry_geometry rejects other lane widths"),
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "binary_product.avx2.mul_elementwise_assign",
        crate::kernel::trace::Substage::MulElementwise,
        crate::kernel::trace::Unit::Elements,
        dst.len() / 32 * (32 / BYTES),
        crate::kernel::Backend::V3,
    );
}
