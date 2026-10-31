//! Descriptor-recursive varying products for the binary fields, in lanes.
//!
//! These kernels multiply two varying vectors of raw binary-field elements,
//! one full element per SIMD lane, for any `Descriptor` whose root
//! presentation is characteristic two and whose degree fits the lane width.
//! Unlike the pinned-tower families, the arithmetic is derived from the
//! descriptor at call time: prime leaves AND their lanes, polynomial and
//! GF(2)-relation leaves run the bit-serial shift/reduce loop, basis nodes
//! conjugate into their base through the presentation's columns, quadratic
//! relations run the tower identity, and every other relation runs schoolbook
//! convolution with top-down reduction. Each recursive call keeps one
//! canonical base coordinate word per lane, so lane-width shifts, adds, and
//! XORs never bleed across elements.
//!
//! # Backends
//!
//! `sse41` serves 16-byte vectors on `Backend::V2`; `avx2` serves
//! 32-byte vectors on `Backend::V3`. Both children expand the shared
//! `binary_product_lanes!` macro once per lane width, so one algorithm
//! source serves every `BYTES` in `{2, 4, 8}` on both instruction sets.

use crate::field::extension::arithmetic::word_one;
use crate::field::extension::descriptor::{Descriptor, NODES};

pub(crate) mod avx2;
pub(crate) mod sse41;

/// Buffer geometry every entry checks before any write.
///
/// The lane width must be one of the packed widths, the compared buffers
/// must match in length and hold whole elements, and the descriptor must be
/// characteristic two with its root degree inside the lane width.
///
/// # Panics
///
/// Panics naming the failing operand when any check fails.
pub(crate) fn check_entry_geometry<const BYTES: usize>(
    name: &str,
    descriptor: &Descriptor,
    dst_len: usize,
    other_len: usize,
    third_len: usize,
) {
    assert!(
        matches!(BYTES, 2 | 4 | 8),
        "{name}: {BYTES} is not a packed lane width",
    );
    assert_eq!(
        dst_len, other_len,
        "{name}: dst is {dst_len} bytes but a is {other_len} bytes",
    );
    assert_eq!(
        dst_len, third_len,
        "{name}: dst is {dst_len} bytes but b is {third_len} bytes",
    );
    assert!(
        dst_len.is_multiple_of(BYTES),
        "{name}: buffer of {dst_len} bytes ends with a partial {BYTES}-byte element",
    );
    assert_eq!(
        descriptor.characteristic(),
        2,
        "{name}: descriptor is not characteristic two",
    );
    assert!(
        (descriptor.degree() as usize) <= BYTES * 8,
        "{name}: degree {} does not fit the {BYTES}-byte lane width",
        descriptor.degree(),
    );
}

/// `word_one` of every used node, computed once per call.
///
/// The identity of an ordered-basis presentation is not the word one, so the
/// coefficient scale below compares against the actual word.
///
/// # Panics
///
/// Panics when the descriptor is not characteristic two; entries check that
/// before calling.
pub(crate) fn identity_words(descriptor: &Descriptor) -> [u64; NODES] {
    let mut ones = [0u64; NODES];
    let mut index = 0u8;
    while index < descriptor.node_count() {
        ones[index as usize] = word_one(descriptor, index);
        index += 1;
    }
    ones
}

/// The interpreter word of one raw element chunk, little-endian.
pub(crate) fn element_word(chunk: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..chunk.len()].copy_from_slice(chunk);
    u64::from_le_bytes(bytes)
}

/// The `bits` low ones of a word, never shifting a `u64` by 64.
#[inline]
pub(crate) fn low_word(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Expand the shared descriptor-recursive algorithm for one instruction
/// set, one vector type, and one lane width.
///
/// The invocation sits inside a module and generates the lane operations,
/// the shift and mask helpers, the node recursion, and the two lane drivers.
/// Every generated function carries the tier's `rite` proof, because the
/// underlying intrinsics demand a feature-enabled context. Arguments, in
/// order: the tier ident, the vector type, the vector width in bytes, the
/// lane width in bytes, and the intrinsics for zero, XOR, AND, OR, lane
/// equality, broadcast, left and right lane shifts, and the vector load and
/// store.
macro_rules! binary_product_lanes {
    (
        $tier:ident, $vec:ty, $vector_bytes:literal, $lane_bytes:literal,
        zero: $zero:ident, xor: $xor:ident, and: $and:ident, or: $or:ident,
        cmpeq: $cmpeq:ident, splat: $splat:ident,
        shl: $shl:ident, shr: $shr:ident,
        load: $load:ident, store: $store:ident
    ) => {
        mod ops {
            // The op bodies call the caller-named intrinsics.
            use core::arch::x86_64::*;

            /// The all-zero vector.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn zero() -> $vec {
                $zero()
            }

            /// Lane-wise XOR.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn xor(a: $vec, b: $vec) -> $vec {
                $xor(a, b)
            }

            /// Lane-wise AND.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn and(a: $vec, b: $vec) -> $vec {
                $and(a, b)
            }

            /// Lane-wise OR.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn or(a: $vec, b: $vec) -> $vec {
                $or(a, b)
            }

            /// Lane-wise equality, all-ones per equal lane.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn cmpeq(a: $vec, b: $vec) -> $vec {
                $cmpeq(a, b)
            }

            /// A lane word broadcast to every lane.
            // The lane word truncates to the lane width.
            #[allow(clippy::cast_possible_truncation)]
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn splat_u64(word: u64) -> $vec {
                $splat(word.cast_signed() as _)
            }

            /// Lane-wise left shift by a compile-time count.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn shl<const COUNT: i32>(v: $vec) -> $vec {
                $shl::<COUNT>(v)
            }

            /// Logical lane-wise right shift by a compile-time count.
            #[inline]
            #[archmage::rite($tier)]
            pub(super) fn shr<const COUNT: i32>(v: $vec) -> $vec {
                $shr::<COUNT>(v)
            }

            /// Load one full vector from an exact-size chunk.
            #[inline]
            #[archmage::rite($tier, import_intrinsics)]
            pub(super) fn load(chunk: &[u8]) -> $vec {
                $load(<&[u8; $vector_bytes]>::try_from(chunk).expect("whole vector chunk"))
            }

            /// Store one full vector into an exact-size chunk.
            #[inline]
            #[archmage::rite($tier, import_intrinsics)]
            pub(super) fn store(chunk: &mut [u8], v: $vec) {
                $store(
                    <&mut [u8; $vector_bytes]>::try_from(chunk).expect("whole vector chunk"),
                    v,
                );
            }
        }

        /// Logical right shift by a runtime lane-constant count.
        ///
        /// Right shifts compose exactly, so the binary digits of `count`
        /// select compile-time stages. Every count this algorithm passes
        /// stays below the lane width, so the stages above it never run.
        #[inline]
        #[archmage::rite($tier)]
        fn shr_bits(v: $vec, count: u32) -> $vec {
            let mut shifted = v;
            if count & 32 != 0 {
                shifted = ops::shr::<32>(shifted);
            }
            if count & 16 != 0 {
                shifted = ops::shr::<16>(shifted);
            }
            if count & 8 != 0 {
                shifted = ops::shr::<8>(shifted);
            }
            if count & 4 != 0 {
                shifted = ops::shr::<4>(shifted);
            }
            if count & 2 != 0 {
                shifted = ops::shr::<2>(shifted);
            }
            if count & 1 != 0 {
                shifted = ops::shr::<1>(shifted);
            }
            shifted
        }

        /// Left shift by a runtime lane-constant count.
        ///
        /// A left shift is multiplication by a power of two, and that
        /// composition survives wrapping at the lane width, so the same
        /// digit selection applies.
        #[inline]
        #[archmage::rite($tier)]
        fn shl_bits(v: $vec, count: u32) -> $vec {
            let mut shifted = v;
            if count & 32 != 0 {
                shifted = ops::shl::<32>(shifted);
            }
            if count & 16 != 0 {
                shifted = ops::shl::<16>(shifted);
            }
            if count & 8 != 0 {
                shifted = ops::shl::<8>(shifted);
            }
            if count & 4 != 0 {
                shifted = ops::shl::<4>(shifted);
            }
            if count & 2 != 0 {
                shifted = ops::shl::<2>(shifted);
            }
            if count & 1 != 0 {
                shifted = ops::shl::<1>(shifted);
            }
            shifted
        }

        /// All-ones per lane where bit `bit` of the lane is one.
        #[inline]
        #[archmage::rite($tier)]
        fn bit_mask(v: $vec, bit: u32) -> $vec {
            let one = ops::splat_u64(1);
            ops::cmpeq(ops::and(shr_bits(v, bit), one), one)
        }

        /// A lane word of `bits` low ones broadcast to every lane.
        #[inline]
        #[archmage::rite($tier)]
        fn low_mask(bits: u32) -> $vec {
            ops::splat_u64(crate::kernel::x86::binary_product::low_word(bits))
        }

        /// Apply a byte-linear map lane-wise: bit `i` of each lane selects
        /// `columns[i]`.
        ///
        /// This mirrors the presentation's own `apply`, bits zero through
        /// seven, so the conjugation is exact in every presentation,
        /// including ordered bases.
        #[inline]
        #[archmage::rite($tier)]
        fn apply_columns(columns: &[u8; 8], v: $vec) -> $vec {
            let mut acc = ops::zero();
            let mut i = 0u32;
            while i < 8 {
                acc = ops::xor(
                    acc,
                    ops::and(
                        ops::splat_u64(u64::from(columns[i as usize])),
                        bit_mask(v, i),
                    ),
                );
                i += 1;
            }
            acc
        }

        /// The reduction word of a polynomial leaf: the full polynomial's
        /// low bits, without the leading term.
        ///
        /// # Panics
        ///
        /// Panics when the degree reaches 64, where the leading-bit mask
        /// would overflow the word.
        #[inline]
        #[archmage::rite($tier)]
        fn polynomial_reduction(full: u128, degree: u32) -> u64 {
            assert!(
                degree < 64,
                "polynomial leaf of degree {degree} exceeds the word capacity",
            );
            #[allow(clippy::cast_possible_truncation)]
            let low = full as u64;
            low & crate::kernel::x86::binary_product::low_word(degree)
        }

        /// The reduction word of a relation directly over GF(2): one packed
        /// coordinate bit per coefficient.
        ///
        /// # Panics
        ///
        /// Panics when the relation exceeds the word capacity; the
        /// coordinate capacity bounds every relation's degree.
        #[inline]
        #[archmage::rite($tier)]
        fn gf2_reduction_word(
            coefficients: &[u64; $crate::field::extension::descriptor::COORDINATES],
            degree: usize,
        ) -> u64 {
            assert!(
                degree <= 64,
                "relation of degree {degree} exceeds the word capacity",
            );
            let mut word = 0u64;
            let mut j = 0;
            while j < degree {
                word |= (coefficients[j] & 1) << j;
                j += 1;
            }
            word
        }

        /// The bit-serial product of two lane vectors modulo one reduction
        /// word.
        ///
        /// Each of the `degree` iterations accumulates the shifted-reduced
        /// left operand when the right operand's low bit is one, then folds
        /// the operand's exiting top bit through the reduction word and
        /// clears the exited bit, which the lane shift retains whenever the
        /// degree sits below the lane width. Both operands stay below
        /// `2^degree`, so every intermediate stays inside the lane and
        /// degree 64 never shifts a word by 64.
        #[inline]
        #[archmage::rite($tier)]
        fn bit_serial_product(degree: u32, reduction: u64, x: $vec, y: $vec) -> $vec {
            let red = ops::splat_u64(reduction);
            let degree_mask = low_mask(degree);
            let mut acc = ops::zero();
            let mut xi = x;
            let mut yi = y;
            let mut bit = 0;
            while bit < degree {
                acc = ops::xor(acc, ops::and(xi, bit_mask(yi, 0)));
                yi = ops::shr::<1>(yi);
                let top = bit_mask(xi, degree - 1);
                let shifted = ops::and(ops::shl::<1>(xi), degree_mask);
                xi = ops::xor(shifted, ops::and(red, top));
                bit += 1;
            }
            acc
        }

        /// The scale of one base element by a scalar relation coefficient.
        ///
        /// Zero and the base's actual identity word short-circuit; every
        /// other coefficient broadcasts through the same recursive product.
        #[inline]
        #[archmage::rite($tier)]
        fn coefficient_scale(
            descriptor: &$crate::field::extension::Descriptor,
            ones: &[u64; $crate::field::extension::descriptor::NODES],
            base: u8,
            coefficient: u64,
            v: $vec,
        ) -> $vec {
            if coefficient == 0 {
                ops::zero()
            } else if coefficient == ones[base as usize] {
                v
            } else {
                node_product(descriptor, ones, base, ops::splat_u64(coefficient), v)
            }
        }

        /// The tower identity for a quadratic relation: one base product
        /// split into low and high halves around the relation's two
        /// coefficients.
        #[inline]
        #[archmage::rite($tier)]
        fn quadratic_product(
            descriptor: &$crate::field::extension::Descriptor,
            ones: &[u64; $crate::field::extension::descriptor::NODES],
            base: u8,
            base_degree: u32,
            coefficients: &[u64; $crate::field::extension::descriptor::COORDINATES],
            x: $vec,
            y: $vec,
        ) -> $vec {
            let low = low_mask(base_degree);
            let linear = coefficients[1];
            let constant = coefficients[0];
            let x0 = ops::and(x, low);
            let x1 = shr_bits(x, base_degree);
            let y0 = ops::and(y, low);
            let y1 = shr_bits(y, base_degree);
            let ac = node_product(descriptor, ones, base, x0, y0);
            let bd = node_product(descriptor, ones, base, x1, y1);
            let mid = ops::xor(
                ops::xor(
                    node_product(descriptor, ones, base, ops::xor(x0, x1), ops::xor(y0, y1)),
                    ac,
                ),
                bd,
            );
            let lo = ops::xor(ac, coefficient_scale(descriptor, ones, base, constant, bd));
            let hi = ops::xor(mid, coefficient_scale(descriptor, ones, base, linear, bd));
            ops::or(ops::and(lo, low), shl_bits(ops::and(hi, low), base_degree))
        }

        /// Workspace bound of the schoolbook reduction: one accumulator per
        /// product coefficient.
        ///
        /// A relation of relative degree `n` over a base of degree at least
        /// two satisfies `2 * n <= 64`, so `n <= 32` and the accumulator
        /// never holds more than `2 * RELATION_WINDOW - 1` vectors.
        const RELATION_WINDOW: usize = 32;

        /// Schoolbook convolution with top-down reduction for the remaining
        /// relations.
        ///
        /// The shape is the descriptor interpreter's own schoolbook: one
        /// bounded vector accumulator per product coefficient, then the
        /// relation folds every coefficient from the top down, each fold
        /// feeding the accumulators below it. No allocation is ever taken.
        /// Coordinate extraction and reassembly shift by whole multiples of
        /// the base degree, all below the lane width.
        ///
        /// # Panics
        ///
        /// Panics when the relation exceeds the window bound; validated
        /// descriptors never do.
        #[inline]
        #[archmage::rite($tier)]
        fn schoolbook_product(
            descriptor: &$crate::field::extension::Descriptor,
            ones: &[u64; $crate::field::extension::descriptor::NODES],
            base: u8,
            relative_degree: usize,
            base_degree: u32,
            coefficients: &[u64; $crate::field::extension::descriptor::COORDINATES],
            operands: ($vec, $vec),
        ) -> $vec {
            let (x, y) = operands;
            assert!(
                relative_degree <= RELATION_WINDOW,
                "relation of relative degree {relative_degree} exceeds the schoolbook window",
            );
            let low = low_mask(base_degree);
            let mut y_coords = [ops::zero(); RELATION_WINDOW];
            let mut window = [ops::zero(); 2 * RELATION_WINDOW];
            // The right operand's coordinates, extracted once.
            let mut j = 0;
            while j < relative_degree {
                // The coordinate shift counts stay below the lane width:
                // the relation's total degree never exceeds the packed
                // word.
                #[allow(clippy::cast_possible_truncation)]
                let shift = (j as u32) * base_degree;
                y_coords[j] = ops::and(shr_bits(y, shift), low);
                j += 1;
            }
            // Convolution: one left coordinate per accumulator line.
            let mut i = 0;
            while i < relative_degree {
                #[allow(clippy::cast_possible_truncation)]
                let shift = (i as u32) * base_degree;
                let xi = ops::and(shr_bits(x, shift), low);
                j = 0;
                while j < relative_degree {
                    window[i + j] = ops::xor(
                        window[i + j],
                        node_product(descriptor, ones, base, xi, y_coords[j]),
                    );
                    j += 1;
                }
                i += 1;
            }
            // The relation folds every coefficient from the top down, each
            // fold feeding the accumulators below it, including the top
            // coefficients the following folds consume.
            let mut k = 2 * relative_degree - 2;
            while k >= relative_degree {
                let top = window[k];
                j = 0;
                while j < relative_degree {
                    window[k - relative_degree + j] = ops::xor(
                        window[k - relative_degree + j],
                        node_product(descriptor, ones, base, top, ops::splat_u64(coefficients[j])),
                    );
                    j += 1;
                }
                k -= 1;
            }
            let mut result = ops::zero();
            i = 0;
            while i < relative_degree {
                // As above: the shift count is a coordinate offset below
                // the lane width.
                #[allow(clippy::cast_possible_truncation)]
                let shift = (i as u32) * base_degree;
                result = ops::or(result, shl_bits(ops::and(window[i], low), shift));
                i += 1;
            }
            result
        }

        /// The product of two lane vectors in the node at `index`.
        ///
        /// Every recursive call receives canonical coordinate words, bits
        /// above the node's degree clear, which keeps shifts and masks
        /// inside their lanes. The traversal only reaches used nodes of a
        /// validated characteristic-two descriptor.
        ///
        /// # Panics
        ///
        /// Panics when the descriptor is not characteristic two; entries
        /// check that once before the lane loop.
        #[inline]
        #[archmage::rite($tier)]
        fn node_product(
            descriptor: &$crate::field::extension::Descriptor,
            ones: &[u64; $crate::field::extension::descriptor::NODES],
            index: u8,
            x: $vec,
            y: $vec,
        ) -> $vec {
            use crate::field::extension::descriptor::Node;
            match descriptor.node(index) {
                Node::Prime { modulus: 2, .. } => ops::and(x, y),
                Node::Polynomial { degree, full } => {
                    let degree = u32::from(*degree);
                    bit_serial_product(degree, polynomial_reduction(*full, degree), x, y)
                }
                Node::Basis {
                    base,
                    to_base,
                    from_base,
                } => {
                    let xb = apply_columns(to_base, x);
                    let yb = apply_columns(to_base, y);
                    let product = node_product(descriptor, ones, *base, xb, yb);
                    apply_columns(from_base, product)
                }
                Node::Extension {
                    base,
                    relative_degree: 2,
                    base_degree,
                    coefficients,
                    ..
                } => quadratic_product(descriptor, ones, *base, *base_degree, coefficients, x, y),
                Node::Extension {
                    base,
                    relative_degree,
                    base_degree,
                    coefficients,
                    ..
                } => {
                    if *base_degree == 1 {
                        // Coordinates over GF(2) are canonical bits, so the
                        // relation itself is the reduction word.
                        bit_serial_product(
                            u32::try_from(*relative_degree)
                                .expect("relation degree fits the coordinate capacity"),
                            gf2_reduction_word(coefficients, *relative_degree),
                            x,
                            y,
                        )
                    } else {
                        schoolbook_product(
                            descriptor,
                            ones,
                            *base,
                            *relative_degree,
                            *base_degree,
                            coefficients,
                            (x, y),
                        )
                    }
                }
                Node::Unused => unreachable!("descriptor traversal reaches an unused node"),
                // Entries check the root characteristic, which every node
                // shares.
                Node::Prime { .. } => {
                    unreachable!("odd-characteristic node in a binary descriptor")
                }
            }
        }

        /// `dst[i] = a[i] * b[i]` over whole vectors, then the scalar
        /// descriptor tail.
        #[inline]
        #[archmage::rite($tier)]
        pub(super) fn mul_elementwise_impl<const BYTES: usize>(
            descriptor: &$crate::field::extension::Descriptor,
            dst: &mut [u8],
            a: &[u8],
            b: &[u8],
        ) {
            // The entry dispatches one lane width per module.
            debug_assert_eq!(BYTES, $lane_bytes);
            let ones = crate::kernel::x86::binary_product::identity_words(descriptor);
            let input_mask = low_mask(descriptor.degree());
            let root = descriptor.root_index();
            let mut dst_chunks = dst.chunks_exact_mut($vector_bytes);
            let mut a_chunks = a.chunks_exact($vector_bytes);
            let mut b_chunks = b.chunks_exact($vector_bytes);
            for ((dst_chunk, x_chunk), y_chunk) in dst_chunks
                .by_ref()
                .zip(a_chunks.by_ref())
                .zip(b_chunks.by_ref())
            {
                let x = ops::and(ops::load(x_chunk), input_mask);
                let y = ops::and(ops::load(y_chunk), input_mask);
                ops::store(dst_chunk, node_product(descriptor, &ones, root, x, y));
            }
            for ((d, x), y) in dst_chunks
                .into_remainder()
                .chunks_exact_mut($lane_bytes)
                .zip(a_chunks.remainder().chunks_exact($lane_bytes))
                .zip(b_chunks.remainder().chunks_exact($lane_bytes))
            {
                let word = crate::kernel::x86::binary_product::element_word(x);
                let other = crate::kernel::x86::binary_product::element_word(y);
                let product =
                    crate::field::extension::arithmetic::word_mul(descriptor, root, word, other);
                d.copy_from_slice(&product.to_le_bytes()[..$lane_bytes]);
            }
        }

        /// `dst[i] *= src[i]` over whole vectors, then the scalar
        /// descriptor tail. The destination is read and rewritten in place,
        /// with no copy and no allocation.
        #[inline]
        #[archmage::rite($tier)]
        pub(super) fn mul_elementwise_assign_impl<const BYTES: usize>(
            descriptor: &$crate::field::extension::Descriptor,
            dst: &mut [u8],
            src: &[u8],
        ) {
            // The entry dispatches one lane width per module.
            debug_assert_eq!(BYTES, $lane_bytes);
            let ones = crate::kernel::x86::binary_product::identity_words(descriptor);
            let input_mask = low_mask(descriptor.degree());
            let root = descriptor.root_index();
            let mut dst_chunks = dst.chunks_exact_mut($vector_bytes);
            let mut src_chunks = src.chunks_exact($vector_bytes);
            for (dst_chunk, y_chunk) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
                let x = ops::and(ops::load(dst_chunk), input_mask);
                let y = ops::and(ops::load(y_chunk), input_mask);
                ops::store(dst_chunk, node_product(descriptor, &ones, root, x, y));
            }
            for (d, y) in dst_chunks
                .into_remainder()
                .chunks_exact_mut($lane_bytes)
                .zip(src_chunks.remainder().chunks_exact($lane_bytes))
            {
                let word = crate::kernel::x86::binary_product::element_word(d);
                let other = crate::kernel::x86::binary_product::element_word(y);
                let product =
                    crate::field::extension::arithmetic::word_mul(descriptor, root, word, other);
                d.copy_from_slice(&product.to_le_bytes()[..$lane_bytes]);
            }
        }
    };
}

pub(crate) use binary_product_lanes;

#[cfg(test)]
mod tests {
    use std::vec;

    use archmage::{SimdToken, X64V2Token, X64V3Token};

    use crate::field::binary::tower::{
        FanPaar4Field, FanPaar8Field, FanPaar16Field, FanPaar32Field, FanPaar64Field,
        Rijndael16Field, Rijndael32Field,
    };
    use crate::field::binary::{AES, Normal, RS};
    use crate::field::{Extension, ExtensionSpec, FiniteField, Packed, Polynomial, Relation};
    use crate::kernel::Backend;
    use crate::kernel::scalar;
    use crate::kernel::tests::{expect_geometry_panic, noise};

    use super::{avx2, check_entry_geometry, sse41};

    /// Row lengths straddling every 16- and 32-byte lane boundary, all even
    /// for the two-byte fields.
    const LENGTHS_2: &[usize] = &[
        0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
    ];

    /// Lengths for four-byte elements, still straddling both lane widths.
    const LENGTHS_4: &[usize] = &[
        0, 4, 8, 12, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512,
    ];

    /// Lengths for eight-byte elements.
    const LENGTHS_8: &[usize] = &[0, 8, 16, 24, 32, 40, 72, 128, 136, 248, 256, 264, 504, 512];

    /// Whether the host resolves to one of `supported`, honoring
    /// `SIMD_BACKEND`; the shared substitute for feature-detection gates.
    fn host_supports(supported: &'static [Backend]) -> bool {
        let resolved = simdispatch::Selection::new("SIMD_BACKEND")
            .supports(supported)
            .resolve();
        if resolved != Backend::Scalar {
            std::eprintln!(
                "binary_product_differential executing tier={} case={}",
                resolved.name(),
                std::thread::current().name().unwrap_or("unnamed"),
            );
        }
        resolved != Backend::Scalar
    }

    fn v2_token() -> Option<X64V2Token> {
        if !host_supports(&[Backend::V2]) {
            return None;
        }
        Some(X64V2Token::summon().expect("the selected V2 tier must supply its token"))
    }

    fn v3_token() -> Option<X64V3Token> {
        if !host_supports(&[Backend::V3]) {
            return None;
        }
        Some(X64V3Token::summon().expect("the selected V3 tier must supply its token"))
    }

    // Custom rows under test. Every relation was verified irreducible
    // independently of this crate's own validation before being pinned.

    /// A quadratic over the Fan-Paar GF(2^8) row: `t^2 + t + 0x10`.
    #[derive(Clone, Copy)]
    struct FpQuad16;
    impl ExtensionSpec<2> for FpQuad16 {
        type Base = FanPaar8Field;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom GF(2^16) over Fan-Paar GF(2^8)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x10, 0x001]);
    }
    type FpQuad16Field = FiniteField<2, 16, Extension<FpQuad16, 2>>;

    /// A quadratic over the Reed-Solomon leaf: `t^2 + t + 0x20`.
    #[derive(Clone, Copy)]
    struct RsQuad16;
    impl ExtensionSpec<2> for RsQuad16 {
        type Base = FiniteField<2, 8, Polynomial<RS>>;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom GF(2^16) over RS";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x20, 0x001]);
    }
    type RsQuad16Field = FiniteField<2, 16, Extension<RsQuad16, 2>>;

    /// A quadratic over the normal-basis GF(2^8) row: `t^2 + t + 1`, whose
    /// base encodes one as `0xff` rather than the word one.
    #[derive(Clone, Copy)]
    struct NormalQuad16;
    impl ExtensionSpec<2> for NormalQuad16 {
        type Base = FiniteField<2, 8, Normal<AES, 0x20>>;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom GF(2^16) over normal GF(2^8)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x01, 0x01]);
    }
    type NormalQuad16Field = FiniteField<2, 16, Extension<NormalQuad16, 2>>;

    /// GF(2^16) directly over GF(2) by `x^16 + x^12 + x^3 + x + 1`.
    #[derive(Clone, Copy)]
    struct Gf2Tower16;
    impl ExtensionSpec<16> for Gf2Tower16 {
        type Base = FiniteField<2, 1, Polynomial<3>>;
        type Storage = Packed<u16>;
        const NAME: &'static str = "GF(2^16) over GF(2)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[
            1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        ]);
    }
    type Gf2Tower16Field = FiniteField<2, 16, Extension<Gf2Tower16, 16>>;

    /// GF(2^16) as a quartic relation over the Fan-Paar GF(2^4) row:
    /// `t^4 + t^3 + 2*t + 5`. This is the schoolbook path at two-byte lanes.
    #[derive(Clone, Copy)]
    struct Fp4Quartic;
    impl ExtensionSpec<4> for Fp4Quartic {
        type Base = FanPaar4Field;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom GF(2^16) quartic over Fan-Paar GF(2^4)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[5, 2, 0, 1]);
    }
    type Fp4QuarticField = FiniteField<2, 16, Extension<Fp4Quartic, 4>>;

    /// A quadratic over the Fan-Paar GF(2^16) row: `t^2 + t + 0x100`.
    #[derive(Clone, Copy)]
    struct FpQuad32;
    impl ExtensionSpec<2> for FpQuad32 {
        type Base = FanPaar16Field;
        type Storage = Packed<u32>;
        const NAME: &'static str = "custom GF(2^32) over Fan-Paar GF(2^16)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x100, 0x0001]);
    }
    type FpQuad32Field = FiniteField<2, 32, Extension<FpQuad32, 2>>;

    /// A quadratic over the Rijndael GF(2^16) row: `t^2 + t + 0x2000`.
    #[derive(Clone, Copy)]
    struct R16Quad32;
    impl ExtensionSpec<2> for R16Quad32 {
        type Base = Rijndael16Field;
        type Storage = Packed<u32>;
        const NAME: &'static str = "custom GF(2^32) over Rijndael GF(2^16)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x2000, 0x0001]);
    }
    type R16Quad32Field = FiniteField<2, 32, Extension<R16Quad32, 2>>;

    /// GF(2^32) as a quartic relation over the AES leaf:
    /// `t^4 + t^3 + t + 7`. This is the schoolbook path at four-byte lanes.
    #[derive(Clone, Copy)]
    struct AesQuartic;
    impl ExtensionSpec<4> for AesQuartic {
        type Base = FiniteField<2, 8, Polynomial<AES>>;
        type Storage = Packed<u32>;
        const NAME: &'static str = "custom GF(2^32) quartic over AES";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x7, 0x1, 0x0, 0x1]);
    }
    type AesQuarticField = FiniteField<2, 32, Extension<AesQuartic, 4>>;

    /// A quadratic over the Fan-Paar GF(2^32) row: `t^2 + t + 0x2000_0000`.
    #[derive(Clone, Copy)]
    struct FpQuad64;
    impl ExtensionSpec<2> for FpQuad64 {
        type Base = FanPaar32Field;
        type Storage = Packed<u64>;
        const NAME: &'static str = "custom GF(2^64) over Fan-Paar GF(2^32)";
        const RELATION: Relation =
            Relation::from_binary_words::<Self::Base>(&[0x2000_0000, 0x0000_0001]);
    }
    type FpQuad64Field = FiniteField<2, 64, Extension<FpQuad64, 2>>;

    /// A quadratic over the Rijndael GF(2^32) row: `t^2 + t + 0x2000_0000`.
    #[derive(Clone, Copy)]
    struct R32Quad64;
    impl ExtensionSpec<2> for R32Quad64 {
        type Base = Rijndael32Field;
        type Storage = Packed<u64>;
        const NAME: &'static str = "custom GF(2^64) over Rijndael GF(2^32)";
        const RELATION: Relation =
            Relation::from_binary_words::<Self::Base>(&[0x2000_0000, 0x0000_0001]);
    }
    type R32Quad64Field = FiniteField<2, 64, Extension<R32Quad64, 2>>;

    /// GF(2^64) directly over GF(2) by `x^64 + x^4 + x^3 + x + 1`: the
    /// degree-64 edge of the bit-serial path.
    #[derive(Clone, Copy)]
    struct Gf2Tower64;
    impl ExtensionSpec<64> for Gf2Tower64 {
        type Base = FiniteField<2, 1, Polynomial<3>>;
        type Storage = Packed<u64>;
        const NAME: &'static str = "GF(2^64) over GF(2)";
        const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&GF2_64_WORDS);
    }
    type Gf2Tower64Field = FiniteField<2, 64, Extension<Gf2Tower64, 64>>;

    /// The degree-64 relation's non-leading coefficients, constant first.
    const GF2_64_WORDS: [u64; 64] = {
        let mut words = [0u64; 64];
        words[0] = 1;
        words[1] = 1;
        words[3] = 1;
        words[4] = 1;
        words
    };

    /// GF(2^64) as an octic relation over the AES leaf:
    /// `t^8 + t^3 + 3*t + 1`. This is the schoolbook path at eight-byte
    /// lanes.
    #[derive(Clone, Copy)]
    struct AesOctic;
    impl ExtensionSpec<8> for AesOctic {
        type Base = FiniteField<2, 8, Polynomial<AES>>;
        type Storage = Packed<u64>;
        const NAME: &'static str = "custom GF(2^64) octic over AES";
        const RELATION: Relation =
            Relation::from_binary_words::<Self::Base>(&[0x1, 0x3, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0]);
    }
    type AesOcticField = FiniteField<2, 64, Extension<AesOctic, 8>>;

    fn field_descriptor<F: crate::field::Field>() -> &'static crate::field::extension::Descriptor {
        F::DESCRIPTOR
    }

    /// Checks whole outputs and untouched guards at aligned and misaligned starts.
    fn check_field<F: crate::field::FieldBuffer, const BYTES: usize>(
        lengths: &[usize],
        v2: Option<X64V2Token>,
        v3: Option<X64V3Token>,
    ) {
        let descriptor = field_descriptor::<F>();
        for &len in lengths {
            for offset in [0, 1, 3, 15, 31] {
                let a = noise(len + offset + 37, 0x1A1);
                let b = noise(len + offset + 37, 0x1B2);
                let window = offset..offset + len;
                let mut want = vec![0u8; len];
                scalar::mul_elementwise::<F>(&mut want, &a[window.clone()], &b[window.clone()]);
                for tier in [Backend::V2, Backend::V3] {
                    if (tier == Backend::V2 && v2.is_none())
                        || (tier == Backend::V3 && v3.is_none())
                    {
                        continue;
                    }
                    let mut got = noise(a.len(), 0x1C3);
                    let mut expected = got.clone();
                    expected[window.clone()].copy_from_slice(&want);
                    let mut assign = a.clone();
                    let mut expected_assign = a.clone();
                    expected_assign[window.clone()].copy_from_slice(&want);
                    match tier {
                        Backend::V2 => {
                            let token = v2.expect("checked tier availability");
                            sse41::mul_elementwise_sse41::<BYTES>(
                                token,
                                descriptor,
                                &mut got[window.clone()],
                                &a[window.clone()],
                                &b[window.clone()],
                            );
                            sse41::mul_elementwise_assign_sse41::<BYTES>(
                                token,
                                descriptor,
                                &mut assign[window.clone()],
                                &b[window.clone()],
                            );
                        }
                        Backend::V3 => {
                            let token = v3.expect("checked tier availability");
                            avx2::mul_elementwise_avx2::<BYTES>(
                                token,
                                descriptor,
                                &mut got[window.clone()],
                                &a[window.clone()],
                                &b[window.clone()],
                            );
                            avx2::mul_elementwise_assign_avx2::<BYTES>(
                                token,
                                descriptor,
                                &mut assign[window.clone()],
                                &b[window.clone()],
                            );
                        }
                        _ => unreachable!("only tested tiers are enumerated"),
                    }
                    assert_eq!(
                        got,
                        expected,
                        "{} {tier:?}, len {len}, offset {offset}",
                        F::NAME
                    );
                    assert_eq!(
                        assign,
                        expected_assign,
                        "{} {tier:?} assign, len {len}, offset {offset}",
                        F::NAME,
                    );
                }
            }
        }
    }

    fn check_all_widths(v2: Option<X64V2Token>, v3: Option<X64V3Token>) {
        check_field::<FanPaar16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<Rijndael16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<FpQuad16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<RsQuad16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<NormalQuad16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<Gf2Tower16Field, 2>(LENGTHS_2, v2, v3);
        check_field::<Fp4QuarticField, 2>(LENGTHS_2, v2, v3);
        check_field::<FanPaar32Field, 4>(LENGTHS_4, v2, v3);
        check_field::<FpQuad32Field, 4>(LENGTHS_4, v2, v3);
        check_field::<R16Quad32Field, 4>(LENGTHS_4, v2, v3);
        check_field::<AesQuarticField, 4>(LENGTHS_4, v2, v3);
        check_field::<FanPaar64Field, 8>(LENGTHS_8, v2, v3);
        check_field::<FpQuad64Field, 8>(LENGTHS_8, v2, v3);
        check_field::<R32Quad64Field, 8>(LENGTHS_8, v2, v3);
        check_field::<Gf2Tower64Field, 8>(LENGTHS_8, v2, v3);
        check_field::<AesOcticField, 8>(LENGTHS_8, v2, v3);
    }

    /// SSE4.1 bodies against the reference on every shipped and custom row.
    #[test]
    fn sse41_lanes_match_reference() {
        let Some(token) = v2_token() else {
            eprintln!("skipping: no SSE4.1 tier on this host");
            return;
        };
        check_all_widths(Some(token), None);
    }

    /// AVX2 bodies against the reference on every shipped and custom row.
    #[test]
    fn avx2_lanes_match_reference() {
        let Some(token) = v3_token() else {
            eprintln!("skipping: no AVX2 tier on this host");
            return;
        };
        check_all_widths(None, Some(token));
    }

    /// The geometry checks reject mismatches, partial elements, unsupported
    /// lane widths, odd characteristics, and degrees above the lane width.
    #[test]
    fn geometry_checks_fail_loudly() {
        let descriptor = field_descriptor::<FanPaar16Field>();
        let name = "binary_product::test";
        for (dst, a, b) in [(6, 8, 8), (8, 6, 8), (8, 8, 6)] {
            expect_geometry_panic(
                || check_entry_geometry::<2>(name, descriptor, dst, a, b),
                "binary_product::test: dst",
            );
        }
        expect_geometry_panic(
            || check_entry_geometry::<2>(name, descriptor, 3, 3, 3),
            "binary_product::test: buffer",
        );
        expect_geometry_panic(
            || check_entry_geometry::<3>(name, descriptor, 3, 3, 3),
            "binary_product::test: 3 is not a packed lane width",
        );
        let odd = field_descriptor::<crate::field::QuadMersenne31>();
        expect_geometry_panic(
            || check_entry_geometry::<2>(name, odd, 4, 4, 4),
            "binary_product::test: descriptor is not characteristic two",
        );
        let wide = field_descriptor::<Gf2Tower64Field>();
        expect_geometry_panic(
            || check_entry_geometry::<2>(name, wide, 8, 8, 8),
            "binary_product::test: degree",
        );
    }
}
