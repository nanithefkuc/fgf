//! Batch reciprocal and division rows: Montgomery inversion over lanes.
//!
//! The scalar [`FieldElem::inv`](crate::field::FieldElem::inv) maps zero to
//! zero and [`FieldElem::div`](crate::field::FieldElem::div) maps a zero
//! divisor to zero; these row forms keep that totality lane by lane. Zeros
//! map to zero and never poison the shared batch product: the running
//! prefix skips zero lanes, the single inverted product only touches
//! nonzero lanes, and zero lanes are written as zero on the way out.

use crate::field::{Elem, FieldElem};
use crate::kernel::FieldKernels;

use super::{check_pair, check_width};

/// Montgomery chunk: stack scratch sized to cover a 512-bit AVX-512 lane
/// of the widest 8-byte lanes without touching the heap.
const CHUNK_ELEMS: usize = 64;

/// Stack scratch per staged chunk: `CHUNK_ELEMS` lanes of 8-byte storage.
/// Wider component elements stage fewer lanes per chunk; the 64-coordinate
/// degree cap keeps every element within this many bytes.
const SCRATCH_BYTES: usize = CHUNK_ELEMS * 8;

/// Lanes per chunk that fit both the prefix array and the byte scratch.
fn chunk_elems<F: FieldKernels>() -> usize {
    assert!(
        F::BYTES <= SCRATCH_BYTES,
        "element width exceeds the batch scratch"
    );
    (SCRATCH_BYTES / F::BYTES).min(CHUNK_ELEMS)
}

/// `dst[i] = 1 / src[i]`, with `1 / 0 == 0`.
///
/// # Panics
///
/// Panics on a length mismatch or a partial trailing element.
pub fn inv_into<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("inv_into", "dst", dst.len(), "src", src.len());
    // The single-chunk kernel stages one prefix element per lane in a
    // 64-slot stack array; longer rows run chunk by chunk.
    let bytes = F::BYTES;
    let mut offset = 0;
    while offset < dst.len() {
        let len = (dst.len() - offset).min(CHUNK_ELEMS * bytes);
        batch_invert_into::<F>(&mut dst[offset..offset + len], &src[offset..offset + len]);
        offset += len;
    }
}

/// `dst[i] = 1 / dst[i]`, with `1 / 0 == 0`.
///
/// # Panics
///
/// Panics if `dst` holds a partial trailing element.
pub fn inv_assign<F: FieldKernels>(dst: &mut [u8]) {
    check_width::<F>("inv_assign", dst.len());
    // The backward pass reads prefix products staged ahead of the write
    // cursor; an in-place run would clobber lanes it still needs. Work
    // chunk by chunk through stack scratch instead.
    let bytes = F::BYTES;
    let mut scratch = [0u8; SCRATCH_BYTES];
    let mut offset = 0;
    while offset < dst.len() {
        let len = (dst.len() - offset).min(chunk_elems::<F>() * bytes);
        scratch[..len].copy_from_slice(&dst[offset..offset + len]);
        batch_invert_into::<F>(&mut dst[offset..offset + len], &scratch[..len]);
        offset += len;
    }
}

/// `dst[i] = a[i] / b[i]`, with `x / 0 == 0`.
///
/// # Panics
///
/// Panics on a length mismatch or a partial trailing element.
pub fn div_elementwise<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_pair::<F>("div_elementwise", "dst", dst.len(), "a", a.len());
    check_pair::<F>("div_elementwise", "dst", dst.len(), "b", b.len());
    batch_divide_into::<F>(dst, a, b);
}

/// `dst[i] /= src[i]`, with `x / 0 == 0`.
///
/// # Panics
///
/// Panics on a length mismatch or a partial trailing element.
pub fn div_elementwise_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("div_elementwise_assign", "dst", dst.len(), "src", src.len());
    // `dst` is both the dividend and the output while `src` is the
    // divisor: stage both chunks on the stack first so the kernel never
    // reads a lane it already overwrote, and never touches the heap.
    let bytes = F::BYTES;
    let mut num = [0u8; SCRATCH_BYTES];
    let mut den = [0u8; SCRATCH_BYTES];
    let mut offset = 0;
    while offset < dst.len() {
        let len = (dst.len() - offset).min(chunk_elems::<F>() * bytes);
        num[..len].copy_from_slice(&dst[offset..offset + len]);
        den[..len].copy_from_slice(&src[offset..offset + len]);
        batch_divide_into::<F>(&mut dst[offset..offset + len], &num[..len], &den[..len]);
        offset += len;
    }
}

/// Montgomery batch inversion of one `dst.len() <= CHUNK_ELEMS * BYTES`
/// chunk: prefix products as `Elem`s on the stack, one scalar inversion,
/// then a backward pass writing reciprocals. Zero lanes hold the running
/// product unchanged and read back zero.
fn batch_invert_into<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    let bytes = F::BYTES;
    let elems = src.len() / bytes;
    // `Elem<F>` is `Copy`; the prefix array costs one element per lane
    // on the stack, no heap, no decode overhead beyond the lane reads.
    let mut prefix = [Elem::<F>::ZERO; CHUNK_ELEMS];
    let mut acc = Elem::<F>::ONE;
    for (slot, lane) in prefix.iter_mut().take(elems).zip(src.chunks_exact(bytes)) {
        let value = F::decode(lane);
        if !value.is_zero() {
            acc = acc.mul(value);
        }
        *slot = acc;
    }
    // One inversion of the nonzero product. An all-zero chunk inverts the
    // identity, and every lane below still maps to zero.
    let mut suffix = acc.inv();
    for i in (0..elems).rev() {
        let value = F::decode(&src[i * bytes..(i + 1) * bytes]);
        if value.is_zero() {
            F::encode(&mut dst[i * bytes..(i + 1) * bytes], Elem::<F>::ZERO);
        } else {
            let before = if i == 0 {
                Elem::<F>::ONE
            } else {
                prefix[i - 1]
            };
            F::encode(&mut dst[i * bytes..(i + 1) * bytes], before.mul(suffix));
            suffix = suffix.mul(value);
        }
    }
}

/// Elementwise division via one batch inversion of the divisor per chunk.
fn batch_divide_into<F: FieldKernels>(dst: &mut [u8], num: &[u8], den: &[u8]) {
    let bytes = F::BYTES;
    let elems = dst.len() / bytes;
    let mut recip = [0u8; SCRATCH_BYTES];
    let mut offset = 0;
    while offset < elems {
        let count = (elems - offset).min(chunk_elems::<F>());
        let (rbuf, _) = recip.split_at_mut(count * bytes);
        batch_invert_into::<F>(rbuf, &den[offset * bytes..(offset + count) * bytes]);
        for index in 0..count {
            let lane = offset + index;
            let value = F::decode(&num[lane * bytes..(lane + 1) * bytes]);
            let inverse = F::decode(&rbuf[index * bytes..(index + 1) * bytes]);
            F::encode(
                &mut dst[lane * bytes..(lane + 1) * bytes],
                value.mul(inverse),
            );
        }
        offset += count;
    }
}
