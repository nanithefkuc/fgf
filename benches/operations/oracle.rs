//! Independent element-wise references for the benchmark cells.
//!
//! Every oracle decodes the stable byte lanes, applies the field's safe
//! scalar [`FieldElem`] arithmetic, and re-encodes. No kernel or prepared
//! coefficient state is read here, so a mismatch between a cell and its
//! oracle is a kernel defect rather than a shared implementation.

use fgf::{Elem, FieldBuffer, FieldElem};

fn assert_width<F: FieldBuffer>(name: &str, len: usize) {
    assert_eq!(
        len % F::BYTES,
        0,
        "{name}: {len} bytes is not a whole number of {}-byte elements",
        F::BYTES
    );
}

pub fn mul_add<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    assert_width::<F>("oracle::mul_add", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::decode(lane).add(coeff.mul(F::decode(source)));
        F::encode(lane, value);
    }
}

pub fn mul_into<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    dst.fill(0);
    mul_add(dst, coeff, src);
}

pub fn mul_assign<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>) {
    assert_width::<F>("oracle::mul_assign", dst.len());
    for lane in dst.chunks_exact_mut(F::BYTES) {
        let value = coeff.mul(F::decode(lane));
        F::encode(lane, value);
    }
}

pub fn elementwise_into<F: FieldBuffer>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_width::<F>("oracle::elementwise_into", dst.len());
    for (lane, (left, right)) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES).zip(b.chunks_exact(F::BYTES)))
    {
        F::encode(lane, F::decode(left).mul(F::decode(right)));
    }
}

pub fn elementwise_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    assert_width::<F>("oracle::elementwise_assign", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(lane, F::decode(lane).mul(F::decode(source)));
    }
}

pub fn add_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    assert_width::<F>("oracle::add_assign", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(lane, F::decode(lane).add(F::decode(source)));
    }
}

pub fn sub_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    assert_width::<F>("oracle::sub_assign", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(lane, F::decode(lane).sub(F::decode(source)));
    }
}

pub fn add_scalar<F: FieldBuffer>(dst: &mut [u8], value: Elem<F>) {
    assert_width::<F>("oracle::add_scalar", dst.len());
    for lane in dst.chunks_exact_mut(F::BYTES) {
        F::encode(lane, F::decode(lane).add(value));
    }
}

pub fn sub_scalar<F: FieldBuffer>(dst: &mut [u8], value: Elem<F>) {
    assert_width::<F>("oracle::sub_scalar", dst.len());
    for lane in dst.chunks_exact_mut(F::BYTES) {
        F::encode(lane, F::decode(lane).sub(value));
    }
}

pub fn inv_into<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    assert_width::<F>("oracle::inv_into", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(lane, F::decode(source).inv());
    }
}

pub fn inv_assign<F: FieldBuffer>(dst: &mut [u8]) {
    assert_width::<F>("oracle::inv_assign", dst.len());
    for lane in dst.chunks_exact_mut(F::BYTES) {
        F::encode(lane, F::decode(lane).inv());
    }
}

pub fn div_elementwise_into<F: FieldBuffer>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_width::<F>("oracle::div_elementwise_into", dst.len());
    for (lane, (left, right)) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES).zip(b.chunks_exact(F::BYTES)))
    {
        F::encode(lane, F::decode(left).div(F::decode(right)));
    }
}

pub fn div_elementwise_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    assert_width::<F>("oracle::div_elementwise_assign", dst.len());
    for (lane, source) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(lane, F::decode(lane).div(F::decode(source)));
    }
}

pub fn scatter_add<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    for (index, &coeff) in coeffs.iter().enumerate() {
        mul_add(
            &mut rows[index * row_len..(index + 1) * row_len],
            coeff,
            src,
        );
    }
}

pub fn scatter_into<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    for (index, &coeff) in coeffs.iter().enumerate() {
        mul_into(
            &mut rows[index * row_len..(index + 1) * row_len],
            coeff,
            src,
        );
    }
}

pub fn gather_add<F: FieldBuffer>(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add(dst, coeff, src);
    }
}

pub fn gather_into<F: FieldBuffer>(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
    dst.fill(0);
    gather_add(dst, coeffs, srcs);
}

pub fn matrix_add<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    row_count: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in coeffs.iter().enumerate().take(row_count) {
            mul_add(&mut rows[row * row_len..(row + 1) * row_len], coeff, src);
        }
    }
}

pub fn matrix_into<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    row_count: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    rows.fill(0);
    matrix_add(rows, row_len, row_count, terms);
}

pub fn matrix_at_add<F: FieldBuffer>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (index, &start) in row_starts.iter().enumerate() {
            mul_add(&mut dst[start..start + row_len], coeffs[index], src);
        }
    }
}

pub fn matrix_at_into<F: FieldBuffer>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &start in row_starts {
        dst[start..start + row_len].fill(0);
    }
    matrix_at_add(dst, row_len, row_starts, terms);
}

fn get_bit(buf: &[u8], bit: usize) -> u8 {
    (buf[bit / 8] >> (bit % 8)) & 1
}

fn flip_bit(buf: &mut [u8], bit: usize, value: u8) {
    let mask = 1u8 << (bit % 8);
    if value == 0 {
        buf[bit / 8] &= !mask;
    } else {
        buf[bit / 8] |= mask;
    }
}

pub fn bit_xor_assign(dst: &mut [u8], src: &[u8]) {
    for (left, &right) in dst.iter_mut().zip(src) {
        *left ^= right;
    }
}

pub fn bit_and_into(dst: &mut [u8], a: &[u8], b: &[u8]) {
    for (lane, (&left, &right)) in dst.iter_mut().zip(a.iter().zip(b)) {
        *lane = left & right;
    }
}

pub fn bit_and_assign(dst: &mut [u8], src: &[u8]) {
    for (left, &right) in dst.iter_mut().zip(src) {
        *left &= right;
    }
}

pub fn bit_andnot_assign(dst: &mut [u8], mask: &[u8]) {
    for (left, &right) in dst.iter_mut().zip(mask) {
        *left &= !right;
    }
}

pub fn bit_xor_range(dst: &mut [u8], bits: usize, from: usize, to: usize, src: &[u8]) {
    assert!(to <= bits, "oracle bit range exceeds the logical length");
    for bit in from..to {
        flip_bit(dst, bit, get_bit(dst, bit) ^ get_bit(src, bit));
    }
}

pub fn bit_clear_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    assert!(to <= bits, "oracle bit range exceeds the logical length");
    for bit in from..to {
        flip_bit(dst, bit, 0);
    }
}

pub fn bit_set_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    assert!(to <= bits, "oracle bit range exceeds the logical length");
    for bit in from..to {
        flip_bit(dst, bit, 1);
    }
}

pub fn bit_gather(dst: &mut [u8], selector: u64, srcs: &[&[u8]]) {
    for (index, &src) in srcs.iter().enumerate() {
        if selector >> index & 1 == 1 {
            bit_xor_assign(dst, src);
        }
    }
}

pub fn bit_weight(buf: &[u8], bits: usize) -> usize {
    (0..bits).map(|bit| usize::from(get_bit(buf, bit))).sum()
}

pub fn bit_dot(a: &[u8], b: &[u8], bits: usize) -> u8 {
    let mut parity = 0u8;
    for bit in 0..bits {
        parity ^= get_bit(a, bit) & get_bit(b, bit);
    }
    parity
}
