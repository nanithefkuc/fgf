//! Adapters for the bit-packed GF(2) public operations.
//!
//! The geometry's `len` counts bits; buffers hold `bytes_for(len)` bytes
//! with zero padding, matching the representation contract.

use super::*;

fn packed<F>(case: &Case, salt: u64) -> (Vec<u8>, usize) {
    let _fixture = fixture_marker::<F>();
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut buffer, start) = guarded_bytes(bytes, case.geometry.misalign);
    let noise = super::super::noise(bytes, seed(case, salt) | 1);
    buffer[start..start + bytes].copy_from_slice(&noise);
    clear_padding(&mut buffer[start..start + bytes], bits);
    (buffer, start)
}

fn clear_padding(buffer: &mut [u8], bits: usize) {
    let tail = bits % 8;
    if tail != 0
        && let Some(last) = buffer.last_mut()
    {
        *last &= (1u8 << tail) - 1;
    }
}

fn bit(buffer: &[u8], index: usize) -> bool {
    buffer[index / 8] & (1 << (index % 8)) != 0
}

fn write_bit(buffer: &mut [u8], index: usize, value: bool) {
    let mask = 1u8 << (index % 8);
    if value {
        buffer[index / 8] |= mask;
    } else {
        buffer[index / 8] &= !mask;
    }
}

/// The declared geometry's full bit range.
///
/// The exercised extent equals the geometry the work budget measures, so a
/// vectorized range can satisfy its cell.
pub(super) fn range(case: &Case) -> (usize, usize) {
    (0, case.geometry.len)
}

fn pair<F>(case: &Case, call: impl Fn(&mut [u8], &[u8]), op: impl Fn(u8, u8) -> u8) -> Outcome {
    let (mut dst, dst_start) = packed::<F>(case, 0);
    let (src, src_start) = packed::<F>(case, 0x51ed_2701);
    let bytes = crate::bits::bytes_for(case.geometry.len);
    let mut expected = dst.clone();
    for index in 0..bytes {
        expected[dst_start + index] = op(dst[dst_start + index], src[src_start + index]);
    }
    let entries = observe_unit(|| {
        call(
            &mut dst[dst_start..dst_start + bytes],
            &src[src_start..src_start + bytes],
        );
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_xor_assign<F>(case: &Case, _cell: &Cell) -> Outcome {
    pair::<F>(case, crate::bits::xor_assign, |a, b| a ^ b)
}

pub(super) fn bits_and_assign<F>(case: &Case, _cell: &Cell) -> Outcome {
    pair::<F>(case, crate::bits::and_assign, |a, b| a & b)
}

pub(super) fn bits_andnot_assign<F>(case: &Case, _cell: &Cell) -> Outcome {
    pair::<F>(case, crate::bits::andnot_assign, |a, b| a & !b)
}

pub(super) fn bits_and_into<F>(case: &Case, _cell: &Cell) -> Outcome {
    let (mut dst, dst_start) = packed::<F>(case, 0);
    let (a, a_start) = packed::<F>(case, 0x51ed_2701);
    let (b, b_start) = packed::<F>(case, 0x4d2);
    let bytes = crate::bits::bytes_for(case.geometry.len);
    let mut expected = dst.clone();
    for index in 0..bytes {
        expected[dst_start + index] = a[a_start + index] & b[b_start + index];
    }
    let entries = observe_unit(|| {
        crate::bits::and_into(
            &mut dst[dst_start..dst_start + bytes],
            &a[a_start..a_start + bytes],
            &b[b_start..b_start + bytes],
        );
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_clear_range<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let (from, to) = range(case);
    let (mut dst, start) = packed::<F>(case, 0);
    let bytes = crate::bits::bytes_for(bits);
    let mut expected = dst.clone();
    for index in from..to {
        write_bit(&mut expected[start..start + bytes], index, false);
    }
    let entries = observe_unit(|| {
        crate::bits::clear_range(&mut dst[start..start + bytes], bits, from, to);
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_set_range<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut dst, start) = packed::<F>(case, 0);
    let (from, to) = range(case);
    let mut expected = dst.clone();
    for index in from..to {
        write_bit(&mut expected[start..start + bytes], index, true);
    }
    let entries = observe_unit(|| {
        crate::bits::set_range(&mut dst[start..start + bytes], bits, from, to);
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_xor_range<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut dst, start) = packed::<F>(case, 0);
    let (src, src_start) = packed::<F>(case, 0x51ed_2701);
    let (from, to) = range(case);
    let mut expected = dst.clone();
    for index in from..to {
        let value = bit(&expected[start..start + bytes], index)
            ^ bit(&src[src_start..src_start + bytes], index);
        write_bit(&mut expected[start..start + bytes], index, value);
    }
    let entries = observe_unit(|| {
        crate::bits::xor_range(
            &mut dst[start..start + bytes],
            &src[src_start..src_start + bytes],
            bits,
            from,
            to,
        );
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_xor_range_with<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut dst, start) = packed::<F>(case, 0);
    let (src, src_start) = packed::<F>(case, 0x51ed_2701);
    let (from, to) = range(case);
    let prepared = crate::bits::XorRange::new(bits, from, to);
    let mut expected = dst.clone();
    for index in from..to {
        let value = bit(&expected[start..start + bytes], index)
            ^ bit(&src[src_start..src_start + bytes], index);
        write_bit(&mut expected[start..start + bytes], index, value);
    }
    let entries = observe_unit(|| {
        crate::bits::xor_range_with(
            &mut dst[start..start + bytes],
            &src[src_start..src_start + bytes],
            &prepared,
        );
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}

pub(super) fn bits_weight<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let (buffer, start) = packed::<F>(case, 0);
    let bytes = crate::bits::bytes_for(bits);
    let expected = (0..bits)
        .filter(|&index| bit(&buffer[start..start + bytes], index))
        .count();
    let (actual, entries) = observe(|| crate::bits::weight(&buffer[start..start + bytes], bits));
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome
        .values
        .push(("weight", actual as u128, expected as u128));
    outcome
}

pub(super) fn bits_dot_product<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (a, a_start) = packed::<F>(case, 0);
    let (b, b_start) = packed::<F>(case, 0x51ed_2701);
    let parity = (0..bits)
        .filter(|&index| {
            bit(&a[a_start..a_start + bytes], index) & bit(&b[b_start..b_start + bytes], index)
        })
        .count()
        % 2;
    let (actual, entries) = observe(|| {
        crate::bits::dot_product(
            &a[a_start..a_start + bytes],
            &b[b_start..b_start + bytes],
            bits,
        )
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome
        .values
        .push(("dot_product", actual.to_index(), parity as u128));
    outcome
}

pub(super) fn bits_xor_gather<F>(case: &Case, _cell: &Cell) -> Outcome {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut dst, start) = packed::<F>(case, 0);
    let srcs: Vec<(Vec<u8>, usize)> = (0..case.geometry.sources)
        .map(|index| packed::<F>(case, 0x51ed_2701 + index as u64))
        .collect();
    let sources = case.geometry.sources;
    assert!(
        sources > 0 && sources < 64,
        "gather source count fits a selector"
    );
    let selector = (1u64 << sources) - 1;
    let mut expected = dst.clone();
    for (index, (source, source_start)) in srcs.iter().enumerate() {
        if selector & (1u64 << index) == 0 {
            continue;
        }
        for offset in 0..bytes {
            expected[start + offset] ^= source[source_start + offset];
        }
    }
    let refs: Vec<&[u8]> = srcs
        .iter()
        .map(|(source, source_start)| &source[*source_start..*source_start + bytes])
        .collect();
    let entries = observe_unit(|| {
        crate::bits::xor_gather(&mut dst[start..start + bytes], bits, selector, &refs);
    });
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.checks.push(BufferCheck {
        name: "dst",
        actual: dst,
        expected,
    });
    outcome
}
