//! Production-route smoke tests: one public call per bulk family, checked
//! against an independent element-wise reference.
//!
//! These run against the ordinary production library at a vector-sized
//! geometry. They carry no trace assertions and no dispatch expectations;
//! the backend-specific machinery lives in the crate's route runner.
#![allow(clippy::chunks_exact_to_as_chunks)]

use fgf::bits;
use fgf::field::{Elem, FieldBuffer, FieldElem, FiniteField, Polynomial, Residue};
use fgf::ops;
use fgf::{Extension, Frobenius, Mersenne31, PreparedEmbedding, Rijndael16};

type Aes = FiniteField<2, 8, Polynomial<0x11B>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type R65537 = FiniteField<65537, 1, Residue>;
type Gf1 = FiniteField<2, 1, Polynomial<3>>;

/// A vector-sized element count.
const LEN: usize = 512;

/// Deterministic, canonical element pattern.
fn pattern<F: FieldBuffer>(len: usize, salt: u64) -> Vec<u8> {
    // An odd, bijective mixer: distinct salts stay distinct, including
    // adjacent ones used for consecutive source rows.
    let mut state = salt
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x1234_5678);
    let order = <F as fgf::field::Field>::ORDER;
    let mut out = vec![0u8; len * F::BYTES];
    for lane in out.chunks_exact_mut(F::BYTES) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let index = u128::from(state) % order;
        F::encode(
            lane,
            Elem::<F>::from_index(index).expect("index below the order"),
        );
    }
    out
}

#[test]
fn smoke_sources_are_distinct() {
    type F = Aes;
    assert_ne!(pattern::<F>(LEN, 0xc0), pattern::<F>(LEN, 0xc1));
    assert_ne!(pattern::<F>(LEN, 0x11), pattern::<F>(LEN, 0x22));
    assert_ne!(pattern::<F>(LEN, 0), pattern::<F>(LEN, 1));
}

fn element<F: FieldBuffer>(bytes: &[u8], index: usize) -> Elem<F> {
    F::decode(&bytes[index * F::BYTES..(index + 1) * F::BYTES])
}

fn coefficient<F: FieldBuffer>() -> Elem<F> {
    if <F as fgf::field::Field>::ORDER > 4 {
        Elem::<F>::from_index(3).expect("index below the order")
    } else {
        Elem::<F>::ONE
    }
}

#[test]
fn row_mul_add_matches_elementwise_reference() {
    type F = Aes;
    let mut dst = pattern::<F>(LEN, 0x11);
    let src = pattern::<F>(LEN, 0x22);
    let coeff = coefficient::<F>();
    let original = dst.clone();
    ops::mul_add::<F>(&mut dst, coeff, &src);
    for index in 0..LEN {
        let expected = element::<F>(&original, index).add(coeff.mul(element::<F>(&src, index)));
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn rows_add_assign_rows_matches_elementwise_reference() {
    type F = R65537;
    let rows = 4;
    let row_len = 64;
    let mut dst = pattern::<F>(rows * row_len, 0x33);
    let src = pattern::<F>(rows * row_len, 0x44);
    let original = dst.clone();
    ops::add_assign_rows::<F>(&mut dst, row_len * F::BYTES, &src);
    for index in 0..rows * row_len {
        let expected = element::<F>(&original, index).add(element::<F>(&src, index));
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn scatter_mul_add_matches_elementwise_reference() {
    type F = Gf16;
    let rows = 4;
    let mut dst = pattern::<F>(rows * LEN, 0x55);
    let src = pattern::<F>(LEN, 0x66);
    let coeffs: Vec<Elem<F>> = (1..=rows)
        .map(|index| Elem::<F>::from_index(index as u128).expect("index below the order"))
        .collect();
    let original = dst.clone();
    ops::mul_add_scatter::<F>(&mut dst, LEN * F::BYTES, &coeffs, &src);
    for (row, &coeff) in coeffs.iter().enumerate() {
        for index in 0..LEN {
            let at = row * LEN + index;
            let expected = element::<F>(&original, at).add(coeff.mul(element::<F>(&src, index)));
            assert_eq!(element::<F>(&dst, at), expected, "row {row} lane {index}");
        }
    }
}

#[test]
fn gather_mul_add_matches_elementwise_reference() {
    type F = Aes;
    let sources = 4;
    let mut dst = pattern::<F>(LEN, 0x77);
    let srcs: Vec<Vec<u8>> = (0..sources)
        .map(|index| pattern::<F>(LEN, 0x80 + index as u64))
        .collect();
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let coeffs: Vec<Elem<F>> = (1..=sources)
        .map(|index| Elem::<F>::from_index(index as u128).expect("index below the order"))
        .collect();
    let original = dst.clone();
    ops::mul_add_gather::<F>(&mut dst, &coeffs, &refs);
    for index in 0..LEN {
        let mut expected = element::<F>(&original, index);
        for source in 0..sources {
            expected = expected.add(coeffs[source].mul(element::<F>(&srcs[source], index)));
        }
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn matrix_mul_add_matches_elementwise_reference() {
    type F = Gf16;
    let rows = 3;
    let sources = 4;
    let mut dst = pattern::<F>(rows * LEN, 0x99);
    let srcs: Vec<Vec<u8>> = (0..sources)
        .map(|index| pattern::<F>(LEN, 0xa0 + index as u64))
        .collect();
    let coeffs: Vec<Vec<Elem<F>>> = (0..sources)
        .map(|source| {
            (0..rows)
                .map(|row| {
                    Elem::<F>::from_index((1 + row + 3 * source) as u128)
                        .expect("index below the order")
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<F>], &[u8])> = coeffs
        .iter()
        .map(Vec::as_slice)
        .zip(srcs.iter().map(Vec::as_slice))
        .collect();
    let original = dst.clone();
    ops::mul_add_matrix::<F>(&mut dst, LEN * F::BYTES, rows, &terms);
    let by_row: Vec<Vec<Elem<F>>> = (0..rows)
        .map(|row| (0..sources).map(|source| coeffs[source][row]).collect())
        .collect();
    for (row, row_coeffs) in by_row.iter().enumerate() {
        for index in 0..LEN {
            let at = row * LEN + index;
            let mut expected = element::<F>(&original, at);
            for (source, &coeff) in row_coeffs.iter().enumerate() {
                expected = expected.add(coeff.mul(element::<F>(&srcs[source], index)));
            }
            assert_eq!(element::<F>(&dst, at), expected, "row {row} lane {index}");
        }
    }
}

#[test]
fn matrix_at_mul_add_matches_elementwise_reference() {
    type F = R65537;
    let rows = 3;
    let sources = 2;
    let pitch = LEN + 8;
    let mut dst = pattern::<F>(rows * pitch, 0xb1);
    let starts: Vec<usize> = (0..rows).map(|row| row * pitch * F::BYTES).collect();
    let srcs: Vec<Vec<u8>> = (0..sources)
        .map(|index| pattern::<F>(LEN, 0xc0 + index as u64))
        .collect();
    let coeffs: Vec<Vec<Elem<F>>> = (0..sources)
        .map(|source| {
            (0..rows)
                .map(|row| {
                    Elem::<F>::from_index((2 + row + source) as u128)
                        .expect("index below the order")
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<F>], &[u8])> = coeffs
        .iter()
        .map(Vec::as_slice)
        .zip(srcs.iter().map(Vec::as_slice))
        .collect();
    let original = dst.clone();
    ops::mul_add_matrix_at::<F>(&mut dst, LEN * F::BYTES, &starts, &terms);
    let by_row: Vec<Vec<Elem<F>>> = (0..rows)
        .map(|row| (0..sources).map(|source| coeffs[source][row]).collect())
        .collect();
    for (row, row_coeffs) in by_row.iter().enumerate() {
        for index in 0..LEN {
            let at = row * pitch + index;
            let mut expected = element::<F>(&original, at);
            for (source, &coeff) in row_coeffs.iter().enumerate() {
                expected = expected.add(coeff.mul(element::<F>(&srcs[source], index)));
            }
            assert_eq!(element::<F>(&dst, at), expected, "row {row} lane {index}");
        }
        for index in LEN..pitch {
            let at = row * pitch + index;
            assert_eq!(
                &dst[at * F::BYTES..(at + 1) * F::BYTES],
                &original[at * F::BYTES..(at + 1) * F::BYTES]
            );
        }
    }
}

#[test]
fn reduction_dot_product_matches_elementwise_reference() {
    type F = Mersenne31;
    let a = pattern::<F>(LEN, 0xd1);
    let b = pattern::<F>(LEN, 0xd2);
    let mut expected = Elem::<F>::ZERO;
    for index in 0..LEN {
        expected = expected.add(element::<F>(&a, index).mul(element::<F>(&b, index)));
    }
    assert_eq!(ops::dot_product::<F>(&a, &b), expected);
}

#[test]
fn elementwise_mul_matches_elementwise_reference() {
    type F = Aes;
    let a = pattern::<F>(LEN, 0xe1);
    let b = pattern::<F>(LEN, 0xe2);
    let mut dst = vec![0u8; LEN * F::BYTES];
    ops::mul_elementwise::<F>(&mut dst, &a, &b);
    for index in 0..LEN {
        let expected = element::<F>(&a, index).mul(element::<F>(&b, index));
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn scalar_add_assign_scalar_matches_elementwise_reference() {
    type F = Gf16;
    let mut dst = pattern::<F>(LEN, 0xf1);
    let value = coefficient::<F>();
    let original = dst.clone();
    ops::add_assign_scalar::<F>(&mut dst, value);
    for index in 0..LEN {
        let expected = element::<F>(&original, index).add(value);
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn pack_round_trip_matches_elements() {
    type F = Aes;
    let bytes = pattern::<F>(LEN, 0x12);
    let elements: Vec<Elem<F>> = bytes.chunks_exact(F::BYTES).map(F::decode).collect();
    let packed = ops::pack_to_vec::<F>(&elements);
    assert_eq!(packed, bytes);
    let unpacked = ops::unpack_to_vec::<F>(&bytes);
    assert_eq!(unpacked, elements);
}

#[test]
fn bits_xor_range_matches_reference() {
    let bits_len = LEN * 8;
    let mut dst = vec![0u8; bits::bytes_for(bits_len)];
    let mut src = vec![0u8; bits::bytes_for(bits_len)];
    for (index, byte) in dst.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    for (index, byte) in src.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(53).wrapping_add(7);
    }
    let original = dst.clone();
    let (from, to) = (13, 1009);
    bits::xor_range(&mut dst, &src, bits_len, from, to);
    for index in 0..bits_len {
        let expected = if (from..to).contains(&index) {
            let d = original[index / 8] & (1 << (index % 8)) != 0;
            let s = src[index / 8] & (1 << (index % 8)) != 0;
            d ^ s
        } else {
            original[index / 8] & (1 << (index % 8)) != 0
        };
        assert_eq!(
            dst[index / 8] & (1 << (index % 8)) != 0,
            expected,
            "bit {index}"
        );
    }
}

#[test]
fn bits_gather_matches_reference() {
    let bits_len = LEN * 8;
    let mut dst = vec![0x5au8; bits::bytes_for(bits_len)];
    let srcs: Vec<Vec<u8>> = (0..4u8)
        .map(|source| {
            (0..bits::bytes_for(bits_len))
                .map(|index| index as u8 ^ source)
                .collect()
        })
        .collect();
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let selector = 0b1011u64;
    let original = dst.clone();
    bits::xor_gather(&mut dst, bits_len, selector, &refs);
    for (index, byte) in original.iter().enumerate() {
        let mut expected = *byte;
        for (source, block) in srcs.iter().enumerate() {
            if selector & (1 << source) != 0 {
                expected ^= block[index];
            }
        }
        assert_eq!(dst[index], expected, "byte {index}");
    }
}

#[test]
fn map_frobenius_matches_element_pow() {
    type F = Aes;
    let src = pattern::<F>(LEN, 0x31);
    let mut dst = vec![0u8; LEN * F::BYTES];
    let map = Frobenius::<F>::new(1);
    ops::map::frobenius_into_with::<F>(&mut dst, &map, &src);
    for index in 0..LEN {
        let expected = element::<F>(&src, index).pow(2);
        assert_eq!(element::<F>(&dst, index), expected, "lane {index}");
    }
}

#[test]
fn map_bits_embed_matches_reference() {
    type S = Gf1;
    type T = Aes;
    let bits_len = LEN * 8;
    let packed = vec![0xa5u8; bits::bytes_for(bits_len)];
    let mut lanes = vec![0u8; bits_len * T::BYTES];
    let map = PreparedEmbedding::<S, T>::new(
        &fgf::field::Embedding::<S, T>::new().expect("GF(2) embeds in every binary row"),
    );
    ops::map_bits::embed_into_with::<S, T>(&mut lanes, &map, &packed, bits_len);
    for index in 0..bits_len {
        let expected = if packed[index / 8] & (1 << (index % 8)) != 0 {
            Elem::<T>::ONE
        } else {
            Elem::<T>::ZERO
        };
        assert_eq!(element::<T>(&lanes, index), expected, "bit {index}");
    }
}
