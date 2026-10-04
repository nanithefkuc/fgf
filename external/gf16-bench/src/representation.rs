//! Independent field arithmetic and validated linear maps between GF16 encodings.

use reed_solomon_erasure::{Field, galois_16};
use reed_solomon_simd::engine::{CANTOR_BASIS, tables};

use crate::native::Native;

fn polynomial_mul(mut a: u32, mut b: u32, polynomial: u32, bits: u32) -> u16 {
    let mut result = 0;
    for _ in 0..bits {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & (1 << bits) != 0 {
            a ^= polynomial;
        }
    }
    result as u16
}

fn byte_mul(a: u16, b: u16, polynomial: u32) -> u16 {
    polynomial_mul(u32::from(a), u32::from(b), polynomial, 8)
}

pub fn tower_mul(x: u16, y: u16) -> u16 {
    let (a, b, c, d) = (x & 255, x >> 8, y & 255, y >> 8);
    let ac = byte_mul(a, c, 0x11b);
    let bd = byte_mul(b, d, 0x11b);
    (ac ^ byte_mul(0x20, bd, 0x11b)) | ((byte_mul(a, d, 0x11b) ^ byte_mul(b, c, 0x11b) ^ bd) << 8)
}

fn erasure_mul(x: u16, y: u16) -> u16 {
    let (a, b, c, d) = (x >> 8, x & 255, y >> 8, y & 255);
    let ac = byte_mul(a, c, 0x11d);
    let hi = byte_mul(ac, 2, 0x11d) ^ byte_mul(a, d, 0x11d) ^ byte_mul(b, c, 0x11d);
    let lo = byte_mul(b, d, 0x11d) ^ byte_mul(ac, 128, 0x11d);
    (hi << 8) | lo
}

fn linear(x: u16, basis: &[u16; 16]) -> u16 {
    let mut result = 0;
    for (bit, &image) in basis.iter().enumerate() {
        if x & (1 << bit) != 0 {
            result ^= image;
        }
    }
    result
}

struct Map {
    forward: Vec<u16>,
}

impl Map {
    fn new(mul: impl Fn(u16, u16) -> u16) -> Self {
        let beta = (0..=u16::MAX)
            .find(|&x| {
                let x2 = mul(x, x);
                let x3 = mul(x2, x);
                let x4 = mul(x2, x2);
                let x8 = mul(x4, x4);
                x8 ^ x4 ^ x3 ^ x ^ 1 == 0
            })
            .expect("GF16 contains an AES-byte-field root");
        let mut basis = [0u16; 16];
        basis[0] = 1;
        for bit in 1..8 {
            basis[bit] = mul(basis[bit - 1], beta);
        }
        let delta = linear(0x20, &basis);
        let root = (0..=u16::MAX)
            .find(|&x| mul(x, x) ^ x == delta)
            .expect("the tower quadratic splits over GF16");
        for bit in 0..8 {
            basis[bit + 8] = mul(basis[bit], root);
        }
        let forward: Vec<_> = (0..=u16::MAX).map(|x| linear(x, &basis)).collect();
        let mut seen = vec![false; 65536];
        for &value in &forward {
            assert!(!seen[usize::from(value)], "field map must be injective");
            seen[usize::from(value)] = true;
        }
        assert_eq!(forward[0], 0);
        assert_eq!(forward[1], 1);
        for i in 0..16 {
            for j in 0..16 {
                let (x, y) = (1 << i, 1 << j);
                assert_eq!(
                    forward[usize::from(tower_mul(x, y))],
                    mul(forward[usize::from(x)], forward[usize::from(y)]),
                    "field map must preserve every basis product"
                );
            }
        }
        Self { forward }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Layout {
    Tower,
    Complete,
    Cantor,
    Erasure,
}

pub struct Representations {
    complete: Map,
    cantor: Map,
    erasure: Map,
}

impl Representations {
    pub fn new(native: &Native) -> Self {
        let mut cantor_to_poly = [0u16; 65536];
        let mut poly_to_cantor = [0u16; 65536];
        for value in 0..=u16::MAX {
            let image = linear(value, &CANTOR_BASIS);
            cantor_to_poly[usize::from(value)] = image;
            poly_to_cantor[usize::from(image)] = value;
        }
        for value in 0..=u16::MAX {
            assert_eq!(
                poly_to_cantor[usize::from(cantor_to_poly[usize::from(value)])],
                value,
                "Cantor basis must be invertible"
            );
        }
        let cantor_mul = |a: u16, b: u16| {
            let product = polynomial_mul(
                u32::from(cantor_to_poly[usize::from(a)]),
                u32::from(cantor_to_poly[usize::from(b)]),
                0x1002d,
                16,
            );
            poly_to_cantor[usize::from(product)]
        };
        for i in 0..16 {
            for j in 0..16 {
                let (x, y) = (1 << i, 1 << j);
                assert_eq!(
                    fgf::Elem::<fgf::Gf16>::from_raw(x)
                        .mul(fgf::Elem::<fgf::Gf16>::from_raw(y))
                        .to_raw(),
                    tower_mul(x, y)
                );
                assert_eq!(
                    native.complete_mul(x, y),
                    polynomial_mul(u32::from(x), u32::from(y), 0x1100b, 16)
                );
                assert_eq!(native.leopard_mul(x, y), cantor_mul(x, y));
                let lut = tables::get_exp_log();
                assert_eq!(
                    tables::mul(x, lut.log[usize::from(y)], &lut.exp, &lut.log),
                    cantor_mul(x, y)
                );
                assert_eq!(
                    u16::from_be_bytes(galois_16::Field::mul(x.to_be_bytes(), y.to_be_bytes())),
                    erasure_mul(x, y)
                );
            }
        }
        assert_eq!(tower_mul(0x100, 0x100), 0x120);
        assert_eq!(tower_mul(2, 2), 4);
        assert_eq!(cantor_mul(2, 2), 3);
        Self {
            complete: Map::new(|a, b| polynomial_mul(u32::from(a), u32::from(b), 0x1100b, 16)),
            cantor: Map::new(cantor_mul),
            erasure: Map::new(erasure_mul),
        }
    }

    pub fn element(&self, value: u16, layout: Layout) -> u16 {
        match layout {
            Layout::Tower => value,
            Layout::Complete => self.complete.forward[usize::from(value)],
            Layout::Cantor => self.cantor.forward[usize::from(value)],
            Layout::Erasure => self.erasure.forward[usize::from(value)],
        }
    }

    pub fn pack(&self, dst: &mut [u8], canonical: &[u8], layout: Layout) {
        assert_eq!(dst.len(), canonical.len());
        assert!(dst.len().is_multiple_of(64));
        for (out, input) in dst
            .as_chunks_mut::<64>()
            .0
            .iter_mut()
            .zip(canonical.as_chunks::<64>().0)
        {
            for (index, symbol) in input.as_chunks::<2>().0.iter().enumerate() {
                let value = self.element(u16::from_le_bytes([symbol[0], symbol[1]]), layout);
                match layout {
                    Layout::Cantor => {
                        out[index] = value as u8;
                        out[index + 32] = (value >> 8) as u8;
                    }
                    Layout::Erasure => {
                        out[2 * index..2 * index + 2].copy_from_slice(&value.to_be_bytes())
                    }
                    Layout::Tower | Layout::Complete => {
                        out[2 * index..2 * index + 2].copy_from_slice(&value.to_le_bytes())
                    }
                }
            }
        }
    }
}
