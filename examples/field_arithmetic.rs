//! Scalar arithmetic and stable encodings across every supported field.
//!
//! Run with `just example field_arithmetic`, or add `--no-default-features`
//! to exercise the portable library without its allocation feature.

#![forbid(unsafe_code)]

use fgf::field::Elem;
use fgf::gf8::{AES, REED_SOLOMON};
use fgf::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Field, Gf2, Gf8, Gf16, Gf32, Gf64, Goldilocks,
    Mersenne31, QuadMersenne31, gf2, gf8, gf16,
};

// Concrete arithmetic can also prepare constants at compile time.
const SCALE: gf16::Elem = gf16::Elem::from_raw(0x0108);
const PRODUCT: gf16::Elem = gf16::Elem::from_raw(0x1234).mul(SCALE);

fn main() {
    assert_eq!(PRODUCT.div(SCALE), gf16::Elem::from_raw(0x1234));

    // Identical raw bytes are not a conversion between different field bases.
    let aes = gf8::Elem::<AES>::from_raw(0x80) * gf8::Elem::<AES>::from_raw(2);
    let rs = gf8::Elem::<REED_SOLOMON>::from_raw(0x80) * gf8::Elem::<REED_SOLOMON>::from_raw(2);
    assert_eq!(aes.to_raw(), 0x1b);
    assert_eq!(rs.to_raw(), 0x1d);
    println!("0x80 * 2: AES basis = {aes}, Reed–Solomon basis = {rs}");

    // Gf2 is not a Field: its vector surface packs bits rather than bytes.
    assert_eq!(gf2::Elem::ONE + gf2::Elem::ONE, gf2::Elem::ZERO);
    assert_eq!(gf2::Elem::ONE * gf2::Elem::ONE, gf2::Elem::ONE);
    println!("{}: 1 + 1 = 0, 1 * 1 = 1; vectors use fgf::bits", Gf2::NAME);
    describe::<Gf8<AES>>();
    describe::<Gf8<REED_SOLOMON>>();
    describe::<Gf16>();
    describe::<Gf32>();
    describe::<Gf64>();
    describe::<FanPaar8>();
    describe::<FanPaar16>();
    describe::<FanPaar32>();
    describe::<FanPaar64>();
    describe::<Mersenne31>();
    describe::<Goldilocks>();
    describe::<QuadMersenne31>();
}

fn describe<F: Field>() {
    let a = F::GENERATOR;
    let b = a.square();
    let product = a.mul(b);
    assert_eq!(product.div(b), a);
    assert_eq!(a.add(b).sub(b), a);

    // Division is total in fgf, not evidence that zero has an inverse.
    // An algorithm requiring inversion must reject zero itself.
    assert!(!b.is_zero());
    assert_eq!(F::Elem::ZERO.inv(), F::Elem::ZERO);
    assert_eq!(a.div(F::Elem::ZERO), F::Elem::ZERO);

    // Field::encode/decode work with slices; concrete elements also have
    // fixed-size to_bytes/from_bytes methods.
    let mut storage = [0u8; 8];
    let bytes = &mut storage[..F::BYTES];
    F::encode(bytes, product);
    assert_eq!(F::decode(bytes), product);
    println!(
        "{}: characteristic={}, order={}, width={} bytes, product={product:?}, encoding={bytes:02x?}",
        F::NAME,
        F::CHARACTERISTIC,
        F::ORDER,
        F::BYTES,
    );
}
