//! Scalar arithmetic and stable encodings across every supported field.
//!
//! Run with `just example field_arithmetic`, or add `--no-default-features`
//! to exercise the portable library without its allocation feature.

#![forbid(unsafe_code)]

use fgf::field::{FieldElem, PrimeIdentity};
use fgf::poly::{AES, REED_SOLOMON};
use fgf::{
    Elem, FanPaar8, FanPaar16, FanPaar32, FanPaar64, Field, FieldBuffer, Gf, Gf1, Gf8, Gf16, Gf32,
    Gf64, Goldilocks, HasGenerator, Mersenne31, Poly, QuadMersenne31,
};

// Concrete arithmetic can also prepare constants at compile time.
const SCALE: Elem<Gf16> = Elem::<Gf16>::from_raw(0x0108);
const PRODUCT: Elem<Gf16> = Elem::<Gf16>::from_raw(0x1234).mul(SCALE);

fn main() {
    assert_eq!(PRODUCT.div(SCALE), Elem::<Gf16>::from_raw(0x1234));

    // Identical raw bytes are not a conversion between different field bases.
    let aes = Elem::<Gf<8, Poly<AES>>>::from_raw(0x80) * Elem::<Gf<8, Poly<AES>>>::from_raw(2);
    let rs = Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0x80)
        * Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(2);
    assert_eq!(aes.to_raw(), 0x1b);
    assert_eq!(rs.to_raw(), 0x1d);
    println!("0x80 * 2: AES basis = {aes}, Reed–Solomon basis = {rs}");

    // Gf1 has no byte-per-element encoding: its vector surface packs bits
    // rather than bytes.
    assert_eq!(Elem::<Gf1>::ONE + Elem::<Gf1>::ONE, Elem::<Gf1>::ZERO);
    assert_eq!(Elem::<Gf1>::ONE * Elem::<Gf1>::ONE, Elem::<Gf1>::ONE);
    println!("{}: 1 + 1 = 0, 1 * 1 = 1; vectors use fgf::bits", Gf1::NAME);
    describe::<Gf8<Poly<AES>>>();
    describe::<Gf8<Poly<REED_SOLOMON>>>();
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

fn describe<F: FieldBuffer + HasGenerator>() {
    let a = Elem::<F>::GENERATOR;
    let b = a.square();
    let product = a.mul(b);
    assert_eq!(product.div(b), a);
    assert_eq!(a.add(b).sub(b), a);

    // Division is total in fgf, not evidence that zero has an inverse.
    // An algorithm requiring inversion must reject zero itself.
    assert!(!b.is_zero());
    assert_eq!(Elem::<F>::ZERO.inv(), Elem::<F>::ZERO);
    assert_eq!(a.div(Elem::<F>::ZERO), Elem::<F>::ZERO);

    // FieldBuffer::encode/decode work with slices; concrete elements also
    // have fixed-size to_bytes/from_bytes methods.
    let mut storage = [0u8; 8];
    let bytes = &mut storage[..F::BYTES];
    F::encode(bytes, product);
    assert_eq!(F::decode(bytes), product);
    println!(
        "{}: characteristic={}, order={}, width={} bytes, product={product:?}, encoding={bytes:02x?}",
        F::NAME,
        <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC,
        F::ORDER,
        F::BYTES,
    );
}
