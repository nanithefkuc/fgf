//! Explore subfield embeddings, Frobenius conjugates, trace, and norm.
//!
//! Run with `just example extension_fields`, optionally with
//! `--no-default-features`. Binary trace and norm are expressed through public
//! scalar arithmetic; QuadMersenne31 also supplies conjugate and norm methods.

#![forbid(unsafe_code)]

use fgf::Field;
use fgf::{Elem, FanPaar8, FanPaar16, Gf, Gf8, Gf16, Mersenne31, Poly, QuadMersenne31};

use fgf::poly::AES;

fn main() {
    // In the AES-rooted tower, x = a + b*u and u^2 + u + DELTA = 0.
    let a = Elem::<Gf<8, Poly<AES>>>::from_raw(0x53);
    let b = Elem::<Gf<8, Poly<AES>>>::from_raw(0xca);
    let x = Elem::<Gf16>::from_components(a, b);
    let conjugate = x.pow(Gf8::<Poly<AES>>::ORDER);
    assert_eq!(conjugate, Elem::<Gf16>::from_components(a + b, b));
    assert_eq!(conjugate.pow(Gf8::<Poly<AES>>::ORDER), x);
    assert_eq!(x.pow(Gf16::ORDER), x);

    // Relative trace and norm land in the embedded byte subfield.
    let trace = x + conjugate;
    let norm = x * conjugate;
    assert_eq!(
        trace,
        Elem::<Gf16>::from_components(b, Elem::<Gf<8, Poly<AES>>>::ZERO)
    );
    let (norm_base, norm_extension) = norm.to_components();
    assert_eq!(norm_extension, Elem::<Gf<8, Poly<AES>>>::ZERO);
    assert_eq!(
        norm_base,
        a.square() + a * b + Elem::<Gf<8, Poly<AES>>>::from_raw(0x20) * b.square()
    );
    assert_eq!(x * x.inv(), Elem::<Gf16>::ONE);
    assert_eq!(x.to_bytes(), [a.to_raw(), b.to_raw()]);
    println!("AES-rooted tower: x={x}, conjugate={conjugate}, trace={trace}, norm={norm_base}");

    // Fan–Paar has another basis. Embed via components, not by relabelling
    // an AES or polynomial-basis payload as a Fan–Paar payload.
    let c = Elem::<FanPaar8>::from_raw(0x53);
    let d = Elem::<FanPaar8>::from_raw(0xca);
    let embedded_c = Elem::<FanPaar16>::from_components(c, Elem::<FanPaar8>::ZERO);
    let embedded_d = Elem::<FanPaar16>::from_components(d, Elem::<FanPaar8>::ZERO);
    assert_eq!(
        embedded_c * embedded_d,
        Elem::<FanPaar16>::from_components(c * d, Elem::<FanPaar8>::ZERO)
    );
    let y = Elem::<FanPaar16>::from_components(c, d);
    assert_eq!(y.mul_alpha(), y * Elem::<FanPaar16>::from_raw(0x0100));
    let y_conjugate = y.pow(256);
    assert_eq!((y + y_conjugate).to_components().1, Elem::<FanPaar8>::ZERO);
    assert_eq!((y * y_conjugate).to_components().1, Elem::<FanPaar8>::ZERO);
    println!(
        "Fan–Paar tower: y={y}, y * tower generator={}",
        y.mul_alpha()
    );

    // Odd-characteristic quadratic extension: z = re + im*i, i^2 = -1.
    let re = Elem::<Mersenne31>::from_raw(7);
    let im = Elem::<Mersenne31>::from_raw(11);
    let z = Elem::<QuadMersenne31>::from_components(re, im);
    assert_eq!(
        Elem::<QuadMersenne31>::I.square(),
        -Elem::<QuadMersenne31>::ONE
    );
    assert_eq!(z.pow(Mersenne31::ORDER), z.conjugate());
    assert_eq!(z.norm(), re.square() + im.square());
    assert_eq!(
        z * z.conjugate(),
        Elem::<QuadMersenne31>::from_components(z.norm(), Elem::<Mersenne31>::ZERO)
    );
    assert_eq!(z * z.inv(), Elem::<QuadMersenne31>::ONE);
    println!(
        "QuadMersenne31: z={z}, conjugate={}, norm={}",
        z.conjugate(),
        z.norm()
    );
}
