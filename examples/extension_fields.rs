//! Explore subfield embeddings, Frobenius conjugates, trace, and norm.
//!
//! Run with `just example extension_fields`, optionally with
//! `--no-default-features`. Binary trace and norm are expressed through public
//! scalar arithmetic; QuadMersenne31 also supplies conjugate and norm methods.

#![forbid(unsafe_code)]

use fgf::Field;
use fgf::fan_paar::{fp8, fp16};
use fgf::{Gf8, Gf16, Mersenne31, gf16, mersenne31, quad_mersenne31};

use fgf::gf8::{AES, Elem};

fn main() {
    // In the AES-rooted tower, x = a + b*u and u^2 + u + DELTA = 0.
    let a = Elem::<AES>::from_raw(0x53);
    let b = Elem::<AES>::from_raw(0xca);
    let x = gf16::Elem::from_components(a, b);
    let conjugate = x.pow(Gf8::<AES>::ORDER as u64);
    assert_eq!(conjugate, gf16::Elem::from_components(a + b, b));
    assert_eq!(conjugate.pow(Gf8::<AES>::ORDER as u64), x);
    assert_eq!(x.pow(Gf16::ORDER as u64), x);

    // Relative trace and norm land in the embedded byte subfield.
    let trace = x + conjugate;
    let norm = x * conjugate;
    assert_eq!(trace, gf16::Elem::from_components(b, Elem::<AES>::ZERO));
    let (norm_base, norm_extension) = norm.to_components();
    assert_eq!(norm_extension, Elem::<AES>::ZERO);
    assert_eq!(norm_base, a.square() + a * b + gf16::DELTA * b.square());
    assert_eq!(x * x.inv(), gf16::Elem::ONE);
    assert_eq!(x.to_bytes(), [a.to_raw(), b.to_raw()]);
    println!("AES-rooted tower: x={x}, conjugate={conjugate}, trace={trace}, norm={norm_base}");

    // Fan–Paar has another basis. Embed via components, not by relabelling
    // an AES or polynomial-basis payload as a Fan–Paar payload.
    let c = fp8::Elem::from_raw(0x53);
    let d = fp8::Elem::from_raw(0xca);
    let embedded_c = fp16::Elem::from_components(c, fp8::Elem::ZERO);
    let embedded_d = fp16::Elem::from_components(d, fp8::Elem::ZERO);
    assert_eq!(
        embedded_c * embedded_d,
        fp16::Elem::from_components(c * d, fp8::Elem::ZERO)
    );
    let y = fp16::Elem::from_components(c, d);
    assert_eq!(y.mul_alpha(), y * fp16::ALPHA);
    let y_conjugate = y.pow(256);
    assert_eq!((y + y_conjugate).to_components().1, fp8::Elem::ZERO);
    assert_eq!((y * y_conjugate).to_components().1, fp8::Elem::ZERO);
    println!(
        "Fan–Paar tower: y={y}, y * tower generator={}",
        y.mul_alpha()
    );

    // Odd-characteristic quadratic extension: z = re + im*i, i^2 = -1.
    let re = mersenne31::Elem::from_raw(7);
    let im = mersenne31::Elem::from_raw(11);
    let z = quad_mersenne31::Elem::from_components(re, im);
    assert_eq!(
        quad_mersenne31::Elem::I.square(),
        -quad_mersenne31::Elem::ONE
    );
    assert_eq!(z.pow(Mersenne31::CHARACTERISTIC), z.conjugate());
    assert_eq!(z.norm(), re.square() + im.square());
    assert_eq!(
        z * z.conjugate(),
        quad_mersenne31::Elem::from_components(z.norm(), mersenne31::Elem::ZERO)
    );
    assert_eq!(z * z.inv(), quad_mersenne31::Elem::ONE);
    println!(
        "QuadMersenne31: z={z}, conjugate={}, norm={}",
        z.conjugate(),
        z.norm()
    );
}
