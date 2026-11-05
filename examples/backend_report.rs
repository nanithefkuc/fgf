//! Report resolved process and field backends, optionally requiring a process tier.
//!
//! Run with `just example backend_report --all-features -- v4x` on a V4x
//! acceptance host, or omit the expected tier to report without asserting it.

#![forbid(unsafe_code)]

use fgf::binary::Polynomial;
use fgf::{Backend, FieldBuffer, FiniteField, backend, backend_for};

fn field<F: FieldBuffer>(label: &str) {
    println!("backend {label}={}", backend_for::<F>().name());
}

fn main() {
    let actual = backend();
    println!(
        "requested={}",
        std::env::var("SIMD_BACKEND").unwrap_or_else(|_| "unset".to_owned())
    );
    println!("actual={}", actual.name());
    if let Some(expected) = std::env::args().nth(1) {
        let expected: Backend = expected.parse().expect("expected tier is a backend name");
        assert_eq!(
            actual, expected,
            "the acceptance host must execute the requested tier"
        );
    }
    field::<FiniteField<2, 8, Polynomial<{ fgf::binary::AES }>>>("gf8b");
    field::<FiniteField<2, 8, Polynomial<{ fgf::binary::RS }>>>("gf8d");
    field::<FiniteField<2, 16, fgf::extension::Extension<fgf::binary::Rijndael16, 2>>>("gf16b");
    field::<FiniteField<2, 32, fgf::extension::Extension<fgf::binary::Rijndael32, 2>>>("gf32b");
    field::<FiniteField<2, 64, fgf::extension::Extension<fgf::binary::Rijndael64, 2>>>("gf64b");
    field::<fgf::prime::Mersenne31>("m31");
    field::<fgf::prime::Goldilocks>("gld");
    field::<fgf::extension::QuadMersenne31>("qm31");
    field::<FiniteField<101, 1>>("res101");
    field::<FiniteField<65_521, 1>>("res65521");
    field::<FiniteField<1_000_000_007, 1>>("res1e9p7");
    field::<FiniteField<18_446_744_073_709_551_557, 1>>("reswide");
}
