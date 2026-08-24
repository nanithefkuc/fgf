//! Scalar field axioms and cross-backend algebra checks.
//!
//! The independent oracle for GF(2^8) is shift-and-XOR; every tower oracle
//! uses schoolbook expansion over its already-tested base field. Nothing here
//! uses the Karatsuba form under test, so reduction bugs remain visible rather
//! than self-consistent.

use fgf::field::Field;
use fgf::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8B, Gf8D, Gf16, Gf32, Gf64, Goldilocks,
    Mersenne31, QuadMersenne31, fan_paar, gf2, gf8b, gf8d, gf16, gf32, gf64, goldilocks,
    mersenne31, quad_mersenne31,
};

/// Every nonzero element, plus zero, in ascending order.
fn all_gf8() -> impl Iterator<Item = gf8b::Elem> {
    (0..=u8::MAX).map(gf8b::Elem)
}

/// A spread of GF(2^16) elements: boundaries, both component planes, and a
/// deterministic pseudo-random spray. Exhaustive would be 4 billion pairs.
fn sample_gf16() -> Vec<gf16::Elem> {
    let mut values = vec![
        gf16::Elem(0),
        gf16::Elem(1),
        gf16::Elem(0x0100),
        gf16::Elem(0x00ff),
        gf16::Elem(0xff00),
        gf16::Elem(0xffff),
        gf16::GENERATOR,
    ];
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        values.push(gf16::Elem((state >> 32) as u16));
    }
    values
}

// ---------------------------------------------------------------------------
// GF(2^8)
// ---------------------------------------------------------------------------

#[test]
fn gf8_table_multiply_matches_shift_and_xor() {
    for a in all_gf8() {
        for b in all_gf8() {
            assert_eq!(
                a.mul(b),
                a.mul_xtime(b),
                "table and xtime disagree on {a:?} * {b:?}"
            );
        }
    }
}

#[test]
fn gf8_inverse_matches_fermat_and_round_trips() {
    assert_eq!(
        gf8b::Elem::ZERO.inv_xtime(),
        gf8b::Elem::ZERO,
        "inv_xtime(0) must be 0"
    );
    assert_eq!(gf8b::Elem::ZERO.inv(), gf8b::Elem::ZERO, "inv(0) must be 0");
    for a in all_gf8().skip(1) {
        assert_eq!(a.inv(), a.inv_xtime(), "inverse backends disagree on {a:?}");
        assert_eq!(a.mul(a.inv()), gf8b::Elem::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), gf8b::Elem::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf8_generator_has_full_order() {
    // 0x03 must generate all 255 nonzero elements and no fewer.
    let mut seen = [false; 256];
    let mut value = gf8b::Elem::ONE;
    for step in 0..255u32 {
        assert!(!seen[value.0 as usize], "generator repeats at step {step}");
        seen[value.0 as usize] = true;
        value = value.mul(Gf8B::GENERATOR);
    }
    assert_eq!(value, gf8b::Elem::ONE, "generator order is not 255");
    assert!(
        seen.iter().skip(1).all(|&hit| hit),
        "orbit misses an element"
    );
}

#[test]
fn gf8_field_axioms() {
    let sample: Vec<_> = all_gf8().step_by(7).collect();
    for &a in &sample {
        assert_eq!(a.add(gf8b::Elem::ZERO), a);
        assert_eq!(a.mul(gf8b::Elem::ONE), a);
        assert_eq!(a.mul(gf8b::Elem::ZERO), gf8b::Elem::ZERO);
        assert_eq!(a.add(a), gf8b::Elem::ZERO, "characteristic two");
        assert_eq!(a.sub(a), gf8b::Elem::ZERO);
        for &b in &sample {
            assert_eq!(a.add(b), b.add(a), "addition commutes");
            assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
            for &c in &sample {
                assert_eq!(a.add(b).add(c), a.add(b.add(c)), "addition associates");
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "multiplication associates"
                );
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "multiplication distributes"
                );
            }
        }
    }
}

#[test]
fn gf8_pow_matches_repeated_multiplication() {
    for a in all_gf8().step_by(11) {
        let mut expected = gf8b::Elem::ONE;
        for exponent in 0..20u64 {
            assert_eq!(a.pow(exponent), expected, "{a:?}^{exponent}");
            expected = expected.mul(a);
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^8) under 0x11D
// ---------------------------------------------------------------------------

/// Every element of the 0x11D field, in ascending raw order.
fn all_gf8d() -> impl Iterator<Item = gf8d::Elem> {
    (0..=u8::MAX).map(gf8d::Elem)
}

#[test]
fn gf8d_table_multiply_matches_shift_and_xor() {
    for a in all_gf8d() {
        for b in all_gf8d() {
            assert_eq!(
                a.mul(b),
                a.mul_xtime(b),
                "table and xtime disagree on {a:?} * {b:?}"
            );
        }
    }
}

#[test]
fn gf8d_inverse_matches_fermat_and_round_trips() {
    assert_eq!(
        gf8d::Elem::ZERO.inv_xtime(),
        gf8d::Elem::ZERO,
        "inv_xtime(0) must be 0"
    );
    assert_eq!(gf8d::Elem::ZERO.inv(), gf8d::Elem::ZERO, "inv(0) must be 0");
    for a in all_gf8d().skip(1) {
        assert_eq!(a.inv(), a.inv_xtime(), "inverse backends disagree on {a:?}");
        assert_eq!(a.mul(a.inv()), gf8d::Elem::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), gf8d::Elem::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf8d_generator_has_full_order() {
    // Under 0x11D the primitive element is x = 2, not 3.
    assert_eq!(Gf8D::GENERATOR, gf8d::Elem(0x02), "0x11D generator is 2");
    let mut seen = [false; 256];
    let mut value = gf8d::Elem::ONE;
    for step in 0..255u32 {
        assert!(!seen[value.0 as usize], "generator repeats at step {step}");
        seen[value.0 as usize] = true;
        value = value.mul(Gf8D::GENERATOR);
    }
    assert_eq!(value, gf8d::Elem::ONE, "generator order is not 255");
    assert!(
        seen.iter().skip(1).all(|&hit| hit),
        "orbit misses an element"
    );
}

#[test]
fn gf8d_field_axioms() {
    let sample: Vec<_> = all_gf8d().step_by(7).collect();
    for &a in &sample {
        assert_eq!(a.add(gf8d::Elem::ZERO), a);
        assert_eq!(a.mul(gf8d::Elem::ONE), a);
        assert_eq!(a.mul(gf8d::Elem::ZERO), gf8d::Elem::ZERO);
        assert_eq!(a.add(a), gf8d::Elem::ZERO, "characteristic two");
        assert_eq!(a.sub(a), gf8d::Elem::ZERO);
        for &b in &sample {
            assert_eq!(a.add(b), b.add(a), "addition commutes");
            assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
            for &c in &sample {
                assert_eq!(a.add(b).add(c), a.add(b.add(c)), "addition associates");
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "multiplication associates"
                );
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "multiplication distributes"
                );
            }
        }
    }
}

#[test]
fn gf8d_pow_matches_repeated_multiplication() {
    for a in all_gf8d().step_by(11) {
        let mut expected = gf8d::Elem::ONE;
        for exponent in 0..20u64 {
            assert_eq!(a.pow(exponent), expected, "{a:?}^{exponent}");
            expected = expected.mul(a);
        }
    }
}

#[test]
fn gf8d_known_answer_products() {
    // The 0x11D field used by ISA-L / klauspost-reedsolomon, independently
    // computed from the shift/XOR oracle.
    assert_eq!(gf8d::Elem(0x53).mul(gf8d::Elem(0xca)), gf8d::Elem(0x8f));
    assert_eq!(gf8d::Elem(0x57).mul(gf8d::Elem(0x83)), gf8d::Elem(0x31));
    assert_eq!(gf8d::Elem(0xff).mul(gf8d::Elem(0xff)), gf8d::Elem(0xe2));
}

#[test]
fn field_polynomials_are_introspectable() {
    assert_eq!(Gf8B::field_poly(), 0x11B);
    assert_eq!(Gf8D::field_poly(), 0x11D);
    assert_eq!(gf8b::REDUCTION_POLY, 0x11B);
    assert_eq!(gf8d::REDUCTION_POLY, 0x11D);
}

#[test]
fn gf8d_is_distinct_from_gf8b() {
    // Same raw bytes, genuinely different products: the guard against filling
    // the 0x11D tables from the 0x11B arithmetic.
    assert_eq!(gf8b::Elem(0x53).mul(gf8b::Elem(0xca)), gf8b::Elem(0x01));
    assert_eq!(gf8b::Elem(0xff).mul(gf8b::Elem(0xff)), gf8b::Elem(0x13));
    let differ = (0u16..=255)
        .flat_map(|a| (0u16..=255).map(move |b| (a as u8, b as u8)))
        .filter(|&(a, b)| {
            gf8b::Elem(a).mul(gf8b::Elem(b)).to_raw() != gf8d::Elem(a).mul(gf8d::Elem(b)).to_raw()
        })
        .count();
    assert_eq!(
        differ, 63_232,
        "0x11B and 0x11D must differ on most products"
    );
}

// ---------------------------------------------------------------------------
// GF(2^16)
// ---------------------------------------------------------------------------

/// Schoolbook `(a + b*u)(c + d*u)` reduced by `u^2 = u + DELTA`, using the
/// GF(2^8) shift-and-XOR multiply. Independent of the Karatsuba form under
/// test and of the log tables.
fn gf16_mul_oracle(x: gf16::Elem, y: gf16::Elem) -> gf16::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul_xtime(c);
    let ad = a.mul_xtime(d);
    let bc = b.mul_xtime(c);
    let bd = b.mul_xtime(d);
    // ac + (ad + bc)u + bd*u^2, and u^2 = u + DELTA.
    let constant = ac.add(gf16::DELTA.mul_xtime(bd));
    let extension = ad.add(bc).add(bd);
    gf16::Elem::from_components(constant, extension)
}

#[test]
fn gf16_karatsuba_matches_schoolbook() {
    let sample = sample_gf16();
    for &a in &sample {
        for &b in &sample {
            assert_eq!(a.mul(b), gf16_mul_oracle(a, b), "{a:?} * {b:?}");
        }
    }
}

#[test]
fn gf16_square_matches_multiply() {
    for a in sample_gf16() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
    }
}

#[test]
fn gf16_inverse_round_trips() {
    assert_eq!(gf16::Elem::ZERO.inv(), gf16::Elem::ZERO, "inv(0) must be 0");
    for a in sample_gf16() {
        if a == gf16::Elem::ZERO {
            continue;
        }
        assert_eq!(a.mul(a.inv()), gf16::Elem::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), gf16::Elem::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf16_generator_has_full_order() {
    // Order must be exactly 65535: g^65535 == 1 and g^(65535/p) != 1 for each
    // prime factor p of 65535 = 3 * 5 * 17 * 257.
    let g = Gf16::GENERATOR;
    assert_eq!(g.pow(65_535), gf16::Elem::ONE, "g^65535 != 1");
    for factor in [3u64, 5, 17, 257] {
        assert_ne!(
            g.pow(65_535 / factor),
            gf16::Elem::ONE,
            "generator order divides 65535/{factor}"
        );
    }
}

#[test]
fn gf16_field_axioms() {
    let sample: Vec<_> = sample_gf16().into_iter().step_by(37).collect();
    for &a in &sample {
        assert_eq!(a.add(gf16::Elem::ZERO), a);
        assert_eq!(a.mul(gf16::Elem::ONE), a);
        assert_eq!(a.mul(gf16::Elem::ZERO), gf16::Elem::ZERO);
        assert_eq!(a.add(a), gf16::Elem::ZERO, "characteristic two");
        for &b in &sample {
            assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
            for &c in &sample {
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "multiplication associates"
                );
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "multiplication distributes"
                );
            }
        }
    }
}

#[test]
fn gf16_embeds_the_base_field() {
    // Elements with a zero extension component must multiply exactly as
    // GF(2^8) does. If the tower reduction were wrong this would break.
    for a in all_gf8().step_by(5) {
        for b in all_gf8().step_by(7) {
            let lifted = gf16::Elem::from_components(a, b)
                .mul(gf16::Elem::from_components(gf8b::Elem(0), gf8b::Elem(0)));
            assert_eq!(lifted, gf16::Elem::ZERO);

            let x = gf16::Elem::from_components(a, gf8b::Elem(0));
            let y = gf16::Elem::from_components(b, gf8b::Elem(0));
            assert_eq!(
                x.mul(y),
                gf16::Elem::from_components(a.mul(b), gf8b::Elem(0)),
                "base-field embedding broken for {a:?} * {b:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^32) and GF(2^64)
// ---------------------------------------------------------------------------

fn sample_gf32() -> Vec<gf32::Elem> {
    let mut values = vec![
        gf32::Elem::ZERO,
        gf32::Elem::ONE,
        gf32::Elem(u32::MAX),
        gf32::Elem(0x0000_ffff),
        gf32::Elem(0xffff_0000),
        gf32::Elem::from_components(gf32::DELTA, gf16::Elem::ZERO),
        Gf32::GENERATOR,
    ];
    let mut state = 0x243f_6a88u32;
    for _ in 0..48 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(gf32::Elem(state));
    }
    values
}

fn sample_gf64() -> Vec<gf64::Elem> {
    let mut values = vec![
        gf64::Elem::ZERO,
        gf64::Elem::ONE,
        gf64::Elem(u64::MAX),
        gf64::Elem(0x0000_0000_ffff_ffff),
        gf64::Elem(0xffff_ffff_0000_0000),
        Gf64::GENERATOR,
    ];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(gf64::Elem(state));
    }
    values
}

fn gf32_mul_oracle(x: gf32::Elem, y: gf32::Elem) -> gf32::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    gf32::Elem::from_components(ac.add(gf32::DELTA.mul(bd)), ad.add(bc).add(bd))
}

fn gf64_mul_oracle(x: gf64::Elem, y: gf64::Elem) -> gf64::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    gf64::Elem::from_components(ac.add(gf64::DELTA.mul(bd)), ad.add(bc).add(bd))
}

#[test]
fn gf32_tower_arithmetic() {
    let sample = sample_gf32();
    for (i, &a) in sample.iter().enumerate() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
        assert_eq!(a.add(a), gf32::Elem::ZERO);
        if a != gf32::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), gf32::Elem::ONE, "inverse({a:?})");
        }
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        assert_eq!(a.mul(b), gf32_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }
}

#[test]
fn gf64_tower_arithmetic() {
    let sample = sample_gf64();
    for (i, &a) in sample.iter().enumerate() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
        assert_eq!(a.add(a), gf64::Elem::ZERO);
        if a != gf64::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), gf64::Elem::ONE, "inverse({a:?})");
        }
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        assert_eq!(a.mul(b), gf64_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }
}

#[test]
fn larger_tower_generators_have_full_order() {
    let g32 = Gf32::GENERATOR;
    let order32 = u32::MAX as u64;
    assert_eq!(g32.pow(order32), gf32::Elem::ONE);
    for factor in [3u64, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), gf32::Elem::ONE);
    }

    let g64 = Gf64::GENERATOR;
    let order64 = u64::MAX;
    assert_eq!(g64.pow(order64), gf64::Elem::ONE);
    for factor in [3u64, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(g64.pow(order64 / factor), gf64::Elem::ONE);
    }
}

// ---------------------------------------------------------------------------
// Prime fields GF(2^31 - 1) and GF(2^64 - 2^32 + 1)
// ---------------------------------------------------------------------------

// The independent oracle is `u128 % p` schoolbook arithmetic: no fold, no
// Shoup, no lane tricks, so a reduction bug stays visible instead of being
// self-consistent with the implementation under test.
const M31_P: u128 = 0x7FFF_FFFF;
const GLD_P: u128 = 0xFFFF_FFFF_0000_0001;

fn sample_m31() -> Vec<mersenne31::Elem> {
    // Canonical boundaries, non-canonical raw lanes (>= p), and a spray.
    let mut values: Vec<mersenne31::Elem> = [
        0,
        1,
        2,
        3,
        0x7FFF_FFFD,
        0x7FFF_FFFE, // 0, 1, 2, 3, p-2, p-1
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF, // p, p+1, 2p+1 (non-canonical)
        0x5555_5555,
        0xAAAA_AAAA,
        7,
    ]
    .into_iter()
    .map(mersenne31::Elem)
    .collect();
    let mut state = 0x243f_6a88u32;
    for _ in 0..64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(mersenne31::Elem(state));
    }
    values
}

fn sample_gld() -> Vec<goldilocks::Elem> {
    let mut values: Vec<goldilocks::Elem> = [
        0,
        1,
        2,
        3,
        0xFFFF_FFFE_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000, // p-2, p-1
        0xFFFF_FFFF_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0xFFFF_FFFF_FFFF_FFFF, // p, p+1, 2^64-1
        0x5555_5555_5555_5555,
        0xAAAA_AAAA_AAAA_AAAA,
        7,
    ]
    .into_iter()
    .map(goldilocks::Elem)
    .collect();
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..48 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(goldilocks::Elem(state));
    }
    values
}

fn m31_mul_oracle(x: mersenne31::Elem, y: mersenne31::Elem) -> mersenne31::Elem {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    mersenne31::Elem((a * b % M31_P) as u32)
}
fn m31_add_oracle(x: mersenne31::Elem, y: mersenne31::Elem) -> mersenne31::Elem {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    mersenne31::Elem(((a + b) % M31_P) as u32)
}
fn m31_sub_oracle(x: mersenne31::Elem, y: mersenne31::Elem) -> mersenne31::Elem {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    mersenne31::Elem(((a + M31_P - b) % M31_P) as u32)
}
fn gld_mul_oracle(x: goldilocks::Elem, y: goldilocks::Elem) -> goldilocks::Elem {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    goldilocks::Elem((a * b % GLD_P) as u64)
}
fn gld_add_oracle(x: goldilocks::Elem, y: goldilocks::Elem) -> goldilocks::Elem {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    goldilocks::Elem(((a + b) % GLD_P) as u64)
}
fn gld_sub_oracle(x: goldilocks::Elem, y: goldilocks::Elem) -> goldilocks::Elem {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    goldilocks::Elem(((a + GLD_P - b) % GLD_P) as u64)
}

#[test]
fn m31_known_answer_products() {
    use mersenne31::Elem;
    assert_eq!(Elem(0x5555_5555).mul(Elem(0x5555_5555)), Elem(0x71C7_1C71));
    assert_eq!(Elem(0x5555_5555).mul(Elem(0x7FFF_FFFE)), Elem(0x2AAA_AAAA));
    assert_eq!(Elem(0x7FFF_FFFE).mul(Elem(0x7FFF_FFFE)), Elem(0x0000_0001));
    assert_eq!(Elem(0x7FFF_FFFD).mul(Elem(0x7FFF_FFFE)), Elem(0x0000_0002));
    assert_eq!(Elem(0x5555_5555).mul(Elem(0x5EAD_BEF0)), Elem(0x74E4_94FA));
    assert_eq!(Elem(2).inv(), Elem(0x4000_0000));
    assert_eq!(Elem(0x5555_5555).inv(), Elem(3));
    // Fold known-answers: non-canonical lanes reduce branchlessly.
    assert_eq!(mersenne31::reduce(0x8000_0000), 1);
    assert_eq!(mersenne31::reduce(0xFFFF_FFFF), 1);
    assert_eq!(mersenne31::reduce(0x7FFF_FFFF), 0);
}

#[test]
fn gld_known_answer_products() {
    use goldilocks::Elem;
    assert_eq!(
        Elem(0x5555_5555_5555_5555).mul(Elem(0xAAAA_AAAA_AAAA_AAAA)),
        Elem(0xFFFF_FFFE_5555_5557)
    );
    assert_eq!(
        Elem(0xFFFF_FFFF_0000_0000).mul(Elem(0xFFFF_FFFF_0000_0000)),
        Elem(0x0000_0000_0000_0001)
    );
    assert_eq!(
        Elem(0x7FFF_FFFF_FFFF_FFFF).mul(Elem(0x7FFF_FFFF_FFFF_FFFF)),
        Elem(0xFFFF_FFFD_C000_0003)
    );
    assert_eq!(
        Elem(0x0000_0000_7FFF_FFFF).mul(Elem(0xFFFF_FFFF_0000_0000)),
        Elem(0xFFFF_FFFE_8000_0002)
    );
    assert_eq!(Elem(2).inv(), Elem(0x7FFF_FFFF_8000_0001));
    assert_eq!(Elem(7).inv(), Elem(0x2492_4924_6DB6_DB6E));
    // Fold known-answers pinning the split reduction chain.
    assert_eq!(goldilocks::canonical(goldilocks::MODULUS), 0);
    assert_eq!(goldilocks::canonical(u64::MAX), 0xFFFF_FFFE);
    assert_eq!(goldilocks::reduce128(1u128 << 64), 0xFFFF_FFFF); // 2^64 = 2^32 - 1
    assert_eq!(goldilocks::reduce128(1u128 << 96), 0xFFFF_FFFF_0000_0000); // 2^96 = -1
}

#[test]
fn m31_field_axioms() {
    let sample = sample_m31();
    for (i, &a) in sample.iter().enumerate() {
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        // Every output is canonical (< p) on any input, canonical or not.
        assert!(a.mul(b).to_raw() < 0x7FFF_FFFF, "mul canonical {a:?}*{b:?}");
        assert!(a.add(b).to_raw() < 0x7FFF_FFFF, "add canonical");
        assert!(a.sub(b).to_raw() < 0x7FFF_FFFF, "sub canonical");
        assert!(a.neg().to_raw() < 0x7FFF_FFFF, "neg canonical");
        // Differentials against the u128 oracle.
        assert_eq!(a.mul(b), m31_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.add(b), m31_add_oracle(a, b), "{a:?} + {b:?}");
        assert_eq!(a.sub(b), m31_sub_oracle(a, b), "{a:?} - {b:?}");
        // Negation laws and sub == add of neg.
        assert_eq!(a.add(a.neg()), mersenne31::Elem::ZERO);
        assert_eq!(a.neg().neg(), a.canonical());
        assert_eq!(a.sub(b), a.add(b.neg()));
        // Ring laws.
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        // Inverse/division totality and round trip.
        assert_eq!(a.div(mersenne31::Elem::ZERO), mersenne31::Elem::ZERO);
        if a.canonical() != mersenne31::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), mersenne31::Elem::ONE, "inv({a:?})");
            assert_eq!(a.div(a), mersenne31::Elem::ONE);
        }
    }
    assert_eq!(mersenne31::Elem::ZERO.inv(), mersenne31::Elem::ZERO);
}

#[test]
fn gld_field_axioms() {
    let sample = sample_gld();
    for (i, &a) in sample.iter().enumerate() {
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        assert!(a.mul(b).to_raw() < goldilocks::MODULUS, "mul canonical");
        assert!(a.add(b).to_raw() < goldilocks::MODULUS, "add canonical");
        assert!(a.sub(b).to_raw() < goldilocks::MODULUS, "sub canonical");
        assert!(a.neg().to_raw() < goldilocks::MODULUS, "neg canonical");
        assert_eq!(a.mul(b), gld_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.add(b), gld_add_oracle(a, b), "{a:?} + {b:?}");
        assert_eq!(a.sub(b), gld_sub_oracle(a, b), "{a:?} - {b:?}");
        assert_eq!(a.add(a.neg()), goldilocks::Elem::ZERO);
        assert_eq!(a.neg().neg(), a.canonical());
        assert_eq!(a.sub(b), a.add(b.neg()));
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        assert_eq!(a.div(goldilocks::Elem::ZERO), goldilocks::Elem::ZERO);
        if a.canonical() != goldilocks::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), goldilocks::Elem::ONE, "inv({a:?})");
            assert_eq!(a.div(a), goldilocks::Elem::ONE);
        }
    }
    assert_eq!(goldilocks::Elem::ZERO.inv(), goldilocks::Elem::ZERO);
}

#[test]
fn prime_generators_have_full_order() {
    // Multiplicative order is exactly p - 1: full order, and not a proper
    // divisor for any prime factor q of p - 1.
    let g = mersenne31::GENERATOR;
    let order = 0x7FFF_FFFEu64; // p - 1 = 2 * 3^2 * 7 * 11 * 31 * 151 * 331
    assert_eq!(g.pow(order), mersenne31::Elem::ONE);
    for q in [2u64, 3, 7, 11, 31, 151, 331] {
        assert_ne!(g.pow(order / q), mersenne31::Elem::ONE, "M31 factor {q}");
    }

    let g = goldilocks::GENERATOR;
    let order = 0xFFFF_FFFF_0000_0000u64; // p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537
    assert_eq!(g.pow(order), goldilocks::Elem::ONE);
    for q in [2u64, 3, 5, 17, 257, 65_537] {
        assert_ne!(g.pow(order / q), goldilocks::Elem::ONE, "GLD factor {q}");
    }
}

#[test]
fn prime_arithmetic_is_total_over_raw_lanes() {
    // Non-canonical raw lanes (>= p) are legal input; every output is
    // canonical and equals the oracle on the reduced operands.
    let raws31 = [
        0x7FFF_FFFFu32,
        0x8000_0000,
        0xFFFF_FFFF,
        0xC000_0000,
        0xBFFF_FFFE,
    ];
    for &ra in &raws31 {
        for &rb in &raws31 {
            let a = mersenne31::Elem::from_raw(ra);
            let b = mersenne31::Elem::from_raw(rb);
            assert!(a.mul(b).to_raw() < 0x7FFF_FFFF);
            assert_eq!(a.mul(b), m31_mul_oracle(a, b), "raw {ra:#x} * {rb:#x}");
            assert_eq!(a.add(b), m31_add_oracle(a, b));
            assert_eq!(a.sub(b), m31_sub_oracle(a, b));
        }
    }
    let raws64 = [
        goldilocks::MODULUS,
        goldilocks::MODULUS + 1,
        u64::MAX,
        0xFFFF_FFFF_8000_0000,
    ];
    for &ra in &raws64 {
        for &rb in &raws64 {
            let a = goldilocks::Elem::from_raw(ra);
            let b = goldilocks::Elem::from_raw(rb);
            assert!(a.mul(b).to_raw() < goldilocks::MODULUS);
            assert_eq!(a.mul(b), gld_mul_oracle(a, b), "raw {ra:#x} * {rb:#x}");
            assert_eq!(a.add(b), gld_add_oracle(a, b));
            assert_eq!(a.sub(b), gld_sub_oracle(a, b));
        }
    }
}

// ---------------------------------------------------------------------------
// GF((2^31 - 1)^2) - QuadMersenne31
// ---------------------------------------------------------------------------

// The independent oracle is schoolbook (a+bi)(c+di) over u128 % p base
// arithmetic: no fold, no lane tricks.
fn qm_mul_oracle(x: quad_mersenne31::Elem, y: quad_mersenne31::Elem) -> quad_mersenne31::Elem {
    let ar = x.0 as u128 % M31_P;
    let ai = x.1 as u128 % M31_P;
    let br = y.0 as u128 % M31_P;
    let bi = y.1 as u128 % M31_P;
    let re = (ar * br + M31_P * M31_P - ai * bi) % M31_P;
    let im = (ar * bi + ai * br) % M31_P;
    quad_mersenne31::Elem(re as u32, im as u32)
}
fn qm_add_oracle(x: quad_mersenne31::Elem, y: quad_mersenne31::Elem) -> quad_mersenne31::Elem {
    let re = (x.0 as u128 % M31_P + y.0 as u128 % M31_P) % M31_P;
    let im = (x.1 as u128 % M31_P + y.1 as u128 % M31_P) % M31_P;
    quad_mersenne31::Elem(re as u32, im as u32)
}
fn qm_sub_oracle(x: quad_mersenne31::Elem, y: quad_mersenne31::Elem) -> quad_mersenne31::Elem {
    let re = (x.0 as u128 % M31_P + M31_P - y.0 as u128 % M31_P) % M31_P;
    let im = (x.1 as u128 % M31_P + M31_P - y.1 as u128 % M31_P) % M31_P;
    quad_mersenne31::Elem(re as u32, im as u32)
}

fn sample_qm() -> Vec<quad_mersenne31::Elem> {
    // Boundary pairs plus a deterministic spray of limb pairs.
    let mut values: Vec<quad_mersenne31::Elem> = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 3),
        (0x7FFF_FFFE, 0x7FFF_FFFD), // p-2, p-3
        (0x7FFF_FFFF, 0x8000_0000), // raw p and p+1
        (0xFFFF_FFFF, 0x5555_5555),
        (7, 12),
    ]
    .into_iter()
    .map(|(re, im)| quad_mersenne31::Elem(re, im))
    .collect();
    let mut state = 0x243f_6a88u32;
    for _ in 0..48 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(quad_mersenne31::Elem(state, state.rotate_left(7)));
    }
    values
}

#[test]
fn qm31_known_answer_products() {
    use quad_mersenne31::Elem;
    // i^2 = -1
    assert_eq!(Elem(0, 1).square(), Elem(0x7FFF_FFFE, 0));
    assert_eq!(Elem(0, 1).mul(Elem(0, 1)), Elem(0x7FFF_FFFE, 0));
    // (a+ai)^2 = 0 + 2a^2 i with a^2 = 0x71C7_1C71 (the frozen M31 KAT).
    assert_eq!(
        Elem(0x5555_5555, 0x5555_5555).square(),
        Elem(
            0,
            mersenne31::Elem(0x71C7_1C71)
                .add(mersenne31::Elem(0x71C7_1C71))
                .0
        )
    );
    // Conjugate/norm inverse: inv(a+bi) = (a-bi)/(a^2+b^2).
    let x = Elem(0x1234_5678, 0x9abc_def0 % 0x7FFF_FFFF);
    assert_eq!(x.mul(x.inv()), Elem::ONE);
    assert_eq!(Elem::ZERO.inv(), Elem::ZERO);
}

#[test]
fn qm31_field_axioms() {
    let sample = sample_qm();
    for (i, &a) in sample.iter().enumerate() {
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        // Every output is canonical per limb on any input.
        assert!(
            a.mul(b).0 < 0x7FFF_FFFF && a.mul(b).1 < 0x7FFF_FFFF,
            "mul canonical"
        );
        assert!(
            a.add(b).0 < 0x7FFF_FFFF && a.add(b).1 < 0x7FFF_FFFF,
            "add canonical"
        );
        assert!(
            a.sub(b).0 < 0x7FFF_FFFF && a.sub(b).1 < 0x7FFF_FFFF,
            "sub canonical"
        );
        assert!(
            a.neg().0 < 0x7FFF_FFFF && a.neg().1 < 0x7FFF_FFFF,
            "neg canonical"
        );
        // Differentials against the u128 oracle.
        assert_eq!(a.mul(b), qm_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.add(b), qm_add_oracle(a, b), "{a:?} + {b:?}");
        assert_eq!(a.sub(b), qm_sub_oracle(a, b), "{a:?} - {b:?}");
        // Negation laws and sub == add of neg.
        assert_eq!(a.add(a.neg()), quad_mersenne31::Elem::ZERO);
        assert_eq!(a.neg().neg(), a.canonical());
        assert_eq!(a.sub(b), a.add(b.neg()));
        // Ring laws.
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        // Inverse/division totality and round trip.
        assert_eq!(
            a.div(quad_mersenne31::Elem::ZERO),
            quad_mersenne31::Elem::ZERO
        );
        if a.canonical() != quad_mersenne31::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), quad_mersenne31::Elem::ONE, "inv({a:?})");
            assert_eq!(a.div(a), quad_mersenne31::Elem::ONE);
        }
        // Norm is the base-field element a^2 + b^2.
        let n = a.norm();
        assert_eq!(
            n,
            mersenne31::Elem(a.0)
                .mul(mersenne31::Elem(a.0))
                .add(mersenne31::Elem(a.1).mul(mersenne31::Elem(a.1)))
                .0
        );
    }
    assert_eq!(
        quad_mersenne31::Elem::ZERO.inv(),
        quad_mersenne31::Elem::ZERO
    );
}

#[test]
fn qm31_generator_has_full_order() {
    // p^2 - 1 = 2^32 * 3^2 * 7 * 11 * 31 * 151 * 331; order is exactly that.
    let g = quad_mersenne31::GENERATOR;
    let order = 0x3FFF_FFFF_0000_0000u64;
    assert_eq!(g.pow(order), quad_mersenne31::Elem::ONE);
    for q in [2u64, 3, 7, 11, 31, 151, 331] {
        assert_ne!(
            g.pow(order / q),
            quad_mersenne31::Elem::ONE,
            "QM factor {q}"
        );
    }
}

#[test]
fn qm31_arithmetic_is_total_over_raw_limbs() {
    let raws = [
        (0x7FFF_FFFFu32, 0x8000_0000),
        (0xFFFF_FFFF, 0xC000_0000),
        (0xBFFF_FFFE, 0xFFFF_FFFF),
    ];
    for &(ra, rai) in &raws {
        for &(rb, rbi) in &raws {
            let a = quad_mersenne31::Elem(ra, rai);
            let b = quad_mersenne31::Elem(rb, rbi);
            let m = a.mul(b);
            assert!(m.0 < 0x7FFF_FFFF && m.1 < 0x7FFF_FFFF);
            assert_eq!(m, qm_mul_oracle(a, b));
            assert_eq!(a.add(b), qm_add_oracle(a, b));
            assert_eq!(a.sub(b), qm_sub_oracle(a, b));
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical Fan-Paar tower
// ---------------------------------------------------------------------------

#[test]
fn fan_paar_matches_canonical_vectors() {
    assert_eq!(
        fan_paar::fp8::Elem(0x1b).mul(fan_paar::fp8::Elem(0xa8)),
        fan_paar::fp8::Elem(0x09)
    );
    assert_eq!(
        fan_paar::fp16::Elem(0x48a8).mul(fan_paar::fp16::Elem(0xf8a4)),
        fan_paar::fp16::Elem(0x3656)
    );
    assert_eq!(
        fan_paar::fp16::Elem(0xf8a4).square(),
        fan_paar::fp16::Elem(0xe7e6)
    );
    assert_eq!(
        fan_paar::fp64::Elem(0xc84d_6191_1083_1cef)
            .mul(fan_paar::fp64::Elem(0x0000_0000_0000_a14f)),
        fan_paar::fp64::Elem(0x3565_086d_6b9e_f595)
    );
}

#[test]
fn fan_paar_arithmetic_round_trips() {
    macro_rules! check {
        ($module:ident, $($value:expr),+ $(,)?) => {
            $(
                let a = fan_paar::$module::Elem($value);
                assert_eq!(a.square(), a.mul(a));
                assert_eq!(a.add(a), fan_paar::$module::Elem::ZERO);
                if a != fan_paar::$module::Elem::ZERO {
                    assert_eq!(a.mul(a.inv()), fan_paar::$module::Elem::ONE);
                }
            )+
        };
    }

    check!(fp8, 0, 1, 0x2d, 0x53, 0xff);
    check!(fp16, 0, 1, 0xe2de, 0x1234, 0xffff);
    check!(fp32, 0, 1, 0x03e2_1cea, 0xdead_beef, u32::MAX);
    check!(
        fp64,
        0,
        1,
        0x070f_870d_cd9c_1d88,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    );
}

#[test]
fn fan_paar_generators_have_full_order() {
    let g8 = FanPaar8::GENERATOR;
    for factor in [3u64, 5, 17] {
        assert_ne!(g8.pow(255 / factor), fan_paar::fp8::Elem::ONE);
    }
    assert_eq!(g8.pow(255), fan_paar::fp8::Elem::ONE);

    let g16 = FanPaar16::GENERATOR;
    for factor in [3u64, 5, 17, 257] {
        assert_ne!(g16.pow(65_535 / factor), fan_paar::fp16::Elem::ONE);
    }
    assert_eq!(g16.pow(65_535), fan_paar::fp16::Elem::ONE);

    let g32 = FanPaar32::GENERATOR;
    let order32 = u32::MAX as u64;
    for factor in [3u64, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), fan_paar::fp32::Elem::ONE);
    }
    assert_eq!(g32.pow(order32), fan_paar::fp32::Elem::ONE);

    let g64 = FanPaar64::GENERATOR;
    for factor in [3u64, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(g64.pow(u64::MAX / factor), fan_paar::fp64::Elem::ONE);
    }
    assert_eq!(g64.pow(u64::MAX), fan_paar::fp64::Elem::ONE);
}

#[test]
fn fan_paar_subfield_encodings_are_nested() {
    for (a, b) in [(0x1bu8, 0xa8u8), (0x53, 0xca), (0xff, 0x42)] {
        let product = fan_paar::fp8::Elem(a).mul(fan_paar::fp8::Elem(b)).0;
        assert_eq!(
            fan_paar::fp16::Elem(a.into())
                .mul(fan_paar::fp16::Elem(b.into()))
                .0,
            product.into()
        );
        assert_eq!(
            fan_paar::fp32::Elem(a.into())
                .mul(fan_paar::fp32::Elem(b.into()))
                .0,
            product.into()
        );
        assert_eq!(
            fan_paar::fp64::Elem(a.into())
                .mul(fan_paar::fp64::Elem(b.into()))
                .0,
            product.into()
        );
    }
}

// ---------------------------------------------------------------------------
// Representation
// ---------------------------------------------------------------------------

#[test]
fn byte_representation_round_trips() {
    for a in all_gf8() {
        let mut buffer = [0u8; 1];
        Gf8B::write(&mut buffer, a);
        assert_eq!(Gf8B::read(&buffer), a);
    }
    for a in sample_gf16() {
        let mut buffer = [0u8; 2];
        Gf16::write(&mut buffer, a);
        assert_eq!(Gf16::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf32() {
        let mut buffer = [0u8; 4];
        Gf32::write(&mut buffer, a);
        assert_eq!(Gf32::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf64() {
        let mut buffer = [0u8; 8];
        Gf64::write(&mut buffer, a);
        assert_eq!(Gf64::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_m31() {
        let mut buffer = [0u8; 4];
        Mersenne31::write(&mut buffer, a);
        assert_eq!(Mersenne31::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gld() {
        let mut buffer = [0u8; 8];
        Goldilocks::write(&mut buffer, a);
        assert_eq!(Goldilocks::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    macro_rules! check_fan_paar_repr {
        ($field:ty, $elem:expr, $bytes:literal) => {{
            let value = $elem;
            let mut buffer = [0u8; $bytes];
            <$field>::write(&mut buffer, value);
            assert_eq!(<$field>::read(&buffer), value);
            assert_eq!(buffer, value.to_bytes());
        }};
    }
    check_fan_paar_repr!(FanPaar8, fan_paar::fp8::Elem(0xa5), 1);
    check_fan_paar_repr!(FanPaar16, fan_paar::fp16::Elem(0xa55a), 2);
    check_fan_paar_repr!(FanPaar32, fan_paar::fp32::Elem(0xa55a_1234), 4);
    check_fan_paar_repr!(FanPaar64, fan_paar::fp64::Elem(0xa55a_1234_dead_beef), 8);
}

#[test]
fn field_constants_are_consistent() {
    assert_eq!(Gf8B::BYTES, 1);
    assert_eq!(Gf8B::ORDER, 1u128 << Gf8B::BITS);
    assert_eq!(Gf16::BYTES, 2);
    assert_eq!(Gf16::ORDER, 1u128 << Gf16::BITS);
    assert_eq!(Gf32::BYTES, 4);
    assert_eq!(Gf32::ORDER, 1u128 << Gf32::BITS);
    assert_eq!(Gf64::BYTES, 8);
    assert_eq!(Gf64::ORDER, 1u128 << Gf64::BITS);
    // Prime fields: ORDER is the modulus, not 2^BITS, and BITS is the lane
    // width (8 * BYTES), not log2(ORDER).
    assert_eq!(Mersenne31::BYTES, 4);
    assert_eq!(Mersenne31::ORDER, 0x7FFF_FFFF);
    assert_eq!(QuadMersenne31::BYTES, 8);
    assert_eq!(QuadMersenne31::ORDER, 0x3FFF_FFFF_0000_0001);
    assert_eq!(Goldilocks::BYTES, 8);
    assert_eq!(Goldilocks::ORDER, 0xFFFF_FFFF_0000_0001);
    for (bytes, bits) in [
        (Gf8B::BYTES, Gf8B::BITS),
        (Gf16::BYTES, Gf16::BITS),
        (Gf32::BYTES, Gf32::BITS),
        (Gf64::BYTES, Gf64::BITS),
        (FanPaar8::BYTES, FanPaar8::BITS),
        (FanPaar16::BYTES, FanPaar16::BITS),
        (FanPaar32::BYTES, FanPaar32::BITS),
        (FanPaar64::BYTES, FanPaar64::BITS),
        (Mersenne31::BYTES, Mersenne31::BITS),
        (Goldilocks::BYTES, Goldilocks::BITS),
        (QuadMersenne31::BYTES, QuadMersenne31::BITS),
    ] {
        assert_eq!(bytes * 8, bits as usize);
    }
}

// ---------------------------------------------------------------------------
// Shared trait, operator, and formatting surface
// ---------------------------------------------------------------------------
//
// The per-field tests above drive the inherent `const` methods. Generic
// consumers reach the same algebra through the `field::Elem` trait, the
// `core::ops` operator impls, the `Sum`/`Product` folds, and the
// `Debug`/`Display`/`Default`/`Hash` impls — none of which the inherent
// callsites touch. This section exercises those surfaces for every field
// against the same laws, so a broken delegation cannot hide behind a correct
// inherent body.

use fgf::field::Elem as _;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hashes_equal<T: Hash>(a: &T, b: &T) -> bool {
    let mut ha = DefaultHasher::new();
    let mut hb = DefaultHasher::new();
    a.hash(&mut ha);
    b.hash(&mut hb);
    ha.finish() == hb.finish()
}

fn empty_sum_of<E: fgf::field::Elem + Sum>(_seed: E) -> E {
    std::iter::empty::<E>().sum()
}

fn empty_product_of<E: fgf::field::Elem + Product>(_seed: E) -> E {
    std::iter::empty::<E>().product()
}

fn sum_of<E: fgf::field::Elem + Sum>(it: impl Iterator<Item = E>) -> E {
    it.sum()
}

fn product_of<E: fgf::field::Elem + Product>(it: impl Iterator<Item = E>) -> E {
    it.product()
}

fn sum_of_ref<'a, E: fgf::field::Elem + Sum<&'a E>>(it: impl Iterator<Item = &'a E>) -> E {
    it.sum()
}

fn product_of_ref<'a, E: fgf::field::Elem + Product<&'a E>>(it: impl Iterator<Item = &'a E>) -> E {
    it.product()
}

fn empty_sum_of_ref<'a, E: fgf::field::Elem + Sum<&'a E>>(_seed: &'a E) -> E {
    std::iter::empty::<&'a E>().sum()
}

fn empty_product_of_ref<'a, E: fgf::field::Elem + Product<&'a E>>(_seed: &'a E) -> E {
    std::iter::empty::<&'a E>().product()
}

use std::iter::{Product, Sum};

/// Every `field::Elem`/`field::Field` surface reachable from generic code:
/// the trait's arithmetic (including defaulted methods), the total
/// zero conventions, `Debug`/`Hash`/`Default`, and the byte codec.
fn exercise_surface<F: Field>(samples: &[F::Elem]) {
    let zero = F::Elem::ZERO;
    let one = F::Elem::ONE;
    assert!(samples.len() >= 2, "surface sweep needs samples");
    assert_eq!(
        F::Elem::default(),
        zero,
        "Default must be the additive identity"
    );

    // Field facts that hold for every field, checked through the trait.
    assert!(!F::NAME.is_empty());
    assert_eq!(F::BITS as usize, 8 * F::BYTES);
    assert_eq!(F::elem_count(3 * F::BYTES), 3);
    assert!(!F::GENERATOR.is_zero());
    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(one.is_one());
    assert!(!zero.is_one());

    for &a in samples {
        // Stable encoding round trip through the Field contract.
        let mut buffer = [0u8; 16];
        F::write(&mut buffer[..F::BYTES], a);
        assert_eq!(F::read(&buffer[..F::BYTES]), a, "write/read round trip");

        // Laws that hold in every field, through the trait methods.
        assert_eq!(a.add(zero), a, "a + 0");
        assert_eq!(a.sub(zero), a, "a - 0");
        assert_eq!(a.sub(a), zero, "a - a");
        assert_eq!(a.add(a.neg()), zero, "a + (-a)");
        assert_eq!(a.mul(one), a, "a * 1");
        assert_eq!(a.mul(zero), zero, "a * 0");
        assert_eq!(a.square(), a.mul(a), "square");
        assert_eq!(a.pow(0), one, "a^0");
        assert_eq!(a.pow(1), a, "a^1");
        assert_eq!(a.pow(3), a.mul(a).mul(a), "a^3");
        assert_eq!(zero.pow(7), zero, "0^7");
        assert_eq!(a.inv().mul(a), if a.is_zero() { zero } else { one }, "inv");
        assert_eq!(a.div(a), if a.is_zero() { zero } else { one }, "a / a");
        assert_eq!(a.div(zero), zero, "a / 0");
        assert_eq!(zero.div(a), zero, "0 / a");

        // Formatting and hashing are supertraits of every element.
        assert!(!format!("{a:?}").is_empty(), "Debug");
        assert_eq!(format!("{a:?}"), format!("{:?}", a.clone()), "Clone/Debug");
        assert!(hashes_equal(&a, &a.clone()), "equal elements hash equally");
    }
}

/// The per-concrete-type surfaces generic code cannot reach: the
/// `core::ops` operator overloads, the `Sum`/`Product` folds, and
/// `Display`. Each field implements these on its own element type, so each
/// gets instantiated here against the same trait-checked laws.
macro_rules! exercise_operators {
    ($samples:expr) => {{
        let samples: Vec<_> = $samples;
        let [a, b, ..] = samples[..] else {
            panic!("operator sweep needs samples");
        };
        let zero = a - a;
        let one = a.pow(0);
        assert_eq!(a + b, a.add(b), "Add");
        assert_eq!(a - b, a.sub(b), "Sub");
        assert_eq!(a * b, a.mul(b), "Mul");
        assert_eq!(a / b, a.div(b), "Div");
        let mut assigned = a;
        assigned += b;
        assert_eq!(assigned, a.add(b), "AddAssign");
        assigned -= b;
        assert_eq!(assigned, a, "SubAssign");
        assigned *= b;
        assert_eq!(assigned, a.mul(b), "MulAssign");
        assigned /= b;
        assert_eq!(assigned, if b.is_zero() { zero } else { a }, "DivAssign");

        let empty_sum = empty_sum_of(a);
        assert_eq!(empty_sum, zero, "empty Sum must be ZERO");
        let empty_product = empty_product_of(a);
        assert_eq!(empty_product, one, "empty Product must be ONE");
        let empty_ref_sum = empty_sum_of_ref(&a);
        assert_eq!(empty_ref_sum, zero, "empty by-reference Sum must be ZERO");
        let empty_ref_product = empty_product_of_ref(&a);
        assert_eq!(
            empty_ref_product, one,
            "empty by-reference Product must be ONE"
        );
        assert_eq!(
            sum_of(samples.iter().copied()),
            samples.iter().fold(zero, |acc, &x| acc.add(x)),
            "Sum (owned) must fold by addition"
        );
        assert_eq!(
            sum_of_ref(samples.iter()),
            samples.iter().fold(zero, |acc, &x| acc.add(x)),
            "Sum (by reference) must fold by addition"
        );
        assert_eq!(
            product_of(samples.iter().copied()),
            samples.iter().fold(one, |acc, &x| acc.mul(x)),
            "Product (owned) must fold by multiplication"
        );
        assert_eq!(
            product_of_ref(samples.iter()),
            samples.iter().fold(one, |acc, &x| acc.mul(x)),
            "Product (by reference) must fold by multiplication"
        );

        assert!(!format!("{a}").is_empty(), "Display");
    }};
}

#[test]
fn gf8b_trait_operator_and_formatting_surface() {
    let samples: Vec<_> = all_gf8().step_by(97).collect();
    exercise_surface::<Gf8B>(&samples);
    exercise_operators!(samples);
}

#[test]
fn gf8d_trait_operator_and_formatting_surface() {
    let samples: Vec<_> = all_gf8d().step_by(97).collect();
    exercise_surface::<Gf8D>(&samples);
    exercise_operators!(samples);
}

#[test]
fn gf16_trait_operator_and_formatting_surface() {
    let samples: Vec<_> = sample_gf16().into_iter().step_by(61).collect();
    exercise_surface::<Gf16>(&samples);
    exercise_operators!(samples);
}

#[test]
fn gf32_trait_operator_and_formatting_surface() {
    let samples: Vec<_> = sample_gf32().into_iter().step_by(7).collect();
    exercise_surface::<Gf32>(&samples);
    exercise_operators!(samples);
}

#[test]
fn gf64_trait_operator_and_formatting_surface() {
    let samples: Vec<_> = sample_gf64().into_iter().step_by(7).collect();
    exercise_surface::<Gf64>(&samples);
    exercise_operators!(samples);
}

#[test]
fn fan_paar_trait_operator_and_formatting_surface() {
    let fp8: Vec<_> = (0..=u8::MAX).step_by(97).map(fan_paar::fp8::Elem).collect();
    exercise_surface::<FanPaar8>(&fp8);
    exercise_operators!(fp8);
    let fp16: Vec<_> = [0, 1, 0x0100, 0xffff, 0xa55a, 0x1234]
        .into_iter()
        .map(fan_paar::fp16::Elem)
        .collect();
    exercise_surface::<FanPaar16>(&fp16);
    exercise_operators!(fp16);
    let fp32: Vec<_> = [0, 1, 0x10000, 0xffff_ffff, 0xa55a_1234]
        .into_iter()
        .map(fan_paar::fp32::Elem)
        .collect();
    exercise_surface::<FanPaar32>(&fp32);
    exercise_operators!(fp32);
    let fp64: Vec<_> = [0, 1, 1 << 32, u64::MAX, 0xa55a_1234_dead_beef]
        .into_iter()
        .map(fan_paar::fp64::Elem)
        .collect();
    exercise_surface::<FanPaar64>(&fp64);
    exercise_operators!(fp64);
}

#[test]
fn prime_trait_operator_and_formatting_surface() {
    // Only the odd-character fields implement unary negation.
    {
        let a = mersenne31::Elem(7).canonical();
        assert_eq!(-a, a.neg(), "M31 Neg");
        let a = goldilocks::Elem(7).canonical();
        assert_eq!(-a, a.neg(), "Goldilocks Neg");
        let a = quad_mersenne31::Elem(7, 9).canonical();
        assert_eq!(-a, a.neg(), "QM31 Neg");
    }
    // The surface laws compare elements for equality, so they need canonical
    // representatives: the raw samples deliberately include non-canonical
    // lanes, and prime-field arithmetic canonicalizes its outputs.
    let m31: Vec<_> = sample_m31()
        .into_iter()
        .step_by(7)
        .map(|a| a.canonical())
        .collect();
    exercise_surface::<Mersenne31>(&m31);
    exercise_operators!(m31);
    let gld: Vec<_> = sample_gld()
        .into_iter()
        .step_by(7)
        .map(|a| a.canonical())
        .collect();
    exercise_surface::<Goldilocks>(&gld);
    exercise_operators!(gld);
    let qm: Vec<_> = sample_qm()
        .into_iter()
        .step_by(7)
        .map(|a| a.canonical())
        .collect();
    exercise_surface::<QuadMersenne31>(&qm);
    exercise_operators!(qm);
}

/// Inherent helpers that exist beside the trait surface: raw/byte conversions
/// and the tower/extension projections. Each is a distinct public entry point
/// generic code cannot reach, so each gets called here.
#[test]
fn inherent_conversion_helpers_round_trip() {
    // GF(2^8) flat fields.
    for a in all_gf8().step_by(53) {
        assert_eq!(gf8b::Elem::from_raw(a.to_raw()), a);
        assert_eq!(gf8b::Elem::from_bytes(a.to_bytes()), a);
    }
    for a in all_gf8d().step_by(53) {
        assert_eq!(gf8d::Elem::from_raw(a.to_raw()), a);
        assert_eq!(gf8d::Elem::from_bytes(a.to_bytes()), a);
    }
    // Towers: component projection is a bijection with from_components.
    for a in sample_gf16().into_iter().step_by(61) {
        let (lo, hi) = a.components();
        assert_eq!(gf16::Elem::from_components(lo, hi), a);
        assert_eq!(gf16::Elem::from_raw(a.to_raw()), a);
        assert_eq!(gf16::Elem::from_bytes(a.to_bytes()), a);
    }
    for a in sample_gf32().into_iter().step_by(11) {
        let (lo, hi) = a.components();
        assert_eq!(gf32::Elem::from_components(lo, hi), a);
        assert_eq!(gf32::Elem::from_raw(a.to_raw()), a);
        assert_eq!(gf32::Elem::from_bytes(a.to_bytes()), a);
    }
    for a in sample_gf64().into_iter().step_by(11) {
        let (lo, hi) = a.components();
        assert_eq!(gf64::Elem::from_components(lo, hi), a);
        assert_eq!(gf64::Elem::from_raw(a.to_raw()), a);
        assert_eq!(gf64::Elem::from_bytes(a.to_bytes()), a);
    }
    // Prime fields: canonical representatives and raw lanes.
    for a in sample_m31().into_iter().step_by(7) {
        assert_eq!(
            a.canonical().to_raw(),
            a.to_raw() % 0x7FFF_FFFF,
            "canonical"
        );
        assert_eq!(mersenne31::Elem::from_raw(a.to_raw()), a);
        assert_eq!(mersenne31::Elem::from_bytes(a.to_bytes()), a);
        assert_eq!(
            mersenne31::reduce(a.to_raw()),
            a.canonical().to_raw(),
            "reduce"
        );
    }
    for a in sample_gld().into_iter().step_by(7) {
        assert!(
            a.canonical().to_raw() < 0xFFFF_FFFF_0000_0001,
            "canonical is below the modulus"
        );
        assert_eq!(goldilocks::Elem::from_raw(a.to_raw()), a);
        assert_eq!(goldilocks::Elem::from_bytes(a.to_bytes()), a);
    }
    // Quadratic extension: conjugation and the norm land in the base field.
    for a in sample_qm().into_iter().step_by(7) {
        let a = a.canonical();
        assert_eq!(
            quad_mersenne31::Elem::from_raw(a.to_raw().0, a.to_raw().1),
            a
        );
        assert_eq!(quad_mersenne31::Elem::from_bytes(a.to_bytes()), a);
        let (re, im) = a.components();
        assert_eq!(quad_mersenne31::Elem::from_components(re, im), a);
        assert_eq!(a.conjugate().conjugate(), a, "conjugation is an involution");
        assert_eq!(
            a.conjugate().norm(),
            a.norm(),
            "norm is fixed under conjugation"
        );
        assert!(a.norm() < 0x7FFF_FFFF, "norm lands in the base field");
        let (re, im) = a.mul(a.conjugate()).canonical().components();
        assert_eq!(re.to_raw(), a.norm(), "a * conj(a) is the norm, really");
        assert!(im.to_raw() == 0, "a * conj(a) is real");
    }
    // Fan–Paar levels expose the same raw/byte/component surface.
    macro_rules! fp_level {
        ($elem:ty, $value:expr) => {{
            let a = $value;
            assert_eq!(<$elem>::from_raw(a.to_raw()), a);
            assert_eq!(<$elem>::from_bytes(a.to_bytes()), a);
            let (lo, hi) = a.components();
            assert_eq!(<$elem>::from_components(lo, hi), a);
        }};
    }
    fp_level!(fan_paar::fp16::Elem, fan_paar::fp16::Elem(0xa55a));
    fp_level!(fan_paar::fp32::Elem, fan_paar::fp32::Elem(0xa55a_1234));
    fp_level!(
        fan_paar::fp64::Elem,
        fan_paar::fp64::Elem(0xa55a_1234_dead_beef)
    );
    assert_eq!(
        fan_paar::fp8::Elem(0xa5).mul_alpha(),
        fan_paar::fp8::Elem(0xa5).mul(fan_paar::fp8::ALPHA)
    );
}

// ---------------------------------------------------------------------------
// GF(2)
// ---------------------------------------------------------------------------

/// The whole field, exhaustively: the four ordered pairs.
fn all_gf2_pairs() -> impl Iterator<Item = (gf2::Elem, gf2::Elem)> {
    let elems = [gf2::Elem(0), gf2::Elem(1)];
    elems.into_iter().flat_map(move |a| {
        let elems = [gf2::Elem(0), gf2::Elem(1)];
        elems.into_iter().map(move |b| (a, b))
    })
}

#[test]
fn gf2_add_and_mul_are_xor_and_and() {
    for (a, b) in all_gf2_pairs() {
        assert_eq!(a.add(b).to_raw(), a.to_raw() ^ b.to_raw());
        assert_eq!(a.mul(b).to_raw(), a.to_raw() & b.to_raw());
        assert_eq!(a.sub(b), a.add(b), "sub is add in characteristic two");
        assert_eq!(a.neg(), a, "neg is the identity");
        assert_eq!(a.square(), a, "x^2 = x");
        assert_eq!(a + b, a.add(b));
        assert_eq!(a - b, a.sub(b));
        assert_eq!(a * b, a.mul(b));
        assert_eq!(-a, a);
    }
}

#[test]
fn gf2_inverse_division_and_power_conventions() {
    for (a, b) in all_gf2_pairs() {
        assert_eq!(a.inv(), a, "inv is the identity on canonical values");
        let quotient = a.div(b).to_raw();
        let expected = if b.to_raw() == 0 { 0 } else { a.to_raw() };
        assert_eq!(quotient, expected, "x / 0 == 0 and x / 1 == x");
        assert_eq!(a / b, a.div(b));
    }
    for a in [gf2::Elem(0), gf2::Elem(1)] {
        assert_eq!(a.pow(0), gf2::Elem::ONE, "pow(_, 0) is one");
        for exponent in [1u64, 2, 3, 63, u64::MAX] {
            assert_eq!(a.pow(exponent), a, "x^n = x for n > 0");
        }
    }
    // Division stays total in const context.
    const _: () = assert!(gf2::Elem(1).div(gf2::Elem(0)).to_raw() == 0);
    const _: () = assert!(gf2::Elem(1).pow(0).to_raw() == 1);
}

#[test]
fn gf2_constants_and_predicates() {
    assert_eq!(gf2::ORDER, 2);
    assert_eq!(gf2::Gf2::NAME, "GF(2)");
    // The multiplicative group is trivial: the generator is one and has
    // order 1.
    assert_eq!(gf2::GENERATOR, gf2::Elem::ONE);
    for exponent in [0u64, 1, 2, 100] {
        assert_eq!(gf2::GENERATOR.pow(exponent), gf2::Elem::ONE);
    }
    assert!(gf2::Elem(0).is_zero());
    assert!(gf2::Elem(1).is_one());
    assert_eq!(gf2::Elem::ZERO.to_raw(), 0);
    assert_eq!(gf2::Elem::ONE.to_raw(), 1);
    assert_eq!(gf2::Elem::default(), gf2::Elem::ZERO);
}

#[test]
fn gf2_elem_trait_bodies_match_inherent() {
    use fgf::field::Elem;

    // Through a generic: only the trait's methods are visible, so the
    // delegating bodies themselves execute.
    fn through_trait<E: Elem>(a: E, b: E) {
        let zero = E::ZERO;
        let one = E::ONE;
        assert_eq!(a.add(zero), a, "a + 0");
        assert_eq!(a.sub(a), zero, "a - a");
        assert_eq!(a.sub(b), a.add(b.neg()), "a - b in characteristic two");
        assert_eq!(a.mul(one), a, "a * 1");
        assert_eq!(a.mul(zero), zero, "a * 0");
        assert_eq!(a.square(), a, "x^2 = x");
        assert_eq!(a.pow(0), one, "a^0");
        assert_eq!(a.pow(5), a, "x^5 = x");
        assert_eq!(
            a.inv().mul(a),
            if a.is_zero() { zero } else { one },
            "inv round trip"
        );
        assert_eq!(a.div(zero), zero, "a / 0");
        assert_eq!(a.div(one), a, "a / 1");
        assert_eq!(a.is_zero(), a == zero);
        assert_eq!(a.is_one(), a == one);
    }

    for (a, b) in all_gf2_pairs() {
        through_trait(a, b);
        assert_eq!(<gf2::Elem as Elem>::add(a, b), gf2::Elem::add(a, b));
        assert_eq!(<gf2::Elem as Elem>::sub(a, b), gf2::Elem::sub(a, b));
        assert_eq!(<gf2::Elem as Elem>::neg(a), gf2::Elem::neg(a));
        assert_eq!(<gf2::Elem as Elem>::mul(a, b), gf2::Elem::mul(a, b));
        assert_eq!(<gf2::Elem as Elem>::square(a), gf2::Elem::square(a));
        assert_eq!(<gf2::Elem as Elem>::inv(a), gf2::Elem::inv(a));
        assert_eq!(<gf2::Elem as Elem>::div(a, b), gf2::Elem::div(a, b));
        assert_eq!(<gf2::Elem as Elem>::pow(a, 7), gf2::Elem::pow(a, 7));
        assert_eq!(<gf2::Elem as Elem>::is_zero(a), gf2::Elem::is_zero(a));
        assert_eq!(<gf2::Elem as Elem>::is_one(a), gf2::Elem::is_one(a));
    }
}

#[test]
fn gf2_raw_lanes_are_total_and_outputs_canonical() {
    // Any raw byte is a legal input; only bit 0 is meaningful, and every
    // arithmetic output is canonical — the crate's totality convention.
    for raw in [0u8, 1, 2, 3, 0x7f, 0x80, 0xfe, 0xff] {
        let a = gf2::Elem(raw);
        assert_eq!(a.add(gf2::Elem(1)).to_raw(), (raw & 1) ^ 1);
        assert_eq!(a.mul(gf2::Elem(1)).to_raw(), raw & 1);
        assert_eq!(a.square().to_raw(), raw & 1);
        assert_eq!(a.inv().to_raw(), raw & 1);
        assert_eq!(a.div(gf2::Elem(0)).to_raw(), 0);
        assert_eq!(a.is_zero(), raw & 1 == 0);
        assert_eq!(a.is_one(), raw & 1 == 1);
    }
    assert_eq!(gf2::Elem::from_raw(0xfe).to_raw(), 0, "from_raw masks");
    assert_eq!(
        gf2::Elem::from_bytes([0xfe]).to_bytes(),
        [0],
        "byte round trip masks"
    );
    assert_eq!(gf2::Elem(1).canonical(), gf2::Elem::ONE);
}

#[test]
fn gf2_operators_folds_and_display() {
    use std::fmt::Write as _;

    let mut a = gf2::Elem(1);
    a += gf2::Elem(1);
    assert_eq!(a, gf2::Elem::ZERO);
    a -= gf2::Elem(1);
    assert_eq!(a, gf2::Elem::ONE);
    a *= gf2::Elem(0);
    assert_eq!(a, gf2::Elem::ZERO);
    a += gf2::Elem(1);
    a /= gf2::Elem(1);
    assert_eq!(a, gf2::Elem::ONE);
    a /= gf2::Elem::ZERO;
    assert_eq!(a, gf2::Elem::ZERO);

    let xor_sum: gf2::Elem = [gf2::Elem(1), gf2::Elem(1), gf2::Elem(1)].into_iter().sum();
    assert_eq!(xor_sum, gf2::Elem::ONE, "sum of three ones");
    let and_product: gf2::Elem = [gf2::Elem(1), gf2::Elem(1)].into_iter().product();
    assert_eq!(and_product, gf2::Elem::ONE);
    let borrowed_sum: gf2::Elem = [&gf2::Elem(1), &gf2::Elem(1)].into_iter().sum();
    assert_eq!(borrowed_sum, gf2::Elem::ZERO);
    let borrowed_product: gf2::Elem = [&gf2::Elem(1), &gf2::Elem(1)].into_iter().product();
    assert_eq!(borrowed_product, gf2::Elem::ONE);

    let mut text = String::new();
    write!(text, "{} {:?}", gf2::Elem(1), gf2::Elem(0)).unwrap();
    assert_eq!(text, "1 Gf2(0)");
}

// ---------------------------------------------------------------------------
// Trait default bodies
// ---------------------------------------------------------------------------

mod toy {
    //! Minimal GF(2^3) over `x^3 + x + 1` implementing only the required
    //! `Elem`/`Field` methods. The trait's defaulted `neg`, `square`, `pow`,
    //! `is_zero`, `is_one`, and `elem_count` bodies run here — every real
    //! field overrides the algebraic ones, so this is the only implementor
    //! where those defaults execute.
    use fgf::field::{Elem, Field};

    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
    pub struct Elem7(pub u8);

    #[derive(Debug, Clone, Copy)]
    pub struct Gf8Toy;

    impl Elem for Elem7 {
        const ZERO: Self = Self(0);
        const ONE: Self = Self(1);

        fn add(self, rhs: Self) -> Self {
            Self(self.0 ^ rhs.0)
        }
        fn sub(self, rhs: Self) -> Self {
            self.add(rhs)
        }
        fn mul(self, rhs: Self) -> Self {
            let mut acc = 0u8;
            let mut a = self.0;
            let b = rhs.0;
            for i in 0..8 {
                if (b >> i) & 1 == 1 {
                    acc ^= a;
                }
                let overflow = a & 0x04 != 0;
                a <<= 1;
                if overflow {
                    a ^= 0x0B; // x^3 = x + 1
                }
            }
            Self(acc & 7)
        }
        fn inv(self) -> Self {
            if self.0 == 0 {
                return Self::ZERO;
            }
            for candidate in 1..8u8 {
                let product = Self(self.0).mul(Self(candidate));
                if product == Self::ONE {
                    return Self(candidate);
                }
            }
            unreachable!("every nonzero element of GF(8) is invertible");
        }
        fn div(self, rhs: Self) -> Self {
            if self.0 == 0 || rhs.0 == 0 {
                return Self::ZERO;
            }
            self.mul(rhs.inv())
        }
    }

    impl Field for Gf8Toy {
        type Elem = Elem7;

        const NAME: &'static str = "GF(2^3) toy";
        const BITS: u32 = 8;
        const BYTES: usize = 1;
        const ORDER: u128 = 8;
        const GENERATOR: Elem7 = Elem7(2);

        fn read(bytes: &[u8]) -> Elem7 {
            Elem7(bytes[0] & 7)
        }
        fn write(bytes: &mut [u8], value: Elem7) {
            bytes[0] = value.0;
        }
    }
}

#[test]
fn elem_trait_defaults_are_correct_for_minimal_implementors() {
    let samples: Vec<toy::Elem7> = (0..8u8).map(toy::Elem7).collect();
    exercise_surface::<toy::Gf8Toy>(&samples);

    // The defaulted bodies specifically: neg is the identity in
    // characteristic two, square is mul, pow is square-and-multiply.
    for a in samples {
        assert_eq!(a.neg(), a, "default neg in characteristic two");
        assert_eq!(a.square(), a.mul(a), "default square");
        assert_eq!(a.pow(5), a.mul(a).mul(a).mul(a).mul(a), "default pow");
    }
    // GF(8)* has order 7: the generator must have full order through the
    // defaulted pow.
    let g = toy::Gf8Toy::GENERATOR;
    assert_eq!(g.pow(7), toy::Elem7::ONE);
    assert_ne!(g.pow(1), toy::Elem7::ONE);
}

#[test]
fn qm31_pow_u128_matches_pow_and_repeated_squaring() {
    let a = quad_mersenne31::Elem(0x1234_5678, 0x9abc_def0).canonical();
    // Inside u64 the two exponentiation paths must agree exactly.
    for exponent in [0u128, 1, 2, 3, 7, 255, 1 << 32, u64::MAX as u128] {
        assert_eq!(
            a.pow_u128(exponent),
            a.pow(exponent as u64),
            "pow_u128({exponent})"
        );
    }
    // Past u64, check against manual square-and-multiply over mul.
    let mut expected = a;
    for _ in 0..70 {
        expected = expected.square();
    }
    assert_eq!(a.pow_u128(1u128 << 70), expected, "pow_u128(2^70)");
}
