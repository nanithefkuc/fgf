//! Scalar field axioms and cross-backend algebra checks.
//!
//! The independent oracle for GF(2^8) is shift-and-XOR; every tower oracle
//! uses schoolbook expansion over its already-tested base field. Nothing here
//! uses the Karatsuba form under test, so reduction bugs remain visible rather
//! than self-consistent.

use fgf::field::{Field, FieldBuffer, PrimeIdentity};
use fgf::poly::{AES, RS};
use fgf::{
    Binary, BinaryField, Cantor, Elem, Goldilocks, HasGenerator, Mersenne31, Normal, Polynomial,
    QuadMersenne31, goldilocks, mersenne31, quad_mersenne31,
};

/// Every element of `Binary<8, Polynomial<POLY>>`, in ascending raw order.
fn all_gf8_elems<const POLY: u128>() -> impl Iterator<Item = Elem<Binary<8, Polynomial<POLY>>>> {
    (0..=u8::MAX).map(Elem::<Binary<8, Polynomial<POLY>>>::from_raw)
}

/// A spread of GF(2^16) elements: boundaries, both component planes, and a
/// deterministic pseudo-random spray. Exhaustive would be 4 billion pairs.
fn sample_gf16() -> Vec<Elem<Gf16>> {
    let mut values = vec![
        Elem::<Gf16>::from_raw(0),
        Elem::<Gf16>::from_raw(1),
        Elem::<Gf16>::from_raw(0x0100),
        Elem::<Gf16>::from_raw(0x00ff),
        Elem::<Gf16>::from_raw(0xff00),
        Elem::<Gf16>::from_raw(0xffff),
        Elem::<Gf16>::GENERATOR,
    ];
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        values.push(Elem::<Gf16>::from_raw((state >> 32) as u16));
    }
    values
}

// ---------------------------------------------------------------------------
// GF(2^8): the laws run on every flat field
// ---------------------------------------------------------------------------
/// Independent shift-and-XOR GF(2^8) multiply over a field's public
/// reduction polynomial: the oracle for the table backends, local to this
/// test crate so it stays independent of every crate-internal path.
fn xtime_mul(a: u8, b: u8, poly: u128) -> u8 {
    let mut a = a;
    let mut acc = 0u8;
    for i in 0..8 {
        if (b >> i) & 1 == 1 {
            acc ^= a;
        }
        let overflow = a & 0x80 != 0;
        a <<= 1;
        if overflow {
            a ^= (poly & 0xFF) as u8;
        }
    }
    acc
}

/// GF(2^8) inverse oracle: `a^254` by square-and-multiply, MSB first, with
/// the crate's `inv(0) == 0` convention.
fn xtime_inv(a: u8, poly: u128) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut result = 1u8;
    for i in 0..8 {
        result = xtime_mul(result, result, poly);
        // 254 = 0b1111_1110: multiply by `a` on every set exponent bit,
        // scanning most-significant first.
        if (254u32 >> (7 - i)) & 1 == 1 {
            result = xtime_mul(result, a, poly);
        }
    }
    result
}

fn table_multiply_matches_shift_and_xor<const POLY: u128>() {
    for a in all_gf8_elems::<POLY>() {
        for b in all_gf8_elems::<POLY>() {
            assert_eq!(
                a.mul(b),
                Elem::<Binary<8, Polynomial<POLY>>>::from_raw(xtime_mul(
                    a.to_raw(),
                    b.to_raw(),
                    Polynomial::<POLY>::POLY
                )),
                "table and xtime disagree on {a:?} * {b:?}"
            );
        }
    }
}

fn inverse_matches_fermat_and_round_trips<const POLY: u128>() {
    assert_eq!(
        xtime_inv(
            Elem::<Binary<8, Polynomial<POLY>>>::ZERO.to_raw(),
            Polynomial::<POLY>::POLY
        ),
        0,
        "inv_xtime(0) must be 0"
    );
    assert_eq!(
        Elem::<Binary<8, Polynomial<POLY>>>::ZERO.inv(),
        Elem::<Binary<8, Polynomial<POLY>>>::ZERO,
        "inv(0) must be 0"
    );
    for a in all_gf8_elems::<POLY>().skip(1) {
        assert_eq!(
            a.inv().to_raw(),
            xtime_inv(a.to_raw(), Polynomial::<POLY>::POLY),
            "inverse backends disagree on {a:?}"
        );
        assert_eq!(
            a.mul(a.inv()),
            Elem::<Binary<8, Polynomial<POLY>>>::ONE,
            "{a:?} * inv({a:?}) != 1"
        );
        assert_eq!(
            a.div(a),
            Elem::<Binary<8, Polynomial<POLY>>>::ONE,
            "{a:?} / {a:?} != 1"
        );
    }
}

fn generator_has_full_order<const POLY: u128>() {
    let mut seen = [false; 256];
    let mut value = Elem::<Binary<8, Polynomial<POLY>>>::ONE;
    for step in 0..255u32 {
        assert!(
            !seen[value.to_raw() as usize],
            "generator repeats at step {step}"
        );
        seen[value.to_raw() as usize] = true;
        value = value.mul(Elem::<Binary<8, Polynomial<POLY>>>::GENERATOR);
    }
    assert_eq!(
        value,
        Elem::<Binary<8, Polynomial<POLY>>>::ONE,
        "generator order is not 255"
    );
    assert!(
        seen.iter().skip(1).all(|&hit| hit),
        "orbit misses an element"
    );
}

fn field_axioms<const POLY: u128>() {
    let sample: Vec<_> = all_gf8_elems::<POLY>().step_by(7).collect();
    for &a in &sample {
        assert_eq!(a.add(Elem::<Binary<8, Polynomial<POLY>>>::ZERO), a);
        assert_eq!(a.mul(Elem::<Binary<8, Polynomial<POLY>>>::ONE), a);
        assert_eq!(
            a.mul(Elem::<Binary<8, Polynomial<POLY>>>::ZERO),
            Elem::<Binary<8, Polynomial<POLY>>>::ZERO
        );
        assert_eq!(
            a.add(a),
            Elem::<Binary<8, Polynomial<POLY>>>::ZERO,
            "characteristic two"
        );
        assert_eq!(a.sub(a), Elem::<Binary<8, Polynomial<POLY>>>::ZERO);
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

fn pow_matches_repeated_multiplication<const POLY: u128>() {
    for a in all_gf8_elems::<POLY>().step_by(11) {
        let mut expected = Elem::<Binary<8, Polynomial<POLY>>>::ONE;
        for exponent in 0..20u128 {
            assert_eq!(a.pow(exponent), expected, "{a:?}^{exponent}");
            expected = expected.mul(a);
        }
    }
}

/// Run one scalar law on every flat field: the two named conventions and
/// two further irreducible polynomials spelled directly.
macro_rules! every_gf8_field {
    ($law:ident) => {
        $law::<AES>();
        $law::<RS>();
        // Data Matrix and the CCSDS Reed–Solomon field: irreducible
        // polynomials no named constant covers.
        $law::<0x12D>();
        $law::<0x187>();
    };
}

#[test]
fn gf8_table_multiply_matches_shift_and_xor() {
    every_gf8_field!(table_multiply_matches_shift_and_xor);
}

#[test]
fn gf8_inverse_matches_fermat_and_round_trips() {
    every_gf8_field!(inverse_matches_fermat_and_round_trips);
}

#[test]
fn gf8_generator_has_full_order() {
    every_gf8_field!(generator_has_full_order);
}

#[test]
fn gf8_field_axioms() {
    every_gf8_field!(field_axioms);
}

#[test]
fn gf8_pow_matches_repeated_multiplication() {
    every_gf8_field!(pow_matches_repeated_multiplication);
}

#[test]
fn gf8_known_answer_products() {
    // The derived generators are public facts: 0x03 under AES, 0x02 under
    // RS.
    assert_eq!(
        Elem::<Binary<8, Polynomial<AES>>>::GENERATOR,
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x03)
    );
    assert_eq!(
        Elem::<Binary<8, Polynomial<RS>>>::GENERATOR,
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x02)
    );
    // AES: the classic Rijndael example.
    assert_eq!(
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x53)
            .mul(Elem::<Binary<8, Polynomial<AES>>>::from_raw(0xca)),
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x01)
    );
    assert_eq!(
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(0xff)
            .mul(Elem::<Binary<8, Polynomial<AES>>>::from_raw(0xff)),
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x13)
    );
    // RS: the ISA-L / klauspost-reedsolomon field, independently
    // computed from the shift/XOR oracle.
    assert_eq!(
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x53)
            .mul(Elem::<Binary<8, Polynomial<RS>>>::from_raw(0xca)),
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x8f)
    );
    assert_eq!(
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x57)
            .mul(Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x83)),
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x31)
    );
    assert_eq!(
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0xff)
            .mul(Elem::<Binary<8, Polynomial<RS>>>::from_raw(0xff)),
        Elem::<Binary<8, Polynomial<RS>>>::from_raw(0xe2)
    );
}

#[test]
fn reduction_polynomials_are_introspectable() {
    assert_eq!(Polynomial::<AES>::POLY, 0x11B);
    assert_eq!(Polynomial::<RS>::POLY, 0x11D);
    assert_eq!(Polynomial::<0x12D>::POLY, 0x12D);
    assert_eq!(Polynomial::<0x187>::POLY, 0x187);
}

#[test]
fn gf8_fields_are_pairwise_distinct() {
    // Same raw bytes, genuinely different products: the guard against
    // filling one polynomial's tables from another's arithmetic.
    fn disagreement_count<const P: u128, const Q: u128>() -> usize {
        (0u16..=255)
            .flat_map(|a| (0u16..=255).map(move |b| (a as u8, b as u8)))
            .filter(|&(a, b)| {
                Elem::<Binary<8, Polynomial<P>>>::from_raw(a)
                    .mul(Elem::<Binary<8, Polynomial<P>>>::from_raw(b))
                    .to_raw()
                    != Elem::<Binary<8, Polynomial<Q>>>::from_raw(a)
                        .mul(Elem::<Binary<8, Polynomial<Q>>>::from_raw(b))
                        .to_raw()
            })
            .count()
    }
    assert_eq!(
        disagreement_count::<AES, RS>(),
        63_232,
        "0x11B and 0x11D must differ on most products"
    );
    // `x^4 * x^4` reduces to the low byte of the polynomial, so two distinct
    // polynomials can never induce the same multiplication: every pair
    // disagrees, and most pairs disagree almost everywhere.
    assert_ne!(disagreement_count::<AES, 0x12D>(), 0);
    assert_ne!(disagreement_count::<AES, 0x187>(), 0);
    assert_ne!(disagreement_count::<RS, 0x12D>(), 0);
    assert_ne!(disagreement_count::<RS, 0x187>(), 0);
    assert_ne!(disagreement_count::<0x12D, 0x187>(), 0);
}

// ---------------------------------------------------------------------------
// Small polynomial degrees and ordered bases
// ---------------------------------------------------------------------------

/// Independent shift-and-reduce multiply under a degree-`n` polynomial.
///
/// The oracle for every degree-2/4 field and basis map, local to this test
/// crate so it stays independent of the descriptor interpreter.
fn small_mul(poly: u128, n: u32, a: u8, b: u8) -> u8 {
    let mut acc: u128 = 0;
    for k in 0..n {
        if (b >> k) & 1 == 1 {
            acc ^= u128::from(a) << k;
        }
    }
    let mut k = 2 * n;
    while k > n {
        k -= 1;
        if (acc >> k) & 1 == 1 {
            acc ^= poly << (k - n);
        }
    }
    u8::try_from(acc).expect("reduction leaves n bits")
}

/// Exhaustive products, inversion, and generator order of one small
/// polynomial field, against the shift-and-reduce oracle.
macro_rules! small_poly_field {
    ($poly:expr, $n:literal, $mask:literal, [$($factor:literal),+]) => {{
        for a in 0..=$mask {
            for b in 0..=$mask {
                assert_eq!(
                    Elem::<Binary<$n, Polynomial<$poly>>>::from_raw(a)
                        .mul(Elem::<Binary<$n, Polynomial<$poly>>>::from_raw(b))
                        .to_raw(),
                    small_mul($poly, $n, a, b),
                    "{a:#x} * {b:#x} under {:#x}",
                    $poly
                );
            }
        }
        for a in 0..=$mask {
            let elem = Elem::<Binary<$n, Polynomial<$poly>>>::from_raw(a);
            if a != 0 {
                assert_eq!(
                    elem.mul(elem.inv()),
                    Elem::<Binary<$n, Polynomial<$poly>>>::ONE,
                    "inverse of {a:#x}"
                );
            } else {
                assert_eq!(elem.inv(), elem, "inv(0) == 0");
            }
        }
        let order = (1u128 << $n) - 1;
        let generator = Elem::<Binary<$n, Polynomial<$poly>>>::GENERATOR;
        assert_eq!(generator.pow(order), Elem::<Binary<$n, Polynomial<$poly>>>::ONE);
        for factor in [$($factor),+] {
            assert_ne!(
                generator.pow(order / factor),
                Elem::<Binary<$n, Polynomial<$poly>>>::ONE,
                "cofactor {factor}"
            );
        }
    }};
}

#[test]
fn small_polynomial_products_match_shift_and_xor() {
    small_poly_field!(0x7, 2, 0x3, [3]);
    small_poly_field!(0x13, 4, 0xF, [3, 5]);
    small_poly_field!(0x19, 4, 0xF, [3, 5]);
    small_poly_field!(0x1F, 4, 0xF, [3, 5]);
}

/// The normal-basis coordinate columns of `e` over `poly`, computed by
/// repeated squaring through the oracle: column `i` is `e^(2^i)`.
fn normal_columns(poly: u128, n: u32, e: u8) -> [u8; 8] {
    let mut columns = [0u8; 8];
    let mut power = e;
    for column in columns.iter_mut().take(n as usize) {
        *column = power;
        power = small_mul(poly, n, power, power);
    }
    columns
}

/// The Cantor coordinate columns of `seed` over `poly`, computed through
/// the oracle: `chain[n - 1]` is the seed and each lower entry squares and
/// adds its successor.
fn cantor_columns(poly: u128, n: u32, seed: u8) -> [u8; 8] {
    let mut chain = [0u8; 8];
    chain[n as usize - 1] = seed;
    for i in (0..n as usize - 1).rev() {
        let top = chain[i + 1];
        chain[i] = small_mul(poly, n, top, top) ^ top;
    }
    chain
}

/// Apply coordinate columns: bit `i` of `word` selects `columns[i]`.
fn apply_columns(columns: [u8; 8], word: u64) -> u64 {
    let mut image = 0u64;
    for (i, &column) in columns.iter().enumerate() {
        if (word >> i) & 1 == 1 {
            image ^= u64::from(column);
        }
    }
    image
}

/// Product agreement after mapping back, actual one and zero, and generator
/// order of one ordered-basis presentation, against oracle-built columns.
macro_rules! basis_field {
    ($field:ty, $poly:expr, $n:literal, $columns:expr, $one:literal) => {{
        let columns = $columns;
        let map = |word: u8| apply_columns(columns, u64::from(word)) as u8;
        let count = 1u32 << $n;
        for x in 0..count {
            for y in 0..count {
                let product = Elem::<$field>::from_raw(x as u8)
                    .mul(Elem::<$field>::from_raw(y as u8))
                    .to_raw();
                assert_eq!(
                    map(product),
                    small_mul($poly, $n, map(x as u8), map(y as u8)),
                    "{x:#x} * {y:#x}"
                );
            }
        }
        assert_eq!(Elem::<$field>::ZERO.to_raw(), 0, "zero is zero");
        assert_eq!(
            Elem::<$field>::ONE.to_raw(),
            $one,
            "actual one of the basis"
        );
        assert_eq!(
            Elem::<$field>::ONE.mul(Elem::<$field>::ONE),
            Elem::<$field>::ONE
        );
        let order = (1u128 << $n) - 1;
        let generator = Elem::<$field>::GENERATOR;
        assert_eq!(generator.pow(order), Elem::<$field>::ONE);
        for factor in [3u128, 5, 17] {
            if order % factor != 0 {
                continue;
            }
            assert_ne!(
                generator.pow(order / factor),
                Elem::<$field>::ONE,
                "cofactor {factor}"
            );
        }
    }};
}

#[test]
fn normal_forward_coordinates_match_independent_powers() {
    basis_field!(
        Binary<2, Normal<0x7, 2>>,
        0x7,
        2,
        normal_columns(0x7, 2, 2),
        0x3
    );
    basis_field!(
        Binary<4, Normal<0x13, 8>>,
        0x13,
        4,
        normal_columns(0x13, 4, 8),
        0xF
    );
    basis_field!(
        Binary<8, Normal<0x11B, 0x20>>,
        0x11B,
        8,
        normal_columns(0x11B, 8, 0x20),
        0xFF
    );
    basis_field!(
        Binary<8, Normal<0x11D, 0x20>>,
        0x11D,
        8,
        normal_columns(0x11D, 8, 0x20),
        0xFF
    );
}

#[test]
fn cantor_forward_coordinates_match_the_independent_chain() {
    basis_field!(
        Binary<2, Cantor<0x7, 2>>,
        0x7,
        2,
        cantor_columns(0x7, 2, 2),
        0x1
    );
    basis_field!(
        Binary<4, Cantor<0x13, 8>>,
        0x13,
        4,
        cantor_columns(0x13, 4, 8),
        0x1
    );
    basis_field!(
        Binary<8, Cantor<0x11B, 0x20>>,
        0x11B,
        8,
        cantor_columns(0x11B, 8, 0x20),
        0x1
    );
    basis_field!(
        Binary<8, Cantor<0x11D, 0x20>>,
        0x11D,
        8,
        cantor_columns(0x11D, 8, 0x20),
        0x1
    );
}

#[test]
fn binary_coordinates_round_trip_and_reject_excess_bits() {
    fn round_trip<F: BinaryField>(valid: u64, excess: &[u64]) {
        assert_eq!(
            F::to_coordinates(F::from_coordinates(valid).expect("valid word")),
            valid
        );
        for &bad in excess {
            assert!(
                F::from_coordinates(bad).is_err(),
                "excess bits {bad:#x} must be rejected"
            );
        }
    }
    round_trip::<Gf1>(1, &[2, 0xFF, u64::MAX]);
    round_trip::<Binary<2, Polynomial<0x7>>>(3, &[4, 0xFF, u64::MAX]);
    round_trip::<Binary<2, Normal<0x7, 2>>>(3, &[4, 0xFF, u64::MAX]);
    round_trip::<Binary<2, Cantor<0x7, 2>>>(3, &[4, 0xFF, u64::MAX]);
    round_trip::<Binary<4, Polynomial<0x13>>>(0xF, &[0x10, 0xFF, u64::MAX]);
    round_trip::<Binary<4, Normal<0x13, 8>>>(0xF, &[0x10, 0xFF, u64::MAX]);
    round_trip::<Binary<4, Cantor<0x13, 8>>>(0xF, &[0x10, 0xFF, u64::MAX]);
    round_trip::<Binary<8, Polynomial<AES>>>(0xFF, &[0x100, 0xFFFF, u64::MAX]);
    round_trip::<Binary<8, Normal<0x11B, 0x20>>>(0xFF, &[0x100, 0xFFFF, u64::MAX]);
    round_trip::<Binary<8, Cantor<0x11B, 0x20>>>(0xFF, &[0x100, 0xFFFF, u64::MAX]);
    round_trip::<Gf16>(0xFFFF, &[0x1_0000, u64::MAX]);
}

// ---------------------------------------------------------------------------
// GF(2^16)
// ---------------------------------------------------------------------------

/// Schoolbook `(a + b*u)(c + d*u)` reduced by `u^2 = u + DELTA`, using the
/// GF(2^8) shift-and-XOR multiply. Independent of the Karatsuba form under
/// test and of the log tables.
fn gf16_mul_oracle(x: Elem<Gf16>, y: Elem<Gf16>) -> Elem<Gf16> {
    let (a, b) = x.to_components();
    let (c, d) = y.to_components();
    let xt = |p: Elem<Binary<8, Polynomial<AES>>>, q: Elem<Binary<8, Polynomial<AES>>>| {
        Elem::<Binary<8, Polynomial<AES>>>::from_raw(xtime_mul(
            p.to_raw(),
            q.to_raw(),
            Polynomial::<AES>::POLY,
        ))
    };
    let ac = xt(a, c);
    let ad = xt(a, d);
    let bc = xt(b, c);
    let bd = xt(b, d);
    // ac + (ad + bc)u + bd*u^2, and u^2 = u + DELTA.
    let constant = ac.add(xt(Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x20), bd));
    let extension = ad.add(bc).add(bd);
    Elem::<Gf16>::from_components(constant, extension)
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
    assert_eq!(
        Elem::<Gf16>::ZERO.inv(),
        Elem::<Gf16>::ZERO,
        "inv(0) must be 0"
    );
    for a in sample_gf16() {
        if a == Elem::<Gf16>::ZERO {
            continue;
        }
        assert_eq!(a.mul(a.inv()), Elem::<Gf16>::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), Elem::<Gf16>::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf16_generator_has_full_order() {
    // Order must be exactly 65535: g^65535 == 1 and g^(65535/p) != 1 for each
    // prime factor p of 65535 = 3 * 5 * 17 * 257.
    let g = Elem::<Gf16>::GENERATOR;
    assert_eq!(g.pow(65_535), Elem::<Gf16>::ONE, "g^65535 != 1");
    for factor in [3u128, 5, 17, 257] {
        assert_ne!(
            g.pow(65_535 / factor),
            Elem::<Gf16>::ONE,
            "generator order divides 65535/{factor}"
        );
    }
}

#[test]
fn gf16_field_axioms() {
    let sample: Vec<_> = sample_gf16().into_iter().step_by(37).collect();
    for &a in &sample {
        assert_eq!(a.add(Elem::<Gf16>::ZERO), a);
        assert_eq!(a.mul(Elem::<Gf16>::ONE), a);
        assert_eq!(a.mul(Elem::<Gf16>::ZERO), Elem::<Gf16>::ZERO);
        assert_eq!(a.add(a), Elem::<Gf16>::ZERO, "characteristic two");
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
    for a in all_gf8_elems::<AES>().step_by(5) {
        for b in all_gf8_elems::<AES>().step_by(7) {
            let lifted = Elem::<Gf16>::from_components(a, b).mul(Elem::<Gf16>::from_components(
                Elem::<Binary<8, Polynomial<AES>>>::from_raw(0),
                Elem::<Binary<8, Polynomial<AES>>>::from_raw(0),
            ));
            assert_eq!(lifted, Elem::<Gf16>::ZERO);

            let x =
                Elem::<Gf16>::from_components(a, Elem::<Binary<8, Polynomial<AES>>>::from_raw(0));
            let y =
                Elem::<Gf16>::from_components(b, Elem::<Binary<8, Polynomial<AES>>>::from_raw(0));
            assert_eq!(
                x.mul(y),
                Elem::<Gf16>::from_components(
                    a.mul(b),
                    Elem::<Binary<8, Polynomial<AES>>>::from_raw(0)
                ),
                "base-field embedding broken for {a:?} * {b:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^32) and GF(2^64)
// ---------------------------------------------------------------------------

fn sample_gf32() -> Vec<Elem<Gf32>> {
    let mut values = vec![
        Elem::<Gf32>::ZERO,
        Elem::<Gf32>::ONE,
        Elem::<Gf32>::from_raw(u32::MAX),
        Elem::<Gf32>::from_raw(0x0000_ffff),
        Elem::<Gf32>::from_raw(0xffff_0000),
        Elem::<Gf32>::from_components(Elem::<Gf16>::from_raw(0x2000), Elem::<Gf16>::ZERO),
        Elem::<Gf32>::GENERATOR,
    ];
    let mut state = 0x243f_6a88u32;
    for _ in 0..48 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(Elem::<Gf32>::from_raw(state));
    }
    values
}

fn sample_gf64() -> Vec<Elem<Gf64>> {
    let mut values = vec![
        Elem::<Gf64>::ZERO,
        Elem::<Gf64>::ONE,
        Elem::<Gf64>::from_raw(u64::MAX),
        Elem::<Gf64>::from_raw(0x0000_0000_ffff_ffff),
        Elem::<Gf64>::from_raw(0xffff_ffff_0000_0000),
        Elem::<Gf64>::GENERATOR,
    ];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(Elem::<Gf64>::from_raw(state));
    }
    values
}

fn gf32_mul_oracle(x: Elem<Gf32>, y: Elem<Gf32>) -> Elem<Gf32> {
    let (a, b) = x.to_components();
    let (c, d) = y.to_components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    Elem::<Gf32>::from_components(
        ac.add(Elem::<Gf16>::from_raw(0x2000).mul(bd)),
        ad.add(bc).add(bd),
    )
}

fn gf64_mul_oracle(x: Elem<Gf64>, y: Elem<Gf64>) -> Elem<Gf64> {
    let (a, b) = x.to_components();
    let (c, d) = y.to_components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    Elem::<Gf64>::from_components(
        ac.add(Elem::<Gf32>::from_raw(0x2000_0000).mul(bd)),
        ad.add(bc).add(bd),
    )
}

#[test]
fn gf32_tower_arithmetic() {
    let sample = sample_gf32();
    for (i, &a) in sample.iter().enumerate() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
        assert_eq!(a.add(a), Elem::<Gf32>::ZERO);
        if a != Elem::<Gf32>::ZERO {
            assert_eq!(a.mul(a.inv()), Elem::<Gf32>::ONE, "inverse({a:?})");
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
        assert_eq!(a.add(a), Elem::<Gf64>::ZERO);
        if a != Elem::<Gf64>::ZERO {
            assert_eq!(a.mul(a.inv()), Elem::<Gf64>::ONE, "inverse({a:?})");
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
    let g32 = Elem::<Gf32>::GENERATOR;
    let order32 = u32::MAX as u128;
    assert_eq!(g32.pow(order32), Elem::<Gf32>::ONE);
    for factor in [3u128, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), Elem::<Gf32>::ONE);
    }

    let g64 = Elem::<Gf64>::GENERATOR;
    let order64 = u128::from(u64::MAX);
    assert_eq!(g64.pow(order64), Elem::<Gf64>::ONE);
    for factor in [3u128, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(g64.pow(order64 / factor), Elem::<Gf64>::ONE);
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

fn sample_m31() -> Vec<Elem<Mersenne31>> {
    // Canonical boundaries, non-canonical raw lanes (>= p), and a spray.
    let mut values: Vec<Elem<Mersenne31>> = [
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
    .map(Elem::<Mersenne31>::from_raw)
    .collect();
    let mut state = 0x243f_6a88u32;
    for _ in 0..64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(Elem::<Mersenne31>::from_raw(state));
    }
    values
}

fn sample_gld() -> Vec<Elem<Goldilocks>> {
    let mut values: Vec<Elem<Goldilocks>> = [
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
    .map(Elem::<Goldilocks>::from_raw)
    .collect();
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..48 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(Elem::<Goldilocks>::from_raw(state));
    }
    values
}

fn m31_mul_oracle(x: Elem<Mersenne31>, y: Elem<Mersenne31>) -> Elem<Mersenne31> {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    Elem::<Mersenne31>::from_raw((a * b % M31_P) as u32)
}
fn m31_add_oracle(x: Elem<Mersenne31>, y: Elem<Mersenne31>) -> Elem<Mersenne31> {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    Elem::<Mersenne31>::from_raw(((a + b) % M31_P) as u32)
}
fn m31_sub_oracle(x: Elem<Mersenne31>, y: Elem<Mersenne31>) -> Elem<Mersenne31> {
    let a = x.to_raw() as u128 % M31_P;
    let b = y.to_raw() as u128 % M31_P;
    Elem::<Mersenne31>::from_raw(((a + M31_P - b) % M31_P) as u32)
}
fn gld_mul_oracle(x: Elem<Goldilocks>, y: Elem<Goldilocks>) -> Elem<Goldilocks> {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    Elem::<Goldilocks>::from_raw((a * b % GLD_P) as u64)
}
fn gld_add_oracle(x: Elem<Goldilocks>, y: Elem<Goldilocks>) -> Elem<Goldilocks> {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    Elem::<Goldilocks>::from_raw(((a + b) % GLD_P) as u64)
}
fn gld_sub_oracle(x: Elem<Goldilocks>, y: Elem<Goldilocks>) -> Elem<Goldilocks> {
    let a = u128::from(x.to_raw()) % GLD_P;
    let b = u128::from(y.to_raw()) % GLD_P;
    Elem::<Goldilocks>::from_raw(((a + GLD_P - b) % GLD_P) as u64)
}

#[test]
fn m31_known_answer_products() {
    use fgf::Mersenne31 as _M31;
    type Elem = fgf::Elem<_M31>;
    assert_eq!(
        Elem::from_raw(0x5555_5555).mul(Elem::from_raw(0x5555_5555)),
        Elem::from_raw(0x71C7_1C71)
    );
    assert_eq!(
        Elem::from_raw(0x5555_5555).mul(Elem::from_raw(0x7FFF_FFFE)),
        Elem::from_raw(0x2AAA_AAAA)
    );
    assert_eq!(
        Elem::from_raw(0x7FFF_FFFE).mul(Elem::from_raw(0x7FFF_FFFE)),
        Elem::from_raw(0x0000_0001)
    );
    assert_eq!(
        Elem::from_raw(0x7FFF_FFFD).mul(Elem::from_raw(0x7FFF_FFFE)),
        Elem::from_raw(0x0000_0002)
    );
    assert_eq!(
        Elem::from_raw(0x5555_5555).mul(Elem::from_raw(0x5EAD_BEF0)),
        Elem::from_raw(0x74E4_94FA)
    );
    assert_eq!(Elem::from_raw(2).inv(), Elem::from_raw(0x4000_0000));
    assert_eq!(Elem::from_raw(0x5555_5555).inv(), Elem::from_raw(3));
    // Fold known-answers: non-canonical lanes reduce branchlessly.
    assert_eq!(mersenne31::reduce(0x8000_0000), 1);
    assert_eq!(mersenne31::reduce(0xFFFF_FFFF), 1);
    assert_eq!(mersenne31::reduce(0x7FFF_FFFF), 0);
}

#[test]
fn gld_known_answer_products() {
    use fgf::Goldilocks as _Gld;
    type Elem = fgf::Elem<_Gld>;
    assert_eq!(
        Elem::from_raw(0x5555_5555_5555_5555).mul(Elem::from_raw(0xAAAA_AAAA_AAAA_AAAA)),
        Elem::from_raw(0xFFFF_FFFE_5555_5557)
    );
    assert_eq!(
        Elem::from_raw(0xFFFF_FFFF_0000_0000).mul(Elem::from_raw(0xFFFF_FFFF_0000_0000)),
        Elem::from_raw(0x0000_0000_0000_0001)
    );
    assert_eq!(
        Elem::from_raw(0x7FFF_FFFF_FFFF_FFFF).mul(Elem::from_raw(0x7FFF_FFFF_FFFF_FFFF)),
        Elem::from_raw(0xFFFF_FFFD_C000_0003)
    );
    assert_eq!(
        Elem::from_raw(0x0000_0000_7FFF_FFFF).mul(Elem::from_raw(0xFFFF_FFFF_0000_0000)),
        Elem::from_raw(0xFFFF_FFFE_8000_0002)
    );
    assert_eq!(
        Elem::from_raw(2).inv(),
        Elem::from_raw(0x7FFF_FFFF_8000_0001)
    );
    assert_eq!(
        Elem::from_raw(7).inv(),
        Elem::from_raw(0x2492_4924_6DB6_DB6E)
    );
    // Fold known-answers pinning the split reduction chain.
    assert_eq!(goldilocks::reduce(goldilocks::MODULUS), 0);
    assert_eq!(goldilocks::reduce(u64::MAX), 0xFFFF_FFFE);
    assert_eq!(goldilocks::reduce_wide(1u128 << 64), 0xFFFF_FFFF); // 2^64 = 2^32 - 1
    assert_eq!(goldilocks::reduce_wide(1u128 << 96), 0xFFFF_FFFF_0000_0000); // 2^96 = -1
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
        assert_eq!(a.add(a.neg()), Elem::<Mersenne31>::ZERO);
        assert_eq!(a.neg().neg(), a);
        assert_eq!(a.sub(b), a.add(b.neg()));
        // Ring laws.
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        // Inverse/division totality and round trip.
        assert_eq!(a.div(Elem::<Mersenne31>::ZERO), Elem::<Mersenne31>::ZERO);
        if a != Elem::<Mersenne31>::ZERO {
            assert_eq!(a.mul(a.inv()), Elem::<Mersenne31>::ONE, "inv({a:?})");
            assert_eq!(a.div(a), Elem::<Mersenne31>::ONE);
        }
    }
    assert_eq!(Elem::<Mersenne31>::ZERO.inv(), Elem::<Mersenne31>::ZERO);
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
        assert_eq!(a.add(a.neg()), Elem::<Goldilocks>::ZERO);
        assert_eq!(a.neg().neg(), a);
        assert_eq!(a.sub(b), a.add(b.neg()));
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        assert_eq!(a.div(Elem::<Goldilocks>::ZERO), Elem::<Goldilocks>::ZERO);
        if a != Elem::<Goldilocks>::ZERO {
            assert_eq!(a.mul(a.inv()), Elem::<Goldilocks>::ONE, "inv({a:?})");
            assert_eq!(a.div(a), Elem::<Goldilocks>::ONE);
        }
    }
    assert_eq!(Elem::<Goldilocks>::ZERO.inv(), Elem::<Goldilocks>::ZERO);
}

#[test]
fn prime_generators_have_full_order() {
    // Multiplicative order is exactly p - 1: full order, and not a proper
    // divisor for any prime factor q of p - 1.
    let g = Elem::<Mersenne31>::GENERATOR;
    let order = 0x7FFF_FFFEu128; // p - 1 = 2 * 3^2 * 7 * 11 * 31 * 151 * 331
    assert_eq!(g.pow(order), Elem::<Mersenne31>::ONE);
    for q in [2u128, 3, 7, 11, 31, 151, 331] {
        assert_ne!(g.pow(order / q), Elem::<Mersenne31>::ONE, "M31 factor {q}");
    }

    let g = Elem::<Goldilocks>::GENERATOR;
    let order = 0xFFFF_FFFF_0000_0000u128; // p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537
    assert_eq!(g.pow(order), Elem::<Goldilocks>::ONE);
    for q in [2u128, 3, 5, 17, 257, 65_537] {
        assert_ne!(g.pow(order / q), Elem::<Goldilocks>::ONE, "GLD factor {q}");
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
            let a = Elem::<Mersenne31>::from_raw(ra);
            let b = Elem::<Mersenne31>::from_raw(rb);
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
            let a = Elem::<Goldilocks>::from_raw(ra);
            let b = Elem::<Goldilocks>::from_raw(rb);
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
fn qm_mul_oracle(x: Elem<QuadMersenne31>, y: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let (xr, xi) = x.to_raw();
    let (yr, yi) = y.to_raw();
    let ar = xr as u128 % M31_P;
    let ai = xi as u128 % M31_P;
    let br = yr as u128 % M31_P;
    let bi = yi as u128 % M31_P;
    let re = (ar * br + M31_P * M31_P - ai * bi) % M31_P;
    let im = (ar * bi + ai * br) % M31_P;
    Elem::<QuadMersenne31>::from_raw(re as u32, im as u32)
}
fn qm_add_oracle(x: Elem<QuadMersenne31>, y: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let (xr, xi) = x.to_raw();
    let (yr, yi) = y.to_raw();
    let re = (xr as u128 % M31_P + yr as u128 % M31_P) % M31_P;
    let im = (xi as u128 % M31_P + yi as u128 % M31_P) % M31_P;
    Elem::<QuadMersenne31>::from_raw(re as u32, im as u32)
}
fn qm_sub_oracle(x: Elem<QuadMersenne31>, y: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let (xr, xi) = x.to_raw();
    let (yr, yi) = y.to_raw();
    let re = (xr as u128 % M31_P + M31_P - yr as u128 % M31_P) % M31_P;
    let im = (xi as u128 % M31_P + M31_P - yi as u128 % M31_P) % M31_P;
    Elem::<QuadMersenne31>::from_raw(re as u32, im as u32)
}

fn sample_qm() -> Vec<Elem<QuadMersenne31>> {
    // Boundary pairs plus a deterministic spray of limb pairs.
    let mut values: Vec<Elem<QuadMersenne31>> = [
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
    .map(|(re, im)| Elem::<QuadMersenne31>::from_raw(re, im))
    .collect();
    let mut state = 0x243f_6a88u32;
    for _ in 0..48 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(Elem::<QuadMersenne31>::from_raw(
            state,
            state.rotate_left(7),
        ));
    }
    values
}

#[test]
fn qm31_known_answer_products() {
    use fgf::QuadMersenne31 as _Qm;
    type Elem = fgf::Elem<_Qm>;
    // i^2 = -1
    assert_eq!(
        Elem::from_raw(0, 1).square(),
        Elem::from_raw(0x7FFF_FFFE, 0)
    );
    assert_eq!(
        Elem::from_raw(0, 1).mul(Elem::from_raw(0, 1)),
        Elem::from_raw(0x7FFF_FFFE, 0)
    );
    // (a+ai)^2 = 0 + 2a^2 i with a^2 = 0x71C7_1C71 (the frozen M31 KAT).
    let two_a_sq = fgf::Elem::<fgf::Mersenne31>::from_raw(0x71C7_1C71)
        .add(fgf::Elem::<fgf::Mersenne31>::from_raw(0x71C7_1C71))
        .to_raw();
    assert_eq!(
        Elem::from_raw(0x5555_5555, 0x5555_5555).square(),
        Elem::from_raw(0, two_a_sq)
    );
    // Conjugate/norm inverse: inv(a+bi) = (a-bi)/(a^2+b^2).
    let x = Elem::from_raw(0x1234_5678, 0x9abc_def0 % 0x7FFF_FFFF);
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
        let m = a.mul(b);
        assert!(
            m.to_raw().0 < 0x7FFF_FFFF && m.to_raw().1 < 0x7FFF_FFFF,
            "mul canonical"
        );
        let s = a.add(b);
        assert!(
            s.to_raw().0 < 0x7FFF_FFFF && s.to_raw().1 < 0x7FFF_FFFF,
            "add canonical"
        );
        let d = a.sub(b);
        assert!(
            d.to_raw().0 < 0x7FFF_FFFF && d.to_raw().1 < 0x7FFF_FFFF,
            "sub canonical"
        );
        let g = a.neg();
        assert!(
            g.to_raw().0 < 0x7FFF_FFFF && g.to_raw().1 < 0x7FFF_FFFF,
            "neg canonical"
        );
        // Differentials against the u128 oracle.
        assert_eq!(a.mul(b), qm_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.add(b), qm_add_oracle(a, b), "{a:?} + {b:?}");
        assert_eq!(a.sub(b), qm_sub_oracle(a, b), "{a:?} - {b:?}");
        // Negation laws and sub == add of neg.
        assert_eq!(a.add(a.neg()), Elem::<QuadMersenne31>::ZERO);
        assert_eq!(a.neg().neg(), a);
        assert_eq!(a.sub(b), a.add(b.neg()));
        // Ring laws.
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.square(), a.mul(a));
        // Inverse/division totality and round trip.
        assert_eq!(
            a.div(Elem::<QuadMersenne31>::ZERO),
            Elem::<QuadMersenne31>::ZERO
        );
        if a != Elem::<QuadMersenne31>::ZERO {
            assert_eq!(a.mul(a.inv()), Elem::<QuadMersenne31>::ONE, "inv({a:?})");
            assert_eq!(a.div(a), Elem::<QuadMersenne31>::ONE);
        }
        // Norm is the base-field element a^2 + b^2.
        let n = a.norm();
        let (ar, ai) = a.to_raw();
        assert_eq!(
            n,
            Elem::<Mersenne31>::from_raw(ar)
                .mul(Elem::<Mersenne31>::from_raw(ar))
                .add(Elem::<Mersenne31>::from_raw(ai).mul(Elem::<Mersenne31>::from_raw(ai)))
        );
    }
    assert_eq!(
        Elem::<QuadMersenne31>::ZERO.inv(),
        Elem::<QuadMersenne31>::ZERO
    );
}

#[test]
fn qm31_generator_has_full_order() {
    // p^2 - 1 = 2^32 * 3^2 * 7 * 11 * 31 * 151 * 331; order is exactly that.
    let g = Elem::<QuadMersenne31>::GENERATOR;
    let order = 0x3FFF_FFFF_0000_0000u128;
    assert_eq!(g.pow(order), Elem::<QuadMersenne31>::ONE);
    for q in [2u128, 3, 7, 11, 31, 151, 331] {
        assert_ne!(
            g.pow(order / q),
            Elem::<QuadMersenne31>::ONE,
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
            let a = Elem::<QuadMersenne31>::from_raw(ra, rai);
            let b = Elem::<QuadMersenne31>::from_raw(rb, rbi);
            let m = a.mul(b);
            assert!(m.to_raw().0 < 0x7FFF_FFFF && m.to_raw().1 < 0x7FFF_FFFF);
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
        Elem::<FanPaar8Field>::from_raw(0x1b).mul(Elem::<FanPaar8Field>::from_raw(0xa8)),
        Elem::<FanPaar8Field>::from_raw(0x09)
    );
    assert_eq!(
        Elem::<FanPaar16Field>::from_raw(0x48a8).mul(Elem::<FanPaar16Field>::from_raw(0xf8a4)),
        Elem::<FanPaar16Field>::from_raw(0x3656)
    );
    assert_eq!(
        Elem::<FanPaar16Field>::from_raw(0xf8a4).square(),
        Elem::<FanPaar16Field>::from_raw(0xe7e6)
    );
    assert_eq!(
        Elem::<FanPaar64Field>::from_raw(0xc84d_6191_1083_1cef)
            .mul(Elem::<FanPaar64Field>::from_raw(0x0000_0000_0000_a14f)),
        Elem::<FanPaar64Field>::from_raw(0x3565_086d_6b9e_f595)
    );
}

#[test]
fn fan_paar_arithmetic_round_trips() {
    macro_rules! check {
        ($field:ty, $($value:expr),+ $(,)?) => {
            $(
                let a = Elem::<$field>::from_raw($value);
                assert_eq!(a.square(), a.mul(a));
                assert_eq!(a.add(a), Elem::<$field>::ZERO);
                if a != Elem::<$field>::ZERO {
                    assert_eq!(a.mul(a.inv()), Elem::<$field>::ONE);
                }
            )+
        };
    }

    check!(FanPaar8Field, 0, 1, 0x2d, 0x53, u8::MAX);
    check!(FanPaar16Field, 0, 1, 0xe2de, 0x1234, u16::MAX);
    check!(FanPaar32Field, 0, 1, 0x03e2_1cea, 0xdead_beef, u32::MAX);
    check!(
        FanPaar64Field,
        0,
        1,
        0x070f_870d_cd9c_1d88,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    );
}

#[test]
fn fan_paar_generators_have_full_order() {
    let g8 = Elem::<FanPaar8Field>::GENERATOR;
    for factor in [3u128, 5, 17] {
        assert_ne!(g8.pow(255 / factor), Elem::<FanPaar8Field>::ONE);
    }
    assert_eq!(g8.pow(255), Elem::<FanPaar8Field>::ONE);

    let g16 = Elem::<FanPaar16Field>::GENERATOR;
    for factor in [3u128, 5, 17, 257] {
        assert_ne!(g16.pow(65_535 / factor), Elem::<FanPaar16Field>::ONE);
    }
    assert_eq!(g16.pow(65_535), Elem::<FanPaar16Field>::ONE);

    let g32 = Elem::<FanPaar32Field>::GENERATOR;
    let order32 = u32::MAX as u128;
    for factor in [3u128, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), Elem::<FanPaar32Field>::ONE);
    }
    assert_eq!(g32.pow(order32), Elem::<FanPaar32Field>::ONE);

    let g64 = Elem::<FanPaar64Field>::GENERATOR;
    for factor in [3u128, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(
            g64.pow(u128::from(u64::MAX) / factor),
            Elem::<FanPaar64Field>::ONE
        );
    }
    assert_eq!(g64.pow(u128::from(u64::MAX)), Elem::<FanPaar64Field>::ONE);
}

#[test]
fn fan_paar_subfield_encodings_are_nested() {
    for (a, b) in [(0x1bu8, 0xa8u8), (0x53, 0xca), (0xff, 0x42)] {
        let product = Elem::<FanPaar8Field>::from_raw(a)
            .mul(Elem::<FanPaar8Field>::from_raw(b))
            .to_raw();
        assert_eq!(
            Elem::<FanPaar16Field>::from_raw(u16::from(a))
                .mul(Elem::<FanPaar16Field>::from_raw(u16::from(b)))
                .to_raw(),
            u16::from(product)
        );
        assert_eq!(
            Elem::<FanPaar32Field>::from_raw(u32::from(a))
                .mul(Elem::<FanPaar32Field>::from_raw(u32::from(b)))
                .to_raw(),
            u32::from(product)
        );
        assert_eq!(
            Elem::<FanPaar64Field>::from_raw(u64::from(a))
                .mul(Elem::<FanPaar64Field>::from_raw(u64::from(b)))
                .to_raw(),
            u64::from(product)
        );
    }
}

// ---------------------------------------------------------------------------
// Representation
// ---------------------------------------------------------------------------

#[test]
fn byte_representation_round_trips() {
    for a in all_gf8_elems::<AES>() {
        let mut buffer = [0u8; 1];
        Binary::<8, Polynomial<AES>>::encode(&mut buffer, a);
        assert_eq!(Binary::<8, Polynomial<AES>>::decode(&buffer), a);
    }
    for a in sample_gf16() {
        let mut buffer = [0u8; 2];
        Gf16::encode(&mut buffer, a);
        assert_eq!(Gf16::decode(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf32() {
        let mut buffer = [0u8; 4];
        Gf32::encode(&mut buffer, a);
        assert_eq!(Gf32::decode(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf64() {
        let mut buffer = [0u8; 8];
        Gf64::encode(&mut buffer, a);
        assert_eq!(Gf64::decode(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_m31() {
        let mut buffer = [0u8; 4];
        Mersenne31::encode(&mut buffer, a);
        assert_eq!(Mersenne31::decode(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gld() {
        let mut buffer = [0u8; 8];
        Goldilocks::encode(&mut buffer, a);
        assert_eq!(Goldilocks::decode(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    macro_rules! check_fan_paar_repr {
        ($field:ty, $elem:expr, $bytes:literal) => {{
            let value = $elem;
            let mut buffer = [0u8; $bytes];
            <$field>::encode(&mut buffer, value);
            assert_eq!(<$field>::decode(&buffer), value);
            assert_eq!(buffer, value.to_bytes());
        }};
    }
    check_fan_paar_repr!(FanPaar8Field, Elem::<FanPaar8Field>::from_raw(0xa5), 1);
    check_fan_paar_repr!(FanPaar16Field, Elem::<FanPaar16Field>::from_raw(0xa55a), 2);
    check_fan_paar_repr!(
        FanPaar32Field,
        Elem::<FanPaar32Field>::from_raw(0xa55a_1234),
        4
    );
    check_fan_paar_repr!(
        FanPaar64Field,
        Elem::<FanPaar64Field>::from_raw(0xa55a_1234_dead_beef),
        8
    );
}

#[test]
fn field_constants_are_consistent() {
    assert_eq!(<Binary<8, Polynomial<AES>> as FieldBuffer>::BYTES, 1);
    assert_eq!(
        Binary::<8, Polynomial<AES>>::ORDER,
        1u128 << Binary::<8, Polynomial<AES>>::DEGREE
    );
    assert_eq!(<Gf16 as FieldBuffer>::BYTES, 2);
    assert_eq!(Gf16::ORDER, 1u128 << Gf16::DEGREE);
    assert_eq!(<Gf32 as FieldBuffer>::BYTES, 4);
    assert_eq!(Gf32::ORDER, 1u128 << Gf32::DEGREE);
    assert_eq!(<Gf64 as FieldBuffer>::BYTES, 8);
    assert_eq!(Gf64::ORDER, 1u128 << Gf64::DEGREE);
    // Prime fields: ORDER is the modulus, not 2^STORAGE_BITS, and
    // STORAGE_BITS is the lane width (8 * BYTES), not log2(ORDER).
    assert_eq!(<Mersenne31 as FieldBuffer>::BYTES, 4);
    assert_eq!(Mersenne31::ORDER, 0x7FFF_FFFF);
    assert_eq!(<QuadMersenne31 as FieldBuffer>::BYTES, 8);
    assert_eq!(QuadMersenne31::ORDER, 0x3FFF_FFFF_0000_0001);
    assert_eq!(<Goldilocks as FieldBuffer>::BYTES, 8);
    assert_eq!(Goldilocks::ORDER, 0xFFFF_FFFF_0000_0001);
    for (bytes, bits) in [
        (
            <Binary<8, Polynomial<AES>> as FieldBuffer>::BYTES,
            <Binary<8, Polynomial<AES>> as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <Gf16 as FieldBuffer>::BYTES,
            <Gf16 as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <Gf32 as FieldBuffer>::BYTES,
            <Gf32 as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <Gf64 as FieldBuffer>::BYTES,
            <Gf64 as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <FanPaar8Field as FieldBuffer>::BYTES,
            <FanPaar8Field as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <FanPaar16Field as FieldBuffer>::BYTES,
            <FanPaar16Field as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <FanPaar32Field as FieldBuffer>::BYTES,
            <FanPaar32Field as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <FanPaar64Field as FieldBuffer>::BYTES,
            <FanPaar64Field as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <Mersenne31 as FieldBuffer>::BYTES,
            <Mersenne31 as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <Goldilocks as FieldBuffer>::BYTES,
            <Goldilocks as FieldBuffer>::STORAGE_BITS,
        ),
        (
            <QuadMersenne31 as FieldBuffer>::BYTES,
            <QuadMersenne31 as FieldBuffer>::STORAGE_BITS,
        ),
    ] {
        assert_eq!(bytes * 8, bits as usize);
    }
}

/// Embed the integer `n` into the field by double-and-add over `ONE`.
///
/// The only portable integer embedding: a raw byte pattern is not an integer
/// in a binary extension field, and a lane value is not one in QM31.
fn embed<F: Field>(mut n: u64) -> Elem<F> {
    use fgf::field::FieldElem as _;
    let mut term = Elem::<F>::ONE;
    let mut total = Elem::<F>::ZERO;
    while n != 0 {
        if n & 1 != 0 {
            total = total.add(term);
        }
        term = term.add(term);
        n >>= 1;
    }
    total
}

#[test]
fn field_characteristic_is_not_derived_from_order() {
    use fgf::field::FieldElem as _;

    fn characteristic<F: Field>() -> u64 {
        <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC
    }

    // Binary towers: order is 2^m, characteristic is two.
    for c in [
        characteristic::<Binary<8, Polynomial<AES>>>(),
        characteristic::<Binary<8, Polynomial<RS>>>(),
        characteristic::<Gf16>(),
        characteristic::<Gf32>(),
        characteristic::<Gf64>(),
        characteristic::<FanPaar8Field>(),
        characteristic::<FanPaar16Field>(),
        characteristic::<FanPaar32Field>(),
        characteristic::<FanPaar64Field>(),
    ] {
        assert_eq!(c, 2);
    }

    // Prime fields of prime order: the two facts coincide.
    assert_eq!(
        u128::from(characteristic::<Mersenne31>()),
        Mersenne31::ORDER
    );
    assert_eq!(
        u128::from(characteristic::<Goldilocks>()),
        Goldilocks::ORDER
    );

    // The extension: characteristic is the base prime `p`, while the order
    // is `p²`. Deriving one from the other is the defect this pins.
    assert_eq!(
        characteristic::<QuadMersenne31>(),
        characteristic::<Mersenne31>()
    );
    assert_ne!(
        u128::from(characteristic::<QuadMersenne31>()),
        QuadMersenne31::ORDER
    );
    let base = u128::from(characteristic::<QuadMersenne31>());
    assert_eq!(base * base, QuadMersenne31::ORDER);

    // Behaviour, not just constants: `p · x = 0` and `(p − 1) · x ≠ 0`.
    macro_rules! check_embedding {
        ($($field:ty),+) => {$({
            assert!(embed::<$field>(characteristic::<$field>()).is_zero());
            assert!(!embed::<$field>(characteristic::<$field>() - 1).is_zero());
        })+};
    }
    check_embedding!(
        Binary<8, Polynomial<AES>>,
        Binary<8, Polynomial<RS>>,
        Gf16,
        Gf32,
        Gf64,
        FanPaar8Field,
        FanPaar16Field,
        FanPaar32Field,
        FanPaar64Field,
        Mersenne31,
        Goldilocks,
        QuadMersenne31
    );
}

// ---------------------------------------------------------------------------
// Shared trait, operator, and formatting surface
// ---------------------------------------------------------------------------
//
// The per-field tests above drive the inherent `const` methods. Generic
// consumers reach the same algebra through the `field::FieldElem` trait, the
// `core::ops` operator impls, the `Sum`/`Product` folds, and the
// `Debug`/`Display`/`Default`/`Hash` impls — none of which the inherent
// callsites touch. This section exercises those surfaces for every field
// against the same laws, so a broken delegation cannot hide behind a correct
// inherent body.

use fgf::field::FieldElem as _;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hashes_equal<T: Hash>(a: &T, b: &T) -> bool {
    let mut ha = DefaultHasher::new();
    let mut hb = DefaultHasher::new();
    a.hash(&mut ha);
    b.hash(&mut hb);
    ha.finish() == hb.finish()
}

fn empty_sum_of<E: fgf::field::FieldElem + Sum>(_seed: E) -> E {
    std::iter::empty::<E>().sum()
}

fn empty_product_of<E: fgf::field::FieldElem + Product>(_seed: E) -> E {
    std::iter::empty::<E>().product()
}

fn sum_of<E: fgf::field::FieldElem + Sum>(it: impl Iterator<Item = E>) -> E {
    it.sum()
}

fn product_of<E: fgf::field::FieldElem + Product>(it: impl Iterator<Item = E>) -> E {
    it.product()
}

fn sum_of_ref<'a, E: fgf::field::FieldElem + Sum<&'a E>>(it: impl Iterator<Item = &'a E>) -> E {
    it.sum()
}

fn product_of_ref<'a, E: fgf::field::FieldElem + Product<&'a E>>(
    it: impl Iterator<Item = &'a E>,
) -> E {
    it.product()
}

fn empty_sum_of_ref<'a, E: fgf::field::FieldElem + Sum<&'a E>>(_seed: &'a E) -> E {
    std::iter::empty::<&'a E>().sum()
}

fn empty_product_of_ref<'a, E: fgf::field::FieldElem + Product<&'a E>>(_seed: &'a E) -> E {
    std::iter::empty::<&'a E>().product()
}

use std::iter::{Product, Sum};

type Gf1 = fgf::Binary<1, fgf::Polynomial<3>>;
type Gf16 = fgf::Binary<16, fgf::Tower<fgf::Rijndael16>>;
type Gf32 = fgf::Binary<32, fgf::Tower<fgf::Rijndael32>>;
type Gf64 = fgf::Binary<64, fgf::Tower<fgf::Rijndael64>>;
type FanPaar8Field = fgf::Binary<8, fgf::Tower<fgf::FanPaar8>>;
type FanPaar16Field = fgf::Binary<16, fgf::Tower<fgf::FanPaar16>>;
type FanPaar32Field = fgf::Binary<32, fgf::Tower<fgf::FanPaar32>>;
type FanPaar64Field = fgf::Binary<64, fgf::Tower<fgf::FanPaar64>>;

/// Every `field::FieldElem`/`field::Field` surface reachable from generic code:
/// the trait's arithmetic (including defaulted methods), the total
/// zero conventions, `Debug`/`Hash`/`Default`, and the byte codec.
fn exercise_surface<F: FieldBuffer + HasGenerator>(samples: &[Elem<F>]) {
    let zero = Elem::<F>::ZERO;
    let one = Elem::<F>::ONE;
    assert!(samples.len() >= 2, "surface sweep needs samples");
    assert_eq!(
        Elem::<F>::default(),
        zero,
        "Default must be the additive identity"
    );

    // Field facts that hold for every field, checked through the trait.
    assert!(!F::NAME.is_empty());
    assert_eq!(F::STORAGE_BITS as usize, 8 * F::BYTES);
    assert!(!Elem::<F>::GENERATOR.is_zero());
    assert!(zero.is_zero());
    assert!(!one.is_zero());
    assert!(one.is_one());
    assert!(!zero.is_one());

    for &a in samples {
        // Stable encoding round trip through the Field contract.
        let mut buffer = [0u8; 16];
        F::encode(&mut buffer[..F::BYTES], a);
        assert_eq!(F::decode(&buffer[..F::BYTES]), a, "write/read round trip");

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

        assert!(hashes_equal(&a, &a.clone()), "equal elements hash equally");
    }
}

#[test]
fn element_formatting_reports_insufficient_writer_capacity() {
    use std::fmt::{self, Write};

    struct SliceWriter<'a> {
        bytes: &'a mut [u8],
        used: usize,
    }

    impl Write for SliceWriter<'_> {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            let end = self.used.checked_add(text.len()).ok_or(fmt::Error)?;
            let destination = self.bytes.get_mut(self.used..end).ok_or(fmt::Error)?;
            destination.copy_from_slice(text.as_bytes());
            self.used = end;
            Ok(())
        }
    }

    fn check<E: Default + fmt::Debug + fmt::Display>() {
        let value = E::default();
        let mut storage = [];
        let mut writer = SliceWriter {
            bytes: &mut storage,
            used: 0,
        };
        assert!(matches!(write!(writer, "{value:?}"), Err(fmt::Error)));
        assert!(matches!(write!(writer, "{value}"), Err(fmt::Error)));
    }

    check::<Elem<Binary<8, Polynomial<AES>>>>();
    check::<Elem<Binary<8, Polynomial<RS>>>>();
    check::<Elem<Binary<8, Polynomial<0x12D>>>>();
    check::<Elem<Binary<8, Polynomial<0x187>>>>();
    check::<Elem<Gf16>>();
    check::<Elem<Gf32>>();
    check::<Elem<Gf64>>();
    check::<Elem<FanPaar8Field>>();
    check::<Elem<FanPaar16Field>>();
    check::<Elem<FanPaar32Field>>();
    check::<Elem<FanPaar64Field>>();
    check::<Elem<Mersenne31>>();
    check::<Elem<Goldilocks>>();
    check::<Elem<QuadMersenne31>>();
    check::<Elem<Gf1>>();
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
        assert_eq!(-a, a.neg(), "Neg");

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
    }};
}

#[test]
fn gf8_trait_operator_and_formatting_surface() {
    fn surface<const POLY: u128>() {
        let samples: Vec<_> = all_gf8_elems::<POLY>().step_by(97).collect();
        exercise_surface::<Binary<8, Polynomial<POLY>>>(&samples);
        exercise_operators!(samples);
    }
    every_gf8_field!(surface);
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
    let fp8: Vec<_> = (0..=u8::MAX)
        .step_by(97)
        .map(Elem::<FanPaar8Field>::from_raw)
        .collect();
    exercise_surface::<FanPaar8Field>(&fp8);
    exercise_operators!(fp8);
    let fp16: Vec<_> = [0, 1, 0x0100, 0xffff, 0xa55a, 0x1234]
        .into_iter()
        .map(Elem::<FanPaar16Field>::from_raw)
        .collect();
    exercise_surface::<FanPaar16Field>(&fp16);
    exercise_operators!(fp16);
    let fp32: Vec<_> = [0, 1, 0x10000, 0xffff_ffff, 0xa55a_1234]
        .into_iter()
        .map(Elem::<FanPaar32Field>::from_raw)
        .collect();
    exercise_surface::<FanPaar32Field>(&fp32);
    exercise_operators!(fp32);
    let fp64: Vec<_> = [0, 1, 1 << 32, u64::MAX, 0xa55a_1234_dead_beef]
        .into_iter()
        .map(Elem::<FanPaar64Field>::from_raw)
        .collect();
    exercise_surface::<FanPaar64Field>(&fp64);
    exercise_operators!(fp64);
}

#[test]
fn prime_trait_operator_and_formatting_surface() {
    // Equality follows field values, so the raw samples — which deliberately
    // include non-canonical lanes — can run through the surface laws
    // uncanonicalized: a non-canonical lane and its canonical image compare
    // equal everywhere below.
    let m31: Vec<_> = sample_m31().into_iter().step_by(7).collect();
    exercise_surface::<Mersenne31>(&m31);
    exercise_operators!(m31);
    let gld: Vec<_> = sample_gld().into_iter().step_by(7).collect();
    exercise_surface::<Goldilocks>(&gld);
    exercise_operators!(gld);
    let qm: Vec<_> = sample_qm().into_iter().step_by(7).collect();
    exercise_surface::<QuadMersenne31>(&qm);
    exercise_operators!(qm);
}

/// Inherent helpers that exist beside the trait surface: raw/byte conversions
/// and the tower/extension projections. Each is a distinct public entry point
/// generic code cannot reach, so each gets called here.
#[test]
fn inherent_conversion_helpers_round_trip() {
    // GF(2^8) flat fields.
    fn gf8_conversions<const POLY: u128>() {
        for a in all_gf8_elems::<POLY>().step_by(53) {
            assert_eq!(Elem::<Binary<8, Polynomial<POLY>>>::from_raw(a.to_raw()), a);
            assert_eq!(
                Elem::<Binary<8, Polynomial<POLY>>>::from_bytes(a.to_bytes()),
                a
            );
        }
    }
    every_gf8_field!(gf8_conversions);
    // Towers: component projection is a bijection with from_components.
    for a in sample_gf16().into_iter().step_by(61) {
        let (lo, hi) = a.to_components();
        assert_eq!(Elem::<Gf16>::from_components(lo, hi), a);
        assert_eq!(Elem::<Gf16>::from_raw(a.to_raw()), a);
        assert_eq!(Elem::<Gf16>::from_bytes(a.to_bytes()), a);
    }
    for a in sample_gf32().into_iter().step_by(11) {
        let (lo, hi) = a.to_components();
        assert_eq!(Elem::<Gf32>::from_components(lo, hi), a);
        assert_eq!(Elem::<Gf32>::from_raw(a.to_raw()), a);
        assert_eq!(Elem::<Gf32>::from_bytes(a.to_bytes()), a);
    }
    for a in sample_gf64().into_iter().step_by(11) {
        let (lo, hi) = a.to_components();
        assert_eq!(Elem::<Gf64>::from_components(lo, hi), a);
        assert_eq!(Elem::<Gf64>::from_raw(a.to_raw()), a);
        assert_eq!(Elem::<Gf64>::from_bytes(a.to_bytes()), a);
    }
    // Prime fields: storage is canonical, and lanes below the modulus
    // survive construction unchanged.
    for a in sample_m31().into_iter().step_by(7) {
        assert_eq!(a.to_raw(), a.to_raw() % 0x7FFF_FFFF, "canonical storage");
        assert_eq!(Elem::<Mersenne31>::from_raw(a.to_raw()), a);
        assert_eq!(Elem::<Mersenne31>::from_bytes(a.to_bytes()), a);
        assert_eq!(
            mersenne31::reduce(a.to_raw()),
            a.to_raw(),
            "reduce is the identity on storage"
        );
    }
    for a in sample_gld().into_iter().step_by(7) {
        assert!(
            a.to_raw() < 0xFFFF_FFFF_0000_0001,
            "storage is below the modulus"
        );
        assert_eq!(Elem::<Goldilocks>::from_raw(a.to_raw()), a);
        assert_eq!(Elem::<Goldilocks>::from_bytes(a.to_bytes()), a);
    }
    // Quadratic extension: conjugation and the norm land in the base field.
    for a in sample_qm().into_iter().step_by(7) {
        assert_eq!(
            Elem::<QuadMersenne31>::from_raw(a.to_raw().0, a.to_raw().1),
            a
        );
        assert_eq!(Elem::<QuadMersenne31>::from_bytes(a.to_bytes()), a);
        let (re, im) = a.to_components();
        assert_eq!(Elem::<QuadMersenne31>::from_components(re, im), a);
        assert_eq!(a.conjugate().conjugate(), a, "conjugation is an involution");
        assert_eq!(
            a.conjugate().norm(),
            a.norm(),
            "norm is fixed under conjugation"
        );
        assert!(
            a.norm().to_raw() < 0x7FFF_FFFF,
            "norm lands in the base field"
        );
        let (re, im) = a.mul(a.conjugate()).to_components();
        assert_eq!(
            re.to_raw(),
            a.norm().to_raw(),
            "a * conj(a) is the norm, really"
        );
        assert!(im.to_raw() == 0, "a * conj(a) is real");
    }
    // Fan–Paar levels expose the same raw/byte/component surface.
    macro_rules! fp_level {
        ($elem:ty, $value:expr) => {{
            let a = $value;
            assert_eq!(<$elem>::from_raw(a.to_raw()), a);
            assert_eq!(<$elem>::from_bytes(a.to_bytes()), a);
            let (lo, hi) = a.to_components();
            assert_eq!(<$elem>::from_components(lo, hi), a);
        }};
    }
    fp_level!(
        Elem<FanPaar16Field>,
        Elem::<FanPaar16Field>::from_raw(0xa55a)
    );
    fp_level!(
        Elem<FanPaar32Field>,
        Elem::<FanPaar32Field>::from_raw(0xa55a_1234)
    );
    fp_level!(
        Elem<FanPaar64Field>,
        Elem::<FanPaar64Field>::from_raw(0xa55a_1234_dead_beef)
    );
    assert_eq!(
        Elem::<FanPaar8Field>::from_raw(0xa5).mul_alpha(),
        Elem::<FanPaar8Field>::from_raw(0xa5).mul(Elem::<FanPaar8Field>::from_raw(0x10))
    );
}

// ---------------------------------------------------------------------------
// GF(2)
// ---------------------------------------------------------------------------

/// The whole field, exhaustively: the four ordered pairs.
fn all_gf1_pairs() -> impl Iterator<Item = (Elem<Gf1>, Elem<Gf1>)> {
    let elems = [Elem::<Gf1>::from_raw(0), Elem::<Gf1>::from_raw(1)];
    elems.into_iter().flat_map(move |a| {
        let elems = [Elem::<Gf1>::from_raw(0), Elem::<Gf1>::from_raw(1)];
        elems.into_iter().map(move |b| (a, b))
    })
}

#[test]
fn gf1_add_and_mul_are_xor_and_and() {
    for (a, b) in all_gf1_pairs() {
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
fn gf1_inverse_division_and_power_conventions() {
    for (a, b) in all_gf1_pairs() {
        assert_eq!(a.inv(), a, "inv is the identity on canonical values");
        let quotient = a.div(b).to_raw();
        let expected = if b.to_raw() == 0 { 0 } else { a.to_raw() };
        assert_eq!(quotient, expected, "x / 0 == 0 and x / 1 == x");
        assert_eq!(a / b, a.div(b));
    }
    for a in [Elem::<Gf1>::from_raw(0), Elem::<Gf1>::from_raw(1)] {
        assert_eq!(a.pow(0), Elem::<Gf1>::ONE, "pow(_, 0) is one");
        for exponent in [1u128, 2, 3, 63, u128::from(u64::MAX)] {
            assert_eq!(a.pow(exponent), a, "x^n = x for n > 0");
        }
    }
    // Division stays total in const context.
    const _: () = assert!(Elem::<Gf1>::from_raw(1).div(Elem::<Gf1>::ZERO).to_raw() == 0);
    const _: () = assert!(Elem::<Gf1>::from_raw(1).pow(0).to_raw() == 1);
}

#[test]
fn gf1_constants_and_predicates() {
    assert_eq!(Gf1::ORDER, 2);
    assert_eq!(<Gf1 as Field>::NAME, "GF(2)");
    // The multiplicative group is trivial: the generator is one and has
    // order 1.
    assert_eq!(Elem::<Gf1>::GENERATOR, Elem::<Gf1>::ONE);
    for exponent in [0u128, 1, 2, 100] {
        assert_eq!(Elem::<Gf1>::GENERATOR.pow(exponent), Elem::<Gf1>::ONE);
    }
    assert!(Elem::<Gf1>::from_raw(0).is_zero());
    assert!(Elem::<Gf1>::from_raw(1).is_one());
    assert_eq!(Elem::<Gf1>::ZERO.to_raw(), 0);
    assert_eq!(Elem::<Gf1>::ONE.to_raw(), 1);
    assert_eq!(Elem::<Gf1>::default(), Elem::<Gf1>::ZERO);
}

#[test]
fn gf1_elem_trait_bodies_match_inherent() {
    // Through a generic: only the trait's methods are visible, so the
    // delegating bodies themselves execute.
    fn through_trait<E: fgf::field::FieldElem>(a: E, b: E) {
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

    for (a, b) in all_gf1_pairs() {
        through_trait(a, b);
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::add(a, b),
            Elem::<Gf1>::add(a, b)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::sub(a, b),
            Elem::<Gf1>::sub(a, b)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::neg(a),
            Elem::<Gf1>::neg(a)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::mul(a, b),
            Elem::<Gf1>::mul(a, b)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::square(a),
            Elem::<Gf1>::square(a)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::inv(a),
            Elem::<Gf1>::inv(a)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::div(a, b),
            Elem::<Gf1>::div(a, b)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::pow(a, 7),
            Elem::<Gf1>::pow(a, 7)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::is_zero(a),
            Elem::<Gf1>::is_zero(a)
        );
        assert_eq!(
            <Elem<Gf1> as fgf::field::FieldElem>::is_one(a),
            Elem::<Gf1>::is_one(a)
        );
    }
}

#[test]
fn gf1_raw_lanes_are_total_and_outputs_canonical() {
    // Any raw byte is a legal input; only bit 0 is meaningful, and every
    // arithmetic output is canonical — the crate's totality convention.
    for raw in [0u8, 1, 2, 3, 0x7f, 0x80, 0xfe, 0xff] {
        let a = Elem::<Gf1>::from_raw(raw);
        assert_eq!(a.add(Elem::<Gf1>::from_raw(1)).to_raw(), (raw & 1) ^ 1);
        assert_eq!(a.mul(Elem::<Gf1>::from_raw(1)).to_raw(), raw & 1);
        assert_eq!(a.square().to_raw(), raw & 1);
        assert_eq!(a.inv().to_raw(), raw & 1);
        assert_eq!(a.div(Elem::<Gf1>::from_raw(0)).to_raw(), 0);
        assert_eq!(a.is_zero(), raw & 1 == 0);
        assert_eq!(a.is_one(), raw & 1 == 1);
    }
    assert_eq!(Elem::<Gf1>::from_raw(0xfe).to_raw(), 0, "from_raw masks");
    assert_eq!(
        Elem::<Gf1>::from_bytes([0xfe]).to_bytes(),
        [0],
        "byte round trip masks"
    );
    assert_eq!(Elem::<Gf1>::from_raw(1).canonical(), Elem::<Gf1>::ONE);
}

#[test]
fn gf1_operators_and_folds() {
    let mut a = Elem::<Gf1>::from_raw(1);
    a += Elem::<Gf1>::from_raw(1);
    assert_eq!(a, Elem::<Gf1>::ZERO);
    a -= Elem::<Gf1>::from_raw(1);
    assert_eq!(a, Elem::<Gf1>::ONE);
    a *= Elem::<Gf1>::from_raw(0);
    assert_eq!(a, Elem::<Gf1>::ZERO);
    a += Elem::<Gf1>::from_raw(1);
    a /= Elem::<Gf1>::from_raw(1);
    assert_eq!(a, Elem::<Gf1>::ONE);
    a /= Elem::<Gf1>::ZERO;
    assert_eq!(a, Elem::<Gf1>::ZERO);

    let one = Elem::<Gf1>::from_raw(1);
    let xor_sum: Elem<Gf1> = [one, one, one].into_iter().sum();
    assert_eq!(xor_sum, Elem::<Gf1>::ONE, "sum of three ones");
    let and_product: Elem<Gf1> = [one, one].into_iter().product();
    assert_eq!(and_product, Elem::<Gf1>::ONE);
    let borrowed_sum: Elem<Gf1> = [&one, &one].into_iter().sum();
    assert_eq!(borrowed_sum, Elem::<Gf1>::ZERO);
    let borrowed_product: Elem<Gf1> = [&one, &one].into_iter().product();
    assert_eq!(borrowed_product, Elem::<Gf1>::ONE);
}

// ---------------------------------------------------------------------------
// Trait default bodies
// ---------------------------------------------------------------------------

mod toy {
    //! Minimal GF(2^3) over `x^3 + x + 1` implementing only the required
    //! [`Field`](fgf::field::Field) raw methods. The blanket
    //! [`FieldElem`](fgf::field::FieldElem) implementation supplies `neg`,
    //! `square`, `pow`, `div`, `is_zero`, and `is_one` over them, so this is
    //! the only field where those blanket bodies execute against custom
    //! arithmetic.
    use fgf::Elem;
    use fgf::field::FieldElem as _;
    use fgf::field::{Field, FieldBuffer, HasGenerator, PrimeCharacteristic};

    #[derive(Debug, Clone, Copy)]
    pub struct Gf8Toy;

    /// Shift-and-XOR multiply under `x^3 + x + 1`, masked to three bits.
    const fn raw_mul(a: u8, b: u8) -> u8 {
        let mut acc = 0u8;
        let mut x = a & 7;
        let mut y = b & 7;
        let mut i = 0;
        while i < 3 {
            if y & 1 == 1 {
                acc ^= x;
            }
            let overflow = x & 0x04 != 0;
            x <<= 1;
            if overflow {
                x ^= 0x0B; // x^3 = x + 1
            }
            y >>= 1;
            i += 1;
        }
        acc & 7
    }

    /// Brute-force inverse over the seven nonzero lanes; zero maps to zero.
    const fn raw_inv(a: u8) -> u8 {
        let a = a & 7;
        if a == 0 {
            return 0;
        }
        let mut candidate = 1u8;
        while candidate < 8 {
            if raw_mul(a, candidate) == 1 {
                return candidate;
            }
            candidate += 1;
        }
        0
    }

    impl Field for Gf8Toy {
        type Raw = u8;
        type Characteristic = PrimeCharacteristic<2>;

        const NAME: &'static str = "GF(2^3) toy";
        const DEGREE: u32 = 3;
        const ORDER: u128 = 8;
        const ZERO_RAW: u8 = 0;
        const ONE_RAW: u8 = 1;
        const VALID: () = ();

        fn canonical_raw(raw: u8) -> u8 {
            raw & 7
        }
        fn add_raw(a: u8, b: u8) -> u8 {
            (a ^ b) & 7
        }
        fn sub_raw(a: u8, b: u8) -> u8 {
            (a ^ b) & 7
        }
        fn neg_raw(a: u8) -> u8 {
            a & 7
        }
        fn mul_raw(a: u8, b: u8) -> u8 {
            raw_mul(a, b)
        }
        fn inv_raw(a: u8) -> u8 {
            raw_inv(a)
        }
    }

    impl FieldBuffer for Gf8Toy {
        const BYTES: usize = 1;
        const STORAGE_BITS: u32 = 8;

        fn decode(bytes: &[u8]) -> Elem<Self> {
            let bytes: [u8; 1] = bytes.try_into().expect("toy element width");
            Elem::<Self>::from_raw(bytes[0])
        }
        fn encode(bytes: &mut [u8], value: Elem<Self>) {
            assert_eq!(bytes.len(), 1, "toy element width");
            bytes[0] = value.to_raw();
        }
    }

    impl HasGenerator for Gf8Toy {
        const GENERATOR_RAW: u8 = 2;
    }
}

#[test]
fn elem_trait_defaults_are_correct_for_minimal_implementors() {
    use fgf::Elem;
    let samples: Vec<Elem<toy::Gf8Toy>> = (0..8u8).map(Elem::<toy::Gf8Toy>::from_raw).collect();
    exercise_surface::<toy::Gf8Toy>(&samples);

    // The blanket bodies specifically: neg is the identity in
    // characteristic two, square is mul, pow is square-and-multiply.
    for a in samples {
        assert_eq!(a.neg(), a, "blanket neg in characteristic two");
        assert_eq!(a.square(), a.mul(a), "blanket square");
        assert_eq!(a.pow(5), a.mul(a).mul(a).mul(a).mul(a), "blanket pow");
    }
    // GF(8)* has order 7: the generator must have full order through the
    // blanket pow.
    let g = Elem::<toy::Gf8Toy>::from_raw(2);
    assert_eq!(g.pow(7), Elem::<toy::Gf8Toy>::ONE);
    assert_ne!(g.pow(1), Elem::<toy::Gf8Toy>::ONE);
}

#[test]
fn generic_from_raw_agrees_with_inherent_constructors() {
    // Concrete types only: on a concrete `Elem<F>` the inherent `const`
    // constructor wins even with `FieldElem` in scope, while the
    // fully-qualified trait spelling takes the blanket implementation.
    // A generic `Elem::<F>::from_raw` would resolve to the trait on both
    // sides and prove nothing.
    macro_rules! check_agreement {
        ($($field:ty: [$($raw:expr),+]),+) => {$({
            $(assert_eq!(
                <Elem<$field> as fgf::field::FieldElem>::from_raw($raw),
                Elem::<$field>::from_raw($raw),
                "generic and inherent from_raw disagree",
            );)+
        })+};
    }

    check_agreement!(
        Binary<8, Polynomial<AES>>: [0, 1, 0x53, 0xFF],
        Binary<8, Polynomial<RS>>: [0, 1, 0x53, 0xFF],
        Gf16: [0, 1, 0x0108, 0xFFFF],
        Gf32: [0, 1, 0x0001_0002, u32::MAX],
        Gf64: [0, 1, 0x0000_0001_0000_0004, u64::MAX],
        FanPaar8Field: [0, 1, 0x2D, 0xFF],
        FanPaar16Field: [0, 1, 0xE2DE, 0xFFFF],
        FanPaar32Field: [0, 1, 0x03E2_1CEA, u32::MAX],
        FanPaar64Field: [0, 1, 0x070F_870D_CD9C_1D88, u64::MAX],
        Mersenne31: [0, 1, 0x7FFF_FFFE, 0x7FFF_FFFF, u32::MAX],
        Goldilocks: [0, 1, 0xFFFF_FFFF_FFFF_FFFE, u64::MAX],
        Gf1: [0, 1, 2, 0xFF]
    );
    // QuadMersenne31 takes two limbs, so it does not fit the macro above.
    for &(re, im) in &[(0, 0), (1, 0), (0x7FFF_FFFF, 0), (u32::MAX, u32::MAX)] {
        assert_eq!(
            <Elem<QuadMersenne31> as fgf::field::FieldElem>::from_raw((re, im)),
            Elem::<QuadMersenne31>::from_raw(re, im),
            "generic and inherent from_raw disagree",
        );
    }
}

#[test]
fn qm31_wide_pow_matches_repeated_squaring() {
    let a = Elem::<QuadMersenne31>::from_raw(0x1234_5678, 0x9abc_def0);
    // Across the u64 range the const inherent pow agrees with manual
    // square-and-multiply.
    for exponent in [0u128, 1, 2, 3, 7, 255, 1 << 32, u64::MAX as u128] {
        let mut expected = Elem::<QuadMersenne31>::ONE;
        let mut base = a;
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 != 0 {
                expected = expected.mul(base);
            }
            base = base.square();
            remaining >>= 1;
        }
        assert_eq!(a.pow(exponent), expected, "pow({exponent})");
    }
    // Past u64, check against manual square-and-multiply over mul.
    let mut expected = a;
    for _ in 0..70 {
        expected = expected.square();
    }
    assert_eq!(a.pow(1u128 << 70), expected, "pow(2^70)");
}

// ---------------------------------------------------------------------------
// Value equality, hashing, and ordering
// ---------------------------------------------------------------------------
//
// `Eq`/`Hash`/`Ord` are public contract: they follow the field value.
// `from_raw` accepts non-canonical raw inputs and canonicalizes them into
// storage, so aliased inputs name one value; these tests pin that aliased
// inputs compare equal, hash equally, sort as one element, and collapse in
// hash containers. The binary full-width fields have one representation per
// value and are covered by the derived-implementation coherence everywhere
// above.

/// One aliased pair naming one field value: equal, same hash,
/// `Ordering::Equal`, and indistinguishable inside a `HashSet`.
fn assert_same_value<E: fgf::field::FieldElem + Ord>(noncanonical: E, canonical: E) {
    use std::cmp::Ordering;
    use std::collections::HashSet;

    assert_eq!(
        noncanonical, canonical,
        "equivalent representatives are one value"
    );
    assert!(
        hashes_equal(&noncanonical, &canonical),
        "equal field values must hash equally"
    );
    assert_eq!(
        noncanonical.cmp(&canonical),
        Ordering::Equal,
        "ordering must agree with equality"
    );
    assert!(!(noncanonical < canonical) && !(canonical < noncanonical));
    let set: HashSet<E> = HashSet::from([noncanonical, canonical]);
    assert_eq!(
        set.len(),
        1,
        "equivalent representatives collapse in a HashSet"
    );
}

#[test]
fn m31_equivalent_representatives_are_one_value() {
    // p ≡ 0, p + 1 ≡ 1, 2^31 ≡ 1, 2^32 − 1 ≡ 1 (mod p).
    let p = mersenne31::MODULUS;
    assert_same_value(Elem::<Mersenne31>::from_raw(p), Elem::<Mersenne31>::ZERO);
    assert_same_value(Elem::<Mersenne31>::from_raw(p + 1), Elem::<Mersenne31>::ONE);
    assert_same_value(
        Elem::<Mersenne31>::from_raw(0x8000_0000),
        Elem::<Mersenne31>::ONE,
    );
    assert_same_value(
        Elem::<Mersenne31>::from_raw(0xFFFF_FFFF),
        Elem::<Mersenne31>::ONE,
    );
    // Ordering is canonical-value order: p (zero) sorts below one, and a
    // raw-bit order would place it far above.
    assert!(Elem::<Mersenne31>::from_raw(p) < Elem::<Mersenne31>::ONE);
    let mut mixed = [
        Elem::<Mersenne31>::from_raw(p + 2), // ≡ 2
        Elem::<Mersenne31>::ZERO,
        Elem::<Mersenne31>::ONE,
        Elem::<Mersenne31>::from_raw(3),
    ];
    mixed.sort();
    assert_eq!(
        mixed,
        [
            Elem::<Mersenne31>::ZERO,
            Elem::<Mersenne31>::ONE,
            Elem::<Mersenne31>::from_raw(2),
            Elem::<Mersenne31>::from_raw(3),
        ]
    );
}

#[test]
fn goldilocks_equivalent_representatives_are_one_value() {
    let p = goldilocks::MODULUS;
    assert_same_value(Elem::<Goldilocks>::from_raw(p), Elem::<Goldilocks>::ZERO);
    assert_same_value(Elem::<Goldilocks>::from_raw(p + 1), Elem::<Goldilocks>::ONE);
    // u64::MAX = p + 2^32 − 2, pinned by the `canonical` known answer.
    assert_same_value(
        Elem::<Goldilocks>::from_raw(u64::MAX),
        Elem::<Goldilocks>::from_raw(0xFFFF_FFFE),
    );
    assert!(Elem::<Goldilocks>::from_raw(p) < Elem::<Goldilocks>::ONE);
    let mut mixed = [
        Elem::<Goldilocks>::from_raw(p + 2), // ≡ 2
        Elem::<Goldilocks>::ZERO,
        Elem::<Goldilocks>::ONE,
        Elem::<Goldilocks>::from_raw(3),
    ];
    mixed.sort();
    assert_eq!(
        mixed,
        [
            Elem::<Goldilocks>::ZERO,
            Elem::<Goldilocks>::ONE,
            Elem::<Goldilocks>::from_raw(2),
            Elem::<Goldilocks>::from_raw(3),
        ]
    );
}

#[test]
fn qm31_equivalent_representatives_are_one_value() {
    let p = quad_mersenne31::MODULUS;
    // Each limb canonicalizes independently: p ≡ 0, 2^31 ≡ 1.
    assert_same_value(
        Elem::<QuadMersenne31>::from_raw(p, 0),
        Elem::<QuadMersenne31>::ZERO,
    );
    assert_same_value(
        Elem::<QuadMersenne31>::from_raw(0x8000_0000, 0x8000_0000),
        Elem::<QuadMersenne31>::from_raw(1, 1),
    );
    // Mixed limbs: one canonical, the other not.
    assert_same_value(
        Elem::<QuadMersenne31>::from_raw(1, p + 5),
        Elem::<QuadMersenne31>::from_raw(1, 5),
    );
    assert_same_value(
        Elem::<QuadMersenne31>::from_raw(0xFFFF_FFFF, 2),
        Elem::<QuadMersenne31>::from_raw(1, 2),
    );
    // Lexicographic order over canonical limbs: raw-bit order would place
    // from_raw(p, 1) above ONE.
    let i = Elem::<QuadMersenne31>::I;
    assert!(Elem::<QuadMersenne31>::ZERO < i);
    assert!(i < Elem::<QuadMersenne31>::ONE);
    assert!(Elem::<QuadMersenne31>::ONE < Elem::<QuadMersenne31>::ONE.add(i));
    assert_eq!(Elem::<QuadMersenne31>::from_raw(p, 1), i);
    assert!(Elem::<QuadMersenne31>::from_raw(p, 1) < Elem::<QuadMersenne31>::ONE);
}

#[test]
fn gf1_masked_inputs_are_one_value() {
    // `from_raw` masks to the low bit, and equality/hash/order follow that
    // bit, so masked high bits never split one field value.
    assert_same_value(Elem::<Gf1>::from_raw(0x02), Elem::<Gf1>::ZERO);
    assert_same_value(Elem::<Gf1>::from_raw(0xFD), Elem::<Gf1>::ONE);
    assert!(Elem::<Gf1>::ZERO < Elem::<Gf1>::ONE);
    let mut bits = [Elem::<Gf1>::from_raw(0x80), Elem::<Gf1>::from_raw(3)];
    bits.sort();
    assert_eq!(bits, [Elem::<Gf1>::ZERO, Elem::<Gf1>::ONE]);
}
