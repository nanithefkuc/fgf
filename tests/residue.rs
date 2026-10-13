//! Residue values and operators against independent references.
//!
//! The production arithmetic bodies are plain widened modular arithmetic;
//! the references here deliberately are not. Multiplication is
//! double-and-add built only from conditional addition, inversion is
//! extended Euclid, and subtraction composes the two. Every prime with
//! order small enough to enumerate fully is checked exhaustively; the
//! full-width primes check carry, borrow, alias, and inverse-zero
//! boundaries.

use core::error::Error;

use fgf::field::finite_field::Residue;
use fgf::field::prime::{reduce, reduce_wide};
use fgf::{
    Components, Elem, Extension, ExtensionField, ExtensionSpec, Field, FieldBuffer, FieldElem,
    FiniteField, FiniteFieldRepr, PrimeCharacteristic, PrimeIdentity, RangeError, Relation,
};

/// `2^64 − 59`, the largest `u64` prime.
const WIDE: u64 = 0xFFFF_FFFF_FFFF_FFC5;
/// `2^64 − 83`.
const WIDE2: u64 = 0xFFFF_FFFF_FFFF_FFAD;
/// `2^63 − 25`.
const WIDE3: u64 = 0x7FFF_FFFF_FFFF_FFE7;

type Res<const P: u64> = FiniteField<P, 1, Residue>;

fn el<const P: u64>(value: u64) -> Elem<Res<P>> {
    Elem::<Res<P>>::from_raw(value)
}

/// `a + b mod p`, one conditional subtraction over the widened sum.
#[allow(clippy::cast_possible_truncation)]
fn add_ref(a: u64, b: u64, p: u64) -> u64 {
    let sum = u128::from(a) + u128::from(b);
    let modulus = u128::from(p);
    if sum >= modulus {
        (sum - modulus) as u64
    } else {
        sum as u64
    }
}

/// `a * b mod p` by double-and-add, conditional addition only.
fn mul_ref(mut a: u64, mut b: u64, p: u64) -> u64 {
    a %= p;
    let mut result = 0;
    while b != 0 {
        if b & 1 == 1 {
            result = add_ref(result, a, p);
        }
        a = add_ref(a, a, p);
        b >>= 1;
    }
    result
}

/// `a^e mod p` from the multiplication reference.
fn pow_ref(mut a: u64, mut e: u64, p: u64) -> u64 {
    a %= p;
    let mut result = 1 % p;
    while e != 0 {
        if e & 1 == 1 {
            result = mul_ref(result, a, p);
        }
        a = mul_ref(a, a, p);
        e >>= 1;
    }
    result
}

/// `a - b mod p`, as addition of the reference negation.
fn sub_ref(a: u64, b: u64, p: u64) -> u64 {
    add_ref(a, p - b % p, p)
}

/// `-a mod p`.
fn neg_ref(a: u64, p: u64) -> u64 {
    sub_ref(0, a, p)
}

/// Modular inverse by extended Euclid over `i128`; `None` when `a` shares
/// a factor with `p`.
#[allow(clippy::cast_possible_truncation)]
fn inv_ref(a: u64, p: u64) -> Option<u64> {
    let (mut t, mut next_t) = (0i128, 1i128);
    let (mut r, mut next_r) = (i128::from(p), i128::from(a % p));
    while next_r != 0 {
        let quotient = r / next_r;
        (t, next_t) = (next_t, t - quotient * next_t);
        (r, next_r) = (next_r, r - quotient * next_r);
    }
    if r != 1 {
        return None;
    }
    Some(t.rem_euclid(i128::from(p)) as u64)
}

#[test]
fn exhaustive_small_primes_match_references() {
    exhaustive::<2>();
    exhaustive::<3>();
    exhaustive::<5>();
    exhaustive::<7>();
    exhaustive::<251>();
}

fn exhaustive<const P: u64>() {
    for a in 0..P {
        let value = el::<P>(a);

        // Storage is the ordinary residue, and every conversion agrees.
        assert_eq!(value.to_residue(), Some(a), "a={a}");
        assert_eq!(value.to_raw(), a);
        assert_eq!(value.canonical().to_raw(), a);
        let width = <Res<P> as FieldBuffer>::BYTES;
        let mut lane = [0u8; 8];
        <Res<P> as FieldBuffer>::encode(&mut lane[..width], value);
        assert_eq!(&lane[..width], &a.to_le_bytes()[..width]);
        assert_eq!(
            <Res<P> as FieldBuffer>::decode(&a.to_le_bytes()[..width]).to_raw(),
            a,
            "a={a}"
        );
        assert_eq!(value.to_string(), a.to_string());

        // Negation, inversion, and totality.
        assert_eq!(value.neg().to_raw(), neg_ref(a, P), "-{a}");
        assert_eq!(value.div(Elem::<Res<P>>::ZERO).to_raw(), 0);
        assert_eq!(value.pow(0).to_raw(), 1, "0^0 == 1 and a^0 == 1, a={a}");
        assert_eq!(
            value.pow(u128::from(P) - 1).to_raw(),
            if a == 0 { 0 } else { 1 },
            "Fermat at a={a}"
        );
        if a == 0 {
            // The total convention, including the explicit P = 2 case.
            assert_eq!(value.inv().to_raw(), 0, "inv(0) at P={P}");
        } else {
            let inverse = inv_ref(a, P).expect("every nonzero residue inverts");
            assert_eq!(value.inv().to_raw(), inverse, "inv({a}) at P={P}");
            assert_eq!(value.mul(value.inv()).to_raw(), 1);
            assert_eq!(value.div(value).to_raw(), 1);
        }
        for e in 1..8u32 {
            assert_eq!(
                value.pow(u128::from(e)).to_raw(),
                pow_ref(a, u64::from(e), P),
                "a={a} e={e} P={P}"
            );
        }

        // The trait surface agrees with the concrete one.
        assert_eq!(FieldElem::is_zero(value), a == 0);
        assert_eq!(FieldElem::is_one(value), a == 1);
        assert_eq!(<Elem<Res<P>> as FieldElem>::from_raw(a).to_raw(), a);
    }

    for a in 0..P {
        for b in 0..P {
            let (left, right) = (el::<P>(a), el::<P>(b));
            assert_eq!(left.add(right).to_raw(), add_ref(a, b, P), "{a}+{b}");
            assert_eq!(left.sub(right).to_raw(), sub_ref(a, b, P), "{a}-{b}");
            assert_eq!(left.mul(right).to_raw(), mul_ref(a, b, P), "{a}*{b}");
            assert_eq!((left + right).to_raw(), add_ref(a, b, P));
            assert_eq!((left - right).to_raw(), sub_ref(a, b, P));
            assert_eq!((left * right).to_raw(), mul_ref(a, b, P));
            assert_eq!((-left).to_raw(), neg_ref(a, P));
            let want = if b == 0 {
                0
            } else {
                mul_ref(a, inv_ref(b, P).expect("nonzero divisor"), P)
            };
            assert_eq!(left.div(right).to_raw(), want, "{a}/{b} at P={P}");
            assert_eq!((left / right).to_raw(), want);
        }
    }
}

#[test]
fn wide_boundaries_match_references() {
    boundaries::<WIDE>();
    boundaries::<WIDE2>();
    boundaries::<WIDE3>();
}

fn boundaries<const P: u64>() {
    // Boundary residues: zero, one, two, and the two residues next to P.
    for a in [0, 1, 2, P - 2, P - 1] {
        let value = el::<P>(a);
        assert_eq!(value.to_residue(), Some(a), "a={a} P={P}");
        assert_eq!(value.neg().to_raw(), neg_ref(a, P));
        assert_eq!(
            value.inv().to_raw(),
            if a == 0 { 0 } else { inv_ref(a, P).unwrap() },
            "inv({a}) at P={P}"
        );
        if a != 0 {
            assert_eq!(value.mul(value.inv()).to_raw(), 1);
        }
        assert_eq!(
            value.pow(u128::from(P) - 1).to_raw(),
            if a == 0 { 0 } else { 1 },
            "Fermat at a={a}"
        );
    }

    // Carries: the widened sum leaves one word.
    assert_eq!(el::<P>(P - 1).add(el::<P>(P - 1)).to_raw(), P - 2);
    assert_eq!(el::<P>(P - 1).add(el::<P>(1)).to_raw(), 0);
    assert_eq!((el::<P>(P - 1) * el::<P>(P - 1)).to_raw(), 1);
    // Borrows: subtraction across zero and reference-checked small cases.
    assert_eq!(el::<P>(0).sub(el::<P>(1)).to_raw(), P - 1);
    assert_eq!(el::<P>(1).sub(el::<P>(2)).to_raw(), P - 1);
    assert_eq!(el::<P>(2).sub(el::<P>(5)).to_raw(), sub_ref(2, 5, P));
    assert_eq!(el::<P>(1).neg().to_raw(), P - 1);
    // The total conventions at full width.
    assert_eq!(el::<P>(0).pow(0).to_raw(), 1);
    assert_eq!(el::<P>(P - 1).square().to_raw(), 1);
    assert_eq!(el::<P>(P - 1).div(Elem::<Res<P>>::ZERO).to_raw(), 0);

    // Products over widened intermediates, including aliased inputs.
    let words = [0, 1, 2, 3, P - 2, P - 1, 1 << 63, (1 << 63) + 1, u64::MAX];
    for a in words {
        for b in words {
            let want = mul_ref(a % P, b % P, P);
            assert_eq!(el::<P>(a).mul(el::<P>(b)).to_raw(), want, "{a}*{b}");
        }
    }

    // Aliases: the high bit, the full word, and interior values above P all
    // reduce into the same elements as their residues.
    let middle = P + (u64::MAX - P) / 2;
    for alias in [P, P + 1, P + 7, middle, u64::MAX, u64::MAX - 1, 1 << 63] {
        let value = el::<P>(alias);
        assert_eq!(value.to_residue(), Some(alias % P), "alias {alias}");
        assert_eq!(value, el::<P>(alias % P), "alias {alias}");
        assert_eq!(
            <Res<P> as FieldBuffer>::decode(&alias.to_le_bytes()).to_raw(),
            alias % P,
            "alias {alias}"
        );
        assert_eq!(value.canonical().to_raw(), alias % P);
        assert_eq!(value.add(value).to_raw(), add_ref(alias % P, alias % P, P));
    }
}

#[test]
fn residue_conversions_reject_out_of_range() {
    assert_eq!(Elem::<Res<7>>::from_residue(0).map(Elem::to_raw), Ok(0));
    assert_eq!(Elem::<Res<7>>::from_residue(6).map(Elem::to_raw), Ok(6));
    assert_eq!(Elem::<Res<7>>::from_residue(7), Err(RangeError));
    assert_eq!(Elem::<Res<7>>::from_residue(8), Err(RangeError));
    assert_eq!(Elem::<Res<7>>::from_residue(u64::MAX), Err(RangeError));
    assert_eq!(Elem::<Res<WIDE>>::from_residue(WIDE), Err(RangeError));
    assert_eq!(
        Elem::<Res<WIDE>>::from_residue(WIDE - 1).map(Elem::to_raw),
        Ok(WIDE - 1)
    );

    // `to_residue` always answers on a degree-one row, aliases included.
    assert_eq!(el::<7>(5).to_residue(), Some(5));
    assert_eq!(el::<WIDE>(u64::MAX).to_residue(), Some(u64::MAX % WIDE));

    const ACCEPTED: bool = Elem::<Res<7>>::from_residue(3).is_ok();
    const REJECTED: bool = Elem::<Res<7>>::from_residue(7).is_err();
    const _: () = assert!(ACCEPTED);
    const _: () = assert!(REJECTED);
}

#[test]
fn range_error_is_a_std_error() {
    fn display_and_source<E: Error>(error: &E) -> bool {
        let _ = error.source();
        !error.to_string().is_empty()
    }
    assert!(display_and_source(&RangeError));
}

#[test]
fn const_context_construction_and_arithmetic() {
    // Everything below is evaluated during compilation.
    const HALF: Elem<Res<7>> = Elem::<Res<7>>::from_raw(2).inv();
    const _: () = assert!(HALF.to_raw() == 4);
    const SUM: Elem<Res<7>> = Elem::<Res<7>>::from_raw(6).add(Elem::<Res<7>>::from_raw(6));
    const _: () = assert!(SUM.to_raw() == 5);
    const NEGATED: Elem<Res<7>> = Elem::<Res<7>>::ONE.neg();
    const _: () = assert!(NEGATED.to_raw() == 6);
    const WIDE_PRODUCT: Elem<Res<WIDE>> =
        Elem::<Res<WIDE>>::from_raw(u64::MAX).mul(Elem::<Res<WIDE>>::from_raw(u64::MAX));
    const _: () = assert!(WIDE_PRODUCT.to_raw() == mul_const::<WIDE>(u64::MAX, u64::MAX));
    const TWO_INVERSE_WIDE: Elem<Res<WIDE>> = Elem::<Res<WIDE>>::from_raw(2).inv();
    const _: () = assert!(TWO_INVERSE_WIDE.to_raw() == (WIDE + 1) >> 1);
    const ZERO_RESIDUE: Option<u64> = Elem::<Res<7>>::ZERO.to_residue();
    const _: () = assert!(matches!(ZERO_RESIDUE, Some(0)));
}

/// Double-and-add multiplication for the const assertions above.
#[allow(clippy::cast_possible_truncation)]
const fn mul_const<const P: u64>(a: u64, b: u64) -> u64 {
    let modulus = P as u128;
    let mut base = a as u128 % modulus;
    let mut exponent = b as u128;
    let mut result: u128 = 0;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = (result + base) % modulus;
        }
        base = base * 2 % modulus;
        exponent >>= 1;
    }
    result as u64
}

#[test]
fn reductions_match_remainder() {
    check_reductions::<2>();
    check_reductions::<3>();
    check_reductions::<251>();
    check_reductions::<0x7FFF_FFFF>();
    check_reductions::<0xFFFF_FFFF_0000_0001>();
    check_reductions::<WIDE>();
    check_reductions::<WIDE2>();
    check_reductions::<WIDE3>();
}

fn check_reductions<const P: u64>() {
    for value in [0, 1, 2, 7, P - 1, P, P + 1, 1 << 63, u64::MAX, u64::MAX - 1] {
        assert_eq!(reduce::<P>(value), value % P, "reduce {value} at {P}");
    }
    let words = [
        0,
        1,
        7,
        u128::from(P) * u128::from(P),
        1 << 64,
        (1 << 64) + 5,
        1 << 127,
        u128::MAX,
        u128::MAX - 1,
    ];
    for wide in words {
        assert_eq!(
            u128::from(reduce_wide::<P>(wide)),
            wide % u128::from(P),
            "reduce_wide {wide} at {P}"
        );
    }
}

#[test]
fn prime_facts_match_pinned_values() {
    fn facts<const P: u64>() -> (u32, u32, u8) {
        (
            <PrimeCharacteristic<P> as PrimeIdentity>::BITS,
            <PrimeCharacteristic<P> as PrimeIdentity>::TWO_ADICITY,
            <PrimeCharacteristic<P> as PrimeIdentity>::RESIDUE_MOD_8,
        )
    }
    assert_eq!(facts::<2>(), (2, 0, 2));
    assert_eq!(facts::<3>(), (2, 1, 3));
    assert_eq!(facts::<7>(), (3, 1, 7));
    assert_eq!(facts::<251>(), (8, 1, 3));
    assert_eq!(facts::<0x7FFF_FFFF>(), (31, 1, 7));
    assert_eq!(facts::<0xFFFF_FFFF_0000_0001>(), (64, 32, 1));
    assert_eq!(facts::<WIDE>(), (64, 2, 5));
}

#[test]
fn row_metadata_matches_modulus() {
    assert_eq!(<Res<7> as Field>::NAME, "GF(p)");
    assert_eq!(<Residue as FiniteFieldRepr<7, 1>>::NAME, "residue");
    assert_eq!(<Res<7> as Field>::DEGREE, 1);
    assert_eq!(<Res<7> as Field>::ORDER, 7);
    assert_eq!(<Res<WIDE> as Field>::ORDER, u128::from(WIDE));
    assert_eq!(
        <<Res<7> as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC,
        7
    );
    assert_eq!(
        <<Res<WIDE> as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC,
        WIDE
    );
    assert_eq!(<Res<7> as Field>::ZERO_RAW, 0);
    assert_eq!(<Res<7> as Field>::ONE_RAW, 1);
    let () = <Res<7> as Field>::VALID;
    let () = <Residue as FiniteFieldRepr<7, 1>>::VALID;
}

#[test]
fn marker_traits_and_debug() {
    fn require_marker_traits<T>()
    where
        T: Clone + Copy + Default + PartialEq + Eq + core::hash::Hash + PartialOrd + Ord,
    {
    }
    require_marker_traits::<Res<7>>();
    require_marker_traits::<Res<WIDE>>();

    // The row is zero-sized; one element stores its canonical u64.
    assert_eq!(size_of::<Res<7>>(), 0);
    assert_eq!(size_of::<Elem<Res<7>>>(), size_of::<u64>());

    let debug = format!("{:?}", Res::<7>::default());
    assert!(
        debug.contains('7') && debug.contains('1') && debug.contains("residue"),
        "{debug}"
    );
    assert_eq!(Res::<7>::default(), Res::<7>::default());
    assert_eq!(format!("{:?}", el::<7>(3)), "GF(p)(3)");
    assert_eq!(el::<7>(3), el::<7>(3));
    assert_ne!(el::<7>(3), el::<7>(4));
}

#[test]
fn generic_field_surface_matches_concrete() {
    fn exercise<F: Field>(x: F::Raw, y: F::Raw) -> (F::Raw, F::Raw) {
        let (left, right) = (Elem::<F>::from_raw(x), Elem::<F>::from_raw(y));
        (
            FieldElem::add(left, right).to_raw(),
            FieldElem::mul(left, FieldElem::inv(right)).to_raw(),
        )
    }
    assert_eq!(
        exercise::<Res<7>>(6, 6),
        (add_ref(6, 6, 7), mul_ref(6, inv_ref(6, 7).unwrap(), 7))
    );
    assert_eq!(
        exercise::<Res<WIDE>>(WIDE - 1, 3),
        (
            add_ref(WIDE - 1, 3, WIDE),
            mul_ref(WIDE - 1, inv_ref(3, WIDE).unwrap(), WIDE)
        )
    );
}

#[test]
fn buffer_round_trip_is_canonical() {
    fn round_trip<F: FieldBuffer>(value: Elem<F>) -> Elem<F> {
        let mut bytes = vec![0u8; F::BYTES];
        F::encode(&mut bytes, value);
        F::decode(&bytes)
    }
    assert_eq!(round_trip(el::<7>(6)), el::<7>(6));
    assert_eq!(
        round_trip(el::<WIDE>(u64::MAX)),
        el::<WIDE>(u64::MAX % WIDE)
    );
    // Decoding an all-zero lane yields the additive identity.
    assert!(round_trip(Elem::<Res<7>>::ZERO).is_zero());
    assert!(round_trip(Elem::<Res<WIDE>>::ZERO).is_zero());
}

/// Lane widths across the characteristic range, at each side of every
/// threshold, with characteristic two keeping its eight-byte lane.
#[test]
fn lane_widths_follow_the_characteristic_rule() {
    fn widths<const P: u64>() -> (usize, u32) {
        (
            <Res<P> as FieldBuffer>::BYTES,
            <Res<P> as FieldBuffer>::STORAGE_BITS,
        )
    }
    assert_eq!(widths::<2>(), (8, 64));
    assert_eq!(widths::<3>(), (1, 8));
    assert_eq!(widths::<251>(), (1, 8));
    assert_eq!(widths::<257>(), (2, 16));
    assert_eq!(widths::<65521>(), (2, 16));
    assert_eq!(widths::<65537>(), (4, 32));
    assert_eq!(widths::<4294967291>(), (4, 32));
    assert_eq!(widths::<4294967311>(), (8, 64));
    assert_eq!(widths::<18446744073709551557>(), (8, 64));
}

/// Frozen lane fixtures: `encode` writes the canonical residue into the
/// exact little-endian bytes computed by hand, at every lane width.
#[test]
fn encode_emits_the_frozen_lanes() {
    fn lane<const P: u64>(value: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; <Res<P> as FieldBuffer>::BYTES];
        <Res<P> as FieldBuffer>::encode(&mut bytes, el::<P>(value));
        bytes
    }

    assert_eq!(lane::<2>(0), [0u8; 8]);
    assert_eq!(lane::<2>(1), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(lane::<2>(2 - 1), [1, 0, 0, 0, 0, 0, 0, 0]);

    assert_eq!(lane::<3>(0), [0]);
    assert_eq!(lane::<3>(1), [1]);
    assert_eq!(lane::<3>(2), [2]);

    assert_eq!(lane::<251>(0), [0]);
    assert_eq!(lane::<251>(1), [1]);
    assert_eq!(lane::<251>(250), [0xFA]);

    assert_eq!(lane::<257>(0), [0, 0]);
    assert_eq!(lane::<257>(1), [1, 0]);
    assert_eq!(lane::<257>(256), [0, 1]);

    assert_eq!(lane::<65521>(0), [0, 0]);
    assert_eq!(lane::<65521>(1), [1, 0]);
    assert_eq!(lane::<65521>(65520), [0xF0, 0xFF]);

    assert_eq!(lane::<65537>(0), [0, 0, 0, 0]);
    assert_eq!(lane::<65537>(1), [1, 0, 0, 0]);
    assert_eq!(lane::<65537>(65536), [0, 0, 1, 0]);

    assert_eq!(lane::<4294967291>(0), [0, 0, 0, 0]);
    assert_eq!(lane::<4294967291>(1), [1, 0, 0, 0]);
    assert_eq!(lane::<4294967291>(4294967290), [0xFA, 0xFF, 0xFF, 0xFF]);

    assert_eq!(lane::<4294967311>(0), [0u8; 8]);
    assert_eq!(lane::<4294967311>(1), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(lane::<4294967311>(4294967310), [0x0E, 0, 0, 0, 1, 0, 0, 0]);

    assert_eq!(lane::<18446744073709551557>(0), [0u8; 8]);
    assert_eq!(lane::<18446744073709551557>(1), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        lane::<18446744073709551557>(18446744073709551556),
        [0xC4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
}

/// Decoding the all-ones lane reduces `2^(8*BYTES) - 1` into the field,
/// against the remainder computed independently over `u128`.
#[test]
fn all_ones_lane_decodes_to_its_reduced_word() {
    fn ones<const P: u64>() -> u64 {
        let bytes = [0xFFu8; 8];
        let width = <Res<P> as FieldBuffer>::BYTES;
        <Res<P> as FieldBuffer>::decode(&bytes[..width]).to_raw()
    }
    assert_eq!(u128::from(ones::<2>()), ((1u128 << 64) - 1) % 2);
    assert_eq!(u128::from(ones::<3>()), ((1u128 << 8) - 1) % 3);
    assert_eq!(u128::from(ones::<251>()), ((1u128 << 8) - 1) % 251);
    assert_eq!(u128::from(ones::<257>()), ((1u128 << 16) - 1) % 257);
    assert_eq!(u128::from(ones::<65521>()), ((1u128 << 16) - 1) % 65521);
    assert_eq!(u128::from(ones::<65537>()), ((1u128 << 32) - 1) % 65537);
    assert_eq!(
        u128::from(ones::<4294967291>()),
        ((1u128 << 32) - 1) % 4294967291
    );
    assert_eq!(
        u128::from(ones::<4294967311>()),
        ((1u128 << 64) - 1) % 4294967311
    );
    assert_eq!(
        u128::from(ones::<18446744073709551557>()),
        ((1u128 << 64) - 1) % 18446744073709551557
    );
}

/// Every residue of the one- and two-byte lanes round-trips through the
/// byte form and encodes to its own little-endian bytes.
#[test]
fn every_residue_round_trips_through_its_lane() {
    fn round_trips<const P: u64>() {
        let width = <Res<P> as FieldBuffer>::BYTES;
        for a in 0..P {
            let mut lane = [0u8; 8];
            <Res<P> as FieldBuffer>::encode(&mut lane[..width], el::<P>(a));
            assert_eq!(&lane[..width], &a.to_le_bytes()[..width], "a={a}");
            assert_eq!(
                <Res<P> as FieldBuffer>::decode(&lane[..width]),
                el::<P>(a),
                "a={a}"
            );
        }
    }
    round_trips::<3>();
    round_trips::<251>();
    round_trips::<257>();
}

#[test]
#[should_panic(expected = "wrong byte width")]
fn decode_rejects_a_wrong_length_slice() {
    let _ = <Res<257> as FieldBuffer>::decode(&[0u8; 1]);
}

#[test]
#[should_panic(expected = "wrong byte width")]
fn encode_rejects_a_wrong_length_slice() {
    let mut lane = [0u8; 1];
    <Res<257> as FieldBuffer>::encode(&mut lane, Elem::<Res<257>>::ONE);
}

/// An extension over a one-byte residue base concatenates the component
/// lanes in component order.
#[test]
fn extension_components_over_a_one_byte_base_pack_tightly() {
    struct SevenSquareSpec;
    impl ExtensionSpec<2> for SevenSquareSpec {
        type Base = FiniteField<7, 1>;
        type Storage = Components;
        const NAME: &'static str = "GF(49)";
        // `t^2 - 3`: `4 = -3` constant first.
        const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
    }
    type SevenSquare = FiniteField<7, 2, Extension<SevenSquareSpec, 2>>;

    assert_eq!(<SevenSquare as FieldBuffer>::BYTES, 2);
    let a = el::<7>(3);
    let b = el::<7>(5);
    let element = <SevenSquare as ExtensionField<2>>::from_components([a, b]);
    let mut lane = [0u8; 2];
    <SevenSquare as FieldBuffer>::encode(&mut lane, element);
    assert_eq!(lane, [3, 5]);
    assert_eq!(<SevenSquare as FieldBuffer>::decode(&lane), element);
}
