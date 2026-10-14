//! Embedding acceptance tests: frozen ladder fixtures, mutual inverses,
//! commuting triangles, membership, trace and norm, against independent
//! shift-reduce oracles computed inside this file.

use fgf::{
    AES, BinaryField, Cantor, Elem, Embedding, EmbeddingError, Extension, ExtensionSpec, FieldElem,
    FiniteField, Isomorphism, Normal, Packed, Polynomial, RS, Relation,
};

type Gf1 = fgf::FiniteField<2, 1, fgf::Polynomial<3>>;
type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf32 = fgf::FiniteField<2, 32, fgf::Extension<fgf::Rijndael32, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;
type FanPaar8Field = fgf::FiniteField<2, 8, fgf::Extension<fgf::FanPaar8, 2>>;
type FanPaar16Field = fgf::FiniteField<2, 16, fgf::Extension<fgf::FanPaar16, 2>>;
type FanPaar32Field = fgf::FiniteField<2, 32, fgf::Extension<fgf::FanPaar32, 2>>;
type FanPaar64Field = fgf::FiniteField<2, 64, fgf::Extension<fgf::FanPaar64, 2>>;

/// Reference GF(2) at the bottom of the ladder.
type Ref2 = FiniteField<2, 2, Polynomial<0x7>>;
/// Reference GF(4) of the ladder.
type Ref4 = FiniteField<2, 4, Polynomial<0x13>>;
/// Reference GF(8) of the ladder: the AES field.
type Ref8 = FiniteField<2, 8, Polynomial<AES>>;

/// Independent shift-reduce product in GF(2)[x] reduced by `field`.
fn oracle_mul(field: u128, n: u32, x: u64, y: u64) -> u64 {
    let mut acc: u128 = 0;
    for k in 0..n {
        if (y >> k) & 1 == 1 {
            acc ^= u128::from(x) << k;
        }
    }
    let mut k = 2 * n;
    while k > n {
        k -= 1;
        if (acc >> k) & 1 == 1 {
            acc ^= field << (k - n);
        }
    }
    acc as u64
}

/// Independent Horner evaluation of `poly` at `r` in GF(2^m) under `field`.
fn oracle_eval(poly: u128, degree: u32, field: u128, m: u32, r: u64) -> u64 {
    let mut acc = 0;
    let mut bit = degree;
    loop {
        acc = oracle_mul(field, m, acc, r);
        if (poly >> bit) & 1 == 1 {
            acc ^= 1;
        }
        if bit == 0 {
            break;
        }
        bit -= 1;
    }
    acc
}

/// Independent smallest nonzero root of the degree-`n` polynomial `poly`
/// in GF(2^m) under `field`.
fn oracle_root(poly: u128, n: u32, field: u128, m: u32) -> u64 {
    for candidate in 1..(1u64 << m) {
        if oracle_eval(poly, n, field, m, candidate) == 0 {
            return candidate;
        }
    }
    panic!("the oracle found no root");
}

/// Independent image of `x` under the power basis of `r`.
fn oracle_powers(field: u128, m: u32, r: u64, x: u64) -> u64 {
    let mut acc = 0;
    let mut power = 1;
    for i in 0..m {
        if (x >> i) & 1 == 1 {
            acc ^= power;
        }
        power = oracle_mul(field, m, power, r);
    }
    acc
}

/// Deterministic sample words for a degree: boundaries plus noise, masked
/// to the degree.
fn samples(degree: u32, extra: u64) -> Vec<u64> {
    let full = if degree >= 64 {
        u64::MAX
    } else {
        (1u64 << degree) - 1
    };
    let mut values = vec![0, 1, 2, 7]
        .into_iter()
        .map(|word| word & full)
        .collect::<Vec<_>>();
    values.push(full);
    values.push(full ^ 1);
    values.push(1u64 << (degree - 1));
    let mut state = 0x0123_4567_89ab_cdefu64 ^ extra;
    for _ in 0..16 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        values.push(state & full);
    }
    values
}

/// A custom degree-32 extension over [`Gf16`], distinct from every shipped
/// description.
#[derive(Clone, Copy)]
struct Custom32;

impl ExtensionSpec<2> for Custom32 {
    type Base = Gf16;
    type Storage = Packed<u32>;
    const NAME: &'static str = "custom degree-32 extension";
    const RELATION: Relation = Relation::from_binary_words::<Gf16>(&[0x2001, 0x0001]);
}

/// A custom degree-16 extension structurally identical to [`Rijndael16`].
#[derive(Clone, Copy)]
struct SameAsRijndael16;

impl ExtensionSpec<2> for SameAsRijndael16 {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "rijndael-shaped custom extension";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<AES>>>(&[0x20, 0x01]);
}

#[test]
fn frozen_ladder_fixtures_recompute_by_brute_force() {
    // The host fixtures: independent oracle roots.
    assert_eq!(oracle_root(0x7, 2, 0x13, 4), 0x6);
    assert_eq!(oracle_root(0x13, 4, 0x11B, 8), 0x5C);

    // The consecutive 1->2 inclusion is the identity on GF(2).
    let one_two = Embedding::<Gf1, Ref2>::new().unwrap();
    for word in 0..2u64 {
        let x = Elem::<Gf1>::from_index(word as u128).unwrap();
        assert_eq!(one_two.embed(x).to_index() as u64, word);
    }

    // The consecutive 2->4 inclusion maps x to the frozen 0x6 and every
    // element to its brute-force image.
    let two_four = Embedding::<Ref2, Ref4>::new().unwrap();
    assert_eq!(
        two_four
            .embed(Elem::<Ref2>::from_index(2).unwrap())
            .to_index() as u64,
        0x6
    );
    for word in 0..4u64 {
        let x = Elem::<Ref2>::from_index(word as u128).unwrap();
        assert_eq!(
            two_four.embed(x).to_index() as u64,
            oracle_powers(0x13, 4, 0x6, word),
            "2->4 image of {word}"
        );
    }

    // The consecutive 4->8 inclusion maps x to the frozen 0x5C.
    let four_eight = Embedding::<Ref4, Ref8>::new().unwrap();
    assert_eq!(
        four_eight
            .embed(Elem::<Ref4>::from_index(2).unwrap())
            .to_index() as u64,
        0x5C
    );
    for word in 0..16u64 {
        let x = Elem::<Ref4>::from_index(word as u128).unwrap();
        assert_eq!(
            four_eight.embed(x).to_index() as u64,
            oracle_powers(0x11B, 8, 0x5C, word),
            "4->8 image of {word}"
        );
    }

    // The composite 2->8 inclusion maps x to the frozen 0xBC: compositions,
    // never separately minimized roots.
    let two_eight = Embedding::<Ref2, Ref8>::new().unwrap();
    assert_eq!(
        two_eight
            .embed(Elem::<Ref2>::from_index(2).unwrap())
            .to_index() as u64,
        0xBC
    );
    let composed = oracle_powers(0x11B, 8, 0x5C, oracle_powers(0x13, 4, 0x6, 2));
    assert_eq!(composed, 0xBC);
    for word in 0..4u64 {
        let x = Elem::<Ref2>::from_index(word as u128).unwrap();
        let expected = oracle_powers(0x11B, 8, 0x5C, oracle_powers(0x13, 4, 0x6, word));
        assert_eq!(two_eight.embed(x).to_index() as u64, expected);
    }
}

#[test]
fn consecutive_inclusions_above_the_byte_field() {
    // The tower-chain inclusions are the low component, element by element.
    let eight_sixteen = Embedding::<Ref8, Gf16>::new().unwrap();
    for byte in 0..=u8::MAX {
        let x = Elem::<Ref8>::from_index(u64::from(byte) as u128).unwrap();
        assert_eq!(
            eight_sixteen.embed(x).to_index() as u64,
            u64::from(byte),
            "8->16 image of byte {byte}"
        );
    }
    let sixteen_thirtytwo = Embedding::<Gf16, Gf32>::new().unwrap();
    for word in 0..=u16::MAX {
        let x = Elem::<Gf16>::from_raw(word);
        assert_eq!(
            sixteen_thirtytwo.embed(x).to_index() as u64,
            u64::from(word),
            "16->32 image of {word}"
        );
    }
    let thirtytwo_sixtyfour = Embedding::<Gf32, Gf64>::new().unwrap();
    for &word in &samples(32, 0x33) {
        let x = Elem::<Gf32>::from_raw(word as u32);
        assert_eq!(
            thirtytwo_sixtyfour.embed(x).to_index() as u64,
            u64::from(word as u32),
            "32->64 image of {word}"
        );
    }

    // Defining-relation images: the target indeterminate satisfies the
    // transported relation of each consecutive step.
    let t16 = Elem::<Gf16>::from_raw(0x0100);
    let b8 = eight_sixteen.embed(Elem::<Ref8>::from_raw(0x20));
    assert_eq!(t16.square().add(t16).add(b8), Elem::<Gf16>::ZERO);

    let t32 = Elem::<Gf32>::from_raw(0x0001_0000);
    let b16 = sixteen_thirtytwo.embed(Elem::<Gf16>::from_raw(0x2000));
    assert_eq!(t32.square().add(t32).add(b16), Elem::<Gf32>::ZERO);

    let t64 = Elem::<Gf64>::from_raw(0x1_0000_0000);
    let b32 = thirtytwo_sixtyfour.embed(Elem::<Gf32>::from_raw(0x2000_0000));
    assert_eq!(t64.square().add(t64).add(b32), Elem::<Gf64>::ZERO);

    // The inclusions are ring homomorphisms on sampled pairs.
    for &a in &samples(8, 0x44) {
        for &b in &samples(8, 0x45) {
            let (x, y) = (
                Elem::<Ref8>::from_raw(a as u8),
                Elem::<Ref8>::from_raw(b as u8),
            );
            assert_eq!(
                eight_sixteen.embed(x.mul(y)),
                eight_sixteen.embed(x).mul(eight_sixteen.embed(y))
            );
        }
    }
}

/// Check that an equal-degree pair of isomorphisms are mutual inverses and
/// ring isomorphisms on sampled coordinate words.
fn check_mutual_isomorphism<S, T>(
    forward: &Isomorphism<S, T>,
    backward: &Isomorphism<T, S>,
    degree: u32,
) where
    S: BinaryField,
    T: BinaryField,
{
    for &word in &samples(degree, 0x50) {
        let x = Elem::<S>::from_index(word as u128).unwrap();
        let image = forward.apply(x);
        assert_eq!(backward.apply(image), x, "round trip failed at {word}");
        assert_eq!(forward.unapply(image), x);
    }
    for &word in &samples(degree, 0x51) {
        let y = Elem::<T>::from_index(word as u128).unwrap();
        assert_eq!(forward.apply(backward.apply(y)), y);
        assert_eq!(backward.unapply(backward.apply(y)), y);
    }
    let left = &samples(degree, 0x52);
    let right = &samples(degree, 0x53);
    for &a in left {
        for &b in right {
            let (x, y) = (
                Elem::<S>::from_index(a as u128).unwrap(),
                Elem::<S>::from_index(b as u128).unwrap(),
            );
            assert_eq!(
                forward.apply(x.add(y)),
                forward.apply(x).add(forward.apply(y)),
                "addition at {a} + {b}"
            );
            assert_eq!(
                forward.apply(x.mul(y)),
                forward.apply(x).mul(forward.apply(y)),
                "multiplication at {a} * {b}"
            );
        }
    }
    assert_eq!(
        forward.apply(Elem::<S>::from_index(S::DESCRIPTION.one() as u128).unwrap()),
        Elem::<T>::from_index(T::DESCRIPTION.one() as u128).unwrap()
    );
}

#[test]
fn equal_degree_presentations_are_mutual_isomorphisms() {
    check_mutual_isomorphism(
        &Isomorphism::<FiniteField<2, 8, Polynomial<AES>>, FiniteField<2, 8, Polynomial<RS>>>::new(
        )
        .unwrap(),
        &Isomorphism::<FiniteField<2, 8, Polynomial<RS>>, FiniteField<2, 8, Polynomial<AES>>>::new(
        )
        .unwrap(),
        8,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FiniteField<2, 8, Polynomial<AES>>, FiniteField<2, 8, Normal<0x11B, 0x20>>>::new().unwrap(),
        &Isomorphism::<FiniteField<2, 8, Normal<0x11B, 0x20>>, FiniteField<2, 8, Polynomial<AES>>>::new().unwrap(),
        8,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FiniteField<2, 8, Polynomial<AES>>, FiniteField<2, 8, Cantor<0x11B, 0x20>>>::new().unwrap(),
        &Isomorphism::<FiniteField<2, 8, Cantor<0x11B, 0x20>>, FiniteField<2, 8, Polynomial<AES>>>::new().unwrap(),
        8,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FanPaar8Field, FiniteField<2, 8, Polynomial<AES>>>::new().unwrap(),
        &Isomorphism::<FiniteField<2, 8, Polynomial<AES>>, FanPaar8Field>::new().unwrap(),
        8,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FanPaar16Field, Gf16>::new().unwrap(),
        &Isomorphism::<Gf16, FanPaar16Field>::new().unwrap(),
        16,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FanPaar32Field, Gf32>::new().unwrap(),
        &Isomorphism::<Gf32, FanPaar32Field>::new().unwrap(),
        32,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FanPaar64Field, Gf64>::new().unwrap(),
        &Isomorphism::<Gf64, FanPaar64Field>::new().unwrap(),
        64,
    );
    check_mutual_isomorphism(
        &Isomorphism::<FiniteField<2, 32, Extension<Custom32, 2>>, Gf32>::new().unwrap(),
        &Isomorphism::<Gf32, FiniteField<2, 32, Extension<Custom32, 2>>>::new().unwrap(),
        32,
    );
}

#[test]
fn equal_descriptions_embed_identically() {
    let identity16 = Isomorphism::<Gf16, Gf16>::new().unwrap();
    for &word in &samples(16, 0x60) {
        let x = Elem::<Gf16>::from_raw(word as u16);
        assert_eq!(identity16.apply(x).to_index() as u64, word);
        assert_eq!(identity16.unapply(x), x);
    }
    // A custom spec structurally identical to a shipped one maps as the
    // identity: descriptions, not Rust types, select the map.
    type Custom16 = FiniteField<2, 16, Extension<SameAsRijndael16, 2>>;
    let structural = Isomorphism::<Custom16, Gf16>::new().unwrap();
    let back = Isomorphism::<Gf16, Custom16>::new().unwrap();
    for &word in &samples(16, 0x61) {
        let word = word as u16;
        assert_eq!(
            structural
                .apply(Elem::<Custom16>::from_raw(word))
                .to_index() as u64,
            u64::from(word)
        );
        assert_eq!(back.apply(Elem::<Gf16>::from_raw(word)).to_raw(), word);
    }
}

/// Check `embed_BC . embed_AB == embed_AC` on sampled coordinate words.
fn check_triangle<A, B, C>(
    ab: &Embedding<A, B>,
    bc: &Embedding<B, C>,
    ac: &Embedding<A, C>,
    degree: u32,
) where
    A: BinaryField,
    B: BinaryField,
    C: BinaryField,
{
    for &word in &samples(degree, 0x70) {
        let x = Elem::<A>::from_index(word as u128).unwrap();
        assert_eq!(
            bc.embed(ab.embed(x)).to_index(),
            ac.embed(x).to_index(),
            "triangle failed at {word}"
        );
    }
}

#[test]
fn inclusion_triangles_commute() {
    let one_two = Embedding::<Gf1, Ref2>::new().unwrap();
    let two_four = Embedding::<Ref2, Ref4>::new().unwrap();
    let four_eight = Embedding::<Ref4, Ref8>::new().unwrap();
    let eight_sixteen = Embedding::<Ref8, Gf16>::new().unwrap();
    let sixteen_thirtytwo = Embedding::<Gf16, Gf32>::new().unwrap();
    let thirtytwo_sixtyfour = Embedding::<Gf32, Gf64>::new().unwrap();
    let one_four = Embedding::<Gf1, Ref4>::new().unwrap();
    let two_eight = Embedding::<Ref2, Ref8>::new().unwrap();
    let one_eight = Embedding::<Gf1, Ref8>::new().unwrap();
    let eight_thirtytwo = Embedding::<Ref8, Gf32>::new().unwrap();
    let sixteen_sixtyfour = Embedding::<Gf16, Gf64>::new().unwrap();
    let one_sixtyfour = Embedding::<Gf1, Gf64>::new().unwrap();
    let two_sixteen = Embedding::<Ref2, Gf16>::new().unwrap();
    let four_sixteen = Embedding::<Ref4, Gf16>::new().unwrap();

    check_triangle(&one_two, &two_four, &one_four, 1);
    check_triangle(&two_four, &four_eight, &two_eight, 2);
    check_triangle(&four_eight, &eight_sixteen, &four_sixteen, 4);
    check_triangle(&eight_sixteen, &sixteen_thirtytwo, &eight_thirtytwo, 8);
    check_triangle(
        &sixteen_thirtytwo,
        &thirtytwo_sixtyfour,
        &sixteen_sixtyfour,
        16,
    );
    check_triangle(
        &one_two,
        &two_sixteen,
        &Embedding::<Gf1, Gf16>::new().unwrap(),
        1,
    );
    check_triangle(
        &two_four,
        &Embedding::<Ref4, Gf32>::new().unwrap(),
        &Embedding::<Ref2, Gf32>::new().unwrap(),
        2,
    );
    check_triangle(
        &one_eight,
        &eight_thirtytwo,
        &Embedding::<Gf1, Gf32>::new().unwrap(),
        1,
    );
    check_triangle(
        &two_eight,
        &Embedding::<Ref8, Gf64>::new().unwrap(),
        &Embedding::<Ref2, Gf64>::new().unwrap(),
        2,
    );
    check_triangle(
        &one_four,
        &four_sixteen,
        &Embedding::<Gf1, Gf16>::new().unwrap(),
        1,
    );
    for &word in &samples(1, 0x71) {
        let x = Elem::<Gf1>::from_index(word as u128).unwrap();
        assert_eq!(
            one_sixtyfour.embed(x).to_index(),
            thirtytwo_sixtyfour
                .embed(
                    sixteen_thirtytwo.embed(
                        eight_sixteen.embed(four_eight.embed(two_four.embed(one_two.embed(x))))
                    )
                )
                .to_index()
        );
    }

    // Mixed presentations: a flat source through a Fan-Paar middle into the
    // Rijndael top.
    let flat_fp = Embedding::<Ref4, FanPaar16Field>::new().unwrap();
    let fp_top = Embedding::<FanPaar16Field, Gf64>::new().unwrap();
    let flat_top = Embedding::<Ref4, Gf64>::new().unwrap();
    check_triangle(&flat_fp, &fp_top, &flat_top, 4);
    // A Fan-Paar byte field through the tower chain: the equal-degree
    // byte map is an isomorphism, composed into the tower inclusion.
    let fp8_aes = Isomorphism::<FanPaar8Field, Ref8>::new().unwrap();
    let fp8_top = Embedding::<FanPaar8Field, Gf64>::new().unwrap();
    let aes_top = Embedding::<Ref8, Gf64>::new().unwrap();
    for &word in &samples(8, 0x72) {
        let x = Elem::<FanPaar8Field>::from_index(word as u128).unwrap();
        assert_eq!(fp8_top.embed(x), aes_top.embed(fp8_aes.apply(x)));
    }
}

#[test]
fn membership_agrees_with_the_frobenius_test() {
    macro_rules! check {
        ($source:ty, $target:ty, $degree:expr, $salt:expr) => {{
            let embedding = Embedding::<$source, $target>::new().unwrap();
            let m: u32 = <$source as fgf::Field>::DEGREE;
            // contains/restrict agree with x^(2^m) == x.
            for &word in &samples($degree, $salt) {
                let y = Elem::<$target>::from_index(word as u128).unwrap();
                let frobenius_fixed = y.pow(1u128 << m) == y;
                assert_eq!(embedding.contains(y), frobenius_fixed, "{word}");
                assert_eq!(embedding.restrict(y).is_some(), frobenius_fixed, "{word}");
            }
            // restrict inverts embed.
            for &word in &samples(m, $salt + 1) {
                let x = Elem::<$source>::from_index(word as u128).unwrap();
                assert_eq!(embedding.restrict(embedding.embed(x)), Some(x));
                assert!(embedding.contains(embedding.embed(x)));
            }
        }};
    }
    check!(Ref2, Ref8, 8, 0x80);
    check!(Ref4, Ref8, 8, 0x81);
    check!(Ref8, Gf16, 16, 0x82);
    check!(Ref8, Gf64, 64, 0x83);
    check!(Gf16, Gf64, 64, 0x84);
    check!(Gf32, Gf64, 64, 0x85);
    check!(FanPaar8Field, FanPaar16Field, 16, 0x86);
    check!(FiniteField<2, 8, Polynomial<RS>>, Gf16, 16, 0x87);
    check!(Gf1, Gf64, 64, 0x88);

    // A concrete nonmember: the image of the 2->8 inclusion is sixteen
    // bytes, and the raw byte 2 is not one of them.
    let two_eight = Embedding::<Ref2, Ref8>::new().unwrap();
    let mut image = std::vec::Vec::new();
    for word in 0..4u64 {
        image.push(
            two_eight
                .embed(Elem::<Ref2>::from_index(word as u128).unwrap())
                .to_index() as u64,
        );
    }
    assert!(!image.contains(&2));
    let nonmember = Elem::<Ref8>::from_index(2).unwrap();
    assert!(!two_eight.contains(nonmember));
    assert_eq!(two_eight.restrict(nonmember), None);

    // A nonmember above the byte field: the high half spoils membership.
    let sixteen_thirtytwo = Embedding::<Gf16, Gf32>::new().unwrap();
    let lifted = Elem::<Gf32>::from_raw(0x0001_0002);
    assert!(!sixteen_thirtytwo.contains(lifted));
    assert_eq!(sixteen_thirtytwo.restrict(lifted), None);
}

#[test]
fn trace_and_norm_match_conjugates() {
    macro_rules! check {
        ($source:ty, $target:ty, $degree:expr, $salt:expr) => {{
            let embedding = Embedding::<$source, $target>::new().unwrap();
            let m: u32 = <$source as fgf::Field>::DEGREE;
            let n: u32 = <$target as fgf::Field>::DEGREE;
            for &word in &samples($degree, $salt) {
                let y = Elem::<$target>::from_index(word as u128).unwrap();
                let mut sum = y;
                let mut product = y;
                let mut power = y;
                for _ in 1..n / m {
                    power = power.pow(1u128 << m);
                    sum = sum.add(power);
                    product = product.mul(power);
                }
                let expected_sum = embedding.restrict(sum).unwrap();
                let expected_product = embedding.restrict(product).unwrap();
                assert_eq!(embedding.trace(y), expected_sum, "trace at {word}");
                assert_eq!(embedding.norm(y), expected_product, "norm at {word}");
            }
        }};
    }
    check!(Ref4, Gf16, 16, 0x90);
    check!(Ref2, Gf16, 16, 0x91);
    check!(Ref8, Gf32, 32, 0x92);
    check!(Gf16, Gf64, 64, 0x93);
    check!(FiniteField<2, 8, Polynomial<RS>>, Gf64, 64, 0x94);
}

#[test]
fn absolute_trace_and_norm_through_gf2() {
    let absolute16 = Embedding::<Gf1, Gf16>::new().unwrap();
    let absolute64 = Embedding::<Gf1, Gf64>::new().unwrap();
    for &word in &samples(16, 0xA0) {
        let y = Elem::<Gf16>::from_raw(word as u16);
        let trace = absolute16.trace(y);
        assert!(trace == Elem::<Gf1>::ZERO || trace == Elem::<Gf1>::ONE);
        assert_eq!(
            trace.to_raw() as u64,
            <Gf16 as BinaryField>::DESCRIPTION.trace(word)
        );
        let norm = absolute16.norm(y);
        let expected = if y.is_zero() {
            Elem::<Gf1>::ZERO
        } else {
            Elem::<Gf1>::ONE
        };
        assert_eq!(norm, expected);
    }
    for &word in &samples(64, 0xA1) {
        let y = Elem::<Gf64>::from_raw(word);
        let trace = absolute64.trace(y);
        assert!(trace == Elem::<Gf1>::ZERO || trace == Elem::<Gf1>::ONE);
        assert_eq!(
            trace.to_raw() as u64,
            <Gf64 as BinaryField>::DESCRIPTION.trace(word)
        );
        assert_eq!(
            absolute64.norm(y),
            if y.is_zero() {
                Elem::<Gf1>::ZERO
            } else {
                Elem::<Gf1>::ONE
            }
        );
    }
}

#[test]
fn equal_degree_constructors_reject_proper_inclusions() {
    assert_eq!(
        Embedding::<FanPaar16Field, Gf16>::new().unwrap_err(),
        EmbeddingError::EqualDegree { degree: 16 }
    );
    assert_eq!(
        Embedding::<FanPaar16Field, Gf16>::from_images(&[]).unwrap_err(),
        EmbeddingError::EqualDegree { degree: 16 }
    );
}

#[test]
fn incompatible_degrees_report_both_sides() {
    assert!(matches!(
        Embedding::<Gf16, FiniteField<2, 8, Polynomial<AES>>>::new(),
        Err(EmbeddingError::IncompatibleDegree {
            source: 16,
            target: 8
        })
    ));
    assert!(matches!(
        Embedding::<FiniteField<2, 8, Polynomial<AES>>, Ref4>::new(),
        Err(EmbeddingError::IncompatibleDegree {
            source: 8,
            target: 4
        })
    ));
    assert!(Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf32>::new().is_ok());
    assert!(Embedding::<Gf1, FanPaar64Field>::new().is_ok());
    assert_eq!(
        Embedding::<Gf64, Gf64>::new().unwrap_err(),
        EmbeddingError::EqualDegree { degree: 64 }
    );
}

#[test]
fn degree64_boundaries_run_without_overflow() {
    let embedding = Embedding::<Gf32, Gf64>::new().unwrap();
    let full = Elem::<Gf64>::from_raw(u64::MAX);
    let _ = embedding.frobenius(full);
    let _ = embedding.trace(full);
    let _ = embedding.norm(full);
    assert!(embedding.contains(embedding.embed(Elem::<Gf32>::from_raw(u32::MAX))));
    assert_eq!(
        embedding.restrict(embedding.embed(Elem::<Gf32>::from_raw(u32::MAX))),
        Some(Elem::<Gf32>::from_raw(u32::MAX))
    );

    let wide = Embedding::<Gf16, Gf64>::new().unwrap();
    let full = Elem::<Gf64>::from_raw(u64::MAX);
    let mut sum = full;
    let mut product = full;
    let mut power = full;
    for _ in 1..4 {
        power = power.pow(1u128 << 16);
        sum = sum.add(power);
        product = product.mul(power);
    }
    assert_eq!(wide.trace(full), wide.restrict(sum).unwrap());
    assert_eq!(wide.norm(full), wide.restrict(product).unwrap());
    let identity = Isomorphism::<Gf64, Gf64>::new().unwrap();
    let full_element = Elem::<Gf64>::from_raw(u64::MAX);
    assert_eq!(identity.apply(full_element), full_element);
    assert_eq!(identity.unapply(full_element), full_element);
}

/// The semantics `VGF2P8AFFINEQB` documents, modeled in scalar code:
/// `dst.bit[b] = parity(map.byte[7 - b] & x)`.
fn affine_apply(map: u64, x: u8) -> u8 {
    let mut out = 0u8;
    for b in 0..8u8 {
        let row = (map >> (8 * u64::from(7 - b))) as u8;
        let parity = (row & x).count_ones() & 1;
        out |= u8::try_from(parity).unwrap() << b;
    }
    out
}

#[test]
fn degree8_maps_preserve_the_selected_aes_conventions() {
    // The frozen Reed-Solomon conjugation constant of the vector kernels.
    const FROZEN_RS: u64 = 0xffaa_cc88_f0a0_c080;

    macro_rules! check_polynomial {
        ($poly:expr) => {{
            let poly: u128 = $poly;
            let backward =
                Isomorphism::<FiniteField<2, 8, Polynomial<{ $poly }>>, Ref8>::new().unwrap();
            let forward =
                Isomorphism::<Ref8, FiniteField<2, 8, Polynomial<{ $poly }>>>::new().unwrap();
            // The independent smallest-root oracle reproduces every image.
            for x in 0..=u8::MAX {
                let expected = if poly == AES {
                    u64::from(x)
                } else {
                    let root = oracle_root(poly, 8, 0x11B, 8);
                    oracle_powers(0x11B, 8, root, u64::from(x))
                };
                assert_eq!(
                    backward
                        .apply(Elem::<FiniteField<2, 8, Polynomial<{ $poly }>>>::from_raw(
                            x
                        ))
                        .to_index() as u64,
                    expected,
                    "selected map of {poly:#x} at {x:#04x}"
                );
            }
            // Mutual inverses over every byte.
            for x in 0..=u8::MAX {
                let element = Elem::<FiniteField<2, 8, Polynomial<{ $poly }>>>::from_raw(x);
                assert_eq!(forward.apply(backward.apply(element)), element);
            }
        }};
    }

    check_polynomial!(AES);
    check_polynomial!(RS);
    check_polynomial!(0x12D);
    check_polynomial!(0x187);

    // The Reed-Solomon map is the frozen involution: the historical qword
    // byte for byte, in both directions, since an involution equals its
    // own inverse.
    let to_aes = Isomorphism::<FiniteField<2, 8, Polynomial<RS>>, Ref8>::new().unwrap();
    let from_aes = Isomorphism::<Ref8, FiniteField<2, 8, Polynomial<RS>>>::new().unwrap();
    for x in 0..=u8::MAX {
        let image = affine_apply(FROZEN_RS, x);
        assert_eq!(
            to_aes
                .apply(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(x))
                .to_index() as u64,
            u64::from(image),
            "frozen involution at {x:#04x}"
        );
        assert_eq!(
            from_aes.apply(Elem::<Ref8>::from_raw(x)).to_raw(),
            image,
            "the frozen map is its own inverse"
        );
    }
}

// ---------------------------------------------------------------------------
// Construction errors and non-quadratic relation nodes
// ---------------------------------------------------------------------------

mod extension_rows {
    use fgf::{Extension, ExtensionSpec, FiniteField, PackedPadded, Relation};

    /// GF(8) as `x^3 + x + 1` over GF(2): a degree outside the ladder.
    pub struct Gf8Spec;
    impl ExtensionSpec<3> for Gf8Spec {
        type Base = FiniteField<2, 1, fgf::Polynomial<3>>;
        type Storage = PackedPadded<u8>;
        const NAME: &'static str = "GF(8) cubic";
        const RELATION: Relation =
            Relation::from_binary_words::<FiniteField<2, 1, fgf::Polynomial<3>>>(&[1, 1, 0]);
    }
    pub type Gf8Cubic = FiniteField<2, 3, Extension<Gf8Spec, 3>>;

    /// GF(64) as the sextic `x^6 + x + 1` over GF(2): a degree-six row
    /// whose presentation shares no tower level with the cubic source.
    pub struct Gf64SexticSpec;
    impl ExtensionSpec<6> for Gf64SexticSpec {
        type Base = FiniteField<2, 1, fgf::Polynomial<3>>;
        type Storage = PackedPadded<u8>;
        const NAME: &'static str = "GF(64) sextic";
        const RELATION: Relation = Relation::from_binary_words::<
            FiniteField<2, 1, fgf::Polynomial<3>>,
        >(&[1, 1, 0, 0, 0, 0]);
    }
    pub type Gf64Sextic = FiniteField<2, 6, Extension<Gf64SexticSpec, 6>>;

    /// GF(2^4) as the quartic `x^4 + x + 1` over GF(2): a non-quadratic
    /// relation node the ladder covers.
    pub struct Gf16QuarticSpec;
    impl ExtensionSpec<4> for Gf16QuarticSpec {
        type Base = FiniteField<2, 1, fgf::Polynomial<3>>;
        type Storage = PackedPadded<u8>;
        const NAME: &'static str = "GF(2^4) quartic";
        const RELATION: Relation =
            Relation::from_binary_words::<FiniteField<2, 1, fgf::Polynomial<3>>>(&[1, 1, 0, 0]);
    }
    pub type Gf16Quartic = FiniteField<2, 4, Extension<Gf16QuarticSpec, 4>>;
}

/// `Embedding::new` rejects non-dividing degrees of supported
/// presentations by variant, before any presentation coverage question.
#[test]
fn incompatible_degree_precedes_every_other_rejection() {
    let error = Embedding::<Gf16, Ref4>::new().unwrap_err();
    assert_eq!(
        error,
        EmbeddingError::IncompatibleDegree {
            source: 16,
            target: 4
        }
    );
    // An unsupported presentation of non-dividing degree still reports the
    // degree first.
    let error = Embedding::<extension_rows::Gf8Cubic, Ref4>::new().unwrap_err();
    assert_eq!(
        error,
        EmbeddingError::IncompatibleDegree {
            source: 3,
            target: 4
        }
    );
}

/// A degree-three binary extension into a degree-six row outside its own
/// tower: the degrees divide, no structural inclusion relates the rows,
/// and the ladder covers neither, so no canonical map is searched for.
#[test]
fn uncovered_presentations_report_no_canonical_map() {
    let error =
        Embedding::<extension_rows::Gf8Cubic, extension_rows::Gf64Sextic>::new().unwrap_err();
    assert_eq!(error, EmbeddingError::NoCanonicalMap);
    // The reverse direction embeds a larger field into a smaller one, so
    // it reports the degree before any presentation question.
    assert!(matches!(
        Embedding::<extension_rows::Gf64Sextic, extension_rows::Gf8Cubic>::new(),
        Err(EmbeddingError::IncompatibleDegree {
            source: 6,
            target: 3
        })
    ));
}

/// A covered non-quadratic relation node maps through the bounded
/// reference like any flat presentation: equal degrees biject through
/// the isomorphism surface, dividing degrees include, and restriction
/// inverts embedding.
#[test]
fn covered_quartic_relation_embeds_through_the_reference() {
    let equal = Isomorphism::<extension_rows::Gf16Quartic, Ref4>::new().unwrap();
    for word in 0..16u64 {
        let element = Elem::<extension_rows::Gf16Quartic>::from_index(word as u128).unwrap();
        assert_eq!(equal.unapply(equal.apply(element)), element);
    }
    let lifted = Embedding::<extension_rows::Gf16Quartic, Ref8>::new().unwrap();
    for word in 0..16u64 {
        let element = Elem::<extension_rows::Gf16Quartic>::from_index(word as u128).unwrap();
        let image = lifted.embed(element);
        assert!(lifted.contains(image));
        assert_eq!(lifted.restrict(image), Some(element));
        assert_eq!(lifted.frobenius(image), image);
    }
    // The embedded copy multiplies like the source, through both maps of
    // the composite: the quartic indeterminate maps to the smallest
    // reference root of `x^4 + x + 1`, and the ladder inclusion maps the
    // degree-four reference's indeterminate to the smallest root of its
    // own polynomial below the AES field.
    let t = Elem::<extension_rows::Gf16Quartic>::from_index(0b0010).unwrap();
    let root = oracle_root(0b10011, 4, 0x13, 4);
    assert_eq!(equal.apply(t).to_index() as u64, root);
    let squared = oracle_mul(0x13, 4, root, root);
    let step = oracle_root(0x13, 4, 0x11B, 8);
    let mut lifted_square = 0u64;
    let mut power = 1u64;
    for i in 0..4 {
        if (squared >> i) & 1 == 1 {
            lifted_square ^= power;
        }
        power = oracle_mul(0x11B, 8, power, step);
    }
    assert_eq!(lifted.embed(t.mul(t)).to_index() as u64, lifted_square);
}

/// Byte fields presented over a normal or Cantor basis embed into the
/// Rijndael tower through the ladder, and restriction inverts the
/// embedding element by element while rejecting elements outside the
/// subfield.
#[test]
fn basis_row_presentations_embed_and_restrict_through_the_ladder() {
    type Normal8 = FiniteField<2, 8, Normal<0x11B, 0x20>>;
    type Cantor8 = FiniteField<2, 8, Cantor<0x11B, 0x20>>;
    let normal = Embedding::<Normal8, Gf16>::new().unwrap();
    let cantor = Embedding::<Cantor8, Gf16>::new().unwrap();
    for word in samples(8, 0x5A) {
        let element = Elem::<Normal8>::from_index(u128::from(word)).unwrap();
        let image = normal.embed(element);
        assert!(normal.contains(image), "normal row: {word}");
        assert_eq!(normal.restrict(image), Some(element), "normal row: {word}");
        let element = Elem::<Cantor8>::from_index(u128::from(word)).unwrap();
        let image = cantor.embed(element);
        assert!(cantor.contains(image), "cantor row: {word}");
        assert_eq!(cantor.restrict(image), Some(element), "cantor row: {word}");
    }
    // The tower indeterminate has a zero low byte, so it lies outside every
    // byte subfield: restriction fails on the basis source too.
    let outside = Elem::<Gf16>::from_raw(0x0100);
    assert_eq!(normal.restrict(outside), None);
    assert_eq!(cantor.restrict(outside), None);
}

/// Restriction answers `None` for every target element outside the source
/// subfield, across flat, tower, and prime-subfield sources.
#[test]
fn restrict_rejects_elements_outside_the_subfield() {
    // GF(4) inside the AES field: `2` and `3` are not fourth roots of
    // themselves, so neither lies in the subfield.
    let two_four = Embedding::<Ref2, Ref8>::new().unwrap();
    for word in [2u128, 3] {
        let element = Elem::<Ref8>::from_index(word).unwrap();
        assert_eq!(two_four.restrict(element), None, "{word}");
    }
    // GF(2^16) inside GF(2^32): a nonzero high component is outside.
    let sixteen_thirtytwo = Embedding::<Gf16, Gf32>::new().unwrap();
    let element = Elem::<Gf32>::from_raw(0x0001_0000);
    assert_eq!(sixteen_thirtytwo.restrict(element), None);
    // GF(2) inside the AES field: only `0` and `1` restrict.
    let one_eight = Embedding::<Gf1, Ref8>::new().unwrap();
    let element = Elem::<Ref8>::from_index(2).unwrap();
    assert_eq!(one_eight.restrict(element), None);
}

/// `from_images` reports a degree mismatch before reading the slice.
#[test]
fn embedding_from_images_rejects_a_degree_mismatch() {
    assert!(matches!(
        Embedding::<Ref8, Ref4>::from_images(&[]),
        Err(EmbeddingError::IncompatibleDegree {
            source: 8,
            target: 4
        })
    ));
}
