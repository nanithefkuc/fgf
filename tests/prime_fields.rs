//! Prime-characteristic contracts: the deterministic primality test and the
//! QuadMersenne31 bulk operations over raw, non-canonical limb bytes.
//!
//! Primality is checked against trial division, against explicit factor
//! lists for composites, and against Lucas certificates (a full
//! factorization of `n − 1` plus a witness of order `n − 1`) for primes too
//! large to trial-divide. The extension oracle reduces each limb with `%`
//! over `u64` and multiplies by the schoolbook `(a+bi)(c+di)` expansion.

use core::hint::black_box;

use fgf::field::is_prime;
use fgf::kernel::vector_elementwise_min_bytes;
use fgf::{Elem, QuadMersenne31, has_vector_elementwise, ops};

/// Primality by trial division up to `√n`.
fn trial_division_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut divisor = 2u64;
    while divisor.saturating_mul(divisor) <= n {
        if n.is_multiple_of(divisor) {
            return false;
        }
        divisor += 1;
    }
    true
}

fn powmod(base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let modulus = u128::from(modulus);
    let mut base = u128::from(base) % modulus;
    let mut result = 1 % modulus;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    u64::try_from(result).expect("residue below a u64 modulus")
}

/// Lucas primality certificate: `factors` is the prime factorization of
/// `n − 1` with multiplicity, each factor proven by trial division, and some
/// small base has multiplicative order exactly `n − 1`.
fn lucas_certified_prime(n: u64, factors: &[u64]) -> bool {
    let product = factors
        .iter()
        .try_fold(1u64, |acc, &factor| acc.checked_mul(factor))
        .expect("factorization of n - 1 fits in u64");
    assert_eq!(product, n - 1, "certificate factors multiply to n - 1");
    for &factor in factors {
        assert!(trial_division_prime(factor), "certificate factor {factor}");
    }
    (2..1_000u64).any(|base| {
        powmod(base, n - 1, n) == 1
            && factors
                .iter()
                .all(|&factor| powmod(base, (n - 1) / factor, n) != 1)
    })
}

/// Compositeness by an explicit nontrivial factorization.
fn factored_composite(n: u64, factors: &[u64]) -> bool {
    let product = factors
        .iter()
        .try_fold(1u64, |acc, &factor| acc.checked_mul(factor))
        .expect("factorization fits in u64");
    factors.len() >= 2 && factors.iter().all(|&factor| factor > 1) && product == n
}

#[test]
fn is_prime_matches_trial_division_below_ten_thousand() {
    for n in 0..10_000u64 {
        assert_eq!(
            is_prime(black_box(n)),
            trial_division_prime(n),
            "is_prime({n})"
        );
    }
}

#[test]
fn is_prime_matches_trial_division_above_two_to_the_32() {
    // Odd and even candidates past 2^32: every witness reduces without
    // vanishing and products need the full 128-bit intermediate.
    let start = 1u64 << 32;
    for n in start - 64..start + 256 {
        assert_eq!(
            is_prime(black_box(n)),
            trial_division_prime(n),
            "is_prime({n})"
        );
    }
}

#[test]
fn is_prime_rejects_strong_pseudoprimes_and_carmichael_numbers() {
    // Strong pseudoprimes to the first few prime bases (the smallest for
    // bases {2}, {2,3}, {2,3,5}, ... {2..23}) and Carmichael numbers.
    let composites: [(u64, &[u64]); 14] = [
        (561, &[3, 11, 17]),
        (1_105, &[5, 13, 17]),
        (1_729, &[7, 13, 19]),
        (2_047, &[23, 89]),
        (1_373_653, &[829, 1_657]),
        (25_326_001, &[2_251, 11_251]),
        (3_215_031_751, &[151, 751, 28_351]),
        (2_152_302_898_747, &[6_763, 10_627, 29_947]),
        (3_474_749_660_383, &[1_303, 16_927, 157_543]),
        (341_550_071_728_321, &[10_670_053, 32_010_157]),
        (3_825_123_056_546_413_051, &[149_491, 747_451, 34_233_211]),
        // Products of the two largest primes below 2^32, and a square.
        (18_446_743_979_220_271_189, &[4_294_967_279, 4_294_967_291]),
        (18_446_744_030_759_878_681, &[4_294_967_291, 4_294_967_291]),
        (u64::MAX, &[3, 5, 17, 257, 641, 65_537, 6_700_417]),
    ];
    for (n, factors) in composites {
        assert!(factored_composite(n, factors), "{n} factorization");
        assert!(!is_prime(black_box(n)), "{n} is composite");
    }
}

#[test]
fn is_prime_accepts_certified_large_primes() {
    let mut goldilocks_minus_one = vec![2u64; 32];
    goldilocks_minus_one.extend_from_slice(&[3, 5, 17, 257, 65_537]);
    let primes: [(u64, &[u64]); 6] = [
        (1_000_000_007, &[2, 500_000_003]),
        (0x7FFF_FFFF, &[2, 3, 3, 7, 11, 31, 151, 331]),
        (4_294_967_291, &[2, 5, 19, 22_605_091]),
        (0xFFFF_FFFF_0000_0001, &goldilocks_minus_one),
        (
            (1 << 61) - 1,
            &[2, 3, 3, 5, 5, 7, 11, 13, 31, 41, 61, 151, 331, 1_321],
        ),
        // The largest prime below 2^64.
        (u64::MAX - 58, &[2, 2, 11, 137, 547, 5_594_472_617_641]),
    ];
    for (n, factors) in primes {
        assert!(lucas_certified_prime(n, factors), "{n} certificate");
        assert!(is_prime(black_box(n)), "{n} is prime");
    }
}

// ---------------------------------------------------------------------------
// QuadMersenne31 over raw limb bytes
// ---------------------------------------------------------------------------

const P: u64 = 0x7FFF_FFFF;

/// Limb spellings that stress the Mersenne fold: canonical boundaries, both
/// representations of zero (`0`, `p`), and the high-bit patterns whose fold
/// lands on or above `p` (`0xFFFF_FFFE ≡ 0`, `0xFFFF_FFFF ≡ 1`).
const LIMBS: [u32; 10] = [
    0,
    1,
    2,
    0x5555_5555,
    0x7FFF_FFFE,
    0x7FFF_FFFF,
    0x8000_0000,
    0xC000_0001,
    0xFFFF_FFFE,
    0xFFFF_FFFF,
];

/// Element lengths in bytes on both sides of the vector thresholds: scalar
/// rows, one AVX2 vector, vector-plus-tail, and multi-vector rows.
const LENS: [usize; 9] = [8, 16, 24, 32, 40, 56, 64, 72, 136];

/// `len / 8` raw elements cycling through the limb spellings with
/// independent strides on each limb.
fn raw_elems(len: usize, re_stride: usize, im_stride: usize, offset: usize) -> Vec<(u32, u32)> {
    (0..len / 8)
        .map(|i| {
            (
                LIMBS[(offset + i * re_stride) % LIMBS.len()],
                LIMBS[(offset + 3 + i * im_stride) % LIMBS.len()],
            )
        })
        .collect()
}

fn to_bytes(elems: &[(u32, u32)]) -> Vec<u8> {
    elems
        .iter()
        .flat_map(|&(re, im)| {
            let mut bytes = [0u8; 8];
            bytes[..4].copy_from_slice(&re.to_le_bytes());
            bytes[4..].copy_from_slice(&im.to_le_bytes());
            bytes
        })
        .collect()
}

/// Canonical pair from raw limbs, reduced with `%`.
fn reduce(value: (u32, u32)) -> (u64, u64) {
    (u64::from(value.0) % P, u64::from(value.1) % P)
}

fn add(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    ((a.0 + b.0) % P, (a.1 + b.1) % P)
}

fn sub(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    ((a.0 + P - b.0) % P, (a.1 + P - b.1) % P)
}

fn mul(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    let re = (a.0 * b.0 % P + P - a.1 * b.1 % P) % P;
    let im = (a.0 * b.1 % P + a.1 * b.0 % P) % P;
    (re, im)
}

fn canonical_bytes(elems: &[(u64, u64)]) -> Vec<u8> {
    let raw: Vec<(u32, u32)> = elems
        .iter()
        .map(|&(re, im)| {
            (
                u32::try_from(re).expect("canonical limb"),
                u32::try_from(im).expect("canonical limb"),
            )
        })
        .collect();
    to_bytes(&raw)
}

fn elem(value: (u64, u64)) -> Elem<QuadMersenne31> {
    Elem::<QuadMersenne31>::from_raw(
        u32::try_from(value.0).expect("canonical limb"),
        u32::try_from(value.1).expect("canonical limb"),
    )
}

fn map(a: &[(u32, u32)], f: impl Fn((u64, u64)) -> (u64, u64)) -> Vec<u8> {
    let out: Vec<(u64, u64)> = a.iter().map(|&x| f(reduce(x))).collect();
    canonical_bytes(&out)
}

fn zip(
    a: &[(u32, u32)],
    b: &[(u32, u32)],
    f: impl Fn((u64, u64), (u64, u64)) -> (u64, u64),
) -> Vec<u8> {
    let out: Vec<(u64, u64)> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f(reduce(x), reduce(y)))
        .collect();
    canonical_bytes(&out)
}

/// Every bulk operation canonicalizes raw destination and source limbs,
/// including the spellings whose Mersenne fold reaches `p`, on rows served
/// by the scalar loops and by the vector kernels.
#[test]
fn quad_mersenne31_ops_canonicalize_raw_limbs() {
    let coeffs = [(3u64, 5u64), (0, 1), (P - 1, 2), (0x5555_5555, 0x1234_5678)];
    for len in LENS {
        let dst_raw = raw_elems(len, 1, 3, 0);
        let src_raw = raw_elems(len, 3, 7, 5);
        let dst = to_bytes(&dst_raw);
        let src = to_bytes(&src_raw);

        let mut got = dst.clone();
        ops::add_assign::<QuadMersenne31>(&mut got, &src);
        assert_eq!(got, zip(&dst_raw, &src_raw, add), "add_assign len {len}");

        let mut got = dst.clone();
        ops::sub_assign::<QuadMersenne31>(&mut got, &src);
        assert_eq!(got, zip(&dst_raw, &src_raw, sub), "sub_assign len {len}");

        let mut got = vec![0u8; len];
        ops::mul_elementwise::<QuadMersenne31>(&mut got, &dst, &src);
        assert_eq!(
            got,
            zip(&dst_raw, &src_raw, mul),
            "mul_elementwise len {len}"
        );

        let mut got = dst.clone();
        ops::mul_elementwise_assign::<QuadMersenne31>(&mut got, &src);
        assert_eq!(
            got,
            zip(&dst_raw, &src_raw, mul),
            "mul_elementwise_assign len {len}"
        );

        for coeff in coeffs {
            let c = elem(coeff);

            let mut got = dst.clone();
            ops::mul_add::<QuadMersenne31>(&mut got, c, &src);
            assert_eq!(
                got,
                zip(&dst_raw, &src_raw, |d, s| add(d, mul(coeff, s))),
                "mul_add len {len} coeff {coeff:?}"
            );

            let mut got = vec![0xA5u8; len];
            ops::mul_into::<QuadMersenne31>(&mut got, c, &src);
            assert_eq!(
                got,
                map(&src_raw, |s| mul(coeff, s)),
                "mul_into len {len} coeff {coeff:?}"
            );

            let mut got = dst.clone();
            ops::mul_assign::<QuadMersenne31>(&mut got, c);
            assert_eq!(
                got,
                map(&dst_raw, |d| mul(coeff, d)),
                "mul_assign len {len} coeff {coeff:?}"
            );

            let mut got = dst.clone();
            ops::add_assign_scalar::<QuadMersenne31>(&mut got, c);
            assert_eq!(
                got,
                map(&dst_raw, |d| add(d, coeff)),
                "add_assign_scalar len {len} coeff {coeff:?}"
            );

            let mut got = dst.clone();
            ops::sub_assign_scalar::<QuadMersenne31>(&mut got, c);
            assert_eq!(
                got,
                map(&dst_raw, |d| sub(d, coeff)),
                "sub_assign_scalar len {len} coeff {coeff:?}"
            );
        }
    }
}

/// The elementwise vector threshold is a whole number of elements, and it
/// is nonzero exactly when a vector kernel serves the elementwise product.
#[test]
fn quad_mersenne31_elementwise_threshold_is_consistent() {
    let threshold = vector_elementwise_min_bytes::<QuadMersenne31>();
    assert_eq!(threshold % 8, 0, "threshold {threshold} splits an element");
    assert_eq!(
        threshold != 0,
        has_vector_elementwise::<QuadMersenne31>(),
        "threshold {threshold} disagrees with elementwise eligibility"
    );
}
