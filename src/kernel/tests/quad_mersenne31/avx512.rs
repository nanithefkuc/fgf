//! AVX-512 `QuadMersenne31` differentials: the independent limb model, the
//! portable reference, unaligned whole-element slices, and
//! validation-before-write geometry for the multiply shapes the tier owns.

use super::*;
use crate::field::mersenne31;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::prime;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

/// Lengths in bytes straddling the 64-byte vector lane: empty, one
/// element, sub-lane tails of one to five elements, exact lanes,
/// lane-plus-tail, and large sizes with whole and partial lanes.
const LENS: &[usize] = &[0, 8, 16, 24, 40, 64, 72, 128, 264, 320, 512, 1024, 2056];

/// Coefficients: zero, one, the imaginary unit, canonical limb
/// extremes, mixed values, the field generator, and raw non-canonical
/// limbs — the last two exercise the coefficient canonicalization the
/// contract allows.
const COEFFS: [Elem<QuadMersenne31>; 9] = [
    Elem::<QuadMersenne31>::ZERO,
    Elem::<QuadMersenne31>::ONE,
    Elem::<QuadMersenne31>::I,
    Elem::<QuadMersenne31>::from_raw([mersenne31::MODULUS - 1, mersenne31::MODULUS - 1]),
    Elem::<QuadMersenne31>::from_raw([1, mersenne31::MODULUS - 1]),
    Elem::<QuadMersenne31>::from_raw([0x5555_5555, 0x1234_5678]),
    Elem::<QuadMersenne31>::from_raw([mersenne31::MODULUS, 0]), // non-canonical zero
    Elem::<QuadMersenne31>::from_raw([u32::MAX, 2]),            // raw limb, value 1 + 2i
    Elem::<QuadMersenne31>::from_raw([1, 12]),                  // the field generator
];

/// A full-work coefficient: canonical, neither zero nor one, both
/// limbs nonzero.
const FULL_WORK: Elem<QuadMersenne31> =
    Elem::<QuadMersenne31>::from_raw([0x1234_5678, 0x3FFF_FFF1]);

/// Random whole elements with canonical limbs, meeting the packed
/// canonical-lane contract.
fn canonical(len: usize, seed: u64) -> Vec<u8> {
    let mut out = noise(len, seed);
    for chunk in out.chunks_exact_mut(8) {
        let e = Elem::<QuadMersenne31>::from_bytes(chunk.try_into().unwrap());
        chunk.copy_from_slice(&e.to_bytes());
    }
    out
}

/// Canonical edge elements tiled to `len` bytes: zero, both-limb and
/// single-limb `p - 1` (the largest canonical limb), one, and a mixed
/// pair — the range edges in the source operands themselves.
fn edges(len: usize) -> Vec<u8> {
    const EDGES: [Elem<QuadMersenne31>; 5] = [
        Elem::<QuadMersenne31>::ZERO,
        Elem::<QuadMersenne31>::from_raw([mersenne31::MODULUS - 1, mersenne31::MODULUS - 1]),
        Elem::<QuadMersenne31>::from_raw([mersenne31::MODULUS - 1, 0]),
        Elem::<QuadMersenne31>::ONE,
        Elem::<QuadMersenne31>::from_raw([0x5555_5555, 0x1234_5678]),
    ];
    let mut out = Vec::with_capacity(len);
    for i in 0..len / 8 {
        out.extend_from_slice(&EDGES[i % EDGES.len()].to_bytes());
    }
    out
}

// Independent `u128 % p` limb model: deliberately sharing no reduction
// shape with the kernels or the field type.
fn o_canon(x: u32) -> u32 {
    u32::try_from(u128::from(x) % u128::from(mersenne31::MODULUS)).unwrap()
}
fn o_add(a: u32, b: u32) -> u32 {
    u32::try_from(
        (u128::from(o_canon(a)) + u128::from(o_canon(b))) % u128::from(mersenne31::MODULUS),
    )
    .unwrap()
}
fn o_sub(a: u32, b: u32) -> u32 {
    let d = i128::from(o_canon(a)) - i128::from(o_canon(b));
    u32::try_from(d.rem_euclid(i128::from(mersenne31::MODULUS))).unwrap()
}
fn o_mul(a: u32, b: u32) -> u32 {
    u32::try_from(
        (u128::from(o_canon(a)) * u128::from(o_canon(b))) % u128::from(mersenne31::MODULUS),
    )
    .unwrap()
}

fn model_elem_add(a: Elem<QuadMersenne31>, b: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let [ar, ai] = a.to_raw();
    let [br, bi] = b.to_raw();
    Elem::<QuadMersenne31>::from_raw([o_add(ar, br), o_add(ai, bi)])
}
fn model_elem_mul(a: Elem<QuadMersenne31>, b: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let [ar, ai] = a.to_raw();
    let [br, bi] = b.to_raw();
    Elem::<QuadMersenne31>::from_raw([
        o_sub(o_mul(ar, br), o_mul(ai, bi)),
        o_add(o_mul(ar, bi), o_mul(ai, br)),
    ])
}

/// Decode a whole-element buffer, apply `f` elementwise, re-encode.
fn model_map(buf: &[u8], f: impl Fn(Elem<QuadMersenne31>) -> Elem<QuadMersenne31>) -> Vec<u8> {
    let mut out = buf.to_vec();
    for chunk in out.chunks_exact_mut(8) {
        let e = f(Elem::<QuadMersenne31>::from_bytes(
            chunk.try_into().unwrap(),
        ));
        chunk.copy_from_slice(&e.to_bytes());
    }
    out
}

/// Decode two whole-element buffers positionally, apply `f` pairwise,
/// re-encode.
fn model_zip(
    a: &[u8],
    b: &[u8],
    f: impl Fn(Elem<QuadMersenne31>, Elem<QuadMersenne31>) -> Elem<QuadMersenne31>,
) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    let mut out = a.to_vec();
    for (ca, cb) in out.chunks_exact_mut(8).zip(b.chunks_exact(8)) {
        let e = f(
            Elem::<QuadMersenne31>::from_bytes(ca.try_into().unwrap()),
            Elem::<QuadMersenne31>::from_bytes(cb.try_into().unwrap()),
        );
        ca.copy_from_slice(&e.to_bytes());
    }
    out
}

/// Every entry matches the independent limb model across boundary
/// lengths, canonical random fixtures, and canonical edge elements in
/// the source operands.
#[test]
fn avx512_matches_independent_model() {
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_elementwise_avx512,
        |token, entry| {
            for &len in LENS {
                for (src, _, b2) in [
                    (
                        canonical(len, 0xa1 + len as u64),
                        canonical(len, 0xb2 + len as u64),
                        canonical(len, 0xc3 + len as u64),
                    ),
                    (edges(len), edges(len), edges(len)),
                ] {
                    let mut got = vec![0u8; len];
                    entry(token, &mut got, &src, &b2);
                    assert_eq!(
                        got,
                        model_zip(&src, &b2, model_elem_mul),
                        "mul_elementwise len {len}"
                    );
                }
            }
        },
    );
    tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx512, |token, entry| {
        for &len in LENS {
            for (src, base, _) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                for &coeff in &COEFFS {
                    let mut got = base.clone();
                    entry(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_zip(&base, &src, |d, s| {
                            model_elem_add(d, model_elem_mul(coeff, s))
                        }),
                        "mul_add len {len} coeff {coeff:?}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx512, |token, entry| {
        for &len in LENS {
            for (src, _, _) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                for &coeff in &COEFFS {
                    let mut got = vec![0u8; len];
                    entry(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_map(&src, |e| model_elem_mul(coeff, e)),
                        "mul_into len {len} coeff {coeff:?}"
                    );
                }
            }
        }
    });
}

/// Every entry agrees with the portable [`prime`] differential oracle
/// over lengths straddling the vector lane.
#[test]
fn avx512_matches_portable_reference() {
    // Sub-lane, exact lanes, lane-plus-tail, and a different tail.
    for &len in &[40usize, 64, 72, 136] {
        let src = edges(len);
        tier_gate::run_entry(
            x86::quad_mersenne31::mul_elementwise_avx512,
            |token, entry| {
                let b2 = canonical(len, 0xc3 + len as u64);
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                entry(token, &mut got, &src, &b2);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &b2);
                assert_eq!(got, want, "mul_elementwise vs reference len {len}");
            },
        );
        tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx512, |token, entry| {
            let base = canonical(len, 0xb2 + len as u64);
            let mut got = base.clone();
            let mut want = base.clone();
            entry(token, &mut got, FULL_WORK, &src);
            prime::mul_add::<QuadMersenne31>(&mut want, FULL_WORK, &src);
            assert_eq!(got, want, "mul_add vs reference len {len}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx512, |token, entry| {
            let mut got = vec![0u8; len];
            let mut want = src.clone();
            entry(token, &mut got, FULL_WORK, &src);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_into vs reference len {len}");
        });
    }
}

/// Unaligned but whole-element slices are legal: every access is an
/// unaligned load or store. Each offset window holds canonical element
/// bytes copied in; only its address is odd.
#[test]
fn avx512_unaligned_slices() {
    let len = 136;
    let src_elems = edges(len);
    let dst_elems = canonical(len, 0x82);
    let coeff = Elem::<QuadMersenne31>::from_raw([3, 5]);
    tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx512, |token, entry| {
        for offset in [0usize, 1, 3, 5] {
            let mut src_storage = vec![0u8; len + 8];
            src_storage[offset..offset + len].copy_from_slice(&src_elems);
            let src = &src_storage[offset..offset + len];
            let mut got = vec![0u8; len];
            entry(token, &mut got, coeff, src);
            assert_eq!(
                got,
                model_map(src, |e| model_elem_mul(coeff, e)),
                "unaligned mul_into offset {offset}"
            );
        }
    });
    tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx512, |token, entry| {
        for offset in [0usize, 1, 3, 5] {
            let mut src_storage = vec![0u8; len + 8];
            src_storage[offset..offset + len].copy_from_slice(&src_elems);
            let src = &src_storage[offset..offset + len];
            // Unaligned destination as well as source.
            let mut dst_storage = vec![0u8; len + 8];
            dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
            let dst = &mut dst_storage[offset..offset + len];
            entry(token, dst, coeff, src);
            assert_eq!(
                &dst_storage[offset..offset + len],
                &model_zip(&dst_elems, &src_elems, |d, s| model_elem_add(
                    d,
                    model_elem_mul(coeff, s)
                ))[..],
                "unaligned mul_add offset {offset}"
            );
        }
    });
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_elementwise_avx512,
        |token, entry| {
            for offset in [0usize, 1, 3, 5] {
                let mut src_storage = vec![0u8; len + 8];
                src_storage[offset..offset + len].copy_from_slice(&src_elems);
                let src = &src_storage[offset..offset + len];
                let mut got = vec![0u8; len];
                entry(token, &mut got, src, &src_elems);
                assert_eq!(
                    got,
                    model_zip(&src_elems, &src_elems, model_elem_mul),
                    "unaligned mul_elementwise offset {offset}"
                );
            }
        },
    );
}

/// The invocation must panic on bad geometry and leave `dst`
/// unchanged: validation precedes every write.
fn expect_geometry_panic(label: &str, dst: &mut [u8], invoke: impl FnOnce(&mut [u8])) {
    let before = dst.to_vec();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invoke(dst))).is_err();
    assert!(panicked, "{label} must panic");
    assert_eq!(
        &*dst,
        &before[..],
        "{label} must not write before validating"
    );
}

/// Geometry validation fires before any write: every entry rejects a
/// partial trailing element, and every multi-operand entry rejects a
/// length mismatch.
#[test]
fn avx512_geometry_validation_precedes_writes() {
    let Some(token) = tier_gate::summon::<archmage::X64V4Token>() else {
        eprintln!("skipping: no AVX-512 on this host");
        return;
    };
    let coeff = Elem::<QuadMersenne31>::from_raw([3, 5]);

    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_add partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_add_avx512(token, d, coeff, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_into partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_into_avx512(token, d, coeff, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_elementwise partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_elementwise_avx512(token, d, &partial_src, &partial_src);
    });

    let mut mismatched = [1u8; 16];
    let short_src = [1u8; 8];
    expect_geometry_panic("mul_add length mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::mul_add_avx512(token, d, coeff, &short_src);
    });
    let mut mismatched = [1u8; 16];
    expect_geometry_panic("mul_into length mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::mul_into_avx512(token, d, coeff, &short_src);
    });
    let mut mismatched = [1u8; 16];
    let full_src = [1u8; 16];
    expect_geometry_panic("mul_elementwise b mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::mul_elementwise_avx512(token, d, &full_src, &short_src);
    });
}
