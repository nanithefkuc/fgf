//! AVX2 `QuadMersenne31` differentials: the independent limb model, the
//! portable reference, the zero/one shortcuts, unaligned whole-element
//! slices, and validation-before-write geometry.

use super::*;
use crate::field::mersenne31;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::prime;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

/// Lengths in bytes straddling the 32-byte vector lane: empty, one
/// element, sub-vector tails of one and three elements, exact vectors,
/// vector-plus-tail, and a large odd size.
const LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];

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
fn model_elem_sub(a: Elem<QuadMersenne31>, b: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
    let [ar, ai] = a.to_raw();
    let [br, bi] = b.to_raw();
    Elem::<QuadMersenne31>::from_raw([o_sub(ar, br), o_sub(ai, bi)])
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

/// Every operation matches the independent limb model across boundary
/// lengths, canonical random fixtures, and canonical edge elements in
/// the source operands.
#[test]
#[allow(clippy::too_many_lines)]
fn avx2_matches_independent_model() {
    tier_gate::run_entry(x86::quad_mersenne31::add_assign_avx2, |token, entry| {
        for &len in LENS {
            for (src, base, _) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                let mut got = base.clone();
                entry(token, &mut got, &src);
                assert_eq!(
                    got,
                    model_zip(&base, &src, model_elem_add),
                    "add_assign len {len}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::quad_mersenne31::sub_assign_avx2, |token, entry| {
        for &len in LENS {
            for (src, base, _) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                let mut got = base.clone();
                entry(token, &mut got, &src);
                assert_eq!(
                    got,
                    model_zip(&base, &src, model_elem_sub),
                    "sub_assign len {len}"
                );
            }
        }
    });
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_elementwise_avx2,
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
    tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx2, |token, entry| {
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
    tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx2, |token, entry| {
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
    tier_gate::run_entry(x86::quad_mersenne31::mul_assign_avx2, |token, entry| {
        for &len in LENS {
            for (_, base, _) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                for &coeff in &COEFFS {
                    let mut got = base.clone();
                    entry(token, &mut got, coeff);
                    assert_eq!(
                        got,
                        model_map(&base, |e| model_elem_mul(coeff, e)),
                        "mul_assign len {len} coeff {coeff:?}"
                    );
                }
            }
        }
    });
}

/// Every operation agrees with the portable [`prime`] differential
/// oracle over lengths straddling the vector lane.
#[test]
fn avx2_matches_portable_reference() {
    // Vector-plus-tail, exact vectors, and a different tail.
    for &len in &[40usize, 64, 72] {
        let src = edges(len);
        tier_gate::run_entry(x86::quad_mersenne31::add_assign_avx2, |token, entry| {
            let base = canonical(len, 0xb2 + len as u64);
            let mut got = base.clone();
            let mut want = base.clone();
            entry(token, &mut got, &src);
            prime::add_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "add_assign vs reference len {len}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::sub_assign_avx2, |token, entry| {
            let base = canonical(len, 0xb2 + len as u64);
            let mut got = base.clone();
            let mut want = base.clone();
            entry(token, &mut got, &src);
            prime::sub_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "sub_assign vs reference len {len}");
        });
        tier_gate::run_entry(
            x86::quad_mersenne31::mul_elementwise_avx2,
            |token, entry| {
                let b2 = canonical(len, 0xc3 + len as u64);
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                entry(token, &mut got, &src, &b2);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &b2);
                assert_eq!(got, want, "mul_elementwise vs reference len {len}");
            },
        );
        tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx2, |token, entry| {
            let base = canonical(len, 0xb2 + len as u64);
            let mut got = base.clone();
            let mut want = base.clone();
            entry(token, &mut got, FULL_WORK, &src);
            prime::mul_add::<QuadMersenne31>(&mut want, FULL_WORK, &src);
            assert_eq!(got, want, "mul_add vs reference len {len}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx2, |token, entry| {
            let mut got = vec![0u8; len];
            let mut want = src.clone();
            entry(token, &mut got, FULL_WORK, &src);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_into vs reference len {len}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::mul_assign_avx2, |token, entry| {
            let base = canonical(len, 0xb2 + len as u64);
            let mut got = base.clone();
            let mut want = base.clone();
            entry(token, &mut got, FULL_WORK);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_assign vs reference len {len}");
        });
    }
}

/// The zero and one coefficient shortcuts preserve the scalar
/// control's observable behavior. The zero arms never read `dst`, so a
/// raw destination pattern defends the byte-preserving contract; the
/// one shortcut adds, so its destination holds canonical elements.
#[test]
fn avx2_zero_one_shortcuts_match_control() {
    let len = 40;
    let src = canonical(len, 0x71);
    // Zero, including a non-canonical limb spelling of it: the no-op
    // for mul_add and the fill for mul_assign and mul_into.
    for coeff in [
        Elem::<QuadMersenne31>::ZERO,
        Elem::<QuadMersenne31>::from_raw([mersenne31::MODULUS, 0]),
    ] {
        tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx2, |token, entry| {
            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            entry(token, &mut got, coeff, &src);
            let prepared = <QuadMersenne31 as KernelDispatch>::prepare(RawDispatch, coeff);
            <QuadMersenne31 as KernelDispatch>::mul_add(RawDispatch, &mut want, &prepared, &src);
            assert_eq!(got, want, "shortcut mul_add {coeff:?}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::mul_assign_avx2, |token, entry| {
            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            entry(token, &mut got, coeff);
            let prepared = <QuadMersenne31 as KernelDispatch>::prepare(RawDispatch, coeff);
            <QuadMersenne31 as KernelDispatch>::mul_assign(RawDispatch, &mut want, &prepared);
            assert_eq!(got, want, "shortcut mul_assign {coeff:?}");
        });
        tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx2, |token, entry| {
            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            entry(token, &mut got, coeff, &src);
            let prepared = <QuadMersenne31 as KernelDispatch>::prepare(RawDispatch, coeff);
            <QuadMersenne31 as KernelDispatch>::mul_into(RawDispatch, &mut want, &prepared, &src);
            assert_eq!(got, want, "shortcut mul_into {coeff:?}");
        });
    }
    // The one shortcut: mul_add adds, mul_assign is the identity.
    tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx2, |token, entry| {
        let mut got = canonical(len, 0x72);
        let mut want = got.clone();
        entry(token, &mut got, Elem::<QuadMersenne31>::ONE, &src);
        <QuadMersenne31 as KernelDispatch>::mul_add(
            RawDispatch,
            &mut want,
            &<QuadMersenne31 as KernelDispatch>::prepare(RawDispatch, Elem::<QuadMersenne31>::ONE),
            &src,
        );
        assert_eq!(got, want, "shortcut mul_add one");
    });
    tier_gate::run_entry(x86::quad_mersenne31::mul_assign_avx2, |token, entry| {
        let mut got = canonical(len, 0x72);
        let mut want = got.clone();
        entry(token, &mut got, Elem::<QuadMersenne31>::ONE);
        <QuadMersenne31 as KernelDispatch>::mul_assign(
            RawDispatch,
            &mut want,
            &<QuadMersenne31 as KernelDispatch>::prepare(RawDispatch, Elem::<QuadMersenne31>::ONE),
        );
        assert_eq!(got, want, "shortcut mul_assign one");
    });
}

/// Unaligned but whole-element slices are legal: every access is an
/// unaligned load or store. Each offset window holds canonical element
/// bytes copied in; only its address is odd.
#[test]
fn avx2_unaligned_slices() {
    let len = 72;
    let src_elems = edges(len);
    let dst_elems = canonical(len, 0x82);
    let coeff = Elem::<QuadMersenne31>::from_raw([3, 5]);
    tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx2, |token, entry| {
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
    tier_gate::run_entry(x86::quad_mersenne31::add_assign_avx2, |token, entry| {
        for offset in [0usize, 1, 3, 5] {
            let mut src_storage = vec![0u8; len + 8];
            src_storage[offset..offset + len].copy_from_slice(&src_elems);
            let src = &src_storage[offset..offset + len];
            // Unaligned destination as well as source.
            let mut dst_storage = vec![0u8; len + 8];
            dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
            let dst = &mut dst_storage[offset..offset + len];
            entry(token, dst, src);
            assert_eq!(
                &dst_storage[offset..offset + len],
                &model_zip(&dst_elems, &src_elems, model_elem_add)[..],
                "unaligned add_assign offset {offset}"
            );
        }
    });
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
fn avx2_geometry_validation_precedes_writes() {
    let Some(token) = tier_gate::summon::<X64V3Token>() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let coeff = Elem::<QuadMersenne31>::from_raw([3, 5]);

    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("add partial", &mut partial, |d| {
        x86::quad_mersenne31::add_assign_avx2(token, d, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("sub partial", &mut partial, |d| {
        x86::quad_mersenne31::sub_assign_avx2(token, d, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_add partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_add_avx2(token, d, coeff, &partial_src);
    });
    let mut partial = [1u8; 12];
    expect_geometry_panic("mul_assign partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_assign_avx2(token, d, coeff);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_into partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_into_avx2(token, d, coeff, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("mul_elementwise partial", &mut partial, |d| {
        x86::quad_mersenne31::mul_elementwise_avx2(token, d, &partial_src, &partial_src);
    });

    let mut mismatched = [1u8; 16];
    let short_src = [1u8; 8];
    expect_geometry_panic("add length mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::add_assign_avx2(token, d, &short_src);
    });
    let mut mismatched = [1u8; 16];
    let full_src = [1u8; 16];
    let short_src = [1u8; 8];
    expect_geometry_panic("mul_elementwise b mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::mul_elementwise_avx2(token, d, &full_src, &short_src);
    });
}

/// Multiples of 8 (one complex element) straddling the 32-byte AVX2
/// lane, with odd element tails: every AVX2 `QuadMersenne31` entry against
/// the portable prime reference through the shared prime drives.
#[test]
#[allow(clippy::too_many_lines)]
fn qm31_avx2_kernels_match_reference() {
    const QM31_LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
    const P: u32 = crate::field::mersenne31::MODULUS;
    const QM31_VALUES: [Elem<QuadMersenne31>; 6] = [
        Elem::<QuadMersenne31>::from_raw([0, 0]),
        Elem::<QuadMersenne31>::from_raw([1, 0]),
        Elem::<QuadMersenne31>::from_raw([0, 1]),
        Elem::<QuadMersenne31>::from_raw([7, 2]),
        Elem::<QuadMersenne31>::from_raw([P - 1, 3]),
        Elem::<QuadMersenne31>::from_raw([P - 1, P - 1]),
    ];
    tier_gate::run_entry(x86::quad_mersenne31::add_assign_avx2, |token, entry| {
        drive_prime_len::<QuadMersenne31>(
            QM31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<QuadMersenne31>(want, src);
            },
            "qm31 avx2",
            "add_assign",
        );
    });
    tier_gate::run_entry(x86::quad_mersenne31::sub_assign_avx2, |token, entry| {
        drive_prime_len::<QuadMersenne31>(
            QM31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<QuadMersenne31>(want, src);
            },
            "qm31 avx2",
            "sub_assign",
        );
    });
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_elementwise_avx2,
        |token, entry| {
            drive_prime_len::<QuadMersenne31>(
                QM31_LENS,
                |got, want, _, src, b2| {
                    *got = vec![0u8; src.len()];
                    *want = vec![0u8; src.len()];
                    entry(token, got, src, b2);
                    prime::mul_elementwise::<QuadMersenne31>(want, src, b2);
                },
                "qm31 avx2",
                "mul_elementwise",
            );
        },
    );
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_elementwise_assign_avx2,
        |token, entry| {
            drive_prime_len::<QuadMersenne31>(
                QM31_LENS,
                |got, want, base, src, _| {
                    entry(token, got, src);
                    prime::mul_elementwise::<QuadMersenne31>(want, base, src);
                },
                "qm31 avx2",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(x86::quad_mersenne31::mul_add_avx2, |token, entry| {
        drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
            QM31_LENS,
            &QM31_VALUES,
            |e| e,
            |got, want, c, _, src| {
                entry(token, got, c, src);
                prime::mul_add::<QuadMersenne31>(want, c, src);
            },
            "qm31 avx2",
            "mul_add",
        );
    });
    tier_gate::run_entry(x86::quad_mersenne31::mul_into_avx2, |token, entry| {
        drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
            QM31_LENS,
            &QM31_VALUES,
            |e| e,
            |got, want, c, _, src| {
                *want = src.to_vec();
                entry(token, got, c, src);
                prime::mul_assign::<QuadMersenne31>(want, c);
            },
            "qm31 avx2",
            "mul_into",
        );
    });
    tier_gate::run_entry(x86::quad_mersenne31::mul_assign_avx2, |token, entry| {
        drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
            QM31_LENS,
            &QM31_VALUES,
            |e| e,
            |got, want, c, _, _| {
                entry(token, got, c);
                prime::mul_assign::<QuadMersenne31>(want, c);
            },
            "qm31 avx2",
            "mul_assign",
        );
    });
    tier_gate::run_entry(
        x86::quad_mersenne31::add_assign_scalar_avx2,
        |token, entry| {
            drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
                QM31_LENS,
                &QM31_VALUES,
                |e| e,
                |got, want, c, _, _| {
                    entry(token, got, c);
                    prime::add_assign_scalar::<QuadMersenne31>(want, c);
                },
                "qm31 avx2",
                "add_assign_scalar",
            );
        },
    );
    tier_gate::run_entry(
        x86::quad_mersenne31::sub_assign_scalar_avx2,
        |token, entry| {
            drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
                QM31_LENS,
                &QM31_VALUES,
                |e| e,
                |got, want, c, _, _| {
                    entry(token, got, c);
                    prime::sub_assign_scalar::<QuadMersenne31>(want, c);
                },
                "qm31 avx2",
                "sub_assign_scalar",
            );
        },
    );
}
