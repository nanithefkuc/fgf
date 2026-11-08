//! `QuadMersenne31` kernel differentials: the dispatch-level zero/one
//! shortcuts here, the AVX2 and AVX-512 suites in the ISA modules.

use super::*;
use crate::field::quad_mersenne31::QuadMersenne31;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;
#[cfg(all(feature = "simd", feature = "simd512", target_arch = "x86_64"))]
mod avx512;

#[test]
fn quad_mersenne31_kernel_handles_zero_and_one_coefficients() {
    // The kernel-level zero/one shortcuts sit below the `ops` wrappers,
    // which short-circuit first; drive them directly.

    let mut zeroed = noise(16, 0x67);
    <QuadMersenne31 as KernelDispatch>::mul_assign(
        RawDispatch,
        &mut zeroed,
        &QuadMersenne31::prepare(RawDispatch, quad_zero()),
    );
    assert!(
        zeroed.iter().all(|&b| b == 0),
        "zero coefficient fills zero"
    );

    let original = noise(16, 0x68);
    let mut scaled = original.clone();
    <QuadMersenne31 as KernelDispatch>::mul_assign(
        RawDispatch,
        &mut scaled,
        &QuadMersenne31::prepare(RawDispatch, quad_one()),
    );
    assert_eq!(scaled, original, "one coefficient is the identity");
}

fn quad_zero() -> Elem<QuadMersenne31> {
    Elem::<QuadMersenne31>::from_raw([0, 0])
}

fn quad_one() -> Elem<QuadMersenne31> {
    Elem::<QuadMersenne31>::from_raw([1, 0])
}

/// Noncanonical limb aliases at every element position, and every ordered
/// pair of boundary words, against the portable reference and an
/// independent `u128 % p` complex model; every stored limb is canonical.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
// Nine entries at a dozen lengths: a table by nature.
#[allow(clippy::too_many_lines)]
fn qm31_entries_canonicalize_noncanonical_lanes() {
    use crate::field::quad_mersenne31::QuadMersenne31;
    use crate::kernel::prime;
    use crate::kernel::x86;

    type Qm = Elem<QuadMersenne31>;
    const P: u32 = crate::field::mersenne31::MODULUS;
    // Limb pairs cycling both canonical and alias limbs through each
    // element position.
    const LIMBS: &[u32] = &[0, 1, P - 1, P, P + 1, 0x8000_0000, 0xFFFF_FFFE, u32::MAX];
    const LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
    const COEFFS: &[(u32, u32)] = &[(0, 0), (1, 0), (0, 1), (3, 5), (P, 7), (u32::MAX, P)];

    fn fill(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while out.len() < len {
            let (re, im) = (LIMBS[i % LIMBS.len()], LIMBS[(i + 3) % LIMBS.len()]);
            out.extend_from_slice(&re.to_le_bytes());
            out.extend_from_slice(&im.to_le_bytes());
            i += 1;
        }
        out.truncate(len);
        out
    }
    fn assert_canonical(label: &str, buf: &[u8]) {
        for (i, limb) in buf.chunks_exact(4).enumerate() {
            let lane = u32::from_le_bytes(limb.try_into().unwrap());
            assert!(lane < P, "{label}: limb {i} holds {lane:#x}, want < {P:#x}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn drive(
        label: &str,
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], Qm, &[u8]),
        mulinto: impl Fn(&mut [u8], Qm, &[u8]),
        mulassign: impl Fn(&mut [u8], Qm),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], Qm),
        subscalar: impl Fn(&mut [u8], Qm),
    ) {
        for &len in LENS {
            let src = fill(len);
            let base = {
                let mut shifted = fill(len + 8);
                shifted.drain(..8);
                shifted.truncate(len);
                shifted
            };
            let other = {
                let mut shifted = fill(len + 16);
                shifted.drain(..16);
                shifted.truncate(len);
                shifted
            };

            let mut got = base.clone();
            let mut want = base.clone();
            add(&mut got, &src);
            prime::add_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "{label} add_assign len {len}");
            assert_canonical(label, &got);

            let mut got = base.clone();
            let mut want = base.clone();
            sub(&mut got, &src);
            prime::sub_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "{label} sub_assign len {len}");
            assert_canonical(label, &got);

            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            elemwise(&mut got, &src, &other);
            prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &other);
            assert_eq!(got, want, "{label} mul_elementwise len {len}");
            assert_canonical(label, &got);

            let mut got = base.clone();
            let mut want = base.clone();
            elemwise_assign(&mut got, &src);
            prime::mul_elementwise::<QuadMersenne31>(&mut want, &base, &src);
            assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
            assert_canonical(label, &got);

            for &(re, im) in COEFFS {
                let coeff = Qm::from_raw([re, im]);
                let mut got = base.clone();
                let mut want = base.clone();
                muladd(&mut got, coeff, &src);
                prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                assert_eq!(got, want, "{label} mul_add len {len} coeff {re:#x},{im:#x}");
                // A zero coefficient is a defined no-op at every level,
                // so untouched input lanes are preserved.
                if coeff != Qm::ZERO {
                    assert_canonical(label, &got);
                }

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                mulinto(&mut got, coeff, &src);
                prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                assert_eq!(
                    got, want,
                    "{label} mul_into len {len} coeff {re:#x},{im:#x}"
                );
                assert_canonical(label, &got);

                let mut got = base.clone();
                let mut want = base.clone();
                mulassign(&mut got, coeff);
                prime::mul_assign::<QuadMersenne31>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "{label} mul_assign len {len} coeff {re:#x},{im:#x}"
                );
                // A unit coefficient is a defined no-op, preserving
                // untouched input lanes.
                if coeff != Qm::ONE {
                    assert_canonical(label, &got);
                }

                let mut got = base.clone();
                let mut want = base.clone();
                addscalar(&mut got, coeff);
                prime::add_assign_scalar::<QuadMersenne31>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "{label} add_assign_scalar len {len} coeff {re:#x},{im:#x}"
                );
                assert_canonical(label, &got);

                let mut got = base.clone();
                let mut want = base.clone();
                subscalar(&mut got, coeff);
                prime::sub_assign_scalar::<QuadMersenne31>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "{label} sub_assign_scalar len {len} coeff {re:#x},{im:#x}"
                );
                assert_canonical(label, &got);
            }
        }
    }

    /// Exhaustive boundary-word pairs for the `QuadMersenne31` entries:
    /// for every ordered pair of the seven boundary words `0, 1, p-1,
    /// p, p+1, 2p, 2p+1 (= u32::MAX)`, one operand holds `(x, y)`
    /// limbs in every element and the other `(y, x)`, so the real and
    /// imaginary limb positions each see every boundary value against
    /// every other in both operand roles. The expectations come from
    /// an independent `u128 % p` complex model, not the crate's
    /// reference.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn drive_pairs(
        label: &str,
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], Qm, &[u8]),
        mulinto: impl Fn(&mut [u8], Qm, &[u8]),
        mulassign: impl Fn(&mut [u8], Qm),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], Qm),
        subscalar: impl Fn(&mut [u8], Qm),
    ) {
        const WORDS: [u32; 7] = [0, 1, P - 1, P, P + 1, 2 * P, u32::MAX];
        // Lengths straddling the AVX2 lane (32 bytes), the AVX-512
        // lane (64), and whole-element tails, sub-lane included.
        const PAIR_LENS: &[usize] = &[8, 32, 40, 64, 72, 136];
        const VALUES: [Qm; 7] = [
            Qm::from_raw([0, 0]),
            Qm::from_raw([1, 1]),
            Qm::from_raw([P - 1, P - 1]),
            Qm::from_raw([P, P]),
            Qm::from_raw([P + 1, P + 1]),
            Qm::from_raw([2 * P, 2 * P]),
            Qm::from_raw([u32::MAX, u32::MAX]),
        ];
        let wide = u128::from(P);
        let canon = |v: u32| u128::from(v) % wide;
        let word = |v: u128| u32::try_from(v).unwrap();
        let fill_pair = |len: usize, re: u32, im: u32| -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
            }
            out.truncate(len);
            out
        };
        // The independent complex model over canonical limb values.
        let mul = |(ar, ai): (u128, u128), (br, bi): (u128, u128)| -> (u128, u128) {
            (
                (ar * br + wide * wide - ai * bi) % wide,
                (ar * bi + ai * br) % wide,
            )
        };

        for &len in PAIR_LENS {
            let pairs = WORDS
                .iter()
                .flat_map(|&x| WORDS.iter().map(move |&y| (x, y)));
            for (x, y) in pairs {
                let cv = (canon(x), canon(y));
                let dst_xy = fill_pair(len, x, y);
                let src_yx = fill_pair(len, y, x);

                let mut got = dst_xy.clone();
                add(&mut got, &src_yx);
                let want = fill_pair(len, word((cv.0 + cv.1) % wide), word((cv.1 + cv.0) % wide));
                assert_eq!(got, want, "{label} pair add {x:#x} {y:#x} len {len}");

                let mut got = dst_xy.clone();
                sub(&mut got, &src_yx);
                let want = fill_pair(
                    len,
                    word((cv.0 + wide - cv.1) % wide),
                    word((cv.1 + wide - cv.0) % wide),
                );
                assert_eq!(got, want, "{label} pair sub {x:#x} {y:#x} len {len}");

                let mut got = vec![0u8; len];
                elemwise(&mut got, &dst_xy, &src_yx);
                let (re, im) = mul(cv, (cv.1, cv.0));
                let want = fill_pair(len, word(re), word(im));
                assert_eq!(got, want, "{label} pair elemwise {x:#x} {y:#x} len {len}");

                let mut got = dst_xy.clone();
                elemwise_assign(&mut got, &src_yx);
                let (re, im) = mul(cv, (cv.1, cv.0));
                assert_eq!(
                    got,
                    fill_pair(len, word(re), word(im)),
                    "{label} pair elemwise_assign {x:#x} {y:#x} len {len}"
                );

                for &(cr, ci) in COEFFS {
                    let cc = (canon(cr), canon(ci));
                    let coeff = Qm::from_raw([cr, ci]);
                    let (pre, pim) = mul(cc, (cv.1, cv.0));

                    let mut got = dst_xy.clone();
                    muladd(&mut got, coeff, &src_yx);
                    // A zero-valued coefficient is a defined no-op that
                    // preserves the untouched limbs byte for byte.
                    let want = if cc == (0, 0) {
                        dst_xy.clone()
                    } else {
                        fill_pair(len, word((cv.0 + pre) % wide), word((cv.1 + pim) % wide))
                    };
                    assert_eq!(
                        got, want,
                        "{label} pair mul_add {x:#x} {y:#x} coeff {cr:#x},{ci:#x} len {len}"
                    );

                    let mut got = vec![0u8; len];
                    mulinto(&mut got, coeff, &src_yx);
                    let want = if cc == (0, 0) {
                        fill_pair(len, 0, 0)
                    } else {
                        fill_pair(len, word(pre), word(pim))
                    };
                    assert_eq!(
                        got, want,
                        "{label} pair mul_into {x:#x} {y:#x} coeff {cr:#x},{ci:#x} len {len}"
                    );

                    let mut got = dst_xy.clone();
                    mulassign(&mut got, coeff);
                    // A unit-valued coefficient is a defined no-op; a
                    // zero-valued one zeroes the buffer.
                    let want = if cc == (0, 0) {
                        fill_pair(len, 0, 0)
                    } else if cc == (1, 0) {
                        dst_xy.clone()
                    } else {
                        let (re, im) = mul(cc, cv);
                        fill_pair(len, word(re), word(im))
                    };
                    assert_eq!(
                        got, want,
                        "{label} pair mul_assign {x:#x} coeff {cr:#x},{ci:#x} len {len}"
                    );
                }

                for value in VALUES {
                    let [vr, vi] = value.to_raw();
                    let vv = (canon(vr), canon(vi));

                    let mut got = dst_xy.clone();
                    addscalar(&mut got, value);
                    assert_eq!(
                        got,
                        fill_pair(len, word((cv.0 + vv.0) % wide), word((cv.1 + vv.1) % wide)),
                        "{label} pair add_scalar {x:#x},{y:#x} value {vr:#x} len {len}"
                    );

                    let mut got = dst_xy.clone();
                    subscalar(&mut got, value);
                    assert_eq!(
                        got,
                        fill_pair(
                            len,
                            word((cv.0 + wide - vv.0) % wide),
                            word((cv.1 + wide - vv.1) % wide)
                        ),
                        "{label} pair sub_scalar {x:#x},{y:#x} value {vr:#x} len {len}"
                    );
                }
            }
        }
    }

    tier_gate::run_entry(x86::quad_mersenne31::add_assign_avx2, |token, entry| {
        drive(
            "qm31 avx2 noncanonical",
            |dst, src| entry(token, dst, src),
            |dst, src| x86::quad_mersenne31::sub_assign_avx2(token, dst, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_add_avx2(token, dst, coeff, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_into_avx2(token, dst, coeff, src),
            |dst, coeff| x86::quad_mersenne31::mul_assign_avx2(token, dst, coeff),
            |dst, a, b| x86::quad_mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::quad_mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::quad_mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::quad_mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
        drive_pairs(
            "qm31 avx2 pairs",
            |dst, src| entry(token, dst, src),
            |dst, src| x86::quad_mersenne31::sub_assign_avx2(token, dst, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_add_avx2(token, dst, coeff, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_into_avx2(token, dst, coeff, src),
            |dst, coeff| x86::quad_mersenne31::mul_assign_avx2(token, dst, coeff),
            |dst, a, b| x86::quad_mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::quad_mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::quad_mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::quad_mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });

    #[cfg(feature = "simd512")]
    tier_gate::run_entry(
        x86::quad_mersenne31::mul_add_avx512,
        |v4x: X64V4xToken, entry| {
            let token = v4x.v4();
            // The AVX-512 QM31 family covers the multiply shapes; the
            // remaining operations run the AVX2 entries above.
            for &len in LENS {
                let src = fill(len);
                let base = fill(len);
                for &(re, im) in COEFFS {
                    let coeff = Qm::from_raw([re, im]);
                    let mut got = base.clone();
                    let mut want = base.clone();
                    entry(token, &mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(got, want, "qm31 avx512 mul_add len {len}");
                    // A zero coefficient is a defined no-op, preserving
                    // untouched input lanes.
                    if coeff != Qm::ZERO {
                        assert_canonical("qm31 avx512 mul_add", &got);
                    }

                    let mut got = vec![0u8; len];
                    let mut want = vec![0u8; len];
                    x86::quad_mersenne31::mul_into_avx512(token, &mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(got, want, "qm31 avx512 mul_into len {len}");
                    assert_canonical("qm31 avx512 mul_into", &got);
                }
                let other = fill(len);
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                x86::quad_mersenne31::mul_elementwise_avx512(token, &mut got, &src, &other);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &other);
                assert_eq!(got, want, "qm31 avx512 mul_elementwise len {len}");
                assert_canonical("qm31 avx512 mul_elementwise", &got);
            }

            // Boundary-word pairs for the three multiply shapes this tier
            // owns, against the same independent `u128 % p` complex model
            // as the AVX2 pair sweep.
            {
                const WORDS: [u32; 7] = [0, 1, P - 1, P, P + 1, 2 * P, u32::MAX];
                const PAIR_LENS: &[usize] = &[8, 32, 40, 64, 72, 136];
                let wide = u128::from(P);
                let canon = |v: u32| u128::from(v) % wide;
                let word = |v: u128| u32::try_from(v).unwrap();
                let fill_pair = |len: usize, re: u32, im: u32| -> Vec<u8> {
                    let mut out = Vec::with_capacity(len);
                    while out.len() < len {
                        out.extend_from_slice(&re.to_le_bytes());
                        out.extend_from_slice(&im.to_le_bytes());
                    }
                    out.truncate(len);
                    out
                };
                let mul = |(ar, ai): (u128, u128), (br, bi): (u128, u128)| -> (u128, u128) {
                    (
                        (ar * br + wide * wide - ai * bi) % wide,
                        (ar * bi + ai * br) % wide,
                    )
                };
                for &len in PAIR_LENS {
                    let pairs = WORDS
                        .iter()
                        .flat_map(|&x| WORDS.iter().map(move |&y| (x, y)));
                    for (x, y) in pairs {
                        let cv = (canon(x), canon(y));
                        let dst_xy = fill_pair(len, x, y);
                        let src_yx = fill_pair(len, y, x);

                        let mut got = vec![0u8; len];
                        x86::quad_mersenne31::mul_elementwise_avx512(
                            token, &mut got, &dst_xy, &src_yx,
                        );
                        let (re, im) = mul(cv, (cv.1, cv.0));
                        assert_eq!(
                            got,
                            fill_pair(len, word(re), word(im)),
                            "qm31 avx512 pairs elemwise {x:#x} {y:#x} len {len}"
                        );

                        for &(cr, ci) in COEFFS {
                            let cc = (canon(cr), canon(ci));
                            let coeff = Qm::from_raw([cr, ci]);
                            let (pre, pim) = mul(cc, (cv.1, cv.0));

                            let mut got = dst_xy.clone();
                            entry(token, &mut got, coeff, &src_yx);
                            let want = if cc == (0, 0) {
                                dst_xy.clone()
                            } else {
                                fill_pair(len, word((cv.0 + pre) % wide), word((cv.1 + pim) % wide))
                            };
                            assert_eq!(
                                got, want,
                                "qm31 avx512 pairs mul_add {x:#x} {y:#x} coeff {cr:#x},{ci:#x} len {len}"
                            );

                            let mut got = vec![0u8; len];
                            x86::quad_mersenne31::mul_into_avx512(token, &mut got, coeff, &src_yx);
                            let want = if cc == (0, 0) {
                                fill_pair(len, 0, 0)
                            } else {
                                fill_pair(len, word(pre), word(pim))
                            };
                            assert_eq!(
                                got, want,
                                "qm31 avx512 pairs mul_into {x:#x} {y:#x} coeff {cr:#x},{ci:#x} len {len}"
                            );
                        }
                    }
                }
            }
        },
    );
}
