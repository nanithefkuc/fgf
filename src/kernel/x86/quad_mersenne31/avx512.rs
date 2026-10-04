//! `QuadMersenne31` kernels over 512-bit lanes (`V4x`).
//!
//! Element-for-element ports of the `avx2` multiply bodies at eight complex
//! elements per vector: the same deferred 64-bit complex products, the same
//! split against `2^31 ≡ 1 (mod p)`, and the same min-chain
//! canonicalization, so every element produces the bytes the AVX2 kernel
//! produces. Only AVX-512F instructions are used, so each entry takes
//! `X64V4Token`. Complete 64-byte lanes run here; the remainder is handed
//! to the matching AVX2 entry.

use crate::field::Elem;
use crate::field::mersenne31;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::x86::mersenne31::P;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Odd 32-bit lanes of a 512-bit vector, as a blend mask.
const ODD_LANES: __mmask16 = 0xAAAA;

/// Duplicate each element's imaginary limb into both lanes of its pair.
#[archmage::rite(v4)]
#[must_use]
fn dup_im(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)))
}

/// Mersenne fold per 32-bit lane: the low 31 bits plus the bit-31 carry.
#[archmage::rite(v4)]
#[must_use]
fn fold(x: __m512i, p: __m512i) -> __m512i {
    _mm512_add_epi32(_mm512_and_si512(x, p), _mm512_srli_epi32::<31>(x))
}

/// Canonicalize lanes in `0..=2p`: two unconditional subtract-and-min steps.
#[archmage::rite(v4)]
#[must_use]
fn min_chain(x: __m512i, p: __m512i) -> __m512i {
    let u = _mm512_min_epu32(x, _mm512_sub_epi32(x, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// `a + b (mod p)` over sixteen folded lanes (`0..=p` in, canonical out).
#[archmage::rite(v4)]
#[must_use]
fn addmod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let t = _mm512_add_epi32(a, b);
    let u = _mm512_min_epu32(t, _mm512_sub_epi32(t, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// Reduce deferred 64-bit product accumulators into packed canonical limb
/// pairs: the even dwords carry the real part and the odd dwords the
/// imaginary part, each split against `2^31 ≡ 1 (mod p)` and canonicalized
/// with the min chain.
#[archmage::rite(v4)]
#[must_use]
fn reduce_pair(re: __m512i, im: __m512i, p: __m512i) -> __m512i {
    let hi = _mm512_mask_blend_epi32(
        ODD_LANES,
        _mm512_srli_epi64::<31>(re),
        _mm512_slli_epi64::<1>(im),
    );
    let hi = _mm512_min_epu32(hi, _mm512_sub_epi32(hi, p));
    let lo = _mm512_and_si512(
        _mm512_mask_blend_epi32(ODD_LANES, re, _mm512_slli_epi64::<32>(im)),
        p,
    );
    let t = _mm512_add_epi32(lo, hi);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// Multiply eight folded complex elements by prepared coefficient
/// broadcasts: `cr` and `ci` as-is, `nci` holding `p - ci`. The real lanes
/// accumulate `re*cr + im*(p-ci) ≡ re*cr - im*ci` and the imaginary lanes
/// `im*cr + re*ci`, both deferred at 64 bits for [`reduce_pair`].
#[archmage::rite(v4)]
#[must_use]
fn cmul(x: __m512i, cr: __m512i, ci: __m512i, nci: __m512i, p: __m512i) -> __m512i {
    let xi = dup_im(x);
    let re = _mm512_add_epi64(_mm512_mul_epu32(x, cr), _mm512_mul_epu32(xi, nci));
    let im = _mm512_add_epi64(_mm512_mul_epu32(xi, cr), _mm512_mul_epu32(x, ci));
    reduce_pair(re, im, p)
}

/// Complex multiply of two folded vectors, `(ar*br - ai*bi, ar*bi + ai*br)`
/// per element, negating the duplicated imaginary limbs of `b` with `p` and
/// reducing the deferred 64-bit accumulators.
#[archmage::rite(v4)]
#[must_use]
fn cmul_vec(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let ai = dup_im(a);
    let bi = dup_im(b);
    let neg_bi = _mm512_xor_si512(bi, p);
    let re = _mm512_add_epi64(_mm512_mul_epu32(a, b), _mm512_mul_epu32(ai, neg_bi));
    let im = _mm512_add_epi64(_mm512_mul_epu32(a, bi), _mm512_mul_epu32(ai, b));
    reduce_pair(re, im, p)
}

/// The coefficient broadcasts every coefficient multiply entry builds:
/// `cr` and `ci` as-is, plus `p - ci` for the negated imaginary product.
/// Both limbs enter canonical.
#[archmage::rite(v4)]
fn coeff_vectors(cr: u32, ci: u32) -> (__m512i, __m512i, __m512i) {
    let nci = mersenne31::MODULUS - ci;
    (
        _mm512_set1_epi32(cr.cast_signed()),
        _mm512_set1_epi32(ci.cast_signed()),
        _mm512_set1_epi32(nci.cast_signed()),
    )
}

/// `dst = coeff * src` in GF((2^31 - 1)^2), AVX-512. The destination is
/// write-only: its prior contents are ignored, never read.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load, and
/// every stored limb is canonical. A zero coefficient zeroes
/// `dst`, mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx512(
    token: archmage::X64V4Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_into_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_into_avx512", dst.len(), 8);
    let (cr, ci) = coeff.to_raw();
    let p = _mm512_set1_epi32(P);
    let (cr_v, ci_v, nci) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        // Canonicalize on load: `cmul`'s deferred reduction assumes limbs
        // below the modulus, and a bare fold can leave `p + 1`.
        let s = min_chain(fold(_mm512_loadu_si512(src_lane), p), p);
        _mm512_storeu_si512(dst_lane, cmul(s, cr_v, ci_v, nci, p));
    }
    if !dst_tail.is_empty() {
        super::mul_into_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst += coeff * src` in GF((2^31 - 1)^2), AVX-512.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load, and
/// every stored limb is canonical. A zero coefficient leaves
/// `dst` untouched and the coefficient `1 + 0i` adds `src` elementwise,
/// mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512(
    token: archmage::X64V4Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_add_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_add_avx512", dst.len(), 8);
    let (cr, ci) = coeff.to_raw();
    if cr == 0 && ci == 0 {
        return;
    }
    if cr == 1 && ci == 0 {
        super::add_assign_avx2(token.v3(), dst, src);
        return;
    }
    let p = _mm512_set1_epi32(P);
    let (cr_v, ci_v, nci) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        // Canonicalize on load, as above.
        let prod = cmul(
            min_chain(fold(_mm512_loadu_si512(src_lane), p), p),
            cr_v,
            ci_v,
            nci,
            p,
        );
        let d = min_chain(fold(_mm512_loadu_si512(&*dst_lane), p), p);
        _mm512_storeu_si512(dst_lane, addmod(d, prod, p));
    }
    if !dst_tail.is_empty() {
        super::mul_add_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i]` in GF((2^31 - 1)^2), AVX-512.
///
/// Every input limb bit pattern is legal: limbs fold to canonical form on
/// load, and every stored limb is canonical.
///
/// # Panics
/// Panics unless all three buffers match in length (whole elements).
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx512(token: archmage::X64V4Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "quad_mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_elementwise_avx512", dst.len(), 8);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a.as_chunks::<64>();
    let (b_lanes, b_tail) = b.as_chunks::<64>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        // The duplicated-imaginary negation below XORs with `p`, which is
        // exact only for canonical limbs: fold and canonicalize on load.
        let va = min_chain(fold(_mm512_loadu_si512(a_lane), p), p);
        let vb = min_chain(fold(_mm512_loadu_si512(b_lane), p), p);
        _mm512_storeu_si512(dst_lane, cmul_vec(va, vb, p));
    }
    if !dst_tail.is_empty() {
        super::mul_elementwise_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::quad_mersenne31::QuadMersenne31;
    use crate::kernel::prime;
    use archmage::SimdToken as _;

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
        Elem::<QuadMersenne31>::from_raw(mersenne31::MODULUS - 1, mersenne31::MODULUS - 1),
        Elem::<QuadMersenne31>::from_raw(1, mersenne31::MODULUS - 1),
        Elem::<QuadMersenne31>::from_raw(0x5555_5555, 0x1234_5678),
        Elem::<QuadMersenne31>::from_raw(mersenne31::MODULUS, 0), // non-canonical zero
        Elem::<QuadMersenne31>::from_raw(u32::MAX, 2),            // raw limb, value 1 + 2i
        Elem::<QuadMersenne31>::from_raw(1, 12),                  // the field generator
    ];

    /// A full-work coefficient: canonical, neither zero nor one, both
    /// limbs nonzero.
    const FULL_WORK: Elem<QuadMersenne31> =
        Elem::<QuadMersenne31>::from_raw(0x1234_5678, 0x3FFF_FFF1);

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (state >> 33).to_le_bytes()[0]
            })
            .collect()
    }

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
            Elem::<QuadMersenne31>::from_raw(mersenne31::MODULUS - 1, mersenne31::MODULUS - 1),
            Elem::<QuadMersenne31>::from_raw(mersenne31::MODULUS - 1, 0),
            Elem::<QuadMersenne31>::ONE,
            Elem::<QuadMersenne31>::from_raw(0x5555_5555, 0x1234_5678),
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
        let (ar, ai) = a.to_raw();
        let (br, bi) = b.to_raw();
        Elem::<QuadMersenne31>::from_raw(o_add(ar, br), o_add(ai, bi))
    }
    fn model_elem_mul(a: Elem<QuadMersenne31>, b: Elem<QuadMersenne31>) -> Elem<QuadMersenne31> {
        let (ar, ai) = a.to_raw();
        let (br, bi) = b.to_raw();
        Elem::<QuadMersenne31>::from_raw(
            o_sub(o_mul(ar, br), o_mul(ai, bi)),
            o_add(o_mul(ar, bi), o_mul(ai, br)),
        )
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

    /// Whether the host resolves to V4x — the same single-source selection
    /// the shared differential tests use, honoring `SIMD_BACKEND`.
    fn host_v4() -> bool {
        simdispatch::Selection::new("SIMD_BACKEND")
            .supports(&[crate::kernel::Backend::V4x])
            .resolve()
            != crate::kernel::Backend::Scalar
    }

    /// Every entry matches the independent limb model across boundary
    /// lengths, canonical random fixtures, and canonical edge elements in
    /// the source operands.
    #[test]
    fn avx512_matches_independent_model() {
        if !host_v4() {
            eprintln!("skipping: no AVX-512 on this host");
            return;
        }
        let token = archmage::X64V4Token::summon().expect("guard passed: AVX-512 summons here");
        for &len in LENS {
            for (src, base, b2) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                let mut got = vec![0u8; len];
                mul_elementwise_avx512(token, &mut got, &src, &b2);
                assert_eq!(
                    got,
                    model_zip(&src, &b2, model_elem_mul),
                    "mul_elementwise len {len}"
                );

                for &coeff in &COEFFS {
                    let mut got = base.clone();
                    mul_add_avx512(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_zip(&base, &src, |d, s| {
                            model_elem_add(d, model_elem_mul(coeff, s))
                        }),
                        "mul_add len {len} coeff {coeff:?}"
                    );

                    let mut got = vec![0u8; len];
                    mul_into_avx512(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_map(&src, |e| model_elem_mul(coeff, e)),
                        "mul_into len {len} coeff {coeff:?}"
                    );
                }
            }
        }
    }

    /// Every entry agrees with the portable [`prime`] differential oracle
    /// over lengths straddling the vector lane.
    #[test]
    fn avx512_matches_portable_reference() {
        if !host_v4() {
            eprintln!("skipping: no AVX-512 on this host");
            return;
        }
        let token = archmage::X64V4Token::summon().expect("guard passed: AVX-512 summons here");
        // Sub-lane, exact lanes, lane-plus-tail, and a different tail.
        for &len in &[40usize, 64, 72, 136] {
            let src = edges(len);
            let base = canonical(len, 0xb2 + len as u64);
            let b2 = canonical(len, 0xc3 + len as u64);

            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            mul_elementwise_avx512(token, &mut got, &src, &b2);
            prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &b2);
            assert_eq!(got, want, "mul_elementwise vs reference len {len}");

            let mut got = base.clone();
            let mut want = base.clone();
            mul_add_avx512(token, &mut got, FULL_WORK, &src);
            prime::mul_add::<QuadMersenne31>(&mut want, FULL_WORK, &src);
            assert_eq!(got, want, "mul_add vs reference len {len}");

            let mut got = vec![0u8; len];
            let mut want = src.clone();
            mul_into_avx512(token, &mut got, FULL_WORK, &src);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_into vs reference len {len}");
        }
    }

    /// Unaligned but whole-element slices are legal: every access is an
    /// unaligned load or store. Each offset window holds canonical element
    /// bytes copied in; only its address is odd.
    #[test]
    fn avx512_unaligned_slices() {
        if !host_v4() {
            eprintln!("skipping: no AVX-512 on this host");
            return;
        }
        let token = archmage::X64V4Token::summon().expect("guard passed: AVX-512 summons here");
        let len = 136;
        let src_elems = edges(len);
        let dst_elems = canonical(len, 0x82);
        let coeff = Elem::<QuadMersenne31>::from_raw(3, 5);
        for offset in [0usize, 1, 3, 5] {
            let mut src_storage = vec![0u8; len + 8];
            src_storage[offset..offset + len].copy_from_slice(&src_elems);
            let src = &src_storage[offset..offset + len];
            let mut got = vec![0u8; len];
            mul_into_avx512(token, &mut got, coeff, src);
            assert_eq!(
                got,
                model_map(src, |e| model_elem_mul(coeff, e)),
                "unaligned mul_into offset {offset}"
            );

            // Unaligned destination as well as source.
            let mut dst_storage = vec![0u8; len + 8];
            dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
            let dst = &mut dst_storage[offset..offset + len];
            mul_add_avx512(token, dst, coeff, src);
            assert_eq!(
                &dst_storage[offset..offset + len],
                &model_zip(&dst_elems, &src_elems, |d, s| model_elem_add(
                    d,
                    model_elem_mul(coeff, s)
                ))[..],
                "unaligned mul_add offset {offset}"
            );

            let mut got = vec![0u8; len];
            mul_elementwise_avx512(token, &mut got, src, &src_elems);
            assert_eq!(
                got,
                model_zip(&src_elems, &src_elems, model_elem_mul),
                "unaligned mul_elementwise offset {offset}"
            );
        }
    }

    /// The invocation must panic on bad geometry and leave `dst`
    /// unchanged: validation precedes every write.
    fn expect_geometry_panic(label: &str, dst: &mut [u8], invoke: impl FnOnce(&mut [u8])) {
        let before = dst.to_vec();
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invoke(dst))).is_err();
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
        if !host_v4() {
            eprintln!("skipping: no AVX-512 on this host");
            return;
        }
        let token = archmage::X64V4Token::summon().expect("guard passed: AVX-512 summons here");
        let coeff = Elem::<QuadMersenne31>::from_raw(3, 5);

        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("mul_add partial", &mut partial, |d| {
            mul_add_avx512(token, d, coeff, &partial_src);
        });
        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("mul_into partial", &mut partial, |d| {
            mul_into_avx512(token, d, coeff, &partial_src);
        });
        let mut partial = [1u8; 12];
        expect_geometry_panic("mul_elementwise partial", &mut partial, |d| {
            mul_elementwise_avx512(token, d, &partial_src, &partial_src);
        });

        let mut mismatched = [1u8; 16];
        let short_src = [1u8; 8];
        expect_geometry_panic("mul_add length mismatch", &mut mismatched, |d| {
            mul_add_avx512(token, d, coeff, &short_src);
        });
        let mut mismatched = [1u8; 16];
        expect_geometry_panic("mul_into length mismatch", &mut mismatched, |d| {
            mul_into_avx512(token, d, coeff, &short_src);
        });
        let mut mismatched = [1u8; 16];
        let full_src = [1u8; 16];
        expect_geometry_panic("mul_elementwise b mismatch", &mut mismatched, |d| {
            mul_elementwise_avx512(token, d, &full_src, &short_src);
        });
    }
}
