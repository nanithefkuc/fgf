//! Byte-lane single-row AVX2 kernels for the generic residue rows.
//!
//! Fixed-coefficient products over one-byte lanes at `P <= 127` through
//! two `vpshufb` nibble tables combined by modular sum: the 32-byte
//! vector body processes 32 raw bytes per iteration and tails run the
//! scalar table form. Entries are safe capability-token functions taking
//! an `X64V3Token` and validating their own geometry.

use super::src_tail_avx2;

// ---------------------------------------------------------------------------
// Byte fixed-coefficient (nibble-table lookups)
// ---------------------------------------------------------------------------

/// Nibble tables for a byte coefficient: `lo[i] = c*i mod P` and
/// `hi[i] = c*16*i mod P` for `i < 16`, each 16-byte table replicated
/// into both 128-bit halves of the 32-byte array.
#[must_use]
fn tables(c: u8, p: u8) -> ([u8; 32], [u8; 32]) {
    let mut lo = [0u8; 32];
    let mut hi = [0u8; 32];
    let wide = u32::from(p);
    for i in 0..16u32 {
        // Products are below `127 * 16 * 15 < 2^16`; residues below `P <= 127` fit `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let l = (u32::from(c) * i % wide) as u8;
        #[allow(clippy::cast_possible_truncation)]
        let h = (u32::from(c) * 16 * i % wide) as u8;
        lo[i as usize] = l;
        lo[i as usize + 16] = l;
        hi[i as usize] = h;
        hi[i as usize + 16] = h;
    }
    (lo, hi)
}

/// `dst[i] = c * src[i] mod P` over all raw bytes through two `vpshufb`
/// nibble tables combined by modular sum, `P <= 127`.
///
/// For moduli `P <= 127` the product `c * x` over a raw byte splits into
/// two nibble lookups: `c*(x mod 16) + c*16*(x div 16)`, each table holding
/// values below `P` and their sum below `2P`, so one unsigned-min
/// correction canonicalizes. Both 16-entry tables are built on the stack
/// per call; the vector body shuffles both nibbles of 32 raw bytes at
/// once; tails run the scalar table form.
///
/// # Panics
///
/// Panics unless `dst` and `src` have equal lengths, or when `p` exceeds
/// 127.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_u8_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8], c: u8, p: u8) {
    assert!(p <= 127, "residue byte: modulus {p} exceeds 127");
    assert_eq!(
        dst.len(),
        src.len(),
        "residue::mul_into_u8_avx2: dst length differs from src"
    );
    let (lo, hi) = tables(c, p);
    let lo_v = _mm256_loadu_si256(&lo);
    let hi_v = _mm256_loadu_si256(&hi);
    let mask0f = _mm256_set1_epi8(0x0F);
    // `p <= 127` fits `i8` exactly; the cast only reinterprets the bits.
    #[allow(clippy::cast_possible_wrap)]
    let cp = _mm256_set1_epi8(p as i8);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, _) = src.as_chunks::<32>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let v = _mm256_loadu_si256(s);
        let rlo = _mm256_shuffle_epi8(lo_v, _mm256_and_si256(v, mask0f));
        let rhi = _mm256_shuffle_epi8(hi_v, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask0f));
        let sum = _mm256_add_epi8(rlo, rhi);
        let sub = _mm256_sub_epi8(sum, cp);
        _mm256_storeu_si256(d, _mm256_min_epu8(sum, sub));
    }
    for (d, s) in dst_tail.iter_mut().zip(src_tail_avx2(src)) {
        let t = lo[*s as usize & 15].wrapping_add(hi[*s as usize >> 4]);
        *d = if t >= p { t - p } else { t };
    }
}

/// `dst[i] += c * src[i] mod P` through the nibble tables, `P <= 127`.
///
/// The raw destination byte reduces through an 8-bit Barrett step
/// (`q = (x * floor(256/P)) >> 8` undershoots `floor(x / P)` by at most
/// one, so `r = x - q*P` sits below `2P` and one unsigned-min correction
/// canonicalizes it) and joins the canonical product below `2P`; the
/// output is canonical.
///
/// # Panics
///
/// Panics unless `dst` and `src` have equal lengths, or when `p` exceeds
/// 127.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_u8_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8], c: u8, p: u8) {
    assert!(p <= 127, "residue byte: modulus {p} exceeds 127");
    assert_eq!(
        dst.len(),
        src.len(),
        "residue::mul_add_u8_avx2: dst length differs from src"
    );
    let (lo, hi) = tables(c, p);
    let lo_v = _mm256_loadu_si256(&lo);
    let hi_v = _mm256_loadu_si256(&hi);
    let mask0f = _mm256_set1_epi8(0x0F);
    // `p <= 127` fits `i8` exactly; the cast only reinterprets the bits.
    #[allow(clippy::cast_possible_wrap)]
    let cp = _mm256_set1_epi8(p as i8);
    // `256 / P <= 85` for `P >= 3`: both casts are exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let mu = _mm256_set1_epi16(((256u32 / u32::from(p)) as u16) as i16);
    let cp16 = _mm256_set1_epi16(i16::from(p));
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, _) = src.as_chunks::<32>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let acc = _mm256_loadu_si256(&*d);
        let acc_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<0>(acc));
        let acc_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(acc));
        let normed = {
            let q_lo = _mm256_srli_epi16::<8>(_mm256_mullo_epi16(acc_lo, mu));
            let q_hi = _mm256_srli_epi16::<8>(_mm256_mullo_epi16(acc_hi, mu));
            let r_lo = _mm256_sub_epi16(acc_lo, _mm256_mullo_epi16(q_lo, cp16));
            let r_hi = _mm256_sub_epi16(acc_hi, _mm256_mullo_epi16(q_hi, cp16));
            let c_lo = _mm256_min_epu16(r_lo, _mm256_sub_epi16(r_lo, cp16));
            let c_hi = _mm256_min_epu16(r_hi, _mm256_sub_epi16(r_hi, cp16));
            // `packus` packs within 128-bit lanes, leaving the middle
            // 8-byte groups swapped; the 64-bit permute restores order.
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(c_lo, c_hi))
        };
        let v = _mm256_loadu_si256(s);
        let rlo = _mm256_shuffle_epi8(lo_v, _mm256_and_si256(v, mask0f));
        let rhi = _mm256_shuffle_epi8(hi_v, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask0f));
        let prod = _mm256_min_epu8(
            _mm256_add_epi8(rlo, rhi),
            _mm256_sub_epi8(_mm256_add_epi8(rlo, rhi), cp),
        );
        let sum = _mm256_add_epi8(normed, prod);
        _mm256_storeu_si256(d, _mm256_min_epu8(sum, _mm256_sub_epi8(sum, cp)));
    }
    for (d, s) in dst_tail.iter_mut().zip(src_tail_avx2(src)) {
        let acc = *d % p;
        let t = lo[*s as usize & 15].wrapping_add(hi[*s as usize >> 4]);
        let prod = if t >= p { t - p } else { t };
        let sum = acc + prod;
        *d = if sum >= p { sum - p } else { sum };
    }
}

/// `dst[i] = dst[i] * c mod P` through the nibble tables, `P <= 127`.
///
/// Arbitrary raw bytes reduce in place; the output is canonical.
///
/// # Panics
///
/// Panics when `p` exceeds 127, or if `dst` holds a partial byte lane
/// (unreachable: every byte is a lane).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_u8_avx2(_token: archmage::X64V3Token, dst: &mut [u8], c: u8, p: u8) {
    assert!(p <= 127, "residue byte: modulus {p} exceeds 127");
    let (lo, hi) = tables(c, p);
    let lo_v = _mm256_loadu_si256(&lo);
    let hi_v = _mm256_loadu_si256(&hi);
    let mask0f = _mm256_set1_epi8(0x0F);
    // `p <= 127` fits `i8` exactly; the cast only reinterprets the bits.
    #[allow(clippy::cast_possible_wrap)]
    let cp = _mm256_set1_epi8(p as i8);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for d in dst_lanes.iter_mut() {
        let v = _mm256_loadu_si256(&*d);
        let rlo = _mm256_shuffle_epi8(lo_v, _mm256_and_si256(v, mask0f));
        let rhi = _mm256_shuffle_epi8(hi_v, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask0f));
        let sum = _mm256_add_epi8(rlo, rhi);
        _mm256_storeu_si256(d, _mm256_min_epu8(sum, _mm256_sub_epi8(sum, cp)));
    }
    for d in dst_tail.iter_mut() {
        let t = lo[*d as usize & 15].wrapping_add(hi[*d as usize >> 4]);
        *d = if t >= p { t - p } else { t };
    }
}
