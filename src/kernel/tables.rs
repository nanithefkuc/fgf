//! Precomputed per-coefficient multiplication tables.
//!
//! Two families, matching the two ways SIMD hardware can multiply in a binary
//! field:
//!
//! - [`ScaleTable`] — split-nibble shuffle tables for one GF(2^8) coefficient.
//!   `c * x` is decomposed as `c * (x & 0xf) ^ c * (x & 0xf0)`, each half a
//!   16-entry lookup that `PSHUFB`/`TBL` performs on every lane at once.
//! - [`TowerCoeff`] — the two broadcast words that turn one GF(2^16) tower
//!   multiply into two byte-wide multiplies of the source and its
//!   adjacent-swapped self. No table at all, which is what makes the GFNI and
//!   `PMULL` paths cheap.
//!
//! Each GF(2^8) polynomial carries one nibble bank and one affine bank, built
//! at compile time from that field's scalar multiply and promoted to rodata
//! once per polynomial a program uses; all 256 coefficients cost 8 KiB plus
//! 2 KiB and stay resident in L1/L2. [`ByteBanks`] exposes a
//! representation's banks, AES nativity, AES isomorphism, and reduction byte
//! to the dispatch layer, which resolves coefficients once and hands the
//! kernels plain values. GF(2^16) has 65536 coefficients, so a
//! full bank would be ~9 MiB and thrash cache. A GF(2^16) coefficient is
//! instead resolved per call into its four base-field factors (two base
//! multiplies, [`TowerCoeff::new`]) and, on shuffle backends, four table
//! copies out of the shared bank ([`TowerTables::new`]) — amortized over the
//! whole buffer, or hoisted out entirely with `Coeff`/`CoeffVec`.

use crate::field::fan_paar::{fp8, fp16};
use crate::field::gf16;
use crate::field::poly::{self, AES};
use crate::field::{Elem, Poly};

/// Split-nibble multiplication tables for one GF(2^8) coefficient.
///
/// The shuffle kernels read only `lo` and `hi`, so one type serves every
/// polynomial; the polynomial decides which bank an entry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleTable {
    /// The raw coefficient byte these tables multiply by.
    pub coeff: u8,
    /// `lo[i] = coeff * i` for the low nibble.
    pub lo: [u8; 16],
    /// `hi[i] = coeff * (i << 4)` for the high nibble.
    pub hi: [u8; 16],
}

impl ScaleTable {
    /// Build the nibble tables for `coeff`.
    #[must_use]
    // Loop counters are bounded by the array sizes (16, 256), so every cast
    // below is exact; `const fn` rules out `try_into`.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new<const POLY: u32>(coeff: Elem<8, Poly<POLY>>) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            lo[i] = poly::Poly::<POLY>::mul_xtime(i as u8, coeff.0);
            hi[i] = poly::Poly::<POLY>::mul_xtime((i as u8) << 4, coeff.0);
            i += 1;
        }
        Self {
            coeff: coeff.0,
            lo,
            hi,
        }
    }
}

/// Build the `VGF2P8AFFINEQB` matrix qword that multiplies by `coeff`.
///
/// Multiplication by a fixed coefficient in any GF(2^8) is an 8×8 GF(2)
/// linear map, and `VGF2P8AFFINEQB` applies an arbitrary such map per byte
/// lane — unlike `GF2P8MULB`, which bakes in the AES polynomial. The
/// instruction computes `dst.bit[b] = parity(map.byte[7 - b] & x)` per lane,
/// so the qword for coefficient `c` stores the product map transposed and
/// row-reversed: byte `r`, bit `k` is bit `7 - r` of the column `c * x^k`.
/// Every entry derives from the field's own reference multiply — never from
/// another library's tables — and is validated against a scalar model of
/// the instruction in this module's tests.
#[must_use]
pub const fn build_affine_map<const POLY: u32>(coeff: Elem<8, Poly<POLY>>) -> u64 {
    let mut columns = [0u8; 8];
    let mut k = 0;
    while k < 8 {
        columns[k] = poly::Poly::<POLY>::mul_xtime(coeff.0, 1 << k);
        k += 1;
    }
    linear_map(columns)
}

/// The `VGF2P8AFFINEQB` matrix qword of the GF(2)-linear byte map whose
/// image of `x^k` is `columns[k]`.
const fn linear_map(columns: [u8; 8]) -> u64 {
    let mut map = 0u64;
    // Row `r` of the instruction's matrix carries, across bits `k`, bit
    // `7 - r` of every column.
    let mut r = 0;
    while r < 8 {
        let mut row = 0u8;
        let mut k = 0;
        while k < 8 {
            row |= ((columns[k] >> (7 - r)) & 1) << k;
            k += 1;
        }
        map |= (row as u64) << (8 * r);
        r += 1;
    }
    map
}

/// The per-representation table banks, one entry per coefficient.
struct Bank<const POLY: u32>;

impl<const POLY: u32> Bank<POLY> {
    /// The nibble tables of every coefficient, 8 KiB.
    ///
    /// Promoted `&'static` items rather than `const` arrays: a `const` array
    /// would be materialized onto the stack at every use site instead of
    /// being borrowed from rodata.
    const SCALE: &'static [ScaleTable; 256] = &build_bank::<POLY>();

    /// The `VGF2P8AFFINEQB` matrix of every coefficient, 2 KiB.
    const AFFINE: &'static [u64; 256] = &build_affine_bank::<POLY>();
}

/// What the GF(2^8) kernels need from a byte representation, as constants
/// and `&'static` data — never as per-call computation.
///
/// The dispatch layer reads this facet once per coefficient and hands the
/// architecture kernels plain values (tables, map qwords, isomorphism
/// qwords, reduction bytes), so no hot kernel body depends on the
/// representation type. Implemented by every [`ByteRepr`] alongside that
/// trait, so the pair cannot drift; in this crate only [`Poly`] carries it.
pub(crate) trait ByteBanks: crate::field::ByteRepr {
    /// Whether this representation's encoding is the GF(2)[x]/0x11B
    /// polynomial basis, so `GF2P8MULB` multiplies in it natively.
    const AES_NATIVE: bool;
    /// The nibble tables of every coefficient, in this representation's
    /// encoding.
    const SCALE: &'static [ScaleTable; 256];
    /// The `VGF2P8AFFINEQB` matrix of every coefficient.
    const AFFINE: &'static [u64; 256];
    /// The isomorphism onto the AES encoding: the identity when
    /// [`AES_NATIVE`](ByteBanks::AES_NATIVE).
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    const ISOMORPHISM: Isomorphism;
    /// The reduction byte the shift-and-reduce elementwise kernels consume:
    /// the low byte of the reference polynomial.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    const REDUCTION_LOW: u8;
}

impl<const POLY: u32> ByteBanks for Poly<POLY> {
    const AES_NATIVE: bool = POLY == AES;
    const SCALE: &'static [ScaleTable; 256] = Bank::<POLY>::SCALE;
    const AFFINE: &'static [u64; 256] = Bank::<POLY>::AFFINE;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    const ISOMORPHISM: Isomorphism = isomorphism::<POLY>();
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    const REDUCTION_LOW: u8 = Poly::<POLY>::REDUCTION_LOW;
}

#[allow(clippy::cast_possible_truncation)]
const fn build_bank<const POLY: u32>() -> [ScaleTable; 256] {
    let mut bank = [ScaleTable::new::<POLY>(Elem::<8, Poly<POLY>>::ZERO); 256];
    let mut i = 0;
    while i < 256 {
        bank[i] = ScaleTable::new::<POLY>(Elem::<8, Poly<POLY>>::from_raw(i as u8));
        i += 1;
    }
    bank
}

#[allow(clippy::cast_possible_truncation)]
const fn build_affine_bank<const POLY: u32>() -> [u64; 256] {
    let mut bank = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        bank[i] = build_affine_map::<POLY>(Elem::<8, Poly<POLY>>::from_raw(i as u8));
        i += 1;
    }
    bank
}

/// The zero coefficient's nibble tables: an all-zero bank entry.
///
/// The zero prepared form's placeholder table; zero coefficients are skipped
/// before any table is read, so its contents matter only in that multiplying
/// by it yields zero.
#[allow(dead_code)]
pub(crate) static ZERO_TABLE: ScaleTable = ScaleTable {
    coeff: 0,
    lo: [0; 16],
    hi: [0; 16],
};

/// Return the shared nibble tables for a GF(2^8) coefficient, from its
/// polynomial's bank.
#[inline]
#[must_use]
pub fn scale_table<const POLY: u32>(coeff: Elem<8, Poly<POLY>>) -> &'static ScaleTable {
    &Bank::<POLY>::SCALE[coeff.0 as usize]
}

/// Return the `VGF2P8AFFINEQB` matrix qword that multiplies by `coeff`, from
/// its polynomial's bank.
#[allow(dead_code)]
#[inline]
#[must_use]
pub fn affine_map<const POLY: u32>(coeff: Elem<8, Poly<POLY>>) -> u64 {
    Bank::<POLY>::AFFINE[coeff.0 as usize]
}

/// A field isomorphism `φ: Gf8<Poly<POLY>> → Gf8<Poly<AES>>` and its inverse,
/// as `VGF2P8AFFINEQB` matrix qwords.
///
/// Conjugating `GF2P8MULB` by it multiplies two varying vectors under any
/// polynomial: `a · b = φ⁻¹(GF2P8MULB(φa, φb))`. Under [`AES`] both maps are
/// the identity.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Isomorphism {
    /// `φ`, into the AES field.
    pub forward: u64,
    /// `φ⁻¹`, back from the AES field.
    pub inverse: u64,
}

/// Return the isomorphism between `Gf8<Poly<POLY>>` and `Gf8<Poly<AES>>`.
#[allow(dead_code)]
#[inline]
#[must_use]
pub const fn isomorphism_to_aes<const POLY: u32>() -> Isomorphism {
    isomorphism::<POLY>()
}

/// Derive the isomorphism onto the AES field.
///
/// `φ(x) = r` for the smallest `r` in GF(2^8)/`0x11B` that is a root of
/// `POLY`; the images of the powers of `x` are the powers of `r`.
/// The inverse columns are the preimages of the AES polynomial basis under
/// that selected map; a separately chosen root need not give its inverse.
#[allow(dead_code)]
const fn isomorphism<const POLY: u32>() -> Isomorphism {
    let to_aes = root_powers::<AES>(poly::Poly::<POLY>::POLY);
    let from_aes = inverse_columns(to_aes);
    Isomorphism {
        forward: linear_map(to_aes),
        inverse: linear_map(from_aes),
    }
}

/// Invert a bijective byte-linear map by finding each basis vector's preimage.
#[allow(dead_code)]
const fn inverse_columns(columns: [u8; 8]) -> [u8; 8] {
    let mut inverse = [0u8; 8];
    let mut basis = 0;
    while basis < 8 {
        let mut candidate = 0u16;
        loop {
            assert!(candidate < 256, "a field isomorphism is bijective");
            let value = candidate.to_le_bytes()[0];
            let mut image = 0u8;
            let mut bit = 0;
            while bit < 8 {
                if (value >> bit) & 1 != 0 {
                    image ^= columns[bit];
                }
                bit += 1;
            }
            if image == 1 << basis {
                inverse[basis] = value;
                break;
            }
            candidate += 1;
        }
        basis += 1;
    }
    inverse
}

/// The powers `r^0..r^7` of the smallest root `r` of `poly` in
/// `Gf8<Poly<FIELD>>`.
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation)]
const fn root_powers<const FIELD: u32>(poly: u32) -> [u8; 8] {
    let mut candidate = 2u32;
    while candidate < 256 {
        let r = candidate.to_le_bytes()[0];
        // Horner evaluation of `poly` at `r`.
        let mut value = 0u8;
        let mut bit = 8;
        loop {
            value = poly::Poly::<FIELD>::mul_xtime(value, r);
            if (poly >> bit) & 1 == 1 {
                value ^= 1;
            }
            if bit == 0 {
                break;
            }
            bit -= 1;
        }
        if value == 0 {
            let mut powers = [0u8; 8];
            let mut power = 1u8;
            let mut k = 0;
            while k < 8 {
                powers[k] = power;
                power = poly::Poly::<FIELD>::mul_xtime(power, r);
                k += 1;
            }
            return powers;
        }
        candidate += 1;
    }
    panic!("an irreducible degree-8 polynomial splits in every GF(2^8)")
}

/// Broadcast factors that express one GF(2^16) tower multiply as two
/// byte-wide GF(2^8) multiplies.
///
/// Interleaved source bytes are `[a, b]` meaning `a + b*u`. Multiplying by
/// `c0 + c1*u` gives
///
/// ```text
/// (c0*a + DELTA*c1*b) + (c1*a + (c0 + c1)*b) * u.
/// ```
///
/// The first term of each component multiplies the source in place; the
/// second multiplies the source with adjacent bytes swapped. So a single
/// alternating-coefficient byte multiply of `src` by [`TowerCoeff::same`],
/// `XORed` with the same of `swap16(src)` by [`TowerCoeff::cross`], produces
/// both components with no planar de-interleave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerCoeff {
    /// The GF(2^16) coefficient.
    pub coeff: gf16::Elem,
    /// Little-endian `[c0, c0 + c1]`, applied to the source directly.
    pub same: u16,
    /// Little-endian `[DELTA*c1, c1]`, applied to the byte-swapped source.
    pub cross: u16,
}

impl TowerCoeff {
    /// Derive the broadcast factors for `coeff`. Two base multiplies.
    #[inline]
    #[must_use]
    pub const fn new(coeff: gf16::Elem) -> Self {
        let (c0, c1) = coeff.to_components();
        let same = u16::from_le_bytes([c0.0, c0.add(c1).0]);
        let cross = u16::from_le_bytes([gf16::DELTA.mul(c1).0, c1.0]);
        Self { coeff, same, cross }
    }

    /// The four base-field coefficients in `[same.0, same.1, cross.0,
    /// cross.1]` order, i.e. `[c0, c0+c1, DELTA*c1, c1]`.
    #[inline]
    #[must_use]
    pub const fn factors(self) -> [Elem<8, Poly<AES>>; 4] {
        let [s0, s1] = self.same.to_le_bytes();
        let [x0, x1] = self.cross.to_le_bytes();
        [
            Elem::<8, Poly<AES>>::from_raw(s0),
            Elem::<8, Poly<AES>>::from_raw(s1),
            Elem::<8, Poly<AES>>::from_raw(x0),
            Elem::<8, Poly<AES>>::from_raw(x1),
        ]
    }
}

/// Nibble tables for the four base-field factors of a GF(2^16) coefficient.
///
/// Used by the shuffle backends (AVX2, SSSE3, NEON), which have no byte-wide
/// field multiply instruction and must emulate one per factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerTables {
    /// The GF(2^16) coefficient.
    pub coeff: gf16::Elem,
    /// Tables for `[c0, c0+c1, DELTA*c1, c1]`. Entries 0 and 1 apply to the
    /// source's even and odd byte lanes; entries 2 and 3 apply to the
    /// adjacent-swapped source's even and odd lanes.
    pub factors: [ScaleTable; 4],
}

impl TowerTables {
    /// Build the four nibble tables for `coeff`.
    #[inline]
    #[must_use]
    pub fn new(coeff: gf16::Elem) -> Self {
        let f = TowerCoeff::new(coeff).factors();
        Self {
            coeff,
            factors: [
                *scale_table(f[0]),
                *scale_table(f[1]),
                *scale_table(f[2]),
                *scale_table(f[3]),
            ],
        }
    }
}

/// Split-nibble multiplication tables for one Fan–Paar `fp8` coefficient.
///
/// The canonical Fan–Paar byte field is *not* the AES field, so `GF2P8MULB`
/// cannot multiply its bytes; every backend emulates the byte product with a
/// 16-entry nibble lookup, exactly as [`ScaleTable`] does for GF(2^8). The
/// two share a shape and a 256-entry static bank, but the multiplication
/// used to *fill* them differs (fp8 tower vs. AES reduction).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpScaleTable {
    /// `lo[i] = coeff * i` for the low nibble.
    pub lo: [u8; 16],
    /// `hi[i] = coeff * (i << 4)` for the high nibble.
    pub hi: [u8; 16],
}

impl FpScaleTable {
    /// Build the nibble tables for `coeff` in the Fan–Paar byte field.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(dead_code)]
    pub const fn new(coeff: fp8::Elem) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            lo[i] = fp8::Elem(i as u8).mul(coeff).0;
            hi[i] = fp8::Elem((i as u8) << 4).mul(coeff).0;
            i += 1;
        }
        Self { lo, hi }
    }
}

/// Shared nibble-table bank, one entry per Fan–Paar `fp8` coefficient.
///
/// A separate bank from [`SCALE_TABLE_BANK`]: fp8 is a different field, so its
/// products are different bytes. 8 KiB, resident in L1/L2, touched only by
/// Fan–Paar kernels.
#[allow(dead_code)]
static FP_SCALE_TABLE_BANK: [FpScaleTable; 256] = build_fp_bank();

#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
const fn build_fp_bank() -> [FpScaleTable; 256] {
    let mut bank = [FpScaleTable::new(fp8::Elem(0)); 256];
    let mut i = 0;
    while i < 256 {
        bank[i] = FpScaleTable::new(fp8::Elem(i as u8));
        i += 1;
    }
    bank
}

/// Return the shared fp8 nibble tables for a Fan–Paar byte coefficient.
#[inline]
#[must_use]
#[allow(dead_code)]
pub fn fp_scale_table(coeff: fp8::Elem) -> &'static FpScaleTable {
    &FP_SCALE_TABLE_BANK[coeff.0 as usize]
}

/// Four nibble-table factors a shuffle kernel consumes lane-by-lane.
///
/// Both GF(2^16) over GF(2^8) and Fan–Paar GF(2^16) over `fp8` reduce a
/// 16-bit fixed-coefficient scale to four byte-wide multiplies — factors 0/1
/// on the source's even/odd bytes, factors 2/3 on the adjacent-swapped
/// source's. The multiply core in `kernel::x86::gf16` reads only the nibble
/// tables, not the field they encode, so this trait lets the two fields share
/// that core. Entries 0 and 1 apply to the source's even and odd byte lanes;
/// 2 and 3 to the swapped source's.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) trait NibbleFactors {
    /// The low-nibble table of factor `i` (`0 <= i < 4`).
    fn lo(&self, i: usize) -> &[u8; 16];
    /// The high-nibble table of factor `i` (`0 <= i < 4`).
    fn hi(&self, i: usize) -> &[u8; 16];
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl NibbleFactors for TowerTables {
    #[inline]
    fn lo(&self, i: usize) -> &[u8; 16] {
        &self.factors[i].lo
    }
    #[inline]
    fn hi(&self, i: usize) -> &[u8; 16] {
        &self.factors[i].hi
    }
}

/// Nibble tables for the four `fp8` factors of a Fan–Paar GF(2^16)
/// coefficient.
///
/// The Fan–Paar tower relation is `X² + alpha·X + 1 = 0` with `alpha` the
/// `fp8` tower generator, so for `c = c0 + c1·X` and `x = x0 + x1·X`:
///
/// ```text
/// r0 = c0·x0 ^ c1·x1
/// r1 = (c0 ^ alpha·c1)·x1 ^ c1·x0
/// ```
///
/// `alpha·(c1·x1) = (alpha·c1)·x1` because `alpha`, `c1`, `x1` all lie in the
/// `fp8` subfield and the subfield commutes — so the `mul_alpha` fold lands in
/// coefficient preparation, not in the kernel. The four factors are therefore
/// `[c0, c0 ^ alpha·c1, c1, c1]`: 0/1 scale the source, 2/3 (both `c1`) the
/// swapped source. This is the same four-table shape as [`TowerTables`], which
/// is why the shuffle multiply core is shared.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpTowerTables {
    /// The Fan–Paar GF(2^16) coefficient, recovered for the portable scalar
    /// tail of a vector kernel.
    pub coeff: fp16::Elem,
    /// Tables for `[c0, c0 ^ alpha·c1, c1, c1]`.
    pub factors: [FpScaleTable; 4],
}

impl FpTowerTables {
    /// Build the four `fp8` nibble tables for `coeff`.
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub fn new(coeff: fp16::Elem) -> Self {
        let (c0, c1) = coeff.to_components();
        // `c1.mul_alpha()` is `alpha·c1` in the fp8 subfield.
        let b = c0.add(c1.mul_alpha());
        let [f0, f1, f2, f3] = [c0, b, c1, c1];
        Self {
            coeff,
            factors: [
                *fp_scale_table(f0),
                *fp_scale_table(f1),
                *fp_scale_table(f2),
                *fp_scale_table(f3),
            ],
        }
    }
}
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl NibbleFactors for FpTowerTables {
    #[inline]
    fn lo(&self, i: usize) -> &[u8; 16] {
        &self.factors[i].lo
    }
    #[inline]
    fn hi(&self, i: usize) -> &[u8; 16] {
        &self.factors[i].hi
    }
}

/// Period-2 alternating subfield coefficients for a two-level tower multiply.
///
/// Generalizes [`TowerCoeff`] one field level up. For a level-2 element
/// `c = c0 + c1·u` over `u² + u + DELTA`, the level-2 multiply is
///
/// ```text
/// c·x = lane_mul(x,      same)  ^  lane_mul(swap(x), cross)
/// ```
///
/// where `lane_mul` multiplies subfield elements under alternating `same`/
/// `cross` coefficients and `swap` exchanges the two subfield halves of each
/// element. `same = [c0, c0 + c1]` applies to the source's even and odd
/// subfield lanes; `cross = [DELTA·c1, c1]` to the half-swapped source. This
/// is exactly the identity the GF(2^16) kernel builds at byte granularity
/// via [`TowerCoeff`], one level up.
///
/// The form carries subfield elements, not byte broadcasts, so it is
/// representation-agnostic: GF(2^32) builds it over `gf16::Elem` and GF(2^64)
/// over `gf32::Elem`, each with its own `DELTA`.
/// This is a representation-agnostic seam: every level-2 SIMD tower — the
/// GFNI x86 path today, the aarch64/wasm paths to come — derives it from the
/// subfield element, so it is dead weight only on targets with no such
/// kernel.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tower2Coeff<E> {
    /// `[c0, c0 + c1]`, applied to the even and odd subfield lanes of the
    /// source.
    pub same: [E; 2],
    /// `[DELTA·c1, c1]`, applied to the even and odd lanes of the half-swapped
    /// source.
    pub cross: [E; 2],
}

impl<E: crate::field::FieldElem> Tower2Coeff<E> {
    /// Derive the alternating subfield coefficient pair for `c0 + c1·u` over
    /// `u² + u + delta`.
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub fn derive(c0: E, c1: E, delta: E) -> Self {
        Self {
            same: [c0, c0.add(c1)],
            cross: [delta.mul(c1), c1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::poly::REED_SOLOMON;

    type E8 = Elem<8, Poly<AES>>;

    fn check_bank<const P: u32>() {
        let runtime_scale = build_bank::<P>();
        let runtime_affine = build_affine_bank::<P>();
        for raw in 0..=u8::MAX {
            let coeff = Elem::<8, Poly<P>>::from_raw(raw);
            for x in 0..=u8::MAX {
                let product = Elem::<8, Poly<P>>::from_raw(x).mul(coeff).0;
                for table in [scale_table::<P>(coeff), &runtime_scale[usize::from(raw)]] {
                    let split = table.lo[(x & 0x0f) as usize] ^ table.hi[(x >> 4) as usize];
                    assert_eq!(split, product, "{coeff:?} nibbles at {x:#04x}");
                }
                for map in [affine_map::<P>(coeff), runtime_affine[usize::from(raw)]] {
                    assert_eq!(apply(map, x), product, "{coeff:?} affine at {x:#04x}");
                }
            }
        }
        for iso in [isomorphism_to_aes::<P>(), isomorphism::<P>()] {
            for a in 0..=u8::MAX {
                assert_eq!(
                    apply(iso.inverse, apply(iso.forward, a)),
                    a,
                    "φ⁻¹φ at {a:#04x}"
                );
                for b in 0..=u8::MAX {
                    let product = Elem::<8, Poly<P>>::from_raw(a)
                        .mul(Elem::<8, Poly<P>>::from_raw(b))
                        .0;
                    let conjugated = apply(
                        iso.inverse,
                        E8::from_raw(apply(iso.forward, a))
                            .mul(E8::from_raw(apply(iso.forward, b)))
                            .0,
                    );
                    assert_eq!(conjugated, product, "φ-conjugated {a:#04x} * {b:#04x}");
                }
            }
        }
    }

    /// The semantics `VGF2P8AFFINEQB` documents, modeled in scalar code:
    /// `dst.bit[b] = parity(map.byte[7 - b] & x)`.
    #[allow(clippy::cast_possible_truncation)]
    fn apply(map: u64, x: u8) -> u8 {
        let mut out = 0u8;
        for b in 0..8u8 {
            let row = (map >> (8 * (7 - b))) as u8;
            let parity = (row & x).count_ones() & 1;
            out |= u8::try_from(parity).unwrap() << b;
        }
        out
    }

    #[test]
    fn banks_reproduce_every_product_under_the_instruction_model() {
        check_bank::<AES>();
        check_bank::<REED_SOLOMON>();
        check_bank::<0x12D>();
        check_bank::<0x187>();
    }

    #[test]
    fn fan_paar_nibble_tables_match_the_wiedemann_recurrence() {
        let runtime = build_fp_bank();
        for bank in [&runtime, &FP_SCALE_TABLE_BANK] {
            for c in 0..=u8::MAX {
                let table = &bank[usize::from(c)];
                for x in 0..=u8::MAX {
                    let product = crate::field::wiedemann::multiply(u64::from(c), u64::from(x), 8);
                    let split = table.lo[usize::from(x & 0x0f)] ^ table.hi[usize::from(x >> 4)];
                    assert_eq!(u64::from(split), product, "{c:#04x} * {x:#04x}");
                }
            }
        }
    }

    #[test]
    fn the_aes_isomorphism_is_the_identity() {
        let iso = isomorphism_to_aes::<AES>();
        for x in 0..=u8::MAX {
            assert_eq!(apply(iso.forward, x), x);
            assert_eq!(apply(iso.inverse, x), x);
        }
    }

    #[test]
    fn the_reed_solomon_isomorphism_is_the_frozen_involution() {
        // The kernels' historical `0x11D` conjugation constant: the map
        // derived from the smallest root is this involution.
        let iso = isomorphism_to_aes::<REED_SOLOMON>();
        assert_eq!(iso.forward, 0xffaa_cc88_f0a0_c080);
        assert_eq!(iso.inverse, iso.forward);
    }

    #[test]
    fn tower_factors_reconstruct_the_product() {
        // Exercise the identity the SIMD kernels rely on, scalar-side.
        for coeff in [0u16, 1, 0x0108, 0x2000, 0xffff, 0x1234] {
            let coeff = gf16::Elem(coeff);
            let tc = TowerCoeff::new(coeff);
            let [f0, f1, f2, f3] = tc.factors();
            for value in [0u16, 1, 0x00ff, 0xff00, 0xbeef] {
                let value = gf16::Elem(value);
                let [a, b] = value.to_bytes();
                let (a, b) = (E8::from_raw(a), E8::from_raw(b));
                // even lane: same.0 * a  ^  cross.0 * b   (b is a's swap partner)
                // odd  lane: same.1 * b  ^  cross.1 * a
                let even = f0.mul(a).add(f2.mul(b));
                let odd = f1.mul(b).add(f3.mul(a));
                let got = gf16::Elem::from_bytes([even.0, odd.0]);
                assert_eq!(got, value.mul(coeff), "{value:?} * {coeff:?}");
            }
        }
    }
}
