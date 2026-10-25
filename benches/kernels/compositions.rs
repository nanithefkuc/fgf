//! Benchmarks checked row compositions and component-base arithmetic.
//!
//! Coefficient preparation remains inside one-shot timings and outside prepared
//! applications. Element arithmetic validates each public operation before its
//! timing; complete destination slices cross the observability barrier.

use std::hint::black_box;

use fgf::{
    AES, Components, Elem, Extension, ExtensionSpec, FieldBuffer, FieldElem, FiniteField,
    Goldilocks, Mersenne31, Normal, Polynomial, QuadMersenne31, Relation, backend_for, ops,
};

use super::noise;

/// A quadratic component row defined by `t² + 2` over Mersenne31.
struct M31QuadSpec;
impl ExtensionSpec<2> for M31QuadSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "M31² / t²+2";
    const RELATION: Relation = Relation::from_coordinates(&[2_u64, 0]);
}
type M31Quad = FiniteField<0x7FFF_FFFF, 2, Extension<M31QuadSpec, 2>>;

/// A quadratic component row defined by `t² + 7` over Goldilocks.
struct GldQuadSpec;
impl ExtensionSpec<2> for GldQuadSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "Goldilocks² / t²+7";
    const RELATION: Relation = Relation::from_coordinates(&[7_u64, 0]);
}
type GldQuad = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GldQuadSpec, 2>>;

/// GF(2¹⁶) over AES GF(2⁸) with t² + t + 0x2C.
type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
struct Aes8QuadSpec;
impl ExtensionSpec<2> for Aes8QuadSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2¹⁶) / AES8 / t²+t+0x2C";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x2C, 1]);
}
type Aes8Quad = FiniteField<2, 16, Extension<Aes8QuadSpec, 2>>;

/// GF(2¹⁶) over normal-basis AES GF(2⁸) with t² + t + 1.
type Aes8Normal = FiniteField<2, 8, Normal<AES, 0x20>>;
struct Aes8NormalQuadSpec;
impl ExtensionSpec<2> for Aes8NormalQuadSpec {
    type Base = Aes8Normal;
    type Storage = Components;
    const NAME: &'static str = "GF(2¹⁶) / normal AES8 / t²+t+1";
    const RELATION: Relation = Relation::from_binary_words::<Aes8Normal>(&[1, 1]);
}
type Aes8NormalQuad = FiniteField<2, 16, Extension<Aes8NormalQuadSpec, 2>>;

/// GF(9) as t² + 1 over GF(3) — the base layer for the nested extension.
struct Gf9BaseSpec;
impl ExtensionSpec<2> for Gf9BaseSpec {
    type Base = FiniteField<3, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(9) / t²+1";
    const RELATION: Relation = Relation::from_coordinates(&[1_u64, 0]);
}
type Gf9 = FiniteField<3, 2, Extension<Gf9BaseSpec, 2>>;

/// A nested component row defined by `t² - (1+u)` over GF(9).
struct Gf81Spec;
impl ExtensionSpec<2> for Gf81Spec {
    type Base = Gf9;
    type Storage = Components;
    const NAME: &'static str = "GF(81) / GF(9) / t²-(1+u)";
    const RELATION: Relation = Relation::from_coordinates(&[2_u64, 2, 0, 0]);
}
type Gf81 = FiniteField<3, 4, Extension<Gf81Spec, 2>>;

/// The largest `u64` prime.
const MAX_PRIME: u64 = 18_446_744_073_709_551_557;

/// A quadratic component row defined by `t² + 2` over the largest u64 prime.
struct MaxPrimeQuadSpec;
impl ExtensionSpec<2> for MaxPrimeQuadSpec {
    type Base = FiniteField<MAX_PRIME, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(p²) / t²+2 maxprime";
    const RELATION: Relation = Relation::from_coordinates(&[2_u64, 0]);
}
type MaxPrimeQuad = FiniteField<MAX_PRIME, 2, Extension<MaxPrimeQuadSpec, 2>>;

struct MersenneCubeSpec;
impl ExtensionSpec<3> for MersenneCubeSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "Mersenne31 cubic components";
    const RELATION: Relation = Relation::from_coordinates(&[0x7fff_ffff - 5, 0, 0]);
}
type MersenneCube = FiniteField<0x7fff_ffff, 3, Extension<MersenneCubeSpec, 3>>;

struct OverQuadSpec;
impl ExtensionSpec<2> for OverQuadSpec {
    type Base = QuadMersenne31;
    type Storage = Components;
    const NAME: &'static str = "components over QuadMersenne31";
    const RELATION: Relation =
        Relation::from_coordinates(&[0x7fff_ffff - 2, 0x7fff_ffff - 1, 0, 0]);
}
type OverQuad = FiniteField<0x7fff_ffff, 4, Extension<OverQuadSpec, 2>>;

struct PackedBaseSpec;
impl ExtensionSpec<2> for PackedBaseSpec {
    type Base = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;
    type Storage = Components;
    const NAME: &'static str = "components over packed GF(2^16)";
    const RELATION: Relation = <fgf::Rijndael32 as ExtensionSpec<2>>::RELATION;
}
type PackedBase = FiniteField<2, 32, Extension<PackedBaseSpec, 2>>;

macro_rules! residue_components {
    ($spec:ident, $row:ident, $p:literal, $degree:literal, [$($coefficient:expr),+]) => {
        struct $spec;
        impl ExtensionSpec<$degree> for $spec {
            type Base = FiniteField<$p, 1>;
            type Storage = Components;
            const NAME: &'static str = stringify!($row);
            const RELATION: Relation = Relation::from_coordinates(&[$($coefficient),+]);
        }
        type $row = FiniteField<$p, $degree, Extension<$spec, $degree>>;
    };
}

residue_components!(Residue3PairSpec, Residue3Pair, 3, 2, [2, 2]);
residue_components!(Residue131PairSpec, Residue131Pair, 131, 2, [5, 0]);
residue_components!(Residue257PairSpec, Residue257Pair, 257, 2, [3, 0]);
residue_components!(Residue65537PairSpec, Residue65537Pair, 65537, 2, [3, 0]);
residue_components!(Residue7CubeSpec, Residue7Cube, 7, 3, [5, 0, 0]);
residue_components!(Residue3QuarticSpec, Residue3Quartic, 3, 4, [2, 0, 2, 0]);
residue_components!(Residue2PairSpec, Residue2Pair, 2, 2, [1, 1]);

struct ResidueDegree8Spec;
impl ExtensionSpec<2> for ResidueDegree8Spec {
    type Base = Residue3Quartic;
    type Storage = Components;
    const NAME: &'static str = "nested residue degree eight";
    const RELATION: Relation = Relation::from_coordinates(&[0, 1, 0, 0, 0, 0, 0, 0]);
}
type ResidueDegree8 = FiniteField<3, 8, Extension<ResidueDegree8Spec, 2>>;

type Cantor8 = FiniteField<2, 8, fgf::Cantor<AES, 0x20>>;
struct CantorPairSpec;
impl ExtensionSpec<2> for CantorPairSpec {
    type Base = Cantor8;
    type Storage = Components;
    const NAME: &'static str = "Cantor components";
    const RELATION: Relation = Relation::from_binary_words::<Cantor8>(&[0x80, 1]);
}
type CantorPair = FiniteField<2, 16, Extension<CantorPairSpec, 2>>;

struct Degree64Spec;
impl ExtensionSpec<8> for Degree64Spec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "degree sixty-four AES components";
    const RELATION: Relation =
        Relation::from_binary_words::<Aes8>(&[0x6b, 0x30, 0xf9, 0x0e, 0xc7, 0xdd, 1, 0xe4]);
}
type Degree64 = FiniteField<2, 64, Extension<Degree64Spec, 8>>;

struct Packed32BaseSpec;
impl ExtensionSpec<2> for Packed32BaseSpec {
    type Base = FiniteField<2, 32, Extension<fgf::Rijndael32, 2>>;
    type Storage = Components;
    const NAME: &'static str = "components over packed GF(2^32)";
    const RELATION: Relation = <fgf::Rijndael64 as ExtensionSpec<2>>::RELATION;
}
type Packed32Base = FiniteField<2, 64, Extension<Packed32BaseSpec, 2>>;

struct Packed8BaseSpec;
impl ExtensionSpec<2> for Packed8BaseSpec {
    type Base = FiniteField<2, 8, Extension<fgf::FanPaar8, 2>>;
    type Storage = Components;
    const NAME: &'static str = "components over packed GF(2^8)";
    const RELATION: Relation = <fgf::FanPaar16 as ExtensionSpec<2>>::RELATION;
}
type Packed8Base = FiniteField<2, 16, Extension<Packed8BaseSpec, 2>>;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Every coefficient is nonzero and non-unit; varies by row/source.
    General,
    /// Mixture of zero, one, and general coefficients.
    Mixed,
    /// Every coefficient is zero.
    Zero,
    /// Every coefficient is one.
    One,
}

const ALL_PATTERNS: &[Pattern] = &[
    Pattern::General,
    Pattern::Mixed,
    Pattern::Zero,
    Pattern::One,
];
const ADDITIVE_PATTERNS: &[Pattern] = &[Pattern::Zero, Pattern::One];

fn controls_only() -> bool {
    std::env::args().any(|arg| arg == "--compositions-controls")
}

fn bench(label: &str, bytes: usize, mut body: impl FnMut()) -> f64 {
    if std::env::args().any(|arg| arg == "--compositions-check") {
        body();
        println!("CHECK\t{label}\t{bytes}");
        0.0
    } else {
        super::bench(label, bytes, body)
    }
}

impl Pattern {
    fn label(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Mixed => "mixed",
            Self::Zero => "zero",
            Self::One => "one",
        }
    }
}

/// Build `n` reduced `u64` coefficients for the given pattern modulo `p`.
/// For GF(2), `General` maps to `One` (no nonzero non-unit exists).
fn build_prime_coeffs(pattern: Pattern, n: usize, p: u64, seed: u64) -> Vec<u64> {
    let mut state = seed | 1;
    match pattern {
        Pattern::Zero => vec![0; n],
        Pattern::One => vec![1 % p; n],
        Pattern::General | Pattern::Mixed => {
            if p == 2 {
                return (0..n)
                    .map(|i| u64::from(!matches!(pattern, Pattern::Mixed) || i % 5 != 0))
                    .collect();
            }
            (0..n)
                .map(|i| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let raw = state;
                    match pattern {
                        Pattern::Mixed => match i % 5 {
                            0 => 0,
                            1 => 1 % p,
                            _ => ((raw % (p - 2)) + 2) % p,
                        },
                        _ => ((raw % (p - 2)) + 2) % p,
                    }
                })
                .collect()
        }
    }
}

/// Convert `u64` coefficients to `Elem<F>` via `from_residue`.
fn to_elems<F: fgf::Field + FieldBuffer>(raws: &[u64]) -> Vec<Elem<F>>
where
    Elem<F>: FieldElem,
{
    raws.iter()
        .map(|&r| Elem::<F>::from_residue(r).unwrap_or_else(|_| panic!("from_residue failed")))
        .collect()
}

const LANE_COUNTS: &[usize] = &[8, 64, 512, 4096];
const NROWS: usize = 4;
const NSOURCES: usize = 3;

fn lane_counts<F: FieldBuffer>() -> Vec<usize> {
    let mut lanes = LANE_COUNTS.to_vec();
    let mut boundaries = vec![16_usize, 32, 64];
    #[cfg(target_arch = "x86_64")]
    if matches!(F::BYTES, 1 | 4 | 8) {
        use fgf::internals::kernel::residue::{VectorOp, avx2_floor_op_len};
        boundaries.push(avx2_floor_op_len(F::BYTES, VectorOp::Add));
    }
    for bytes in boundaries {
        let boundary = bytes.div_ceil(F::BYTES);
        lanes.extend([boundary.saturating_sub(1).max(1), boundary, boundary + 1]);
    }
    lanes.sort_unstable();
    lanes.dedup();
    lanes
}

fn elementwise_controls<F: fgf::FieldKernels>(label: &str, lanes: usize, seed: u64) {
    let len = lanes * F::BYTES;
    let left = canonical_noise::<F>(len, seed ^ 0x3c00);
    let right = canonical_noise::<F>(len, seed ^ 0x4d00);
    let mut expected = vec![0; len];
    for ((dst, left), right) in expected
        .chunks_exact_mut(F::BYTES)
        .zip(left.chunks_exact(F::BYTES))
        .zip(right.chunks_exact(F::BYTES))
    {
        F::encode(dst, F::decode(left).mul(F::decode(right)));
    }
    let mut destination = vec![0; len];
    ops::mul_elementwise::<F>(&mut destination, &left, &right);
    assert_eq!(destination, expected, "{label}: elementwise control");
    for arm in ["a", "b"] {
        bench(
            &format!("compositions/{label}/mul_elementwise_control-{arm}/general/lanes-{lanes}"),
            3 * len,
            || {
                ops::mul_elementwise::<F>(
                    black_box(&mut destination),
                    black_box(&left),
                    black_box(&right),
                );
                black_box(destination.as_slice());
            },
        );
    }
}

/// Decode `n` elements from `buf` starting at the first lane.
fn elems_from_buf<F: FieldBuffer>(buf: &[u8], n: usize) -> Vec<Elem<F>>
where
    Elem<F>: FieldElem,
{
    (0..n)
        .map(|i| F::decode(&buf[i * F::BYTES..][..F::BYTES]))
        .collect()
}

/// Encode elements into a byte buffer.
fn elems_to_buf<F: FieldBuffer>(elems: &[Elem<F>]) -> Vec<u8>
where
    Elem<F>: FieldElem,
{
    let mut buf = vec![0u8; elems.len() * F::BYTES];
    for (i, e) in elems.iter().enumerate() {
        F::encode(&mut buf[i * F::BYTES..][..F::BYTES], *e);
    }
    buf
}

/// Round-trip noise through decode/encode to produce canonical lane bytes.
fn canonical_noise<F: FieldBuffer>(len: usize, seed: u64) -> Vec<u8>
where
    Elem<F>: FieldElem,
{
    let mut bytes = noise(len, seed);
    for lane in bytes.chunks_exact_mut(F::BYTES) {
        let value = F::decode(lane);
        F::encode(lane, value);
    }
    bytes
}

/// Benchmark all patterns and operations for one prime field.
#[allow(clippy::too_many_lines)]
fn prime_panel<F>(label: &str, seed: u64, patterns: &[Pattern])
where
    F: fgf::FieldKernels + FieldBuffer,
    Elem<F>: FieldElem,
{
    let p = <F::Characteristic as fgf::field::PrimeIdentity>::CHARACTERISTIC;
    let bytes = F::BYTES;
    println!(
        "FIELD_BACKEND\tcompositions/{label}\t{}",
        backend_for::<F>().name()
    );
    println!("FIELD_LAYOUT\tcompositions/{label}\t{bytes}");
    println!(
        "compositions prime {label} — backend {}:",
        backend_for::<F>().name()
    );

    for lanes in lane_counts::<F>() {
        let len = lanes * bytes;
        println!("  lanes {lanes} ({len} B):");
        elementwise_controls::<F>(label, lanes, seed);
        if controls_only() {
            continue;
        }

        for &pattern in patterns {
            let pat = pattern.label();

            // Single-row multiplication is measured alongside its compositions.
            {
                let src0 = canonical_noise::<F>(len, seed);
                let mut dst0 = canonical_noise::<F>(len, seed ^ 0x9e37);
                let c_raw = build_prime_coeffs(pattern, 1, p, seed ^ 0xab00)[0];
                let coeff = Elem::<F>::from_residue(c_raw).unwrap_or_else(|_| {
                    panic!("from_residue({c_raw}) failed for {label} {pat} p={p}")
                });

                // Oracle: dst[i] = dst[i] + coeff * src[i]
                let src_elems = elems_from_buf::<F>(&src0, lanes);
                let dst_elems = elems_from_buf::<F>(&dst0, lanes);
                let exp: Vec<Elem<F>> = dst_elems
                    .iter()
                    .zip(&src_elems)
                    .map(|(&d, &s)| d.add(coeff.mul(s)))
                    .collect();
                let exp_buf = elems_to_buf::<F>(&exp);

                let mut probe = dst0.clone();
                ops::mul_add::<F>(&mut probe, black_box(coeff), &src0);
                assert_eq!(
                    &probe, &exp_buf,
                    "compositions oracle mul_add {label} {pat} {lanes}"
                );
                probe.copy_from_slice(&dst0);

                let traffic = len * 2;
                bench(
                    &format!("compositions/{label}/mul_add/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst0);
                        let src = black_box(&src0);
                        ops::mul_add::<F>(dst, black_box(coeff), src);
                        black_box(&dst[..]);
                    },
                );

                let prepared = ops::Coeff::<F>::new(coeff);
                dst0.copy_from_slice(&probe);
                ops::mul_add_with::<F>(&mut dst0, &prepared, &src0);
                assert_eq!(
                    &dst0, &exp_buf,
                    "compositions oracle mul_add_with {label} {pat} {lanes}"
                );
                dst0.copy_from_slice(&probe);
                bench(
                    &format!("compositions/{label}/mul_add_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst0);
                        ops::mul_add_with::<F>(dst, black_box(&prepared), black_box(&src0));
                        black_box(&dst[..]);
                    },
                );
            }

            {
                let src = canonical_noise::<F>(len, seed ^ 0x1000);
                let rows_orig = canonical_noise::<F>(len * NROWS, seed ^ 0x2000);
                let coeff_raws = build_prime_coeffs(pattern, NROWS, p, seed ^ 0x3000);
                let coeff_elems = to_elems::<F>(&coeff_raws);
                let prepared_vec = ops::CoeffVec::<F>::new(&coeff_elems);

                let src_elems = elems_from_buf::<F>(&src, lanes);
                let rows_elems: Vec<Vec<Elem<F>>> = (0..NROWS)
                    .map(|r| elems_from_buf::<F>(&rows_orig[r * len..][..len], lanes))
                    .collect();
                let exp_scatter_add: Vec<Vec<u8>> = rows_elems
                    .iter()
                    .zip(&coeff_elems)
                    .map(|(row, &c)| {
                        elems_to_buf::<F>(
                            &row.iter()
                                .zip(&src_elems)
                                .map(|(&d, &s)| d.add(c.mul(s)))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();

                // mul_add_scatter (unprepared)
                {
                    let mut probe = rows_orig.clone();
                    ops::mul_add_scatter::<F>(&mut probe, len, black_box(&coeff_elems), &src);
                    for r in 0..NROWS {
                        assert_eq!(
                            &probe[r * len..][..len],
                            &exp_scatter_add[r],
                            "compositions oracle mul_add_scatter {label} {pat} {lanes} row{r}"
                        );
                    }
                }

                let traffic = len * (NROWS + 1);
                let mut rows1 = rows_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_scatter/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut rows1);
                        let src = black_box(&src);
                        ops::mul_add_scatter::<F>(rows, len, black_box(&coeff_elems), src);
                        black_box(&rows[..]);
                    },
                );

                // mul_into_scatter (unprepared)
                let exp_scatter_into: Vec<Vec<u8>> = coeff_elems
                    .iter()
                    .map(|&c| {
                        elems_to_buf::<F>(&src_elems.iter().map(|&s| c.mul(s)).collect::<Vec<_>>())
                    })
                    .collect();
                {
                    let mut probe = vec![0u8; len * NROWS];
                    ops::mul_into_scatter::<F>(&mut probe, len, black_box(&coeff_elems), &src);
                    for r in 0..NROWS {
                        assert_eq!(
                            &probe[r * len..][..len],
                            &exp_scatter_into[r],
                            "compositions oracle mul_into_scatter {label} {pat} {lanes} row{r}"
                        );
                    }
                }

                let mut rows2 = vec![0u8; len * NROWS];
                bench(
                    &format!("compositions/{label}/mul_into_scatter/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut rows2);
                        let src = black_box(&src);
                        ops::mul_into_scatter::<F>(rows, len, black_box(&coeff_elems), src);
                        black_box(&rows[..]);
                    },
                );

                // mul_add_scatter_with (prepared)
                {
                    let mut probe = rows_orig.clone();
                    ops::mul_add_scatter_with::<F>(
                        &mut probe,
                        len,
                        black_box(prepared_vec.as_ref()),
                        black_box(&src),
                    );
                    for r in 0..NROWS {
                        assert_eq!(
                            &probe[r * len..][..len],
                            &exp_scatter_add[r],
                            "compositions oracle mul_add_scatter_with {label} {pat} {lanes} row{r}"
                        );
                    }
                }

                let mut rows3 = rows_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_scatter_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut rows3);
                        let src = black_box(&src);
                        ops::mul_add_scatter_with::<F>(
                            rows,
                            len,
                            black_box(prepared_vec.as_ref()),
                            black_box(src),
                        );
                        black_box(&rows[..]);
                    },
                );
            }

            {
                let srcs: Vec<Vec<u8>> = (0..NSOURCES)
                    .map(|i| canonical_noise::<F>(len, seed ^ (0x4000 + i as u64)))
                    .collect();
                let dst_orig = canonical_noise::<F>(len, seed ^ 0x5000);
                let coeff_raws = build_prime_coeffs(pattern, NSOURCES, p, seed ^ 0x6000);
                let coeff_elems = to_elems::<F>(&coeff_raws);
                let prepared_vec = ops::CoeffVec::<F>::new(&coeff_elems);

                let srcs_elems: Vec<Vec<Elem<F>>> =
                    srcs.iter().map(|s| elems_from_buf::<F>(s, lanes)).collect();
                let dst_elems = elems_from_buf::<F>(&dst_orig, lanes);

                // Oracle: dst += sum_i coeff[i] * srcs[i]
                let exp_gather_add: Vec<u8> = elems_to_buf::<F>(
                    &dst_elems
                        .iter()
                        .enumerate()
                        .map(|(j, &d)| {
                            let mut acc = d;
                            for (s_idx, src_l) in srcs_elems.iter().enumerate() {
                                acc = acc.add(coeff_elems[s_idx].mul(src_l[j]));
                            }
                            acc
                        })
                        .collect::<Vec<_>>(),
                );

                let src_refs: Vec<&[u8]> = srcs.iter().map(|v| v.as_slice()).collect();

                // mul_add_gather (unprepared)
                {
                    let mut probe = dst_orig.clone();
                    ops::mul_add_gather::<F>(
                        &mut probe,
                        black_box(&coeff_elems),
                        black_box(&src_refs),
                    );
                    assert_eq!(
                        &probe, &exp_gather_add,
                        "compositions oracle mul_add_gather {label} {pat} {lanes}"
                    );
                }

                let traffic = len * (NSOURCES + 1);
                let mut dst1 = dst_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_gather/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst1);
                        ops::mul_add_gather::<F>(
                            dst,
                            black_box(&coeff_elems),
                            black_box(&src_refs),
                        );
                        black_box(&dst[..]);
                    },
                );

                // mul_into_gather (unprepared)
                let exp_gather_into: Vec<u8> = elems_to_buf::<F>(
                    &(0..lanes)
                        .map(|j| {
                            let mut acc = Elem::<F>::ZERO;
                            for (s_idx, src_l) in srcs_elems.iter().enumerate() {
                                acc = acc.add(coeff_elems[s_idx].mul(src_l[j]));
                            }
                            acc
                        })
                        .collect::<Vec<_>>(),
                );
                {
                    let mut probe = vec![0u8; len];
                    ops::mul_into_gather::<F>(
                        &mut probe,
                        black_box(&coeff_elems),
                        black_box(&src_refs),
                    );
                    assert_eq!(
                        &probe, &exp_gather_into,
                        "compositions oracle mul_into_gather {label} {pat} {lanes}"
                    );
                }

                let mut dst2 = vec![0u8; len];
                bench(
                    &format!("compositions/{label}/mul_into_gather/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst2);
                        ops::mul_into_gather::<F>(
                            dst,
                            black_box(&coeff_elems),
                            black_box(&src_refs),
                        );
                        black_box(&dst[..]);
                    },
                );

                // mul_add_gather_with (prepared)
                {
                    let mut probe = dst_orig.clone();
                    ops::mul_add_gather_with::<F>(
                        &mut probe,
                        black_box(prepared_vec.as_ref()),
                        black_box(&src_refs),
                    );
                    assert_eq!(
                        &probe, &exp_gather_add,
                        "compositions oracle mul_add_gather_with {label} {pat} {lanes}"
                    );
                }

                let mut dst3 = dst_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_gather_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst3);
                        ops::mul_add_gather_with::<F>(
                            dst,
                            black_box(prepared_vec.as_ref()),
                            black_box(&src_refs),
                        );
                        black_box(&dst[..]);
                    },
                );

                let mut dst4 = canonical_noise::<F>(len, seed ^ 0xd44a);
                let mut probe = dst4.clone();
                ops::mul_into_gather_with::<F>(
                    &mut probe,
                    black_box(prepared_vec.as_ref()),
                    black_box(&src_refs),
                );
                assert_eq!(
                    &probe, &exp_gather_into,
                    "compositions oracle mul_into_gather_with {label} {pat} {lanes}"
                );
                bench(
                    &format!("compositions/{label}/mul_into_gather_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let dst = black_box(&mut dst4);
                        ops::mul_into_gather_with::<F>(
                            dst,
                            black_box(prepared_vec.as_ref()),
                            black_box(&src_refs),
                        );
                        black_box(&dst[..]);
                    },
                );
            }

            {
                let ncoeff = NSOURCES * NROWS;
                let src_rows: Vec<Vec<u8>> = (0..NSOURCES)
                    .map(|i| canonical_noise::<F>(len, seed ^ (0x7000 + i as u64)))
                    .collect();
                let dst_rows_orig = canonical_noise::<F>(len * NROWS, seed ^ 0x8000);
                let coeff_raws = build_prime_coeffs(pattern, ncoeff, p, seed ^ 0x9000);
                let coeff_elems = to_elems::<F>(&coeff_raws);
                let coeff_matrix =
                    ops::CoeffMatrix::<F>::from_source_major(NSOURCES, NROWS, &coeff_elems);

                let src_elems_m: Vec<Vec<Elem<F>>> = src_rows
                    .iter()
                    .map(|s| elems_from_buf::<F>(s, lanes))
                    .collect();
                let dst_elems_m: Vec<Vec<Elem<F>>> = (0..NROWS)
                    .map(|o| elems_from_buf::<F>(&dst_rows_orig[o * len..][..len], lanes))
                    .collect();

                let exp_matrix_add: Vec<Vec<u8>> = dst_elems_m
                    .iter()
                    .enumerate()
                    .map(|(o, d_row)| {
                        elems_to_buf::<F>(
                            &d_row
                                .iter()
                                .enumerate()
                                .map(|(j, &d)| {
                                    let mut acc = d;
                                    for s in 0..NSOURCES {
                                        acc = acc
                                            .add(coeff_elems[s * NROWS + o].mul(src_elems_m[s][j]));
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();

                let exp_matrix_into: Vec<Vec<u8>> = (0..NROWS)
                    .map(|o| {
                        elems_to_buf::<F>(
                            &(0..lanes)
                                .map(|j| {
                                    let mut acc = Elem::<F>::ZERO;
                                    for s in 0..NSOURCES {
                                        acc = acc
                                            .add(coeff_elems[s * NROWS + o].mul(src_elems_m[s][j]));
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();

                let src_refs: Vec<&[u8]> = src_rows.iter().map(|v| v.as_slice()).collect();
                let terms: Vec<(&[Elem<F>], &[u8])> = (0..NSOURCES)
                    .map(|s| (&coeff_elems[s * NROWS..][..NROWS], src_refs[s]))
                    .collect();

                // mul_add_matrix (unprepared)
                {
                    let mut probe = dst_rows_orig.clone();
                    ops::mul_add_matrix::<F>(&mut probe, len, NROWS, black_box(&terms));
                    for o in 0..NROWS {
                        assert_eq!(
                            &probe[o * len..][..len],
                            &exp_matrix_add[o],
                            "compositions oracle mul_add_matrix {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let traffic = len * (NROWS + NSOURCES);
                let mut dst_rows1 = dst_rows_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_matrix/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut dst_rows1);
                        ops::mul_add_matrix::<F>(rows, len, NROWS, black_box(&terms));
                        black_box(&rows[..]);
                    },
                );

                // mul_into_matrix (unprepared)
                {
                    let mut probe = vec![0u8; len * NROWS];
                    ops::mul_into_matrix::<F>(&mut probe, len, NROWS, black_box(&terms));
                    for o in 0..NROWS {
                        assert_eq!(
                            &probe[o * len..][..len],
                            &exp_matrix_into[o],
                            "compositions oracle mul_into_matrix {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut dst2 = vec![0u8; len * NROWS];
                bench(
                    &format!("compositions/{label}/mul_into_matrix/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut dst2);
                        ops::mul_into_matrix::<F>(rows, len, NROWS, black_box(&terms));
                        black_box(&rows[..]);
                    },
                );

                // mul_add_matrix_with (prepared)
                {
                    let mut probe = dst_rows_orig.clone();
                    ops::mul_add_matrix_with::<F>(
                        &mut probe,
                        len,
                        black_box(&coeff_matrix),
                        black_box(&src_refs),
                    );
                    for o in 0..NROWS {
                        assert_eq!(
                            &probe[o * len..][..len],
                            &exp_matrix_add[o],
                            "compositions oracle mul_add_matrix_with {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut dst3 = dst_rows_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_matrix_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut dst3);
                        ops::mul_add_matrix_with::<F>(
                            rows,
                            len,
                            black_box(&coeff_matrix),
                            black_box(&src_refs),
                        );
                        black_box(&rows[..]);
                    },
                );

                // mul_into_matrix_with (prepared)
                {
                    let mut probe = vec![0u8; len * NROWS];
                    ops::mul_into_matrix_with::<F>(
                        &mut probe,
                        len,
                        black_box(&coeff_matrix),
                        black_box(&src_refs),
                    );
                    for o in 0..NROWS {
                        assert_eq!(
                            &probe[o * len..][..len],
                            &exp_matrix_into[o],
                            "compositions oracle mul_into_matrix_with {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut dst4 = vec![0u8; len * NROWS];
                bench(
                    &format!("compositions/{label}/mul_into_matrix_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut dst4);
                        ops::mul_into_matrix_with::<F>(
                            rows,
                            len,
                            black_box(&coeff_matrix),
                            black_box(&src_refs),
                        );
                        black_box(&rows[..]);
                    },
                );
            }

            {
                let ncoeff = NSOURCES * NROWS;
                let src_rows: Vec<Vec<u8>> = (0..NSOURCES)
                    .map(|i| canonical_noise::<F>(len, seed ^ (0xa000 + i as u64)))
                    .collect();
                let coeff_raws = build_prime_coeffs(pattern, ncoeff, p, seed ^ 0xb000);
                let coeff_elems = to_elems::<F>(&coeff_raws);
                let coeff_matrix =
                    ops::CoeffMatrix::<F>::from_source_major(NSOURCES, NROWS, &coeff_elems);

                let row_stride = len + bytes;
                let region_len = row_stride * NROWS + bytes;
                let region_orig = canonical_noise::<F>(region_len, seed ^ 0xc000);
                let row_starts: Vec<usize> = (0..NROWS).map(|r| r * row_stride).collect();

                let src_elems_at: Vec<Vec<Elem<F>>> = src_rows
                    .iter()
                    .map(|s| elems_from_buf::<F>(s, lanes))
                    .collect();
                let dst_at_elems: Vec<Vec<Elem<F>>> = row_starts
                    .iter()
                    .map(|&start| elems_from_buf::<F>(&region_orig[start..][..len], lanes))
                    .collect();

                let exp_matrix_at_add: Vec<Vec<u8>> = dst_at_elems
                    .iter()
                    .enumerate()
                    .map(|(o, d_row)| {
                        elems_to_buf::<F>(
                            &d_row
                                .iter()
                                .enumerate()
                                .map(|(j, &d)| {
                                    let mut acc = d;
                                    for s in 0..NSOURCES {
                                        acc = acc.add(
                                            coeff_elems[s * NROWS + o].mul(src_elems_at[s][j]),
                                        );
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();

                let exp_matrix_at_into: Vec<Vec<u8>> = (0..NROWS)
                    .map(|o| {
                        elems_to_buf::<F>(
                            &(0..lanes)
                                .map(|j| {
                                    let mut acc = Elem::<F>::ZERO;
                                    for s in 0..NSOURCES {
                                        acc = acc.add(
                                            coeff_elems[s * NROWS + o].mul(src_elems_at[s][j]),
                                        );
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();

                let src_refs: Vec<&[u8]> = src_rows.iter().map(|v| v.as_slice()).collect();
                let terms: Vec<(&[Elem<F>], &[u8])> = (0..NSOURCES)
                    .map(|s| (&coeff_elems[s * NROWS..][..NROWS], src_refs[s]))
                    .collect();

                // mul_add_matrix_at (unprepared)
                {
                    let mut probe = region_orig.clone();
                    ops::mul_add_matrix_at::<F>(&mut probe, len, &row_starts, black_box(&terms));
                    for o in 0..NROWS {
                        let start = row_starts[o];
                        assert_eq!(
                            &probe[start..][..len],
                            &exp_matrix_at_add[o],
                            "compositions oracle mul_add_matrix_at {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let traffic = len * (NROWS + NSOURCES);
                let mut region1 = region_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_matrix_at/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut region1);
                        ops::mul_add_matrix_at::<F>(rows, len, &row_starts, black_box(&terms));
                        black_box(&rows[..]);
                    },
                );

                // mul_into_matrix_at (unprepared)
                {
                    let mut probe = region_orig.clone();
                    for &start in &row_starts {
                        probe[start..start + len].fill(0);
                    }
                    ops::mul_into_matrix_at::<F>(&mut probe, len, &row_starts, black_box(&terms));
                    for o in 0..NROWS {
                        let start = row_starts[o];
                        assert_eq!(
                            &probe[start..][..len],
                            &exp_matrix_at_into[o],
                            "compositions oracle mul_into_matrix_at {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut region2 = region_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_into_matrix_at/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut region2);
                        ops::mul_into_matrix_at::<F>(rows, len, &row_starts, black_box(&terms));
                        black_box(&rows[..]);
                    },
                );

                // mul_add_matrix_at_with (prepared)
                {
                    let mut probe = region_orig.clone();
                    ops::mul_add_matrix_at_with::<F>(
                        &mut probe,
                        len,
                        &row_starts,
                        black_box(&coeff_matrix),
                        black_box(&src_refs),
                    );
                    for o in 0..NROWS {
                        let start = row_starts[o];
                        assert_eq!(
                            &probe[start..][..len],
                            &exp_matrix_at_add[o],
                            "compositions oracle mul_add_matrix_at_with {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut region3 = region_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_add_matrix_at_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut region3);
                        ops::mul_add_matrix_at_with::<F>(
                            rows,
                            len,
                            &row_starts,
                            black_box(&coeff_matrix),
                            black_box(&src_refs),
                        );
                        black_box(&rows[..]);
                    },
                );

                // mul_into_matrix_at_with (prepared)
                {
                    let mut probe = region_orig.clone();
                    for &start in &row_starts {
                        probe[start..start + len].fill(0);
                    }
                    ops::mul_into_matrix_at_with::<F>(
                        &mut probe,
                        len,
                        &row_starts,
                        black_box(&coeff_matrix),
                        black_box(&src_refs),
                    );
                    for o in 0..NROWS {
                        let start = row_starts[o];
                        assert_eq!(
                            &probe[start..][..len],
                            &exp_matrix_at_into[o],
                            "compositions oracle mul_into_matrix_at_with {label} {pat} {lanes} row{o}"
                        );
                    }
                }

                let mut region4 = region_orig.clone();
                bench(
                    &format!("compositions/{label}/mul_into_matrix_at_with/{pat}/lanes-{lanes}"),
                    traffic,
                    || {
                        let rows = black_box(&mut region4);
                        ops::mul_into_matrix_at_with::<F>(
                            rows,
                            len,
                            &row_starts,
                            black_box(&coeff_matrix),
                            black_box(&src_refs),
                        );
                        black_box(&rows[..]);
                    },
                );
            }
        }
    }
    additive_panel::<F>(label, seed);
}

/// Oracle for component field addition: `Elem::add`.
fn component_elem_add<F: fgf::Field + FieldBuffer>(a: &[u8], b: &[u8]) -> Vec<u8>
where
    Elem<F>: FieldElem,
{
    let mut out = vec![0u8; a.len()];
    for i in (0..a.len()).step_by(F::BYTES) {
        let ae = F::decode(&a[i..][..F::BYTES]);
        let be = F::decode(&b[i..][..F::BYTES]);
        F::encode(&mut out[i..][..F::BYTES], ae.add(be));
    }
    out
}

/// Oracle for component field subtraction: `Elem::sub`.
fn component_elem_sub<F: fgf::Field + FieldBuffer>(a: &[u8], b: &[u8]) -> Vec<u8>
where
    Elem<F>: FieldElem,
{
    let mut out = vec![0u8; a.len()];
    for i in (0..a.len()).step_by(F::BYTES) {
        let ae = F::decode(&a[i..][..F::BYTES]);
        let be = F::decode(&b[i..][..F::BYTES]);
        F::encode(&mut out[i..][..F::BYTES], ae.sub(be));
    }
    out
}

/// Benchmarks additive operations for one field.
fn additive_panel<F>(label: &str, seed: u64)
where
    F: fgf::FieldKernels + FieldBuffer,
    Elem<F>: FieldElem,
{
    if controls_only() {
        return;
    }
    let bytes = F::BYTES;
    println!(
        "compositions additive {label} — backend {}:",
        backend_for::<F>().name()
    );

    for lanes in lane_counts::<F>() {
        let len = lanes * bytes;
        println!("  lanes {lanes} ({len} B):");

        let n = NROWS;

        // add_assign
        {
            let src = canonical_noise::<F>(len, seed ^ 0x1000);
            let mut dst = canonical_noise::<F>(len, seed ^ 0x1100);
            let exp = component_elem_add::<F>(&dst, &src);

            {
                let mut probe = dst.clone();
                ops::add_assign::<F>(&mut probe, black_box(&src));
                assert_eq!(probe, exp, "compositions oracle add_assign {label} {lanes}");
            }

            bench(
                &format!("compositions/{label}/add_assign/general/lanes-{lanes}"),
                len * 2,
                || {
                    let dst = black_box(&mut dst);
                    let src = black_box(&src);
                    ops::add_assign::<F>(dst, black_box(src));
                    black_box(&dst[..]);
                },
            );
        }

        // sub_assign
        {
            let src = canonical_noise::<F>(len, seed ^ 0x1200);
            let mut dst = canonical_noise::<F>(len, seed ^ 0x1300);
            let exp = component_elem_sub::<F>(&dst, &src);

            {
                let mut probe = dst.clone();
                ops::sub_assign::<F>(&mut probe, black_box(&src));
                assert_eq!(probe, exp, "compositions oracle sub_assign {label} {lanes}");
            }

            bench(
                &format!("compositions/{label}/sub_assign/general/lanes-{lanes}"),
                len * 2,
                || {
                    let dst = black_box(&mut dst);
                    let src = black_box(&src);
                    ops::sub_assign::<F>(dst, black_box(src));
                    black_box(&dst[..]);
                },
            );
        }

        // add_assign_rows
        {
            let src_rows = canonical_noise::<F>(len * n, seed ^ 0x1400);
            let mut dst_rows = canonical_noise::<F>(len * n, seed ^ 0x1500);
            let exp = component_elem_add::<F>(&dst_rows, &src_rows);

            {
                let mut probe = dst_rows.clone();
                ops::add_assign_rows::<F>(&mut probe, len, black_box(&src_rows));
                assert_eq!(
                    probe, exp,
                    "compositions oracle add_assign_rows {label} {lanes}"
                );
            }

            bench(
                &format!("compositions/{label}/add_assign_rows/general/lanes-{lanes}"),
                len * n * 2,
                || {
                    let dst = black_box(&mut dst_rows);
                    ops::add_assign_rows::<F>(dst, len, black_box(&src_rows));
                    black_box(&dst[..]);
                },
            );
        }

        // sub_assign_rows
        {
            let src_rows = canonical_noise::<F>(len * n, seed ^ 0x1600);
            let mut dst_rows = canonical_noise::<F>(len * n, seed ^ 0x1700);
            let exp = component_elem_sub::<F>(&dst_rows, &src_rows);

            {
                let mut probe = dst_rows.clone();
                ops::sub_assign_rows::<F>(&mut probe, len, black_box(&src_rows));
                assert_eq!(
                    probe, exp,
                    "compositions oracle sub_assign_rows {label} {lanes}"
                );
            }

            bench(
                &format!("compositions/{label}/sub_assign_rows/general/lanes-{lanes}"),
                len * n * 2,
                || {
                    let dst = black_box(&mut dst_rows);
                    ops::sub_assign_rows::<F>(dst, len, black_box(&src_rows));
                    black_box(&dst[..]);
                },
            );
        }

        // add_gather_offsets
        {
            let src = canonical_noise::<F>(len * (NSOURCES + 2), seed ^ 0x1800);
            let mut dst = canonical_noise::<F>(len, seed ^ 0x1900);
            let offsets: Vec<u32> = (0..NSOURCES).map(|i| (i * len) as u32).collect();

            // Oracle: accumulate each source row into dst
            let exp = {
                let mut acc = dst.clone();
                for &off in &offsets {
                    let slice = &src[off as usize..][..len];
                    let sum = component_elem_add::<F>(&acc, slice);
                    acc.copy_from_slice(&sum);
                }
                acc
            };

            {
                let mut probe = dst.clone();
                ops::add_gather_offsets::<F>(&mut probe, black_box(&src), black_box(&offsets));
                assert_eq!(
                    probe, exp,
                    "compositions oracle add_gather_offsets {label} {lanes}"
                );
            }

            bench(
                &format!("compositions/{label}/add_gather_offsets/general/lanes-{lanes}"),
                len * (NSOURCES + 1),
                || {
                    let dst = black_box(&mut dst);
                    ops::add_gather_offsets::<F>(dst, black_box(&src), black_box(&offsets));
                    black_box(&dst[..]);
                },
            );
        }
    }
}

/// Run the compositions panel. Called from `run_panels` when `--compositions`
/// is given.
pub(crate) fn run() {
    println!("Public row-composition measurements:");
    println!("COMPOSITIONS_GEOMETRY\trows\t{NROWS}\tsources\t{NSOURCES}");

    println!("\nQuadratic prime rows:");
    prime_panel::<QuadMersenne31>("qm31", 0x10_0000, ALL_PATTERNS);

    println!("\nPrime fields:");
    prime_panel::<Mersenne31>("m31", 0x20_0000, ALL_PATTERNS);
    prime_panel::<Goldilocks>("gld", 0x30_0000, ALL_PATTERNS);

    // Residue prime fields
    prime_panel::<FiniteField<101, 1>>("gf101", 0x40_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<127, 1>>("gf127", 0x41_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<131, 1>>("gf131", 0x42_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<65_521, 1>>("gf65521", 0x43_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<65_537, 1>>("gf65537", 0x44_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<1_000_000_007, 1>>("gf1e9p7", 0x45_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<1_280_000_003, 1>>("gf1280000003", 0x46_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<4_294_967_291, 1>>("gf4294967291", 0x47_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<MAX_PRIME, 1>>("gfmaxprime", 0x48_0000, ALL_PATTERNS);

    // Residue variants
    prime_panel::<FiniteField<0x7FFF_FFFF, 1, fgf::Residue>>("m31r", 0x50_0000, ALL_PATTERNS);
    prime_panel::<FiniteField<0xFFFF_FFFF_0000_0001, 1, fgf::Residue>>(
        "gldr",
        0x51_0000,
        ALL_PATTERNS,
    );

    // GF(2) shortcut
    prime_panel::<FiniteField<2, 1>>("gf2", 0x52_0000, ALL_PATTERNS);

    println!("\nComponent additive routes:");
    component_panel::<M31Quad>("m31quad", 0x60_0000);
    component_panel::<GldQuad>("gldquad", 0x61_0000);
    component_panel::<Aes8Quad>("aes8quad", 0x62_0000);
    component_panel::<Aes8NormalQuad>("aes8nquad", 0x63_0000);
    component_panel::<Gf81>("gf81", 0x64_0000);
    component_panel::<MaxPrimeQuad>("maxprimequad", 0x65_0000);
    component_panel::<MersenneCube>("m31cube", 0x66_0000);
    component_panel::<OverQuad>("over-quad", 0x67_0000);
    component_panel::<PackedBase>("packed-base", 0x68_0000);
    component_panel::<Residue3Pair>("residue3-pair", 0x69_0000);
    component_panel::<Residue131Pair>("residue131-pair", 0x6a_0000);
    component_panel::<Residue257Pair>("residue257-pair", 0x6b_0000);
    component_panel::<Residue65537Pair>("residue65537-pair", 0x6c_0000);
    component_panel::<Residue7Cube>("residue7-cube", 0x6d_0000);
    component_panel::<Residue3Quartic>("residue3-quartic", 0x6e_0000);
    component_panel::<Residue2Pair>("residue2-pair", 0x6f_0000);
    component_panel::<ResidueDegree8>("residue-degree8", 0x70_0000);
    component_panel::<CantorPair>("cantor-pair", 0x71_0000);
    component_panel::<Degree64>("degree64", 0x72_0000);
    component_panel::<Packed32Base>("packed32-base", 0x73_0000);
    component_panel::<Packed8Base>("packed8-base", 0x74_0000);
    let mode = if controls_only() {
        "controls"
    } else if std::env::args().any(|arg| arg == "--compositions-check") {
        "check"
    } else {
        "full"
    };
    println!("COMPOSITIONS_DONE\t{mode}");
}

fn component_panel<F: fgf::FieldKernels>(label: &str, seed: u64) {
    prime_panel::<F>(label, seed, ADDITIVE_PATTERNS);
}
