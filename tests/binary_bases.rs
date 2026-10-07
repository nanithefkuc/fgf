//! Binary presentation surfaces outside the arithmetic sweeps: element
//! formatting, custom-tower squaring against a base-field schoolbook, and the
//! wide identity and canonical predicates.

use std::fmt::{self, Write};

use fgf::{
    AES, Binary, BinaryField, Cantor, Elem, Embedding, FieldElem, Normal, Polynomial, RS, Tower,
    TowerSpec,
};

type Gf16 = fgf::Binary<16, fgf::Tower<fgf::Rijndael16>>;
type Gf32 = fgf::Binary<32, fgf::Tower<fgf::Rijndael32>>;
type Gf64 = fgf::Binary<64, fgf::Tower<fgf::Rijndael64>>;

/// Deterministic linear-congruential samples.
fn lcg_samples(count: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state ^ (state >> 29)
        })
        .collect()
}

/// Degree-two and degree-four elements display as hexadecimal digits that
/// parse back to their coordinate word, under every ordered basis.
#[test]
fn small_degree_display_parses_back_to_coordinates() {
    fn check<F: BinaryField>(count: u64)
    where
        Elem<F>: fmt::Display,
    {
        for word in 0..count {
            let element = F::from_coordinates(word).expect("in-range coordinates");
            let text = element.to_string();
            assert_eq!(u64::from_str_radix(&text, 16), Ok(word), "{text:?}");
        }
    }
    check::<Binary<2, Polynomial<0x7>>>(4);
    check::<Binary<2, Normal<0x7, 2>>>(4);
    check::<Binary<2, Cantor<0x7, 2>>>(4);
    check::<Binary<4, Polynomial<0x13>>>(16);
    check::<Binary<4, Normal<0x13, 8>>>(16);
    check::<Binary<4, Cantor<0x13, 8>>>(16);
}

/// A writer with no capacity, failing every nonempty write.
struct FullWriter;

impl Write for FullWriter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if text.is_empty() {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

/// Formatting surfaces of markers, errors, and prepared embeddings return
/// the writer's failure instead of swallowing it.
#[test]
fn binary_formatting_reports_writer_failure() {
    let marker = Binary::<8, Normal<0x11B, 0x20>>::default();
    assert_eq!(write!(FullWriter, "{marker:?}"), Err(fmt::Error));

    let coordinate =
        Binary::<4, Polynomial<0x13>>::from_coordinates(0x10).expect_err("excess coordinate bits");
    assert_eq!(write!(FullWriter, "{coordinate}"), Err(fmt::Error));

    let degree = Embedding::<Gf16, Binary<8, Polynomial<AES>>>::new()
        .expect_err("sixteen does not divide eight");
    assert_eq!(write!(FullWriter, "{degree}"), Err(fmt::Error));

    let embedding =
        Embedding::<Binary<8, Polynomial<AES>>, Gf16>::new().expect("eight divides sixteen");
    assert_eq!(write!(FullWriter, "{embedding:?}"), Err(fmt::Error));
}

/// `t^2 + 2*t + 0x80` over the Reed–Solomon byte field.
#[derive(Clone, Copy)]
struct BasesCustom16;
impl TowerSpec for BasesCustom16 {
    type Base = Binary<8, Polynomial<RS>>;
    const LINEAR_COEFFICIENT: u64 = 2;
    const CONSTANT_COEFFICIENT: u64 = 0x80;
    const NAME: &'static str = "bases custom GF(2^16)";
}

/// `t^2 + t + 0x2001` over [`Gf16`].
#[derive(Clone, Copy)]
struct BasesCustom32;
impl TowerSpec for BasesCustom32 {
    type Base = Gf16;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x2001;
    const NAME: &'static str = "bases custom GF(2^32)";
}

/// `t^2 + t + 0x2ada7f36` over [`Gf32`].
#[derive(Clone, Copy)]
struct BasesCustom64;
impl TowerSpec for BasesCustom64 {
    type Base = Gf32;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x2ada_7f36;
    const NAME: &'static str = "bases custom GF(2^64)";
}

/// A tower over a normal-basis byte field, whose one is `0x00FF` rather
/// than raw one.
#[derive(Clone, Copy)]
struct BasesNormal16;
impl TowerSpec for BasesNormal16 {
    type Base = Binary<8, Normal<0x11B, 0x20>>;
    const LINEAR_COEFFICIENT: u64 = 0xFF;
    const CONSTANT_COEFFICIENT: u64 = 0x1;
    const NAME: &'static str = "bases normal-base GF(2^16)";
}

/// The square of `a + b*t` under `t^2 = A*t + B`, computed in the base:
/// `(a^2 + B*b^2) + (A*b^2)*t`.
fn schoolbook_square<B: BinaryField>(half: u32, linear: u64, constant: u64, word: u64) -> u64 {
    let base = |coordinates: u64| B::from_coordinates(coordinates).expect("base coordinates");
    let mask = (1u64 << half) - 1;
    let low = base(word & mask);
    let high = base(word >> half);
    let low_sq = FieldElem::mul(low, low);
    let high_sq = FieldElem::mul(high, high);
    let new_low = FieldElem::add(low_sq, FieldElem::mul(base(constant), high_sq));
    let new_high = FieldElem::mul(base(linear), high_sq);
    B::to_coordinates(new_low) | (B::to_coordinates(new_high) << half)
}

/// Custom towers square through the general descriptor interpreter; the
/// result matches the base-field schoolbook of the defining relation.
#[test]
fn custom_tower_squares_match_the_base_schoolbook() {
    type F16 = Binary<16, Tower<BasesCustom16>>;
    type F32 = Binary<32, Tower<BasesCustom32>>;
    type F64 = Binary<64, Tower<BasesCustom64>>;
    for raw in 0..=u16::MAX {
        let square = Elem::<F16>::from_raw(raw).square();
        assert_eq!(
            u64::from(square.to_raw()),
            schoolbook_square::<Binary<8, Polynomial<RS>>>(8, 2, 0x80, u64::from(raw)),
            "{raw:#06x}"
        );
    }
    for word in lcg_samples(4096, 0x5eed_0032) {
        #[allow(clippy::cast_possible_truncation)]
        let raw = word as u32;
        let square = Elem::<F32>::from_raw(raw).square();
        assert_eq!(
            u64::from(square.to_raw()),
            schoolbook_square::<Gf16>(16, 1, 0x2001, u64::from(raw)),
            "{raw:#010x}"
        );
    }
    for raw in lcg_samples(4096, 0x5eed_0064) {
        let square = Elem::<F64>::from_raw(raw).square();
        assert_eq!(
            square.to_raw(),
            schoolbook_square::<Gf32>(32, 1, 0x2ada_7f36, raw),
            "{raw:#018x}"
        );
    }
}

/// `is_one` holds exactly for the multiplicative identity — `x * g == g`
/// for a nonzero `g` — including a tower whose one is not raw one, and
/// canonical encodings agree with element equality.
#[test]
fn wide_identity_and_canonical_predicates_follow_the_field() {
    macro_rules! check {
        ($field:ty, $samples:expr, $probe:expr) => {{
            let probe = Elem::<$field>::from_raw($probe);
            let elements: Vec<Elem<$field>> = $samples
                .into_iter()
                .chain([0, 1, Elem::<$field>::ONE.to_raw()])
                .map(Elem::<$field>::from_raw)
                .collect();
            let mut ones = 0;
            for &x in &elements {
                let identity = x.mul(probe) == probe;
                assert_eq!(x.is_one(), identity, "{:#x}", x.to_raw());
                ones += usize::from(identity);
                assert_eq!(x.canonical(), x);
                for &y in &elements {
                    assert_eq!(x.canonical().to_bytes() == y.canonical().to_bytes(), x == y);
                }
            }
            assert!(ones >= 1, "the identity is among the samples");
        }};
    }
    #[allow(clippy::cast_possible_truncation)]
    let narrow = |count, seed| -> Vec<u16> {
        lcg_samples(count, seed)
            .into_iter()
            .map(|word| word as u16)
            .collect()
    };
    #[allow(clippy::cast_possible_truncation)]
    let medium = |count, seed| -> Vec<u32> {
        lcg_samples(count, seed)
            .into_iter()
            .map(|word| word as u32)
            .collect()
    };
    check!(Binary<16, Tower<BasesNormal16>>, narrow(64, 0x1d16), 0x1234);
    check!(Gf16, narrow(64, 0x6f16), 0x1234);
    check!(Gf32, medium(64, 0x6f32), 0x1234_5678);
    check!(Gf64, lcg_samples(64, 0x6f64), 0x1234_5678_9abc_def0);
}
