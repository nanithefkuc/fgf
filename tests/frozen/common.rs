//! Shared reader for the frozen field fixtures.
//!
//! The fixtures are UTF-8 text: one record per line, whitespace-separated
//! fields, hexadecimal words as lowercase `0x…`, decimal counts and indices,
//! `#` comment lines only as the file header. This module parses that shape,
//! mirrors the fixture sampling rule, and re-derives the presentation facts
//! the fixtures record so every test asserts against recomputed values.

/// The capture revision every fixture file names in its header.
pub const TAG: &str = "v3.0.0-rc.1";

/// The parsed record lines of one fixture file.
///
/// The iterator skips the two-line header after checking it, then yields
/// every remaining line with its one-based line number. Empty lines are
/// skipped; any other `#` line panics.
pub struct Records<'a> {
    file: &'static str,
    lines: core::str::Lines<'a>,
    line_number: usize,
    header_done: bool,
}

impl<'a> Records<'a> {
    pub fn new(file: &'static str, text: &'a str) -> Self {
        Self {
            file,
            lines: text.lines(),
            line_number: 0,
            header_done: false,
        }
    }

    fn check_header(&mut self) {
        let describes = self.lines.next().expect("fixture header line");
        self.line_number += 1;
        assert!(
            describes.starts_with("# "),
            "{}: missing fixture header",
            self.file
        );
        let tag = self.lines.next().expect("fixture tag line");
        self.line_number += 1;
        assert_eq!(
            tag,
            format!("# captured from {TAG}"),
            "{}: wrong capture tag",
            self.file
        );
        self.header_done = true;
    }
}

impl<'a> Iterator for Records<'a> {
    type Item = (&'a str, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.header_done {
            self.check_header();
        }
        loop {
            let line = self.lines.next()?;
            self.line_number += 1;
            if line.is_empty() {
                continue;
            }
            assert!(
                !line.starts_with('#'),
                "{}: comment line outside the header",
                self.file
            );
            return Some((line, self.line_number));
        }
    }
}

/// Parse one hexadecimal word field.
pub fn word(file: &'static str, number: usize, field: &str) -> u64 {
    field
        .strip_prefix("0x")
        .and_then(|rest| u64::from_str_radix(rest, 16).ok())
        .unwrap_or_else(|| panic!("{file}:{number}: not a 0x word: {field}"))
}

/// Parse one hexadecimal word field that may be the absent marker.
pub fn opt_word(file: &'static str, number: usize, field: &str) -> Option<u64> {
    if field == "-" {
        None
    } else {
        Some(word(file, number, field))
    }
}

/// The fixture sample generator: SplitMix64 seeded at zero.
pub struct SplitMix64Reader(u64);

impl SplitMix64Reader {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// All-ones at `degree` bits; every bit at degree 64.
pub fn degree_mask(degree: u32) -> u64 {
    if degree >= 64 {
        u64::MAX
    } else {
        (1u64 << degree) - 1
    }
}

/// The 64 sample coordinate words of one presentation, freshly seeded.
pub fn samples(degree: u32) -> [u64; 64] {
    let mut source = SplitMix64Reader::new();
    core::array::from_fn(|_| source.next() & degree_mask(degree))
}

/// The 64 source and 64 target sample words of one ordered pair, drawn in
/// that order from a freshly seeded generator.
pub fn pair_samples(source_degree: u32, target_degree: u32) -> ([u64; 64], [u64; 64]) {
    let mut source = SplitMix64Reader::new();
    let source_words = core::array::from_fn(|_| source.next() & degree_mask(source_degree));
    let target_words = core::array::from_fn(|_| source.next() & degree_mask(target_degree));
    (source_words, target_words)
}

// ---------------------------------------------------------------------------
// Presentation surface mirrored from the capture configuration
// ---------------------------------------------------------------------------

/// Byte encoding of one element, where the presentation carries one.
pub trait ByteEnc: fgf::Field {
    fn encode_bytes(value: fgf::Elem<Self>) -> Vec<u8>;
}

macro_rules! byte_enc_small {
    ($($n:literal),* $(,)?) => {$(
        impl<R: fgf::BinaryRepr<$n>> ByteEnc for fgf::FiniteField<2, $n, R> {
            fn encode_bytes(value: fgf::Elem<Self>) -> Vec<u8> {
                value.to_bytes().into_iter().collect()
            }
        }
    )*};
}

byte_enc_small!(1, 2, 4, 8);

macro_rules! byte_enc_tower {
    ($($n:literal),* $(,)?) => {$(
        impl<S: fgf::TowerSpec> ByteEnc for fgf::FiniteField<2, $n, fgf::Tower<S>>
        where
            S::Base: fgf::BinaryDegree<{ $n / 2 }>,
        {
            fn encode_bytes(value: fgf::Elem<Self>) -> Vec<u8> {
                value.to_bytes().into_iter().collect()
            }
        }
    )*};
}

byte_enc_tower!(16, 32, 64);

/// Decoding a full-width alias through the storage constructor: the decoded
/// raw word and the byte encoding.
pub trait AliasDecode: fgf::Field {
    fn decode_alias(word: u64) -> (u64, Vec<u8>);
}

macro_rules! alias_decode_small {
    ($($n:literal),* $(,)?) => {$(
        impl<R: fgf::BinaryRepr<$n>> AliasDecode for fgf::FiniteField<2, $n, R> {
            fn decode_alias(word: u64) -> (u64, Vec<u8>) {
                let value = fgf::Elem::<Self>::from_raw(word as u8);
                let raw = value.to_raw();
                (u64::from(raw), Self::encode_bytes(value))
            }
        }
    )*};
}

alias_decode_small!(1, 2, 4, 8);

macro_rules! alias_decode_tower {
    ($($n:literal, $raw:ty;)* $(,)?) => {$(
        impl<S: fgf::TowerSpec> AliasDecode for fgf::FiniteField<2, $n, fgf::Tower<S>>
        where
            S::Base: fgf::BinaryDegree<{ $n / 2 }>,
        {
            fn decode_alias(word: u64) -> (u64, Vec<u8>) {
                let value = fgf::Elem::<Self>::from_raw(word as $raw);
                let raw = value.to_raw();
                (u64::from(raw), Self::encode_bytes(value))
            }
        }
    )*};
}

alias_decode_tower!(16, u16; 32, u32; 64, u64;);

/// The extension levels of a presentation, prime field upward.
pub trait Levels: fgf::Field + Sized {
    const LEVEL_COUNT: usize;

    /// The level's generator: its coordinate word in the level's own
    /// presentation and the element of `Self` it lifts to.
    fn level_element(level: usize) -> (u64, fgf::Elem<Self>);
}

macro_rules! poly_levels {
    ($($n:literal),* $(,)?) => {$(
        impl<const P: u128> Levels for fgf::FiniteField<2, $n, fgf::Polynomial<P>> {
            const LEVEL_COUNT: usize = if $n == 1 { 0 } else { 1 };

            fn level_element(_level: usize) -> (u64, fgf::Elem<Self>) {
                let indeterminate = <Self as fgf::BinaryField>::from_coordinates(2)
                    .expect("the indeterminate fits");
                (2, indeterminate)
            }
        }
    )*};
}

poly_levels!(1, 2, 4, 8);

macro_rules! basis_levels {
    ($($n:literal),* $(,)?) => {$(
        impl<const P: u128, const E: u8> Levels for fgf::FiniteField<2, $n, fgf::Normal<P, E>> {
            const LEVEL_COUNT: usize = 1;

            fn level_element(_level: usize) -> (u64, fgf::Elem<Self>) {
                let route = fgf::Embedding::<fgf::FiniteField<2, $n, fgf::Polynomial<P>>, Self>::new()
                    .expect("equal degrees embed");
                let generator =
                    route.embed(fgf::Elem::<fgf::FiniteField<2, $n, fgf::Polynomial<P>>>::from_raw(2));
                (<Self as fgf::BinaryField>::to_coordinates(generator), generator)
            }
        }

        impl<const P: u128, const SEED: u8> Levels for fgf::FiniteField<2, $n, fgf::Cantor<P, SEED>> {
            const LEVEL_COUNT: usize = 1;

            fn level_element(_level: usize) -> (u64, fgf::Elem<Self>) {
                let route = fgf::Embedding::<fgf::FiniteField<2, $n, fgf::Polynomial<P>>, Self>::new()
                    .expect("equal degrees embed");
                let generator =
                    route.embed(fgf::Elem::<fgf::FiniteField<2, $n, fgf::Polynomial<P>>>::from_raw(2));
                (<Self as fgf::BinaryField>::to_coordinates(generator), generator)
            }
        }
    )*};
}

basis_levels!(2, 4, 8);

macro_rules! small_tower_levels {
    ($($n:literal),* $(,)?) => {$(
        impl<S: fgf::TowerSpec> Levels for fgf::FiniteField<2, $n, fgf::Tower<S>>
        where
            S::Base: Levels + fgf::BinaryField + fgf::BinaryDegree<{ $n / 2 }>,
            S::Base: fgf::Field<Raw = u8>,
        {
            const LEVEL_COUNT: usize = <S::Base as Levels>::LEVEL_COUNT + 1;

            fn level_element(level: usize) -> (u64, fgf::Elem<Self>) {
                let base_count = <S::Base as Levels>::LEVEL_COUNT;
                if level < base_count {
                    let (word, base_element) = <S::Base as Levels>::level_element(level);
                    (word, fgf::Elem::<Self>::from_raw(base_element.to_raw()))
                } else {
                    let base_one = <S::Base as fgf::Field>::ONE_RAW;
                    let indeterminate = fgf::Elem::<Self>::from_raw(base_one << ($n / 2));
                    (u64::from(base_one) << ($n / 2), indeterminate)
                }
            }
        }
    )*};
}

small_tower_levels!(2, 4, 8);

macro_rules! wide_tower_levels {
    ($($n:literal, $baseraw:ty;)* $(,)?) => {$(
        impl<S: fgf::TowerSpec> Levels for fgf::FiniteField<2, $n, fgf::Tower<S>>
        where
            S::Base: Levels + fgf::BinaryField + fgf::BinaryDegree<{ $n / 2 }>,
            S::Base: fgf::Field<Raw = $baseraw>,
        {
            const LEVEL_COUNT: usize = <S::Base as Levels>::LEVEL_COUNT + 1;

            fn level_element(level: usize) -> (u64, fgf::Elem<Self>) {
                let base_count = <S::Base as Levels>::LEVEL_COUNT;
                if level < base_count {
                    let (word, base_element) = <S::Base as Levels>::level_element(level);
                    let lifted =
                        fgf::Elem::<Self>::from_components(base_element, fgf::Elem::<S::Base>::ZERO);
                    (word, lifted)
                } else {
                    let indeterminate = fgf::Elem::<Self>::from_components(
                        fgf::Elem::<S::Base>::ZERO,
                        fgf::Elem::<S::Base>::ONE,
                    );
                    let base_one = u64::from(<S::Base as fgf::Field>::ONE_RAW);
                    (base_one << ($n / 2), indeterminate)
                }
            }
        }
    )*};
}

wide_tower_levels!(16, u8; 32, u16; 64, u32;);

/// The manifest facts of one captured presentation.
pub trait Presentation:
    fgf::BinaryField<Raw: Into<u64>> + ByteEnc + AliasDecode + Levels + Sized
{
    const SHORT: &'static str;
}

// ---------------------------------------------------------------------------
// The captured presentation inventory
// ---------------------------------------------------------------------------

pub type Poly1 = fgf::FiniteField<2, 1, fgf::Polynomial<3>>;
pub type Poly2 = fgf::FiniteField<2, 2, fgf::Polynomial<0x7>>;
pub type Poly4 = fgf::FiniteField<2, 4, fgf::Polynomial<0x13>>;
pub type Poly8Aes = fgf::FiniteField<2, 8, fgf::Polynomial<{ fgf::AES }>>;
pub type Poly8Rs = fgf::FiniteField<2, 8, fgf::Polynomial<{ fgf::RS }>>;
pub type Rijndael16F = fgf::FiniteField<2, 16, fgf::Tower<fgf::Rijndael16>>;
pub type Rijndael32F = fgf::FiniteField<2, 32, fgf::Tower<fgf::Rijndael32>>;
pub type Rijndael64F = fgf::FiniteField<2, 64, fgf::Tower<fgf::Rijndael64>>;
pub type FanPaar2F = fgf::FiniteField<2, 2, fgf::Tower<fgf::FanPaar2>>;
pub type FanPaar4F = fgf::FiniteField<2, 4, fgf::Tower<fgf::FanPaar4>>;
pub type FanPaar8F = fgf::FiniteField<2, 8, fgf::Tower<fgf::FanPaar8>>;
pub type FanPaar16F = fgf::FiniteField<2, 16, fgf::Tower<fgf::FanPaar16>>;
pub type FanPaar32F = fgf::FiniteField<2, 32, fgf::Tower<fgf::FanPaar32>>;
pub type FanPaar64F = fgf::FiniteField<2, 64, fgf::Tower<fgf::FanPaar64>>;
pub type Normal2 = fgf::FiniteField<2, 2, fgf::Normal<0x7, 2>>;
pub type Normal4 = fgf::FiniteField<2, 4, fgf::Normal<0x13, 8>>;
pub type Normal8Aes = fgf::FiniteField<2, 8, fgf::Normal<0x11B, 0x20>>;
pub type Normal8Rs = fgf::FiniteField<2, 8, fgf::Normal<0x11D, 0x20>>;
pub type Cantor2 = fgf::FiniteField<2, 2, fgf::Cantor<0x7, 2>>;
pub type Cantor4 = fgf::FiniteField<2, 4, fgf::Cantor<0x13, 8>>;
pub type Cantor8Aes = fgf::FiniteField<2, 8, fgf::Cantor<0x11B, 0x20>>;
pub type Cantor8Rs = fgf::FiniteField<2, 8, fgf::Cantor<0x11D, 0x20>>;

/// A custom degree-8 tower over the degree-four polynomial field:
/// `t^2 + t + 0x8`.
#[derive(Clone, Copy)]
pub struct Custom8Spec;
impl fgf::TowerSpec for Custom8Spec {
    type Base = Poly4;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x8;
    const NAME: &'static str = "custom GF(2^8) tower";
}
pub type Custom8F = fgf::FiniteField<2, 8, fgf::Tower<Custom8Spec>>;

/// A custom degree-16 tower over the Reed-Solomon byte field:
/// `t^2 + 2*t + 0x80`.
#[derive(Clone, Copy)]
pub struct Custom16Spec;
impl fgf::TowerSpec for Custom16Spec {
    type Base = Poly8Rs;
    const LINEAR_COEFFICIENT: u64 = 2;
    const CONSTANT_COEFFICIENT: u64 = 0x80;
    const NAME: &'static str = "custom GF(2^16) tower";
}
pub type Custom16F = fgf::FiniteField<2, 16, fgf::Tower<Custom16Spec>>;

/// A custom degree-16 tower over the AES byte field with a non-unit linear
/// coefficient: `t^2 + 2*t + 0x01`.
#[derive(Clone, Copy)]
pub struct AesLin16Spec;
impl fgf::TowerSpec for AesLin16Spec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 2;
    const CONSTANT_COEFFICIENT: u64 = 0x01;
    const NAME: &'static str = "AES-base non-unit linear tower";
}
pub type AesLin16F = fgf::FiniteField<2, 16, fgf::Tower<AesLin16Spec>>;

/// A custom degree-32 tower over Rijndael16: `t^2 + t + 0x2001`.
#[derive(Clone, Copy)]
pub struct Custom32Spec;
impl fgf::TowerSpec for Custom32Spec {
    type Base = Rijndael16F;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x2001;
    const NAME: &'static str = "custom GF(2^32) tower";
}
pub type Custom32F = fgf::FiniteField<2, 32, fgf::Tower<Custom32Spec>>;

/// A custom degree-64 tower over Rijndael32: `t^2 + t + 0x2ada7f36`.
#[derive(Clone, Copy)]
pub struct Custom64Spec;
impl fgf::TowerSpec for Custom64Spec {
    type Base = Rijndael32F;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x2ada_7f36;
    const NAME: &'static str = "custom GF(2^64) tower";
}
pub type Custom64F = fgf::FiniteField<2, 64, fgf::Tower<Custom64Spec>>;

/// A custom degree-16 tower over the normal-basis byte field: the linear
/// coefficient is the base's identity (all bits set) and the constant term
/// is a trace-one element in basis coordinates.
#[derive(Clone, Copy)]
pub struct NormBase16Spec;
impl fgf::TowerSpec for NormBase16Spec {
    type Base = Normal8Aes;
    const LINEAR_COEFFICIENT: u64 = 0xFF;
    const CONSTANT_COEFFICIENT: u64 = 0x1;
    const NAME: &'static str = "normal-base GF(2^16) tower";
}
pub type NormBase16F = fgf::FiniteField<2, 16, fgf::Tower<NormBase16Spec>>;

/// A second normal-base spec with the same defining data: its descriptor is
/// structurally identical to [`NormBase16Spec`].
#[derive(Clone, Copy)]
pub struct OverNormal16Spec;
impl fgf::TowerSpec for OverNormal16Spec {
    type Base = Normal8Aes;
    const LINEAR_COEFFICIENT: u64 = 0xFF;
    const CONSTANT_COEFFICIENT: u64 = 0x1;
    const NAME: &'static str = "over-normal GF(2^16) tower";
}
pub type OverNormal16F = fgf::FiniteField<2, 16, fgf::Tower<OverNormal16Spec>>;

/// A custom degree-16 tower structurally identical to Rijndael16.
#[derive(Clone, Copy)]
pub struct Same16Spec;
impl fgf::TowerSpec for Same16Spec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x20;
    const NAME: &'static str = "rijndael-shaped custom tower";
}
pub type Same16F = fgf::FiniteField<2, 16, fgf::Tower<Same16Spec>>;

/// An extra custom tower with a non-identity linear coefficient over the AES
/// byte field: `t^2 + 3*t + 0xa0`.
#[derive(Clone, Copy)]
pub struct ExtraALin16Spec;
impl fgf::TowerSpec for ExtraALin16Spec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 0x03;
    const CONSTANT_COEFFICIENT: u64 = 0xA0;
    const NAME: &'static str = "extra non-unit linear tower";
}
pub type ExtraALin16F = fgf::FiniteField<2, 16, fgf::Tower<ExtraALin16Spec>>;

/// An extra custom tower over the AES byte field with the Rijndael relation
/// constant altered by one: `t^2 + t + 0x21`.
#[derive(Clone, Copy)]
pub struct ExtraAConst16Spec;
impl fgf::TowerSpec for ExtraAConst16Spec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x21;
    const NAME: &'static str = "extra altered constant tower";
}
pub type ExtraAConst16F = fgf::FiniteField<2, 16, fgf::Tower<ExtraAConst16Spec>>;

/// An extra custom tower over the Cantor-basis byte field: the linear
/// coefficient is the base identity and the constant term is the transported
/// Rijndael relation constant.
#[derive(Clone, Copy)]
pub struct ExtraCantor16Spec;
impl fgf::TowerSpec for ExtraCantor16Spec {
    type Base = Cantor8Aes;
    const LINEAR_COEFFICIENT: u64 = 0x1;
    const CONSTANT_COEFFICIENT: u64 = 0x80;
    const NAME: &'static str = "extra cantor-base tower";
}
pub type ExtraCantor16F = fgf::FiniteField<2, 16, fgf::Tower<ExtraCantor16Spec>>;

/// Two extra spec types whose descriptors are structurally identical to each
/// other and to no shipped presentation: `t^2 + 5*t + 0x16` over AES.
#[derive(Clone, Copy)]
pub struct ExtraTwin16ASpec;
impl fgf::TowerSpec for ExtraTwin16ASpec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 0x05;
    const CONSTANT_COEFFICIENT: u64 = 0x16;
    const NAME: &'static str = "extra twin tower";
}
pub type ExtraTwin16AF = fgf::FiniteField<2, 16, fgf::Tower<ExtraTwin16ASpec>>;

/// The second twin of the pair; see [`ExtraTwin16ASpec`].
#[derive(Clone, Copy)]
pub struct ExtraTwin16BSpec;
impl fgf::TowerSpec for ExtraTwin16BSpec {
    type Base = Poly8Aes;
    const LINEAR_COEFFICIENT: u64 = 0x05;
    const CONSTANT_COEFFICIENT: u64 = 0x16;
    const NAME: &'static str = "extra twin tower";
}
pub type ExtraTwin16BF = fgf::FiniteField<2, 16, fgf::Tower<ExtraTwin16BSpec>>;

/// Feed the captured presentation inventory to `$mac`.
macro_rules! for_each_presentation {
    ($mac:ident $(, $extra:expr)*) => {
        $mac!($($extra,)* [
            (poly1, Poly1, "poly", "0x3", true),
            (poly2, Poly2, "poly", "0x7", true),
            (poly4, Poly4, "poly", "0x13", true),
            (poly8aes, Poly8Aes, "poly", "0x11B", true),
            (poly8rs, Poly8Rs, "poly", "0x11D", true),
            (rijndael16, Rijndael16F, "tower", "poly8aes 0x1 0x20", false),
            (rijndael32, Rijndael32F, "tower", "rijndael16 0x1 0x2000", false),
            (
                rijndael64, Rijndael64F, "tower", "rijndael32 0x1 0x20000000",
                false
            ),
            (fanpaar2, FanPaar2F, "tower", "poly1 0x1 0x1", false),
            (fanpaar4, FanPaar4F, "tower", "fanpaar2 0x2 0x1", false),
            (fanpaar8, FanPaar8F, "tower", "fanpaar4 0x4 0x1", false),
            (fanpaar16, FanPaar16F, "tower", "fanpaar8 0x10 0x1", false),
            (fanpaar32, FanPaar32F, "tower", "fanpaar16 0x100 0x1", false),
            (fanpaar64, FanPaar64F, "tower", "fanpaar32 0x10000 0x1", false),
            (normal2, Normal2, "normal", "0x7 0x2", true),
            (normal4, Normal4, "normal", "0x13 0x8", true),
            (normal8aes, Normal8Aes, "normal", "0x11B 0x20", true),
            (normal8rs, Normal8Rs, "normal", "0x11D 0x20", true),
            (cantor2, Cantor2, "cantor", "0x7 0x2", true),
            (cantor4, Cantor4, "cantor", "0x13 0x8", true),
            (cantor8aes, Cantor8Aes, "cantor", "0x11B 0x20", true),
            (cantor8rs, Cantor8Rs, "cantor", "0x11D 0x20", true),
            (custom8, Custom8F, "tower", "poly4 0x1 0x8", false),
            (custom16, Custom16F, "tower", "poly8rs 0x2 0x80", false),
            (aeslin16, AesLin16F, "tower", "poly8aes 0x2 0x1", false),
            (custom32, Custom32F, "tower", "rijndael16 0x1 0x2001", false),
            (
                custom64, Custom64F, "tower", "rijndael32 0x1 0x2ada7f36", false
            ),
            (normbase16, NormBase16F, "tower", "normal8aes 0xff 0x1", false),
            (
                overnormal16, OverNormal16F, "tower", "normal8aes 0xff 0x1",
                false
            ),
            (same16, Same16F, "tower", "poly8aes 0x1 0x20", false),
            (extraalin16, ExtraALin16F, "tower", "poly8aes 0x3 0xa0", false),
            (
                extraaconst16, ExtraAConst16F, "tower", "poly8aes 0x1 0x21",
                false
            ),
            (
                extracantor16, ExtraCantor16F, "tower", "cantor8aes 0x1 0x80",
                false
            ),
            (
                extratwin16a, ExtraTwin16AF, "tower", "poly8aes 0x5 0x16", false
            ),
            (
                extratwin16b, ExtraTwin16BF, "tower", "poly8aes 0x5 0x16", false
            )
        ]);
    };
}

macro_rules! define_presentations {
    ([$(($short:ident, $typ:ty, $kind:literal, $data:literal, $flat:literal)),* $(,)?]) => {
        $(
            impl Presentation for $typ {
                const SHORT: &'static str = stringify!($short);
            }
        )*
    };
}

for_each_presentation!(define_presentations);
