//! Frozen field fixtures: presentation manifest, samples, generators, and
//! shipped-instance known answers, asserted against recomputed values.

use crate::common as frozen;

#[allow(unused_imports)]
use frozen::{
    AesLin16F, Cantor2, Cantor4, Cantor8Aes, Cantor8Rs, Custom8F, Custom16F, Custom32F, Custom64F,
    ExtraAConst16F, ExtraALin16F, ExtraCantor16F, ExtraTwin16AF, ExtraTwin16BF, FanPaar2F,
    FanPaar4F, FanPaar8F, FanPaar16F, FanPaar32F, FanPaar64F, NormBase16F, Normal2, Normal4,
    Normal8Aes, Normal8Rs, OverNormal16F, Poly1, Poly2, Poly4, Poly8Aes, Poly8Rs, Rijndael16F,
    Rijndael32F, Rijndael64F, Same16F,
};

use fgf::{
    BinaryField, Elem, Field, FieldElem, Goldilocks, HasGenerator, Mersenne31, QuadMersenne31,
};
use frozen::{AliasDecode, ByteEnc, Levels, Records, SplitMix64Reader, samples};

const MANIFEST: &str = include_str!("fixtures/manifest.txt");
const PRESENTATIONS: &str = include_str!("fixtures/presentations.txt");
const GENERATORS: &str = include_str!("fixtures/generators.txt");
const SHIPPED: &str = include_str!("fixtures/shipped.txt");
const NAMED: &str = include_str!("fixtures/named.txt");
const FLAT: &str = include_str!("fixtures/flat.txt");

fn hex(word: u64) -> String {
    format!("{word:#x}")
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use core::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn fields_of<'a>(file: &'static str, number: usize, line: &'a str) -> Vec<&'a str> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    assert!(!fields.is_empty(), "{file}:{number}: empty record");
    fields
}

/// The manifest lists every captured presentation, in order, with its
/// defining data and degree.
#[test]
fn manifest_lists_every_presentation_in_order() {
    let mut lines = Records::new("manifest.txt", MANIFEST);
    let mut degrees: Vec<(&str, u32)> = Vec::new();
    macro_rules! check {
        ([$(($short:ident, $typ:ty, $kind:literal, $data:literal, $flat:literal)),* $(,)?]) => {{
            $(
                let (line, number) = lines.next().expect("manifest line");
                let degree = <$typ as Field>::DEGREE;
                let expected = format!("{} {} {} {}", stringify!($short), $kind, degree, $data);
                assert_eq!(line, expected, "manifest.txt:{number}");
                if $kind == "tower" {
                    // A tower's first defining-data token names its base, which
                    // precedes it in the manifest at half its degree.
                    let base = $data.split_whitespace().next().expect("tower base");
                    let base_degree = degrees
                        .iter()
                        .find(|(name, _)| *name == base)
                        .map(|&(_, degree)| degree)
                        .unwrap_or_else(|| panic!("manifest.txt:{number}: unknown base {base}"));
                    assert_eq!(2 * base_degree, degree, "manifest.txt:{number}: base degree");
                }
                degrees.push((stringify!($short), degree));
            )*
        }};
    }
    for_each_presentation!(check);
    assert!(lines.next().is_none(), "manifest.txt: trailing records");
}

/// Every sample line of one presentation: coordinate word, canonical raw
/// word, byte encoding.
#[test]
fn sample_words_match_recomputed_values() {
    let mut lines = Records::new("presentations.txt", PRESENTATIONS);
    macro_rules! check {
        ([$(($short:ident, $typ:ty, $($rest:tt)*)),* $(,)?]) => {{
            $(
                for (index, word) in samples(<$typ as Field>::DEGREE)
                    .into_iter()
                    .enumerate()
                {
                    let (line, number) = lines.next().expect("sample line");
                    let fields = fields_of("presentations.txt", number, line);
                    assert_eq!(fields[0], stringify!($short), "presentations.txt:{number}");
                    assert_eq!(fields[1], index.to_string(), "presentations.txt:{number}");
                    assert_eq!(fields[2], hex(word), "presentations.txt:{number}");
                    let value = <$typ as BinaryField>::from_coordinates(word)
                        .expect("a masked sample fits the field degree");
                    assert_eq!(
                        fields[3],
                        hex(value.to_raw().into()),
                        "presentations.txt:{number}"
                    );
                    assert_eq!(
                        fields[4],
                        hex_bytes(&<$typ as ByteEnc>::encode_bytes(value)),
                        "presentations.txt:{number}"
                    );
                }
            )*
        }};
    }
    for_each_presentation!(check);
    assert!(
        lines.next().is_none(),
        "presentations.txt: trailing records"
    );
}

/// Identity words and level generators of every presentation.
#[test]
fn generators_match_recomputed_levels() {
    let mut lines = Records::new("generators.txt", GENERATORS);
    macro_rules! check {
        ([$(($short:ident, $typ:ty, $($rest:tt)*)),* $(,)?]) => {{
            $(
                let (line, number) = lines.next().expect("generator line");
                let fields = fields_of("generators.txt", number, line);
                assert_eq!(fields[0], stringify!($short), "generators.txt:{number}");
                assert_eq!(
                    fields[1],
                    <$typ as Levels>::LEVEL_COUNT.to_string(),
                    "generators.txt:{number}"
                );
                let one = <$typ as BinaryField>::to_coordinates(Elem::<$typ>::ONE);
                assert_eq!(fields[2], hex(one), "generators.txt:{number}");
                assert_eq!(
                    fields.len(),
                    3 + <$typ as Levels>::LEVEL_COUNT,
                    "generators.txt:{number}"
                );
                for (level, field) in fields[3..].iter().enumerate() {
                    let (expected, _) = <$typ as Levels>::level_element(level);
                    assert_eq!(field, &hex(expected), "generators.txt:{number}");
                }
            )*
        }};
    }
    for_each_presentation!(check);
    assert!(lines.next().is_none(), "generators.txt: trailing records");
}

/// Alias decoding through the storage constructor for every flat
/// presentation.
#[test]
fn flat_aliases_match_recomputed_decoding() {
    let mut lines = Records::new("flat.txt", FLAT);
    macro_rules! check {
        ([$(($short:ident, $typ:ty, $kind:literal, $data:literal, $flat:literal)),* $(,)?]) => {{
            $(
                if $flat {
                    let degree = <$typ as Field>::DEGREE;
                    let mut inputs = vec![
                        0,
                        1,
                        2,
                        3,
                        0xFF,
                        0xFFFF_FFFF,
                        0xFFFF_FFFF_FFFF_FFFF,
                        frozen::degree_mask(degree),
                        frozen::degree_mask(degree) + 1,
                    ];
                    inputs.extend_from_slice(&frozen::samples(degree)[..8]);
                    for word in inputs {
                        let (line, number) = lines.next().expect("alias line");
                        let fields = fields_of("flat.txt", number, line);
                        assert_eq!(fields[0], stringify!($short), "flat.txt:{number}");
                        assert_eq!(fields[1], hex(word), "flat.txt:{number}");
                        let (raw, bytes) = <$typ as AliasDecode>::decode_alias(word);
                        assert_eq!(fields[2], hex(raw), "flat.txt:{number}");
                        assert_eq!(fields[3], hex_bytes(&bytes), "flat.txt:{number}");
                    }
                }
            )*
        }};
    }
    for_each_presentation!(check);
    assert!(lines.next().is_none(), "flat.txt: trailing records");
}

/// The selected generator as an element, through the crate trait.
trait GeneratorOf: HasGenerator + Sized {
    fn generator() -> Elem<Self> {
        Elem::<Self>::GENERATOR
    }
}
impl<F: HasGenerator> GeneratorOf for F {}

/// Known answers of one shipped tower: samples, products, inverses,
/// generator powers, byte encodings.
macro_rules! check_shipped_tower {
    ($lines:expr, $file:literal, $short:ident, $typ:ty) => {{
        let (line, number) = $lines.next().expect("field line");
        assert_eq!(
            line,
            concat!("field ", stringify!($short)),
            "{}:{number}",
            $file
        );
        let drawn = samples(<$typ as Field>::DEGREE);
        for (index, word) in drawn.into_iter().enumerate() {
            let (line, number) = $lines.next().expect("sample line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "sample", "{}:{number}", $file);
            assert_eq!(fields[1], index.to_string(), "{}:{number}", $file);
            assert_eq!(fields[2], hex(word), "{}:{number}", $file);
        }
        for index in 0..64 {
            let left = <$typ as BinaryField>::from_coordinates(drawn[index]).expect("sample");
            let right =
                <$typ as BinaryField>::from_coordinates(drawn[(index + 1) % 64]).expect("sample");
            let (line, number) = $lines.next().expect("mul line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "mul", "{}:{number}", $file);
            assert_eq!(fields[1], index.to_string(), "{}:{number}", $file);
            assert_eq!(fields[2], hex(drawn[index]), "{}:{number}", $file);
            assert_eq!(
                fields[3],
                hex(drawn[(index + 1) % 64]),
                "{}:{number}",
                $file
            );
            assert_eq!(
                fields[4],
                hex(<$typ as BinaryField>::to_coordinates(FieldElem::mul(
                    left, right
                ))),
                "{}:{number}",
                $file
            );
        }
        for (index, word) in drawn.into_iter().enumerate() {
            let value = <$typ as BinaryField>::from_coordinates(word).expect("sample");
            let (line, number) = $lines.next().expect("inv line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "inv", "{}:{number}", $file);
            assert_eq!(fields[1], index.to_string(), "{}:{number}", $file);
            assert_eq!(fields[2], hex(word), "{}:{number}", $file);
            assert_eq!(
                fields[3],
                hex(<$typ as BinaryField>::to_coordinates(FieldElem::inv(value))),
                "{}:{number}",
                $file
            );
        }
        let generator = <$typ as GeneratorOf>::generator();
        for exponent in 0..64u64 {
            let power = FieldElem::pow(generator, u128::from(exponent));
            let (line, number) = $lines.next().expect("pow line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "pow", "{}:{number}", $file);
            assert_eq!(fields[1], exponent.to_string(), "{}:{number}", $file);
            assert_eq!(
                fields[2],
                hex(<$typ as BinaryField>::to_coordinates(power)),
                "{}:{number}",
                $file
            );
        }
        for (index, word) in drawn.into_iter().enumerate() {
            let power = FieldElem::pow(generator, u128::from(word));
            let (line, number) = $lines.next().expect("powexp line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "powexp", "{}:{number}", $file);
            assert_eq!(fields[1], index.to_string(), "{}:{number}", $file);
            assert_eq!(fields[2], hex(word), "{}:{number}", $file);
            assert_eq!(
                fields[3],
                hex(<$typ as BinaryField>::to_coordinates(power)),
                "{}:{number}",
                $file
            );
        }
        for (index, word) in drawn.into_iter().enumerate() {
            let value = <$typ as BinaryField>::from_coordinates(word).expect("sample");
            let (line, number) = $lines.next().expect("bytes line");
            let fields = fields_of($file, number, line);
            assert_eq!(fields[0], "bytes", "{}:{number}", $file);
            assert_eq!(fields[1], index.to_string(), "{}:{number}", $file);
            assert_eq!(fields[2], hex(word), "{}:{number}", $file);
            assert_eq!(
                fields[3],
                hex_bytes(&<$typ as ByteEnc>::encode_bytes(value)),
                "{}:{number}",
                $file
            );
        }
    }};
}

/// Shipped-instance records reproduce from the current crate.
#[test]
fn shipped_instances_match_known_answers() {
    let mut lines = Records::new("shipped.txt", SHIPPED);
    check_shipped_tower!(lines, "shipped.txt", rijndael16, Rijndael16F);
    check_shipped_tower!(lines, "shipped.txt", rijndael32, Rijndael32F);
    check_shipped_tower!(lines, "shipped.txt", rijndael64, Rijndael64F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar2, FanPaar2F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar4, FanPaar4F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar8, FanPaar8F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar16, FanPaar16F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar32, FanPaar32F);
    check_shipped_tower!(lines, "shipped.txt", fanpaar64, FanPaar64F);
    check_quad_mersenne(&mut lines);
    assert!(lines.next().is_none(), "shipped.txt: trailing records");
}

/// The quadratic Mersenne extension: canonical limbs beside every input.
fn check_quad_mersenne(lines: &mut Records<'_>) {
    type Q = QuadMersenne31;
    let (line, number) = lines.next().expect("field line");
    assert_eq!(line, "field quadmersenne31", "shipped.txt:{number}");
    let prime = u64::from(fgf::mersenne31::MODULUS);
    let mut source = SplitMix64Reader::new();
    let mut limbs = [(0u32, 0u32); 64];
    for (re, im) in &mut limbs {
        *re = u32::try_from(source.next() % prime).expect("below the prime");
        *im = u32::try_from(source.next() % prime).expect("below the prime");
    }
    let elements: [Elem<Q>; 64] =
        core::array::from_fn(|index| Elem::<Q>::from_raw([limbs[index].0, limbs[index].1]));
    for (index, (re, im)) in limbs.into_iter().enumerate() {
        let (line, number) = lines.next().expect("sample line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "sample", "shipped.txt:{number}");
        assert_eq!(fields[1], index.to_string(), "shipped.txt:{number}");
        assert_eq!(fields[2], hex(u64::from(re)), "shipped.txt:{number}");
        assert_eq!(fields[3], hex(u64::from(im)), "shipped.txt:{number}");
        let raw = elements[index].to_raw();
        assert_eq!(fields[4], hex(u64::from(raw[0])), "shipped.txt:{number}");
        assert_eq!(fields[5], hex(u64::from(raw[1])), "shipped.txt:{number}");
    }
    for index in 0..64 {
        let left = elements[index];
        let right = elements[(index + 1) % 64];
        let product = FieldElem::mul(left, right);
        let (line, number) = lines.next().expect("mul line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "mul", "shipped.txt:{number}");
        assert_eq!(fields[1], index.to_string(), "shipped.txt:{number}");
        assert_eq!(
            fields[2],
            hex(u64::from(left.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[3],
            hex(u64::from(left.to_raw()[1])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[4],
            hex(u64::from(right.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[5],
            hex(u64::from(right.to_raw()[1])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[6],
            hex(u64::from(product.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[7],
            hex(u64::from(product.to_raw()[1])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[8],
            hex_bytes(&left.to_bytes()),
            "shipped.txt:{number}"
        );
    }
    for (index, element) in elements.into_iter().enumerate() {
        let inverse = FieldElem::inv(element);
        let (line, number) = lines.next().expect("inv line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "inv", "shipped.txt:{number}");
        assert_eq!(fields[1], index.to_string(), "shipped.txt:{number}");
        assert_eq!(
            fields[2],
            hex(u64::from(element.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[3],
            hex(u64::from(element.to_raw()[1])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[4],
            hex(u64::from(inverse.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[5],
            hex(u64::from(inverse.to_raw()[1])),
            "shipped.txt:{number}"
        );
    }
    let generator = Elem::<Q>::GENERATOR;
    for exponent in 0..64u64 {
        let power = FieldElem::pow(generator, u128::from(exponent));
        let (line, number) = lines.next().expect("pow line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "pow", "shipped.txt:{number}");
        assert_eq!(fields[1], exponent.to_string(), "shipped.txt:{number}");
        assert_eq!(
            fields[2],
            hex(u64::from(power.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[3],
            hex(u64::from(power.to_raw()[1])),
            "shipped.txt:{number}"
        );
    }
    for (index, element) in elements.into_iter().enumerate() {
        let exponent = element.to_raw()[0];
        let power = FieldElem::pow(generator, u128::from(exponent));
        let (line, number) = lines.next().expect("powexp line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "powexp", "shipped.txt:{number}");
        assert_eq!(fields[1], index.to_string(), "shipped.txt:{number}");
        assert_eq!(fields[2], hex(u64::from(exponent)), "shipped.txt:{number}");
        assert_eq!(
            fields[3],
            hex(u64::from(power.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[4],
            hex(u64::from(power.to_raw()[1])),
            "shipped.txt:{number}"
        );
    }
    for (index, element) in elements.into_iter().enumerate() {
        let (line, number) = lines.next().expect("bytes line");
        let fields = fields_of("shipped.txt", number, line);
        assert_eq!(fields[0], "bytes", "shipped.txt:{number}");
        assert_eq!(fields[1], index.to_string(), "shipped.txt:{number}");
        assert_eq!(
            fields[2],
            hex(u64::from(element.to_raw()[0])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[3],
            hex(u64::from(element.to_raw()[1])),
            "shipped.txt:{number}"
        );
        assert_eq!(
            fields[4],
            hex_bytes(&element.to_bytes()),
            "shipped.txt:{number}"
        );
    }
}

/// Named prime fields: byte decoding, raw-alias decoding, known answers.
#[test]
fn named_prime_fields_match_known_answers() {
    let mut lines = Records::new("named.txt", NAMED);
    check_named::<Mersenne31>(&mut lines);
    check_named::<Goldilocks>(&mut lines);
    assert!(lines.next().is_none(), "named.txt: trailing records");
}

/// The fixture facts of one named prime field.
trait PrimeRow: HasGenerator + Field {
    const NAME: &'static str;
    const PRIME: u64;
    fn wrap(word: u64) -> Elem<Self>;
    fn raw(value: Elem<Self>) -> u64;
    fn encoded(value: Elem<Self>) -> Vec<u8>;
}

impl PrimeRow for Mersenne31 {
    const NAME: &'static str = "mersenne31";
    const PRIME: u64 = fgf::mersenne31::MODULUS as u64;
    fn wrap(word: u64) -> Elem<Self> {
        Elem::<Self>::from_raw(word as u32)
    }
    fn raw(value: Elem<Self>) -> u64 {
        u64::from(value.to_raw())
    }
    fn encoded(value: Elem<Self>) -> Vec<u8> {
        value.to_bytes().into_iter().collect()
    }
}

impl PrimeRow for Goldilocks {
    const NAME: &'static str = "goldilocks";
    const PRIME: u64 = fgf::goldilocks::MODULUS;
    fn wrap(word: u64) -> Elem<Self> {
        Elem::<Self>::from_raw(word)
    }
    fn raw(value: Elem<Self>) -> u64 {
        value.to_raw()
    }
    fn encoded(value: Elem<Self>) -> Vec<u8> {
        value.to_bytes().into_iter().collect()
    }
}

fn check_named<F: PrimeRow>(lines: &mut Records<'_>) {
    let mut source = SplitMix64Reader::new();
    let mut inputs = [0u64; 64];
    for input in &mut inputs {
        *input = if F::PRIME <= u64::from(u32::MAX) {
            source.next() & 0xFFFF_FFFF
        } else {
            source.next()
        };
    }
    for (index, word) in inputs.into_iter().enumerate() {
        let value = F::wrap(word);
        let (line, number) = lines.next().expect("sample line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "sample", "named.txt:{number}");
        assert_eq!(fields[2], index.to_string(), "named.txt:{number}");
        assert_eq!(fields[3], hex(word), "named.txt:{number}");
        assert_eq!(fields[4], hex(F::raw(value)), "named.txt:{number}");
        assert_eq!(
            fields[5],
            hex_bytes(&F::encoded(value)),
            "named.txt:{number}"
        );
    }
    // Every word is representable for both named primes: Goldilocks is
    // near two to the sixty-fourth, so no multiple of its prime above the
    // first fits a machine word.
    let boundary: [u64; 8] = [
        0,
        1,
        F::PRIME - 1,
        F::PRIME,
        F::PRIME + 1,
        F::PRIME + 0x1_0000,
        F::PRIME.rotate_left(1),
        u64::MAX,
    ];
    for word in boundary {
        let value = F::wrap(word);
        let (line, number) = lines.next().expect("alias line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "alias", "named.txt:{number}");
        assert_eq!(fields[2], hex(word), "named.txt:{number}");
        assert_eq!(fields[3], hex(F::raw(value)), "named.txt:{number}");
        assert_eq!(
            fields[4],
            hex_bytes(&F::encoded(value)),
            "named.txt:{number}"
        );
    }
    for index in 0..64 {
        let left = F::wrap(inputs[index]);
        let right = F::wrap(inputs[(index + 1) % 64]);
        let (line, number) = lines.next().expect("mul line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "mul", "named.txt:{number}");
        assert_eq!(fields[2], index.to_string(), "named.txt:{number}");
        assert_eq!(fields[3], hex(inputs[index]), "named.txt:{number}");
        assert_eq!(
            fields[4],
            hex(inputs[(index + 1) % 64]),
            "named.txt:{number}"
        );
        assert_eq!(
            fields[5],
            hex(F::raw(FieldElem::mul(left, right))),
            "named.txt:{number}"
        );
    }
    for (index, word) in inputs.into_iter().enumerate() {
        let value = F::wrap(word);
        let (line, number) = lines.next().expect("inv line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "inv", "named.txt:{number}");
        assert_eq!(fields[2], index.to_string(), "named.txt:{number}");
        assert_eq!(fields[3], hex(word), "named.txt:{number}");
        assert_eq!(
            fields[4],
            hex(F::raw(FieldElem::inv(value))),
            "named.txt:{number}"
        );
    }
    let generator = Elem::<F>::GENERATOR;
    for exponent in 0..64u64 {
        let power = FieldElem::pow(generator, u128::from(exponent));
        let (line, number) = lines.next().expect("pow line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "pow", "named.txt:{number}");
        assert_eq!(fields[2], exponent.to_string(), "named.txt:{number}");
        assert_eq!(fields[3], hex(F::raw(power)), "named.txt:{number}");
    }
    for (index, word) in inputs.into_iter().enumerate() {
        let power = FieldElem::pow(generator, u128::from(word));
        let (line, number) = lines.next().expect("powexp line");
        let fields = fields_of("named.txt", number, line);
        assert_eq!(fields[0], <F as PrimeRow>::NAME, "named.txt:{number}");
        assert_eq!(fields[1], "powexp", "named.txt:{number}");
        assert_eq!(fields[2], index.to_string(), "named.txt:{number}");
        assert_eq!(fields[3], hex(word), "named.txt:{number}");
        assert_eq!(fields[4], hex(F::raw(power)), "named.txt:{number}");
    }
}
