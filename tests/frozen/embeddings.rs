//! Frozen embedding fixtures: every ordered manifest pair, with unit-vector
//! images, level-generator images, embeddings, and target analyses asserted
//! against recomputed values.

use crate::common as frozen;

#[allow(unused_imports)]
use frozen::{
    AesLin16F, Cantor2, Cantor4, Cantor8Aes, Cantor8Rs, Custom8F, Custom16F, Custom32F, Custom64F,
    ExtraAConst16F, ExtraALin16F, ExtraCantor16F, ExtraTwin16AF, ExtraTwin16BF, FanPaar2F,
    FanPaar4F, FanPaar8F, FanPaar16F, FanPaar32F, FanPaar64F, NormBase16F, Normal2, Normal4,
    Normal8Aes, Normal8Rs, OverNormal16F, Poly1, Poly2, Poly4, Poly8Aes, Poly8Rs, Rijndael16F,
    Rijndael32F, Rijndael64F, Same16F,
};

use fgf::{Embedding, EmbeddingError, Field, FieldElem};
use frozen::{Levels, Presentation, Records, pair_samples};

/// The manifest order every fixture file follows.
const SOURCE_LIST: [&str; 35] = [
    "poly1",
    "poly2",
    "poly4",
    "poly8aes",
    "poly8rs",
    "rijndael16",
    "rijndael32",
    "rijndael64",
    "fanpaar2",
    "fanpaar4",
    "fanpaar8",
    "fanpaar16",
    "fanpaar32",
    "fanpaar64",
    "normal2",
    "normal4",
    "normal8aes",
    "normal8rs",
    "cantor2",
    "cantor4",
    "cantor8aes",
    "cantor8rs",
    "custom8",
    "custom16",
    "aeslin16",
    "custom32",
    "custom64",
    "normbase16",
    "overnormal16",
    "same16",
    "extraalin16",
    "extraaconst16",
    "extracantor16",
    "extratwin16a",
    "extratwin16b",
];

/// The records of one ordered pair.
struct PairRecords {
    target: String,
    incompatible: Option<(u32, u32)>,
    units: Vec<(usize, u64, u64)>,
    levels: Vec<(usize, u64, u64)>,
    embeds: Vec<(usize, u64, u64)>,
    targets: Vec<TargetRecord>,
}

/// One target record: input index, input word, membership, restriction,
/// Frobenius image, trace, and norm.
type TargetRecord = (usize, u64, u8, Option<u64>, u64, u64, u64);

/// Parse one source file into per-target record groups.
fn parse_source(file: &'static str, text: &'static str) -> Vec<PairRecords> {
    let mut groups: Vec<PairRecords> = Vec::new();
    for (line, number) in Records::new(file, text) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "pair" => {
                let incompatible = match fields[2] {
                    "ok" => None,
                    "incompat" => Some((
                        fields[3]
                            .parse()
                            .unwrap_or_else(|_| panic!("{file}:{number}: bad degree")),
                        fields[4]
                            .parse()
                            .unwrap_or_else(|_| panic!("{file}:{number}: bad degree")),
                    )),
                    other => panic!("{file}:{number}: unknown pair outcome {other}"),
                };
                groups.push(PairRecords {
                    target: fields[1].to_owned(),
                    incompatible,
                    units: Vec::new(),
                    levels: Vec::new(),
                    embeds: Vec::new(),
                    targets: Vec::new(),
                });
            }
            "unit" => groups
                .last_mut()
                .expect("unit record inside a pair")
                .units
                .push((
                    fields[1].parse().expect("unit index"),
                    frozen::word(file, number, fields[2]),
                    frozen::word(file, number, fields[3]),
                )),
            "level" => groups
                .last_mut()
                .expect("level record inside a pair")
                .levels
                .push((
                    fields[1].parse().expect("level index"),
                    frozen::word(file, number, fields[2]),
                    frozen::word(file, number, fields[3]),
                )),
            "embed" => groups
                .last_mut()
                .expect("embed record inside a pair")
                .embeds
                .push((
                    fields[1].parse().expect("embed index"),
                    frozen::word(file, number, fields[2]),
                    frozen::word(file, number, fields[3]),
                )),
            "target" => groups
                .last_mut()
                .expect("target record inside a pair")
                .targets
                .push((
                    fields[1].parse().expect("target index"),
                    frozen::word(file, number, fields[2]),
                    fields[3]
                        .parse()
                        .unwrap_or_else(|_| panic!("{file}:{number}: bad contains flag")),
                    frozen::opt_word(file, number, fields[4]),
                    frozen::word(file, number, fields[5]),
                    frozen::word(file, number, fields[6]),
                    frozen::word(file, number, fields[7]),
                )),
            other => panic!("{file}:{number}: unknown record kind {other}"),
        }
    }
    groups
}

/// Assert one ordered pair's records against recomputed values.
fn check_pair<S: Presentation, T: Presentation>(records: &PairRecords, file: &str) {
    let source_degree = <S as Field>::DEGREE;
    assert_eq!(records.target, T::SHORT, "{file}: pair order");
    match Embedding::<S, T>::new() {
        Err(EmbeddingError::IncompatibleDegree { source, target }) => {
            let (recorded_source, recorded_target) = records
                .incompatible
                .unwrap_or_else(|| panic!("{file}: expected a success group"));
            assert_eq!(source, recorded_source, "{file}: error source degree");
            assert_eq!(target, recorded_target, "{file}: error target degree");
            assert!(
                records.units.is_empty()
                    && records.levels.is_empty()
                    && records.embeds.is_empty()
                    && records.targets.is_empty(),
                "{file}: an error group carries no result records"
            );
        }
        Err(other) => panic!("{file}: unexpected embedding error {other:?}"),
        Ok(embedding) => {
            assert!(
                records.incompatible.is_none(),
                "{file}: expected an error group"
            );

            // Images of the source coordinate unit vectors.
            assert_eq!(
                records.units.len(),
                source_degree as usize,
                "{file}: unit count"
            );
            for (index, word, image) in &records.units {
                assert_eq!(*word, 1u64 << *index, "{file}: unit word");
                let value = fgf::Elem::<S>::from_index(u128::from(*word))
                    .expect("a unit vector fits the field degree");
                assert_eq!(
                    *image,
                    embedding.embed(value).to_index() as u64,
                    "{file}: unit {index} image"
                );
            }

            // Images of the source level generators, in level order.
            assert_eq!(
                records.levels.len(),
                <S as Levels>::LEVEL_COUNT,
                "{file}: level count"
            );
            for (level, word, image) in &records.levels {
                let (expected_word, element) = <S as Levels>::level_element(*level);
                assert_eq!(*word, expected_word, "{file}: level {level} word");
                assert_eq!(
                    *image,
                    embedding.embed(element).to_index() as u64,
                    "{file}: level {level} image"
                );
            }

            // Embeddings of the 64 source sample words.
            let (source_samples, target_samples) =
                pair_samples(source_degree, <T as Field>::DEGREE);
            assert_eq!(records.embeds.len(), 64, "{file}: embed count");
            for (index, word, image) in &records.embeds {
                assert_eq!(*word, source_samples[*index], "{file}: embed {index} input");
                let value = fgf::Elem::<S>::from_index(u128::from(*word))
                    .expect("a masked sample fits the field degree");
                assert_eq!(
                    *image,
                    embedding.embed(value).to_index() as u64,
                    "{file}: embed {index} image"
                );
            }

            // Membership, restriction, Frobenius, trace, and norm of the 64
            // target sample words.
            assert_eq!(records.targets.len(), 64, "{file}: target count");
            for (index, word, contained, restrict, frobenius, trace, norm) in &records.targets {
                assert_eq!(
                    *word, target_samples[*index],
                    "{file}: target {index} input"
                );
                let value = fgf::Elem::<T>::from_index(u128::from(*word))
                    .expect("a masked sample fits the field degree");
                assert_eq!(
                    *contained,
                    u8::from(embedding.contains(value)),
                    "{file}: target {index} contains"
                );
                assert_eq!(
                    *restrict,
                    embedding
                        .restrict(value)
                        .map(|preimage| preimage.to_index() as u64),
                    "{file}: target {index} restrict"
                );
                assert_eq!(
                    *frobenius,
                    embedding.frobenius(value).to_index() as u64,
                    "{file}: target {index} frobenius"
                );
                assert_eq!(
                    *trace,
                    embedding.trace(value).to_index() as u64,
                    "{file}: target {index} trace"
                );
                assert_eq!(
                    *norm,
                    embedding.norm(value).to_index() as u64,
                    "{file}: target {index} norm"
                );
            }
        }
    }
}

/// Every ordered pair of manifest presentations reproduces its records,
/// one test per source presentation so the runner executes sources in
/// parallel and failures name their source.
macro_rules! source_tests {
    ([$(($sname:ident, $styp:ty, $($srest:tt)*)),* $(,)?]) => {$(
        #[test]
        fn $sname() {
            check_source::<$styp>(stringify!($sname));
        }
    )*};
}
for_each_presentation!(source_tests);

/// The fixture text of one source presentation.
fn source_text(name: &str) -> (&'static str, &'static str) {
    match name {
        "poly1" => (
            "embeddings-poly1.txt",
            include_str!("fixtures/embeddings-poly1.txt"),
        ),
        "poly2" => (
            "embeddings-poly2.txt",
            include_str!("fixtures/embeddings-poly2.txt"),
        ),
        "poly4" => (
            "embeddings-poly4.txt",
            include_str!("fixtures/embeddings-poly4.txt"),
        ),
        "poly8aes" => (
            "embeddings-poly8aes.txt",
            include_str!("fixtures/embeddings-poly8aes.txt"),
        ),
        "poly8rs" => (
            "embeddings-poly8rs.txt",
            include_str!("fixtures/embeddings-poly8rs.txt"),
        ),
        "rijndael16" => (
            "embeddings-rijndael16.txt",
            include_str!("fixtures/embeddings-rijndael16.txt"),
        ),
        "rijndael32" => (
            "embeddings-rijndael32.txt",
            include_str!("fixtures/embeddings-rijndael32.txt"),
        ),
        "rijndael64" => (
            "embeddings-rijndael64.txt",
            include_str!("fixtures/embeddings-rijndael64.txt"),
        ),
        "fanpaar2" => (
            "embeddings-fanpaar2.txt",
            include_str!("fixtures/embeddings-fanpaar2.txt"),
        ),
        "fanpaar4" => (
            "embeddings-fanpaar4.txt",
            include_str!("fixtures/embeddings-fanpaar4.txt"),
        ),
        "fanpaar8" => (
            "embeddings-fanpaar8.txt",
            include_str!("fixtures/embeddings-fanpaar8.txt"),
        ),
        "fanpaar16" => (
            "embeddings-fanpaar16.txt",
            include_str!("fixtures/embeddings-fanpaar16.txt"),
        ),
        "fanpaar32" => (
            "embeddings-fanpaar32.txt",
            include_str!("fixtures/embeddings-fanpaar32.txt"),
        ),
        "fanpaar64" => (
            "embeddings-fanpaar64.txt",
            include_str!("fixtures/embeddings-fanpaar64.txt"),
        ),
        "normal2" => (
            "embeddings-normal2.txt",
            include_str!("fixtures/embeddings-normal2.txt"),
        ),
        "normal4" => (
            "embeddings-normal4.txt",
            include_str!("fixtures/embeddings-normal4.txt"),
        ),
        "normal8aes" => (
            "embeddings-normal8aes.txt",
            include_str!("fixtures/embeddings-normal8aes.txt"),
        ),
        "normal8rs" => (
            "embeddings-normal8rs.txt",
            include_str!("fixtures/embeddings-normal8rs.txt"),
        ),
        "cantor2" => (
            "embeddings-cantor2.txt",
            include_str!("fixtures/embeddings-cantor2.txt"),
        ),
        "cantor4" => (
            "embeddings-cantor4.txt",
            include_str!("fixtures/embeddings-cantor4.txt"),
        ),
        "cantor8aes" => (
            "embeddings-cantor8aes.txt",
            include_str!("fixtures/embeddings-cantor8aes.txt"),
        ),
        "cantor8rs" => (
            "embeddings-cantor8rs.txt",
            include_str!("fixtures/embeddings-cantor8rs.txt"),
        ),
        "custom8" => (
            "embeddings-custom8.txt",
            include_str!("fixtures/embeddings-custom8.txt"),
        ),
        "custom16" => (
            "embeddings-custom16.txt",
            include_str!("fixtures/embeddings-custom16.txt"),
        ),
        "aeslin16" => (
            "embeddings-aeslin16.txt",
            include_str!("fixtures/embeddings-aeslin16.txt"),
        ),
        "custom32" => (
            "embeddings-custom32.txt",
            include_str!("fixtures/embeddings-custom32.txt"),
        ),
        "custom64" => (
            "embeddings-custom64.txt",
            include_str!("fixtures/embeddings-custom64.txt"),
        ),
        "normbase16" => (
            "embeddings-normbase16.txt",
            include_str!("fixtures/embeddings-normbase16.txt"),
        ),
        "overnormal16" => (
            "embeddings-overnormal16.txt",
            include_str!("fixtures/embeddings-overnormal16.txt"),
        ),
        "same16" => (
            "embeddings-same16.txt",
            include_str!("fixtures/embeddings-same16.txt"),
        ),
        "extraalin16" => (
            "embeddings-extraalin16.txt",
            include_str!("fixtures/embeddings-extraalin16.txt"),
        ),
        "extraaconst16" => (
            "embeddings-extraaconst16.txt",
            include_str!("fixtures/embeddings-extraaconst16.txt"),
        ),
        "extracantor16" => (
            "embeddings-extracantor16.txt",
            include_str!("fixtures/embeddings-extracantor16.txt"),
        ),
        "extratwin16a" => (
            "embeddings-extratwin16a.txt",
            include_str!("fixtures/embeddings-extratwin16a.txt"),
        ),
        _ => (
            "embeddings-extratwin16b.txt",
            include_str!("fixtures/embeddings-extratwin16b.txt"),
        ),
    }
}

/// Check every target of one source against its file.
fn check_source<S: Presentation>(name: &'static str) {
    let (file, text) = source_text(name);
    let groups = parse_source(file, text);
    assert_eq!(groups.len(), SOURCE_LIST.len(), "{file}: pair count");
    for (position, records) in groups.iter().enumerate() {
        assert_eq!(
            records.target, SOURCE_LIST[position],
            "{file}: pair order at {position}"
        );
    }

    macro_rules! check_targets {
        ($($ttyp:ty),* $(,)?) => {$(
            let position = SOURCE_LIST
                .iter()
                .position(|candidate| *candidate == <$ttyp as Presentation>::SHORT)
                .expect("manifest name");
            check_pair::<S, $ttyp>(&groups[position], file);
        )*};
    }
    check_targets!(
        frozen::Poly1,
        frozen::Poly2,
        frozen::Poly4,
        frozen::Poly8Aes,
        frozen::Poly8Rs,
        frozen::Rijndael16F,
        frozen::Rijndael32F,
        frozen::Rijndael64F,
        frozen::FanPaar2F,
        frozen::FanPaar4F,
        frozen::FanPaar8F,
        frozen::FanPaar16F,
        frozen::FanPaar32F,
        frozen::FanPaar64F,
        frozen::Normal2,
        frozen::Normal4,
        frozen::Normal8Aes,
        frozen::Normal8Rs,
        frozen::Cantor2,
        frozen::Cantor4,
        frozen::Cantor8Aes,
        frozen::Cantor8Rs,
        frozen::Custom8F,
        frozen::Custom16F,
        frozen::AesLin16F,
        frozen::Custom32F,
        frozen::Custom64F,
        frozen::NormBase16F,
        frozen::OverNormal16F,
        frozen::Same16F,
        frozen::ExtraALin16F,
        frozen::ExtraAConst16F,
        frozen::ExtraCantor16F,
        frozen::ExtraTwin16AF,
        frozen::ExtraTwin16BF,
    );
}
