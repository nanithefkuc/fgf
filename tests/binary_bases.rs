//! Binary presentation surfaces outside the arithmetic sweeps: element
//! formatting, custom-extension squaring against a base-field schoolbook, and the
//! wide identity and canonical predicates.

use std::fmt::{self, Write};

use fgf::{
    AES, BinaryField, Cantor, Elem, Embedding, Extension, ExtensionSpec, Field, FieldBuffer,
    FieldElem, FiniteField, Normal, Packed, Polynomial, RS, RangeError, Relation, Residue,
};

type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf32 = fgf::FiniteField<2, 32, fgf::Extension<fgf::Rijndael32, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;

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
            let element = Elem::<F>::from_index(word as u128).expect("in-range coordinates");
            let text = element.to_string();
            assert_eq!(u64::from_str_radix(&text, 16), Ok(word), "{text:?}");
        }
    }
    check::<FiniteField<2, 2, Polynomial<0x7>>>(4);
    check::<FiniteField<2, 2, Normal<0x7, 2>>>(4);
    check::<FiniteField<2, 2, Cantor<0x7, 2>>>(4);
    check::<FiniteField<2, 4, Polynomial<0x13>>>(16);
    check::<FiniteField<2, 4, Normal<0x13, 8>>>(16);
    check::<FiniteField<2, 4, Cantor<0x13, 8>>>(16);
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
    let marker = FiniteField::<2, 8, Normal<0x11B, 0x20>>::default();
    assert_eq!(write!(FullWriter, "{marker:?}"), Err(fmt::Error));

    let coordinate = Elem::<FiniteField<2, 4, Polynomial<0x13>>>::from_index(0x10)
        .expect_err("excess coordinate bits");
    assert_eq!(write!(FullWriter, "{coordinate}"), Err(fmt::Error));

    let degree = Embedding::<Gf16, FiniteField<2, 8, Polynomial<AES>>>::new()
        .expect_err("sixteen does not divide eight");
    assert_eq!(write!(FullWriter, "{degree}"), Err(fmt::Error));

    let embedding = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf16>::new()
        .expect("eight divides sixteen");
    assert_eq!(write!(FullWriter, "{embedding:?}"), Err(fmt::Error));
}

/// `t^2 + 2*t + 0x80` over the Reed–Solomon byte field.
#[derive(Clone, Copy)]
struct BasesCustom16;
impl ExtensionSpec<2> for BasesCustom16 {
    type Base = FiniteField<2, 8, Polynomial<RS>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "bases custom GF(2^16)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<RS>>>(&[0x80, 0x2]);
}

/// `t^2 + t + 0x2001` over [`Gf16`].
#[derive(Clone, Copy)]
struct BasesCustom32;
impl ExtensionSpec<2> for BasesCustom32 {
    type Base = Gf16;
    type Storage = Packed<u32>;
    const NAME: &'static str = "bases custom GF(2^32)";
    const RELATION: Relation = Relation::from_binary_words::<Gf16>(&[0x2001, 0x0001]);
}

/// `t^2 + t + 0x2ada7f36` over [`Gf32`].
#[derive(Clone, Copy)]
struct BasesCustom64;
impl ExtensionSpec<2> for BasesCustom64 {
    type Base = Gf32;
    type Storage = Packed<u64>;
    const NAME: &'static str = "bases custom GF(2^64)";
    const RELATION: Relation = Relation::from_binary_words::<Gf32>(&[0x2ada_7f36, 0x0000_0001]);
}

/// An extension over a normal-basis byte field, whose one is `0x00FF`
/// rather than raw one.
#[derive(Clone, Copy)]
struct BasesNormal16;
impl ExtensionSpec<2> for BasesNormal16 {
    type Base = FiniteField<2, 8, Normal<0x11B, 0x20>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "bases normal-base GF(2^16)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(&[0x1, 0xFF]);
}

/// The square of `a + b*t` under `t^2 = A*t + B`, computed in the base:
/// `(a^2 + B*b^2) + (A*b^2)*t`.
fn schoolbook_square<B: BinaryField>(half: u32, linear: u64, constant: u64, word: u64) -> u64 {
    let base =
        |coordinates: u64| Elem::<B>::from_index(coordinates as u128).expect("base coordinates");
    let mask = (1u64 << half) - 1;
    let low = base(word & mask);
    let high = base(word >> half);
    let low_sq = FieldElem::mul(low, low);
    let high_sq = FieldElem::mul(high, high);
    let new_low = FieldElem::add(low_sq, FieldElem::mul(base(constant), high_sq));
    let new_high = FieldElem::mul(base(linear), high_sq);
    (new_low.to_index() as u64) | ((new_high.to_index() as u64) << half)
}

/// Custom extensions square through the generic extension body; the result
/// matches the base-field schoolbook of the defining relation.
#[test]
fn custom_tower_squares_match_the_base_schoolbook() {
    type F16 = FiniteField<2, 16, Extension<BasesCustom16, 2>>;
    type F32 = FiniteField<2, 32, Extension<BasesCustom32, 2>>;
    type F64 = FiniteField<2, 64, Extension<BasesCustom64, 2>>;
    for raw in 0..=u16::MAX {
        let square = Elem::<F16>::from_raw(raw).square();
        assert_eq!(
            u64::from(square.to_raw()),
            schoolbook_square::<FiniteField<2, 8, Polynomial<RS>>>(8, 2, 0x80, u64::from(raw)),
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
/// for a nonzero `g` — including an extension whose one is not raw one, and
/// canonical encodings agree with element equality.
#[test]
fn wide_identity_and_canonical_predicates_follow_the_field() {
    macro_rules! check {
        // Custom rows: the buffer encoding.
        (custom $field:ty, $samples:expr, $probe:expr) => {
            check!(@run $field, $samples, $probe, |x: Elem<$field>| {
                let mut bytes = vec![0u8; <$field as FieldBuffer>::BYTES];
                <$field as FieldBuffer>::encode(&mut bytes, x);
                bytes
            })
        };
        (@run $field:ty, $samples:expr, $probe:expr, $encode:expr) => {{
            let encode = $encode;
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
                for &y in &elements {
                    assert_eq!(encode(x) == encode(y), x == y);
                }
            }
            assert!(ones >= 1, "the identity is among the samples");
        }};
        // Shipped rows: the inherent `canonical` and byte surface.
        ($field:ty, $samples:expr, $probe:expr) => {
            check!(@run $field, $samples, $probe, |x: Elem<$field>| {
                assert_eq!(x.canonical(), x);
                x.canonical().to_bytes().to_vec()
            })
        };
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
    check!(custom FiniteField<2, 16, Extension<BasesNormal16, 2>>, narrow(64, 0x1d16), 0x1234);
    check!(Gf16, narrow(64, 0x6f16), 0x1234);
    check!(Gf32, medium(64, 0x6f32), 0x1234_5678);
    check!(Gf64, lcg_samples(64, 0x6f64), 0x1234_5678_9abc_def0);
}

/// The two characteristic-two degree-one presentations are distinct types
/// with different lane widths; every residue moves between them through the
/// const residue conversions and keeps its arithmetic.
#[test]
fn residue_two_and_binary_degree_one_transport_by_residue() {
    type Bin = FiniteField<2, 1, Polynomial<3>>;
    type Res = FiniteField<2, 1, Residue>;
    assert_ne!(std::any::TypeId::of::<Bin>(), std::any::TypeId::of::<Res>());
    // Different storage: one byte for the binary row, a u64 residue for the
    // residue row, so their buffers are never interchangeable.
    assert_eq!(size_of::<<Bin as Field>::Raw>(), 1);
    assert_eq!(size_of::<<Res as Field>::Raw>(), 8);

    for a in 0..2u64 {
        let bin = Elem::<Bin>::from_residue(a).expect("residue below two");
        let res = Elem::<Res>::from_residue(bin.to_residue().expect("degree one"))
            .expect("residue below two");
        assert_eq!(res.to_residue(), Some(a));
        let back = Elem::<Bin>::from_residue(res.to_residue().expect("degree one"))
            .expect("residue below two");
        assert_eq!(back, bin);
        for b in 0..2u64 {
            let (bin_b, res_b) = (
                Elem::<Bin>::from_residue(b).expect("residue below two"),
                Elem::<Res>::from_residue(b).expect("residue below two"),
            );
            assert_eq!(bin.add(bin_b).to_residue(), res.add(res_b).to_residue());
            assert_eq!(bin.mul(bin_b).to_residue(), res.mul(res_b).to_residue());
        }
    }
    for out_of_range in [2, 3, 256, u64::MAX] {
        assert_eq!(Elem::<Bin>::from_residue(out_of_range), Err(RangeError));
    }
}
