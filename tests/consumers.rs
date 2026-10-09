//! Downstream consumer matrix over the public API.
//!
//! Compiles and runs exactly what a downstream crate can see from outside:
//! every shipped row through the generic [`Field`] surface, the degree-one
//! rows through [`PrimeField`], every wire-format row through
//! [`FieldBuffer`], every dispatchable row through [`FieldKernels`], const
//! constructors and arithmetic in real const contexts, and generic code
//! whose raw word type is a primitive. Compile-fail counterparts live in
//! `tests/compile_fail/` and run through `just compile-fail`.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
use fgf::field::{Field, FieldBuffer, FieldElem};
use fgf::poly::{AES, RS};
use fgf::{
    BinaryRepr, Cantor, Elem, FieldKernels, FiniteField, FiniteFieldRepr, Goldilocks, Mersenne31,
    Normal, Polynomial, PrimeField, QuadMersenne31, Residue, Rijndael16, Rijndael32, Rijndael64,
    ops,
};

type Residue2 = FiniteField<2, 1, Residue>;
type Residue7 = FiniteField<7, 1, Residue>;
type ResiduePrime = FiniteField<1_000_000_007, 1, Residue>;
type ResidueWide = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1, Residue>;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Gf32 = FiniteField<2, 32, Extension<Rijndael32, 2>>;
type Gf64 = FiniteField<2, 64, Extension<Rijndael64, 2>>;
type FanPaar2Row = FiniteField<2, 2, Extension<fgf::FanPaar2, 2>>;
type FanPaar4Row = FiniteField<2, 4, Extension<fgf::FanPaar4, 2>>;
type FanPaar8Row = FiniteField<2, 8, Extension<fgf::FanPaar8, 2>>;
type FanPaar16Row = FiniteField<2, 16, Extension<fgf::FanPaar16, 2>>;
type FanPaar32Row = FiniteField<2, 32, Extension<fgf::FanPaar32, 2>>;
type FanPaar64Row = FiniteField<2, 64, Extension<fgf::FanPaar64, 2>>;

/// A downstream extension specification over a shipped byte field with the
/// Rijndael relation `t^2 + t + 0x20`, reaching the library-owned kernels.
#[derive(Clone, Copy)]
struct CustomSpec;

impl ExtensionSpec<2> for CustomSpec {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "custom GF(2^16)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<AES>>>(&[0x20, 1]);
}

type CustomGf16 = FiniteField<2, 16, Extension<CustomSpec, 2>>;

/// Every byte-buffer row at degree eight and above, each driven through
/// both the buffer and the packed-operation surface.
macro_rules! byte_rows {
    ($check:ident) => {
        byte_rows!(@each $check;
            FiniteField<2, 8, Polynomial<AES>> => 0x53,
            FiniteField<2, 8, Polynomial<RS>> => 0x53,
            FiniteField<2, 8, Normal<0x11B, 0x20>> => 0x53,
            FiniteField<2, 8, Cantor<0x11B, 0x20>> => 0x53,
            FanPaar8Row => 0xAB,
            FanPaar16Row => 0x0108,
            FanPaar32Row => 0x5555_5555,
            FanPaar64Row => 0x00C0_FFEE,
            Gf16 => 0x1B70,
            Gf32 => 0x0F0F_0F0F,
            Gf64 => 0xFFFF_0000_0000_FFFF,
            CustomGf16 => 0x0108,
        );
    };
    (@each $check:ident; $($row:ty => $raw:expr,)*) => {
        $($check::<$row>(Elem::<$row>::from_raw($raw));)*
    };
}

/// Drive the generic `Field` surface end to end: identities, arithmetic,
/// inversion totality, and exponentiation, all through the trait.
fn exercise_field<F: Field>(x: F::Raw, y: F::Raw) {
    let zero = Elem::<F>::ZERO;
    let one = Elem::<F>::ONE;
    let left = Elem::<F>::from_raw(x);
    let right = Elem::<F>::from_raw(y);

    assert!(zero.is_zero());
    assert!(one.is_one());
    assert_eq!(left.mul(one), left);
    assert_eq!(left.add(zero), left);
    assert_eq!(left.sub(left), zero);
    assert_eq!(left.add(left.neg()), zero);
    assert_eq!(left.mul(right), right.mul(left));
    assert_eq!(left.mul(right.inv()), left.div(right));
    assert_eq!(left.pow(0), one);
    // The total conventions hold for every element produced here.
    assert_eq!(zero.inv(), zero);
    assert_eq!(left.div(zero), zero);
    // Default is the additive identity.
    assert_eq!(Elem::<F>::default(), zero);
}

/// Degree-one rows: the `PrimeField` marker rides on the `Field` surface.
fn exercise_prime_field<F: PrimeField>() {
    assert_eq!(<F as Field>::DEGREE, 1);
    let one = Elem::<F>::ONE;
    let base = Elem::<F>::from_raw(one.to_raw());
    assert_eq!(base, one);
    exercise_field::<F>(one.to_raw(), one.to_raw());
}

/// Wire-format rows: encode and decode round-trip canonically, and an
/// all-zero lane decodes to the additive identity.
fn exercise_buffer<F: FieldBuffer>(value: Elem<F>) {
    let mut bytes = vec![0u8; F::BYTES];
    F::encode(&mut bytes, value);
    assert_eq!(F::decode(&bytes), value);
    let zero = vec![0u8; F::BYTES];
    assert!(F::decode(&zero).is_zero());
    assert_eq!(F::STORAGE_BITS, (8 * F::BYTES) as u32);
}

/// Dispatchable rows: one packed operation per row through `ops`, plus the
/// backend report every consumer can query.
fn exercise_kernels<F: FieldKernels>(coeff: Elem<F>) {
    let len = 8 * F::BYTES;
    let mut src = vec![0u8; len];
    F::encode(&mut src[..F::BYTES], coeff);
    let mut dst = vec![0u8; len];
    ops::mul_add::<F>(&mut dst, coeff, &src);
    let mut want = vec![0u8; len];
    for (d, s) in want
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).add(F::decode(s).mul(coeff)));
    }
    assert_eq!(dst, want, "{}", <F as Field>::NAME);
    // The prepared form carries its coefficient through.
    let prepared = ops::Coeff::<F>::new(coeff);
    assert_eq!(prepared.value(), coeff);
    let _ = fgf::backend_for::<F>();
}

#[test]
fn every_field_row_drives_the_generic_surface() {
    // Binary polynomial presentations at degrees one, two, four, eight.
    exercise_field::<FiniteField<2, 1, Polynomial<3>>>(0, 1);
    exercise_field::<FiniteField<2, 1, Polynomial<3>>>(1, 1);
    exercise_field::<FiniteField<2, 2, Polynomial<0x7>>>(0x2, 0x1);
    exercise_field::<FiniteField<2, 4, Polynomial<0x13>>>(0x9, 0x5);
    exercise_field::<FiniteField<2, 8, Polynomial<AES>>>(0x53, 0xCA);
    exercise_field::<FiniteField<2, 8, Polynomial<RS>>>(0xFF, 0x01);
    // Ordered bases at degrees two, four, and eight.
    exercise_field::<FiniteField<2, 2, Normal<0x7, 0x2>>>(0x1, 0x2);
    exercise_field::<FiniteField<2, 2, Cantor<0x7, 0x2>>>(0x1, 0x3);
    exercise_field::<FiniteField<2, 4, Normal<0x13, 0x8>>>(0x7, 0x3);
    exercise_field::<FiniteField<2, 4, Cantor<0x13, 0xE>>>(0x7, 0x3);
    exercise_field::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(0x53, 0x01);
    exercise_field::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>(0x53, 0xCA);
    // Shipped Fan-Paar rows at every degree, and the Rijndael rows.
    exercise_field::<FanPaar2Row>(0x2, 0x3);
    exercise_field::<FanPaar4Row>(0x9, 0x5);
    exercise_field::<FanPaar8Row>(0xAB, 0xCD);
    exercise_field::<FanPaar16Row>(0x0108, 0x1B70);
    exercise_field::<FanPaar32Row>(0x5555_5555, 0x0F0F_0F0F);
    exercise_field::<FanPaar64Row>(0x0123_4567_89AB_CDEF, 7);
    exercise_field::<Gf16>(0x0108, 0x1B70);
    exercise_field::<Gf32>(0x5555_5555, 0x0F0F_0F0F);
    exercise_field::<Gf64>(0x0123_4567_89AB_CDEF, 0xFFFF_0000_0000_FFFF);
    // A downstream extension specification over a shipped byte field.
    exercise_field::<CustomGf16>(0x0108, 0x1B70);
    // Named prime rows and the quadratic extension.
    exercise_field::<Mersenne31>(0x5555_5555, 0x71C7_1C71);
    exercise_field::<Goldilocks>(0xFFFF_FFFF_0000_0000, 7);
    exercise_field::<QuadMersenne31>([0x5555_5555, 0x1234_5678], [0x7FFF_FFFF, 0x7FFF_FFFF]);
    // Residue rows across the characteristic range.
    exercise_field::<Residue2>(1, 1);
    exercise_field::<Residue7>(5, 6);
    exercise_field::<ResiduePrime>(123_456_789, 987_654_321);
    exercise_field::<ResidueWide>(u64::MAX, 1 << 63);
}

#[test]
fn degree_one_rows_are_prime_fields() {
    exercise_prime_field::<Residue2>();
    exercise_prime_field::<Residue7>();
    exercise_prime_field::<ResiduePrime>();
    exercise_prime_field::<ResidueWide>();
    exercise_prime_field::<Mersenne31>();
    exercise_prime_field::<Goldilocks>();
    exercise_prime_field::<FiniteField<2, 1, Polynomial<3>>>();
}

#[test]
fn binary_rows_keep_fixed_raw_words_under_generic_presentations() {
    // Generic over the presentation at a fixed binary degree: the raw word
    // stays the row's primitive, and both characteristic-two degree-one
    // presentations coexist with their own widths.
    fn one_raw<R: BinaryRepr<8>>() -> u8 {
        <FiniteField<2, 8, R> as Field>::ONE_RAW
    }
    fn degree_one<R: BinaryRepr<1>>() -> u8 {
        <FiniteField<2, 1, R> as Field>::ONE_RAW
    }
    fn label<const N: u32, R: BinaryRepr<N>>() -> &'static str {
        <R as FiniteFieldRepr<2, N>>::NAME
    }
    assert_eq!(one_raw::<Polynomial<AES>>(), 1);
    assert_eq!(one_raw::<Normal<0x11B, 0x20>>(), 0xFF);
    assert_eq!(degree_one::<Polynomial<3>>(), 1);
    assert_eq!(<Residue2 as Field>::ONE_RAW, 1u64);
    assert_eq!(label::<8, Polynomial<AES>>(), "GF(2^8)/0x11B");
    assert_eq!(label::<4, Cantor<0x13, 0xE>>(), "cantor");
}

#[test]
fn wire_format_rows_round_trip() {
    exercise_buffer::<Mersenne31>(Elem::<Mersenne31>::from_raw(0x5555_5555));
    exercise_buffer::<Goldilocks>(Elem::<Goldilocks>::from_raw(7));
    let quad = Elem::<QuadMersenne31>::from_raw([3, 5]);
    exercise_buffer::<QuadMersenne31>(quad);
    byte_rows!(exercise_buffer);
    exercise_buffer::<Residue7>(Elem::<Residue7>::from_residue(6).expect("in range"));
    exercise_buffer::<ResiduePrime>(Elem::<ResiduePrime>::from_raw(123_456_789));
    exercise_buffer::<ResidueWide>(Elem::<ResidueWide>::from_raw(u64::MAX));
}

#[test]
fn dispatchable_rows_drive_packed_operations() {
    exercise_kernels::<Mersenne31>(Elem::<Mersenne31>::from_raw(7));
    exercise_kernels::<Goldilocks>(Elem::<Goldilocks>::from_raw(7));
    exercise_kernels::<QuadMersenne31>(Elem::<QuadMersenne31>::from_raw([13, 17]));
    byte_rows!(exercise_kernels);
    exercise_kernels::<Residue2>(Elem::<Residue2>::ONE);
    exercise_kernels::<Residue7>(Elem::<Residue7>::from_residue(3).expect("in range"));
    exercise_kernels::<ResiduePrime>(Elem::<ResiduePrime>::from_raw(123_456_789));
    exercise_kernels::<ResidueWide>(Elem::<ResidueWide>::from_raw(u64::MAX));
}

// Const evaluation of the constructors, arithmetic, and conversions: these
// items are computed during compilation of this test binary.
const RESIDUE_SEVEN_INVERSE: Elem<Residue7> = Elem::<Residue7>::from_raw(2).inv();
const _: () = assert!(RESIDUE_SEVEN_INVERSE.to_raw() == 4);

const RESIDUE_WIDE_SUM: Elem<ResidueWide> =
    Elem::<ResidueWide>::from_raw(u64::MAX).add(Elem::<ResidueWide>::from_raw(1));
const _: () = assert!(RESIDUE_WIDE_SUM.to_raw() == 59);

const RESIDUE_REJECTED: bool = Elem::<Residue7>::from_residue(7).is_err();
const RESIDUE_ACCEPTED: bool = Elem::<Residue7>::from_residue(3).is_ok();
const _: () = assert!(RESIDUE_REJECTED);
const _: () = assert!(RESIDUE_ACCEPTED);

const NAMED_FIELD_INVERSE: Elem<Mersenne31> = Elem::<Mersenne31>::from_raw(2).inv();
const _: () = assert!(NAMED_FIELD_INVERSE.to_raw() == 0x4000_0000);

const FAMILY_ADD: Elem<FiniteField<7, 1, Residue>> =
    Elem::<FiniteField<7, 1, Residue>>::from_raw(6)
        .add(Elem::<FiniteField<7, 1, Residue>>::from_raw(6));
const _: () = assert!(FAMILY_ADD.to_raw() == 5);

const BINARY_RESIDUE: Option<u64> = match Elem::<FiniteField<2, 1, Polynomial<3>>>::from_residue(1)
{
    Ok(value) => value.add(value).add(value).to_residue(),
    Err(_) => None,
};
const _: () = assert!(matches!(BINARY_RESIDUE, Some(1)));

#[test]
fn const_contexts_evaluate() {
    // Runtime reads of the compile-time values above. The rejected and
    // accepted conversion results are compile-time checked already.
    assert_eq!(RESIDUE_SEVEN_INVERSE.to_residue(), Some(4));
    assert_eq!(RESIDUE_WIDE_SUM.to_residue(), Some(59));
    assert_eq!(NAMED_FIELD_INVERSE.to_raw(), 0x4000_0000);
    assert_eq!(FAMILY_ADD.to_residue(), Some(5));
}

/// Every shipped binary extension instance with two nonzero raw words.
macro_rules! shipped_binary_instances {
    ($mac:ident) => {
        $mac!(RIJNDAEL16_CONST, Gf16, 0x1B70, 0x0108);
        $mac!(RIJNDAEL32_CONST, Gf32, 0x5555_5555, 0x0F0F_0F0F);
        $mac!(
            RIJNDAEL64_CONST,
            Gf64,
            0x0123_4567_89AB_CDEF,
            0xFFFF_0000_0000_FFFF
        );
        $mac!(FAN_PAAR2_CONST, FanPaar2Row, 0x2, 0x3);
        $mac!(FAN_PAAR4_CONST, FanPaar4Row, 0x9, 0x5);
        $mac!(FAN_PAAR8_CONST, FanPaar8Row, 0xAB, 0xCD);
        $mac!(FAN_PAAR16_CONST, FanPaar16Row, 0x0108, 0x1B70);
        $mac!(FAN_PAAR32_CONST, FanPaar32Row, 0x5555_5555, 0x0F0F_0F0F);
        $mac!(FAN_PAAR64_CONST, FanPaar64Row, 0x0123_4567_89AB_CDEF, 7);
    };
}

/// The inherent const surface of one shipped binary instance, evaluated
/// during compilation: a combined value, the field identities, the byte
/// round trip, and the generator's order.
macro_rules! const_binary_surface {
    ($name:ident, $row:ty, $x:expr, $y:expr) => {
        const $name: Elem<$row> = Elem::<$row>::from_raw($x)
            .mul(Elem::<$row>::from_raw($y))
            .inv()
            .pow(3);
        const _: () = {
            let x = Elem::<$row>::from_raw($x);
            let y = Elem::<$row>::from_raw($y);
            assert!(x.mul(x.inv()).is_one());
            assert!(x.div(y).mul(y).to_raw() == x.to_raw());
            assert!(x.add(y).sub(y).to_raw() == x.to_raw());
            assert!(x.neg().add(x).is_zero());
            assert!(x.square().to_raw() == x.mul(x).to_raw());
            assert!(x.canonical().to_raw() == x.to_raw());
            assert!(Elem::<$row>::from_bytes(x.to_bytes()).to_raw() == x.to_raw());
            assert!($name.pow(3).mul(x.mul(y).pow(9)).is_one());
            let generator = Elem::<$row>::GENERATOR;
            assert!(generator.pow(<$row as Field>::ORDER - 1).is_one());
            assert!(!generator.pow((<$row as Field>::ORDER - 1) / 3).is_one());
        };
    };
}

shipped_binary_instances!(const_binary_surface);

/// `mul_alpha` multiplies by the level indeterminate: component one set,
/// the word `1 << (N / 2)`.
const _: () = {
    let x8 = Elem::<FanPaar8Row>::from_raw(0xAB);
    assert!(x8.mul_alpha().to_raw() == x8.mul(Elem::<FanPaar8Row>::from_raw(0x10)).to_raw());
    let x16 = Elem::<FanPaar16Row>::from_raw(0x1B70);
    assert!(x16.mul_alpha().to_raw() == x16.mul(Elem::<FanPaar16Row>::from_raw(0x100)).to_raw());
    let x32 = Elem::<FanPaar32Row>::from_raw(0x5555_5555);
    assert!(x32.mul_alpha().to_raw() == x32.mul(Elem::<FanPaar32Row>::from_raw(0x1_0000)).to_raw());
};

/// The wide component form: component zero is the constant term, component
/// one the indeterminate, and the split round-trips.
macro_rules! const_components {
    ($($row:ty, $half:literal;)*) => {$(
        const _: () = {
            type Base = <$row as ExtensionField<2>>::Base;
            let one = Elem::<$row>::from_components([Elem::<Base>::ONE, Elem::<Base>::ZERO]);
            assert!(one.is_one());
            let t = Elem::<$row>::from_components([Elem::<Base>::ZERO, Elem::<Base>::ONE]);
            assert!(t.to_raw() == 1 << $half);
            let x = Elem::<$row>::GENERATOR;
            let [low, high] = x.to_components();
            assert!(Elem::<$row>::from_components([low, high]).to_raw() == x.to_raw());
            assert!(Elem::<$row>::from_components([low, Elem::<Base>::ZERO])
                .add(t.mul(Elem::<$row>::from_components([high, Elem::<Base>::ZERO])))
                .to_raw()
                == x.to_raw());
        };
    )*};
}

const_components!(
    Gf16, 8;
    Gf32, 16;
    Gf64, 32;
    FanPaar16Row, 8;
    FanPaar32Row, 16;
    FanPaar64Row, 32;
);

const QUAD_CONST: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw([3, 5])
    .mul(Elem::<QuadMersenne31>::from_raw([0x7FFF_FFFE, 12]))
    .inv()
    .pow(3);
const _: () = {
    type Q = QuadMersenne31;
    let x = Elem::<Q>::from_raw([3, 5]);
    let y = Elem::<Q>::from_raw([0x7FFF_FFFE, 12]);
    assert!(x.mul(x.inv()).is_one());
    assert!(x.div(y).mul(y).to_raw()[0] == 3 && x.div(y).mul(y).to_raw()[1] == 5);
    assert!(x.add(y).sub(y).to_raw()[1] == 5);
    assert!(x.neg().add(x).is_zero());
    assert!(x.square().to_raw()[0] == x.mul(x).to_raw()[0]);
    assert!(x.canonical().to_raw()[1] == 5);
    assert!(Elem::<Q>::from_bytes(x.to_bytes()).to_raw()[1] == 5);
    assert!(QUAD_CONST.pow(3).mul(x.mul(y).pow(9)).is_one());
    // `i^2 = -1`, the conjugate flips the imaginary limb, and the norm is
    // `re^2 + im^2` both directly and as `x * conj(x)`.
    assert!(Elem::<Q>::I.mul(Elem::<Q>::I).add(Elem::<Q>::ONE).is_zero());
    let conjugate = x.conjugate();
    assert!(conjugate.to_raw()[0] == 3 && conjugate.to_raw()[1] == 0x7FFF_FFFF - 5);
    assert!(x.norm().to_raw() == 34);
    let [re, im] = x.mul(conjugate).to_components();
    assert!(re.to_raw() == 34 && im.to_raw() == 0);
    let [re, im] = x.to_components();
    assert!(Elem::<Q>::from_components([re, im]).to_raw()[1] == 5);
    let generator = Elem::<Q>::GENERATOR;
    assert!(generator.pow(<Q as Field>::ORDER - 1).is_one());
};

#[test]
fn shipped_instance_const_surface_matches_the_trait_surface() {
    // The compile-time values above, recomputed through the generic trait
    // surface at runtime.
    macro_rules! check {
        ($name:ident, $row:ty, $x:expr, $y:expr) => {
            let x = <Elem<$row> as FieldElem>::from_raw($x);
            let y = <Elem<$row> as FieldElem>::from_raw($y);
            assert_eq!(
                $name,
                FieldElem::pow(FieldElem::inv(FieldElem::mul(x, y)), 3),
                "{}",
                <$row as Field>::NAME
            );
        };
    }
    shipped_binary_instances!(check);
    check!(QUAD_CONST, QuadMersenne31, [3, 5], [0x7FFF_FFFE, 12]);
}

#[test]
fn raw_words_are_primitives_of_fixed_width() {
    // Generic code driving raw words as primitives instantiates for every
    // fixed-width row, extension storage included.
    fn xor_raws<F: Field>(a: F::Raw, b: F::Raw) -> F::Raw
    where
        F::Raw: core::ops::BitXor<Output = F::Raw>,
    {
        a ^ b
    }
    assert_eq!(
        xor_raws::<FiniteField<2, 8, Polynomial<AES>>>(0xF0, 0x0F),
        0xFF
    );
    assert_eq!(xor_raws::<Gf16>(0x00FF, 0x0F0F), 0x0FF0);
    assert_eq!(xor_raws::<Gf32>(1, 3), 2);
    assert_eq!(xor_raws::<Gf64>(1, 3), 2);
    assert_eq!(xor_raws::<Mersenne31>(0xF0, 0x0F), 0xFF);
    assert_eq!(xor_raws::<Goldilocks>(1, 3), 2);
    assert_eq!(xor_raws::<Residue7>(1, 3), 2);
    assert_eq!(xor_raws::<ResidueWide>(1, 3), 2);

    // Fixed raw widths, pinned per row.
    const fn width<F: Field>() -> usize {
        size_of::<F::Raw>()
    }
    assert_eq!(width::<FiniteField<2, 1, Polynomial<3>>>(), 1);
    assert_eq!(width::<FiniteField<2, 2, Polynomial<0x7>>>(), 1);
    assert_eq!(width::<FiniteField<2, 4, Polynomial<0x13>>>(), 1);
    assert_eq!(width::<FiniteField<2, 8, Polynomial<AES>>>(), 1);
    assert_eq!(width::<Gf16>(), 2);
    assert_eq!(width::<Gf32>(), 4);
    assert_eq!(width::<Gf64>(), 8);
    assert_eq!(width::<Mersenne31>(), 4);
    assert_eq!(width::<Goldilocks>(), 8);
    assert_eq!(width::<QuadMersenne31>(), 8);
    assert_eq!(width::<Residue7>(), 8);
    assert_eq!(width::<ResidueWide>(), 8);
}

#[test]
fn family_generic_code_requires_the_capability() {
    // Abstracting over P, N, and R means naming the representation bound and
    // the capability bound together; metadata membership alone proves
    // nothing.
    fn doubled<const P: u64, const N: u32, R>(
        value: Elem<FiniteField<P, N, R>>,
    ) -> Elem<FiniteField<P, N, R>>
    where
        R: FiniteFieldRepr<P, N>,
        FiniteField<P, N, R>: Field,
    {
        FieldElem::add(value, value)
    }
    assert_eq!(doubled(Elem::<Residue7>::from_raw(3)).to_raw(), 6);
    assert_eq!(doubled(Elem::<Residue2>::ONE).to_raw(), 0);
    // u64::MAX reduces to 58; doubling gives 116, still canonical.
    assert_eq!(
        doubled(Elem::<ResidueWide>::from_raw(u64::MAX)).to_raw(),
        116
    );
}

#[test]
fn residue_conversions_are_independent_of_presentation_spelling() {
    // The default spelling and the explicit presentation are one type.
    fn same_type<F: Field>(elem: Elem<F>) -> Elem<F> {
        elem
    }
    let explicit = Elem::<FiniteField<7, 1, Residue>>::from_raw(5);
    let defaulted = Elem::<FiniteField<7, 1>>::from_raw(5);
    assert_eq!(same_type(explicit), defaulted);
    assert_eq!(
        defaulted.to_residue(),
        Elem::<FiniteField<7, 1>>::from_residue(5)
            .ok()
            .map(Elem::to_raw)
    );
}

// ---------------------------------------------------------------------------
// Generic extension rows over every base shape
// ---------------------------------------------------------------------------

use fgf::{
    BinaryField, Components, Extension, ExtensionField, ExtensionSpec, Packed, PackedPadded,
    Relation,
};

type Residue3 = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1>;

/// GF(49) as `t^2 - 3` over the residue row at seven.
#[derive(Clone, Copy)]
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = Residue7;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    // `4 = -3` constant first, then the zero linear coefficient.
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type Gf49Row = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

/// Goldilocks squared as `t^2 - 7`.
#[derive(Clone, Copy)]
struct GoldilocksSquareSpec;
impl ExtensionSpec<2> for GoldilocksSquareSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(g^2)";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}
type GoldilocksSquare = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GoldilocksSquareSpec, 2>>;

/// Mersenne31 cubed as `t^3 - 5`.
#[derive(Clone, Copy)]
struct MersenneCubeSpec;
impl ExtensionSpec<3> for MersenneCubeSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "GF((2^31 - 1)^3)";
    const RELATION: Relation = Relation::from_coordinates(&[0x7FFF_FFFF - 5, 0, 0]);
}
type MersenneCube = FiniteField<0x7FFF_FFFF, 3, Extension<MersenneCubeSpec, 3>>;

/// BabyBear to the fourth as `t^4 - 11`.
#[derive(Clone, Copy)]
struct BabyBearFourthSpec;
impl ExtensionSpec<4> for BabyBearFourthSpec {
    type Base = FiniteField<0x7800_0001, 1>;
    type Storage = Components;
    const NAME: &'static str = "BabyBear^4";
    const RELATION: Relation = Relation::from_coordinates(&[0x7800_0001 - 11, 0, 0, 0]);
}
type BabyBearFourth = FiniteField<0x7800_0001, 4, Extension<BabyBearFourthSpec, 4>>;

/// KoalaBear to the fourth as `t^4 - 3`.
#[derive(Clone, Copy)]
struct KoalaBearFourthSpec;
impl ExtensionSpec<4> for KoalaBearFourthSpec {
    type Base = FiniteField<0x7F00_0001, 1>;
    type Storage = Components;
    const NAME: &'static str = "KoalaBear^4";
    const RELATION: Relation = Relation::from_coordinates(&[0x7F00_0001 - 3, 0, 0, 0]);
}
type KoalaBearFourth = FiniteField<0x7F00_0001, 4, Extension<KoalaBearFourthSpec, 4>>;

/// A custom GF(p^2) as `t^2 + 1` over Mersenne31, and `t^2 - (2 + i)`
/// above it: an extension whose base is a custom extension row.
#[derive(Clone, Copy)]
struct CustomComplexMersenneSpec;
impl ExtensionSpec<2> for CustomComplexMersenneSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "custom GF(p^2)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type CustomComplexMersenne = FiniteField<0x7FFF_FFFF, 2, Extension<CustomComplexMersenneSpec, 2>>;

#[derive(Clone, Copy)]
struct OverCustomMersenneSpec;
impl ExtensionSpec<2> for OverCustomMersenneSpec {
    type Base = CustomComplexMersenne;
    type Storage = Components;
    const NAME: &'static str = "GF(p^4) over custom";
    // `-(2 + i)` in base components: `(-2, -1)`.
    const RELATION: Relation =
        Relation::from_coordinates(&[0x7FFF_FFFF - 2, 0x7FFF_FFFF - 1, 0, 0]);
}
type OverCustomMersenne = FiniteField<0x7FFF_FFFF, 4, Extension<OverCustomMersenneSpec, 2>>;

/// `t^2 - (2 + i)` over the shipped QuadMersenne31 row.
#[derive(Clone, Copy)]
struct OverQuadMersenneSpec;
impl ExtensionSpec<2> for OverQuadMersenneSpec {
    type Base = QuadMersenne31;
    type Storage = Components;
    const NAME: &'static str = "GF(((2^31 - 1)^2)^2)";
    // `-(2 + i)` in base components: `(-2, -1)`.
    const RELATION: Relation =
        Relation::from_coordinates(&[0x7FFF_FFFF - 2, 0x7FFF_FFFF - 1, 0, 0]);
}
type OverQuadMersenne = FiniteField<0x7FFF_FFFF, 4, Extension<OverQuadMersenneSpec, 2>>;

/// The largest `u64` prime squared as `t^2 - 2`, two a nonresidue.
#[derive(Clone, Copy)]
struct WidePrimeSquareSpec;
impl ExtensionSpec<2> for WidePrimeSquareSpec {
    type Base = Residue3;
    type Storage = Components;
    const NAME: &'static str = "GF(p^2) at the widest prime";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_FFFF_FFC5 - 2, 0]);
}
type WidePrimeSquare = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 2, Extension<WidePrimeSquareSpec, 2>>;

/// A cubic `t^3 + t + 1` over the AES byte field.
#[derive(Clone, Copy)]
struct AesCubicSpec;
impl ExtensionSpec<3> for AesCubicSpec {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = PackedPadded<u32>;
    const NAME: &'static str = "GF(2^24)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<AES>>>(&[1, 1, 0]);
}
type AesCubic = FiniteField<2, 24, Extension<AesCubicSpec, 3>>;

/// `x^13 + x^4 + x^3 + x + 1` over GF(2).
#[derive(Clone, Copy)]
struct Degree13Spec;
impl ExtensionSpec<13> for Degree13Spec {
    type Base = FiniteField<2, 1, Polynomial<3>>;
    type Storage = PackedPadded<u16>;
    const NAME: &'static str = "GF(2^13)";
    const RELATION: Relation = Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[
        1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
}
type Degree13 = FiniteField<2, 13, Extension<Degree13Spec, 13>>;

/// GF(8) as `x^3 + x + 1` over GF(2), and `t^2 + alpha*t + 1` above it
/// with `alpha` the base indeterminate: a quadratic over a non-quadratic
/// characteristic-two extension base.
#[derive(Clone, Copy)]
struct Gf8CubicSpec;
impl ExtensionSpec<3> for Gf8CubicSpec {
    type Base = FiniteField<2, 1, Polynomial<3>>;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(8)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[1, 1, 0]);
}
type Gf8Cubic = FiniteField<2, 3, Extension<Gf8CubicSpec, 3>>;

#[derive(Clone, Copy)]
struct Gf64OverGf8Spec;
impl ExtensionSpec<2> for Gf64OverGf8Spec {
    type Base = Gf8Cubic;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(64)";
    const RELATION: Relation = Relation::from_binary_words::<Gf8Cubic>(&[1, 2]);
}
type Gf64OverGf8 = FiniteField<2, 6, Extension<Gf64OverGf8Spec, 2>>;

/// `t^2 + t + 1` over the AES normal basis.
#[derive(Clone, Copy)]
struct OverNormalSpec;
impl ExtensionSpec<2> for OverNormalSpec {
    type Base = FiniteField<2, 8, Normal<0x11B, 0x20>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16) over normal";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(&[1, 1]);
}
type OverNormal = FiniteField<2, 16, Extension<OverNormalSpec, 2>>;

/// `t^2 + one*t + 3` over the AES Cantor basis, the linear coefficient
/// the basis identity word.
#[derive(Clone, Copy)]
struct OverCantorSpec;
impl ExtensionSpec<2> for OverCantorSpec {
    type Base = FiniteField<2, 8, Cantor<0x11B, 0x20>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16) over cantor";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>(&[3, 0xFF]);
}
type OverCantor = FiniteField<2, 16, Extension<OverCantorSpec, 2>>;

/// `t^2 + t + 1` over the residue row at two, which is not a binary
/// presentation: the row keeps its component bytes but never joins the
/// binary capability.
#[derive(Clone, Copy)]
struct ResidueTwoSquareSpec;
impl ExtensionSpec<2> for ResidueTwoSquareSpec {
    type Base = Residue2;
    type Storage = Components;
    const NAME: &'static str = "GF(4) over residues";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
}
type ResidueTwoSquare = FiniteField<2, 2, Extension<ResidueTwoSquareSpec, 2>>;

/// Product, inverse, and component round trip shared by every extension
/// row above.
fn exercise_extension<F: ExtensionField<D>, const D: usize>(value: Elem<F>, other: Elem<F>) {
    let zero = Elem::<F>::ZERO;
    let one = Elem::<F>::ONE;
    assert_eq!(value.mul(one), value);
    assert_eq!(value.mul(other), other.mul(value));
    assert_eq!(value.mul(value), value.pow(2));
    assert_eq!(value.mul(value.inv()), one);
    assert_eq!(zero.inv(), zero);
    assert_eq!(
        value.mul(other.add(one)),
        value.mul(other).add(value),
        "{}",
        <F as Field>::NAME
    );
    let components = <F as ExtensionField<D>>::to_components(value);
    assert_eq!(
        <F as ExtensionField<D>>::from_components(components),
        value,
        "{}",
        <F as Field>::NAME
    );
}

/// The indeterminate `t` of an extension row: component one.
fn indeterminate<F: ExtensionField<D>, const D: usize>() -> Elem<F> {
    let mut components = <F as ExtensionField<D>>::to_components(Elem::<F>::ZERO);
    components[1] = Elem::<F::Base>::ONE;
    <F as ExtensionField<D>>::from_components(components)
}

#[test]
fn extension_rows_exercise_arithmetic_and_relations() {
    // Odd characteristic, component storage.
    exercise_extension::<Gf49Row, 2>(
        Elem::<Gf49Row>::from_raw([3, 5]),
        Elem::<Gf49Row>::from_raw([6, 2]),
    );
    let t = indeterminate::<Gf49Row, 2>();
    assert_eq!(t.mul(t), Elem::<Gf49Row>::from_raw([3, 0]));

    exercise_extension::<GoldilocksSquare, 2>(
        Elem::<GoldilocksSquare>::from_raw([0xFFFF_FFFF_0000_0001 - 3, 0x0123_4567_89AB_CDEF]),
        Elem::<GoldilocksSquare>::from_raw([7, 9]),
    );
    let t = indeterminate::<GoldilocksSquare, 2>();
    assert_eq!(t.mul(t), Elem::<GoldilocksSquare>::from_raw([7, 0]));

    exercise_extension::<MersenneCube, 3>(
        Elem::<MersenneCube>::from_raw([5, 9, 2]),
        Elem::<MersenneCube>::from_raw([0x7FFF_FFFE, 1, 0x4000_0000]),
    );
    let t = indeterminate::<MersenneCube, 3>();
    assert_eq!(t.mul(t).mul(t), Elem::<MersenneCube>::from_raw([5, 0, 0]));

    exercise_extension::<BabyBearFourth, 4>(
        Elem::<BabyBearFourth>::from_raw([5, 9, 2, 11]),
        Elem::<BabyBearFourth>::from_raw([1, 2, 3, 4]),
    );
    let t = indeterminate::<BabyBearFourth, 4>();
    assert_eq!(t.pow(4), Elem::<BabyBearFourth>::from_raw([11, 0, 0, 0]));

    exercise_extension::<KoalaBearFourth, 4>(
        Elem::<KoalaBearFourth>::from_raw([1, 2, 3, 4]),
        Elem::<KoalaBearFourth>::from_raw([0x7F00_0000, 7, 7, 7]),
    );
    let t = indeterminate::<KoalaBearFourth, 4>();
    assert_eq!(t.pow(4), Elem::<KoalaBearFourth>::from_raw([3, 0, 0, 0]));

    exercise_extension::<OverCustomMersenne, 2>(
        Elem::<OverCustomMersenne>::from_raw([[1, 2], [3, 4]]),
        Elem::<OverCustomMersenne>::from_raw([[0x7FFF_FFFE, 0], [12, 0x4000_0000]]),
    );
    let t = indeterminate::<OverCustomMersenne, 2>();
    // `t^2 = 2 + i` in base components.
    assert_eq!(
        t.mul(t),
        Elem::<OverCustomMersenne>::from_raw([[2, 1], [0, 0]])
    );
    // The base row inverts through its own relation: `i^2 + 1 = 0`.
    let i = indeterminate::<CustomComplexMersenne, 2>();
    assert_eq!(
        i.mul(i),
        Elem::<CustomComplexMersenne>::from_raw([0x7FFF_FFFE, 0])
    );

    // Over the shipped QuadMersenne31: `t^2 = 2 + i` in the base's own
    // limbs, and the byte form concatenates the base encodings.
    exercise_extension::<OverQuadMersenne, 2>(
        Elem::<OverQuadMersenne>::from_raw([[1, 2], [3, 4]]),
        Elem::<OverQuadMersenne>::from_raw([[0x7FFF_FFFE, 0], [12, 0x4000_0000]]),
    );
    let t = indeterminate::<OverQuadMersenne, 2>();
    assert_eq!(
        t.mul(t),
        Elem::<OverQuadMersenne>::from_raw([[2, 1], [0, 0]])
    );
    assert_eq!(<OverQuadMersenne as FieldBuffer>::BYTES, 16);
    let mut bytes = [0u8; 16];
    FieldBuffer::encode(&mut bytes, t.mul(t));
    assert_eq!(
        FieldBuffer::decode(&bytes),
        Elem::<OverQuadMersenne>::from_raw([[2, 1], [0, 0]])
    );
    assert_eq!(&bytes[..4], &2u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
    assert_eq!(&bytes[8..], &[0u8; 8]);

    exercise_extension::<WidePrimeSquare, 2>(
        Elem::<WidePrimeSquare>::from_raw([7, 9]),
        Elem::<WidePrimeSquare>::from_raw([0xFFFF_FFFF_FFFF_FFC4, 0xFFFF_FFFF_FFFF_FFC3]),
    );
    let t = indeterminate::<WidePrimeSquare, 2>();
    assert_eq!(t.mul(t), Elem::<WidePrimeSquare>::from_raw([2, 0]));

    // Characteristic two.
    exercise_extension::<AesCubic, 3>(
        Elem::<AesCubic>::from_raw(0x00AB_CDEF),
        Elem::<AesCubic>::from_raw(0x1200_0034),
    );
    let t = indeterminate::<AesCubic, 3>();
    // `t^3 + t + 1 = 0`: the indeterminate occupies bits `[8, 16)`.
    assert!(t.mul(t).mul(t).add(t).add(Elem::<AesCubic>::ONE).is_zero());

    exercise_extension::<Degree13, 13>(
        Elem::<Degree13>::from_raw(0x1234),
        Elem::<Degree13>::from_raw(0x1B53),
    );
    let t = indeterminate::<Degree13, 13>();
    // `x^13 + x^4 + x^3 + x + 1 = 0`.
    assert!(
        t.pow(13)
            .add(t.pow(4))
            .add(t.pow(3))
            .add(t)
            .add(Elem::<Degree13>::ONE)
            .is_zero()
    );

    exercise_extension::<Gf8Cubic, 3>(
        Elem::<Gf8Cubic>::from_raw(0x5),
        Elem::<Gf8Cubic>::from_raw(0x3),
    );
    let t = indeterminate::<Gf8Cubic, 3>();
    assert!(t.mul(t).mul(t).add(t).add(Elem::<Gf8Cubic>::ONE).is_zero());

    exercise_extension::<Gf64OverGf8, 2>(
        Elem::<Gf64OverGf8>::from_raw(0x2A),
        Elem::<Gf64OverGf8>::from_raw(0x15),
    );
    let t = indeterminate::<Gf64OverGf8, 2>();
    // `t^2 + alpha*t + 1 = 0` with `alpha` the base indeterminate (word 2).
    assert!(
        t.mul(t)
            .add(t.mul(Elem::<Gf64OverGf8>::from_raw(0x02)))
            .add(Elem::<Gf64OverGf8>::ONE)
            .is_zero()
    );

    // Ordered-basis bases: the identity is the basis identity word, and
    // arithmetic runs through the basis.
    exercise_extension::<OverNormal, 2>(
        Elem::<OverNormal>::from_raw(0x1B70),
        Elem::<OverNormal>::from_raw(0x0108),
    );
    assert_eq!(
        <OverNormal as Field>::ONE_RAW,
        <OverNormal as ExtensionField<2>>::from_components([
            Elem::<FiniteField<2, 8, Normal<0x11B, 0x20>>>::ONE,
            Elem::<FiniteField<2, 8, Normal<0x11B, 0x20>>>::ZERO,
        ])
        .to_raw()
    );
    let t = indeterminate::<OverNormal, 2>();
    // The relation words are base coordinate words: `t^2 + 1*t + 1` over
    // the basis elements named by word one, not the basis identity.
    let word_one = Elem::<OverNormal>::from_raw(0x0001);
    assert!(t.mul(t).add(t.mul(word_one)).add(word_one).is_zero());

    exercise_extension::<OverCantor, 2>(
        Elem::<OverCantor>::from_raw(0x1B70),
        Elem::<OverCantor>::from_raw(0x0108),
    );
    assert_eq!(
        <OverCantor as Field>::ONE_RAW,
        <OverCantor as ExtensionField<2>>::from_components([
            Elem::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>::ONE,
            Elem::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>::ZERO,
        ])
        .to_raw()
    );
    let t = indeterminate::<OverCantor, 2>();

    // The relation words are base coordinate words: `t^2 + 0xFF*t + 3`
    // over the basis element named by word `0xFF`.
    assert!(
        t.mul(t)
            .add(t.mul(Elem::<OverCantor>::from_raw(0x00FF)))
            .add(Elem::<OverCantor>::from_raw(0x0003))
            .is_zero()
    );

    // The residue row at two: component bytes, never a binary row.
    exercise_extension::<ResidueTwoSquare, 2>(
        Elem::<ResidueTwoSquare>::from_raw([1, 0]),
        Elem::<ResidueTwoSquare>::from_raw([0, 1]),
    );
    let t = indeterminate::<ResidueTwoSquare, 2>();
    assert!(t.mul(t).add(t).add(Elem::<ResidueTwoSquare>::ONE).is_zero());
    assert_eq!(<ResidueTwoSquare as FieldBuffer>::BYTES, 16);
    let mut bytes = [0u8; 16];
    FieldBuffer::encode(&mut bytes, Elem::<ResidueTwoSquare>::from_raw([1, 1]));
    assert_eq!(
        FieldBuffer::decode(&bytes),
        Elem::<ResidueTwoSquare>::from_raw([1, 1])
    );
    assert_eq!(&bytes[..8], &1u64.to_le_bytes());
    assert_eq!(&bytes[8..], &1u64.to_le_bytes());
}

#[test]
fn characteristic_two_extension_rows_join_the_binary_capability() {
    use fgf::BinaryDegree;

    // The padded GF(8) and GF(64) rows convert through their coordinate
    // words: component `i` at the bits `[i*B, (i+1)*B)`.
    fn coordinate_word<F: BinaryField>(value: Elem<F>) -> u64 {
        F::to_coordinates(value)
    }
    let t = indeterminate::<Gf8Cubic, 3>();
    assert_eq!(coordinate_word(t), 0b010);
    assert_eq!(
        Gf8Cubic::from_coordinates(0b010).map(coordinate_word),
        Ok(0b010)
    );
    assert!(Gf8Cubic::from_coordinates(1 << 3).is_err());
    let u = indeterminate::<Gf64OverGf8, 2>();
    assert_eq!(coordinate_word(u), 0b001_000);
    assert!(Gf64OverGf8::from_coordinates(1 << 6).is_err());
    // The wide rows report their full words.
    assert_eq!(
        coordinate_word(Elem::<OverNormal>::from_raw(0x1B70)),
        0x1B70
    );
    assert_eq!(
        OverCantor::from_coordinates(0x00FF).map(|v| v.to_raw()),
        Ok(0x00FF)
    );
    // Every characteristic-two row above is a binary degree member at its
    // exact degree, whatever the storage. The element argument carries the
    // row into the bound; the value is the degree the bound names.
    fn degree<F: BinaryDegree<N>, const N: u32>(_: Elem<F>) -> u32 {
        N
    }
    assert_eq!(degree::<Gf8Cubic, 3>(Elem::<Gf8Cubic>::from_raw(0x5)), 3);
    assert_eq!(
        degree::<Gf64OverGf8, 6>(Elem::<Gf64OverGf8>::from_raw(0x2A)),
        6
    );
    assert_eq!(
        degree::<Degree13, 13>(Elem::<Degree13>::from_raw(0x1234)),
        13
    );
    assert_eq!(
        degree::<AesCubic, 24>(Elem::<AesCubic>::from_raw(0x00AB_CDEF)),
        24
    );
    assert_eq!(
        degree::<OverNormal, 16>(Elem::<OverNormal>::from_raw(0x1B70)),
        16
    );
}
