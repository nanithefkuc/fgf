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
    Tower, ops,
};

type Residue2 = FiniteField<2, 1, Residue>;
type Residue7 = FiniteField<7, 1, Residue>;
type ResiduePrime = FiniteField<1_000_000_007, 1, Residue>;
type ResidueWide = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1, Residue>;

type Gf16 = FiniteField<2, 16, Tower<Rijndael16>>;
type Gf32 = FiniteField<2, 32, Tower<Rijndael32>>;
type Gf64 = FiniteField<2, 64, Tower<Rijndael64>>;

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
    exercise_field::<FiniteField<2, 4, Cantor<0x13, 0xE>>>(0x7, 0x3);
    exercise_field::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(0x53, 0x01);
    exercise_field::<FiniteField<2, 8, Tower<fgf::FanPaar8>>>(0xAB, 0xCD);
    // Towers at degrees sixteen, thirty-two, and sixty-four.
    exercise_field::<Gf16>(0x0108, 0x1B70);
    exercise_field::<Gf32>(0x5555_5555, 0x0F0F_0F0F);
    exercise_field::<Gf64>(0x0123_4567_89AB_CDEF, 0xFFFF_0000_0000_FFFF);
    // Named prime rows and the quadratic extension.
    exercise_field::<Mersenne31>(0x5555_5555, 0x71C7_1C71);
    exercise_field::<Goldilocks>(0xFFFF_FFFF_0000_0000, 7);
    exercise_field::<QuadMersenne31>((0x5555_5555, 0x1234_5678), (0x7FFF_FFFF, 0x7FFF_FFFF));
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
    let quad = Elem::<QuadMersenne31>::from_raw(3, 5);
    exercise_buffer::<QuadMersenne31>(quad);
    exercise_buffer::<FiniteField<2, 8, Polynomial<AES>>>(
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
    );
    exercise_buffer::<Gf16>(Elem::<Gf16>::from_raw(0x1B70));
    exercise_buffer::<Gf32>(Elem::<Gf32>::from_raw(0x0F0F_0F0F));
    exercise_buffer::<Gf64>(Elem::<Gf64>::from_raw(0xFFFF_0000_0000_FFFF));
    exercise_buffer::<Residue7>(Elem::<Residue7>::from_residue(6).expect("in range"));
    exercise_buffer::<ResiduePrime>(Elem::<ResiduePrime>::from_raw(123_456_789));
    exercise_buffer::<ResidueWide>(Elem::<ResidueWide>::from_raw(u64::MAX));
}

#[test]
fn dispatchable_rows_drive_packed_operations() {
    exercise_kernels::<Mersenne31>(Elem::<Mersenne31>::from_raw(7));
    exercise_kernels::<Goldilocks>(Elem::<Goldilocks>::from_raw(7));
    exercise_kernels::<QuadMersenne31>(Elem::<QuadMersenne31>::from_raw(13, 17));
    exercise_kernels::<FiniteField<2, 8, Polynomial<AES>>>(Elem::<
        FiniteField<2, 8, Polynomial<AES>>,
    >::from_raw(0x53));
    exercise_kernels::<Gf16>(Elem::<Gf16>::from_raw(0x0108));
    exercise_kernels::<Gf32>(Elem::<Gf32>::from_raw(0x5555_5555));
    exercise_kernels::<Gf64>(Elem::<Gf64>::from_raw(0x00C0_FFEE_00C0_FFEE));
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
