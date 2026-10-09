//! The generic extension row: one implementation of every capability.
//!
//! Every `FiniteField<P, N, Extension<S, D>>` row shares this body: the
//! raw-word operations run on the storage's component form over the base's
//! `Field` arithmetic, with the defining relation decoded once per call
//! from the spec. Componentwise addition, subtraction, and negation;
//! schoolbook products reduced by `t^D = -(c_{D-1} t^{D-1} + ... + c_0)`;
//! inversion by the conjugate-norm formula at relative degree two and by
//! extended Euclid over the base above it. Zero maps to zero under
//! inversion and division.

use core::fmt;

use super::descriptor::Descriptor;
use super::private::Sealed;
use super::storage::{ExtensionStorage, StorageBuffer, StorageDisplay, StorageRepr, extract_bits};
use super::{Extension, ExtensionField, ExtensionSpec};
use crate::field::binary::binary_private;
use crate::field::binary::description::BinaryDescription;
use crate::field::binary::{BinaryDegree, BinaryField, CoordinateError};
use crate::field::finite_field::{FiniteField, FiniteFieldRepr, checked_order};
use crate::field::private::RowAccess;
use crate::field::{Elem, Field, FieldBuffer, FieldElem, PrimeCharacteristic, Validate};

/// The raw word of an extension row over `S`.
type RawOf<S, const D: usize> =
    <<S as ExtensionSpec<D>>::Storage as ExtensionStorage<<S as ExtensionSpec<D>>::Base, D>>::Raw;

/// The base raw word of an extension row over `S`.
type BaseRaw<S, const D: usize> = <<S as ExtensionSpec<D>>::Base as Field>::Raw;

/// The base components of a raw word, in component order.
fn split<S: ExtensionSpec<D>, const D: usize>(raw: RawOf<S, D>) -> [BaseRaw<S, D>; D]
where
    S::Storage: StorageRepr<S::Base, D>,
{
    <S::Storage as StorageRepr<S::Base, D>>::split(raw)
}

/// The raw word of base components, canonicalized on the way in.
fn join<S: ExtensionSpec<D>, const D: usize>(components: [BaseRaw<S, D>; D]) -> RawOf<S, D>
where
    S::Storage: StorageRepr<S::Base, D>,
{
    <S::Storage as StorageRepr<S::Base, D>>::join(components)
}

// ---------------------------------------------------------------------------
// Representation metadata
// ---------------------------------------------------------------------------

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> FiniteFieldRepr<P, N>
    for Extension<S, D>
where
    S::Storage: StorageRepr<S::Base, D>,
{
    const NAME: &'static str = S::NAME;
    const VALID: () = {
        let () = <S::Base as Field>::VALID;
        let descriptor = <FiniteField<P, N, Extension<S, D>> as RowAccess>::DESCRIPTOR;
        Descriptor::validate_relation(descriptor);
    };
}

// ---------------------------------------------------------------------------
// Component arithmetic over the base
// ---------------------------------------------------------------------------

/// The zero component vector.
fn zero_components<B: Field, const D: usize>() -> [B::Raw; D] {
    core::array::from_fn(|_| <B as Field>::ZERO_RAW)
}

/// The defining relation as base elements, constant term first.
fn relation_components<S: ExtensionSpec<D>, const D: usize>() -> [BaseRaw<S, D>; D] {
    let flat = S::RELATION.to_coordinates();
    let words = <S::Base as Field>::DEGREE as usize;
    core::array::from_fn(|j| {
        <S::Base as RowAccess>::from_prime_coordinates(&flat[j * words..(j + 1) * words])
    })
}

/// Componentwise addition.
fn add_components<B: Field, const D: usize>(a: &[B::Raw; D], b: &[B::Raw; D]) -> [B::Raw; D] {
    let mut out = zero_components::<B, D>();
    for (slot, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *slot = <B as Field>::add_raw(*x, *y);
    }
    out
}

/// Componentwise subtraction.
fn sub_components<B: Field, const D: usize>(a: &[B::Raw; D], b: &[B::Raw; D]) -> [B::Raw; D] {
    let mut out = zero_components::<B, D>();
    for (slot, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *slot = <B as Field>::sub_raw(*x, *y);
    }
    out
}

/// Componentwise negation.
fn neg_components<B: Field, const D: usize>(a: &[B::Raw; D]) -> [B::Raw; D] {
    core::array::from_fn(|i| <B as Field>::neg_raw(a[i]))
}

/// The degree of a coefficient vector, or `-1` when it is zero.
#[allow(clippy::cast_possible_wrap)]
fn poly_deg<B: Field>(coefficients: &[B::Raw]) -> isize {
    let mut degree: isize = -1;
    for (i, coefficient) in coefficients.iter().enumerate() {
        if *coefficient != <B as Field>::ZERO_RAW {
            degree = i as isize;
        }
    }
    degree
}

/// The schoolbook product of two coefficient vectors, reduced by
/// `t^D = -(c_{D-1} t^{D-1} + ... + c_0)`.
///
/// The product accumulates into one vector for the degrees below `D` and
/// one for the degrees at and above `D`, then folds the high degrees from
/// the top down, so no intermediate wider than the component count exists.
fn mul_components<B: Field, const D: usize>(
    a: &[B::Raw; D],
    b: &[B::Raw; D],
    relation: &[B::Raw; D],
) -> [B::Raw; D] {
    let mut low = zero_components::<B, D>();
    let mut high = zero_components::<B, D>();
    for i in 0..D {
        for j in 0..D {
            let product = <B as Field>::mul_raw(a[i], b[j]);
            if product == <B as Field>::ZERO_RAW {
                continue;
            }
            if i + j < D {
                low[i + j] = <B as Field>::add_raw(low[i + j], product);
            } else {
                high[i + j - D] = <B as Field>::add_raw(high[i + j - D], product);
            }
        }
    }
    // One high coefficient per degree from `2D - 2` down to `D`, folding
    // `top * t^degree` into the lower degrees through the relation.
    for j in (0..D).rev().skip(1) {
        let top = high[j];
        if top == <B as Field>::ZERO_RAW {
            continue;
        }
        for (c, &relation_coefficient) in relation.iter().enumerate() {
            let product = <B as Field>::mul_raw(top, relation_coefficient);
            let target = j + c;
            if target < D {
                low[target] = <B as Field>::sub_raw(low[target], product);
            } else {
                high[target - D] = <B as Field>::sub_raw(high[target - D], product);
            }
        }
    }
    low
}

/// The quotient and remainder of `u` by a nonzero `v`, both of degree
/// below `D`.
///
/// The divisor's leading coefficient is inverted once; the dividend's
/// coefficients are eliminated from the top down.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn divmod<B: Field, const D: usize>(
    u: &[B::Raw; D],
    v: &[B::Raw; D],
) -> ([B::Raw; D], [B::Raw; D]) {
    let divisor_degree = poly_deg::<B>(v) as usize;
    let inverse_lead = <B as Field>::inv_raw(v[divisor_degree]);
    let mut quotient = zero_components::<B, D>();
    let mut remainder = *u;
    let mut k = poly_deg::<B>(u);
    // Eliminating down to the divisor's degree inclusive leaves a remainder
    // strictly below it; stopping above it would return equal-degree
    // dividends unreduced.
    while k >= divisor_degree as isize {
        let ki = k as usize;
        let factor = <B as Field>::mul_raw(remainder[ki], inverse_lead);
        if factor != <B as Field>::ZERO_RAW {
            quotient[ki - divisor_degree] = factor;
            for (j, &coefficient) in v.iter().enumerate().take(divisor_degree + 1) {
                let target = ki - divisor_degree + j;
                remainder[target] = <B as Field>::sub_raw(
                    remainder[target],
                    <B as Field>::mul_raw(factor, coefficient),
                );
            }
        }
        k -= 1;
    }
    (quotient, remainder)
}

/// The quotient and remainder of the defining relation by a nonzero `v`,
/// without materializing the relation's monic leading term.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn relation_divmod<B: Field, const D: usize>(
    relation: &[B::Raw; D],
    v: &[B::Raw; D],
) -> ([B::Raw; D], [B::Raw; D]) {
    let divisor_degree = poly_deg::<B>(v) as usize;
    let inverse_lead = <B as Field>::inv_raw(v[divisor_degree]);
    let mut quotient = zero_components::<B, D>();
    let mut remainder = *relation;
    // Eliminate the monic `t^D` first: its quotient term is the inverse of
    // the divisor's leading coefficient, and the `t^D` it subtracts cancels
    // the implicit leading one.
    quotient[D - divisor_degree] = inverse_lead;
    for (j, &coefficient) in v.iter().enumerate().take(divisor_degree) {
        let target = D - divisor_degree + j;
        remainder[target] = <B as Field>::sub_raw(
            remainder[target],
            <B as Field>::mul_raw(inverse_lead, coefficient),
        );
    }
    let mut k = D as isize - 1;
    // As `divmod`: the last elimination lands in the quotient's constant
    // term and drops the remainder below the divisor's degree.
    while k >= divisor_degree as isize {
        let ki = k as usize;
        let factor = <B as Field>::mul_raw(remainder[ki], inverse_lead);
        if factor != <B as Field>::ZERO_RAW {
            quotient[ki - divisor_degree] = factor;
            for (j, &coefficient) in v.iter().enumerate().take(divisor_degree + 1) {
                let target = ki - divisor_degree + j;
                remainder[target] = <B as Field>::sub_raw(
                    remainder[target],
                    <B as Field>::mul_raw(factor, coefficient),
                );
            }
        }
        k -= 1;
    }
    (quotient, remainder)
}

/// The inverse of a nonzero component vector over the relation.
///
/// At relative degree two, the conjugate-norm formula: for
/// `f = t^2 + c1*t + c0` the norm `a0^2 - c1*a0*a1 + c0*a1^2` is nonzero
/// and the inverse is `((a0 - c1*a1) - a1*t) / norm`. Above it, extended
/// Euclid on `(relation, vector)`: the invariant
/// `r = s * vector (mod relation)` runs from the relation and the vector
/// down to the unit gcd, whose Bézout coefficient scales to the inverse.
/// Every polynomial stays below the component count: each quotient times a
/// Bézout coefficient is reduced through the relation on the way in.
fn inv_components<B: Field, const D: usize>(
    a: &[B::Raw; D],
    relation: &[B::Raw; D],
) -> [B::Raw; D] {
    let one = <B as Field>::ONE_RAW;
    if D == 2 {
        let (a0, a1) = (a[0], a[1]);
        let (c0, c1) = (relation[0], relation[1]);
        let square = <B as Field>::mul_raw(a0, a0);
        let cross = <B as Field>::mul_raw(c1, <B as Field>::mul_raw(a0, a1));
        let scaled = <B as Field>::mul_raw(c0, <B as Field>::mul_raw(a1, a1));
        // The norm of a nonzero element is nonzero over an irreducible
        // relation, so the base inverse is exact.
        let sum = <B as Field>::add_raw(<B as Field>::sub_raw(square, cross), scaled);
        let inverse_norm = <B as Field>::inv_raw(sum);
        let leading = <B as Field>::sub_raw(a0, <B as Field>::mul_raw(c1, a1));
        let r0 = <B as Field>::mul_raw(leading, inverse_norm);
        let r1 = <B as Field>::mul_raw(<B as Field>::neg_raw(a1), inverse_norm);
        let mut out = zero_components::<B, D>();
        out[0] = r0;
        out[1] = r1;
        return out;
    }
    // A nonzero constant inverts in the base.
    if a[1..]
        .iter()
        .all(|&component| component == <B as Field>::ZERO_RAW)
    {
        let inverse = <B as Field>::inv_raw(a[0]);
        let zero = <B as Field>::ZERO_RAW;
        return core::array::from_fn(|i| if i == 0 { inverse } else { zero });
    }
    let (quotient, mut r_prev) = relation_divmod::<B, D>(relation, a);
    // `relation = quotient*a + r_prev`, so `r_prev` is congruent to
    // `-quotient*a` modulo the relation.
    let mut s_prev = neg_components::<B, D>(&quotient);
    let mut r_cur = *a;
    let mut s_cur = {
        let mut s = zero_components::<B, D>();
        s[0] = one;
        s
    };
    loop {
        let (q, rem) = divmod::<B, D>(&r_prev, &r_cur);
        if poly_deg::<B>(&rem) < 0 {
            break;
        }
        let scaled = mul_components::<B, D>(&q, &s_cur, relation);
        let s_next = sub_components::<B, D>(&s_prev, &scaled);
        r_prev = r_cur;
        r_cur = rem;
        s_prev = s_cur;
        s_cur = s_next;
    }
    // The loop leaves a nonzero constant `u` with `s_cur*a = u` modulo the
    // relation.
    let inverse_unit = <B as Field>::inv_raw(r_cur[0]);
    core::array::from_fn(|i| <B as Field>::mul_raw(s_cur[i], inverse_unit))
}

// ---------------------------------------------------------------------------
// Field row
// ---------------------------------------------------------------------------

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> Field
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
{
    type Raw = RawOf<S, D>;
    type Characteristic = PrimeCharacteristic<P>;
    const NAME: &'static str = S::NAME;
    // Degree and order read the metadata proof directly, never the row's
    // identities, so central validation can read them without a cycle.
    const DEGREE: u32 = {
        checked_order::<P, N>();
        N
    };
    const ORDER: u128 = checked_order::<P, N>();
    // The identity words evaluate the row's validity first and then read
    // their row-local data, never the reverse.
    const ZERO_RAW: Self::Raw = {
        let () = <Self as Field>::VALID;
        <S::Storage as StorageRepr<S::Base, D>>::ZERO
    };
    const ONE_RAW: Self::Raw = {
        let () = <Self as Field>::VALID;
        <S::Storage as StorageRepr<S::Base, D>>::ONE
    };
    // Metadata first, then the representation's defining data.
    const VALID: () = {
        checked_order::<P, N>();
        let () = <Extension<S, D> as FiniteFieldRepr<P, N>>::VALID;
    };

    #[inline]
    fn canonical_raw(raw: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        <S::Storage as StorageRepr<S::Base, D>>::canonical(raw)
    }

    #[inline]
    fn add_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        let (a, b) = (split::<S, D>(left), split::<S, D>(right));
        join::<S, D>(add_components::<S::Base, D>(&a, &b))
    }

    #[inline]
    fn sub_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        let (a, b) = (split::<S, D>(left), split::<S, D>(right));
        join::<S, D>(sub_components::<S::Base, D>(&a, &b))
    }

    #[inline]
    fn neg_raw(value: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        join::<S, D>(neg_components::<S::Base, D>(&split::<S, D>(value)))
    }

    #[inline]
    fn mul_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        let (a, b) = (split::<S, D>(left), split::<S, D>(right));
        let relation = relation_components::<S, D>();
        join::<S, D>(mul_components::<S::Base, D>(&a, &b, &relation))
    }

    #[inline]
    fn inv_raw(value: Self::Raw) -> Self::Raw {
        let () = Validate::<Self>::OK;
        let a = split::<S, D>(value);
        if a.iter().all(|&x| x == <S::Base as Field>::ZERO_RAW) {
            return <S::Storage as StorageRepr<S::Base, D>>::ZERO;
        }
        let relation = relation_components::<S, D>();
        join::<S, D>(inv_components::<S::Base, D>(&a, &relation))
    }
}

// ---------------------------------------------------------------------------
// Row hooks
// ---------------------------------------------------------------------------

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> RowAccess
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
{
    const DESCRIPTOR: &'static Descriptor = &Descriptor::extension(
        <S::Base as RowAccess>::DESCRIPTOR,
        P,
        N,
        D,
        S::RELATION.to_coordinates(),
        <S::Storage as StorageRepr<S::Base, D>>::TAG,
    );

    // The row hooks leave the trait's `where Self: Field` clause implicit:
    // every row implements `Field`, and omitting the clause keeps
    // `<Self as Field>::Raw` normalizable in the bodies.
    fn prime_coordinates_into(dst: &mut [u64], raw: <Self as Field>::Raw) {
        let () = Validate::<Self>::OK;
        assert_eq!(
            dst.len(),
            N as usize,
            "prime coordinate count differs from the field degree"
        );
        let words = <S::Base as Field>::DEGREE as usize;
        let components = split::<S, D>(raw);
        for (i, component) in components.iter().enumerate() {
            <S::Base as RowAccess>::prime_coordinates_into(
                &mut dst[i * words..(i + 1) * words],
                *component,
            );
        }
    }

    fn from_prime_coordinates(src: &[u64]) -> <Self as Field>::Raw {
        let () = Validate::<Self>::OK;
        assert_eq!(
            src.len(),
            N as usize,
            "prime coordinate count differs from the field degree"
        );
        let words = <S::Base as Field>::DEGREE as usize;
        let components = core::array::from_fn(|i| {
            <S::Base as RowAccess>::from_prime_coordinates(&src[i * words..(i + 1) * words])
        });
        join::<S, D>(components)
    }

    fn from_prime(value: u64) -> <Self as Field>::Raw {
        let () = Validate::<Self>::OK;
        let mut components = zero_components::<S::Base, D>();
        components[0] = <S::Base as RowAccess>::from_prime(value);
        join::<S, D>(components)
    }

    fn to_prime(raw: <Self as Field>::Raw) -> Option<u64> {
        let () = Validate::<Self>::OK;
        let components = split::<S, D>(raw);
        if components[1..]
            .iter()
            .any(|&component| component != <S::Base as Field>::ZERO_RAW)
        {
            return None;
        }
        <S::Base as RowAccess>::to_prime(components[0])
    }
}

// ---------------------------------------------------------------------------
// Component capability
// ---------------------------------------------------------------------------

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> Sealed
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
{
}

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> ExtensionField<D>
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
{
    type Base = S::Base;

    fn from_components(components: [Elem<S::Base>; D]) -> Elem<Self> {
        let () = Validate::<Self>::OK;
        let raws = components.map(|component| component.raw);
        Elem {
            raw: join::<S, D>(raws),
        }
    }

    fn to_components(value: Elem<Self>) -> [Elem<S::Base>; D] {
        let () = Validate::<Self>::OK;
        split::<S, D>(value.raw).map(|raw| Elem { raw })
    }
}

// ---------------------------------------------------------------------------
// Byte and display forms
// ---------------------------------------------------------------------------

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> FieldBuffer
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageBuffer<S::Base, D>,
{
    const BYTES: usize = <S::Storage as StorageBuffer<S::Base, D>>::BYTES;
    const STORAGE_BITS: u32 = <S::Storage as StorageBuffer<S::Base, D>>::STORAGE_BITS;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let () = Validate::<Self>::OK;
        assert_eq!(
            bytes.len(),
            Self::BYTES,
            "extension element has the wrong byte width"
        );
        let raw = <S::Storage as StorageBuffer<S::Base, D>>::decode(bytes);
        Elem {
            raw: <S::Storage as StorageRepr<S::Base, D>>::canonical(raw),
        }
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        let () = Validate::<Self>::OK;
        assert_eq!(
            bytes.len(),
            Self::BYTES,
            "extension element has the wrong byte width"
        );
        <S::Storage as StorageBuffer<S::Base, D>>::encode(bytes, &value.raw);
    }
}

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> fmt::Display
    for Elem<FiniteField<P, N, Extension<S, D>>>
where
    S::Storage: StorageDisplay<S::Base, D>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The absolute degree never exceeds the hex digit capacity.
        #[allow(clippy::cast_possible_truncation)]
        let digits = N.div_ceil(4) as usize;
        <S::Storage as StorageDisplay<S::Base, D>>::fmt(formatter, &self.raw, digits)
    }
}

// ---------------------------------------------------------------------------
// Binary capability rows
// ---------------------------------------------------------------------------

impl<const N: u32, const D: usize, S: ExtensionSpec<D>> binary_private::Sealed
    for FiniteField<2, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
    S::Base: BinaryField,
{
}

impl<const N: u32, const D: usize, S: ExtensionSpec<D>> BinaryField
    for FiniteField<2, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
    S::Base: BinaryField,
{
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::append_relation(
        <S::Base as BinaryField>::DESCRIPTION,
        N,
        D,
        S::RELATION.to_coordinates(),
        <S::Storage as StorageRepr<S::Base, D>>::TAG,
    );

    #[inline]
    fn to_coordinates(value: Elem<Self>) -> u64 {
        // Component `i` occupies the bits `[i*B, (i+1)*B)` of the base
        // degree `B`, whatever the storage; the absolute degree stays at
        // or below 64, so the shifts never wrap.
        let degree = <S::Base as Field>::DEGREE as usize;
        let mut word = 0u64;
        for (index, component) in split::<S, D>(value.raw).iter().enumerate() {
            let element = Elem::<S::Base> { raw: *component };
            word |= <S::Base as BinaryField>::to_coordinates(element) << (index * degree);
        }
        word
    }

    #[inline]
    fn from_coordinates(value: u64) -> Result<Elem<Self>, CoordinateError> {
        // The fit check bounds the word by the row's absolute degree, so
        // every extracted component word fits the base.
        if N < 64 && value >> N != 0 {
            return Err(CoordinateError);
        }
        let degree = <S::Base as Field>::DEGREE as usize;
        let components = core::array::from_fn(|index| {
            let component = extract_bits(value, index * degree, degree);
            <S::Base as BinaryField>::from_coordinates(component)
                .expect("a fitted component word fits the base degree")
                .to_raw()
        });
        Ok(Elem::<Self>::from_raw(join::<S, D>(components)))
    }
}

impl<const N: u32, const D: usize, S: ExtensionSpec<D>> BinaryDegree<N>
    for FiniteField<2, N, Extension<S, D>>
where
    S::Storage: StorageRepr<S::Base, D>,
    S::Base: BinaryField,
{
}
