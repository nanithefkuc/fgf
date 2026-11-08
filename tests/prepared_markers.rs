//! Compile-time marker guarantees of the prepared-coefficient wrappers.
//!
//! [`ops::Coeff`] and the prepared collections store the field contract's
//! backend-ready coefficient form behind an opaque accessor. That form is
//! always plain data — raw words, broadcast tiles, or byte tables — so the
//! wrappers keep the marker traits they held before the dispatch cutover.
//! These consumers name every shipped field class and custom component and
//! packed extension specs, including specs that are themselves `!Send` and
//! `!Sync`, and assert the exact set of guarantees the wrappers provide:
//! `Send`, `Sync`, `Unpin`, `UnwindSafe`, and `RefUnwindSafe`.

use core::marker::PhantomData;
use core::panic::{RefUnwindSafe, UnwindSafe};

use fgf::binary::{
    AES, Cantor, FanPaar8, FanPaar16, FanPaar32, FanPaar64, Normal, Polynomial, RS, Rijndael16,
    Rijndael32, Rijndael64,
};
use fgf::extension::{Components, Extension, ExtensionSpec, Packed, QuadMersenne31, Relation};
#[cfg(feature = "alloc")]
use fgf::ops;
use fgf::ops::Coeff;
use fgf::prime::{Goldilocks, Mersenne31};
use fgf::{Elem, FiniteField};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Rs8 = FiniteField<2, 8, Polynomial<RS>>;
type Normal8 = FiniteField<2, 8, Normal<0x11B, 0x20>>;
type Cantor8 = FiniteField<2, 8, Cantor<0x11B, 0x20>>;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Gf32 = FiniteField<2, 32, Extension<Rijndael32, 2>>;
type Gf64 = FiniteField<2, 64, Extension<Rijndael64, 2>>;

type FanPaar8Row = FiniteField<2, 8, Extension<FanPaar8, 2>>;
type FanPaar16Row = FiniteField<2, 16, Extension<FanPaar16, 2>>;
type FanPaar32Row = FiniteField<2, 32, Extension<FanPaar32, 2>>;
type FanPaar64Row = FiniteField<2, 64, Extension<FanPaar64, 2>>;

/// A custom component-storage spec over GF(7): `t^2 - 3`.
struct CustomComponents;

impl ExtensionSpec<2> for CustomComponents {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "custom component GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

type CustomComponentsRow = FiniteField<7, 2, Extension<CustomComponents, 2>>;

/// A custom packed spec over the AES byte field: `t^2 + t + 0x20`.
struct CustomPacked;

impl ExtensionSpec<2> for CustomPacked {
    type Base = Aes8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "custom packed GF(2^16)";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x20, 0x01]);
}

type CustomPackedRow = FiniteField<2, 16, Extension<CustomPacked, 2>>;

/// A component-storage spec whose spec type is `!Send` and `!Sync`.
///
/// The marker never reaches the row or its prepared word: `Extension` holds
/// the spec behind `PhantomData<fn() -> S>`, and the prepared component
/// array holds raw base words.
struct MarkerComponents(PhantomData<*const ()>);

impl ExtensionSpec<2> for MarkerComponents {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "marker custom component GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

type MarkerComponentsRow = FiniteField<7, 2, Extension<MarkerComponents, 2>>;

/// A packed-storage spec whose spec type is `!Send` and `!Sync`.
struct MarkerPacked(PhantomData<*const ()>);

impl ExtensionSpec<2> for MarkerPacked {
    type Base = Aes8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "marker custom packed GF(2^16)";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x20, 0x01]);
}

type MarkerPackedRow = FiniteField<2, 16, Extension<MarkerPacked, 2>>;

/// One consumer per wrapper kind and field class.
///
/// The closure body is type-checked but never evaluated, so the checks cost
/// no run time.
const _: fn() = || {
    fn assert_markers<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}

    macro_rules! assert_fields {
        ($($field:ty),+ $(,)?) => {$(
            assert_markers::<Coeff<$field>>();
        )+};
    }

    assert_fields!(
        Aes8,
        Rs8,
        Normal8,
        Cantor8,
        Gf16,
        Gf32,
        Gf64,
        FanPaar8Row,
        FanPaar16Row,
        FanPaar32Row,
        FanPaar64Row,
        Mersenne31,
        Goldilocks,
        QuadMersenne31,
        FiniteField<2, 1>,
        FiniteField<251, 1>,
        FiniteField<65_521, 1>,
        FiniteField<1_000_000_007, 1>,
        FiniteField<4_294_967_291, 1>,
        FiniteField<18_446_744_073_709_551_557, 1>,
        CustomComponentsRow,
        CustomPackedRow,
        MarkerComponentsRow,
        MarkerPackedRow,
    );
};

#[cfg(feature = "alloc")]
const _: fn() = || {
    fn assert_markers<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}

    macro_rules! assert_collections {
        ($($field:ty),+ $(,)?) => {$(
            assert_markers::<ops::CoeffVec<$field>>();
            assert_markers::<ops::CoeffMatrix<$field>>();
            assert_markers::<ops::CoeffRef<'static, $field>>();
            assert_markers::<ops::CoeffVecRef<'static, $field>>();
        )+};
    }

    assert_collections!(
        Aes8,
        Gf16,
        Gf64,
        Mersenne31,
        Goldilocks,
        QuadMersenne31,
        FiniteField<251, 1>,
        FiniteField<1_000_000_007, 1>,
        FiniteField<18_446_744_073_709_551_557, 1>,
        CustomComponentsRow,
        CustomPackedRow,
        MarkerComponentsRow,
        MarkerPackedRow,
    );
};

#[test]
fn shipped_prepared_wrappers_keep_their_markers() {
    // The const consumers above are the assertion; this test exists so the
    // file runs in the ordinary test matrix and reports the guarantee.
    let coeff = Coeff::<Aes8>::new(Elem::<Aes8>::from_raw(0));
    assert_eq!(coeff.value().to_raw(), 0);
}
