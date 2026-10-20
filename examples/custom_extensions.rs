//! Custom extension fields from caller-supplied specifications.
//!
//! Run with `just example custom_extensions`, optionally with
//! `--no-default-features`. An [`ExtensionSpec`] is data only — a base row,
//! a storage, a name, and the defining relation — and the relation is
//! validated in const before any value exists, so an irreducible cubic
//! over GF(5) builds GF(125) exactly as the shipped towers are built.
//!
//! The odd-prime half runs component arithmetic on quadratic GF(49) and
//! cubic GF(125), each with the total inverse convention `inv(0) == 0`.
//! The binary half spells one relation — `t^2 + t + 0x20` over the AES
//! polynomial field — through three storages. `Components` keeps one base
//! element per component and carries a byte form with the packed operation
//! surface. `Packed<u16>` fills its word exactly, and because this spec's
//! presentation equals a shipped instance's, it runs that instance's
//! specialized body and kernels. `PackedPadded<u32>` keeps sixteen zero
//! padding bits and carries neither byte form nor packed operations, so it
//! speaks only through scalar element arithmetic.

#![forbid(unsafe_code)]

use fgf::poly::AES;
use fgf::{
    Backend, Components, Elem, Extension, ExtensionField, ExtensionSpec, Field, FieldBuffer,
    FieldElem, FiniteField, Packed, PackedPadded, Polynomial, Relation, Rijndael16, backend_for,
    ops,
};

// Callers name their own fields; the canonical `FiniteField<P, N, R>` type is the API.
type Gf7 = FiniteField<7, 1>;
type Gf5 = FiniteField<5, 1>;
type Base8 = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;

/// GF(49) as `t^2 - 3` over GF(7), one base element per component.
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = Gf7;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    // `t^2 - 3`: constant coefficient `4 = -3` first, then the linear one.
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

/// GF(125) as `t^3 + t + 1` over GF(5), one base element per component.
struct Gf125Spec;
impl ExtensionSpec<3> for Gf125Spec {
    type Base = Gf5;
    type Storage = Components;
    const NAME: &'static str = "GF(125)";
    // `t^3 + t + 1`: constant, linear, then the zero quadratic
    // coefficient.
    const RELATION: Relation = Relation::from_coordinates(&[1, 1, 0]);
}
type Gf125 = FiniteField<5, 3, Extension<Gf125Spec, 3>>;

/// `t^2 + t + 0x20` over the AES field, coordinates packed into one `u16`.
struct Packed16Spec;
impl ExtensionSpec<2> for Packed16Spec {
    type Base = Base8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "custom GF(2^16), packed";
    const RELATION: Relation = Relation::from_binary_words::<Base8>(&[0x20, 0x01]);
}
type Packed16 = FiniteField<2, 16, Extension<Packed16Spec, 2>>;

/// The same relation in a padded `u32`: the coordinate word plus sixteen
/// always-zero bits.
struct Padded16Spec;
impl ExtensionSpec<2> for Padded16Spec {
    type Base = Base8;
    type Storage = PackedPadded<u32>;
    const NAME: &'static str = "custom GF(2^16), padded";
    const RELATION: Relation = Relation::from_binary_words::<Base8>(&[0x20, 0x01]);
}
type Padded16 = FiniteField<2, 16, Extension<Padded16Spec, 2>>;

/// The same relation over one-component-per-base-element storage.
struct Component16Spec;
impl ExtensionSpec<2> for Component16Spec {
    type Base = Base8;
    type Storage = Components;
    const NAME: &'static str = "custom GF(2^16), components";
    const RELATION: Relation = Relation::from_binary_words::<Base8>(&[0x20, 0x01]);
}
type Component16 = FiniteField<2, 16, Extension<Component16Spec, 2>>;

const LANES: usize = 9;

fn main() {
    gf49();
    gf125();
    storages();
}

/// Quadratic GF(49): component arithmetic, byte form, total inverse.
fn gf49() {
    // The identities are const, over the base identities.
    const ONE: Elem<Gf49> = Elem::<Gf49>::ONE;

    // The indeterminate: components `(0, 1)`.
    let t = <Gf49 as ExtensionField<2>>::from_components([Elem::<Gf7>::ZERO, Elem::<Gf7>::ONE]);

    // The relation folds the square into the base: `t^2 = 3`.
    let square = t * t;
    let [constant, linear] = <Gf49 as ExtensionField<2>>::to_components(square);
    assert_eq!((constant.to_raw(), linear.to_raw()), (3, 0));

    // Component arithmetic is field arithmetic: distribution over fixed
    // component triples.
    let (p, q, r) = (
        <Gf49 as ExtensionField<2>>::from_components([
            Elem::<Gf7>::from_raw(2),
            Elem::<Gf7>::from_raw(5),
        ]),
        <Gf49 as ExtensionField<2>>::from_components([
            Elem::<Gf7>::from_raw(6),
            Elem::<Gf7>::from_raw(1),
        ]),
        <Gf49 as ExtensionField<2>>::from_components([
            Elem::<Gf7>::from_raw(3),
            Elem::<Gf7>::from_raw(4),
        ]),
    );
    assert_eq!((p + q) * r, p * r + q * r);

    // Inversion is total: zero inverts to zero, every nonzero element
    // round-trips, and the nonzero group has order 48.
    assert_eq!(Elem::<Gf49>::ZERO.inv(), Elem::<Gf49>::ZERO);
    let w = t + ONE;
    assert_eq!(w * w.inv(), ONE);
    assert_eq!(t.pow(48), ONE);

    // The byte form concatenates the base encodings: two one-byte lanes,
    // constant component first.
    let mut bytes = [0u8; <Gf49 as FieldBuffer>::BYTES];
    <Gf49 as FieldBuffer>::encode(&mut bytes, w);
    assert_eq!(bytes, [1, 1]);
    assert_eq!(<Gf49 as FieldBuffer>::decode(&bytes), w);

    println!(
        "{}: t^2 = 3, (t + 1) has bytes {bytes:?}, inversion total",
        Gf49::NAME
    );
}

/// Cubic GF(125): schoolbook components under a degree-three relation.
fn gf125() {
    let t = <Gf125 as ExtensionField<3>>::from_components([
        Elem::<Gf5>::ZERO,
        Elem::<Gf5>::ONE,
        Elem::<Gf5>::ZERO,
    ]);

    // The relation folds the cube: `t^3 = -t - 1 = 4t + 4`.
    let cube = t * t * t;
    let [constant, linear, quadratic] = <Gf125 as ExtensionField<3>>::to_components(cube);
    assert_eq!(
        (constant.to_raw(), linear.to_raw(), quadratic.to_raw()),
        (4, 4, 0)
    );

    // Above relative degree two, inversion runs extended Euclid: still
    // total, still exact on nonzero elements, and the nonzero group has
    // order 124.
    assert_eq!(Elem::<Gf125>::ZERO.inv(), Elem::<Gf125>::ZERO);
    let v = <Gf125 as ExtensionField<3>>::from_components([
        Elem::<Gf5>::from_raw(1),
        Elem::<Gf5>::from_raw(3),
        Elem::<Gf5>::from_raw(2),
    ]);
    assert_eq!(v * v.inv(), Elem::<Gf125>::ONE);
    assert_eq!(t.pow(124), Elem::<Gf125>::ONE);

    println!(
        "{}: t^3 + t + 1 folds the cube to {cube}, inversion total",
        Gf125::NAME
    );
}

/// One relation through three storages: what each storage carries.
fn storages() {
    // The packed spec's presentation equals the shipped Rijndael GF(2^16)
    // instance, so it runs that instance's specialized body: equal
    // presentations produce equal bytes.
    let (a, b) = (0x53C7u16, 0x1D2Eu16);
    let packed_product = Elem::<Packed16>::from_raw(a) * Elem::<Packed16>::from_raw(b);
    let shipped_product = Elem::<Gf16>::from_raw(a) * Elem::<Gf16>::from_raw(b);
    assert_eq!(packed_product.to_raw(), shipped_product.to_raw());
    assert_eq!(backend_for::<Packed16>(), backend_for::<Gf16>());

    // The padded storage canonicalizes by clearing the padding bits, and
    // the coordinate word sits in the low half of its wider word.
    let padded_product = Elem::<Padded16>::from_raw(0xBEEF_0000 | u32::from(a))
        * Elem::<Padded16>::from_raw(u32::from(b));
    assert_eq!(padded_product.to_raw(), u32::from(packed_product.to_raw()));
    assert_eq!(
        Elem::<Padded16>::from_raw(0xBEEF_0000 | u32::from(a)).to_raw(),
        u32::from(a)
    );

    // Components keep one base element per side, low component first; the
    // same product lands in the component pair, and the row reports the
    // scalar backend.
    let component = |lo: u8, hi: u8| {
        <Component16 as ExtensionField<2>>::from_components([
            Elem::<Base8>::from_raw(lo),
            Elem::<Base8>::from_raw(hi),
        ])
    };
    let component_product = component(a as u8, (a >> 8) as u8) * component(b as u8, (b >> 8) as u8);
    let [low, high] = <Component16 as ExtensionField<2>>::to_components(component_product);
    let product_word = packed_product.to_raw();
    assert_eq!(
        (low.to_raw(), high.to_raw()),
        (product_word as u8, (product_word >> 8) as u8)
    );
    assert_eq!(backend_for::<Component16>(), Backend::Scalar);

    // Packed operations run over the byte form from stack buffers, with
    // one coefficient prepared once and reused across calls.
    let xs: [Elem<Component16>; LANES] =
        std::array::from_fn(|i| component((i * 13 + 5) as u8, (i * 29 + 3) as u8));
    let ys: [Elem<Component16>; LANES] =
        std::array::from_fn(|i| component((i * 7 + 11) as u8, (i * 19 + 1) as u8));
    let k = component(0x7B, 0x2A);
    let prepared = ops::Coeff::<Component16>::new(k);

    const WIDTH: usize = LANES * <Component16 as FieldBuffer>::BYTES;
    let mut xs_packed = [0u8; WIDTH];
    ops::pack::<Component16>(&mut xs_packed, &xs);
    let mut ys_packed = [0u8; WIDTH];
    ops::pack::<Component16>(&mut ys_packed, &ys);

    // `dst` starts at zero, so one accumulate scales `xs`, and a second
    // accumulate with the same prepared coefficient adds `k * ys`.
    let mut acc = [0u8; WIDTH];
    ops::mul_add_with::<Component16>(&mut acc, &prepared, &xs_packed);
    ops::mul_add_with::<Component16>(&mut acc, &prepared, &ys_packed);
    ops::add_assign::<Component16>(&mut acc, &xs_packed);
    ops::sub_assign::<Component16>(&mut acc, &xs_packed);

    let mut lanes = [Elem::<Component16>::ZERO; LANES];
    ops::unpack::<Component16>(&mut lanes, &acc);
    for (lane, (&x, &y)) in lanes.iter().zip(xs.iter().zip(&ys)) {
        assert_eq!(*lane, k * x + k * y, "lane agrees with element arithmetic");
    }

    println!(
        "{} ({:?}) = {} = {}: one relation, three storages, one arithmetic",
        Component16::NAME,
        backend_for::<Component16>(),
        Packed16::NAME,
        Padded16::NAME
    );
}
