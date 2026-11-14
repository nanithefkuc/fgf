//! Dense extension relations validated under the default const-evaluation lint.
//!
//! Const element construction forces validation; runtime field identities and
//! relation evaluation exercise the accepted rows through the public API.

use fgf::extension::{Components, Extension, ExtensionField, ExtensionSpec, Relation};
use fgf::prime::Residue;
use fgf::{Elem, Field, FieldElem, FiniteField};

type Prime3 = FiniteField<3, 1, Residue>;
/// The residue presentation of `2^31 - 1`.
type M31Prime = FiniteField<0x7FFF_FFFF, 1, Residue>;

// ---------------------------------------------------------------------------
// GF(3^40), GF(3^48), and GF(3^64): dense irreducible prime-base relations.

struct Gf3Dense40Spec;
impl ExtensionSpec<40> for Gf3Dense40Spec {
    type Base = Prime3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^40) dense";
    const RELATION: Relation = Relation::from_coordinates(&[
        2, 0, 2, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 1, 1,
        0, 2, 0, 0, 2, 1, 0, 1, 2, 1,
    ]);
}
type Gf3Dense40 = FiniteField<3, 40, Extension<Gf3Dense40Spec, 40>>;

struct Gf3Dense48Spec;
impl ExtensionSpec<48> for Gf3Dense48Spec {
    type Base = Prime3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^48) dense";
    const RELATION: Relation = Relation::from_coordinates(&[
        2, 0, 1, 1, 2, 0, 1, 1, 1, 0, 2, 0, 2, 2, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0, 2, 0, 1, 1, 0, 2,
        0, 2, 2, 0, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0, 0, 2, 1, 2,
    ]);
}
type Gf3Dense48 = FiniteField<3, 48, Extension<Gf3Dense48Spec, 48>>;

struct Gf3Dense64Spec;
impl ExtensionSpec<64> for Gf3Dense64Spec {
    type Base = Prime3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^64) dense";
    const RELATION: Relation = Relation::from_coordinates(&[
        1, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 0, 1, 2, 1, 1, 1, 2, 0, 2, 0,
        2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 0,
        0, 1, 1, 1,
    ]);
}
type Gf3Dense64 = FiniteField<3, 64, Extension<Gf3Dense64Spec, 64>>;

// ---------------------------------------------------------------------------
// GF(9) relative degree eight over the quadratic base t^2 + 1.

struct Gf9BaseSpec;
impl ExtensionSpec<2> for Gf9BaseSpec {
    type Base = Prime3;
    type Storage = Components;
    const NAME: &'static str = "GF(9)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type Gf9Base = FiniteField<3, 2, Extension<Gf9BaseSpec, 2>>;

struct Gf9Dense8Spec;
impl ExtensionSpec<8> for Gf9Dense8Spec {
    type Base = Gf9Base;
    type Storage = Components;
    const NAME: &'static str = "GF(3^16) dense";
    const RELATION: Relation =
        Relation::from_coordinates(&[1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0, 0, 0, 2]);
}
type Gf9Dense8 = FiniteField<3, 16, Extension<Gf9Dense8Spec, 8>>;

// ---------------------------------------------------------------------------
// Mersenne31 relative degree four.

struct M31Dense4Spec;
impl ExtensionSpec<4> for M31Dense4Spec {
    type Base = M31Prime;
    type Storage = Components;
    const NAME: &'static str = "GF((2^31 - 1)^4) dense";
    const RELATION: Relation =
        Relation::from_coordinates(&[535_977_110, 860_917_705, 1_097_133_730, 1_919_007_738]);
}
type M31Dense4 = FiniteField<0x7FFF_FFFF, 4, Extension<M31Dense4Spec, 4>>;

// ---------------------------------------------------------------------------
// Compile-time acceptance: constructing an element evaluates irreducibility.

const _: Elem<Gf3Dense40> = Elem::<Gf3Dense40>::ZERO;

const _: Elem<Gf3Dense48> = Elem::<Gf3Dense48>::ZERO;

const _: Elem<Gf3Dense64> = Elem::<Gf3Dense64>::ZERO;

const _: Elem<Gf9Dense8> = Elem::<Gf9Dense8>::ZERO;

const _: Elem<M31Dense4> = Elem::<M31Dense4>::ZERO;

// ---------------------------------------------------------------------------
// Runtime relation and arithmetic checks.

/// The indeterminate of an extension row: component one.
fn row_indeterminate<F: ExtensionField<D>, const D: usize>() -> Elem<F> {
    let mut components = <F as ExtensionField<D>>::to_components(Elem::<F>::ZERO);
    components[1] = Elem::<F::Base>::ONE;
    <F as ExtensionField<D>>::from_components(components)
}

/// Evaluates `t^D + c_{D-1} t^(D-1) + ... + c_0` for a prime-base row and
/// reports whether the defining relation vanishes at `t`.
fn gf3_relation_holds<F: ExtensionField<D, Base = Prime3>, const D: usize>(
    relation: &[u64],
) -> bool {
    let t = row_indeterminate::<F, D>();
    let mut value = Elem::<F>::ZERO;
    let mut index = D;
    while index > 0 {
        index -= 1;
        let mut components = [Elem::<Prime3>::ZERO; D];
        components[0] = Elem::<Prime3>::from_raw(relation[index]);
        let coefficient = F::from_components(components);
        value = value.mul(t).add(coefficient);
    }
    value
        .add(t.pow(u128::try_from(D).expect("degree fits in u128")))
        .is_zero()
}

/// The GF(9) relative-degree-eight relation, whose coefficients are base
/// coordinates in pairs.
fn gf9_relation_holds(relation: &[u64]) -> bool {
    let t = row_indeterminate::<Gf9Dense8, 8>();
    let mut value = Elem::<Gf9Dense8>::ZERO;
    let mut index = 8;
    while index > 0 {
        index -= 1;
        let coefficient = Gf9Base::from_components([
            Elem::<Prime3>::from_raw(relation[2 * index]),
            Elem::<Prime3>::from_raw(relation[2 * index + 1]),
        ]);
        let mut components = [Elem::<Gf9Base>::ZERO; 8];
        components[0] = coefficient;
        value = value.mul(t).add(Gf9Dense8::from_components(components));
    }
    value.add(t.pow(8)).is_zero()
}

/// The Mersenne31 relative-degree-four relation over the residue presentation.
fn m31_relation_holds(relation: &[u64]) -> bool {
    let t = row_indeterminate::<M31Dense4, 4>();
    let mut value = Elem::<M31Dense4>::ZERO;
    let mut index = 4;
    while index > 0 {
        index -= 1;
        let mut components = [Elem::<M31Prime>::ZERO; 4];
        components[0] = Elem::<M31Prime>::from_raw(relation[index]);
        value = value.mul(t).add(M31Dense4::from_components(components));
    }
    value.add(t.pow(4)).is_zero()
}

/// Tower arithmetic implied by an irreducible defining relation: the
/// indeterminate is nonzero and invertible, and it satisfies the group law.
fn exercise_tower<F: ExtensionField<D>, const D: usize>() {
    let t = row_indeterminate::<F, D>();
    assert!(t.mul(t.inv()).is_one());
    assert!(t.pow(<F as Field>::ORDER - 1).is_one());
    assert_eq!(t.pow(<F as Field>::ORDER), t);
}

pub fn exercise() {
    let relation = <Gf3Dense40Spec as ExtensionSpec<40>>::RELATION;
    assert!(gf3_relation_holds::<Gf3Dense40, 40>(
        relation.to_coordinates()
    ));
    exercise_tower::<Gf3Dense40, 40>();

    let relation = <Gf3Dense48Spec as ExtensionSpec<48>>::RELATION;
    assert!(gf3_relation_holds::<Gf3Dense48, 48>(
        relation.to_coordinates()
    ));
    exercise_tower::<Gf3Dense48, 48>();

    let relation = <Gf3Dense64Spec as ExtensionSpec<64>>::RELATION;
    assert!(gf3_relation_holds::<Gf3Dense64, 64>(
        relation.to_coordinates()
    ));
    exercise_tower::<Gf3Dense64, 64>();

    let relation = <Gf9Dense8Spec as ExtensionSpec<8>>::RELATION;
    assert!(gf9_relation_holds(relation.to_coordinates()));
    exercise_tower::<Gf9Dense8, 8>();

    let relation = <M31Dense4Spec as ExtensionSpec<4>>::RELATION;
    assert!(m31_relation_holds(relation.to_coordinates()));
    exercise_tower::<M31Dense4, 4>();
}
