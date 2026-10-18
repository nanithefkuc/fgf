//! Generic extension arithmetic against an independent reference.
//!
//! The reference is written inside this file against the defining data
//! alone: elements are flattened prime-subfield coordinate vectors,
//! products are plain schoolbook polynomial multiplication reduced by the
//! defining relation through recursive base multiplication, and inverses
//! are found by brute-force enumeration of the field. Exhaustive sweeps
//! cover every element pair of every row of order at most 625 — odd
//! relations with nonzero middle coefficients, the odd cubic and quartic,
//! and an odd depth-two tower included — and deterministic samples cover
//! the full-width prime rows.

use fgf::field::mersenne31::Mersenne31;
use fgf::{
    AES, Elem, Embedding, Extension, ExtensionField, ExtensionSpec, Field, FieldElem, FiniteField,
    Goldilocks, Normal, PackedPadded, Polynomial, PrimeIdentity, QuadMersenne31, Relation,
};

// ---------------------------------------------------------------------------
// Rows under test
// ---------------------------------------------------------------------------

/// GF(49) as `t^2 - 3` over the residue row at seven.
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

/// GF(4) as `t^2 + t + 1` over the residue row at two, which is not a
/// binary presentation.
struct Gf4Spec;
impl ExtensionSpec<2> for Gf4Spec {
    type Base = FiniteField<2, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(4) over residues";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
}
type Gf4Row = FiniteField<2, 2, Extension<Gf4Spec, 2>>;

/// GF(8) as `x^3 + x + 1` over GF(2).
struct Gf8Spec;
impl ExtensionSpec<3> for Gf8Spec {
    type Base = FiniteField<2, 1, Polynomial<3>>;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(8)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[1, 1, 0]);
}
type Gf8Row = FiniteField<2, 3, Extension<Gf8Spec, 3>>;

/// GF(64) as `t^2 + alpha*t + 1` over GF(8): a depth-two tower whose base
/// is a non-quadratic extension.
struct Gf64Spec;
impl ExtensionSpec<2> for Gf64Spec {
    type Base = Gf8Row;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(64)";
    const RELATION: Relation = Relation::from_binary_words::<Gf8Row>(&[1, 2]);
}
type Gf64Row = FiniteField<2, 6, Extension<Gf64Spec, 2>>;

/// GF(2^4) as `t^2 + t + omega` over the flat GF(4): component storage
/// over a binary base.
type FlatGf4 = FiniteField<2, 2, Polynomial<7>>;
struct BinaryComponentsSpec;
impl ExtensionSpec<2> for BinaryComponentsSpec {
    type Base = FlatGf4;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(2^4) in components";
    const RELATION: Relation = Relation::from_binary_words::<FlatGf4>(&[2, 1]);
}
type BinaryComponents = FiniteField<2, 4, Extension<BinaryComponentsSpec, 2>>;

/// Goldilocks squared as `t^2 - 7`.
struct GoldilocksSquareSpec;
impl ExtensionSpec<2> for GoldilocksSquareSpec {
    type Base = Goldilocks;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(g^2)";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}
type GoldilocksSquare = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GoldilocksSquareSpec, 2>>;

/// A custom GF(p^2) as `t^2 + 1` over Mersenne31, and `t^2 - (2 + i)`
/// above it: an extension over a custom extension base.
struct CustomComplexSpec;
impl ExtensionSpec<2> for CustomComplexSpec {
    type Base = Mersenne31;
    type Storage = fgf::Components;
    const NAME: &'static str = "custom GF(p^2)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type CustomComplex = FiniteField<0x7FFF_FFFF, 2, Extension<CustomComplexSpec, 2>>;

struct OverCustomSpec;
impl ExtensionSpec<2> for OverCustomSpec {
    type Base = CustomComplex;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(p^4) over custom";
    const RELATION: Relation = Relation::from_coordinates(&[0x7FFF_FFFD, 0x7FFF_FFFE, 0, 0]);
}
type OverCustom = FiniteField<0x7FFF_FFFF, 4, Extension<OverCustomSpec, 2>>;

/// The largest `u64` prime squared as `t^2 - 2`, two its smallest
/// quadratic nonresidue.
struct WidePrimeSquareSpec;
impl ExtensionSpec<2> for WidePrimeSquareSpec {
    type Base = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "wide GF(p^2)";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_FFFF_FFC5 - 2, 0]);
}
type WidePrimeSquare = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 2, Extension<WidePrimeSquareSpec, 2>>;

/// GF(49) as `t^2 + t + 3` over the residue row at seven: a non-binomial
/// odd relation.
struct Gf49FullSpec;
impl ExtensionSpec<2> for Gf49FullSpec {
    type Base = FiniteField<7, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(49) full relation";
    const RELATION: Relation = Relation::from_coordinates(&[3, 1]);
}
type Gf49Full = FiniteField<7, 2, Extension<Gf49FullSpec, 2>>;

/// GF(125) as `x^3 + x + 1` over the residue row at five.
struct Gf125Spec;
impl ExtensionSpec<3> for Gf125Spec {
    type Base = FiniteField<5, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(125)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1, 0]);
}
type Gf125 = FiniteField<5, 3, Extension<Gf125Spec, 3>>;

/// GF(343) as `t^3 - 2` over the residue row at seven: the odd cubic
/// Euclid inversion path.
struct Gf343Spec;
impl ExtensionSpec<3> for Gf343Spec {
    type Base = FiniteField<7, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(343)";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}
type Gf343 = FiniteField<7, 3, Extension<Gf343Spec, 3>>;

/// GF(625) as `t^4 + t^3 + t^2 + t + 3` over the residue row at five: a
/// quartic with every middle coefficient nonzero.
struct Gf625Spec;
impl ExtensionSpec<4> for Gf625Spec {
    type Base = FiniteField<5, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(625)";
    const RELATION: Relation = Relation::from_coordinates(&[3, 1, 1, 1]);
}
type Gf625 = FiniteField<5, 4, Extension<Gf625Spec, 4>>;

/// GF(9) as `a^2 + 1` over the residue row at three, and `t^2 - (1 + a)`
/// above it: an odd depth-two tower.
struct Gf9Spec;
impl ExtensionSpec<2> for Gf9Spec {
    type Base = FiniteField<3, 1>;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(9)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type Gf9Row = FiniteField<3, 2, Extension<Gf9Spec, 2>>;

struct Gf81Spec;
impl ExtensionSpec<2> for Gf81Spec {
    type Base = Gf9Row;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(81)";
    // `-(1 + a)` in base components: `(-1, -1)`, then the zero linear
    // coefficient.
    const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
}
type Gf81Row = FiniteField<3, 4, Extension<Gf81Spec, 2>>;

// ---------------------------------------------------------------------------
// The independent reference
// ---------------------------------------------------------------------------

/// One reference field: a prime field, or an extension by a monic
/// relation whose non-leading coefficients are flattened base coordinates,
/// constant term first.
enum Reference {
    /// The prime field of `p`.
    Prime(u64),
    /// The degree-`degree` extension of `base` by `relation`.
    Extension {
        base: Box<Reference>,
        degree: usize,
        relation: Vec<Vec<u64>>,
    },
}

impl Reference {
    /// The prime characteristic.
    fn characteristic(&self) -> u64 {
        match self {
            Reference::Prime(p) => *p,
            Reference::Extension { base, .. } => base.characteristic(),
        }
    }

    /// The flattened coordinate count of one element.
    fn words(&self) -> usize {
        match self {
            Reference::Prime(_) => 1,
            Reference::Extension { base, degree, .. } => base.words() * degree,
        }
    }

    /// The element count, `p^words`, at most 625 for every swept row.
    fn order(&self) -> usize {
        let mut order = 1usize;
        for _ in 0..self.words() {
            order *= self.characteristic() as usize;
        }
        order
    }

    /// The one element in flattened coordinates.
    fn one(&self) -> Vec<u64> {
        match self {
            Reference::Prime(_) => vec![1],
            Reference::Extension { base, .. } => {
                let mut one = vec![0u64; self.words()];
                let base_one = base.one();
                one[..base_one.len()].copy_from_slice(&base_one);
                one
            }
        }
    }

    /// Coordinatewise sum and negation over the prime field.
    fn add(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let p = self.characteristic();
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((u128::from(x) + u128::from(y)) % u128::from(p)) as u64)
            .collect()
    }

    fn neg(&self, a: &[u64]) -> Vec<u64> {
        let p = self.characteristic();
        a.iter().map(|&x| if x == 0 { 0 } else { p - x }).collect()
    }

    /// The schoolbook product reduced by the defining relation, through
    /// recursive base multiplication.
    fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        match self {
            Reference::Prime(p) => {
                vec![((u128::from(a[0]) * u128::from(b[0])) % u128::from(*p)) as u64]
            }
            Reference::Extension {
                base,
                degree,
                relation,
            } => {
                let words = base.words();
                let mut wide = vec![vec![0u64; words]; 2 * degree - 1];
                for i in 0..*degree {
                    for j in 0..*degree {
                        let product = base.mul(
                            &a[i * words..(i + 1) * words],
                            &b[j * words..(j + 1) * words],
                        );
                        let slot = &mut wide[i + j];
                        *slot = base.add(slot, &product);
                    }
                }
                // `t^degree = -(relation[degree-1] t^(degree-1) + ... )`.
                for k in (*degree..2 * degree - 1).rev() {
                    let top = wide[k].clone();
                    if top.iter().all(|&x| x == 0) {
                        continue;
                    }
                    for (j, coefficient) in relation.iter().enumerate() {
                        let product = base.mul(&top, coefficient);
                        let negated = base.neg(&product);
                        let slot = &mut wide[k - degree + j];
                        *slot = base.add(slot, &negated);
                    }
                }
                wide.truncate(*degree);
                wide.concat()
            }
        }
    }

    /// The inverse by brute-force enumeration, zero mapping to zero.
    fn inv(&self, a: &[u64]) -> Vec<u64> {
        if a.iter().all(|&x| x == 0) {
            return vec![0; self.words()];
        }
        let one = self.one();
        for index in 1..self.order() {
            let candidate = self.index(index);
            if self.mul(a, &candidate) == one {
                return candidate;
            }
        }
        panic!("the brute-force reference found no inverse");
    }

    /// The element of a flattened coordinate index.
    fn index(&self, mut index: usize) -> Vec<u64> {
        let p = self.characteristic() as usize;
        (0..self.words())
            .map(|_| {
                let digit = index % p;
                index /= p;
                digit as u64
            })
            .collect()
    }
}

/// The flattened prime coordinates of one row, in component order through
/// the base: the format the reference computes in.
trait Coordinates {
    /// Flattened coordinate count of one element.
    const WORDS: usize;
    /// The flattened coordinates of `self`.
    fn coordinates(self) -> Vec<u64>;
    /// The element of flattened coordinates.
    fn element(coordinates: &[u64]) -> Self;
}

macro_rules! leaf_coordinates {
    ($row:ty) => {
        impl Coordinates for Elem<$row> {
            const WORDS: usize = 1;
            fn coordinates(self) -> Vec<u64> {
                vec![
                    self.to_residue()
                        .expect("a degree-one element carries its residue"),
                ]
            }
            fn element(coordinates: &[u64]) -> Self {
                Self::from_residue(coordinates[0])
                    .expect("the reference stays below the characteristic")
            }
        }
    };
}

macro_rules! extension_coordinates {
    ($row:ty, $base:ty, $degree:literal) => {
        impl Coordinates for Elem<$row> {
            const WORDS: usize = <Elem<$base> as Coordinates>::WORDS * $degree;
            fn coordinates(self) -> Vec<u64> {
                <$row as ExtensionField<$degree>>::to_components(self)
                    .into_iter()
                    .flat_map(<Elem<$base> as Coordinates>::coordinates)
                    .collect()
            }
            fn element(coordinates: &[u64]) -> Self {
                let words = <Elem<$base> as Coordinates>::WORDS;
                let components: [_; $degree] = (0..$degree)
                    .map(|i| {
                        <Elem<$base> as Coordinates>::element(
                            &coordinates[i * words..(i + 1) * words],
                        )
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("the component count matches the relative degree");
                <$row as ExtensionField<$degree>>::from_components(components)
            }
        }
    };
}

leaf_coordinates!(FiniteField<7, 1>);
leaf_coordinates!(FiniteField<2, 1>);
leaf_coordinates!(FiniteField<2, 1, Polynomial<3>>);
leaf_coordinates!(FiniteField<3, 1>);
leaf_coordinates!(FiniteField<5, 1>);

/// The flat GF(4): one coordinate per bit of its byte word.
impl Coordinates for Elem<FlatGf4> {
    const WORDS: usize = 2;
    fn coordinates(self) -> Vec<u64> {
        let word = u64::from(self.to_raw());
        vec![word & 1, word >> 1]
    }
    fn element(coordinates: &[u64]) -> Self {
        FieldElem::from_raw(coordinates[0] as u8 | (coordinates[1] as u8) << 1)
    }
}

leaf_coordinates!(Mersenne31);
leaf_coordinates!(Goldilocks);
leaf_coordinates!(FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1>);

extension_coordinates!(Gf49, FiniteField<7, 1>, 2);
extension_coordinates!(Gf4Row, FiniteField<2, 1>, 2);
extension_coordinates!(Gf8Row, FiniteField<2, 1, Polynomial<3>>, 3);
extension_coordinates!(Gf64Row, Gf8Row, 2);
extension_coordinates!(BinaryComponents, FlatGf4, 2);
extension_coordinates!(GoldilocksSquare, Goldilocks, 2);
extension_coordinates!(CustomComplex, Mersenne31, 2);
extension_coordinates!(OverCustom, CustomComplex, 2);
extension_coordinates!(WidePrimeSquare, FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1>, 2);
extension_coordinates!(Gf49Full, FiniteField<7, 1>, 2);
extension_coordinates!(Gf125, FiniteField<5, 1>, 3);
extension_coordinates!(Gf343, FiniteField<7, 1>, 3);
extension_coordinates!(Gf625, FiniteField<5, 1>, 4);
extension_coordinates!(Gf9Row, FiniteField<3, 1>, 2);
extension_coordinates!(Gf81Row, Gf9Row, 2);
extension_coordinates!(OverNormal16, Normal4, 2);

/// The normal-basis GF(4): two index bits through the row's own
/// presentation, not the polynomial basis.
impl Coordinates for Elem<Normal4> {
    const WORDS: usize = 2;
    fn coordinates(self) -> Vec<u64> {
        let index = self.to_index();
        vec![(index & 1) as u64, ((index >> 1) & 1) as u64]
    }
    fn element(coordinates: &[u64]) -> Self {
        FieldElem::from_index(u128::from(coordinates[0]) | (u128::from(coordinates[1]) << 1))
            .expect("the reference stays below the order")
    }
}

// ---------------------------------------------------------------------------
// Exhaustive sweeps
// ---------------------------------------------------------------------------

/// Every element of a swept row, in flattened coordinate order.
fn every_element<F: Field>() -> Vec<Elem<F>>
where
    Elem<F>: Coordinates,
{
    (0..<F as Field>::ORDER as usize)
        .map(|index| {
            let mut digits = index as u128;
            let p = <F as Field>::Characteristic::CHARACTERISTIC as u128;
            let coordinates: Vec<u64> = (0..<Elem<F> as Coordinates>::WORDS)
                .map(|_| {
                    let digit = (digits % p) as u64;
                    digits /= p;
                    digit
                })
                .collect();
            <Elem<F> as Coordinates>::element(&coordinates)
        })
        .collect()
}

/// Sweep one row exhaustively against `reference`: every product pair,
/// every inverse by brute force, the defining relation at the
/// indeterminate, distributivity, the zero and one laws, and component
/// round trips.
fn sweep<F: ExtensionField<D>, const D: usize>(reference: &Reference)
where
    Elem<F>: Coordinates,
{
    let elements = every_element::<F>();
    let zero = Elem::<F>::ZERO;
    let one = Elem::<F>::ONE;
    assert_eq!(
        one.coordinates(),
        reference.one(),
        "{}: one",
        <F as Field>::NAME
    );
    let mut indeterminate_components = <F as ExtensionField<D>>::to_components(Elem::<F>::ZERO);
    indeterminate_components[1] = Elem::<<F as ExtensionField<D>>::Base>::ONE;
    let t = <F as ExtensionField<D>>::from_components(indeterminate_components);
    for (index, &a) in elements.iter().enumerate() {
        let a_coordinates = a.coordinates();
        // Inverse: identity and brute force.
        if a == zero {
            assert_eq!(a.inv(), zero);
        } else {
            assert_eq!(a.mul(a.inv()), one, "{}: a * a^-1", <F as Field>::NAME);
            assert_eq!(
                a.inv().coordinates(),
                reference.inv(&a_coordinates),
                "{}: brute-force inverse",
                <F as Field>::NAME
            );
        }
        // Component round trip.
        assert_eq!(
            <F as ExtensionField<D>>::from_components(<F as ExtensionField<D>>::to_components(a)),
            a,
            "{}: components",
            <F as Field>::NAME
        );
        // Zero and one laws.
        assert_eq!(a.add(zero), a);
        assert_eq!(a.sub(a), zero);
        assert_eq!(a.add(a.neg()), zero);
        assert_eq!(a.mul(one), a);
        for &b in &elements {
            // Product against the reference.
            let product = a.mul(b);
            assert_eq!(
                product.coordinates(),
                reference.mul(&a_coordinates, &b.coordinates()),
                "{}: product of {index}",
                <F as Field>::NAME
            );
            assert_eq!(a.mul(b), b.mul(a), "{}: commutativity", <F as Field>::NAME);
        }
        // Distributivity on the element itself.
        assert_eq!(
            a.mul(a.add(one)),
            a.mul(a).add(a),
            "{}: distributivity",
            <F as Field>::NAME
        );
        // Addition and negation against the reference.
        assert_eq!(
            a.add(t).coordinates(),
            reference.add(&a_coordinates, &t.coordinates()),
            "{}: addition",
            <F as Field>::NAME
        );
        assert_eq!(
            a.neg().coordinates(),
            reference.neg(&a_coordinates),
            "{}: negation",
            <F as Field>::NAME
        );
    }
    // `f(t) = 0`: the top product reduces through the defining relation,
    // exactly the reduction the reference applies.
    let mut powers = vec![one, t];
    while powers.len() < D {
        powers.push(powers[powers.len() - 1].mul(t));
    }
    let top = powers[D - 1].mul(t);
    assert_eq!(
        top.coordinates(),
        reference.mul(&powers[D - 1].coordinates(), &t.coordinates()),
        "{}: f(t) = 0",
        <F as Field>::NAME
    );
}

/// The reference of GF(49) as `t^2 - 3` over GF(7).
fn gf49_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(7)),
        degree: 2,
        relation: vec![vec![4], vec![0]],
    }
}

/// The reference of GF(4) over the residue row at two.
fn gf4_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(2)),
        degree: 2,
        relation: vec![vec![1], vec![1]],
    }
}

/// The reference of GF(8) as `x^3 + x + 1` over GF(2).
fn gf8_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(2)),
        degree: 3,
        relation: vec![vec![1], vec![1], vec![0]],
    }
}

/// The reference of GF(64) as `t^2 + alpha*t + 1` over GF(8), `alpha` the
/// base indeterminate in flattened coordinates `[0, 1, 0]`.
fn gf64_reference() -> Reference {
    Reference::Extension {
        base: Box::new(gf8_reference()),
        degree: 2,
        relation: vec![vec![1, 0, 0], vec![0, 1, 0]],
    }
}

/// The reference of GF(2^4) as `t^2 + t + omega` over the flat GF(4),
/// `omega` the base indeterminate with coordinates `[0, 1]`.
fn binary_components_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Extension {
            base: Box::new(Reference::Prime(2)),
            degree: 2,
            relation: vec![vec![1], vec![1]],
        }),
        degree: 2,
        relation: vec![vec![0, 1], vec![1, 0]],
    }
}

/// The reference of GF(49) as `t^2 + t + 3` over GF(7).
fn gf49_full_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(7)),
        degree: 2,
        relation: vec![vec![3], vec![1]],
    }
}

/// The reference of GF(125) as `x^3 + x + 1` over GF(5).
fn gf125_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(5)),
        degree: 3,
        relation: vec![vec![1], vec![1], vec![0]],
    }
}

/// The reference of GF(343) as `t^3 - 2` over GF(7).
fn gf343_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(7)),
        degree: 3,
        relation: vec![vec![5], vec![0], vec![0]],
    }
}

/// The reference of GF(625) as `t^4 + t^3 + t^2 + t + 3` over GF(5).
fn gf625_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(5)),
        degree: 4,
        relation: vec![vec![3], vec![1], vec![1], vec![1]],
    }
}

/// The reference of GF(81) as `t^2 - (1 + a)` over GF(9), itself
/// `a^2 + 1` over GF(3); `1 + a` has flattened coordinates `[1, 1]`, so
/// its negation is `[2, 2]`.
fn gf81_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Extension {
            base: Box::new(Reference::Prime(3)),
            degree: 2,
            relation: vec![vec![1], vec![0]],
        }),
        degree: 2,
        relation: vec![vec![2, 2], vec![0, 0]],
    }
}

#[test]
fn gf49_matches_the_reference_exhaustively() {
    sweep::<Gf49, 2>(&gf49_reference());
}

#[test]
fn gf4_over_residues_matches_the_reference_exhaustively() {
    sweep::<Gf4Row, 2>(&gf4_reference());
}

#[test]
fn gf8_matches_the_reference_exhaustively() {
    sweep::<Gf8Row, 3>(&gf8_reference());
}

#[test]
fn gf64_depth_two_tower_matches_the_reference_exhaustively() {
    sweep::<Gf64Row, 2>(&gf64_reference());
}

#[test]
fn binary_components_row_matches_the_reference_exhaustively() {
    sweep::<BinaryComponents, 2>(&binary_components_reference());
}

#[test]
fn gf49_full_relation_matches_the_reference_exhaustively() {
    sweep::<Gf49Full, 2>(&gf49_full_reference());
}

#[test]
fn gf125_matches_the_reference_exhaustively() {
    sweep::<Gf125, 3>(&gf125_reference());
}

#[test]
fn gf343_matches_the_reference_exhaustively() {
    sweep::<Gf343, 3>(&gf343_reference());
}

#[test]
fn gf625_quartic_matches_the_reference_exhaustively() {
    sweep::<Gf625, 4>(&gf625_reference());
}

#[test]
fn gf81_odd_depth_two_tower_matches_the_reference_exhaustively() {
    sweep::<Gf81Row, 2>(&gf81_reference());
}

// ---------------------------------------------------------------------------
// Full-width prime samples
// ---------------------------------------------------------------------------

/// A deterministic word generator for the sample sweeps.
struct Sample {
    state: u64,
}

impl Sample {
    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
}

/// Sample one extension row against `reference`: products, inverses,
/// distributivity, the relation, and component round trips.
fn sample<F: ExtensionField<D>, const D: usize>(reference: &Reference, seed: u64)
where
    Elem<F::Base>: Coordinates,
    Elem<F>: Coordinates,
{
    let mut rng = Sample { state: seed };
    let one = Elem::<F>::ONE;
    let mut indeterminate_components = <F as ExtensionField<D>>::to_components(Elem::<F>::ZERO);
    indeterminate_components[1] = Elem::<F::Base>::ONE;
    let t = <F as ExtensionField<D>>::from_components(indeterminate_components);
    for _ in 0..64 {
        let raw_a = rng.next();
        let raw_b = rng.next();
        let a_coordinates: Vec<u64> = (0..<Elem<F> as Coordinates>::WORDS)
            .map(|i| (raw_a.wrapping_mul(2_654_435_761 + i as u64)) % reference.characteristic())
            .collect();
        let b_coordinates: Vec<u64> = (0..<Elem<F> as Coordinates>::WORDS)
            .map(|i| (raw_b.wrapping_mul(4_050_323_379 + i as u64)) % reference.characteristic())
            .collect();
        let a = <Elem<F> as Coordinates>::element(&a_coordinates);
        let b = <Elem<F> as Coordinates>::element(&b_coordinates);
        assert_eq!(
            a.mul(b).coordinates(),
            reference.mul(&a_coordinates, &b_coordinates),
            "{}: sampled product",
            <F as Field>::NAME
        );
        assert_eq!(a.mul(a.inv()), one);
        // The inverse is checked through the reference product: brute-force
        // enumeration cannot reach full-width prime orders.
        assert_eq!(
            reference.mul(&a_coordinates, &a.inv().coordinates()),
            reference.one(),
            "{}: sampled inverse",
            <F as Field>::NAME
        );
        assert_eq!(
            a.mul(b.add(one)),
            a.mul(b).add(a),
            "{}: sampled distributivity",
            <F as Field>::NAME
        );
        assert_eq!(
            <F as ExtensionField<D>>::from_components(<F as ExtensionField<D>>::to_components(a)),
            a
        );
    }
    // The relation at the indeterminate, against the reference reduction.
    let mut powers = vec![one, t];
    while powers.len() < D {
        powers.push(powers[powers.len() - 1].mul(t));
    }
    let top = powers[D - 1].mul(t);
    assert_eq!(
        top.coordinates(),
        reference.mul(&powers[D - 1].coordinates(), &t.coordinates()),
        "{}: sampled f(t) = 0",
        <F as Field>::NAME
    );
}

/// The reference of the custom GF(p^2) as `t^2 + 1` over Mersenne31.
fn custom_complex_reference() -> Reference {
    Reference::Extension {
        base: Box::new(Reference::Prime(0x7FFF_FFFF)),
        degree: 2,
        relation: vec![vec![1], vec![0]],
    }
}

#[test]
fn goldilocks_square_matches_the_reference_on_samples() {
    sample::<GoldilocksSquare, 2>(
        &Reference::Extension {
            base: Box::new(Reference::Prime(0xFFFF_FFFF_0000_0001)),
            degree: 2,
            relation: vec![vec![0xFFFF_FFFF_0000_0001 - 7], vec![0]],
        },
        0x0B00,
    );
}

#[test]
fn wide_prime_square_matches_the_reference_on_samples() {
    sample::<WidePrimeSquare, 2>(
        &Reference::Extension {
            base: Box::new(Reference::Prime(0xFFFF_FFFF_FFFF_FFC5)),
            degree: 2,
            relation: vec![vec![0xFFFF_FFFF_FFFF_FFC5 - 2], vec![0]],
        },
        0x0D00,
    );
}

#[test]
fn mersenne_custom_tower_matches_the_reference_on_samples() {
    sample::<CustomComplex, 2>(&custom_complex_reference(), 0x1100);
    sample::<OverCustom, 2>(
        &Reference::Extension {
            base: Box::new(custom_complex_reference()),
            degree: 2,
            relation: vec![vec![0x7FFF_FFFD, 0x7FFF_FFFE], vec![0, 0]],
        },
        0x1300,
    );
}

// ---------------------------------------------------------------------------
// Degree-2 routing against the pow-based reference
// ---------------------------------------------------------------------------

/// The explicit pow-based reference of one base-into-row embedding: the
/// scalar oracle the routing must match. The Frobenius is `x^|Base|`,
/// the trace the sum of the two conjugates restricted to the source, and
/// the norm the product restricted to the source; membership is the
/// Frobenius fixed-point test.
fn pow_frobenius<S: Field, T: Field<Characteristic = S::Characteristic>>(
    value: Elem<T>,
) -> Elem<T> {
    value.pow(S::ORDER)
}

/// The pow-based trace reference: `x + x^|S|` restricted to the source.
fn pow_trace<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    value: Elem<T>,
) -> Elem<S> {
    embedding
        .restrict(value + pow_frobenius::<S, T>(value))
        .expect("a conjugate sum lies in the embedding image")
}

/// The pow-based norm reference: `x * x^|S|` restricted to the source.
fn pow_norm<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    value: Elem<T>,
) -> Elem<S> {
    embedding
        .restrict(value * pow_frobenius::<S, T>(value))
        .expect("a conjugate product lies in the embedding image")
}

/// Check one routed degree-2 pair element by element: the compact
/// `frobenius`, `trace`, `norm`, `contains`, and `restrict` agree with
/// the pow reference, and the prepared embedding agrees with both.
#[cfg(feature = "alloc")]
fn check_routed<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    value: Elem<T>,
) {
    use fgf::PreparedEmbedding;
    let prepared = PreparedEmbedding::<S, T>::new(embedding);
    let expected_frobenius = pow_frobenius::<S, T>(value);
    assert_eq!(
        embedding.frobenius(value),
        expected_frobenius,
        "{}: frobenius matches the pow reference",
        <T as Field>::NAME
    );
    assert_eq!(
        prepared.frobenius(value),
        expected_frobenius,
        "{}: prepared frobenius matches the pow reference",
        <T as Field>::NAME
    );
    let expected_trace = pow_trace::<S, T>(embedding, value);
    assert_eq!(
        embedding.trace(value),
        expected_trace,
        "{}: trace matches the pow reference",
        <T as Field>::NAME
    );
    assert_eq!(
        prepared.trace(value),
        expected_trace,
        "{}: prepared trace matches the pow reference",
        <T as Field>::NAME
    );
    let expected_norm = pow_norm::<S, T>(embedding, value);
    assert_eq!(
        embedding.norm(value),
        expected_norm,
        "{}: norm matches the pow reference",
        <T as Field>::NAME
    );
    assert_eq!(
        prepared.norm(value),
        expected_norm,
        "{}: prepared norm matches the pow reference",
        <T as Field>::NAME
    );
    let expected_contains = expected_frobenius == value;
    assert_eq!(
        embedding.contains(value),
        expected_contains,
        "{}: contains matches the fixed-point test",
        <T as Field>::NAME
    );
    assert_eq!(
        prepared.contains(value),
        expected_contains,
        "{}: prepared contains matches the fixed-point test",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.restrict(value).is_some(),
        expected_contains,
        "{}: restrict agrees with contains",
        <T as Field>::NAME
    );
    assert_eq!(
        prepared.restrict(value),
        embedding.restrict(value),
        "{}: prepared restrict matches the compact map",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.frobenius(embedding.frobenius(value)),
        value,
        "{}: the routed Frobenius is an involution",
        <T as Field>::NAME
    );
}

/// Check one routed degree-2 pair without the prepared types (no-alloc
/// builds): the compact maps agree with the pow reference alone.
#[cfg(not(feature = "alloc"))]
fn check_routed<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    value: Elem<T>,
) {
    let expected_frobenius = pow_frobenius::<S, T>(value);
    assert_eq!(
        embedding.frobenius(value),
        expected_frobenius,
        "{}: frobenius matches the pow reference",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.trace(value),
        pow_trace::<S, T>(embedding, value),
        "{}: trace matches the pow reference",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.norm(value),
        pow_norm::<S, T>(embedding, value),
        "{}: norm matches the pow reference",
        <T as Field>::NAME
    );
    let expected_contains = expected_frobenius == value;
    assert_eq!(
        embedding.contains(value),
        expected_contains,
        "{}: contains matches the fixed-point test",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.restrict(value).is_some(),
        expected_contains,
        "{}: restrict agrees with contains",
        <T as Field>::NAME
    );
    assert_eq!(
        embedding.frobenius(embedding.frobenius(value)),
        value,
        "{}: the routed Frobenius is an involution",
        <T as Field>::NAME
    );
}

/// Sweep the base-into-row pair `(Base, Row)` exhaustively: every target
/// element through the routed check.
fn sweep_pair<Base, Row>()
where
    Base: Field,
    Row: Field<Characteristic = Base::Characteristic>,
    Elem<Row>: Coordinates,
{
    let embedding = Embedding::<Base, Row>::new().expect("the base embeds in its row");
    for value in every_element::<Row>() {
        check_routed::<Base, Row>(&embedding, value);
    }
}

/// Sweep the base-into-row pair `(Base, Row)` by index order: every
/// target element through the routed check, for packed rows without a
/// `Coordinates` spelling.
fn sweep_indexed_pair<Base, Row>()
where
    Base: Field,
    Row: Field<Characteristic = Base::Characteristic>,
    Elem<Row>: FieldElem<Field = Row>,
{
    let embedding = Embedding::<Base, Row>::new().expect("the base embeds in its row");
    for index in 0..<Row as Field>::ORDER {
        let value = <Elem<Row> as FieldElem>::from_index(index).expect("index below the order");
        check_routed::<Base, Row>(&embedding, value);
    }
}

/// A quadratic extension of the normal-basis GF(4): the base word one is
/// raw 3, so raw 1 is not field one anywhere on this tower. `t^2 + u*t +
/// b` with `u` the base one (raw 3) and `b` the base element of raw 2 is
/// irreducible by the Artin-Schreier criterion.
type Normal4 = FiniteField<2, 2, Normal<7, 2>>;
struct OverNormal4Spec;
impl ExtensionSpec<2> for OverNormal4Spec {
    type Base = Normal4;
    type Storage = fgf::Components;
    const NAME: &'static str = "GF(16) over normal-basis GF(4)";
    const RELATION: Relation = Relation::from_binary_words::<Normal4>(&[2, 3]);
}
type OverNormal16 = FiniteField<2, 4, Extension<OverNormal4Spec, 2>>;

type Gf16Tower = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;
type FanPaar16Tower = FiniteField<2, 16, Extension<fgf::FanPaar16, 2>>;

#[test]
fn routed_degree2_matches_the_pow_reference_exhaustively() {
    // c1 = 0 (binomial), c1 = 1 in characteristic two, general c1 in odd
    // characteristic, and the normal-basis tower whose raw one is not 1.
    sweep_pair::<FiniteField<7, 1>, Gf49>();
    sweep_pair::<FiniteField<2, 1>, Gf4Row>();
    sweep_pair::<FiniteField<7, 1>, Gf49Full>();
    sweep_pair::<FiniteField<3, 1>, Gf9Row>();
    sweep_pair::<FlatGf4, BinaryComponents>();
    sweep_pair::<Normal4, OverNormal16>();
    sweep_pair::<Gf8Row, Gf64Row>();
    sweep_pair::<Gf9Row, Gf81Row>();
}

#[test]
fn routed_packed_towers_match_the_pow_reference_exhaustively() {
    sweep_indexed_pair::<FiniteField<2, 8, Polynomial<AES>>, Gf16Tower>();
    sweep_indexed_pair::<FiniteField<2, 8, Extension<fgf::FanPaar8, 2>>, FanPaar16Tower>();
}

/// Sample one relative-degree-2 row against the pow reference with the
/// deterministic word generator.
fn sample_pair<F>(seed: u64, characteristic: u64)
where
    F: ExtensionField<2> + Field<Characteristic = <F::Base as Field>::Characteristic>,
    F::Base: Field,
    Elem<F>: Coordinates,
{
    let embedding = Embedding::<F::Base, F>::new().expect("the base embeds in its row");
    let mut rng = Sample { state: seed };
    for _ in 0..64 {
        let raw = rng.next();
        let coordinates: Vec<u64> = (0..<Elem<F> as Coordinates>::WORDS)
            .map(|i| (raw.wrapping_mul(2_654_435_761 + i as u64)) % characteristic)
            .collect();
        check_routed::<F::Base, F>(&embedding, <Elem<F> as Coordinates>::element(&coordinates));
    }
}

/// Sample a packed binary quadratic tower row against the pow reference
/// with the deterministic word generator.
fn sample_packed_pair<Base, Row>(seed: u64)
where
    Base: fgf::BinaryField,
    Row: Field<Characteristic = <Base as Field>::Characteristic> + fgf::FieldBuffer,
    Elem<Row>: FieldElem<Field = Row>,
    Elem<Base>: FieldElem<Field = Base>,
{
    let embedding = Embedding::<Base, Row>::new().expect("the base embeds in its row");
    let mut rng = Sample { state: seed };
    for _ in 0..64 {
        let index = u128::from(rng.next()) % <Row as Field>::ORDER;
        let value = <Elem<Row> as FieldElem>::from_index(index).expect("index below the order");
        check_routed::<Base, Row>(&embedding, value);
    }
}

#[test]
fn quad_mersenne31_routing_matches_the_pow_reference_on_samples() {
    let embedding =
        Embedding::<Mersenne31, QuadMersenne31>::new().expect("Mersenne31 embeds in QM31");
    let mut rng = Sample { state: 0xC01u64 };
    for _ in 0..64 {
        let re = Elem::<Mersenne31>::from_raw((rng.next() % Mersenne31::ORDER as u64) as u32);
        let im = Elem::<Mersenne31>::from_raw((rng.next() % Mersenne31::ORDER as u64) as u32);
        let value = Elem::<QuadMersenne31>::from_components([re, im]);
        check_routed::<Mersenne31, QuadMersenne31>(&embedding, value);
        // The inherent `const` conjugate/norm agree with the routed maps.
        let conjugate = value.conjugate();
        assert_eq!(conjugate.conjugate(), value, "conjugation is an involution");
        assert_eq!(conjugate, embedding.frobenius(value));
        assert_eq!(embedding.norm(value), value.norm());
        assert_eq!(
            value * conjugate,
            Elem::<QuadMersenne31>::from_components([value.norm(), Elem::<Mersenne31>::ZERO])
        );
    }
}

#[test]
fn goldilocks_square_routing_matches_the_pow_reference_on_samples() {
    sample_pair::<GoldilocksSquare>(0x0B00, Goldilocks::ORDER as u64);
}

#[test]
fn wide_prime_square_routing_matches_the_pow_reference_on_samples() {
    sample_pair::<WidePrimeSquare>(0x0D00, 0xFFFF_FFFF_FFFF_FFC5);
}

#[test]
fn mersenne_custom_quadratic_routing_matches_the_pow_reference_on_samples() {
    sample_pair::<CustomComplex>(0x1100, Mersenne31::ORDER as u64);
}

#[test]
fn routed_wide_towers_match_the_pow_reference_on_samples() {
    sample_packed_pair::<Gf16Tower, FiniteField<2, 32, Extension<fgf::Rijndael32, 2>>>(0xBEEF);
    sample_packed_pair::<FanPaar16Tower, FiniteField<2, 32, Extension<fgf::FanPaar32, 2>>>(0xCAFE);
}
