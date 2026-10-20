//! The registered field presentations and their routing identities.
//!
//! Every panel iterates the same two lists: [`for_each_field`] covers all
//! forty-five representative presentations for scalar, conversion, and query
//! surfaces; [`for_each_bulk_field`] covers the bulk-kernel subset. Labels
//! are the exact `field` strings of every record and are frozen across
//! rounds and hosts. The custom specs are user-side [`ExtensionSpec`]
//! implementations exercising the open extension surface: a custom
//! Rijndael-constant tower, packed towers over the RS polynomial and over
//! normal- and Cantor-basis bases, and an odd-degree `PackedPadded` row that
//! is scalar-only by construction. Every relation constant was chosen so the
//! defining polynomial is irreducible — `t²+t+c` over a binary base needs
//! trace one on `c`, which the row validation re-proves at first use.

use fgf::{
    AES, Cantor, Components, Extension, ExtensionSpec, FanPaar2, FanPaar4, FanPaar8, FanPaar16,
    FanPaar32, FanPaar64, FiniteField, Goldilocks, KernelBackend, Mersenne31, Normal, Packed,
    PackedPadded, Polynomial, QuadMersenne31, RS, Relation, Residue, Rijndael16, Rijndael32,
    Rijndael64, backend_for,
};

// --- shipped rows -----------------------------------------------------------

pub type Gf2p1Poly = FiniteField<2, 1, Polynomial<3>>;
pub type Gf2p2Poly = FiniteField<2, 2, Polynomial<7>>;
pub type Gf2p4Poly = FiniteField<2, 4, Polynomial<19>>;
pub type Gf2p8Aes = FiniteField<2, 8, Polynomial<AES>>;
pub type Gf2p8Rs = FiniteField<2, 8, Polynomial<RS>>;
pub type Gf2p2Normal = FiniteField<2, 2, Normal<0x7, 2>>;
pub type Gf2p4Normal = FiniteField<2, 4, Normal<0x13, 8>>;
pub type Gf2p8Normal = FiniteField<2, 8, Normal<AES, 0x20>>;
pub type Gf2p2Cantor = FiniteField<2, 2, Cantor<0x7, 2>>;
pub type Gf2p4Cantor = FiniteField<2, 4, Cantor<0x13, 8>>;
pub type Gf2p8Cantor = FiniteField<2, 8, Cantor<AES, 0x20>>;
pub type Gf2p2FanPaar = FiniteField<2, 2, Extension<FanPaar2, 2>>;
pub type Gf2p4FanPaar = FiniteField<2, 4, Extension<FanPaar4, 2>>;
pub type Gf2p8FanPaar = FiniteField<2, 8, Extension<FanPaar8, 2>>;
pub type Gf2p16FanPaar = FiniteField<2, 16, Extension<FanPaar16, 2>>;
pub type Gf2p32FanPaar = FiniteField<2, 32, Extension<FanPaar32, 2>>;
pub type Gf2p64FanPaar = FiniteField<2, 64, Extension<FanPaar64, 2>>;
pub type Gf2p16Rijndael = FiniteField<2, 16, Extension<Rijndael16, 2>>;
pub type Gf2p32Rijndael = FiniteField<2, 32, Extension<Rijndael32, 2>>;
pub type Gf2p64Rijndael = FiniteField<2, 64, Extension<Rijndael64, 2>>;

// --- custom extensions ------------------------------------------------------

/// A custom packed tower over the RS polynomial: `t² + t + 0x21`, whose
/// constant has binary trace one, unlike the shipped `0x20` tower.
pub struct CustomRs16;

impl ExtensionSpec<2> for CustomRs16 {
    type Base = Gf2p8Rs;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/RS custom";
    const RELATION: Relation = Relation::from_binary_words::<Gf2p8Rs>(&[0x21, 0x01]);
}

/// A custom packed tower over the AES polynomial: `t² + t + 0x2C`.
pub struct CustomRijndael16;

impl ExtensionSpec<2> for CustomRijndael16 {
    type Base = Gf2p8Aes;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/AES custom";
    const RELATION: Relation = Relation::from_binary_words::<Gf2p8Aes>(&[0x2C, 0x01]);
}

/// A custom packed tower over the shipped normal basis. Both relation
/// coefficients use word `0x01`, not the normal-basis unit `0xFF`:
/// `t² + βt + β`, where `β` is basis vector zero.
pub struct CustomNormal16;

impl ExtensionSpec<2> for CustomNormal16 {
    type Base = Gf2p8Normal;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/normal custom";
    const RELATION: Relation = Relation::from_binary_words::<Gf2p8Normal>(&[0x01, 0x01]);
}

/// A custom packed tower over the shipped Cantor basis. In Cantor
/// coordinates only the top basis vector carries binary trace one, so the
/// constant word `0x80` has trace one.
pub struct CustomCantor16;

impl ExtensionSpec<2> for CustomCantor16 {
    type Base = Gf2p8Cantor;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/cantor custom";
    const RELATION: Relation = Relation::from_binary_words::<Gf2p8Cantor>(&[0x80, 0x01]);
}

/// An odd-degree `PackedPadded` row: `t⁵ + t² + 1` over GF(2), five
/// coordinate bits in an eight-bit word. Five-bit lanes have no vector
/// kernel route, so every bulk operation here runs the portable path.
pub struct OddPadded5;

impl ExtensionSpec<5> for OddPadded5 {
    type Base = Gf2p1Poly;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(2^5) padded";
    const RELATION: Relation = Relation::from_binary_words::<Gf2p1Poly>(&[1, 0, 1, 0, 0]);
}

pub type Gf2p16CustomRs = FiniteField<2, 16, Extension<CustomRs16, 2>>;
pub type Gf2p16CustomRijndael = FiniteField<2, 16, Extension<CustomRijndael16, 2>>;
pub type Gf2p16CustomNormal = FiniteField<2, 16, Extension<CustomNormal16, 2>>;
pub type Gf2p16CustomCantor = FiniteField<2, 16, Extension<CustomCantor16, 2>>;
pub type Gf2p5Padded = FiniteField<2, 5, Extension<OddPadded5, 5>>;

// --- residue rows and specialized primes ------------------------------------

pub type Gf2p1Residue = FiniteField<2, 1>;
pub type Gf101 = FiniteField<101, 1>;
pub type Gf127 = FiniteField<127, 1>;
pub type Gf131 = FiniteField<131, 1>;
pub type Gf65521 = FiniteField<65_521, 1>;
pub type Gf65537 = FiniteField<65_537, 1>;
pub type Gf1e9p7 = FiniteField<1_000_000_007, 1>;
pub type GfU32Max = FiniteField<4_294_967_291, 1>;
/// The largest `u64` prime.
pub const MAX_PRIME: u64 = 18_446_744_073_709_551_557;
pub type GfMaxPrime = FiniteField<MAX_PRIME, 1>;
pub type M31Residue = FiniteField<0x7FFF_FFFF, 1, Residue>;
pub type GoldilocksResidue = FiniteField<0xFFFF_FFFF_0000_0001, 1, Residue>;

// --- odd-characteristic and wide prime extensions ----------------------------

/// `t² + t + 3` over GF(7): the discriminant `1 − 12 ≡ 3` is a quadratic
/// non-residue modulo seven.
pub struct Gf49Spec;

impl ExtensionSpec<2> for Gf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(7^2)";
    const RELATION: Relation = Relation::from_coordinates(&[3, 1]);
}

/// `t³ + 2t + 1` over GF(5): no root in GF(5), and a cubic without a root is
/// irreducible.
pub struct Gf125Spec;

impl ExtensionSpec<3> for Gf125Spec {
    type Base = FiniteField<5, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(5^3)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 2, 0]);
}

/// `t² + 1` over GF(3): no root, since squares modulo three are `{0, 1}`.
pub struct Gf9Spec;

impl ExtensionSpec<2> for Gf9Spec {
    type Base = FiniteField<3, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(3^2)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}

/// `t² + (1+u)` over GF(9), where `u² = -1`. The constant is a
/// nonsquare: `(1+u)^4 = -1`, and `-1` is a square in GF(9).
pub struct Gf81Spec;

impl ExtensionSpec<2> for Gf81Spec {
    type Base = FiniteField<3, 2, Extension<Gf9Spec, 2>>;
    type Storage = Components;
    const NAME: &'static str = "GF(3^4)";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1, 0, 0]);
}

/// `t² + 7` over Goldilocks: seven is the smallest quadratic non-residue of
/// `2⁶⁴ − 2³² + 1`.
pub struct GoldilocksSquareSpec;

impl ExtensionSpec<2> for GoldilocksSquareSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(goldilocks^2)";
    const RELATION: Relation = Relation::from_coordinates(&[7, 0]);
}

/// `t² + 2` over the largest `u64` prime: two is a quadratic non-residue.
pub struct MaxPrimeSquareSpec;

impl ExtensionSpec<2> for MaxPrimeSquareSpec {
    type Base = GfMaxPrime;
    type Storage = Components;
    const NAME: &'static str = "GF(maxprime^2)";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0]);
}

pub type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;
pub type Gf125 = FiniteField<5, 3, Extension<Gf125Spec, 3>>;
pub type Gf9 = FiniteField<3, 2, Extension<Gf9Spec, 2>>;
pub type Gf81 = FiniteField<3, 4, Extension<Gf81Spec, 2>>;
pub type GoldilocksSquared =
    FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GoldilocksSquareSpec, 2>>;
pub type MaxPrimeSquared = FiniteField<MAX_PRIME, 2, Extension<MaxPrimeSquareSpec, 2>>;

/// The specialized presentations, spelled through their own rows.
pub type M31 = Mersenne31;
pub type Gld = Goldilocks;
pub type Qm31 = QuadMersenne31;

/// The backend name this field's bulk operations resolve to in the running
/// process: each field's own routing, not the process backend.
pub fn backend_name<F: fgf::FieldKernels>() -> String {
    backend_for::<F>().name().to_string()
}

/// Scalar execution route for arithmetic and presentation-only operations.
pub fn scalar_backend_name() -> String {
    "scalar".to_string()
}

/// Iterate every representative presentation: the full forty-five-field list
/// for scalar, conversion, and query panels.
#[allow(unused_macros)]
macro_rules! for_each_field {
    ($callback:ident) => {
        $callback!("gf2p8-aes", crate::fields::Gf2p8Aes);
        $callback!("gf2p8-rs", crate::fields::Gf2p8Rs);
        $callback!("gf2p8-normal", crate::fields::Gf2p8Normal);
        $callback!("gf2p8-cantor", crate::fields::Gf2p8Cantor);
        $callback!("gf2p8-fanpaar", crate::fields::Gf2p8FanPaar);
        $callback!("gf2p16-fanpaar", crate::fields::Gf2p16FanPaar);
        $callback!("gf2p32-fanpaar", crate::fields::Gf2p32FanPaar);
        $callback!("gf2p64-fanpaar", crate::fields::Gf2p64FanPaar);
        $callback!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael);
        $callback!("gf2p32-rijndael", crate::fields::Gf2p32Rijndael);
        $callback!("gf2p64-rijndael", crate::fields::Gf2p64Rijndael);
        $callback!("gf2p16-custom-rs", crate::fields::Gf2p16CustomRs);
        $callback!(
            "gf2p16-custom-rijndael",
            crate::fields::Gf2p16CustomRijndael
        );
        $callback!("gf2p16-custom-normal", crate::fields::Gf2p16CustomNormal);
        $callback!("gf2p16-custom-cantor", crate::fields::Gf2p16CustomCantor);
        $callback!("gf2p1-residue", crate::fields::Gf2p1Residue);
        $callback!("gf101", crate::fields::Gf101);
        $callback!("gf127", crate::fields::Gf127);
        $callback!("gf131", crate::fields::Gf131);
        $callback!("gf65521", crate::fields::Gf65521);
        $callback!("gf65537", crate::fields::Gf65537);
        $callback!("gf1e9p7", crate::fields::Gf1e9p7);
        $callback!("gfu32max", crate::fields::GfU32Max);
        $callback!("gfmaxp", crate::fields::GfMaxPrime);
        $callback!("m31", crate::fields::M31);
        $callback!("m31-residue", crate::fields::M31Residue);
        $callback!("goldilocks", crate::fields::Gld);
        $callback!("goldilocks-residue", crate::fields::GoldilocksResidue);
        $callback!("qm31", crate::fields::Qm31);
        $callback!("gf49", crate::fields::Gf49);
        $callback!("gf125", crate::fields::Gf125);
        $callback!("gf9", crate::fields::Gf9);
        $callback!("gf81", crate::fields::Gf81);
        $callback!("goldilocks-squared", crate::fields::GoldilocksSquared);
        $callback!("maxprime-squared", crate::fields::MaxPrimeSquared);
    };
}
pub(crate) use for_each_field;

/// Iterate the seven presentation-only binary rows at degrees one, two, and
/// four: polynomial, normal-basis, and Cantor-basis presentations with no
/// kernel route. These rows carry the full scalar, conversion, and
/// presentation surface but no `FieldKernels` implementation, so their
/// records report the `scalar` backend.
#[allow(unused_macros)]
macro_rules! for_each_field_small {
    ($callback:ident) => {
        $callback!("gf2p1-poly", crate::fields::Gf2p1Poly);
        $callback!("gf2p2-poly", crate::fields::Gf2p2Poly);
        $callback!("gf2p4-poly", crate::fields::Gf2p4Poly);
        $callback!("gf2p2-normal", crate::fields::Gf2p2Normal);
        $callback!("gf2p4-normal", crate::fields::Gf2p4Normal);
        $callback!("gf2p2-cantor", crate::fields::Gf2p2Cantor);
        $callback!("gf2p4-cantor", crate::fields::Gf2p4Cantor);
        $callback!("gf2p2-fanpaar", crate::fields::Gf2p2FanPaar);
        $callback!("gf2p4-fanpaar", crate::fields::Gf2p4FanPaar);
        $callback!("gf2p5-padded", crate::fields::Gf2p5Padded);
    };
}
pub(crate) use for_each_field_small;

/// Iterate the bulk-kernel subset: every presentation whose packed-lane
/// geometry carries bulk panels, in the same registration order.
#[allow(unused_macros)]
macro_rules! for_each_bulk_field {
    ($callback:ident) => {
        $crate::fields::for_each_field!($callback);
    };
}
pub(crate) use for_each_bulk_field;

/// Capability lines: one `FIELD_BACKEND` record per registered field, in
/// registration order, so every run states where each field's routing sits.
pub fn print_field_backends() {
    macro_rules! line {
        ($label:literal, $F:ty) => {
            println!("FIELD_BACKEND\t{}\t{}", $label, backend_name::<$F>());
        };
    }
    for_each_field!(line);
    macro_rules! scalar_line {
        ($label:literal, $F:ty) => {
            println!("FIELD_BACKEND\t{}\tscalar", $label);
        };
    }
    for_each_field_small!(scalar_line);
    // The process-level capability queries the panels report.
    let process = fgf::backend();
    println!(
        "FIELD_BACKEND\tprocess\t{}\tnative_mul={}\tblocked_rows={}",
        process.name(),
        process.has_native_mul(),
        process.has_blocked_rows()
    );
}
