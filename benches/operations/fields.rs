//! Field presentations covered by the operations benchmark.
//!
//! The labels are frozen across rounds, hosts, and arms. They name the
//! presentation and storage class, not an alias in the API: two labels with
//! the same cardinality remain distinct when their polynomial, basis, tower
//! relation, or storage differ.

use fgf::FiniteField;
use fgf::binary::{
    AES, FanPaar16, FanPaar32 as FanPaar32Spec, FanPaar64 as FanPaar64Spec, Polynomial, RS,
    Rijndael16, Rijndael32, Rijndael64,
};
use fgf::extension::{Components, Extension, ExtensionSpec, Packed, QuadMersenne31, Relation};
use fgf::prime::{Goldilocks, Mersenne31};
use fgf::{FieldBuffer, backend_for};

// --- shipped rows -----------------------------------------------------------

pub type Gf8Aes = FiniteField<2, 8, Polynomial<AES>>;
pub type Gf8Rs = FiniteField<2, 8, Polynomial<RS>>;
pub type Gf16Rijndael = FiniteField<2, 16, Extension<Rijndael16, 2>>;
pub type Gf32Rijndael = FiniteField<2, 32, Extension<Rijndael32, 2>>;
pub type Gf64Rijndael = FiniteField<2, 64, Extension<Rijndael64, 2>>;
pub type Gf16FanPaar = FiniteField<2, 16, Extension<FanPaar16, 2>>;
pub type FanPaar32 = FiniteField<2, 32, Extension<FanPaar32Spec, 2>>;
pub type FanPaar64 = FiniteField<2, 64, Extension<FanPaar64Spec, 2>>;

/// The largest `u64` prime.
pub const MAX_PRIME: u64 = 18_446_744_073_709_551_557;

pub type ResidueU8 = FiniteField<101, 1>;
/// A one-byte residue row above the `P <= 127` vector cutoff.
pub type ResidueU8High = FiniteField<251, 1>;
pub type ResidueU32 = FiniteField<1_000_000_007, 1>;
pub type ResidueU64 = FiniteField<MAX_PRIME, 1>;

// --- custom extension specs -------------------------------------------------

/// A packed quadratic tower over the Reed-Solomon byte field:
/// `t^2 + 2t + 0x80`.
pub struct PackedRs16Spec;

impl ExtensionSpec<2> for PackedRs16Spec {
    type Base = Gf8Rs;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/RS packed";
    const RELATION: Relation = Relation::from_binary_words::<Gf8Rs>(&[0x80, 2]);
}

pub type PackedRs16 = FiniteField<2, 16, Extension<PackedRs16Spec, 2>>;

/// A packed quadratic tower over the AES byte field: `t^2 + t + 0x2C`.
pub struct CustomRijndael16Spec;

impl ExtensionSpec<2> for CustomRijndael16Spec {
    type Base = Gf8Aes;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/AES custom";
    const RELATION: Relation = Relation::from_binary_words::<Gf8Aes>(&[0x2C, 0x01]);
}

pub type CustomRijndael16 = FiniteField<2, 16, Extension<CustomRijndael16Spec, 2>>;

/// GF(49) as `t^2 - 3` over the residue row at seven, one coordinate per
/// component.
pub struct ComponentsGf49Spec;

impl ExtensionSpec<2> for ComponentsGf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(7^2)/components";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

pub type ComponentsGf49 = FiniteField<7, 2, Extension<ComponentsGf49Spec, 2>>;

// --- labels -----------------------------------------------------------------

pub const GF8_AES: &str = "gf8-aes";
pub const GF8_RS: &str = "gf8-rs";
pub const GF16_RIJNDAEL: &str = "gf16-rijndael";
pub const GF32_RIJNDAEL: &str = "gf32-rijndael";
pub const GF64_RIJNDAEL: &str = "gf64-rijndael";
pub const FANPAAR16: &str = "fanpaar16";
pub const FANPAAR32: &str = "fanpaar32";
pub const FANPAAR64: &str = "fanpaar64";
pub const RESIDUE_101: &str = "residue-101";
pub const RESIDUE_251: &str = "residue-251";
pub const RESIDUE_1E9P7: &str = "residue-1000000007";
pub const RESIDUE_MAX: &str = "residue-18446744073709551557";
pub const M31: &str = "m31";
pub const GOLDILOCKS: &str = "goldilocks";
pub const QM31: &str = "qm31";
pub const PACKED_RS16: &str = "packed-rs16";
pub const COMPONENTS_GF49: &str = "components-gf49";
pub const CUSTOM_RIJNDAEL16: &str = "custom-rijndael16";
pub const GF2_BITS: &str = "gf2-bits";

// --- families ---------------------------------------------------------------

pub const FAMILY_GF: &str = "gf";
pub const FAMILY_PRIME: &str = "prime";
pub const FAMILY_M31: &str = "m31";
pub const FAMILY_GDL: &str = "gdl";
pub const FAMILY_EXT: &str = "ext";
pub const FAMILY_GF2: &str = "gf2";

/// Backend and compiler-owned element-byte geometry per registered field label.
pub fn print_backends() {
    fn row<F: FieldBuffer>(label: &str) {
        println!("FIELD_BACKEND\t{label}\t{}", backend_for::<F>().name());
        println!("FIELD_ELEMENT_BYTES\t{label}\t{}", F::BYTES);
    }
    row::<Gf8Aes>(GF8_AES);
    row::<Gf8Rs>(GF8_RS);
    row::<Gf16Rijndael>(GF16_RIJNDAEL);
    row::<Gf32Rijndael>(GF32_RIJNDAEL);
    row::<Gf64Rijndael>(GF64_RIJNDAEL);
    row::<Gf16FanPaar>(FANPAAR16);
    row::<FanPaar32>(FANPAAR32);
    row::<FanPaar64>(FANPAAR64);
    row::<ResidueU8>(RESIDUE_101);
    row::<ResidueU8High>(RESIDUE_251);
    row::<ResidueU32>(RESIDUE_1E9P7);
    row::<ResidueU64>(RESIDUE_MAX);
    row::<Mersenne31>(M31);
    row::<Goldilocks>(GOLDILOCKS);
    row::<QuadMersenne31>(QM31);
    row::<PackedRs16>(PACKED_RS16);
    row::<ComponentsGf49>(COMPONENTS_GF49);
    row::<CustomRijndael16>(CUSTOM_RIJNDAEL16);
    // Bit-packed GF(2) has no `FieldBuffer` lane form; its arithmetic is the
    // GF(2) interop row's byte XOR route.
    println!(
        "FIELD_BACKEND\t{GF2_BITS}\t{}",
        backend_for::<FiniteField<2, 1>>().name()
    );
}
