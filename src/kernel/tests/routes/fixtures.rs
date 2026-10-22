//! Extension specifications named by the generated ledger cases.
//!
//! Every specification mirrors the ledger rule that cites it: the base row,
//! the storage, and the defining relation. Shipped presentations need no
//! entry here; these are the custom rows a case names by type.

use crate::field::{
    Components, ExtensionSpec, FiniteField, Goldilocks, Mersenne31, Packed, Polynomial, Relation,
    Rijndael16, Rijndael32,
};

type Residue3 = FiniteField<3, 1, crate::field::Residue>;
type Residue7 = FiniteField<7, 1, crate::field::Residue>;
type Aes8 = FiniteField<2, 8, Polynomial<0x11B>>;
type Normal8 = FiniteField<2, 8, crate::field::Normal<0x11B, 0x20>>;

/// GF(2^8) as `t^2 + t + 8` over GF(2^4), packed in a byte word.
pub(super) struct Custom8Spec;
impl ExtensionSpec<2> for Custom8Spec {
    type Base = FiniteField<2, 4, Polynomial<0x13>>;
    type Storage = Packed<u8>;
    const NAME: &'static str = "custom GF(2^8) extension";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 4, Polynomial<0x13>>>(&[0x8, 0x1]);
}

/// GF(2^16) as `t^2 + 2t + 0x80` over the Reed-Solomon byte field.
pub(super) struct Custom16Spec;
impl ExtensionSpec<2> for Custom16Spec {
    type Base = FiniteField<2, 8, Polynomial<0x11D>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "custom GF(2^16) extension";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<0x11D>>>(&[0x80, 0x2]);
}

/// GF(2^32) as `t^2 + t + 0x2001` over the Rijndael GF(2^16) row.
pub(super) struct Custom32Spec;
impl ExtensionSpec<2> for Custom32Spec {
    type Base = FiniteField<2, 16, crate::field::Extension<Rijndael16, 2>>;
    type Storage = Packed<u32>;
    const NAME: &'static str = "custom GF(2^32) extension";
    const RELATION: Relation = Relation::from_binary_words::<
        FiniteField<2, 16, crate::field::Extension<Rijndael16, 2>>,
    >(&[0x2001, 0x0001]);
}

/// GF(2^64) as `t^2 + t + 0x2ada7f36` over the Rijndael GF(2^32) row.
pub(super) struct Custom64Spec;
impl ExtensionSpec<2> for Custom64Spec {
    type Base = FiniteField<2, 32, crate::field::Extension<Rijndael32, 2>>;
    type Storage = Packed<u64>;
    const NAME: &'static str = "custom GF(2^64) extension";
    const RELATION: Relation = Relation::from_binary_words::<
        FiniteField<2, 32, crate::field::Extension<Rijndael32, 2>>,
    >(&[0x2ada_7f36, 0x0000_0001]);
}

/// GF(9) as `t^2 + 2t + 2` over GF(3).
pub(super) struct G9Spec;
impl ExtensionSpec<2> for G9Spec {
    type Base = Residue3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as t^2 + 2t + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 2]);
}

/// GF(9) as `a^2 + 1` over GF(3), a distinct quadratic presentation.
pub(super) struct G9AltSpec;
impl ExtensionSpec<2> for G9AltSpec {
    type Base = Residue3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as a^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}

/// GF(81) as `x^4 + 2x^2 + 2` over GF(3).
pub(super) struct G81QuarticSpec;
impl ExtensionSpec<4> for G81QuarticSpec {
    type Base = Residue3;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as x^4 + 2x^3 + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
}

/// GF(3^4) as `x^4 + 2x^2 + 2` over GF(3), the lower tower step.
pub(super) struct G3e4Spec;
impl ExtensionSpec<4> for G3e4Spec {
    type Base = Residue3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^4) as x^4 + 2x^2 + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
}

/// GF(3^8) as `t^2 - x` over GF(3^4).
pub(super) struct G3e8Spec;
impl ExtensionSpec<2> for G3e8Spec {
    type Base = FiniteField<3, 4, crate::field::Extension<G3e4Spec, 4>>;
    type Storage = Components;
    const NAME: &'static str = "GF(3^8) as t^2 - x";
    const RELATION: Relation = Relation::from_coordinates(&[0, 1, 0, 0, 0, 0, 0, 0]);
}

/// GF(7^3) as `x^3 - 2` over GF(7).
pub(super) struct G343Spec;
impl ExtensionSpec<3> for G343Spec {
    type Base = Residue7;
    type Storage = Components;
    const NAME: &'static str = "GF(343) as x^3 - 2";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}

/// GF(131^2) as `t^2 + 5` over GF(131).
pub(super) struct Gf131QuadraticSpec;
impl ExtensionSpec<2> for Gf131QuadraticSpec {
    type Base = FiniteField<131, 1, crate::field::Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(131^2) as t^2 + 5";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0]);
}

/// GF(257^2) as `t^2 + 3` over GF(257).
pub(super) struct Gf257QuadraticSpec;
impl ExtensionSpec<2> for Gf257QuadraticSpec {
    type Base = FiniteField<257, 1, crate::field::Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(257^2) as t^2 + 3";
    const RELATION: Relation = Relation::from_coordinates(&[3, 0]);
}

/// GF(65537^2) as `t^2 + 3` over GF(65537).
pub(super) struct Gf65537QuadraticSpec;
impl ExtensionSpec<2> for Gf65537QuadraticSpec {
    type Base = FiniteField<65537, 1, crate::field::Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(65537^2) as t^2 + 3";
    const RELATION: Relation = Relation::from_coordinates(&[3, 0]);
}

/// A quadratic over Mersenne31 as `t^2 + 2`, distinct from `QuadMersenne31`.
pub(super) struct MersenneQuadraticSpec;
impl ExtensionSpec<2> for MersenneQuadraticSpec {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "custom quadratic M31";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0]);
}

/// A quadratic over Goldilocks as `t^2 + 7`.
pub(super) struct GoldilocksSquaredSpec;
impl ExtensionSpec<2> for GoldilocksSquaredSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(goldilocks^2)";
    const RELATION: Relation = Relation::from_coordinates(&[7, 0]);
}

/// GF(2^16) as `t^2 + 0x53*t + 1` over the AES byte field.
pub(super) struct OverAesSpec;
impl ExtensionSpec<2> for OverAesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) as t^2 + 0x53*t + 1";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[1, 0x53]);
}

/// GF(2^16) as `t^2 + 0x35*t + 1` over the normal-basis byte field.
pub(super) struct NormalQuadraticSpec;
impl ExtensionSpec<2> for NormalQuadraticSpec {
    type Base = Normal8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) as t^2 + 0x35*t + 1";
    const RELATION: Relation = Relation::from_binary_words::<Normal8>(&[1, 0x35]);
}

/// GF(2^64) as a degree-eight relation over the AES byte field.
pub(super) struct Gf64OverAesSpec;
impl ExtensionSpec<8> for Gf64OverAesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^64) as a degree-eight relation over AES";
    const RELATION: Relation =
        Relation::from_binary_words::<Aes8>(&[107, 48, 249, 14, 199, 221, 1, 228]);
}

/// GF(p^2) as `t^2 + 3` over the widest residue row.
pub(super) struct U64QuadraticSpec;
impl ExtensionSpec<2> for U64QuadraticSpec {
    type Base = FiniteField<18446744073709551557, 1, crate::field::Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(p^2) as t^2 + 3 over residue-u64";
    const RELATION: Relation = Relation::from_coordinates(&[3, 0]);
}

/// GF(2^16) as `t^2 + t + 0x80` over the Cantor byte field.
pub(super) struct CantorQuadraticSpec;
impl ExtensionSpec<2> for CantorQuadraticSpec {
    type Base = FiniteField<2, 8, crate::field::Cantor<0x11B, 0x20>>;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) as t^2 + t + 0x80 over Cantor";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, crate::field::Cantor<0x11B, 0x20>>>(
            &[0x80, 0x1],
        );
}

/// GF(2^24) as `t^3 + x` over the AES byte field.
pub(super) struct D3AesSpec;
impl ExtensionSpec<3> for D3AesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^24) as t^3 + 2";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[2, 0, 0]);
}
