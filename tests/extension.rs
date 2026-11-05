//! Generic extension rows and their independent references in one binary.
//!
//! The split is by role: `rows` holds each field type's own laws,
//! encodings, and packed surface; `reference` holds arithmetic against an
//! independent in-test schoolbook and brute-force reference. The GF(49)
//! and GF(343) rows and the deterministic sample generator are shared:
//! both halves drive them.

#[path = "extension/reference.rs"]
mod reference;
#[path = "extension/rows.rs"]
mod rows;

use fgf::{Components, Extension, ExtensionSpec, FiniteField, Relation};

/// GF(49) as `t^2 - 3` over the residue row at seven.
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    // `t^2 - 3`: `4 = -3` constant first.
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;
type Base7 = FiniteField<7, 1>;

/// GF(343) as `t^3 - 2` over the residue row at seven: the odd cubic
/// Euclid inversion path.
struct Gf343Spec;
impl ExtensionSpec<3> for Gf343Spec {
    type Base = Base7;
    type Storage = Components;
    const NAME: &'static str = "GF(343)";
    // `t^3 - 2`: `5 = -2` constant first.
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}

type Gf343 = FiniteField<7, 3, Extension<Gf343Spec, 3>>;

/// A deterministic sample generator for larger fields.
struct Sample {
    state: u64,
}

impl Sample {
    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}
