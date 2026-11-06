//! General field extensions: one presentation over every canonical row.
//!
//! [`FiniteField<P, N, Extension<S, D>>`](crate::FiniteField) names
//! GF(P^N) as a relative degree-`D` extension of the base row `S::Base`.
//! [`ExtensionSpec`] is open and data-only: a base row, a storage
//! selection, a name, and the relation. The defining relation is
//! validated by associated constants that every value-producing operation
//! evaluates — the field metadata first
//! (prime characteristic, binary degree cap, cardinality bound), then
//! the base's validity, the characteristic agreement, the degree
//! identity, the relation's length and coordinate range, storage
//! admissibility, the descriptor capacity, and finally the
//! irreducibility of the relation over its base — so an invalid row is a
//! compile error in any build that generates code using it. Merely naming
//! the type evaluates nothing.
//!
//! # Layout
//!
//! [`Components`] keeps one base element per component; over a byte-backed
//! base the stable byte form concatenates the base encodings.
//! [`Packed<W>`](Packed) and [`PackedPadded<W>`](PackedPadded) keep the
//! row's coordinate word in one of the sealed [`PackedWord`] integers,
//! component `i` at the bits `[i*B, (i+1)*B)` of the base degree `B`, and
//! require a binary base. `Packed<W>` fills its word exactly and encodes
//! little-endian; `PackedPadded<W>` keeps the coordinate word below its
//! word width with every padding bit zero and has no byte form. Every
//! row's raw word, bytes, and coordinate word are canonical: constructors
//! canonicalize components through the base and clear padding bits.
//!
//! [`QuadMersenne31`] is the extension row
//! `FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>`: pairs
//! `[re, im]` of [`Mersenne31`](crate::prime::Mersenne31) limbs with
//! `i² = −1`, under the relation `t^2 + 1` that [`ComplexMersenne31`] names.
//! Since `p = 2³¹ − 1 ≡ 3 (mod 4)`, `−1` is a quadratic non-residue, so
//! `Fₚ[i]/(i²+1)` is a field. The raw word is `[u32; 2]`, real limb first,
//! and the stable byte form is the interleaved little-endian pair
//! `[re: u32, im: u32]`, 8 bytes per element.
//!
//! Every row shares one generic body over the base's arithmetic: component
//! addition, subtraction, and negation; schoolbook products reduced by
//! `t^D = -(c_{D-1} t^{D-1} + ... + c_0)`; inversion by the conjugate-norm
//! formula at relative degree two and by extended Euclid above it. Zero
//! maps to zero under inversion and division, and `0^0 = 1`. Elements are
//! [`Elem`](crate::Elem) values with the crate-wide operators;
//! [`ExtensionField::from_components`] and
//! [`ExtensionField::to_components`] convert against the base. Generic and
//! custom rows carry const identities and `to_raw`, and no const element
//! arithmetic. Rows with a byte form implement
//! [`FieldBuffer`](crate::FieldBuffer) and the packed operation surface.
//!
//! The shipped instances — the Rijndael and Fan-Paar towers in
//! [`binary`](crate::binary) and [`QuadMersenne31`] — are rows of
//! this family with a full `const` element surface and a checked
//! generator. A row whose descriptor equals a shipped instance's — the same
//! base presentations, relations, and storages at every level — produces
//! the same bytes as that instance in every operation.
//!
//! # Preconditions
//!
//! Validation is exact, and a relation whose validation exceeds rustc's
//! `long_running_const_eval` limit fails to compile as a capacity limit
//! rather than a reducibility verdict; fgf does not `allow` that lint on
//! its own items, so an over-budget relation stays a compile error. Deep
//! towers also cost compiler time within the limit.
//!
//! # Totality and canonicalization
//!
//! For [`QuadMersenne31`], every raw limb bit pattern is a legal input:
//! [`Elem::from_raw`](crate::FieldElem::from_raw) reduces each limb into
//! canonical storage on the way in, and every arithmetic output is canonical
//! (`< p` per limb). Storage is always canonical, so equality, hashing, and
//! ordering compare raw words directly. Arithmetic is variable-time and not
//! intended for secret data.
//!
//! # Examples
//!
//! GF(49) as `t^2 - 3` over GF(7):
//!
//! ```
//! use fgf::{
//!     Elem, FieldElem, FiniteField, extension::{Components, Extension, ExtensionField, ExtensionSpec, Relation},
//! };
//!
//! struct Gf49Spec;
//! impl ExtensionSpec<2> for Gf49Spec {
//!     type Base = FiniteField<7, 1>;
//!     type Storage = Components;
//!     const NAME: &'static str = "GF(49)";
//!     // `t^2 - 3`: constant coefficient `4 = -3` first, then the linear
//!     // coefficient.
//!     const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
//! }
//!
//! type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;
//!
//! // The identities are const, over the base identities.
//! const ONE: Elem<Gf49> = Elem::<Gf49>::ONE;
//!
//! // `t^2 = 3` in components: `(0, 1)^2 = (3, 0)`.
//! let t = <Gf49 as ExtensionField<2>>::from_components([
//!     Elem::<FiniteField<7, 1>>::ZERO,
//!     Elem::<FiniteField<7, 1>>::ONE,
//! ]);
//! let square = t * t;
//! let components = <Gf49 as ExtensionField<2>>::to_components(square);
//! assert_eq!(components[0].to_raw(), 3);
//! assert_eq!(components[1].to_raw(), 0);
//!
//! // Every nonzero element inverts, and the defining relation holds.
//! let inverse = (t + ONE).inv();
//! assert_eq!((t + ONE) * inverse, ONE);
//! ```
//!
//! `QuadMersenne31` known answers, with `i² = −1`:
//!
//! ```
//! use fgf::{Elem, extension::QuadMersenne31};
//!
//! // i² = −1  →  (0 + i)² = (−1 + 0·i)
//! const I: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw([0, 1]);
//! const _: () = assert!(I.square().to_raw()[0] == 0x7FFF_FFFE);
//! const _: () = assert!(I.square().to_raw()[1] == 0);
//!
//! // Known product, pinned against the frozen M31 known answer
//! // a² = 0x71C7_1C71: (a + a·i)² = 0 + 2a²·i with 2a² mod p = 0x638E_38E3.
//! const A: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw([0x5555_5555, 0x5555_5555]);
//! const _: () = assert!(A.mul(A).to_raw()[0] == 0);
//! const _: () = assert!(A.mul(A).to_raw()[1] == 0x638E_38E3);
//!
//! // Division is total.
//! const _: () = assert!(A.div(Elem::<QuadMersenne31>::ZERO).is_zero());
//!
//! // The generator has full order p² − 1.
//! assert_eq!(
//!     Elem::<QuadMersenne31>::GENERATOR.pow(0x3FFF_FFFF_0000_0000),
//!     Elem::<QuadMersenne31>::ONE
//! );
//! ```

pub use crate::field::extension::{
    Components, Extension, ExtensionField, ExtensionSpec, ExtensionStorage, Packed, PackedPadded,
    PackedWord, Relation,
};
pub use crate::field::quad_mersenne31::{ComplexMersenne31, QuadMersenne31};
