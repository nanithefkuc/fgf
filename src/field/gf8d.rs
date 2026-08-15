//! GF(2^8) under the polynomial `0x11D`.
//!
//! The field is `GF(2)[x] / (x^8 + x^4 + x^3 + x^2 + 1)`. It is the same
//! construction as [`gf8b`](crate::field::gf8b), differing only in the
//! reduction polynomial (`0x11D` rather than the AES `0x11B`) and its
//! multiplicative generator (`0x02` rather than `0x03`; under `0x11D` the
//! element `x` is primitive and `3` is not — the mirror of the AES field).
//!
//! This is the field used by Intel ISA-L, `klauspost/reedsolomon`, and the
//! classical Reed–Solomon tables. It exists so codecs above `fgf` can produce
//! and consume shards that are byte-identical to those ecosystems. The bytes
//! are a wire convention: the polynomial, the generator `0x02`, and the
//! encoding are frozen.
//!
//! `0x11D` has no native hardware multiply — `GF2P8MULB` (GFNI) implements
//! only the AES field — so its accelerated backends use the polynomial-agnostic
//! nibble-shuffle kernels (and, where measured, a `VGF2P8AFFINEQB` affine map).
//! Scalar arithmetic here is `const`, exactly as in [`gf8b`](crate::field::gf8b),
//! so `0x11D` coding matrices are equally `const` items.
//!
//! ```
//! use fgf::gf8d::Elem;
//!
//! // The generator is 2, and it has full multiplicative order.
//! assert_eq!(Elem::from_raw(fgf::gf8d::GENERATOR.to_raw()), Elem(0x02));
//! assert_eq!(Elem(0x02).pow(255), Elem::ONE);
//!
//! // Known-answer products under 0x11D. The 0x11B (AES) field gives 0x01,
//! // 0xc1, and 0x13 for the same inputs — the fields are genuinely distinct.
//! assert_eq!(Elem(0x53).mul(Elem(0xca)), Elem(0x8f));
//! assert_eq!(Elem(0x57).mul(Elem(0x83)), Elem(0x31));
//! assert_eq!(Elem(0xff).mul(Elem(0xff)), Elem(0xe2));
//!
//! // Inverse and division round-trip; division is total in `const` context.
//! assert_eq!(Elem(0x53).mul(Elem(0x53).inv()), Elem::ONE);
//! const _: () = assert!(Elem(0x57).div(Elem::ZERO).to_raw() == 0);
//! ```

crate::field::flat8::flat_gf8!(Gf8D, 0x11D, 0x1D, 0x02, "GF(2^8)/0x11D");
