//! Shipped degree-one fields and the prime-characteristic identity.
//!
//! [`Residue`] is the ordinary-residue presentation of the canonical family
//! at degree one: [`FiniteField<P, 1>`](crate::FiniteField) is GF(P) for
//! every checked `u64` prime, with no named modulus. The named folds
//! ([`Mersenne31`], [`Goldilocks`]) are standalone degree-one rows over
//! their fold presentations [`MersenneFold`] and [`GoldilocksFold`].
//! [`PrimeField`] is the sealed degree-one capability marker, and
//! [`is_prime`] is the crate's deterministic Miller–Rabin check.
//!
//! Every degree-one row exposes its characteristic through
//! [`PrimeIdentity`]: one identity type per prime, independent of
//! coordinates or a named modulus. A prime identity is available at any
//! checked `u64` prime, and a composite modulus fails the same way.
//!
//! # Totality and canonicalization
//!
//! Every raw lane is a legal input: [`Elem::from_raw`](crate::FieldElem::from_raw)
//! reduces it into the canonical range `0..p` on the way in, and every
//! arithmetic output is canonical without panicking on out-of-range
//! operands. This is the prime-field form of the crate-wide `inv(0) == 0` and
//! `x / 0 == 0` convention: division is total in every build profile and
//! under `const` evaluation, which is not a claim that zero is invertible.
//! Storage is always canonical, so equality, hashing, and ordering compare
//! raw words directly; inspect the stored bits with
//! [`Elem::to_raw`](crate::Elem::to_raw). Arithmetic is variable-time and not
//! intended for secret data.
//!
//! Mersenne31 folds the high bit back with `2^31 ≡ 1 (mod p)` and one
//! conditional subtract. Goldilocks reduces a 128-bit product with
//! `2^64 ≡ 2^32 − 1 (mod p)` and `2^96 ≡ −1`.
//!
//! # Examples
//!
//! A prime identity reads back at any prime and fails at a composite:
//!
//! ```
//! use fgf::prime::{PrimeCharacteristic, PrimeIdentity};
//!
//! // Seven is prime, so its identity reads back.
//! const P: u64 = <PrimeCharacteristic<7> as PrimeIdentity>::CHARACTERISTIC;
//! const _: () = assert!(P == 7);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::prime::{PrimeCharacteristic, PrimeIdentity};
//!
//! // Ninety-one is composite, so reading the identity fails.
//! const Q: u64 = <PrimeCharacteristic<91> as PrimeIdentity>::CHARACTERISTIC;
//! ```
//!
//! A shipped prime field constructs its identities generically:
//!
//! ```
//! use fgf::{Elem, prime::Mersenne31};
//!
//! const ONE: Elem<Mersenne31> = Elem::<Mersenne31>::ONE;
//! assert_eq!(ONE.to_raw(), 1);
//! ```
//!
//! `Mersenne31` known answers, pinned against the modular reduction:
//!
//! ```
//! use fgf::{Elem, prime::Mersenne31};
//!
//! const X: Elem<Mersenne31> = Elem::<Mersenne31>::from_raw(0x5555_5555);
//! const _: () = assert!(X.mul(X).to_raw() == 0x71C7_1C71);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem<Mersenne31> = Elem::<Mersenne31>::from_raw(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x4000_0000);
//! const _: () = assert!(Elem::<Mersenne31>::from_raw(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::<Mersenne31>::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1.
//! assert_eq!(
//!     Elem::<Mersenne31>::GENERATOR.pow(0x7FFF_FFFE),
//!     Elem::<Mersenne31>::ONE
//! );
//! ```
//!
//! `Goldilocks` known answers:
//!
//! ```
//! use fgf::{Elem, prime::Goldilocks};
//!
//! const X: Elem<Goldilocks> = Elem::<Goldilocks>::from_raw(0xFFFF_FFFF_0000_0000);
//! const _: () = assert!(X.mul(X).to_raw() == 0x0000_0000_0000_0001);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem<Goldilocks> = Elem::<Goldilocks>::from_raw(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x7FFF_FFFF_8000_0001);
//! const _: () = assert!(Elem::<Goldilocks>::from_raw(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::<Goldilocks>::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1 = 2^64 − 2^32.
//! assert_eq!(
//!     Elem::<Goldilocks>::GENERATOR.pow(0xFFFF_FFFF_0000_0000),
//!     Elem::<Goldilocks>::ONE
//! );
//! ```

pub use crate::field::finite_field::Residue;
pub use crate::field::goldilocks::{Goldilocks, GoldilocksFold};
pub use crate::field::mersenne31::{Mersenne31, MersenneFold};
pub use crate::field::prime::PrimeField;
pub use crate::field::{PrimeCharacteristic, PrimeIdentity, is_prime};
