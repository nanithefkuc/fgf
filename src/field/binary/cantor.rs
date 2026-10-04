//! Cantor-basis binary presentations at degrees 2, 4, and 8.
//!
//! [`Cantor`] builds its basis from `SEED` through the relation
//! `c_(N-1) = SEED` and `c_i = c_(i+1)^2 + c_(i+1)`; bit `i` selects `c_i`.
//! Validity explicitly asserts that the chain reaches `c_0 == ONE` and
//! checks basis rank by bounded enumeration, never by a matrix solver.
//!
//! Smallest valid test instances use the same seeds as the normal basis:
//! `SEED = 2` over `P = 0x7`, `SEED = 8` over `P = 0x13`, and
//! `SEED = 0x20` over [`AES`](super::poly::AES) and
//! [`REED_SOLOMON`](super::poly::REED_SOLOMON).
//!
//! ```compile_fail,E0080
//! // SEED = 0 builds the zero chain, which never reaches one.
//! let _ = fgf::Elem::<fgf::Gf<8, fgf::Cantor<0x11B, 0>>>::from_raw(1);
//! ```

use super::description::{BinaryDescription, ByteLogExp};
use super::normal::inverse_columns;
use super::poly::Poly;
use super::{BinaryRepr, private};

/// A Cantor-basis presentation over `Poly<P>` seeded by `SEED`.
///
/// `SEED` is a raw element of `Poly<P>`; the basis chain runs
/// `c_(N-1) = SEED` down through `c_i = c_(i+1)^2 + c_(i+1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Cantor<const P: u128, const SEED: u8>;

impl<const P: u128, const SEED: u8> private::Sealed for Cantor<P, SEED> {}

/// Cantor chain in polynomial coordinates: `cols[i]` is `c_i`.
pub(crate) const fn to_base<const P: u128, const SEED: u8, const N: u8>() -> [u8; 8] {
    let desc = BinaryDescription::polynomial(N, P);
    let mut chain = [0u8; 8];
    chain[(N - 1) as usize] = SEED;
    let mut i = (N - 1) as usize;
    while i > 0 {
        let top = chain[i] as u64;
        let sq = desc.mul(top, top);
        #[allow(clippy::cast_possible_truncation)]
        let byte = (sq ^ top) as u8;
        chain[i - 1] = byte;
        i -= 1;
    }
    chain
}

macro_rules! cantor_row {
    ($n:literal) => {
        impl<const P: u128, const SEED: u8> BinaryRepr<$n> for Cantor<P, SEED> {
            const NAME: &'static str = "cantor";
            const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::append_basis(
                &BinaryDescription::polynomial($n, P),
                to_base::<P, SEED, $n>(),
                inverse_columns(to_base::<P, SEED, $n>(), $n),
            );
            const LOG_EXP: Option<&'static ByteLogExp> = None;
            const VALID: () = {
                assert!((SEED as u64) < (1u64 << $n), "cantor seed has excess bits");
                assert!(
                    to_base::<P, SEED, $n>()[0] == 1,
                    "cantor chain does not reach one"
                );
                <Self as BinaryRepr<$n>>::DESCRIPTION.validate();
            };
        }
    };
}

cantor_row!(2);
cantor_row!(4);

/// Transport the polynomial generator of `P` into Cantor coordinates.
pub(crate) const fn transported_generator<const P: u128, const SEED: u8, const N: u8>() -> u64 {
    let poly_desc = BinaryDescription::polynomial(N, P);
    let poly_gen = Poly::<P>::smallest_generator(&poly_desc);
    let from = inverse_columns(to_base::<P, SEED, N>(), N);
    let mut image = 0u64;
    let mut i = 0;
    while i < 8 {
        if (poly_gen >> i) & 1 == 1 {
            image ^= from[i] as u64;
        }
        i += 1;
    }
    image
}

use super::super::HasGenerator;
use super::Gf;

impl<const P: u128, const SEED: u8> HasGenerator for Gf<2, Cantor<P, SEED>> {
    const GENERATOR_RAW: u8 = {
        let g = transported_generator::<P, SEED, 2>();
        assert!(
            <Cantor<P, SEED> as BinaryRepr<2>>::DESCRIPTION.is_generator(g),
            "basis generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128, const SEED: u8> HasGenerator for Gf<4, Cantor<P, SEED>> {
    const GENERATOR_RAW: u8 = {
        let g = transported_generator::<P, SEED, 4>();
        assert!(
            <Cantor<P, SEED> as BinaryRepr<4>>::DESCRIPTION.is_generator(g),
            "basis generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128, const SEED: u8> HasGenerator for Gf<8, Cantor<P, SEED>> {
    const GENERATOR_RAW: u8 = {
        let g = transported_generator::<P, SEED, 8>();
        assert!(
            <Cantor<P, SEED> as BinaryRepr<8>>::DESCRIPTION.is_generator(g),
            "basis generator does not have full order"
        );
        #[allow(clippy::cast_possible_truncation)]
        let generator = g as u8;
        generator
    };
}

impl<const P: u128, const SEED: u8> BinaryRepr<8> for Cantor<P, SEED> {
    const NAME: &'static str = "cantor";
    const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::append_basis(
        &BinaryDescription::polynomial(8, P),
        to_base::<P, SEED, 8>(),
        inverse_columns(to_base::<P, SEED, 8>(), 8),
    );
    const LOG_EXP: Option<&'static ByteLogExp> = Some(&ByteLogExp::build(
        &BinaryDescription::append_basis(
            &BinaryDescription::polynomial(8, P),
            to_base::<P, SEED, 8>(),
            inverse_columns(to_base::<P, SEED, 8>(), 8),
        ),
        {
            let poly_desc = BinaryDescription::polynomial(8, P);
            let poly_gen = Poly::<P>::smallest_generator(&poly_desc);
            let mut image = 0u64;
            let mut i = 0;
            let from = inverse_columns(to_base::<P, SEED, 8>(), 8);
            while i < 8 {
                if (poly_gen >> i) & 1 == 1 {
                    image ^= from[i] as u64;
                }
                i += 1;
            }
            image
        },
    ));
    const VALID: () = {
        assert!(
            to_base::<P, SEED, 8>()[0] == 1,
            "cantor chain does not reach one"
        );
        <Self as BinaryRepr<8>>::DESCRIPTION.validate();
        assert!(
            <Self as BinaryRepr<8>>::DESCRIPTION.is_generator({
                let poly_desc = BinaryDescription::polynomial(8, P);
                let poly_gen = Poly::<P>::smallest_generator(&poly_desc);
                let mut image = 0u64;
                let mut i = 0;
                let from = inverse_columns(to_base::<P, SEED, 8>(), 8);
                while i < 8 {
                    if (poly_gen >> i) & 1 == 1 {
                        image ^= from[i] as u64;
                    }
                    i += 1;
                }
                image
            }),
            "basis generator does not have full order"
        );
    };
}
