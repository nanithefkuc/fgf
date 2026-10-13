//! Named prime fields as rows of the canonical family.
//!
//! `Mersenne31` and `Goldilocks` are aliases of their fold rows, not marker
//! structs; the default `Residue` row at the same modulus is a distinct
//! presentation reached by residue conversion, never by reinterpreting a
//! buffer.

use core::any::TypeId;

use fgf::{
    Elem, Field, FieldBuffer, FiniteField, FiniteFieldRepr, Goldilocks, GoldilocksFold,
    HasGenerator, Mersenne31, MersenneFold, PrimeField, RangeError,
};

const M31: u64 = 0x7FFF_FFFF;
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

fn require_prime_field<F: PrimeField>() {}

#[test]
fn aliases_are_the_spelled_out_rows() {
    assert_eq!(
        TypeId::of::<Mersenne31>(),
        TypeId::of::<FiniteField<M31, 1, MersenneFold>>()
    );
    assert_eq!(
        TypeId::of::<Goldilocks>(),
        TypeId::of::<FiniteField<GOLDILOCKS, 1, GoldilocksFold>>()
    );
    assert_ne!(
        TypeId::of::<Mersenne31>(),
        TypeId::of::<FiniteField<M31, 1>>()
    );
    assert_ne!(
        TypeId::of::<Goldilocks>(),
        TypeId::of::<FiniteField<GOLDILOCKS, 1>>()
    );
    require_prime_field::<Mersenne31>();
    require_prime_field::<Goldilocks>();
}

#[test]
fn presentation_labels_and_field_names() {
    assert_eq!(
        <MersenneFold as FiniteFieldRepr<M31, 1>>::NAME,
        "mersenne-fold"
    );
    assert_eq!(
        <GoldilocksFold as FiniteFieldRepr<GOLDILOCKS, 1>>::NAME,
        "goldilocks-fold"
    );
    assert_eq!(Mersenne31::NAME, "GF(2^31 - 1)");
    assert_eq!(Goldilocks::NAME, "GF(2^64 - 2^32 + 1)");
    // Marker Debug carries the characteristic, degree, and presentation
    // label; its punctuation is not a contract. The degree digit is looked
    // for after the characteristic is removed, since the characteristic's
    // own digits contain a `1`.
    for (debug, p, label) in [
        (format!("{:?}", Mersenne31::default()), M31, "mersenne-fold"),
        (
            format!("{:?}", Goldilocks::default()),
            GOLDILOCKS,
            "goldilocks-fold",
        ),
    ] {
        let without_p = debug.replacen(&p.to_string(), "", 1);
        assert!(
            debug.contains(&p.to_string()) && without_p.contains('1') && debug.contains(label),
            "{debug}"
        );
    }
}

/// Residues worth transporting: identities, the generator, `p − 1`, and a
/// spread of interior values.
fn sample_residues(modulus: u64) -> [u64; 7] {
    [0, 1, 2, 7, modulus / 3, modulus - 2, modulus - 1]
}

#[test]
fn residue_transport_preserves_arithmetic() {
    type ResM31 = FiniteField<M31, 1>;
    type ResGl = FiniteField<GOLDILOCKS, 1>;

    for &a in &sample_residues(M31) {
        for &b in &sample_residues(M31) {
            let (na, nb) = (
                Elem::<Mersenne31>::from_residue(a).unwrap(),
                Elem::<Mersenne31>::from_residue(b).unwrap(),
            );
            let (ra, rb) = (
                Elem::<ResM31>::from_residue(na.to_residue().unwrap()).unwrap(),
                Elem::<ResM31>::from_residue(nb.to_residue().unwrap()).unwrap(),
            );
            for (named, residue) in [
                (na.add(nb), ra.add(rb)),
                (na.sub(nb), ra.sub(rb)),
                (na.mul(nb), ra.mul(rb)),
                (na.div(nb), ra.div(rb)),
            ] {
                assert_eq!(named.to_residue(), residue.to_residue(), "{a} {b}");
            }
        }
    }

    for &a in &sample_residues(GOLDILOCKS) {
        for &b in &sample_residues(GOLDILOCKS) {
            let (na, nb) = (
                Elem::<Goldilocks>::from_residue(a).unwrap(),
                Elem::<Goldilocks>::from_residue(b).unwrap(),
            );
            let (ra, rb) = (
                Elem::<ResGl>::from_residue(na.to_residue().unwrap()).unwrap(),
                Elem::<ResGl>::from_residue(nb.to_residue().unwrap()).unwrap(),
            );
            for (named, residue) in [
                (na.add(nb), ra.add(rb)),
                (na.sub(nb), ra.sub(rb)),
                (na.mul(nb), ra.mul(rb)),
                (na.div(nb), ra.div(rb)),
            ] {
                assert_eq!(named.to_residue(), residue.to_residue(), "{a} {b}");
            }
        }
    }
}

#[test]
fn named_and_residue_lanes_differ_in_width() {
    // Same modulus, different wire layouts: transport must go through the
    // residue, because the buffers are not interchangeable.
    assert_eq!(Mersenne31::BYTES, 4);
    assert_eq!(<FiniteField<M31, 1> as FieldBuffer>::BYTES, 8);
    assert_eq!(Goldilocks::BYTES, 8);
    assert_eq!(<FiniteField<GOLDILOCKS, 1> as FieldBuffer>::BYTES, 8);
}

#[test]
fn residue_conversions_reject_out_of_range() {
    assert_eq!(Elem::<Mersenne31>::from_residue(M31), Err(RangeError));
    assert_eq!(Elem::<Mersenne31>::from_residue(u64::MAX), Err(RangeError));
    // Values whose low 32 bits look like valid residues are still rejected:
    // the range check runs before any narrowing.
    assert_eq!(Elem::<Mersenne31>::from_residue(1 << 32), Err(RangeError));
    assert_eq!(
        Elem::<Mersenne31>::from_residue((1 << 32) + 7),
        Err(RangeError)
    );
    assert_eq!(
        Elem::<Goldilocks>::from_residue(GOLDILOCKS),
        Err(RangeError)
    );
    assert_eq!(Elem::<Goldilocks>::from_residue(u64::MAX), Err(RangeError));
    // The raw constructors still reduce aliases; only the checked residue
    // constructor rejects them.
    assert_eq!(
        Elem::<Mersenne31>::from_raw(u32::MAX).to_residue(),
        Some(u64::from(u32::MAX) % M31)
    );
    assert_eq!(
        Elem::<Goldilocks>::from_raw(u64::MAX).to_residue(),
        Some(u64::MAX - GOLDILOCKS)
    );
}

#[test]
fn residue_conversions_are_const() {
    const G: Option<u64> = match Elem::<Mersenne31>::from_residue(7) {
        Ok(value) => value.mul(value).to_residue(),
        Err(_) => None,
    };
    const H: Option<u64> = match Elem::<Goldilocks>::from_residue(GOLDILOCKS - 1) {
        Ok(value) => value.mul(value).to_residue(),
        Err(_) => None,
    };
    assert_eq!(G, Some(49));
    assert_eq!(H, Some(1));
    assert_eq!(Mersenne31::GENERATOR_RAW, 7);
    assert_eq!(Goldilocks::GENERATOR_RAW, 7);
}
