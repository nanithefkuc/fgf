//! Conversion and encoding surfaces: [`FieldBuffer`] decode/encode,
//! presentation indices, residues and their range errors, extension
//! components, primality, and the packing conversions.
//!
//! Decode/encode and index/residue conversions run one call per steady body;
//! packing runs an 8192-lane bulk body over preallocated buffers, and
//! `pack_to_vec` is timed end-to-end because its owned allocation is the
//! work under measurement. Every case verifies its own round trip before
//! timing: decode after encode returns the element, `to_index` after
//! `from_index` returns the index, and out-of-range residues and indices are
//! rejected with [`RangeError`] rather than narrowed. Each case body
//! re-derives its fixtures from the case's registered label, so the run
//! entry points stay plain function pointers with deterministic state.

use std::hint::black_box;

use fgf::extension::ExtensionField;
use fgf::prime::PrimeIdentity;
use fgf::{Elem, RangeError};
use fgf::{Field, FieldBuffer, FieldElem};

use crate::fields::{self, for_each_field, for_each_field_small};
use crate::fixture;
use crate::registry::{self, Case, Scope};
use crate::scalar::seed_for;

const LANES: usize = 8192;

const API_DECODE: &[&str] = &["fgf::FieldBuffer::decode"];
const API_ENCODE: &[&str] = &["fgf::FieldBuffer::encode"];
const API_INDEX: &[&str] = &[
    "fgf::FieldElem::from_index",
    "fgf::FieldElem::to_index",
    "fgf::RangeError",
];
const API_RESIDUE: &[&str] = &[
    "fgf::FieldElem::from_residue",
    "fgf::FieldElem::to_residue",
    "fgf::RangeError",
];
const API_COMPONENTS: &[&str] = &[
    "fgf::extension::ExtensionField::from_components",
    "fgf::extension::ExtensionField::to_components",
];
const API_PACK: &[&str] = &["fgf::ops::pack"];
const API_UNPACK: &[&str] = &["fgf::ops::unpack"];
const API_UNPACK_TO_VEC: &[&str] = &["fgf::ops::unpack_to_vec"];
const API_PACK_TO_VEC: &[&str] = &["fgf::ops::pack_to_vec"];
const API_IS_PRIME: &[&str] = &["fgf::prime::is_prime"];

fn register_decode<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/decode/{label}"),
        api: API_DECODE,
        field: label.to_string(),
        operation: "decode".to_string(),
        scope: Scope::Steady,
        bytes: F::BYTES,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elem = fixture::elements::<F>(1, &mut seed).remove(0);
            let mut lane = vec![0u8; F::BYTES];
            <F as FieldBuffer>::encode(&mut lane, elem);
            assert_eq!(
                <F as FieldBuffer>::decode(&lane),
                elem,
                "{}: decode oracle",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let value = <F as FieldBuffer>::decode(black_box(&lane));
                fixture::observe(value.to_raw());
            });
        },
    };
    registry::add(case);
}

fn register_encode<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/encode/{label}"),
        api: API_ENCODE,
        field: label.to_string(),
        operation: "encode".to_string(),
        scope: Scope::Steady,
        bytes: F::BYTES,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elem = fixture::elements::<F>(1, &mut seed).remove(0);
            let mut expected = vec![0u8; F::BYTES];
            <F as FieldBuffer>::encode(&mut expected, elem);
            let backend = fields::scalar_backend_name();
            let mut lane = vec![0u8; F::BYTES];
            <F as FieldBuffer>::encode(&mut lane, elem);
            assert_eq!(lane, expected, "{}: encode oracle", case.case);
            registry::time(case, ctx, &backend, || {
                <F as FieldBuffer>::encode(&mut lane, black_box(elem));
                fixture::observe(&lane);
            });
        },
    };
    registry::add(case);
}

fn register_index<F: Field>(label: &'static str) {
    let case = Case {
        case: format!("convert/index/{label}"),
        api: API_INDEX,
        field: label.to_string(),
        operation: "index round trip".to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elem = fixture::elements::<F>(1, &mut seed).remove(0);
            let index = elem.to_index();
            let valid = index % F::ORDER.max(1);
            assert_eq!(
                <Elem<F> as FieldElem>::from_index(valid)
                    .expect("valid index")
                    .to_index(),
                valid,
                "{}: index round trip",
                case.case
            );
            assert!(
                matches!(
                    <Elem<F> as FieldElem>::from_index(F::ORDER),
                    Err(RangeError { .. })
                ),
                "{}: out-of-range index",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let decoded =
                    <Elem<F> as FieldElem>::from_index(black_box(valid)).expect("valid index");
                fixture::observe(decoded.to_index());
            });
        },
    };
    registry::add(case);
}

fn register_residue<F: Field>(label: &'static str) {
    let case = Case {
        case: format!("convert/residue/{label}"),
        api: API_RESIDUE,
        field: label.to_string(),
        operation: "residue conversion".to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
            let mut seed = seed_for(case.field.as_str());
            let elem = fixture::elements::<F>(1, &mut seed).remove(0);
            let residue = elem.to_residue().unwrap_or(0);
            assert_eq!(
                <Elem<F> as FieldElem>::from_residue(residue)
                    .expect("valid residue")
                    .to_residue(),
                Some(residue),
                "{}: residue round trip",
                case.case
            );
            assert!(
                matches!(
                    <Elem<F> as FieldElem>::from_residue(characteristic),
                    Err(RangeError { .. })
                ),
                "{}: out-of-range residue",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let converted = <Elem<F> as FieldElem>::from_residue(black_box(residue))
                    .expect("valid residue");
                fixture::observe(converted.to_residue());
            });
        },
    };
    registry::add(case);
}

/// Component conversions for one extension presentation at relative degree
/// `D`.
fn register_components<F: ExtensionField<D>, const D: usize>(label: &'static str) {
    let case = Case {
        case: format!("convert/components/{label}"),
        api: API_COMPONENTS,
        field: label.to_string(),
        operation: "component conversion".to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: D,
        sources: D,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let components: [Elem<F::Base>; D] = fixture::elements::<F::Base>(D, &mut seed)
                .try_into()
                .expect("degree components");
            let rebuilt = <F as ExtensionField<D>>::from_components(components);
            assert_eq!(
                <F as ExtensionField<D>>::to_components(rebuilt),
                components,
                "{}: component round trip",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let value = <F as ExtensionField<D>>::from_components(black_box(components));
                fixture::observe(<F as ExtensionField<D>>::to_components(value));
            });
        },
    };
    registry::add(case);
}

fn register_pack<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/pack/{label}"),
        api: API_PACK,
        field: label.to_string(),
        operation: "pack".to_string(),
        scope: Scope::Steady,
        bytes: LANES * F::BYTES,
        lanes: LANES,
        sources: LANES,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elems = fixture::elements::<F>(LANES, &mut seed);
            let mut packed = vec![0u8; LANES * F::BYTES];
            fgf::ops::pack::<F>(&mut packed, &elems);
            let mut round_trip = elems.clone();
            fgf::ops::unpack::<F>(&mut round_trip, &packed);
            assert_eq!(round_trip, elems, "{}: pack oracle", case.case);
            let backend = fields::scalar_backend_name();
            let mut steady = vec![0u8; LANES * F::BYTES];
            registry::time(case, ctx, &backend, || {
                // `pack` overwrites every byte, so the preallocated buffer
                // needs no refill between calls.
                fgf::ops::pack::<F>(&mut steady, black_box(&elems));
                fixture::observe(steady.len());
            });
        },
    };
    registry::add(case);
}

fn register_unpack<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/unpack/{label}"),
        api: API_UNPACK,
        field: label.to_string(),
        operation: "unpack".to_string(),
        scope: Scope::Steady,
        bytes: LANES * F::BYTES,
        lanes: LANES,
        sources: 1,
        destinations: LANES,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elems = fixture::elements::<F>(LANES, &mut seed);
            let packed = fgf::ops::pack_to_vec::<F>(&elems);
            let mut round_trip = elems.clone();
            fgf::ops::unpack::<F>(&mut round_trip, &packed);
            assert_eq!(round_trip, elems, "{}: unpack oracle", case.case);
            let backend = fields::scalar_backend_name();
            let mut steady = elems.clone();
            registry::time(case, ctx, &backend, || {
                // `unpack` overwrites every element, so the preallocated
                // destination needs no refill between calls.
                fgf::ops::unpack::<F>(&mut steady, black_box(&packed));
                fixture::observe(steady.len());
            });
        },
    };
    registry::add(case);
}

fn register_unpack_to_vec<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/unpack-to-vec/{label}"),
        api: API_UNPACK_TO_VEC,
        field: label.to_string(),
        operation: "unpack_to_vec".to_string(),
        scope: Scope::EndToEnd,
        bytes: LANES * F::BYTES,
        lanes: LANES,
        sources: 1,
        destinations: LANES,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elems = fixture::elements::<F>(LANES, &mut seed);
            let packed = fgf::ops::pack_to_vec::<F>(&elems);
            assert_eq!(
                fgf::ops::unpack_to_vec::<F>(&packed),
                elems,
                "{}: unpack_to_vec oracle",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let round_trip = fgf::ops::unpack_to_vec::<F>(black_box(&packed));
                fixture::observe(round_trip.len());
            });
        },
    };
    registry::add(case);
}

fn register_pack_to_vec<F: FieldBuffer>(label: &'static str) {
    let case = Case {
        case: format!("convert/pack-to-vec/{label}"),
        api: API_PACK_TO_VEC,
        field: label.to_string(),
        operation: "pack_to_vec".to_string(),
        scope: Scope::EndToEnd,
        bytes: LANES * F::BYTES,
        lanes: LANES,
        sources: LANES,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let elems = fixture::elements::<F>(LANES, &mut seed);
            let expected = fgf::ops::pack_to_vec::<F>(&elems);
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let packed = fgf::ops::pack_to_vec::<F>(black_box(&elems));
                fixture::observe(packed.len());
            });
            drop(expected);
        },
    };
    registry::add(case);
}

/// Primality latency for a full-width prime, semiprime, and divisibility shortcut.
fn register_is_prime() {
    for route in ["large-prime", "semiprime", "even-composite"] {
        registry::add(Case {
            case: format!("convert/is-prime/{route}"),
            api: API_IS_PRIME,
            field: "u64".to_string(),
            operation: route.to_string(),
            scope: Scope::Steady,
            bytes: 0,
            lanes: 1,
            sources: 1,
            destinations: 1,
            control: false,
            phase: None,
            run: |case, ctx| {
                use fgf::prime::is_prime;
                let (candidate, expected) = match case.operation.as_str() {
                    "large-prime" => (18_446_744_073_709_551_557, true),
                    "semiprime" => (4_294_967_291_u64 * 4_294_967_279, false),
                    "even-composite" => (u64::MAX - 1, false),
                    _ => unreachable!("registered primality route"),
                };
                assert_eq!(
                    is_prime(candidate),
                    expected,
                    "{}: primality oracle",
                    case.case
                );
                registry::time(case, ctx, "scalar", || {
                    fixture::observe(is_prime(black_box(candidate)));
                });
            },
        });
    }
}

/// Register the conversion panel over every registered field, plus the
/// field-independent primality cases.
pub fn register() {
    macro_rules! walk {
        ($label:literal, $F:ty) => {
            register_decode::<$F>($label);
            register_encode::<$F>($label);
            register_index::<$F>($label);
            register_residue::<$F>($label);
            register_pack::<$F>($label);
            register_unpack::<$F>($label);
            register_unpack_to_vec::<$F>($label);
            register_pack_to_vec::<$F>($label);
        };
    }
    for_each_field!(walk);
    macro_rules! scalar_walk {
        ($label:literal, $F:ty) => {
            register_index::<$F>($label);
            register_residue::<$F>($label);
        };
    }
    for_each_field_small!(scalar_walk);

    register_is_prime();

    macro_rules! components_walk {
        ($label:literal, $F:ty, $d:literal) => {
            register_components::<$F, $d>($label);
        };
    }
    components_walk!("gf2p2-fanpaar", crate::fields::Gf2p2FanPaar, 2);
    components_walk!("gf2p4-fanpaar", crate::fields::Gf2p4FanPaar, 2);
    components_walk!("gf2p8-fanpaar", crate::fields::Gf2p8FanPaar, 2);
    components_walk!("gf2p16-fanpaar", crate::fields::Gf2p16FanPaar, 2);
    components_walk!("gf2p32-fanpaar", crate::fields::Gf2p32FanPaar, 2);
    components_walk!("gf2p64-fanpaar", crate::fields::Gf2p64FanPaar, 2);
    components_walk!("gf2p16-rijndael", crate::fields::Gf2p16Rijndael, 2);
    components_walk!("gf2p32-rijndael", crate::fields::Gf2p32Rijndael, 2);
    components_walk!("gf2p64-rijndael", crate::fields::Gf2p64Rijndael, 2);
    components_walk!("gf2p16-custom-rs", crate::fields::Gf2p16CustomRs, 2);
    components_walk!(
        "gf2p16-custom-rijndael",
        crate::fields::Gf2p16CustomRijndael,
        2
    );
    components_walk!("gf2p16-custom-normal", crate::fields::Gf2p16CustomNormal, 2);
    components_walk!("gf2p16-custom-cantor", crate::fields::Gf2p16CustomCantor, 2);
    components_walk!("gf2p5-padded", crate::fields::Gf2p5Padded, 5);
    components_walk!("gf49", crate::fields::Gf49, 2);
    components_walk!("gf125", crate::fields::Gf125, 3);
    components_walk!("gf9", crate::fields::Gf9, 2);
    components_walk!("gf81", crate::fields::Gf81, 2);
    components_walk!("goldilocks-squared", crate::fields::GoldilocksSquared, 2);
    components_walk!("maxprime-squared", crate::fields::MaxPrimeSquared, 2);
    components_walk!("qm31", crate::fields::Qm31, 2);
}
