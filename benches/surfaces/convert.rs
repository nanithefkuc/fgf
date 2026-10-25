//! Conversion and encoding surfaces: [`FieldBuffer`] decode/encode,
//! presentation indices, residues and their range errors, extension
//! components, the integer reduction functions, and the packing conversions.
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
//!
//! Timed packing calls pass their complete output slices through the
//! optimization barrier. Allocating variants keep allocation and destruction
//! inside the timed body.

use std::hint::black_box;

use fgf::field::{Field, FieldBuffer, FieldElem, PrimeIdentity};
use fgf::{Elem, ExtensionField, RangeError};

use crate::fields::{self, for_each_field, for_each_field_small};
use crate::fixture;
use crate::registry::{self, Case, Scope};
use crate::scalar::seed_for;

const LANES: usize = 8192;

const API_DECODE: &[&str] = &["fgf::field::FieldBuffer::decode"];
const API_ENCODE: &[&str] = &["fgf::field::FieldBuffer::encode"];
const API_INDEX: &[&str] = &[
    "fgf::field::FieldElem::from_index",
    "fgf::field::FieldElem::to_index",
    "fgf::RangeError",
];
const API_RESIDUE: &[&str] = &[
    "fgf::field::FieldElem::from_residue",
    "fgf::field::FieldElem::to_residue",
    "fgf::RangeError",
];
const API_COMPONENTS: &[&str] = &[
    "fgf::ExtensionField::from_components",
    "fgf::ExtensionField::to_components",
];
const API_PACK: &[&str] = &["fgf::ops::pack"];
const API_UNPACK: &[&str] = &["fgf::ops::unpack"];
const API_UNPACK_TO_VEC: &[&str] = &["fgf::ops::unpack_to_vec"];
const API_PACK_TO_VEC: &[&str] = &["fgf::ops::pack_to_vec"];
const API_REDUCE: &[&str] = &[
    "fgf::field::prime::reduce",
    "fgf::field::prime::reduce_wide",
];
const API_FOLD_REDUCE_M31: &[&str] = &["fgf::mersenne31::reduce"];
const API_FOLD_REDUCE_GLD: &[&str] = &["fgf::goldilocks::reduce", "fgf::goldilocks::reduce_wide"];
const API_IS_PRIME: &[&str] = &["fgf::field::is_prime"];

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
            assert_eq!(
                <Elem<F> as FieldElem>::from_index(F::ORDER),
                Err(RangeError),
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
            assert_eq!(
                <Elem<F> as FieldElem>::from_residue(characteristic),
                Err(RangeError),
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
                fixture::observe(steady.as_slice());
            });
            fixture::assert_bytes_equal(&case.case, &steady, &packed);
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
                fixture::observe(steady.as_slice());
            });
            assert_eq!(steady, elems, "{}: timed unpack oracle", case.case);
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
                fixture::observe(round_trip.as_slice());
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
            assert_eq!(
                fgf::ops::unpack_to_vec::<F>(&expected),
                elems,
                "{}: pack_to_vec oracle",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let packed = fgf::ops::pack_to_vec::<F>(black_box(&elems));
                fixture::observe(packed.as_slice());
            });
        },
    };
    registry::add(case);
}

/// The generic `prime::reduce`/`reduce_wide` pair at one modulus.
fn register_reduce<const P: u64>(label: &'static str) {
    let case = Case {
        case: format!("convert/reduce/{label}"),
        api: API_REDUCE,
        field: label.to_string(),
        operation: "integer reduction".to_string(),
        scope: Scope::Steady,
        bytes: 0,
        lanes: 1,
        sources: 1,
        destinations: 1,
        control: false,
        phase: None,
        run: |case, ctx| {
            let mut seed = seed_for(case.field.as_str());
            let value = fixture::lcg(&mut seed);
            let wide = u128::from(value) << 64 | u128::from(fixture::lcg(&mut seed));
            assert_eq!(
                fgf::field::prime::reduce::<P>(value),
                value % P,
                "{}: reduce oracle",
                case.case
            );
            assert_eq!(
                u128::from(fgf::field::prime::reduce_wide::<P>(wide)),
                wide % u128::from(P),
                "{}: reduce_wide oracle",
                case.case
            );
            let backend = fields::scalar_backend_name();
            registry::time(case, ctx, &backend, || {
                let reduced = fgf::field::prime::reduce::<P>(black_box(value));
                let widened = fgf::field::prime::reduce_wide::<P>(black_box(wide));
                fixture::observe((reduced, widened));
            });
        },
    };
    registry::add(case);
}

/// The specialized fold reductions: `mersenne31::reduce` and the Goldilocks
/// pair, each validated against the generic reduction.
fn register_fold_reduce(label: &'static str) {
    if label == "m31" {
        let case = Case {
            case: "convert/fold-reduce/m31".to_string(),
            api: API_FOLD_REDUCE_M31,
            field: "m31".to_string(),
            operation: "specialized reduction".to_string(),
            scope: Scope::Steady,
            bytes: 0,
            lanes: 1,
            sources: 1,
            destinations: 1,
            control: false,
            phase: None,
            run: |case, ctx| {
                let mut seed = seed_for(case.field.as_str());
                let value = (fixture::lcg(&mut seed) & u64::from(u32::MAX)) as u32;
                assert_eq!(
                    u64::from(fgf::mersenne31::reduce(value)),
                    fgf::field::prime::reduce::<0x7FFF_FFFF>(u64::from(value)),
                    "{}: m31 reduce oracle",
                    case.case
                );
                let backend = fields::scalar_backend_name();
                registry::time(case, ctx, &backend, || {
                    fixture::observe(fgf::mersenne31::reduce(black_box(value)));
                });
            },
        };
        registry::add(case);
    } else {
        let case = Case {
            case: "convert/fold-reduce/goldilocks".to_string(),
            api: API_FOLD_REDUCE_GLD,
            field: "goldilocks".to_string(),
            operation: "specialized reduction".to_string(),
            scope: Scope::Steady,
            bytes: 0,
            lanes: 1,
            sources: 1,
            destinations: 1,
            control: false,
            phase: None,
            run: |case, ctx| {
                let mut seed = seed_for(case.field.as_str());
                let value = fixture::lcg(&mut seed);
                let wide = u128::from(value) << 64 | u128::from(fixture::lcg(&mut seed));
                assert_eq!(
                    fgf::goldilocks::reduce(value),
                    fgf::field::prime::reduce::<0xFFFF_FFFF_0000_0001>(value),
                    "{}: goldilocks reduce oracle",
                    case.case
                );
                assert_eq!(
                    u128::from(fgf::goldilocks::reduce_wide(wide)),
                    wide % u128::from(0xFFFF_FFFF_0000_0001u64),
                    "{}: goldilocks reduce_wide oracle",
                    case.case
                );
                let backend = fields::scalar_backend_name();
                registry::time(case, ctx, &backend, || {
                    let reduced = fgf::goldilocks::reduce(black_box(value));
                    let widened = fgf::goldilocks::reduce_wide(black_box(wide));
                    fixture::observe((reduced, widened));
                });
            },
        };
        registry::add(case);
    }
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
                use fgf::field::is_prime;
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
/// field-independent reduction, fold-reduction, and primality cases.
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

    macro_rules! residue_walk {
        ($label:literal, $p:literal) => {
            register_reduce::<$p>($label);
        };
    }
    residue_walk!("gf2p1-residue", 2);
    residue_walk!("gf101", 101);
    residue_walk!("gf127", 127);
    residue_walk!("gf131", 131);
    residue_walk!("gf65521", 65_521);
    residue_walk!("gf65537", 65_537);
    residue_walk!("gf1e9p7", 1_000_000_007);
    residue_walk!("gfu32max", 4_294_967_291);
    residue_walk!("gfmaxp", 18_446_744_073_709_551_557);
    residue_walk!("m31-residue", 0x7FFF_FFFF);
    residue_walk!("goldilocks-residue", 0xFFFF_FFFF_0000_0001);

    register_fold_reduce("m31");
    register_fold_reduce("goldilocks");
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
