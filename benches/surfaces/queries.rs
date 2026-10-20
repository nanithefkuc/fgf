//! Backend and field capability queries: the process backend ladder, the
//! per-field routing resolution, the vector-elementwise capability pair, and
//! the binary description accessors.
//!
//! These are the trivial timings the measurement engine must not slow down:
//! every body is one monomorphized call with black-boxed results, timed at
//! query scope with zero logical bytes.

use std::hint::black_box;

use fgf::{Backend, FieldKernels, KernelBackend};

use crate::fields;
use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

const API_PROCESS: &[&str] = &[
    "fgf::backend",
    "fgf::KernelBackend::has_native_mul",
    "fgf::KernelBackend::has_blocked_rows",
    "fgf::Backend::name",
];
const API_TIERS: &[&str] = &["fgf::FGF_TIERS"];
const API_PARSE: &[&str] = &[
    "fgf::ParseBackendError",
    "core::str::FromStr for fgf::Backend",
];
const API_FOR_FIELD: &[&str] = &["fgf::backend_for", "fgf::FieldKernels::backend"];
const API_VECTOR_ELEMENTWISE: &[&str] = &[
    "fgf::has_vector_elementwise",
    "fgf::kernel::FieldKernels::vector_elementwise_min_bytes",
];
const API_DESCRIPTION: &[&str] = &[
    "fgf::BinaryField::DESCRIPTION",
    "fgf::BinaryDescription::degree",
    "fgf::BinaryDescription::one",
    "fgf::BinaryDescription::mul",
    "fgf::BinaryDescription::inv",
    "fgf::BinaryDescription::trace",
    "fgf::BinaryDescription::same_structure",
    "fgf::BinaryDescription::is_generator",
];

fn query_case(
    case: String,
    field: String,
    operation: &'static str,
    api: &'static [&'static str],
    run: fn(&Case, &CaseCtx),
) {
    registry::add(Case {
        case,
        api,
        field,
        operation: operation.to_string(),
        scope: Scope::Query,
        bytes: 0,
        lanes: 0,
        sources: 0,
        destinations: 0,
        control: false,
        phase: None,
        run,
    });
}

/// The process-level backend ladder and its capability bits.
fn run_process(case: &Case, ctx: &CaseCtx) {
    let process = fgf::backend();
    assert!(
        fgf::FGF_TIERS.contains(&process),
        "{}: process backend is a declared tier",
        case.case
    );
    let backend = process.name().to_string();
    registry::time(case, ctx, &backend, || {
        let process: Backend = fgf::backend();
        let name = process.name();
        fixture::observe((
            black_box(name),
            process.has_native_mul(),
            process.has_blocked_rows(),
        ));
    });
}

/// The declared tier ladder itself.
fn run_tiers(case: &Case, ctx: &CaseCtx) {
    let tiers = fgf::FGF_TIERS;
    assert!(!tiers.is_empty(), "{}: tier ladder is nonempty", case.case);
    assert_eq!(
        *tiers.last().expect("nonempty"),
        Backend::Scalar,
        "{}: scalar terminates the ladder",
        case.case
    );
    registry::time(case, ctx, "scalar", || {
        let mut observed = 0usize;
        for tier in fgf::FGF_TIERS {
            observed += usize::from(!tier.name().is_empty());
        }
        fixture::observe(observed);
    });
}

/// Backend name parsing, including the failure value.
fn run_parse(case: &Case, ctx: &CaseCtx) {
    let ok: Backend = "v3".parse().expect("v3 is a tier");
    assert_eq!(ok.name(), "v3");
    assert!("not-a-tier".parse::<Backend>().is_err(), "{}", case.case);
    let mut seed = seed_for(&case.case);
    let name = match fixture::lcg(&mut seed) % 3 {
        0 => "v2",
        1 => "v3",
        _ => "scalar",
    };
    registry::time(case, ctx, "scalar", || {
        let _ = fixture::observe(name.parse::<Backend>().map(|backend| backend.name()));
    });
}

/// Per-field routing resolution over every bulk field.
fn run_for_field<F: FieldKernels>(case: &Case, ctx: &CaseCtx) {
    let resolved = fgf::backend_for::<F>();
    assert!(
        fgf::FGF_TIERS.contains(&resolved),
        "{}: field backend is a declared tier",
        case.case
    );
    let backend = resolved.name().to_string();
    registry::time(case, ctx, &backend, || {
        fixture::observe(fgf::backend_for::<F>().name());
    });
}

/// The vector-elementwise capability pair for one field.
fn run_vector_elementwise<F: FieldKernels>(case: &Case, ctx: &CaseCtx) {
    let resolved = fgf::backend_for::<F>();
    let has = fgf::has_vector_elementwise::<F>();
    if has {
        assert!(
            resolved != Backend::Scalar,
            "{}: vector elementwise implies a vector-capable tier",
            case.case
        );
    }
    let backend = resolved.name().to_string();
    registry::time(case, ctx, &backend, || {
        fixture::observe((
            fgf::has_vector_elementwise::<F>(),
            fgf::kernel::vector_elementwise_min_bytes::<F>(),
        ));
    });
}

/// The binary description accessors for one degree-eight presentation.
fn run_description<F: FieldKernels + fgf::BinaryField>(case: &Case, ctx: &CaseCtx) {
    let description = <F as fgf::BinaryField>::DESCRIPTION;
    assert_eq!(
        description.degree(),
        8,
        "{}: degree-eight description",
        case.case
    );
    assert_ne!(
        description.one(),
        0,
        "{}: description one is nonzero",
        case.case
    );
    assert!(
        description.same_structure(description),
        "{}: structure self-agreement",
        case.case
    );
    let a = 0x53u64;
    let b = 0xCAu64;
    assert_eq!(
        description.mul(description.inv(a), a),
        description.one(),
        "{}: description inversion oracle",
        case.case
    );
    let backend = fields::scalar_backend_name();
    registry::time(case, ctx, &backend, || {
        fixture::observe((
            description.degree(),
            description.mul(black_box(a), black_box(b)),
            description.inv(black_box(a)),
            description.trace(black_box(a)),
            description.is_generator(black_box(a)),
        ));
    });
}

/// Register the capability-query panel.
pub fn register() {
    query_case(
        "query/backend/process".to_string(),
        "process".to_string(),
        "process backend",
        API_PROCESS,
        run_process,
    );
    query_case(
        "query/backend/tiers".to_string(),
        "process".to_string(),
        "tier ladder",
        API_TIERS,
        run_tiers,
    );
    query_case(
        "query/backend/parse".to_string(),
        "process".to_string(),
        "backend parsing",
        API_PARSE,
        run_parse,
    );

    macro_rules! walk {
        ($label:literal, $F:ty) => {
            query_case(
                format!("query/backend/for-field/{}", $label),
                $label.to_string(),
                "field backend",
                API_FOR_FIELD,
                run_for_field::<$F>,
            );
            query_case(
                format!("query/backend/vector-elementwise/{}", $label),
                $label.to_string(),
                "vector elementwise capability",
                API_VECTOR_ELEMENTWISE,
                run_vector_elementwise::<$F>,
            );
        };
    }
    crate::fields::for_each_bulk_field!(walk);

    macro_rules! descriptions {
        ($label:literal, $F:ty) => {
            query_case(
                format!("query/description/{}", $label),
                $label.to_string(),
                "binary description accessors",
                API_DESCRIPTION,
                run_description::<$F>,
            );
        };
    }
    descriptions!("gf2p8-aes", crate::fields::Gf2p8Aes);
    descriptions!("gf2p8-rs", crate::fields::Gf2p8Rs);
    descriptions!("gf2p8-normal", crate::fields::Gf2p8Normal);
    descriptions!("gf2p8-cantor", crate::fields::Gf2p8Cantor);
}
