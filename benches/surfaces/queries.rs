//! Backend and field capability queries: the process backend, the per-field
//! routing resolution, and the vector-elementwise capability pair.
//!
//! These are the trivial timings the measurement engine must not slow down:
//! every body is one monomorphized call with black-boxed results, timed at
//! query scope with zero logical bytes.

use std::hint::black_box;

use fgf::{Backend, FieldKernels};

use crate::fixture;
use crate::registry::{self, Case, CaseCtx, Scope};
use crate::scalar::seed_for;

const API_PROCESS: &[&str] = &["fgf::backend", "fgf::Backend::name"];
const API_PARSE: &[&str] = &["core::str::FromStr for fgf::Backend"];
const API_FOR_FIELD: &[&str] = &["fgf::backend_for", "fgf::FieldKernels::backend"];
const API_VECTOR_ELEMENTWISE: &[&str] = &[
    "fgf::FieldKernels::has_vector_elementwise",
    "fgf::FieldKernels::vector_elementwise_min_bytes",
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

/// The process-level backend resolution and its name.
fn run_process(case: &Case, ctx: &CaseCtx) {
    let process = fgf::backend();
    let backend = process.name().to_string();
    assert!(!backend.is_empty(), "{}: backend names itself", case.case);
    registry::time(case, ctx, &backend, || {
        let process: Backend = fgf::backend();
        let name = process.name();
        fixture::observe(black_box(name));
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
    let backend = resolved.name().to_string();
    registry::time(case, ctx, &backend, || {
        fixture::observe(fgf::backend_for::<F>().name());
    });
}

/// The vector-elementwise capability pair for one field.
fn run_vector_elementwise<F: FieldKernels>(case: &Case, ctx: &CaseCtx) {
    let resolved = fgf::backend_for::<F>();
    let has = <F as FieldKernels>::has_vector_elementwise();
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
            <F as FieldKernels>::has_vector_elementwise(),
            <F as FieldKernels>::vector_elementwise_min_bytes(),
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
}
