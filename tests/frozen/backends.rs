//! Frozen backend-context records: rows whose architecture, feature
//! configuration, requested tier, and resolved tier match the running
//! process are compared against the actual per-field backend reports; every
//! other context is reported as unavailable rather than passed. Each record
//! also names the capturing host's best tier — the resolution with no
//! override in that build — and a no-override record is compared only when
//! that value equals this process's backend, which pins it to this host.
//! Override runs that resolve to the same tier produce the same per-field
//! reports on every captured host, so both host records compare.

use crate::common as frozen;

#[allow(unused_imports)]
use frozen::{
    AesLin16F, Cantor2, Cantor4, Cantor8Aes, Cantor8Rs, Custom8F, Custom16F, Custom32F, Custom64F,
    ExtraAConst16F, ExtraALin16F, ExtraCantor16F, ExtraTwin16AF, ExtraTwin16BF, FanPaar2F,
    FanPaar4F, FanPaar8F, FanPaar16F, FanPaar32F, FanPaar64F, NormBase16F, Normal2, Normal4,
    Normal8Aes, Normal8Rs, OverNormal16F, Poly1, Poly2, Poly4, Poly8Aes, Poly8Rs, Rijndael16F,
    Rijndael32F, Rijndael64F, Same16F,
};

use frozen::Records;

const BACKENDS: &str = include_str!("fixtures/backends.txt");

/// One backend-context group from the fixture.
struct ContextRecords {
    arch: String,
    features: String,
    requested: String,
    host_best: String,
    process: String,
    rows: Vec<(String, String, u8)>,
}

/// Feed the kernel-reporting rows to `$mac`.
macro_rules! kernel_rows {
    ($mac:ident) => {
        $mac!(
            (poly8aes, frozen::Poly8Aes),
            (poly8rs, frozen::Poly8Rs),
            (normal8aes, frozen::Normal8Aes),
            (normal8rs, frozen::Normal8Rs),
            (cantor8aes, frozen::Cantor8Aes),
            (cantor8rs, frozen::Cantor8Rs),
            (custom8, frozen::Custom8F),
            (fanpaar8, frozen::FanPaar8F),
            (rijndael16, frozen::Rijndael16F),
            (rijndael32, frozen::Rijndael32F),
            (rijndael64, frozen::Rijndael64F),
            (fanpaar16, frozen::FanPaar16F),
            (fanpaar32, frozen::FanPaar32F),
            (fanpaar64, frozen::FanPaar64F),
            (custom16, frozen::Custom16F),
            (aeslin16, frozen::AesLin16F),
            (custom32, frozen::Custom32F),
            (custom64, frozen::Custom64F),
            (normbase16, frozen::NormBase16F),
            (overnormal16, frozen::OverNormal16F),
            (same16, frozen::Same16F),
            (extraalin16, frozen::ExtraALin16F),
            (extraaconst16, frozen::ExtraAConst16F),
            (extracantor16, frozen::ExtraCantor16F),
            (extratwin16a, frozen::ExtraTwin16AF),
            (extratwin16b, frozen::ExtraTwin16BF),
            (mersenne31, fgf::Mersenne31),
            (goldilocks, fgf::Goldilocks),
            (quadmersenne31, fgf::QuadMersenne31),
        );
    };
}

/// Parse every context group in file order.
fn parse_contexts() -> Vec<ContextRecords> {
    let mut contexts: Vec<ContextRecords> = Vec::new();
    for (line, number) in Records::new("backends.txt", BACKENDS) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "context" => {
                assert_eq!(fields.len(), 6, "backends.txt:{number}: context shape");
                contexts.push(ContextRecords {
                    arch: fields[1].to_owned(),
                    features: fields[2].to_owned(),
                    requested: fields[3].to_owned(),
                    host_best: fields[4].to_owned(),
                    process: fields[5].to_owned(),
                    rows: Vec::new(),
                });
            }
            "row" => {
                assert_eq!(fields.len(), 4, "backends.txt:{number}: row shape");
                let contained: u8 = fields[3]
                    .parse()
                    .unwrap_or_else(|_| panic!("backends.txt:{number}: bad vector flag"));
                contexts
                    .last_mut()
                    .expect("row inside a context")
                    .rows
                    .push((fields[1].to_owned(), fields[2].to_owned(), contained));
            }
            other => panic!("backends.txt:{number}: unknown record kind {other}"),
        }
    }
    contexts
}

/// The feature configuration of this test binary, as the capture spelled it.
fn enabled_features() -> String {
    let mut names: Vec<&'static str> = [
        ("alloc", cfg!(feature = "alloc")),
        ("internals", cfg!(feature = "internals")),
        ("simd", cfg!(feature = "simd")),
        ("simd512", cfg!(feature = "simd512")),
        ("std", cfg!(feature = "std")),
    ]
    .into_iter()
    .filter(|(_, enabled)| *enabled)
    .map(|(name, _)| name)
    .collect();
    names.sort_unstable();
    let joined = names.join(",");
    if joined.is_empty() {
        "-".to_owned()
    } else {
        joined
    }
}

/// The requested tier of this process.
fn requested_tier() -> String {
    std::env::var("SIMD_BACKEND").unwrap_or_else(|_| "none".to_owned())
}

/// Every matching context reproduces its rows; every other context is
/// reported unavailable.
#[test]
fn backend_contexts_match_reports() {
    let contexts = parse_contexts();
    assert!(!contexts.is_empty(), "backends.txt: no context records");

    let arch = std::env::consts::ARCH;
    let features = enabled_features();
    let requested = requested_tier();
    let process = fgf::backend().name();
    let mut compared = 0;
    for context in &contexts {
        let matches = context.arch == arch
            && context.features == features
            && context.requested == requested
            && context.process == process
            && (requested != "none" || context.host_best == process);
        if !matches {
            println!(
                "backends.txt: context unavailable in this process: arch {} features {} requested {} host best {} process {}",
                context.arch,
                context.features,
                context.requested,
                context.host_best,
                context.process
            );
            continue;
        }
        compared += 1;
        assert_eq!(
            context.process, process,
            "backends.txt: process backend for requested {requested}"
        );
        macro_rules! check_rows {
            ($(($short:ident, $typ:ty)),* $(,)?) => {{
                assert_eq!(
                    context.rows.len(),
                    0 $(+ { let _ = stringify!($short); 1 })*,
                    "backends.txt: row count for requested {requested}"
                );
                for (index, (row, backend, vector)) in context.rows.iter().enumerate() {
                    let expected = [
                        $((stringify!($short), fgf::backend_for::<$typ>(), fgf::has_vector_elementwise::<$typ>())),*
                    ];
                    let (name, backend_expected, vector_expected) = expected[index];
                    assert_eq!(row, name, "backends.txt: row order for requested {requested}");
                    assert_eq!(
                        backend,
                        backend_expected.name(),
                        "backends.txt: backend of {name} for requested {requested}"
                    );
                    assert_eq!(
                        *vector,
                        u8::from(vector_expected),
                        "backends.txt: vector elementwise of {name} for requested {requested}"
                    );
                }
            }};
        }
        kernel_rows!(check_rows);
    }
    // A process configuration no capture covered (another architecture,
    // feature set, or requested tier) has nothing to compare: it is reported
    // unavailable rather than failed.
    if compared == 0 {
        println!("backends.txt: no captured context for this process; backend records unavailable");
    } else {
        println!("backends.txt: compared {compared} context(s) for this process");
    }
}
