# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.93'
API_NIGHTLY := 'nightly-2026-09-30'

# Stable public-surface gate. The inventory tools are external development
# tools pinned here and in the CI installer; both read the rustdoc JSON format
# `API_NIGHTLY` emits. `API_PATHS_MODE` is `enforce`: every stable item has
# exactly one path. `API_MAJOR` is the major version whose published releases
# form the semver baseline.
API_PUBLIC_API := '0.52.0'
API_SEMVER_CHECKS := '0.51.0'
API_FORMAT := '61'
API_TARGETS := 'x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu wasm32-unknown-unknown'
API_PATHS_MODE := 'enforce'
API_MAJOR := '4'

# Every Cargo feature except `default` and `internals`; `_api-generate` fails
# when it disagrees with the manifest, so a new feature cannot drop out of the
# `all` inventory cell and the `stable-max` semver set.
API_STABLE_FEATURES := 'alloc,std,simd,simd512'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v4x v3_gfni_crypto v3 v2 scalar'

# Owned-unsafe coverage is provided by the promoted GFNI recipe below.
MIRI := ''

# Coverage exclusions, classified by the capability a file's lines need.
# Hosted runners cannot rely on GFNI or AVX-512, so exactly those file lists,
# plus the two binary-field dispatch modules whose V4x and GFNI arms
# dominate their bodies, are excluded from the hosted gate; every other
# kernel file, including the portable and narrower-ISA branches, stays
# included. Each list is explicit per file: a new GFNI, AVX-512, or dispatch
# exclusion must be added here with a T4 justification in the crate's
# `AGENTS.md`, never silently. The acceptance-host recipe `cover-native`
# excludes nothing and reports each list separately.
# GFNI-only bodies: GF2P8MULB lanes and their coefficient-map helpers.
COV_IGNORE_GFNI := 'kernel/x86/gf8/gfni\.rs$|kernel/x86/gf8/gfni/elementwise\.rs$|kernel/x86/gf8/gfni/gather\.rs$|kernel/x86/gf8/gfni/matrix\.rs$|kernel/x86/gf8/gfni/rows\.rs$|kernel/x86/gf8/gfni/scatter\.rs$|kernel/x86/gf8/gfni/single\.rs$|kernel/x86/gf16/gfni\.rs$|kernel/x86/gf16/gfni/elementwise\.rs$|kernel/x86/gf16/gfni/gather\.rs$|kernel/x86/gf16/gfni/matrix\.rs$|kernel/x86/gf16/gfni/scatter\.rs$|kernel/x86/gf16/gfni/single\.rs$|kernel/x86/gf32\.rs$|kernel/x86/gf64\.rs$'
# AVX-512 bodies: 64-byte lanes, masked and streaming stores, V4x kernels.
COV_IGNORE_AVX512 := 'kernel/x86/bytes/avx512\.rs$|kernel/x86/gf8/avx512\.rs$|kernel/x86/gf8/avx512/elementwise\.rs$|kernel/x86/gf8/avx512/gather\.rs$|kernel/x86/gf8/avx512/matrix\.rs$|kernel/x86/gf8/avx512/scatter\.rs$|kernel/x86/gf8/avx512/single\.rs$|kernel/x86/gf16/avx512\.rs$|kernel/x86/goldilocks/avx512\.rs$|kernel/x86/mersenne31/avx512\.rs$|kernel/x86/quad_mersenne31/avx512\.rs$|kernel/x86/residue/avx512\.rs$|kernel/x86/residue/avx512/matrix\.rs$|kernel/x86/residue/avx512/single\.rs$|kernel/x86/residue/avx512/single/u8\.rs$|kernel/x86/residue/avx512/single/u32\.rs$|kernel/x86/residue/avx512/single/u64\.rs$'
# Dispatch modules whose generic arms are selected per field type and tier.
COV_IGNORE_DISPATCH := 'kernel/gf8\.rs$|kernel/tower\.rs$'
COV_IGNORE := '"' + COV_IGNORE_GFNI + '|' + COV_IGNORE_AVX512 + '|' + COV_IGNORE_DISPATCH + '"'

# Export the existing merged profiles, including after a failed coverage gate.
# CI uploads the export; run `just cover` first.
[private]
[group('cover')]
cover-lcov:
    cargo llvm-cov report --lcov --output-path coverage.lcov --ignore-filename-regex {{COV_IGNORE}}

# Native coverage on an acceptance host that executes every declared ISA
# (AVX-512 and GFNI included): the same per-tier walk as the hosted `cover`,
# a report of every file, and a separate extraction of the GFNI, AVX-512,
# and dispatch lists, so a PR touching those files reports figures for
# exactly what it changed. The figures are informational: `#[arcane]` kernel
# bodies carry almost no coverage mapping, so the ISA lists measure their
# wrapper lines, and the dispatch totals score each generic function by its
# best monomorphized instance. The 95% gate applies to the hosted scope. The
# extraction fails when a list matches no file or two lists match one file.
[group('cover')]
cover-native:
    #!/usr/bin/env bash
    set -euo pipefail
    export FGF_REQUIRE_TIER=v4x
    just _cover-runs
    report=target/coverage-native.lcov
    cargo llvm-cov report --lcov --output-path "$report"
    cargo llvm-cov report --summary-only
    python3 - "$report" gfni '{{COV_IGNORE_GFNI}}' avx512 '{{COV_IGNORE_AVX512}}' dispatch '{{COV_IGNORE_DISPATCH}}' <<'PY'
    import re
    import sys

    report = sys.argv[1]
    groups = {name: re.compile(pattern) for name, pattern in zip(sys.argv[2::2], sys.argv[3::2])}
    lines = {name: [0, 0] for name in groups}
    matched = {name: 0 for name in groups}
    current = None
    for raw in open(report, encoding="utf-8"):
        if raw.startswith("SF:"):
            path = raw[3:].strip()
            hits = [name for name, pattern in groups.items() if pattern.search(path)]
            if len(hits) > 1:
                sys.exit(f"cover-native: {path} matches lists {hits}; keep the lists disjoint")
            current = hits[0] if hits else None
            if current:
                matched[current] += 1
        elif current and raw.startswith("LF:"):
            lines[current][0] += int(raw[3:])
        elif current and raw.startswith("LH:"):
            lines[current][1] += int(raw[3:])
    for name, (found, hit) in lines.items():
        if matched[name] == 0:
            sys.exit(f"cover-native: the {name} file list matched no report file")
        pct = 100.0 * hit / found if found else 100.0
        print(f"cover-native: {name} files {matched[name]}: lines {hit}/{found} ({pct:.2f}%)")
    PY
    cargo llvm-cov report --ignore-filename-regex {{COV_IGNORE}} --fail-under-lines {{COV_MIN}}


# Run one educational example, passing additional Cargo options through.
[group('examples')]
example NAME *ARGS:
    cargo run --example {{NAME}} {{ARGS}}

# Every example runs under the default features and under `alloc` without
# `std`; the allocation-free subset also runs with every library feature
# disabled. Timing probes are separate diagnostics.
# Run every educational example in each feature configuration.
[group('examples')]
examples:
    #!/usr/bin/env bash
    set -euo pipefail
    all=(field_arithmetic shard_recovery streaming_encode binary_syndrome extension_fields embeddings prime_vectors generic_prime custom_extensions basis_maps packed_no_alloc)
    portable=(field_arithmetic binary_syndrome extension_fields embeddings custom_extensions basis_maps packed_no_alloc)
    for name in "${all[@]}"; do
        cargo run --example "${name}"
    done
    for name in "${all[@]}"; do
        cargo run --example "${name}" --no-default-features --features alloc
    done
    for name in "${portable[@]}"; do
        cargo run --example "${name}" --no-default-features
    done

# The shared `features` recipe walks `--no-default-features`, default, and
# `--all-features`. `alloc` sits between the first two: it turns on the
# prepared coefficient collections (`CoeffVec`, `CoeffMatrix`) and
# `pack_to_vec` without `std`, so neither end of the shared walk builds it.
# CI runs this alongside `just features`.
[group('test')]
features-alloc:
    cargo test --no-default-features --features alloc
    cargo clippy --no-default-features --features alloc --all-targets -- -D warnings

# Exposure must compile without activating SIMD or its capability-token
# dependency, and every SIMD feature must compile without exposure.
[group('test')]
features-internals:
    cargo check --all-targets --no-default-features
    cargo check --all-targets --no-default-features --features internals
    cargo check --all-targets --no-default-features --features alloc,internals
    cargo check --all-targets --no-default-features --features std,internals
    cargo check --all-targets
    cargo check --all-targets --features simd512
    cargo check --all-targets --all-features

# The gather validation regression executes as i686 release code with
# overflow checks disabled, under no-default and default plus internals.
# Default-feature i686 uses the portable backend; direct x64-kernel cases run
# in the native test matrix. The listing guard rejects an empty target.
# The i686 run needs `gcc-multilib` for the 32-bit link environment.
[group('test')]
test-gather-validation-32:
    #!/usr/bin/env bash
    set -euo pipefail
    rustup target add i686-unknown-linux-gnu
    export RUSTFLAGS="${RUSTFLAGS:-} -C overflow-checks=off"
    export FGF_REQUIRE_WRAPPING=1
    cargo check --release --target i686-unknown-linux-gnu --all-features --all-targets
    cargo check --release --target i686-unknown-linux-gnu --all-targets
    for args in "--no-default-features" "--features internals"; do
        echo "── i686 release gather_validation ${args}"
        listed=$(cargo test --release --target i686-unknown-linux-gnu ${args} --test gather_validation -- --list)
        grep -Fxq 'overflowing_offset_rejects_before_mutation_on_32_bit: test' <<<"$listed" || {
            printf 'i686_overflow_regression_missing features=%s\n' "$args" >&2
            exit 1
        }
        cargo test --release --target i686-unknown-linux-gnu ${args} --test gather_validation
        cargo run --release --target i686-unknown-linux-gnu ${args} --example backend_report -- scalar
    done

# Compile the non-x86 library paths, including Wasm's explicit SIMD feature.
[group('test')]
cross-check:
    cargo check --target aarch64-unknown-linux-gnu --all-features --lib
    cargo check --target wasm32-unknown-unknown --all-features --lib
    RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" cargo check --target wasm32-unknown-unknown --all-features --lib

# Type-check every standalone comparison package under `external/` against
# this working copy, so public API changes cannot strand them.
[group('test')]
external-check:
    #!/usr/bin/env bash
    set -euo pipefail
    for manifest in external/*/Cargo.toml; do
        echo "external-check: ${manifest%/Cargo.toml}"
        cargo check --manifest-path "${manifest}" --all-targets
    done

# Kernel registry identity, documents, and artifact. Generation runs the
# static checks: every public `kernel::x86` function takes a mapped
# capability token first (or sits on the audited tokenless list), every
# mapped token family has entries, and every token-first function under
# `kernel::x86` is reachable through `internals` (or sits on the private
# helper allowlist). CI runs this recipe, uploads the registry, and the
# CI `isa-accept` job downloads and verifies the artifact instead of
# regenerating it.
KR_TARGET := 'x86_64-unknown-linux-gnu'
KR_DIR := 'target/kernel-registry'

# Behavioral Python tests, then generation from the pinned nightly's
# all-features x86_64 rustdoc documents (one with --document-hidden-items,
# one with --document-private-items). Needs the pinned API_NIGHTLY.
[group('gate')]
kernel-registry-check:
    API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" python3 -m unittest -q api/test_kernel_registry.py
    just _kernel-registry-generate

[private]
_kernel-registry-generate:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf {{KR_DIR}}
    mkdir -p {{KR_DIR}}
    export RUSTDOCFLAGS="-Z unstable-options --document-hidden-items"
    cargo +{{API_NIGHTLY}} rustdoc --lib --all-features --target {{KR_TARGET}} \
        --target-dir {{KR_DIR}}/doc-public \
        -- -Z unstable-options --output-format json --document-hidden-items
    doc={{KR_DIR}}/doc-public/{{KR_TARGET}}/doc/fgf.json
    [[ -f "$doc" ]] || { printf 'kernel-registry: public rustdoc JSON missing after the run\n' >&2; exit 1; }
    mv "$doc" {{KR_DIR}}/public.json
    export RUSTDOCFLAGS="-Z unstable-options --document-private-items"
    cargo +{{API_NIGHTLY}} rustdoc --lib --all-features --target {{KR_TARGET}} \
        --target-dir {{KR_DIR}}/doc-private \
        -- -Z unstable-options --output-format json --document-private-items
    doc={{KR_DIR}}/doc-private/{{KR_TARGET}}/doc/fgf.json
    [[ -f "$doc" ]] || { printf 'kernel-registry: private rustdoc JSON missing after the run\n' >&2; exit 1; }
    mv "$doc" {{KR_DIR}}/private.json
    API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" python3 api/kernel_registry.py generate \
        --public-doc {{KR_DIR}}/public.json \
        --private-doc {{KR_DIR}}/private.json \
        --format {{API_FORMAT}} --channel {{API_NIGHTLY}} \
        --target {{KR_TARGET}} --features all-features \
        --out {{KR_DIR}}/registry.json

# Require an actual tier and positive direct-kernel execution evidence before
# sweeping downgrade requests. Dedicated acceptance hosts supply their real
# tier. Every run — the initial request and each downgrade iteration — is
# audited per entry against the kernel registry at `min(required floor,
# resolved request ceiling)`, with markers above the ceiling rejected.
# REGISTRY (as in CI) uses a downloaded artifact, which is verified against
# the tested tree and never regenerated; without it the run regenerates the
# registry first, so a native host always audits its own tree.
[group('gate')]
isa-accept EXPECTED REGISTRY="":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{EXPECTED}}" in
        v4x|v3_gfni_crypto|v3|v2|scalar) ;;
        *) printf 'unsupported_acceptance_tier tier=%s\n' "{{EXPECTED}}" >&2; exit 2 ;;
    esac
    registry="{{REGISTRY}}"
    if [[ -n "${registry}" ]]; then
        [[ -f "${registry}" ]] || {
            printf 'isa-accept: registry artifact %s is missing\n' "${registry}" >&2
            exit 1
        }
        API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" \
            python3 api/kernel_registry.py check-identity "${registry}" --target {{KR_TARGET}}
        echo "isa-accept: registry artifact identity verified"
    else
        just kernel-registry-check
        registry={{KR_DIR}}/registry.json
    fi
    export SIMD_BACKEND="{{EXPECTED}}"
    export FGF_REQUIRE_TIER="{{EXPECTED}}"
    bin=$(cargo build --all-features --example backend_report --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    reported=$("${bin}" "{{EXPECTED}}")
    printf '%s\n' "${reported}"
    for field in gf8b gf8d gf16b; do
        grep -Fxq "backend ${field}={{EXPECTED}}" <<<"${reported}" || {
            printf 'isa_field_mismatch field=%s expected=%s\n' "${field}" "{{EXPECTED}}" >&2
            exit 1
        }
    done
    audit() {
        local resolved
        resolved=$(sed -n 's/^actual=//p' <<<"${1}")
        [[ -n "${resolved}" ]] || {
            printf 'isa_accept_unresolved_backend report:\n%s\n' "${1}" >&2
            exit 1
        }
        API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" \
            python3 api/kernel_registry.py audit "${registry}" \
            --floor "{{EXPECTED}}" --actual "${resolved}" --log "${2}"
    }
    log=$(mktemp)
    trap 'rm -f "$log"' EXIT
    cargo test --all-features -- --nocapture 2>&1 | tee "$log"
    audit "${reported}" "$log"
    for tier in {{TIERS}}; do
        printf 'ISA_REQUEST\t%s\n' "${tier}"
        iter_report=$(SIMD_BACKEND="${tier}" "${bin}")
        printf '%s\n' "${iter_report}"
        SIMD_BACKEND="${tier}" cargo test --all-features -- --nocapture 2>&1 | tee "$log"
        audit "${iter_report}" "$log"
    done

# Negative consumer snippets under `tests/compile_fail/`: each must fail to
# compile on the crate MSRV. The first line names the expectation: a
# panic fragment or rustc diagnostic code/text (`expect:`) that must appear
# in the error, or `expect-rejection` when only rejection is the contract.
# Comment lines two and three carry optional directives by prefix:
# `// expect-code: E0080` adds a required diagnostic code, so
# const-validation regressions pin both the named fragment and the
# evaluation error, and `// reject: <fragment>` forbids an actual
# diagnostic fragment, in either order. A reject assertion names the
# forbidden panic text of a condition that must not fire (a degree
# assertion reached before a field-validity failure, say), never text the
# snippet's own source quotes. A rejection-only case fails when stderr
# holds an unresolved-name error (`E0405`, `E0412`, `E0422`, `E0423`,
# `E0425`, `E0432`, `E0433`): such a snippet was rejected for a
# missing import, not the bound under test. Every case also fails when it
# trips on a private module path (`E0603`), because a privatized module is
# not the contract the snippet names.
# Every snippet runs against the all-features rlib; the sealed/row-hook
# raw-arithmetic probes, the raw/proof/prepared/query/removal privacy
# probes, and the map/error-boundary negatives additionally run against
# the stable-maximal rlib (every stable feature, `internals` off), each
# built separately with its exact artifact path retained, so one
# configuration's rlib can never serve the other's run. Diagnostics are
# labeled by configuration.
# Cargo does not treat that directory as test targets, so these run only
# through this recipe.
[group('test')]
compile-fail:
    #!/usr/bin/env bash
    set -euo pipefail
    rlib_of() {
        cargo +{{MSRV}} build --lib "${@}" --message-format=json |
            jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "fgf") | .filenames[]? | select(endswith(".rlib"))] | first // empty'
    }
    scratch=$(mktemp -d)
    trap 'rm -rf "${scratch}"' EXIT
    # The root package's rlib lives at one canonical path, so the second
    # build overwrites the first: each configuration's artifact is copied
    # to its own file before the next build runs.
    rlib_all_src=$(rlib_of --all-features)
    if [[ -z "${rlib_all_src}" ]]; then
        echo "compile-fail: all-features fgf rlib not found in the build output" >&2
        exit 1
    fi
    rlib_all="${scratch}/libfgf-all.rlib"
    cp "${rlib_all_src}" "${rlib_all}"
    rlib_stable_src=$(rlib_of --no-default-features --features {{API_STABLE_FEATURES}})
    if [[ -z "${rlib_stable_src}" ]]; then
        echo "compile-fail: stable-maximal fgf rlib not found in the build output" >&2
        exit 1
    fi
    rlib_stable="${scratch}/libfgf-stable.rlib"
    cp "${rlib_stable_src}" "${rlib_stable}"
    failures=0
    run_case() {
        local config="${1}" rlib="${2}" snippet="${3}"
        local header fragment code reject err directive
        header=$(head -n 1 "${snippet}")
        case "${header}" in
            "// expect-rejection") fragment="" ;;
            "// expect: "*) fragment="${header##*expect: }" ;;
            *)
                echo "compile-fail: ${snippet}: missing expectation header" >&2
                exit 1
                ;;
        esac
        # Lines two and three carry optional directives by prefix, in
        # either order.
        code=""
        reject=""
        for directive in "$(sed -n '2p' "${snippet}")" "$(sed -n '3p' "${snippet}")"; do
            case "${directive}" in
                "// expect-code: "*) code="${directive##*expect-code: }" ;;
                "// reject: "*) reject="${directive##*reject: }" ;;
                *) : ;;
            esac
        done

        if err=$(rustc +{{MSRV}} --edition 2024 --crate-type lib --extern fgf="${rlib}" -L dependency=target/debug/deps --out-dir "${scratch}" "${snippet}" 2>&1); then
            echo "compile-fail: FAIL [${config}] ${snippet} compiled successfully" >&2
            failures=$((failures + 1))
        elif grep -qE -- 'error\[E0603\]: module `[^`]+` is private' <<< "${err}"; then
            echo "compile-fail: FAIL [${config}] ${snippet}: rejected for a private module path" >&2
            failures=$((failures + 1))
        elif [[ -n "${fragment}" ]] && ! grep -qF -- "${fragment}" <<< "${err}"; then
            echo "compile-fail: FAIL [${config}] ${snippet} expected fragment: ${fragment}" >&2
            failures=$((failures + 1))
        elif [[ -n "${code}" ]] && ! grep -qF -- "error[${code}]" <<< "${err}"; then
            echo "compile-fail: FAIL [${config}] ${snippet} expected code: ${code}" >&2
            failures=$((failures + 1))
        elif [[ -n "${reject}" ]] && grep -qF -- "${reject}" <<< "${err}"; then
            echo "compile-fail: FAIL [${config}] ${snippet} forbidden fragment present: ${reject}" >&2
            failures=$((failures + 1))
        elif [[ -z "${fragment}" ]] && grep -qE -- 'error\[E(0405|0412|0422|0423|0425|0432|0433)\]' <<< "${err}"; then
            echo "compile-fail: FAIL [${config}] ${snippet}: rejected for an unresolved name" >&2
            failures=$((failures + 1))
        else
            echo "compile-fail: ok [${config}] $(basename "${snippet}")"
        fi
    }
    for snippet in tests/compile_fail/*.rs; do
        run_case all-features "${rlib_all}" "${snippet}"
    done
    for snippet in \
        tests/compile_fail/sealed_*.rs \
        tests/compile_fail/row_hook_*.rs \
        tests/compile_fail/raw_*_without_proof.rs \
        tests/compile_fail/field_buffer_*.rs \
        tests/compile_fail/field_kernels_*.rs \
        tests/compile_fail/embedding_*.rs \
        tests/compile_fail/isomorphism_*.rs \
        tests/compile_fail/map_*.rs \
        tests/compile_fail/maps_*.rs \
        tests/compile_fail/image_error_*.rs \
        tests/compile_fail/range_error_*.rs
    do
        [[ -e "${snippet}" ]] || continue
        run_case stable-maximal "${rlib_stable}" "${snippet}"
    done
    if [[ "${failures}" -ne 0 ]]; then
        echo "compile-fail: ${failures} case(s) failed" >&2
        exit 1
    fi

# MSRV acceptance: every negative consumer snippet rejects and the positive
# consumer matrix runs on the crate MSRV toolchain.
[group('test')]
msrv-accept:
    just compile-fail
    cargo +{{MSRV}} test --all-features --test consumers

# Check the published dependency lock in an isolated checkout.
[group('test')]
msrv-locked:
    cargo +{{MSRV}} check --locked --all-features --all-targets

# Exercise the exact package allowlist outside umbrella patches without
# requiring or modifying a commit in the developer's working tree.
[group('test')]
package-check:
    #!/usr/bin/env bash
    set -euo pipefail
    scratch=$(mktemp -d /tmp/fgf-package-check.XXXXXXXX)
    trap 'rm -rf "$scratch"' EXIT
    mkdir "$scratch/source" "$scratch/package"
    tar --exclude=.git --exclude=target --exclude=bench-records --exclude=__pycache__ --exclude=.lucid -cf - . \
        | tar -xf - -C "$scratch/source"
    (
        cd "$scratch/source"
        cargo package --allow-dirty --no-verify
    )
    archives=("$scratch/source/target/package/"*.crate)
    [[ "${#archives[@]}" -eq 1 ]] || { printf 'package_archive_count count=%s\n' "${#archives[@]}" >&2; exit 1; }
    tar -xf "${archives[0]}" -C "$scratch/package"
    (
        cd "$scratch/package"/*
        cargo +{{MSRV}} check --locked --all-features --all-targets
        for name in custom_extensions basis_maps packed_no_alloc; do
            cargo +{{MSRV}} run --locked --example "$name" --no-default-features
        done
    )

# Regenerate the stable API inventory from the exact `cargo package` archive
# and compare it with the committed `api/`. The archive excludes `api/`, so
# the committed inventory, its generators, and the recipe files are copied
# beside the extracted source; the inventory is then derived from the
# packaged source alone.
[group('release')]
package-api:
    #!/usr/bin/env bash
    set -euo pipefail
    scratch=$(mktemp -d /tmp/fgf-package-api.XXXXXXXX)
    trap 'rm -rf "$scratch"' EXIT
    mkdir "$scratch/source" "$scratch/package"
    tar --exclude=.git --exclude=target --exclude=bench-records --exclude=__pycache__ --exclude=.lucid -cf - . \
        | tar -xf - -C "$scratch/source"
    (
        cd "$scratch/source"
        cargo package --allow-dirty --no-verify
    )
    archives=("$scratch/source/target/package/"*.crate)
    [[ "${#archives[@]}" -eq 1 ]] || { printf 'package_archive_count count=%s\n' "${#archives[@]}" >&2; exit 1; }
    tar -xf "${archives[0]}" -C "$scratch/package"
    packaged=("$scratch/package"/*)
    cp -r api justfile crate.just "${packaged[0]}/"
    (
        cd "${packaged[0]}"
        just api
    )
    printf 'package-api: %s exposes the committed inventory\n' "$(basename "${archives[0]}")"

# Paired measurement configuration for `bench-pair`. The runner overlays the
# snapshot's bench target, manifests, toolchain, features, and flags onto
# checkouts of both library revisions, so `PAIR_FEATURES` and `PAIR_ALIGN`
# describe the measurement snapshot rather than the caller's tree. `PAIR_RUNS`
# and `PAIR_DRIFT_CAP` are per-host values using exact hostnames, written
# 'host:value host:value'; a single numeric value is accepted for a
# single-host campaign. Hosts without an entry are refused by acceptance. The
# values below are the native calibration recommendations; acceptance is still
# gated by the host, tier, toolchain, and flag bindings.
# The calibrated host, tier, toolchain, and flag bindings are checked against
# the live host before any acceptance measurement; `PAIR_CALIBRATED_FLAGS` must
# equal `PAIR_ALIGN` for the host it was calibrated on.
PAIR_RUNS := 'Comet:24 zbox:24'
# Start-control IQR drift caps are half the noninferiority margin
# (PAIR_DELTA / 2) on every host: drift below that cannot create or hide a
# decision. The calibration recommendation's cap remains a diagnostic.
PAIR_DRIFT_CAP := 'Comet:0.015 zbox:0.015'
PAIR_ROUNDS := '7'
PAIR_DELTA := '0.03'
PAIR_GAMMA := '0.03'
PAIR_FEATURES := 'simd512'
PAIR_ALIGN := '-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6'
# Exact hostnames; no substring match. Tiers are per host, written
# 'host1:t1,t2 host2:t3,t4' (for example
# 'willow-cove:v4x,v3,scalar golden-cove:v3_gfni_crypto,v3,scalar').
PAIR_CALIBRATED_HOSTS := 'Comet zbox'
PAIR_CALIBRATED_TIERS := 'Comet:v3_gfni_crypto,v3,v2,scalar zbox:v4x,v3,v2,scalar'
PAIR_CALIBRATED_TOOLCHAIN := '1.98.1/48a229ceaefd4985c50990b14116b6d856af0985'
PAIR_CALIBRATED_FLAGS := '-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6'
PAIR_SNAPSHOT := 'HEAD'
# Calibration collects its own budget; PAIR_RUNS never caps it. The native
# calibration sessions collected 24 runs, so the budget must be at least the
# recommended run count above.
PAIR_CALIBRATION_RUNS := '24'
# Recommended drift cap = observed maximum across calibration sessions times
# this stated safety factor.
PAIR_DRIFT_SAFETY_FACTOR := '1.25'
# SMT sampling is fixed here: a window below 200 ms is too short to resolve
# load, and a busy threshold above 0.05 admits a loaded sibling.
PAIR_SMT_WINDOW := '0.2'
PAIR_SMT_BUSY_THRESHOLD := '0.05'
# Empty means taskset-only pinning. Setting this to e.g. '--idle 0' adds chrt.
PAIR_CHRT := ''
# Runs required when a signed unstable cell is also a superiority target;
# uncalibrated until an owner sets it from native evidence.
PAIR_UNSTABLE_TARGET_RUNS := 'uncalibrated'

# Standalone processes are the repetition axis for descriptive snapshots.
# This budget describes spread; it never supplies a paired acceptance bound.
BASELINE_RUNS := '5'
BASELINE_ROUNDS := '1'

# Public scalar, storage, map, and bulk breadth, without a duplicated pair.
[group('bench')]
bench-baseline *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    exec python3 benches/bench_baseline.py \
        --snapshot '{{PAIR_SNAPSHOT}}' \
        --runs '{{BASELINE_RUNS}}' \
        --rounds '{{BASELINE_ROUNDS}}' \
        --features '{{PAIR_FEATURES}}' \
        --rustflags '{{PAIR_ALIGN}}' \
        --smt-window '{{PAIR_SMT_WINDOW}}' \
        --smt-busy-threshold '{{PAIR_SMT_BUSY_THRESHOLD}}' \
        --chrt '{{PAIR_CHRT}}' \
        {{ARGS}}

# Paired before/after measurement of the public operations bench. Calibration,
# threshold, and layout-only campaigns are the same recipe with `--mode` and
# their declaration arguments; further arguments pass through to the runner.
# See `python3 benches/bench_pair.py --help` for the full surface.
[group('bench')]
bench-pair *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    exec python3 benches/bench_pair.py \
        --snapshot '{{PAIR_SNAPSHOT}}' \
        --runs '{{PAIR_RUNS}}' \
        --drift-cap '{{PAIR_DRIFT_CAP}}' \
        --rounds '{{PAIR_ROUNDS}}' \
        --delta '{{PAIR_DELTA}}' \
        --gamma '{{PAIR_GAMMA}}' \
        --features '{{PAIR_FEATURES}}' \
        --rustflags '{{PAIR_ALIGN}}' \
        --calibrated-hosts '{{PAIR_CALIBRATED_HOSTS}}' \
        --calibrated-tiers '{{PAIR_CALIBRATED_TIERS}}' \
        --calibrated-toolchain '{{PAIR_CALIBRATED_TOOLCHAIN}}' \
        --calibrated-flags '{{PAIR_CALIBRATED_FLAGS}}' \
        --calibration-runs '{{PAIR_CALIBRATION_RUNS}}' \
        --drift-safety-factor '{{PAIR_DRIFT_SAFETY_FACTOR}}' \
        --smt-window '{{PAIR_SMT_WINDOW}}' \
        --smt-busy-threshold '{{PAIR_SMT_BUSY_THRESHOLD}}' \
        --chrt '{{PAIR_CHRT}}' \
        --unstable-target-runs '{{PAIR_UNSTABLE_TARGET_RUNS}}' \
        --policy-json '{"runs":"{{PAIR_RUNS}}","drift_cap":"{{PAIR_DRIFT_CAP}}","rounds":"{{PAIR_ROUNDS}}","delta":"{{PAIR_DELTA}}","gamma":"{{PAIR_GAMMA}}","calibrated_hosts":"{{PAIR_CALIBRATED_HOSTS}}","calibrated_tiers":"{{PAIR_CALIBRATED_TIERS}}","calibrated_toolchain":"{{PAIR_CALIBRATED_TOOLCHAIN}}","calibrated_flags":"{{PAIR_CALIBRATED_FLAGS}}","smt_window":"{{PAIR_SMT_WINDOW}}","smt_busy_threshold":"{{PAIR_SMT_BUSY_THRESHOLD}}","chrt":"{{PAIR_CHRT}}","unstable_target_runs":"{{PAIR_UNSTABLE_TARGET_RUNS}}"}' \
        {{ARGS}}

# Instrumented coverage map of the public operations bench at one tier. The
# bench is rebuilt on the pinned nightly with branch coverage, `--check` runs
# once per cell with its own profile, and the merged map relates every
# kernel-registry entry and threshold branch to the cells that reached it.
# The threshold inventory path comes from `FGF_THRESHOLD_INVENTORY`; the
# scanner is read beside it.
[group('bench')]
bench-coverage-map TIER *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    inventory="${FGF_THRESHOLD_INVENTORY:-}"
    [[ -n "${inventory}" ]] || {
        printf 'bench-coverage-map: set FGF_THRESHOLD_INVENTORY to the threshold inventory.json\n' >&2
        exit 2
    }
    registry=target/kernel-registry/registry.json
    private=target/kernel-registry/private.json
    if [[ ! -f "${registry}" || ! -f "${private}" ]] || \
       ! API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" \
         python3 api/kernel_registry.py check-identity "${registry}" --target {{KR_TARGET}}; then
        just kernel-registry-check
    fi
    exec python3 benches/coverage_map.py \
        --tier {{TIER}} \
        --inventory "${inventory}" \
        --scan "$(dirname "${inventory}")/scan.py" \
        --toolchain {{API_NIGHTLY}} \
        {{ARGS}}

# Build a portable benchmark artifact for interleaved remote comparisons.
[group('bench')]
bench-build NAME:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p target/fgf-bench
    cargo bench --all-features --bench "{{NAME}}" --no-run --message-format=json-render-diagnostics |
        jq -ers --arg name "{{NAME}}" \
            '[.[] | select(.reason == "compiler-artifact" and .target.name == $name and .target.kind == ["bench"] and .executable != null)] | if length == 1 then .[0].executable else error("expected one benchmark executable") end' \
            >"target/fgf-bench/{{NAME}}.path"

# Miri over the dispatched GFNI paths: the owned-unsafe residue that the
# scalar configuration does not compile. Runtime detection reports nothing
# under the interpreter, so the tier is promoted at compile time; MIRIFLAGS
# stays fixed for this recipe. The ops lines cover the public dispatch seam;
# the internals lines provide direct-kernel residue coverage. Non-temporal
# stores stay proof-review-only: the fence intrinsic is unsupported under
# Miri.
# Everything listed here must pass.
[group('test')]
unsafe-check-gfni:
    #!/usr/bin/env bash
    set -euo pipefail
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v3 -C target-feature=+gfni,+vaes,+vpclmulqdq"
    export MIRIFLAGS="-Zmiri-disable-isolation"
    while IFS= read -r args; do
        [[ -z "${args}" ]] && continue
        echo "── cargo +nightly miri test ${args}"
        # shellcheck disable=SC2086
        cargo +nightly miri test ${args}
    done <<< "{{MIRI_GFNI}}"

# `cargo miri test` argument sets for `unsafe-check-gfni`, one per line. CI's
# `miri-gfni` job invokes that recipe, so this variable is the single source
# of the list.
MIRI_GFNI := '''--test ops gf8_matrix_scattered_matches_contiguous -- --exact
--test ops gf16_mul_into_matrix_overwrites -- --exact
--features internals --test internals proven_gf8_mul_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_rs_gfni_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_scatter_gather_match_scalar -- --exact
--features internals --test internals proven_gf8_matrix_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_matrix_scattered_matches_scalar -- --exact
--features internals --test internals proven_gf16_kernels_match_scalar -- --exact
--features internals --test internals proven_gf16_degenerate_boundaries -- --exact'''

# Each round shuffles the family's units and prints the realized order, so no
# unit owns the warm or cold position. `gf` interleaves the binary-field self
# suite, the ISA-L and klauspost trio, and the native GF16 harness; `gdl`
# (Goldilocks) and `m31` (Mersenne31 and QuadMersenne31) interleave the self
# suite with the Plonky3 harness; `gf16` repeats the native GF16 harness alone
# and passes ARGS to it. The harnesses live in `external/` as separate
# unpublished packages: linking a competitor needs a build script, and `fgf`
# itself carries none. Every unit runs pinned to `FEC_GOLDEN_CORE`.
# Five-round competitor campaign for one family: gf, gdl, m31, or gf16.
[group('bench')]
bench-comp FAMILY *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    pin=(taskset -c "${FEC_GOLDEN_CORE}")
    case "{{FAMILY}}" in
        gf)
            pkg-config --modversion libisal >/dev/null || {
                echo "libisal not found through pkg-config; install Intel ISA-L" >&2
                exit 1
            }
            command -v go >/dev/null || {
                echo "go not found in PATH; install the Go toolchain" >&2
                exit 1
            }
            just _trio-build
            just gf16-build
            units=(self trio gf16)
            ;;
        gdl|m31)
            RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path external/prime-bench/Cargo.toml --features fgf/simd512
            units=(self prime)
            ;;
        gf16)
            just gf16-build
            units=(gf16)
            pin=(chrt --idle 0 taskset -c "${FEC_GOLDEN_CORE}")
            ;;
        *)
            printf 'unsupported_family family=%s\n' "{{FAMILY}}" >&2
            exit 2
            ;;
    esac
    echo "pinned to core ${FEC_GOLDEN_CORE}"
    for run in 1 2 3 4 5; do
        echo "== round ${run}"
        printf '%s\n' "${units[@]}" | shuf | while read -r unit; do
            echo "-- round ${run} unit ${unit}"
            case "${unit}" in
                self)
                    just bench operations --time --rounds 1 --family "{{FAMILY}}"
                    if [[ "{{FAMILY}}" == gf ]]; then
                        just bench compare
                    fi
                    ;;
                trio) "${pin[@]}" external/bench-trio/target/release/fgf-bench-trio ;;
                gf16) "${pin[@]}" external/gf16-bench/target/release/fgf-gf16-bench {{ARGS}} ;;
                prime) "${pin[@]}" external/prime-bench/target/release/fgf-bench-prime {{FAMILY}} ;;
            esac
        done
    done

# Build the byte-field competitor harness without running a timing campaign.
[private]
_trio-build:
    cargo build --release --manifest-path external/bench-trio/Cargo.toml

# Native GF16 field comparison; every arm is required and validated before timing.
[private]
[group('bench')]
gf16-build:
    cargo build --release --manifest-path external/gf16-bench/Cargo.toml

[private]
[group('bench')]
gf16-check *ARGS: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --check {{ARGS}}

# Validate the portable public benchmark inventory and fixture results. Native
# competitors require capable acceptance hardware and have a separate recipe.
[group('bench')]
benchmark-check:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 -m unittest discover -s benches -p 'test_*.py'
    bin=$(cargo build --profile test --all-features --bench surfaces --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    "${bin}" --check | tee "$scratch/check.log"
    just benchmark-api-json
    python3 benches/check_surfaces.py --rustdoc target/doc/fgf.json --check-log "$scratch/check.log"

# Compiler-derived stable callable set: no hand-maintained source-text manifest.
[private]
[group('bench')]
benchmark-api-json:
    cargo +{{API_NIGHTLY}} rustdoc --all-features --lib -- -Z unstable-options --output-format json

# Full exact native competitor differentials; never silently skipped on a host
# lacking the required AVX2/GFNI/crypto capability.
[group('bench')]
benchmark-check-native: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --presentations --check
    just gf16-check

# Render the committed measurement record, or verify it without writing.
[group('bench')]
bench-publish *ARGS:
    python3 benches/publish.py {{ARGS}}

# Documentation in the published default-feature configuration, the one
# docs.rs builds.
[group('doc')]
doc-stable:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# The full release gate: the shared pull-request gate, the promoted GFNI
# Miri residue that CI runs as a separate job, and every crate-specific
# acceptance recipe. Cross-compilation needs the aarch64 and wasm32 targets
# installed, `unsafe-check-gfni` needs nightly Miri, the gather validation
# needs `gcc-multilib`, and the external packages need their native
# toolchains (Go, Intel ISA-L, C/C++). Publication is gated separately by
# `just publish-dry` from a clean tree; `isa-accept` needs the host's pinned
# tier and stays a per-host call.
[group('gate')]
release-accept:
    just validate
    just unsafe-check-gfni
    just features-internals
    just features-alloc
    just msrv-locked
    just msrv-accept
    just cross-check
    just doc-stable
    just external-check
    just test-gather-validation-32
    just examples
    just benchmark-check
    just benchmark-check-native
    just package-check
    just api
    just kernel-registry-check
    just package-api
    just api-semver-release

# Generate the stable inventories, their metadata, and the one-path report
# into `target/api/generated`. Every configuration builds unchanged and no
# inventory line is filtered: the `all` cell builds every feature except
# `internals`, and a separate `--all-features` rustdoc document must have the
# same stable export path set. The rustdoc JSON is deleted before
# each run and copied to `$out/json` after it, so a skipped rustdoc pass
# fails instead of reusing a stale document. Each inventory closes with the
# external re-export section, and the metadata records the resolved version
# of every re-exported dependency. Tool versions are checked before anything
# is generated. It runs as a dependency so variable overrides on the command
# line reach it.
[private]
_api-generate:
    #!/usr/bin/env bash
    set -euo pipefail
    have=$(cargo public-api --version 2>/dev/null || true)
    [[ "$have" == "cargo-public-api {{API_PUBLIC_API}}" ]] || {
        printf 'api: need cargo-public-api %s, found "%s"; run: cargo binstall cargo-public-api@%s\n' "{{API_PUBLIC_API}}" "$have" "{{API_PUBLIC_API}}" >&2
        exit 2
    }
    have=$(cargo semver-checks --version 2>/dev/null || true)
    [[ "$have" == "cargo-semver-checks {{API_SEMVER_CHECKS}}" ]] || {
        printf 'api: need cargo-semver-checks %s, found "%s"; run: cargo binstall cargo-semver-checks@%s\n' "{{API_SEMVER_CHECKS}}" "$have" "{{API_SEMVER_CHECKS}}" >&2
        exit 2
    }
    API_NIGHTLY="{{API_NIGHTLY}}" API_FORMAT="{{API_FORMAT}}" python3 -m unittest -q api/test_surface.py
    declared=$(cargo metadata --no-deps --format-version 1 \
        | jq -r '.packages[0].features | keys - ["default", "internals"] | sort | join(",")')
    pinned=$(tr ',' '\n' <<<"{{API_STABLE_FEATURES}}" | sort | paste -sd,)
    [[ "$declared" == "$pinned" ]] || {
        printf 'api: API_STABLE_FEATURES is %s but the manifest declares %s\n' "$pinned" "$declared" >&2
        exit 2
    }
    out=target/api/generated
    rm -rf "$out"
    mkdir -p "$out/inventory" "$out/json"
    export RUSTDOCFLAGS="-Z unstable-options --document-hidden-items"
    configs=("portable|--no-default-features" "alloc|--no-default-features --features alloc" "default|" "all|--no-default-features --features {{API_STABLE_FEATURES}}")
    {
        printf 'nightly %s\n' "{{API_NIGHTLY}}"
        printf 'rustdoc-json-format %s\n' "{{API_FORMAT}}"
        printf 'cargo-public-api %s (-s: blanket impls omitted)\n' "{{API_PUBLIC_API}}"
        printf 'cargo-semver-checks %s\n' "{{API_SEMVER_CHECKS}}"
        printf 'rustdocflags %s\n' "$RUSTDOCFLAGS"
        printf 'targets %s\n' "{{API_TARGETS}}"
        for entry in "${configs[@]}"; do flags="${entry#*|}"; printf 'config %s%s\n' "${entry%%|*}" "${flags:+ $flags}"; done
    } >"$out/metadata.txt"
    failed=0
    for target in {{API_TARGETS}}; do
        for entry in "${configs[@]}"; do
            config="${entry%%|*}"
            # shellcheck disable=SC2206
            flags=(${entry#*|})
            doc="target/api/$target/doc/fgf.json"
            rm -f "$doc"
            cargo +{{API_NIGHTLY}} public-api -s --color never --target "$target" \
                --target-dir target/api "${flags[@]}" >"$out/inventory/$config-$target.txt" \
                2>"$out/$config-$target.log" || {
                cat "$out/$config-$target.log" >&2
                exit 1
            }
            [[ -f "$doc" ]] || {
                printf 'api: %s %s: rustdoc JSON missing after the run\n' "$config" "$target" >&2
                exit 1
            }
            cp "$doc" "$out/json/$config-$target.json"
            printf '# external re-exports\n' >>"$out/inventory/$config-$target.txt"
            python3 api/surface.py reexports --format {{API_FORMAT}} \
                "$out/json/$config-$target.json" >>"$out/inventory/$config-$target.txt"
            if ! report=$(python3 api/surface.py paths --format {{API_FORMAT}} \
                "$out/json/$config-$target.json" --mode {{API_PATHS_MODE}}); then
                printf '%s\n' "$report"
                failed=1
            fi
            printf 'api: %-8s %-26s %s\n' "$config" "$target" "$(head -n 1 <<<"$report" | sed 's/^surface: //')"
            [[ "$config-$target" == all-x86_64-unknown-linux-gnu ]] && printf '%s\n' "$report" >"$out/paths.txt"
        done
        doc="target/api/$target/doc/fgf.json"
        rm -f "$doc"
        cargo +{{API_NIGHTLY}} rustdoc --lib --all-features --target "$target" \
            --target-dir target/api -- -Z unstable-options --output-format json --document-hidden-items \
            2>"$out/all-features-$target.log" || {
            cat "$out/all-features-$target.log" >&2
            exit 1
        }
        [[ -f "$doc" ]] || {
            printf 'api: all-features %s: rustdoc JSON missing after the run\n' "$target" >&2
            exit 1
        }
        cp "$doc" "$out/json/all-features-$target.json"
        python3 api/surface.py stable-paths-equal --format {{API_FORMAT}} \
            "$out/json/all-features-$target.json" "$out/json/all-$target.json" || failed=1
    done
    metadata=$(cargo metadata --format-version 1)
    while IFS= read -r dep; do
        # Versions come from fgf's own normal dependency edges, matched on the
        # rustdoc crate name, so another copy elsewhere in the graph is ignored.
        versions=$(jq -r --arg name "$dep" '
            .resolve.root as $root
            | (.resolve.nodes[] | select(.id == $root) | .deps[]
               | select(any(.dep_kinds[]; .kind == null)) | select(.name == $name) | .pkg) as $pkg
            | .packages[] | select(.id == $pkg) | .version' <<<"$metadata" | sort -u)
        [[ -n "$versions" && $(wc -l <<<"$versions") -eq 1 ]] || {
            printf 'api: re-exported crate %s does not resolve to exactly one normal dependency of fgf (found: %s)\n' "$dep" "${versions:-none}" >&2
            exit 1
        }
        printf 'dependency %s %s\n' "$dep" "$versions"
    done < <(grep -h '^reexport ' "$out"/inventory/*.txt | sed 's/^reexport [^=]*= //' | cut -d: -f1 | sort -u) >>"$out/metadata.txt"
    if [[ "{{API_PATHS_MODE}}" == report ]]; then cat "$out/paths.txt"; else tail -n 1 "$out/paths.txt"; fi
    for entry in "${configs[@]}"; do
        config="${entry%%|*}"
        files=()
        for target in {{API_TARGETS}}; do files+=("$out/inventory/$config-$target.txt"); done
        python3 api/surface.py same "$config" "${files[@]}" || failed=1
    done
    exit "$failed"

# Check the committed stable API inventory: regenerate every inventory and
# fail on any difference from `api/`. Read-only.
[group('gate')]
api: _api-generate
    #!/usr/bin/env bash
    set -euo pipefail
    out=target/api/generated
    if ! diff -u api/metadata.txt "$out/metadata.txt" || ! diff -ru api/inventory "$out/inventory"; then
        printf 'api: the public surface differs from the committed inventory; an authorized API change runs `just api-update`\n' >&2
        exit 1
    fi
    echo "api: stable surface matches the committed inventory"

# Rewrite the committed inventory. Run only inside an explicitly authorized
# API change; review the resulting diff.
[group('gate')]
api-update: _api-generate
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf api/inventory
    cp -r target/api/generated/inventory api/inventory
    cp target/api/generated/metadata.txt api/metadata.txt
    echo "api: inventory rewritten; review git diff -- api/"

# Semver check of the stable feature sets against the highest published
# non-yanked, non-prerelease `API_MAJOR`.x release, sorted numerically
# whatever order crates.io returns, or against BASELINE when given for
# development verification. `internals` is never enabled.
[group('gate')]
api-semver BASELINE="":
    #!/usr/bin/env bash
    set -euo pipefail
    have=$(cargo semver-checks --version 2>/dev/null || true)
    [[ "$have" == "cargo-semver-checks {{API_SEMVER_CHECKS}}" ]] || {
        printf 'api-semver: need cargo-semver-checks %s, found "%s"\n' "{{API_SEMVER_CHECKS}}" "$have" >&2
        exit 2
    }
    baseline="{{BASELINE}}"
    if [[ -z "$baseline" ]]; then
        listing=$(mktemp)
        trap 'rm -f "$listing"' EXIT
        page="?per_page=100"
        while [[ -n "$page" ]]; do
            body=$(curl -fsS -A "fgf-api-semver" "https://crates.io/api/v1/crates/{{CRATE}}/versions$page") || {
                printf 'api-semver: could not query crates.io for the published baseline\n' >&2
                exit 1
            }
            jq -r '.versions[] | select(.yanked | not) | .num' <<<"$body" >>"$listing"
            page=$(jq -r '.meta.next_page // empty' <<<"$body")
        done
        baseline=$(grep -E "^{{API_MAJOR}}\.[0-9]+\.[0-9]+$" "$listing" | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1 || true)
        [[ -n "$baseline" ]] || {
            printf 'api-semver: no published non-prerelease %s.x release to compare against\n' "{{API_MAJOR}}" >&2
            exit 1
        }
    fi
    current=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')
    printf 'api-semver: candidate %s against published %s\n' "$current" "$baseline"
    configs=("portable|--only-explicit-features" "alloc|--only-explicit-features --features alloc" "default|--default-features" "stable-max|--only-explicit-features --features {{API_STABLE_FEATURES}}")
    for entry in "${configs[@]}"; do
        # shellcheck disable=SC2206
        flags=(${entry#*|})
        printf 'api-semver: %s\n' "${entry%%|*}"
        cargo +{{API_NIGHTLY}} semver-checks check-release --baseline-version "$baseline" "${flags[@]}"
    done

# Release gate for `api-semver`, decided by the version's major number: a
# major below `API_MAJOR` is skipped, `API_MAJOR.0.0` and its prereleases are
# skipped because the first release of a major has no baseline of its own,
# every other `API_MAJOR` version including prereleases runs the check, and a
# major above `API_MAJOR` fails until the constant is raised.
[private]
api-semver-release VERSION="":
    #!/usr/bin/env bash
    set -euo pipefail
    version="{{VERSION}}"
    [[ -n "$version" ]] || version=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')
    if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
        printf 'api-semver: cannot parse version %s\n' "$version" >&2
        exit 2
    fi
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    patch="${BASH_REMATCH[3]}"
    if (( major > {{API_MAJOR}} )); then
        printf 'api-semver: %s is above API_MAJOR %s; raise API_MAJOR in crate.just\n' "$version" "{{API_MAJOR}}" >&2
        exit 1
    fi
    if (( major < {{API_MAJOR}} || (minor == 0 && patch == 0) )); then
        printf 'api-semver: skipped for %s; releases before %s and the first %s.0.0 of the major have no published %s.x baseline\n' \
            "$version" "{{API_MAJOR}}" "{{API_MAJOR}}" "{{API_MAJOR}}"
    else
        just api-semver
    fi
