# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.93'
API_NIGHTLY := 'nightly-2026-09-30'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v4x v3_gfni_crypto v3 v2 scalar'

# Owned-unsafe coverage is provided by the promoted GFNI recipe below.
MIRI := ''

# Coverage excludes the full x86 binary-field kernel subtrees and the complete
# gf8/tower dispatch files, including their portable and narrower-ISA branches.
# AVX-512 QuadMersenne31, residue, and GF(2) interop bodies and the GF(2)
# interop dispatch are also excluded. AVX2 residue kernels and mixed-ISA prime
# dispatch modules remain included.
COV_IGNORE := '"kernel/x86/(gf8|gf16|gf32|gf64)(/|\.rs$)|kernel/x86/quad_mersenne31/avx512\.rs$|kernel/x86/residue/avx512(/|\.rs$)|kernel/x86/gf1/avx512\.rs$|kernel/gf1/dispatch\.rs$|kernel/(gf8|tower)\.rs$"'

# Export the existing merged profiles, including after a failed coverage gate.
[group('cover')]
cover-lcov:
    cargo llvm-cov report --lcov --output-path coverage.lcov --ignore-filename-regex {{COV_IGNORE}}


# Run one educational example, passing additional Cargo options through.
[group('examples')]
example NAME *ARGS:
    cargo run --example {{NAME}} {{ARGS}}

# Run all educational examples; the timing probe is a separate diagnostic.
[group('examples')]
examples *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    for name in field_arithmetic shard_recovery streaming_encode binary_syndrome extension_fields embeddings prime_vectors generic_prime custom_extensions basis_maps packed_no_alloc; do
        cargo run --example "${name}" {{ARGS}}
    done

# The library-noalloc examples: they run with every feature off, the
# strongest portability claim an example can make. The prepared-collection
# and timing-probe examples stay out because they need `alloc` or `std`.
[group('examples')]
examples-portable *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    for name in field_arithmetic binary_syndrome extension_fields embeddings custom_extensions basis_maps packed_no_alloc; do
        cargo run --example "${name}" --no-default-features {{ARGS}}
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

# Print process and representative-field backends without inferring execution
# coverage from a downgrade request.
[group('test')]
backend-report:
    #!/usr/bin/env bash
    set -euo pipefail
    bin=$(cargo build --all-features --example backend_report --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    "${bin}"

# Require an actual tier and positive direct-kernel execution evidence before
# sweeping downgrade requests. Dedicated acceptance hosts supply their real tier.
[group('gate')]
isa-accept EXPECTED:
    #!/usr/bin/env bash
    set -euo pipefail
    export SIMD_BACKEND="{{EXPECTED}}"
    bin=$(cargo build --all-features --example backend_report --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    reported=$("${bin}" "{{EXPECTED}}")
    printf '%s\n' "${reported}"
    for field in gf8b gf8d gf16b; do
        grep -Fxq "backend ${field}={{EXPECTED}}" <<<"${reported}" || {
            printf 'isa_field_mismatch field=%s expected=%s\n' "${field}" "{{EXPECTED}}" >&2
            exit 1
        }
    done
    for features in default all; do
        just route-accept "{{EXPECTED}}" "$features"
    done
    log=$(mktemp)
    trap 'rm -f "$log"' EXIT
    cargo test --all-features -- --nocapture 2>&1 | tee "$log"
    case "{{EXPECTED}}" in
        v4x) required=(v4x v3_gfni_crypto v3 v2) ;;
        v3_gfni_crypto) required=(v3_gfni_crypto v3 v2) ;;
        v3) required=(v3 v2) ;;
        v2) required=(v2) ;;
        scalar) required=() ;;
        *) printf 'unsupported_acceptance_tier tier=%s\n' "{{EXPECTED}}" >&2; exit 2 ;;
    esac
    for tier in "${required[@]}"; do
        grep -Fq "kernel_differential executing tier=${tier} " "$log" || {
            printf 'isa_execution_missing expected=%s\n' "$tier" >&2
            exit 1
        }
    done
    if [[ "{{EXPECTED}}" == v4x ]]; then
        grep -Fq 'executing kernel=avx512' "$log" || {
            printf 'isa_execution_missing kernel=avx512\n' >&2
            exit 1
        }
        for case in gf8_avx512_kernels_match_reference gf16_wide_kernels_match_reference prime_avx512_kernels_match_reference; do
            grep -Fq "kernel_differential executing tier=v4x case=kernel::tests::x86::$case" "$log" || {
                printf 'isa_execution_missing case=%s\n' "$case" >&2
                exit 1
            }
        done
    fi
    for tier in {{TIERS}}; do
        printf 'ISA_REQUEST\t%s\n' "${tier}"
        SIMD_BACKEND="${tier}" "${bin}"
        SIMD_BACKEND="${tier}" cargo test --all-features -- --nocapture
    done

# Negative consumer snippets under `tests/compile_fail/`: each must fail to
# compile on the crate MSRV. The first line names the expectation: a
# crate-owned panic fragment (`expect:`) that must appear in the error, or
# `expect-rejection` for trait-bound and privacy rejections, where only the
# failure itself is the contract. A rejection case fails when stderr holds
# an unresolved-name error (`E0412`, `E0425`, `E0432`, `E0433`): such a
# snippet was rejected for a missing import, not the bound under test.
# Cargo does not treat that directory as test targets, so these run only
# through this recipe.
[group('test')]
compile-fail:
    #!/usr/bin/env bash
    set -euo pipefail
    rlib=$(cargo +{{MSRV}} build --lib --all-features --message-format=json | jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "fgf") | .filenames[]? | select(endswith(".rlib"))] | first // empty')
    if [[ -z "${rlib}" ]]; then
        echo "compile-fail: fgf rlib not found in the build output" >&2
        exit 1
    fi
    scratch=$(mktemp -d)
    trap 'rm -rf "${scratch}"' EXIT
    failures=0
    for snippet in tests/compile_fail/*.rs; do
        header=$(head -n 1 "${snippet}")
        case "${header}" in
            "// expect-rejection") fragment="" ;;
            "// expect: "*) fragment="${header##*expect: }" ;;
            *)
                echo "compile-fail: ${snippet}: missing expectation header" >&2
                exit 1
                ;;
        esac
        if err=$(rustc +{{MSRV}} --edition 2024 --crate-type lib --extern fgf="${rlib}" -L dependency=target/debug/deps --out-dir "${scratch}" "${snippet}" 2>&1); then
            echo "compile-fail: FAIL ${snippet} compiled successfully" >&2
            failures=$((failures + 1))
        elif [[ -n "${fragment}" ]] && ! grep -qF -- "${fragment}" <<< "${err}"; then
            echo "compile-fail: FAIL ${snippet} expected fragment: ${fragment}" >&2
            failures=$((failures + 1))
        elif [[ -z "${fragment}" ]] && grep -qE -- 'error\[E(0412|0425|0432|0433)\]' <<< "${err}"; then
            echo "compile-fail: FAIL ${snippet}: rejected for an unresolved name" >&2
            failures=$((failures + 1))
        else
            echo "compile-fail: ok $(basename "${snippet}")"
        fi
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

# `cargo miri test` argument sets for `unsafe-check-gfni`, one per line. The
# CI `miri-gfni` job mirrors this list; keep the two in sync.
MIRI_GFNI := '''--test ops gf8_matrix_scattered_matches_contiguous -- --exact
--test ops gf16_mul_into_matrix_overwrites -- --exact
--features internals --test internals proven_gf8_mul_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_rs_gfni_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_scatter_gather_match_scalar -- --exact
--features internals --test internals proven_gf8_matrix_kernels_match_scalar -- --exact
--features internals --test internals proven_gf8_matrix_scattered_matches_scalar -- --exact
--features internals --test internals proven_gf16_kernels_match_scalar -- --exact
--features internals --test internals proven_gf16_degenerate_boundaries -- --exact'''

# Field-family benchmark suites. `bench-<field>` runs that family's self
# benchmarks in one pass. `bench-<field>-comp` runs five complete rounds
# interleaving the family's self suite with every wired competitor harness,
# shuffling the unit order within each round and printing the realized
# order; the benchmark record aggregates across rounds. `gf` is the binary
# tower fields, `gdl` Goldilocks, `m31` Mersenne31 and QuadMersenne31,
# `gf2` bit-packed GF(2). The competitor harnesses live in `external/` as
# separate, unpublished packages — the link to the system ISA-L and the Go
# c-archive needs a build script, and `fgf` itself carries none.
[group('bench')]
bench-gf:
    just bench kernels --gf
    just bench compare

# Build the byte-field competitor harness without running a timing campaign.
[private]
_trio-build:
    cargo build --release --manifest-path external/bench-trio/Cargo.toml

# The binary-field competitor trio. `external/bench-trio/` links the system
# Intel ISA-L (pkg-config `libisal`) and klauspost/reedsolomon as a Go C
# archive (Go toolchain in PATH) into one binary that interleaves `fgf`,
# ISA-L, and klauspost over one fixture set, rotating through all six arm
# orders. The build fails loudly when a toolchain is absent rather than
# dropping the arm. Each round shuffles the self suite, the byte-field trio,
# and the native GF16 harness so no unit owns the warm or cold position.
[group('bench')]
bench-gf-comp:
    #!/usr/bin/env bash
    set -euo pipefail
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
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    for run in 1 2 3 4 5; do
        echo "== round ${run}"
        printf '%s\n' self trio gf16 | shuf | while read -r unit; do
            echo "-- round ${run} unit ${unit}"
            case "${unit}" in
                self)
                    just bench kernels --gf
                    just bench compare
                    ;;
                trio)
                    "${pin[@]}" external/bench-trio/target/release/fgf-bench-trio
                    ;;
                gf16)
                    "${pin[@]}" external/gf16-bench/target/release/fgf-gf16-bench
                    ;;
            esac
        done
    done

[group('bench')]
bench-gdl:
    just bench kernels --gdl

# Field-map operations across every measured pair geometry: scalar chains,
# region rows over the compact-map control loop, and construction
# amortization, with machine records for the `maps` publisher suite.
[group('bench')]
bench-field-maps:
    just bench field_maps

# The prime-field competitor: `external/prime-bench/` compares `fgf`
# against Plonky3 on each library's native layout, built with
# `-C target-cpu=native` because Plonky3 selects its packed kernels at
# compile time. Each round shuffles its units — the self suite and the
# harness — so no unit systematically owns the warm or cold position.
[group('bench')]
bench-gdl-comp:
    #!/usr/bin/env bash
    set -euo pipefail
    RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path external/prime-bench/Cargo.toml --features fgf/simd512
    echo "bench-gdl-comp: five rounds, units self p3"
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    for run in 1 2 3 4 5; do
        echo "== round ${run}"
        printf '%s\n' self p3 | shuf | while read -r unit; do
            echo "-- round ${run} unit ${unit}"
            case "${unit}" in
                self)
                    just bench kernels --gdl
                    ;;
                p3)
                    "${pin[@]}" external/prime-bench/target/release/fgf-bench-prime gdl
                    ;;
            esac
        done
    done

[group('bench')]
bench-m31:
    just bench kernels --m31
    just bench-save prime_ntt
    just bench prime_ntt

[group('bench')]
bench-m31-comp:
    #!/usr/bin/env bash
    set -euo pipefail
    RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path external/prime-bench/Cargo.toml --features fgf/simd512
    echo "bench-m31-comp: five rounds, units self p3"
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    for run in 1 2 3 4 5; do
        echo "== round ${run}"
        printf '%s\n' self p3 | shuf | while read -r unit; do
            echo "-- round ${run} unit ${unit}"
            case "${unit}" in
                self)
                    just bench kernels --m31
                    ;;
                p3)
                    "${pin[@]}" external/prime-bench/target/release/fgf-bench-prime m31
                    ;;
            esac
        done
    done

[group('bench')]
bench-gf2:
    just bench kernels --gf2

[group('bench')]
bench-gf2-comp:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "bench-gf2-comp: no competitor harness is wired for bit-packed GF(2) yet"
    for run in 1 2 3 4 5; do
        echo "== round ${run} order self"
        just bench kernels --gf2
    done

# Native GF16 field comparison; every arm is required and validated before timing.
[group('bench')]
gf16-build:
    cargo build --release --manifest-path external/gf16-bench/Cargo.toml

[group('bench')]
gf16-check *ARGS: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --check {{ARGS}}

# Validate the portable public benchmark inventory and fixture results. Native
# competitors require capable acceptance hardware and have a separate recipe.
[group('bench')]
benchmark-check:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 -m unittest discover -s benches -p 'test_surfaces.py'
    bin=$(cargo build --profile test --all-features --bench surfaces --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    "${bin}" --check | tee "$scratch/check.log"
    just benchmark-api-json
    python3 benches/check_surfaces.py --rustdoc target/doc/fgf.json --check-log "$scratch/check.log"

# Compiler-derived stable callable set: no hand-maintained source-text manifest.
[group('bench')]
benchmark-api-json:
    cargo +{{API_NIGHTLY}} rustdoc --all-features --lib -- -Z unstable-options --output-format json

# Full exact native competitor differentials; never silently skipped on a host
# lacking the required AVX2/GFNI/crypto capability.
[group('bench')]
benchmark-check-native: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --presentations --check
    just gf16-check

[group('bench')]
gf16-smoke *ARGS: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --smoke {{ARGS}}

[group('bench')]
gf16-fmt:
    cargo fmt --manifest-path external/gf16-bench/Cargo.toml

[group('bench')]
gf16-lint:
    cargo fmt --manifest-path external/gf16-bench/Cargo.toml --check
    cargo clippy --release --manifest-path external/gf16-bench/Cargo.toml --all-targets -- -D warnings

[group('bench')]
bench-gf16-comp *ARGS: gf16-build
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    for run in 1 2 3 4 5; do
        echo "== GF16 round ${run}"
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" external/gf16-bench/target/release/fgf-gf16-bench {{ARGS}}
    done

# Build all artifacts before the paired version campaign begins.
[group('bench')]
bench-refresh-build SURFACES="yes":
    #!/usr/bin/env bash
    set -euo pipefail
    just bench-build kernels
    just bench-build compare
    # The frozen v2 tree predates the maps suite and carries no field_maps bench.
    if [[ -f benches/field_maps.rs ]]; then
        just bench-build field_maps
    fi
    case "{{SURFACES}}" in yes) just bench-build surfaces ;; no) ;; *) exit 2 ;; esac
    just _trio-build
    just gf16-build
    RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path external/prime-bench/Cargo.toml --features fgf/simd512

# Run one prebuilt family round; the paired runner owns repetition and version order.
[group('bench')]
bench-refresh-round FAMILY:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    run_self() {
        local name="$1" bin
        shift
        if [[ -n "${FGF_SELF_ARTIFACTS:-}" ]]; then
            bin="$FGF_SELF_ARTIFACTS/$name"
        else
            IFS= read -r bin <"target/fgf-bench/$name.path"
        fi
        taskset -c "$FEC_GOLDEN_CORE" "$bin" "$@"
    }
    case "{{FAMILY}}" in
        gf) units=(self trio gf16) ;;
        gdl|m31) units=(self prime) ;;
        gf2|prime|maps) units=(self) ;;
        surfaces) units=(surfaces presentations) ;;
        *) printf 'invalid_family family=%s\n' "{{FAMILY}}" >&2; exit 2 ;;
    esac
    printf '%s\n' "${units[@]}" | shuf | while read -r unit; do
        printf 'UNIT\t%s\n' "$unit"
        case "$unit" in
            self)
                if [[ "{{FAMILY}}" == maps ]]; then
                    run_self field_maps
                else
                    run_self kernels --{{FAMILY}}
                    if [[ "{{FAMILY}}" == gf ]]; then
                        run_self compare
                    fi
                fi
                ;;
            trio) taskset -c "$FEC_GOLDEN_CORE" external/bench-trio/target/release/fgf-bench-trio ;;
            gf16) taskset -c "$FEC_GOLDEN_CORE" external/gf16-bench/target/release/fgf-gf16-bench ;;
            prime) taskset -c "$FEC_GOLDEN_CORE" external/prime-bench/target/release/fgf-bench-prime {{FAMILY}} ;;
            surfaces) run_self surfaces ;;
            presentations)
                taskset -c "$FEC_GOLDEN_CORE" external/gf16-bench/target/release/fgf-gf16-bench --presentations ;;
        esac
    done

# Compare two source roots on one host without reusing previous timings.
[group('bench')]
bench-paired V2_ROOT OUTPUT:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    : "${SIMD_BACKEND:?set SIMD_BACKEND to the intended host tier}"
    export RAYON_NUM_THREADS=1 GOMAXPROCS=1
    v2=$(realpath "{{V2_ROOT}}")
    v3="$PWD"
    out=$(realpath -m "{{OUTPUT}}")
    mkdir -p "$out"
    [[ ! -e "$out/complete.txt" && ! -e "$out/order.tsv" ]] || {
        printf 'output_already_used path=%s\n' "$out" >&2
        exit 2
    }
    exec 9>"$HOME/.fgf-benchmark-campaign.lock"
    flock -n 9 || { printf 'campaign_host_busy\n' >&2; exit 2; }
    just --justfile "$v2/justfile" --working-directory "$v2" bench-refresh-build no
    just --justfile "$v3/justfile" --working-directory "$v3" bench-refresh-build yes
    for source in "$v2" "$v3"; do
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just --justfile "$source/justfile" --working-directory "$source" test
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just --justfile "$source/justfile" --working-directory "$source" gf16-check
    done
    for round in 1 2 3 4 5; do
        for family in gf gdl m31 gf2 maps; do
            # Field maps have no released-v2 counterpart: the v3 tree alone runs them.
            if [[ "$family" == maps ]]; then
                versions=(v3)
            else
                mapfile -t versions < <(printf '%s\n' v2 v3 | shuf)
            fi
            for version in "${versions[@]}"; do
                source="$v3"
                [[ "$version" == v2 ]] && source="$v2"
                if [[ -n "${FGF_PORTABLE_ARTIFACTS:-}" ]]; then
                    export FGF_SELF_ARTIFACTS="$FGF_PORTABLE_ARTIFACTS/$version"
                fi
                printf '%s\t%s\t%s\n' "$round" "$family" "$version" | tee -a "$out/order.tsv"
                chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" \
                    just --justfile "$source/justfile" --working-directory "$source" bench-refresh-round "$family" \
                    >"$out/round-$round-$version-$family.log" 2>&1
            done
        done
        case "$((round % 3))" in
            1) variants=(v3 avx2 scalar) ;;
            2) variants=(avx2 scalar v3) ;;
            0) variants=(scalar v3 avx2) ;;
        esac
        for variant in "${variants[@]}"; do
            case "$variant" in
                v3) tier="$SIMD_BACKEND" ;;
                avx2) tier=v3 ;;
                scalar) tier=scalar ;;
            esac
            if [[ -n "${FGF_PORTABLE_ARTIFACTS:-}" ]]; then
                export FGF_SELF_ARTIFACTS="$FGF_PORTABLE_ARTIFACTS/v3"
            fi
            printf '%s\tprime\t%s\n' "$round" "$variant" | tee -a "$out/order.tsv"
            SIMD_BACKEND="$tier" chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" \
                just --justfile "$v3/justfile" --working-directory "$v3" bench-refresh-round prime \
                >"$out/round-$round-$variant-prime.log" 2>&1
        done
        # The public-API surface family is current-v3-only: it has no v2
        # counterpart, so it never enters the version shuffle and each of
        # its rounds completes after the legacy families.
        printf '%s\tsurfaces\tv3\n' "$round" | tee -a "$out/order.tsv"
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" \
            just --justfile "$v3/justfile" --working-directory "$v3" bench-refresh-round surfaces \
            >"$out/round-$round-v3-surfaces.log" 2>&1
    done
    printf 'CAMPAIGN_COMPLETE\n' >"$out/complete.txt"

# Materialize the frozen release with benchmark-only public API adapters.
[group('bench')]
bench-prepare-v2 DESTINATION:
    python3 benches/prepare_v2.py --output "{{DESTINATION}}"

# Generate or verify version snapshots and paired comparisons from raw runs.
[group('bench')]
bench-publish RECORDS OUTPUT *ARGS:
    python3 benches/publish.py --records "{{RECORDS}}" --output "{{OUTPUT}}" {{ARGS}}

# Publisher regression: feed real `field_maps` producer output through the
# maps grammar (per-record checks plus the completeness inventory), so a
# grammar change that wrongly requires lane-region or pow-control rows for
# scalar-only or bits-only pairs fails here instead of in a campaign.
# Builds the bench, captures its `--list-labels` inventory (label grammar
# only, no timing), and validates the full producer inventory.
[group('bench')]
bench-maps-publisher:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release --bench field_maps
    bin=$(ls -t target/release/deps/field_maps-* | grep -v '\.d$' | head -n 1)
    "$bin" --list-labels > "$bin-maps-inventory.txt"
    python3 benches/test_maps_publisher.py "$bin-maps-inventory.txt"

# ── Public-route acceptance ───────────────────────────────────────────────
#
# `src/kernel/tests/routes.rs` runs every generated ledger case through the
# public operation surface, judges the output against an element-wise
# reference, and judges the test-only trace against the cell's work budget.
# The recipes below drive that runner, audit its receipts with
# `benches/simd_ledger.py accept`, and prove the audit can fail.

# The rustdoc callable set, the test list, the runner's case list, and the
# public benchmark surface stream, all produced from one source tree.
# Inventories every ledger check resolves against.
[private]
_route-inventories FEATURES DIR:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{DIR}}"
    case "{{FEATURES}}" in
        default) flags=() ;;
        all) flags=(--all-features) ;;
        *) printf 'route_inventory: unsupported features %s\n' "{{FEATURES}}" >&2; exit 2 ;;
    esac
    cargo +{{API_NIGHTLY}} rustdoc --all-features --lib -- -Z unstable-options --output-format json
    cargo test --lib "${flags[@]}" -- --list >"{{DIR}}/tests.txt"
    FGF_ROUTE_LIST=1 cargo test --lib "${flags[@]}" kernel::tests::routes::accept -- --nocapture >"{{DIR}}/cases.txt"
    bin=$(cargo build --profile test --all-features --bench surfaces --message-format=json-render-diagnostics | jq -rs '[.[] | select(.executable != null) | .executable] | last')
    "${bin}" --check >"{{DIR}}/surfaces.txt"

# Run every case and print receipts with the observed served tier and work.
# Routing mismatches never fail the run; a wrong output still does. An
# ambient `SIMD_BACKEND` selects the request cell.
# Survey one feature configuration.
[group('gate')]
route-survey FEATURES LOG:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{FEATURES}}" in
        default) flags=() ;;
        all) flags=(--all-features) ;;
        *) printf 'route_survey: unsupported features %s\n' "{{FEATURES}}" >&2; exit 2 ;;
    esac
    export FGF_ROUTE_SURVEY=1
    mkdir -p "$(dirname "{{LOG}}")"
    cargo test --lib "${flags[@]}" kernel::tests::routes::accept -- --nocapture 2>&1 | tee "{{LOG}}"

# Validate every ledger reference, run the required cases, then audit the
# log with the expectations passed as arguments rather than read from it.
# Fail-closed acceptance for one request and feature configuration.
[group('gate')]
route-accept REQUEST FEATURES:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{FEATURES}}" in
        default) flags=() ;;
        all) flags=(--all-features) ;;
        *) printf 'route_accept: unsupported features %s\n' "{{FEATURES}}" >&2; exit 2 ;;
    esac
    case "{{REQUEST}}" in
        v4x|v3_gfni_crypto|v3|v2|scalar) ;;
        *) printf 'route_accept: unsupported request %s\n' "{{REQUEST}}" >&2; exit 2 ;;
    esac
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    just _route-inventories {{FEATURES}} "$scratch"
    python3 benches/simd_ledger.py check --full \
        --rustdoc target/doc/fgf.json \
        --test-list "$scratch/tests.txt" \
        --case-list "$scratch/cases.txt" \
        --surface-inventory "$scratch/surfaces.txt"
    source_id=$(
        {
            git rev-parse HEAD
            git diff HEAD
            git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum
        } | sha256sum | cut -d' ' -f1
    )
    cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
    node=$(cat /proc/sys/kernel/hostname 2>/dev/null || uname -n)
    host_id="${node}|${cpu}"
    log="$scratch/route.log"
    SIMD_BACKEND="{{REQUEST}}" FGF_ROUTE_ACCEPT=1 \
        FGF_SOURCE_ID="$source_id" FGF_HOST_ID="$host_id" \
        FGF_FEATURES="{{FEATURES}}" \
        cargo test --lib "${flags[@]}" kernel::tests::routes::accept -- --nocapture | tee "$log"
    python3 benches/simd_ledger.py accept --log "$log" \
        --features "{{FEATURES}}" --request "{{REQUEST}}" \
        --source-id "$source_id" --host-id "$host_id"

# Compile against a temporary ledger copy with one required cell, run each
# injected fault and a downgraded request, and assert the audit rejects
# every one with its reason code. The canonical ledger and generated case
# table are restored on exit.
# Prove the acceptance pipeline can fail.
[group('gate')]
route-accept-negative REQUEST:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{REQUEST}}" in
        v4x)
            obligation='ops::mul_add@residue-u32'
            features=all
            flags=(--all-features)
            serving=v4x
            field=v4x
            ;;
        v3_gfni_crypto|v3)
            obligation='ops::mul_add@residue-u32'
            features=default
            flags=()
            serving=v3
            field=v3
            ;;
        v2)
            obligation='ops::mul_add@gf8-poly-aes'
            features=default
            flags=()
            serving=v2
            field=v2
            ;;
        *) printf 'route_accept_negative: no vector budget for request %s\n' "{{REQUEST}}" >&2; exit 2 ;;
    esac
    scratch=$(mktemp -d)
    cases=src/kernel/tests/routes/cases.rs
    cp "$cases" "$scratch/cases.rs"
    trap 'cp "$scratch/cases.rs" "$cases"; rm -rf "$scratch"' EXIT
    ledger="$scratch/ledger.json"
    survey="$scratch/survey.log"
    FGF_ROUTE_SURVEY=1 SIMD_BACKEND="{{REQUEST}}" \
        cargo test --lib "${flags[@]}" kernel::tests::routes::accept -- --nocapture \
        >"$survey" 2>&1
    python3 - "$ledger" "$survey" "{{REQUEST}}" "$obligation" "$serving" "$field" <<'PY'
    import json
    import sys

    out, survey_path, request, obligation, serving, field = sys.argv[1:7]
    ledger = json.load(open("benches/simd_ledger.json"))
    seen = {}
    for line in open(survey_path):
        if not line.startswith("ROUTE_RECEIPT\t"):
            continue
        receipt = json.loads(line.split("\t", 1)[1])
        key = (receipt["obligation"], receipt["features"], receipt["request_raw"])
        identity = (receipt["process"], receipt["field"])
        if key in seen and seen[key] != identity:
            sys.exit(f"route_accept_negative: obligation {key[0]!r} resolves "
                     f"two field backends {seen[key]!r} and {identity!r}")
        seen[key] = identity
    obligations = ledger.setdefault("obligations", {})
    for (name, features, requested), (process, field_backend) in seen.items():
        cells = obligations.setdefault(name, {}).setdefault("cells", {})
        cells[f"{features}/{requested}"] = {
            "status": "direct",
            "required": False,
            "min_serving": requested,
            "work": [],
            "process": process,
            "field": field_backend,
        }
    obligations.setdefault(obligation, {}).setdefault("cells", {})[f"default/{request}"] = {
        "status": "direct",
        "required": True,
        "min_serving": serving,
        "work": ["row-mul-add"],
        "process": request,
        "field": field,
    }
    json.dump(ledger, open(out, "w"), indent=1, sort_keys=False)
    PY
    python3 benches/simd_ledger.py --ledger "$ledger" generate --out "$cases"
    # The generated map construction data carries caller images as target
    # byte encodings. Until the canonical generator emits that spelling,
    # reuse the canonical table's map constants, which the temporary ledger
    # leaves unchanged.
    python3 - "$cases" "$scratch/cases.rs" <<'PY'
    import re
    import sys

    generated, canonical = sys.argv[1], sys.argv[2]
    pattern = re.compile(r"const (MAP_\d+): MapSpec = MapSpec \{ .* \};")

    def constants(path):
        found = {}
        for line in open(path):
            match = pattern.match(line.rstrip("\n"))
            if match:
                found[match.group(1)] = match.group(0)
        return found

    wanted = constants(canonical)
    lines = []
    for line in open(generated):
        match = pattern.match(line.rstrip("\n"))
        if match and match.group(1) in wanted:
            lines.append(wanted[match.group(1)] + "\n")
        else:
            lines.append(line)
    open(generated, "w").writelines(lines)
    PY
    run_negative() {
        local backend="$1" fault="$2" log="$3" source="$4" rc=0
        FGF_ROUTE_FAULT="$fault" SIMD_BACKEND="$backend" FGF_ROUTE_ACCEPT=1 \
            FGF_SOURCE_ID="$source" FGF_HOST_ID=negative FGF_FEATURES="$features" \
            cargo test --lib "${flags[@]}" kernel::tests::routes::accept -- --nocapture \
            >"$log" 2>&1 || rc=$?
        printf '%s' "$rc"
    }
    expect_reject() {
        local code="$1" log="$2" err="$3" rc=0
        python3 benches/simd_ledger.py --ledger "$ledger" accept --log "$log" \
            --features "$features" --request "{{REQUEST}}" \
            --source-id negative --host-id negative 2>"$err" || rc=$?
        if [[ "$rc" -eq 0 ]]; then
            printf 'route_accept_negative: audit accepted %s\n' "$code" >&2
            return 1
        fi
        grep -Fq "simd_ledger: reject $code" "$err" || {
            printf 'route_accept_negative: expected reject %s, got:\n' "$code" >&2
            cat "$err" >&2
            return 1
        }
        printf 'ROUTE_NEGATIVE\tcode=%s\n' "$code"
    }
    base="$scratch/base.log"
    rc=$(run_negative "{{REQUEST}}" "" "$base" negative)
    [[ "$rc" -eq 0 ]] || {
        printf 'route_accept_negative: baseline run failed rc=%s\n' "$rc" >&2
        tail -40 "$base" >&2
        exit 1
    }
    python3 benches/simd_ledger.py --ledger "$ledger" accept --log "$base" \
        --features "$features" --request "{{REQUEST}}" \
        --source-id negative --host-id negative
    for fault in corrupt-output suppress-trace halve-work lower-tier; do
        log="$scratch/fault-$fault.log"
        rc=$(run_negative "{{REQUEST}}" "$fault" "$log" negative)
        [[ "$rc" -ne 0 ]] || {
            printf 'route_accept_negative: fault %s was not rejected\n' "$fault" >&2
            exit 1
        }
        expect_reject fail_receipt "$log" "$scratch/fault-$fault.err"
    done
    # Omitting a receipt leaves the runner's verdict untouched; the audit
    # must reject the shortened log.
    drop="$scratch/fault-drop-receipt.log"
    run_negative "{{REQUEST}}" drop-receipt "$drop" negative >/dev/null
    expect_reject count_mismatch "$drop" "$scratch/fault-drop-receipt.err"
    case "{{REQUEST}}" in
        v4x) lower=v3_gfni_crypto ;;
        v3_gfni_crypto) lower=v3 ;;
        v3) lower=v2 ;;
        v2) lower=scalar ;;
        *) lower='' ;;
    esac
    if [[ -n "$lower" ]]; then
        forced="$scratch/forced-$lower.log"
        rc=$(run_negative "$lower" "" "$forced" negative)
        [[ "$rc" -eq 0 ]] || {
            printf 'route_accept_negative: downgraded run failed rc=%s\n' "$rc" >&2
            exit 1
        }
        expect_reject request_mismatch "$forced" "$scratch/forced-$lower.err"
    fi
    printf 'ROUTE_NEGATIVE_DONE\trequest=%s\n' "{{REQUEST}}"

# Full ledger resolution, byte-identical case generation, and the ledger
# unit tests. Run on the machine that holds the collected evidence records.
# The rerunnable route gate.
[group('gate')]
route-gate:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p bench-records/routes
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    just _route-inventories default "$scratch"
    python3 benches/simd_ledger.py check --full \
        --rustdoc target/doc/fgf.json \
        --test-list "$scratch/tests.txt" \
        --case-list "$scratch/cases.txt" \
        --surface-inventory "$scratch/surfaces.txt" \
        --records bench-records/routes
    python3 benches/simd_ledger.py generate --check --out src/kernel/tests/routes/cases.rs
    python3 -m unittest discover -s benches -p 'test_simd_ledger.py'

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
    just external-check
    just test-gather-validation-32
    just examples
    just examples --no-default-features --features alloc
    just examples-portable
    just benchmark-check
    just benchmark-check-native
    just package-check

# Route ledger and tooling: schema-only validation plus the Python unit tests.
# `just benchmark-api-json` must have already run (or package-check/validate
# will have produced target/doc/fgf.json).
[group('bench')]
simd-ledger:
    just benchmark-api-json
    python3 benches/simd_ledger.py inventory --rustdoc target/doc/fgf.json
    python3 benches/simd_ledger.py check --schema-only
    python3 -m unittest discover -s benches -p 'test_simd_ledger.py'
