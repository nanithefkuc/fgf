# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.93'
API_NIGHTLY := 'nightly-2026-09-30'

# Stable public-surface gate. The inventory tools are external development
# tools pinned here and in the CI installer; both read the rustdoc JSON format
# `API_NIGHTLY` emits. `API_PATHS_MODE` is `report` until every stable item
# has one path, then `enforce`. `API_MAJOR` is the major version whose
# published releases form the semver baseline.
API_PUBLIC_API := '0.52.0'
API_SEMVER_CHECKS := '0.51.0'
API_FORMAT := '61'
API_TARGETS := 'x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu wasm32-unknown-unknown'
API_PATHS_MODE := 'report'
API_MAJOR := '4'

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
# CI uploads the export; run `just cover` first.
[private]
[group('cover')]
cover-lcov:
    cargo llvm-cov report --lcov --output-path coverage.lcov --ignore-filename-regex {{COV_IGNORE}}


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
                    just bench kernels --{{FAMILY}}
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
    python3 -m unittest discover -s benches -p 'test_surfaces.py'
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

# Build all artifacts before the paired version campaign begins.
[private]
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
[private]
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
    just benchmark-check
    just benchmark-check-native
    just package-check
    just api
    just api-semver-release

# Generate the stable inventories, their metadata, and the one-path report
# into `target/api/generated`. Every configuration builds unchanged; lines
# naming `fgf::internals` are dropped from the inventory. Tool versions are
# checked before anything is generated. It runs as a dependency so variable
# overrides on the command line reach it.
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
    API_NIGHTLY="{{API_NIGHTLY}}" python3 -m unittest -q api/test_surface.py
    out=target/api/generated
    rm -rf "$out"
    mkdir -p "$out/inventory"
    export RUSTDOCFLAGS="-Z unstable-options --document-hidden-items"
    configs=("portable|--no-default-features" "alloc|--no-default-features --features alloc" "default|" "all|--all-features")
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
            cargo +{{API_NIGHTLY}} public-api -s --color never --target "$target" \
                --target-dir target/api "${flags[@]}" 2>"$out/$config-$target.log" \
                | python3 api/surface.py project >"$out/inventory/$config-$target.txt" || {
                cat "$out/$config-$target.log" >&2
                exit 1
            }
            json="target/api/$target/doc/fgf.json"
            if ! report=$(python3 api/surface.py paths "$json" --mode {{API_PATHS_MODE}}); then
                printf '%s\n' "$report"
                failed=1
            fi
            printf 'api: %-8s %-26s %s\n' "$config" "$target" "$(head -n 1 <<<"$report" | sed 's/^surface: //')"
            [[ "$config-$target" == all-x86_64-unknown-linux-gnu ]] && printf '%s\n' "$report" >"$out/paths.txt"
        done
    done
    tail -n 1 "$out/paths.txt"
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

# Semver check of the stable feature sets against the latest published
# non-prerelease `API_MAJOR`.x release, or against BASELINE when given for
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
        baseline=$(curl -fsS -A "fgf-api-semver" "https://crates.io/api/v1/crates/{{CRATE}}/versions?per_page=100" \
            | jq -r --arg major "{{API_MAJOR}}" '[.versions[] | select(.yanked | not) | .num | select(test("^" + $major + "\\.[0-9]+\\.[0-9]+$"))] | first // empty') || {
            printf 'api-semver: could not query crates.io for the published baseline\n' >&2
            exit 1
        }
        [[ -n "$baseline" ]] || {
            printf 'api-semver: no published non-prerelease %s.x release to compare against\n' "{{API_MAJOR}}" >&2
            exit 1
        }
    fi
    current=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')
    printf 'api-semver: candidate %s against published %s\n' "$current" "$baseline"
    configs=("portable|--only-explicit-features" "alloc|--only-explicit-features --features alloc" "default|--default-features" "stable-max|--only-explicit-features --features alloc,std,simd,simd512")
    for entry in "${configs[@]}"; do
        # shellcheck disable=SC2206
        flags=(${entry#*|})
        printf 'api-semver: %s\n' "${entry%%|*}"
        cargo +{{API_NIGHTLY}} semver-checks check-release --baseline-version "$baseline" "${flags[@]}"
    done

# Release gate for `api-semver`: the first release of `API_MAJOR` has no
# published baseline of its own major and is skipped, as is every other
# version string; later `API_MAJOR`.x releases run the check.
[private]
api-semver-release VERSION="":
    #!/usr/bin/env bash
    set -euo pipefail
    version="{{VERSION}}"
    [[ -n "$version" ]] || version=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')
    if [[ "$version" =~ ^{{API_MAJOR}}\.[0-9]+\.[0-9]+$ && "$version" != "{{API_MAJOR}}.0.0" ]]; then
        just api-semver
    else
        printf 'api-semver: skipped for %s; only %s.x releases after %s.0.0 have a published baseline of their major\n' "$version" "{{API_MAJOR}}" "{{API_MAJOR}}"
    fi
