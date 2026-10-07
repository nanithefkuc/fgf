# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.93'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v4x v3_gfni_crypto v3 v2 scalar'

# Owned-unsafe coverage is provided by the promoted GFNI recipe below.
MIRI := ''

# Coverage exclusions for code that cannot execute on a CI host. The binary
# field kernels require GFNI, and QuadMersenne31's AVX-512 kernel requires V4x.
# The GitHub-hosted coverage runner cannot select those tiers; mixed-ISA prime
# dispatch modules remain included.
COV_IGNORE := '"kernel/x86/(gf8|gf16|gf32|gf64)(/|\.rs$)|kernel/x86/quad_mersenne31/avx512\.rs$|kernel/(gf8|tower)\.rs$"'

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
    for name in field_arithmetic shard_recovery streaming_encode binary_syndrome extension_fields embeddings prime_vectors; do
        cargo run --example "${name}" {{ARGS}}
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

# Compile the non-x86 library paths, including Wasm's explicit SIMD feature.
[group('test')]
cross-check:
    cargo check --target aarch64-unknown-linux-gnu --all-features --lib
    cargo check --target wasm32-unknown-unknown --all-features --lib
    RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" cargo check --target wasm32-unknown-unknown --all-features --lib

# Build a portable benchmark artifact for interleaved remote comparisons.
[group('bench')]
bench-build NAME:
    cargo bench --all-features --bench {{NAME}} --no-run

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
gf16-check: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --check

[group('bench')]
gf16-smoke: gf16-build
    external/gf16-bench/target/release/fgf-gf16-bench --smoke

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
bench-refresh-build:
    just bench-build kernels
    just bench-build compare
    just _trio-build
    just gf16-build
    RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path external/prime-bench/Cargo.toml --features fgf/simd512

# Run one family round; the paired runner owns repetition and version order.
[group('bench')]
bench-refresh-round FAMILY:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FEC_GOLDEN_CORE:?set FEC_GOLDEN_CORE to an isolated core}"
    case "{{FAMILY}}" in
        gf) units=(self trio gf16) ;;
        gdl|m31) units=(self prime) ;;
        gf2) units=(self) ;;
        *) printf 'invalid_family family=%s\n' "{{FAMILY}}" >&2; exit 2 ;;
    esac
    printf '%s\n' "${units[@]}" | shuf | while read -r unit; do
        printf 'UNIT\t%s\n' "$unit"
        case "$unit" in
            self)
                if [[ -n "${FGF_SELF_ARTIFACTS:-}" ]]; then
                    taskset -c "$FEC_GOLDEN_CORE" "$FGF_SELF_ARTIFACTS/kernels" --{{FAMILY}}
                    if [[ "{{FAMILY}}" == gf ]]; then
                        taskset -c "$FEC_GOLDEN_CORE" "$FGF_SELF_ARTIFACTS/compare"
                    fi
                else
                    just bench kernels --{{FAMILY}}
                    if [[ "{{FAMILY}}" == gf ]]; then
                        just bench compare
                    fi
                fi
                ;;
            trio) taskset -c "$FEC_GOLDEN_CORE" external/bench-trio/target/release/fgf-bench-trio ;;
            gf16) taskset -c "$FEC_GOLDEN_CORE" external/gf16-bench/target/release/fgf-gf16-bench ;;
            prime) taskset -c "$FEC_GOLDEN_CORE" external/prime-bench/target/release/fgf-bench-prime {{FAMILY}} ;;
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
    for source in "$v2" "$v3"; do
        just --justfile "$source/justfile" --working-directory "$source" bench-refresh-build
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just --justfile "$source/justfile" --working-directory "$source" test
        chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just --justfile "$source/justfile" --working-directory "$source" gf16-check
    done
    for round in 1 2 3 4 5; do
        for family in gf gdl m31 gf2; do
            printf '%s\n' v2 v3 | shuf | while read -r version; do
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
