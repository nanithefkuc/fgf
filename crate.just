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

# Coverage exclusions for code that cannot execute on a CI host. The x86
# kernel subtrees need their own ISA; `kernel/gf8.rs` and `kernel/tower.rs`
# carry the GFNI dispatch arms, which no GitHub-hosted runner can select.
COV_IGNORE := '"kernel/x86/(gf8|gf16|gf32|gf64)(/|\.rs$)|kernel/(gf8|tower)\.rs$"'

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
--features internals --test internals proven_gf8d_gfni_kernels_match_scalar -- --exact
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

# The binary-field competitor trio. `external/bench-trio/` links the system
# Intel ISA-L (pkg-config `libisal`) and klauspost/reedsolomon as a Go C
# archive (Go toolchain in PATH) into one binary that interleaves `fgf`,
# ISA-L, and klauspost over one fixture set, rotating through all six arm
# orders. The build fails loudly when a toolchain is absent rather than
# dropping the arm. Each round shuffles its units — the self suite, then
# one trio invocation — so no unit systematically owns the warm or cold
# position.
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
    cargo build --release --manifest-path external/bench-trio/Cargo.toml
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    for run in 1 2 3 4 5; do
        echo "== round ${run}"
        printf '%s\n' self trio | shuf | while read -r unit; do
            echo "-- round ${run} unit ${unit}"
            case "${unit}" in
                self)
                    just bench kernels --gf
                    just bench compare
                    ;;
                trio)
                    "${pin[@]}" external/bench-trio/target/release/fgf-bench-trio
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

# The QuadMersenne31 threshold evidence in `prime_ntt` keeps its paired
# two-execution protocol outside the rounds: its estimator pools the scalar
# and duplicated scalar arms over both executions, and re-rounding it would
# redefine that recorded estimator.
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
    just bench-save prime_ntt
    just bench prime_ntt

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
