# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.93'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v3_gfni_crypto v3 v2 scalar'

# `cargo miri test` argument sets, one per line. Empty when the crate declares
# no unsafe surface, in which case `just unsafe-check` reports and skips.
MIRI := '''--no-default-features --test ops gf8_mul_add_matches_oracle -- --exact
--no-default-features --test ops gf16_mul_into_matrix_overwrites -- --exact
--no-default-features --test ops gf8_matrix_scattered_matches_contiguous -- --exact
--no-default-features --test ops elementwise_products_match_field_arithmetic -- --exact
--no-default-features --test ops mersenne31_public_ops_match_oracle -- --exact
--no-default-features --test ops goldilocks_public_ops_match_oracle -- --exact
--no-default-features --test ops gf16_add_assign_rows_matches_oracle -- --exact
--no-default-features --test ops gf16_add_gather_matches_oracle -- --exact'''

# Coverage exclusions for code that cannot execute on a CI host. The x86
# kernel subtrees need their own ISA; `kernel/gf8.rs` and `kernel/tower.rs`
# carry the GFNI dispatch arms, which no GitHub-hosted runner can select.
COV_IGNORE := '"kernel/x86/(gf8|gf16|gf32|gf64|avx512)(/|\.rs$)|kernel/(gf8|tower)\.rs$"'

# The shared `features` recipe walks `--no-default-features`, default, and
# `--all-features`. `alloc` sits between the first two: it turns on the
# prepared coefficient collections (`CoeffVec`, `CoeffMatrix`) and
# `pack_to_vec` without `std`, so neither end of the shared walk builds it.
# CI runs this alongside `just features`.
[group('test')]
features-alloc:
    cargo test --no-default-features --features alloc
    cargo clippy --no-default-features --features alloc --all-targets -- -D warnings

# Exposure must compile without activating SIMD or its capability-token dependency.
[group('test')]
features-internals:
    cargo check --all-targets --no-default-features
    cargo check --all-targets --no-default-features --features internals
    cargo check --all-targets --no-default-features --features alloc,internals
    cargo check --all-targets --no-default-features --features std,internals
    cargo check --all-targets
    cargo check --all-targets --all-features

# The ISA-L competitor comparison. `bench-isal/` is a separate, unpublished
# package: linking the system ISA-L needs a build script, and `fgf` itself
# carries none. Requires `libisal` discoverable through pkg-config; the
# build fails loudly when it is absent rather than dropping the arm.
[group('bench')]
bench-isal:
    #!/usr/bin/env bash
    set -euo pipefail
    pkg-config --modversion libisal >/dev/null || {
        echo "libisal not found through pkg-config; install Intel ISA-L" >&2
        exit 1
    }
    cargo build --release --manifest-path bench-isal/Cargo.toml
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    "${pin[@]}" bench-isal/target/release/fgf-bench-isal

# The klauspost/reedsolomon competitor comparison. `bench-klauspost/` is a
# separate, unpublished package in the shape of `bench-isal/`: linking the Go
# c-archive needs a build script, and `fgf` itself carries none. Requires the
# Go toolchain in PATH; the build fails loudly when it is absent rather than
# dropping the arm.
[group('bench')]
bench-klauspost:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v go >/dev/null || {
        echo "go not found in PATH; install the Go toolchain" >&2
        exit 1
    }
    cargo build --release --manifest-path bench-klauspost/Cargo.toml
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    "${pin[@]}" bench-klauspost/target/release/fgf-bench-klauspost

# The prime-field NTT consumer harness. `external-bench/prime-ntt/` is a
# separate, unpublished package that path-depends on the local `fgf` and the
# local `butterfly-fft` (with its `internals` feature) and patches the
# registry's `fgf` and `simdispatch` onto those same local copies, so the
# published pins of either crate stay untouched. The bench validates every
# geometry — forced fused against forced packed against production forward,
# inverse roundtrip, and a direct DFT up to size 1024 — before any arm is
# timed, and panics on mismatch. ARGS pass through to Criterion as filters;
# `--test` runs each arm once without timing.
[group('bench')]
bench-prime-ntt *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo bench --manifest-path external-bench/prime-ntt/Cargo.toml --bench ntt_consumer --no-run
    pin=()
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        pin=(taskset -c "${FEC_GOLDEN_CORE}")
        echo "pinned to core ${FEC_GOLDEN_CORE}"
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
    fi
    "${pin[@]}" cargo bench --manifest-path external-bench/prime-ntt/Cargo.toml --bench ntt_consumer -- {{ARGS}}
