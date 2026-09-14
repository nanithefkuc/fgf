# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to fgf belong in this file and are documented in `AGENTS.md`.

CRATE := 'fgf'
MSRV := '1.89'

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

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
COV_IGNORE := '"kernel/x86/(gf8|gf16|gf32|gf64|avx512)(/|\.rs$)"'

# The shared `features` recipe walks `--no-default-features`, default, and
# `--all-features`. `alloc` sits between the first two: it turns on the
# prepared coefficient collections (`CoeffVec`, `CoeffMatrix`) and
# `pack_to_vec` without `std`, so neither end of the shared walk builds it.
# CI runs this alongside `just features`.
[group('test')]
features-alloc:
    cargo test --no-default-features --features alloc
    cargo clippy --no-default-features --features alloc --all-targets -- -D warnings
