#!/usr/bin/env bash
# Cargo target runner for the hosted coverage job. Hosted x86-64 runners differ
# in CPU generation, and `SIMD_BACKEND` only downgrades, so the GFNI and
# AVX-512 tiers run under Intel SDE emulating Sapphire Rapids: those passes
# execute the same paths on every runner. Every other run executes natively.
set -euo pipefail
case "${SIMD_BACKEND:-}" in
    v4x | v3_gfni_crypto)
        exec "${SDE_PATH:?SDE_PATH must name the SDE install directory}/sde64" -spr -- "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
