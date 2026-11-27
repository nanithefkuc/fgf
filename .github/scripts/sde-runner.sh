#!/usr/bin/env bash
# Cargo target runner for the emulated hosted coverage shards. Hosted x86-64
# runners differ in CPU generation, and `SIMD_BACKEND` only downgrades, so the
# GFNI and AVX-512 shards run every test process under Intel SDE emulating
# Sapphire Rapids. CI installs this runner only for those shards.
set -euo pipefail
exec "${SDE_PATH:?SDE_PATH must name the SDE install directory}/sde64" -spr -- "$@"
