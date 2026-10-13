# Wider binary fields: `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/wide-binary.md); the other snapshot at [v2](../v2/wide-binary.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `gf32b`: `v4x` / `v3_gfni_crypto`; `gf64b`: `v4x` / `v3_gfni_crypto`; `fp16`: `v4x` / `v3_gfni_crypto`; `fp32`: `v4x` / `v3_gfni_crypto`; `fp64`: `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |

## Self-timings

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Field | Operation | mul_add (GiB/s) |
| --- | --- | --- |
| fp16 | mul_add | 9.97 / 16.5 |
| fp32 | mul_add | 3.71 / 6.08 |
| fp64 | mul_add | 1.36 / 1.82 |
| gf32b | mul_add | 17.0 / 30.1 |
| gf64b | mul_add | 9.09 / 16.3 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
