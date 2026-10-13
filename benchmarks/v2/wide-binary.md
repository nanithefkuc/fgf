# Wider binary fields: `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/wide-binary.md); the other snapshot at [v3](../v3/wide-binary.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `-` |
| Resolved field backends | `gf32b`: `v4x` / -; `gf64b`: `v4x` / -; `fp16`: `v4x` / -; `fp32`: `v4x` / -; `fp64`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |

## Self-timings

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Field | Operation | mul_add (GiB/s) |
| --- | --- | --- |
| fp16 | mul_add | 9.97 / - |
| fp32 | mul_add | 3.71 / - |
| fp64 | mul_add | 1.37 / - |
| gf32b | mul_add | 17.1 / - |
| gf64b | mul_add | 9.09 / - |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
