# Wider binary fields: `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64`

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better. Latency: lower is better. A dash means no measurement.

[All families](../../BENCHMARKS.md) · [Field definitions](../labels.md)

## Find a result

- [Single-row multiplication](#single-row-multiplication)
- [Competitors](#competitors)
- [Measurement details](#measurement-details)

## Self-timings

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Field | Operation | mul_add (GiB/s) |
| --- | --- | --- |
| fp16 | mul_add | 9.54 / 16.5 |
| fp32 | mul_add | 3.65 / 6.08 |
| fp64 | mul_add | 1.37 / 2.12 |
| gf32b | mul_add | 15.7 / 30.2 |
| gf64b | mul_add | 8.70 / 16.3 |

## Competitors

No matched competitor measurements are available.

## Measurement details

<details>
<summary>Setup, resolved backends, versions, and caveats</summary>

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` |
| Revision | `63b8d168a7e4b81043fafe4db1ffb9918dc1c231` plus the measured working-tree archive |
| Toolchain | 1.98.1 |
| Willow Cove (i5-1135G7) process | `v4x` |
| Golden Cove (i7-12700K) process | `v3_gfni_crypto` |
| `gf32b` backend | `v4x` / `v3_gfni_crypto` |
| `gf64b` backend | `v4x` / `v3_gfni_crypto` |
| `fp16` backend | `v4x` / `v3_gfni_crypto` |
| `fp32` backend | `v4x` / `v3_gfni_crypto` |
| `fp64` backend | `v4x` / `v3_gfni_crypto` |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by aggregate latency per call; numerators are stated per panel. |

- The unchanged byte-XOR control brackets every self-suite process; its values are retained in the raw evidence.

</details>
