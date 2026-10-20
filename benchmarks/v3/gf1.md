# Bit-packed GF(2): `gf1`

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better. Latency: lower is better. A dash means no measurement.

[All families](../../BENCHMARKS.md) · [Field definitions](../labels.md)

## Find a result

- [Buffer operations](#buffer-operations)
- [Short ranges](#short-ranges)
- [Competitors](#competitors)
- [Measurement details](#measurement-details)

## Self-timings

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits [N/4, 7N/8); `bits::XorRange` is the prepared geometry that `bits::xor_range_with` applies.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| bits::XorRange 5/8 | 58.8 / 82.7 | 66.0 / 71.5 | 21.0 / 38.2 | 9.84 / 24.0 |
| bits::and_into | 19.3 / 66.0 | 19.3 / 46.7 | 5.85 / 18.9 | 4.65 / 10.8 |
| bits::dot_product | 20.3 / 38.2 | 18.3 / 41.7 | 10.7 / 36.4 | 8.79 / 18.6 |
| bits::weight | 118 / 13.9 | 67.5 / 14.4 | 38.7 / 14.4 | 17.9 / 13.9 |
| bits::xor_assign | 43.3 / 101 | 41.4 / 63.4 | 8.60 / 24.4 | 6.11 / 14.7 |
| bits::xor_range 5/8 | 53.7 / 74.3 | 65.9 / 71.4 | 21.0 / 38.2 | 9.75 / 23.8 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.39 / 2.16 |
| bits::xor_range | 11.8 / 5.61 |

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
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by aggregate latency per call; numerators are stated per panel. |

- The unchanged byte-XOR control brackets every self-suite process; its values are retained in the raw evidence.

</details>
